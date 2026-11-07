"""mymind.py: the modern (Llama / Qwen3 / minimind-3) decoder built in P1-02, in one importable file.
Changes from mygpt.py (GPT-2): RMSNorm, RoPE, SwiGLU, grouped-query attention, QK-norm, no biases, KV cache.
Module names follow the Llama/Qwen convention (q_proj, gate_proj, ...) so released weights map 1:1."""
import math, torch, torch.nn as nn, torch.nn.functional as F

MIND_CONFIG_64M = {   # minimind-3 (64M): same shapes as jingyaogong/minimind-3
    'vocab_size': 6400, 'context_length': 2048, 'emb_dim': 768, 'n_layers': 8,
    'n_heads': 8, 'n_kv_heads': 4, 'ffn_dim': 2432, 'rope_theta': 1e6, 'norm_eps': 1e-6, 'drop_rate': 0.0}

def ffn_dim_for(d):   # minimind's rule: about pi * d, rounded up to a multiple of 64
    return math.ceil(d * math.pi / 64) * 64

class RMSNorm(nn.Module):
    """LayerNorm without the mean subtraction and without the shift: x / rms(x) * weight."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        xf = x.float()
        return (self.weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))).type_as(x)

def rope_tables(head_dim, max_len, theta=1e6):
    """cos/sin of angle pos * freq_i, one row per position. Pair i of a head rotates at freq theta^(-2i/d)."""
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    ang = torch.outer(torch.arange(max_len).float(), freqs)          # (max_len, head_dim/2)
    ang = torch.cat([ang, ang], dim=-1)                               # (max_len, head_dim)
    return ang.cos(), ang.sin()

def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)

def apply_rope(x, cos, sin):
    """x: (B, T, H, D); cos/sin: (T, D). Rotates dim i together with dim i + D/2."""
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return (x * cos + rotate_half(x) * sin).type_as(x)

class KVCache:
    """Keys/values of every layer for the tokens seen so far, so generation only runs the newest token."""
    def __init__(self, n_layers):
        self.kv, self.pos = [None] * n_layers, 0
    def update(self, layer, k, v):     # k, v: (B, T_new, H_kv, D)
        if self.kv[layer] is not None:
            k = torch.cat([self.kv[layer][0], k], dim=1); v = torch.cat([self.kv[layer][1], v], dim=1)
        self.kv[layer] = (k, v)
        return k, v

class GQAttention(nn.Module):
    """Causal attention where n_heads query heads share n_kv_heads key/value heads."""
    def __init__(self, cfg, layer):
        super().__init__()
        d, self.h, self.h_kv = cfg['emb_dim'], cfg['n_heads'], cfg['n_kv_heads']
        assert d % self.h == 0 and self.h % self.h_kv == 0
        self.hd, self.layer, self.p = d // self.h, layer, cfg['drop_rate']
        self.q_proj = nn.Linear(d, self.h * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.h_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.h_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.h * self.hd, d, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.hd, cfg['norm_eps']), RMSNorm(self.hd, cfg['norm_eps'])   # QK-norm
    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.h, self.hd))
        k = self.k_norm(self.k_proj(x).view(B, T, self.h_kv, self.hd))
        v = self.v_proj(x).view(B, T, self.h_kv, self.hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer, k, v)        # cache stores only h_kv heads: that is the memory saving
        S = k.shape[1]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # (B, heads, len, hd)
        rep = self.h // self.h_kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        drop = self.p if self.training else 0.0
        if S == T:      # no earlier tokens: plain causal mask
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        else:           # T new tokens after S - T cached ones: new token j sees everything up to its own position
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.h * self.hd))

class SwiGLU(nn.Module):
    """down( silu(gate(x)) * up(x) ): one branch decides how much of the other passes through."""
    def __init__(self, cfg):
        super().__init__()
        d, f = cfg['emb_dim'], cfg.get('ffn_dim') or ffn_dim_for(cfg['emb_dim'])
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MindBlock(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(cfg['emb_dim'], cfg['norm_eps']), RMSNorm(cfg['emb_dim'], cfg['norm_eps'])
        self.attn, self.ffn = GQAttention(cfg, layer), SwiGLU(cfg)
        self.drop = nn.Dropout(cfg['drop_rate'])
    def forward(self, x, cos, sin, cache=None):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, cache))
        return x + self.drop(self.ffn(self.ffn_norm(x)))

class MindModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])
        self.blocks = nn.ModuleList([MindBlock(cfg, i) for i in range(cfg['n_layers'])])
        self.final_norm = RMSNorm(cfg['emb_dim'], cfg['norm_eps'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
        self.apply(self._init_weights)
        self.out_head.weight = self.tok_emb.weight          # tied, as in GPT-2 and minimind
        cos, sin = rope_tables(cfg['emb_dim'] // cfg['n_heads'], cfg['context_length'], cfg['rope_theta'])
        self.register_buffer('rope_cos', cos, persistent=False); self.register_buffer('rope_sin', sin, persistent=False)
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx, cache=None):
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        assert start + T <= self.cfg['context_length'], 'sequence longer than context_length'
        cos, sin = self.rope_cos[start:start + T], self.rope_sin[start:start + T]
        x = self.drop(self.tok_emb(idx))
        for b in self.blocks:
            x = b(x, cos, sin, cache)
        if cache is not None:
            cache.pos += T
        return self.out_head(self.final_norm(x))

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.0, top_k=None, eos_id=None, use_cache=True):
    """temperature=0 -> greedy. With use_cache the prompt runs once, then one token per step."""
    model.eval()
    ctx = model.cfg['context_length']
    cache = KVCache(len(model.blocks)) if use_cache else None
    nxt = idx
    for _ in range(max_new_tokens):
        if use_cache and cache.pos + nxt.shape[1] > ctx:
            break                                        # cache full; keep it simple and stop
        logits = model(nxt, cache) if use_cache else model(idx[:, -ctx:])
        logits = logits[:, -1, :].float()
        if top_k is not None:
            kth = torch.topk(logits, top_k).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float('-inf'))
        if temperature > 0:
            next_id = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
        else:
            next_id = logits.argmax(-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
        nxt = next_id
        if eos_id is not None and (next_id == eos_id).all():
            break
    return idx

def load_minimind_weights(model, sd):
    """P1-02: copy jingyaogong/minimind-3 weights (its model.safetensors dict) into MindModel.
    Same layout, only the names differ; no transposes needed (minimind uses nn.Linear like us)."""
    def put(param, key):
        v = sd[key]
        assert param.shape == v.shape, f'{key}: {tuple(param.shape)} vs {tuple(v.shape)}'
        with torch.no_grad():
            param.copy_(v.to(param.dtype))
    put(model.tok_emb.weight, 'model.embed_tokens.weight')
    for i, b in enumerate(model.blocks):
        p = f'model.layers.{i}.'
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            put(getattr(b.attn, name).weight, p + f'self_attn.{name}.weight')
        put(b.attn.q_norm.weight, p + 'self_attn.q_norm.weight'); put(b.attn.k_norm.weight, p + 'self_attn.k_norm.weight')
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            put(getattr(b.ffn, name).weight, p + f'mlp.{name}.weight')
        put(b.attn_norm.weight, p + 'input_layernorm.weight'); put(b.ffn_norm.weight, p + 'post_attention_layernorm.weight')
    put(model.final_norm.weight, 'model.norm.weight')
    return model
