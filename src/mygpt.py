"""mygpt.py: the GPT built step by step in notebooks P0-01..P0-03, in one importable file.
Later notebooks import from here instead of re-pasting the classes."""
import math, torch, torch.nn as nn

GPT_CONFIG_124M = {'vocab_size': 50257, 'context_length': 1024, 'emb_dim': 768,
                   'n_heads': 12, 'n_layers': 12, 'drop_rate': 0.1, 'qkv_bias': False}

class MultiHeadAttention(nn.Module):          # P0-02
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.h, self.hd = num_heads, d_out // num_heads
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())
    def forward(self, x):
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.h, self.hd).transpose(1, 2)
        Q, K, V = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        scores = (Q @ K.transpose(-2, -1) / self.hd**0.5).masked_fill(self.mask[:T, :T], float('-inf'))
        A = self.dropout(torch.softmax(scores, dim=-1))
        return self.out_proj((A @ V).transpose(1, 2).contiguous().view(B, T, self.h * self.hd))

class LayerNorm(nn.Module):                   # P0-03
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim)); self.shift = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True); var = x.var(-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['emb_dim']
        self.net = nn.Sequential(nn.Linear(d, 4 * d), GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg['emb_dim'], cfg['emb_dim'], cfg['context_length'],
                                      cfg['n_heads'], cfg['drop_rate'], cfg['qkv_bias'])
        self.ff = FeedForward(cfg)
        self.norm1, self.norm2 = LayerNorm(cfg['emb_dim']), LayerNorm(cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])
    def forward(self, x):
        x = x + self.drop(self.att(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg, tie_weights=False):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
        self.apply(self._init_weights)   # GPT-2 style: small random weights (std 0.02)
        if tie_weights:
            self.out_head.weight = self.tok_emb.weight
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        return self.out_head(self.final_norm(self.blocks(self.drop(x))))

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None):
    """temperature=0 -> greedy (P0-03). >0 -> sample (P0-04)."""
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(idx[:, -context_size:])[:, -1, :]
        if top_k is not None:
            kth = torch.topk(logits, top_k).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float('-inf'))
        if temperature > 0:
            next_id = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
        else:
            next_id = logits.argmax(-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
