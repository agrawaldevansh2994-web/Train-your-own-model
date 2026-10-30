import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# P0-03 · The GPT block, then a whole GPT
**Build Your Own AI · Phase 0, step 3.** CPU or Colab, about 30 minutes.

P0-01 turned text into token vectors. P0-02 let tokens look at each other (attention). Now we wrap attention with the parts that make it trainable at depth, stack blocks, and get a full **GPT-2 (124M)** architecture that writes text.

The block, top to bottom:
```
x ─┬─ LayerNorm ─ MultiHeadAttention ─ Dropout ─(+)─┬─ LayerNorm ─ FeedForward(GELU) ─ Dropout ─(+)─> out
   └────────────── shortcut ──────────────────────┘ └──────────────── shortcut ─────────────────┘
```
We build each piece, check what it does with numbers, then assemble.""")
code("""import sys, subprocess
try:
    import tiktoken
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tiktoken'])
    import tiktoken
import torch, torch.nn as nn, math
torch.manual_seed(123)
print('torch', torch.__version__)""")
md("""## 0. Config
These are GPT-2 small's real numbers. Everything below reads from this dict.""")
code("""GPT_CONFIG_124M = {
    'vocab_size': 50257,     # GPT-2 BPE vocab (P0-01)
    'context_length': 1024,  # max tokens the model sees
    'emb_dim': 768,          # d_model
    'n_heads': 12,           # attention heads (P0-02)
    'n_layers': 12,          # transformer blocks stacked
    'drop_rate': 0.1,
    'qkv_bias': False,
}""")
md("Multi-head attention, exactly as you built it in P0-02:")
code("""class MultiHeadAttention(nn.Module):
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
        return self.out_proj((A @ V).transpose(1, 2).contiguous().view(B, T, self.h * self.hd))""")
md("""## 1. LayerNorm: keep activations in a sane range
Deep nets drift: each layer's outputs can grow or shrink until gradients explode or vanish. **LayerNorm** rescales each token's vector to mean 0, variance 1, then lets the model learn a scale (γ) and shift (β):
$$\\text{LN}(x) = \\gamma \\cdot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta$$
It normalises over the **feature** dimension (last axis), per token, so it doesn't depend on batch size.""")
code("""class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))   # gamma
        self.shift = nn.Parameter(torch.zeros(dim))  # beta
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift

x = torch.randn(2, 5) * 4 + 3
ln = LayerNorm(5)
y = ln(x)
r = lambda t, d: [round(v, d) for v in t.tolist()]
print('before: mean', r(x.mean(-1), 2), ' var', r(x.var(-1, unbiased=False), 2))
print('after : mean', r(y.mean(-1), 4), ' var', r(y.var(-1, unbiased=False), 4))
print('matches nn.LayerNorm:', torch.allclose(y, nn.LayerNorm(5)(x), atol=1e-5))""")
md("""## 2. GELU and the feed-forward network
Attention mixes information *between* tokens. The **feed-forward network (FFN)** then thinks about each token *on its own*: expand 768 → 3072, apply a nonlinearity, shrink back to 768. Most of a GPT's parameters live here.

The nonlinearity is **GELU**, a smooth ReLU: small negative inputs leak a little through instead of being cut to exactly zero, which gives smoother gradients.
$$\\text{GELU}(x) \\approx 0.5x\\left(1+\\tanh\\left[\\sqrt{2/\\pi}\\,(x+0.044715x^3)\\right]\\right)$$""")
code("""class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

xs = torch.tensor([-3., -1., -0.5, 0., 0.5, 1., 3.])
print('x    ', xs.tolist())
print('ReLU ', torch.relu(xs).tolist())
print('GELU ', [round(v, 3) for v in GELU()(xs).tolist()])

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['emb_dim']
        self.net = nn.Sequential(nn.Linear(d, 4 * d), GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        return self.net(x)

ffn = FeedForward(GPT_CONFIG_124M)
print('FFN out shape:', tuple(ffn(torch.randn(2, 3, 768)).shape))
print(f'FFN params: {sum(p.numel() for p in ffn.parameters())/1e6:.2f}M (attention was 2.36M)')""")
md("""## 3. Shortcut (residual) connections
Instead of `x = layer(x)`, a GPT does `x = x + layer(x)`. The `+ x` gives gradients a straight highway back to the early layers.
Below: a 5-layer net with tiny-ish weights. Compare the gradient that reaches **layer 0** with and without shortcuts.""")
code("""class DeepNet(nn.Module):
    def __init__(self, sizes, shortcut):
        super().__init__()
        self.shortcut = shortcut
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i+1]), GELU()) for i in range(len(sizes)-1)])
    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = x + out if (self.shortcut and x.shape == out.shape) else out
        return x

def grads(model):
    x = torch.tensor([[1., 0., -1.]])
    loss = nn.MSELoss()(model(x), torch.tensor([[0.]]))
    loss.backward()
    for name, p in model.named_parameters():
        if 'weight' in name:
            print(f'  {name:18s} mean |grad| = {p.grad.abs().mean().item():.6f}')

sizes = [3, 3, 3, 3, 3, 1]
torch.manual_seed(123); print('WITHOUT shortcuts'); grads(DeepNet(sizes, False))
torch.manual_seed(123); print('WITH shortcuts');    grads(DeepNet(sizes, True))""")
md("Without shortcuts the gradient shrinks layer by layer toward the input (the **vanishing gradient** problem). With them, early layers still get a real learning signal. This is why you can stack 12, 48 or 100+ blocks.")
md("""## 4. The transformer block
Pre-LayerNorm (normalise *before* each sublayer, as GPT-2 and every modern LLM do), sublayer, dropout, add the shortcut. Shape in = shape out, so blocks stack like Lego.""")
code("""class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg['emb_dim'], cfg['emb_dim'], cfg['context_length'],
                                      cfg['n_heads'], cfg['drop_rate'], cfg['qkv_bias'])
        self.ff = FeedForward(cfg)
        self.norm1, self.norm2 = LayerNorm(cfg['emb_dim']), LayerNorm(cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])
    def forward(self, x):
        x = x + self.drop(self.att(self.norm1(x)))   # tokens talk to each other
        x = x + self.drop(self.ff(self.norm2(x)))    # each token thinks alone
        return x

torch.manual_seed(123)
blk = TransformerBlock(GPT_CONFIG_124M)
x = torch.rand(2, 4, 768)
print('in ', tuple(x.shape), '-> out', tuple(blk(x).shape))
print(f'one block: {sum(p.numel() for p in blk.parameters())/1e6:.2f}M params')""")
md("""## 5. The whole GPT
token embedding + **position** embedding (attention by itself has no idea of order, so we add a learned vector per position) → dropout → N blocks → final LayerNorm → linear **head** back to 50,257 vocab scores (**logits**).""")
code("""class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(self.drop(x))
        return self.out_head(self.final_norm(x))     # (B, T, vocab)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
tok = tiktoken.get_encoding('gpt2')
batch = torch.stack([torch.tensor(tok.encode('Every effort moves you')),
                     torch.tensor(tok.encode('Every day holds a'))])
logits = model(batch)
print('input', tuple(batch.shape), '-> logits', tuple(logits.shape))""")
md("### Counting parameters: where does \"124M\" come from?")
code("""total = sum(p.numel() for p in model.parameters())
emb = model.tok_emb.weight.numel()
head = model.out_head.weight.numel()
att = sum(p.numel() for b in model.blocks for p in b.att.parameters())
ff = sum(p.numel() for b in model.blocks for p in b.ff.parameters())
print(f'total params           : {total/1e6:.1f}M')
print(f'  token embedding      : {emb/1e6:.1f}M')
print(f'  output head          : {head/1e6:.1f}M')
print(f'  attention (12 blocks): {att/1e6:.1f}M')
print(f'  feed-forward (12)    : {ff/1e6:.1f}M')
print(f'with weight tying (head shares tok_emb, like GPT-2): {(total-head)/1e6:.1f}M')
print(f'memory at float32: {total*4/1024**2:.0f} MB')""")
md("GPT-2 reuses the token-embedding matrix as the output head (**weight tying**), which is how 163M becomes the famous 124M. Note: FFN is ~2x attention; the vocab matrices are a big chunk too. That's why small models like minimind use a much smaller vocab.")
md("""## 6. Generate text (from an untrained brain)
Generation is a loop: run the model, take the logits at the **last** position, pick the most likely token (greedy), append it, repeat. Crop to `context_length` so we never exceed what the position table knows.""")
code("""def generate_greedy(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(idx[:, -context_size:])[:, -1, :]   # (B, vocab) at last position
        next_id = torch.argmax(logits, dim=-1, keepdim=True)   # greedy pick
        idx = torch.cat([idx, next_id], dim=1)
    return idx

model.eval()   # dropout off
start = 'Hello, I am'
ids = torch.tensor(tok.encode(start)).unsqueeze(0)
out = generate_greedy(model, ids, max_new_tokens=8, context_size=GPT_CONFIG_124M['context_length'])
print('token ids:', out[0].tolist())
print('text     :', repr(tok.decode(out[0].tolist())))""")
md("""Gibberish, and that's correct: every weight is random. The architecture is complete; what's missing is **learning**.
At random init each of the 50,257 tokens is about equally likely, so the expected loss is about ln(50257):""")
code("""import torch.nn.functional as F
targets = torch.stack([torch.tensor(tok.encode(' effort moves you forward')), torch.tensor(tok.encode(' day holds a chance'))])
loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
print(f'untrained loss: {loss.item():.2f}   ln(vocab) = {math.log(50257):.2f}')
print(f'perplexity: {loss.exp().item():,.0f}  (model is as unsure as picking among ~that many words)')""")
md("""## Your turn
1. Shrink the model: `n_layers=2, emb_dim=128, n_heads=4, context_length=256`. Parameter count now? Which part dominates, and why?
2. In section 3, change `sizes` to 10 layers of width 3. How small does layer 0's gradient get without shortcuts?
3. Swap `torch.argmax` for `torch.multinomial(torch.softmax(logits, -1), 1)`. Run generation twice. Why does the output change?
4. Remove `pos_emb`. Would "dog bites man" and "man bites dog" look different to the model? (Hint: attention is a weighted *sum*.)

**Next: P0-04 pretraining.** We train this exact architecture (shrunk) on *The Verdict* on CPU, watch the loss fall from ~10.9, add temperature/top-k sampling, and save a checkpoint. Your first model that actually learned something.""")
nb.cells=C; nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nbf.write(nb,'P0-03-gpt-block.ipynb')
