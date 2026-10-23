import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# P0-02 · Attention from scratch
**Build Your Own AI · Phase 0, step 2.** CPU or Colab, about 25 minutes.

In P0-01 every token became a vector of shape `(batch, T, d_model)`. But each vector only knows *its own* token. In "the bank of the river", `bank` needs to look at `river` to mean the right thing.

**Attention** lets every token look at the other tokens and pull in what's relevant. We build it in five steps:
1. attention with no weights (just dot products),
2. scaled dot-product attention with learnable **Q, K, V**,
3. the **causal mask** (a GPT may not peek at the future),
4. dropout,
5. **multi-head** attention: the exact module inside GPT-2, Qwen, minimind and nanochat.""")
code("""import torch, torch.nn as nn, math
torch.manual_seed(123)
print('torch', torch.__version__)""")
md("""## 1. Attention with no weights
Six tokens, each a 3-d vector (pretend these came out of the embedding layer).
For one *query* token, score every token by **dot product** (how aligned are the vectors), turn scores into weights with **softmax** (positive, sum to 1), then take the **weighted average**.""")
code("""words = ['Your', 'journey', 'starts', 'with', 'one', 'step']
x = torch.tensor([[0.43, 0.15, 0.89],
                  [0.55, 0.87, 0.66],
                  [0.57, 0.85, 0.64],
                  [0.22, 0.58, 0.33],
                  [0.77, 0.25, 0.10],
                  [0.05, 0.80, 0.55]])
q = x[1]                       # query = 'journey'
scores = x @ q                 # dot product with every token
weights = torch.softmax(scores, dim=0)
context = weights @ x          # weighted average of all tokens
for w, s, a in zip(words, scores, weights):
    print(f'{w:8s} score {s:.3f}  weight {a:.3f}')
print('sum of weights:', weights.sum().item())
print('context vector for journey:', context)""")
md("Same thing for **all** tokens at once is one matrix multiply: `scores = x @ x.T` is `(T, T)`, row *i* is how much token *i* attends to each token.")
code("""A = torch.softmax(x @ x.T, dim=-1)   # (6, 6) attention matrix
print(A.round(decimals=2))
print('all rows sum to 1:', torch.allclose(A.sum(-1), torch.ones(6)))
Z = A @ x                              # (6, 3) new vectors, each a mix of the sentence
print(Z)""")
md("""## 2. Learnable Q, K, V (scaled dot-product attention)
Raw dot products only compare tokens as they are. A model wants to *learn* what to look for. So each token is projected three ways:
- **Query** `q = x W_q`: what am I looking for?
- **Key** `k = x W_k`: what do I contain?
- **Value** `v = x W_v`: what do I hand over if picked?

$$\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V$$

Why divide by $\\sqrt{d_k}$? Dot products of long vectors get big, softmax then saturates to one-hot and gradients vanish. The demo below shows it.""")
code("""d_in, d_out = 3, 2
W_q = nn.Parameter(torch.rand(d_in, d_out))
W_k = nn.Parameter(torch.rand(d_in, d_out))
W_v = nn.Parameter(torch.rand(d_in, d_out))
Q, K, V = x @ W_q, x @ W_k, x @ W_v
print('Q', tuple(Q.shape), 'K', tuple(K.shape), 'V', tuple(V.shape))
scores = Q @ K.T
A = torch.softmax(scores / math.sqrt(d_out), dim=-1)
print('attention:', tuple(A.shape)); print(A.round(decimals=3))
print('output:', tuple((A @ V).shape))""")
code("""# Why scale? Dot products of random 512-d vectors:
d = 512
a = torch.randn(8, d); b = torch.randn(8, d)
s = (a @ b.T)[0]
print('raw score std     :', s.std().item())
print('softmax raw max   :', torch.softmax(s, -1).max().item(), '<- nearly one-hot')
print('softmax scaled max:', torch.softmax(s / math.sqrt(d), -1).max().item(), '<- spread out, learnable')""")
code("""class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=False)   # nn.Linear = better-initialised matmul
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        A = torch.softmax(Q @ K.transpose(-2, -1) / K.shape[-1]**0.5, dim=-1)
        return A @ V
torch.manual_seed(789)
print(SelfAttention(3, 2)(x))""")
md("""## 3. Causal mask: no peeking
A GPT is trained to predict token *t+1* from tokens *≤ t*. If position 2 could see position 5, it would just copy the answer.
So we set every score **above the diagonal** to `-inf` before softmax; `exp(-inf) = 0`, so the future gets exactly zero weight.""")
code("""T = x.shape[0]
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
print(mask.int())
torch.manual_seed(789)
sa = SelfAttention(3, 2)
Q, K = sa.W_q(x), sa.W_k(x)
scores = (Q @ K.T).masked_fill(mask, float('-inf'))
A = torch.softmax(scores / K.shape[-1]**0.5, dim=-1)
print(A.round(decimals=2))   # lower-triangular, rows still sum to 1""")
md("Row 0 (`Your`) can only see itself (weight 1.0). Row 5 (`step`) sees the whole sentence. This one line is what makes a model **autoregressive**.")
md("""## 4. Dropout
During training, randomly zero some attention weights (and scale the rest up) so the model can't lean on one connection. Off at inference (`model.eval()`).""")
code("""torch.manual_seed(123)
drop = nn.Dropout(0.5)
print(drop(A).round(decimals=2))""")
md("""## 5. Multi-head attention
One head learns one kind of relationship. **Multiple heads** run in parallel on slices of the vector (one might track grammar, another names, another position) and their outputs are concatenated, then mixed by an output projection.

Trick: don't loop over heads. Project once to `d_out`, then **reshape** `(B, T, d_out)` → `(B, heads, T, head_dim)` and let batched matmul do all heads at once.""")
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
        split = lambda t: t.view(B, T, self.h, self.hd).transpose(1, 2)   # (B, h, T, hd)
        Q, K, V = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        scores = Q @ K.transpose(-2, -1) / self.hd**0.5                   # (B, h, T, T)
        scores = scores.masked_fill(self.mask[:T, :T], float('-inf'))
        A = self.dropout(torch.softmax(scores, dim=-1))
        out = (A @ V).transpose(1, 2).contiguous().view(B, T, self.h * self.hd)  # concat heads
        return self.out_proj(out)

torch.manual_seed(123)
batch = torch.stack([x, x])             # (2, 6, 3)
mha = MultiHeadAttention(d_in=3, d_out=4, context_length=6, num_heads=2)
print('in', tuple(batch.shape), '-> out', tuple(mha(batch).shape))""")
md("### Real size: GPT-2 small\n12 heads, d_model 768, context 1024. Same class, just bigger numbers.")
code("""gpt2_mha = MultiHeadAttention(768, 768, 1024, num_heads=12)
n = sum(p.numel() for p in gpt2_mha.parameters())
print(f'one attention block: {n/1e6:.2f}M params (x 12 layers = {12*n/1e6:.1f}M)')
xb = torch.randn(2, 1024, 768)
print('output', tuple(gpt2_mha(xb).shape))""")
code("""# Sanity check against PyTorch's built-in fused kernel (what you'd use in production)
import torch.nn.functional as F
torch.manual_seed(0)
B, h, T, hd = 2, 4, 16, 8
Q, K, V = torch.randn(B, h, T, hd), torch.randn(B, h, T, hd), torch.randn(B, h, T, hd)
mask = torch.triu(torch.ones(T, T), 1).bool()
ours = torch.softmax((Q @ K.transpose(-2, -1) / hd**0.5).masked_fill(mask, float('-inf')), -1) @ V
theirs = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print('matches F.scaled_dot_product_attention:', torch.allclose(ours, theirs, atol=1e-5))""")
md("""## Your turn
1. In section 1, change the query to `x[4]` ('one'). Which word does it attend to most? Why (look at the vectors)?
2. Remove the `/ sqrt(d)` in the `SelfAttention` class and use d_out=512. Print `A.max()`. What happened?
3. Delete the mask line in `MultiHeadAttention`. Why would training then look great but generation fail?
4. Count: for context T=1024, the attention matrix per head has how many entries? What happens at T=128k? (This is why long context is expensive.)

**Next: P0-03 GPT block**: LayerNorm, feed-forward (GELU), residual connections, stack N blocks, and generate text (badly, untrained) from your own model.""")
nb.cells=C; nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nbf.write(nb,'P0-02-attention.ipynb')
