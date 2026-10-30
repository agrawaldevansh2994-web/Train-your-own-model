import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# P0-04 · Pretraining: teach your GPT to write
**Build Your Own AI · Phase 0, step 4.** CPU about 5 min, Colab GPU about 1 min.

Runs on **Colab or Kaggle** (on Kaggle, turn on Internet in the notebook settings; it needs a phone-verified account).

Until now the model was a random brain (loss 11, gibberish). **Pretraining** is just this loop, repeated:
1. show the model text, ask it to predict every next token,
2. measure how wrong it was (**cross-entropy loss**),
3. nudge every weight a little in the direction that reduces the loss (**backprop + AdamW**).

Everything ChatGPT knows started with this exact loop, on trillions of tokens instead of one short story.

**How the notebooks fit together:** P0-01 to P0-03 each built one piece. That code now lives in one file, `src/mygpt.py`, in this repo. This notebook downloads it and imports the model, so it runs on its own; you don't need the earlier notebooks open. Each Colab tab is a separate machine with separate memory; the only thing that carries forward is what you **save** (the checkpoint at the end).""")
code("""import os, urllib.request
REPO = 'https://raw.githubusercontent.com/agrawaldevansh2994-web/Train-your-own-model/main/'
for path in ['src/mygpt.py']:
    if not os.path.exists(os.path.basename(path)) and not os.path.exists('../' + path):
        urllib.request.urlretrieve(REPO + path, os.path.basename(path))
import sys; sys.path += ['.', '../src']
import subprocess
try:
    import tiktoken, trackio
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tiktoken', 'trackio'])
    import tiktoken, trackio
import torch, torch.nn as nn, torch.nn.functional as F, math, time
from mygpt import GPTModel, generate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch', torch.__version__, '| device:', device)""")
md("## 1. Data: the same story from P0-01, split 90/10")
code("""url = 'https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt'
if not os.path.exists('the-verdict.txt'):
    urllib.request.urlretrieve(url, 'the-verdict.txt')
text = open('the-verdict.txt', encoding='utf-8').read()
tok = tiktoken.get_encoding('gpt2')
ids = tok.encode(text)
split = int(0.9 * len(ids))
train_ids, val_ids = ids[:split], ids[split:]
print(f'{len(text):,} chars -> {len(ids):,} tokens | train {len(train_ids):,} | val {len(val_ids):,}')""")
md("""Sliding-window dataset from P0-01: input = a chunk of tokens, target = the same chunk shifted one to the right. Every position is its own prediction task.

**Validation** text is never trained on. If train loss falls but val loss rises, the model is memorising, not learning.""")
code("""class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, ids, ctx, stride):
        self.x, self.y = [], []
        for i in range(0, len(ids) - ctx, stride):
            self.x.append(torch.tensor(ids[i:i + ctx]))
            self.y.append(torch.tensor(ids[i + 1:i + ctx + 1]))
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

CFG = {   # shrunk GPT: same architecture, laptop-sized
    'vocab_size': 50257, 'context_length': 128, 'emb_dim': 256,
    'n_heads': 4, 'n_layers': 4, 'drop_rate': 0.1, 'qkv_bias': False,
}
torch.manual_seed(123)
train_dl = torch.utils.data.DataLoader(TokenDataset(train_ids, CFG['context_length'], CFG['context_length']), batch_size=2, shuffle=True, drop_last=True)
val_dl = torch.utils.data.DataLoader(TokenDataset(val_ids, CFG['context_length'], CFG['context_length']), batch_size=2)
x, y = next(iter(train_dl))
print('batches: train', len(train_dl), '| val', len(val_dl), '| batch shape', tuple(x.shape))
print('input :', repr(tok.decode(x[0, :10].tolist())))
print('target:', repr(tok.decode(y[0, :10].tolist())))""")
md("""## 2. The loss
The model outputs logits `(B, T, 50257)`. Softmax turns each row into probabilities; **cross-entropy** is the average of `-log(p[correct token])`.
- Perfect guess (p=1) → loss 0.
- Uniform guess (p=1/50257) → loss ln(50257) ≈ 10.82.

**Perplexity** = exp(loss) ≈ "how many tokens the model is torn between".""")
code("""torch.manual_seed(123)
model = GPTModel(CFG, tie_weights=True).to(device)   # weight tying from P0-03; small std-0.02 init (see mygpt.py)
print(f'params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

def batch_loss(xb, yb):
    logits = model(xb.to(device))
    return F.cross_entropy(logits.flatten(0, 1), yb.to(device).flatten())

@torch.no_grad()
def avg_loss(dl, max_batches=None):
    model.eval(); tot, n = 0.0, 0
    for i, (xb, yb) in enumerate(dl):
        if max_batches and i >= max_batches: break
        tot += batch_loss(xb, yb).item(); n += 1
    model.train(); return tot / n

l = avg_loss(train_dl)
print(f'loss before training: {l:.3f}  (perplexity {math.exp(l):,.0f}; uniform would be {math.log(50257):.2f})')""")
md("""## 3. The training loop
`loss.backward()` computes, for every one of the millions of weights, which direction would lower the loss (the gradient). **AdamW** takes a small step that way, with per-weight adaptive step sizes and weight decay (a gentle pull toward zero that fights overfitting).""")
code("""def sample(prompt='Every effort moves you', n=25, **kw):
    model.eval()
    out = generate(model, torch.tensor([tok.encode(prompt)], device=device), n, CFG['context_length'], **kw)
    model.train()
    return tok.decode(out[0].tolist()).replace('\\n', ' ')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
EPOCHS = 12
history = {'step': [], 'train': [], 'val': []}
# Trackio: free, local experiment tracker (same API as Weights & Biases). Logs every epoch to a small local database.
trackio.init(project='train-your-own-model', name='p0-04-tiny-gpt', config={**CFG, 'lr': 1e-3, 'epochs': EPOCHS, 'batch_size': 2})
step, t0 = 0, time.time()
for epoch in range(EPOCHS):
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = batch_loss(xb, yb)
        loss.backward()                       # gradients for every weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                      # nudge the weights
        step += 1
    tr, va = avg_loss(train_dl), avg_loss(val_dl)
    history['step'].append(step); history['train'].append(tr); history['val'].append(va)
    trackio.log({'train_loss': tr, 'val_loss': va, 'epoch': epoch + 1}, step=step)
    print(f'epoch {epoch+1:2d} | step {step:3d} | train {tr:.3f} | val {va:.3f} | {time.time()-t0:4.0f}s')
    if epoch in (0, 3, EPOCHS - 1):
        print('   ->', repr(sample()))
trackio.finish()""")
md("""**Trackio dashboard (optional).** Run `trackio.show()` in a new cell to open the interactive loss chart. Every run you do (different lr, more epochs) appears as a separate line, so you can compare experiments. That's how you'll track the longer Phase 1 runs.""")
md("## 4. Read the curve")
code("""import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3.5))
plt.plot(history['step'], history['train'], 'o-', label='train')
plt.plot(history['step'], history['val'], 's--', label='validation')
plt.xlabel('step'); plt.ylabel('cross-entropy loss'); plt.legend(); plt.title('Your first pretraining run')
plt.grid(alpha=.3); plt.tight_layout(); plt.show()
best = min(range(len(history['val'])), key=lambda i: history['val'][i])
print(f'best val loss {history["val"][best]:.3f} at epoch {best+1}; final train {history["train"][-1]:.3f}')""")
md("""What you should see: train loss keeps dropping (to about 5 after 12 epochs), while val loss bottoms out around 6.4 after ~8 epochs and then **flattens or creeps back up**. That widening gap is **overfitting**: with only ~4,600 training tokens, the model memorises the story instead of learning English. Real pretraining fixes this with more data, not more epochs: nanochat uses billions of tokens. Same loop, more text.""")
md("""## 5. Decoding: temperature and top-k
Greedy (always the top token) is deterministic and repetitive. Instead we **sample** from the probabilities, with two dials:
- **Temperature** divides logits before softmax. T<1 sharpens (safer), T>1 flattens (wilder).
- **Top-k** keeps only the k likeliest tokens, so a weird token can't sneak in.""")
code("""model.eval()
with torch.no_grad():
    logits = model(torch.tensor([tok.encode('Every effort moves')], device=device))[0, -1]
top = torch.topk(logits, 5)
for T in (0.5, 1.0, 2.0):
    p = torch.softmax(logits / T, -1)
    row = ', '.join(f'{tok.decode([i])!r}:{p[i]:.2f}' for i in top.indices.tolist())
    print(f'T={T}: {row}')""")
code("""torch.manual_seed(42)
print('greedy        :', repr(sample()))
print('T=0.7, top-k 20:', repr(sample(temperature=0.7, top_k=20)))
print('T=1.5, top-k 50:', repr(sample(temperature=1.5, top_k=50)))""")
md("""## 6. Save a checkpoint
A Colab session's disk is wiped when it ends. Save weights + optimizer state + config so you (or P0-05) can resume. In Colab, mount Drive to keep it.""")
code("""ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': CFG, 'history': history}
SAVE = 'p0_04_tiny_gpt.pt'
try:
    from google.colab import drive
    drive.mount('/content/drive'); os.makedirs('/content/drive/MyDrive/train-your-own-model', exist_ok=True)
    SAVE = '/content/drive/MyDrive/train-your-own-model/p0_04_tiny_gpt.pt'
except ImportError:
    if os.path.isdir('/kaggle/working'):   # Kaggle: download it from the notebook's Output tab
        SAVE = '/kaggle/working/p0_04_tiny_gpt.pt'
torch.save(ckpt, SAVE)
print(f'saved {SAVE} ({os.path.getsize(SAVE)/1e6:.0f} MB; ~2/3 of that is AdamW state, weights alone are ~65 MB)')

# reload into a fresh model and check it gives the same answer
model.eval()   # dropout off, or the two outputs would differ randomly\nck = torch.load(SAVE, map_location=device)
m2 = GPTModel(ck['config'], tie_weights=True).to(device); m2.load_state_dict(ck['model']); m2.eval()
xb = torch.tensor([tok.encode('Every effort moves')], device=device)
with torch.no_grad():
    print('reloaded model identical:', torch.allclose(model(xb), m2(xb), atol=1e-5))""")
md("""## Your turn
1. Train for 30 epochs. What happens to val loss? Write down the epoch you'd stop at (that's **early stopping**).
2. Set `drop_rate=0.0` and retrain. Does the train/val gap grow or shrink?
3. Try `lr=1e-2` and `lr=1e-4`. Which explodes, which crawls?
4. Paste in a longer text you like (a public-domain book from Project Gutenberg) and retrain. Does val loss get better?
5. Sample with `temperature=0.01` and compare to greedy. Why are they the same?

**Next: P0-05 load real GPT-2 weights** into *your* `GPTModel` and watch your own code write fluent English. That closes Phase 0; then Phase 1 (minimind) trains a full pipeline on GPU.""")
nb.cells=C; nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nbf.write(nb,'P0-04-pretraining.ipynb')
