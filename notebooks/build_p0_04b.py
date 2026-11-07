import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# P0-04b · More data: the real cure for overfitting
**Build Your Own AI · Phase 0, optional step after P0-04.** Colab T4 GPU about 10-15 min. On CPU it falls back to a short test run.

In P0-04 the model memorised *The Verdict*: train loss kept falling while validation loss turned up. Dropout and smaller models only slowed that down. Here we test the real fix. We train **the exact same 16M model** three times, changing only **how much text** it learns from:

| run | training text | how often it sees each token |
|---|---|---|
| small | 5k tokens (about the size of *The Verdict*) | hundreds of times |
| medium | 50k tokens | dozens of times |
| full | about 300k tokens | about a dozen times |

Same model, same number of training steps, same validation text. Only the data changes.

**Data:** Tiny Shakespeare, the same text Andrej Karpathy uses in "Let's build GPT". He tokenizes it by character (65 symbols); we use the GPT-2 tokenizer from P0-01, so our losses are not comparable with his.

**Before you start (Colab):** Runtime → Change runtime type → **T4 GPU**.""")
code("""import os, sys, subprocess, urllib.request, time, math
REPO = 'https://raw.githubusercontent.com/agrawaldevansh2994-web/Train-your-own-model/main/'
if not os.path.exists('mygpt.py') and not os.path.exists('../src/mygpt.py'):
    urllib.request.urlretrieve(REPO + 'src/mygpt.py', 'mygpt.py')
sys.path += ['.', '../src']
try:
    import tiktoken, trackio
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tiktoken', 'trackio'])
    import tiktoken, trackio
import torch, torch.nn.functional as F
from mygpt import GPTModel, generate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = device == 'cuda'
print('torch', torch.__version__, '| device:', device, '|', torch.cuda.get_device_name(0) if GPU else 'no GPU: Runtime > Change runtime type > T4 GPU')""")
md("## 1. Data: Tiny Shakespeare, 90% to train on, 10% held out")
code("""url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
if not os.path.exists('tinyshakespeare.txt'):
    urllib.request.urlretrieve(url, 'tinyshakespeare.txt')
text = open('tinyshakespeare.txt', encoding='utf-8').read()
tok = tiktoken.get_encoding('gpt2')
ids = torch.tensor(tok.encode(text))
split = int(0.9 * len(ids))
train_all, val_ids = ids[:split], ids[split:]
print(f'{len(text):,} chars -> {len(ids):,} tokens | train {len(train_all):,} | val {len(val_ids):,}')
print(f'that is {len(train_all) / 4116:.0f}x the training text of P0-04\\n')
print(text[:300])""")
md("""All three runs are scored on the **same** held-out 10%, so their validation losses are directly comparable. The small and medium runs train on the first 5k and 50k tokens of the training part.""")
code("""CFG = {   # identical to P0-04
    'vocab_size': 50257, 'context_length': 128, 'emb_dim': 256,
    'n_heads': 4, 'n_layers': 4, 'drop_rate': 0.1, 'qkv_bias': False,
}
T = CFG['context_length']
STEPS = 2000 if GPU else 150        # same training budget for every run
BATCH = 16 if GPU else 4
EVAL_EVERY = STEPS // 20
LR = 1e-3
SIZES = {'small (5k tokens)': 5_000, 'medium (50k tokens)': 50_000, f'full ({len(train_all) // 1000}k tokens)': len(train_all)}
if not GPU: print('CPU quick test: 150 steps per run. The real experiment needs a GPU.')
for name, n in SIZES.items():
    print(f'{name:22s} each token seen about {STEPS * BATCH * T / n:6.1f} times')

def get_batch(data):
    # random windows, like Andrej's get_batch (P0-04 walked through fixed chunks instead)
    i = torch.randint(0, len(data) - T - 1, (BATCH,))
    x = torch.stack([data[j:j + T] for j in i]); y = torch.stack([data[j + 1:j + T + 1] for j in i])
    return x.to(device), y.to(device)

def windows(data, cap):
    n = min((len(data) - 1) // T, cap)
    return data[:n * T].view(n, T).to(device), data[1:n * T + 1].view(n, T).to(device)

@torch.no_grad()
def loss_on(model, X, Y, bs=32):
    model.eval(); tot = 0.0
    for k in range(0, len(X), bs):
        with torch.autocast(device, dtype=torch.float16, enabled=GPU):
            logits = model(X[k:k + bs])
        tot += F.cross_entropy(logits.float().flatten(0, 1), Y[k:k + bs].flatten(), reduction='sum').item()
    model.train(); return tot / Y.numel()

Xva, Yva = windows(val_ids, 256 if GPU else 24)
print(f'validation: {Xva.numel():,} tokens, the same for every run')""")
md("""## 2. Train the same model three times
Each run gets the same number of steps. The only difference is the pool of text the batches are drawn from. On a T4 each run takes a few minutes. Every run is also logged to Trackio, so `trackio.show()` puts all three on one dashboard.""")
code("""def train(name, data):
    torch.manual_seed(123)
    model = GPTModel(CFG, tie_weights=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    scaler = torch.amp.GradScaler(device, enabled=GPU)   # fp16 on the T4 for speed
    Xtr, Ytr = windows(data, 256 if GPU else 24)
    h = {'step': [], 'train': [], 'val': []}
    trackio.init(project='train-your-own-model', name=f'p0-04b-{len(data) // 1000}k', embed=False, config={**CFG, 'train_tokens': len(data), 'steps': STEPS, 'batch': BATCH, 'lr': LR})
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = get_batch(data)
        with torch.autocast(device, dtype=torch.float16, enabled=GPU):
            logits = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        opt.zero_grad(); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % EVAL_EVERY == 0:
            tr, va = loss_on(model, Xtr, Ytr), loss_on(model, Xva, Yva)
            h['step'].append(step); h['train'].append(tr); h['val'].append(va)
            trackio.log({'train_loss': tr, 'val_loss': va}, step=step)
            if step % (EVAL_EVERY * 5) == 0:
                print(f'  step {step:5d} | train {tr:.3f} | val {va:.3f} | gap {va - tr:5.2f} | {time.time() - t0:4.0f}s')
    trackio.finish()
    return model, h

results = {}
for name, n in SIZES.items():
    print(f'--- {name}')
    results[name] = train(name, train_all[:n])""")
md("## 3. Compare the three runs")
code("""import matplotlib.pyplot as plt
colors = ['tab:red', 'tab:orange', 'tab:green']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={'width_ratios': [1.6, 1]})
for c, (name, (m, h)) in zip(colors, results.items()):
    a1.plot(h['step'], h['val'], '-', color=c, lw=2.5, label=f'{name}: validation')
    a1.plot(h['step'], h['train'], ':', color=c, lw=2, label=f'{name}: train')
a1.axhline(math.log(50257), color='gray', ls='--', lw=1); a1.text(h['step'][0], math.log(50257) + .1, 'random guessing 10.82', color='gray')
a1.set_xlabel('training step', fontsize=12); a1.set_ylabel('cross-entropy loss', fontsize=12); a1.set_title('Same model, same steps, different amounts of text', fontsize=13)
a1.legend(fontsize=9, ncol=2, loc='upper right', bbox_to_anchor=(1, .93)); a1.grid(alpha=.3)
names = list(results); best = [min(results[n][1]['val']) for n in names]; final = [results[n][1]['val'][-1] for n in names]; gaps = [results[n][1]['val'][-1] - results[n][1]['train'][-1] for n in names]
x = range(len(names))
a2.bar([i - .2 for i in x], best, .4, color=colors, alpha=.55, label='best validation loss')
a2.bar([i + .2 for i in x], final, .4, color=colors, label='final validation loss')
for i in x: a2.text(i + .2, final[i] + .08, f'gap {gaps[i]:.2f}', ha='center', fontsize=10)
a2.set_xticks(list(x)); a2.set_xticklabels([n.split(' (')[0] + '\\n' + n.split('(')[1].rstrip(')') for n in names]); a2.set_ylim(0, max(final) + 1)
a2.set_ylabel('validation loss (lower is better)', fontsize=12); a2.set_title('More data, lower validation loss', fontsize=13); a2.legend(fontsize=9, loc='lower right'); a2.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig('p0_04b_compare.png', dpi=150); plt.show()
for n, b, f, g in zip(names, best, final, gaps):
    print(f'{n:22s} best val {b:.2f} | final val {f:.2f} | final gap {g:.2f} (unseen text {math.exp(g):.1f}x more surprising)')""")
md("""**How to read it.** All three start the same way. Then:
- **small**: train loss dives toward zero while validation turns upward early. It has memorised its 5k tokens, exactly like P0-04.
- **medium**: overfits too, but later and less.
- **full**: train and validation stay close and validation ends lowest. With about 300k tokens it cannot memorise everything in the same number of steps, so it is forced to learn patterns that also work on unseen text.

Nothing about the model changed. This is why real pretraining runs use billions of tokens, and why the fix for P0-04 was never more epochs.""")
md("## 4. What each model writes")
code("""prompt = torch.tensor([tok.encode('ROMEO:')], device=device)
for name, (m, h) in results.items():
    m.eval(); torch.manual_seed(42)
    out = generate(m, prompt, 60, T, temperature=0.8, top_k=40)
    print(f'--- {name}\\n{tok.decode(out[0].tolist())}\\n')""")
md("""Small data tends to produce fragments of its own 5k tokens, or loops. The full-data model should look more like Shakespeare in shape: speaker names, line breaks, archaic words. It is still a 16M model trained for minutes, so do not expect sense.""")
md("""## 5. Save the full-data model
Same format as P0-04, so you can load it later with `GPTModel(ck['config'], tie_weights=True)`.""")
code("""m_full, h_full = results[names[-1]]
ckpt = {'model': m_full.state_dict(), 'config': CFG, 'history': h_full, 'train_tokens': len(train_all)}
SAVE = 'p0_04b_shakespeare.pt'
try:
    from google.colab import drive
    drive.mount('/content/drive'); os.makedirs('/content/drive/MyDrive/train-your-own-model', exist_ok=True)
    SAVE = '/content/drive/MyDrive/train-your-own-model/p0_04b_shakespeare.pt'
except ImportError:
    if os.path.isdir('/kaggle/working'): SAVE = '/kaggle/working/p0_04b_shakespeare.pt'
torch.save(ckpt, SAVE)
print(f'saved {SAVE} ({os.path.getsize(SAVE) / 1e6:.0f} MB)')""")
md("""## Your turn
1. Set `STEPS = 6000` and rerun. Does the full-data model start to overfit too? At what point does even 300k tokens become "too little"?
2. Set `'drop_rate': 0.0` in `CFG`. Which run suffers most without dropout?
3. Make the model bigger (`'emb_dim': 384, 'n_layers': 6, 'n_heads': 6`). With the full data, does the bigger model win on validation? With the small data?
4. Andrej's character-level model reaches a validation loss near 1.5. Why can't you compare that with your number? (Hint: what is the loss of random guessing with 65 symbols, versus 50,257 tokens?)

**Next: P0-05** loads OpenAI's GPT-2, which saw about 40 GB of text, into this same architecture.""")
nb.cells=C
nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nb.metadata['accelerator']='GPU'
nb.metadata['colab']={'provenance':[],'gpuType':'T4'}
nbf.write(nb,'P0-04b-more-data.ipynb')
