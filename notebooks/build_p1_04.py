import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# P1-04 · Pretrain your own 16M model on TinyStories
**Build Your Own AI · Phase 1, step 4.** GPU (Kaggle or Colab T4) for the real run. On a laptop CPU it runs a short test (about 10 minutes) to check that everything works and the loss falls.

Everything from P1-01 to P1-03 comes together here: our 4,096-token tokenizer, the modern block from `mymind.py`, and 552M tokens of stories on disk. New in this notebook, compared with P0-04:

| | P0-04 (Phase 0) | P1-04 |
|---|---|---|
| data | 4k tokens, many epochs | 552M tokens, less than one pass: nothing to memorise |
| learning rate | constant | **warmup then cosine decay** |
| precision | fp32 | **mixed precision**: fp16 (T4) or bf16 (newer GPUs) with a gradient scaler |
| progress | loss | loss, **bits per byte**, and a sample story at every eval |
| long runs | finished in minutes | **checkpoint every 20 minutes and resume**, because Kaggle and Colab sessions end |

**Before you start:**
- **Kaggle:** Settings → Accelerator → **GPU T4 x2** (we use one of the two), and Internet **on**.
- **Colab:** Runtime → Change runtime type → **T4 GPU**. The checkpoint goes to your Google Drive.
- The first run downloads and tokenizes the data (2.2 GB, see P1-03). On a fresh Kaggle or Colab machine that takes roughly 15-30 minutes (a guess; P1-03 took 11 minutes to tokenize on the laptop).""")
code("""import os, sys, subprocess, urllib.request, math, time, json
REPO = 'https://raw.githubusercontent.com/agrawaldevansh2994-web/Train-your-own-model/main/'
for f in ['mymind.py', 'tsdata.py']:
    if not os.path.exists(f) and not os.path.exists('../src/' + f):
        urllib.request.urlretrieve(REPO + 'src/' + f, f)
sys.path += ['.', '../src']
for pkg in ['tokenizers', 'trackio']:
    try: __import__(pkg)
    except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg])
import numpy as np, torch, torch.nn.functional as F, trackio
from mymind import MindModel, generate
import tsdata
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = device == 'cuda'
AMP = torch.bfloat16 if GPU and torch.cuda.is_bf16_supported() else torch.float16
print('torch', torch.__version__, '| device:', torch.cuda.get_device_name(0) if GPU else 'CPU (short test run only)', '| mixed precision:', str(AMP).split('.')[-1] if GPU else 'off')""")
md("""## 1. Data
`tsdata.prepare` is the P1-03 pipeline in one function: download, tokenize once, write `uint16` files. It skips any step whose output already exists, so a second run starts in seconds. On CPU we only take the first 50 MB of the training file (enough for a short test), unless P1-03's full file is already in `data/`.""")
code("""TRAIN_MB = None if GPU or os.path.exists('data/ts_train_full.bin') else 50
train_bin, val_bin, meta = tsdata.prepare('data', train_mb=TRAIN_MB)
tok = tsdata.load_tokenizer('data')
EOT = tok.token_to_id('<|endoftext|>')
print(json.dumps(meta, indent=1))
train_data, val_data = tsdata.Batches(train_bin, seed=0), tsdata.Batches(val_bin, seed=1)""")
md("""## 2. Model and training settings
The model is the one we sized in P1-02: 15.9M parameters, 10% of them in the embedding.
- `rope_theta` 10,000: the classic value for short contexts. minimind uses 1,000,000 because it reads up to 32k tokens; our stories fit in 512.
- **AdamW** with betas (0.9, 0.95), as in GPT-3, Llama and minimind. **Weight decay** only on the weight matrices; norms and the embedding are left alone (the usual rule).
- **Batch** 32 windows x 512 tokens = 16,384 tokens per step.
- **How long:** `TRAIN_TOKENS` = 300M, close to the Chinchilla 20 tokens per parameter (318M) from P1-03. That is about 18,300 steps.""")
code("""MODEL_CFG = {'vocab_size': meta['vocab_size'], 'context_length': 512, 'emb_dim': 384, 'n_layers': 8, 'n_heads': 6, 'n_kv_heads': 2,
             'ffn_dim': None, 'rope_theta': 10_000.0, 'norm_eps': 1e-6, 'drop_rate': 0.0}
if GPU:
    TRAIN = {'batch': 32, 'seq': 512, 'train_tokens': 300_000_000, 'lr': 1e-3, 'min_lr': 1e-4, 'warmup': 500,
             'weight_decay': 0.1, 'grad_clip': 1.0, 'eval_every': 500, 'eval_windows': 100, 'ckpt_minutes': 20, 'max_hours': 11}
else:   # CPU: the same model, small batches, a few hundred steps
    TRAIN = {'batch': 8, 'seq': 256, 'train_tokens': 300 * 8 * 256, 'lr': 1e-3, 'min_lr': 1e-4, 'warmup': 30,
             'weight_decay': 0.1, 'grad_clip': 1.0, 'eval_every': 50, 'eval_windows': 16, 'ckpt_minutes': 20, 'max_hours': 1}
TOKENS_PER_STEP = TRAIN['batch'] * TRAIN['seq']
STEPS = TRAIN['train_tokens'] // TOKENS_PER_STEP

def make_model():
    torch.manual_seed(1337)
    return MindModel(MODEL_CFG).to(device)
def make_optimizer(model):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and 'tok_emb' not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or 'tok_emb' in n]
    return torch.optim.AdamW([{'params': decay, 'weight_decay': TRAIN['weight_decay']}, {'params': no_decay, 'weight_decay': 0.0}],
                             lr=TRAIN['lr'], betas=(0.9, 0.95), fused=GPU)
model = make_model()
n_params = sum(p.numel() for p in model.parameters())
print(f'{n_params / 1e6:.1f}M parameters | {TOKENS_PER_STEP:,} tokens per step | {STEPS:,} steps | {TRAIN["train_tokens"] / 1e6:.0f}M tokens = {TRAIN["train_tokens"] / meta["train_tokens"]:.2f} of the training file')""")
md("""## 3. The learning-rate schedule
P0-04 used one fixed learning rate. Real runs change it over time:
- **Warmup:** start near 0 and rise linearly. At step 0 the weights are random and the gradients are large and noisy; big steps then can push the model somewhere bad that it never recovers from.
- **Cosine decay:** after warmup, glide down along half a cosine to `min_lr`. Big steps early cover ground fast; small steps late settle into the bottom of the valley.""")
code("""def lr_at(step):
    if step < TRAIN['warmup']:
        return TRAIN['lr'] * (step + 1) / TRAIN['warmup']
    progress = min(1.0, (step - TRAIN['warmup']) / max(1, STEPS - TRAIN['warmup']))
    return TRAIN['min_lr'] + 0.5 * (TRAIN['lr'] - TRAIN['min_lr']) * (1 + math.cos(math.pi * progress))
import matplotlib.pyplot as plt
s = np.arange(STEPS)
plt.figure(figsize=(8, 3)); plt.plot(s, [lr_at(i) for i in s]); plt.xlabel('step'); plt.ylabel('learning rate'); plt.title('warmup, then cosine decay'); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
for i in [0, TRAIN['warmup'] // 2, TRAIN['warmup'], STEPS // 2, STEPS - 1]:
    print(f'step {i:6d}: lr {lr_at(i):.2e}')""")
md("""## 4. One training step, and how fast it runs
**Mixed precision.** Inside `torch.autocast`, matrix multiplies run in 16-bit floats, which is 2 to 8 times faster on a GPU and halves the memory. The weights themselves stay in 32-bit. fp16 can only represent numbers down to about 6e-8, and small gradients would round to zero, so the **GradScaler** multiplies the loss by a big number before `backward()` and divides the gradients back before the optimizer step. If that ever overflows to infinity, it skips the step and lowers the factor. bf16 (A100, L4, newer) has fp32's range, so it needs no scaler; the T4 has no bf16.

Before committing hours, we **measure**: run 25 real steps on a throwaway copy and project the time for the whole run.""")
code("""scaler = torch.amp.GradScaler('cuda', enabled=GPU and AMP == torch.float16)
def train_step(model, opt, step):
    for g in opt.param_groups: g['lr'] = lr_at(step)
    x, y = train_data.get(TRAIN['batch'], TRAIN['seq'], device)
    with torch.autocast(device, dtype=AMP, enabled=GPU):
        logits = model(x)
    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN['grad_clip'])
    scaler.step(opt); scaler.update()
    return loss.item(), gnorm.item()

bench_model = make_model(); bench_opt = make_optimizer(bench_model)
n_warm, n_timed = (5, 20) if GPU else (2, 4)    # the first steps are slow (memory allocation, kernel choice): skip them
for i in range(n_warm + n_timed):
    if i == n_warm:
        if GPU: torch.cuda.synchronize()
        t0 = time.time()
    loss, _ = train_step(bench_model, bench_opt, i)
if GPU: torch.cuda.synchronize()
tok_per_s = n_timed * TOKENS_PER_STEP / (time.time() - t0)
print(f'MEASURED on this machine: {tok_per_s:,.0f} tokens/s')
print(f'projected time for {TRAIN["train_tokens"] / 1e6:.0f}M tokens: {TRAIN["train_tokens"] / tok_per_s / 3600:.2f} h (plus evals, about 5% more)')
if GPU: print(f'GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB of {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
del bench_model, bench_opt""")
md("""If the projection is longer than your session (Kaggle 12 h, Colab varies), that is fine: the loop below saves a checkpoint every 20 minutes and stops cleanly at `max_hours`. Run the notebook again in a new session and it **resumes** from the checkpoint. To make the run shorter instead, lower `TRAIN['train_tokens']` and rerun from section 2.

## 5. Evaluation: loss, bits per byte, and a sample
- **Validation loss** on the same 100 windows every time (the first 51k tokens of the validation file), so points are comparable.
- **Bits per byte** = loss / ln 2 / (bytes per token). Loss in nats per token depends on the tokenizer; bits per byte asks "how many bits does the model need per byte of text", which any model can be scored on. P1-05 uses it to compare with TinyStories-33M.
- A **sample story** from the same prompt and seed, to watch the writing improve.""")
code("""Xv, Yv = val_data.fixed(TRAIN['eval_windows'], TRAIN['seq'])
BPT = meta['bytes_per_token_val']
@torch.no_grad()
def evaluate(model):
    model.eval(); tot = 0.0
    for k in range(0, len(Xv), 16):
        x, y = Xv[k:k + 16].to(device), Yv[k:k + 16].to(device)
        with torch.autocast(device, dtype=AMP, enabled=GPU):
            logits = model(x)
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction='sum').item()
    model.train()
    loss = tot / Yv.numel()
    return loss, loss / math.log(2) / BPT
def sample(model, prompt='Once upon a time', n=60, seed=0):
    torch.manual_seed(seed)
    ids = torch.tensor([tok.encode(prompt).ids], device=device)
    out = generate(model, ids, n, temperature=0.8, top_k=40, eos_id=EOT)
    model.train()
    return tok.decode(out[0].tolist())
loss0, bpb0 = evaluate(model)
print(f'untrained: val loss {loss0:.3f} (random guessing: ln 4096 = {math.log(4096):.3f}) | {bpb0:.2f} bits per byte')
print('sample:', repr(sample(model)[:150]))""")
md("""## 6. Train
The loop: pick the learning rate, take a step, and every `eval_every` steps evaluate, print a sample and log to Trackio. Every 20 minutes (and at the end) the full state goes to a checkpoint: model, optimizer, scaler, step, history and config. `RESUME = True` picks up from that file if it exists.

Where the checkpoint goes: Google Drive on Colab (it survives the session), `/kaggle/working` on Kaggle (download it from the Output tab before the session ends, or commit the notebook version), `checkpoints/` on the laptop.""")
code("""CKPT = os.path.join(tsdata.save_dir(), 'p1_04_tinystories.pt')
RESUME = True

model = make_model(); opt = make_optimizer(model)
history = {'step': [], 'tokens': [], 'train_loss': [], 'val_loss': [], 'bpb': [], 'lr': [], 'samples': []}
start = 0
if RESUME and os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
    history, start = ck['history'], ck['step']
    print(f'resumed from {CKPT} at step {start:,}')
def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'step': step,
                'config': MODEL_CFG, 'train': TRAIN, 'history': history, 'meta': meta}, CKPT + '.tmp')
    os.replace(CKPT + '.tmp', CKPT)     # write then rename: a crash mid-save never destroys the old checkpoint

trackio.init(project='train-your-own-model', name='p1-04-tinystories', resume='allow', embed=False, config={**MODEL_CFG, **TRAIN, 'params': n_params})
t_start = t_ckpt = time.time(); t_log = time.time(); recent = []
for step in range(start, STEPS):
    loss, gnorm = train_step(model, opt, step)
    recent.append(loss)
    if (step + 1) % 50 == 0:
        dt = time.time() - t_log; t_log = time.time()
        trackio.log({'train_loss': float(np.mean(recent[-50:])), 'lr': lr_at(step), 'grad_norm': gnorm, 'tokens_per_s': 50 * TOKENS_PER_STEP / dt}, step=step + 1)
    if (step + 1) % TRAIN['eval_every'] == 0 or step + 1 == STEPS:
        vl, bpb = evaluate(model); tl = float(np.mean(recent)); recent = []
        s = sample(model)
        for k, v in [('step', step + 1), ('tokens', (step + 1) * TOKENS_PER_STEP), ('train_loss', tl), ('val_loss', vl), ('bpb', bpb), ('lr', lr_at(step)), ('samples', s)]:
            history[k].append(v)
        trackio.log({'val_loss': vl, 'bits_per_byte': bpb}, step=step + 1)
        hrs = (time.time() - t_start) / 3600
        print(f'step {step + 1:6,d}/{STEPS:,} | {(step + 1) * TOKENS_PER_STEP / 1e6:6.1f}M tok | train {tl:.3f} | val {vl:.3f} | {bpb:.3f} bpb | lr {lr_at(step):.1e} | {hrs:.2f} h')
        print('   ', repr(s[:160]))
    if time.time() - t_ckpt > TRAIN['ckpt_minutes'] * 60:
        save(step + 1); t_ckpt = time.time()
    if (time.time() - t_start) / 3600 > TRAIN['max_hours']:
        save(step + 1); print(f'stopped at max_hours; checkpoint saved at step {step + 1:,}. Run again to resume.'); break
else:
    save(STEPS); print(f'done: {STEPS:,} steps, checkpoint {CKPT} ({os.path.getsize(CKPT) / 1e6:.0f} MB)')
trackio.finish()""")
md("""## 7. What happened
Loss falls very fast at first (the model learns spelling, common words and punctuation), then slowly (grammar, then story structure: characters, a problem, an ending). Because we never repeat data, **train and validation loss should stay close**: compare that with P0-04, where they split apart within a few epochs.""")
code("""h = history
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5))
a1.plot(np.array(h['tokens']) / 1e6, h['train_loss'], label='train (average since last eval)')
a1.plot(np.array(h['tokens']) / 1e6, h['val_loss'], label='validation')
a1.set_xlabel('tokens seen (millions)'); a1.set_ylabel('loss (nats per token)'); a1.set_title('Loss'); a1.legend(); a1.grid(alpha=.3)
a2.plot(np.array(h['tokens']) / 1e6, h['bpb'], color='tab:green')
a2.set_xlabel('tokens seen (millions)'); a2.set_ylabel('bits per byte'); a2.set_title('Validation bits per byte (lower is better)'); a2.grid(alpha=.3)
plt.tight_layout(); plt.savefig('p1_04_training.png', dpi=150); plt.show()
print(f"final: train {h['train_loss'][-1]:.3f} | val {h['val_loss'][-1]:.3f} | gap {h['val_loss'][-1] - h['train_loss'][-1]:+.3f} | {h['bpb'][-1]:.3f} bits per byte")
print('\\nthe same prompt and seed over training:')
picks = sorted(set(list(range(0, len(h['step']), max(1, len(h['step']) // 5))) + [len(h['step']) - 1]))
for st, s in [(h['step'][i], h['samples'][i]) for i in picks]:
    print(f'step {st:6,d}: {s[:200]!r}')""")
md("""## 8. Write some stories
Different prompts, sampled with temperature 0.8 and top-k 40 (P0-04). Generation stops at `<|endoftext|>`: the model decides when its story is over.""")
code("""model.eval()
for i, p in enumerate(['Once upon a time', 'Lily wanted a puppy.', 'The little robot', 'Tom and his dad went to the park.']):
    print(f'--- {p}\\n{sample(model, p, n=200, seed=i)}\\n')""")
md("""## Your turn
1. Plot `history['lr']` against `history['val_loss']`. Where in the schedule does the loss improve fastest?
2. Set `'warmup': 0` and `'lr': 3e-3`, run 2,000 steps, and compare the first evals with a normal run. What does warmup protect against?
3. After training, run `evaluate` on 1,000 windows instead of 100. How much does the number move? How many windows do you need for a stable comparison?
4. Turn on `torch.compile(model)` right after `make_model()` and rerun the benchmark cell. Is the speedup worth the compile time for this run length?

**Next: P1-05** scores this model properly: bits per byte on the whole validation set, the same measurement for the official TinyStories models, and side-by-side stories.""")
nb.cells=C
nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nb.metadata['accelerator']='GPU'
nb.metadata['colab']={'provenance':[],'gpuType':'T4'}
nbf.write(nb,'P1-04-pretrain.ipynb')
