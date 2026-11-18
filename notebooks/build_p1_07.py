import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# P1-07 · LoRA: fine-tune by training 3% of the weights
**Build Your Own AI · Phase 1, step 7.** GPU (Kaggle or Colab T4) for the real run, about as long as P1-06; a laptop CPU runs a short test. Needs the checkpoints from P1-04 (and P1-06, to compare).

In P1-06 every one of the 15.9M weights changed. For a 7B model that means a 28 GB copy of the weights per fine-tune, plus optimizer state three times that size. **LoRA** (Low-Rank Adaptation, Hu et al. 2021) freezes the original weights and learns a small **correction** next to each chosen weight matrix:

$$W x \\;\\longrightarrow\\; W x + \\frac{\\alpha}{r} \\, B A x \\qquad A: r \\times d_{in}, \\; B: d_{out} \\times r$$

With rank `r` = 8 and a 384 × 384 matrix, `B A` has 384·8 + 8·384 = 6,144 numbers instead of 147,456. `B` starts at **zero**, so at step 0 the model is exactly the base model; training only grows the correction.

Questions this notebook answers with numbers: how many weights do we train, how big is the saved file, and how close does LoRA get to full fine-tuning on the same task?

minimind implements LoRA by hand too (`model/model_lora.py`, about 60 lines); this is the same idea.""")
code("""import os, sys, subprocess, urllib.request, math, time, random
REPO = 'https://raw.githubusercontent.com/agrawaldevansh2994-web/Train-your-own-model/main/'
for f in ['mymind.py', 'tsdata.py', 'sftdata.py']:
    if not os.path.exists(f) and not os.path.exists('../src/' + f):
        urllib.request.urlretrieve(REPO + 'src/' + f, f)
sys.path += ['.', '../src']
for pkg in ['tokenizers']:
    try: __import__(pkg)
    except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg])
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from mymind import MindModel
import tsdata, sftdata
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = device == 'cuda'
AMP = torch.bfloat16 if GPU and torch.cuda.is_bf16_supported() else torch.float16
tok = tsdata.load_tokenizer('data'); EOT = tok.token_to_id('<|endoftext|>')
BASE_CKPT = tsdata.find_checkpoint('p1_04_tinystories.pt'); assert BASE_CKPT, 'P1-04 checkpoint not found'
ck = torch.load(BASE_CKPT, map_location=device); CFG = ck['config']
def load_base():
    m = MindModel(CFG).to(device); m.load_state_dict(ck['model']); return m
print('device:', device, '| base:', BASE_CKPT)""")
md("""## 1. A LoRA layer
`LoRALinear` wraps an existing `nn.Linear`. The original weight is frozen (`requires_grad = False`). Only `A` and `B` train.""")
code("""class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))   # zero: starts as the base model
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
    def merged_weight(self):
        return self.base.weight + self.scale * self.B @ self.A

lin = nn.Linear(384, 384, bias=False); lora = LoRALinear(lin, r=8)
x = torch.randn(2, 384)
print('equal to the base layer at the start:', torch.allclose(lora(x), lin(x)))
print(f'trainable: {sum(p.numel() for p in lora.parameters() if p.requires_grad):,} (A and B) vs {lin.weight.numel():,} in the frozen weight')""")
md("""## 2. Put LoRA into the model
We wrap every attention and feed-forward matrix in every block (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) and freeze everything else, embedding and norms included.""")
code("""TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
def add_lora(model, r=8, alpha=16):
    for p in model.parameters(): p.requires_grad_(False)
    for block in model.blocks:
        for parent in [block.attn, block.ffn]:
            for name in TARGETS:
                if hasattr(parent, name): setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha))
    return model
model = add_lora(load_base(), r=8)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad); total = sum(p.numel() for p in model.parameters())
print(f'trainable {trainable:,} of {total:,} parameters = {trainable / total:.1%}')
with torch.no_grad():
    ids = torch.randint(0, 4096, (1, 32), device=device)
    print('identical to the base model before training:', torch.allclose(model(ids), load_base()(ids), atol=1e-5))""")
md("""## 3. The same SFT task as P1-06
`sftdata.py` holds P1-06's data code: parse TinyStories-Instruct, build the chat prompt, mask the loss to the answer, and the "uses all required words" score.""")
code("""train_ex, val_ex = sftdata.load_examples('data', train_mb=200 if GPU else 20)
train_set = sftdata.to_examples(tok, train_ex, CFG['context_length'])
val_set = sftdata.to_examples(tok, val_ex[:400], CFG['context_length'])
word_eval = [d for d in val_ex[400:] if sftdata.words_of(d)][:(100 if GPU else 12)]
@torch.no_grad()
def val_loss(model, bs=16):
    model.eval(); tot = cnt = 0
    for k in range(0, len(val_set), bs):
        x, y = sftdata.collate(val_set[k:k + bs], device, EOT)
        with torch.autocast(device, dtype=AMP, enabled=GPU):
            logits = model(x)
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100, reduction='sum').item(); cnt += (y != -100).sum().item()
    model.train(); return tot / cnt
print(f'{len(train_set):,} training examples | base model answer loss {val_loss(load_base()):.3f}')""")
md("""## 4. Train only A and B
LoRA usually wants a **higher** learning rate than full fine-tuning (the update starts at zero and has few parameters to move), so 1e-3 here against 3e-4 in P1-06. Same examples, same number of steps.""")
code("""LORA = {'lr': 1e-3, 'min_lr': 1e-4, 'warmup': 100 if GPU else 10, 'batch': 32 if GPU else 8,
        'examples': min(len(train_set), 100_000) if GPU else 8 * 120, 'grad_clip': 1.0}
STEPS = LORA['examples'] // LORA['batch']
def lr_at(step):
    if step < LORA['warmup']: return LORA['lr'] * (step + 1) / LORA['warmup']
    prog = (step - LORA['warmup']) / max(1, STEPS - LORA['warmup'])
    return LORA['min_lr'] + 0.5 * (LORA['lr'] - LORA['min_lr']) * (1 + math.cos(math.pi * prog))
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LORA['lr'], betas=(0.9, 0.95), weight_decay=0.0)
scaler = torch.amp.GradScaler('cuda', enabled=GPU and AMP == torch.float16)
random.seed(0); order = random.sample(range(len(train_set)), LORA['examples'])
model.train(); t0 = time.time(); recent = []
for step in range(STEPS):
    for g in opt.param_groups: g['lr'] = lr_at(step)
    x, y = sftdata.collate([train_set[i] for i in order[step * LORA['batch']:(step + 1) * LORA['batch']]], device, EOT)
    with torch.autocast(device, dtype=AMP, enabled=GPU):
        logits = model(x)
    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, LORA['grad_clip']); scaler.step(opt); scaler.update()
    recent.append(loss.item())
    if step == 19: print(f'MEASURED: about {(time.time() - t0) / 20 * STEPS / 60:.0f} min for {STEPS:,} steps')
    if (step + 1) % max(1, STEPS // 6) == 0 or step + 1 == STEPS:
        print(f'step {step + 1:5,d}/{STEPS:,} | train {np.mean(recent):.3f} | val {val_loss(model):.3f} | {(time.time() - t0) / 60:.1f} min'); recent = []""")
md("""## 5. What gets saved
Only `A` and `B` need saving: the base model is unchanged and already on disk. That is the practical win of LoRA: one base model, many small adapters (one per task, per customer, per style).""")
code("""adapter = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
SAVE = os.path.join(tsdata.save_dir(), 'p1_07_lora_adapter.pt')
torch.save({'adapter': adapter, 'r': 8, 'alpha': 16, 'targets': TARGETS, 'base': BASE_CKPT, 'config': CFG}, SAVE)
full_mb = sum(p.numel() for p in load_base().parameters()) * 4 / 1e6
print(f'adapter file: {os.path.getsize(SAVE) / 1e6:.2f} MB | a full fp32 copy of the model: {full_mb:.0f} MB')""")
md("""## 6. Merge: no extra cost at inference
After training, `W + (α/r) B A` is just another matrix of the same shape. Fold it into the base weight and the LoRA layers disappear: the merged model is a normal `MindModel`, as fast as the original.""")
code("""merged = load_base()
with torch.no_grad():
    for bm, bl in zip(merged.blocks, model.blocks):
        for pm, pl in [(bm.attn, bl.attn), (bm.ffn, bl.ffn)]:
            for name in TARGETS:
                if hasattr(pl, name): getattr(pm, name).weight.copy_(getattr(pl, name).merged_weight())
merged.eval(); model.eval()
with torch.no_grad():
    ids = torch.randint(0, 4096, (2, 64), device=device)
    print('max difference, LoRA model vs merged model:', (model(ids) - merged(ids)).abs().max().item())""")
md("""## 7. LoRA against full fine-tuning
On the laptop CPU test all word scores will be near 0% (the base model barely trained). Compare the answer losses there; after the GPU runs, compare the word scores too.""")
code("""rows = [('base (P1-04)', load_base())]
FULL = tsdata.find_checkpoint('p1_06_sft.pt')
if FULL:
    full = MindModel(CFG).to(device); full.load_state_dict(torch.load(FULL, map_location=device)['model']); rows.append(('full SFT (P1-06)', full))
rows.append(('LoRA r=8 (merged)', merged))
print(f"{'model':20s} {'trained params':>15s} {'answer loss':>12s} {'uses all words':>15s}")
for name, m in rows:
    n = {'base (P1-04)': 0, 'full SFT (P1-06)': sum(p.numel() for p in m.parameters())}.get(name, trainable)
    print(f'{name:20s} {n:15,d} {val_loss(m):12.3f} {sftdata.word_score(m, tok, word_eval, device):15.0%}')
if not FULL: print('(no P1-06 checkpoint found, so no full-SFT row)')
d = word_eval[0]
print(f'\\n=== INSTRUCTION\\n{sftdata.instruction(d)}\\n=== LoRA\\n{sftdata.write(merged, tok, d, device)[:600]}')""")
md("""## Your turn
1. Try `r` = 1, 4, 32. Plot answer loss against trainable parameters. Where do the returns stop?
2. Put LoRA only on `q_proj` and `v_proj` (the original paper's choice). How much worse is it than all seven matrices?
3. Train LoRA with the P1-06 learning rate (3e-4). What happens, and why does LoRA want a larger one?
4. Train a second adapter on a different style (only stories with `BadEnding` in Features) and swap adapters on the same base model. What does each one write for the same instruction?

**Next: P1-08** preference tuning: DPO teaches the model which of two answers people prefer, without a reward model.""")
nb.cells=C
nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nb.metadata['accelerator']='GPU'
nb.metadata['colab']={'provenance':[],'gpuType':'T4'}
nbf.write(nb,'P1-07-lora.ipynb')
