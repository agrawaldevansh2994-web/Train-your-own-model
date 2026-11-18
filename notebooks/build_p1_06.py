import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# P1-06 · SFT: teach your model to follow instructions
**Build Your Own AI · Phase 1, step 6.** GPU (Kaggle or Colab T4) about 15-30 minutes for the real run (a guess until you run the timing cell); a laptop CPU runs a short test. Needs the checkpoint from P1-04.

After pretraining, the model can only **continue** text. Give it "Write a story about a dog" and it continues that sentence like any other. **Supervised fine-tuning (SFT)** shows it thousands of examples of an instruction followed by a good answer, in a fixed **chat format**, until it learns: when a user turn ends, write the answer, then stop.

| | pretraining (P1-04) | SFT (this notebook) |
|---|---|---|
| data | raw stories | (instruction, story) pairs |
| format | one long stream | `<|im_start|>user ... <|im_end|>` then `<|im_start|>assistant ... <|im_end|>` |
| loss on | every token | **only the answer tokens** |
| learning rate | 1e-3 | 3e-4, a gentler nudge to a model that already knows English |
| tokens | 300M | about 30M |

**The data: TinyStories-Instruct**, from the same authors. Every story comes with some of: a summary, words it must use, features (dialogue, a bad ending, a twist, a moral), and a sentence it must contain. We turn those into an instruction. Because "use these words" can be checked by a program, we get an **automatic score** for instruction following, and we will reuse that score as the reward for reinforcement learning (GRPO) in a later step.""")
code("""import os, sys, subprocess, urllib.request, math, time, json, re, random
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
tok = tsdata.load_tokenizer('data')
EOT, IM_START, IM_END = (tok.token_to_id(t) for t in ['<|endoftext|>', '<|im_start|>', '<|im_end|>'])
print('device:', torch.cuda.get_device_name(0) if GPU else 'CPU (short test run only)', '| special ids', EOT, IM_START, IM_END)""")
md("""## 1. The base model from P1-04""")
code("""BASE_CKPT = tsdata.find_checkpoint('p1_04_tinystories.pt')
assert BASE_CKPT, 'No P1-04 checkpoint found. Put p1_04_tinystories.pt in checkpoints/ (or set BASE_CKPT).'
ck = torch.load(BASE_CKPT, map_location=device)
CFG = ck['config']
def load_base():
    m = MindModel(CFG).to(device); m.load_state_dict(ck['model']); return m
base = load_base().eval()
print(f'{BASE_CKPT}: pretrained on {ck["history"]["tokens"][-1] / 1e6:.0f}M tokens, val loss {ck["history"]["val_loss"][-1]:.3f}')""")
md("""## 2. The data
The raw file looks like this: fields in **any order**, sometimes after the story.""")
code("""BASE = 'https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/'
TRAIN_MB = 200 if GPU else 20          # 200 MB is about 180k examples, more than one epoch needs
os.makedirs('data', exist_ok=True)
tsdata.fetch(BASE + 'TinyStories-Instruct-train.txt', f'data/tsi_train_{TRAIN_MB}mb.txt', TRAIN_MB * 1_000_000)
tsdata.fetch(BASE + 'TinyStories-Instruct-valid.txt', 'data/tsi_valid.txt')
raw_val = open('data/tsi_valid.txt', encoding='utf-8', errors='replace').read().split('<|endoftext|>')[1:-1]
print(raw_val[4][:700])""")
md("""Parse each record into its fields, then write the fields as a plain-English instruction. The story is the answer.""")
code("""FIELD = re.compile(r'^(Features|Words|Summary|Random sentence|Story):', re.M)
FEATURES = {'Dialogue': 'has dialogue', 'BadEnding': 'has a bad ending', 'MoralValue': 'teaches a lesson', 'Twist': 'has a plot twist',
            'Foreshadowing': 'uses foreshadowing', 'Conflict': 'has a conflict'}
def parse(rec):
    parts = FIELD.split(rec)
    d = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    return d if d.get('Story') and d.get('Summary') else None
def instruction(d):
    lines = ['Write a short story.', f"It is about: {d['Summary']}"]
    if 'Words' in d: lines.append(f"Use the words: {d['Words']}.")
    if 'Features' in d: lines.append('It ' + ' and '.join(FEATURES.get(f.strip(), f.strip().lower()) for f in d['Features'].split(',')) + '.')
    if 'Random sentence' in d: lines.append(f'Include this sentence: "{d["Random sentence"]}"')
    return '\\n'.join(lines)
def load(path):
    recs = open(path, encoding='utf-8', errors='replace').read().split('<|endoftext|>')[1:-1]
    return [d for d in map(parse, recs) if d]
train_ex, val_ex = load(f'data/tsi_train_{TRAIN_MB}mb.txt'), load('data/tsi_valid.txt')
print(f'{len(train_ex):,} training examples | {len(val_ex):,} validation examples')
print('\\nINSTRUCTION:\\n' + instruction(val_ex[4]) + '\\n\\nANSWER:\\n' + val_ex[4]['Story'][:300] + '...')""")
md("""## 3. The chat format and the loss mask
Each example becomes one token sequence:
```
<|im_start|>user
Write a short story. It is about: ...<|im_end|>
<|im_start|>assistant
Once upon a time ...<|im_end|>
```
These are the special tokens we reserved in P1-01 and never used in pretraining. The model has to learn them now.

**Loss mask.** We only train on the answer (and the closing `<|im_end|>`, so the model learns to stop). The instruction tokens get target `-100`, which `cross_entropy` ignores. Training on the instruction too would teach the model to *write instructions*, which we do not want.""")
code("""def encode(d):
    prompt = tok.encode(f"<|im_start|>user\\n{instruction(d)}<|im_end|>\\n<|im_start|>assistant\\n").ids
    answer = tok.encode(d['Story']).ids + [IM_END]
    return prompt, answer
p, a = encode(val_ex[4])
ids = p + a
targets = [-100] * len(p) + a                  # targets[i] is what the model should predict after seeing ids[:i+1] (shifted below)
print(f'{len(p)} instruction tokens (no loss) + {len(a)} answer tokens (loss)')
print('start  :', ' | '.join(tok.id_to_token(i) for i in ids[:12]))
print('switch :', ' | '.join(f'[{tok.id_to_token(i)}]' if k >= len(p) else tok.id_to_token(i) for k, i in enumerate(ids[len(p) - 4:len(p) + 4], start=len(p) - 4)))
print('(in brackets: tokens the model is trained to produce)')
MAX_LEN = CFG['context_length']
def to_example(d):
    p, a = encode(d)
    if len(p) + len(a) > MAX_LEN: return None          # rare long ones: skip rather than cut the ending off
    return p, a
train_set = [e for e in map(to_example, train_ex) if e]; val_set = [e for e in map(to_example, val_ex[:400]) if e]
print(f'\\nkept {len(train_set):,} of {len(train_ex):,} training examples (the rest are longer than {MAX_LEN} tokens)')
print(f'average example: {np.mean([len(p) + len(a) for p, a in train_set]):.0f} tokens, of which {np.mean([len(a) for p, a in train_set]):.0f} are answer')""")
md("""A batch pads examples to the same length. Padding positions also get target `-100`.""")
code("""def collate(batch):
    L = max(len(p) + len(a) for p, a in batch)
    x = torch.full((len(batch), L), EOT, dtype=torch.long); y = torch.full((len(batch), L), -100, dtype=torch.long)
    for r, (p, a) in enumerate(batch):
        seq = p + a
        x[r, :len(seq)] = torch.tensor(seq)
        y[r, len(p) - 1:len(seq) - 1] = torch.tensor(a)      # predict each answer token from the position before it
    return x.to(device), y.to(device)
x, y = collate(train_set[:4])
print('batch', tuple(x.shape), '| trained positions per row:', (y != -100).sum(1).tolist())

@torch.no_grad()
def sft_val_loss(model, n=len(val_set), bs=16):
    model.eval(); tot = cnt = 0
    for k in range(0, n, bs):
        x, y = collate(val_set[k:k + bs])
        with torch.autocast(device, dtype=AMP, enabled=GPU):
            logits = model(x)
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100, reduction='sum').item()
        cnt += (y != -100).sum().item()
    model.train(); return tot / cnt
print(f'base model, loss on answer tokens: {sft_val_loss(base):.3f}')""")
md("""## 4. The score: does the story use the required words?
For validation examples that list words, generate a story and check that every word appears (as part of a longer word counts too: "jumped" contains "jump"). This is crude: it ignores whether the story matches the summary or is any good. But it is automatic, and it measures exactly one instruction-following skill.""")
code("""def words_of(d): return [w.strip().lower() for w in d.get('Words', '').split(',') if w.strip()]
word_eval = [d for d in val_ex[400:] if words_of(d)][:(100 if GPU else 12)]
@torch.no_grad()
def write(model, d, temperature=0.7, n=300, seed=0):
    torch.manual_seed(seed)
    ids = torch.tensor([tok.encode(f"<|im_start|>user\\n{instruction(d)}<|im_end|>\\n<|im_start|>assistant\\n").ids], device=device)
    out = generate(model, ids, n, temperature=temperature, top_k=40, eos_id=IM_END)
    return tok.decode(out[0, ids.shape[1]:].tolist())
def word_score(model):
    hits = []
    for i, d in enumerate(word_eval):
        story = write(model, d, seed=i).lower()
        hits.append(all(w in story for w in words_of(d)))
    return float(np.mean(hits))
t0 = time.time(); base_score = word_score(base)
print(f'base model: {base_score:.0%} of {len(word_eval)} stories use all the required words ({time.time() - t0:.0f}s)')
print('\\nbase model given an instruction:\\n' + write(base, word_eval[0])[:400])""")
md("""The base model has never seen `<|im_start|>`, so it has no idea a story is wanted. Whatever it writes is a continuation of something that looks like a strange story. Any required words it hits are luck (or come from words common in all stories).

## 5. Fine-tune
Same machinery as P1-04: AdamW, warmup and cosine, mixed precision, a checkpoint at the end. One pass over the examples.""")
code("""SFT = {'lr': 3e-4, 'min_lr': 3e-5, 'warmup': 100 if GPU else 10, 'batch': 32 if GPU else 8,
       'examples': min(len(train_set), 100_000) if GPU else 8 * 120, 'eval_every': 500 if GPU else 40, 'grad_clip': 1.0}
STEPS = SFT['examples'] // SFT['batch']
def lr_at(step):
    if step < SFT['warmup']: return SFT['lr'] * (step + 1) / SFT['warmup']
    prog = (step - SFT['warmup']) / max(1, STEPS - SFT['warmup'])
    return SFT['min_lr'] + 0.5 * (SFT['lr'] - SFT['min_lr']) * (1 + math.cos(math.pi * prog))
model = load_base().train()
opt = torch.optim.AdamW(model.parameters(), lr=SFT['lr'], betas=(0.9, 0.95), weight_decay=0.0, fused=GPU)
scaler = torch.amp.GradScaler('cuda', enabled=GPU and AMP == torch.float16)
random.seed(0); order = random.sample(range(len(train_set)), SFT['examples'])
history = {'step': [], 'train_loss': [], 'val_loss': []}
trackio.init(project='train-your-own-model', name='p1-06-sft', embed=False, config={**SFT, **CFG})
t0 = time.time(); recent = []; n_tok = 0
for step in range(STEPS):
    for g in opt.param_groups: g['lr'] = lr_at(step)
    x, y = collate([train_set[i] for i in order[step * SFT['batch']:(step + 1) * SFT['batch']]])
    with torch.autocast(device, dtype=AMP, enabled=GPU):
        logits = model(x)
    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), SFT['grad_clip'])
    scaler.step(opt); scaler.update()
    recent.append(loss.item()); n_tok += x.numel()
    if step == 19: print(f'MEASURED: {n_tok / (time.time() - t0):,.0f} tokens/s -> about {(time.time() - t0) / 20 * STEPS / 60:.0f} min for {STEPS:,} steps')
    if (step + 1) % SFT['eval_every'] == 0 or step + 1 == STEPS:
        vl = sft_val_loss(model); tl = float(np.mean(recent)); recent = []
        history['step'].append(step + 1); history['train_loss'].append(tl); history['val_loss'].append(vl)
        trackio.log({'train_loss': tl, 'val_loss': vl, 'lr': lr_at(step)}, step=step + 1)
        print(f'step {step + 1:5,d}/{STEPS:,} | train {tl:.3f} | val {vl:.3f} | {(time.time() - t0) / 60:.1f} min')
trackio.finish()
SAVE = os.path.join(tsdata.save_dir(), 'p1_06_sft.pt')
torch.save({'model': model.state_dict(), 'config': CFG, 'history': history, 'sft': SFT, 'base': BASE_CKPT}, SAVE)
print(f'saved {SAVE}')""")
md("""## 6. Before and after
On the laptop CPU test the base model saw only about 1M tokens, so expect both word scores near 0%: the code runs, but the numbers mean something only after the GPU run.""")
code("""t0 = time.time(); sft_score = word_score(model)
print(f'loss on answer tokens : base {sft_val_loss(base):.3f} -> SFT {sft_val_loss(model):.3f}')
print(f'uses all required words: base {base_score:.0%} -> SFT {sft_score:.0%} (of {len(word_eval)} stories, {time.time() - t0:.0f}s)')
for i in [0, 1]:
    d = word_eval[i]
    print(f'\\n=== INSTRUCTION\\n{instruction(d)}\\n=== BASE MODEL\\n{write(base, d)[:500]}\\n=== SFT MODEL\\n{write(model, d)[:700]}')""")
md("""## 7. Try your own instruction
The model only knows what TinyStories-Instruct taught it: a summary, words, features, a sentence. Instructions far from that ("write a poem", "what is 2 + 2") are outside anything it has seen.""")
code("""mine = {'Summary': 'A little cat named Biscuit is afraid of the rain, but her friend the duck shows her that puddles are fun.',
        'Words': 'umbrella, splash, brave', 'Features': 'Dialogue'}
print(instruction(mine), '\\n'); print(write(model, mine, seed=1))""")
md("""## Your turn
1. Train on every token (set `y` to the whole shifted sequence, prompt included). Compare the word score and look at what the model writes. Why does the mask matter?
2. Set the SFT learning rate to 1e-3 (the pretraining value). Does the story quality or the val loss suffer? This is the classic risk of fine-tuning too hard ("catastrophic forgetting").
3. Write an instruction that asks for something TinyStories never contains (a recipe, a poem). What does the model do?
4. The word score ignores whether the story fits the summary. Design a second automatic check. (Hint: does the story mention the character names from the summary?)

**Next: P1-07** LoRA: the same fine-tuning while training only a tiny fraction of the weights.""")
nb.cells=C
nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nb.metadata['accelerator']='GPU'
nb.metadata['colab']={'provenance':[],'gpuType':'T4'}
nbf.write(nb,'P1-06-sft.ipynb')
