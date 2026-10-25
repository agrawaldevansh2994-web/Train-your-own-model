import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# P0-05 · Load real GPT-2 weights into your own model
**Build Your Own AI · Phase 0, step 5 (the last step of Phase 0).** About 3 min on CPU, including a 548 MB download. No GPU needed.

In P0-04 you trained a 16M model on one short story (about 5,000 tokens). It learned commas and a few words.
OpenAI trained **GPT-2 (124M)** with *the exact same architecture you built*, on about 40 GB of web text (WebText).
Here you pour OpenAI's trained numbers into **your** `GPTModel` class. If your code is right, it will write fluent English.

What you learn:
- what a checkpoint really is: a dictionary of named tensors,
- how to map someone else's weight names and layouts onto your own code (the everyday skill behind Phase 1 to 3),
- how to check that a port is exactly correct, not just "looks fine",
- how much data does: same code, very different model.""")
code("""import os, sys, urllib.request
REPO = 'https://raw.githubusercontent.com/agrawaldevansh2994-web/Train-your-own-model/main/'
if not os.path.exists('mygpt.py') and not os.path.exists('../src/mygpt.py'):
    urllib.request.urlretrieve(REPO + 'src/mygpt.py', 'mygpt.py')
sys.path += ['.', '../src']
import subprocess
for pkg in ['tiktoken', 'safetensors']:
    try: __import__(pkg)
    except ImportError: subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', pkg], check=True)
import torch, tiktoken, math
from safetensors.torch import load_file
from mygpt import GPTModel, GPT_CONFIG_124M, generate, load_gpt2_weights
torch.manual_seed(123)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch', torch.__version__, '| device:', device)""")
md("""## 1. Download the weights
OpenAI's GPT-2 is hosted on Hugging Face as `model.safetensors`: one file, 548 MB.
In Colab it's cached in your Google Drive, so the next session skips the download.""")
code("""URL = 'https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors'
CACHE = 'gpt2.safetensors'
try:
    from google.colab import drive
    drive.mount('/content/drive')
    os.makedirs('/content/drive/MyDrive/train-your-own-model', exist_ok=True)
    CACHE = '/content/drive/MyDrive/train-your-own-model/gpt2.safetensors'
except ImportError:
    pass   # local run: cache next to the notebook
if not os.path.exists(CACHE):
    print('downloading 548 MB ...'); urllib.request.urlretrieve(URL, CACHE)
sd = load_file(CACHE)
print(f'{len(sd)} tensors, {sum(v.numel() for v in sd.values())/1e6:.1f}M numbers, {os.path.getsize(CACHE)/1e6:.0f} MB')""")
md("""## 2. Look inside: a checkpoint is just named tensors
Compare these names with your classes in `mygpt.py`:

| OpenAI / Hugging Face name | your code |
|---|---|
| `wte`, `wpe` | `tok_emb`, `pos_emb` |
| `h.0` ... `h.11` | `blocks[0]` ... `blocks[11]` |
| `ln_1`, `ln_2`, `ln_f` | `norm1`, `norm2`, `final_norm` (`weight`/`bias` = your `scale`/`shift`) |
| `attn.c_attn` | `W_q`, `W_k`, `W_v` **packed into one matrix** |
| `attn.c_proj` | `out_proj` |
| `mlp.c_fc`, `mlp.c_proj` | `ff.net[0]`, `ff.net[2]` |""")
code("""for k in ['wte.weight', 'wpe.weight', 'h.0.ln_1.weight', 'h.0.attn.c_attn.weight', 'h.0.attn.c_attn.bias',
          'h.0.attn.c_proj.weight', 'h.0.mlp.c_fc.weight', 'h.0.mlp.c_proj.weight', 'ln_f.weight']:
    print(f'{k:26s} {tuple(sd[k].shape)}')
extra = [k for k in sd if k.endswith('attn.bias') and sd[k].dim() > 1]
print('\\nnot weights (old causal-mask buffers, skipped):', len(extra), extra[:1])""")
md("""Two layout surprises, and both are classic porting bugs:
1. **Transposed.** `c_fc` is `(768, 3072)`, but your `nn.Linear(768, 3072).weight` is `(3072, 768)`. OpenAI used a "Conv1D" layer that stores weights as (in, out); PyTorch stores (out, in). So every matrix gets a `.T`.
2. **Packed.** `c_attn` is `(768, 2304)` = Q, K and V side by side (3 × 768). Split it into 3.

Your model also needs `qkv_bias=True`: GPT-2 has biases on Q/K/V, the small configs we used before did not.

The function below lives in `mygpt.py` as `load_gpt2_weights`. It copies every tensor and **asserts the shape**, so a wrong mapping fails loudly instead of silently producing garbage.""")
code("""import inspect
print(inspect.getsource(load_gpt2_weights))""")
md("## 3. Pour the weights into your model")
code("""cfg = {**GPT_CONFIG_124M, 'qkv_bias': True}
model = GPTModel(cfg)
tok = tiktoken.get_encoding('gpt2')
prompt = torch.tensor([tok.encode('Every effort moves you')])

model.eval()
print('BEFORE (random):', tok.decode(generate(model, prompt, 15, 1024)[0].tolist()))
load_gpt2_weights(model, sd)
model.eval()
n = sum(p.numel() for p in model.parameters())
print(f'params after loading: {n/1e6:.1f}M (tied embeddings, matches GPT-2 124M)')
print('AFTER  (GPT-2)  :', tok.decode(generate(model, prompt, 15, 1024)[0].tolist()))""")
md("""## 4. Is it *exactly* right? Compare with Hugging Face's own GPT-2
"Looks fluent" isn't proof: a slightly wrong port can still write OK text. The real test: feed the same tokens to Hugging Face's official implementation and compare the raw logits. They should match to about 1e-4 (floating-point noise).

Colab has `transformers` preinstalled. If it's missing, this cell skips.""")
code("""try:
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained('openai-community/gpt2').eval()
    x = torch.tensor([tok.encode('The quick brown fox jumps over the lazy dog because')])
    with torch.no_grad():
        mine, theirs = model(x), hf(x).logits
    diff = (mine - theirs).abs().max().item()
    print(f'max |logit difference| = {diff:.2e}  ->', 'MATCH' if diff < 1e-3 else 'MISMATCH, check the mapping')
    del hf
except ImportError:
    print('transformers not installed; skipped (the text test above is your check)')""")
md("""## 5. What does it know? Next-token probabilities
Same softmax over 50,257 tokens as P0-04, but now the probability goes to sensible words.""")
code("""def top5(text):
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor([tok.encode(text)]))[0, -1], -1)
    top = torch.topk(p, 5)
    print(f'{text!r:45s} ->', ', '.join(f'{tok.decode([i])!r} {v:.2f}' for v, i in zip(top.values.tolist(), top.indices.tolist())))
for t in ['The capital of France is', 'I drank a cup of', 'def add(a, b):\\n    return', 'Mumbai is a city in']:
    top5(t)""")
md("""## 6. Your P0-04 model vs GPT-2 on the same validation text
Same story, same 10% validation split that P0-04 held out. Loss = average surprise per token (lower is better); perplexity = e^loss ≈ "how many tokens it's choosing between".""")
code("""text = urllib.request.urlopen('https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt').read().decode()
ids = tok.encode(text); val = ids[int(0.9 * len(ids)):]
def val_loss(m, ctx):
    m.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val) - 1, ctx):
            x = torch.tensor([val[i:i + ctx]]); y = torch.tensor([val[i + 1:i + ctx + 1]])
            x = x[:, :y.shape[1]]
            tot += torch.nn.functional.cross_entropy(m(x)[0], y[0], reduction='sum').item(); n += y.numel()
    return tot / n
g = val_loss(model, 128)
print(f'GPT-2 124M   val loss {g:.2f} | perplexity {math.exp(g):,.0f}')
ck_paths = ['/content/drive/MyDrive/train-your-own-model/p0_04_tiny_gpt.pt', 'p0_04_tiny_gpt.pt']
ck = next((p for p in ck_paths if os.path.exists(p)), None)
if ck:
    c = torch.load(ck, map_location='cpu')
    tiny = GPTModel(c['cfg'], tie_weights=True); tiny.load_state_dict(c['model'])
    t = val_loss(tiny, 128)
    print(f'your P0-04   val loss {t:.2f} | perplexity {math.exp(t):,.0f}')
else:
    print('your P0-04 checkpoint not found (run P0-04 first); it typically ends around val loss 6.5-7')""")
md("""GPT-2 has **never seen this story** (it's validation text for both), yet it's far less surprised by it. The architecture is identical; the difference is about 8 million web pages of training. That's why Phase 1 and 2 are about data and compute, not new layers.""")
md("## 7. Sampling with a real model")
code("""torch.manual_seed(42)
for name, kw in [('greedy', dict(temperature=0.0)), ('T=0.7, top-k 40', dict(temperature=0.7, top_k=40)), ('T=1.2, top-k 100', dict(temperature=1.2, top_k=100))]:
    out = generate(model, torch.tensor([tok.encode('The best way to learn machine learning is')]), 40, 1024, **kw)
    print(f'--- {name}\\n{tok.decode(out[0].tolist())}\\n')""")
md("""Notice that greedy decoding tends to loop and repeat itself. Sampling is more varied, but at high temperature it drifts off topic. That's the same trade-off you saw in P0-04, now visible in real sentences.

## What you built in Phase 0
| step | piece |
|---|---|
| P0-01 | tokenizer: text → token ids |
| P0-02 | attention: tokens look at each other |
| P0-03 | the full GPT: blocks, residuals, LayerNorm, generation |
| P0-04 | pretraining loop: loss, AdamW, overfitting, checkpoints |
| P0-05 | your code + OpenAI's weights = working GPT-2 |

**Next, Phase 1:** train a modern small LLM (minimind, 64M) end to end on a Colab GPU: its own tokenizer, pretraining on real data, then chat fine-tuning (SFT), LoRA, and preference/RL tuning.""")
nb['cells']=C
nb['metadata']={'kernelspec':{'name':'python3','display_name':'Python 3'},'colab':{'provenance':[]}}
nbf.write(nb,'P0-05-load-gpt2-weights.ipynb'); print('built')
