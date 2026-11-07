import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# P1-02 · The modern block: from GPT-2 to minimind, Llama and Qwen
**Build Your Own AI · Phase 1, step 2.** Laptop CPU or Colab, about 5 minutes (downloads minimind-3, 128 MB).

The GPT you built in Phase 0 is the 2019 design. Every open model since Llama (2023) changed the same handful of parts. minimind-3 has them all, and its config file literally says `Qwen3ForCausalLM`.

| part | GPT-2 (our `mygpt.py`) | modern (`mymind.py`: minimind-3, Llama 3, Qwen3) | why |
|---|---|---|---|
| normalisation | LayerNorm: subtract mean, divide by std, scale, shift | **RMSNorm**: divide by root-mean-square, scale | cheaper, trains just as well |
| position | learned table added to the input (1,024 × 768 numbers) | **RoPE**: rotate q and k by an angle that depends on position | no parameters, attention sees relative distance |
| feed-forward | Linear, GELU, Linear (4 × wider) | **SwiGLU**: two branches, one gates the other, then Linear | better quality for the same parameters |
| attention heads | every head has its own q, k, v | **GQA**: 8 query heads share 4 key/value heads | half the memory when generating |
| attention stability | nothing | **QK-norm**: RMSNorm on each head's q and k | attention scores cannot blow up |
| biases | in every Linear | **none** | fewer parameters, nothing lost |
| generation | rerun the whole text for every new token | **KV cache**: keep old keys and values | much faster generation |

We build each piece, look at what it does with small numbers, then assemble them in `src/mymind.py`. The proof is the same as in P0-05: **load minimind-3's real weights into our code and match Hugging Face's logits.**""")
code("""import os, sys, subprocess, urllib.request, math, time
REPO = 'https://raw.githubusercontent.com/agrawaldevansh2994-web/Train-your-own-model/main/'
for f in ['mygpt.py', 'mymind.py']:
    if not os.path.exists(f) and not os.path.exists('../src/' + f):
        urllib.request.urlretrieve(REPO + 'src/' + f, f)
sys.path += ['.', '../src']
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
for pkg in ['transformers', 'safetensors', 'tokenizers']:
    try: __import__(pkg)
    except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg])
import torch, torch.nn as nn, torch.nn.functional as F
from mygpt import GPTModel, LayerNorm
from mymind import RMSNorm, rope_tables, apply_rope, SwiGLU, GQAttention, MindModel, KVCache, MIND_CONFIG_64M, generate, load_minimind_weights
torch.set_printoptions(precision=3, sci_mode=False)
print('torch', torch.__version__)""")
md("""## 1. RMSNorm
LayerNorm (P0-03) does two things: **centre** the vector (subtract its mean) and **scale** it (divide by its standard deviation), then applies a learned scale and shift. RMSNorm drops the centring and the shift:

$$\\text{RMSNorm}(x) = \\frac{x}{\\sqrt{\\text{mean}(x^2) + \\epsilon}} \\cdot w$$

The 2019 paper that proposed it found the centring was not what made LayerNorm work. Dropping it saves compute and one vector of parameters per norm.""")
code("""x = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
ln, rms = LayerNorm(4), RMSNorm(4)
for name, y in [('input', x), ('LayerNorm', ln(x)), ('RMSNorm', rms(x))]:
    print(f'{name:10s} {y.detach()[0]}  mean {y.mean().item():5.2f}  root-mean-square {y.pow(2).mean().sqrt().item():.2f}')
print('\\nparameters: LayerNorm', sum(p.numel() for p in ln.parameters()), '| RMSNorm', sum(p.numel() for p in rms.parameters()))""")
md("""Both outputs have root-mean-square 1, so the next layer always sees numbers of the same size. Only LayerNorm also forces the mean to 0. The model learns fine without that.

## 2. RoPE: rotary position embeddings
GPT-2 **adds** a learned position vector to each token at the input. That costs 1,024 × 768 parameters, gives nothing beyond position 1,024, and attention has to work out "how far apart are these two tokens" by itself.

RoPE instead **rotates** the query and key vectors. Split each head's vector into pairs of numbers; treat each pair as a point on a plane; rotate it by the angle `position × freq`. Pair 0 spins fast, the last pair very slowly.

The key fact: if q is rotated by angle `m·θ` and k by `n·θ`, their dot product only depends on the difference `(m − n)·θ`. So the attention score depends on **how far apart** two tokens are, not where they are. Check it:""")
code("""torch.manual_seed(0)
hd = 8
cos, sin = rope_tables(hd, 200, theta=10_000)
q, k = torch.randn(1, 1, 1, hd), torch.randn(1, 1, 1, hd)      # (batch, time, heads, head_dim)
def score(m, n):   # query at position m, key at position n
    qm = apply_rope(q, cos[m:m + 1], sin[m:m + 1]); kn = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
    return (qm * kn).sum().item()
for m, n in [(3, 1), (10, 8), (150, 148), (3, 0), (150, 100)]:
    print(f'query at {m:3d}, key at {n:3d}  distance {m - n:3d}  score {score(m, n):7.4f}')""")
md("""Same distance, same score, at any position. Different distance, different score.

How fast does each pair spin? With minimind-3's settings (head size 96, `theta` = 1,000,000) the first pair completes a full turn every 6 tokens, while the last pairs barely move across thousands of tokens. Fast pairs tell nearby tokens apart; slow pairs carry long-range position.""")
code("""hd, theta = 96, 1e6
freqs = 1.0 / theta ** (torch.arange(0, hd, 2).float() / hd)
for i in [0, 1, 8, 16, 32, 47]:
    print(f'pair {i:2d}: one full turn every {2 * math.pi / freqs[i].item():>14,.0f} tokens')
print('\\nRoPE parameters: 0 | GPT-2 learned positions:', f'{1024 * 768:,}')""")
md("""## 3. SwiGLU feed-forward
GPT-2's feed-forward is `Linear → GELU → Linear`. SwiGLU uses three matrices:

$$\\text{FFN}(x) = W_{down}\\big(\\text{silu}(W_{gate}\\,x) \\odot W_{up}\\,x\\big)$$

`up` computes candidate features, `gate` decides how much of each one passes (silu is a smooth "mostly 0 for negatives, about x for positives" curve), `⊙` multiplies them number by number. Because there are three matrices instead of two, the hidden width is made smaller to keep the size similar: Llama uses about 2.7 × d, minimind about π × d (2,432 for d = 768).""")
code("""xs = torch.tensor([-3.0, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0])
print('x      ', xs); print('silu(x)', F.silu(xs)); print('gelu(x)', F.gelu(xs))
cfg = dict(MIND_CONFIG_64M)
ffn_old = 2 * 768 * (4 * 768) + 4 * 768 + 768                 # GPT-2: two Linears with biases
ffn_new = sum(p.numel() for p in SwiGLU(cfg).parameters())    # three Linears, no biases
print(f'\\nfeed-forward params at d=768: GPT-2 {ffn_old:,} | SwiGLU (width {cfg["ffn_dim"]}) {ffn_new:,}')""")
md("""silu and GELU are nearly the same curve; the real change is the gate. SwiGLU here is a bit bigger than GPT-2's FFN per layer.

## 4. Grouped-query attention (GQA)
In P0-02 every head had its own query, key and value. In GQA, **groups of query heads share one key and one value head**. minimind-3 has 8 query heads and 4 key/value heads, so each k/v head serves 2 query heads.

Why bother? When generating, the model stores every past token's keys and values (the KV cache, section 7). Fewer k/v heads means a smaller cache. The quality loss is small.""")
code("""att = GQAttention(cfg, layer=0)
x = torch.randn(1, 5, 768)
print('q_proj', tuple(att.q_proj.weight.shape), '-> 8 query heads of 96')
print('k_proj', tuple(att.k_proj.weight.shape), '-> 4 key heads of 96')
print('v_proj', tuple(att.v_proj.weight.shape), '-> 4 value heads of 96')
print('which k/v head each query head uses:', torch.arange(4).repeat_interleave(2).tolist())
print('output', tuple(att(x, *rope_tables(96, 5)).shape))

def kv_bytes(layers, kv_heads, head_dim, ctx, bytes_per=2):   # keys + values, fp16
    return 2 * layers * kv_heads * head_dim * ctx * bytes_per
print(f"\\n{'KV cache in fp16':34s} {'per token':>10s} {'8k tokens':>10s}")
for name, L, h, hkv, hd in [('minimind-3, GQA 8 q / 4 kv', 8, 8, 4, 96), ('minimind-3 if plain multi-head', 8, 8, 8, 96),
                            ('Llama 3 8B, GQA 32 q / 8 kv', 32, 32, 8, 128), ('Llama 3 8B if plain multi-head', 32, 32, 32, 128)]:
    print(f'{name:34s} {kv_bytes(L, hkv, hd, 1) / 1024:8.0f} KB {kv_bytes(L, hkv, hd, 8192) / 2**30:8.2f} GB')""")
md("""For a big model the saving is what makes long conversations fit on one GPU: 1 GB instead of 4 GB for one 8k-token chat with Llama 3 8B.

## 5. QK-norm
Attention scores are `q · k`. If training pushes the q and k weights larger, scores grow with the **square** of that growth, softmax turns into a hard one-hot, and training can blow up. Qwen3 and OLMo 2 apply an RMSNorm to each head's q and k **before** RoPE, which fixes their size:""")
code("""torch.manual_seed(0)
x = torch.randn(6, 96); Wq, Wk = torch.randn(96, 96) * 0.1, torch.randn(96, 96) * 0.1
norm = RMSNorm(96)
for grow in [1, 3, 10]:
    q, k = x @ (Wq * grow), x @ (Wk * grow)
    raw = (q @ k.T / 96**0.5); normed = (norm(q) @ norm(k).T / 96**0.5).detach()
    print(f'weights x{grow:2d} | biggest raw score {raw.abs().max():8.2f} -> top softmax prob {raw.softmax(-1).max():.3f} '
          f'| with QK-norm {normed.abs().max():5.2f} -> {normed.softmax(-1).max():.3f}')""")
md("""Without QK-norm, growing the weights 10× makes scores 100× larger and attention collapses onto one token. With it, the scores stay the same size; the model can still sharpen attention on purpose through the learned norm weights.

## 6. The whole model
`src/mymind.py` assembles the pieces exactly like `mygpt.py` did: embedding, a stack of blocks (norm → attention → add, norm → feed-forward → add), final norm, output head tied to the embedding. Below, the same width and depth built both ways with minimind-3's 6,400-token vocabulary.""")
code("""new = MindModel(MIND_CONFIG_64M)
old = GPTModel({'vocab_size': 6400, 'context_length': 2048, 'emb_dim': 768, 'n_heads': 8, 'n_layers': 8, 'drop_rate': 0.0, 'qkv_bias': True}, tie_weights=True)
count = lambda m, f: sum(p.numel() for n, p in m.named_parameters() if f(n))
rows = [('token embedding (tied with output)', count(old, lambda n: 'tok_emb' in n), count(new, lambda n: 'tok_emb' in n)),
        ('position embedding', count(old, lambda n: 'pos_emb' in n), 0),
        ('attention, all layers', count(old, lambda n: '.att.' in n), count(new, lambda n: '.attn.' in n)),
        ('feed-forward, all layers', count(old, lambda n: '.ff.' in n), count(new, lambda n: '.ffn.' in n)),
        ('norms', count(old, lambda n: 'norm' in n), count(new, lambda n: 'norm' in n))]
print(f"{'':36s} {'GPT-2 style':>13s} {'modern':>13s}")
for name, a, b in rows: print(f'{name:36s} {a:13,d} {b:13,d}')
print(f"{'total':36s} {sum(p.numel() for p in old.parameters()):13,d} {sum(p.numel() for p in new.parameters()):13,d}")""")
md("""Both come out near 63-64M. The modern model has no position table, spends less on attention (k and v are half size) and more on the feed-forward. **63.9M** is minimind's "64M".

## 7. The proof: run minimind-3's real weights in our code
minimind-3 is published on Hugging Face as a standard `Qwen3ForCausalLM`. We download it, copy its weights into our `MindModel` with `load_minimind_weights` (only names change; no transposes, unlike GPT-2's in P0-05), and compare the logits with Hugging Face's own implementation.""")
code("""from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM
path = snapshot_download('jingyaogong/minimind-3', allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json'])
sd = load_file(os.path.join(path, 'model.safetensors'))
print(f'{len(sd)} tensors, dtype {next(iter(sd.values())).dtype}, e.g. {list(sd)[:3]}')
ours = load_minimind_weights(MindModel(MIND_CONFIG_64M), sd).eval()
hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
print('Hugging Face class:', type(hf).__name__)
mm_tok = Tokenizer.from_file(os.path.join(path, 'tokenizer.json'))
chat = lambda q: f'<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n'
ids = torch.tensor([mm_tok.encode(chat('What is the capital of France?')).ids])
with torch.no_grad():
    diff = (ours(ids) - hf(ids).logits).abs().max().item()
print(f'max logit difference, ours vs Hugging Face: {diff:.2e}', '(match)' if diff < 1e-3 else '(MISMATCH)')""")
md("""Our short `mymind.py` computes exactly the same numbers as Hugging Face's Qwen3 code. Now let it talk. minimind-3 is a chat model (pretrained, then SFT), so the prompt uses its chat format: `<|im_start|>user ... <|im_end|>` then `<|im_start|>assistant`. We stop at `<|im_end|>`, the token that means "my answer is finished". Remember it learned mostly from Chinese text, and it is only 64M parameters.""")
code("""eos = mm_tok.token_to_id('<|im_end|>')
for q in ['What is the capital of France?', 'Write one sentence about the sea.', 'Why is the sky blue?']:
    ids = torch.tensor([mm_tok.encode(chat(q)).ids])
    out = generate(ours, ids, 80, eos_id=eos)
    print(f'Q: {q}\\nA: {mm_tok.decode(out[0, ids.shape[1]:].tolist()).strip()}\\n')""")
md("""Facts it saw often (Paris, Rayleigh scattering) come out right. Open-ended prompts tend to loop ("a vast and endlessly embrace..."): we use greedy decoding, always taking the single most likely token, and a 64M model easily falls into a cycle. Sampling with `temperature=0.7, top_k=40` (P0-04) usually breaks the loop. Answers are cut off at 80 tokens.""")
md("""## 8. KV cache
Without a cache, generating token 100 runs all 100 tokens through the model again, though tokens 1 to 99 have not changed. Their keys and values are the same as last time. A KV cache stores them per layer, so each step only runs the **one new token** and attends over the stored keys and values.

`generate(..., use_cache=True)` runs the prompt once, then feeds one token at a time. Both ways must produce identical tokens (greedy); only the speed differs.""")
code("""ids = torch.tensor([mm_tok.encode('Once upon a time').ids])
t0 = time.time(); a = generate(ours, ids, 200, use_cache=False); t_plain = time.time() - t0
t0 = time.time(); b = generate(ours, ids, 200, use_cache=True); t_cache = time.time() - t0
print('identical tokens:', torch.equal(a, b))
print(f'200 new tokens: no cache {t_plain:.1f}s | KV cache {t_cache:.1f}s | {t_plain / t_cache:.1f}x faster')
cache = KVCache(len(ours.blocks)); ours(ids, cache)
k, v = cache.kv[0]
print('cache after the prompt, layer 0 keys:', tuple(k.shape), '= (batch, tokens, kv heads, head size)')""")
md("""The longer the text, the bigger the win, since the plain version's work grows with every token. The cache keeps 4 k/v heads, not 8: that is GQA's saving from section 4.

## Your turn
1. In the RoPE demo, set `theta=10` and print scores for distances 1 to 20. What goes wrong when every pair spins fast?
2. Build `MindModel` with `n_kv_heads` = 1 (multi-query attention, used by PaLM) and 8 (plain multi-head). How do the parameter count and the KV cache size change?
3. Break minimind-3 on purpose: after loading, set every `q_norm.weight` and `k_norm.weight` to 1.0, or skip RoPE (`cos=1, sin=0`). What does it say about France now?
4. Our TinyStories model will be much smaller. Build `MindModel` with `vocab_size` 4096 (P1-01's tokenizer), `emb_dim` 384, `n_layers` 8, `n_heads` 6, `n_kv_heads` 2, `ffn_dim` None, `context_length` 512. How many parameters, and what share is the embedding?

**Next: P1-03** turns 2.2 GB of TinyStories into token ids on disk, ready for training.""")
nb.cells=C
nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nbf.write(nb,'P1-02-modern-block.ipynb')
