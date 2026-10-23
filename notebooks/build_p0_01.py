import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# P0-01 · Tokenizer from scratch
**Build Your Own AI · Phase 0, step 1.** Runs on your laptop CPU or in Colab, about 20 minutes.

An LLM never sees letters. It sees **integers** (token ids). This notebook builds that bridge three ways:
1. a simple word tokenizer (and why it breaks),
2. a **byte-level BPE** tokenizer trained by you, merge by merge (the same idea GPT-2/4, minimind and nanochat use),
3. turning ids into **(input, target)** training pairs and **embedding vectors**: the exact tensors a GPT eats.

Run each cell top to bottom and read the printouts. Write what surprised you in `lab-notes/P0-01-tokenizer.md`.""")
code("""# Setup (Colab already has torch; on your laptop: pip install torch tiktoken)
import sys, subprocess
try:
    import tiktoken
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tiktoken'])
    import tiktoken
import re, urllib.request, collections, torch
print('torch', torch.__version__)""")
md("## 1. The data\nA 20 KB short story (Edith Wharton, *The Verdict*, public domain). Tiny on purpose, so every step is visible.")
code("""url = 'https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt'
text = urllib.request.urlopen(url).read().decode('utf-8')
print('characters:', len(text))
print(text[:300])""")
md("""## 2. Word tokenizer: the naive way
Split on spaces and punctuation, give every unique piece an id. Simple, but watch what happens to a word it never saw.""")
code("""pieces = [p for p in re.split(r'([,.:;?_!"()\\']|--|\\s)', text) if p.strip()]
print('pieces:', len(pieces), pieces[:20])
vocab_words = sorted(set(pieces)) + ['<|endoftext|>', '<|unk|>']
w2i = {w:i for i,w in enumerate(vocab_words)}; i2w = {i:w for w,i in w2i.items()}
print('vocab size:', len(w2i))

def word_encode(s):
    ps = [p for p in re.split(r'([,.:;?_!"()\\']|--|\\s)', s) if p.strip()]
    return [w2i.get(p, w2i['<|unk|>']) for p in ps]
def word_decode(ids):
    return re.sub(r'\\s+([,.?!"()\\'])', r'\\1', ' '.join(i2w[i] for i in ids))

s = 'Hello, do you like tea? Jack painted it.'
ids = word_encode(s)
print(ids)
print(word_decode(ids))   # 'Hello' never appears in the story, so it becomes <|unk|>: information lost""")
md("""**The problem:** any new word becomes `<|unk|>`, and a real-world word vocab would need millions of entries.
**The fix:** work below the word level. Start from raw **bytes** (256 symbols, covers every language and emoji) and *learn* which byte pairs to glue together.""")
md("""## 3. Byte-level BPE, trained by you
Algorithm (Byte Pair Encoding):
1. Turn text into bytes: ids 0-255.
2. Count every adjacent pair `(a, b)`.
3. The most frequent pair becomes a **new token** id 256, 257, ...; replace it everywhere.
4. Repeat for `num_merges` steps. Vocab size = 256 + num_merges.

GPT-2 did 50,000 merges; GPT-4 about 100k; minimind about 6,400. We do 300 so you can watch it.""")
code("""def get_pair_counts(ids):
    return collections.Counter(zip(ids, ids[1:]))

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

ids = list(text.encode('utf-8'))
print('start: bytes =', len(ids))
num_merges = 300
merges = {}                                  # (a, b) -> new id
vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
for k in range(num_merges):
    counts = get_pair_counts(ids)
    pair = max(counts, key=counts.get)
    new_id = 256 + k
    ids = merge(ids, pair, new_id)
    merges[pair] = new_id
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    if k < 15 or k % 50 == 0:
        print(f'merge {k:3d}: {vocab[pair[0]]!r} + {vocab[pair[1]]!r} -> {vocab[new_id]!r}  (seen {counts[pair]}x)')
print('end: tokens =', len(ids), '| compression', round(len(text.encode()) / len(ids), 2), 'bytes per token')""")
md("Watch the merges: first spaces and common letter pairs (`e `, `th`), then whole words (` the`, ` and`). Nobody told it English; it found the structure from counts.")
code("""def bpe_encode(s):
    ids = list(s.encode('utf-8'))
    while len(ids) >= 2:
        counts = get_pair_counts(ids)
        pair = min(counts, key=lambda p: merges.get(p, float('inf')))  # earliest-learned merge first
        if pair not in merges: break
        ids = merge(ids, pair, merges[pair])
    return ids

def bpe_decode(ids):
    return b''.join(vocab[i] for i in ids).decode('utf-8', errors='replace')

s = 'Hello, do you like tea? Jack painted it. नमस्ते 🙂'
ids = bpe_encode(s)
print(ids)
print([bpe_decode([i]) for i in ids])
print('round trip ok:', bpe_decode(ids) == s)   # no <|unk|> ever, even Hindi and emoji""")
md("The `�` marks are not errors: each Devanagari letter is 3 bytes and the emoji is 4, and our tiny tokenizer never learned to merge them, so you see single raw bytes. Decoded together they round-trip perfectly.")
md("## 4. Compare with the real GPT-2 tokenizer\nSame algorithm, trained on ~40 GB of web text with 50,000 merges.")
code("""gpt2 = tiktoken.get_encoding('gpt2')
for name, enc in [('ours (556 vocab)', bpe_encode), ('gpt2 (50257 vocab)', gpt2.encode)]:
    t = enc(text)
    print(f'{name:20s} tokens: {len(t):6d}  bytes/token: {len(text.encode())/len(t):.2f}')
print([gpt2.decode([i]) for i in gpt2.encode('Hello, do you like tea? नमस्ते')])""")
md("Bigger vocab = fewer tokens per text = more text fits in the model's context, but a bigger embedding table to learn. Note how GPT-2 shreds Hindi into many tokens: its training data was mostly English. **That's why tokenizers are trained per language mix.**")
md("""## 5. From ids to training pairs
A GPT learns one thing: **predict the next token**. Slide a window over the ids; the target is the input shifted by one.""")
code("""ids_all = gpt2.encode(text)
context = 8
x = ids_all[50:50+context]; y = ids_all[51:51+context]
for i in range(context):
    print(repr(gpt2.decode(x[:i+1])), '->', repr(gpt2.decode([y[i]])))""")
code("""from torch.utils.data import Dataset, DataLoader
class NextTokenDataset(Dataset):
    def __init__(self, ids, context, stride):
        self.x, self.y = [], []
        for i in range(0, len(ids) - context, stride):
            self.x.append(torch.tensor(ids[i:i+context]))
            self.y.append(torch.tensor(ids[i+1:i+context+1]))
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

ds = NextTokenDataset(ids_all, context=4, stride=4)
dl = DataLoader(ds, batch_size=8, shuffle=False)
xb, yb = next(iter(dl))
print('samples:', len(ds)); print('x', xb.shape, '\\n', xb); print('y', yb.shape, '\\n', yb)""")
md("""## 6. From ids to vectors (embeddings)
An embedding layer is just a lookup table: row `id` = a learnable vector. A GPT adds a second table for **position**, since attention alone doesn't know order.

`input = token_emb[ids] + pos_emb[0..T-1]` gives shape **(batch, T, d_model)**, the tensor that enters the transformer in step P0-03.""")
code("""torch.manual_seed(123)
vocab_size, d_model, T = gpt2.n_vocab, 256, 4
tok_emb = torch.nn.Embedding(vocab_size, d_model)
pos_emb = torch.nn.Embedding(T, d_model)
h = tok_emb(xb) + pos_emb(torch.arange(T))
print('token table:', tuple(tok_emb.weight.shape), '=', tok_emb.weight.numel()/1e6, 'M params')
print('input to GPT:', tuple(h.shape), '(batch, tokens, d_model)')""")
md("""## Your turn (5 min each)
1. Change `num_merges` to 50 and 1000. How do compression and the learned tokens change?
2. Encode a Hinglish sentence you'd actually type with ours vs GPT-2. Who uses fewer tokens? Why?
3. Set `stride=1` in the dataset. How many samples now? What's the trade-off?
4. With GPT-2's vocab and d_model=768, how many params is the token table alone? (Hint: 50257 x 768.)

**Next: P0-02 attention** (why the model looks back at earlier tokens, and the math of Q, K, V).""")
nb.cells=C; nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nbf.write(nb,'P0-01-tokenizer.ipynb')
