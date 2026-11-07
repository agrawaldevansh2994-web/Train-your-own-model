import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# P1-01 · Train a real tokenizer
**Build Your Own AI · Phase 1, step 1.** Laptop CPU or Colab, about 5 minutes (most of it is a 100 MB download and five tokenizer trainings).

In P0-01 you trained BPE by hand on a 20 KB story, one merge at a time in Python. Here is the same algorithm at real scale: **100 MB of English** and thousands of merges, using Hugging Face's `tokenizers` library (written in Rust, so it trains in seconds).

**The data: TinyStories.** About 2.7 million short stories written by GPT-4 using only words a 3-to-4-year-old knows. It was made to show that tiny models (1M to 33M parameters) can learn to write fluent English, which is exactly the size we can train in Phase 1. We use the V2 file (GPT-4 only, cleaner than V1).

**The question this notebook answers:** GPT-2's tokenizer has 50,257 tokens. How many do we need for our data, and what does each token cost us in model parameters?""")
code("""import os, sys, subprocess, urllib.request, time
for pkg in ['tokenizers', 'tiktoken']:
    try: __import__(pkg)
    except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg])
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import tiktoken
print('tokenizers', __import__('tokenizers').__version__)""")
md("""## 1. The data
The full training file is 2.2 GB. A tokenizer does not need all of it: 100 MB (about 120k stories) is plenty to learn which character pairs are common. We download only the first 100 MB with an HTTP range request, and the separate 22 MB validation file to measure the result on stories the tokenizer never saw.""")
code("""BASE = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/'
os.makedirs('data', exist_ok=True)
def fetch(name, out, nbytes=None):
    if os.path.exists(out): return
    req = urllib.request.Request(BASE + name, headers={'Range': f'bytes=0-{nbytes - 1}'} if nbytes else {})
    t0 = time.time(); open(out, 'wb').write(urllib.request.urlopen(req).read())
    print(f'downloaded {out}: {os.path.getsize(out) / 1e6:.0f} MB in {time.time() - t0:.0f}s')
fetch('TinyStoriesV2-GPT4-train.txt', 'data/tinystories_train_100mb.txt', 100_000_000)
fetch('TinyStoriesV2-GPT4-valid.txt', 'data/tinystories_valid.txt')

def stories(path):
    parts = open(path, encoding='utf-8', errors='ignore').read().split('<|endoftext|>')
    return [s.strip() for s in parts[1:-1] if s.strip()]   # first and last pieces can be cut mid-story, drop them
train = stories('data/tinystories_train_100mb.txt')
val = stories('data/tinystories_valid.txt')
print(f'train: {len(train):,} stories | val: {len(val):,} stories | average story {sum(map(len, train)) // len(train)} characters\\n')
print(train[0])""")
md("""The files separate stories with the text `<|endoftext|>`. We split on it, and later give it its own special token so the model learns where one story ends and the next begins.

## 2. Train a 4,096-token BPE
Same steps as P0-01, now done by the library:
- **Byte-level:** text is turned into bytes first, so the 256 single bytes are always in the vocabulary. Any text can be encoded, even Hindi or emoji, and nothing becomes `<|unk|>`.
- **Pre-tokenizer:** split into words first (keeping the leading space, like GPT-2), so merges never cross word boundaries.
- **Merges:** repeatedly glue the most frequent adjacent pair into a new token, until the vocabulary reaches `vocab_size`.
- **Special tokens:** reserved ids that are never split. `<|endoftext|>` separates stories. `<|im_start|>` and `<|im_end|>` mark chat turns; we will not use them until SFT (P1-06), but they have to exist from the start because the vocabulary is fixed once the model trains. minimind puts the same three at ids 0, 1, 2.""")
code("""SPECIAL = ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
def train_bpe(texts, vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    return tok

t0 = time.time()
tok = train_bpe(train, 4096)
print(f'trained a {tok.get_vocab_size():,}-token BPE on {len(train):,} stories in {time.time() - t0:.0f}s')""")
md("""## 3. Look inside
The vocabulary is: 3 special tokens, then the 256 bytes, then 3,837 merges in the order they were learned. Byte-level BPE shows a space as `Ġ` and a newline as `Ċ`: those are just printable stand-ins for the raw bytes.""")
code("""vocab = tok.get_vocab_size()
show = lambda ids: ' '.join(repr(tok.id_to_token(i)) for i in ids)
print('special  :', show(range(0, 3)))
print('bytes    :', show(range(3, 13)), '...')
print('merge 1+ :', show(range(259, 279)))
print('merge 1k :', show(range(1259, 1274)))
print('last     :', show(range(vocab - 15, vocab)))
names = ['ĠLily', 'ĠTim', 'ĠBen', 'ĠMia', 'ĠLucy', 'ĠSara']
print('names    :', {n: tok.token_to_id(n) for n in names})""")
md("""The first merges are letter pairs and space-plus-letter (`Ġt`, `he`, `Ġa`), the same as you saw in P0-01. By merge 1,000 the tokens are whole words. The last tokens are rarer words that only just made the cut. Character names like ` Lily` and ` Tim` got their own token early (ids in the 300s to 900s) because TinyStories uses them constantly.

Now encode a sentence and decode it back. Byte-level BPE must give back **exactly** the original text, for every story.""")
code("""s = val[0][:160]
enc = tok.encode(s)
print(s, '\\n')
print(len(enc.ids), 'tokens:', ' | '.join(t.replace('Ġ', '·').replace('Ċ', '⏎') for t in enc.tokens))
print('\\nround trip exact on all val stories:', all(tok.decode(tok.encode(v).ids) == v for v in val))
odd = 'Lily said “hi” 👋 नमस्ते'
print(f'never-seen text: {odd!r} -> {len(tok.encode(odd).ids)} tokens -> {tok.decode(tok.encode(odd).ids)!r}')""")
md("""The emoji and the Hindi word are not in TinyStories, so they fall back to single bytes (many tokens, but nothing lost). If you ever want Hindi or Hinglish, the fix is to put some in the tokenizer's training text.

## 4. How big should the vocabulary be?
Train five tokenizers on the same stories and measure them on the **validation** stories:
- **characters per token:** higher means each token carries more text, so a context window holds more story.
- **embedding parameters:** every token needs a row in the embedding table, `vocab_size × emb_dim` numbers (tied with the output layer, so counted once). Our TinyStories model will be small, around `emb_dim` 384; minimind-3 uses 768.""")
code("""gpt2 = tiktoken.get_encoding('gpt2')
sample = val[:5000]; n_chars = sum(map(len, sample))
def measure(n_tokens, V, name):
    return {'name': name, 'vocab': V, 'chars/token': n_chars / n_tokens, 'tokens/story': n_tokens / len(sample),
            'emb@384': V * 384 / 1e6, 'emb@768': V * 768 / 1e6}
rows = []
for V in [512, 1024, 2048, 4096, 8192]:
    t = tok if V == 4096 else train_bpe(train, V)
    rows.append(measure(sum(len(e.ids) for e in t.encode_batch(sample)), V, f'ours {V}'))
rows.append(measure(sum(len(x) for x in gpt2.encode_ordinary_batch(sample)), gpt2.n_vocab, 'GPT-2'))
print(f"{'tokenizer':11s} {'vocab':>6s} {'chars/token':>12s} {'tokens/story':>13s} {'emb params d=384':>17s} {'d=768':>8s}")
for r in rows:
    print(f"{r['name']:11s} {r['vocab']:6,d} {r['chars/token']:12.2f} {r['tokens/story']:13.0f} {r['emb@384']:15.1f}M {r['emb@768']:7.1f}M")""")
code("""import matplotlib.pyplot as plt
ours = rows[:-1]; g = rows[-1]
fig, a1 = plt.subplots(figsize=(9, 5))
a1.plot([r['vocab'] for r in ours], [r['chars/token'] for r in ours], 'o-', color='tab:blue', lw=2.5, label='ours (trained on TinyStories)')
a1.scatter([g['vocab']], [g['chars/token']], color='tab:red', s=80, zorder=3, label='GPT-2 (trained on web text)')
a1.set_xscale('log', base=2); a1.set_xticks([r['vocab'] for r in ours + [g]], [f"{r['vocab']:,}" for r in ours + [g]]); a1.set_ylim(2.2, 4.3); a1.set_xlabel('vocabulary size (log scale)', fontsize=12); a1.set_ylabel('characters per token on TinyStories val', fontsize=12)
for r in ours + [g]:
    a1.annotate(f"{r['emb@384']:.1f}M emb params", (r['vocab'], r['chars/token']), textcoords='offset points', xytext=(-30, -18), fontsize=9)
a1.set_title('Compression flattens out fast; embedding cost keeps growing', fontsize=13); a1.legend(loc='lower right'); a1.grid(alpha=.3)
plt.tight_layout(); plt.savefig('p1_01_vocab_size.png', dpi=150); plt.show()""")
md("""**How to read it.** Going from 512 to 4,096 tokens makes stories far shorter. After that the curve flattens: TinyStories has a small vocabulary, so 4,096 tokens already covers nearly every common word. GPT-2's 50,257 tokens compress our stories about as well as our 4,096 (and our 8,192 actually beats GPT-2 here, because it was trained on this kind of text), yet at `emb_dim` 384 they would cost about 19M parameters in the embedding table alone. That is roughly the size of the whole model we plan to train.

This is minimind's reasoning for its 6,400-token vocabulary, and it is why we pick **4,096** for TinyStories.

## 5. The catch: a tokenizer only fits the text it was trained on
Our tokenizer has never seen science, code, or adult vocabulary. Compare it with GPT-2 on text that is not a children's story.""")
code("""texts = {
    'TinyStories val': val[1][:300],
    'science (web-style)': 'Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs mostly blue and red wavelengths, which is why leaves look green.',
    'Python code': 'def train(model, data):\\n    for x, y in data:\\n        loss = model(x).cross_entropy(y)\\n        loss.backward()',
}
print(f"{'text':22s} {'chars':>6s} {'ours 4k':>8s} {'GPT-2':>7s}")
for name, t in texts.items():
    print(f'{name:22s} {len(t):6d} {len(tok.encode(t).ids):8d} {len(gpt2.encode_ordinary(t)):7d}')
print('\\nscience, ours:', ' | '.join(tok.encode(texts['science (web-style)']).tokens[:20]))""")
md("""On stories the two are close. On science text and code ours needs noticeably more tokens, because words like ` Photosynthesis` or ` wavelengths` were never common enough to earn a merge. That is fine for a TinyStories model. When we later pretrain on general web text (FineWeb-Edu, P1-05b), we train a new tokenizer on that text.

## 6. Save it
One JSON file holds the whole tokenizer: vocabulary, merges, special tokens, pre-tokenizer and decoder. Training is deterministic, so rerunning this notebook gives the same file. A copy is committed in the repo at `tokenizers/tinystories-4k.json`, and later notebooks download that copy.""")
code("""tok.save('data/tinystories-4k.json')
tok2 = Tokenizer.from_file('data/tinystories-4k.json')
print(f"saved data/tinystories-4k.json ({os.path.getsize('data/tinystories-4k.json') / 1e3:.0f} KB)")
print('reloaded tokenizer gives same ids:', tok2.encode(val[2]).ids == tok.encode(val[2]).ids)
print('special token ids:', {s: tok2.token_to_id(s) for s in SPECIAL})
print('a story ends like this:', tok2.encode(val[3][-30:] + '<|endoftext|>').tokens[-6:])""")
md("""## Your turn
1. Train on only the first 1,000 stories instead of all of them. How do characters per token and the last learned tokens change? How much data does a tokenizer really need?
2. Count how many of the 4,096 tokens never appear when you encode the whole validation set. Those rows of the embedding table would never get trained.
3. Add a few paragraphs of Hindi or Hinglish to `train` and retrain. Does `नमस्ते` get shorter? What did it cost the English stories?
4. minimind-3 uses 6,400 tokens for Chinese plus English. Why does a bilingual model need a bigger vocabulary than a TinyStories one?

**Next: P1-02** upgrades our GPT-2 block to the modern design that minimind, Llama and Qwen use (RMSNorm, RoPE, SwiGLU, grouped-query attention, KV cache), and checks it against minimind-3's real weights.""")
nb.cells=C
nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nbf.write(nb,'P1-01-tokenizer.ipynb')
