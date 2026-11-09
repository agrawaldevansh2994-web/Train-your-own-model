import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# P1-03 · Data pipeline: from 2.2 GB of text to a file of token ids
**Build Your Own AI · Phase 1, step 3.** Laptop CPU. Tokenizing the 2.2 GB file took about 11 minutes on our laptop; the download took anywhere from 2 to 40 minutes depending on the connection (the cell resumes if the connection drops). Needs about 3.5 GB of free disk. To try it fast, set `TRAIN_MB = 200` in the first data cell.

Training reads token ids, not text. Tokenizing 2.2 GB of stories on every run would waste time, so we do it **once** and save the ids to disk. The training loop then only does one cheap thing: cut random windows out of that file.

Three decisions this notebook makes:
- **One long stream of ids**, stories separated by the `<|endoftext|>` token (id 0). No padding, so every position in every batch is real data.
- **`uint16` on disk.** Our vocabulary is 4,096, and one `uint16` holds up to 65,535, so each token takes 2 bytes. (GPT-2's 50,257 vocab also fits; a vocab above 65,535 would need `uint32`.)
- **`np.memmap`.** The file is opened without being loaded into RAM: the OS reads only the windows we ask for. The 1 GB file costs almost no memory.""")
code("""import os, sys, subprocess, urllib.request, time, json, codecs
for pkg in ['tokenizers', 'numpy']:
    try: __import__(pkg)
    except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg])
import numpy as np, torch
from tokenizers import Tokenizer
REPO = 'https://raw.githubusercontent.com/agrawaldevansh2994-web/Train-your-own-model/main/'
os.makedirs('data', exist_ok=True)
tok_path = '../tokenizers/tinystories-4k.json'
if not os.path.exists(tok_path):
    tok_path = 'data/tinystories-4k.json'
    if not os.path.exists(tok_path): urllib.request.urlretrieve(REPO + 'tokenizers/tinystories-4k.json', tok_path)
tok = Tokenizer.from_file(tok_path)
EOT = tok.token_to_id('<|endoftext|>')
print('vocab', tok.get_vocab_size(), '| end-of-story id', EOT, '| fits in uint16:', tok.get_vocab_size() < 2**16)""")
md("""## 1. Download
The tokenizer is the one P1-01 trained. We need the full training file (2.2 GB) and the validation file (22 MB). `TRAIN_MB = None` means everything; set a number to download only the first part (a quick test).""")
code("""BASE = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/'
TRAIN_MB = None          # None = the whole 2.2 GB file; e.g. 200 = first 200 MB only
def fetch(name, out, nbytes=None, tries=30):
    # resumable: a 2 GB download over home wifi WILL get cut sometimes, so keep the .part file and continue from its end
    if os.path.exists(out): print('have', out, f'({os.path.getsize(out) / 1e6:.0f} MB)'); return
    part, t0 = out + '.part', time.time()
    for attempt in range(tries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if nbytes and have >= nbytes: break
        last = f'{nbytes - 1}' if nbytes else ''
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE + name, headers={'Range': f'bytes={have}-{last}'})) as r, open(part, 'ab') as f:
                expect = have + int(r.headers['Content-Length'])
                while chunk := r.read(1 << 22): f.write(chunk)
            if os.path.getsize(part) >= expect: break              # got everything the server promised
        except urllib.error.HTTPError as e:
            if e.code == 416: break                                  # nothing left to ask for: the file is complete
            print('http error', e.code)
        except (OSError, ConnectionError) as e:
            print(f'connection dropped at {os.path.getsize(part) / 1e6:.0f} MB ({type(e).__name__}); resuming ({attempt + 1}/{tries})')
        time.sleep(3)
    else:
        raise RuntimeError(f'could not finish {name}; run this cell again to resume from the .part file')
    os.replace(part, out)
    print(f'downloaded {out}: {os.path.getsize(out) / 1e6:.0f} MB in {time.time() - t0:.0f}s')
suffix = 'full' if TRAIN_MB is None else f'{TRAIN_MB}mb'
fetch('TinyStoriesV2-GPT4-train.txt', f'data/ts_train_{suffix}.txt', None if TRAIN_MB is None else TRAIN_MB * 1_000_000)
fetch('TinyStoriesV2-GPT4-valid.txt', 'data/ts_valid.txt')""")
md("""## 2. Tokenize and write the `.bin` file
The text file is 2.2 GB, too big to tokenize in one go comfortably, so we read it in 64 MB chunks. A chunk boundary will usually fall in the middle of a story, so each chunk is split on `<|endoftext|>` and the unfinished last piece is **carried over** to the start of the next chunk. The same can happen inside a character: a curly quote `“` is 3 bytes, and a chunk can end after the first one. The incremental UTF-8 decoder keeps those bytes and finishes the character with the next chunk.

For every complete story we encode it and append the `<|endoftext|>` id, then write the ids as `uint16` bytes. `encode_batch` tokenizes many stories in parallel (Rust, all cores).""")
code("""SEP = '<|endoftext|>'
def tokenize_file(txt_path, bin_path, skip_first_piece=False, chunk_bytes=64_000_000):
    n_tok = n_story = n_chars = 0; carry = ''; first = True; t0 = time.time()
    utf8 = codecs.getincrementaldecoder('utf-8')(errors='replace')   # holds a character cut in half by the chunk edge until the next chunk
    with open(txt_path, 'rb') as src, open(bin_path, 'wb') as dst:
        while True:
            raw = src.read(chunk_bytes)
            text = carry + utf8.decode(raw, final=not raw)
            pieces = text.split(SEP)
            carry = pieces.pop()                                    # unfinished story: wait for the next chunk
            if first and skip_first_piece: pieces = pieces[1:]; first = False   # the valid file starts mid-story
            first = False
            stories = [p.strip() for p in pieces if p.strip()]
            if stories:
                ids = [i for e in tok.encode_batch(stories) for i in (*e.ids, EOT)]
                np.asarray(ids, dtype=np.uint16).tofile(dst)
                n_tok += len(ids); n_story += len(stories); n_chars += sum(map(len, stories))
            if not raw: break
    dt = time.time() - t0
    print(f'{bin_path}: {n_story:,} stories -> {n_tok:,} tokens | {n_tok / n_story:.0f} tokens/story | {n_chars / n_tok:.2f} chars/token | {dt:.0f}s ({n_tok / dt / 1e6:.2f}M tokens/s)')
    return n_tok, n_story

train_bin, val_bin = f'data/ts_train_{suffix}.bin', 'data/ts_val.bin'
n_val, s_val = tokenize_file('data/ts_valid.txt', val_bin, skip_first_piece=True)
n_train, s_train = tokenize_file(f'data/ts_train_{suffix}.txt', train_bin)""")
md("""The validation file starts in the middle of a story in the published data (we saw this in P1-01), so its first piece is dropped. For the training file with `TRAIN_MB` set, the last unfinished story is dropped by the same carry logic.

## 3. Check it
Three checks before we trust half a billion numbers.

**Round trip.** Split the validation ids back into stories at every `<|endoftext|>`, decode each, and compare with the original text. It must match for every story.""")
code("""val_ids = np.fromfile(val_bin, dtype=np.uint16)
print('file size', f'{os.path.getsize(val_bin):,} bytes = 2 x {len(val_ids):,} tokens | max id {val_ids.max()} (vocab {tok.get_vocab_size()}) | ids are valid: {val_ids.max() < tok.get_vocab_size()}')
cuts = np.flatnonzero(val_ids == EOT)
starts = np.r_[0, cuts[:-1] + 1]
orig = [s.strip() for s in open('data/ts_valid.txt', encoding='utf-8', errors='ignore').read().split(SEP)[1:-1] if s.strip()]
n_check = len(orig)
ok = sum(tok.decode(val_ids[a:b].tolist()) == o for a, b, o in zip(starts[:n_check], cuts[:n_check], orig[:n_check]))
print(f'stories decoded back to the exact original text: {ok} of {n_check}')
print('stories in file:', len(cuts), '| stories in text:', len(orig))""")
md("""**Chunk edges.** The validation file is smaller than one 64 MB chunk, so the check above never crossed a chunk edge. Tokenize it again in 1 MB chunks (23 edges, cutting stories and characters) and demand the identical file.""")
code("""tokenize_file('data/ts_valid.txt', 'data/ts_val_1mb_chunks.bin', skip_first_piece=True, chunk_bytes=1_000_000)
same = np.array_equal(np.fromfile('data/ts_val_1mb_chunks.bin', dtype=np.uint16), val_ids)
print('1 MB chunks give the identical token file:', same); os.remove('data/ts_val_1mb_chunks.bin')""")
md("""**By eye.** The first 60 ids of the file and what they decode to. Id 0 is the story separator.""")
code("""print(val_ids[:60].tolist())
print(tok.decode(val_ids[:60].tolist(), skip_special_tokens=False))""")
md("""## 4. The batch loader
`np.memmap` gives an array that looks like it is in memory but is read from disk on demand. A **batch** is `B` random windows of `T` tokens, with the target being the same window shifted by one token (exactly the input/target pairs of P0-01, but now random windows instead of fixed chunks).

Windows are cut anywhere, so one will often start in the middle of a story and run over a story boundary. That is fine: the model sees `<|endoftext|>` and learns that a new, unrelated story starts after it.""")
code("""def get_batch(split, B, T, device='cpu', rng=np.random.default_rng()):
    data = np.memmap(train_bin if split == 'train' else val_bin, dtype=np.uint16, mode='r')
    i = rng.integers(0, len(data) - T - 1, size=B)
    x = torch.from_numpy(np.stack([data[j:j + T] for j in i]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[j + 1:j + 1 + T] for j in i]).astype(np.int64))
    return x.to(device), y.to(device)

x, y = get_batch('train', 4, 24)
print('x', tuple(x.shape), x.dtype, '| y is x shifted by one:', bool((x[:, 1:] == y[:, :-1]).all()))
for r in range(2):
    print(f'\\nwindow {r}: ' + repr(tok.decode(x[r].tolist(), skip_special_tokens=False)))
big_x, _ = get_batch('train', 256, 512)
print('\\nwindows (of 256 sampled) that contain a story boundary:', int((big_x == EOT).any(1).sum()))
t0 = time.time()
for _ in range(20): get_batch('train', 32, 512)
print(f'20 batches of 32 x 512 tokens: {time.time() - t0:.2f}s, about {(time.time() - t0) / 20 * 1000:.0f} ms per batch')""")
md("""Loading a batch takes a few milliseconds, so the GPU will never wait for data.

## 5. How much data is enough?
Two numbers for the model we plan to train in P1-04 (15.9M parameters, from P1-02):
- the **Chinchilla rule of thumb**: about 20 training tokens per parameter for a compute-efficient run;
- **bytes per token**, which P1-04 needs to report loss as *bits per byte*. A loss in nats per token is only comparable between models with the same tokenizer; bits per byte is comparable across tokenizers (including TinyStories-33M's GPT-Neo tokenizer).""")
code("""n_params = 15.9e6
bytes_val = sum(len(o.encode('utf-8')) for o in orig)
bytes_per_token = bytes_val / (len(val_ids) - len(cuts))        # bytes of text per real token (the separators are not text)
print(f'validation: {len(val_ids):,} tokens, {bytes_val / 1e6:.1f} MB of text, {bytes_per_token:.2f} bytes per token')
print(f'training  : {n_train:,} tokens in the file\\n')
print(f"{'tokens seen in training':26s} {'epochs of this file':>20s} {'tokens per parameter':>22s}")
for name, tks in [('20M (laptop CPU, 4 h)', 20e6), ('Chinchilla: 20 per param', 20 * n_params), ('300M (a T4 run, guess)', 300e6), ('the whole file once', n_train)]:
    print(f'{name:26s} {tks / n_train:20.2f} {tks / n_params:22.1f}')
json.dump({'train_tokens': n_train, 'val_tokens': int(len(val_ids)), 'bytes_per_token_val': bytes_per_token, 'vocab_size': tok.get_vocab_size()}, open('data/ts_meta.json', 'w'))
print('\\nsaved data/ts_meta.json')""")
md("""The file holds more than a compute-efficient run for this model needs, so the run covers only part of it: most stories are seen once or not at all (windows are drawn at random, so a few get picked twice). That is what P0-04b taught: with enough data there is nothing to memorise, and train and validation loss stay close.

On this laptop, training measured about 1,350 tokens per second (batch 8, length 512). At that speed, even 20M tokens takes about 4 hours. That is why P1-04 does its real run on a GPU.

## Your turn
1. Time `tokenize_file` on the validation file with `encode_batch` replaced by a plain loop `[tok.encode(s) for s in stories]`. How much slower is it?
2. `get_batch` reopens the memmap on every call. Move it outside the function. Does the batch time change? (Hint: the first call after a restart is slow because the OS must read the file; later calls hit its cache.)
3. Remove the `EOT` token from the stream (write only the story ids). Decode a window. Can you still see where one story ends? What would a model trained on this write after "The end."?
4. Change `dtype` to `np.uint8`. What happens, and why does a vocabulary of 4,096 not fit?

**Next: P1-04** trains the 15.9M model on this file. First a short CPU run to check that the loss falls, then the real run on a free Kaggle or Colab T4.""")
nb.cells=C
nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nbf.write(nb,'P1-03-data-pipeline.ipynb')
