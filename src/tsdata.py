"""tsdata.py: the TinyStories data pipeline from P1-03 in one importable file (fetch, tokenize_file, prepare, get_batch).
P1-03 shows this code cell by cell; later notebooks import it. Keep the two in sync."""
import os, time, json, codecs, urllib.request, urllib.error
import numpy as np, torch

BASE = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/'
REPO = 'https://raw.githubusercontent.com/agrawaldevansh2994-web/Train-your-own-model/main/'
SEP = '<|endoftext|>'

def fetch(url, out, nbytes=None, tries=30):
    """Resumable download: keeps out + '.part' and continues from its end after a dropped connection."""
    if os.path.exists(out): return
    part, t0 = out + '.part', time.time()
    for attempt in range(tries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if nbytes and have >= nbytes: break
        last = f'{nbytes - 1}' if nbytes else ''
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': f'bytes={have}-{last}'})) as r, open(part, 'ab') as f:
                expect = have + int(r.headers['Content-Length'])
                while chunk := r.read(1 << 22): f.write(chunk)
            if os.path.getsize(part) >= expect: break
        except urllib.error.HTTPError as e:
            if e.code == 416: break
            print('http error', e.code)
        except (OSError, ConnectionError) as e:
            print(f'connection dropped at {os.path.getsize(part) / 1e6:.0f} MB ({type(e).__name__}); resuming ({attempt + 1}/{tries})')
        time.sleep(3)
    else:
        raise RuntimeError(f'could not finish {url}; run again to resume')
    os.replace(part, out)
    print(f'downloaded {out}: {os.path.getsize(out) / 1e6:.0f} MB in {time.time() - t0:.0f}s')

def load_tokenizer(data_dir='data'):
    from tokenizers import Tokenizer
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tokenizers', 'tinystories-4k.json')
    if not os.path.exists(path):
        path = os.path.join(data_dir, 'tinystories-4k.json')
        if not os.path.exists(path): urllib.request.urlretrieve(REPO + 'tokenizers/tinystories-4k.json', path)
    return Tokenizer.from_file(path)

def tokenize_file(tok, txt_path, bin_path, skip_first_piece=False, chunk_bytes=64_000_000):
    """Stories -> one uint16 stream, <|endoftext|> after each. Returns (tokens, stories, utf-8 bytes of story text)."""
    eot = tok.token_to_id(SEP)
    n_tok = n_story = n_bytes = 0; carry = ''; first = True; t0 = time.time()
    utf8 = codecs.getincrementaldecoder('utf-8')(errors='replace')
    with open(txt_path, 'rb') as src, open(bin_path, 'wb') as dst:
        while True:
            raw = src.read(chunk_bytes)
            pieces = (carry + utf8.decode(raw, final=not raw)).split(SEP)
            carry = pieces.pop()
            if first and skip_first_piece: pieces = pieces[1:]
            first = False
            stories = [p.strip() for p in pieces if p.strip()]
            if stories:
                ids = [i for e in tok.encode_batch(stories) for i in (*e.ids, eot)]
                np.asarray(ids, dtype=np.uint16).tofile(dst)
                n_tok += len(ids); n_story += len(stories); n_bytes += sum(len(s.encode('utf-8')) for s in stories)
            if not raw: break
    print(f'{bin_path}: {n_story:,} stories -> {n_tok:,} tokens in {time.time() - t0:.0f}s')
    return n_tok, n_story, n_bytes

def prepare(data_dir='data', train_mb=None):
    """Download + tokenize TinyStoriesV2-GPT4 once. train_mb=None: the whole 2.2 GB; a number: only the first MB.
    Returns (train_bin, val_bin, meta). Skips work whose output already exists."""
    os.makedirs(data_dir, exist_ok=True)
    suffix = 'full' if train_mb is None else f'{train_mb}mb'
    train_txt, train_bin = f'{data_dir}/ts_train_{suffix}.txt', f'{data_dir}/ts_train_{suffix}.bin'
    val_txt, val_bin, meta_path = f'{data_dir}/ts_valid.txt', f'{data_dir}/ts_val.bin', f'{data_dir}/ts_meta_{suffix}.json'
    if os.path.exists(meta_path) and os.path.exists(train_bin) and os.path.exists(val_bin):
        return train_bin, val_bin, json.load(open(meta_path))
    tok = load_tokenizer(data_dir)
    fetch(BASE + 'TinyStoriesV2-GPT4-valid.txt', val_txt)
    fetch(BASE + 'TinyStoriesV2-GPT4-train.txt', train_txt, None if train_mb is None else train_mb * 1_000_000)
    n_val, s_val, b_val = tokenize_file(tok, val_txt, val_bin, skip_first_piece=True)
    n_train, s_train, _ = tokenize_file(tok, train_txt, train_bin)
    meta = {'train_tokens': n_train, 'val_tokens': n_val, 'val_stories': s_val, 'val_bytes': b_val,
            'bytes_per_token_val': b_val / n_val, 'vocab_size': tok.get_vocab_size()}   # per token INCLUDING separators: what bits/byte needs
    json.dump(meta, open(meta_path, 'w'))
    return train_bin, val_bin, meta

def save_dir():
    """Where checkpoints go: Google Drive on Colab (survives the session), /kaggle/working on Kaggle, checkpoints/ elsewhere."""
    try:
        from google.colab import drive
        drive.mount('/content/drive'); d = '/content/drive/MyDrive/train-your-own-model'
    except ImportError:
        d = '/kaggle/working' if os.path.isdir('/kaggle/working') else 'checkpoints'
    os.makedirs(d, exist_ok=True)
    return d

def find_checkpoint(fname):
    """Look for a checkpoint from an earlier notebook in all the usual places. Returns a path or None."""
    import glob
    for c in [os.path.join(save_dir(), fname), *glob.glob(f'/kaggle/input/*/{fname}'), *glob.glob(f'/kaggle/input/*/*/{fname}'),
              f'checkpoints/{fname}', f'../checkpoints/{fname}', fname]:
        if os.path.exists(c): return c
    return None

def val_stories(data_dir='data'):
    """The validation stories as text (first piece dropped: the published file starts mid-story)."""
    os.makedirs(data_dir, exist_ok=True)
    path = f'{data_dir}/ts_valid.txt'
    fetch(BASE + 'TinyStoriesV2-GPT4-valid.txt', path)
    return [s.strip() for s in open(path, encoding='utf-8', errors='replace').read().split(SEP)[1:-1] if s.strip()]

class Batches:
    """Random (x, y) windows from a uint16 token file via np.memmap. y is x shifted by one."""
    def __init__(self, path, seed=0):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.rng = np.random.default_rng(seed)
    def get(self, B, T, device='cpu'):
        i = self.rng.integers(0, len(self.data) - T - 1, size=B)
        x = torch.from_numpy(np.stack([self.data[j:j + T] for j in i]).astype(np.int64))
        y = torch.from_numpy(np.stack([self.data[j + 1:j + 1 + T] for j in i]).astype(np.int64))
        if device == 'cuda':
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)
    def fixed(self, n, T):
        """n consecutive non-overlapping windows from the start: the same eval set every time."""
        n = min(n, (len(self.data) - 1) // T)
        x = torch.from_numpy(self.data[:n * T].astype(np.int64)).view(n, T)
        y = torch.from_numpy(self.data[1:n * T + 1].astype(np.int64)).view(n, T)
        return x, y
