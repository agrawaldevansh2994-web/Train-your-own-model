"""sftdata.py: the TinyStories-Instruct SFT data code from P1-06 in one importable file.
P1-06 shows this code cell by cell; P1-07 onward import it. Keep the two in sync."""
import os, re, numpy as np, torch
import tsdata
from mymind import generate

BASE = 'https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/'
FIELD = re.compile(r'^(Features|Words|Summary|Random sentence|Story):', re.M)
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
    return '\n'.join(lines)

def prompt_text(d):
    return f"<|im_start|>user\n{instruction(d)}<|im_end|>\n<|im_start|>assistant\n"

def load_examples(data_dir='data', train_mb=200):
    os.makedirs(data_dir, exist_ok=True)
    tr, va = f'{data_dir}/tsi_train_{train_mb}mb.txt', f'{data_dir}/tsi_valid.txt'
    tsdata.fetch(BASE + 'TinyStories-Instruct-train.txt', tr, train_mb * 1_000_000)
    tsdata.fetch(BASE + 'TinyStories-Instruct-valid.txt', va)
    load = lambda p: [d for d in map(parse, open(p, encoding='utf-8', errors='replace').read().split('<|endoftext|>')[1:-1]) if d]
    return load(tr), load(va)

def encode(tok, d):
    im_end = tok.token_to_id('<|im_end|>')
    return tok.encode(prompt_text(d)).ids, tok.encode(d['Story']).ids + [im_end]

def to_examples(tok, exs, max_len):
    out = []
    for d in exs:
        p, a = encode(tok, d)
        if len(p) + len(a) <= max_len: out.append((p, a))
    return out

def collate(batch, device, pad_id):
    """Pad to the longest; targets are -100 except the answer tokens (loss mask)."""
    L = max(len(p) + len(a) for p, a in batch)
    x = torch.full((len(batch), L), pad_id, dtype=torch.long); y = torch.full((len(batch), L), -100, dtype=torch.long)
    for r, (p, a) in enumerate(batch):
        seq = p + a
        x[r, :len(seq)] = torch.tensor(seq)
        y[r, len(p) - 1:len(seq) - 1] = torch.tensor(a)
    return x.to(device), y.to(device)

def words_of(d):
    return [w.strip().lower() for w in d.get('Words', '').split(',') if w.strip()]

@torch.no_grad()
def write(model, tok, d, device, temperature=0.7, n=300, seed=0):
    torch.manual_seed(seed)
    ids = torch.tensor([tok.encode(prompt_text(d)).ids], device=device)
    out = generate(model, ids, n, temperature=temperature, top_k=40, eos_id=tok.token_to_id('<|im_end|>'))
    return tok.decode(out[0, ids.shape[1]:].tolist())

def word_score(model, tok, evalset, device):
    """Share of stories that contain every required word (substring match, so 'jumped' counts for 'jump')."""
    hits = [all(w in write(model, tok, d, device, seed=i).lower() for w in words_of(d)) for i, d in enumerate(evalset)]
    return float(np.mean(hits))
