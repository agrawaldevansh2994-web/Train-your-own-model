# P1-01 Train a real tokenizer (2026-11-07)
Notebook: notebooks/P1-01-tokenizer.ipynb (built by build_p1_01.py). Output: tokenizers/tinystories-4k.json.
## What it does
Trains a byte-level BPE (Hugging Face `tokenizers`, Rust) on the first 100 MB of TinyStoriesV2-GPT4 (122k stories). Special tokens `<|endoftext|>`, `<|im_start|>`, `<|im_end|>` at ids 0-2 (same as minimind). Sweeps vocab 512 to 8,192 and compares with GPT-2's 50,257 on the 22 MB validation file.
## The math / code
Same algorithm as P0-01: bytes first (256 base tokens, nothing is ever unknown), split into words, then repeatedly merge the most frequent adjacent pair. Embedding cost = vocab_size x emb_dim (tied with the output head, so counted once).
## Reference run (laptop CPU, 2026-11-07, about 3 min)
| tokenizer | chars/token on val | tokens/story | emb params at d=384 |
|---|---|---|---|
| ours 512 | 2.41 | 331 | 0.2M |
| ours 1,024 | 3.17 | 251 | 0.4M |
| ours 2,048 | 3.67 | 217 | 0.8M |
| ours 4,096 | 3.97 | 201 | 1.6M |
| ours 8,192 | 4.10 | 194 | 3.1M |
| GPT-2 50,257 | 4.06 | 196 | 19.3M |
Off-domain: science sentence 54 tokens (ours) vs 27 (GPT-2); Python snippet 73 vs 52. Training 4,096 takes about 25 s. Deterministic: same data gives the same file (md5 checked twice).
## What I observed
(fill in after running)
## Questions / next
P1-02 the modern block.
