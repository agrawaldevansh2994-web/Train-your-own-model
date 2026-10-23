# P0-01 Tokenizer (2026-10-23)
Notebook: notebooks/P0-01-tokenizer.ipynb (own code, data = The Verdict, public domain, via rasbt raw URL).
## What it does
Text -> integer ids. Word tokenizer (breaks on unseen words), byte-level BPE trained from scratch, GPT-2 tiktoken comparison, next-token (x, y) pairs, token + position embeddings.
## The math / code
BPE: count adjacent pairs, merge the most frequent into a new id, repeat. Vocab = 256 + merges. Encoding applies merges in learned order.
Embedding: lookup table (vocab, d_model); input = tok_emb[ids] + pos_emb[0..T-1] -> (B, T, d_model).
## Reference run (CPU)
- Story 20,479 chars; word vocab 1,132; "Hello" -> <|unk|>.
- 300 merges: first 'e ', ' t', 'd '; by merge 14 ' the '; 2.28 bytes/token. GPT-2: 3.98 bytes/token (5,145 tokens vs our 8,980).
- Hindi + emoji round-trip exactly in our BPE (raw bytes); GPT-2 also splits Hindi into many byte tokens.
- Dataset ctx 4 stride 4: 1,286 samples. GPT-2 token table at d_model 256 = 12.87M params.
## What I observed
(fill in after running)
## Questions / next
P0-02 attention.
