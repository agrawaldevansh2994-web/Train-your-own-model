# P0-03 GPT block + full GPT (2026-10-24)
Notebook: notebooks/P0-03-gpt-block.ipynb (own code).
## What it does
Wraps attention with LayerNorm, a GELU feed-forward net and shortcut connections into a transformer block, stacks 12 of them into GPT-2 small (124M), and generates text greedily from random weights.
## The math / code
LN(x) = gamma*(x-mu)/sqrt(var+eps) + beta, per token over features.
FFN: Linear(768->3072) -> GELU -> Linear(3072->768).
Block (pre-LN): x += drop(att(ln1(x))); x += drop(ff(ln2(x))).
GPT: tok_emb + pos_emb -> dropout -> 12 blocks -> final LN -> out_head (logits over 50,257).
## Reference run (CPU)
- LayerNorm output: mean 0, var 1; matches nn.LayerNorm.
- FFN 4.72M params vs attention 2.36M; one block 7.09M.
- Layer-0 gradient: 0.0002 without shortcuts vs 0.22 with.
- Total 163.0M (tok_emb 38.6M, head 38.6M, attention 28.3M, FFN 56.7M); with weight tying 124.4M; 622 MB at fp32.
- Untrained greedy: 'Hello, I am Featureiman Byeswickattribute argue logger Normandy'.
- Untrained loss 11.04 vs ln(50257) = 10.82.
## What I observed
(fill in after running)
## Questions / next
P0-04 pretrain a shrunk GPT on The Verdict on CPU; temperature/top-k; checkpoint.
