# P0-02 Attention (2026-10-23)
Notebook: notebooks/P0-02-attention.ipynb (own code).
## What it does
Lets each token mix in information from other tokens: weightless dot-product attention, learnable Q/K/V, causal mask, dropout, multi-head attention.
## The math / code
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V. Q = xW_q, K = xW_k, V = xW_v.
Causal mask: scores above the diagonal set to -inf before softmax, so the future gets zero weight.
Multi-head: project to d_out, reshape (B,T,d_out) -> (B,h,T,hd), batched matmul, concat, out_proj.
## Reference run (CPU)
- 'journey' attends most to itself (0.238) and 'starts' (0.233).
- Random 512-d vectors: raw score std 15.8, softmax max 0.998 (one-hot); scaled max 0.26.
- Causal matrix is lower-triangular; row 0 = [1, 0, ...].
- GPT-2 small attention block (768, 12 heads) = 2.36M params; x12 layers = 28.3M.
- Our implementation matches F.scaled_dot_product_attention(is_causal=True).
## What I observed
(fill in after running)
## Questions / next
P0-03 GPT block (LayerNorm, GELU FFN, residuals, generate).
