# P0-05 Load real GPT-2 weights (2026-10-25)
Notebook: notebooks/P0-05-load-gpt2-weights.ipynb. Loader: `load_gpt2_weights` in src/mygpt.py.
## What it does
Downloads OpenAI GPT-2 124M (Hugging Face `model.safetensors`, 548 MB, cached in Drive), maps its 148 weight tensors onto our GPTModel, verifies against Hugging Face's implementation, and compares it with the P0-04 model.
## The math / code
- HF names map onto ours: wte/wpe -> tok_emb/pos_emb, h.i -> blocks[i], ln_1/ln_2/ln_f -> norm1/norm2/final_norm, c_attn -> W_q/W_k/W_v, c_proj -> out_proj, mlp.c_fc/c_proj -> ff.net[0]/[2].
- Conv1D stores weights as (in, out), so transpose them. c_attn (768, 2304) packs Q|K|V, so chunk it into 3. GPT-2 needs qkv_bias=True. Output head is tied to wte.
- Every copy asserts its shape, so a bad mapping fails loudly.
## Reference run (CPU)
- 124.4M params after loading. Max |logit diff| vs HF GPT2LMHeadModel 3.24e-04: MATCH.
- 'Every effort moves you' -> random: 'aminsamins flowing DON petition...'; GPT-2: ' forward. The first step is to understand the importance of your work'.
- 'I drank a cup of' -> coffee 0.27, tea 0.16. 'Mumbai is a city in' -> which 0.24, India 0.20.
- Val loss on The Verdict's held-out 10%: GPT-2 3.74 (perplexity 42) vs P0-04 model ~6.7 after 1 epoch. Same architecture, the difference is data.
- Greedy decoding repeats itself; T=1.2 drifts.
## What I observed
(fill in: your P0-04 vs GPT-2 val loss from section 6)
## Next
Phase 1: minimind end to end on a Colab GPU.
