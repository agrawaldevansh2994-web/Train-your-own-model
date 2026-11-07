# P1-02 The modern block (2026-11-07)
Notebook: notebooks/P1-02-modern-block.ipynb (built by build_p1_02.py). Model code: src/mymind.py.
## What it does
Upgrades the GPT-2 block from Phase 0 to the design used by minimind-3, Llama 3 and Qwen3, one piece at a time, then proves it by loading minimind-3's released weights (jingyaogong/minimind-3, 128 MB fp16) and matching Hugging Face's Qwen3ForCausalLM.
## The math / code
- RMSNorm: x / sqrt(mean(x^2) + eps) * w. No mean subtraction, no shift.
- RoPE: rotate each pair (i, i + d/2) of q and k by pos * theta^(-2i/d). q.k then depends only on distance m - n. 0 parameters.
- SwiGLU: down(silu(gate(x)) * up(x)), width ceil(pi * d / 64) * 64 = 2,432 at d = 768.
- GQA: 8 query heads share 4 k/v heads (repeat_interleave). Halves the KV cache.
- QK-norm: RMSNorm on each head's q and k before RoPE. Keeps scores bounded.
- No biases, tied embeddings, KV cache in generate().
## Reference run (laptop CPU, 2026-11-07, about 1 min after the download)
- minimind-3 config: 63,912,192 params (GPT-2-style model of the same width/depth/vocab: 63,192,576).
- Ours vs HF Qwen3ForCausalLM: max logit diff 0.0 (bit-exact; perturbing one weight by 0.1 gives 0.084, so the check is real). Vs minimind's own model_minimind.py (manual attention path): 3.5e-05.
- Greedy answers: "The capital of France is Paris." / sky: starts a correct Rayleigh-scattering explanation / sea: loops ("a vast and endlessly embrace").
- KV cache: identical tokens, 200 tokens in about 4 s vs 15 s without (3.6-4x on this laptop).
- TinyStories-size config (vocab 4096, d 384, 8 layers, 6 q / 2 kv heads, ctx 512): 15.9M params, embedding 10%.
## What I observed
(fill in after running)
## Questions / next
P1-03 data pipeline: TinyStories to token ids on disk.
