# Train Your Own Model

Building my own AI end to end and learning every step: tokenizer, pretraining, fine-tuning, RL, evals, serving, and an agent on top.

| Phase | What | Based on | Status |
|---|---|---|---|
| 0 | GPT from scratch on CPU: tokenizer, attention, GPT block | own notebooks, after rasbt/LLMs-from-scratch | done |
| 1 | Train an English LLM from zero (TinyStories ~16M, then FineWeb-Edu 64M): pretrain, SFT, LoRA, DPO, GRPO | jingyaogong/minimind architecture, own code | in progress |
| 2 | ChatGPT-style pipeline with evals and web chat | karpathy/nanochat | |
| 3 | Fine-tune open weights on my own data, run locally (GGUF/Ollama) | unslothai/unsloth | |
| 4 | Tool-calling agent served behind an API | | |

## Notebooks
| Step | Topic | Colab | Kaggle |
|---|---|---|---|
| P0-01 | Tokenizer from scratch: word tokenizer, byte-level BPE, GPT-2 comparison, next-token pairs, embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P0-01-tokenizer.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P0-01-tokenizer.ipynb) |
| P0-02 | Attention from scratch: dot-product, Q/K/V, scaling, causal mask, dropout, multi-head | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P0-02-attention.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P0-02-attention.ipynb) |
| P0-03 | GPT block + full GPT-2 (124M): LayerNorm, GELU FFN, residuals, param count, first generation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P0-03-gpt-block.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P0-03-gpt-block.ipynb) |
| P0-04 | Pretraining: train a 16M GPT on a short story, loss curves, overfitting, temperature/top-k, checkpoints | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P0-04-pretraining.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P0-04-pretraining.ipynb) |
| P0-04b | More data (optional, GPU): same 16M model trained on 5k, 50k and 300k tokens of Tiny Shakespeare, showing data is the cure for overfitting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P0-04b-more-data.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P0-04b-more-data.ipynb) |
| P0-05 | Load OpenAI's GPT-2 124M weights into your own model, verify against Hugging Face, compare with P0-04 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P0-05-load-gpt2-weights.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P0-05-load-gpt2-weights.ipynb) |
| P1-01 | Train a 4,096-token BPE on TinyStories, vocab size vs compression vs embedding cost, compare with GPT-2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P1-01-tokenizer.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P1-01-tokenizer.ipynb) |
| P1-02 | The modern block (RMSNorm, RoPE, SwiGLU, GQA, QK-norm, KV cache), verified by running minimind-3's weights in your own code | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P1-02-modern-block.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/agrawaldevansh2994-web/Train-your-own-model/blob/main/notebooks/P1-02-modern-block.ipynb) |

Kaggle: turn on Internet in notebook settings (needs a phone-verified account). Training runs log loss curves with [Trackio](https://github.com/gradio-app/trackio) (free, local, W&B-compatible); run `trackio.show()` to see them.

`src/mygpt.py` holds the model built in P0-01 to P0-03 in one file; P0-04 onward import it. `src/mymind.py` is the modern (minimind-3 / Qwen3-shaped) model from P1-02; Phase 1 imports it. `tokenizers/` holds tokenizers trained in Phase 1.

## Layout
- `notebooks/` runnable notebooks (each regenerated by its `build_*.py`)
- `lab-notes/` one note per step: what it does, the math, what I observed
- `curriculum.md` phases and status, `costs.md` compute options and costs

Training compute: Google Colab Pro and Kaggle. Laptop (CPU only) for Phase 0 and local inference.
