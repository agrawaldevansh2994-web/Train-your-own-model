# Train Your Own Model

Devansh's learning repo: build an LLM end to end (tokenizer → pretraining → SFT/LoRA/DPO/GRPO → evals → serving → agent). Phases and status in `curriculum.md`, compute pricing in `costs.md`. Remote: github.com/agrawaldevansh2994-web/Train-your-own-model (branch `main`). GitHub is the source of truth; work done in other sessions is pushed there, so `git pull` before starting.

**Next session: read `HANDOFF.md` first** (Phase 0 is done; it says how to start Phase 1).

## Layout
- `notebooks/build_pN_XX.py` → generates `notebooks/PN-XX-*.ipynb` (Phase 0 and 1). **Edit the build script, never the .ipynb by hand**, then regenerate: `cd notebooks && python build_p1_XX.py`. Committed notebooks have no outputs.
- `src/mygpt.py`: the model from P0-01..P0-03 in one file (MultiHeadAttention, LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, generate, load_gpt2_weights). P0-04+ import it; on Colab/Kaggle they download it from the raw GitHub URL, locally they use `../src`. Changing it changes every later notebook: keep it in sync with the classes shown in P0-02/P0-03.
- `src/mymind.py`: the modern model from P1-02 (RMSNorm, RoPE, SwiGLU, GQA, QK-norm, KV cache; MindModel, generate, load_minimind_weights, MIND_CONFIG_64M). Module names follow Llama/Qwen (q_proj, gate_proj...). Verified bit-exact against HF Qwen3ForCausalLM on jingyaogong/minimind-3. Phase 1 notebooks import it the same way as mygpt.
- `tokenizers/*.json`: tokenizers trained in Phase 1 (small, committed; later notebooks download them). `notebooks/data/` holds downloaded datasets (gitignored).
- `lab-notes/P0-XX-*.md`: one short note per step (what, math, observed). Devansh fills in observations after running.
- `_old_local_backup/`: pre-clone local copies of the notebooks (gitignored).

## Conventions
- Notebooks are teaching material: each one runs top to bottom on its own, on laptop CPU, Colab, or Kaggle (Kaggle needs Internet on). Each has a "Your turn" section and ends pointing to the next step.
- Code style: compact, few comments, the same idioms as the build scripts (`md(...)`/`code(...)` lambdas, config dicts like `GPT_CONFIG_124M`). Prose: short, plain, no em-dash filler.
- Checkpoints save `{'model', 'optimizer', 'config', 'history'}`; load with key `'config'`.
- Big files (`*.pt`, `*.safetensors`, `*.gguf`, `checkpoints/`, `.env`) are gitignored; never commit weights or tokens.
- Training runs log to Trackio (`trackio.init/log/finish`).

## Environment
- Laptop: Windows 11, Intel Core 5 120U, 16 GB RAM, **no CUDA**. System Python 3.14 has no torch. Local runs use the project venv `.venv/` (gitignored): `.venv/Scripts/python`. Recreate: `python -m venv .venv`, then `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt`.
- GPU work: Colab Pro (via Google AI Pro) or Kaggle T4x2 (fp16, no bf16). **No paid compute run without an explicit go from Devansh.**

## Verifying changes
"Looks right" isn't enough: execute the notebook (e.g. `jupyter nbconvert --to notebook --execute` or nbclient in a scratch copy, so downloads and checkpoints don't land in the repo) and read the real outputs before calling it done. P0-05's ground truth is a logit match against Hugging Face GPT-2 (max diff < 1e-3).

Baseline from a full run in `.venv` on 2026-11-07 (torch 2.15.0+cpu): P0-04 uses an 80/20 split (4,116 train / 1,029 val tokens), 16.1M params, 20 epochs, ~3 min → train 3.51 / val 7.47 (best val 6.97 at epoch 8, gap 3.96), plus two extra models for the underfit (8.62/8.93) and best-fit (5.45/6.81) comparison; checkpoint 193 MB. P0-05: HF logit diff 6.1e-05, GPT-2 val loss 3.74 vs P0-04 6.95 on the last 10% of the story.

P0-04b on Colab T4 (2000 steps, 90/10 split of Tiny Shakespeare), best val / final val / final gap: 5k tokens 7.85 / 11.37 / 11.33; 50k tokens 6.69 / 10.66 / 10.43; 304k tokens 4.85 / 4.90 / 1.53.
