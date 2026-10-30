# Train Your Own Model

Devansh's learning repo: build an LLM end to end (tokenizer → pretraining → SFT/LoRA/DPO/GRPO → evals → serving → agent). Phases and status in `curriculum.md`, compute pricing in `costs.md`. Remote: github.com/agrawaldevansh2994-web/Train-your-own-model (branch `main`). GitHub is the source of truth; work done in other sessions is pushed there, so `git pull` before starting.

## Layout
- `notebooks/build_p0_XX.py` → generates `notebooks/P0-XX-*.ipynb`. **Edit the build script, never the .ipynb by hand**, then regenerate: `cd notebooks && python build_p0_XX.py`. Committed notebooks have no outputs.
- `src/mygpt.py`: the model from P0-01..P0-03 in one file (MultiHeadAttention, LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, generate, load_gpt2_weights). P0-04+ import it; on Colab/Kaggle they download it from the raw GitHub URL, locally they use `../src`. Changing it changes every later notebook: keep it in sync with the classes shown in P0-02/P0-03.
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

Baseline from a full run in `.venv` on 2026-10-30 (torch 2.15.0+cpu, ~6.5 min total): P0-04 16.1M params, 12 epochs → train 4.82 / val 6.46 (best val 6.42 at epoch 8), checkpoint 193 MB; P0-05 HF logit diff 6.1e-05, GPT-2 val loss 3.74 vs P0-04 6.46.
