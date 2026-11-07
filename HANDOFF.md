# Handoff: start Phase 1 (minimind)

Written 2026-11-07 at the end of the Phase 0 session. Read `CLAUDE.md` and the brain index (`C:\Users\agraw\Harnessing brain\index.md`) first, then this file. Delete or rewrite this file once Phase 1 is underway.

## State
- **Phase 0 is done.** P0-01 to P0-05 and the optional P0-04b are built, tested and pushed (last pushed commit `8a2759c`). Devansh ran all of them, P0-04b on a Colab T4.
- `curriculum.md` still shows P0-03, P0-04 and P0-05 as `[~]`. Tick them to `[x]` (add P0-04b) and update the Phase 1 line when Phase 1 starts.
- Key results, also in `CLAUDE.md`: GPT-2 loaded into our `GPTModel` with logit diff 6.1e-05. P0-04b best val 7.85 / 6.69 / 4.85 for 5k / 50k / 304k training tokens. The lesson is that the same model with more data fixes overfitting.

## First action in the new session
1. `git pull`, then `git status`.
2. Look at the real minimind repo (https://github.com/jingyaogong/minimind, Apache-2.0) before planning: its layout, config, dataset links and training scripts. Do not plan from memory. Devansh also pointed out that Andrej's repos (nanoGPT, ng-video-lecture) are open to look things up in.
3. Propose a Phase 1 plan and get a go before building. Suggested notebook split, in the same style as Phase 0 (a `build_p1_XX.py` generating each notebook, each one runnable top to bottom and ending with "Your turn"):
   - P1-01 train a tokenizer (~6,400 tokens) and compare with GPT-2's BPE from P0-01
   - P1-02 pretrain on a real dataset (GPU, Colab or Kaggle)
   - P1-03 SFT (chat format)
   - P1-04 LoRA
   - P1-05 DPO, then GRPO
   - evals along the way (lm-evaluation-harness is listed in `curriculum.md`)

## Rules and preferences to keep
- **No paid compute run without an explicit go from Devansh.** Colab Pro (Google AI Pro) or Kaggle T4x2 (fp16, no bf16). State a time estimate before any GPU run, and say whether it is measured or a guess.
- **Edit the build scripts, never the `.ipynb` files.** Committed notebooks have no outputs. Verify by executing a scratch copy (nbclient) and reading real outputs. In Phase 0 this caught wrong numbers in the notebook text more than once.
- **Laptop is CPU only.** Use `.venv/Scripts/python`. Test notebooks in a small CPU quick mode (as P0-04b does) before pushing. Colab and Kaggle code paths were only tested in P0-04b, so expect to debug the first run.
- **Devansh fills in `lab-notes/` himself.** Do not write his observations. Offer help when he asks.
- **Teaching style that worked:** short, plain prose with no em-dash filler. Explain with tables, using real numbers from his runs, and keep the "Your turn" exercises. He learns by watching Andrej's video alongside and by playing with visuals, so interactive HTML explainers helped (for example `explainers/neural-net-lab.html`).
- **Report faithfully:** if a result is surprising or a notebook fails, say so with the output.
- Keep Trackio logging (`embed=False` on Colab, because the embedded dashboard does not survive a downloaded notebook). Save any comparison chart as a PNG.
- Big files (`*.pt`, `*.safetensors`, `*.gguf`, `checkpoints/`) stay out of git.

## Open items (small, none blocking)
- Uncommitted and Devansh's: the `lab-notes/P0-0X` edits, the executed P0-01 to P0-03 and P0-05 notebooks, `notebooks/the-verdict.txt` (downloaded data, safe to ignore or gitignore). Do not commit these without asking.
- `.gitignore` and `CLAUDE.md` have local edits (`explainers/` ignored, the P0-04b baseline added). They are fine to commit with the next push.
- The P0-01 markdown still says the tokenizer learns "whole phrases (`hat `, `in the `)". That overstates it; only a few real phrases form. Reword it in `build_p0_01.py` when convenient.
- Optional P0-05b (not built): GPT-2 medium vs 124M, plus fine-tuning GPT-2 on Tiny Shakespeare against the from-scratch 304k model. Only if Devansh asks.
- Optional P0-04 cell showing one prediction being nudged step by step (code is in the session transcript, not built).
- `explainers/` (gitignored, local only): `neural-net-lab.html` and `Q-and-A-phase-0.md` (his questions with corrected answers).
- Cleanup: about 550 MB of GPT-2 weights in `~/.cache/huggingface` and `notebooks/gpt2.safetensors` can be deleted.
