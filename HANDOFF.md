# Handoff: Phase 1 in progress

Updated 2026-11-07. Read `CLAUDE.md` and the brain index (`C:\Users\agraw\Harnessing brain\index.md`) first, then this file.

## Decisions (made with Devansh on 2026-11-07)
- **minimind's own data is mostly Chinese.** Samples: pretrain_t2t_mini 74-86% Chinese characters, 0% English rows; sft_t2t_mini 52-59%. So we do **not** run minimind's scripts. We use minimind (and Hark's notes) as the reference, build on our own code, and train on English data.
- **Path: TinyStories first, then a scale-up run** (same pattern as P0-04 → P0-04b):
  1. ~16M model on TinyStories with our 4,096-token tokenizer: laptop plus free Kaggle/Colab.
  2. P1-05b: 64M (minimind-3 shape) on ~1.3B tokens of FineWeb-Edu with a new tokenizer trained on that text. **Guesses, not measured:** ~10-25 h on one T4 (2-3 Kaggle sessions with checkpoints), or ~1-2 h on a rented H100 (E2E spot ₹70/h, on demand ₹379/h). Devansh offered to rent a GPU if needed. Measure tokens/s on the first real run before choosing, and get his go first.
  3. SFT on `HuggingFaceTB/smol-smoltalk` (Apache-2.0), DPO on the English half of minimind's dpo.jsonl (sampled from llamafactory/DPO-En-Zh-20k), a small GRPO toy with an automatic reward.
- Hark's notes (pasted by Devansh), which we adopted: a modern block on our own code, TinyStories, fp16 + GradScaler on T4, checkpoint every ~30 min, Trackio. Our corrections to Hark: raw val loss is not comparable with TinyStories-33M (different tokenizer), so use bits per byte; nanochat runs on 1 GPU too (8xH100 is only for the speedrun).

## Done
- **P1-01 tokenizer** (`build_p1_01.py`, `tokenizers/tinystories-4k.json`): 4,096 vocab gives 3.97 chars/token vs GPT-2's 4.06 on TinyStories val, with 12x fewer tokens.
- **P1-02 modern block** (`build_p1_02.py`, `src/mymind.py`): RMSNorm, RoPE, SwiGLU, GQA, QK-norm, KV cache. minimind-3 weights in our MindModel are bit-exact against HF Qwen3ForCausalLM.
- Both notebooks were executed in a scratch copy on CPU; their outputs are recorded in their lab notes.

## Done too: P1-03 data pipeline (2026-11-09)
`build_p1_03.py`: full TinyStoriesV2-GPT4 tokenized once: 552,589,195 train tokens, 5,578,628 val tokens, 3.98 bytes per token (saved to data/ts_meta.json for bits per byte). Tokenizing took 631-730 s on the laptop; the download is flaky (one 2 GB attempt was reset at 1.09 GB), so `fetch` resumes. Measured on this laptop: training the 15.9M model runs at about 1,350 tokens/s on CPU, so the real P1-04 run needs a GPU (a T4 speed of 30-60k tokens/s is still a guess, measure it first). Chinchilla for 15.9M = 318M tokens = 0.58 of the file. On Colab/Kaggle P1-04 must re-run the P1-03 prep (about 1 GB output; Kaggle does not keep files between sessions, so consider saving the .bin as a Kaggle dataset).

## Built 2026-11-18 (CPU-tested in scratch; not yet run on a GPU)
- **Purpose changed (Devansh, 2026-11-18):** Phase 1 is for learning the pipeline only. The real goal is Phase 3: fine-tune a big open model on his data. Keep Phase 1 lean: a CPU smoke test per notebook, short GPU runs, explanation tables instead of him rerunning everything. Avoid heavy CPU jobs while he uses the laptop.
- `src/tsdata.py` (P1-03 pipeline + Batches, save_dir, find_checkpoint, val_stories) and `src/sftdata.py` (P1-06 data code). Keep in sync with the notebooks that show the code.
- **P1-04 pretrain:** 15.9M MindModel, rope_theta 1e4, AdamW (0.9, 0.95), wd 0.1 on matrices only, warmup + cosine 1e-3 -> 1e-4, fp16 + GradScaler (bf16 if supported), eval = val loss on 100 fixed windows + bits per byte + a sample, checkpoint every 20 min with resume and max_hours. GPU defaults: batch 32 x 512, 300M tokens (18.3k steps). A benchmark cell MEASURES tokens/s and projects run time before training. CPU test: 300 steps of 8 x 256, val loss 8.43 -> 3.92 (1.43 bpb), gap +0.11, 349 tokens/s (laptop under load). Resume was tested (it picked up at step 300).
- **P1-05 eval:** bits per byte vs roneneldan/TinyStories-1M/8M/33M on the same val stories (only stories that fit in 512 tokens), loss by position, probes, side-by-side stories. The first CPU run hit a bug (append needed a tuple); it is fixed but **not yet rerun**: rerun it in scratch on the run4 checkpoint before committing.
- **P1-06 SFT:** TinyStories-Instruct parsed into chat-format instructions, loss on answer tokens only, "uses all required words" score. CPU test: answer loss 4.07 -> 3.84, word score 0% -> 0% (base saw only 1M tokens; meaningful only after the GPU run).
- **P1-07 LoRA:** hand-written LoRALinear on all 7 matrices, r=8: 471,040 trainable (2.9%), adapter 1.9 MB, merged vs LoRA max diff 4.8e-06. CPU: answer loss 3.95 (full SFT 3.84).
- **Next:** rerun P1-05 when the laptop is free; P1-08 = one short notebook with DPO and GRPO, both judged by the word score; then Phase 3. The P1-04 GPU run needs Devansh's go (free Kaggle T4, but state the measured time first).

## Rules (unchanged)
- **This repo is public.** Phase 3's real target (fine-tuning on Devansh's private judgment data) lives in a separate private repo, later. Never put private data, quotes from it, or weights trained on it here. Describe methods only. Nothing to do about this until Phase 3.
- No paid compute without an explicit go. State a time estimate before any GPU run and say whether it is measured or a guess.
- Edit build scripts, never the `.ipynb`. Verify by executing a scratch copy and reading the real outputs.
- Devansh fills in "What I observed" in lab notes himself.
- Uncommitted and his: the P0 lab-note edits, the executed P0 notebooks, and `notebooks/the-verdict.txt`. Do not commit these without asking.
- Open small items from Phase 0: the P0-01 "whole phrases" wording, optional P0-05b, and cleaning up the GPT-2 weights cache. The minimind-3 download adds 128 MB to `~/.cache/huggingface`.
