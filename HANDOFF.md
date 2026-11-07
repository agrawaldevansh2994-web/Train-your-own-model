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

## Next: P1-03 data pipeline, then P1-04 pretrain
- P1-03: download the full TinyStoriesV2-GPT4-train.txt (2.2 GB) and tokenize it with tokenizers/tinystories-4k.json (`encode_batch`, fast). Append `<|endoftext|>` (id 0) after every story. Write uint16 `.bin` files (vocab < 65,536) plus val, and load them with np.memmap and random windows (like nanoGPT). Report the total token count (expect ~550M from 3.97 chars/token; not measured).
- P1-04: MindModel `{'vocab_size': 4096, 'context_length': 512, 'emb_dim': 384, 'n_layers': 8, 'n_heads': 6, 'n_kv_heads': 2, 'ffn_dim': None, ...}` = 15.9M params. Use AdamW with warmup + cosine, fp16 autocast + GradScaler on T4, Trackio with `embed=False`, and checkpoints `{'model','optimizer','config','history'}` to Drive / `/kaggle/working`. Include a CPU quick mode. Eval: val loss, bits per byte, fixed prompts.
- Phase 1 notebooks import `mymind.py` with the same download-if-missing pattern as P1-02.

## Rules (unchanged)
- No paid compute without an explicit go. State a time estimate before any GPU run and say whether it is measured or a guess.
- Edit build scripts, never the `.ipynb`. Verify by executing a scratch copy and reading the real outputs.
- Devansh fills in "What I observed" in lab notes himself.
- Uncommitted and his: the P0 lab-note edits, the executed P0 notebooks, and `notebooks/the-verdict.txt`. Do not commit these without asking.
- Open small items from Phase 0: the P0-01 "whole phrases" wording, optional P0-05b, and cleaning up the GPT-2 weights cache. The minimind-3 download adds 128 MB to `~/.cache/huggingface`.
