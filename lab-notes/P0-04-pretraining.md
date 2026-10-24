# P0-04 Pretraining a tiny GPT (2026-10-24)
Notebook: notebooks/P0-04-pretraining.ipynb. Imports the model from src/mygpt.py (P0-01..03 code in one file).
## What it does
Trains a shrunk GPT (ctx 128, d 256, 4 heads, 4 layers, tied weights, 16.1M params) on The Verdict (4,630 train / 515 val tokens) with AdamW. Plots train/val loss, demos temperature + top-k sampling, saves and reloads a checkpoint (to Google Drive in Colab).
## The math / code
loss = mean(-log p[correct next token]) (cross-entropy); perplexity = exp(loss).
Loop: zero_grad -> forward -> loss -> backward -> clip grad norm 1.0 -> AdamW step (lr 1e-3, wd 0.1).
Sampling: logits / T, keep top-k, softmax, multinomial.
## Reference run (CPU, 1 epoch smoke test)
- Init loss 10.885 (uniform 10.82). Without std-0.02 init the tied embedding gave loss 171: init matters.
- Epoch 1: train 6.502, val 6.736; greedy output 'Every effort moves you,,,,,,...' (learned commas first).
- Checkpoint 193 MB (weights ~65 MB + AdamW state); reload identical (after model.eval()).
- Full 12-epoch run not recorded here: the build machine ran out of RAM. Expect train loss well under val by epoch ~6-8 (overfitting).
## What I observed
(fill in after running: epoch where val loss bottoms out, best sample)
## Questions / next
P0-05 load real GPT-2 weights into GPTModel.
