# P1-03 Data pipeline (2026-11-09)
Notebook: notebooks/P1-03-data-pipeline.ipynb (built by build_p1_03.py). Output (gitignored): notebooks/data/ts_train_full.bin, ts_val.bin, ts_meta.json.
## What it does
Downloads TinyStoriesV2-GPT4 (2.2 GB train, 22 MB valid) with a resumable download, tokenizes it once with tokenizers/tinystories-4k.json in 64 MB chunks, and writes one stream of uint16 ids with `<|endoftext|>` (id 0) after every story. A `get_batch` function cuts random windows from the file through np.memmap.
## The math / code
- uint16: vocab 4,096 < 65,536, so 2 bytes per token. 552M tokens = 1.1 GB.
- Chunk edges: the unfinished last story of a chunk is carried to the next chunk, and an incremental UTF-8 decoder finishes a character cut by the edge.
- Batch: x = data[j : j+T], y = data[j+1 : j+T+1] for random j.
- Bits per byte (for P1-04 evals) = loss in nats per token x (1 / ln 2) / bytes per token.
## Reference run (laptop CPU, 2026-11-09)
- Train: 2,717,494 stories, 552,589,195 tokens, 203 tokens/story, 3.95 chars/token.
- Valid: 27,628 stories, 5,578,628 tokens, 3.98 bytes per token (22.1 MB of text).
- Tokenizing the train file: 631 s and 730 s in two runs (0.76-0.88M tokens/s; the laptop is noisy). Download: 2 to 40 min depending on the day; one attempt was cut at 1.09 GB by a connection reset, hence the resume code.
- Checks: ids max 4095; all 27,628 validation stories decode to the exact original text; tokenizing the valid file in 1 MB chunks gives the identical file; y is x shifted by one; 252 of 256 random 512-token windows contain a story boundary; batch of 32 x 512 loads in about 10 ms.
- Data per parameter for the planned 15.9M model: Chinchilla (20 tokens per parameter) = 318M tokens = 0.58 of this file.
## What I observed
(fill in after running)
## Questions / next
P1-04 pretraining: CPU quick run to check the loss falls, then the real run on a T4.
