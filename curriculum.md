# Build Your Own AI: curriculum
Started 2026-10-23. Status: [ ] to do, [~] in progress, [x] done. Hardware: HP Laptop 15-fd0xxx, Intel Core 5 120U, 16 GB RAM, Intel integrated graphics (no NVIDIA/CUDA), ~143 GB free disk, Windows 64-bit. Plan: P0 + local inference (Ollama, CPU) on laptop; all GPU training on Colab Pro (via Google AI Pro) or Kaggle.

| # | Phase | Repo | What you learn | What you build | Deliverable | Time | Est. cost |
|---|---|---|---|---|---|---|---|
| 0 | Foundations | rasbt/LLMs-from-scratch ch.1-4 | tokenization/BPE, embeddings, self-attention math, causal mask, multi-head, GPT block, sampling | a tiny GPT in PyTorch on CPU | notebook + lab notes 00-04 | 2-3 wks | ₹0 |
| 1 | Train a real small LLM | jingyaogong/minimind (Apache-2.0) | tokenizer training, pretrain loop, SFT, LoRA, DPO, GRPO, eval | 64M model from zero, your own weights | minimind checkpoint + chat demo | 1-2 wks | ₹0 (Kaggle) to ~₹300 (rented 3090, 4-6 h) |
| 2 | Scale up | karpathy/nanochat (MIT) | Rust BPE, data pipeline, scaling depth, midtrain, RL, evals (CORE, GSM8K), web chat | ChatGPT-style model; mini d12 run or full speedrun | eval report card + web chat | 1 wk | mini ~₹500-1,400; full 8xH100 speedrun ~₹9,100-12,100 |
| 3 | Fine-tune open weights on your data | unslothai/unsloth (Apache-2.0) | data formatting, chat templates, QLoRA, overfitting checks, GGUF export, quantization | Qwen/Gemma small model tuned on your own data, running in Ollama | GGUF on laptop + before/after eval | 1-2 wks | ₹0 (Kaggle/Colab free) to ~₹150 |
| 4 | Agent + serving | phase-3 model + tool calling | function calling, ReAct loop, RAG, OpenAI-compatible API (llama.cpp server/vLLM), auth, logging | agent with tools, served behind an API | live endpoint + demo | 2 wks | ₹0 local; optional always-on CPU VM ~₹3,400/mo (E2E C3.8GB) |

Total estimate (see costs.md): lean path ~₹700-2,300 (~$8-24); full path with 8xH100 speedrun ~₹10,000-13,500 (~$105-140). Excludes optional hosting.

## Status
- [x] P0-01 tokenizer  - [x] P0-02 attention  - [~] P0-03 GPT block + generate  - [~] P0-04 pretrain tiny GPT  - [~] P0-05 load GPT-2 weights
- [ ] P1 setup + tokenizer  - [ ] P1 pretrain  - [ ] P1 SFT  - [ ] P1 LoRA  - [ ] P1 DPO/RL
- [ ] P2 choose mini vs full  - [ ] P2 run  - [ ] P2 evals
- [ ] P3 dataset  - [ ] P3 QLoRA  - [ ] P3 GGUF + Ollama
- [ ] P4 tools  - [ ] P4 API  - [ ] P4 deploy
Open: which phase to start (recommended P0 on laptop); license of LLMs-from-scratch (NOASSERTION, check LICENSE file before reusing code publicly).
