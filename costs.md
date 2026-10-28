# Compute options and costs (verified 2026-10-23, ₹94.90/USD from open.er-api.com)
Rule: no paid run without a deliberate go. Scratch runs happen on a CPU-only box.

| Option | GPU | Price | INR | Source |
|---|---|---|---|---|
| His laptop | none (Intel iGPU, CPU only, 16 GB RAM) | owned | ₹0 | P0 + running 1-4B GGUF models in Ollama; 7-8B Q4 fits but slow |
| CPU scratch box | none (CPU) | free | ₹0 | - |
| Kaggle notebooks | 2x T4 16GB (P100 retired 15 Sep 2026) | free, ~30 GPU-h/week, 12h sessions, phone verify | ₹0 | kaggle.com/docs/notebooks, P100 sunset post |
| Colab free | T4 (not guaranteed) | free | ₹0 | colab signup page |
| Colab Pro | T4/L4/A100 via 100 CU/mo (T4 ~1.2 CU/h, A100 ~5.4 CU/h) | $9.99/mo (US list) | ~₹948 + FX fee | colab.research.google.com/signup |
| Colab Pro via Google AI Pro (my plan) | Colab Pro benefits; AI Pro lists 200 Colab CU/mo (~165 T4-h or ~37 A100-h); Kaggle quota rises to ~45 GPU-h/week when linked | included | ₹0 extra | aiplantracker 22 Oct 2026; Kaggle discussion 741987 |
| RunPod pod | RTX 3090 24GB | $0.50/h | ₹47/h | runpod.io/pricing (8 Oct 2026) |
| RunPod pod | RTX A5000 24GB | $0.27/h | ₹26/h | same |
| RunPod pod | A100 80GB | $1.79/h | ₹170/h | same |
| RunPod pod | H100 SXM 80GB | $3.99/h (8x = $31.92/h) | ₹379/h (8x ₹3,029/h) | same |
| Vast.ai (marketplace) | RTX 3090 | from ~$0.12/h, median ~$0.28/h | ₹11-27/h | vast.ai/pricing/gpu/RTX-3090; aicomputeradar |
| Vast.ai | H100 | median ~$3.29/h | ₹312/h | aicomputeradar |
| Lambda | A10 24GB | $1.29/h + GST | ₹122/h | lambda.ai/pricing |
| Lambda | H100 SXM (8x) | $3.99/GPU/h | 8x ₹3,029/h | same |
| E2E Networks (India, UPI, INR billing) | T4 / L4 / A100 80GB / H100 | $0.45 / $0.57 / $2.10 / $3.99 per h + GST | ₹43 / ₹54 / ₹199 / ₹379 per h | e2enetworks.com/pricing |
| E2E CPU VM (serving) | C3.8GB 4 vCPU | $35.64/mo | ~₹3,380/mo | same |
| Google Cloud free trial | ends early Nov 2026; GPUs generally need an upgraded paid account (check) | - | - | - |

## Cost along the process
| Phase | Lean path | Full path |
|---|---|---|
| P0 foundations | ₹0 (laptop CPU) | ₹0 |
| P1 minimind | ₹0 Colab Pro A100/L4 CU or Kaggle T4x2 / ~₹160 Vast 3090 6h | ~₹285 RunPod 3090 6h |
| P2 nanochat | mini depth-12 on Colab Pro A100 (CU, likely ₹0) or rented 1x H100/A100 ~₹500-1,400 | full speedrun 8xH100 3-4h, ₹9,100-12,100 ($96-128) |
| P3 Unsloth QLoRA | ₹0 Kaggle/Colab | ~₹150 (L4/3090, 2-3h) |
| P4 serve + agent | ₹0 on laptop | +~₹3,380/mo if always-on CPU VM |
| Buffer (setup, idle, reruns ~25%) | ~₹200-500 | ~₹1,000 |
| **Total** | **~₹700-2,300 (~$8-24)** | **~₹10,000-13,500 (~$105-140)** + optional hosting |
Note: storage on rented boxes adds a little (RunPod volume $0.10/GB/mo). Stop pods when idle.

## Update 2026-10-28 (vendor pages read that day)
- E2E Networks new rate card (27 Oct 2026, excl. 18% GST): T4 ₹30/h, L4 ₹49/h, A100-80GB ₹189/h, H100 ₹379/h. **Spot (interruptible): H100 ₹70/h, A100-80GB ₹66/h, H200 ₹88/h.** Changes P2: an 8x spot H100 speedrun at about ₹560/h ≈ ₹2,250-2,650 + GST for 3.5-4 h, vs ₹9,100-12,100 on demand, if 8 spot GPUs are available and the run checkpoints (reclaim risk). Not verified that 8x spot is offered together.
- Modal: $30/mo free credits (card likely needed). T4 ~$0.59/h, A10 ~$1.10/h, H100 ~$3.95/h. Fallback after Kaggle/Colab.
- HF Jobs needs bought credits (t4-small $0.40/h, a10g $1.00/h, default timeout 30 min). HF ZeroGPU: free demo Spaces (5 min/day GPU on a free account).
- Gone/changed: GitHub Models retired 30 Jul 2026; Cerebras has no free tier; free HF accounts get no Inference Providers credits; Lightning AI offline since 16 Oct 2026 incident; LiteLLM 1.107.0 (and 1.82.7/8) malicious, so pin or avoid.
- Note: some sources still list a Kaggle P100, but the 2026-10-23 check found P100 retired 15 Sep 2026. Treat it as T4x2 (fp16; no bf16 either way).
