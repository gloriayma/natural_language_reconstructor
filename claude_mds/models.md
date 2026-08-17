# Model selection (open-source model panel)

The model panel used across the interp experiments, with reasons. History at the bottom.

Tied/untied verified two ways: `tie_word_embeddings` in each repo's `config.json` (absent =
HF default True for GPT-2/Gemma-2), and at runtime by comparing embedding vs lm_head tensor
pointers (recorded in each run's `meta.json` — the two have agreed everywhere so far).

| model | params | layers | tied? | vocab | why chosen |
|---|---|---|---|---|---|
| gpt2 | 124M | 12 | tied | 50k | classic baseline, most-studied in interp; the "lower-bound worst modern-ish model" |
| gpt2-xl | 1.5B | 48 | tied | 50k | size scaling within the same family/tokenizer |
| EleutherAI/pythia-410m | 410M | 24 | untied | 50k | interp-focused suite, untied at small scale |
| EleutherAI/pythia-6.9b | 6.9B | 32 | untied | 50k | same family larger — untied size contrast |
| Qwen/Qwen2.5-0.5B | 0.5B | 24 | **tied** | 152k | modern small model; within-family tied/untied contrast vs 7B |
| Qwen/Qwen2.5-7B | 7B | 28 | **untied** | 152k | the other half of the Qwen contrast |
| google/gemma-2-2b | 2.6B | 26 | tied | 256k | tied at an unusually huge vocab; Google's open model |
| mistralai/Mistral-7B-v0.3 | 7B | 32 | untied | 32k | different family; old-style small vocab |
| allenai/OLMo-2-1124-7B | 7B | 32 | untied | 100k | fully open (weights+data+code) — truest "open source" |
| openai/gpt-oss-20b | 20B MoE | 24 | untied | 201k | OpenAI's open frontier-lab model; stretch: MXFP4-quantized |
| Qwen/Qwen2.5-72B | 72B | 80 | untied | 152k | **best open model this compute can handle** (ungated) |
| meta-llama/Llama-3.2-1B | 1B | 16 | tied | 128k | ⏳ pending HF license — within-family contrast w/ 8B |
| meta-llama/Llama-3.1-8B | 8B | 32 | untied | 128k | ⏳ pending HF license |

## Access

Existing token has Gemma + Mistral access. **meta-llama repos return 403** — to unblock, log
into HF as `gloriama` and accept the license at
https://huggingface.co/meta-llama/Llama-3.2-1B and https://huggingface.co/meta-llama/Llama-3.1-8B,
then uncomment the two Llama lines in `experiments/submit_all.sh` and run
`bash experiments/submit_all.sh meta-llama`.

## Frontier / closed models

Not runnable (no weight access). Publicly inferable: GPT-2 (OpenAI) was tied; OpenAI's open
gpt-oss (2025) is untied at 201k vocab; the pattern across Meta/Google/Qwen open releases is
small models tie (saves a big fraction of params), large models untie. Closed frontier configs
(GPT-4/5-class, Claude) are not published.

## Excluded, and why

- DeepSeek-V3/R1 (671B): originally "over compute budget", revised after learning `gpu_batch`
  allows multi-node jobs (no node cap) — now *possible in principle* (multi-node GPU, or even
  a 2TB cpu_high_mem node holds ~1.34TB bf16 for this hook-only experiment) but excluded on
  cost (allocation-billed) and multi-node/FP8-dequant complexity. Revisit if a 600B-class
  data point becomes worth it.
- Llama-3.1-70B: gated (and Qwen2.5-72B fills the slot ungated). Qwen3-235B: same multi-node
  cost/complexity argument as DeepSeek.
- BLOOM/OPT: superseded; gpt2 is a more interesting "lower bound".

## Finetuning-capable subset (for later experiments)

- Full finetune on one H100: gpt2, gpt2-xl, pythia-410m, Qwen2.5-0.5B, gemma-2-2b, Llama-3.2-1B.
- 7–8B class: full FT on a 4×H100 node (FSDP/ZeRO) or LoRA on one H100.
- Qwen2.5-72B: QLoRA on a 4×H100 node; full FT would need multi-node `gpu_batch` (possible —
  no node cap — but allocation billing makes it a deliberate spend, ~$5/hr per node).

## History

- **2026-07-05**: Panel chosen for the embed→unembed round-trip experiment; tied/untied
  verified from configs; meta-llama 403 discovered (user pinged to accept license).
