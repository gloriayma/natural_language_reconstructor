<!-- Suggested CLAUDE.md — what would have let me skip a session of discovery.
     Drop-in replacement: your original conventions are kept (lightly sharpened),
     everything else is what I had to find out the hard way on 2026-07-05. -->

# Project

Interpretability experiments on open-source LLMs (embeddings, activations, finetuning),
run on the Arc "Chimera" SLURM cluster.

# Answering questions about "a model"

When I ask about "a model", answer for: the best open-source models, frontier models
(ChatGPT/Claude) if findable, and the lower-bound worst modern model. Give a general sense.

# Write-ups (claude_mds/)

Whenever you do anything, write it up in `claude_mds/` so I can replicate it — including
snags, surprises, and reversals. Keep docs **atomic: one per reusable topic** (e.g. the model
panel, the word list, cluster learnings) plus one per experiment for method+results.
Revisiting a topic = append a dated entry to its History section, don't start a new doc.
Cross-link docs instead of duplicating content. In the end I want few docs, each with a
rich history.

# Cluster (Chimera) — read before running anything

- **Your dev shell is already inside a SLURM allocation** (cpu partition, ~4 CPU / 16GB).
  `srun` from it fails with an "Invalid generic resource (gres) specification" error.
  Use **`sbatch`** for batch work; use **`sh_gpu [N]`** for interactive GPU allocations
  (sets CUDA_VISIBLE_DEVICES; defaults 8 cores + 80GB RAM per GPU; override with
  `--cpus-per-gpu` / `--mem-per-gpu`).
- **`sbatch --wrap` bodies run under `sh`, not bash** — `conda activate` will silently fail
  into system python. Either use a `#!/bin/bash` script with
  `source /opt/conda/etc/profile.d/conda.sh` at the top, or skip activation and call
  `~/miniconda/envs/<env>/bin/python` by absolute path.
- **Hardware**: H100-80GB ×4 per GPU node (`gpu`, `gpu_batch`, `gpu_high_mem`); GPU nodes
  have 176–184 cores, 1–2TB RAM. >4 GPUs means a **multi-node `gpu_batch` job**
  (`--nodes=N --gpus=M`, no node cap). `cpu_high_mem` nodes: 2TB RAM / 184 cores.
  `gpu_cpu` is whole-node-reservation labs only. MIG (sub-80GB slices) is currently broken.
- **Time limits**: `cpu`/`gpu` 1–5 days, `*_batch` 14 days; there's a per-user concurrent
  job cap, so large submissions queue and trickle through.
- **Billing — it's real money, charged on ALLOCATION not utilization**: $1.23/H100-hour
  (standard), $1.29 (high-mem queue); storage $150/TB beyond the free allocation. Keep
  sbatch time limits tight, never leave interactive allocations idle, and prune unused
  model weights from the HF cache. Jobs bill my lab's cost center by default (`-A vci` /
  `-A adi` only if granted).

# Environment

- Conda lives at `~/miniconda`; home is shared across nodes, so envs work on compute nodes.
  Use the `interp` env for interp work (torch + transformers already installed).
- Big files go under `/large_storage/hsulab/userspace/gloriama/interp/`. Set
  `HF_HOME=/large_storage/hsulab/userspace/gloriama/interp/hf_cache` before anything
  touches HuggingFace. The pipe to HF is fast (~145GB in ~15 min).
- HF token: `~/.cache/huggingface/token` (account `gloriama`). It has gated access to
  Gemma and Mistral; **meta-llama is NOT approved** (as of 2026-07-05) — curl-check gated
  repos (`.../resolve/main/config.json`, expect 200) before queueing jobs on them, and tell
  me which licenses I need to click.

# Git

Before branching off in a branch or new worktree, check main and pull so it's up to date
with remote.
