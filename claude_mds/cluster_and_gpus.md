# Cluster & GPU learnings (Arc SLURM cluster — "Chimera")

What I learned about the compute environment while setting up the interp experiments,
plus corrections from Gloria's notes on the cluster docs (2026-07-05, marked [docs]).
History at the bottom.

## The picture

- **GPUs: NVIDIA H100 80GB, 4 per GPU node** (`gpu`, `gpu_batch`, `gpu_high_mem` partitions).
  GPU nodes have 176–184 cores and 1–2TB RAM. The GPU type is NOT visible in `sinfo`
  features (all `(null)`) — I had to sbatch a 2-minute `nvidia-smi -L` job to find out.
- **A single node maxes at 4×H100. For more, go multi-node** [docs]: `gpu_batch` allows
  multi-node jobs with no node cap (`--nodes=2 --gpus=8`, etc.). So the capability ceiling is
  not "70B on one node" — bigger models are possible across nodes, at multi-node complexity
  and cost.
- **CPU high-mem nodes**: 2TB RAM, 184 cores (`cpu_high_mem`, `cpu_batch_high_mem`), some idle.
- **`gpu_cpu` partition** (160c/800GB, 4 GPUs) exists but is only for labs with whole-node
  reservations [docs].
- **MIG (20/40GB GPU slices) is currently broken** by a SLURM upgrade — no sub-80GB slices to
  save cost for now [docs].
- **Partitions/time limits**: `cpu`/`gpu` 1–5 days; `*_batch` 14 days; also `*_preemptible`.
- **Per-user concurrent job cap** (`QOSMaxJobsPerUserLimit`): submitting ~11 jobs queues most of
  them; they trickle through as slots free.
- **Network to HuggingFace is fast**: Qwen2.5-72B (145GB of safetensors) downloaded + loaded +
  ran in a ~17-minute job.

## Billing (it's real money) [docs]

- **$1.23 per H100 GPU-hour** (standard) / **$1.29** (high-mem queue), **charged on
  allocation, not utilization** — an idle interactive GPU shell or a hung job bills the whole
  time it holds the allocation. Keep sbatch time limits tight and kill stray allocations.
- Jobs hit the lab/cost-center by default; `-A vci` / `-A adi` can target other accounts if
  you have access.
- **Storage: $150/TB beyond the free allocation.** The HF cache
  (`.../gloriama/interp/hf_cache`) grows fast — Qwen2.5-72B alone is 145GB. Prune models that
  won't be reused.

## Interactive GPU work: `sh_gpu` [docs]

`sh_gpu [N]` is the documented tool for grabbing an interactive GPU allocation (sets
`CUDA_VISIBLE_DEVICES` for you). Defaults per GPU: 8 CPU cores + 80GB RAM; override with
`--cpus-per-gpu` / `--mem-per-gpu`. Use this instead of fighting `srun` from the dev shell
(see trap #1). Remember allocation billing: exit when done.

## Traps (each cost real time)

1. **The dev shell is itself inside a SLURM allocation** (cpu partition, 4 CPU / 16GB).
   Consequence: `srun -p gpu --gpus=1 ...` fails with
   `Unable to create step for job <id>: Invalid generic resource (gres) specification` —
   it tries to add a step to the *current* job instead of creating a new one.
   **Use `sbatch` for batch work, `sh_gpu` for interactive GPU work.**
2. **`sbatch --wrap` bodies run under `sh` (dash), not bash**, on compute nodes:
   `source: not found` → `conda activate` fails → system python runs instead →
   `ModuleNotFoundError`. Two fixes:
   - Documented [docs]: in a proper `#!/bin/bash` sbatch script, put
     `source /opt/conda/etc/profile.d/conda.sh` at the top, then `conda activate <env>` works.
   - Also valid (what `experiments/submit_all.sh` does): skip activation entirely and call
     `~/miniconda/envs/<env>/bin/python` by absolute path.
3. Only ~16GB RAM in the dev shell: models above ~2B (fp32) must go through sbatch.

## Storage & env

- Big files under `/large_storage/hsulab/userspace/gloriama/interp/` (12T free on the mount;
  per CLAUDE.md — but note the $150/TB charge above). HF cache:
  `export HF_HOME=/large_storage/hsulab/userspace/gloriama/interp/hf_cache`.
- Conda env `interp` (python 3.11): `pip install torch transformers accelerate sentencepiece
  protobuf safetensors` → torch 2.12.1+cu130, transformers 5.13.0.
- HF token at `~/.cache/huggingface/token` (account `gloriama`, oauth, expires 2026-08-01).

## History

- **2026-07-05**: Basics discovered during setup for the embed→unembed round-trip experiment.
  First 11-job batch lost to trap #2 (jobs 2655387–2655397, cancelled, resubmitted).
- **2026-07-05 (later)**: Gloria's notes from the cluster docs added: billing model ($1.23–1.29
  /H100-hr on allocation; $150/TB storage; `-A vci`/`-A adi`), `sh_gpu`, the
  `/opt/conda/etc/profile.d/conda.sh` sbatch fix, multi-node `gpu_batch` (>4 GPUs), `gpu_cpu`
  reservation-only, MIG broken. Tightened sbatch time limits in `submit_all.sh` in response
  (allocation-billed hangs are expensive).
