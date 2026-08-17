#!/bin/bash
# Submit one round-trip job per model. Usage:
#   bash submit_all.sh            # submit every model
#   bash submit_all.sh gpt2 Qwen  # submit only models whose name matches a pattern
#
# Columns: model | dtype | mem | cpus | time | partition
set -u
ROOT=/home/gloriama/dev/activation_balls
export HF_HOME=/large_storage/hsulab/userspace/gloriama/interp/hf_cache
mkdir -p "$HF_HOME"

MODELS=(
  "gpt2|float32|16G|8|1:00:00|cpu"
  "gpt2-xl|float32|32G|8|1:00:00|cpu"
  "EleutherAI/pythia-410m|float32|16G|8|1:00:00|cpu"
  "EleutherAI/pythia-6.9b|bfloat16|48G|16|2:00:00|cpu"
  "Qwen/Qwen2.5-0.5B|float32|16G|8|1:00:00|cpu"
  "Qwen/Qwen2.5-7B|bfloat16|48G|16|2:00:00|cpu"
  "google/gemma-2-2b|float32|32G|8|1:00:00|cpu"
  "mistralai/Mistral-7B-v0.3|bfloat16|48G|16|2:00:00|cpu"
  "allenai/OLMo-2-1124-7B|bfloat16|48G|16|2:00:00|cpu"
  # time limits kept tight: the cluster bills on allocation, so a hung job holding
  # a big allocation costs real money (see claude_mds/cluster_and_gpus.md)
  "openai/gpt-oss-20b|bfloat16|120G|16|2:00:00|cpu"
  "Qwen/Qwen2.5-72B|bfloat16|280G|24|3:00:00|cpu_high_mem"
  # pending HF license approval (403 as of 2026-07-05):
  # "meta-llama/Llama-3.2-1B|float32|16G|8|1:00:00|cpu"
  # "meta-llama/Llama-3.1-8B|bfloat16|48G|16|2:00:00|cpu"
)

for spec in "${MODELS[@]}"; do
  IFS='|' read -r model dtype mem cpus time part <<< "$spec"
  if [ $# -gt 0 ]; then
    match=0
    for pat in "$@"; do [[ "$model" == *"$pat"* ]] && match=1; done
    [ $match -eq 0 ] && continue
  fi
  safe=${model//\//__}
  outdir="$ROOT/results/$safe"
  mkdir -p "$outdir"
  # NOTE: --wrap scripts run under sh (not bash) on the compute nodes, so no
  # `source`/`conda activate` — call the env's python by absolute path instead.
  sbatch -p "$part" --mem="$mem" -c "$cpus" -t "$time" -J "rt-$safe" \
    -o "$outdir/slurm-%j.out" --wrap "
export HF_HOME=$HF_HOME
export OMP_NUM_THREADS=$cpus
$HOME/miniconda/envs/interp/bin/python $ROOT/experiments/roundtrip.py --model '$model' --dtype $dtype --outdir '$outdir'
"
done
squeue -u "$USER" -o "%.10i %.14j %.9P %.8T %.10M %R" | head -20
