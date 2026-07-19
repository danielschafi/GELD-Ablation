#!/bin/bash
# Submit the ablation job array (9 combinations, 2 running at a time).
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p slurm_logs
sbatch slurm_ablation.bash
