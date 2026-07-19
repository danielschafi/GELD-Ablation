#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --partition=students
#SBATCH --gpus=a100:1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --array=0-8%2
#SBATCH --output=slurm_logs/ablation_%A_%a.log
#SBATCH --mail-type=end
#SBATCH --mail-user=daniel.schafi@bluewin.ch

# Job array over all (stage_1, stage_2) combinations (3 x 3 = 9 tasks).
# %2 throttles to at most 2 tasks running at once, matching the 2 available GPUs.
# SLURM runs this from the submission directory (the repo root), which is what
# `uv run -m test` needs — do not cd to dirname "$0" (that points at SLURM's
# staged copy of the script, not the repo).
set -euo pipefail

STAGE_1_OPTS=("BeamSearch" "knn-BeamSearch" "Neural")
STAGE_2_OPTS=("None" "Neural-RC" "BeamSearch-RC")

s1_idx=$((SLURM_ARRAY_TASK_ID / 3))
s2_idx=$((SLURM_ARRAY_TASK_ID % 3))
export STAGE_1="${STAGE_1_OPTS[$s1_idx]}"
export STAGE_2="${STAGE_2_OPTS[$s2_idx]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: STAGE_1=${STAGE_1}  STAGE_2=${STAGE_2}"
uv run -m test
