#!/bin/bash
#SBATCH --job-name=train-1000
#SBATCH --partition=students
#SBATCH --gpus=a100:1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=48
#SBATCH --time=35:00:00
#SBATCH --output=train_stage_2_%j.log

#SBATCH --mail-type=end
#SBATCH --mail-user=daniel.schafi@bluewin.ch   


uv run -m train_new