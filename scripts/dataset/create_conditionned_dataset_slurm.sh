#!/bin/bash

#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --mem=128000
#SBATCH --cpus-per-task=32
#SBATCH --partition=cpu
#SBATCH --job-name=create_cond_chronos_dataset
uv run scripts/dataset/create_conditionned_dataset.py --subset_info_file scripts/dataset/subsets_new.json --batch_size 1024 --max_timesteps 1024 --arrow_output_path $PROJECTS/cond_chronos.arrow