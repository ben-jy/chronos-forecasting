#!/bin/bash

#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --mem=32000
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40_48gb:1
#SBATCH --job-name=lotsa-chronos-tiny
uv run scripts/training/train.py --config scripts/training/configs/lotsa-chronos-t5-tiny.yaml