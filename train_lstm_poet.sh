#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=lstm_poetry
#SBATCH --output=lstm_poetry-%j.out
#SBATCH --error=lstm_poetry-%j.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=1:00:00
#SBATCH --mem=50gb
source /home/mwolter1/.bashrc
conda activate jax
python src/train.py