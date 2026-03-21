#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuH200x8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --job-name=megafold_1x1
#SBATCH --output=logs/%x_%j.out

module load pytorch-conda/2.8
source activate venv

export MASTER_PORT="12345"
export MASTER_ADDR="localhost"

deepspeed --num_gpus=1 train.py --config configs/megafold_1x1.yaml --trainer_name initial_training