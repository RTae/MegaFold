#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuH200x8
#SBATCH --gpus=4
#SBATCH --mem=32G
#SBATCH --job-name=megafold_4x2
#SBATCH --output=logs/%x_%j.out

module load pytorch-conda/2.8
source activate venv

export MASTER_PORT="12345"
export MASTER_ADDR="localhost"


deepspeed --num_gpus=4 train.py --config configs/megafold_4x2.yaml --trainer_name initial_training