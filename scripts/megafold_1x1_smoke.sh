#!/usr/bin/env bash
set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-29517}"
cd "$(dirname "$0")/.."
/home/rtae/miniconda3/envs/venv/bin/deepspeed --master_port "$MASTER_PORT" --num_gpus=1 train.py --config configs/megafold_1x1_smoke.yaml --trainer_name initial_training
