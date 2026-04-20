#!/usr/bin/env bash
# Capture pair_bias tensors from every Pairformer attention module using the
# low-memory smoke config. Writes a single .pt file keyed by module name.
set -euo pipefail

MASTER_PORT=${MASTER_PORT:-29519}
OUTPUT=${OUTPUT:-outputs/pair_bias_structured/run.pt}
CONFIG=${CONFIG:-configs/megafold_1x1_smoke.yaml}
TRAINER_NAME=${TRAINER_NAME:-initial_training}

mkdir -p "$(dirname "$OUTPUT")"

echo "Writing captures to $OUTPUT"

deepspeed --master_port "$MASTER_PORT" --num_gpus=1 \
    scripts/capture_pair_bias_structured.py \
    --config "$CONFIG" \
    --trainer_name "$TRAINER_NAME" \
    --output "$OUTPUT"
