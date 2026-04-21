#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NSYS_CMD="${NSYS_CMD:-nsys}"
DEEPSPEED_CMD="${DEEPSPEED_CMD:-deepspeed}"
OUTPUT_DIR="${NSYS_OUTPUT_DIR:-${ROOT_DIR}/nsys_reports}"
CONFIG="configs/megafold_1x1_smoke.yaml"
TRAINER_NAME="initial_training"
NUM_GPUS=1
MAX_STEPS="${MEGAFOLD_MAX_STEPS:-3}"
MASTER_PORT="${MASTER_PORT:-29517}"
OUTPUT_NAME=""
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/profile_nsys_train.sh [options] [-- extra train.py args]

Options:
  --config PATH         Training config to run.
  --trainer-name NAME   Trainer name inside the YAML.
  --gpus N              Number of GPUs for DeepSpeed.
  --max-steps N         Stop after N optimizer steps. Use 0 for no limit.
  --output NAME         Output basename under nsys_reports/.
  --help                Show this message.

Examples:
  scripts/profile_nsys_train.sh
  scripts/profile_nsys_train.sh --max-steps 5 --output smoke_trace
  scripts/profile_nsys_train.sh --config configs/megafold_1x1.yaml --gpus 1 --max-steps 10
  scripts/profile_nsys_train.sh --config configs/megafold_1x2.yaml --gpus 2 --max-steps 4
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --trainer-name)
            TRAINER_NAME="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${OUTPUT_NAME}" ]]; then
    config_name="$(basename "${CONFIG}" .yaml)"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_NAME="${config_name}_g${NUM_GPUS}_s${MAX_STEPS}_${timestamp}"
fi

OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}"

echo "Tracing MegaFold training with Nsight Systems"
echo "  Config: ${CONFIG}"
echo "  Trainer: ${TRAINER_NAME}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Max steps: ${MAX_STEPS}"
echo "  Report: ${OUTPUT_PATH}.nsys-rep"

MEGAFOLD_NVTX=1 \
MEGAFOLD_MAX_STEPS="${MAX_STEPS}" \
"${NSYS_CMD}" profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=none \
    --cpuctxsw=none \
    --wait=all \
    --capture-range=nvtx \
    --nvtx-capture=train \
    --stop-on-range-end=true \
    --force-overwrite=true \
    -o "${OUTPUT_PATH}" \
    "${DEEPSPEED_CMD}" --master_port "${MASTER_PORT}" --num_gpus="${NUM_GPUS}" train.py \
    --config "${CONFIG}" \
    --trainer_name "${TRAINER_NAME}" \
    "${EXTRA_ARGS[@]}"

echo "Trace written to ${OUTPUT_PATH}.nsys-rep"