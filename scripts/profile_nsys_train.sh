#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NSYS_CMD="${NSYS_CMD:-nsys}"
DEEPSPEED_CMD="${DEEPSPEED_CMD:-}"
PYTHON_CMD="${PYTHON_CMD:-}"
OUTPUT_DIR="${NSYS_OUTPUT_DIR:-${ROOT_DIR}/nsys_reports}"
CONFIG="configs/megafold_1x1_smoke.yaml"
TRAINER_NAME="initial_training"
NUM_GPUS=1
MAX_STEPS="${MEGAFOLD_MAX_STEPS:-3}"
CAPTURE_STEP=""
MASTER_PORT="${MASTER_PORT:-29517}"
OUTPUT_NAME=""
EXTRA_ARGS=()

if [[ -z "${DEEPSPEED_CMD}" ]]; then
    if command -v deepspeed >/dev/null 2>&1; then
        DEEPSPEED_CMD="deepspeed"
    elif [[ -x "/home/rtae/miniconda3/envs/venv/bin/deepspeed" ]]; then
        DEEPSPEED_CMD="/home/rtae/miniconda3/envs/venv/bin/deepspeed"
    else
        DEEPSPEED_CMD="deepspeed"
    fi
fi

if [[ -z "${PYTHON_CMD}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    elif [[ -x "/home/rtae/miniconda3/envs/venv/bin/python3.13" ]]; then
        PYTHON_CMD="/home/rtae/miniconda3/envs/venv/bin/python3.13"
    else
        PYTHON_CMD="python"
    fi
fi

usage() {
    cat <<'EOF'
Usage: scripts/profile_nsys_train.sh [options] [-- extra train.py args]

Options:
  --config PATH         Training config to run.
  --trainer-name NAME   Trainer name inside the YAML.
  --gpus N              Number of GPUs for DeepSpeed.
  --max-steps N         Stop after N optimizer steps. Use 0 for no limit.
    --capture-step N      Capture only NVTX range for optimizer step N.
  --output NAME         Output basename under nsys_reports/.
  --help                Show this message.

Examples:
  scripts/profile_nsys_train.sh
    scripts/profile_nsys_train.sh --capture-step 1 --output steady_state_step1
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
        --capture-step)
            CAPTURE_STEP="$2"
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

CAPTURE_LABEL="train"
EFFECTIVE_MAX_STEPS="${MAX_STEPS}"

if [[ -n "${CAPTURE_STEP}" ]]; then
    CAPTURE_LABEL="train.step_${CAPTURE_STEP}"
    min_steps=$((CAPTURE_STEP + 1))
    if [[ "${EFFECTIVE_MAX_STEPS}" == "0" || "${EFFECTIVE_MAX_STEPS}" -lt "${min_steps}" ]]; then
        EFFECTIVE_MAX_STEPS="${min_steps}"
    fi
fi

if [[ -z "${OUTPUT_NAME}" ]]; then
    config_name="$(basename "${CONFIG}" .yaml)"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    if [[ -n "${CAPTURE_STEP}" ]]; then
        OUTPUT_NAME="${config_name}_g${NUM_GPUS}_cap${CAPTURE_STEP}_s${EFFECTIVE_MAX_STEPS}_${timestamp}"
    else
        OUTPUT_NAME="${config_name}_g${NUM_GPUS}_s${EFFECTIVE_MAX_STEPS}_${timestamp}"
    fi
fi

OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}"

TRAIN_CMD=()
if [[ "${NUM_GPUS}" == "1" ]]; then
    TRAIN_CMD=(
        "${PYTHON_CMD}" train.py
        --config "${CONFIG}"
        --trainer_name "${TRAINER_NAME}"
    )
else
    TRAIN_CMD=(
        "${DEEPSPEED_CMD}" --master_port "${MASTER_PORT}" --num_gpus="${NUM_GPUS}" train.py
        --config "${CONFIG}"
        --trainer_name "${TRAINER_NAME}"
    )
fi

if (( ${#EXTRA_ARGS[@]} > 0 )); then
    TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Tracing MegaFold training with Nsight Systems"
echo "  Config: ${CONFIG}"
echo "  Trainer: ${TRAINER_NAME}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Max steps: ${EFFECTIVE_MAX_STEPS}"
if [[ -n "${CAPTURE_STEP}" ]]; then
    echo "  Capture step: ${CAPTURE_STEP}"
fi
echo "  Report: ${OUTPUT_PATH}.nsys-rep"

MEGAFOLD_NVTX=1 \
MEGAFOLD_MAX_STEPS="${EFFECTIVE_MAX_STEPS}" \
"${NSYS_CMD}" profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=none \
    --cpuctxsw=none \
    --wait=all \
    --trace-fork-before-exec=true \
    --capture-range=nvtx \
    --capture-range-end=stop \
    --nvtx-capture="${CAPTURE_LABEL}" \
    --force-overwrite=true \
    -o "${OUTPUT_PATH}" \
    "${TRAIN_CMD[@]}"

echo "Trace written to ${OUTPUT_PATH}.nsys-rep"