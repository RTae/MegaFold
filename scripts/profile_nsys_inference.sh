#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

NSYS_CMD="${NSYS_CMD:-$(command -v nsys || true)}"
PYTHON_CMD="${PYTHON_CMD:-$(command -v python || true)}"

if [[ -z "${NSYS_CMD}" ]]; then
    echo "Could not find 'nsys' in PATH. Install NVIDIA Nsight Systems first." >&2
    exit 1
fi

if [[ -z "${PYTHON_CMD}" ]]; then
    echo "Could not find 'python' in PATH." >&2
    exit 1
fi

OUTPUT_DIR="nsys_reports"
OUTPUT_NAME=""
CAPTURE_LABEL="inference"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --capture-label)
            CAPTURE_LABEL="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${OUTPUT_NAME}" ]]; then
    timestamp="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_NAME="inference_${timestamp}"
fi

OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}"

echo "Tracing MegaFold inference with Nsight Systems"
echo "  Capture label: ${CAPTURE_LABEL}"
echo "  Report: ${OUTPUT_PATH}.nsys-rep"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
MEGAFOLD_NVTX=1 \
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
    "${PYTHON_CMD}" -m megafold.cli "${EXTRA_ARGS[@]}"

if [[ ! -f "${OUTPUT_PATH}.nsys-rep" ]]; then
    echo "Nsight Systems did not generate ${OUTPUT_PATH}.nsys-rep" >&2
    exit 1
fi

echo "Generated ${OUTPUT_PATH}.nsys-rep"