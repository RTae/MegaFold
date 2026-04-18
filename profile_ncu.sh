#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# profile_ncu.sh — Wrapper around profile_ncu_runner.py for Nsight Compute
#
# Usage:
#   ./profile_ncu.sh <impl> [runner args...]
#   ./profile_ncu.sh all [runner args...]
#   ./profile_ncu.sh summary
#
# Examples:
#   ./profile_ncu.sh megafold --mode bwd --n-ctx 384
#   ./profile_ncu.sh flashbias --mode full --n-ctx 512 --warmup 10 --iters 5
#   ./profile_ncu.sh all --mode bwd --n-ctx 384
#   ./profile_ncu.sh summary
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_SCRIPT="${SCRIPT_DIR}/profile_ncu_runner.py"

PYTHON_CMD="${PYTHON_CMD:-python}"
NCU_CMD="${NCU_CMD:-ncu}"
OUTPUT_DIR="${NCU_OUTPUT_DIR:-${SCRIPT_DIR}/ncu_reports}"

ALL_IMPLS=(sdpa triton flashbias megafold fa3)

usage() {
    echo "Usage: $0 <impl|all|summary> [runner args...]"
    echo ""
    echo "Implementations: ${ALL_IMPLS[*]}"
    echo ""
    echo "Runner args are forwarded to profile_ncu_runner.py run <impl>."
    echo "Common runner args:"
    echo "  --mode {fwd,bwd,full}  Pass to profile (default: bwd)"
    echo "  --n-ctx N              Sequence length (default: 384)"
    echo "  --n-seq N              MSA sequences (default: 1)"
    echo "  --n-heads N            Head count (default: 4)"
    echo "  --head-dim N           Head dimension (default: 32)"
    echo "  --warmup N             Warmup iterations (default: 5)"
    echo "  --iters N              Profiled iterations (default: 3)"
    echo "  --seed N               Random seed (default: 0)"
    exit 1
}

smoke_test() {
    echo "=== Smoke test ==="
    ${PYTHON_CMD} "${RUNNER_SCRIPT}" smoke
}

run_one() {
    local name="$1"; shift
    mkdir -p "${OUTPUT_DIR}"

    # Parse --mode and --n-ctx from runner args to build a unique filename
    local mode="bwd" ctx="384"
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        case "${args[$i]}" in
            --mode)   mode="${args[$((i+1))]}" ;;
            --n-ctx)  ctx="${args[$((i+1))]}" ;;
        esac
    done
    local out_file="${OUTPUT_DIR}/${name}_${mode}_ctx${ctx}"

    echo "=== Profiling: ${name} mode=${mode} n-ctx=${ctx} ==="
    echo "  Output: ${out_file}.ncu-rep"

    ${NCU_CMD} --profile-from-start off \
        --set full \
        -o "${out_file}" \
        ${PYTHON_CMD} "${RUNNER_SCRIPT}" run "${name}" "$@"
}

run_all() {
    for impl in "${ALL_IMPLS[@]}"; do
        run_one "${impl}" "$@" || echo "  (${impl} skipped or failed)"
    done
}

print_summary() {
    echo "=== NCU report files ==="
    if [[ -d "${OUTPUT_DIR}" ]]; then
        ls -lh "${OUTPUT_DIR}"/*.ncu-rep 2>/dev/null || echo "  No .ncu-rep files found in ${OUTPUT_DIR}/"
    else
        echo "  Output directory ${OUTPUT_DIR}/ does not exist."
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    usage
fi

cmd="$1"; shift

case "${cmd}" in
    smoke)
        smoke_test
        ;;
    sdpa|triton|flashbias|megafold|fa3)
        smoke_test
        run_one "${cmd}" "$@"
        ;;
    all)
        smoke_test
        run_all "$@"
        ;;
    summary)
        print_summary
        ;;
    *)
        echo "Error: unknown command '${cmd}'"
        usage
        ;;
esac
