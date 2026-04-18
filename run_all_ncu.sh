#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPLS=(sdpa sdpa_no_bias fa1_bias flashbias megafold fa3 fa4)
MODES=(fwd bwd full)
CTXS=(256 384 512 1024)

for ctx in "${CTXS[@]}"; do
    for mode in "${MODES[@]}"; do
        for impl in "${IMPLS[@]}"; do
            echo "=== impl=${impl}  mode=${mode}  n-ctx=${ctx} ==="
            "${SCRIPT_DIR}/profile_ncu.sh" "${impl}" --mode "${mode}" --n-ctx "${ctx}" "$@" \
                || echo "  FAILED: ${impl} ${mode} ${ctx}"
        done
    done
done

echo ""
"${SCRIPT_DIR}/profile_ncu.sh" summary
