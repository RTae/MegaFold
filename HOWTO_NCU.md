# NCU Profiling HOWTO

## Quick start

```bash
# Smoke test — confirms CUDA works
python profile_ncu_runner.py smoke

# Profile MegaFold backward pass (default mode)
python profile_ncu_runner.py run megafold --mode bwd --n-ctx 384

# Profile FlashBias forward+backward
python profile_ncu_runner.py run flashbias --mode full --n-ctx 512
```

## Implementations

| Name       | Kernel                         | Layout              | Notes                                      |
|------------|--------------------------------|---------------------|--------------------------------------------|
| `sdpa`     | `torch.nn.functional.scaled_dot_product_attention` | `(B, H, N_seq, N, D)` | Falls back to math/efficient backend due to `attn_mask` |
| `triton`   | FA1-Triton with dense bias     | `(B, N, H, D)` LLM  | Not directly comparable to megafold        |
| `flashbias`| FlashBias Triton low-rank bias | `(B, N, H, D)` LLM  | Not directly comparable to megafold        |
| `megafold` | MegaFold `TritonEvoformer`     | `(B, N_seq, N, H, D)` AF3 | Primary target                        |
| `fa3`      | FlashAttention-3 (no bias)     | `(B, N, H, D)`      | Compute ceiling baseline; only `N_SEQ=1`   |

## Modes

| Mode   | What runs in the profiled region                                |
|--------|-----------------------------------------------------------------|
| `fwd`  | Forward pass only                                               |
| `bwd`  | Forward once (outside profiler), then backward in each iteration|
| `full` | Forward + backward each iteration; `do` pre-allocated once      |

Default mode is `bwd` — this captures the dQ kernel and `atomic_add` to `d_pair_bias`.

## CLI reference

```
python profile_ncu_runner.py run <impl> [options]

positional:
  impl          {sdpa, triton, flashbias, megafold, fa3}

options:
  --mode        {fwd, bwd, full}     (default: bwd)
  --n-ctx N     Sequence length      (default: 384)
  --n-seq N     MSA sequences        (default: 1)
  --n-heads N   Head count           (default: 4)
  --head-dim N  Head dimension       (default: 32)
  --warmup N    Warmup iterations    (default: 5)
  --iters N     Profiled iterations  (default: 3)
  --seed N      Random seed          (default: 0)
```

## Running under NCU

```bash
# Utilization summary
ncu --profile-from-start off \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed \
    -o megafold_bwd \
    python profile_ncu_runner.py run megafold --mode bwd --n-ctx 384

# Roofline
ncu --profile-from-start off --set roofline -o megafold_roofline \
    python profile_ncu_runner.py run megafold --mode bwd --n-ctx 384

# Full report (slow — many replays)
ncu --profile-from-start off --set full -o megafold_full \
    python profile_ncu_runner.py run megafold --mode bwd --n-ctx 384
```

Or use the shell wrapper:

```bash
chmod +x profile_ncu.sh

# Profile one impl
./profile_ncu.sh megafold --mode bwd --n-ctx 384

# Profile all impls
./profile_ncu.sh all --mode bwd --n-ctx 384

# List report files
./profile_ncu.sh summary
```

## Comparison groups

When analyzing results, group implementations by layout:

- **AF3-style** (apples-to-apples): `megafold` vs `sdpa`
- **LLM-style bias kernels**: `triton` vs `flashbias`
- **Compute ceiling**: `fa3` (no bias overhead)

The `triton`/`flashbias` kernels use LLM-style `(B, N, H, D)` tensors without
the `N_SEQ` broadcast dimension. At `N_SEQ=1` the total FLOPs are comparable to
`megafold`, but memory access patterns differ.
