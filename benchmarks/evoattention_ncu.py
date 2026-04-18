"""
NCU (Nsight Compute) profiling target for EvoAttention.

Designed to be run under `ncu --profile-from-start off` so that only the
profiled region (delimited by cudaProfilerStart/Stop) is captured.
Warmup iterations run first to trigger Triton autotuning and GPU caching
before the profiler capture window opens.

--------------------------------------------------------------------
Typical usage
--------------------------------------------------------------------

1. SM / memory utilization summary:
   ncu --profile-from-start off \\
       --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\\
lts__throughput.avg.pct_of_peak_sustained_elapsed \\
       -o evoattention_util \\
       python benchmarks/evoattention_ncu.py --provider triton --n-ctx 512

2. Roofline / compute vs memory breakdown:
   ncu --profile-from-start off \\
       --set roofline \\
       -o evoattention_roofline \\
       python benchmarks/evoattention_ncu.py --provider triton --n-ctx 512

3. Full detailed report (slow – replays every kernel many times):
   ncu --profile-from-start off \\
       --set full \\
       -o evoattention_full \\
       python benchmarks/evoattention_ncu.py --provider triton --n-ctx 512

4. SFU (exp/softmax) throughput vs matmul:
   ncu --profile-from-start off \\
       --metrics smsp__sass_thread_inst_executed_op_sfma_pred_on.sum,\\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\\
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum \\
       -o evoattention_sfu \\
       python benchmarks/evoattention_ncu.py --provider triton --n-ctx 512

Flags
-----
  --provider   {triton, deepspeed, torch}   kernel to profile
  --mode       {fwd, bwd, full}             which pass to profile
  --n-ctx      int                          sequence length (default 512)
  --warmup     int                          warmup iterations before profiler capture (default 20)
  --iters      int                          profiled iterations (default 5)
"""

import argparse
import torch
import torch.cuda.nvtx as nvtx

# ── parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NCU profiling target for EvoAttention")
parser.add_argument("--provider", choices=["triton", "deepspeed", "torch"], default="triton")
parser.add_argument("--mode",     choices=["fwd", "bwd", "full"],           default="full")
parser.add_argument("--n-ctx",    type=int, default=512,   help="Sequence length to profile")
parser.add_argument("--warmup",   type=int, default=20,    help="Warmup iterations (triggers autotuning)")
parser.add_argument("--iters",    type=int, default=5,     help="Profiled iterations")
args = parser.parse_args()

# ── lazy imports (avoid loading DS / Triton unless needed) ────────────────────
if args.provider in ("triton",):
    from megafold.model.FusedEvoAttention.untuned_evoattention import TritonEvoformer
if args.provider == "deepspeed":
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

# ── constants (must match evoattention_speed.py) ──────────────────────────────
BATCH, N_HEADS, HEAD_DIM, N_SEQ = 4, 16, 64, 1
H       = N_HEADS
N_CTX   = args.n_ctx
dtype   = torch.bfloat16
device  = "cuda"

# ── helper ────────────────────────────────────────────────────────────────────
def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def _torch_attention(q, k, v, res_mask, pair_bias):
    softmax_scale = 1 / (q.shape[-1] ** 0.5)
    ref_P = torch.matmul(q * softmax_scale, k.transpose(3, 4)) + pair_bias
    ref_P = ref_P.masked_fill(~res_mask, max_neg_value(ref_P))
    ref_P = torch.softmax(ref_P.float(), dim=-1).to(q.dtype)
    return torch.matmul(ref_P, v)

# ── build tensors & fn ────────────────────────────────────────────────────────
if args.provider == "triton":
    q = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    res_mask  = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, N_CTX), dtype=torch.bool,   device=device)
    pair_bias = torch.randn((BATCH, 1, H, N_CTX, N_CTX), dtype=torch.float32,        device=device, requires_grad=True)

    if args.mode == "fwd":
        def fn():
            return TritonEvoformer(q, k, v, res_mask, pair_bias)
    elif args.mode == "bwd":
        o = TritonEvoformer(q, k, v, res_mask, pair_bias)          # build graph once
        do = torch.randn_like(o)
        def fn():
            return o.backward(do, retain_graph=True)
    else:  # full
        _o_shape = (BATCH, N_SEQ, N_CTX, H, HEAD_DIM)
        do = torch.randn(_o_shape, dtype=dtype, device=device)  # fixed grad, no randn inside fn
        def fn():
            o = TritonEvoformer(q, k, v, res_mask, pair_bias)
            o.backward(do, retain_graph=True)

elif args.provider == "deepspeed":
    q = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, N_SEQ, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    res_mask  = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, N_CTX), dtype=torch.bool,   device=device).bfloat16()
    pair_bias = torch.randn((BATCH, 1, H, N_CTX, N_CTX), dtype=dtype,               device=device, requires_grad=True)

    if args.mode == "fwd":
        def fn():
            return DS4Sci_EvoformerAttention(q, k, v, [res_mask, pair_bias])
    elif args.mode == "bwd":
        o = DS4Sci_EvoformerAttention(q, k, v, [res_mask, pair_bias])
        do = torch.randn_like(o)
        def fn():
            return o.backward(do, retain_graph=True)
    else:
        _o_shape = (BATCH, N_SEQ, N_CTX, H, HEAD_DIM)
        do = torch.randn(_o_shape, dtype=dtype, device=device)  # fixed grad, no randn inside fn
        def fn():
            o = DS4Sci_EvoformerAttention(q, k, v, [res_mask, pair_bias])
            o.backward(do, retain_graph=True)

else:  # torch
    q = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_SEQ, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    res_mask  = torch.randint(0, 2, (BATCH, 1, N_SEQ, 1, N_CTX), dtype=torch.bool,    device=device)
    pair_bias = torch.randn((BATCH, H, 1, N_CTX, N_CTX), dtype=torch.float32,         device=device, requires_grad=True)

    if args.mode == "fwd":
        def fn():
            return _torch_attention(q, k, v, res_mask, pair_bias)
    elif args.mode == "bwd":
        o = _torch_attention(q, k, v, res_mask, pair_bias).reshape(BATCH, N_SEQ, N_CTX, H, HEAD_DIM)
        do = torch.randn_like(o)
        def fn():
            return o.backward(do, retain_graph=True)
    else:
        _o_shape = (BATCH, N_SEQ, N_CTX, H, HEAD_DIM)
        do = torch.randn(_o_shape, dtype=dtype, device=device)  # fixed grad, no randn inside fn
        def fn():
            o = _torch_attention(q, k, v, res_mask, pair_bias).reshape(BATCH, N_SEQ, N_CTX, H, HEAD_DIM)
            o.backward(do, retain_graph=True)

# ── warmup (triggers Triton autotuning; runs outside profiler capture) ─────────
print(f"[ncu target] provider={args.provider}  mode={args.mode}  N_CTX={N_CTX}")
print(f"[ncu target] warming up {args.warmup} iterations ...")
for _ in range(args.warmup):
    fn()
torch.cuda.synchronize()

# ── profiled region ────────────────────────────────────────────────────────────
# NCU captures everything between cudaProfilerStart and cudaProfilerStop.
# Run with:  ncu --profile-from-start off ...
print(f"[ncu target] starting profiler capture ({args.iters} iterations) ...")
torch.cuda.cudart().cudaProfilerStart()
for i in range(args.iters):
    nvtx.range_push(f"evoattention_{args.provider}_{args.mode}_iter{i}")
    fn()
    nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("[ncu target] profiler capture done.")
