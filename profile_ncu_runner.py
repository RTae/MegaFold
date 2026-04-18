#!/usr/bin/env python3
"""Minimal Python entrypoint for Nsight Compute profiling."""

import argparse
import importlib
import math
import os
import sys

import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


BATCH = 1
N_SEQ = 1
N_CTX = 384
N_HEADS = 4
HEAD_DIM = 32
RANK = 8
QKV_DTYPE = torch.bfloat16
PAIR_BIAS_DTYPE = torch.float32
DEVICE = torch.device("cuda")


def ensure_cuda_available():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Nsight Compute profiling.")


def max_neg_value(tensor):
    """Return the most negative finite value for a tensor dtype."""
    return -torch.finfo(tensor.dtype).max


def _run_profile(fn, warmup, iters, impl_name, mode):
    """Run warmup, then profiled iterations inside cudaProfilerStart/Stop."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(iters):
        nvtx.range_push(f"{impl_name}_{mode}_iter{i}")
        fn()
        nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def _build_mode_fn(mode, forward_fn):
    """Create the callable used for profiling."""
    if mode == "fwd":
        return forward_fn
    if mode == "bwd":
        output = forward_fn()
        do = torch.randn_like(output)
        return lambda: output.backward(do, retain_graph=True)
    # full: pre-allocate do so torch.randn doesn't pollute the NCU trace
    probe = forward_fn()
    do = torch.randn_like(probe)
    del probe
    def full():
        o = forward_fn()
        o.backward(do, retain_graph=True)
    return full


def _warn_unavailable(name, exc):
    print(f"Warning: {name} is unavailable ({exc}). Skipping.")
    return False


def make_megafold_tensors(n_seq, n_ctx, n_heads, head_dim):
    """Create AF3-style Evoformer tensors."""
    q = torch.randn(
        (BATCH, n_seq, n_ctx, n_heads, head_dim),
        dtype=QKV_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    k = torch.randn(
        (BATCH, n_seq, n_ctx, n_heads, head_dim),
        dtype=QKV_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    v = torch.randn(
        (BATCH, n_seq, n_ctx, n_heads, head_dim),
        dtype=QKV_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    res_mask = torch.randint(
        0,
        2,
        (BATCH, n_seq, 1, 1, n_ctx),
        dtype=torch.bool,
        device=DEVICE,
    )
    pair_bias = torch.randn(
        (BATCH, 1, n_heads, n_ctx, n_ctx),
        dtype=PAIR_BIAS_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    return q, k, v, res_mask, pair_bias


def make_sdpa_tensors(n_seq, n_ctx, n_heads, head_dim):
    """Create baseline SDPA tensors with the torch layout."""
    q = torch.randn(
        (BATCH, n_heads, n_seq, n_ctx, head_dim),
        dtype=QKV_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    k = torch.randn(
        (BATCH, n_heads, n_seq, n_ctx, head_dim),
        dtype=QKV_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    v = torch.randn(
        (BATCH, n_heads, n_seq, n_ctx, head_dim),
        dtype=QKV_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    res_mask = torch.randint(
        0,
        2,
        (BATCH, 1, n_seq, 1, n_ctx),
        dtype=torch.bool,
        device=DEVICE,
    )
    pair_bias = torch.randn(
        (BATCH, n_heads, 1, n_ctx, n_ctx),
        dtype=PAIR_BIAS_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    return q, k, v, res_mask, pair_bias


def make_llm_qkv(n_ctx, n_heads, head_dim):
    """Create LLM-style attention tensors."""
    q = torch.randn(
        (BATCH, n_ctx, n_heads, head_dim),
        dtype=QKV_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    k = torch.randn(
        (BATCH, n_ctx, n_heads, head_dim),
        dtype=QKV_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    v = torch.randn(
        (BATCH, n_ctx, n_heads, head_dim),
        dtype=QKV_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    return q, k, v


def make_dense_bias(n_ctx, n_heads):
    """Create a dense pair bias tensor for LLM-style kernels."""
    return torch.randn(
        (BATCH, n_heads, n_ctx, n_ctx),
        dtype=PAIR_BIAS_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )


def make_low_rank_bias_factors(n_ctx, n_heads):
    """Create low-rank bias factors for FlashBias."""
    q_bias = torch.randn(
        (BATCH, n_heads, n_ctx, RANK),
        dtype=QKV_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    k_bias = torch.randn(
        (BATCH, n_heads, n_ctx, RANK),
        dtype=QKV_DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    return (
        q_bias.permute(0, 2, 1, 3).contiguous(),
        k_bias.permute(0, 2, 1, 3).contiguous(),
    )


def sdpa_with_bias(q, k, v, res_mask, pair_bias):
    """Torch SDPA baseline. Uses math/efficient backend (not FlashAttn) due to attn_mask."""
    softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    attn_mask = pair_bias.expand(-1, -1, q.shape[2], -1, -1)
    expanded_mask = res_mask.expand(-1, q.shape[1], -1, q.shape[3], -1)
    attn_mask = attn_mask.masked_fill(~expanded_mask, max_neg_value(attn_mask))
    output = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=softmax_scale,
        is_causal=False,
    )
    return output.permute(0, 2, 3, 1, 4).contiguous()


def run_sdpa(args):
    q, k, v, res_mask, pair_bias = make_sdpa_tensors(
        args.n_seq,
        args.n_ctx,
        args.n_heads,
        args.head_dim,
    )
    fn = _build_mode_fn(
        args.mode,
        lambda: sdpa_with_bias(q, k, v, res_mask, pair_bias),
    )
    _run_profile(fn, args.warmup, args.iters, "sdpa", args.mode)
    return True


def run_sdpa_no_bias(args):
    """Torch SDPA without bias — compute ceiling baseline. Only valid at N_SEQ=1."""
    if args.n_seq != 1:
        print("Warning: sdpa_no_bias flattens N_SEQ*N_CTX into one sequence. "
              "Results are only meaningful at N_SEQ=1. Skipping.")
        return False

    q, k, v, _, _ = make_megafold_tensors(
        args.n_seq,
        args.n_ctx,
        args.n_heads,
        args.head_dim,
    )
    batch, n_seq, n_ctx, n_heads, head_dim = q.shape
    softmax_scale = 1.0 / math.sqrt(head_dim)

    def forward():
        # Reshape to (B, N_HEADS, N_SEQ*N_CTX, HEAD_DIM) for SDPA
        q_flat = q.reshape(batch, n_seq * n_ctx, n_heads, head_dim).permute(0, 2, 1, 3)
        k_flat = k.reshape(batch, n_seq * n_ctx, n_heads, head_dim).permute(0, 2, 1, 3)
        v_flat = v.reshape(batch, n_seq * n_ctx, n_heads, head_dim).permute(0, 2, 1, 3)
        return F.scaled_dot_product_attention(
            query=q_flat,
            key=k_flat,
            value=v_flat,
            dropout_p=0.0,
            scale=softmax_scale,
            is_causal=False,
        )

    fn = _build_mode_fn(args.mode, forward)
    _run_profile(fn, args.warmup, args.iters, "sdpa_no_bias", args.mode)
    return True


def run_fa1_bias(args):
    """FA1-Triton+bias kernel — LLM-style layout, not directly comparable to megafold."""
    try:
        from flash_bias.flash_attn_triton import FlashAttnFunc
        attention_triton = FlashAttnFunc.apply
    except Exception as exc:
        return _warn_unavailable("flash_bias Triton", exc)

    q, k, v = make_llm_qkv(args.n_ctx, args.n_heads, args.head_dim)
    pair_bias = make_dense_bias(args.n_ctx, args.n_heads)
    softmax_scale = 1.0 / math.sqrt(args.head_dim)
    fn = _build_mode_fn(
        args.mode,
        lambda: attention_triton(q, k, v, pair_bias, False, softmax_scale),
    )
    _run_profile(fn, args.warmup, args.iters, "fa1_bias", args.mode)
    return True


def run_flashbias(args):
    """FlashBias Triton kernel — LLM-style layout, not directly comparable to megafold."""
    try:
        from flash_bias.flash_bias_triton import FlashBiasFunc
        flashbias_triton = FlashBiasFunc.apply
    except Exception as exc:
        return _warn_unavailable("flash_bias FlashBias", exc)

    q, k, v = make_llm_qkv(args.n_ctx, args.n_heads, args.head_dim)
    q_bias, k_bias = make_low_rank_bias_factors(args.n_ctx, args.n_heads)
    softmax_scale = 1.0 / math.sqrt(args.head_dim)
    fn = _build_mode_fn(
        args.mode,
        lambda: flashbias_triton(q, k, v, q_bias, k_bias, None, False, softmax_scale),
    )
    _run_profile(fn, args.warmup, args.iters, "flashbias", args.mode)
    return True


def run_megafold(args):
    try:
        from megafold.model.FusedEvoAttention.evoattention import TritonEvoformer
    except Exception as exc:
        return _warn_unavailable("MegaFold EvoAttention", exc)

    q, k, v, res_mask, pair_bias = make_megafold_tensors(
        args.n_seq,
        args.n_ctx,
        args.n_heads,
        args.head_dim,
    )
    fn = _build_mode_fn(
        args.mode,
        lambda: TritonEvoformer(q, k, v, res_mask, pair_bias),
    )
    _run_profile(fn, args.warmup, args.iters, "megafold", args.mode)
    return True


def run_fa3_no_bias(args):
    """FlashAttention-3 without bias — compute ceiling baseline. Only valid at N_SEQ=1."""
    if args.n_seq != 1:
        print("Warning: fa3 flattens N_SEQ*N_CTX into one sequence. "
              "Results are only meaningful at N_SEQ=1. Skipping.")
        return False

    flash_attn_func = None
    last_exc = None
    for module_name in (
        "flash_attn_interface",
        "flash_attn",
        "flash_attn.flash_attn_interface",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            last_exc = exc
            continue
        flash_attn_func = getattr(module, "flash_attn_func", None)
        if flash_attn_func is not None:
            break

    if flash_attn_func is None:
        return _warn_unavailable("FlashAttention-3", last_exc or "flash_attn_func not found")

    q, k, v, _, _ = make_megafold_tensors(
        args.n_seq,
        args.n_ctx,
        args.n_heads,
        args.head_dim,
    )
    batch, n_seq, n_ctx, n_heads, head_dim = q.shape

    def forward():
        q_flat = q.reshape(batch, n_seq * n_ctx, n_heads, head_dim)
        k_flat = k.reshape(batch, n_seq * n_ctx, n_heads, head_dim)
        v_flat = v.reshape(batch, n_seq * n_ctx, n_heads, head_dim)
        output = flash_attn_func(
            q_flat,
            k_flat,
            v_flat,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=False,
        )
        return output.reshape(batch, n_seq, n_ctx, n_heads, head_dim)

    fn = _build_mode_fn(
        args.mode,
        forward,
    )
    _run_profile(fn, args.warmup, args.iters, "fa3", args.mode)
    return True


def run_fa4_no_bias(args):
    """FlashAttention-4 (CuTE) without bias — compute ceiling baseline. Only valid at N_SEQ=1."""
    if args.n_seq != 1:
        print("Warning: fa4 flattens N_SEQ*N_CTX into one sequence. "
              "Results are only meaningful at N_SEQ=1. Skipping.")
        return False

    try:
        from flash_attn.cute import flash_attn_func
    except Exception as exc:
        return _warn_unavailable("FlashAttention-4", exc)

    q, k, v, _, _ = make_megafold_tensors(
        args.n_seq,
        args.n_ctx,
        args.n_heads,
        args.head_dim,
    )
    batch, n_seq, n_ctx, n_heads, head_dim = q.shape

    def forward():
        q_flat = q.reshape(batch, n_seq * n_ctx, n_heads, head_dim)
        k_flat = k.reshape(batch, n_seq * n_ctx, n_heads, head_dim)
        v_flat = v.reshape(batch, n_seq * n_ctx, n_heads, head_dim)
        result = flash_attn_func(q_flat, k_flat, v_flat, causal=False)
        output = result[0] if isinstance(result, tuple) else result
        return output.reshape(batch, n_seq, n_ctx, n_heads, head_dim)

    fn = _build_mode_fn(
        args.mode,
        forward,
    )
    _run_profile(fn, args.warmup, args.iters, "fa4", args.mode)
    return True


IMPLEMENTATIONS = {
    "sdpa": run_sdpa,
    "sdpa_no_bias": run_sdpa_no_bias,
    "fa1_bias": run_fa1_bias,
    "flashbias": run_flashbias,
    "megafold": run_megafold,
    "fa3": run_fa3_no_bias,
    "fa4": run_fa4_no_bias,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Dispatch one Pairformer attention implementation for ncu profiling."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("smoke", help="Check that Python can allocate a CUDA tensor.")

    run_parser = subparsers.add_parser("run", help="Run one implementation for profiling.")
    run_parser.add_argument("impl", choices=sorted(IMPLEMENTATIONS))
    run_parser.add_argument("--mode", choices=["fwd", "bwd", "full"], default="bwd")
    run_parser.add_argument("--n-ctx", type=int, default=N_CTX)
    run_parser.add_argument("--n-seq", type=int, default=N_SEQ)
    run_parser.add_argument("--n-heads", type=int, default=N_HEADS)
    run_parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    run_parser.add_argument("--warmup", type=int, default=5)
    run_parser.add_argument("--iters", type=int, default=3)
    run_parser.add_argument("--seed", type=int, default=0)

    return parser


def run_smoke_test():
    tensor = torch.randn(1, device=DEVICE)
    torch.cuda.synchronize()
    print(f"CUDA smoke test passed: {tensor.item():.6f}")


def main():
    args = build_parser().parse_args()
    ensure_cuda_available()

    if args.command == "smoke":
        run_smoke_test()
        return

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ran = IMPLEMENTATIONS[args.impl](args)
    if not ran:
        return

    print(
        f"Completed {args.impl} mode={args.mode} at N_CTX={args.n_ctx}, "
        f"N_SEQ={args.n_seq}, H={args.n_heads}, D={args.head_dim}, "
        f"warmup={args.warmup}, iters={args.iters}"
    )


if __name__ == "__main__":
    main()