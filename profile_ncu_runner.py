#!/usr/bin/env python3
"""
Minimal Python entrypoint for Nsight Compute profiling.

This keeps tensor construction and attention dispatch in Python so the shell
wrapper can focus on invoking ``ncu`` and reporting summaries.
"""

import argparse
import math
import os
import sys

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


BATCH = 1
N_TOKENS = 1218
N_HEADS = 16
HEAD_DIM = 128
RANK = 8
DTYPE = torch.float16
DEVICE = torch.device("cuda")
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def ensure_cuda_available():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Nsight Compute profiling.")


def make_qkv():
    q = torch.randn(BATCH, N_TOKENS, N_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def make_dense_bias():
    return torch.randn(BATCH, N_HEADS, N_TOKENS, N_TOKENS, device=DEVICE, dtype=DTYPE)


def make_low_rank_bias_factors():
    q_bias = torch.randn(BATCH, N_HEADS, N_TOKENS, RANK, device=DEVICE, dtype=DTYPE)
    k_bias = torch.randn_like(q_bias)
    return (
        q_bias.permute(0, 2, 1, 3).contiguous(),
        k_bias.permute(0, 2, 1, 3).contiguous(),
    )


def run_sdpa(warmup, iters):
    from flash_bias.attention_func import attention_sdpa

    q, k, v = make_qkv()
    bias = make_dense_bias()

    for _ in range(warmup):
        attention_sdpa(q, k, v, SOFTMAX_SCALE, bias, False)
    torch.cuda.synchronize()

    for _ in range(iters):
        attention_sdpa(q, k, v, SOFTMAX_SCALE, bias, False)
    torch.cuda.synchronize()


def run_triton(warmup, iters):
    from flash_bias.attention_func import attention_triton

    q, k, v = make_qkv()
    bias = make_dense_bias()

    for _ in range(warmup):
        attention_triton(q, k, v, bias, False, SOFTMAX_SCALE)
    torch.cuda.synchronize()

    for _ in range(iters):
        attention_triton(q, k, v, bias, False, SOFTMAX_SCALE)
    torch.cuda.synchronize()


def run_flashbias(warmup, iters):
    from flash_bias.attention_func import flashbias_sdpa

    q, k, v = make_qkv()
    q_bias, k_bias = make_low_rank_bias_factors()

    for _ in range(warmup):
        flashbias_sdpa(q, k, v, q_bias, k_bias, SOFTMAX_SCALE, None, False)
    torch.cuda.synchronize()

    for _ in range(iters):
        flashbias_sdpa(q, k, v, q_bias, k_bias, SOFTMAX_SCALE, None, False)
    torch.cuda.synchronize()


IMPLEMENTATIONS = {
    "sdpa": run_sdpa,
    "triton": run_triton,
    "flashbias": run_flashbias,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Dispatch one Pairformer attention implementation for ncu profiling."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("smoke", help="Check that Python can allocate a CUDA tensor.")

    run_parser = subparsers.add_parser("run", help="Run one implementation for profiling.")
    run_parser.add_argument("impl", choices=sorted(IMPLEMENTATIONS))
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
    IMPLEMENTATIONS[args.impl](args.warmup, args.iters)
    print(
        f"Completed {args.impl} at N={N_TOKENS}, H={N_HEADS}, D={HEAD_DIM}, "
        f"warmup={args.warmup}, iters={args.iters}"
    )


if __name__ == "__main__":
    main()