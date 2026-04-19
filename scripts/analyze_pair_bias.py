#!/usr/bin/env python3
"""Summarize structural regularity in captured pair-bias tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def normalize_pair_bias(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    tensor = tensor.detach().float().cpu()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim >= 4:
        tensor = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])

    if tensor.ndim != 3:
        raise ValueError(f"Expected a 2D/3D/4D/5D pair-bias tensor, got shape {tuple(tensor.shape)}")

    is_square = tensor.shape[-1] == tensor.shape[-2]
    return tensor, is_square


def summarize_pair_bias(tensor: torch.Tensor) -> dict:
    heads, is_square = normalize_pair_bias(tensor)
    q_len, k_len = heads.shape[-2], heads.shape[-1]

    summary = {
        "shape": list(heads.shape),
        "is_square": is_square,
        "mean": heads.mean().item(),
        "std": heads.std().item(),
        "min": heads.min().item(),
        "max": heads.max().item(),
        "row_smoothness_abs_mean": (heads[:, 1:, :] - heads[:, :-1, :]).abs().mean().item() if q_len > 1 else 0.0,
        "col_smoothness_abs_mean": (heads[:, :, 1:] - heads[:, :, :-1]).abs().mean().item() if k_len > 1 else 0.0,
    }

    if is_square:
        n = q_len
        symmetry_abs_mean = (heads - heads.transpose(-1, -2)).abs().mean().item()
        diag = heads.diagonal(dim1=-2, dim2=-1)
        diag_mean = diag.mean().item()

        if n > 1:
            offdiag_mask = ~torch.eye(n, dtype=torch.bool)
            offdiag_mean = heads[:, offdiag_mask].mean().item()
        else:
            offdiag_mean = 0.0

        distance_profile = {}
        for offset in range(min(n, 32)):
            diagonals = [torch.diagonal(heads, offset=offset, dim1=-2, dim2=-1).reshape(-1)]
            if offset > 0:
                diagonals.append(torch.diagonal(heads, offset=-offset, dim1=-2, dim2=-1).reshape(-1))
            merged = torch.cat(diagonals)
            distance_profile[offset] = {
                "mean": merged.mean().item(),
                "std": merged.std().item(),
            }

        low_rank_energy = []
        for head in heads[: min(4, heads.shape[0])]:
            singular_values = torch.linalg.svdvals(head)
            energy = (singular_values.square()).sum()
            rank8_energy = (singular_values[: min(8, singular_values.numel())].square()).sum()
            low_rank_energy.append((rank8_energy / energy).item() if energy > 0 else 0.0)

        summary.update(
            {
                "symmetry_abs_mean": symmetry_abs_mean,
                "diag_mean": diag_mean,
                "offdiag_mean": offdiag_mean,
                "rank8_energy_fraction_first_heads": low_rank_energy,
                "distance_profile": distance_profile,
            }
        )
    else:
        midpoint = k_len // 2
        summary.update(
            {
                "left_half_mean": heads[..., :midpoint].mean().item() if midpoint > 0 else 0.0,
                "right_half_mean": heads[..., midpoint:].mean().item(),
                "edge_column_mean": torch.cat([heads[..., :1], heads[..., -1:]], dim=-1).mean().item(),
                "center_column_mean": heads[..., max(midpoint - 1, 0) : min(midpoint + 1, k_len)].mean().item(),
            }
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tensor_path", type=Path, help="Path to the captured .pt file")
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the summary as JSON",
    )
    args = parser.parse_args()

    tensor = torch.load(args.tensor_path, map_location="cpu")
    summary = summarize_pair_bias(tensor)
    print(json.dumps(summary, indent=2))

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
