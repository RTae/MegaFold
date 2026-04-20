"""Stage A — Explore MegaFold's attention module hierarchy.

Run via the same DeepSpeed launcher as training, for example:

    deepspeed --num_gpus=1 scripts/explore_pair_bias_modules.py \
        --config configs/megafold_1x1_smoke.yaml \
        --trainer_name initial_training

The script:
1. Builds the MegaFold Trainer (same code path as ``train.py``).
2. Walks ``trainer.model`` and prints every ``AttentionPairBias`` and
   ``TriangleAttention`` module it finds.
3. Shows the ``to_attn_bias`` submodule shape for the first match so you can
   confirm where the pair-bias tensor is produced.

It does NOT run a forward pass, so it is cheap and safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path regardless of where DeepSpeed spawns the process.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from loguru import logger

from megafold.configs import create_trainer_from_conductor_yaml

TARGET_CLASSES = ("AttentionPairBias", "TriangleAttention")


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap common wrappers (Fabric / DeepSpeed / DDP) to the raw MegaFold module."""
    for attr in ("module", "_forward_module"):
        inner = getattr(model, attr, None)
        if isinstance(inner, torch.nn.Module):
            return _unwrap(inner)
    return model


def list_attention_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    hits = []
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if any(tok in class_name for tok in TARGET_CLASSES):
            hits.append((name, module))
    return hits


def describe_to_attn_bias(module: torch.nn.Module) -> str:
    to_attn_bias = getattr(module, "to_attn_bias", None)
    if to_attn_bias is None:
        return "  (no .to_attn_bias submodule)"

    lines = [f"  to_attn_bias: {type(to_attn_bias).__name__}"]
    for i, child in enumerate(to_attn_bias.children() if hasattr(to_attn_bias, "children") else []):
        params = sum(p.numel() for p in child.parameters())
        lines.append(f"    [{i}] {type(child).__name__}  params={params}")
    return "\n".join(lines)


def main(config_path: str, trainer_name: str) -> None:
    assert os.path.exists(config_path), f"Config file not found at {config_path}."
    torch.set_float32_matmul_precision("high")

    logger.info("Building trainer from {}", config_path)
    trainer = create_trainer_from_conductor_yaml(config_path, trainer_name=trainer_name)

    model = _unwrap(trainer.model)
    hits = list_attention_modules(model)

    logger.info("Found {} attention modules", len(hits))
    print("=" * 90)
    print(f"Total attention modules matching {TARGET_CLASSES}: {len(hits)}")
    print("=" * 90)

    for name, module in hits:
        print(f"- {name}  ({type(module).__name__})")

    if not hits:
        print("No matching modules were found; check TARGET_CLASSES for this build.")
        return

    print()
    print("=" * 90)
    print("First module details — inspect its pair-bias projection submodule")
    print("=" * 90)
    first_name, first_mod = hits[0]
    print(first_name)
    print(describe_to_attn_bias(first_mod))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore MegaFold pair-bias modules.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--trainer_name", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    main(args.config, args.trainer_name)
