"""Stage B — Capture structured pair_bias tensors from MegaFold.

Pair_bias in MegaFold is NOT a module input; it is computed inside each
attention module as::

    attn_bias = module.to_attn_bias(pairwise_repr)

so this script hooks the forward output of each ``to_attn_bias`` submodule
inside every ``AttentionPairBias`` / ``TriangleAttention``. The first call
per module is saved, keyed by the module's fully qualified path.

The script itself runs inside the normal DeepSpeed launcher, so the easiest
way to use it is via the smoke config with ``num_train_steps: 1``:

    deepspeed --num_gpus=1 scripts/capture_pair_bias_structured.py \
        --config configs/megafold_1x1_smoke.yaml \
        --trainer_name initial_training \
        --output outputs/pair_bias_structured/run.pt

After the first training step finishes, the script saves the captured tensors
and lets the trainer exit.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from loguru import logger

from megafold.configs import create_trainer_from_conductor_yaml

TARGET_CLASSES = ("AttentionPairBias", "TriangleAttention")


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    for attr in ("module", "_forward_module"):
        inner = getattr(model, attr, None)
        if isinstance(inner, torch.nn.Module):
            return _unwrap(inner)
    return model


def register_pair_bias_hooks(
    model: torch.nn.Module,
    captures: dict,
    *,
    pairformer_only: bool,
) -> list:
    handles = []
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if not any(tok in class_name for tok in TARGET_CLASSES):
            continue
        if pairformer_only and "pairformer" not in name.lower():
            continue

        to_attn_bias = getattr(module, "to_attn_bias", None)
        if to_attn_bias is None:
            continue

        hook_name = f"{name}.to_attn_bias"

        def make_hook(key: str):
            def _hook(_mod, _inputs, output):
                if key in captures:
                    return  # first call only
                if not torch.is_tensor(output):
                    return
                if output.dim() not in (4, 5):
                    return
                captures[key] = output.detach().to(dtype=torch.float32, device="cpu").clone()

            return _hook

        handles.append(to_attn_bias.register_forward_hook(make_hook(hook_name)))

    return handles


def save_captures(captures: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captures, output_path)
    total_bytes = sum(t.element_size() * t.numel() for t in captures.values())
    logger.info(
        "Saved {} captures totaling {:.2f} MB to {}",
        len(captures),
        total_bytes / 1e6,
        output_path,
    )


def main(config_path: str, trainer_name: str, output: str, pairformer_only: bool) -> None:
    assert os.path.exists(config_path), f"Config file not found at {config_path}."
    torch.set_float32_matmul_precision("high")

    output_path = Path(output)

    logger.info("Building trainer from {}", config_path)
    trainer = create_trainer_from_conductor_yaml(config_path, trainer_name=trainer_name)

    model = _unwrap(trainer.model)
    captures: dict[str, torch.Tensor] = {}
    handles = register_pair_bias_hooks(model, captures, pairformer_only=pairformer_only)
    logger.info(
        "Registered {} to_attn_bias hooks (pairformer_only={})",
        len(handles),
        pairformer_only,
    )

    try:
        trainer.load_from_checkpoint_folder()
    except Exception as exc:  # noqa: BLE001 - trainer utility may be strict about optional ckpts
        logger.warning("load_from_checkpoint_folder failed: {}", exc)

    try:
        trainer()
    except Exception as exc:  # noqa: BLE001 - we still want to flush whatever we captured
        logger.warning("trainer() raised ({}); flushing captures anyway.", exc)
    finally:
        for handle in handles:
            handle.remove()

        if captures:
            save_captures(captures, output_path)
        else:
            logger.warning("No pair-bias tensors were captured. Check TARGET_CLASSES / filters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture MegaFold pair_bias tensors.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--trainer_name", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/pair_bias_structured/run.pt",
        help="Where to write the {module_name: tensor} torch.save file.",
    )
    parser.add_argument(
        "--include_all",
        action="store_true",
        help="Also capture non-Pairformer attention modules (atom transformer, diffusion, etc.).",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    main(
        args.config,
        args.trainer_name,
        output=args.output,
        pairformer_only=not args.include_all,
    )
