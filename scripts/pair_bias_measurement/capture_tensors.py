"""Stage B: Capture pair_bias tensors from every Pairformer attention module.

Registers forward hooks on the ``to_attn_bias`` submodule of every
``AttentionPairBias`` / ``TriangleAttention`` in the Pairformer stack, runs one
forward pass, and saves all captured tensors to a ``.pt`` file keyed by the
parent module's qualified name.

The captured object is a ``dict[str, torch.Tensor]`` where each value has
shape ``[B, heads, i, j]`` in ``float32`` on CPU. The analysis notebook
(``analyze_patterns.ipynb``) consumes this file directly.

Why hook ``to_attn_bias`` output, not the attention module's inputs?
  In this codebase ``pair_bias`` is *computed* inside each attention module
  from ``pairwise_repr`` — it is never a positional argument. See
  ``megafold/model/megafold.py:626`` (AttentionPairBias) and
  ``megafold/model/megafold.py:684`` (TriangleAttention). The output of
  ``to_attn_bias`` is exactly the ``[b, heads, i, j]`` tensor we want.

Usage
-----
    # default: drive Pairformer stack with synthetic inputs, no data needed
    python scripts/pair_bias_measurement/capture_tensors.py \
        --config configs/megafold_1x1.yaml \
        --n 384 \
        --output captures/run1.pt

    # optional: full model forward using the trainer dataloader + a checkpoint
    python scripts/pair_bias_measurement/capture_tensors.py \
        --conductor-config configs/megafold_1x1.yaml \
        --trainer-name initial_training \
        --full-forward \
        --output captures/run1.pt
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch import nn

from megafold.configs import create_megafold_from_yaml
from megafold.model.megafold import AttentionPairBias, TriangleAttention


PAIR_BIAS_CLASSES: Tuple[type, ...] = (AttentionPairBias, TriangleAttention)


# ---------------------------------------------------------------------------
# hook installation
# ---------------------------------------------------------------------------


def install_pair_bias_hooks(
    model: nn.Module,
    captures: Dict[str, List[torch.Tensor]],
    only_pairformer: bool = True,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Install a forward hook on every ``to_attn_bias`` submodule.

    The hook saves ``output.detach().cpu().float()`` under the *parent* module's
    qualified name so keys match the spec ("…blocks.k.attention_pair_bias").
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for name, module in model.named_modules():
        if not isinstance(module, PAIR_BIAS_CLASSES):
            continue
        if only_pairformer and "pairformer" not in name.lower():
            continue
        if not hasattr(module, "to_attn_bias"):
            continue

        key = _canonical_key(name, module)

        def _make_hook(key: str):
            def _hook(_m, _inputs, output):
                if not isinstance(output, torch.Tensor):
                    return
                if output.dim() != 4:
                    # defensively skip anything we do not recognize
                    return
                captures[key].append(output.detach().to("cpu", dtype=torch.float32))

            return _hook

        handles.append(module.to_attn_bias.register_forward_hook(_make_hook(key)))

    return handles


def _canonical_key(qualified_name: str, module: nn.Module) -> str:
    """Map a raw ``named_modules`` path to a spec-friendly key.

    Examples::

        pairformer_stack.layers.0.0.tri_att_start  -> pairformer_stack.blocks.0.triangle_attention_starting
        pairformer_stack.layers.0.0.tri_att_end    -> pairformer_stack.blocks.0.triangle_attention_ending
        pairformer_stack.layers.0.1.fn             -> pairformer_stack.blocks.0.attention_pair_bias
    """
    # find the "...layers.<idx>..." anchor if present
    parts = qualified_name.split(".")
    block_idx = None
    for i, p in enumerate(parts):
        if p in ("layers", "blocks") and i + 1 < len(parts) and parts[i + 1].isdigit():
            block_idx = int(parts[i + 1])
            break

    if isinstance(module, TriangleAttention):
        # `need_transpose` is True only on the ending variant.
        role = "triangle_attention_ending" if getattr(module, "need_transpose", False) else "triangle_attention_starting"
    elif isinstance(module, AttentionPairBias):
        role = "attention_pair_bias"
    else:  # should not happen given the filter
        role = type(module).__name__.lower()

    if block_idx is None:
        return f"{qualified_name}::{role}"
    # try to keep everything up to the "layers" segment as the stack prefix
    prefix = qualified_name.split(".layers.")[0] if ".layers." in qualified_name else \
        qualified_name.split(".blocks.")[0]
    return f"{prefix}.blocks.{block_idx}.{role}"


# ---------------------------------------------------------------------------
# forward drivers
# ---------------------------------------------------------------------------


def _synthetic_pairformer_forward(
    model: nn.Module, batch: int, n: int, device: torch.device, dtype: torch.dtype
) -> None:
    """Drive ``model.pairformer_stack`` with synthetic inputs.

    This exercises every pair_bias module in the stack without needing MSAs,
    templates, or a real data pipeline. The spec accepts synthetic inputs as a
    fallback path (Part 3, Unknown 2) — and for pattern *measurement* random
    pairwise_repr is only useful if weights are trained; see the captured
    file's ``_meta["weights_initialized"]`` flag for the caveat.
    """
    dim_single = getattr(model, "dim_single", 384)
    dim_pairwise = getattr(model, "dim_pairwise", 128)

    single_repr = torch.randn(batch, n, dim_single, device=device, dtype=dtype)
    pairwise_repr = torch.randn(batch, n, n, dim_pairwise, device=device, dtype=dtype)
    mask = torch.ones(batch, n, device=device, dtype=torch.bool)

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")
    ):
        model.pairformer_stack(single_repr=single_repr, pairwise_repr=pairwise_repr, mask=mask)


def _full_model_forward(
    conductor_config: str, trainer_name: str, model_hooks_installer
) -> nn.Module:
    """Load the full trainer and run one batch. Returns the underlying model."""
    from megafold.configs import create_trainer_from_conductor_yaml

    trainer = create_trainer_from_conductor_yaml(conductor_config, trainer_name=trainer_name)
    try:
        trainer.load_from_checkpoint_folder()
    except Exception as exc:  # pragma: no cover — best effort only
        print(f"[capture] WARN: could not load checkpoint ({exc}); using initialized weights")

    model = trainer.model
    model.eval()

    # let the caller install hooks now that the real model is ready
    model_hooks_installer(model)

    batch = next(iter(trainer.dataloader))
    device = next(model.parameters()).device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        model(batch)
    return model


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _weights_look_initialized(model: nn.Module) -> bool:
    """Heuristic: has ``to_attn_bias`` linear weight been moved off zero init?

    ``AttentionPairBias`` zeros its ``to_attn_bias.linear_weight`` at
    construction time (see ``megafold/model/megafold.py:569``), so any nonzero
    value is a signal that *some* training or load happened.
    """
    for _, module in model.named_modules():
        if isinstance(module, AttentionPairBias) and hasattr(module, "to_attn_bias"):
            for p in module.to_attn_bias.parameters():
                if p.abs().sum().item() > 0:
                    return True
    return False


def main(
    config: str,
    output: str,
    n: int,
    batch: int,
    full_forward: bool,
    conductor_config: str | None,
    trainer_name: str | None,
) -> None:
    captures: Dict[str, List[torch.Tensor]] = defaultdict(list)
    handles: List[torch.utils.hooks.RemovableHandle] = []

    if full_forward:
        assert conductor_config and trainer_name, (
            "--full-forward requires --conductor-config and --trainer-name"
        )

        def _install(real_model: nn.Module) -> None:
            handles.extend(install_pair_bias_hooks(real_model, captures))
            print(f"[capture] installed {len(handles)} hooks on full model")

        model = _full_model_forward(conductor_config, trainer_name, _install)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        print(f"[capture] loading model from {config} …")
        model = create_megafold_from_yaml(config, dotpath="model")
        model = model.to(device=device, dtype=torch.float32)
        model.eval()
        handles.extend(install_pair_bias_hooks(model, captures))
        print(f"[capture] installed {len(handles)} hooks; running pairformer forward …")
        _synthetic_pairformer_forward(model, batch=batch, n=n, device=device, dtype=dtype)

    for h in handles:
        h.remove()

    # keep the first call only — each module is called once per forward.
    saved: Dict[str, torch.Tensor] = {
        name: tensors[0] for name, tensors in captures.items() if tensors
    }

    meta = {
        "num_captures": len(saved),
        "full_forward": full_forward,
        "weights_initialized": _weights_look_initialized(model),
        "config": conductor_config if full_forward else config,
        "trainer_name": trainer_name if full_forward else None,
        "n": None if full_forward else n,
        "batch": None if full_forward else batch,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    torch.save({"_meta": meta, "captures": saved}, output)

    total_mb = sum(t.element_size() * t.numel() for t in saved.values()) / 1e6
    print(f"[capture] saved {len(saved)} tensors ({total_mb:.1f} MB) to {output}")
    print(f"[capture] meta: {meta}")
    if not meta["weights_initialized"]:
        print(
            "[capture] WARNING: to_attn_bias weights appear to be at their zero "
            "init. Patterns measured on this capture will reflect random-weight "
            "noise, not learned structure. Load a checkpoint or use --full-forward "
            "with a trained model for meaningful results."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/megafold_1x1.yaml",
                        help="model yaml for the synthetic pairformer-only path")
    parser.add_argument("--output", type=str, default="captures/run1.pt")
    parser.add_argument("--n", type=int, default=384)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--full-forward", action="store_true",
                        help="drive the full MegaFold model via the trainer dataloader")
    parser.add_argument("--conductor-config", type=str, default=None)
    parser.add_argument("--trainer-name", type=str, default=None)
    args = parser.parse_args()
    main(
        config=args.config,
        output=args.output,
        n=args.n,
        batch=args.batch,
        full_forward=args.full_forward,
        conductor_config=args.conductor_config,
        trainer_name=args.trainer_name,
    )
