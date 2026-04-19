"""Stage A: Explore MegaFold's Pairformer attention module structure.

Goal
----
Find the PyTorch module names and input / output tensor shapes for the three
attention types in each Pairformer block so we know exactly where to hook to
capture the ``pair_bias`` tensor.

Codebase note
-------------
In MegaFold, ``pair_bias`` is **computed inside** each attention module from
``pairwise_repr`` via a ``to_attn_bias`` submodule; it is *not* passed as an
input argument. Concretely:

* ``AttentionPairBias.to_attn_bias`` is
  ``Sequential(LayernormLinear, Rearrange("b ... h -> b h ..."))`` and emits a
  tensor of shape ``[b, heads, i, j]`` (see ``megafold/model/megafold.py:570``).
* ``TriangleAttention.to_attn_bias`` is
  ``Sequential(LinearNoBias, Rearrange("... i j h -> ... h i j"))`` and emits a
  tensor of shape ``[b, heads, i, j]`` (see ``megafold/model/megafold.py:656``).

So the right hook point is the **output** of each module's ``to_attn_bias``
submodule. This script confirms that layout and prints a sample shape.

Usage
-----
    python scripts/pair_bias_measurement/explore_model.py \
        --config configs/megafold_1x1.yaml \
        --n 384
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import torch
from torch import nn

from megafold.configs import create_megafold_from_yaml
from megafold.model.megafold import AttentionPairBias, TriangleAttention


# ---------------------------------------------------------------------------
# module discovery
# ---------------------------------------------------------------------------


PAIR_BIAS_CLASSES: Tuple[type, ...] = (AttentionPairBias, TriangleAttention)


def find_pair_bias_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return ``(qualified_name, module)`` for every pair-bias-emitting module.

    A pair-bias-emitting module is one that contains a ``to_attn_bias``
    submodule whose output is the ``[b, heads, i, j]`` bias tensor.
    """
    found: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, PAIR_BIAS_CLASSES) and hasattr(module, "to_attn_bias"):
            found.append((name, module))
    return found


# ---------------------------------------------------------------------------
# hook that records first call's I/O shapes for a single module
# ---------------------------------------------------------------------------


def _tensor_summary(x: object) -> str:
    if isinstance(x, torch.Tensor):
        return f"Tensor shape={tuple(x.shape)} dtype={x.dtype}"
    return f"{type(x).__name__}"


class ShapeRecorder:
    """Callable forward-hook that records inputs / outputs on the first call."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.recorded: Dict[str, List[str]] | None = None

    def __call__(self, module: nn.Module, inputs: tuple, output: object) -> None:
        if self.recorded is not None:
            return
        in_lines = [f"  [{i}] {_tensor_summary(x)}" for i, x in enumerate(inputs)]
        if isinstance(output, torch.Tensor):
            out_lines = [f"  {_tensor_summary(output)}"]
        elif isinstance(output, (tuple, list)):
            out_lines = [f"  [{i}] {_tensor_summary(x)}" for i, x in enumerate(output)]
        else:
            out_lines = [f"  {type(output).__name__}"]
        self.recorded = {"inputs": in_lines, "outputs": out_lines}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _build_synthetic_inputs(
    model: nn.Module, batch: int, n: int, device: torch.device, dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    """Build a minimal input dict that exercises ``model.pairformer_stack``.

    We feed the Pairformer stack directly instead of running the full MegaFold
    forward pass, because the full pass requires atom / MSA / template inputs
    that are expensive to synthesize. Every attention module in the Pairformer
    is still driven, which is the only place pair_bias lives.
    """
    dim_single = getattr(model, "dim_single", 384)
    dim_pairwise = getattr(model, "dim_pairwise", 128)

    single_repr = torch.randn(batch, n, dim_single, device=device, dtype=dtype)
    pairwise_repr = torch.randn(batch, n, n, dim_pairwise, device=device, dtype=dtype)
    mask = torch.ones(batch, n, device=device, dtype=torch.bool)
    return {"single_repr": single_repr, "pairwise_repr": pairwise_repr, "mask": mask}


def main(config_path: str, n: int, batch: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[explore] loading model from {config_path} …")
    model = create_megafold_from_yaml(config_path, dotpath="model")
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    modules = find_pair_bias_modules(model)
    pairformer_only = [(nm, m) for nm, m in modules if "pairformer" in nm.lower()]
    print(f"[explore] found {len(modules)} pair-bias modules total")
    print(f"[explore] of which {len(pairformer_only)} live under a *pairformer* path")

    if not pairformer_only:
        print("[explore] no pairformer modules found; nothing more to do")
        return

    # tally by class / by role so the user sees 48 × 3 quickly
    by_kind: Dict[str, int] = {}
    for nm, m in pairformer_only:
        key = f"{type(m).__name__}/{_role(nm)}"
        by_kind[key] = by_kind.get(key, 0) + 1
    print("[explore] counts by (class / role):")
    for k, v in sorted(by_kind.items()):
        print(f"    {k:60s} {v}")

    # register one shape recorder on the first module AND on its to_attn_bias
    first_name, first_module = pairformer_only[0]
    module_rec = ShapeRecorder(first_name)
    subm_rec = ShapeRecorder(f"{first_name}.to_attn_bias")
    handles = [
        first_module.register_forward_hook(module_rec),
        first_module.to_attn_bias.register_forward_hook(subm_rec),
    ]

    print(f"\n[explore] running forward on pairformer_stack with synthetic N={n}, B={batch}")
    inputs = _build_synthetic_inputs(model, batch=batch, n=n, device=device, dtype=dtype)
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            model.pairformer_stack(**inputs)
    finally:
        for h in handles:
            h.remove()

    print("\n" + "=" * 72)
    print(f"First module: {first_name}")
    print(f"Type:         {type(first_module).__name__}")
    print("-" * 72)
    if module_rec.recorded is None:
        print("(hook did not fire — forward pass did not reach this module)")
    else:
        print("MODULE INPUTS:")
        for line in module_rec.recorded["inputs"]:
            print(line)
        print("MODULE OUTPUT:")
        for line in module_rec.recorded["outputs"]:
            print(line)
    print("-" * 72)
    if subm_rec.recorded is None:
        print("(to_attn_bias hook did not fire)")
    else:
        print("to_attn_bias INPUTS:")
        for line in subm_rec.recorded["inputs"]:
            print(line)
        print("to_attn_bias OUTPUT:  <-- THIS IS THE pair_bias TENSOR")
        for line in subm_rec.recorded["outputs"]:
            print(line)
    print("=" * 72)


def _role(qualified_name: str) -> str:
    """Best-effort role label (``triangle_start`` / ``triangle_end`` / ``pair_bias``)."""
    n = qualified_name.lower()
    if "triangle" in n and "start" in n:
        return "triangle_attention_starting"
    if "triangle" in n and "end" in n:
        return "triangle_attention_ending"
    if "tri_att" in n:
        return "triangle_attention"
    return "attention_pair_bias"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/megafold_1x1.yaml")
    parser.add_argument("--n", type=int, default=384, help="sequence length")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    main(args.config, n=args.n, batch=args.batch)
