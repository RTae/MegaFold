# Pair-bias data-pattern measurement

Three-stage workflow that measures four structural properties of the
`pair_bias` tensors produced inside each Pairformer attention module —
**symmetry**, **tile-level spatial sparsity**, **temporal reuse across blocks**,
and **cross-head correlation** — so you can decide whether a data-aware kernel
is worth building.

## Files

| Stage | File                          | Purpose                                                                             |
|-------|-------------------------------|-------------------------------------------------------------------------------------|
| A     | `explore_model.py`            | Enumerate the 48 × 3 Pairformer attention modules and print I/O shapes.             |
| B     | `capture_tensors.py`          | Hook every `to_attn_bias` submodule, run one forward pass, save tensors to `.pt`.   |
| C     | `analyze_patterns.ipynb`      | Load captures, compute the four metrics, plot, print a strength summary.           |

## Codebase-specific hook point

In MegaFold, `pair_bias` is **not an input** to attention modules — it is computed
inside them from `pairwise_repr` via a `to_attn_bias` submodule. The right hook
point is therefore the **output of `to_attn_bias`**, which has shape
`[B, heads, i, j]`:

* `AttentionPairBias.to_attn_bias` — `megafold/model/megafold.py:570`
* `TriangleAttention.to_attn_bias` — `megafold/model/megafold.py:656`

Both `capture_tensors.py` and `explore_model.py` target that submodule.

## Quickstart

```bash
# A — confirm the hook layout (synthetic inputs, no data needed)
python scripts/pair_bias_measurement/explore_model.py \
    --config configs/megafold_1x1.yaml --n 384

# B — capture pair_bias from every pairformer module
python scripts/pair_bias_measurement/capture_tensors.py \
    --config configs/megafold_1x1.yaml \
    --n 384 \
    --output captures/run1.pt

# C — open the notebook and point CAPTURE_PATH at the file you just wrote
jupyter lab scripts/pair_bias_measurement/analyze_patterns.ipynb
```

### Running with a trained model

The default path uses synthetic inputs and unloaded weights, which is enough to
validate the pipeline end-to-end but **will not** produce meaningful pattern
measurements — random weights give noise, not signal. Stage B warns about this
in the capture file's `_meta["weights_initialized"]` flag.

To measure on a trained model, drive a full forward pass through the trainer:

```bash
python scripts/pair_bias_measurement/capture_tensors.py \
    --full-forward \
    --conductor-config configs/megafold_1x1.yaml \
    --trainer-name initial_training \
    --output captures/run_trained.pt
```

That path requires a populated data cache and a checkpoint in the config's
`checkpoint_folder`.

## Strength thresholds (from spec)

| Property            | Metric                                                           | STRONG       | MODERATE      |
|---------------------|------------------------------------------------------------------|--------------|---------------|
| Symmetry            | `‖sym‖² / (‖sym‖² + ‖anti‖²)`                                    | > 0.9        | 0.7 – 0.9     |
| Spatial sparsity    | frac of 64×64 tiles with mean\|B\| < 10% of median                | > 30 %       | 15 – 30 %     |
| Temporal reuse      | median `‖B_{k+1} − B_k‖ / ‖B_k‖`                                 | < 0.1        | 0.1 – 0.3     |
| Cross-head correl.  | mean off-diagonal head–head Pearson correlation                  | > 0.8        | 0.5 – 0.8     |
