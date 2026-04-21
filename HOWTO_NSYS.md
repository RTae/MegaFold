# Nsight Systems HOWTO

This repo already had Nsight Compute support. This file covers tracing the real training path through `train.py` with NVIDIA Nsight Systems.

## What was added

- `train.py` now emits a top-level NVTX range named `train` when `MEGAFOLD_NVTX=1`.
- `megafold/trainer.py` now emits a step-level NVTX range named `train.step_N`, plus nested ranges for dataloader, forward, backward, optimizer, EMA, zero-grad, and scheduler.
- `megafold/model/megafold.py` and `megafold/utils/model_utils.py` now emit finer-grained NVTX ranges for trunk, diffusion, heads, and major loss terms.
- `scripts/profile_nsys_train.sh` launches `train.py` directly for 1-GPU tracing and uses DeepSpeed only for multi-GPU tracing.

## Recommended first trace

Use the smoke config and limit the run to a few optimizer steps so the report stays readable.

```bash
scripts/profile_nsys_train.sh
```

That defaults to:

- config: `configs/megafold_1x1_smoke.yaml`
- GPUs: `1`
- max steps: `3`
- output dir: `nsys_reports/`

## Capture exactly one steady-state step

If you want to skip the noisy first step and capture only a later training step, use `--capture-step`.

```bash
# Capture only training step 1, but allow the job to run long enough to reach it
scripts/profile_nsys_train.sh --capture-step 1 --output steady_state_step1
```

Step indices are zero-based:

- `--capture-step 0` captures the first training step
- `--capture-step 1` captures the second training step
- `--capture-step 2` captures the third training step

When `--capture-step N` is set, the wrapper automatically increases `MEGAFOLD_MAX_STEPS` to at least `N + 1` if needed.

## Common examples

```bash
# Capture only the second training step on the smoke config
scripts/profile_nsys_train.sh --capture-step 1 --output smoke_step1

# 5 optimizer steps on the smoke config
scripts/profile_nsys_train.sh --max-steps 5 --output smoke_5steps

# 10 steps on the standard 1x1 config
scripts/profile_nsys_train.sh \
  --config configs/megafold_1x1.yaml \
  --gpus 1 \
  --max-steps 10 \
  --output megafold_1x1_trace

# 4 steps on the 2-GPU config
scripts/profile_nsys_train.sh \
  --config configs/megafold_1x2.yaml \
  --gpus 2 \
  --max-steps 4 \
  --output megafold_1x2_trace
```

## Manual command

If you want to run `nsys` directly instead of using the wrapper:

```bash
MEGAFOLD_NVTX=1 \
MEGAFOLD_MAX_STEPS=2 \
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=none \
  --cpuctxsw=none \
  --wait=all \
  --capture-range=nvtx \
  --capture-range-end=stop \
  --nvtx-capture=train.step_1 \
  --force-overwrite=true \
  -o nsys_reports/manual_trace \
  deepspeed --master_port 29517 --num_gpus=1 train.py \
  --config configs/megafold_1x1_smoke.yaml \
  --trainer_name initial_training
```

## Important notes

- Keep the YAML field `profile: false` when using `nsys`. That field enables PyTorch profiler, which is a different tool and adds extra overhead.
- `MEGAFOLD_MAX_STEPS` is an environment override for short debug or tracing runs. Set it to `0` to disable the limit.
- Set `MEGAFOLD_AUTOGRAD_NVTX=1` if you want PyTorch autograd to emit extra NVTX markers inside backward for a more detailed kernel timeline.
- By default the wrapper captures the NVTX range named `train`, so Python startup and DeepSpeed launcher setup stay out of the final trace.
- With `--capture-step N`, the wrapper captures only the NVTX range `train.step_N`, which gives you one exact training step instead of the whole run.
- For single-GPU configs like `configs/megafold_1x1_smoke.yaml`, the wrapper now calls `train.py` directly to avoid DeepSpeed launcher indirection preventing NVTX-triggered capture.

## Opening the report

```bash
nsys-ui nsys_reports/<name>.nsys-rep
```

If you only have the CLI available:

```bash
nsys stats nsys_reports/<name>.nsys-rep
```