# Nsight Systems HOWTO

This repo already had Nsight Compute support. This file covers tracing the real training path through `train.py` with NVIDIA Nsight Systems.

## What was added

- `train.py` now emits a top-level NVTX range named `train` when `MEGAFOLD_NVTX=1`.
- `megafold/trainer.py` now emits nested NVTX ranges for dataloader, forward, backward, optimizer, EMA, zero-grad, scheduler, and validation.
- `scripts/profile_nsys_train.sh` launches `nsys` against the normal DeepSpeed training entrypoint.

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

## Common examples

```bash
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
MEGAFOLD_MAX_STEPS=3 \
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=none \
  --cpuctxsw=none \
  --wait=all \
  --capture-range=nvtx \
  --nvtx-capture=train \
  --stop-on-range-end=true \
  --force-overwrite=true \
  -o nsys_reports/manual_trace \
  deepspeed --master_port 29517 --num_gpus=1 train.py \
  --config configs/megafold_1x1_smoke.yaml \
  --trainer_name initial_training
```

## Important notes

- Keep the YAML field `profile: false` when using `nsys`. That field enables PyTorch profiler, which is a different tool and adds extra overhead.
- `MEGAFOLD_MAX_STEPS` is an environment override for short debug or tracing runs. Set it to `0` to disable the limit.
- The wrapper captures only the NVTX range named `train`, so Python startup and DeepSpeed launcher setup stay out of the final trace.

## Opening the report

```bash
nsys-ui nsys_reports/<name>.nsys-rep
```

If you only have the CLI available:

```bash
nsys stats nsys_reports/<name>.nsys-rep
```