# Copyright 2026 UIUC SSAIL
# Part of MegaFold project
#
# Licensed under the MIT License. See LICENSE file in the project root
# for full license text.
#
# If you use this code, please cite:
#
#   La, H., Gupta, A., Morehead, A., Cheng, J., & Zhang, M. (2025).
#   MegaFold: System-Level Optimizations for Accelerating Protein
#   Structure Prediction Models. arXiv:2506.20686
#   https://arxiv.org/abs/2506.20686
#
# BibTeX:
#   @misc{la2025megafoldsystemleveloptimizationsaccelerating,
#       title={MegaFold: System-Level Optimizations for Accelerating
#              Protein Structure Prediction Models},
#       author={Hoa La and Ahan Gupta and Alex Morehead
#               and Jianlin Cheng and Minjia Zhang},
#       year={2025},
#       eprint={2506.20686},
#       archivePrefix={arXiv},
#       primaryClass={q-bio.BM},
#       url={https://arxiv.org/abs/2506.20686},
#   }

from __future__ import annotations

import glob
import os
import pprint
import traceback
from contextlib import contextmanager, nullcontext
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
import torchinfo
from adam_atan2_pytorch.foreach import AdamAtan2
from beartype.typing import Any, Callable, List, Literal, Set
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.lion import DeepSpeedCPULion
from ema_pytorch import EMA
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy, DeepSpeedStrategy
from lightning.pytorch.utilities.memory import garbage_collection_cuda
from lion_pytorch.foreach import Lion
from pydantic import BaseModel
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader as OrigDataLoader
from torch.utils.data import Dataset, Sampler
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm
from wrapt_timeout_decorator import timeout
from torch.utils.data.distributed import DistributedSampler
from megafold.data import mmcif_writing
from megafold.inputs import (
    BatchedAtomInput,
    collate_inputs_to_batched_atom_input,
    compose_calls,
)
from megafold.model.megafold import ComputeConfidenceScore, ComputeModelSelectionScore, Sample
from megafold.tensor_typing import package_available, should_typecheck, typecheck
from megafold.utils.model_utils import at_most_one_of, divisible_by
from megafold.utils.trainer_utils import (
    CycleIterator,
    capture_hparams,
    choose_logger,
    generate_id,
    get_default_supported_precision,
    get_logger_experiment_id,
    parse_devices,
    parse_dtype,
)
from megafold.utils.utils import default, exists, not_exists
from megafold.distributed.parallel_info import setup_2d_parallel_groups, get_parallel_info
import time 
from deepspeed.utils.timer import SynchronizedWallClockTimer
import torch.distributed as dist
from datetime import timedelta
import deepspeed
import random
import numpy as np
from loguru import logger


def nvtx_enabled() -> bool:
    value = os.environ.get("MEGAFOLD_NVTX", "")
    return value.lower() not in {"", "0", "false", "no"}


@contextmanager
def nvtx_range(message: str):
    if not nvtx_enabled() or not torch.cuda.is_available():
        yield
        return

    import torch.cuda.nvtx as nvtx

    nvtx.range_push(message)
    try:
        yield
    finally:
        nvtx.range_pop()


def seed_everything(seed: int):
    """
    Seed all random number generators for reproducibility.
    Replicates Lightning Fabric's seed_everything() behavior.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU


# constants

PHASES = Literal["train", "val", "test"]
FORWARD_MAX_SECONDS_PER_INPUT = 120
SAMPLING_MAX_SECONDS_PER_INPUT = 300


# =================================================================================
# === 2D Parallelism Setup =======================================================
# =================================================================================


def create_2d_dataloader(dataset, batch_size, parallel_info, DataLoader_, is_training=True, **kwargs):
    """
    Create dataloader for 2D parallelism.
    
    Key insight: Since ranks within the same SP group have the same ddp_rank,
    DistributedSampler automatically gives them identical data samples.
    
    Args:
        dataset: The dataset to wrap
        batch_size: Batch size per GPU
        parallel_info: 2D parallel configuration info
        DataLoader_: Custom DataLoader class (with preprocessing logic)
        is_training: Whether this is a training dataloader (affects sampler behavior)
        **kwargs: Additional dataloader arguments
    """
    # Handle sampler logic exactly like original code
    dataloader_kwargs = dict(kwargs)
    
    if 'sampler' in dataloader_kwargs:
        # Custom sampler provided - use it but warn about potential conflicts
        # Note: This case is rare and may need special handling
        print("Warning: Custom sampler provided with 2D parallelism. This may conflict with DistributedSampler.")
        pass
    else:
        # Create DistributedSampler for 2D parallelism
        # Use shuffle from kwargs (True for training, False for val/test)
        shuffle_data = dataloader_kwargs.pop('shuffle', False) if is_training else False
        
        # For SP groups that should see the same data, use SP-based sampling
        # This ensures consistent data ordering across different SP configurations
        if parallel_info['ddp_size'] == 1:
            # Single SP group case: all ranks see all data (no distributed sampling)
            sampler = None  # Use default sequential sampler
            dataloader_kwargs['shuffle'] = shuffle_data  # Restore shuffle for non-distributed case
        else:
            # Multiple SP groups case: use DDP-based sampling
            sampler = DistributedSampler(
                dataset,
                num_replicas=parallel_info['ddp_size'],  # Number of DDP groups
                rank=parallel_info['ddp_rank'],          # DDP rank
                shuffle=shuffle_data
            )
            dataloader_kwargs['sampler'] = sampler
        
        # Remove shuffle from kwargs only if we're using a sampler
        if sampler is not None:
            dataloader_kwargs.pop('shuffle', None)
    
    return DataLoader_(
        dataset,
        batch_size=batch_size,
        **dataloader_kwargs
    )


# helpers


@contextmanager
def to_device_and_back(module: Module, device: torch.device):
    """Move module to device and back after context."""
    orig_device = next(module.parameters()).device
    need_move_device = orig_device != device

    if need_move_device:
        module.to(device)

    yield

    if need_move_device:
        module.to(orig_device)


# dataloader and collation fn


@typecheck
def DataLoader(
    *args,
    atoms_per_window: int | None = None,
    map_input_fn: Callable | None = None,
    transform_to_atom_inputs: bool = True,
    **kwargs,
) -> OrigDataLoader:
    """DataLoader with collation function."""
    collate_fn = partial(
        collate_inputs_to_batched_atom_input,
        atoms_per_window=atoms_per_window,
        transform_to_atom_inputs=transform_to_atom_inputs,
    )

    if exists(map_input_fn):
        collate_fn = partial(collate_fn, map_input_fn=map_input_fn)

    return OrigDataLoader(*args, collate_fn=collate_fn, **kwargs)


# default scheduler used in paper w/ warmup


def default_lambda_lr_fn(
    steps: int,
    warmup_steps: int = 3000,
    decay_every_n_steps: int = 5e4,
    decay_rate: float = 0.95,
    disabled: bool = False,
) -> float:
    """Default lambda function for scheduler."""
    if disabled:
        return 1.0

    # warmup for `warmup_steps` steps

    if steps < warmup_steps:
        return steps / warmup_steps

    # decay `decay_rate` every `decay_every_n_steps` steps

    steps -= warmup_steps
    return decay_rate ** (steps / decay_every_n_steps)


# main class


class Trainer:
    """Section 5.4."""

    @typecheck
    def __init__(
        self,
        model: BaseModel,  # NOTE: must be a `MegaFoldConfig` instance
        *,
        dataset: Dataset,
        num_train_steps: int,
        global_batch_size: int,
        num_valid_steps: int | None = None,
        num_test_steps: int | None = None,
        devices: int | str = "auto",
        num_nodes: int = 1,
        seed: int = 42,
        grad_accum_every: int = 1,
        clear_cuda_cache_every: int = 1,
        confidence_head_interval: int = 10,
        offload_optimizer: bool = False,
        allgather_bucket_size: int = 200_000_000,
        reduce_bucket_size: int = 200_000_000,
        samples_on_cpu: bool = False,
        map_dataset_input_fn: Callable | None = None,
        valid_dataset: Dataset | None = None,
        valid_every: int = 1000,
        test_dataset: Dataset | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        ema_decay: float = 0.999,
        lr: float = 1e-3,
        default_optimizer_kwargs: dict = dict(
            betas=(0.9, 0.95),
            eps=1e-8,
        ),
        clip_grad_norm: float = 10.0,
        default_lambda_lr: Callable = default_lambda_lr_fn,
        train_sampler: Sampler | None = None,
        fabric: Fabric | None = None,
        profile: bool = False,
        profiler_kwargs: dict = dict(),
        logger_name: Literal["wandb", "tensorboard", "csv"] | None = None,
        logger_kwargs: dict = dict(),
        train_log_interval: int = 1,
        accelerator: Literal["cpu", "gpu", "tpu", "mps", "auto"] = "auto",
        strategy: Literal["auto", "ddp", "deepspeed", "deepspeed_2d"] = "deepspeed_2d",
        data_parallel_size: int = 1,
        sequence_parallel_size: int = 1,
        strategy_stage: int = 0,
        checkpoint_prefix: str = "megafold.ckpt.",
        checkpoint_every: int = 25,
        checkpoint_folder: str = "./checkpoints",
        overwrite_checkpoints: bool = False,
        fabric_kwargs: dict = dict(),
        precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
        use_ema: bool = True,
        ema_kwargs: dict = dict(use_foreach=True),
        ema_on_cpu: bool = False,
        use_adam_atan2: bool = False,
        use_lion: bool = False,
        use_torch_compile: bool = False,
        is_fine_tuning: bool = False,
        summarize_model: bool = True,
        num_samples_per_example: int = 5,
        visualize_train_samples_every_n_steps: int = 0,
        visualize_valid_samples_every_n_steps: int = 0,
        visualize_test_samples_every_n_steps: int = 0,
        watch_model: Literal["gradients", "parameters", "all"] | None = None,
        watch_model_freq: int = 1,
        dl_kwargs: dict = dict(),
    ):
        super().__init__()

        # precision

        self.precision_str = precision or get_default_supported_precision(training=True) # bf16-mixed
        self.dtype = parse_dtype(self.precision_str) # torch.bfloat16

        # Initialize 2D parallelism flag
        self.use_native_deepspeed_2d = False

        # strategy

        devices = parse_devices(devices)
        self.devices = devices

        is_single_gpu_native_2d = (
            strategy == "deepspeed_2d"
            and num_nodes == 1
            and data_parallel_size == 1
            and sequence_parallel_size == 1
        )

        if is_single_gpu_native_2d:
            logger.warning(
                "Single-GPU run detected; downgrading strategy from deepspeed_2d to auto to avoid unnecessary distributed rendezvous."
            )
            strategy = "auto"

        if strategy == "ddp":
            # if necessary, address potential DDP activation checkpointing issue: https://discuss.pytorch.org/t/ddp-and-gradient-checkpointing/132244
            strategy = DDPStrategy(find_unused_parameters=False, static_graph=False)
        elif strategy == "deepspeed":
            ds_config = DeepSpeedStrategy(
                zero_optimization=False,
                stage=strategy_stage,
                offload_optimizer=offload_optimizer,
                allgather_bucket_size=allgather_bucket_size,
                reduce_bucket_size=reduce_bucket_size,
            ).config

            # override certain default config values
            ds_config["gradient_clipping"] = clip_grad_norm
            ds_config["train_micro_batch_size_per_gpu"] = 1
            ds_config["gradient_accumulation_steps"] = grad_accum_every

            strategy = DeepSpeedStrategy(config=ds_config)
        elif strategy == "deepspeed_2d":
            # Use native DeepSpeed with 2D parallelism (no Fabric)
            self.use_native_deepspeed_2d = True
        elif strategy != "auto":
            raise ValueError(f"Unknown strategy: {strategy}")

        self.using_deepspeed_strategy = isinstance(strategy, DeepSpeedStrategy)

        # logger

        loggers = None

        if exists(logger_name):
            loggers = [choose_logger(logger_name, **logger_kwargs)]

        self.train_log_interval = train_log_interval

        # store parallelism settings for all strategies
        self.global_batch_size = global_batch_size
        self.data_parallel_size = data_parallel_size
        self.sequence_parallel_size = sequence_parallel_size

        # initialize DS/Fabric

        if self.use_native_deepspeed_2d:
            # Skip Fabric setup for native DeepSpeed 2D
            self.fabric = None

            # Set device early to avoid NCCL warnings
            # For multi-node: LOCAL_RANK should be 0-7 on each node, not global rank
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            # Ensure local_rank is within valid GPU range (0-7 per node)
            num_gpus_per_node = torch.cuda.device_count()
            local_rank = local_rank % num_gpus_per_node
            torch.cuda.set_device(local_rank)
            
            # Ensure all new groups created (by us or libraries) inherit a long default timeout
            # Initialize distributed backend with extended timeout
            if not dist.is_initialized():
                deepspeed.init_distributed(timeout=timedelta(seconds=120000))

        else:
            if not_exists(fabric):
                fabric = Fabric(
                    accelerator=accelerator,
                    devices=devices,
                    num_nodes=num_nodes,
                    strategy=strategy,
                    # NOTE: we use 32-bit precision by default to avoid weight casting issues with DeepSpeed
                    precision=precision or "32-true", # "32-true"
                    loggers=loggers,
                    **fabric_kwargs,
                )

            self.fabric = fabric
            self.fabric.launch()

        # dataset arguments

        dataset_ = dataset.datasets[0] if isinstance(dataset, ConcatDataset) else dataset
        cropping_config = getattr(dataset_, "cropping_config", {})
        self.crop_size = cropping_config.get("n_res", int(1e6))

        # hyperparameters

        hparams = capture_hparams()

        if self.use_native_deepspeed_2d:
            if dist.get_rank() == 0:
                print(pprint.pformat(hparams))
                # TODO: Add hyperparameter logging for 2D mode if needed
        else:
            self.fabric.print(pprint.pformat(hparams))
            if logger_name in ("tensorboard", "wandb"):
                self.fabric.logger.log_hyperparams(hparams)

        # checkpointing logic

        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_every = checkpoint_every
        self.overwrite_checkpoints = overwrite_checkpoints
        self.checkpoint_folder = Path(checkpoint_folder)

        self.checkpoint_folder.mkdir(exist_ok=True, parents=True)
        assert (
            self.checkpoint_folder.is_dir()
        ), f"Checkpoint folder {self.checkpoint_folder} does not exist."

        # random seed

        latest_step = self.get_latest_step_from_checkpoint_folder()

        # NOTE: we use same seed for every process to init model the same way;
        # we also add the latest step to the seed to ensure that the dataloaders
        # are initialized uniquely if we are resuming training from a checkpoint,
        # e.g., since the PDBDataset is a map-style dataset that internally samples
        # PDB IDs in an iterable (state-less) manner using the WeightedPDBSampler;
        # this is designed as such to ensure that map-style distillation datasets
        # are directly compatible with the PDBDataset via simple concatenation.
        self.print(f"Seeding everything with seed {seed + latest_step}.")
        if self.use_native_deepspeed_2d:
            seed_everything(seed + latest_step)
        else:
            self.fabric.seed_everything(seed + latest_step)

        # PAE-specific loss adjustment

        if latest_step < 5000:
            # NOTE: we do this to prevent the PAE weights
            # (which importantly are used for sample ranking)
            # from getting stuck in local minima early in training
            # when the model is poor at denoising structures
            model.train_pae = False

        # efficient model instantiation

        if self.use_native_deepspeed_2d:
            # Initialize model on CPU for efficient DeepSpeed loading
            model = model.create_instance()
        else:
            with self.fabric.init_module():
                model = model.create_instance()  # NOTE: parameters are placed on the meta-device

        # exponential moving average (EMA)

        self.ema_model = None
        self.has_ema = use_ema

        if self.has_ema:
            self.ema_model = EMA(
                model,
                beta=ema_decay,
                update_every=checkpoint_every,
                inv_gamma=1.0,
                power=1.0,
                include_online_model=False,
                allow_different_devices=True,
                coerce_dtype=True,
                **ema_kwargs,
            )

            self.ema_device = "cpu" if ema_on_cpu else self.device
            self.ema_model.to(self.ema_device)

        # maybe torch compile

        if use_torch_compile:
            assert (
                not should_typecheck
            ), "Does not work well with jaxtyping + beartype, please invoke your training script with the environment flag `TYPECHECK=False` - ex. `TYPECHECK=False python train_megafold.py`"
            model = torch.compile(model)

        # reseed everything (since for some reason model initialization resets `torch.initial_seed()`)

        if self.use_native_deepspeed_2d:
            seed_everything(seed + latest_step)
        else:
            self.fabric.seed_everything(seed + latest_step)

        # if map dataset function given, curry into DataLoader

        DataLoader_ = partial(DataLoader, atoms_per_window=model.atoms_per_window)

        if exists(map_dataset_input_fn):
            DataLoader_ = partial(DataLoader_, map_input_fn=map_dataset_input_fn)

        # maybe distillation dataset training

        train_dl_kwargs = dict()

        if exists(train_sampler):
            train_dl_kwargs.update(sampler=train_sampler)
            print("Warning: Custom sampler provided with 2D parallelism. This may conflict with DistributedSampler.") 
        else:
            # For debugging/comparison purposes, you can set shuffle=False here
            # Set shuffle=False for reproducible comparison between 1x1GPU and 1x2GPU runs
            train_dl_kwargs.update(shuffle=False, drop_last=True)

        # training, validation, and test steps

        self.num_train_steps = num_train_steps
        self.num_valid_steps = num_valid_steps
        self.num_test_steps = num_test_steps

        # optimizer

        if not_exists(optimizer):
            optimizer_klass = (
                partial(DeepSpeedCPUAdam, adamw_mode=False)
                if self.using_deepspeed_strategy and offload_optimizer
                else Adam
            )

            assert at_most_one_of(use_adam_atan2, use_lion)

            if use_adam_atan2:
                default_optimizer_kwargs.pop("eps", None)
                optimizer_klass = AdamAtan2
                assert not (self.using_deepspeed_strategy and offload_optimizer), (
                    "AdamAtan2 is not supported with DeepSpeed optimizer offloading. "
                    "Please set `use_adam_atan2=False` or `offload_optimizer=False`."
                )
            elif use_lion:
                default_optimizer_kwargs.pop("eps", None)
                optimizer_klass = (
                    DeepSpeedCPULion
                    if self.using_deepspeed_strategy and offload_optimizer
                    else Lion
                )

            # Remove 'type' key if present (comes from config but not accepted by optimizer)
            default_optimizer_kwargs.pop("type", None)
            optimizer = optimizer_klass(model.parameters(), lr=lr, **default_optimizer_kwargs)

        elif (
            self.using_deepspeed_strategy
            and offload_optimizer
            and not (
                isinstance(optimizer, DeepSpeedCPUAdam) or isinstance(optimizer, DeepSpeedCPULion)
            )
        ):
            raise ValueError(
                "When using DeepSpeed optimizer offloading, the optimizer must be an instance of DeepSpeedCPUAdam or DeepSpeedCPULion."
            )

        # scheduler

        if not_exists(scheduler):
            scheduler = LambdaLR(optimizer, lr_lambda=default_lambda_lr)

        # setup for model and optimizer

        if self.use_native_deepspeed_2d:
            # DeepSpeed configuration - use external optimizer and scheduler
            # NOTE: Keep DeepSpeed in 32-bit like Fabric, let model handle mixed precision via autocast
            # For 2D parallelism: We need to satisfy DeepSpeed's assertion while maintaining correct training.  DeepSpeed assertion: train_batch_size == micro_batch × grad_accum × world_sizeIn 2D parallelism: Only data_parallel_size GPUs process different data. Solution: Set train_batch_size to what DeepSpeed expects based on world_size
            world_size = dist.get_world_size()
            deepspeed_train_batch_size = self.batch_size() * grad_accum_every * world_size
            
            ds_config = {
                "train_micro_batch_size_per_gpu": self.batch_size(),
                "train_batch_size": deepspeed_train_batch_size,
                "gradient_accumulation_steps": grad_accum_every,
                "zero_optimization": {"stage": strategy_stage},
                "gradient_clipping": clip_grad_norm,
                "bf16": {"enabled": False},  # Let model handle mixed precision like Fabric
            }
            
            # Initialize DeepSpeed engine with user's optimizer and scheduler
            model_engine, ds_optimizer, _, ds_scheduler = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                optimizer=optimizer,
                lr_scheduler=scheduler,
                config=ds_config
            )
            
            self.model, self.model_optimizer, self.scheduler = model_engine, ds_optimizer, ds_scheduler
            self.model_engine = model_engine
            
            # After DeepSpeed initialization, setup 2D parallelism groups.
            # This avoids communicator creation races during DS engine bootstrapping.
            self.parallel_info = setup_2d_parallel_groups(
                self.data_parallel_size,
                self.sequence_parallel_size,
                timeout_seconds=120000,
            )
            
            # Print 2D parallelism info
            self.print("DeepSpeed 2D Parallelism initialized:")
            self.print(f"  - Data parallel size: {self.data_parallel_size}")
            self.print(f"  - Sequence parallel size: {self.sequence_parallel_size}")
            self.print(f"  - DDP groups: {self.parallel_info['ddp_groups']}")
            self.print(f"  - SP groups: {self.parallel_info['sp_groups']}")
            self.print(f"  - DeepSpeed precision: 32-bit (matches Fabric)")
            self.print(f"  - Model dtype: {self.dtype} (via autocast)")
            self.print(f"  - Mixed precision: Model-managed via torch.autocast (same as Fabric)")
            
        else:
            # Original Fabric setup
            model, optimizer = self.fabric.setup(model, optimizer)
            self.model, self.model_optimizer, self.scheduler = model, optimizer, scheduler
            self.model_engine = None

        
        # dataloaders
        self.num_nodes = num_nodes
        self.valid_every = valid_every
        self.needs_valid = exists(valid_dataset)
        self.needs_test = exists(test_dataset)

        if self.use_native_deepspeed_2d:
            # Create dataloaders with 2D parallelism using original logic            
            # Training dataloader
            self.dataloader = create_2d_dataloader(
                dataset, 
                batch_size=self.batch_size(),
                parallel_info=self.parallel_info,
                DataLoader_=DataLoader_,
                is_training=True,
                **dl_kwargs, 
                **train_dl_kwargs
            )
            dataloaders = [self.dataloader]
            
            # Validation dataloader (no shuffle, no drop_last, no custom sampler)
            if self.needs_valid:
                self.valid_dataloader = create_2d_dataloader(
                    valid_dataset,
                    batch_size=self.batch_size(),
                    parallel_info=self.parallel_info,
                    DataLoader_=DataLoader_,
                    is_training=False,
                    **dl_kwargs
                )
                dataloaders.append(self.valid_dataloader)
            
            # Test dataloader (no shuffle, no drop_last, no custom sampler)
            if self.needs_test:
                self.test_dataloader = create_2d_dataloader(
                    test_dataset,
                    batch_size=self.batch_size(),
                    parallel_info=self.parallel_info,
                    DataLoader_=DataLoader_,
                    is_training=False,
                    **dl_kwargs
                )
                dataloaders.append(self.test_dataloader)
        else:
            # Original dataloader creation
            self.dataloader = DataLoader_(
                dataset, batch_size=self.batch_size(), **dl_kwargs, **train_dl_kwargs
            )
            dataloaders = [self.dataloader]

            # validation dataloader on the EMA model
            if self.needs_valid:
                self.valid_dataloader = DataLoader_(
                    valid_dataset, batch_size=self.batch_size(), **dl_kwargs
                )
                dataloaders.append(self.valid_dataloader)

            # testing dataloader on EMA model
            if self.needs_test:
                self.test_dataloader = DataLoader_(
                    test_dataset, batch_size=self.batch_size(), **dl_kwargs
                )
                dataloaders.append(self.test_dataloader)


        # setup dataloaders with Fabric (only for non-2D case)
        if not self.use_native_deepspeed_2d:
            setup_result = self.fabric.setup_dataloaders(*dataloaders)
            if len(dataloaders) == 1:
                dataloaders = [setup_result]
            else:
                dataloaders = list(setup_result)

            # Reassign dataloaders after Fabric setup
            self.dataloader = dataloaders[0]

            if self.needs_valid:
                self.valid_dataloader = dataloaders[1]

            if self.needs_test:
                self.test_dataloader = dataloaders[-1]

        if self.is_main and summarize_model:
            torchinfo.summary(model)
        # maximum norm gradient clipping

        self.clip_grad_norm = clip_grad_norm

        # gradient accumulation

        self.grad_accum_every = grad_accum_every

        # CUDA memory clearing

        self.clear_cuda_cache_every = clear_cuda_cache_every

        # steps

        self.steps = 0

        # confidence head interval

        self.confidence_head_interval = confidence_head_interval

        # path caching for the last loaded model, if any

        if self.use_native_deepspeed_2d:
            self.train_id = None  # Will be generated later
        else:
            self.train_id = get_logger_experiment_id(self.fabric.loggers)

        self.last_loaded_train_id = None
        self.model_loaded_from_path: Path | None = None

        # model selection

        self.is_fine_tuning = is_fine_tuning
        self.num_samples_per_example = num_samples_per_example

        self.compute_model_selection_score = ComputeModelSelectionScore(
            is_fine_tuning=is_fine_tuning
        )

        self.best_model_selection_step = -1
        self.best_model_selection_score = -float("inf")
        self.best_top_ranked_lddt = -float("inf")

        self.samples_on_cpu = samples_on_cpu

        # visualization parameters

        self.visualize_train_samples_every_n_steps = visualize_train_samples_every_n_steps
        self.visualize_valid_samples_every_n_steps = visualize_valid_samples_every_n_steps
        self.visualize_test_samples_every_n_steps = visualize_test_samples_every_n_steps

        if logger_name == "wandb" and exists(watch_model) and not self.use_native_deepspeed_2d:
            assert package_available(
                "wandb"
            ), "Please install and use the `wandb` package to log model gradients/parameters."

            self.fabric.logger.experiment.watch(model, log=watch_model, log_freq=watch_model_freq)

        # profiler

        self.profile = profile

        if self.profile:
            assert "log_dir" in profiler_kwargs, "Please provide a `log_dir` for the profiler."

            self.profiler_log_dir = profiler_kwargs["log_dir"]

            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profiler_log_dir),
                profile_memory=True,
                with_stack=True,
            )

    @property
    def device(self) -> torch.device:
        """Get device."""
        if self.use_native_deepspeed_2d:
            # Use local rank, not global rank for device assignment
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            num_gpus_per_node = torch.cuda.device_count()
            local_rank = local_rank % num_gpus_per_node
            return torch.device(f"cuda:{local_rank}")
        return self.fabric.device

    @property
    def is_main(self) -> bool:
        """Check if main rank."""
        if self.use_native_deepspeed_2d:
            return dist.get_rank() == 0
        return self.fabric.global_rank == 0

    def generate_train_id(self):
        """Generate a unique training id."""
        if exists(self.train_id):
            return

        self.train_id = generate_id()

    @property
    def train_id_with_prev(self) -> str:
        """Get train id with previous train id."""
        if not_exists(self.last_loaded_train_id):
            return self.train_id

        ckpt_num = str(self.model_loaded_from_path).split(".")[-2]

        return f"{self.last_loaded_train_id}.{ckpt_num}-{self.train_id}"

    # saving and loading

    def save_checkpoint(self):
        """Save checkpoint."""
        assert exists(self.train_id_with_prev), "Train ID not generated."

        # formulate checkpoint path and save

        os.makedirs(self.checkpoint_folder, exist_ok=True)

        checkpoint_path = (
            self.checkpoint_folder
            / f"({self.train_id_with_prev})_{self.checkpoint_prefix}{self.steps}.pt"
        )

        self.save(checkpoint_path, overwrite=self.overwrite_checkpoints)

    def save(self, path: str | Path, overwrite=False, prefix: str | None = None):
        """Save model and optimizer states."""
        self.wait()

        if isinstance(path, str):
            path = Path(path)

        assert not path.is_dir() and (overwrite or not path.exists()), (
            f"Checkpoint file {path} already exists. "
            "Please set `overwrite=True` to overwrite the file."
        )

        path.parent.mkdir(exist_ok=True, parents=True)

        package = dict(
            version=self.model.state_dict_with_init_args["version"],
            init_args_and_kwargs=self.model.state_dict_with_init_args["init_args_and_kwargs"],
            model=self.model,
            ema_model=self.ema_model.state_dict() if self.has_ema else None,
            model_optimizer=self.model_optimizer,
            scheduler=self.scheduler,
            steps=self.steps,
            id=self.train_id,
            best_model_selection_step=self.best_model_selection_step,
            best_model_selection_score=self.best_model_selection_score,
            best_top_ranked_lddt=self.best_top_ranked_lddt,
        )

        self.print(f"Saving checkpoint to {str(path)}")
        if self.use_native_deepspeed_2d:
            # Use DeepSpeed's native checkpointing
            self.model_engine.save_checkpoint(str(path.parent), tag=path.stem)
        else:
            self.fabric.save(path, package)

        self.wait()

    def get_latest_step_from_checkpoint_folder(
        self, prefix=None, excluded_prefixes: Set[str] | None = {"collated_"}
    ) -> int:
        """Get latest step from checkpoint folder."""
        path = self.checkpoint_folder

        if isinstance(path, str):
            path = Path(path)

        assert path.is_dir(), f"Checkpoint folder {path} does not exist."

        prefix = default(prefix, self.checkpoint_prefix)

        model_paths = [
            p
            for p in path.glob(f"**/*_{prefix}*.pt")
            if not any(p.name.startswith(e) for e in excluded_prefixes)
        ]

        if not model_paths:
            self.print(f"WARNING: No files found in directory {path}. Skipping seed loading.")
            return 0

        model_paths = sorted(model_paths, key=lambda p: int(str(p).split(".")[-2]))
        latest_step = int(str(model_paths[-1]).split(".")[-2])

        return latest_step

    def load_from_checkpoint_folder(self, **kwargs):
        """Load from checkpoint folder."""
        self.load(path=self.checkpoint_folder, **kwargs)

    def load(
        self,
        path: str | Path,
        strict=True,
        prefix=None,
        only_model=False,
        reset_steps=False,
        load_best_model=False,
        excluded_prefixes: Set[str] | None = {"collated_"},
    ):
        """Load model and optimizer states."""
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            self.print(
                f"WARNING: {str(path)} cannot be found. Skipping checkpoint loading from this folder."
            )
            return

        # if the path is a directory, then automatically load latest checkpoint

        if path.is_dir():
            prefix = default(prefix, self.checkpoint_prefix)

            model_paths = [
                p
                for p in path.glob(f"**/*_{prefix}*.pt")
                if not any(p.name.startswith(e) for e in excluded_prefixes)
            ]

            if not model_paths:
                self.print(
                    f"WARNING: No files found in directory {path}. Skipping checkpoint loading."
                )
                return

            if load_best_model:
                paths = [
                    p
                    for p in model_paths
                    if int(str(p).split(".")[-2]) == self.best_model_selection_step
                ]
                assert (
                    paths
                ), f"No best model at step {self.best_model_selection_step} with model selection score {self.best_model_selection_score:.6f} found at {path}."

                path = paths[0]
                self.print(
                    f"Best model found at step {self.best_model_selection_step} with model selection score {self.best_model_selection_score:.6f}."
                )
            else:
                model_paths = sorted(model_paths, key=lambda p: int(str(p).split(".")[-2]))
                path = model_paths[-1]

        # load model from path

        package = dict(
            version=self.model.state_dict_with_init_args["version"],
            init_args_and_kwargs=self.model.state_dict_with_init_args["init_args_and_kwargs"],
            model=self.model,
            ema_model=self.ema_model.state_dict() if self.has_ema else None,
            model_optimizer=self.model_optimizer,
            scheduler=self.scheduler,
            steps=self.steps,
            id=self.train_id,
            best_model_selection_step=self.best_model_selection_step,
            best_model_selection_score=self.best_model_selection_score,
            best_top_ranked_lddt=self.best_top_ranked_lddt,
        )

        self.print(f"Loading checkpoint from {path}")
        if self.use_native_deepspeed_2d:
            # Use DeepSpeed's native checkpoint loading
            _, _ = self.model_engine.load_checkpoint(str(path.parent), tag=path.stem)
            # Load additional package data manually for 2D mode
            # TODO: Implement proper package loading for 2D mode
        else:
            self.fabric.load(path, package, strict=strict)

        # load EMA model weights

        if self.has_ema:
            # NOTE: `strict=False` to allow loading of EMA model weights even if PLM/NLM model weights are not present
            self.ema_model.load_state_dict(package["ema_model"], strict=False)

        # ensure that the model is loaded from the same version

        self.model._version = package["version"]
        self.model._args_and_kwargs = package["init_args_and_kwargs"]

        package_version = package["version"]
        current_version = version("megafold")

        if package_version != current_version:
            self.print(
                f"WARNING: Loading a saved model from version {package_version}, but you are on version {current_version}."
            )

        # for eventually saving entire training history in filename

        self.model_loaded_from_path = path
        self.last_loaded_train_id = package["id"]

        if only_model:
            return

        # install remaining metadata

        if reset_steps:
            self.steps = 0
        else:
            self.steps = package.get("steps", 0)

        self.best_model_selection_step = package.get("best_model_selection_step", -1)
        self.best_model_selection_score = package.get("best_model_selection_score", -float("inf"))
        self.best_top_ranked_lddt = package.get("best_top_ranked_lddt", -float("inf"))

    # shortcut methods

    def wait(self):
        """Wait for all ranks to sync."""
        if self.use_native_deepspeed_2d:
            dist.barrier()
        else:
            self.fabric.barrier()

    def print(self, *args, **kwargs):
        """Print to stdout."""
        if self.use_native_deepspeed_2d:
            if dist.get_rank() == 0:
                print(*args, **kwargs)
        else:
            self.fabric.print(*args, **kwargs)

    def log(self, name: str, value: Any):
        """Log dictionary."""
        if self.use_native_deepspeed_2d:
            # Simple logging for 2D mode (could be enhanced)
            if dist.get_rank() == 0:
                print(f"LOG {name}: {value}")
        else:
            self.fabric.log(name, value, step=self.steps)

    def log_dict(self, **log_data):
        """Log dictionary."""
        if self.use_native_deepspeed_2d:
            # Simple logging for 2D mode (could be enhanced)
            if dist.get_rank() == 0:
                for k, v in log_data.items():
                    print(f"LOG {k}: {v}")
        else:
            self.fabric.log_dict(log_data, step=self.steps)

    def batch_size(self) -> int:
        """Number of samples load to each GPU (between optimizer steps per data-parallel rank)."""
        batch_size = self.global_batch_size // self.data_parallel_size
        assert batch_size > 0, "Effective batch size must be greater than 0."
        return batch_size

    # MSA caching

    def cache_msas(self, split: Literal["train", "val", "test"]):
        """Cache MSAs for a given dataset split."""
        dataloader = self.dataloader

        if split == "val":
            dataloader = self.valid_dataloader
            assert self.needs_valid, "Validation dataloader not available."
        elif split == "test":
            dataloader = self.test_dataloader
            assert self.needs_test, "Test dataloader not available."

        for _ in tqdm(dataloader, desc=f"Caching MSAs for {split} split..."):
            pass

        self.print(f"Finished caching MSAs for {split} split.")

    # input caching

    def cache_inputs(self, split: Literal["train", "val", "test"]):
        """Cache input features for a given dataset split."""
        dataloader = self.dataloader

        if split == "val":
            dataloader = self.valid_dataloader
            assert self.needs_valid, "Validation dataloader not available."
        elif split == "test":
            dataloader = self.test_dataloader
            assert self.needs_test, "Test dataloader not available."

        for _ in tqdm(dataloader, desc=f"Caching inputs for {split} split..."):
            pass

        self.print(f"Finished caching inputs for {split} split.")

    # sampling and visualization

    @typecheck
    @torch.inference_mode()
    def visualize(
        self,
        sampled_atom_pos: Float["b m 3"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        filepaths: List[str],
        batch_idx: int,
        phase: PHASES,
        sample_idx: int = 1,
        filename_suffixes: List[str] | None = None,
        b_factors: Float["b m"] | None = None,  # type: ignore
        allow_atom_mask_mismatch: bool = False,
        verbose: bool = False,
    ) -> None:
        """Visualize samples pre-generated for the examples in a batch.

        :param sampled_atom_pos: The sampled atom positions for the batch.
        :param atom_mask: The atom mask for the batch.
        :param filepaths: The filepaths of the input examples.
        :param batch_idx: The index of the current batch.
        :param phase: The phase of the current step.
        :param sample_idx: The index of the sample to visualize.
        :param filename_suffixes: The suffixes to append to the filenames.
        :param b_factors: The B-factors or equivalent mmCIF field values to list for each atom.
        :param allow_atom_mask_mismatch: Whether to allow the atom mask to mismatch the atom
            positions.
        :param verbose: Whether to print verbose output.
        """
        if verbose:
            self.print(f"Visualizing {phase} samples...")

        samples_output_dir = os.path.join(self.checkpoint_folder, f"{phase}_samples")
        os.makedirs(samples_output_dir, exist_ok=True)

        batch_size = len(atom_mask)

        for b in range(batch_size):
            input_filepath = filepaths[b]
            file_id = os.path.splitext(os.path.basename(input_filepath))[0]
            filename_suffix = filename_suffixes[b] if exists(filename_suffixes) else ""

            output_filepath = os.path.join(
                samples_output_dir,
                os.path.basename(input_filepath).replace(
                    ".cif",
                    f"-sampled-step-{self.steps}-batch-{batch_idx}-example-{b}-sample-{sample_idx}{filename_suffix}.cif",
                ),
            )

            example_atom_mask = atom_mask[b]
            sampled_atom_positions = sampled_atom_pos[b][example_atom_mask].float().cpu().numpy()
            example_b_factors = (
                b_factors[b][example_atom_mask].float().cpu().numpy()
                if exists(b_factors)
                else None
            )

            mmcif_writing.write_mmcif_from_filepath_and_id(
                input_filepath=input_filepath,
                output_filepath=output_filepath,
                file_id=file_id,
                gapless_poly_seq=True,
                insert_orig_atom_names=True,
                insert_megafold_mmcif_metadata=True,
                sampled_atom_positions=sampled_atom_positions,
                b_factors=example_b_factors,
                allow_atom_mask_mismatch=allow_atom_mask_mismatch,
            )

    @typecheck
    @torch.no_grad()
    def sample_and_visualize(
        self,
        model: Module,
        batch: BatchedAtomInput,
        batch_idx: int,
        phase: PHASES,
        sample_idx: int = 1,
        filename_suffixes: List[str] | None = None,
        allow_atom_mask_mismatch: bool = False,
        verbose: bool = False,
    ) -> None:
        """Visualize samples generated for the examples in the input batch.

        :param model: The model to use for sampling.
        :param batch: A batch of `AtomInput` data.
        :param batch_idx: The index of the current batch.
        :param phase: The phase of the current step.
        :param sample_idx: The index of the sample to visualize.
        :param filename_suffixes: The suffixes to append to the filenames.
        :param allow_atom_mask_mismatch: Whether to allow the atom mask to mismatch the atom positions.
        :param verbose: Whether to print verbose output.
        """
        if verbose:
            self.print(f"Sampling and visualizing {phase} samples...")

        batch_sampled_atom_pos = timeout(
            dec_timeout=SAMPLING_MAX_SECONDS_PER_INPUT,
            use_signals=True,
            timeout_exception=BaseException,
        )(model.__call__)(
            **batch.dict(),
            dtype=self.dtype,
            return_loss=False,
            num_sample_steps=200,
            num_recycling_steps=4,
            verbose=verbose,
        )

        samples_output_dir = os.path.join(self.checkpoint_folder, f"{phase}_samples")
        os.makedirs(samples_output_dir, exist_ok=True)

        for example_idx, sampled_atom_pos in enumerate(batch_sampled_atom_pos):
            input_filepath = batch.filepath[example_idx]
            file_id = os.path.splitext(os.path.basename(input_filepath))[0]
            filename_suffix = filename_suffixes[example_idx] if exists(filename_suffixes) else ""

            output_filepath = os.path.join(
                samples_output_dir,
                os.path.basename(input_filepath).replace(
                    ".cif",
                    f"-sampled-step-{self.steps}-batch-{batch_idx}-example-{example_idx}-sample-{sample_idx}{filename_suffix}.cif",
                ),
            )

            atom_mask = ~batch.missing_atom_mask[example_idx]
            sampled_atom_positions = sampled_atom_pos[atom_mask].cpu().numpy()

            mmcif_writing.write_mmcif_from_filepath_and_id(
                input_filepath=input_filepath,
                output_filepath=output_filepath,
                file_id=file_id,
                gapless_poly_seq=True,
                insert_orig_atom_names=True,
                insert_megafold_mmcif_metadata=True,
                sampled_atom_positions=sampled_atom_positions,
                allow_atom_mask_mismatch=allow_atom_mask_mismatch,
            )

    # main train forwards

    def __call__(self, verbose: Literal["", "standard", "extra"] = "extra"):
        """Train model."""
        self.generate_train_id()

        # cycle through dataloader

        dl = CycleIterator(self.dataloader)

        # set up metric accumulation

        self.wait()

        # maybe start profiling

        if self.profile:
            self.print("Starting profiler...")
            self.profiler.start()

        # prepare model selection buffers on the correct device

        samples_device = "cpu" if self.samples_on_cpu else self.device

        self.compute_model_selection_score.dist_breaks = (
            self.compute_model_selection_score.dist_breaks.to(samples_device)
        )
        self.compute_model_selection_score.lddt_thresholds = (
            self.compute_model_selection_score.lddt_thresholds.to(samples_device)
        )

        self.compute_model_selection_score.compute_confidence_score.pae_breaks = (
            self.compute_model_selection_score.compute_confidence_score.pae_breaks.to(
                samples_device
            )
        )
        self.compute_model_selection_score.compute_confidence_score.pde_breaks = (
            self.compute_model_selection_score.compute_confidence_score.pde_breaks.to(
                samples_device
            )
        )

        # prepare optimizer gradient clearing procedure

        zero_grad = (
            self.model_optimizer.zero_grad
            if hasattr(self.model_optimizer, "zero_grad")
            else (
                compose_calls(
                    self.model_optimizer.clear_lp_grads, self.model_optimizer.clear_hp_grads
                )
            )
        )

        # while less than required number of training steps

        grad_accum_iter = 0
        prevTime = None
        timeSoFar = [] 
        lossSoFar = [] 
        debug_max_steps = int(os.environ.get("MEGAFOLD_MAX_STEPS", "0"))

        while self.steps < self.num_train_steps:
            if debug_max_steps and self.steps >= debug_max_steps:
                self.print(
                    f"Stopping early at step {self.steps} because MEGAFOLD_MAX_STEPS={debug_max_steps}."
                )
                break
            self.model.train()

            grad_accum_iter += 1
            is_accumulating = grad_accum_iter < self.grad_accum_every

            # fetch training batch

            if verbose:
                self.print(
                    f"Step {self.steps}, Accum {grad_accum_iter} | Fetching training batch..."
                )

            # track time 
            if prevTime is not None:
                diff = time.time() - prevTime
                self.print(f"Time taken for training step {self.steps}: {diff}")
                timeSoFar.append(diff)
                self.print(f"Time over the steps: {timeSoFar}")
                self.print(f"Loss over the steps: {lossSoFar}")
                self.print("Memory usage: " + SynchronizedWallClockTimer.memory_usage())
            prevTime = time.time()

            with nvtx_range(f"train.step_{self.steps}.dataloader"):
                train_batch = next(dl)
            
            # Move batch to device for distributed training
            if self.use_native_deepspeed_2d or self.using_deepspeed_strategy:
                # For DeepSpeed, need to explicitly move data to GPU
                train_batch = train_batch.to(self.device)

            input = train_batch.dict()

            current_rank = (
                dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            )

            # Get filepath info (it's a list since batch can contain multiple files)
            filepaths = input.get("filepath", ["Unknown"])
            filepath_str = filepaths[0] if filepaths and filepaths[0] is not None else "Unknown"

            print(f"\nRank {current_rank} | Filepath: {filepath_str} | Sequence length: {input['is_molecule_types'].shape[1]} | MSA length: {input['msa'].shape[1]}")

            # maybe profile

            if self.profile:
                self.profiler.step()
                if self.steps >= 1 + 1 + 3:
                    break

            # forward pass

            # Handle gradient synchronization
            sync_context = (
                nullcontext() if self.use_native_deepspeed_2d 
                else self.fabric.no_backward_sync(
                    self.model, enabled=is_accumulating and not self.using_deepspeed_strategy
                )
            )
            
            with sync_context:
                loss_breakdown = None

                try:
                    if verbose == "extra":
                        self.print(f"Step {self.steps}, Accum {grad_accum_iter} | Forward pass...")

                    with nvtx_range(f"train.step_{self.steps}.forward"):
                        loss, loss_breakdown = timeout(
                            dec_timeout=FORWARD_MAX_SECONDS_PER_INPUT if self.steps > 0 else 12000, # first step can be slow
                            use_signals=True,
                            timeout_exception=BaseException,
                        )(self.model.__call__)(
                            **train_batch.dict(),
                            dtype=self.dtype,
                            return_loss_breakdown=True,
                            call_confidence_head=self.steps % self.confidence_head_interval == 0,
                            # verbose=verbose == "extra",
                        )

                        print(f"Rank {current_rank} | Memory after forward pass: {SynchronizedWallClockTimer.memory_usage()}")

                        # Pre-backward high memory guard: abort this step when cached memory exceeds 70% of HBM
                        current_device = (
                            self.device.index
                            if isinstance(self.device, torch.device)
                            else torch.cuda.current_device()
                        )
                        total_hbm_bytes = torch.cuda.get_device_properties(current_device).total_memory
                        threshold_bytes = int(0.7 * total_hbm_bytes)
                        # memory_cached(), not memory_allocated(), since fairer for pytorch allocator
                        cached_bytes = torch.cuda.memory_reserved(current_device)

                        if cached_bytes >= threshold_bytes:
                            raise RuntimeError(
                                f"PreBackwardHighMemory guard: cached={cached_bytes} bytes >= 70% HBM ({threshold_bytes} bytes)"
                            )

                except BaseException as e:
                    self.print(
                        f"Step {self.steps}, Accum {grad_accum_iter} | Skipping training batch due to forward base exception: {e}, {traceback.format_exc()}"
                    )
                    loss = torch.tensor([torch.nan], device=self.device)

                    if "out of memory" in str(e):
                        self.print(
                            f"Step {self.steps}, Accum {grad_accum_iter} | Failing on training batch forward due to GPU being out of memory."
                        )
                    if "PreBackwardHighMemory" in str(e):
                        self.print(
                            f"Step {self.steps}, Accum {grad_accum_iter} | PreBackwardHighMemory guard triggered; skipping this step."
                        )

                except Exception as e:
                    self.print(
                        f"Step {self.steps}, Accum {grad_accum_iter} | Skipping training batch due to forward exception: {e}, {traceback.format_exc()}"
                    )
                    loss = torch.tensor([torch.nan], device=self.device)

                    if "out of memory" in str(e):
                        self.print(
                            f"Step {self.steps}, Accum {grad_accum_iter} | Failing on training batch forward due to GPU being out of memory."
                        )
                    if "PreBackwardHighMemory" in str(e):
                        self.print(
                            f"Step {self.steps}, Accum {grad_accum_iter} | PreBackwardHighMemory guard triggered; skipping this step."
                        )

                # skip step if any device fails its forward pass (e.g., by running out of memory)
                self.wait()
                
                # print loss for each GPU
                print(f"\nRank {current_rank} | Loss: {loss.item():.3f}")

                # Gather losses just to check loss is not nan/inf           
                if self.use_native_deepspeed_2d:
                    parallel_info = get_parallel_info()
                    if parallel_info["ddp_size"] > 1:
                        gathered_losses = [torch.zeros_like(loss) for _ in range(parallel_info["ddp_size"])]
                        dist.all_gather(gathered_losses, loss, group=parallel_info["current_ddp_process_group"]) # only gather within the same DP group because loss of SP group is the same 
                        losses = torch.stack(gathered_losses)
                    else:
                        losses = loss
                else:
                    losses = self.fabric.all_gather(loss)

                if torch.isnan(losses).any() or torch.isinf(losses).any():
                    self.print(
                        f"Step {self.steps}, Accum {grad_accum_iter} | Skipping training batch due to invalid (e.g., NaN or inf) loss."
                    )

                    # clean up the computational graph using a cool-down period

                    self.wait()
                    zero_grad()

                    del train_batch, loss, losses
                    if exists(loss_breakdown):
                        del loss_breakdown

                    garbage_collection_cuda()

                    grad_accum_iter = 0
                    is_accumulating = grad_accum_iter < self.grad_accum_every

                    # Do not track timing for this skipped step
                    prevTime = None

                    self.wait()
                    continue # If nan loss, since all processes have allgathered, they will all "continue" and get the next batch here

                # backward pass

                try:
                    if verbose == "extra":
                        self.print(
                            f"Step {self.steps}, Accum {grad_accum_iter} | Backward pass..."
                        )
                    with nvtx_range(f"train.step_{self.steps}.backward"):
                        if self.use_native_deepspeed_2d:
                            # Native DeepSpeed 2D handles gradient accumulation internally
                            self.model_engine.backward(loss)
                        else:
                            self.fabric.backward(
                                loss / (1.0 if self.using_deepspeed_strategy else self.grad_accum_every)
                            )
                except Exception as e:
                    self.print(
                        f"Step {self.steps}, Accum {grad_accum_iter} | Failing on training batch backward due to exception: {e}, {traceback.format_exc()}"
                    )
                    raise e ## raise error to terminate the entire process (we don't want this, we just want to skip any OOM process)
                    ## the OOM gpu will terminate but since it's in the backward pass -- other ranks just keep waiting for grad synch from this OOM rank and thus silently failed

                print(f"Rank {current_rank} | Memory after backward pass: {SynchronizedWallClockTimer.memory_usage()}")

            # proceed only after accumulating all gradients

            if not is_accumulating:
                # loss metrics

                for k, v in loss_breakdown._asdict().items():
                    # lazily create breakdown metrics
                    if not hasattr(self, f"mean_loss_breakdown_{k}"):
                        setattr(
                            self,
                            f"mean_loss_breakdown_{k}",
                            MeanMetric(sync_on_compute=True).to(self.device),
                        )
                    mean_train_metric = getattr(self, f"mean_loss_breakdown_{k}")
                    mean_train_metric.update(v.detach() if torch.is_tensor(v) else v)

                # gradient clipping

                if self.clip_grad_norm > 0 and not self.using_deepspeed_strategy and not self.use_native_deepspeed_2d: # NOTE: DeepSpeed handles gradient clipping internally
                    if verbose == "extra":
                        self.print(
                            f"Step {self.steps} | Clipping gradients to a maximum norm of {self.clip_grad_norm}..."
                        )
                    self.fabric.clip_gradients(
                        self.model, self.model_optimizer, max_norm=self.clip_grad_norm
                    )

                # optimizer step

                if verbose == "extra":
                    self.print(f"Step {self.steps} | Optimization...")

                with nvtx_range(f"train.step_{self.steps}.optimizer"):
                    if self.use_native_deepspeed_2d:
                        # DeepSpeed handles optimizer step
                        self.model_engine.step()
                    else:
                        self.model_optimizer.step()

                # update exponential moving average

                if verbose == "extra":
                    self.print(f"Step {self.steps} | EMA update...")

                # NOTE: it is assumed that for non-parameter-sharding training strategies
                # such as DeepSpeed ZeRO Stage 2, the model parameters at this point on each
                # device are identical for the current EMA weight update, such that the
                # rank zero EMA weights can subsequently be treated as global EMA weights

                with nvtx_range(f"train.step_{self.steps}.ema"):
                    if self.has_ema:
                        self.ema_model.update()

                # zero gradients

                if self.fabric and (not isinstance(self.fabric.strategy, DeepSpeedStrategy)) and (not self.use_native_deepspeed_2d):
                    # NOTE: DeepSpeed handles gradient zeroing internally

                    if verbose == "extra":
                        self.print(f"Step {self.steps} | Zeroing gradients...")

                    with nvtx_range(f"train.step_{self.steps}.zero_grad"):
                        zero_grad()

                # update scheduler

                if verbose == "extra":
                    self.print(f"Step {self.steps} | Scheduler update...")

                with nvtx_range(f"train.step_{self.steps}.scheduler"):
                    if self.use_native_deepspeed_2d:
                        # DeepSpeed automatically manages scheduler step when lr_scheduler is passed to initialize()
                        # Do NOT call scheduler.step() manually - DeepSpeed handles it internally
                        pass
                    else:
                        self.scheduler.step()

                # increment steps

                self.steps += 1
                grad_accum_iter = 0

                # visualize samples

                seq_len = train_batch.molecule_atom_lens.shape[-1]
                filepaths_available = hasattr(train_batch, "filepath") and exists(
                    train_batch.filepath
                )
                visualize_samples = (
                    # NOTE: we cannot visualize cropped examples, since the sampled atom positions
                    # would then not be of the same shape as the original atom positions
                    filepaths_available
                    and self.visualize_train_samples_every_n_steps > 0
                    and self.steps % self.visualize_train_samples_every_n_steps == 0
                    and seq_len < self.crop_size
                )

                if visualize_samples:
                    if verbose == "extra":
                        self.print(f"Step {self.steps} | Sample visualization...")

                    eval_model = default(self.ema_model, self.model)

                    with torch.no_grad(), to_device_and_back(eval_model, self.device):
                        eval_model.eval()

                        try:
                            self.sample_and_visualize(
                                eval_model,
                                train_batch,
                                self.steps,
                                phase="train",
                                # verbose=verbose in ("standard", "extra"),
                            )

                        except BaseException as e:
                            self.print(
                                f"Step {self.steps} | Skipping sample visualization due to base exception: {e}, {traceback.format_exc()}"
                            )
                            garbage_collection_cuda()

                        except Exception as e:
                            self.print(
                                f"Step {self.steps} | Skipping sample visualization due to exception: {e}, {traceback.format_exc()}"
                            )
                            garbage_collection_cuda()

                # log

                if self.steps % self.train_log_interval == 0:
                    if verbose == "extra":
                        self.print(f"Step {self.steps} | Logging...")

                    metrics = {
                        "step": self.steps,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                    }

                    for k in loss_breakdown._asdict():
                        # NOTE: these are expensive device-to-host synchronizations
                        mean_train_metric = getattr(self, f"mean_loss_breakdown_{k}")
                        metrics[f"train/{k}"] = mean_train_metric.compute().item()

                    self.print(
                        f"Step {metrics['step']} |"
                        f" Train loss: {metrics['train/total_loss']:.6f} (step)"
                    )
                    lossSoFar.append(metrics['train/total_loss'])
                    

                    self.log_dict(**metrics)

                # maybe validate with EMA model

                force_save_best_checkpoint = False

                if self.needs_valid and divisible_by(self.steps, self.valid_every):
                    if verbose == "extra":
                        self.print(f"Step {self.steps} | Validating...")

                    # set up metric accumulation

                    with nvtx_range(f"train.step_{self.steps}.validation"):
                        mean_model_selection_score = MeanMetric(sync_on_compute=True).to(self.device)
                        mean_top_ranked_lddt = MeanMetric(sync_on_compute=True).to(self.device)

                        self.wait()

                        if verbose:
                            self.print("Validating...")

                        eval_model = default(self.ema_model, self.model)

                        with torch.no_grad(), to_device_and_back(eval_model, self.device):
                            eval_model.eval()

                            for valid_batch_idx, valid_batch in enumerate(self.valid_dataloader):
                                # Move batch to device for distributed training
                                if self.use_native_deepspeed_2d or self.using_deepspeed_strategy:
                                    valid_batch = valid_batch.to(self.device)
                                    
                                if (
                                    exists(self.num_valid_steps)
                                    and valid_batch_idx >= self.num_valid_steps
                                ):
                                    self.print(
                                        f"Step {self.steps} |"
                                        f" Stopping validation early after seeing {self.num_valid_steps} val batches."
                                    )
                                    del valid_batch
                                    garbage_collection_cuda()
                                    break

                                if verbose == "extra":
                                    self.print(
                                        f"Step {self.steps} | Running val step {valid_batch_idx}..."
                                    )

                                # generate multiple samples per example in each batch

                            valid_samples: List[Sample] = []

                            try:
                                for _ in range(self.num_samples_per_example):
                                    valid_sampled_atom_pos, valid_logits = timeout(
                                        dec_timeout=SAMPLING_MAX_SECONDS_PER_INPUT,
                                        use_signals=True,
                                        timeout_exception=BaseException,
                                    )(eval_model.__call__)(
                                        **valid_batch.dict(),
                                        dtype=self.dtype,
                                        return_loss=False,
                                        return_confidence_head_logits=True,
                                        return_distogram_head_logits=True,
                                        num_sample_steps=200,
                                        num_recycling_steps=4,
                                        # verbose=verbose == "extra",
                                    )
                                    valid_plddt = ComputeConfidenceScore.compute_plddt(
                                        valid_logits.plddt.to(samples_device)
                                    )
                                    valid_samples.append(
                                        (
                                            valid_sampled_atom_pos.to(samples_device),
                                            valid_logits.pde.to(samples_device),
                                            valid_plddt.to(samples_device),
                                            valid_logits.distance.to(samples_device),
                                        )
                                    )

                            except BaseException as e:
                                self.print(
                                    f"Step {self.steps} |"
                                    f" Skipping validation step {valid_batch_idx} due to base exception: {e}, {traceback.format_exc()}"
                                )
                                mean_model_selection_score.update(
                                    torch.tensor([torch.nan], device=self.device)
                                )
                                mean_top_ranked_lddt.update(
                                    torch.tensor([torch.nan], device=self.device)
                                )

                                del valid_batch
                                garbage_collection_cuda()

                            except Exception as e:
                                self.print(
                                    f"Step {self.steps} |"
                                    f" Skipping validation step {valid_batch_idx} due to exception: {e}, {traceback.format_exc()}"
                                )
                                mean_model_selection_score.update(
                                    torch.tensor([torch.nan], device=self.device)
                                )
                                mean_top_ranked_lddt.update(
                                    torch.tensor([torch.nan], device=self.device)
                                )

                                del valid_batch
                                garbage_collection_cuda()

                            # NOTE: we must wait until all ranks are synchronized each validation step
                            # before we decide which ranks can compute valid scores and visualize samples
                            self.wait()
                            if len(valid_samples) != self.num_samples_per_example:
                                continue

                            valid_score_details = self.compute_model_selection_score.compute_model_selection_score(
                                valid_batch,
                                valid_samples,
                                is_fine_tuning=self.is_fine_tuning,
                                return_details=True,
                                # NOTE: the AF3 supplement (Section 5.7) suggests that DM did not compute validation RASA for unresolved regions
                                compute_rasa=False,
                                device=samples_device,
                            )

                            valid_top_sample = valid_score_details.scored_samples[
                                valid_score_details.best_gpde_index
                            ]
                            (
                                valid_top_sample_idx,
                                valid_top_batch_sampled_atom_pos,
                                valid_top_sample_plddt,
                                valid_top_model_selection_score,
                                _,
                            ) = valid_top_sample

                            # compute the unweighted lDDT score

                            valid_unweighted_score_details = (
                                self.compute_model_selection_score.compute_model_selection_score(
                                    valid_batch,
                                    valid_samples,
                                    is_fine_tuning=self.is_fine_tuning,
                                    return_details=True,
                                    return_unweighted_scores=True,
                                    compute_rasa=False,
                                    device=samples_device,
                                )
                            )

                            valid_unweighted_top_sample = (
                                valid_unweighted_score_details.scored_samples[
                                    valid_unweighted_score_details.best_gpde_index
                                ]
                            )
                            valid_top_ranked_lddt = valid_unweighted_top_sample[3]

                            mean_model_selection_score.update(
                                valid_score_details.score.mean().detach()
                            )
                            mean_top_ranked_lddt.update(valid_top_ranked_lddt.mean().detach())

                            # visualize (top) samples

                            seq_len = valid_batch.molecule_atom_lens.shape[-1]
                            filepaths_available = hasattr(valid_batch, "filepath") and exists(
                                valid_batch.filepath
                            )
                            visualize_samples = (
                                # NOTE: we cannot visualize cropped examples, since the sampled atom positions
                                # would then not be of the same shape as the original atom positions
                                filepaths_available
                                and self.visualize_valid_samples_every_n_steps > 0
                                and self.steps % self.visualize_valid_samples_every_n_steps == 0
                                and seq_len < self.crop_size
                            )

                            if visualize_samples:
                                assert exists(
                                    valid_top_batch_sampled_atom_pos
                                ), "The top sampled validation atom positions must be provided to visualize them."
                                filename_suffixes = [
                                    f"-score-{score:.4f}"
                                    for score in valid_top_model_selection_score.tolist()
                                ]
                                filepaths = (
                                    list(valid_batch.filepath)
                                    if hasattr(valid_batch, "filepath")
                                    and exists(valid_batch.filepath)
                                    else None
                                )
                                if exists(filepaths):
                                    self.visualize(
                                        sampled_atom_pos=valid_top_batch_sampled_atom_pos,
                                        atom_mask=~valid_batch.missing_atom_mask,
                                        filepaths=filepaths,
                                        batch_idx=valid_batch_idx,
                                        phase="val",
                                        sample_idx=valid_top_sample_idx,
                                        filename_suffixes=filename_suffixes,
                                        b_factors=valid_top_sample_plddt,
                                        # verbose=verbose in ("standard", "extra"),
                                    )

                    # log

                    valid_model_selection_score = (
                        mean_model_selection_score.compute().item()
                    )  # NOTE: expensive device-to-host synchronization
                    valid_top_ranked_lddt = (
                        mean_top_ranked_lddt.compute().item()
                    )  # NOTE: expensive device-to-host synchronization

                    valid_metrics = {
                        "val/model_selection_score": valid_model_selection_score,
                        "val/top_ranked_lddt": valid_top_ranked_lddt,
                    }

                    self.print(
                        f"Step {self.steps} |"
                        f" Val model selection score: {valid_metrics['val/model_selection_score']:.6f} (epoch),",
                        f" Val top ranked lDDT: {valid_metrics['val/top_ranked_lddt']:.6f} (epoch)",
                    )

                    self.log_dict(**valid_metrics)

                    self.wait()

                    # track best model selection score

                    if (
                        valid_metrics["val/model_selection_score"]
                        > self.best_model_selection_score
                    ):
                        if verbose:
                            self.print(
                                f"Step {self.steps} |"
                                f" New best val model selection score: {valid_metrics['val/model_selection_score']:.6f} (epoch),",
                                f" New best val top ranked lDDT: {valid_metrics['val/top_ranked_lddt']:.6f} (epoch)",
                            )

                        self.best_model_selection_step = self.steps
                        self.best_model_selection_score = valid_metrics[
                            "val/model_selection_score"
                        ]
                        self.best_top_ranked_lddt = valid_metrics["val/top_ranked_lddt"]

                        force_save_best_checkpoint = True

                # maybe save a checkpoint

                if force_save_best_checkpoint or divisible_by(self.steps, self.checkpoint_every):
                    if verbose == "extra":
                        self.print(
                            f"Step {self.steps} | Saving a{' new best ' if force_save_best_checkpoint else ' '}checkpoint..."
                        )
                    self.save_checkpoint()

                # clear CUDA cache

                if (
                    self.clear_cuda_cache_every > 0
                    and self.steps % self.clear_cuda_cache_every == 0
                ):
                    if verbose == "extra":
                        self.print(f"Step {self.steps} | Clearing CUDA cache...")
                    torch.cuda.empty_cache()

        # maybe finish profiling

        if self.profile:
            self.print("Stopping profiler...")
            self.profiler.stop()

        # maybe test

        if self.needs_test:
            self.wait()

            self.load_from_checkpoint_folder(load_best_model=True)

            # set up metric accumulation

            mean_model_selection_score = MeanMetric(sync_on_compute=True).to(self.device)
            mean_top_ranked_lddt = MeanMetric(sync_on_compute=True).to(self.device)

            self.wait()

            if verbose:
                self.print("Testing...")

            eval_model = default(self.ema_model, self.model)

            with torch.no_grad(), to_device_and_back(eval_model, self.device):
                eval_model.eval()

                for test_batch_idx, test_batch in enumerate(self.test_dataloader):
                    # Move batch to device for distributed training
                    if self.use_native_deepspeed_2d or self.using_deepspeed_strategy:
                        test_batch = test_batch.to(self.device)
                        
                    if exists(self.num_test_steps) and test_batch_idx >= self.num_test_steps:
                        self.print(
                            f"Step {self.steps} |"
                            f" Stopping testing early after seeing {self.num_test_steps} test batches."
                        )
                        del test_batch
                        garbage_collection_cuda()
                        break

                    if verbose == "extra":
                        self.print(f"Step {self.steps} | Running test step {test_batch_idx}...")

                    # generate multiple samples per example in each batch

                    test_samples: List[Sample] = []

                    try:
                        for _ in range(self.num_samples_per_example):
                            test_sampled_atom_pos, test_logits = timeout(
                                dec_timeout=SAMPLING_MAX_SECONDS_PER_INPUT,
                                use_signals=True,
                                timeout_exception=BaseException,
                            )(eval_model.__call__)(
                                **test_batch.dict(),
                                dtype=self.dtype,
                                return_loss=False,
                                return_confidence_head_logits=True,
                                return_distogram_head_logits=True,
                                num_sample_steps=200,
                                num_recycling_steps=4,
                                # verbose=verbose == "extra",
                            )
                            test_plddt = ComputeConfidenceScore.compute_plddt(
                                test_logits.plddt.to(samples_device)
                            )
                            test_samples.append(
                                (
                                    test_sampled_atom_pos.to(samples_device),
                                    test_logits.pde.to(samples_device),
                                    test_plddt.to(samples_device),
                                    test_logits.distance.to(samples_device),
                                )
                            )

                    except BaseException as e:
                        self.print(
                            f"Step {self.steps} |"
                            f" Skipping test step {test_batch_idx} due to base exception: {e}, {traceback.format_exc()}"
                        )
                        mean_model_selection_score.update(
                            torch.tensor([torch.nan], device=self.device)
                        )
                        mean_top_ranked_lddt.update(torch.tensor([torch.nan], device=self.device))

                        del test_batch
                        garbage_collection_cuda()

                    except Exception as e:
                        self.print(
                            f"Step {self.steps} |"
                            f" Skipping test step {test_batch_idx} due to exception: {e}, {traceback.format_exc()}"
                        )
                        mean_model_selection_score.update(
                            torch.tensor([torch.nan], device=self.device)
                        )
                        mean_top_ranked_lddt.update(torch.tensor([torch.nan], device=self.device))

                        del test_batch
                        garbage_collection_cuda()

                    # NOTE: we must wait until all ranks are synchronized each test step
                    # before we decide which ranks can compute valid scores and visualize samples
                    self.wait()
                    if len(test_samples) != self.num_samples_per_example:
                        continue

                    test_score_details = self.compute_model_selection_score.compute_model_selection_score(
                        test_batch,
                        test_samples,
                        is_fine_tuning=self.is_fine_tuning,
                        return_details=True,
                        return_unweighted_scores=False,
                        # NOTE: the AF3 supplement (Section 5.7) suggests that DM computed RASA only for the test set's unresolved regions
                        # NOTE: cannot find where to get the unresolved chain IDs and residue masks from to match the AF3 supplement
                        compute_rasa=True,
                        unresolved_cid=None,
                        unresolved_residue_mask=None,
                        device=samples_device,
                    )

                    test_top_sample = test_score_details.scored_samples[
                        test_score_details.best_gpde_index
                    ]
                    (
                        test_top_sample_idx,
                        test_top_batch_sampled_atom_pos,
                        test_top_sample_plddt,
                        test_top_model_selection_score,
                        _,
                    ) = test_top_sample

                    # compute the unweighted lDDT score

                    test_unweighted_score_details = (
                        self.compute_model_selection_score.compute_model_selection_score(
                            test_batch,
                            test_samples,
                            is_fine_tuning=self.is_fine_tuning,
                            return_details=True,
                            return_unweighted_scores=True,
                            compute_rasa=False,
                            device=samples_device,
                        )
                    )

                    test_unweighted_top_sample = test_unweighted_score_details.scored_samples[
                        test_unweighted_score_details.best_gpde_index
                    ]
                    test_top_ranked_lddt = test_unweighted_top_sample[3]

                    mean_model_selection_score.update(test_score_details.score.mean().detach())
                    mean_top_ranked_lddt.update(test_top_ranked_lddt.mean().detach())

                    # visualize (top) samples

                    seq_len = test_batch.molecule_atom_lens.shape[-1]
                    filepaths_available = hasattr(test_batch, "filepath") and exists(
                        test_batch.filepath
                    )
                    visualize_samples = (
                        # NOTE: we cannot visualize cropped examples, since the sampled atom positions
                        # would then not be of the same shape as the original atom positions
                        filepaths_available
                        and self.visualize_test_samples_every_n_steps > 0
                        and self.steps % self.visualize_test_samples_every_n_steps == 0
                        and seq_len < self.crop_size
                    )

                    if visualize_samples:
                        assert exists(
                            test_top_batch_sampled_atom_pos
                        ), "The top sampled test atom positions must be provided to visualize them."
                        filename_suffixes = [
                            f"-score-{score:.4f}"
                            for score in test_top_model_selection_score.tolist()
                        ]
                        filepaths = (
                            list(test_batch.filepath)
                            if hasattr(test_batch, "filepath") and exists(test_batch.filepath)
                            else None
                        )
                        if exists(filepaths):
                            self.visualize(
                                sampled_atom_pos=test_top_batch_sampled_atom_pos,
                                atom_mask=~test_batch.missing_atom_mask,
                                filepaths=filepaths,
                                batch_idx=test_batch_idx,
                                phase="test",
                                sample_idx=test_top_sample_idx,
                                filename_suffixes=filename_suffixes,
                                b_factors=test_top_sample_plddt,
                                # verbose=verbose in ("standard", "extra"),
                            )

            # log

            test_model_selection_score = (
                mean_model_selection_score.compute().item()
            )  # NOTE: expensive device-to-host synchronization
            test_top_ranked_lddt = (
                mean_top_ranked_lddt.compute().item()
            )  # NOTE: expensive device-to-host synchronization

            test_metrics = {
                "test/model_selection_score": test_model_selection_score,
                "test/top_ranked_lddt": test_top_ranked_lddt,
            }

            self.print(
                f"Step {self.steps} |"
                f" Test model selection score: {test_metrics['test/model_selection_score']:.6f} (epoch),",
                f" Test top ranked lDDT: {test_metrics['test/top_ranked_lddt']:.6f} (epoch)",
            )

            self.log_dict(**test_metrics)

        self.wait()

        # maybe log profiler artifacts

        if self.profile and self.is_main:
            assert package_available(
                "wandb"
            ), "Please install and use the `wandb` package to log profiler artifacts."
            import wandb

            profile_art = wandb.Artifact("trace", type="profile")

            trace_files = list(glob.glob(os.path.join(self.profiler_log_dir, "*.pt.trace.json")))
            assert trace_files, "No trace files found."

            profile_art.add_file(trace_files[0], "trace.pt.trace.json")
            if not self.use_native_deepspeed_2d:
                self.fabric.logger.experiment.log_artifact(profile_art)

            self.print("Profiler artifacts logged.")

        print("Training complete.")
