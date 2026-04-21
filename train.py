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

import argparse
import os
from contextlib import contextmanager

import torch
from loguru import logger

from megafold.configs import create_trainer_from_conductor_yaml


def _nvtx_enabled() -> bool:
    value = os.environ.get("MEGAFOLD_NVTX", "")
    return value.lower() not in {"", "0", "false", "no"}


@contextmanager
def _nvtx_range(message: str):
    if not _nvtx_enabled() or not torch.cuda.is_available():
        yield
        return

    import torch.cuda.nvtx as nvtx

    nvtx.range_push(message)
    try:
        yield
    finally:
        nvtx.range_pop()


def main(config_path: str, trainer_name: str):
    """Main function for training.

    :param config_path: Path to a conductor config file.
    :param trainer_name: Name of the trainer to use.
    """
    assert os.path.exists(config_path), f"Config file not found at {config_path}."
    torch.set_float32_matmul_precision("high")

    trainer = create_trainer_from_conductor_yaml(config_path, trainer_name=trainer_name)
    trainer.load_from_checkpoint_folder()

    logger.info("Trainer starting!")
    with _nvtx_range("train"):
        trainer()
    logger.info("Trainer finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MegaFold.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a conductor config file.",
    )
    parser.add_argument(
        "--trainer_name",
        type=str,
        required=True,
        help="Name of the trainer to use.",
    )
    # Add support for DeepSpeed's --local_rank argument
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by DeepSpeed launcher).",
    )
    args = parser.parse_args()
    main(args.config, args.trainer_name)
