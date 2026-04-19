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

"""
Parallel information management for 2D parallelism.
This module is separate from trainer to avoid circular imports.
"""

import torch.distributed as dist
import datetime

# Global storage for parallel info (accessible from anywhere)
_PARALLEL_INFO = None


def get_parallel_info():
    """Get the current parallel configuration."""
    global _PARALLEL_INFO
    if _PARALLEL_INFO is None:
        return {
            'ddp_groups': [[0]],
            'sp_groups': [[0]],
            'current_ddp_group': [0],
            'current_sp_group': [0],
            'current_ddp_process_group': None,
            'current_sp_process_group': None,
            'ddp_rank': 0,
            'sp_rank': 0,
            'ddp_size': 1,
            'sp_size': 1,
        }
    return _PARALLEL_INFO


def set_parallel_info(parallel_info):
    """Store parallel configuration globally"""
    global _PARALLEL_INFO
    _PARALLEL_INFO = parallel_info


def setup_2d_parallel_groups(data_parallel_size=2, sequence_parallel_size=2, timeout_seconds=3600):
    """
    Setup 2D parallelism process groups.
    
    Parameters:
    - data_parallel_size: Number of SP groups (each SP group gets different data)
    - sequence_parallel_size: Size of each SP group (ranks that process same data, different sequence chunks)
    
    For 4 GPUs with data_parallel_size=2, sequence_parallel_size=2:
    - SP groups: [0,1] and [2,3] (same data within group, different sequence chunks)
    - DDP groups: [0,2] and [1,3] (different data between corresponding ranks)
    
    Standard layout: contiguous ranks form SP groups.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    assert world_size == data_parallel_size * sequence_parallel_size, (
        f"World size {world_size} != dp_size {data_parallel_size} * "
        f"sp_size {sequence_parallel_size}"
    )
    
    # Create timeout for process groups (convert seconds to timedelta)
    timeout = datetime.timedelta(seconds=timeout_seconds)
    
    # Create SP groups: ranks that get same data, different sequence chunks
    # Standard layout: contiguous ranks in same SP group
    sp_groups = []  # [0, 1], [2, 3]
    sp_process_groups = []
    for sp_group_id in range(data_parallel_size): # 0 -> [0, 1]; 1 -> [2, 3]
        group_ranks = []
        for local_rank in range(sequence_parallel_size):
            group_ranks.append(sp_group_id * sequence_parallel_size + local_rank)
        sp_groups.append(group_ranks)
        pg = dist.new_group(group_ranks, timeout=timeout)
        sp_process_groups.append(pg)

    # Create DDP groups: ranks that get different data
    # Each DDP group contains ranks with same local position from different SP groups
    ddp_groups = []  # [0, 2], [1, 3]
    ddp_process_groups = []
    for local_rank in range(sequence_parallel_size):
        group_ranks = []
        for sp_group_id in range(data_parallel_size):
            group_ranks.append(sp_group_id * sequence_parallel_size + local_rank)
        ddp_groups.append(group_ranks)
        pg = dist.new_group(group_ranks, timeout=timeout)
        ddp_process_groups.append(pg)

    # Determine which groups current rank belongs to
    current_ddp_group = None
    current_sp_group = None
    current_ddp_process_group = None
    current_sp_process_group = None
    ddp_rank = None
    sp_rank = None

    for i, group in enumerate(ddp_groups):
        if rank in group:
            current_ddp_group = group
            current_ddp_process_group = ddp_process_groups[i]
            ddp_rank = group.index(rank)
            break

    for i, group in enumerate(sp_groups):
        if rank in group:
            current_sp_group = group
            current_sp_process_group = sp_process_groups[i]
            sp_rank = group.index(rank)
            break

    parallel_info = {
        'ddp_groups': ddp_groups,
        'sp_groups': sp_groups,
        'current_ddp_group': current_ddp_group,
        'current_sp_group': current_sp_group,
        'current_ddp_process_group': current_ddp_process_group,  # DDP PG
        'current_sp_process_group': current_sp_process_group,  # SP PG object
        'ddp_rank': ddp_rank,
        'sp_rank': sp_rank,  # Local rank within an SP group
        'ddp_size': len(current_ddp_group),
        'sp_size': len(current_sp_group)
    }

    # Store globally for easy access from anywhere
    set_parallel_info(parallel_info)

    return parallel_info