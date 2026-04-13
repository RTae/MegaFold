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

import torch 
from deepspeed.utils.timer import SynchronizedWallClockTimer 
from megafold.model.FusedTransition.fused_transition import FusedTransition
from benchmarks.transition_speed import Transition

device = 'cuda'
dtype = torch.bfloat16
M, d, expansion_factor = 147456, 64, 4 
provider = "triton"


if provider == "triton":
    x = torch.randn((M, d), device=device, dtype=dtype)
    triton_transition = FusedTransition(dim=d, expansion_factor=expansion_factor, device=device, dtype=dtype)
    o = triton_transition(x)
    do = torch.randn(o.shape, device=device, dtype=dtype)
    o.backward(do, retain_graph=True)

elif provider == "torch":
    x = torch.randn((M, d), device=device, dtype=dtype, requires_grad=True)
    torch_transition = Transition(dim=d, expansion_factor=expansion_factor, device=device, dtype=dtype)
    o = torch_transition(x)
    do = torch.randn(o.shape, device=device, dtype=dtype)
    o.backward(do, retain_graph=True)


mem_usage = SynchronizedWallClockTimer.memory_usage()
print("Memory usage after pass: ", mem_usage)
