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
import torch.nn as nn 
from deepspeed.utils.timer import SynchronizedWallClockTimer 
from megafold.model.FusedLayernormLinear.fused_layernorm_linear import LayernormLinear

device = 'cuda'
dtype = torch.float16
M, N, K = 147456, 128, 128
provider = "triton"


if provider == "triton":
    a = torch.randn((M, K), dtype=dtype, device=device)
    fused_kernel = LayernormLinear(K, N, has_layernorm_bias=True, has_linear_bias=False, dtype=dtype, device=device)
    c = fused_kernel(a)
    dC = torch.randn_like(c)
    c.backward(dC)
    
elif provider == "torch":   
    a = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    ln = nn.LayerNorm(K, dtype=dtype, device=device)
    linear = nn.Linear(K, N, bias=False, dtype=dtype, device=device)    
    c = linear(ln(a))
    dC = torch.randn_like(c)
    c.backward(dC)
    
mem_usage = SynchronizedWallClockTimer.memory_usage()
print("Memory usage after pass: ", mem_usage)
