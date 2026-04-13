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

from typing import Tuple
import torch
import torch.distributed as dist
from torch import Tensor
from megafold.distributed.parallel_info import get_parallel_info
from einops import rearrange


__all__ = [
    'scatter',
    'gather',
    'gather_async',
    'gather_async_opp',
    'col_to_row',
    'row_to_col',
    'reduce_sum',
    'reduce_scatter_sum'
]


def divide(numerator, denominator):
    assert numerator // denominator == numerator / denominator
    return numerator // denominator


def _split(tensor: Tensor, dim: int = -1) -> Tensor:
    parallel_info = get_parallel_info()
    if parallel_info["sp_size"] == 1:
        return tensor

    split_size = divide(tensor.shape[dim], parallel_info["sp_size"])
    tensor_list = torch.split(tensor, split_size, dim=dim)

    output = tensor_list[parallel_info["sp_rank"]].contiguous()

    return output


def _gather(tensor: Tensor, dim: int = -1) -> Tensor:
    parallel_info = get_parallel_info()
    if parallel_info["sp_size"] == 1:
        return tensor

    # Standard execution flow (works on regular tensors)
    if dim == 1 and list(tensor.shape)[0] == 1:
        output_shape = list(tensor.shape)
        output_shape[1] *= parallel_info["sp_size"]
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_list = output.chunk(parallel_info["sp_size"], dim=1)
        dist.all_gather(list(tensor_list),
                        tensor,
                        group=parallel_info["current_sp_process_group"],
                        async_op=False)
    else:
        tensor_list = [
            torch.empty_like(tensor) for _ in range(parallel_info["sp_size"])
        ]
        dist.all_gather(tensor_list,
                        tensor,
                        group=parallel_info["current_sp_process_group"],
                        async_op=False)
        output = torch.cat(tensor_list, dim=dim)
    
    return output


def scatter(input: Tensor, dim: int = -1) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Scatter.apply(input, dim)
    else:
        input = _split(input, dim=dim)
    return input


class Scatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Scatter", input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _split(input, dim=dim)

    @staticmethod
    def backward(ctx: "Scatter", grad_output: Tensor) -> Tuple[Tensor]:
        dim, = ctx.saved_tensors
        return _gather(grad_output, dim=int(dim)), None


def gather(input: Tensor, dim: int = -1) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Gather.apply(input, dim)
    else:
        input = _gather(input, dim=dim)
    return input


class Gather(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Gather", input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _gather(input, dim=dim)

    @staticmethod
    def backward(ctx: "Gather", grad_output: Tensor) -> Tuple[Tensor]:
        dim, = ctx.saved_tensors
        return _split(grad_output, dim=int(dim)), None


def _gather_async(tensor: Tensor, dim: int = -1) -> Tensor:
    parallel_info = get_parallel_info()
    if parallel_info["sp_size"]== 1:
        return tensor, None

    output_shape = list(tensor.shape)
    output_shape[dim] *= parallel_info["sp_size"]
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    tensor_list = output.chunk(parallel_info["sp_size"], dim=dim)
    work = dist.all_gather(list(tensor_list),
                           tensor,
                           group=parallel_info["current_sp_process_group"],
                           async_op=True)
    return output, work


def gather_async(input: Tensor, dim: int = -1) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input, work = GatherAsync.apply(input, dim)
    else:
        input, work = _gather_async(input, dim=dim)
    return input, work


def gather_async_opp(output: Tensor, work, dim: int = -1) -> Tensor:
    if work:
        work.wait()
    return output


class GatherAsyncOpp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "GatherAsyncOpp", input: Tensor) -> Tensor:
        parallel_info = get_parallel_info()
        sp_size = parallel_info["sp_size"]
        output = rearrange(input, 'n (x h) w c -> n h (x w) c', x=sp_size)
        return output

    @staticmethod
    def backward(ctx: "GatherAsyncOpp", grad_output: Tensor) -> Tuple[Tensor]:
        parallel_info = get_parallel_info()
        sp_size = parallel_info["sp_size"]
        n, h, w, c = grad_output.shape
        return grad_output.resize_(n, h * sp_size, int(w / sp_size), c)


class GatherAsync(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "GatherAsync", input: Tensor, dim: int = -1) -> Tensor:
        ctx.dim = dim
        return _gather_async(input, dim=dim)

    @staticmethod
    def backward(ctx: "GatherAsync", grad_output: Tensor, grad_work=None) -> Tuple[Tensor]:
        out = _split(grad_output, dim=ctx.dim)
        return out, None


def _all_to_all(tensor: Tensor, in_dim: int = -1, out_dim: int = -1) -> Tensor:
    parallel_info = get_parallel_info()
    if parallel_info["sp_size"] == 1:
        return tensor

    split_size = divide(tensor.shape[in_dim], parallel_info["sp_size"])
    input_tensor_list = torch.split(tensor, split_size, dim=in_dim)

    input_tensor_list = [tensor_.contiguous() for tensor_ in input_tensor_list]
    if out_dim == 1:
        # Create contiguous output tensors for all_to_all
        output_tensor_list = [torch.empty_like(input_tensor_list[0]).contiguous() for _ in range(parallel_info["sp_size"])]
        
        try:
            dist.all_to_all(output_tensor_list,
                            input_tensor_list,
                            group=parallel_info["current_sp_process_group"],
                            async_op=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
        output = torch.cat(output_tensor_list, dim=out_dim)
    else:
        # Create contiguous output tensors for all_to_all
        output_tensor_list = [torch.empty_like(tensor_).contiguous() for tensor_ in input_tensor_list]

        try:
            dist.all_to_all(output_tensor_list,
                            input_tensor_list,
                            group=parallel_info["current_sp_process_group"],
                            async_op=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
            
        output = torch.cat(output_tensor_list, dim=out_dim)

    return output


def col_to_row(input_: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input_.requires_grad:
        input_ = All_to_All.apply(input_, 1, 2)
    else:
        input_ = _all_to_all(input_, in_dim=1, out_dim=2)
    return input_


def row_to_col(input_: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input_.requires_grad:
        input_ = All_to_All.apply(input_, 2, 1)
    else:
        input_ = _all_to_all(input_, in_dim=2, out_dim=1)
    return input_


class All_to_All(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "All_to_All", input_: Tensor, in_dim: int = -1, out_dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([in_dim, out_dim]))
        return _all_to_all(input_, in_dim=in_dim, out_dim=out_dim)

    @staticmethod
    def backward(ctx: "All_to_All", grad_output: Tensor) -> Tuple[Tensor]:
        saved_tensors = ctx.saved_tensors[0]
        return _all_to_all(grad_output, in_dim=int(saved_tensors[1]),
                           out_dim=int(saved_tensors[0])), None, None


def reduce_sum(input: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = ReduceSum.apply(input)
    else:
        input = _reduce_sum(input)
    return input

def _reduce_sum(tensor: Tensor) -> Tensor:
    parallel_info = get_parallel_info()
    if parallel_info["sp_size"] == 1:
        return tensor

    dist.all_reduce(tensor,
                    op=dist.ReduceOp.SUM,
                    group=parallel_info["current_sp_process_group"],
                    async_op=False)

    return tensor

class ReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "ReduceSum", input: Tensor) -> Tensor:
        return _reduce_sum(input)

    @staticmethod
    def backward(ctx: "ReduceSum", grad_output: Tensor) -> Tensor:
        return grad_output


def _reduce_scatter_sum(tensor: Tensor, dim: int = 1) -> Tensor:
    parallel_info = get_parallel_info()
    if parallel_info["sp_size"] == 1:
        return tensor

    # split input along dim into sp_size chunks
    split_size = divide(tensor.shape[dim], parallel_info["sp_size"])
    input_list = list(torch.split(tensor.contiguous(), split_size, dim=dim))

    # output shape equals the chunk shape along dim
    out_shape = list(tensor.shape)
    out_shape[dim] = split_size
    output = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)

    dist.reduce_scatter(
        output,
        input_list,
        op=dist.ReduceOp.SUM,
        group=parallel_info["current_sp_process_group"],
        async_op=False,
    )
    return output


def reduce_scatter_sum(input: Tensor, dim: int = 1) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        return ReduceScatterSum.apply(input, dim)
    return _reduce_scatter_sum(input, dim=dim)


class ReduceScatterSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "ReduceScatterSum", input: Tensor, dim: int = 1) -> Tensor:
        ctx.dim = dim
        return _reduce_scatter_sum(input, dim=dim)

    @staticmethod
    def backward(ctx: "ReduceScatterSum", grad_output: Tensor) -> Tuple[Tensor, None]:
        # Gradient of reduce_scatter(sum) is all_gather along the same dim
        grad_input = _gather(grad_output, dim=ctx.dim)
        return grad_input, None
