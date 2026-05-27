from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
import math
from typing import Any
import torch.nn.functional as F

from cs336_basics.model import Embedding, Linear


def get_flashattention_autograd_function_pytorch() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyFlashAttnAutogradFunctionClass
    raise NotImplementedError


def get_flashattention_autograd_function_triton() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyTritonFlashAttentionAutogradFunctionClass
    raise NotImplementedError

class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

        for param in module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._sync_all_grads)

    def _sync_all_grads(self, param):
        param.grad /= dist.get_world_size()
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    return DDP(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    ddp_model.finish_gradient_synchronization()

class AllGatherWeight(torch.autograd.Function):
    """Gather row-sharded weights in forward; reduce-scatter their grads in backward."""

    @staticmethod
    def forward(ctx, shard: torch.Tensor, full_rows: int, master_dtype: torch.dtype) -> torch.Tensor:
        ctx.full_rows = full_rows
        ctx.master_dtype = master_dtype  # always fp32
        world_size = dist.get_world_size()
        shard_size = math.ceil(full_rows / world_size)

        pad_rows = shard_size - shard.shape[0]
        padded = F.pad(shard, (0, 0, 0, pad_rows)) if pad_rows > 0 else shard
        chunks = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(chunks, padded.contiguous())
        return torch.cat(chunks, dim=0)[:full_rows]

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        world_size = dist.get_world_size()
        shard_size = math.ceil(ctx.full_rows / world_size)

        grad = grad.to(ctx.master_dtype)

        pad_rows = world_size * shard_size - ctx.full_rows
        padded = F.pad(grad, (0, 0, 0, pad_rows)) if pad_rows > 0 else grad
        grad_shard = torch.empty(shard_size, *grad.shape[1:], dtype=grad.dtype, device=grad.device)
        dist.reduce_scatter(grad_shard, list(padded.chunk(world_size, dim=0)))
        grad_shard /= world_size
        return grad_shard, None, None


def _shard_weight(layer: nn.Module, compute_dtype: torch.dtype | None) -> tuple[nn.Parameter, int, torch.dtype | None]:
    full_rows = layer.weight.shape[0]
    shard_size = math.ceil(full_rows / dist.get_world_size())
    start = dist.get_rank() * shard_size
    end = min(start + shard_size, full_rows)
    shard = nn.Parameter(layer.weight.data[start:end].clone())
    return shard, full_rows, compute_dtype


class ShardedLinear(Linear):
    def __init__(self, layer: Linear, compute_dtype: torch.dtype | None):
        nn.Module.__init__(self)
        self.weight, self.full_rows, self.compute_dtype = _shard_weight(layer, compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(self.compute_dtype) if self.compute_dtype else self.weight
        full_weight = AllGatherWeight.apply(weight, self.full_rows, self.weight.dtype)
        return F.linear(x, full_weight)


class ShardedEmbedding(Embedding):
    def __init__(self, layer: Embedding, compute_dtype: torch.dtype | None):
        nn.Module.__init__(self)
        self.weight, self.full_rows, self.compute_dtype = _shard_weight(layer, compute_dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(self.compute_dtype) if self.compute_dtype else self.weight
        full_weight = AllGatherWeight.apply(weight, self.full_rows, self.weight.dtype)
        return full_weight[token_ids]


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self._replace_with_fsdp_layers(module, compute_dtype)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def _replace_with_fsdp_layers(self, module: nn.Module, compute_dtype: torch.dtype | None) -> None:
        for name, child in module.named_children():
            if isinstance(child, Linear):
                setattr(module, name, ShardedLinear(child, compute_dtype))
            elif isinstance(child, Embedding):
                setattr(module, name, ShardedEmbedding(child, compute_dtype))
            else:
                self._replace_with_fsdp_layers(child, compute_dtype)
    
    def finish_gradient_synchronization(self):
        for module in self.module.modules():
            if isinstance(module, (ShardedLinear, ShardedEmbedding)):
                continue
            for param in module.parameters(recurse=False):
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad /= dist.get_world_size()

def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    fully-sharded data parallel training, including weight sharding,
    all-gather for forward/backward, and gradient reduce-scatter.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with FSDP.
        compute_dtype: optional torch.dtype
            If provided, weights are cast to this dtype before communication
            and compute, saving bandwidth. Master weights stay in fp32.
    Returns:
        Instance of an FSDP class.
    """
    return FSDP(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the FSDP-wrapped model.
    """
    fsdp_model.finish_gradient_synchronization()

def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    All-gather sharded parameters from the FSDP model to reconstruct full
    parameter tensors. Replicated parameters are returned as-is.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
    Returns:
        State dictionary mapping parameter names to full (unsharded) tensors.
    """
    state_dict = {}
    for name, module in fsdp_model.module.named_modules():
        if isinstance(module, (ShardedLinear, ShardedEmbedding)):
            prefix = (name + '.') if name else ''
            state_dict[prefix + 'weight'] = AllGatherWeight.apply(
                module.weight.data, module.full_rows, module.weight.dtype
            )
    for name, param in fsdp_model.module.named_parameters():
        if name not in state_dict:
            state_dict[name] = param.data
    return state_dict

class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any) -> None:
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = kwargs
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._all_params: list[torch.Tensor] = []
        super().__init__(params, {})

    def _shard_size(self) -> int:
        return math.ceil(len(self._all_params) / self._world_size)

    def _owner_rank(self, param_idx: int) -> int:
        return min(param_idx // self._shard_size(), self._world_size - 1)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        params = list(param_group.get("params", []))
        super().add_param_group(param_group)
        self._all_params.extend(params)

        my_params = [
            p for i, p in enumerate(self._all_params)
            if self._owner_rank(i) == self._rank
        ]
        self._inner = self._optimizer_cls([{"params": my_params}], **self._optimizer_kwargs)

    def step(self, closure=None, **kwargs):
        loss = self._inner.step(closure, **kwargs)
        for i, p in enumerate(self._all_params):
            dist.broadcast(p.data, src=self._owner_rank(i))
        return loss


def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
