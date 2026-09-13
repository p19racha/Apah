"""Tensor parallel ColumnParallelLinear and RowParallelLinear layer modules and model sharding logic."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from apah.engine.parallel.process_group import get_tp_rank, get_tp_world_size, is_tp_initialized


def shard_attention_heads(num_heads: int, tp_world_size: int) -> int:
    """Validate and compute number of attention heads per GPU.

    Args:
        num_heads: Total number of attention heads across model.
        tp_world_size: Tensor parallel world size.

    Returns:
        Number of attention heads assigned per GPU rank.

    Raises:
        ValueError: If num_heads is not evenly divisible by tp_world_size.
    """
    if tp_world_size <= 1:
        return num_heads
    if num_heads % tp_world_size != 0:
        raise ValueError(
            f"Number of attention heads ({num_heads}) is not evenly divisible by "
            f"tensor parallel world size ({tp_world_size})."
        )
    return num_heads // tp_world_size


class ColumnParallelLinear(nn.Module):
    """Linear layer with output features sharded across tensor parallel GPUs (QKV, MLP up/gate)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_world_size: Optional[int] = None,
        tp_rank: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_world_size = tp_world_size or get_tp_world_size()
        self.tp_rank = tp_rank if tp_rank is not None else get_tp_rank()

        if self.out_features % self.tp_world_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be evenly divisible by tp_world_size ({self.tp_world_size})."
            )

        self.out_features_per_partition = self.out_features // self.tp_world_size

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, self.in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local matmul without cross-GPU communication."""
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Linear layer with input features sharded across GPUs (Attention o_proj, MLP down_proj).

    Requires an all-reduce operation across GPUs after the local matmul to sum partial results.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_world_size: Optional[int] = None,
        tp_rank: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_world_size = tp_world_size or get_tp_world_size()
        self.tp_rank = tp_rank if tp_rank is not None else get_tp_rank()

        if self.in_features % self.tp_world_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be evenly divisible by tp_world_size ({self.tp_world_size})."
            )

        self.in_features_per_partition = self.in_features // self.tp_world_size

        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features_per_partition)
        )
        if bias and self.tp_rank == 0:
            # Bias is only added on rank 0 to prevent multi-counting during all-reduce
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local matmul and perform all-reduce across tensor parallel ranks."""
        local_output = F.linear(x, self.weight)

        if is_tp_initialized() and self.tp_world_size > 1:
            dist.all_reduce(local_output, op=dist.ReduceOp.SUM)

        if self.bias is not None:
            local_output += self.bias

        return local_output


def convert_model_to_tensor_parallel(
    model: nn.Module,
    tp_world_size: int,
    tp_rank: int,
) -> nn.Module:
    """Walk a Hugging Face Causal LM and replace standard Linear layers with Column/RowParallel equivalents.

    Note on PagedAttention:
        Each GPU rank process initializes its own isolated PagedKVCache block pool with sharded head counts
        (num_heads_per_gpu = num_heads // tp_world_size, num_kv_heads_per_gpu = num_kv_heads // tp_world_size).
        This guarantees that PagedAttention memory management is per-GPU aware across the tensor parallel group.

    Args:
        model: Loaded Hugging Face PreTrainedModel.
        tp_world_size: Tensor parallel world size.
        tp_rank: Rank index of current process.

    Returns:
        Tensor-parallel sharded model module.
    """
    if tp_world_size <= 1:
        return model

    # Locate transformer layers list
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        return model

    for layer_idx, layer in enumerate(layers):
        # 1. Shard Attention Module (q_proj, k_proj, v_proj -> ColumnParallel; o_proj -> RowParallel)
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            # Update head counts on attention module
            if hasattr(attn, "num_heads"):
                attn.num_heads = shard_attention_heads(attn.num_heads, tp_world_size)
            if hasattr(attn, "num_key_value_heads"):
                attn.num_key_value_heads = shard_attention_heads(attn.num_key_value_heads, tp_world_size)
            if hasattr(attn, "hidden_size") and hasattr(attn, "num_heads"):
                attn.hidden_size = attn.num_heads * getattr(attn, "head_dim", attn.hidden_size // (attn.num_heads * tp_world_size))

            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                orig_proj = getattr(attn, proj_name, None)
                if isinstance(orig_proj, nn.Linear):
                    col_linear = ColumnParallelLinear(
                        in_features=orig_proj.in_features,
                        out_features=orig_proj.out_features,
                        bias=orig_proj.bias is not None,
                        tp_world_size=tp_world_size,
                        tp_rank=tp_rank,
                    )
                    # Shard weight along output dimension (dim 0)
                    out_per_gpu = orig_proj.out_features // tp_world_size
                    shard_w = orig_proj.weight.data[tp_rank * out_per_gpu : (tp_rank + 1) * out_per_gpu, :].clone()
                    col_linear.weight.data.copy_(shard_w)
                    if orig_proj.bias is not None:
                        shard_b = orig_proj.bias.data[tp_rank * out_per_gpu : (tp_rank + 1) * out_per_gpu].clone()
                        col_linear.bias.data.copy_(shard_b)
                    setattr(attn, proj_name, col_linear)

            o_proj = getattr(attn, "o_proj", None)
            if isinstance(o_proj, nn.Linear):
                row_linear = RowParallelLinear(
                    in_features=o_proj.in_features,
                    out_features=o_proj.out_features,
                    bias=o_proj.bias is not None,
                    tp_world_size=tp_world_size,
                    tp_rank=tp_rank,
                )
                # Shard weight along input dimension (dim 1)
                in_per_gpu = o_proj.in_features // tp_world_size
                shard_w = o_proj.weight.data[:, tp_rank * in_per_gpu : (tp_rank + 1) * in_per_gpu].clone()
                row_linear.weight.data.copy_(shard_w)
                if o_proj.bias is not None and tp_rank == 0:
                    row_linear.bias.data.copy_(o_proj.bias.data.clone())
                setattr(attn, "o_proj", row_linear)

        # 2. Shard MLP Module (gate_proj, up_proj -> ColumnParallel; down_proj -> RowParallel)
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for proj_name in ["gate_proj", "up_proj"]:
                orig_proj = getattr(mlp, proj_name, None)
                if isinstance(orig_proj, nn.Linear):
                    col_linear = ColumnParallelLinear(
                        in_features=orig_proj.in_features,
                        out_features=orig_proj.out_features,
                        bias=orig_proj.bias is not None,
                        tp_world_size=tp_world_size,
                        tp_rank=tp_rank,
                    )
                    out_per_gpu = orig_proj.out_features // tp_world_size
                    shard_w = orig_proj.weight.data[tp_rank * out_per_gpu : (tp_rank + 1) * out_per_gpu, :].clone()
                    col_linear.weight.data.copy_(shard_w)
                    if orig_proj.bias is not None:
                        shard_b = orig_proj.bias.data[tp_rank * out_per_gpu : (tp_rank + 1) * out_per_gpu].clone()
                        col_linear.bias.data.copy_(shard_b)
                    setattr(mlp, proj_name, col_linear)

            down_proj = getattr(mlp, "down_proj", None)
            if isinstance(down_proj, nn.Linear):
                row_linear = RowParallelLinear(
                    in_features=down_proj.in_features,
                    out_features=down_proj.out_features,
                    bias=down_proj.bias is not None,
                    tp_world_size=tp_world_size,
                    tp_rank=tp_rank,
                )
                in_per_gpu = down_proj.in_features // tp_world_size
                shard_w = down_proj.weight.data[:, tp_rank * in_per_gpu : (tp_rank + 1) * in_per_gpu].clone()
                row_linear.weight.data.copy_(shard_w)
                if down_proj.bias is not None and tp_rank == 0:
                    row_linear.bias.data.copy_(down_proj.bias.data.clone())
                setattr(mlp, "down_proj", row_linear)

    return model
