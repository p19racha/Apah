"""Unit tests for tensor parallel sharding, ColumnParallelLinear, and RowParallelLinear modules."""

import pytest
import torch
import torch.nn as nn

from apah.engine.parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    convert_model_to_tensor_parallel,
    shard_attention_heads,
)


def test_shard_attention_heads_validation():
    """Test shard_attention_heads validation logic for divisible and non-divisible world sizes."""
    assert shard_attention_heads(32, 1) == 32
    assert shard_attention_heads(32, 2) == 16
    assert shard_attention_heads(32, 4) == 8
    assert shard_attention_heads(16, 8) == 2

    with pytest.raises(ValueError) as exc_info:
        shard_attention_heads(32, 3)

    assert "is not evenly divisible" in str(exc_info.value)


def test_column_parallel_linear_shape_and_forward():
    """Test ColumnParallelLinear weight partitioning and forward pass output shape."""
    in_features = 128
    out_features = 256
    tp_world_size = 2
    tp_rank = 0

    col_layer = ColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        tp_world_size=tp_world_size,
        tp_rank=tp_rank,
    )

    # Shard out_features should be 256 // 2 = 128
    assert col_layer.out_features_per_partition == 128
    assert col_layer.weight.shape == (128, 128)
    assert col_layer.bias.shape == (128,)

    x = torch.randn(4, in_features)
    out = col_layer(x)
    assert out.shape == (4, 128)


def test_row_parallel_linear_shape_and_forward():
    """Test RowParallelLinear weight partitioning and forward pass output shape."""
    in_features = 256
    out_features = 128
    tp_world_size = 2
    tp_rank = 0

    row_layer = RowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        tp_world_size=tp_world_size,
        tp_rank=tp_rank,
    )

    # Shard in_features should be 256 // 2 = 128
    assert row_layer.in_features_per_partition == 128
    assert row_layer.weight.shape == (128, 128)
    assert row_layer.bias.shape == (128,)

    x = torch.randn(4, 128)  # Input partitioned across GPUs
    out = row_layer(x)
    assert out.shape == (4, 128)


class DummyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 16
        self.num_key_value_heads = 16
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)


class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(64, 128)
        self.up_proj = nn.Linear(64, 128)
        self.down_proj = nn.Linear(128, 64)


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttn()
        self.mlp = DummyMLP()


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyLayer()])


def test_convert_model_to_tensor_parallel():
    """Test convert_model_to_tensor_parallel converts Linear layers to Column/Row parallel."""
    model = DummyModel()
    sharded = convert_model_to_tensor_parallel(model, tp_world_size=2, tp_rank=0)

    layer0 = sharded.model.layers[0]
    attn = layer0.self_attn
    mlp = layer0.mlp

    assert isinstance(attn.q_proj, ColumnParallelLinear)
    assert isinstance(attn.k_proj, ColumnParallelLinear)
    assert isinstance(attn.v_proj, ColumnParallelLinear)
    assert isinstance(attn.o_proj, RowParallelLinear)

    assert isinstance(mlp.gate_proj, ColumnParallelLinear)
    assert isinstance(mlp.up_proj, ColumnParallelLinear)
    assert isinstance(mlp.down_proj, RowParallelLinear)

    assert attn.num_heads == 8
    assert attn.num_key_value_heads == 8
