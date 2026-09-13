"""Tests for PagedCache memory manager (BlockPool and BlockTable)."""

import math
import pytest
import torch
from apah.engine.paged_cache import DEFAULT_BLOCK_SIZE, BlockPool, BlockTable


def test_block_pool_allocation_and_deallocation():
    """Verify block pool allocation and freeing accounting."""
    pool = BlockPool(
        num_blocks=10,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        block_size=16,
        dtype=torch.float32,
        device="cpu",
    )

    assert pool.num_total_blocks() == 10
    assert pool.num_free_blocks() == 10

    b0 = pool.allocate_block()
    b1 = pool.allocate_block()
    b2 = pool.allocate_block()

    assert len({b0, b1, b2}) == 3
    assert pool.num_free_blocks() == 7

    pool.free_block(b1)
    assert pool.num_free_blocks() == 8

    pool.free_block(b0)
    pool.free_block(b2)
    assert pool.num_free_blocks() == 10


def test_block_pool_exhaustion():
    """Verify MemoryError is raised when BlockPool runs out of blocks."""
    pool = BlockPool(
        num_blocks=2,
        num_layers=1,
        num_kv_heads=2,
        head_dim=32,
        block_size=16,
        dtype=torch.float32,
        device="cpu",
    )

    _ = pool.allocate_block()
    _ = pool.allocate_block()

    with pytest.raises(MemoryError, match="BlockPool exhausted"):
        pool.allocate_block()


def test_block_table_growth_exact():
    """Verify BlockTable grows by exactly 1 block every BLOCK_SIZE tokens."""
    block_size = 16
    pool = BlockPool(
        num_blocks=20,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
    )

    table = BlockTable(block_pool=pool, block_size=block_size)

    # Initial prompt allocation of 30 tokens -> requires ceil(30/16) = 2 blocks
    table.allocate_prompt_blocks(30)
    assert len(table.get_physical_blocks()) == 2
    assert table.num_tokens == 30
    assert pool.num_free_blocks() == 18

    # Tokens 31 and 32 fill up the 2nd block
    table.append_token()  # 31
    assert len(table.get_physical_blocks()) == 2

    table.append_token()  # 32 (block 2 is full)
    assert len(table.get_physical_blocks()) == 2

    # Token 33 triggers allocation of the 3rd block
    table.append_token()  # 33
    assert len(table.get_physical_blocks()) == 3
    assert pool.num_free_blocks() == 17

    table.free()
    assert pool.num_free_blocks() == 20
    assert len(table.get_physical_blocks()) == 0


def test_stress_allocation_deallocation_50_sequences():
    """Stress test: allocate and free blocks across 50 simulated sequences of varying lengths."""
    pool = BlockPool(
        num_blocks=500,
        num_layers=4,
        num_kv_heads=8,
        head_dim=64,
        block_size=16,
        dtype=torch.float32,
        device="cpu",
    )

    assert pool.num_free_blocks() == 500

    tables = []
    for i in range(50):
        prompt_len = (i % 10 + 1) * 12
        table = BlockTable(block_pool=pool, block_size=16)
        table.allocate_prompt_blocks(prompt_len)

        for _ in range(15):
            table.append_token()

        tables.append(table)

    assert pool.num_free_blocks() < 500

    for table in tables:
        table.free()

    assert pool.num_free_blocks() == 500
    assert pool.num_total_blocks() == 500
