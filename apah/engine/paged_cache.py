"""Paged KV cache memory manager containing PhysicalBlock, BlockPool, and BlockTable."""

import logging
import math
from typing import Any, List, Optional, Set, Tuple
import torch

logger = logging.getLogger("apah.engine.paged_cache")

DEFAULT_BLOCK_SIZE = 16


class BlockPool:
    """GPU memory pool managing pre-allocated fixed-size physical blocks."""

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        """Initialize BlockPool and pre-allocate block storage tensor on GPU.

        Layout choice:
            `block_storage` tensor shape:
            [num_blocks, num_layers, 2, num_kv_heads, block_size, head_dim]
            where index 0 along dim=2 is Key (K), and index 1 along dim=2 is Value (V).

        Args:
            num_blocks: Total number of physical blocks in the pool.
            num_layers: Number of transformer layers in model.
            num_kv_heads: Number of key/value heads per layer.
            head_dim: Dimensionality of each attention head.
            block_size: Number of token slots per block.
            dtype: PyTorch data type for KV tensors.
            device: Target execution device.
        """
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        # Pre-allocate contiguous GPU block storage tensor
        self.block_storage = torch.zeros(
            (num_blocks, num_layers, 2, num_kv_heads, block_size, head_dim),
            dtype=dtype,
            device=self.device,
        )

        # Track free and allocated block IDs
        self.free_blocks: List[int] = list(range(num_blocks))
        self.allocated_blocks: Set[int] = set()

    @classmethod
    def create_from_model_config(
        cls,
        config: Any,
        gpu_mem_fraction: float = 0.2,
        block_size: int = DEFAULT_BLOCK_SIZE,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> "BlockPool":
        """Compute block count from available memory budget and instantiate BlockPool."""
        num_layers = getattr(config, "num_hidden_layers", 32)
        num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 32))
        hidden_size = getattr(config, "hidden_size", 4096)
        num_attn_heads = getattr(config, "num_attention_heads", 32)
        head_dim = getattr(config, "head_dim", hidden_size // num_attn_heads)

        bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
        block_bytes = num_layers * 2 * num_kv_heads * block_size * head_dim * bytes_per_elem

        if "cuda" in device and torch.cuda.is_available():
            dev_idx = torch.device(device).index or 0
            free_mem_bytes, _ = torch.cuda.mem_get_info(dev_idx)
            budget_bytes = int(free_mem_bytes * gpu_mem_fraction)
            num_blocks = max(16, budget_bytes // block_bytes)
        else:
            num_blocks = 256  # Fallback for CPU / unit testing

        logger.info(
            f"Pre-allocated BlockPool: {num_blocks} blocks ({num_blocks * block_bytes / (1024**2):.2f} MB GPU memory) "
            f"with block_size={block_size}"
        )

        return cls(
            num_blocks=num_blocks,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )

    def allocate_block(self) -> int:
        """Allocate a single physical block from pool.

        Returns:
            Allocated block ID integer.

        Raises:
            MemoryError: If block pool is exhausted.
        """
        if not self.free_blocks:
            raise MemoryError(
                f"BlockPool exhausted: All {self.num_blocks} physical blocks are currently allocated."
            )
        block_id = self.free_blocks.pop(0)
        self.allocated_blocks.add(block_id)
        return block_id

    def free_block(self, block_id: int) -> None:
        """Return a physical block to the free pool."""
        if block_id in self.allocated_blocks:
            self.allocated_blocks.remove(block_id)
            self.free_blocks.append(block_id)
        else:
            logger.warning(f"Attempted to free unallocated or double-freed block_id={block_id}")

    def num_free_blocks(self) -> int:
        """Count of available unallocated blocks."""
        return len(self.free_blocks)

    def num_total_blocks(self) -> int:
        """Count of total blocks in pool."""
        return self.num_blocks


class BlockTable:
    """Per-sequence block table mapping logical blocks to physical BlockPool blocks."""

    def __init__(self, block_pool: BlockPool, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        """Initialize BlockTable for a sequence.

        Args:
            block_pool: Reference to global BlockPool.
            block_size: Tokens per block.
        """
        self.block_pool = block_pool
        self.block_size = block_size
        self.physical_block_ids: List[int] = []
        self.num_tokens: int = 0

    def allocate_prompt_blocks(self, prompt_len: int) -> None:
        """Allocate initial physical blocks required for prompt tokens."""
        needed_blocks = math.ceil(prompt_len / self.block_size)
        for _ in range(needed_blocks):
            block_id = self.block_pool.allocate_block()
            self.physical_block_ids.append(block_id)
        self.num_tokens = prompt_len

    def append_token(self) -> Tuple[int, int]:
        """Advance sequence token count and allocate a new physical block if current block is full.

        Returns:
            Tuple of (physical_block_id, offset_within_block).
        """
        self.num_tokens += 1
        block_index = (self.num_tokens - 1) // self.block_size
        offset = (self.num_tokens - 1) % self.block_size

        if block_index >= len(self.physical_block_ids):
            new_block_id = self.block_pool.allocate_block()
            self.physical_block_ids.append(new_block_id)

        target_block_id = self.physical_block_ids[block_index]
        return target_block_id, offset

    def get_physical_blocks(self) -> List[int]:
        """List of physical block IDs assigned to this sequence."""
        return self.physical_block_ids.copy()

    def get_fragmentation(self) -> int:
        """Return number of unused token slots in current last block."""
        if not self.physical_block_ids:
            return 0
        rem = self.num_tokens % self.block_size
        return (self.block_size - rem) if rem > 0 else 0

    def free(self) -> None:
        """Release all physical blocks owned by this block table back to pool."""
        for block_id in self.physical_block_ids:
            self.block_pool.free_block(block_id)
        self.physical_block_ids.clear()
        self.num_tokens = 0
