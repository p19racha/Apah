"""PagedAttention compute wrapper for writing, reading, and performing attention over paged KV blocks."""

import logging
import math
from typing import Any, List, Tuple
import torch

from apah.engine.paged_cache import BlockPool, BlockTable

logger = logging.getLogger("apah.engine.attention")


def get_num_layers(past_key_values: Any) -> int:
    """Get total number of layers from past_key_values cache object or tuple."""
    if hasattr(past_key_values, "layers"):
        return len(past_key_values.layers)
    elif hasattr(past_key_values, "key_cache"):
        return len(past_key_values.key_cache)
    return len(past_key_values)


def extract_kv_pair_for_layer(past_key_values: Any, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract (key_tensor, value_tensor) for a given layer index."""
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values
    elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    else:
        return past_key_values[layer_idx][0], past_key_values[layer_idx][1]


def write_past_kv_to_paged_cache(
    block_pool: BlockPool,
    block_table: BlockTable,
    past_key_values: Any,
    is_prefill: bool = False,
) -> None:
    """Write Key and Value tensors into BlockPool storage using sequence's BlockTable.

    Args:
        block_pool: Reference to global BlockPool.
        block_table: Sequence's BlockTable.
        past_key_values: Hugging Face model output past_key_values cache.
        is_prefill: If True, writes full prompt sequence KV cache across blocks. If False, writes last decode token.
    """
    if past_key_values is None:
        return

    num_layers = get_num_layers(past_key_values)
    k_layer0, _ = extract_kv_pair_for_layer(past_key_values, 0)
    block_size = block_pool.block_size
    physical_blocks = block_table.get_physical_blocks()

    if is_prefill:
        # --- Prefill Prompt KV Write ---
        seq_len = k_layer0.shape[2]
        for token_idx in range(seq_len):
            block_idx = token_idx // block_size
            offset = token_idx % block_size
            p_block_id = physical_blocks[block_idx]

            for layer_i in range(num_layers):
                k_layer, v_layer = extract_kv_pair_for_layer(past_key_values, layer_i)
                k_t = k_layer[0, :, token_idx, :]  # [num_kv_heads, head_dim]
                v_t = v_layer[0, :, token_idx, :]  # [num_kv_heads, head_dim]

                block_pool.block_storage[p_block_id, layer_i, 0, :, offset, :] = k_t
                block_pool.block_storage[p_block_id, layer_i, 1, :, offset, :] = v_t

    else:
        # --- Decode Step KV Write ---
        token_idx = block_table.num_tokens - 1
        block_idx = token_idx // block_size
        offset = token_idx % block_size
        p_block_id = physical_blocks[block_idx]

        for layer_i in range(num_layers):
            k_layer, v_layer = extract_kv_pair_for_layer(past_key_values, layer_i)
            k_t = k_layer[0, :, -1, :]  # [num_kv_heads, head_dim]
            v_t = v_layer[0, :, -1, :]  # [num_kv_heads, head_dim]

            block_pool.block_storage[p_block_id, layer_i, 0, :, offset, :] = k_t
            block_pool.block_storage[p_block_id, layer_i, 1, :, offset, :] = v_t


def read_past_kv_from_paged_cache(
    block_pool: BlockPool,
    block_table: BlockTable,
) -> Any:
    """Gather non-contiguous KV blocks into Hugging Face compatible DynamicCache or past_key_values tuple.

    Args:
        block_pool: Reference to global BlockPool.
        block_table: Sequence's BlockTable.

    Returns:
        DynamicCache or past_key_values tuple containing gathered K and V tensors.
    """
    total_tokens = block_table.num_tokens
    if total_tokens == 0:
        return ()

    physical_blocks = block_table.get_physical_blocks()
    block_size = block_pool.block_size
    num_layers = block_pool.num_layers

    gathered_k_list = []
    gathered_v_list = []

    for p_block_id in physical_blocks:
        k_b = block_pool.block_storage[p_block_id, :, 0, :, :, :]  # [num_layers, num_kv_heads, block_size, head_dim]
        v_b = block_pool.block_storage[p_block_id, :, 1, :, :, :]  # [num_layers, num_kv_heads, block_size, head_dim]
        gathered_k_list.append(k_b)
        gathered_v_list.append(v_b)

    full_k = torch.cat(gathered_k_list, dim=2)  # [num_layers, num_kv_heads, num_blocks*block_size, head_dim]
    full_v = torch.cat(gathered_v_list, dim=2)

    valid_k = full_k[:, :, :total_tokens, :]
    valid_v = full_v[:, :, :total_tokens, :]

    try:
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for layer_i in range(num_layers):
            k_l = valid_k[layer_i].unsqueeze(0)  # [1, num_kv_heads, total_tokens, head_dim]
            v_l = valid_v[layer_i].unsqueeze(0)  # [1, num_kv_heads, total_tokens, head_dim]
            cache.update(k_l, v_l, layer_i)
        return cache
    except Exception:
        past_kv = []
        for layer_i in range(num_layers):
            k_l = valid_k[layer_i].unsqueeze(0)
            v_l = valid_v[layer_i].unsqueeze(0)
            past_kv.append((k_l, v_l))
        return tuple(past_kv)


def paged_attention_forward(
    query: torch.Tensor,
    block_table: BlockTable,
    block_pool: BlockPool,
    layer_idx: int,
) -> torch.Tensor:
    """Pure PyTorch gather-based PagedAttention forward pass for a sequence.

    Note:
        This pure PyTorch gather-based implementation gathers non-contiguous physical
        blocks into a temporary tensor for scaled dot-product attention to verify
        correctness and memory savings. A fused Triton/CUDA kernel is a future optimization (Phase 4b).

    Args:
        query: Query tensor of shape [batch=1, num_heads, q_len, head_dim].
        block_table: Sequence's BlockTable.
        block_pool: Global BlockPool.
        layer_idx: Layer index to process.

    Returns:
        Attention output tensor of shape [batch=1, num_heads, q_len, head_dim].
    """
    total_tokens = block_table.num_tokens
    physical_blocks = block_table.get_physical_blocks()

    k_blocks = [block_pool.block_storage[p_id, layer_idx, 0, :, :, :] for p_id in physical_blocks]
    v_blocks = [block_pool.block_storage[p_id, layer_idx, 1, :, :, :] for p_id in physical_blocks]

    full_k = torch.cat(k_blocks, dim=1)[:, :total_tokens, :]  # [num_kv_heads, total_tokens, head_dim]
    full_v = torch.cat(v_blocks, dim=1)[:, :total_tokens, :]  # [num_kv_heads, total_tokens, head_dim]

    k = full_k.unsqueeze(0)  # [1, num_kv_heads, total_tokens, head_dim]
    v = full_v.unsqueeze(0)  # [1, num_kv_heads, total_tokens, head_dim]

    # Support Grouped Query Attention (GQA) where query heads > key/value heads
    num_queries_per_kv = query.shape[1] // k.shape[1]
    if num_queries_per_kv > 1:
        k = k.repeat_interleave(num_queries_per_kv, dim=1)
        v = v.repeat_interleave(num_queries_per_kv, dim=1)

    attn_output = torch.nn.functional.scaled_dot_product_attention(query, k, v)
    return attn_output
