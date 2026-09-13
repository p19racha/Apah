"""Apah continuous batching scheduler using block-based PagedKV cache memory management."""

import asyncio
import logging
import math
import time
from typing import Dict, List, Optional
import torch

from apah.engine.attention import read_past_kv_from_paged_cache, write_past_kv_to_paged_cache
from apah.engine.paged_cache import DEFAULT_BLOCK_SIZE, BlockPool, BlockTable
from apah.engine.runtime import ApahRuntime
from apah.engine.sequence import Sequence, SequenceStatus

logger = logging.getLogger("apah.engine.scheduler")


class ContinuousBatchingScheduler:
    """Iteration-level continuous batching scheduler backed by PagedKV cache block tables."""

    def __init__(
        self,
        runtime: Optional[ApahRuntime] = None,
        max_batch_size: int = 8,
        block_size: int = DEFAULT_BLOCK_SIZE,
        gpu_mem_fraction: float = 0.2,
    ) -> None:
        """Initialize ContinuousBatchingScheduler with PagedKV cache.

        Args:
            runtime: ApahRuntime instance wrapping model and tokenizer.
            max_batch_size: Maximum concurrent active sequences in batch.
            block_size: Number of tokens per physical memory block.
            gpu_mem_fraction: Memory budget fraction for block pool pre-allocation.
        """
        self.runtime: Optional[ApahRuntime] = None
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.gpu_mem_fraction = gpu_mem_fraction

        self.block_pool: Optional[BlockPool] = None
        self.waiting_queue: List[Sequence] = []
        self.active_batch: List[Sequence] = []

        self._is_running: bool = False
        self._loop_task: Optional[asyncio.Task] = None

        self.total_tokens_generated: int = 0
        self._token_timestamps: List[float] = []

        if runtime is not None:
            self.set_runtime(runtime)

    def set_runtime(self, runtime: Optional[ApahRuntime]) -> None:
        """Set active runtime and pre-allocate BlockPool."""
        self.runtime = runtime
        if runtime is not None and getattr(runtime, "model", None) is not None:
            config = runtime.model.config
            dtype = runtime.model.dtype if hasattr(runtime.model, "dtype") else torch.float16
            device = str(runtime.device)
            self.block_pool = BlockPool.create_from_model_config(
                config=config,
                gpu_mem_fraction=self.gpu_mem_fraction,
                block_size=self.block_size,
                dtype=dtype,
                device=device,
            )
        else:
            self.block_pool = None

    def add_request(self, sequence: Sequence) -> None:
        """Enqueue a new sequence request into the waiting queue."""
        sequence.status = SequenceStatus.WAITING
        self.waiting_queue.append(sequence)
        logger.debug(f"Enqueued sequence '{sequence.seq_id}' into waiting queue (len={len(self.waiting_queue)})")

    @property
    def active_count(self) -> int:
        """Number of active sequences currently in execution batch."""
        return len(self.active_batch)

    @property
    def waiting_count(self) -> int:
        """Number of sequences waiting in queue."""
        return len(self.waiting_queue)

    @property
    def throughput_tok_s(self) -> float:
        """Calculate aggregate generation throughput (tokens/sec) over recent window."""
        now = time.time()
        self._token_timestamps = [t for t in self._token_timestamps if now - t <= 10.0]
        if len(self._token_timestamps) < 2:
            return 0.0
        elapsed = now - self._token_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return len(self._token_timestamps) / elapsed

    def get_stats(self) -> Dict[str, float]:
        """Get telemetry statistics including PagedCache block usage and fragmentation."""
        num_total = float(self.block_pool.num_total_blocks()) if self.block_pool else 0.0
        num_free = float(self.block_pool.num_free_blocks()) if self.block_pool else 0.0
        num_alloc = num_total - num_free

        frag_tokens = 0
        for seq in self.active_batch:
            if seq.block_table is not None:
                frag_tokens += seq.block_table.get_fragmentation()

        return {
            "active_batch_size": float(self.active_count),
            "waiting_queue_length": float(self.waiting_count),
            "aggregate_throughput_tok_s": round(self.throughput_tok_s, 2),
            "total_tokens_generated": float(self.total_tokens_generated),
            "total_blocks": num_total,
            "free_blocks": num_free,
            "allocated_blocks": num_alloc,
            "fragmentation_tokens": float(frag_tokens),
        }

    def _admit_waiting_sequences(self) -> None:
        """Admit WAITING sequences into active batch if sufficient free blocks are available in BlockPool."""
        if not self.waiting_queue or self.block_pool is None:
            return

        while len(self.active_batch) < self.max_batch_size and self.waiting_queue:
            seq = self.waiting_queue[0]
            needed_blocks = math.ceil(len(seq.prompt_token_ids) / self.block_size)

            if self.block_pool.num_free_blocks() < needed_blocks:
                logger.warning(
                    f"Insufficient free blocks ({self.block_pool.num_free_blocks()}) "
                    f"to admit sequence '{seq.seq_id}' (requires {needed_blocks} blocks). Holding admission."
                )
                break

            seq = self.waiting_queue.pop(0)
            seq.block_table = BlockTable(block_pool=self.block_pool, block_size=self.block_size)
            seq.block_table.allocate_prompt_blocks(len(seq.prompt_token_ids))

            seq.status = SequenceStatus.RUNNING
            self.active_batch.append(seq)
            logger.info(f"Admitted sequence '{seq.seq_id}' to active batch with {needed_blocks} blocks.")

    async def _step_batch(self) -> None:
        """Execute one iteration-level decode/prefill step using PagedKV cache."""
        if not self.active_batch or self.runtime is None or self.runtime.model is None or self.block_pool is None:
            return

        device = self.runtime.device
        tokenizer = self.runtime.tokenizer
        finished_sequences: List[Sequence] = []

        for seq in list(self.active_batch):
            try:
                if seq.next_token_id is None:
                    # --- Prefill Phase ---
                    prompt_ids = torch.tensor([seq.prompt_token_ids], dtype=torch.long, device=device)
                    seq_len = prompt_ids.shape[1]
                    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

                    with torch.no_grad():
                        outputs = self.runtime.model(
                            input_ids=prompt_ids,
                            attention_mask=attention_mask,
                            use_cache=True,
                        )

                    # Write prefill prompt KV cache into paged block storage
                    write_past_kv_to_paged_cache(self.block_pool, seq.block_table, outputs.past_key_values, is_prefill=True)

                    logits = outputs.logits[:, -1, :]
                    next_token_tensor = self.runtime._sample_next_token(logits, seq.temperature, seq.do_sample)
                    seq.next_token_id = next_token_tensor

                else:
                    # --- Decode Step ---
                    # 1. Advance token count and allocate new physical block if needed
                    seq.block_table.append_token()

                    # 2. Gather K/V cache from non-contiguous paged block storage
                    past_kv = read_past_kv_from_paged_cache(self.block_pool, seq.block_table)

                    # 3. Model forward pass for new decode token
                    current_seq_len = seq.block_table.num_tokens
                    attention_mask = torch.ones((1, current_seq_len), dtype=torch.long, device=device)

                    with torch.no_grad():
                        outputs = self.runtime.model(
                            input_ids=seq.next_token_id,
                            attention_mask=attention_mask,
                            past_key_values=past_kv,
                            use_cache=True,
                        )

                    # 4. Write new token's K/V into paged block storage
                    write_past_kv_to_paged_cache(self.block_pool, seq.block_table, outputs.past_key_values, is_prefill=False)

                    logits = outputs.logits[:, -1, :]
                    next_token_tensor = self.runtime._sample_next_token(logits, seq.temperature, seq.do_sample)
                    seq.next_token_id = next_token_tensor

                # Process output token
                token_val = int(seq.next_token_id.item())
                seq.output_token_ids.append(token_val)

                now = time.time()
                self.total_tokens_generated += 1
                self._token_timestamps.append(now)

                current_decoded = tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
                new_chunk = current_decoded[len(seq.accumulated_text) :]
                seq.accumulated_text = current_decoded

                if new_chunk:
                    seq.token_queue.put_nowait(new_chunk)

                if seq.is_finished():
                    finished_sequences.append(seq)

            except Exception as e:
                logger.error(f"Error executing step for sequence '{seq.seq_id}': {e}", exc_info=True)
                seq.mark_finished(error=e)
                finished_sequences.append(seq)

        # Evict finished sequences and free their physical blocks back to pool
        for seq in finished_sequences:
            if seq in self.active_batch:
                self.active_batch.remove(seq)
            seq.mark_finished()  # Triggers block_table.free()
            logger.info(f"Evicted finished sequence '{seq.seq_id}' and returned blocks to BlockPool.")

    async def run_loop(self) -> None:
        """Continuous background execution loop."""
        logger.info("ContinuousBatchingScheduler background loop started.")
        self._is_running = True

        while self._is_running:
            try:
                if self.runtime is None or getattr(self.runtime, "model", None) is None:
                    await asyncio.sleep(0.05)
                    continue

                self._admit_waiting_sequences()

                if not self.active_batch:
                    await asyncio.sleep(0.005)
                    continue

                await self._step_batch()
                await asyncio.sleep(0.001)

            except asyncio.CancelledError:
                logger.info("ContinuousBatchingScheduler loop cancelled.")
                self._is_running = False
                break
            except Exception as e:
                logger.error(f"Unexpected error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(0.01)

    def start(self) -> None:
        """Start the background scheduler loop if not running."""
        if not self._is_running or self._loop_task is None or self._loop_task.done():
            self._is_running = True
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            self._loop_task = loop.create_task(self.run_loop())

    def stop(self) -> None:
        """Stop the scheduler loop and clean up all sequences and physical blocks."""
        self._is_running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            self._loop_task = None

        for seq in list(self.waiting_queue) + list(self.active_batch):
            seq.mark_finished()

        self.waiting_queue.clear()
        self.active_batch.clear()
