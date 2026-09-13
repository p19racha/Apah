"""Sequence data structure representing an in-flight generation request in Apah."""

import asyncio
from enum import Enum
import uuid
from typing import Any, List, Optional, Set, Tuple

from apah.engine.paged_cache import BlockTable


class SequenceStatus(str, Enum):
    """Execution status of a sequence."""

    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class Sequence:
    """Represents a single generation request managed by the continuous batching scheduler."""

    def __init__(
        self,
        prompt: str,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: Optional[bool] = None,
        stop_token_ids: Optional[Set[int]] = None,
        seq_id: Optional[str] = None,
    ) -> None:
        """Initialize Sequence.

        Args:
            prompt: Original text prompt.
            prompt_token_ids: Tokenized prompt ID list.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            do_sample: Explicit boolean for temperature sampling vs greedy decoding.
            stop_token_ids: Set of token IDs that trigger termination.
            seq_id: Unique string sequence identifier.
        """
        self.seq_id = seq_id if seq_id is not None else f"seq-{uuid.uuid4().hex[:10]}"
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.status = SequenceStatus.WAITING

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        if do_sample is None:
            self.do_sample = (temperature != 1.0) and (temperature > 0.0)
        else:
            self.do_sample = do_sample

        self.stop_token_ids: Set[int] = stop_token_ids if stop_token_ids is not None else set()

        # BlockTable reference for paged KV cache allocation
        self.block_table: Optional[BlockTable] = None

        # Next token ID tensor of shape (1, 1) for next decode step
        self.next_token_id: Optional[Any] = None

        # Text decoding state tracking for incremental streaming
        self.accumulated_text: str = ""

        # Queue for streaming text tokens back to HTTP client. Puts None when finished.
        self.token_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        # Future for completion notification
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        self.completion_future: asyncio.Future[str] = loop.create_future()

    def is_finished(self) -> bool:
        """Check if sequence has finished generation."""
        if self.status == SequenceStatus.FINISHED:
            return True

        if len(self.output_token_ids) >= self.max_new_tokens:
            return True

        if self.output_token_ids and self.output_token_ids[-1] in self.stop_token_ids:
            return True

        return False

    def mark_finished(self, error: Optional[Exception] = None) -> None:
        """Mark sequence as finished, free physical blocks, and notify completion handlers."""
        if self.status != SequenceStatus.FINISHED:
            self.status = SequenceStatus.FINISHED

            # Free block table blocks immediately back to BlockPool
            if self.block_table is not None:
                self.block_table.free()
                self.block_table = None

            self.next_token_id = None

            if error:
                if not self.completion_future.done():
                    self.completion_future.set_exception(error)
                self.token_queue.put_nowait(None)
            else:
                if not self.completion_future.done():
                    self.completion_future.set_result(self.accumulated_text)
                self.token_queue.put_nowait(None)
