"""Idle manager for monitoring model inactivity and auto-unloading idle models from GPU memory."""

import asyncio
import logging
import time
from typing import Dict, Optional

from apah.engine.scheduler import ContinuousBatchingScheduler

logger = logging.getLogger("apah.engine.idle_manager")


class IdleManager:
    """Manager tracking model activity and auto-unloading idle models past configurable timeout."""

    def __init__(
        self,
        scheduler: ContinuousBatchingScheduler,
        idle_timeout_seconds: int = 300,
        enabled: bool = True,
        check_interval_seconds: float = 5.0,
    ):
        self.scheduler = scheduler
        self.idle_timeout_seconds = idle_timeout_seconds
        self.enabled = enabled
        self.check_interval_seconds = check_interval_seconds

        self._last_activity_time: float = time.time()
        self._is_running: bool = False
        self._loop_task: Optional[asyncio.Task] = None

    def record_activity(self) -> None:
        """Update last-activity timestamp for the currently active model."""
        self._last_activity_time = time.time()

    def get_idle_seconds(self) -> float:
        """Return elapsed seconds since last model activity."""
        return max(0.0, time.time() - self._last_activity_time)

    def get_idle_status(self) -> Dict[str, float]:
        """Return dict mapping current loaded model name to its idle duration in seconds."""
        from apah.engine.server import server_state
        if server_state.is_loaded and server_state.model_name is not None:
            return {server_state.model_name: self.get_idle_seconds()}
        return {}

    async def run_loop(self) -> None:
        """Background loop checking idle duration and triggering auto-unload when threshold is exceeded."""
        self._is_running = True
        logger.info(
            f"IdleManager background loop started (enabled={self.enabled}, "
            f"timeout={self.idle_timeout_seconds}s, check_interval={self.check_interval_seconds}s)."
        )

        while self._is_running:
            try:
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break

            if not self.enabled or self.idle_timeout_seconds <= 0:
                continue

            from apah.engine.server import server_state
            if not server_state.is_loaded or server_state.model_name is None:
                continue

            idle_duration = self.get_idle_seconds()
            if idle_duration >= self.idle_timeout_seconds:
                # Race condition guard: re-check active sequences and waiting queue atomically
                stats = self.scheduler.get_stats()
                active_seqs = int(stats.get("active_batch_size", 0))
                waiting_seqs = int(stats.get("waiting_queue_length", 0))

                if active_seqs == 0 and waiting_seqs == 0:
                    model_name = server_state.model_name
                    mem_before = server_state.gpu_memory_mb
                    logger.info(
                        f"Auto-unloading idle model '{model_name}'. "
                        f"Idle duration: {idle_duration:.1f}s >= threshold ({self.idle_timeout_seconds}s). "
                        f"Active requests: {active_seqs}, Waiting queue: {waiting_seqs}."
                    )
                    server_state.unload()
                    mem_after = server_state.gpu_memory_mb
                    freed_mb = max(0.0, mem_before - mem_after)
                    logger.info(
                        f"Successfully auto-unloaded '{model_name}'. GPU memory freed: {freed_mb:.2f} MB."
                    )

    def start(self) -> None:
        """Start IdleManager background task in active event loop."""
        if self._loop_task is None or self._loop_task.done():
            self._is_running = True
            try:
                loop = asyncio.get_running_loop()
                self._loop_task = loop.create_task(self.run_loop())
            except RuntimeError:
                pass

    def stop(self) -> None:
        """Stop IdleManager background task."""
        self._is_running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            self._loop_task = None
