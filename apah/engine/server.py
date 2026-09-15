"""Apah FastAPI server core module managing global runtime state, scheduler, NVML GPU monitor, and idle manager."""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import time
from typing import Optional
import torch
import uvicorn
from fastapi import FastAPI

from apah.engine.gpu_monitor import GPUMonitor
from apah.engine.idle_manager import IdleManager
from apah.engine.runtime import ApahRuntime
from apah.engine.scheduler import ContinuousBatchingScheduler
from apah.security.audit_log import AuditLogger
from apah.security.network_guard import (
    enable_airgap_mode,
    format_airgap_banner,
    is_airgap_enabled,
)
from apah.security.sandbox import format_sandbox_banner, get_path_guard

logger = logging.getLogger("apah.engine.server")


class ServerState:
    """Singleton state holder for loaded ApahRuntime instance, scheduler, GPU monitor, idle manager, and audit logger."""

    def __init__(self) -> None:
        self.runtime: Optional[ApahRuntime] = None
        self.model_name: Optional[str] = None
        self.load_time: Optional[float] = None
        self.scheduler: ContinuousBatchingScheduler = ContinuousBatchingScheduler()
        self.gpu_monitor: GPUMonitor = GPUMonitor()
        self.idle_manager: IdleManager = IdleManager(scheduler=self.scheduler)
        self.audit_logger: AuditLogger = AuditLogger()

    @property
    def is_loaded(self) -> bool:
        return self.runtime is not None and getattr(self.runtime, "model", None) is not None

    @property
    def uptime_seconds(self) -> float:
        if self.load_time is None or not self.is_loaded:
            return 0.0
        return time.time() - self.load_time

    @property
    def gpu_memory_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    def set_runtime(self, runtime: Optional[ApahRuntime], model_name: Optional[str]) -> None:
        """Set active runtime and update scheduler and idle manager."""
        self.runtime = runtime
        self.model_name = model_name
        self.load_time = time.time() if runtime is not None else None
        self.scheduler.set_runtime(runtime)
        if runtime is not None:
            self.scheduler.start()
            self.idle_manager.record_activity()
        else:
            self.scheduler.stop()

    def unload(self) -> None:
        """Unload runtime model, stop scheduler, and clear memory."""
        self.scheduler.stop()
        if self.runtime is not None:
            try:
                self.runtime.unload()
            except Exception as e:
                logger.warning(f"Error during runtime unload: {e}")
        self.runtime = None
        self.model_name = None
        self.load_time = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


server_state = ServerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context managing scheduler, GPU monitor, and idle manager background tasks."""
    logger.info("Initializing server lifespan...")
    if is_airgap_enabled():
        enable_airgap_mode()
    server_state.scheduler.start()
    server_state.idle_manager.start()
    yield
    logger.info("Shutting down server lifespan...")
    server_state.idle_manager.stop()
    server_state.scheduler.stop()
    server_state.gpu_monitor.shutdown()
    server_state.unload()


def print_startup_banner(audit_log_content: bool = False) -> None:
    """Print visible, loud server startup banner indicating security posture."""
    airgap_str = "ENABLED (Process-level socket isolation active)" if is_airgap_enabled() else "DISABLED (Standard network permitted)"
    content_str = "FULL CONTENT LOGGING" if audit_log_content else "METADATA ONLY (Data sensitive mode)"
    
    logger.info("================================================================================")
    logger.info("                 APAH SOVEREIGN ON-PREMISE AI WORKBENCH")
    logger.info("================================================================================")
    logger.info(f"  AIR-GAP MODE:      {airgap_str}")
    logger.info(f"  AUDIT LOGGING:     ENABLED ({content_str})")
    logger.info(f"  PATH SANDBOX:      ACTIVE (Allowed dirs: {[str(p) for p in get_path_guard().allowed_paths]})")
    logger.info("================================================================================")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from apah.api.routes import router
    from apah.compat.ollama_shim import ollama_router
    from apah.dashboard.backend.routes import dashboard_router, setup_dashboard_static

    app = FastAPI(
        title="Apah LLM Inference Server",
        version="0.5.0",
        description="FastAPI LLM inference server with continuous batching scheduler and GPU monitoring.",
        lifespan=lifespan,
    )
    app.include_router(router)
    app.include_router(ollama_router)
    app.include_router(dashboard_router)
    setup_dashboard_static(app)
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 11500,
    gpu_mem_fraction: float = 0.9,
    idle_timeout: int = 300,
    no_idle_unload: bool = False,
    audit_log_content: bool = False,
) -> None:
    """Start the Apah FastAPI Uvicorn server."""
    if os.environ.get("APAH_AIRGAP_MODE", "0").lower() in ("1", "true", "yes"):
        enable_airgap_mode()

    server_state.audit_logger.log_content = audit_log_content
    server_state.idle_manager.idle_timeout_seconds = idle_timeout
    server_state.idle_manager.enabled = not no_idle_unload

    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
            logger.info(f"Set CUDA process memory fraction cap to {gpu_mem_fraction}")
        except Exception as e:
            logger.warning(f"Could not set CUDA per-process memory fraction: {e}")

    print_startup_banner(audit_log_content=audit_log_content)

    app = create_app()
    logger.info(f"Starting Apah server on {host}:{port} (idle_timeout={idle_timeout}s, auto_unload={not no_idle_unload})...")
    uvicorn.run(app, host=host, port=port)

