"""Dashboard routes for static file serving and WebSockets (logs live-tailing, GPU/scheduler stats)."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from apah.engine.server import server_state

logger = logging.getLogger("apah.dashboard.backend.routes")

dashboard_router = APIRouter()

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@dashboard_router.get("/dashboard", response_class=HTMLResponse, summary="Serve Apah Dashboard index.html")
@dashboard_router.get("/dashboard/", response_class=HTMLResponse, summary="Serve Apah Dashboard index.html")
async def get_dashboard_index():
    """Serve local web control dashboard index.html page."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h3>Dashboard frontend index.html not found.</h3>", status_code=404)
    return FileResponse(index_path)


@dashboard_router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint streaming live-tail audit log lines."""
    await websocket.accept()
    logger.info("WebSocket /ws/logs client connected.")
    try:
        log_dir = server_state.audit_logger.log_dir

        # 1. Send recent existing log lines on initial connection
        initial_lines: List[str] = []
        if log_dir.exists():
            log_files = sorted(log_dir.glob("apah_audit_*.jsonl"))
            for lf in log_files:
                try:
                    with open(lf, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        initial_lines.extend(lines)
                except Exception:
                    pass

        # Send last 50 lines to client
        recent_lines = initial_lines[-50:]
        for line in recent_lines:
            await websocket.send_text(line)

        # 2. Live tail loop
        last_file: Optional[Path] = None
        last_pos: int = 0

        if log_dir.exists():
            log_files = sorted(log_dir.glob("apah_audit_*.jsonl"))
            if log_files:
                last_file = log_files[-1]
                last_pos = last_file.stat().st_size

        while True:
            await asyncio.sleep(0.5)
            log_files = sorted(log_dir.glob("apah_audit_*.jsonl")) if log_dir.exists() else []
            if not log_files:
                continue

            current_file = log_files[-1]
            if last_file != current_file:
                last_file = current_file
                last_pos = 0

            if current_file.exists():
                curr_size = current_file.stat().st_size
                if curr_size > last_pos:
                    with open(current_file, "r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        new_text = f.read()
                        last_pos = f.tell()
                        for line in new_text.splitlines():
                            line_str = line.strip()
                            if line_str:
                                await websocket.send_text(line_str)
                elif curr_size < last_pos:
                    # File was truncated or rotated
                    last_pos = 0
    except WebSocketDisconnect:
        logger.info("WebSocket /ws/logs client disconnected.")
    except Exception as e:
        logger.warning(f"WebSocket /ws/logs error: {e}")


@dashboard_router.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    """WebSocket endpoint pushing real-time GPU and continuous batching scheduler stats every 1-2 seconds."""
    await websocket.accept()
    logger.info("WebSocket /ws/stats client connected.")
    try:
        while True:
            gpu_stats = [
                {
                    "index": s.index,
                    "name": s.name,
                    "memory_used_mb": s.memory_used_mb,
                    "memory_total_mb": s.memory_total_mb,
                    "memory_free_mb": s.memory_free_mb,
                    "utilization_pct": s.utilization_pct,
                    "temperature_c": s.temperature_c,
                }
                for s in server_state.gpu_monitor.get_all_device_stats()
            ]

            scheduler_stats = server_state.scheduler.get_stats()
            tp_size = getattr(server_state.runtime, "tp_world_size", 1) if server_state.is_loaded else 1
            idle_sec = server_state.idle_manager.get_idle_seconds() if server_state.is_loaded else 0.0

            payload = {
                "gpu": gpu_stats,
                "scheduler": scheduler_stats,
                "is_loaded": server_state.is_loaded,
                "model_name": server_state.model_name,
                "uptime_seconds": server_state.uptime_seconds,
                "idle_seconds": idle_sec,
                "gpu_memory_mb": server_state.gpu_memory_mb,
                "tp_world_size": tp_size,
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        logger.info("WebSocket /ws/stats client disconnected.")
    except Exception as e:
        logger.warning(f"WebSocket /ws/stats error: {e}")
