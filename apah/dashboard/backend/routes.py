"""Dashboard backend routes serving live WebSocket logs, real-time GPU/scheduler stats, and static frontend assets."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from apah.engine.server import server_state

logger = logging.getLogger("apah.dashboard.backend.routes")

dashboard_router = APIRouter()


@dashboard_router.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    """WebSocket streaming audit log events live (tail -f style)."""
    await websocket.accept()
    log_path = server_state.audit_logger.log_path

    try:
        # If log file exists, send initial tail (up to last 50 lines)
        sent_lines = 0
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    tail_lines = lines[-50:] if len(lines) > 50 else lines
                    for line in tail_lines:
                        line_str = line.strip()
                        if line_str:
                            await websocket.send_text(line_str)
                            sent_lines += 1
            except Exception as e:
                logger.warning(f"Error reading initial audit log lines: {e}")

        # Stream new log entries as they are written
        last_size = log_path.stat().st_size if log_path.exists() else 0

        while True:
            await asyncio.sleep(0.5)

            if not log_path.exists():
                continue

            current_size = log_path.stat().st_size
            if current_size > last_size:
                with open(log_path, "r", encoding="utf-8") as f:
                    f.seek(last_size)
                    new_lines = f.readlines()
                    last_size = f.tell()

                for line in new_lines:
                    line_str = line.strip()
                    if line_str:
                        await websocket.send_text(line_str)
            elif current_size < last_size:
                # Log file truncated or rotated
                last_size = 0

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected from /ws/logs")
    except Exception as e:
        logger.error(f"Error in /ws/logs stream: {e}")


@dashboard_router.websocket("/ws/stats")
async def ws_stats(websocket: WebSocket):
    """WebSocket pushing live GPU hardware stats and scheduler performance metrics every 1-2 seconds."""
    await websocket.accept()

    try:
        while True:
            gpu_stats = server_state.gpu_monitor.get_all_device_stats()
            sched_stats = server_state.scheduler.get_stats()
            tp_size = getattr(server_state.runtime, "tp_world_size", 1) if server_state.runtime else 1

            payload = {
                "gpu": [
                    {
                        "index": s.index,
                        "name": s.name,
                        "memory_used_mb": s.memory_used_mb,
                        "memory_total_mb": s.memory_total_mb,
                        "memory_free_mb": s.memory_free_mb,
                        "utilization_pct": s.utilization_pct,
                        "temperature_c": s.temperature_c,
                    }
                    for s in gpu_stats
                ],
                "scheduler": {
                    "active_batch_size": int(sched_stats.get("active_batch_size", 0)),
                    "waiting_queue_length": int(sched_stats.get("waiting_queue_length", 0)),
                    "aggregate_throughput_tok_s": float(sched_stats.get("aggregate_throughput_tok_s", 0.0)),
                    "total_tokens_generated": int(sched_stats.get("total_tokens_generated", 0)),
                },
                "server": {
                    "model_name": server_state.model_name,
                    "is_loaded": server_state.is_loaded,
                    "gpu_memory_mb": round(server_state.gpu_memory_mb, 2),
                    "uptime_seconds": round(server_state.uptime_seconds, 1),
                    "idle_seconds": round(server_state.idle_manager.get_idle_seconds(), 1),
                    "tp_world_size": tp_size,
                    "auto_unload_enabled": server_state.idle_manager.enabled,
                    "idle_timeout_seconds": server_state.idle_manager.idle_timeout_seconds,
                },
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected from /ws/stats")
    except Exception as e:
        logger.error(f"Error in /ws/stats stream: {e}")


def get_frontend_dir() -> Path:
    """Return path to frontend static directory."""
    return Path(__file__).resolve().parent.parent / "frontend"


def setup_dashboard_static(app: FastAPI) -> None:
    """Mount dashboard frontend static files and root dashboard endpoints."""
    frontend_dir = get_frontend_dir()

    if not frontend_dir.exists():
        logger.warning(f"Dashboard frontend directory not found at {frontend_dir}")
        return

    index_html = frontend_dir / "index.html"

    @app.get("/dashboard", response_class=FileResponse, include_in_schema=False)
    async def dashboard_index():
        if index_html.exists():
            return FileResponse(str(index_html), media_type="text/html")
        return RedirectResponse(url="/dashboard/")

    app.mount("/dashboard", StaticFiles(directory=str(frontend_dir), html=True), name="dashboard_static")
