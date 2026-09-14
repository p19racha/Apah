"""Apah desktop standalone app entrypoint starting FastAPI server and system tray icon."""

import argparse
import logging
import os
import sys
import threading
import time
import webbrowser

import uvicorn

from apah.desktop.tray_icon import ApahTrayApp
from apah.engine.server import create_app, enable_airgap_mode, is_airgap_enabled, server_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("apah.desktop.main")


class DesktopServerManager:
    """Manages FastAPI uvicorn server thread lifecycle for desktop app."""

    def __init__(self, host: str = "127.0.0.1", port: int = 11500):
        self.host = host
        self.port = port
        self.server: uvicorn.Server = None
        self.server_thread: threading.Thread = None
        self.is_running = False

    def start(self):
        """Start uvicorn server in daemon background thread."""
        if self.is_running:
            return

        if is_airgap_enabled():
            enable_airgap_mode()

        app = create_app()
        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="info",
            loop="asyncio",
        )
        self.server = uvicorn.Server(config)

        def _run():
            logger.info(f"Starting desktop uvicorn server on http://{self.host}:{self.port}...")
            self.server.run()

        self.server_thread = threading.Thread(target=_run, daemon=True)
        self.server_thread.start()
        self.is_running = True

    def stop(self):
        """Gracefully stop uvicorn server and unload GPU memory."""
        if not self.is_running:
            return
        logger.info("Stopping desktop uvicorn server...")
        if self.server:
            self.server.should_exit = True
        server_state.unload()
        self.is_running = False

    def restart(self):
        """Restart desktop server instance."""
        logger.info("Restarting desktop uvicorn server...")
        self.stop()
        time.sleep(1)
        self.start()


def open_dashboard_in_browser(host: str = "127.0.0.1", port: int = 11500):
    """Open default system web browser to dashboard URL."""
    url = f"http://{host}:{port}/dashboard"
    logger.info(f"Opening dashboard in default browser: {url}")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="Apah LLM Inference Engine Desktop App")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind local server")
    parser.add_argument("--port", type=int, default=11500, help="Port to bind local server")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser on startup")
    args = parser.parse_args()

    # 1. Start Server Manager
    manager = DesktopServerManager(host=args.host, port=args.port)
    manager.start()

    # Wait briefly for server startup
    time.sleep(1.0)

    # 2. Auto-open dashboard in browser on launch unless disabled
    if not args.no_browser:
        open_dashboard_in_browser(host=args.host, port=args.port)

    # 3. Callbacks for tray menu
    def _open_dashboard():
        open_dashboard_in_browser(host=args.host, port=args.port)

    def _restart():
        manager.restart()

    def _quit():
        logger.info("Quit requested from tray icon. Freeing GPU memory and exiting...")
        manager.stop()
        sys.exit(0)

    # 4. Start system tray app (blocking UI loop)
    try:
        tray_app = ApahTrayApp(
            on_open_dashboard=_open_dashboard,
            on_restart_server=_restart,
            on_quit=_quit,
            host=args.host,
            port=args.port,
        )
        tray_app.run()
    except Exception as e:
        logger.warning(f"Could not initialize system tray icon ({e}). Running in headless mode.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            _quit()


if __name__ == "__main__":
    main()
