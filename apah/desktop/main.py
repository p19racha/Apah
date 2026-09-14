"""Apah Desktop Application Entrypoint — Background FastAPI Server + System Tray Icon."""

import argparse
import logging
import os
import sys
import threading
import time
import webbrowser
import uvicorn

from apah.engine.server import create_app, server_state
from apah.security.network_guard import enable_airgap_mode, is_airgap_enabled

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("apah.desktop.main")


class ApahDesktopApp:
    """Desktop Application manager wrapping background Uvicorn server and system tray icon."""

    def __init__(self, host: str = "127.0.0.1", port: int = 11500, auto_open: bool = True):
        self.host = host
        self.port = port
        self.auto_open = auto_open
        self.dashboard_url = f"http://{self.host}:{self.port}/dashboard"
        self.server: uvicorn.Server = None
        self.server_thread: threading.Thread = None
        self.tray_icon = None
        self._is_shutting_down = False

    def start_server_thread(self) -> None:
        """Start FastAPI server in a background daemon thread."""
        app = create_app()
        config = uvicorn.Config(app=app, host=self.host, port=self.port, log_level="info")
        self.server = uvicorn.Server(config=config)

        def _run():
            logger.info(f"Starting background Uvicorn server on http://{self.host}:{self.port}...")
            self.server.run()

        self.server_thread = threading.Thread(target=_run, daemon=True)
        self.server_thread.start()

    def open_dashboard(self) -> None:
        """Open local dashboard in system default browser."""
        logger.info(f"Opening dashboard URL: {self.dashboard_url}")
        try:
            webbrowser.open(self.dashboard_url)
        except Exception as e:
            logger.warning(f"Could not open default browser: {e}")

    def restart_server(self) -> None:
        """Restart background FastAPI server."""
        logger.info("Restarting server...")
        if self.server:
            self.server.should_exit = True
        time.sleep(1.0)
        self.start_server_thread()

    def quit(self) -> None:
        """Gracefully shut down server, unload GPU models, and exit application."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        logger.info("Shutting down Apah Desktop App...")

        # Stop system tray icon if running
        if self.tray_icon:
            try:
                self.tray_icon.stop()
            except Exception:
                pass

        # Unload GPU memory
        try:
            server_state.unload()
        except Exception as e:
            logger.warning(f"Error unloading runtime on quit: {e}")

        # Signal server loop to exit
        if self.server:
            self.server.should_exit = True

        logger.info("Apah Desktop App stopped cleanly.")
        sys.exit(0)

    def run(self) -> None:
        """Main application execution loop."""
        if is_airgap_enabled():
            enable_airgap_mode()

        self.start_server_thread()

        # Wait briefly for server startup
        time.sleep(1.2)

        if self.auto_open:
            self.open_dashboard()

        # Attempt to launch system tray icon
        try:
            from apah.desktop.tray_icon import setup_tray_menu

            self.tray_icon = setup_tray_menu(
                on_open_dashboard=self.open_dashboard,
                on_restart_server=self.restart_server,
                on_quit=self.quit,
                is_running=True,
            )
            logger.info("Launching System Tray Icon...")
            self.tray_icon.run()
        except Exception as e:
            logger.warning(f"Could not start GUI system tray icon ({e}). Running in headless server mode.")
            logger.info(f"Apah Dashboard is accessible at: {self.dashboard_url}")
            # Keep main thread alive in headless mode
            try:
                while not self._is_shutting_down:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.quit()


def main():
    parser = argparse.ArgumentParser(description="Apah Sovereign AI Desktop & Dashboard Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=11500, help="Port to bind (default: 11500)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser on launch")

    args = parser.parse_args()

    app = ApahDesktopApp(host=args.host, port=args.port, auto_open=not args.no_browser)
    app.run()


if __name__ == "__main__":
    main()
