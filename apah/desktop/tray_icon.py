"""System tray icon generator and menu builder using pystray and Pillow."""

import logging
from typing import Callable, Optional
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("apah.desktop.tray_icon")

# Safe dynamic import of pystray to support headless/WSL environments without X11 display
pystray = None
PYSTRAY_IMPORT_ERROR: Optional[str] = None

try:
    import pystray
except Exception as e:
    PYSTRAY_IMPORT_ERROR = str(e)


def create_tray_icon_image(size: int = 64) -> Image.Image:
    """Generate a clean, high-DPI system tray icon image with Apah branding."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw rounded circle background in Apah Primary Blue (#2563eb)
    draw.ellipse((4, 4, size - 4, size - 4), fill=(37, 99, 235, 255))

    # Draw 'A' letter in white
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=int(size * 0.55))
    except Exception:
        font = ImageFont.load_default()

    # Draw white capital 'A'
    draw.text((size // 2, size // 2), "A", fill=(255, 255, 255, 255), font=font, anchor="mm")
    return img


class ApahTrayApp:
    """System tray app manager providing desktop menu and browser control."""

    def __init__(
        self,
        on_open_dashboard: Callable[[], None],
        on_restart_server: Callable[[], None],
        on_quit: Callable[[], None],
        host: str = "127.0.0.1",
        port: int = 11500,
    ):
        self.on_open_dashboard = on_open_dashboard
        self.on_restart_server = on_restart_server
        self.on_quit = on_quit
        self.host = host
        self.port = port
        self.icon = None
        self.server_status = "Running"

    def _build_menu(self):
        if pystray is None:
            raise RuntimeError(f"pystray is not available: {PYSTRAY_IMPORT_ERROR}")

        return pystray.Menu(
            pystray.MenuItem(
                "Open Dashboard",
                self._handle_open_dashboard,
                default=True,
            ),
            pystray.MenuItem(
                f"Server: {self.server_status} ({self.port})",
                action=lambda icon, item: None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Restart Server",
                self._handle_restart_server,
            ),
            pystray.MenuItem(
                "Quit Apah",
                self._handle_quit,
            ),
        )

    def _handle_open_dashboard(self, icon=None, item=None):
        logger.info("Tray menu: Open Dashboard clicked.")
        self.on_open_dashboard()

    def _handle_restart_server(self, icon=None, item=None):
        logger.info("Tray menu: Restart Server clicked.")
        self.on_restart_server()

    def _handle_quit(self, icon=None, item=None):
        logger.info("Tray menu: Quit Apah clicked.")
        if self.icon:
            self.icon.stop()
        self.on_quit()

    def run(self):
        """Run system tray event loop."""
        if pystray is None or PYSTRAY_IMPORT_ERROR is not None:
            raise RuntimeError(f"System tray icon unavailable: {PYSTRAY_IMPORT_ERROR}")

        try:
            image = create_tray_icon_image()
            self.icon = pystray.Icon(
                "apah_tray",
                image,
                "Apah LLM Engine",
                self._build_menu(),
            )
            logger.info("Starting system tray icon loop...")
            self.icon.run()
        except Exception as e:
            logger.warning(f"Failed to start system tray icon loop: {e}")
            raise RuntimeError(f"System tray GUI display error: {e}")
