"""System tray icon helper for Apah Desktop App using pystray and Pillow."""

import logging
from typing import Callable, Optional
from PIL import Image, ImageDraw

logger = logging.getLogger("apah.desktop.tray_icon")


def generate_default_icon_image() -> Image.Image:
    """Generate a high-resolution 64x64 PIL Image icon for system tray if no image file is provided."""
    image = Image.new("RGBA", (64, 64), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Rounded blue square
    draw.rounded_rectangle((4, 4, 60, 60), radius=12, fill="#2563eb")

    # White 'A' letter logo
    draw.polygon([(32, 14), (48, 50), (40, 50), (32, 34), (24, 50), (16, 50)], fill="#ffffff")
    draw.ellipse((27, 24, 37, 34), fill="#60a5fa")

    return image


def setup_tray_menu(
    on_open_dashboard: Callable[[], None],
    on_restart_server: Callable[[], None],
    on_quit: Callable[[], None],
    is_running: bool = True,
    icon_image_path: Optional[str] = None,
):
    """Create and return a pystray.Icon instance with standard Apah menu items."""
    import pystray
    from pystray import MenuItem as item

    if icon_image_path:
        try:
            image = Image.open(icon_image_path)
        except Exception as e:
            logger.warning(f"Could not load tray icon from '{icon_image_path}': {e}. Using default icon.")
            image = generate_default_icon_image()
    else:
        image = generate_default_icon_image()

    status_text = "Server: Running" if is_running else "Server: Stopped"

    menu = pystray.Menu(
        item("Open Dashboard", lambda icon, item: on_open_dashboard(), default=True),
        item(status_text, None, enabled=False),
        item("Restart Server", lambda icon, item: on_restart_server()),
        pystray.Menu.SEPARATOR,
        item("Quit", lambda icon, item: on_quit()),
    )

    icon = pystray.Icon("apah", image, "Apah AI Engine", menu=menu)
    return icon
