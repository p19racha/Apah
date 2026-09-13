"""Rich output and formatting helpers for Apah CLI."""

from rich.console import Console

# Shared Rich Console instance
console = Console()


def human_size(size_bytes: int) -> str:
    """Format byte counts into human-readable strings (e.g. 4.2 GB)."""
    if size_bytes < 0:
        size_bytes = 0
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(size_bytes)
    unit_index = 0

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} B"
    
    formatted = f"{size:.1f}"
    if formatted.endswith(".0"):
        formatted = formatted[:-2]
    return f"{formatted} {units[unit_index]}"


def human_duration(seconds: float) -> str:
    """Format duration in seconds into human-readable strings (e.g. 1h 2m 5s)."""
    total_secs = int(max(0, seconds))
    if total_secs == 0:
        return "0s"

    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)
