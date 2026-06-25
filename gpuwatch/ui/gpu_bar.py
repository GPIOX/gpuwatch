"""
Rich bar rendering for GPU utilization and memory.

Produces colored progress bar strings suitable for Rich renderables.
Uses manual Unicode block characters for reliable rendering across Rich versions.
"""

from __future__ import annotations

from rich.style import Style
from rich.text import Text


def _bar_style(percent: float) -> Style:
    """Color bar: green < 50%, yellow < 80%, red >= 80%."""
    if percent < 50:
        return Style(color="green")
    elif percent < 80:
        return Style(color="yellow")
    else:
        return Style(color="red")


def _render_bar(percent: float, width: int) -> str:
    """Build a bar string with █ (filled) and ░ (empty) characters."""
    pct = max(0.0, min(percent, 100.0))
    filled = int(round(pct / 100.0 * width))
    filled = min(filled, width)
    empty = width - filled
    return "█" * filled + "░" * empty


def utilization_bar(percent: int, width: int = 10) -> Text:
    """Render a utilization bar like: '████████░░ 72%'."""
    pct = max(0, min(percent, 100))
    style = _bar_style(pct)
    bar_str = _render_bar(pct, width)
    result = Text(bar_str, style=style)
    result.append(f" {pct:3d}%", style=style)
    return result


def _format_mem(mb: int) -> str:
    """Format memory value like nvitop:
    >= 20480 MiB (20 GiB) → XX.XXGiB
    <  20480 MiB          → XXXXXMiB
    """
    if mb >= 20480:
        return f"{mb / 1024:.2f}GiB"
    else:
        return f"{mb}MiB"


def memory_bar(used_mb: int, total_mb: int, width: int = 16) -> Text:
    """Render a memory bar like: '████████░░░░ 21.24GiB / 23.99GiB'."""
    if total_mb <= 0:
        return Text("─" * width + " N/A")
    pct = (used_mb / total_mb) * 100.0
    style = _bar_style(pct)
    bar_str = _render_bar(pct, width)
    used_str = _format_mem(used_mb)
    total_str = _format_mem(total_mb)
    result = Text(bar_str, style=style)
    result.append(f" {used_str} / {total_str}", style=style)
    return result


def temp_str(celsius: int) -> Text:
    """Render temperature with color: blue < 50, green < 70, yellow < 85, red >= 85."""
    if celsius <= 0:
        return Text("N/A", style=Style(color="bright_black"))
    if celsius < 50:
        style = Style(color="blue")
    elif celsius < 70:
        style = Style(color="green")
    elif celsius < 85:
        style = Style(color="yellow")
    else:
        style = Style(color="red")
    return Text(f"{celsius}°C", style=style)


def power_str(watts: float, limit_watts: float) -> Text:
    """Render power draw with color relative to limit:
    green < 50%, yellow < 80%, red >= 80%.
    """
    if limit_watts <= 0:
        return Text(f"{watts:.0f}W", style=Style(color="bright_black"))
    ratio = (watts / limit_watts) * 100
    if ratio < 50:
        style = Style(color="green")
    elif ratio < 80:
        style = Style(color="yellow")
    else:
        style = Style(color="red")
    return Text(f"{watts:.0f}W", style=style)
