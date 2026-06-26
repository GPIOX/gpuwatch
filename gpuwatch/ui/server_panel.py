"""
Single-server GPU panel widget.

Renders GPU utilization bars, memory usage, temperatures, power draw,
and running processes for one server. Updates on each polling cycle.
"""

from __future__ import annotations

import time

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from ..models import ServerSnapshot

from .gpu_bar import _format_mem, memory_bar, power_str, temp_str, utilization_bar


def _truncate(text: str, max_len: int = 70) -> str:
    """Truncate a string if too long, appending '…'."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"




class ServerPanel(Static):
    """A panel displaying one server's GPU status."""

    def __init__(self, host: str, label: str) -> None:
        super().__init__("")
        self._host = host
        self._label = label
        self._snapshot: ServerSnapshot | None = None
        self.compact: bool = False
        self.name_width: int = 10  # set by Dashboard, updated dynamically

    @property
    def host(self) -> str:
        return self._host

    def update_snapshot(self, snapshot: ServerSnapshot) -> None:
        """Update with a new snapshot and re-render."""
        self._snapshot = snapshot
        self.refresh(layout=True)

    def render(self) -> Panel:
        if self._snapshot is None:
            return Panel(
                Text("Waiting for first poll...", style="dim"),
                title=self._label,
                border_style="bright_black",
            )

        snap = self._snapshot
        return Panel(
            self._build_content(snap),
            title=self._build_title(snap),
            border_style="bright_black",
        )

    def _build_title(self, snap: ServerSnapshot) -> Text:
        """Build the panel title line: 'two4090     OK  42ms  12:31:04'."""
        title = Text()
        title.append(snap.label, style="bold cyan")

        # Status indicator
        status_colors = {
            "ok": "green",
            "connecting": "yellow",
            "timeout": "red",
            "stale": "yellow",
            "error": "red",
            "auth_error": "red",
            "no_python": "red",
            "down": "red",
        }
        color = status_colors.get(snap.status, "red")
        status_labels = {
            "ok": "OK",
            "connecting": "CONNECTING",
            "timeout": "TIMEOUT",
            "stale": "STALE",
            "error": "ERROR",
            "auth_error": "AUTH ERR",
            "no_python": "NO PYTHON",
            "down": "DOWN",
        }
        label = status_labels.get(snap.status, snap.status.upper())
        title.append(f"  {label}", style=f"bold {color}")

        # Latency
        if snap.latency_ms is not None:
            title.append(
                f"  {snap.latency_ms:.0f}ms", style="bright_black"
            )

        # Last update time
        if snap.updated_at:
            ts = time.strftime("%H:%M:%S", time.localtime(snap.updated_at))
            title.append(f"  {ts}", style="bright_black")

        return title

    def _build_content(self, snap: ServerSnapshot) -> Table:
        """Build a Rich Table of GPU rows + process subtables."""
        if snap.status == "connecting":
            t = Table(show_header=False, expand=True, box=None)
            t.add_row(Text("Connecting...", style="yellow"))
            return t

        # Choose rendering style
        if self.compact:
            gpu_table = self._build_compact(snap)
        else:
            gpu_table = self._build_full(snap)

        # Prepend error banner if there's an error (preserves GPU data below it)
        if snap.error:
            wrapper = Table(show_header=False, expand=True, box=None)
            err_text = Text("Error: ", style="bold red")
            err_text.append(snap.error, style="red")
            wrapper.add_row(err_text)
            if snap.gpus:
                wrapper.add_row(Text(""))
                wrapper.add_row(Text("Showing last known data:", style="dim"))
            wrapper.add_row(Text(""))
            wrapper.add_row(gpu_table)
            return wrapper

        return gpu_table

    def _build_full(self, snap: ServerSnapshot) -> Table:
        """GPU metric rows in fixed columns, process info as indented text below."""
        wrapper = Table(show_header=False, expand=True, box=None, padding=(0, 1))
        # Single-column wrapper so process text can flow freely
        wrapper.add_column("body", justify="left")

        # ── GPU metric grid (nested fixed-column table) ──
        gpu_grid = Table(show_header=False, expand=True, box=None, padding=0)
        gpu_grid.add_column("gpu", width=5, justify="left")
        gpu_grid.add_column("name", width=self.name_width, justify="left")
        gpu_grid.add_column("util", width=15, justify="left")
        gpu_grid.add_column("mem", width=36, justify="left")
        gpu_grid.add_column("temp", width=5, justify="left")
        gpu_grid.add_column("power", width=6, justify="left")

        for gpu in snap.gpus:
            gpu_grid.add_row(
                Text(f"GPU {gpu.index}", style="bold cyan"),
                Text(gpu.name, style="white"),
                utilization_bar(gpu.utilization_gpu, width=11),
                memory_bar(gpu.memory_used_mb, gpu.memory_total_mb, width=18),
                temp_str(gpu.temperature_c),
                power_str(gpu.power_watts, gpu.power_limit_watts),
            )

        wrapper.add_row(gpu_grid)

        # ── Process details (free-form indented text below GPU grid) ──
        for gpu in snap.gpus:
            if gpu.processes:
                wrapper.add_row(Text(""))  # spacer
                wrapper.add_row(Text(
                    f"  GPU {gpu.index}  PID      Mem     Command",
                    style="bold underline",
                ))
                for proc in gpu.processes:
                    mem_str = _format_mem(proc.gpu_memory_mb)
                    cmd = _truncate(proc.cmdline or proc.name)
                    wrapper.add_row(Text(
                        f"        {proc.pid:<7} {mem_str:>9}  {cmd}",
                        style="green",
                    ))

            if gpu.other_users:
                wrapper.add_row(Text(""))  # spacer
                for ou in gpu.other_users:
                    mem_str = _format_mem(ou.total_memory_mb)
                    wrapper.add_row(Text(
                        f"  GPU {gpu.index}  {ou.user}: {ou.process_count} proc, {mem_str}",
                        style="dim",
                    ))

        return wrapper

    def _build_compact(self, snap: ServerSnapshot) -> Table:
        """Compact: one line per GPU, process summary inline."""
        gpu_grid = Table(show_header=False, expand=True, box=None, padding=0)
        gpu_grid.add_column("gpu", width=5, justify="left")
        gpu_grid.add_column("name", width=self.name_width, justify="left")
        gpu_grid.add_column("util", width=14, justify="left")
        gpu_grid.add_column("mem", width=37, justify="left")
        gpu_grid.add_column("temp", width=5, justify="left")
        gpu_grid.add_column("power", width=6, justify="left")
        gpu_grid.add_column("proc", width=20, justify="left")

        for gpu in snap.gpus:
            proc_parts: list[str] = [p.name for p in gpu.processes]
            if gpu.other_users:
                other_total = sum(ou.total_memory_mb for ou in gpu.other_users)
                other_count = sum(ou.process_count for ou in gpu.other_users)
                proc_parts.append(f"+{other_count}o/{_format_mem(other_total)}")
            proc_str = ", ".join(proc_parts) if proc_parts else "—"

            gpu_grid.add_row(
                Text(f"GPU {gpu.index}", style="bold cyan"),
                Text(gpu.name, style="white"),
                utilization_bar(gpu.utilization_gpu, width=10),
                memory_bar(gpu.memory_used_mb, gpu.memory_total_mb, width=18),
                temp_str(gpu.temperature_c),
                power_str(gpu.power_watts, gpu.power_limit_watts),
                Text(_truncate(proc_str, 23), style="green"),
            )

        return gpu_grid
