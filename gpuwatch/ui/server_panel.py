"""
Single-server GPU panel widget.

Renders GPU utilization bars, memory usage, temperatures, power draw,
and running processes for one server. Updates on each polling cycle.
"""

from __future__ import annotations

import time

from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from ..models import ServerSnapshot
from .gpu_bar import memory_bar, temp_str, utilization_bar


class ServerPanel(Static):
    """A panel displaying one server's GPU status."""

    def __init__(self, host: str, label: str) -> None:
        super().__init__("")
        self._host = host
        self._label = label
        self._snapshot: ServerSnapshot | None = None
        self.compact: bool = False

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
        return Panel(self._build_content(snap), title=self._build_title(snap))

    def _build_title(self, snap: ServerSnapshot) -> Text:
        """Build the panel title line: 'two4090     OK  42ms  12:31:04'."""
        title = Text()
        title.append(snap.label, style="bold white")

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
            wrapper.add_row(
                Text(f"[bold red]Error:[/] {snap.error}", style="red")
            )
            if snap.gpus:
                wrapper.add_row(Text(""))
                wrapper.add_row(Text("Showing last known data:", style="dim"))
            wrapper.add_row(Text(""))
            wrapper.add_row(gpu_table)
            return wrapper

        return gpu_table

    def _build_full(self, snap: ServerSnapshot) -> Table:
        """Full layout: one row per GPU + process details."""
        gpu_table = Table(
            show_header=False, expand=True, box=None, padding=(0, 1)
        )

        for gpu in snap.gpus:
            util_text = utilization_bar(gpu.utilization_gpu, width=12)
            mem_text = memory_bar(
                gpu.memory_used_mb, gpu.memory_total_mb, width=18
            )
            temp_text = temp_str(gpu.temperature_c)
            power_text = Text(
                f"{gpu.power_watts:.0f}/{gpu.power_limit_watts:.0f}W",
                style="bright_black",
            )

            gpu_row = Text()
            gpu_row.append(f"GPU {gpu.index}  ", style="bold cyan")
            gpu_row.append(f"{gpu.name:24s}", style="white")
            gpu_row.append("  ")
            gpu_row.append(util_text)
            gpu_row.append("  ")
            gpu_row.append(mem_text)
            gpu_row.append("  ")
            gpu_row.append(temp_text)
            gpu_row.append("  ")
            gpu_row.append(power_text)
            gpu_table.add_row(gpu_row)

            if gpu.processes:
                proc_label = Text("      PID   Mem       Process", style="dim")
                gpu_table.add_row(proc_label)
                for proc in gpu.processes:
                    user_tag = f"({proc.user})" if proc.user else ""
                    proc_text = Text()
                    proc_text.append(
                        f"      {proc.pid:<5} ", style="bright_black"
                    )
                    proc_text.append(
                        f"{proc.gpu_memory_mb:>5}MB  ", style="yellow"
                    )
                    proc_text.append(f"{proc.name} ", style="green")
                    proc_text.append(user_tag, style="bright_black")
                    gpu_table.add_row(proc_text)

            if gpu.index < len(snap.gpus) - 1:
                gpu_table.add_row(Text("", style="dim"))

        return gpu_table

    def _build_compact(self, snap: ServerSnapshot) -> Table:
        """Compact layout: one line per GPU, process name inline."""
        gpu_table = Table(
            show_header=False, expand=True, box=None, padding=(0, 1)
        )

        # Header
        header = Text()
        header.append(
            f"{'GPU':<5} {'Name':<22} {'Util':>5}  {'Memory':<24} {'Temp':>4}  {'Power':>10}  Proc",
            style="bold underline",
        )
        gpu_table.add_row(header)

        for gpu in snap.gpus:
            util_text = utilization_bar(gpu.utilization_gpu, width=10)
            mem_text = memory_bar(
                gpu.memory_used_mb, gpu.memory_total_mb, width=20
            )
            temp_text = temp_str(gpu.temperature_c)
            power_text = Text(
                f"{gpu.power_watts:.0f}/{gpu.power_limit_watts:.0f}W",
                style="bright_black",
            )

            # Top process name (or "-")
            if gpu.processes:
                top_proc = gpu.processes[0]
                proc_str = f"{top_proc.name}"
                if top_proc.user:
                    proc_str += f"({top_proc.user})"
            else:
                proc_str = "-"

            row = Text()
            row.append(f"GPU{gpu.index:<2} ", style="bold cyan")
            row.append(f"{gpu.name:22s}", style="white")
            row.append(" ")
            row.append(util_text)
            row.append(" ")
            row.append(mem_text)
            row.append(" ")
            row.append(temp_text)
            row.append("  ")
            row.append(power_text)
            row.append("  ")
            row.append(proc_str, style="green")
            gpu_table.add_row(row)

        return gpu_table
