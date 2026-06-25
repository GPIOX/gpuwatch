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
import os

from .gpu_bar import _format_mem, memory_bar, power_str, temp_str, utilization_bar


def _truncate_cmd(cmd: str, max_len: int = 70) -> str:
    """Truncate a command string if too long, appending '…'."""
    if len(cmd) <= max_len:
        return cmd
    return cmd[: max_len - 1] + "…"


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
        """Full layout: one row per GPU + own processes + aggregated other users."""
        gpu_table = Table(
            show_header=False, expand=True, box=None, padding=(0, 1)
        )

        for gpu in snap.gpus:
            util_text = utilization_bar(gpu.utilization_gpu, width=12)
            mem_text = memory_bar(
                gpu.memory_used_mb, gpu.memory_total_mb, width=18
            )
            temp_text = temp_str(gpu.temperature_c)
            power_text = power_str(gpu.power_watts, gpu.power_limit_watts)

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

            # ── Own processes (highlighted with full cmdline) ──
            if gpu.processes:
                gpu_table.add_row(Text(""))  # spacer
                proc_label = Text(
                    "  PID      GPU Mem   Command", style="bold underline"
                )
                gpu_table.add_row(proc_label)
                for proc in gpu.processes:
                    mem_str = _format_mem(proc.gpu_memory_mb)
                    cmd = _truncate_cmd(proc.cmdline or proc.name)

                    proc_text = Text()
                    proc_text.append(
                        f"  {proc.pid:<7} ", style="cyan"
                    )
                    proc_text.append(
                        f"{mem_str:>9} ", style="yellow"
                    )
                    proc_text.append(f"  {cmd}", style="green")
                    gpu_table.add_row(proc_text)

            # ── Other users (aggregated, dimmed) ──
            if gpu.other_users:
                gpu_table.add_row(Text(""))  # spacer
                if gpu.processes:
                    gpu_table.add_row(Text("  ─────────────────────────────────────────────", style="dim"))
                for ou in gpu.other_users:
                    mem_str = _format_mem(ou.total_memory_mb)
                    other_text = Text()
                    other_text.append(
                        f"  {ou.user:<15} ", style="bright_black"
                    )
                    other_text.append(
                        f"{ou.process_count} proc, {mem_str}",
                        style="dim",
                    )
                    gpu_table.add_row(other_text)

            if gpu.index < len(snap.gpus) - 1:
                gpu_table.add_row(Text("", style="dim"))

        return gpu_table

    def _build_compact(self, snap: ServerSnapshot) -> Table:
        """Compact layout: one line per GPU, own process + other user summary inline."""
        gpu_table = Table(
            show_header=False, expand=True, box=None, padding=(0, 1)
        )

        header = Text()
        header.append(
            f"{'GPU':<5} {'Name':<22} {'Util':>5}  {'Memory':<24} {'Temp':>4}  {'Power':>10}  Processes",
            style="bold underline",
        )
        gpu_table.add_row(header)

        for gpu in snap.gpus:
            util_text = utilization_bar(gpu.utilization_gpu, width=10)
            mem_text = memory_bar(
                gpu.memory_used_mb, gpu.memory_total_mb, width=20
            )
            temp_text = temp_str(gpu.temperature_c)
            power_text = power_str(gpu.power_watts, gpu.power_limit_watts)

            # Build process summary
            proc_parts: list[str] = []
            for proc in gpu.processes:
                proc_parts.append(f"{proc.name}")
            if gpu.other_users:
                other_total = sum(ou.total_memory_mb for ou in gpu.other_users)
                other_count = sum(ou.process_count for ou in gpu.other_users)
                proc_parts.append(
                    f"+{other_count} other ({_format_mem(other_total)})"
                )
            proc_str = ", ".join(proc_parts) if proc_parts else "-"

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
