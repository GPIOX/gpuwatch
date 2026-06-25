"""
GPU Watch — Textual TUI application.

Interactive multi-server GPU monitoring with real-time NVML polling.
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header

from .collector import Collector
from .config import discover_servers
from .models import ServerConfig, ServerSnapshot
from .ui.dashboard import Dashboard
from .ui.server_selector import ServerItem, ServerSelector


class GPUWatchApp(App):
    """Main Textual app for GPU Watch."""

    TITLE = "GPU Watch"
    SUB_TITLE = "Multi-server GPU monitor"

    CSS = """
    #main-container {
        layout: horizontal;
        height: 1fr;
    }

    ServerSelector {
        width: 28;
        height: 1fr;
    }

    Dashboard {
        width: 1fr;
        height: 1fr;
    }

    ServerPanel {
        height: auto;
        min-height: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "refresh_all", "Refresh", show=True),
        Binding("c", "toggle_compact", "Compact", show=True),
    ]

    def __init__(self, refresh: float = 1.5, timeout: float = 5.0) -> None:
        super().__init__()
        self._refresh_interval = refresh
        self._timeout = timeout
        self._servers: list[ServerConfig] = []
        self._collector: Collector | None = None
        self._selector: ServerSelector | None = None
        self._dashboard: Dashboard | None = None
        self._compact_mode: bool = False

    def compose(self) -> ComposeResult:
        self._servers = discover_servers()
        yield Header()
        with Horizontal(id="main-container"):
            server_list = [(s.host, s.label, s.enabled) for s in self._servers]
            self._selector = ServerSelector(server_list)
            yield self._selector

            self._dashboard = Dashboard()
            yield self._dashboard
        yield Footer()

    def on_mount(self) -> None:
        """Start the collector after the UI is ready."""
        self._collector = Collector(
            self._servers,
            refresh_seconds=self._refresh_interval,
            timeout_seconds=self._timeout,
        )
        self._collector.on_update(self._on_snapshot_update)

        # Auto-start servers that were enabled in config
        for server in self._servers:
            if server.enabled:
                if self._dashboard:
                    self._dashboard.ensure_panel(server.host, server.label)
                asyncio.create_task(self._collector.start(server.host))

    # ── collector callback ────────────────────────────────────────────

    def _on_snapshot_update(self, host: str, snapshot: ServerSnapshot) -> None:
        """Callback from collector: update the UI with a new snapshot (synchronous)."""
        if self._dashboard:
            self._dashboard.update_panel(host, snapshot)

        if self._selector:
            status_map = {
                "ok": f"OK {snapshot.latency_ms:.0f}ms" if snapshot.latency_ms else "OK",
                "connecting": "[yellow]...[/]",
                "timeout": "[red]TIMEOUT[/]",
                "stale": "[yellow]STALE[/]",
                "error": "[red]ERR[/]",
                "auth_error": "[red]AUTH[/]",
                "no_python": "[red]NO PY3[/]",
                "down": "[red]DOWN[/]",
            }
            status = status_map.get(snapshot.status, snapshot.status)
            self._selector.update_status(host, status)

    # ── server toggle handler (bubbles up from ServerItem) ───────────

    async def on_server_item_toggled(self, event: ServerItem.Toggled) -> None:
        """Handle server toggle events. Messages bubble from ServerItem."""
        if self._collector is None:
            return

        server = next((s for s in self._servers if s.host == event.host), None)
        if server is None:
            return

        server.enabled = event.enabled

        if event.enabled:
            if self._dashboard:
                self._dashboard.ensure_panel(server.host, server.label)
            await self._collector.start(event.host)
        else:
            await self._collector.stop(event.host)
            if self._dashboard:
                self._dashboard.remove_panel(event.host)

    # ── actions ───────────────────────────────────────────────────────

    def action_refresh_all(self) -> None:
        """Force refresh: restart polling for all enabled servers."""
        if not hasattr(self, "_refresh_lock"):
            self._refresh_lock = asyncio.Lock()

        # Remember current focus so we can restore it after refresh
        focused = self.screen.focused

        async def _restart():
            if self._collector is None:
                return
            async with self._refresh_lock:
                for server in self._servers:
                    if server.enabled:
                        await self._collector.stop(server.host)
                        await self._collector.start(server.host)
            # Restore focus after refresh completes
            if focused is not None:
                try:
                    focused.focus()
                except Exception:
                    pass

        asyncio.create_task(_restart())

    def action_toggle_compact(self) -> None:
        """Toggle compact display mode."""
        self._compact_mode = not self._compact_mode
        if self._dashboard:
            self._dashboard.compact = self._compact_mode
        mode = "on" if self._compact_mode else "off"
        self.notify(f"Compact mode {mode}", timeout=1)

    async def action_quit(self) -> None:
        """Clean shutdown: stop all collectors before exiting."""
        if self._collector:
            await self._collector.stop_all()
        self.exit()


def main() -> None:
    """Entry point for `gpuwatch` command."""
    app = GPUWatchApp(refresh=1.5, timeout=5.0)
    app.run()
