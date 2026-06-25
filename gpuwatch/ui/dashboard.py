"""
Dashboard — scrollable container for server panels on the right side.
"""

from __future__ import annotations

from textual.containers import VerticalScroll

from ..models import ServerSnapshot
from .server_panel import ServerPanel


class Dashboard(VerticalScroll, can_focus=False):
    """Scrollable container holding all active server panels.

    Not focusable — focus stays on the ServerSelector sidebar.
    Tracks the global max GPU name width so all panels align.
    """

    DEFAULT_CSS = """
    Dashboard {
        width: 1fr;
        height: 1fr;
        border: solid $primary-background;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._panels: dict[str, ServerPanel] = {}
        self._compact: bool = False
        self._name_width: int = 10

    @property
    def compact(self) -> bool:
        return self._compact

    @compact.setter
    def compact(self, value: bool) -> None:
        self._compact = value
        for panel in self._panels.values():
            panel.compact = value
            panel.refresh(layout=True)

    def _compute_name_width(self) -> int:
        """Find the longest GPU name across all active panels."""
        max_len = 10
        for panel in self._panels.values():
            snap = panel._snapshot
            if snap and snap.gpus:
                for g in snap.gpus:
                    max_len = max(max_len, len(g.name))
        return min(max_len, 35)

    def _apply_name_width(self) -> None:
        """Recompute and push name width to all panels."""
        w = self._compute_name_width()
        if w != self._name_width:
            self._name_width = w
            for panel in self._panels.values():
                panel.name_width = w
                panel.refresh(layout=True)

    def ensure_panel(self, host: str, label: str) -> ServerPanel:
        """Get or create a panel widget for a server."""
        if host not in self._panels:
            panel = ServerPanel(host, label)
            panel.compact = self._compact
            panel.name_width = self._name_width
            self._panels[host] = panel
            self.mount(panel)
            self.refresh(layout=True)
        return self._panels[host]

    def update_panel(self, host: str, snapshot: ServerSnapshot) -> None:
        """Update the panel for a specific server with new snapshot data."""
        panel = self._panels.get(host)
        if panel is not None:
            panel.update_snapshot(snapshot)
            self._apply_name_width()

    def remove_panel(self, host: str) -> None:
        """Remove a server panel (when monitoring is stopped)."""
        panel = self._panels.pop(host, None)
        if panel is not None:
            panel.remove()
            self.refresh(layout=True)
            self._apply_name_width()
