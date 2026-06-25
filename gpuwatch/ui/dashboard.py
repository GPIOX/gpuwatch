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
    Scrolling works via mouse wheel / touchpad.
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

    @property
    def compact(self) -> bool:
        return self._compact

    @compact.setter
    def compact(self, value: bool) -> None:
        self._compact = value
        for panel in self._panels.values():
            panel.compact = value
            panel.refresh(layout=True)

    def ensure_panel(self, host: str, label: str) -> ServerPanel:
        """Get or create a panel widget for a server."""
        if host not in self._panels:
            panel = ServerPanel(host, label)
            panel.compact = self._compact
            self._panels[host] = panel
            self.mount(panel)
            self.refresh(layout=True)
        return self._panels[host]

    def update_panel(self, host: str, snapshot: ServerSnapshot) -> None:
        """Update the panel for a specific server with new snapshot data."""
        panel = self._panels.get(host)
        if panel is not None:
            panel.update_snapshot(snapshot)

    def remove_panel(self, host: str) -> None:
        """Remove a server panel (when monitoring is stopped)."""
        panel = self._panels.pop(host, None)
        if panel is not None:
            panel.remove()
            self.refresh(layout=True)
