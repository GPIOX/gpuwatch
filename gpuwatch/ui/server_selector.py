"""
Server selector sidebar with checkboxes.

Shows a list of discovered GPU servers with [x]/[ ] toggles.
Space key toggles monitoring on/off.
"""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Label, Static


class ServerItem(Static, can_focus=True):
    """A single server row with checkbox and label.

    Messages bubble naturally to parent widgets via Textual's DOM,
    so the App can handle ServerItem.Toggled directly.
    """

    class Toggled(Message):
        """Emitted when a server is toggled on/off. Bubbles up."""

        def __init__(self, host: str, enabled: bool) -> None:
            super().__init__()
            self.host = host
            self.enabled = enabled

    def __init__(self, host: str, label: str, enabled: bool = False) -> None:
        super().__init__()
        self.host = host
        self.server_label = label
        self._enabled = enabled
        self._status: str = ""

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_status(self, status: str) -> None:
        """Show a small status indicator after the label."""
        self._status = status
        self.refresh()

    def toggle(self) -> None:
        """Toggle monitoring state."""
        self._enabled = not self._enabled
        self.refresh()
        self.post_message(self.Toggled(self.host, self._enabled))

    def render(self) -> str:
        check = "[bold green]◉[/]" if self._enabled else "[dim]○[/]"
        has_focus = "[cyan]▸[/]" if self.has_focus else " "
        status = f" {self._status}" if self._status else ""
        return f"{has_focus} {check} {self.server_label}{status}"

    def on_key(self, event: events.Key) -> None:
        """Space toggles the checkbox."""
        if event.key == "space":
            self.toggle()
            event.prevent_default()
            event.stop()


class ServerSelector(Vertical):
    """Sidebar listing all discovered servers with toggles.

    Handles up/down arrow keys for navigating between ServerItem children.
    Messages from ServerItem bubble through here to the App automatically.
    """

    DEFAULT_CSS = """
    ServerSelector {
        width: 28;
        height: 1fr;
        border: solid $primary-background;
        padding: 1 0;
    }

    ServerItem {
        height: 1;
        padding: 0 1;
    }
    ServerItem:focus {
        background: $boost;
    }
    """

    def __init__(self, servers: list[tuple[str, str, bool]]) -> None:
        super().__init__()
        self._items: dict[str, ServerItem] = {}
        self._server_list = servers
        self._ordered: list[str] = []  # host order for navigation

    def compose(self) -> ComposeResult:
        yield Label("  [bold]Servers[/]")
        for host, label, enabled in self._server_list:
            item = ServerItem(host, label, enabled=enabled)
            self._items[host] = item
            self._ordered.append(host)
            yield item

    def get_item(self, host: str) -> ServerItem | None:
        return self._items.get(host)

    def update_status(self, host: str, status: str) -> None:
        """Update the status hint for a server item."""
        item = self._items.get(host)
        if item is not None:
            item.set_status(status)

    def on_key(self, event: events.Key) -> None:
        """Arrow keys navigate between server items."""
        if event.key not in ("up", "down"):
            return

        focused = self.screen.focused
        current_host = None
        if isinstance(focused, ServerItem):
            current_host = focused.host

        # Find current index (or -1 if nothing focused)
        try:
            idx = self._ordered.index(current_host) if current_host else -1
        except ValueError:
            idx = -1

        if event.key == "up":
            idx = max(0, idx - 1) if idx >= 0 else 0
        else:  # down
            idx = min(len(self._ordered) - 1, idx + 1)

        if 0 <= idx < len(self._ordered):
            target = self._items.get(self._ordered[idx])
            if target is not None:
                target.focus()

        event.prevent_default()
        event.stop()
