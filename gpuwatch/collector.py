"""
Async polling manager — one long-lived coroutine per selected server.

Fires SSH probe calls on a fixed interval, parses JSON responses into
ServerSnapshot objects, and exposes the latest snapshot for the TUI.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable

from . import ssh_executor
from .models import ServerConfig, ServerSnapshot

logger = logging.getLogger(__name__)


class Collector:
    """Manages polling tasks for multiple servers.

    Each selected server gets a dedicated async task that loops:
        SSH probe → parse JSON → store snapshot → sleep → repeat
    """

    def __init__(
        self,
        servers: list[ServerConfig],
        refresh_seconds: float = 1.5,
        timeout_seconds: float = 5.0,
    ):
        self._servers: dict[str, ServerConfig] = {s.host: s for s in servers}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._snapshots: dict[str, ServerSnapshot | None] = {
            s.host: None for s in servers
        }
        self._refresh = refresh_seconds
        self._timeout = timeout_seconds
        self._on_update: Callable[[str, ServerSnapshot], None] | None = None

    @property
    def snapshots(self) -> dict[str, ServerSnapshot | None]:
        return dict(self._snapshots)

    def on_update(self, callback: Callable[[str, ServerSnapshot], None]) -> None:
        """Register a callback invoked on each new snapshot. Thread-safe via asyncio."""
        self._on_update = callback

    async def start(self, host: str) -> None:
        """Start polling a server. Idempotent — no-op if already running."""
        if host in self._tasks:
            return
        config = self._servers.get(host)
        if config is None:
            logger.warning("Unknown server: %s", host)
            return

        # Start with a connecting placeholder
        self._snapshots[host] = ServerSnapshot.error_snapshot(
            host=host,
            label=config.label,
            status="connecting",
            error="Connecting...",
        )
        self._notify(host, self._snapshots[host])  # type: ignore[arg-type]

        self._tasks[host] = asyncio.create_task(self._poll_loop(host, config.label))

    async def stop(self, host: str) -> None:
        """Stop polling a server. Idempotent."""
        task = self._tasks.pop(host, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def stop_all(self) -> None:
        """Cancel all polling tasks."""
        for host in list(self._tasks):
            await self.stop(host)

    async def _poll_loop(self, host: str, label: str) -> None:
        """Single-server polling loop."""
        consecutive_failures = 0

        while True:
            try:
                json_str, latency_ms = await ssh_executor.run_probe(
                    host, timeout=self._timeout
                )
                data = json.loads(json_str)

                if data.get("ok"):
                    snapshot = ServerSnapshot.from_probe(host, label, data, latency_ms)
                    consecutive_failures = 0
                else:
                    snapshot = ServerSnapshot.error_snapshot(
                        host=host,
                        label=label,
                        status="error",
                        error=data.get("error", "Unknown remote error"),
                        previous=self._snapshots.get(host),
                    )
                    consecutive_failures += 1

            except ssh_executor.SSHTimeoutError as e:
                consecutive_failures += 1
                prev = self._snapshots.get(host)
                status = "stale" if prev and prev.gpus else "timeout"
                snapshot = ServerSnapshot.error_snapshot(
                    host=host, label=label, status=status, error=str(e), previous=prev
                )

            except ssh_executor.SSHAuthError as e:
                consecutive_failures += 1
                snapshot = ServerSnapshot.error_snapshot(
                    host=host,
                    label=label,
                    status="auth_error",
                    error=e.stderr,
                    previous=self._snapshots.get(host),
                )

            except ssh_executor.RemotePythonNotFound as e:
                consecutive_failures += 1
                snapshot = ServerSnapshot.error_snapshot(
                    host=host,
                    label=label,
                    status="no_python",
                    error="python3 not found on remote server",
                    previous=self._snapshots.get(host),
                )

            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                consecutive_failures += 1
                prev = self._snapshots.get(host)
                snapshot = ServerSnapshot.error_snapshot(
                    host=host,
                    label=label,
                    status="error",
                    error=f"Parse error: {e}",
                    previous=prev,
                )

            except Exception as e:
                consecutive_failures += 1
                logger.exception("Unexpected error polling %s", host)
                prev = self._snapshots.get(host)
                snapshot = ServerSnapshot.error_snapshot(
                    host=host,
                    label=label,
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                    previous=prev,
                )

            self._snapshots[host] = snapshot
            self._notify(host, snapshot)

            # Back off slightly on repeated failures to avoid hammering
            delay = self._refresh
            if consecutive_failures > 3:
                delay = min(self._refresh * 2, 10.0)
            if consecutive_failures > 10:
                delay = min(self._refresh * 4, 30.0)

            await asyncio.sleep(delay)

    def _notify(self, host: str, snapshot: ServerSnapshot) -> None:
        """Fire the update callback from the event loop thread."""
        if self._on_update:
            try:
                self._on_update(host, snapshot)
            except Exception:
                logger.exception("Callback error for %s", host)
