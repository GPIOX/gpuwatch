"""
Data models for GPU monitoring.

All data is ephemeral — received as JSON from remote probes,
parsed in memory, rendered to TUI. Nothing persists to disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class GPUProcess:
    """A process running on a specific GPU."""

    pid: int
    name: str
    gpu_memory_mb: int
    user: str | None = None

    @classmethod
    def from_probe(cls, data: dict[str, Any]) -> GPUProcess:
        return cls(
            pid=data["pid"],
            name=data.get("name", "?"),
            gpu_memory_mb=data["gpu_memory_mb"],
            user=data.get("user"),
        )


@dataclass
class GPUInfo:
    """Snapshot of a single GPU's state."""

    index: int
    uuid: str
    name: str
    utilization_gpu: int  # 0–100
    utilization_mem: int  # 0–100
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    temperature_c: int
    power_watts: float
    power_limit_watts: float
    processes: list[GPUProcess] = field(default_factory=list)

    @property
    def memory_percent(self) -> float:
        """Memory usage as percentage."""
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100

    @classmethod
    def from_probe(cls, data: dict[str, Any]) -> GPUInfo:
        processes = [
            GPUProcess.from_probe(p) for p in data.get("processes", [])
        ]
        return cls(
            index=data["index"],
            uuid=data.get("uuid", "unknown"),
            name=data.get("name", "unknown"),
            utilization_gpu=data.get("utilization_gpu", 0),
            utilization_mem=data.get("utilization_mem", 0),
            memory_total_mb=data.get("memory_total_mb", 0),
            memory_used_mb=data.get("memory_used_mb", 0),
            memory_free_mb=data.get("memory_free_mb", 0),
            temperature_c=data.get("temperature_c", 0),
            power_watts=data.get("power_watts", 0.0),
            power_limit_watts=data.get("power_limit_watts", 0.0),
            processes=processes,
        )


ServerStatus = Literal[
    "ok", "connecting", "timeout", "error", "stale", "auth_error", "no_python", "down"
]


@dataclass
class ServerSnapshot:
    """Snapshot of one server's GPU state at a point in time."""

    host: str
    label: str
    status: ServerStatus
    gpus: list[GPUInfo]
    error: str | None = None
    updated_at: float = 0.0
    latency_ms: float | None = None

    @classmethod
    def from_probe(
        cls,
        host: str,
        label: str,
        data: dict[str, Any],
        latency_ms: float,
    ) -> ServerSnapshot:
        """Build a snapshot from successful probe output."""
        gpus = [GPUInfo.from_probe(g) for g in data.get("gpus", [])]
        return cls(
            host=host,
            label=label,
            status="ok",
            gpus=gpus,
            updated_at=time.time(),
            latency_ms=latency_ms,
        )

    @classmethod
    def error_snapshot(
        cls,
        host: str,
        label: str,
        status: ServerStatus,
        error: str,
        previous: ServerSnapshot | None = None,
    ) -> ServerSnapshot:
        """Build a snapshot representing an error state, preserving old GPU data if available."""
        return cls(
            host=host,
            label=label,
            status=status,
            gpus=previous.gpus if previous else [],
            error=error,
            updated_at=time.time(),
            latency_ms=previous.latency_ms if previous else None,
        )


@dataclass
class ServerConfig:
    """Configuration for a monitored server."""

    host: str  # SSH alias
    label: str  # display name
    enabled: bool = False  # whether polling is active
