"""
Server discovery from ~/.ssh/config and optional local YAML config.

Parses SSH config for Host aliases, filters out non-GPU entries,
and allows a local YAML file to override labels / ordering.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .models import ServerConfig

logger = logging.getLogger(__name__)

# Keyword-based filtering: only show hosts that look like GPU servers
_GPU_KEYWORDS = {
    "4090",
    "3090",
    "4080",
    "3080",
    "a100",
    "a6000",
    "a5000",
    "a40",
    "h100",
    "h800",
    "v100",
    "t4",
    "gpu",
    "dgx",
}


def _looks_like_gpu(host: str) -> bool:
    """Heuristic: does the host alias contain a known GPU keyword?"""
    lower = host.lower()
    return any(kw in lower for kw in _GPU_KEYWORDS)


def parse_ssh_config(path: str | None = None) -> list[dict[str, str]]:
    """Parse ~/.ssh/config and return a list of Host entries.

    Each entry is a dict with keys: 'host', 'hostname', 'port', 'user'.
    Wildcard hosts (Host *) are excluded.
    """
    if path is None:
        path = os.path.expanduser("~/.ssh/config")

    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.warning("SSH config not found: %s", path)
        return []
    except OSError as e:
        logger.warning("Cannot read SSH config: %s", e)
        return []

    hosts: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw in lines:
        line = raw.strip()

        # Skip comments and blank lines
        if not line or line.startswith("#"):
            continue

        # Split on whitespace; handle quoted values
        parts = line.split()
        if len(parts) < 2:
            continue

        keyword = parts[0].lower()
        value = " ".join(parts[1:])

        if keyword == "host":
            # Save previous host entry
            if current and current.get("host"):
                hosts.append(current)
            # Start new entry; skip wildcards
            if "*" in value or "?" in value:
                current = None
            else:
                current = {"host": value}
        elif current is not None and keyword in ("hostname", "port", "user"):
            current[keyword] = value

    # Don't forget the last entry
    if current and current.get("host"):
        hosts.append(current)

    return hosts


def load_yaml_config(path: str | None = None) -> dict[str, Any] | None:
    """Load optional YAML config. Returns None if not found."""
    if path is None:
        path = os.path.expanduser("~/.config/gpuwatch/servers.yml")

    try:
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None
    except ImportError:
        logger.debug("pyyaml not installed, skipping YAML config")
        return None
    except Exception as e:
        logger.warning("Failed to load YAML config %s: %s", path, e)
        return None


def discover_servers(yaml_path: str | None = None) -> list[ServerConfig]:
    """Discover GPU servers from SSH config, optionally overlaid with YAML.

    Strategy:
    1. Parse ~/.ssh/config and find all hosts matching GPU keywords.
    2. If a YAML config exists, use its server list (with labels) and
       cross-reference against SSH config hosts.
    3. If no YAML config, auto-discover from SSH config with generated labels.
    """
    ssh_hosts = parse_ssh_config()
    ssh_map: dict[str, dict[str, str]] = {h["host"]: h for h in ssh_hosts}

    yaml = load_yaml_config(yaml_path)

    if yaml and "servers" in yaml:
        # Use YAML-defined server list, enriching from SSH config
        result: list[ServerConfig] = []
        for entry in yaml["servers"]:
            host = entry["host"]
            ssh_info = ssh_map.get(host, {})
            label = entry.get("label", host)
            result.append(
                ServerConfig(
                    host=host,
                    label=label,
                    enabled=entry.get("enabled", False),
                )
            )
        return result

    # Auto-discovery: find GPU-sounding hosts in SSH config
    gpu_hosts = [h for h in ssh_hosts if _looks_like_gpu(h["host"])]

    if not gpu_hosts:
        logger.warning(
            "No GPU servers found in SSH config. "
            "Create ~/.config/gpuwatch/servers.yml to configure manually."
        )
        return []

    return [
        ServerConfig(
            host=h["host"],
            label=h["host"],  # Use host alias as label
            enabled=False,
        )
        for h in gpu_hosts
    ]
