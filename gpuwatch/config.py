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

# Known code-hosting / non-server domains to skip in auto-discovery
_SKIP_HOSTNAMES = {
    "github.com", "gitlab.com", "bitbucket.org",
    "codeberg.org", "gitee.com",
}


def parse_ssh_config(path: str | None = None) -> list[dict[str, str]]:
    """Parse ~/.ssh/config and return a list of Host entries.

    Each entry is a dict with keys: 'host', 'hostname', 'port', 'user'.
    Wildcard hosts (Host *) are excluded.
    """
    if path is None:
        path = os.path.expanduser("~/.ssh/config")

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (FileNotFoundError, UnicodeDecodeError):
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
            # Parse aliases; SSH supports "Host gpu1 gpu2"
            import shlex
            try:
                aliases = shlex.split(value)
            except ValueError:
                aliases = value.split()
            non_wildcard = [a for a in aliases if "*" not in a and "?" not in a]
            if non_wildcard:
                # Use first non-wildcard alias as primary
                current = {"host": non_wildcard[0]}
            else:
                current = None
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

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, UnicodeDecodeError):
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
                    ssh_user=ssh_info.get("user"),
                )
            )
        return result

    # Show all non-wildcard hosts, skipping known code-hosting domains.
    servers = [
        h for h in ssh_hosts
        if h["host"].lower() not in _SKIP_HOSTNAMES
        and h.get("hostname", "").lower() not in _SKIP_HOSTNAMES
    ]
    return [
        ServerConfig(
            host=h["host"],
            label=h["host"],
            enabled=False,
            ssh_user=h.get("user"),
        )
        for h in servers
    ]
