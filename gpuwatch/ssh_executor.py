"""
Asynchronous SSH executor using system `ssh` binary.

Runs the NVML probe script on remote servers via SSH stdin,
captures JSON stdout, respects ~/.ssh/config automatically.

No extra SSH library needed — uses asyncio subprocess.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


# Read the bundled probe script once at import time
_PROBE_PATH = Path(__file__).parent / "nvml_probe.py"
_PROBE_SCRIPT = _PROBE_PATH.read_text(encoding="utf-8")


class SSHTimeoutError(asyncio.TimeoutError):
    """Raised when an SSH command exceeds its time limit."""


class SSHCommandError(Exception):
    """Raised when the remote command exits with a non-zero status."""

    def __init__(self, returncode: int, stderr: str):
        self.returncode = returncode
        self.stderr = stderr.strip()
        super().__init__(f"SSH exited {returncode}: {self.stderr}")


class SSHAuthError(SSHCommandError):
    """Raised on SSH authentication failure (exit 255)."""


class RemotePythonNotFound(SSHCommandError):
    """Raised when python3 is not available on the remote server."""


async def run_probe(
    host_alias: str, timeout: float = 5.0, own_user: str | None = None
) -> tuple[str, float]:
    """Execute the NVML probe on a remote server via SSH.

    Args:
        host_alias: SSH host alias (from ~/.ssh/config).
        timeout: Maximum time to wait for the SSH command (seconds).
        own_user: If set, passed to probe as --own-user for highlighting.

    Returns:
        (stdout_string, latency_ms) on success.

    Raises:
        SSHTimeoutError: if the command times out.
        SSHAuthError: if SSH authentication fails.
        RemotePythonNotFound: if python3 is missing on the remote.
        SSHCommandError: for other non-zero exits.
    """
    loop = asyncio.get_running_loop()
    start = loop.time()

    # SSH multiplexing: reuse connections to reduce per-poll overhead.
    # %C is a hash of %l%h%p%r%j — safe for Unix socket path limits.
    import os as _os
    _ctrl_dir = _os.path.expanduser("~/.ssh/controlmasters")
    _os.makedirs(_ctrl_dir, mode=0o700, exist_ok=True)

    # Build remote command: python3 - [--own-user <user>]
    remote_cmd = ["python3", "-"]
    if own_user:
        remote_cmd.extend(["--own-user", own_user])

    proc = await asyncio.create_subprocess_exec(
        "ssh",
        "-T",  # disable pseudo-terminal allocation
        "-o", "BatchMode=yes",  # never prompt for password
        "-o", "ConnectTimeout=3",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ControlMaster=auto",
        "-o", "ControlPersist=60s",
        "-o", f"ControlPath={_ctrl_dir}/%C",
        "--",  # prevent alias starting with '-' being parsed as option
        host_alias,
        *remote_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=_PROBE_SCRIPT.encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise SSHTimeoutError(
            f"SSH to {host_alias} timed out after {timeout}s"
        )
    finally:
        # Ensure the subprocess is killed/cleaned up on any failure
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass

    latency_ms = (loop.time() - start) * 1000

    if proc.returncode == 255:
        stderr_str = stderr.decode("utf-8", errors="replace")
        if "Permission denied" in stderr_str:
            raise SSHAuthError(proc.returncode, stderr_str)
        raise SSHCommandError(proc.returncode, stderr_str)

    if proc.returncode == 127:
        raise RemotePythonNotFound(
            proc.returncode,
            stderr.decode("utf-8", errors="replace"),
        )

    if proc.returncode != 0:
        raise SSHCommandError(
            proc.returncode,
            stderr.decode("utf-8", errors="replace"),
        )

    return stdout.decode("utf-8"), latency_ms
