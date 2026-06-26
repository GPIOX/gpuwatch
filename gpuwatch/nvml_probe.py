#!/usr/bin/env python3
"""
NVML GPU probe -- runs on remote server via SSH stdin, outputs JSON to stdout.

Zero dependencies: uses only Python stdlib ctypes + libnvidia-ml.so (part of
NVIDIA driver). No pip install required on the remote side.

Usage (on remote server):
    python3 nvml_probe.py          # runs locally
    ssh host 'python3 -' < nvml_probe.py   # sent over SSH stdin

Output: JSON object to stdout, diagnostics to stderr.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import sys
import time
from typing import Any


# ---------------------------------------------------------------------------
# NVML constants
# ---------------------------------------------------------------------------

NVML_TEMPERATURE_GPU = 0
NVML_SUCCESS = 0
NVML_ERROR_NOT_INITIALIZED = 3
NVML_ERROR_INSUFFICIENT_SIZE = 4
NVML_ERROR_NO_PERMISSION = 7

NVML_DEVICE_NAME_BUFFER_SIZE = 96
NVML_DEVICE_UUID_BUFFER_SIZE = 96

# ---------------------------------------------------------------------------
# C struct definitions
# ---------------------------------------------------------------------------


class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlUtilization(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


class NvmlProcessInfo(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
    ]


# ---------------------------------------------------------------------------
# NVML function signatures
# ---------------------------------------------------------------------------


def _setup_nvml(lib) -> None:
    """Declare argtypes/restype for all NVML functions used."""
    # Init / shutdown
    lib.nvmlInit.restype = ctypes.c_int
    lib.nvmlShutdown.restype = ctypes.c_int

    # Device count
    lib.nvmlDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    lib.nvmlDeviceGetCount.restype = ctypes.c_int

    # Device handle
    lib.nvmlDeviceGetHandleByIndex.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    lib.nvmlDeviceGetHandleByIndex.restype = ctypes.c_int

    # Device name
    lib.nvmlDeviceGetName.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    lib.nvmlDeviceGetName.restype = ctypes.c_int

    # Device UUID
    lib.nvmlDeviceGetUUID.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    lib.nvmlDeviceGetUUID.restype = ctypes.c_int

    # Memory info
    lib.nvmlDeviceGetMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NvmlMemory),
    ]
    lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int

    # Utilization rates
    lib.nvmlDeviceGetUtilizationRates.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NvmlUtilization),
    ]
    lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int

    # Temperature
    lib.nvmlDeviceGetTemperature.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.nvmlDeviceGetTemperature.restype = ctypes.c_int

    # Power usage (milliwatts)
    lib.nvmlDeviceGetPowerUsage.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.nvmlDeviceGetPowerUsage.restype = ctypes.c_int

    # Power management limit (milliwatts)
    lib.nvmlDeviceGetPowerManagementLimit.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    lib.nvmlDeviceGetPowerManagementLimit.restype = ctypes.c_int

    # Compute running processes
    lib.nvmlDeviceGetComputeRunningProcesses.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(NvmlProcessInfo),
    ]
    lib.nvmlDeviceGetComputeRunningProcesses.restype = ctypes.c_int


# ---------------------------------------------------------------------------
# Process info helpers (pure Python, /proc filesystem)
# ---------------------------------------------------------------------------


def _read_proc_comm(pid: int) -> str | None:
    """Read process name from /proc/<pid>/comm."""
    try:
        path = f"/proc/{pid}/comm"
        with open(path, "r") as f:
            return f.read().strip()
    except (OSError, PermissionError):
        return None


def _read_proc_cmdline(pid: int) -> str | None:
    """Read full command line from /proc/<pid>/cmdline.

    Arguments are separated by null bytes; we replace them with spaces.
    Returns None on failure (process exited, permission denied, etc.).
    """
    try:
        path = f"/proc/{pid}/cmdline"
        with open(path, "rb") as f:
            raw = f.read()
        if not raw:
            return None
        # Replace null bytes with spaces
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except (OSError, PermissionError):
        return None


def _read_proc_uid(pid: int) -> int | None:
    """Read UID (owner) of /proc/<pid>/status. Returns None on failure."""
    try:
        path = f"/proc/{pid}/status"
        with open(path, "r") as f:
            for line in f:
                if line.startswith("Uid:"):
                    # "Uid:\t1000\t1000\t1000\t1000"
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except (OSError, PermissionError):
        pass
    return None


def _uid_to_name(uid: int) -> str | None:
    """Convert numeric UID to username."""
    try:
        import pwd

        return pwd.getpwuid(uid).pw_name
    except (ImportError, KeyError):
        return str(uid)


def _run_nvsmi_processes() -> dict[str, list[dict[str, Any]]]:
    """Fallback: run nvidia-smi to get GPU process info.

    NVML process queries may return NVML_ERROR_NO_PERMISSION when the
    current user cannot read other users' process details. nvidia-smi
    handles this via driver-level access, so we use it as a fallback.

    Returns: dict mapping GPU UUID -> list of {pid, name, used_memory_mb}
    """
    import subprocess

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    result: dict[str, list[dict[str, Any]]] = {}
    for line in output.decode("utf-8", errors="replace").strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",", 3)]
        if len(parts) < 4:
            continue
        gpu_uuid, pid_str, proc_name, mem_str = parts
        try:
            pid = int(pid_str)
            mem_mb = int(mem_str)
        except ValueError:
            continue
        if gpu_uuid not in result:
            result[gpu_uuid] = []
        result[gpu_uuid].append(
            {"pid": pid, "name": proc_name, "used_memory_mb": mem_mb}
        )
    return result


def _gpu_processes(
    lib,
    handle,
    gpu_uuid: str,
    own_user: str | None,
    nvsmi_data: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect processes for one GPU.

    Tries NVML first. Falls back to pre-fetched nvidia-smi data if
    NVML returns NVML_ERROR_NO_PERMISSION.
    """
    raw_procs: list[dict[str, Any]] = []
    use_nvsmi = False

    # Try NVML
    try:
        proc_count = ctypes.c_uint(0)
        rc = lib.nvmlDeviceGetComputeRunningProcesses(
            handle, ctypes.byref(proc_count), None
        )
        if rc == NVML_ERROR_NO_PERMISSION:
            use_nvsmi = True
        elif rc in (NVML_SUCCESS, NVML_ERROR_INSUFFICIENT_SIZE) and proc_count.value > 0:
            buf = (NvmlProcessInfo * proc_count.value)()
            rc2 = lib.nvmlDeviceGetComputeRunningProcesses(
                handle, ctypes.byref(proc_count), buf
            )
            if rc2 == NVML_SUCCESS:
                for j in range(proc_count.value):
                    pi = buf[j]
                    name = _read_proc_comm(pi.pid)
                    uid = _read_proc_uid(pi.pid)
                    username = _uid_to_name(uid) if uid is not None else None
                    gpu_mem = pi.usedGpuMemory
                    if gpu_mem >= (1 << 63):
                        gpu_mem = 0
                    raw_procs.append(
                        {
                            "pid": pi.pid,
                            "gpu_memory_mb": int(gpu_mem // (1024 * 1024)),
                            "name": name or "?",
                            "user": username,
                        }
                    )
    except Exception:
        pass

    # Fallback to nvidia-smi data. Fetch on demand if the pre-check
    # on GPU 0 didn't catch a permission issue on another GPU.
    if use_nvsmi:
        if nvsmi_data is None:
            nvsmi_data = _run_nvsmi_processes()
        for pi in nvsmi_data.get(gpu_uuid, []):
            # Resolve user from /proc for own/other classification
            uid = _read_proc_uid(pi["pid"])
            username = _uid_to_name(uid) if uid is not None else None
            raw_procs.append(
                {
                    "pid": pi["pid"],
                    "gpu_memory_mb": pi["used_memory_mb"],
                    "name": pi["name"],
                    "user": username,
                }
            )

    # -- Classify: own vs other --
    own_procs: list[dict[str, Any]] = []
    other_map: dict[str, dict[str, int]] = {}

    for rp in raw_procs:
        user = rp.get("user")
        if own_user and user == own_user:
            if not use_nvsmi:
                # NVML path: we already have user info, add cmdline
                cmdline = _read_proc_cmdline(rp["pid"])
            else:
                # nvidia-smi path: cmdline from /proc
                cmdline = _read_proc_cmdline(rp["pid"])
            own_procs.append(
                {
                    "pid": rp["pid"],
                    "gpu_memory_mb": rp["gpu_memory_mb"],
                    "name": rp["name"],
                    "user": user,
                    "cmdline": cmdline,
                }
            )
        else:
            key = user or "?"
            if key not in other_map:
                other_map[key] = {"count": 0, "mem": 0}
            other_map[key]["count"] += 1
            other_map[key]["mem"] += rp["gpu_memory_mb"]

    other_list = [
        {"user": u, "process_count": d["count"], "total_memory_mb": d["mem"]}
        for u, d in sorted(other_map.items())
    ]
    return own_procs, other_list


# ---------------------------------------------------------------------------
# Main probe logic
# ---------------------------------------------------------------------------


def probe(own_user: str | None = None) -> dict[str, Any]:
    """Collect GPU information via NVML and return as a dict.

    Args:
        own_user: If set, processes from this user are returned in full
                  detail (with cmdline); other users' processes are
                  aggregated per user.
    """
    t_start = time.monotonic()

    # Find and load libnvidia-ml
    lib_path = ctypes.util.find_library("nvidia-ml")
    if lib_path is None:
        # Try common locations directly
        for candidate in (
            "libnvidia-ml.so.1",
            "libnvidia-ml.so",
            "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
            "/usr/lib64/libnvidia-ml.so.1",
            "/usr/lib/libnvidia-ml.so.1",
        ):
            try:
                lib = ctypes.CDLL(candidate)
                lib_path = candidate
                break
            except OSError:
                continue
    else:
        lib = ctypes.CDLL(lib_path)

    if lib_path is None:
        return {
            "ok": False,
            "error": "Cannot find libnvidia-ml.so -- NVIDIA driver not installed?",
            "elapsed_ms": (time.monotonic() - t_start) * 1000,
        }

    _setup_nvml(lib)

    # Initialize NVML
    rc = lib.nvmlInit()
    if rc != NVML_SUCCESS:
        return {
            "ok": False,
            "error": f"nvmlInit failed with code {rc}",
            "elapsed_ms": (time.monotonic() - t_start) * 1000,
        }

    try:
        # Get GPU count
        count = ctypes.c_uint(0)
        rc = lib.nvmlDeviceGetCount(ctypes.byref(count))
        if rc != NVML_SUCCESS:
            return {
                "ok": False,
                "error": f"nvmlDeviceGetCount failed with code {rc}",
                "elapsed_ms": (time.monotonic() - t_start) * 1000,
            }

        gpus: list[dict[str, Any]] = []
        handle = ctypes.c_void_p()

        # Pre-check: does NVML allow process queries? If not, pre-fetch
        # nvidia-smi process data once for all GPUs.
        nvsmi_data: dict[str, list[dict[str, Any]]] | None = None
        if count.value > 0:
            proc_count = ctypes.c_uint(0)
            test_handle = ctypes.c_void_p()
            rc0 = lib.nvmlDeviceGetHandleByIndex(0, ctypes.byref(test_handle))
            rc1 = lib.nvmlDeviceGetComputeRunningProcesses(
                test_handle, ctypes.byref(proc_count), None
            )
            if rc1 == NVML_ERROR_NO_PERMISSION:
                nvsmi_data = _run_nvsmi_processes()

        for i in range(count.value):
            gpu: dict[str, Any] = {"index": i}

            # Get device handle
            rc = lib.nvmlDeviceGetHandleByIndex(i, ctypes.byref(handle))
            if rc != NVML_SUCCESS:
                gpu["error"] = f"get handle failed: {rc}"
                gpus.append(gpu)
                continue

            # Name
            try:
                name_buf = ctypes.create_string_buffer(NVML_DEVICE_NAME_BUFFER_SIZE)
                lib.nvmlDeviceGetName(handle, name_buf, NVML_DEVICE_NAME_BUFFER_SIZE)
                gpu["name"] = name_buf.value.decode("utf-8", errors="replace")
            except Exception:
                gpu["name"] = "unknown"

            # UUID
            try:
                uuid_buf = ctypes.create_string_buffer(NVML_DEVICE_UUID_BUFFER_SIZE)
                lib.nvmlDeviceGetUUID(handle, uuid_buf, NVML_DEVICE_UUID_BUFFER_SIZE)
                gpu["uuid"] = uuid_buf.value.decode("utf-8", errors="replace")
            except Exception:
                gpu["uuid"] = "unknown"

            # Memory
            try:
                mem = NvmlMemory()
                lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
                gpu["memory_total_mb"] = int(mem.total // (1024 * 1024))
                gpu["memory_used_mb"] = int(mem.used // (1024 * 1024))
                gpu["memory_free_mb"] = int(mem.free // (1024 * 1024))
            except Exception:
                gpu["memory_total_mb"] = 0
                gpu["memory_used_mb"] = 0
                gpu["memory_free_mb"] = 0

            # Utilization
            try:
                util = NvmlUtilization()
                lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util))
                gpu["utilization_gpu"] = util.gpu
                gpu["utilization_mem"] = util.memory
            except Exception:
                gpu["utilization_gpu"] = 0
                gpu["utilization_mem"] = 0

            # Temperature
            try:
                temp = ctypes.c_uint(0)
                lib.nvmlDeviceGetTemperature(
                    handle, NVML_TEMPERATURE_GPU, ctypes.byref(temp)
                )
                gpu["temperature_c"] = temp.value
            except Exception:
                gpu["temperature_c"] = 0

            # Power usage (NVML returns milliwatts)
            try:
                power = ctypes.c_uint(0)
                lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power))
                gpu["power_watts"] = round(power.value / 1000.0, 1)
            except Exception:
                gpu["power_watts"] = 0.0

            # Power limit
            try:
                power_limit = ctypes.c_uint(0)
                lib.nvmlDeviceGetPowerManagementLimit(
                    handle, ctypes.byref(power_limit)
                )
                gpu["power_limit_watts"] = round(power_limit.value / 1000.0, 1)
            except Exception:
                gpu["power_limit_watts"] = 0.0

            # Compute processes: own user in detail, others aggregated
            processes, other_users = _gpu_processes(
                lib, handle,
                gpu_uuid=gpu.get("uuid", ""),
                own_user=own_user,
                nvsmi_data=nvsmi_data,
            )
            gpu["processes"] = processes
            gpu["other_users"] = other_users
            gpus.append(gpu)

        elapsed = (time.monotonic() - t_start) * 1000

        return {"ok": True, "gpus": gpus, "elapsed_ms": round(elapsed, 1)}

    finally:
        lib.nvmlShutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--own-user", default=None, help="Highlight processes for this user")
    args = parser.parse_args()

    result = probe(own_user=args.own_user)
    json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
