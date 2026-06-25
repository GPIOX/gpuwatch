#!/usr/bin/env python3
"""
NVML GPU probe — runs on remote server via SSH stdin, outputs JSON to stdout.

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


# ---------------------------------------------------------------------------
# Main probe logic
# ---------------------------------------------------------------------------


def probe() -> dict[str, Any]:
    """Collect GPU information via NVML and return as a dict."""
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
            "error": "Cannot find libnvidia-ml.so — NVIDIA driver not installed?",
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

            # Compute processes (also try graphics processes)
            processes: list[dict[str, Any]] = []
            try:
                proc_count = ctypes.c_uint(0)
                # First call to get count — may return INSUFFICIENT_SIZE when
                # count > 0 and buffer is NULL. Both are valid.
                rc = lib.nvmlDeviceGetComputeRunningProcesses(
                    handle, ctypes.byref(proc_count), None
                )
                if rc in (NVML_SUCCESS, NVML_ERROR_INSUFFICIENT_SIZE) and proc_count.value > 0:
                    buf = (NvmlProcessInfo * proc_count.value)()
                    rc2 = lib.nvmlDeviceGetComputeRunningProcesses(
                        handle, ctypes.byref(proc_count), buf
                    )
                    if rc2 == NVML_SUCCESS:
                        for j in range(proc_count.value):
                            pi = buf[j]
                            name = _read_proc_comm(pi.pid)
                            uid = _read_proc_uid(pi.pid)
                            gpu_mem = pi.usedGpuMemory
                            # NVML can return max uint64 for unavailable values
                            if gpu_mem >= (1 << 63):
                                gpu_mem = 0
                            processes.append(
                                {
                                    "pid": pi.pid,
                                    "gpu_memory_mb": int(
                                        gpu_mem // (1024 * 1024)
                                    ),
                                    "name": name or "?",
                                    "user": _uid_to_name(uid) if uid is not None else None,
                                }
                            )
            except Exception:
                pass

            gpu["processes"] = processes
            gpus.append(gpu)

        elapsed = (time.monotonic() - t_start) * 1000

        return {"ok": True, "gpus": gpus, "elapsed_ms": round(elapsed, 1)}

    finally:
        lib.nvmlShutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    result = probe()
    json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
