"""Utility functions for momo-kibidango."""

from __future__ import annotations

import logging
import os
import re
import sys

import psutil


def get_device() -> str:
    """Detect the best available compute device.

    Returns ``"cuda"`` if NVIDIA GPUs are available, ``"mps"`` on Apple
    Silicon with Metal support, otherwise ``"cpu"``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_memory_gb() -> float:
    """Return current process RSS memory usage in gigabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def get_available_memory_gb(device: str) -> float:
    """Return available memory in GB for the given device.

    Parameters
    ----------
    device:
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns
    -------
    float
        Available memory in gigabytes.  For ``"cpu"`` this is the system's
        available RAM.  For ``"cuda"`` it queries the current default GPU.
        For ``"mps"`` it falls back to system RAM (Apple does not expose a
        dedicated API for Metal memory headroom).
    """
    if device == "cuda":
        try:
            import torch

            free, _total = torch.cuda.mem_get_info()
            return free / (1024 ** 3)
        except Exception:
            pass

    # For cpu / mps / fallback, report system available RAM.
    mem = psutil.virtual_memory()
    return mem.available / (1024 ** 3)


def estimate_model_memory_gb(model_id: str, dtype: str = "float16") -> float:
    """Heuristic estimate of GPU/RAM needed to load a model.

    The estimate is based on well-known parameter-count patterns found in
    model names (e.g. ``"7B"``, ``"0.5B"``, ``"70B"``).  If no pattern is
    matched a conservative 7 B-parameter default is assumed.

    Parameters
    ----------
    model_id:
        HuggingFace-style model identifier (e.g. ``"Qwen/Qwen2.5-7B-Instruct"``).
    dtype:
        Data type string.  Supported: ``"float32"``, ``"bfloat16"``,
        ``"float16"``, ``"int8"``, ``"int4"``.

    Returns
    -------
    float
        Estimated memory footprint in gigabytes (weights only, no KV-cache).
    """
    # Bytes per parameter for each dtype
    bytes_per_param: dict[str, float] = {
        "float32": 4.0,
        "bfloat16": 2.0,
        "float16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
    }
    bpp = bytes_per_param.get(dtype, 2.0)

    # Try to extract parameter count from the model name
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", model_id)
    if match:
        params_b = float(match.group(1))
    else:
        # Conservative fallback
        params_b = 7.0

    # params_b is in billions; multiply by 1e9 then by bytes, convert to GB
    memory_bytes = params_b * 1e9 * bpp
    memory_gb = memory_bytes / (1024 ** 3)

    # Add ~10% overhead for optimizer states / buffers during loading
    return round(memory_gb * 1.1, 2)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the package-level logger.

    Uses a structured format suitable for production log aggregation.
    Calling this function multiple times is safe; handlers are only added
    once.

    Parameters
    ----------
    level:
        Log level name (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).

    Returns
    -------
    logging.Logger
        The ``momo_kibidango`` root logger.
    """
    logger = logging.getLogger("momo_kibidango")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
