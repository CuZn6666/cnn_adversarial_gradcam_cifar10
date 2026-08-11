from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any

import pytest


@dataclass(frozen=True)
class CuPyRuntime:
    cp: Any
    version: str
    cuda_runtime_version: int
    device_count: int
    gpu_name: str
    simple_computation: float


def _decode_device_name(raw_name: Any) -> str:
    if isinstance(raw_name, bytes):
        return raw_name.decode("utf-8", errors="replace")
    return str(raw_name)


def load_cupy_runtime() -> CuPyRuntime:
    """Load CuPy and verify that a CUDA device can execute a small operation."""
    if importlib.util.find_spec("cupy") is None:
        raise RuntimeError("cupy is not installed.")

    import cupy as cp

    try:
        cuda_runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
    except Exception as error:  # pragma: no cover - requires CUDA failure.
        raise RuntimeError("CuPy is installed but CUDA runtime is unavailable.") from error

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as error:  # pragma: no cover - requires CUDA failure.
        raise RuntimeError("CuPy is installed but CUDA device query failed.") from error

    if device_count <= 0:
        raise RuntimeError("CuPy is installed but no CUDA GPU is visible.")

    try:
        properties = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = _decode_device_name(properties.get("name", "unknown"))
    except Exception:  # pragma: no cover - device name is diagnostic only.
        gpu_name = "unknown"

    try:
        values = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float32)
        result = cp.sum(values * values)
        cp.cuda.Stream.null.synchronize()
        simple_computation = float(cp.asnumpy(result))
    except Exception as error:  # pragma: no cover - requires CUDA failure.
        raise RuntimeError("CuPy allocation or computation failed.") from error

    return CuPyRuntime(
        cp=cp,
        version=str(cp.__version__),
        cuda_runtime_version=cuda_runtime_version,
        device_count=device_count,
        gpu_name=gpu_name,
        simple_computation=simple_computation,
    )


def require_cupy_runtime() -> CuPyRuntime:
    try:
        return load_cupy_runtime()
    except RuntimeError as error:
        pytest.skip(str(error))
