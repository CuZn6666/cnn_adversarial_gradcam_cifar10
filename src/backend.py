"""Array backend helpers for NumPy-first tensor code."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import (
    sliding_window_view as _numpy_sliding_window_view,
)

try:  # pragma: no cover - exercised only when CuPy is installed.
    import cupy as _cupy
except ImportError:  # pragma: no cover - local reference environment.
    _cupy = None


ArrayModule = Any


def cupy_available() -> bool:
    """Return whether CuPy can be selected as an optional backend."""
    return _cupy is not None


def backend_name(backend: ArrayModule) -> str:
    """Return a stable display name for a supported array backend."""
    resolved = resolve_backend(backend)
    if resolved is np:
        return "numpy"
    return "cupy"


def resolve_backend(backend: str | ArrayModule = "numpy") -> ArrayModule:
    """Resolve a backend name or module to an array namespace."""
    if isinstance(backend, str):
        normalized = backend.lower()
        if normalized == "numpy":
            return np
        if normalized == "cupy":
            if _cupy is None:
                raise ImportError(
                    "CuPy backend requested, but cupy is not installed."
                )
            return _cupy
        raise ValueError("backend must be 'numpy', 'cupy', numpy, or cupy.")

    if backend is np or getattr(backend, "__name__", None) == "numpy":
        return np
    if getattr(backend, "__name__", None) == "cupy":
        if _cupy is None:
            raise ImportError(
                "CuPy backend requested, but cupy is not installed."
            )
        return _cupy
    raise ValueError("backend must be 'numpy', 'cupy', numpy, or cupy.")


def is_cupy_array(value: Any) -> bool:
    """Return whether value is a CuPy ndarray."""
    return _cupy is not None and isinstance(value, _cupy.ndarray)


def get_array_module(*arrays: Any) -> ArrayModule:
    """Return the array module used by the provided arrays."""
    for array in arrays:
        if is_cupy_array(array):
            return _cupy
    return np


def ensure_same_backend(*arrays: Any) -> ArrayModule:
    """Validate that arrays in one tensor operation use one backend."""
    detected: list[ArrayModule] = []
    for array in arrays:
        if not hasattr(array, "shape"):
            continue
        module = get_array_module(array)
        if module not in detected:
            detected.append(module)

    if len(detected) > 1:
        raise TypeError(
            "Mixed NumPy/CuPy arrays are not allowed in one tensor "
            "operation. Use to_backend(...) at an explicit boundary."
        )
    return detected[0] if detected else np


def ensure_backend_array(
    array: Any,
    backend: str | ArrayModule,
    *,
    name: str = "array",
) -> ArrayModule:
    """Validate that an array belongs to the selected backend."""
    expected = resolve_backend(backend)
    actual = ensure_same_backend(array)
    if actual is not expected:
        raise TypeError(
            f"{name} uses {backend_name(actual)} but expected "
            f"{backend_name(expected)}. Use to_backend(...) at an explicit "
            "CPU/GPU boundary."
        )
    return expected


def to_numpy(array: Any) -> np.ndarray:
    """Convert an array or scalar to a NumPy ndarray at an explicit boundary."""
    if is_cupy_array(array):
        return _cupy.asnumpy(array)
    return np.asarray(array)


def to_backend(
    array: Any,
    backend: str | ArrayModule,
    *,
    dtype: Any | None = None,
) -> Any:
    """Convert an array to the selected backend at an explicit boundary."""
    xp = resolve_backend(backend)
    if xp is np:
        return np.asarray(to_numpy(array), dtype=dtype)
    return xp.asarray(array, dtype=dtype)


def to_python_bool(value: Any) -> bool:
    """Convert a backend scalar to a Python bool at a synchronization point."""
    if is_cupy_array(value):
        value = _cupy.asnumpy(value)
    if isinstance(value, np.ndarray):
        return bool(value.item())
    return bool(value)


def to_python_float(value: Any) -> float:
    """Convert a backend scalar to a Python float at a synchronization point."""
    if is_cupy_array(value):
        value = _cupy.asnumpy(value)
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def to_python_int(value: Any) -> int:
    """Convert a backend scalar to a Python int at a synchronization point."""
    if is_cupy_array(value):
        value = _cupy.asnumpy(value)
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def isfinite_all(array: Any) -> bool:
    """Return whether all array elements are finite."""
    xp = get_array_module(array)
    return to_python_bool(xp.isfinite(array).all())


def sliding_window_view(
    array: Any,
    window_shape: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Any:
    """Backend-aware sliding-window view."""
    xp = get_array_module(array)
    if xp is np:
        return _numpy_sliding_window_view(
            array,
            window_shape,
            axis=axis,
        )

    try:
        cupy_sliding_window_view = xp.lib.stride_tricks.sliding_window_view
    except AttributeError as error:  # pragma: no cover - CuPy-version guard.
        raise RuntimeError(
            "Selected backend does not provide sliding_window_view."
        ) from error
    return cupy_sliding_window_view(array, window_shape, axis=axis)
