from __future__ import annotations

import pytest

from tests.cupy_test_utils import require_cupy_runtime


@pytest.fixture(scope="session")
def cupy_runtime():
    return require_cupy_runtime()


@pytest.fixture(scope="session")
def cp(cupy_runtime):
    return cupy_runtime.cp
