import os
import sys
import pytest

# Ensure build, bindings, and python package directories are in sys.path
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "bindings"))
python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
for d in [build_dir, src_dir, python_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

try:
    import rocketsim_cuda
    _NATIVE_AVAILABLE = hasattr(rocketsim_cuda, "SimContext")
except Exception:
    _NATIVE_AVAILABLE = False


@pytest.fixture(autouse=True)
def require_native_extension(request):
    """
    Enforce that rocketsim_cuda native extension (.pyd) is compiled.
    Tests must fail and abort with an explicit error if the binary is missing.
    No test skipping permitted.
    """
    if not _NATIVE_AVAILABLE:
        raise RuntimeError(
            "rocketsim_cuda native extension (.pyd) is NOT compiled or importable from build/. "
            "Please build the project first via scripts/build_and_test.ps1 or cmake."
        )
