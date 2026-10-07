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
    Skip tests requiring compiled C++/CUDA extension if rocketsim_cuda.pyd is not built.
    When the native extension is present, all tests execute normally.
    """
    if not _NATIVE_AVAILABLE:
        module_name = request.module.__name__
        pure_python_modules = [
            "test_parity_regression_guard",
            "test_dodge_adversarial",
            "test_multi_car_kickoff",
        ]
        if not any(pure in module_name for pure in pure_python_modules):
            pytest.skip("rocketsim_cuda native extension (.pyd) not compiled in build/")
