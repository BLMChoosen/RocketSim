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

if sys.platform == "win32":
    if os.path.isdir(build_dir):
        try:
            os.add_dll_directory(build_dir)
        except OSError:
            pass
    _cuda_candidates = [
        os.environ.get("CUDA_PATH", ""),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
    ]
    for _c in _cuda_candidates:
        if _c and os.path.isdir(os.path.join(_c, "bin")):
            try:
                os.add_dll_directory(os.path.join(_c, "bin"))
                break
            except OSError:
                pass

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
