"""
RocketSim-CUDA: Ultra-parallel native C++/CUDA RocketSim physics engine with zero-copy PyTorch/DLPack tensors.
"""

import sys
import os
import importlib.util

_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "bindings"))
for d in [_build_dir, _src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

if sys.platform == "win32":
    if os.path.isdir(_build_dir):
        try:
            os.add_dll_directory(_build_dir)
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

# Load binary extension
_pyd = None
if os.path.isdir(_build_dir):
    for fname in os.listdir(_build_dir):
        if fname.startswith("rocketsim_cuda") and (fname.endswith(".pyd") or fname.endswith(".so")):
            _pyd = os.path.join(_build_dir, fname)
            break

if _pyd:
    spec = importlib.util.spec_from_file_location("rocketsim_cuda", _pyd)
    if spec and spec.loader:
        _native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_native)
        for k, v in _native.__dict__.items():
            if not k.startswith("__"):
                globals()[k] = v

from gym_env import RocketSimBatchedEnv

__all__ = [
    "SimContext",
    "GpuTensorView",
    "GpuEvent",
    "Event",
    "RocketSimBatchedEnv",
    "zeros",
    "get_vram_info",
    "TICK_RATE",
    "DELTA_TIME",
]
