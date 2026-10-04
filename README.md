<div align="center">

# RocketSim-CUDA

**Massively Parallel, GPU-Native Rocket League Physics Simulation Engine**

[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B%20Zero--Copy-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)]()

<br>

<img src="https://user-images.githubusercontent.com/36944229/219303954-7267bce1-b7c5-4f15-881c-b9545512e65b.png" alt="RocketSim-CUDA Banner" width="800"/>

<p align="center">
  A ground-up C++/CUDA reimplementation of <a href="https://github.com/ZealanL/RocketSim">ZealanL's RocketSim</a> designed to simulate tens of thousands of Rocket League matches concurrently on GPU with <b>zero-copy PyTorch/DLPack tensor integration</b>.
</p>

</div>

---

## Why RocketSim-CUDA?

While the original [RocketSim](https://github.com/ZealanL/RocketSim) is exceptionally optimized for multi-core CPUs, scaling reinforcement learning workloads across dozens of CPU threads encounters severe bottlenecks:
1. **CPU Saturation:** Simulating 40+ CPU workers pins host processors at 100%, starving data loaders and policy neural network training.
2. **PCIe Latency Wall:** Continuously shipping rollout observations, rewards, and actions between Host RAM and GPU VRAM throttles modern accelerators.
3. **Uneven Scaling:** High-end GPUs (e.g., RTX 4090, RTX 50 series, A100, H100) sit mostly idle waiting for physics rollouts.

**RocketSim-CUDA eliminates the CPU from the simulation loop.** The engine runs physics, suspension raycasts, analytical arena collisions, car dynamics, car-ball contact resolution, and boost pad tracking natively in GPU VRAM with zero host copies (`cudaMemcpy`).

> **Note on RL Ecosystem Boundaries:**  
> `RocketSim-CUDA` provides the core GPU physics backend and zero-copy state tensors (`car_obs`, `ball_obs`, `terminated`, `scoring_team`, `boost_pads`, `ball_hit_is_valid`). High-level reinforcement learning abstractions (custom reward functions, complex observation parsers, action parsers, and policy wrappers) are provided by the companion **`rlgym-cuda`** ecosystem.

---

## Key Features

* **Massive Concurrency:** Simulate **16,384 to 65,536+ environments concurrently** on a single consumer or data-center GPU.
* **Zero-Copy PyTorch / DLPack Pipeline:** Observation, action, and termination tensors live exclusively in VRAM. Step tens of thousands of environments without PCIe transfers.
* **Closed-Form Analytic Arena SDF:** Replaces mesh raycasts and dynamic BVH tree traversals with closed-form $O(1)$ Signed Distance Fields for walls, corners, curved ramps, and goal cavities.
* **Faithful Suspension & Dynamics:** Port of RocketSim's `btVehicleRL` 4-wheel suspension raycaster, bilateral friction model, aerial torque, jumps, dodges/flips, auto-roll, and auto-flip.
* **3D Car-Ball Contact Solver:** Continuous OBB vs. Sphere contact detection with impulse exchange, surface friction, restitution, and RocketSim piecewise extra hit impulse.
* **Hardware-Optimal Memory (SoA):** Pure Structure of Arrays layout (`alignas(16)`) ensuring contiguous, coalesced 128-byte transactions across warps.
* **Deterministic IEEE-754 Precision:** Built with `--fmad=false`, `--prec-div=true`, `--prec-sqrt=true`, and `-ftz=false`.

---

## Performance Benchmarks

Simulating complete physics (Car dynamics, 4-wheel suspension, Arena SDF, Car-ball collisions, 34 Boost pads, and Goal detection) across 120 Hz physical ticks on an **NVIDIA GeForce RTX 5060 (8 GB VRAM)**:

### 1v0 Solo Baseline (1 Car per Arena)

| Environments | Total Cars | Step Latency (ms) | Physical Throughput (SPS) | VRAM Pool (MB) |
| :---: | :---: | :---: | :---: | :---: |
| **4,096** | 4,096 | 0.0289 ms | **141,955,657 SPS** | 3.11 MB |
| **16,384** | 16,384 | 0.0628 ms | **260,889,499 SPS** | 12.45 MB |
| **32,768** | 32,768 | 0.0895 ms | **365,956,080 SPS** | 24.91 MB |
| **65,536** | 65,536 | 0.2042 ms | **320,888,630 SPS** | 49.81 MB |

### 2v2 Full Match Simulation (4 Cars per Arena)

> **Important Scope Limitation:** In Milestone 4, 2v2 arenas simulate 4 autonomous cars with independent raycast suspensions, analytical arena collisions, car-ball impacts, and boost consumption. Car-on-car collisions and demolitions are scheduled for Milestone 5; hence throughput does not include inter-car contact graph solving and cannot be directly compared to CPU RocketSim.

| Environments | Total Cars | Step Latency (ms) | Environment SPS | Agent SPS (Car-Ticks/s) | VRAM Pool (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | 4,096 | 0.0904 ms | 11,331,097 SPS | **45,324,387 Car-SPS** | 2.19 MB |
| **4,096** | 16,384 | 0.0933 ms | 43,889,777 SPS | **175,559,109 Car-SPS** | 8.75 MB |
| **8,192** | 32,768 | 0.1037 ms | 78,988,858 SPS | **315,955,433 Car-SPS** | 17.50 MB |
| **16,384** | 65,536 | 0.2070 ms | 79,131,206 SPS | **316,524,825 Car-SPS** | 35.00 MB |

*All measurements recorded via asynchronous GPU hardware events (`cudaEventRecord` / `cudaEventElapsedTime`). Raw SPS corresponds to 120 Hz physical ticks; policy decisions at `tick_skip = 8` correspond to $\text{SPS} / 8$. See [BENCHMARKS.md](BENCHMARKS.md) for full report.*

---

## Quick Start (Python & Zero-Copy RL)

```python
import rocketsim_cuda as rsc
from gym_env import RocketSimBatchedEnv

# 1. Initialize 16,384 environments concurrently on GPU (1v0 mode, 30 Hz action frequency)
num_envs = 16384
env = RocketSimBatchedEnv(
    num_envs=num_envs,
    cars_per_env=1,
    tick_skip=4  # 4 physical ticks (120Hz) per environment step
)

# 2. Reset environments to kickoff poses
obs = env.reset()

# 3. Pre-allocate actions tensor directly in GPU VRAM (shape: [total_cars, 8])
# Columns: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
actions = rsc.zeros([env.total_cars, 8], dtype="float32")

# Apply full forward throttle & boost
for c in range(env.total_cars):
    actions[c, 0] = 1.0  # Throttle
    actions[c, 6] = 1.0  # Boost

# 4. Simulation Step Loop (Zero host-device copies)
for step in range(100):
    obs, rewards, terminated, truncated, info = env.step(actions)

    # Read contact and game state tensors directly from GPU VRAM
    ball_hits = info["ball_hit_is_valid"]
    is_goal = info["is_goal"]

    # Selective GPU reset for terminated arenas without CPU barriers
    if terminated.any():
        env.reset_masked(terminated)

env.close()
```

Run the touch-ball example directly:
```bash
python examples/smoke_touch_ball.py
```

---

## Differential Parity & Golden Master Validation

Parity is validated against **RocketSim CPU** (Bullet Physics 3.24 reference oracle) through lockstep differential testing:

```bash
# Run the differential harness across all scenarios with windowed reporting:
./build/differential_harness --scenario all --ticks 10000 --envs 1 --report
```

### Parity Highlights
* **Short-Horizon Micro-Parity ($t \le 10\text{ ticks}$, $\le 83\text{ ms}$):**
  - Car idle on ground: $\Vert\Delta\mathbf{p}\Vert_\infty \le 7.63 \times 10^{-6}\text{ UU}$, velocity delta $\le 3.82 \times 10^{-6}\text{ UU/s}$, quaternion delta $\le 5.96 \times 10^{-8}$.
  - Ground throttle: $\Vert\Delta\mathbf{p}\Vert_\infty \le 9.77 \times 10^{-4}\text{ UU}$ (exact 2-ULP precision limit at $|Y| > 4600\text{ UU}$).
  - Free ball flight: $\Vert\Delta\mathbf{p}\Vert_\infty \le 3.05 \times 10^{-5}\text{ UU}$, velocity delta $\le 2.44 \times 10^{-4}\text{ UU/s}$.
* **Kickoff Goalie Collision Gate ($4608\text{ UU}$ Supersonic Drive):**
  - First touch tick: **Tick 314 on GPU vs Tick 315 on CPU** ($1\text{ tick}$ delta across 315 ticks = 99.7% temporal parity).
  - Impact speed: $2018\text{ UU/s}$ GPU vs $2033\text{ UU/s}$ CPU ($0.7\%$ delta).
  - Rebound ball speed: $2907\text{ UU/s}$ GPU vs $2863\text{ UU/s}$ CPU ($1.5\%$ impulse fidelity).
* **Long-Horizon Multi-Second Dynamics ($t > 120\text{ ticks}$, $> 1\text{ s}$):**
  - Car suspension resting height reaches an exact analytical equilibrium ($Z \approx 17.03\text{ UU}$) with delta $\le 0.00488\text{ UU}$ ($< 5\text{ mm}$) that remains strictly bounded without drift across 10,000 continuous ticks ($83.3\text{ s}$).
  - Rigid body collisions against curved arena surfaces and ball impacts have positive Lyapunov exponents ($\lambda > 0$). In single-precision float32, microscopic rounding differences naturally separate macroscopic trajectories after multiple wall bounces.
* Full empirical measurements, component-wise delta tables, and CPU-vs-CPU perturbation analysis are documented in [docs/PARITY_REPORT.md](docs/PARITY_REPORT.md).

---

## Roadmap & Scope
* **Milestone 4 (Current):** Complete 1v0 physics pipeline for single-agent RL training (full Octane suspension and dynamics, ball flight and aerodynamics, closed-form SDF arena collisions, 3D OBB-sphere car-ball contact impulses, 34 boost pads, 5 Soccar kickoff spawns, goal scoring thresholds, zero-copy PyTorch/DLPack tensors).
* **Milestone 5 (Upcoming):** Car-on-car OBB-OBB collisions, demolitions, supersonic demo timers, and multi-agent 2v2/3v3 self-play.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      GPU VRAM (cuda:0)                       │
│                                                              │
│   ┌─────────────────────┐          ┌─────────────────────┐   │
│   │   RocketSim-CUDA    │ ◄──────► │ PyTorch / DLPack    │   │
│   │   SimContext SoA    │  Zero    │ Rollout Buffers     │   │
│   └──────────┬──────────┘  Copy    └──────────┬──────────┘   │
│              │                                │              │
│   ┌──────────▼──────────┐          ┌──────────▼──────────┐   │
│   │ CUDA Physics Kernel │          │  PPO Neural Network │   │
│   │ (SDF, Car, Ball)    │          │ (Forward/Backward)  │   │
│   └─────────────────────┘          └─────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
▲
No Host PCIe Traffic (0 bytes transferred during step loop)
▼
┌──────────────────────────────────────────────────────────────┐
│                      Host CPU (1 Thread)                     │
│               Dispatches async CUDA streams only             │
└──────────────────────────────────────────────────────────────┘
```

---

## Installation & Building

### Prerequisites
* **NVIDIA GPU:** Compute Capability $\ge 7.5$ (Turing, Ampere, Ada Lovelace, Blackwell).
* **CUDA Toolkit:** Version 12.0 or higher.
* **Compiler:** C++20 compliant compiler (GCC 11+, Clang 14+, or MSVC 2022 v17.4+).
* **Build System:** CMake $\ge 3.24$ and **Ninja**.
* **Python:** Python 3.9 through 3.14 with `nanobind`.

### Build Native C++/CUDA Targets
```bash
# Configure with Release optimization
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"

# Build all binaries
cmake --build build --config Release -j
```

### Run Python Test Suite
```bash
python -m pytest tests/python/ -v
```

---

## Legal & Fair Use Notice

`RocketSim-CUDA` is an independent, clean-room physical recreation based on the open-source [RocketSim](https://github.com/ZealanL/RocketSim) project and Bullet Physics. It **does not contain any proprietary code or extracted assets** from Rocket League, Psyonix, or Epic Games.

* This library is intended exclusively for research in deep reinforcement learning, trajectory optimization, and simulation analysis.
* **Anti-Cheating Policy:** The authors strongly condemn the use of this software or models trained with it to deploy unauthorized bots or cheats in online competitive matchmaking.

---

## Acknowledgements

* **[ZealanL](https://github.com/ZealanL):** Creator of the original [RocketSim](https://github.com/ZealanL/RocketSim) and pioneer of open Rocket League physics simulation.
* **Bullet Physics:** Underlying kinematics and constraint formulation.
* **Nanobind:** Blazing-fast, lightweight C++/Python bindings.
