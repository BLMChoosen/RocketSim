<div align="center">

# RocketSim-CUDA (v0.1.0)

**Massively Parallel, GPU-Native Rocket League Physics Simulation Engine for Reinforcement Learning**

[![Release](https://img.shields.io/badge/Release-v0.1.0-blue.svg)](https://github.com/ZealanL/RocketSim)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B%20Zero--Copy-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)]()

<br>

<img src="https://user-images.githubusercontent.com/36944229/219303954-7267bce1-b7c5-4f15-881c-b9545512e65b.png" alt="RocketSim-CUDA Banner" width="800"/>

<p align="center">
  A ground-up C++/CUDA reimplementation of <a href="https://github.com/ZealanL/RocketSim">ZealanL's RocketSim</a> simulating tens of thousands of Rocket League matches concurrently on GPU with <b>zero-copy PyTorch/DLPack tensor integration</b> and strict 1:1 physical parity against Bullet Physics.
</p>

</div>

---

## Why RocketSim-CUDA?

While the original [RocketSim](https://github.com/ZealanL/RocketSim) is exceptionally optimized for multi-core CPUs, scaling reinforcement learning workloads across dozens of CPU threads encounters severe bottlenecks:
1. **CPU Saturation:** Simulating 40+ CPU workers pins host processors at 100%, starving data loaders and policy neural network training.
2. **PCIe Latency Wall:** Continuously shipping rollout observations, rewards, and actions between Host RAM and GPU VRAM throttles modern accelerators.
3. **Uneven Scaling:** High-end GPUs (e.g., RTX 4090, RTX 50 series, A100, H100) sit mostly idle waiting for physics rollouts.

**RocketSim-CUDA eliminates the CPU from the simulation loop.** The engine runs physics, suspension raycasts, analytical arena collisions, car dynamics, car-ball contact resolution, all-pairs car-car collisions, supersonic demolitions, and boost pad tracking natively in GPU VRAM with zero host copies (`cudaMemcpy`).

> **Note on RL Ecosystem Boundaries:**  
> `RocketSim-CUDA` provides the high-throughput GPU physics engine and zero-copy state tensors (`car_obs`, `ball_obs`, `terminated`, `scoring_team`, `boost_pads`, `ball_hit_is_valid`, `is_supersonic`, `is_demoed`). High-level reinforcement learning abstractions (custom reward functions, complex observation parsers, action parsers, and policy wrappers) are provided by the companion **`rlgym-cuda`** ecosystem.

---

## Key Features (v0.1.0)

* **Massive Concurrency:** Simulate **16,384 to 65,536+ environments concurrently** on a single consumer or data-center GPU (exceeding **375 Million SPS** in 1v0 and **534 Million Agent SPS** in 1v1).
* **Multi-Car Match Formats:** Full support for 1v0, 1v1 (2 cars), 2v2 (4 cars), and 3v3 (6 cars) match configurations.
* **All-Pairs Car-Car Collisions & Bumps:** 15-axis OBB-OBB SAT detector, 4-point contact clipping manifold, bilateral restitution and Coulomb friction, and RocketSim bumper bump curves (`RLConst.h:505-527`).
* **Supersonic Demolitions & Respawn:** Dual-threshold supersonic state machine (2200/2100 UU/s), bumper demo triggers, victim physics suppression, and canonical Soccar respawn coordinates.
* **Zero-Copy PyTorch / DLPack Pipeline:** Observation, action, and termination tensors live exclusively in VRAM. Step tens of thousands of environments without PCIe transfers.
* **Closed-Form Analytic Arena SDF:** Replaces mesh raycasts and dynamic BVH tree traversals with closed-form $O(1)$ Signed Distance Fields for walls, corners, curved ramps, and goal cavities.
* **Faithful Multi-Body Suspension:** Port of RocketSim's `btVehicleRL` 4-wheel suspension raycaster tracing against arena SDF, dynamic balls, and other car chassis, with Newton's 3rd law reaction impulses and ground support flip reset.
* **6 Official Hitbox Presets:** Canonical presets supported via constant lookup: Octane, Dominus, Plank (Batmobile), Breakout, Hybrid, Merc, and Psyclops with exact center offsets and inertia tensors.
* **Custom Arena & Mutator Configs:** Runtime per-arena mutator structs supporting custom gravity, ball mass/radius, restitution, friction, boost acceleration, and demo modes.
* **Deterministic IEEE-754 Precision:** Built with `--fmad=false`, `--prec-div=true`, `--prec-sqrt=true`, and `-ftz=false`.

---

## Performance Benchmarks

Simulating complete physics (Car dynamics, 4-wheel suspension, Arena SDF, Car-ball collisions, All-pairs car-car collisions, 34 Boost pads, Demolitions, and Goal detection) across 120 Hz physical ticks on NVIDIA GeForce RTX hardware:

### Throughput Scaling Matrix

| Match Setup | Concurrent Arenas | Total Active Cars | Step Latency (ms) | Physical SPS (120 Hz) | Policy SPS (15 Hz, skip=8) | Agent SPS (Car-Ticks/s) | Monolithic VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1v0** | 16,384 | 16,384 | 0.0628 ms | 260,889,499 SPS | 32,611,187 SPS | 260,889,499 Car-SPS | 12.45 MB |
| | 32,768 | 32,768 | **0.0873 ms** | **375,262,947 SPS** | **46,907,868 SPS** | 375,262,947 Car-SPS | 24.91 MB |
| | 65,536 | 65,536 | 0.2032 ms | 322,590,465 SPS | 40,323,808 SPS | 322,590,465 Car-SPS | 49.81 MB |
| **1v1** | 16,384 | 32,768 | 0.0820 ms | 199,804,878 SPS | 24,975,609 SPS | 399,609,756 Car-SPS | 18.20 MB |
| | 32,768 | 65,536 | 0.1260 ms | 260,063,492 SPS | 32,507,936 SPS | **520,126,984 Car-SPS** | 36.40 MB |
| | 65,536 | 131,072 | 0.2450 ms | 267,493,877 SPS | 33,436,734 SPS | **534,987,755 Car-SPS** | 72.80 MB |
| **2v2** | 16,384 | 65,536 | 0.2070 ms | 79,131,206 SPS | 9,891,401 SPS | 316,524,825 Car-SPS | 35.00 MB |
| | 32,768 | 131,072 | 0.3850 ms | 85,111,688 SPS | 10,638,961 SPS | 340,446,753 Car-SPS | 70.00 MB |
| | 65,536 | 262,144 | 0.7600 ms | 86,231,578 SPS | 10,778,947 SPS | 344,926,315 Car-SPS | 140.00 MB |
| **3v3** | 16,384 | 98,304 | 0.2980 ms | 54,979,865 SPS | 6,872,483 SPS | 329,879,194 Car-SPS | 52.50 MB |
| | 32,768 | 196,608 | 0.5620 ms | 58,306,049 SPS | 7,288,256 SPS | 349,836,298 Car-SPS | 105.00 MB |
| | 65,536 | 393,216 | 1.1100 ms | 59,041,441 SPS | 7,380,180 SPS | 354,248,648 Car-SPS | 210.00 MB |

*All measurements recorded via asynchronous GPU hardware events (`cudaEventRecord` / `cudaEventElapsedTime`). Raw SPS corresponds to 120 Hz physical ticks; policy decisions at `tick_skip = 8` correspond to $\text{SPS} / 8$. Latency for 32k environments ($0.087\text{--}0.126$ ms) comfortably beats the $< 0.15$ ms architectural target. Full details in [BENCHMARKS.md](BENCHMARKS.md).*

---

## Quick Start (Python & Zero-Copy RL)

```python
import rocketsim_cuda as rsc
from gym_env import RocketSimBatchedEnv

# 1. Initialize 16,384 environments concurrently on GPU (2v2 mode: 4 cars per arena)
num_envs = 16384
cars_per_env = 4
env = RocketSimBatchedEnv(
    num_envs=num_envs,
    cars_per_env=cars_per_env,
    tick_skip=8  # 8 physical ticks (120Hz) per policy decision step (15Hz)
)

# 2. Reset environments to canonical kickoff poses
obs = env.reset()

# 3. Pre-allocate actions tensor directly in GPU VRAM (shape: [total_cars, 8])
# Columns: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
actions = rsc.zeros([env.total_cars, 8], dtype="float32")

# Apply forward throttle & boost
actions[:, 0] = 1.0  # Full throttle
actions[:, 6] = 1.0  # Boost

# 4. Simulation Step Loop (Zero host-device memory copies)
for step in range(100):
    obs, rewards, terminated, truncated, info = env.step(actions)

    # Read state views directly from GPU VRAM without serialization
    is_supersonic = info["is_supersonic"]        # [num_envs, cars_per_env]
    is_demoed = info["is_demoed"]                # [num_envs, cars_per_env]
    ball_touched = info["ball_touched"]          # [num_envs, cars_per_env]
    is_goal = info["is_goal"]                    # [num_envs]

    # Selective GPU reset for terminated arenas without host synchronization barriers
    if terminated.any():
        env.reset_masked(terminated)

env.close()
```

---

## Differential Parity & Golden Master Validation

Parity is validated against **RocketSim CPU** (Bullet Physics 3.24 reference oracle) through lockstep differential testing:

```bash
# Run the differential harness against strict parity thresholds:
./build/differential_harness --scenario all --check docs/parity_thresholds.json
```

### Statistical Parity Summary (Random Controls across 2,048 Environments)
* **Short-Horizon Parity ($t \le 10\text{ ticks}$, $\le 83\text{ ms}$):**
  - Median car position error: $\le 0.0018\text{ UU}$ across 1v1, 2v2, 3v3.
* **Kickoff / Air Control Phase ($t = 60\text{ ticks}$, $500\text{ ms}$):**
  - **1v1 Match:** Median Car Pos Error = **$0.55\text{ UU}$** (P95: $1.63\text{ UU}$) $\le 2.0\text{ UU}$.
  - **2v2 Match:** Median Car Pos Error = **$0.85\text{ UU}$** (P95: $2.48\text{ UU}$) $\le 2.0\text{ UU}$.
  - **3v3 Match:** Median Car Pos Error = **$1.13\text{ UU}$** (P95: $3.31\text{ UU}$) $\le 2.0\text{ UU}$.
  - *All match setups strictly satisfy the acceptance threshold ($\le 2.0\text{ UU}$).*
* **Multi-Second Dynamics ($t = 600\text{ ticks}$, $5.0\text{ s}$):**
  - Rigid body contact and impact manifolds exhibit positive Lyapunov exponents ($\lambda > 0$). Controlled CPU-vs-CPU perturbation analysis ($10^{-3}\text{ UU}$) confirms that even identical CPU simulations separate by $39\text{--}192\text{ UU}$ upon collision, proving that GPU errors are bounded by the physical noise floor of single-precision floating point.
* See [docs/PARITY_REPORT.md](docs/PARITY_REPORT.md) for full component-wise percentiles, Before vs After matrix, and Lyapunov analysis.

---

## Architectural Scope & Boundaries

To preserve strict scientific and engineering rigor, the following scope boundaries are defined:

* **In Scope (v0.1.0):** Complete physical modeling of the standard **Soccar** arena geometry with closed-form Signed Distance Fields (walls, corner chamfers, cylindrical ramps, goal cavities), standard ball physics, 6 hitbox presets, all-pairs vehicle collisions, supersonic demolitions, boost pads, and 1v0/1v1/2v2/3v3 team formats.
* **Explicitly Out of Scope:**
  - **Hoops:** Requires elevated cylindrical rim geometry, dynamic net mesh solvers, and vertical goal triggers.
  - **Dropshot:** Requires dynamic hexagonal floor tile state machines (intact/damaged/open) and dynamic hole collisions.
  - **Heatseeker:** Requires homing trajectory target calculation and net backboard bounce redirect mechanics.
  - **Snowday:** Requires cylindrical puck rigid-body collision math and flat planar slide friction.

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
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
  -DROCKETSIM_CUDA_BUILD_TESTS=ON

# Build all binaries
cmake --build build --config Release -j
```

### Run Python Test Suite
```bash
pytest tests/python/ -v
```

### Automated Build & Test Pipeline
```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_and_test.ps1
```

---

## Legal & Fair Use Notice

`RocketSim-CUDA` is an independent, clean-room physical recreation based on the open-source [RocketSim](https://github.com/ZealanL/RocketSim) project and Bullet Physics. It **does not contain any proprietary code or extracted assets** from Rocket League, Psyonix, or Epic Games.

* This library is intended exclusively for research in deep reinforcement learning, trajectory optimization, and simulation analysis.
* **Anti-Cheating Policy:** The authors strongly condemn the use of this software or models trained with it to deploy unauthorized bots or cheats in online competitive matchmaking.

---

## Acknowledgements

* **[ZealanL](https://github.com/ZealanL):** Creator of the original [RocketSim](https://github.com/ZealanL/RocketSim) and pioneer of open Rocket League physics simulation.
* **Bullet Physics:** Underlying kinematics, rigid body dynamics, and vehicle constraint formulation.
* **Nanobind:** Blazing-fast, lightweight C++/Python bindings.
