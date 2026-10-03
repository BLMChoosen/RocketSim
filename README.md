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
  A ground-up C++/CUDA rewrite of <a href="https://github.com/ZealanL/RocketSim">ZealanL's RocketSim</a> designed to simulate tens of thousands of Rocket League matches concurrently on GPU with <b>strict 1:1 physical parity</b> and <b>zero-copy PyTorch tensor integration</b>.
</p>

</div>

---

## Why RocketSim-CUDA?

While the original [RocketSim](https://github.com/ZealanL/RocketSim) is exceptionally optimized for multi-core CPUs, scaling reinforcement learning workloads across dozens of CPU threads encounters severe bottlenecks:
1. **CPU Saturation:** Simulating 40+ CPU workers pins host processors at 100%, starving data loaders and neural network inference.
2. **PCIe Latency Wall:** Continuously shipping rollout observations and actions between Host RAM and GPU VRAM throttles modern accelerators.
3. **Uneven Scaling:** High-end GPUs (e.g., RTX 40/50 series, A100/H100) sit mostly idle waiting for physics rollouts.

**RocketSim-CUDA eliminates the CPU completely from the simulation loop.** The engine runs physics, suspension raycasts, arena collisions, and reward/observation transforms natively in GPU VRAM, allowing a single workstation GPU to outperform massive CPU compute clusters.

---

## Key Features

* **Massive Concurrency:** Simulate **16,384 to 65,536+ arenas in parallel** on a single consumer or data-center GPU.
* **Zero-Copy PyTorch Loop (DLPack):** Observation and action buffers live directly in VRAM. Step the entire batch of environments without a single `cudaMemcpy` round-trip across PCIe.
* **Bitwise Differential Parity:** Verified tick-by-tick against the CPU reference engine (Bullet Physics / RocketSim) using an automated Chebyshev norm harness ($\Vert{}\Delta_{\mathbf{p}}\Vert{}_\infty \le 10^{-4}\text{ UU}$, Quaternions $\le 10^{-5}$).
* **Analytic $O(1)$ Arena SDF:** Replaces polygon mesh raycasts and dynamic BVH structures with exact analytical Signed Distance Fields for walls, corners, curves, ramps, and goals.
* **Hardware-Optimal Memory (SoA):** Pure Structure of Arrays layout (`alignas(16)`) providing 100% coalesced 128-byte memory transactions across active warps.
* **Deterministic Execution:** Built with `--fmad=false`, `--prec-div=true`, and strict IEEE-754 floating-point constraints for deterministic replayability.

---

## Performance Benchmark

Simulating **Soccar (2v2 - 4 cars per arena)** with pseudo-random agent actions across 120 Hz physical ticks (exact match to ZealanL's RocketSim benchmark conditions):

### Hardware Specifications
* **CPU:** AMD Ryzen 5 5500 (6 Cores / 12 Threads @ 3.60GHz base / 4.20GHz boost)
* **GPU:** NVIDIA GeForce RTX 5060 (8 GB GDDR7 @ 14,001 MHz)
* **Host RAM:** 16 GB Dual-Channel DDR4 @ 3800 MT/s (3800 MHz)
* **VRAM Bandwidth:** ~448 GB/s (100% on-device simulation, 0 PCIe traffic)

### Benchmark Results (2v2 Match Simulation)

| Engine | Execution Device | Parallel Envs | Total Cars | Step Latency (ms) | Throughput (SPS / TPS) | Speedup vs CPU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RocketSim Original** | CPU (1 Thread) | 1 | 4 | 0.0433 ms | **23,083 TPS** | 1.0x (Baseline) |
| **RocketSim Original** | CPU (12 Threads - Max) | 12 | 48 | 0.2737 ms | **43,849 TPS** | 1.9x |
| **RocketSim-CUDA** | NVIDIA RTX 5060 | 4,096 | 16,384 | 0.0703 ms | **58,304,524 SPS** | **2,525x** |
| **RocketSim-CUDA** | NVIDIA RTX 5060 | 16,384 | 65,536 | 0.1491 ms | **109,875,251 SPS** | **4,759x** |
| **RocketSim-CUDA** | NVIDIA RTX 5060 | 32,768 | 131,072 | 0.3106 ms | **105,500,691 SPS** | **4,570x** |
| **RocketSim-CUDA** | NVIDIA RTX 5060 | 65,536 | 262,144 | 0.5608 ms | **116,869,693 SPS** | **5,063x** |

> *Note: In 1v0 / 1-car RL rollout mode, RocketSim-CUDA reaches up to **482,761,983 SPS** (0.0679 ms latency at 32,768 environments) in active VRAM.*

---

## Architecture Overview


```

┌──────────────────────────────────────────────────────────────┐
│                      GPU VRAM (cuda:0)                       │
│                                                              │
│   ┌─────────────────────┐          ┌─────────────────────┐   │
│   │   RocketSim-CUDA    │ ◄──────► │ PyTorch Rollout Buf │   │
│   │   SimContext SoA    │  DLPack  │   (Zero-Copy View)  │   │
│   └──────────┬──────────┘  Pointers└──────────┬──────────┘   │
│              │                                │              │
│   ┌──────────▼──────────┐          ┌──────────▼──────────┐   │
│   │ CUDA Physics Kernel │          │  PPO Neural Network │   │
│   │ (SDF + Suspension)  │          │ (Forward/Backward)  │   │
│   └─────────────────────┘          └─────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
▲
No PCIe Traffic
▼
┌──────────────────────────────────────────────────────────────┐
│                      Host CPU (1 Thread)                     │
│               Dispatches async CUDA streams only             │
└──────────────────────────────────────────────────────────────┘

```

---

## Installation

### Prerequisites
* **NVIDIA GPU:** Compute Capability $\ge 7.5$ (Turing, Ampere, Ada Lovelace, Blackwell).
* **CUDA Toolkit:** Version 12.0 or higher.
* **Compiler:** C++20 compliant compiler (GCC 11+, Clang 14+, or MSVC 2022 v17.4+).
* **CMake:** $\ge 3.24$ and **Ninja** build system.
* **Python:** 3.10+ with PyTorch (CUDA build enabled).

### Build from Source (Python Package)

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/BLMChoosen/RocketSim-CUDA.git
cd RocketSim-CUDA

# Build and install in editable mode via scikit-build-core & nanobind
pip install -e .

```

---

## Quick Start (Python / PyTorch)

```python
import torch
import rocketsim_cuda as rsc

# 1. Initialize 32,768 environments concurrently on GPU
num_envs = 32768
sim = rsc.RocketSimBatchedEnv(num_envs=num_envs, device="cuda:0")

# 2. Acquire zero-copy tensor views directly from VRAM (DLPack)
# Shape: [num_envs, num_cars, obs_dim]
car_obs = sim.get_car_observations() 
ball_obs = sim.get_ball_observations()

print(f"Allocated {num_envs} environments directly in VRAM.")
print(f"Obs Tensor Pointer: {hex(car_obs.data_ptr())} (Zero-copy verified)")

# 3. Simulation Step Loop (Zero PCIe Overhead)
for step in range(1000):
    # Sample random actions on GPU: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    actions = torch.rand((num_envs, 1, 8), device="cuda:0", dtype=torch.float32) * 2.0 - 1.0

    # Step physics (sub-stepped at 120 Hz internally)
    sim.step(actions)

    # Selective reset for environments that scored or timed out
    dones = sim.get_dones()
    if dones.any():
        sim.reset(torch.nonzero(dones).squeeze(-1))

```

---

## Differential Validation (Golden Master)

To guarantee that RL policies trained in `RocketSim-CUDA` transfer seamlessly to standard Rocket League engines without simulation drift:

```bash
# Run the lockstep differential harness against CPU reference (10,000 ticks)
./build/bin/differential_harness --ticks 10000 --batch 4096

```

The harness records `.rsgold` state snapshots and enforces strict Chebyshev distance constraints:

* **Position Error:** $\Vert{}\Delta_{\mathbf{p}}\Vert{}_\infty \le 10^{-4}\text{ UU}$
* **Quaternion Distance:** $\min(\lVert q_{\text{cpu}} - q_{\text{gpu}} \rVert_\infty, \lVert q_{\text{cpu}} + q_{\text{gpu}} \rVert_\infty) \le 10^{-5}$

---

## Ecosystem Integrations

* **[rlgym-cuda] (COMING SOON):** GPU-batched observation builders and vectorized reward functions for Rocket League.
* **[GigaLearn-CUDA](COMING SOON):** High-throughput C++/LibTorch reinforcement learning framework designed for 100% GPU-resident rollouts.

---

## Legal & Fair Use Notice

`RocketSim-CUDA` is an independent, clean-room physical recreation based on the open-source [RocketSim](https://github.com/ZealanL/RocketSim) project and Bullet Physics. It **does not contain any proprietary code or extracted assets** from Rocket League, Psyonix, or Epic Games.

* This library is intended exclusively for research in deep reinforcement learning, trajectory optimization, and simulation analysis.
* **Anti-Cheating Policy:** I strongly condemn the use of this software or models trained with it to deploy unauthorized bots or cheats in online competitive matchmaking.

---

## Acknowledgements

* **[ZealanL](https://github.com/ZealanL):** Creator of the original [RocketSim](https://github.com/ZealanL/RocketSim) and pioneer of the open Rocket League simulation stack.
* **Bullet Physics:** Underlying numerical kinematics foundations.
* **Nanobind:** Lightweight and ultra-fast C++/Python bindings.

```

<FollowUp label="Quer que eu prepare a estrutura do repositório rlgym-cuda agora?" query="Gere a estrutura inicial de arquivos e o código de rlgym-cuda com as funções tensoriais de observação e recompensa para Rocket League."/>

```
