# RocketSim-CUDA Comprehensive Benchmark Report

Generated at: 2026-10-04T04:08:08Z

## 1. 1v0 Solo Configuration (1 Car per Environment)

> **Features Active:** Full Ball Trajectory, Octane Dynamics & Raycast Suspension, Analytical Soccar Arena SDF, Car-Ball OBB Collision, 34 Boost Pads, Episode Lifecycle Terminations.

| Environments |   Total Cars |  Step Latency (ms) |   Throughput (SPS) | Pool VRAM (MB) |
|--------------|--------------|--------------------|--------------------|----------------|
|        4,096 |        4,096 |             0.0289 |        141,955,657 |           3.11 |
|       16,384 |       16,384 |             0.0628 |        260,889,499 |          12.45 |
|       32,768 |       32,768 |             0.0895 |        365,956,080 |          24.91 |
|       65,536 |       65,536 |             0.2042 |        320,888,630 |          49.81 |

## 2. 2v2 Team Match Configuration (4 Cars per Environment)

> **Features Active:** 4 Autonomous Cars (2 Blue vs 2 Orange), Ball Trajectory & Aerodynamics, 4-Wheel Raycast Suspension per Car, Arena SDF, Car-Ball Collisions, 34 Boost Pads, Team Scoring & Terminations.

| Environments |   Total Cars |  Step Latency (ms) |        Env SPS |  Agent SPS (Car-Ticks) | Pool VRAM (MB) |
|--------------|--------------|--------------------|----------------|------------------------|----------------|
|        1,024 |        4,096 |             0.0910 |     11,255,306 |             45,021,226 |           2.19 |
|        4,096 |       16,384 |             0.0947 |     43,231,560 |            172,926,240 |           8.75 |
|        8,192 |       32,768 |             0.1046 |     78,299,036 |            313,196,144 |          17.50 |
|       16,384 |       65,536 |             0.2065 |     79,323,726 |            317,294,905 |          35.00 |

## 3. Methodology & Architectural Invariants

* **Hardware Timing:** Asynchronous timing recorded directly on the GPU execution stream using `cudaEventRecord` / `cudaEventElapsedTime`.
* **Zero-Copy Pipeline:** Pure on-device tensor execution. Observation, reward, action, and termination pointers reside exclusively in GPU VRAM without host PCIe round-trips.
* **Pre-Allocated Memory Arena:** 100% pre-allocated Structure-of-Arrays (SoA) layout with 128-byte cache-line alignment. Exactly 0 bytes allocated during simulation steps (`cudaMalloc` / `cudaFree` = 0).
* **Numerical Standard:** IEEE-754 strict floating-point compiler flags (`--fmad=false --prec-div=true --prec-sqrt=true -ftz=false`).
