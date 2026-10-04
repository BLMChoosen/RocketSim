# RocketSim-CUDA Comprehensive Benchmark Report

Generated at: 2026-10-04T04:51:49Z

## 1. 1v0 Solo Configuration (1 Car per Environment)

> **Features Active:** Full Ball Trajectory, Octane Dynamics & 4-Wheel Raycast Suspension with Bilateral Friction, Analytical Soccar Arena SDF, Car-Ball OBB-Sphere Collision, 34 Boost Pads, Goal Scoring & Episode Lifecycle Terminations.

> **Note on Step Frequencies:** SPS is reported in raw **120 Hz physical simulation ticks per second**. With standard RLGym policy substepping (`tick_skip = 8`, 15 Hz decision rate), the policy step rate is $\text{Physical SPS} / 8$.

| Environments |   Total Cars |  Step Latency (ms) |   Physical SPS (120Hz) |  Policy SPS (15Hz, skip=8) | Pool VRAM (MB) |
|--------------|--------------|--------------------|------------------------|----------------------------|----------------|
|        4,096 |        4,096 |             0.0295 |            138,997,985 |                 17,374,748 |           3.11 |
|       16,384 |       16,384 |             0.0644 |            254,390,980 |                 31,798,873 |          12.45 |
|       32,768 |       32,768 |             0.0873 |            375,262,947 |                 46,907,868 |          24.91 |
|       65,536 |       65,536 |             0.2032 |            322,590,465 |                 40,323,808 |          49.81 |

## 2. 2v2 Team Match Configuration (4 Cars per Environment)

> **Features Active:** 4 Autonomous Cars (2 Blue vs 2 Orange), Ball Trajectory & Aerodynamics, 4-Wheel Raycast Suspension per Car, Arena SDF, Car-Ball Collisions, 34 Boost Pads, Team Scoring & Terminations.
> **Important Scope Limitation:** Milestone 4 does **NOT** simulate car-on-car collisions or demolitions (deferred to Milestone 5). Hence, 2v2 performance cannot be directly compared to CPU RocketSim (which solves car-car contact graphs).

| Environments |   Total Cars |  Step Latency (ms) |   Physical Env SPS |  Policy Env SPS (skip=8) |  Agent SPS (Car-Ticks) | Pool VRAM (MB) |
|--------------|--------------|--------------------|--------------------|--------------------------|------------------------|----------------|
|        1,024 |        4,096 |             0.0904 |         11,331,097 |                1,416,387 |             45,324,387 |           2.19 |
|        4,096 |       16,384 |             0.0933 |         43,889,777 |                5,486,222 |            175,559,109 |           8.75 |
|        8,192 |       32,768 |             0.1037 |         78,988,858 |                9,873,607 |            315,955,433 |          17.50 |
|       16,384 |       65,536 |             0.2070 |         79,131,206 |                9,891,401 |            316,524,825 |          35.00 |

## 3. Methodology & Architectural Invariants

* **Hardware Timing:** Asynchronous timing recorded directly on the GPU execution stream using `cudaEventRecord` / `cudaEventElapsedTime`.
* **Zero-Copy Pipeline:** Pure on-device tensor execution. Observation, reward, action, and termination pointers reside exclusively in GPU VRAM without host PCIe round-trips.
* **Pre-Allocated Memory Arena:** 100% pre-allocated Structure-of-Arrays (SoA) layout with 128-byte cache-line alignment. Exactly 0 bytes allocated during simulation steps (`cudaMalloc` / `cudaFree` = 0).
* **Numerical Standard:** IEEE-754 strict floating-point compiler flags (`--fmad=false --prec-div=true --prec-sqrt=true -ftz=false`).
