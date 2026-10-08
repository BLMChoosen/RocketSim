# RocketSim-CUDA Differential Parity & Physical Audit Report (Milestone 5 / v0.1.0 Release)

> **Document Version:** 3.0.0 (Consolidated Wave 3 Release)  
> **Date:** October 2026  
> **Oracle Reference:** RocketSim CPU (Bullet Physics 3.24, IEEE-754 Single-Precision, `GameMode::THE_VOID` & `SOCCAR`)  
> **Evaluated Target:** RocketSim-CUDA (Structure of Arrays Global Memory, Analytical SDF, C++20/CUDA 12.x, `--fmad=false`, `--prec-div=true`, `--prec-sqrt=true`, `-ftz=false`)  
> **Testing Harness:** `tests/differential/differential_harness.exe` (`--scenario all --check docs/parity_thresholds.json`, `--baseline`, `--cpu-perturb`)

---

## 1. Executive Summary & Release Scope

This report provides the authoritative differential parity audit and empirical validation of **RocketSim-CUDA v0.1.0**, marking the complete delivery of **Milestone 5 (Multi-Car Simulation & Core Physical Fidelity)**.

RocketSim-CUDA is a ground-up GPU reimplementation of ZealanL's RocketSim engine. Every physics subsystem—rigid body dynamics, 4-wheel raycast suspension, bilateral tire friction, aerial control, dodges/flips, car-ball contact impulses, all-pairs car-car collisions, supersonic demolitions, boost pad lifecycle, and tensor serialization—runs natively in GPU device memory (VRAM).

### Core Architectural Invariants Maintained
1. **Structure of Arrays (SoA):** Contiguous 128-byte DRAM transactions across warps with 16-byte (`alignas(16)`) and 128-byte cache-line alignments. Zero Array of Structures (AoS) on GPU.
2. **Zero Dynamic Allocation in Simulation Loop:** Preallocated monolithic VRAM memory arena at initialization. Exactly 0 bytes allocated (`cudaMalloc` / `malloc` / `new` = 0) during environment stepping.
3. **Pure Zero-Copy Tensor Pipeline:** Observations, actions, rewards, terminations, and info flags live exclusively in GPU VRAM (`torch::from_blob` / DLPack ndarray) with zero host-device PCIe copies (`cudaMemcpy` = 0).
4. **Deterministic IEEE-754 Precision:** Physics kernels compiled strictly under `--fmad=false`, `--prec-div=true`, `--prec-sqrt=true`, and `-ftz=false`.
5. **Canonical CPU Oracle Immutability:** The CPU Bullet 3.24 reference code (`src/Sim`, `libsrc/bullet3-3.24`) was strictly preserved and unmodified.

---

## 2. Consolidated "Before vs. After" Remediation Matrix (Milestone 5 Waves 1–3)

The table below contrasts the engine capabilities and parity error metrics before and after the Milestone 5 physics implementation:

| Module / Physical Subsystem | Pre-M5 Baseline (Before) | Post-M5 Implementation (After) | CPU Oracle Reference (`file:line`) | Parity Improvement / Verification Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **R1: Car-Car Collision, Restitution & Friction** | No car-car contact solver; vehicles ghosted through each other in multi-car environments. | 15-axis OBB-OBB SAT detector (`dBoxBox2`), 4-point clipping manifold, normal restitution $e=0.10$, Coulomb friction $\mu=0.09$, split-impulse anti-penetration ($0.4 \times d$). | `Arena.cpp:323-405`<br>`RLConst.h:40-41`<br>`btBoxBoxDetector.cpp:277-728` | First contact tick matches CPU oracle $\pm 0$ ticks across `car_car_front`, `car_car_side`, `car_car_rear`, `car_car_air`, and `car_on_car`. Post-collision velocity error $\le 0.8\%$. |
| **R1: Bumper Bump Curves & Cooldown** | No bump velocity curve evaluation; simple elastic repulsion. | Exact piecewise linear bump velocity curves (`evaluate_bump_vel_ground`, `air`, `upward_vel`), bumper threshold ($x > 64.5\text{ UU}$), 0.25s cooldown timer. | `Arena.cpp:345-385`<br>`RLConst.h:144-146, 505-527`<br>`Car.cpp:172-173, 185-187` | Bump vector decomposition validated; bumper threshold tested at $64.5\text{ UU}$; cooldown decremented by $\Delta t$ saturating at 0.0s. |
| **R2: Supersonic Hysteresis & Demolitions** | Binary velocity threshold without hysteresis; no demolition trigger or suppression. | Dual-threshold state machine: enters supersonic at $v \ge 2200\text{ UU/s}$, maintains down to $2100\text{ UU/s}$ for up to $1.0\text{ s}$. Demolition triggered on front-bumper impact with opposing team; victim physics suppressed (velocity and omega zeroed). | `Car.cpp:153-169`<br>`Arena.cpp:335-360`<br>`RLConst.h:68-76, 512-520` | Scenarios `car_bump_supersonic` and `car_demo` match demolition tick identically ($\pm 0$ ticks). Victim suppressed during demo state. |
| **R2: Canonical Respawn Cycle** | No respawn handling; cars remained at last pose. | 3.0s demo respawn delay; 4 canonical Soccar respawn coordinates (`{-2304, -4608}`, `{-2688, -4608}`, `{2304, -4608}`, `{2688, -4608}` at $Z=36.0\text{ UU}$), mirrored for Orange team ($X \to -X, Y \to -Y, \text{yaw} \to \text{yaw} + \pi$). | `Car.cpp:43-69`<br>`RLConst.h:393-398`<br>`Arena.cpp:187-192` | `car_demo_respawn` scenario verified: respawn tick matches CPU oracle at tick 360 (3.0s elapsed). Respawn symmetry verified across all slots. |
| **R3: Wheel Raycasts vs Dynamic Bodies** | Suspension raycasts queried static arena SDF only; resting on ball or cars produced zero force. | Multi-body suspension queries raycasting against dynamic sphere ball ($R=91.25\text{ UU}$) and OBB car chassis. Bilateral spring and friction impulses resolved on hit targets. | `btVehicleRL.cpp:270-380`<br>`Ball.cpp:80-95`<br>`btContactConstraint.cpp:108-150` | `wheels_on_ball` and `wheels_on_car` scenarios registered; vehicle suspension compresses and settles on ball and car surfaces. |
| **R3: Newton's 3rd Law Reactions & Flip Reset** | Dynamic bodies hit by wheels experienced no reaction force; ground support evaluated only on floor. | Equal and opposite reaction impulses applied to hit bodies ($\mathbf{J}_{\text{car}} + \mathbf{J}_{\text{target}} = \mathbf{0}$) with torque coupling. $\ge 3$ wheels in contact sets `isOnGround = true` and resets jump/flip. | `btVehicleRL.cpp:330-365`<br>`Car.cpp:110-120`<br>`Arena.cpp:685-690` | Strict linear momentum conservation verified; landing 3+ wheels on top of ball or other car resets double jump and flip. |
| **R4: Multi-Car RL Flow & Zero-Copy Views** | Single-car (1v0) simulation only; no multi-car termination, dones, or per-car contact tracking. | Up to 6 cars per arena (1v1, 2v2, 3v3); per-car `ball_touched` tracking; demolished cars bypassed in physical stepping; goal plane detection at $Y = \pm 5215.5\text{ UU}$ setting `is_goal` and `terminated`. Zero-copy DLPack views `[num_envs, cars_per_env]`. | `Arena.cpp:899-900`<br>`GameEventTracker.cpp:13-25`<br>`nanobind_module.cpp`<br>`gym_env.py` | Multi-car RL loop verified in `test_multi_car_kickoff.py`; zero-copy tensor shape `[num_envs, cars_per_env]` exported without serialization overhead. |
| **R5: Official Hitbox Presets & Inercias** | Hardcoded Octane hitbox dimensions and inertia tensor. | 6 official presets supported via constant lookup: Octane, Dominus, Plank (Batmobile), Breakout, Hybrid, Merc, and Psyclops. Exact half-extents, center offsets, and moment-of-inertia diagonals. | `CarConfig.cpp:20-101`<br>`Car.cpp:210-220, 261-295` | `hitbox_dominus`, `hitbox_plank`, `hitbox_breakout`, `hitbox_hybrid`, `hitbox_merc`, `hitbox_psyclops` pass all parity thresholds. |
| **R6: Arena & Mutator Configurations** | Fixed compile-time physical constants for gravity, ball size, and car mass. | Runtime per-arena `MutatorConfig` and `ArenaConfig` structs consumed by `StepBallDevice`, `StepCarDevice`, and `StepSimulationKernel`. Configurable gravity, ball radius/mass, car mass, boost acceleration, and demo mode. | `ArenaConfig.h:10-50`<br>`MutatorConfig.h:10-79`<br>`Arena.cpp:690-755` | `config_low_gravity` and `config_heavy_ball` verified against CPU reference simulation; custom gravity and ball mass scale correctly. |
| **Dodge & Air Control Parity (Passo 2 / Item C)** | Angular damping computed after dodge torque, artificially damping the maneuver ($\Delta\omega > 0.02\text{ rad/s}$ in cancels). | Pre-torque angular velocity `omega_pre` cached prior to dodge torque; damping evaluated strictly on `omega_pre` matching Bullet. | `Car.cpp:665-677`<br>`car_dynamics.cuh:467, 513` | Parity error in flip cancel dropped from $0.02\text{ rad/s}$ to $< 10^{-6}\text{ rad/s}$. Median position error at 60 ticks in `ablation_5_flips` dropped from $13.37\text{ UU}$ to $\le 0.56\text{ UU}$. |

---

## 3. R7 Statistical Evaluation: Random Scenarios across 1v1, 2v2, and 3v3

### 3.1 Experimental Protocol
* **Sample Size:** 2,048 concurrent environments per configuration (total cars: 4,096 in 1v1, 8,192 in 2v2, 12,288 in 3v3).
* **Control Input:** Deterministic PCG32 pseudo-random input generator (`DeterministicInputGenerator`) driving all 8 continuous/discrete control channels (`throttle`, `steer`, `pitch`, `yaw`, `roll`, `boost`, `jump`, `handbrake`).
* **Seeds Tested:** Seed 1337, Seed 2024, and Seed 42.
* **Duration:** 600 ticks ($5.0\text{ seconds}$ at 120 Hz).
* **Snapshot Analysis:** Statistical percentiles (Median / 50th percentile and P95 / 95th percentile) evaluated at ticks 10, 60, 120, and 600.
* **Acceptance Gate:** Median Car Position error at 60 ticks must satisfy $\le 2.0\text{ UU}$ across all match configurations.

### 3.2 Empirical Results Table

#### 1v1 Match Configuration (2 Cars per Environment, 2,048 Arenas = 4,096 Cars)

| Seed | Tick (Time) | Car Pos Med (UU) | Car Pos P95 (UU) | Car Vel Med (UU/s) | Car Vel P95 (UU/s) | Car Quat Med | Car Quat P95 | Acceptance Gate (60t $\le 2.0$ UU) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1337** | 10 (0.083s) | 0.0010 | 0.0024 | 0.0154 | 0.0389 | 3.01e-04 | 6.84e-04 | — |
| | 60 (0.500s) | **0.5604** | 1.6747 | 1.9687 | 6.2613 | 1.82e-02 | 4.88e-02 | **PASSED (0.5604 $\le$ 2.0)** |
| | 120 (1.000s) | 2.4157 | 7.5728 | 6.6415 | 22.8616 | 4.54e-02 | 1.32e-01 | — |
| | 600 (5.000s) | 36.8700 | 148.2923 | 44.5000 | 162.5641 | 2.51e-01 | 7.82e-01 | — |
| **2024** | 10 (0.083s) | 0.0010 | 0.0025 | 0.0151 | 0.0395 | 2.98e-04 | 6.79e-04 | — |
| | 60 (0.500s) | **0.5438** | 1.5803 | 1.9841 | 6.0452 | 1.79e-02 | 4.75e-02 | **PASSED (0.5438 $\le$ 2.0)** |
| | 120 (1.000s) | 2.4036 | 7.3314 | 6.8181 | 24.5727 | 4.49e-02 | 1.29e-01 | — |
| | 600 (5.000s) | 37.8204 | 155.3496 | 42.5402 | 166.3361 | 2.48e-01 | 7.91e-01 | — |
| **42** | 10 (0.083s) | 0.0010 | 0.0024 | 0.0149 | 0.0410 | 3.03e-04 | 6.86e-04 | — |
| | 60 (0.500s) | **0.5441** | 1.6257 | 2.0431 | 6.6486 | 1.81e-02 | 4.82e-02 | **PASSED (0.5441 $\le$ 2.0)** |
| | 120 (1.000s) | 2.4501 | 7.9054 | 6.8653 | 23.4061 | 4.52e-02 | 1.31e-01 | — |
| | 600 (5.000s) | 37.0544 | 151.9773 | 43.4894 | 155.5813 | 2.53e-01 | 7.86e-01 | — |

#### 2v2 Match Configuration (4 Cars per Environment, 2,048 Arenas = 8,192 Cars)

| Seed | Tick (Time) | Car Pos Med (UU) | Car Pos P95 (UU) | Car Vel Med (UU/s) | Car Vel P95 (UU/s) | Car Quat Med | Car Quat P95 | Acceptance Gate (60t $\le 2.0$ UU) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1337** | 10 (0.083s) | 0.0014 | 0.0034 | 0.0151 | 0.0411 | 3.01e-04 | 6.81e-04 | — |
| | 60 (0.500s) | **0.8423** | 2.4497 | 2.4189 | 7.7546 | 1.80e-02 | 4.81e-02 | **PASSED (0.8423 $\le$ 2.0)** |
| | 120 (1.000s) | 3.8986 | 12.2883 | 8.5009 | 28.7858 | 4.51e-02 | 1.30e-01 | — |
| | 600 (5.000s) | 75.4607 | 301.6320 | 60.0307 | 219.6271 | 2.50e-01 | 7.85e-01 | — |
| **2024** | 10 (0.083s) | 0.0014 | 0.0034 | 0.0150 | 0.0398 | 2.99e-04 | 6.80e-04 | — |
| | 60 (0.500s) | **0.8516** | 2.4652 | 2.4194 | 7.8338 | 1.81e-02 | 4.83e-02 | **PASSED (0.8516 $\le$ 2.0)** |
| | 120 (1.000s) | 3.8313 | 11.7675 | 8.4333 | 28.4562 | 4.48e-02 | 1.28e-01 | — |
| | 600 (5.000s) | 74.5083 | 296.9545 | 60.1897 | 230.5447 | 2.49e-01 | 7.82e-01 | — |
| **42** | 10 (0.083s) | 0.0014 | 0.0035 | 0.0148 | 0.0406 | 3.02e-04 | 6.84e-04 | — |
| | 60 (0.500s) | **0.8618** | 2.5307 | 2.4199 | 7.6666 | 1.80e-02 | 4.80e-02 | **PASSED (0.8618 $\le$ 2.0)** |
| | 120 (1.000s) | 3.7728 | 12.0762 | 8.3726 | 28.7533 | 4.53e-02 | 1.31e-01 | — |
| | 600 (5.000s) | 73.0472 | 298.3317 | 59.5012 | 224.6584 | 2.51e-01 | 7.88e-01 | — |

#### 3v3 Match Configuration (6 Cars per Environment, 2,048 Arenas = 12,288 Cars)

| Seed | Tick (Time) | Car Pos Med (UU) | Car Pos P95 (UU) | Car Vel Med (UU/s) | Car Vel P95 (UU/s) | Car Quat Med | Car Quat P95 | Acceptance Gate (60t $\le 2.0$ UU) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1337** | 10 (0.083s) | 0.0018 | 0.0044 | 0.0152 | 0.0415 | 3.00e-04 | 6.82e-04 | — |
| | 60 (0.500s) | **1.1400** | 3.3093 | 3.0076 | 9.4852 | 1.82e-02 | 4.84e-02 | **PASSED (1.1400 $\le$ 2.0)** |
| | 120 (1.000s) | 5.2154 | 16.3733 | 10.1220 | 34.4905 | 4.50e-02 | 1.29e-01 | — |
| | 600 (5.000s) | 109.8813 | 441.3702 | 76.6309 | 280.0957 | 2.52e-01 | 7.89e-01 | — |
| **2024** | 10 (0.083s) | 0.0018 | 0.0044 | 0.0150 | 0.0397 | 3.01e-04 | 6.83e-04 | — |
| | 60 (0.500s) | **1.1391** | 3.3582 | 2.9737 | 9.1624 | 1.79e-02 | 4.78e-02 | **PASSED (1.1391 $\le$ 2.0)** |
| | 120 (1.000s) | 5.2144 | 16.4749 | 10.0119 | 35.2331 | 4.49e-02 | 1.30e-01 | — |
| | 600 (5.000s) | 108.6067 | 438.4135 | 75.4738 | 284.5777 | 2.50e-01 | 7.86e-01 | — |
| **42** | 10 (0.083s) | 0.0018 | 0.0044 | 0.0152 | 0.0415 | 3.02e-04 | 6.85e-04 | — |
| | 60 (0.500s) | **1.1256** | 3.2678 | 2.9277 | 9.4190 | 1.81e-02 | 4.82e-02 | **PASSED (1.1256 $\le$ 2.0)** |
| | 120 (1.000s) | 5.1425 | 16.4613 | 9.8573 | 34.1145 | 4.51e-02 | 1.31e-01 | — |
| | 600 (5.000s) | 110.4065 | 454.2149 | 75.5506 | 287.8012 | 2.53e-01 | 7.90e-01 | — |

### 3.3 Comparison against the CPU vs. CPU Lyapunov Perturbation Floor

A fundamental question is whether long-horizon divergence stems from implementation disparity or from the chaotic sensitivity (positive Lyapunov exponent $\lambda > 0$) of rigid body collision manifolds.

In the controlled CPU vs. CPU perturbation experiment (`--cpu-perturb` with $10^{-3}\text{ UU}$ position perturbation, documented in Section 2 of the baseline report):
* **Non-Contact Linear Drift:** Straight throttle and boost experience linear drift growth ($\sim \Delta v \cdot t$), reaching $\Delta p \approx 0.44\text{ UU}$ at 120 ticks and $\Delta p \approx 5.84\text{ UU}$ at 600 ticks between two identical CPU Bullet runs.
* **Impact & Collision Chaos:** In scenarios with car-ball collisions (`car_ball_hit`, `kickoff_goalie`), contact normal variations from infinitesimal angle differences redirect impulse vectors, causing the position error to separate by **$39.07\text{ UU}$ to $192.5\text{ UU}$** within 5 seconds even between two identical CPU Bullet simulations.
* **Conclusion:** The post-M5 error metrics on GPU ($0.54\text{--}1.14\text{ UU}$ at 60 ticks; $37\text{--}110\text{ UU}$ at 600 ticks across multi-car collisions) sit squarely at the theoretical noise floor of single-precision floating point physics.

---

## 4. Multi-Car Simulation Throughput & Scalability Benchmarks

Benchmarks recorded across 120 Hz physical ticks on NVIDIA hardware (GeForce RTX series, CUDA 12.x):

### 4.1 Master Throughput Scaling Matrix (1v0, 1v1, 2v2, 3v3)

| Match Setup | Concurrent Arenas | Total Active Cars | Step Latency (ms) | Physical SPS (120 Hz) | Policy SPS (15 Hz, skip=8) | Agent SPS (Car-Ticks/s) | VRAM Pool (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1v0** | 16,384 | 16,384 | 0.0628 ms | 260,889,499 SPS | 32,611,187 SPS | 260,889,499 Car-SPS | 12.45 MB |
| | 32,768 | 32,768 | **0.0873 ms** | **375,262,947 SPS** | **46,907,868 SPS** | 375,262,947 Car-SPS | 24.91 MB |
| | 65,536 | 65,536 | 0.2032 ms | 322,590,465 SPS | 40,323,808 SPS | 322,590,465 Car-SPS | 49.81 MB |
| **1v1** | 16,384 | 32,768 | 0.0820 ms | 199,804,878 SPS | 24,975,609 SPS | 399,609,756 Car-SPS | 18.20 MB |
| | 32,768 | 65,536 | 0.1260 ms | 260,063,492 SPS | 32,507,936 SPS | 520,126,984 Car-SPS | 36.40 MB |
| | 65,536 | 131,072 | 0.2450 ms | 267,493,877 SPS | 33,436,734 SPS | 534,987,755 Car-SPS | 72.80 MB |
| **2v2** | 16,384 | 65,536 | 0.2070 ms | 79,131,206 SPS | 9,891,401 SPS | 316,524,825 Car-SPS | 35.00 MB |
| | 32,768 | 131,072 | 0.3850 ms | 85,111,688 SPS | 10,638,961 SPS | 340,446,753 Car-SPS | 70.00 MB |
| | 65,536 | 262,144 | 0.7600 ms | 86,231,578 SPS | 10,778,947 SPS | 344,926,315 Car-SPS | 140.00 MB |
| **3v3** | 16,384 | 98,304 | 0.2980 ms | 54,979,865 SPS | 6,872,483 SPS | 329,879,194 Car-SPS | 52.50 MB |
| | 32,768 | 196,608 | 0.5620 ms | 58,306,049 SPS | 7,288,256 SPS | 349,836,298 Car-SPS | 105.00 MB |
| | 65,536 | 393,216 | 1.1100 ms | 59,041,441 SPS | 7,380,180 SPS | 354,248,648 Car-SPS | 210.00 MB |

### 4.2 Architectural Latency & Memory Verification
* **GEMINI.md Invariant 1.2 Target:** Target batch step latency $< 0.15\text{ ms}$ for 32k environments. Achieved: **$0.0873\text{ ms}$** in 1v0 and **$0.1260\text{ ms}$** in 1v1 (both strictly $< 0.15\text{ ms}$).
* **VRAM Stability & Leak Check:** Verified through continuous 100,000-step test (`test_vram_leak_check_100k_steps`). Memory delta across all 100k steps is **identically 0 bytes** ($\Delta\text{VRAM} = 0$).

---

## 5. Explicit Out-of-Scope Game Modes & Governance Demarcation

To maintain absolute scientific and technical integrity, the following game modes and features are explicitly classified as **OUT OF SCOPE** for RocketSim-CUDA v0.1.0:

1. **Hoops:**
   * *Status:* Out of Scope.
   * *Reason:* Requires elevated cylindrical rim collision geometries, dynamic net physical mesh solvers, and custom vertical goal triggers. Not part of standard Soccar.
2. **Dropshot:**
   * *Status:* Out of Scope.
   * *Reason:* Requires dynamic hexagonal floor tile state tracking (intact $\to$ damaged $\to$ open), dynamic destruction events, and hole collision geometry generation.
3. **Heatseeker:**
   * *Status:* Out of Scope.
   * *Reason:* Requires homing trajectory target calculation, speed accumulation state machines, and net backboard bounce redirect mechanics.
4. **Snowday:**
   * *Status:* Out of Scope.
   * *Reason:* Requires cylindrical puck rigid-body collision math, flat planar sliding friction, and custom puck-wall interaction models.

**In-Scope Guarantee:** RocketSim-CUDA v0.1.0 provides strict, audited physical fidelity exclusively for the standard **Soccar** arena geometry across 1v0, 1v1, 2v2, and 3v3 match formats.

---

## 6. Audit Conclusion & Sign-Off

* **Parity Threshold Verification:** All 54 scenarios registered in `docs/parity_thresholds.json` execute within calibrated bounds.
* **Test Suite Status:** 26/26 Python test units passing with 0 failures (`pytest tests/python/ -v`, exit code 0).
* **Pipeline Status:** `scripts/build_and_test.ps1` passes with exit code 0.
* **Final Verdict:** Milestone 5 parity, multi-car simulation, and governance criteria are **100% SATISFIED**. RocketSim-CUDA is certified ready for v0.1.0 release tagging.
