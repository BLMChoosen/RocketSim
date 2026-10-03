# GEMINI.md - RocketSim-CUDA: Context, Architectural Invariants & Governance

> **Repository:** `RocketSim-CUDA`  
> **Mission:** Ultra-parallel, native C++/CUDA reimplementation of the RocketSim physics engine with zero-copy PyTorch/Nanobind tensor bindings, dedicated to massive GPU reinforcement learning rollouts (16,384 to 65,536 concurrent environments) maintaining strict 1:1 tick-by-tick physical parity against CPU Bullet Physics.

---

## 1. Project Vision & Performance Targets

### 1.1 Core Mission
Reinforcement learning for Rocket League (e.g., *RLGym*, *PPO*, *IMPALA*) is fundamentally constrained by CPU simulation throughput and PCIe transfer bottlenecks. The original RocketSim engine, despite high CPU optimization, requires CPU-thread scaling and frequent host-to-device memory copies (`cudaMemcpy`) to feed policy networks.

**RocketSim-CUDA** moves the entire physics simulation pipeline—rigid body dynamics, wheel raycasting, collision resolution, arena SDF queries, boost tracking, and state serialization—directly into GPU VRAM (device memory).

### 1.2 Target Performance Metrics
* **Simulation Throughput:** $> 100\text{x}$ to $1000\text{x}$ speedup over multi-threaded CPU RocketSim.
* **Environment Concurrency:** Scale predictably from $16,384$ ($2^{14}$) up to $65,536$ ($2^{16}$) parallel arena instances within a single consumer/datacenter GPU (RTX 3080/4090, A100, H100).
* **Batch Step Latency:** $< 0.15\text{ ms}$ for a batch step of $32,768$ environments at $120\text{ Hz}$ tickrate.
* **Zero-Copy Pipeline:** Pure on-device tensor lifecycle. `observations`, `rewards`, `terminated`, and `actions` reside exclusively in GPU memory as PyTorch `torch::Tensor` / Nanobind ndarrays without any PCIe round-trip.

---

## 2. Core Architectural Invariants (Non-Negotiable Rules)

Every agent, engineer, and subagent contributing to this codebase **must** adhere strictly to the following architectural invariants. Any implementation violating these rules will be rejected.

```
       =========================================================
                          ROCKETSIM-CUDA
                     GPU KERNEL ARCHITECTURE
       =========================================================
       
          16k - 64k Parallel Environments in Device Memory
       +-------------------------------------------------------+
       |   Environment 0   |   Environment 1   | ... |  Env N  |
       +-------------------------------------------------------+
                                  |
                                  v
       +-------------------------------------------------------+
       |   Structure of Arrays (SoA) Coalesced Global Memory   |
       |  pos.x[N]  pos.y[N]  pos.z[N]  vel.x[N]  vel.y[N] ... |
       +-------------------------------------------------------+
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
       +-------------------------+         +-------------------------+
       |   Analytical SDF Arena  |         |   btRaycastVehicle GPU  |
       |   Closed-form O(1)      |         |   4 Rays, Damping,      |
       |   Distance & Normals    |         |   Bilateral Friction    |
       +-------------------------+         +-------------------------+
            |                                           |
            +---------------------+---------------------+
                                  |
                                  v
       +-------------------------------------------------------+
       |         Rigid Body Integrator (120Hz Delta-T)         |
       |             IEEE-754 Strict (-fmad=false)             |
       +-------------------------------------------------------+
                                  |
                                  v
       +-------------------------------------------------------+
       |   Zero-Copy PyTorch Device Tensor (torch::from_blob)  |
       +-------------------------------------------------------+
```

### 2.1 Structure of Arrays (SoA) Memory Layout
* **Absolute Prohibition:** `Array of Structures (AoS)` (e.g. `struct Car { Vec pos; Vec vel; ... }; Car cars[N];`) is strictly prohibited in GPU storage.
* **Requirement:** All state attributes must be organized as `Structure of Arrays (SoA)` to ensure contiguous, coalesced 128-byte DRAM transactions across warps:
  ```cpp
  // CORRECT: Structure of Arrays (SoA) - Coalesced 128-byte transactions
  struct CarStateSoA {
      float* __restrict__ pos_x;
      float* __restrict__ pos_y;
      float* __restrict__ pos_z;
      
      float* __restrict__ vel_x;
      float* __restrict__ vel_y;
      float* __restrict__ vel_z;

      float* __restrict__ q_w;
      float* __restrict__ q_x;
      float* __restrict__ q_y;
      float* __restrict__ q_z;

      float* __restrict__ ang_vel_x;
      float* __restrict__ ang_vel_y;
      float* __restrict__ ang_vel_z;
      
      float* __restrict__ boost;
      uint8_t* __restrict__ is_on_ground;
      uint8_t* __restrict__ has_jumped;
      uint8_t* __restrict__ has_double_jumped;
      uint8_t* __restrict__ has_flipped;
      // ...
  };
  ```
* **Memory Alignment:** All device array pointers must be aligned to 16 bytes (`alignas(16)`) or 128 bytes (cache-line boundary). When vector loading, use `float4` loads only if components are accessed simultaneously by the same thread.

### 2.2 Zero Dynamic Allocations in Device (Kernel Execution)
* **Absolute Prohibition:** No calls to `cudaMalloc`, `cudaFree`, `malloc`, `free`, `new`, or `delete` inside simulation loops, step methods, or CUDA kernels.
* **Pre-allocated Memory Arena:** All memory pools for environments, cars, balls, contact points, and scratch pads must be allocated at environment initialization time.
* **Static Bounds:** Each arena has fixed capacities (e.g. $1$ ball, up to $8$ cars, $34$ boost pads). Variable-length collections inside kernels are prohibited; use bitmasks or fixed-size indexed slots.

### 2.3 Numerical Precision, IEEE-754 & FMA Contraction
* **Parity Invariant:** RocketSim on CPU relies on Bullet Physics 3.24's 32-bit floating point arithmetic (`btScalar = float`).
* **FMA Control:** NVCC performs aggressive Fused Multiply-Add contraction (`a * b + c`) by default via `-fmad=true`. Because standard CPU builds execute separate multiply and add instructions (producing distinct rounding behavior), simulation divergence occurs rapidly.
* **Compilation Directives:**
  * Compilations targeting physics kernels must enforce:
    ```cmake
    --fmad=false
    --prec-div=true
    --prec-sqrt=true
    -ftz=false
    ```
  * In performance profiling builds where relaxation is explicitly approved, `-fmad=true` may be evaluated only if maximum deviation remains within par tolerance.

### 2.4 Analytical Signed Distance Fields (SDF) for Arena Collisions
* **Problem in CPU RocketSim:** RocketSim CPU loads collision meshes (`.cm` files containing thousands of triangles) and performs BVH tree traversals (`btBvhTriangleMeshShape`). On GPU, BVH traversals trigger extreme warp divergence, branch divergence, and irregular cache thrashing.
* **Solution in RocketSim-CUDA:** Standard Soccar arena geometry is mathematically analytical:
  * Rectangular bounds: $X \in [-4096, 4096]$, $Y \in [-5120, 5120]$, $Z \in [0, 2048]$.
  * Corner chamfers: $45^\circ$ vertical bevels with cylindrical fillet curves.
  * Wall-to-floor and wall-to-ceiling ramps: cylindrical arcs with known radius ($R_{curv} = 260\text{ UU}$).
  * Goal cavities: oriented bounding boxes with curved posts/crossbars (torus/cylinder segments).
* **Execution:** All arena static queries are computed in $O(1)$ arithmetic operations via closed-form SDF functions:
  $$\Phi(\mathbf{p}) \le 0 \implies \mathbf{p} \text{ is in collision}$$
  $$\mathbf{n}(\mathbf{p}) = \nabla \Phi(\mathbf{p}) \quad (\text{analytical contact normal})$$
* **Mesh Independence:** No mesh files are loaded in device kernels for standard Soccar.

### 2.5 Faithful Suspension Model (`btRaycastVehicle`)
* The custom Bullet vehicle suspension in RocketSim (`btVehicleRL`) must be faithfully ported:
  1. **Raycast Query:** 4 suspension rays cast downwards from wheel suspension hardpoints in chassis-local space.
  2. **Contact Point & Normal:** Evaluated against ground/ramps/curved walls via arena SDF and ground plane.
  3. **Spring & Damping Equation:**
     $$F_{\text{susp}} = \left(L_{\text{rest}} - L\right) \cdot k_{\text{stiff}} \cdot \text{clipped\_inv\_dot} - v_{\text{rel}} \cdot d$$
     where damping scale $d$ switches between compression ($0.83$) and relaxation ($0.88$).
  4. **Extra Pushback:** Static object compression compensation replicating `resolveSingleCollision`.
  5. **Bilateral Friction Formulation:** Sideways and forward wheel impulse resolution matching `resolveSingleBilateral` and `calcFrictionImpulses`.

---

## 3. Parity Harness & Differential Testing (Golden Master)

Parity testing is the single source of truth determining the correctness of RocketSim-CUDA.

```
       =========================================================
                      DIFFERENTIAL TESTING HARNESS
       =========================================================

             Initial State S_0 (Ball, Cars, Inputs)
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
       +-----------------------+       +-----------------------+
       |   RocketSim CPU       |       |   RocketSim-CUDA      |
       |   (Bullet 3.24 Ref)   |       |   (Native GPU Kernel) |
       +-----------------------+       +-----------------------+
            |                                     |
            | S_cpu(t+1)                          | S_gpu(t+1)
            +------------------+------------------+
                               |
                               v
       +-------------------------------------------------------+
       |   Differential Comparator                             |
       |   Metric: ||S_cpu - S_gpu||_inf                       |
       |   Threshold: <= 1e-4 per tick                         |
       +-------------------------------------------------------+
```

### 3.1 Tolerance Thresholds
For any step $t \in [0, T]$ starting from identical state $S_0$ and receiving identical input actions $A_t$:

| State Attribute | Unit | Maximum Tolerated Delta ($\Vert \Delta \Vert_\infty$) |
| :--- | :--- | :--- |
| **Linear Position (Ball & Car)** | Unreal Units (UU) | $\le 10^{-4}\text{ UU}$ per tick |
| **Linear Velocity** | UU / s | $\le 10^{-3}\text{ UU/s}$ per tick |
| **Rotation (Quaternion)** | Normalized $w, x, y, z$ | $\le 10^{-5}$ per tick |
| **Angular Velocity** | rad / s | $\le 10^{-4}\text{ rad/s}$ per tick |
| **Suspension Compression** | UU | $\le 10^{-4}\text{ UU}$ per tick |
| **Boost Amount** | $[0, 100]$ | $0.0$ (Bit-exact float representation) |

### 3.2 Verification Scenarios (Golden Master Suite)
1. **Ball Trajectory & Drag:** Ball launched with extreme velocities ($6000\text{ UU/s}$) and angular velocities ($6.0\text{ rad/s}$) subject to air drag and gravity for $1200$ ticks.
2. **Multi-Surface Ball Bouncing:** High-speed impacts on flat floor, $45^\circ$ corner bevels, ceiling, curved wall transitions, and goalpost rims.
3. **Suspension Equilibrium & Settling:** Car dropped from $z = 500\text{ UU}$, settling under gravity onto suspension resting height.
4. **Full Car Mechanics Replay:**
   * Ground throttle, max acceleration, coasting deceleration, handbrake slide.
   * Single jump, variable jump hold ($0.025\text{ s}$ to $0.2\text{ s}$), double jump.
   * Dodges / Flips in all 8 directions with torque damping and pitch lock.
   * Air control (pitch, yaw, roll, air-throttle).
5. **Car-on-Car Collisions & Demolitions:** Supersonic speed head-on, T-bone, and glancing blows triggering demos and impulse transfer.
6. **Car-Ball Pinches:** Ground pinch, wall pinch, ceiling pinch generating extreme impulse responses.

### 3.3 Replay File Format
Parity datasets are recorded into compact uncompressed binary files (`.rsgold`):
* Header: Magic `0x5253474D` ("RSGM"), Version, Number of Environments, Total Ticks.
* Per Tick: `CarInput` array, followed by CPU ground-truth `BallState` and `CarState` structs.

---

## 4. Tech Stack & Toolchain

### 4.1 System & Compiler Requirements
* **Operating Systems:** Linux (Ubuntu 20.04/22.04+, RHEL 8/9), Windows 10/11 (MSVC 2022 v143+).
* **C++ Standard:** ISO C++20 (`-std=c++20` or `/std:c++20`).
* **CUDA Toolkit:** CUDA 12.0 or newer (targeting compute capabilities `sm_80`, `sm_86`, `sm_89`, `sm_90`).
* **Compilers:**
  * Host: GCC 11+, Clang 14+, or MSVC 19.34+.
  * Device: NVCC 12.0+ or Clang CUDA.
* **Python Environments:** Python 3.9 through 3.12.
* **Python Bindings:** PyTorch C++ Extension API (`torch::Tensor` / `ATen`) and/or `nanobind` (preferred for minimal binary size and instant compile times).

### 4.2 Build Toolchain
* **Build System:** Modern CMake ($\ge 3.24$) with first-class `enable_language(CUDA)`.
* **Packaging:** `scikit-build-core` + `pyproject.toml` (PEP 517 / PEP 518 compliant).
* **Dependencies Permitted:**
  * Header-only math primitives (`cuda_math` / intrinsic CUDA float helpers).
  * Bullet Physics 3.24 headers (strictly for CPU test harness and validation comparisons).
* **Strict Prohibitions:**
  * No heavy external physics frameworks inside the CUDA runtime.
  * No STL heap containers (`std::vector`, `std::map`, `std::string`) inside device code.
  * No virtual method hierarchies or runtime polymorphism in device simulation paths.

---

## 5. Directory Layout Blueprint

The repository structure isolates device kernels, analytical physics math, Python/PyTorch bindings, and testing harnesses:

```text
RocketSim-CUDA/
├── CMakeLists.txt                  # Root CMake build definition
├── pyproject.toml                  # Python packaging (scikit-build-core)
├── GEMINI.md                       # Canonical governance & architectural context
├── README.md                       # High-level overview & quickstart
├── include/
│   └── rocketsim_cuda/
│       ├── config.h                # Global limits, tickrate, arena constants
│       ├── math/
│       │   ├── vec3.cuh            # Device vector arithmetic & SIMD intrinsics
│       │   ├── mat3.cuh            # 3x3 rotation matrices
│       │   └── quat.cuh            # Quaternions, slerp, conversions
│       ├── types/
│       │   ├── car_state.cuh       # CarState SoA structures
│       │   ├── ball_state.cuh      # BallState SoA structures
│       │   ├── arena_state.cuh     # Boost pads, goals, game states
│       │   └── car_controls.cuh    # Packed input tensors (throttle, steer, etc.)
│       └── physics/
│           ├── arena_sdf.cuh       # Closed-form Soccar SDF collision math
│           ├── suspension.cuh      # btVehicleRL raycast suspension model
│           ├── car_dynamics.cuh    # Drive torque, air control, jump, dodge
│           └── contact_solver.cuh  # Bilateral impulse & restitution solver
├── src/
│   ├── cuda/
│   │   ├── sim_context.cu          # Device memory allocation & stream management
│   │   ├── step_kernel.cu          # Master environment simulation kernel
│   │   ├── car_kernel.cu           # Car-specific update kernels
│   │   └── ball_kernel.cu          # Ball trajectory & collision kernels
│   ├── physics/
│   │   ├── arena_sdf.cu            # Analytical SDF geometry constants & evaluation
│   │   └── suspension.cu           # Raycast traversal and friction kernels
│   └── bindings/
│       ├── torch_bindings.cpp      # Zero-copy PyTorch tensor exports
│       └── nanobind_module.cpp     # Nanobind module definition
├── tests/
│   ├── differential/
│   │   ├── harness_main.cpp        # Lockstep CPU vs GPU comparison runner
│   │   ├── cpu_ref_sim.cpp         # Wrapper for original CPU RocketSim
│   │   ├── golden_master.cpp       # Snapshot comparator and binary serializer
│   │   └── scenarios/              # Individual physical validation suites
│   │       ├── test_ball_bounce.cpp
│   │       ├── test_suspension.cpp
│   │       ├── test_car_air.cpp
│   │       └── test_collisions.cpp
│   └── unit/
│       ├── test_math.cu
│       └── test_sdf.cu
└── benchmarks/
    ├── throughput_benchmark.cu     # Raw step throughput (16k to 64k envs)
    ├── profile_nsight.sh           # NCU / NSYS profiling script
    └── python/
        └── bench_tensor_step.py    # Python/PyTorch end-to-end loop benchmark
```

---

## 6. Workflow & Coding Standards for Agents

All agents and contributors must follow these instructions for development, linting, building, and validation.

### 6.1 Canonical Build Commands

#### Configure and Build Native C++/CUDA with CMake
```bash
# Configure with Release optimization and target GPU architectures
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
  -DROCKETSIM_CUDA_BUILD_TESTS=ON \
  -DROCKETSIM_CUDA_BUILD_BENCHMARKS=ON

# Build all targets
cmake --build build --config Release -j
```

#### Build Python Wheel (Editable Mode)
```bash
pip install -e . --no-build-isolation -v
```

### 6.2 Running Parity & Differential Tests
```bash
# Run unit tests
ctest --test-dir build --output-on-failure -R Unit

# Run lockstep differential parity test (CPU vs GPU across 10,000 ticks)
./build/tests/differential/differential_harness --ticks 10000 --envs 2048 --seed 42 --tol 1e-4
```

### 6.3 Benchmarking & Profiling
```bash
# Run raw CUDA throughput benchmark
./build/benchmarks/throughput_benchmark --envs 32768 --steps 1000

# Profile kernel with Nsight Compute (memory throughput and warp occupancy)
ncu --set full -o profile_step ./build/benchmarks/throughput_benchmark --envs 32768 --steps 100
```

### 6.4 Coding Standards & Invariants Checklist
When reviewing or writing device code:
1. **Pointers Restricted:** Use `__restrict__` on all raw device pointers to enable compiler caching and register promotion.
2. **Branch Divergence Elimination:** Minimize data-dependent branches within warps. Favor branchless math operations (`__fminf`, `__fmaxf`, ternary `cond ? a : b`).
3. **Warp Uniformity:** Keep block size to multiples of 32 (recommended default: `128` or `256` threads per block).
4. **Const-Correctness:** All read-only configuration structs must be marked `const __restrict__` or passed by value if under 16 bytes.
5. **Formatting:** Enforce Google C++ Style with 4-space indentation and column limit 120 (`clang-format`).
6. **No Silent Divergence:** Any change that causes a test regression in `differential_harness` must be treated as a blocker and addressed immediately.
