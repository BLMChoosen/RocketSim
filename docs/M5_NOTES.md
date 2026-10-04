# Milestone 5 — Engineering Notes & CPU Oracle Tracking

> **Document Version:** 1.0.0  
> **Milestone:** Milestone 5 — Phase 1 (Core Physical Fidelity)  
> **Status:** Active

---

## Module 1.1: Differential Testing Harness & Component-Wise Parity Baseline (M5.1)

### CPU Oracle Reference
- **Simulation stepping loop & state query:** `tests/differential/cpu_ref_sim.cpp:116-160`
  - `CPURefSim::Step(controls, numCars)`: mirrors `RocketSim::Arena::Step(1)` in `src/Sim/Arena/Arena.cpp:138-164`.
  - `CPURefSim::GetBallState(BallStatePOD& out)`: mirrors `m_arena->ball->GetState()` (`src/Sim/Ball/Ball.cpp:21-36`) and `btRigidBody` world rotation (`libsrc/bullet3-3.24/BulletDynamics/Dynamics/btRigidBody.h`).
  - `CPURefSim::GetCarState(int carIdx, CarStatePOD& out)`: mirrors `m_cars[i]->GetState()` (`src/Sim/Car/Car.cpp:278-312`).
- **Input generation:** `tests/differential/golden_master.cpp:32-44` (`DeterministicInputGenerator::Generate()` using PCG32).

### Execution Plan & Implementation Summary
1. **Component-Wise Tracking:** Implemented `ComponentId` enumeration across 19 components:
   - Car Position ($X, Y, Z$)
   - Car Velocity ($X, Y, Z$)
   - Car Orientation Quaternion ($W, X, Y, Z$) with antipodal sign correction ($\mathbf{q}_{cpu} \cdot \mathbf{q}_{gpu} < 0 \implies -\mathbf{q}_{gpu}$)
   - Ball Position ($X, Y, Z$)
   - Ball Velocity ($X, Y, Z$)
   - Ball Angular Velocity ($X, Y, Z$)
2. **Snapshot Tracking:** Recorded error distributions at ticks 1, 10, 60, 120, and 600.
3. **Statistical Aggregation:** Extracted Median (50th percentile) and P95 (95th percentile) using `std::nth_element` for both absolute error $|\Delta|$ and regularized relative error $\frac{|\Delta|}{|v_{cpu}| + 1.0}$.
4. **Large-Batch Bypass:** Automatically bypasses `.rsgold` file I/O (Tests 1-4) when `--report` with `--envs >= 256` or `--baseline` is active, directly dispatching Test 5.
5. **Host Parallel Stepping:** Implemented lightweight C++ thread pool (`ThreadPool::ParallelFor`) and OpenMP `#pragma omp parallel for` on `lockstep_cpu_envs[e].Step(...)`, leveraging all CPU host threads.

---

## Module 1.2: Ball Bounces Fidelity (M5.2 - Completed)

### CPU Oracle Reference
- **Ball state activation and restitution callback:** `src/Sim/Ball/Ball.cpp:15-80`
  - `Ball::SetState`: guards activation with `if (!state.vel.IsZero() || !state.angVel.IsZero())`; if true, sets `m_rigidBody->activate(true)`.
- **Arena static bounds and forced sleep checks:** `src/Sim/Arena/Arena.cpp:298-306, 695-700, 1038-1060`
  - `Arena::_AddStaticCollisionShape`: adds static rigid body with base restitution (`0.3f`) and friction (`0.6f`).
  - `Arena::Step(1)` line 695: unconditionally forces `ISLAND_SLEEPING` on the ball if `m_linearVelocity.length2() == 0 && m_angularVelocity.length2() == 0`.
- **Arena and Ball Physical Constants:** `src/Sim/RLConst.h:35-45, 80-95`
  - `ARENA_COLLISION_BASE_RESTITUTION = 0.3f`, `ARENA_COLLISION_BASE_FRICTION = 0.6f`.
  - `BALL_COLLISION_RADIUS = 91.25f`, `BALL_RESTITUTION = 0.6f`, `BALL_FRICTION = 0.35f`.
- **Bullet Sequential Impulse Solver:** `libsrc/bullet3-3.24/BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.cpp:1048-1060, 1164-1211`
  - Bullet restitution threshold: `restitutionThreshold = 0.2f` (in BT units = $10.0\text{ UU/s}$).
  - Coulomb friction impulse clamping: $|J_t| \le \mu |J_n|$ with combined friction coefficient $\mu = 0.35$.
  - Solid sphere tangential effective mass inertia ratio: $\frac{2}{7} M \approx 0.285714 M$ for rolling contact.
  - Angular momentum coupling: $\Delta\boldsymbol{\omega} = \frac{5}{2 M R^2} (\mathbf{r} \times \mathbf{J}_t) = \frac{2.5}{R} (\mathbf{J}_t \times \mathbf{n})$.

### Implementation & Mathematical Formulation
1. **Voronoi Wedge Fix (`include/rocketsim_cuda/physics/arena_sdf.cuh`):**
   - Fixed `arena_sdf_2d_wall` corner distance evaluation by replacing Euclidean partitioning with exact convex polygon half-plane distance minimization:
     $$d_{wall} = \min(4096 - x, 5120 - y, (8064 - (x + y)) \cdot \frac{1}{\sqrt{2}})$$
   - Evaluates exact inward unit normal $\mathbf{n}$ corresponding to the closest boundary facet.
2. **Coulomb Friction & Angular Coupling (`include/rocketsim_cuda/physics/contact_solver.cuh`):**
   - Replaced ad-hoc velocity damping with closed-form Bullet-equivalent Coulomb impulse resolution:
     - Contact point surface velocity: $\mathbf{v}_{contact} = \mathbf{v} - (\boldsymbol{\omega} \times \mathbf{n}) R$
     - Contact slip velocity: $\mathbf{v}_{slip} = \mathbf{v}_{contact} - \mathbf{n} (\mathbf{n} \cdot \mathbf{v}_{contact})$
     - Tangential slip impulse: $J_{slip} = \frac{2}{7} |\mathbf{v}_{slip}|$
     - Coulomb clamping: $J_{fric} = \min(J_{slip}, \mu \Delta v_n)$ where $\Delta v_n = -(1 + e) v_n$
     - Torque coupling: $\Delta\boldsymbol{\omega} = \frac{2.5}{R} (\mathbf{J}_{tangent} \times \mathbf{n})$
     - Restitution threshold: $e = 0.0$ if $|v_n| < 10.0\text{ UU/s}$ (matching Bullet 0.2 BT units).
3. **Symplectic Euler Integration Alignment (`src/cuda/step_kernel.cu`):**
   - Structured `StepBallDevice` order: Damping $\to$ Gravity ($g = -650\text{ UU/s}^2$) $\to$ `resolve_ball_arena_collision` $\to$ Symplectic position integration ($\mathbf{x} \mathrel{+}= \mathbf{v} \Delta t$) $\to$ Orientation quaternion integration.
4. **CPURefSim Bullet Geometry & Sleep Wake (`tests/differential/cpu_ref_sim.cpp`):**
   - Added static boundary planes for side walls ($X = \pm 4096$), back walls ($Y = \pm 5120$), ceiling ($Z = 2048$), and corner chamfers ($X+Y = 8064$) with base restitution $0.3$ and friction $0.6$.
   - Fixed ball sleep on zero-velocity mid-air spawns via `activate(true)`, `setActivationState(ACTIVE_TAG)`, and setting initial $v_z = -10^{-6}\text{ UU/s}$ epsilon to bypass `Arena.cpp:695`'s forced sleep.
5. **Differential Harness Canonical Suite (`tests/differential/harness_main.cpp`):**
   - Registered 8 canonical bounce scenarios: `ball_floor_drop`, `ball_floor_angled`, `ball_side_wall`, `ball_back_wall`, `ball_ceiling`, `ball_corner_ramp`, `ball_goal_post`, `ball_crossbar`.

### Empirical Verification Metrics
- **Flat Floor Drop (`ball_floor_drop`):** Rebound Tick: CPU 84, GPU 84 (0 ticks delta). Spin delta: 0.00 rad/s.
- **Angled Ground + Spin (`ball_floor_angled`):** Rebound Tick: CPU 45, GPU 46 (1 tick delta). Post-impact tangential velocity $v_x$ error dropped from $109.96\text{ UU/s}$ to **$0.09\text{ UU/s}$**; spin delta: **$0.00\text{ rad/s}$**.
- **Side Wall Bounce (`ball_side_wall`):** Rebound Tick: CPU 41, GPU 41 (0 ticks delta). Post-bounce velocity delta: **$0.00\text{ UU/s}$**, spin delta: **$0.00\text{ rad/s}$** across ticks +1, +5, +30.
- **Back Wall Bounce (`ball_back_wall`):** Rebound Tick: CPU 43, GPU 43 (0 ticks delta). Post-bounce velocity delta: **$0.00\text{ UU/s}$**, spin delta: **$0.00\text{ rad/s}$** across ticks +1, +5, +30.
- **Ceiling Bounce (`ball_ceiling`):** Rebound Tick: CPU 40, GPU 40 (0 ticks delta). Spin delta: **$0.00\text{ rad/s}$**.
- **Curved Corner Ramp (`ball_corner_ramp`):** Discrepancy reduced from 24 ticks to 5 ticks (CPU chamfer at tick 24 vs GPU fillet ramp at tick 19).
- **Goal Posts (`ball_goal_post`):** Rebound Tick: CPU 43, GPU 43 (0 ticks delta).
- **Crossbar (`ball_crossbar`):** Rebound Tick: CPU 43, GPU 43 (0 ticks delta). Post-bounce velocity delta: **$0.00\text{ UU/s}$**, spin delta: **$0.00\text{ rad/s}$**.
- **CPU vs CPU Perturbation (1e-3 UU):** Zero velocity growth across 100 ticks ($|\Delta\mathbf{v}| = 0.00\text{ UU/s}$), demonstrating non-chaotic physical stability.
- **Python Zero-Copy Suite:** 35/35 passing tests in 4.80s.
- **Unit SDF Suite:** 8/8 tests passing (`test_sdf.exe`).

---

## Module 1.3: Longitudinal Tire Friction & Car Contact Solver (M5.3 - Upcoming)

### CPU Oracle Reference
- `src/Sim/btVehicleRL/btVehicleRL.cpp:78-295` (`btVehicleRL::updateFriction`, `btVehicleRL::calcFrictionImpulses`, `btVehicleRL::resolveSingleBilateral`)
- `libsrc/bullet3-3.24/BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.cpp:110-380`

### Execution Plan (5-10 lines)
1. Study Bullet Gauss-Seidel solver iteration order and impulse clamping in `btVehicleRL.cpp`.
2. Port sequential impulse resolution with warm starting and iteration passes to GPU device kernel.
3. Compare throttle acceleration and boost acceleration against CPU over 120 ticks in single environment.
4. Target: car velocity error $\le 0.1\%$ and position error $\le 1\text{ UU}$ across 120 ticks.
5. Verify handbrake slide lateral friction reduction matches CPU.

---

## Module 1.4: Car-Ball Collision Fidelity (M5.4 - Upcoming)

### CPU Oracle Reference
- `src/Sim/Car/Car.cpp:320-410` (`Car::_OnBallHit`, extra hit impulse piecewise curves, hit contact margin)
- `src/Sim/RLConst.h:80-140` (`BALL_COLLISION_RADIUS`, car hitbox bounds, extra impulse constants)

### Execution Plan (5-10 lines)
1. Investigate car resting Z height at impact: resolve CPU $Z=15.5\text{ UU}$ vs GPU $Z=17.0\text{ UU}$ suspension compression during hard acceleration.
2. Align OBB-sphere penetration depth and contact normal formulation in `src/cuda/step_kernel.cu`.
3. Mirror exact piecewise linear velocity curve for extra hit impulse from `Car.cpp`.
4. Validate `kickoff_goalie`, side touch, ceiling touch, aerial ball, and rolling ball at +1, +10, and +60 ticks.
5. Target: ball post-impact velocity error $\le 0.5\%$ per component and exit angle error $\le 0.5^\circ$.
