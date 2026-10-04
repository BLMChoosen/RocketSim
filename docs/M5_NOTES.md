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

## Module 1.2: Ball Bounces Fidelity (M5.2 - Upcoming)

### CPU Oracle Reference
- `src/Sim/Ball/Ball.cpp:15-80` (`Ball::Step` and restitution/friction callbacks)
- `src/Sim/Arena/Arena.cpp:45-120` (`Arena::InitArena` collision meshes and contact callbacks)
- `libsrc/bullet3-3.24/BulletCollision/NarrowPhaseCollision/btPersistentManifold.h`

### Execution Plan (5-10 lines)
1. Isolate 1 environment, single-bounce drops across 8 distinct surfaces: floor, angled ground with spin, side wall, back wall, ceiling, curved corner ramps, goal posts, crossbar.
2. Measure velocity and angular velocity deltas at +1, +5, and +30 ticks post-impact against CPU reference.
3. Run CPU vs CPU perturbation baseline ($10^{-3}$ delta) to isolate Lyapunov exponent from systematic error.
4. Correct analytical SDF normal evaluation in `include/rocketsim_cuda/physics/arena_sdf.cuh` where ramp or goalpost normals diverge.
5. Re-evaluate bounce telemetry tables and verify convergence across all bounce scenarios.

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
