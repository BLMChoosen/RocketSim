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

## Module 1.3: Longitudinal Tire Friction & Car Contact Solver (M5.3 - Completed)

### CPU Oracle Reference
- **Lifecycle partitioning & execution order:** `src/Sim/Car/Car.cpp:103, 122, 141`
  - `_bulletVehicle.updateVehicleFirst(tickTime)` (raycast & friction impulses computed from $t-1$ cached controls).
  - `Car::_UpdateWheels(tickTime, ...)` (computes $t+1$ engine/brake/steer forces and friction curves using pre-impulse velocity).
  - `_bulletVehicle.updateVehicleSecond(tickTime)` (computes suspension forces, applies suspension and tire friction impulses to rigid body).
- **Drive torque & brake torque scaling:** `src/Sim/Car/Car.cpp:380-415`, `src/RLConst.h:84-85`
  - `THROTTLE_TORQUE_AMOUNT * UU_TO_BT = (180.0f * 400.0f) * 0.02f = 1440.0f`.
  - `BRAKE_TORQUE_AMOUNT * UU_TO_BT = (180.0f * (14.25f + 1.0f / 3.0f)) * 0.02f = 52.5f`.
- **Friction curves & unprojected lateral direction:** `src/Sim/Car/Car.cpp:440-455`
  - `latDir = wheel.m_worldTransform.getBasis().getColumn(1)` (unprojected wheel axle vector).
  - `longDir = latDir.cross(wheel.m_raycastInfo.m_contactNormalWS)`.
  - `baseFriction = abs(crossVec.dot(latDir))`.
  - `frictionCurveInput = baseFriction / (abs(crossVec.dot(longDir)) + baseFriction)`.
- **Suspension force calculation & pushback gating:** `src/Sim/btVehicleRL/btVehicleRL.cpp:270-303`
  - Gated on `if (wheel.m_wheelsSuspensionForce != 0)` before applying `(suspForce * dt) + extraPushback`.
- **Bilateral friction & planar offset:** `src/Sim/btVehicleRL/btVehicleRL.cpp:306-395`, `libsrc/bullet3-3.24/BulletDynamics/ConstraintSolver/btContactConstraint.cpp:147`
  - Bilateral constraint damping: `contactDamping = 0.2f`.
  - Planar offset: `wheelRelPos = wheelContactOffset - upDir * contactUpDot` to eliminate sliding roll torque.

### Implementation Summary
1. **Lifecycle Alignment (`src/cuda/step_kernel.cu`):**
   - Reordered `StepCarsDevice`: Wheel raycasts $\to$ Wheel dynamics (`update_car_wheel_dynamics` reading pre-impulse velocity) $\to$ Air control, jump, auto-flip, auto-roll, boost $\to$ `apply_suspension_and_friction` (accumulating suspension and tire friction impulses) $\to$ Symplectic linear integration with external forces $\to$ Angular integration and chassis collision resolution.
2. **Suspension Pushback Gating (`include/rocketsim_cuda/physics/suspension.cuh`):**
   - Gated `extra_pushback` and normal suspension impulse application strictly on `if (susp_force > 0.0f)`.
   - Verified planar tire friction offset and 0.2f bilateral damping.
3. **Unprojected Wheel Frame in Friction (`include/rocketsim_cuda/physics/car_dynamics.cuh`):**
   - Switched `base_friction` to use unprojected `lat_dir = basis.right * cos(steer) - basis.forward * sin(steer)` and `long_dir = lat_dir.cross(hit_normal)` matching `Car.cpp:440-455`.

### Empirical Verification Metrics
- **Throttle Scenario (`throttle`, 120 ticks, 1 env):**
  - **Car Pos Y Error:** Reduced from **$2.715\text{ UU}$** to **$0.01367\text{ UU}$** ($198\times$ improvement, target $\le 1.0\text{ UU}$ **PASSED**).
  - **Car Vel Y Error:** Reduced from **$4.814\text{ UU/s}$ ($0.53\%$)** to **$0.0005493\text{ UU/s}$ ($6.044 \times 10^{-7} = 0.00006\%$)** ($8760\times$ improvement, target $\le 0.1\%$ **PASSED**).
- **Boost Scenario (`boost`, 120 ticks, 1 env):**
  - **Car Pos Y Error:** Reduced from **$1.856\text{ UU}$** to **$0.01025\text{ UU}$** ($181\times$ improvement, target $\le 1.0\text{ UU}$ **PASSED**).
  - **Car Vel Y Error:** Reduced from **$2.582\text{ UU/s}$ ($0.168\%$)** to **$0.001831\text{ UU/s}$ ($1.195 \times 10^{-6} = 0.00012\%$)** ($1400\times$ improvement, target $\le 0.1\%$ **PASSED**).
- **Unit & Integration Suites:**
  - Python test suite: **35/35 tests passing** in 4.50s.
  - Analytical SDF unit test suite: **8/8 tests passing** (`test_sdf.exe`).

---

## Module 1.4: Car-Ball Collision Fidelity & Impact Height Parity (M5.4 - Completed)

### CPU Oracle Reference
- **Collision margin & shape extents:** `libsrc/bullet3-3.24/BulletCollision/CollisionShapes/btCollisionMargin.h:22` (`CONVEX_DISTANCE_MARGIN = btScalar(0.04)` = $2.0\text{ UU}$).
  - `libsrc/bullet3-3.24/BulletCollision/CollisionShapes/btBoxShape.cpp:22-23` (`m_implicitShapeDimensions = boxHalfExtents - margin`).
- **Sphere-box narrowphase algorithm:** `libsrc/bullet3-3.24/BulletCollision/CollisionDispatch/btSphereBoxCollisionAlgorithm.cpp:96-200`
  - `getSphereDistance`: clamps sphere center to `boxHalfExtentsWithoutMargin`, evaluates `dist2 = normal.length2()`, sets `pointOnBox = closestPoint + normal * boxMargin`.
  - `getSpherePenetration`: internal projection to closest face when sphere center is inside inner box extents.
- **Hitbox dimensions & inertia:** `src/Sim/Car/Car.cpp:208-245`, `src/Sim/Car/CarConfig/CarConfig.cpp:21, 32`
  - Octane `hitboxSize = Vec(120.507f, 86.6994f, 38.6591f)`, `hitboxPosOffset = Vec(13.8757f, 0.0f, 20.755f)`.
  - Half-extents: $\mathbf{h} = (60.2535, 43.3497, 19.32955)\text{ UU}$, Mass $M_c = 180.0\text{ BT}$, Ball mass $M_b = 30.0\text{ BT}$.
- **Split impulse & constraint solver:** `src/Sim/Arena/Arena.cpp:473-476`, `libsrc/bullet3-3.24/BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.cpp:958, 974`
  - `m_splitImpulsePenetrationThreshold = 1.0e30f; m_erp2 = 0.8f;`
  - Mass ratio distribution: $M_c / (M_c + M_b) = 6/7$ to ball, $M_b / (M_c + M_b) = 1/7$ to car.
- **Piecewise extra hit impulse curve:** `src/Sim/Ball/Ball.cpp:261-285`, `src/RLConst.h:135-140, 496-503`
  - `BALL_CAR_EXTRA_IMPULSE_FACTOR_CURVE`: $(0, 0.65) \to (500, 0.65) \to (2300, 0.55) \to (4600, 0.30)$.
  - `BALL_CAR_EXTRA_IMPULSE_Z_SCALE = 0.35f`, `BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE = 0.65f`.

### Implementation Summary
1. **Bullet Margin & Edge Rounding (`include/rocketsim_cuda/physics/contact_solver.cuh`):**
   - Implemented `BOX_MARGIN = 2.0f` (`CONVEX_DISTANCE_MARGIN = 0.04 BT = 2.0 UU`) in `test_car_ball_collision`.
   - Clamped sphere center to `inner_half = hitbox_half - 2.0f` and offset contact point: $\mathbf{x}_{box} = \mathbf{q}_{inner} + \mathbf{n}_{local} \times 2.0\text{ UU}$.
   - Ported Bullet's `getSpherePenetration` for internal projection when sphere center penetrates the inner box.
   - Eliminates the $+0.604^\circ$ normal pitch discrepancy on OBB chamfer edges.
2. **Exact Ball Surface Lever Arm (`include/rocketsim_cuda/physics/contact_solver.cuh`):**
   - Replaced $\mathbf{r}_b = \mathbf{x}_{contact} - \mathbf{x}_{ball}$ with strict sphere surface lever arm $\mathbf{r}_b = -\mathbf{n}_{world} R_{ball}$, eliminating the $14\%$ lever arm compression error.
3. **Split-Impulse Penetration Separation (`include/rocketsim_cuda/physics/contact_solver.cuh`):**
   - Replaced 100% ball push with exact Bullet split impulse separation:
     $$\Delta\mathbf{x}_{ball} = +\mathbf{n}_{world} \cdot (p \cdot 0.8 \cdot \frac{6}{7}), \quad \Delta\mathbf{x}_{car} = -\mathbf{n}_{world} \cdot (p \cdot 0.8 \cdot \frac{1}{7})$$
   - Written back updated car position $(\mathbf{x}_{car, x}, \mathbf{x}_{car, y}, \mathbf{x}_{car, z})$.
4. **Post-Solve Downward Velocity Displacement (`src/cuda/step_kernel.cu`):**
   - Applied $\Delta Z_{vel} = v_z \Delta t$ to car position upon collision detection in `StepSimulationKernel`.
   - Replicates Bullet's post-solve transform integration (`integrateTransforms`), dropping car impact height to $Z = 15.50\text{ UU}$.
5. **Differential Harness Gate Analysis (`tests/differential/harness_main.cpp`):**
   - Extended impact gate telemetry to track and report both `kickoff_goalie` and `car_ball_hit` scenarios.

### Empirical Verification Metrics
- **Kickoff Goalie Scenario (`kickoff_goalie`, 400 ticks, 1 env):**
  - **Car Pos Z at Impact:** CPU $15.50\text{ UU}$, GPU $15.49\text{ UU}$ ($\Delta = \mathbf{0.01\text{ UU}}$). Disparity resolved.
  - **Ball Exit Deflection Angle:**
    - $+1$ Tick: CPU $18.53^\circ$, GPU $18.66^\circ$ ($\Delta = \mathbf{0.12^\circ} \le 0.5^\circ$).
    - $+10$ Ticks: CPU $17.65^\circ$, GPU $17.77^\circ$ ($\Delta = \mathbf{0.12^\circ} \le 0.5^\circ$).
    - $+60$ Ticks: CPU $12.56^\circ$, GPU $12.66^\circ$ ($\Delta = \mathbf{0.10^\circ} \le 0.5^\circ$).
  - **Ball Velocity Parity (+1 Tick):**
    - $V_z$: CPU $959.8\text{ UU/s}$, GPU $962.1\text{ UU/s}$ ($\Delta = 2.4\text{ UU/s} = \mathbf{0.25\%} \le 0.5\%$).
    - $V_y$: CPU $2862.8\text{ UU/s}$, GPU $2849.7\text{ UU/s}$ ($\Delta = 13.2\text{ UU/s} = \mathbf{0.46\%} \le 0.5\%$).
    - Total speed: CPU $3019.4\text{ UU/s}$, GPU $3007.7\text{ UU/s}$ ($\Delta = 11.7\text{ UU/s} = \mathbf{0.38\%} \le 0.5\%$).
- **Car-Ball Hit Scenario (`car_ball_hit`, 200 ticks, 1 env):**
  - **Car Pos Z at Impact:** CPU $16.36\text{ UU}$, GPU $16.37\text{ UU}$ ($\Delta = \mathbf{0.01\text{ UU}}$).
  - **Ball Exit Deflection Angle:**
    - $+1$ Tick: CPU $17.01^\circ$, GPU $16.80^\circ$ ($\Delta = \mathbf{0.21^\circ} \le 0.5^\circ$).
    - $+10$ Ticks: CPU $15.74^\circ$, GPU $15.53^\circ$ ($\Delta = \mathbf{0.21^\circ} \le 0.5^\circ$).
    - $+60$ Ticks: CPU $8.39^\circ$, GPU $8.17^\circ$ ($\Delta = \mathbf{0.22^\circ} \le 0.5^\circ$).
  - **Ball Velocity Parity (+1 Tick):**
    - $V_y$: CPU $2033.9\text{ UU/s}$, GPU $2034.5\text{ UU/s}$ ($\Delta = 0.6\text{ UU/s} = \mathbf{0.03\%} \le 0.5\%$).
    - Total speed: CPU $2126.8\text{ UU/s}$, GPU $2125.0\text{ UU/s}$ ($\Delta = 1.8\text{ UU/s} = \mathbf{0.08\%} \le 0.5\%$).
- **Unit & Integration Suites:**
  - Python test suite: **35/35 tests passing** in 4.52s.
  - Analytical SDF unit test suite: **8/8 tests passing** (`test_sdf.exe`).

