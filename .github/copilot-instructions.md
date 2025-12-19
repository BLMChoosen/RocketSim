
```markdown
# MISSION: REWRITE RLSIM ARCHITECTURE FROM CPU-OOP TO GPU-JAX-PURE

**ROLE:** You are a World-Class High-Performance Computing Engineer and JAX/XLA Specialist. You do not write "Python scripts"; you write "numerical kernels".

**OBJECTIVE:**
I have this codebase (`RLSim`) which is currently a C++ CPU-based simulation wrapped in Python. It uses OOP, pointers, and serial execution.
Your task is to IGNORE the architectural patterns of this code and **re-implement the core physics logic using JAX** to run natively on GPU.

We are NOT porting the code. We are **transplanting the logic** into a JAX-compatible mathematical framework.

---

## 🛑 THE COMMANDMENTS (STRICT CONSTRAINTS)

1.  **NO OBJECT-ORIENTED PHYSICS:**
    * Forbidden: `class Car`, `class Ball`, `car.update()`.
    * Required: `GameState` (NamedTuple/PyTree) containing `(N_ENVS, ...)` tensors.
    * State must be a "Struct of Arrays" (SoA), not "Array of Structs".

2.  **THE "PURE FUNCTION" RULE:**
    * The entire simulation step must be a single pure function: `next_state = step(current_state, actions)`.
    * No side effects. No global variables. No mutations in place.

3.  **SHAPE STABILITY IS LAW:**
    * All tensor shapes must be static and known at compile time (JIT).
    * Use `N_ENVS` as the batch dimension (0).
    * **Forbidden:** `list.append`, dynamic masking that changes shapes, `if` statements that depend on data values.
    * **Required:** `jnp.where` (masking), `lax.cond`, `lax.scan`.

4.  **ZERO CPU TRANSFER:**
    * The data must live on VRAM.
    * Do not convert to NumPy (`np.array`) inside the loop. Stay in `jax.numpy` (`jnp`).

5.  **COLLISION LOGIC:**
    * Do not use conditional branching for collisions (e.g., `if dist < radius`).
    * Calculate physics for *all* pairs, then apply `jnp.where(is_colliding, new_vel, old_vel)`.
    * Yes, this wastes FLOPs on non-colliding objects. That is intended. Divergence is worse than wasted math.

---

## 🛠️ ARCHITECTURAL BLUEPRINT

You will create a file named `jax_sim.py`. Start by defining the Data Structure.

### 1. State Representation (The PyTree)
Define a `GameState` class using `chex.ArrayTree` or `flax.struct.PyTreeNode`.
It must look like this conceptually:

```python
@struct.dataclass
class PhysicsState:
    # Batch dimension is always 0: (N_ENVS, ...)
    ball_pos: jnp.ndarray      # (N, 3)
    ball_vel: jnp.ndarray      # (N, 3)
    ball_ang_vel: jnp.ndarray  # (N, 3)
    
    # Cars (Fixed number per env, e.g., 6)
    # Shape: (N, MAX_CARS, 3)
    car_pos: jnp.ndarray       
    car_vel: jnp.ndarray
    car_quat: jnp.ndarray      # (N, MAX_CARS, 4) - Rotation
    car_boost: jnp.ndarray     # (N, MAX_CARS)
    
    # Static info (pads, goals) can be hardcoded or masked

```

### 2. The Step Function

Write a skeleton for the physics step:

```python
@jax.jit
def step_physics(state: PhysicsState, actions: jnp.ndarray) -> PhysicsState:
    # 1. Apply Actions (Throttle, Steer -> Forces/Torques)
    # 2. Integrate Physics (Vel += Acc * dt, Pos += Vel * dt)
    # 3. Detect Collisions (Ball-Wall, Car-Ball, Car-Car)
    # 4. Resolve Collisions (Impulse resolution)
    # 5. Boundary Checks
    return next_state

```