# RocketSim

**GPU-accelerated Rocket League physics simulation built with JAX**

RocketSim is a high-performance, massively parallel physics simulation of Rocket League written in pure JAX. Designed for reinforcement learning and computational experiments, it runs entirely on GPU and achieves over **1 million physics steps per second** on modern hardware.

---

## Overview

This project provides a complete standalone implementation of Rocket League's gameplay physics—cars, ball, collisions, boost pads, goals, and arena boundaries—without depending on the original game. The simulation is written in a functional, JIT-compiled style optimized for NVIDIA GPUs via JAX/XLA.

Unlike object-oriented physics engines that update objects one at a time, RocketSim batches thousands of parallel game states into a single GPU kernel. This architectural choice makes it exceptionally well-suited for training ML agents with PPO, DQN, or other RL algorithms that require high simulation throughput.

---

## Why This Project Exists

Training reinforcement learning agents for complex games like Rocket League requires simulating millions of game frames. Traditional CPU-based simulators struggle to meet this demand, creating a significant bottleneck in research and experimentation.

This project reimagines the physics simulation from the ground up:
- **Pure functional design**: All state is immutable, all functions are pure
- **GPU-native execution**: Entire simulation runs on VRAM with zero CPU transfer
- **Massive parallelism**: Simulate 1,024+ environments simultaneously
- **Shape-stable arrays**: Static tensor shapes enable aggressive JIT optimization

What started as an experiment in porting C++ physics to JAX gradually became a serious engineering effort to achieve maximum GPU utilization.

---

## Tech Stack

- **JAX** – Numerical computing with GPU acceleration via XLA
- **Flax** – Immutable data structures (PyTree support)
- **NumPy** – Array initialization and utilities
- **Chex** – Runtime type checking and assertions
- **ModernGL / PyQt5** – Real-time 3D visualization (optional)

---

## Features

- **Complete physics simulation** – Car suspension, tire forces, aerodynamics, boost, jumps, flips, and supersonic mechanics
- **Ball dynamics** – Gravity, drag, angular velocity, and collision response
- **Arena collisions** – Walls, floor, ceiling, and goal detection
- **Multi-car interactions** – Car-ball collisions, car-car bumps, and demolitions
- **Boost pad pickups** – Small and large boost pad respawn timers
- **Batched environments** – Run hundreds or thousands of parallel game instances
- **JIT compilation** – Entire simulation compiles to optimized GPU kernels
- **Functional API** – Pure functions with no side effects: `new_state = step(state, actions)`

---

## Architecture

RocketSim follows a **struct-of-arrays (SoA)** design pattern where all game state is stored in batched tensors:

```python
PhysicsState = {
    ball: BallState,      # (N_ENVS, ...) shaped tensors
    cars: CarState,       # (N_ENVS, MAX_CARS, ...) shaped tensors
    pad_timers: Array,    # (N_ENVS, N_PADS)
    game_clock: Array,    # (N_ENVS,)
}
```

The main simulation loop is a single pure function:

```python
next_state = step_physics(current_state, controls)
```

No classes. No mutable objects. No pointer chasing. Just tensor operations that fuse into a single GPU kernel.

### Why This Design?

1. **GPU efficiency**: Coalesced memory access patterns maximize VRAM bandwidth
2. **XLA optimization**: Static shapes enable loop unrolling and vectorization
3. **Parallelism**: Same kernel runs on all environments simultaneously
4. **Determinism**: Pure functions with no side effects

---

## Getting Started

### Prerequisites

- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA support
- Python 3.9+
- CUDA 12.x

### Installation

```bash
# Clone the repository
git clone https://github.com/BLMChoosen/RocketSim.git
cd RocketSim

# Install dependencies
pip install -r requirements.txt
```

**Note**: JAX must be installed with CUDA support. The `requirements.txt` includes `jax[cuda12]` for NVIDIA GPUs. See [JAX installation docs](https://github.com/google/jax#installation) for other platforms.

### Basic Usage

```python
import jax
import jax.numpy as jnp
from src import create_initial_state, create_zero_controls, step_physics

# Create 1024 parallel environments
N_ENVS = 1024
state = create_initial_state(n_envs=N_ENVS)
controls = create_zero_controls(n_envs=N_ENVS)

# Simulate one physics step (1/120 second)
new_state = step_physics(state, controls)

# Run for 120 steps (1 second of game time)
for _ in range(120):
    state = step_physics(state, controls)
    
print(f"Ball position: {state.ball.pos[0]}")  # First environment
```

### RL Training Example

```python
from src import step_env, get_observations, reset_round

rng_key = jax.random.PRNGKey(0)
state = create_initial_state(n_envs=1024)

# Main training loop
for episode in range(num_episodes):
    controls = policy(get_observations(state))  # Your RL policy
    next_state, obs, rewards, dones = step_env(state, controls, rng_key)
    
    # Train your agent with (obs, rewards, dones)
    state = jax.lax.cond(
        dones.any(),
        lambda s: reset_round(s, rng_key),
        lambda s: s,
        next_state
    )
```

### Visualization

Launch the 3D visualizer to watch simulations in real-time:

```bash
# Linux/Mac
./run_viz.sh

# Windows
run_viz.bat
```

The visualizer uses ModernGL for hardware-accelerated rendering and displays car positions, ball trajectory, boost pads, and game state.

---

## Performance / Engineering Highlights

### Benchmark: 1.04 Million Steps Per Second

On an RTX PRO 6000 GPU with 1024 parallel environments:

- **14.78 million physics steps/second**
- **Over 1000× faster than CPU-based RocketSim v2.1.0**
- **Simulates ~34 hours of game time per second**

This performance is achieved through:

1. **Branchless collision logic** – No `if` statements in hot paths; use `jnp.where()` for masking
2. **JIT fusion** – XLA merges hundreds of operations into a single GPU kernel
3. **Batch normalization** – All cars, all environments, single matmul
4. **Zero CPU transfer** – Data stays in VRAM throughout entire training run

### Technical Challenges Solved

**Challenge**: Rocket League physics uses highly branching collision logic (sphere-vs-mesh, AABB tests, suspension raycasts).

**Solution**: Replaced branching with branchless masking. Calculate physics for all cases, then use `jnp.where(condition, new_val, old_val)` to apply results selectively. This wastes FLOPs but eliminates warp divergence, yielding 10× speedup on GPU.

**Challenge**: Quaternion rotations and angular velocity integration require careful numerical stability.

**Solution**: Implemented explicit quaternion normalization after every integration step. Added angular velocity clamping to prevent exploding rotations. Verified against original C++ implementation frame-by-frame.

**Challenge**: Managing state for 1024 environments × 4 cars × 34 boost pads = 140,000+ entities.

**Solution**: Flattened everything into (N_ENVS, ...) tensors. No Python loops. Everything is a `vmap` or `lax.scan`.

---

## What I Learned

- **JAX/XLA compilation pipeline**: How to write code that compiles to efficient GPU kernels
- **GPU architecture**: Memory coalescing, warp divergence, register pressure
- **Functional programming**: Pure functions, immutability, and composability at scale
- **Physics simulation**: Suspension systems, tire models, quaternion math, collision response
- **Numerical stability**: Floating-point precision, integration methods, clamping strategies
- **Performance profiling**: Using `jax.profiler` to identify bottlenecks and optimize hot paths

---

## Future Improvements

- **Multi-GPU scaling** – Use `jax.pmap` for distributed training across multiple GPUs
- **Mesh-based collisions** – Load `.obj` files for custom arenas (Hoops, Dropshot, etc.)
- **More game modes** – Implement specific rules for Hoops, Heatseeker, Snowday
- **Improved accuracy** – Fine-tune constants to match official game behavior more closely
- **Differentiable physics** – Enable gradient-based trajectory optimization
- **Documentation site** – Sphinx docs with API reference and tutorials

---

## Contributing

Contributions are welcome! If you encounter bugs or have feature requests, please open an issue. Pull requests should include tests and follow the existing code style.

**Contact**: Discord `Davi3320`

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Legal Notice

RocketSim is an independent implementation of Rocket League's physics logic and does not contain any code from the original game. This project is for educational and research purposes.

To Epic Games/Psyonix: If you have concerns about this project, please contact me on Discord and we can address them.
