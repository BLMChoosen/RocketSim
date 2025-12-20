"""
RocketSim - JAX-Based Rocket League Physics Simulation
=======================================================

A GPU-accelerated, JAX-native physics simulation of Rocket League.
Designed for massively parallel reinforcement learning.

Architecture:
- All state is immutable (Flax structs)
- All functions are pure (no side effects)
- Batch dimension (N_ENVS) is always axis 0
- Shape-stable for JIT compilation

Quick Start:
    >>> import jax
    >>> from rocket_sim import create_initial_state, create_zero_controls, step_physics
    >>> 
    >>> state = create_initial_state(n_envs=1024)
    >>> controls = create_zero_controls(n_envs=1024)
    >>> new_state = step_physics(state, controls)

For RL training:
    >>> from rocket_sim import step_env, get_observations, reset_round
    >>> 
    >>> rng_key = jax.random.PRNGKey(0)
    >>> next_state, obs, rewards, dones = step_env(state, controls, rng_key)
"""

from __future__ import annotations

# Type definitions
from .types import (
    BallState,
    CarState,
    CarControls,
    PhysicsState,
)

# Constants (for advanced users)
from .constants import (
    DT,
    GRAVITY_Z,
    BALL_RADIUS,
    BALL_MAX_SPEED,
    CAR_MAX_SPEED,
    BOOST_MAX,
    ARENA_EXTENT_X,
    ARENA_EXTENT_Y,
    ARENA_HEIGHT,
)

# Math utilities
from .math_utils import (
    quat_multiply,
    quat_normalize,
    quat_from_angular_velocity,
    quat_rotate_vector,
    quat_to_rotation_matrix,
    quat_from_yaw,
    get_forward_up_right,
    get_car_forward_dir,
    get_car_up_dir,
    get_car_right_dir,
)

# Main simulation functions
from .game import (
    # State initialization
    create_initial_state,
    create_initial_ball_state,
    create_initial_car_state,
    create_zero_controls,
    # Main simulation
    step_physics,
    step_env,
    reset_round,
    # Observations
    get_observations,
    # Sub-systems
    resolve_boost_pads,
    check_goal,
    step_cars,
)

# Physics functions (for customization)
from .physics import (
    step_ball,
    apply_gravity,
    apply_ball_drag,
    integrate_position,
    integrate_rotation,
    solve_suspension_and_tires,
)

# Mechanics (for customization)
from .mechanics import (
    handle_jump,
    handle_flip_or_double_jump,
    apply_flip_z_damping,
    apply_boost,
    update_supersonic_status,
)

# Collision (for customization)
from .collision import (
    arena_sdf,
    resolve_ball_arena_collision,
    resolve_car_arena_collision,
    resolve_car_ball_collision,
    resolve_car_car_collision,
)

__version__ = "0.1.0"
__all__ = [
    # Types
    "BallState",
    "CarState", 
    "CarControls",
    "PhysicsState",
    # Constants
    "DT",
    "GRAVITY_Z",
    "BALL_RADIUS",
    "BALL_MAX_SPEED",
    "CAR_MAX_SPEED",
    "BOOST_MAX",
    "ARENA_EXTENT_X",
    "ARENA_EXTENT_Y",
    "ARENA_HEIGHT",
    # Main API
    "create_initial_state",
    "create_initial_ball_state",
    "create_initial_car_state",
    "create_zero_controls",
    "step_physics",
    "step_env",
    "reset_round",
    "get_observations",
    # Math
    "quat_multiply",
    "quat_normalize",
    "quat_from_angular_velocity",
    "quat_rotate_vector",
    "quat_to_rotation_matrix",
    "quat_from_yaw",
    "get_forward_up_right",
    "get_car_forward_dir",
    "get_car_up_dir",
    "get_car_right_dir",
    # Physics
    "step_ball",
    "apply_gravity",
    "apply_ball_drag",
    "integrate_position",
    "integrate_rotation",
    "solve_suspension_and_tires",
    # Mechanics
    "handle_jump",
    "handle_flip_or_double_jump",
    "apply_flip_z_damping",
    "apply_boost",
    "update_supersonic_status",
    # Collision
    "arena_sdf",
    "resolve_ball_arena_collision",
    "resolve_car_arena_collision",
    "resolve_car_ball_collision",
    "resolve_car_car_collision",
    "resolve_boost_pads",
    "check_goal",
    "step_cars",
]
