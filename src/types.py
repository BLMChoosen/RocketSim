"""
State Data Structures (PyTrees)
===============================
Flax struct dataclasses for physics state.
State is represented as "Struct of Arrays" for vectorized computation.
Batch dimension (N_ENVS) is ALWAYS axis 0.
"""

from __future__ import annotations
import jax.numpy as jnp
import flax.struct as struct


@struct.dataclass
class BallState:
    """
    Ball rigid body state.
    
    All arrays have shape (N_ENVS, ...) where N_ENVS is the batch dimension.
    
    Attributes:
        pos: Ball position in world space. Shape: (N, 3)
        vel: Ball linear velocity. Shape: (N, 3)
        ang_vel: Ball angular velocity. Shape: (N, 3)
    """
    pos: jnp.ndarray      # (N, 3) - Position [x, y, z]
    vel: jnp.ndarray      # (N, 3) - Linear velocity [vx, vy, vz]
    ang_vel: jnp.ndarray  # (N, 3) - Angular velocity [wx, wy, wz]


@struct.dataclass
class CarState:
    """
    Car rigid body and internal state.
    
    All arrays have shape (N_ENVS, MAX_CARS, ...) for per-car data,
    or (N_ENVS, ...) for per-environment data.
    """
    # Rigid body state
    pos: jnp.ndarray         # (N, MAX_CARS, 3) - Position [x, y, z]
    vel: jnp.ndarray         # (N, MAX_CARS, 3) - Linear velocity
    ang_vel: jnp.ndarray     # (N, MAX_CARS, 3) - Angular velocity
    quat: jnp.ndarray        # (N, MAX_CARS, 4) - Rotation quaternion [w, x, y, z]
    
    # Internal game state
    boost_amount: jnp.ndarray    # (N, MAX_CARS) - Boost 0-100
    is_on_ground: jnp.ndarray    # (N, MAX_CARS) - bool
    wheel_contacts: jnp.ndarray  # (N, MAX_CARS, 4) - per-wheel contact bool
    handbrake_val: jnp.ndarray   # (N, MAX_CARS) - Analog handbrake 0-1
    
    # Jump/flip state
    is_jumping: jnp.ndarray        # (N, MAX_CARS)
    has_jumped: jnp.ndarray        # (N, MAX_CARS)
    jump_timer: jnp.ndarray        # (N, MAX_CARS)
    has_flipped: jnp.ndarray       # (N, MAX_CARS)
    has_double_jumped: jnp.ndarray # (N, MAX_CARS)
    is_flipping: jnp.ndarray       # (N, MAX_CARS)
    flip_timer: jnp.ndarray        # (N, MAX_CARS)
    flip_rel_torque: jnp.ndarray   # (N, MAX_CARS, 3)
    air_time: jnp.ndarray          # (N, MAX_CARS)
    air_time_since_jump: jnp.ndarray  # (N, MAX_CARS)
    last_jump_pressed: jnp.ndarray   # (N, MAX_CARS)
    
    # Status flags
    is_demoed: jnp.ndarray       # (N, MAX_CARS)
    demo_respawn_timer: jnp.ndarray # (N, MAX_CARS)
    is_supersonic: jnp.ndarray   # (N, MAX_CARS)
    supersonic_timer: jnp.ndarray  # (N, MAX_CARS)
    is_boosting: jnp.ndarray     # (N, MAX_CARS)
    boosting_time: jnp.ndarray   # (N, MAX_CARS)
    
    # Team (immutable per episode)
    team: jnp.ndarray            # (N, MAX_CARS) - int (0=Blue, 1=Orange)


@struct.dataclass
class CarControls:
    """Control inputs for cars."""
    throttle: jnp.ndarray   # (N, MAX_CARS) - [-1, 1]
    steer: jnp.ndarray      # (N, MAX_CARS) - [-1, 1]
    pitch: jnp.ndarray      # (N, MAX_CARS) - [-1, 1]
    yaw: jnp.ndarray        # (N, MAX_CARS) - [-1, 1]
    roll: jnp.ndarray       # (N, MAX_CARS) - [-1, 1]
    jump: jnp.ndarray       # (N, MAX_CARS) - bool
    boost: jnp.ndarray      # (N, MAX_CARS) - bool
    handbrake: jnp.ndarray  # (N, MAX_CARS) - bool


@struct.dataclass
class PhysicsState:
    """
    Complete physics state for all environments.
    Top-level state container for N_ENVS parallel environments.
    """
    ball: BallState
    cars: CarState
    tick_count: jnp.ndarray      # (N,)
    pad_is_active: jnp.ndarray   # (N, N_PADS_TOTAL)
    pad_timers: jnp.ndarray      # (N, N_PADS_TOTAL)
    blue_score: jnp.ndarray      # (N,)
    orange_score: jnp.ndarray    # (N,)
