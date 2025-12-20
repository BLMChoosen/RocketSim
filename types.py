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
    
    Attributes:
        pos: Car position in world space. Shape: (N, MAX_CARS, 3)
        vel: Car linear velocity. Shape: (N, MAX_CARS, 3)
        ang_vel: Car angular velocity. Shape: (N, MAX_CARS, 3)
        quat: Car rotation quaternion [w, x, y, z]. Shape: (N, MAX_CARS, 4)
        boost_amount: Current boost (0-100). Shape: (N, MAX_CARS)
        is_on_ground: Ground contact flag. Shape: (N, MAX_CARS)
        wheel_contacts: Per-wheel contact flags. Shape: (N, MAX_CARS, 4)
        is_jumping: Currently in jump state. Shape: (N, MAX_CARS)
        jump_timer: Time since jump started. Shape: (N, MAX_CARS)
        has_flipped: Has used flip/double-jump. Shape: (N, MAX_CARS)
        has_double_jumped: Has used double jump. Shape: (N, MAX_CARS)
        is_demoed: Car is demolished. Shape: (N, MAX_CARS)
        is_supersonic: Currently supersonic. Shape: (N, MAX_CARS)
        team: Team index (0=Blue, 1=Orange). Shape: (N, MAX_CARS)
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
    
    # Jump/flip state
    is_jumping: jnp.ndarray        # (N, MAX_CARS) - Currently holding jump
    has_jumped: jnp.ndarray        # (N, MAX_CARS) - Used first jump to get airborne
    jump_timer: jnp.ndarray        # (N, MAX_CARS) - Time since jump started
    has_flipped: jnp.ndarray       # (N, MAX_CARS) - Has executed a flip
    has_double_jumped: jnp.ndarray # (N, MAX_CARS) - Has used double jump (no stick input)
    is_flipping: jnp.ndarray       # (N, MAX_CARS) - Currently in flip animation
    flip_timer: jnp.ndarray        # (N, MAX_CARS) - Time since flip started
    flip_rel_torque: jnp.ndarray   # (N, MAX_CARS, 3) - Flip torque direction (body frame)
    air_time: jnp.ndarray          # (N, MAX_CARS) - Total time airborne
    air_time_since_jump: jnp.ndarray  # (N, MAX_CARS) - Time airborne after jump ended
    last_jump_pressed: jnp.ndarray   # (N, MAX_CARS) - Jump button state last frame
    
    # Status flags
    is_demoed: jnp.ndarray       # (N, MAX_CARS) - bool
    demo_respawn_timer: jnp.ndarray # (N, MAX_CARS) - float
    is_supersonic: jnp.ndarray   # (N, MAX_CARS) - bool
    supersonic_timer: jnp.ndarray  # (N, MAX_CARS) - float
    
    # Team (immutable per episode)
    team: jnp.ndarray            # (N, MAX_CARS) - int (0=Blue, 1=Orange)


@struct.dataclass
class CarControls:
    """
    Control inputs for cars.
    
    Attributes:
        throttle: Forward/back input [-1, 1]. Shape: (N, MAX_CARS)
        steer: Steering input [-1, 1]. Shape: (N, MAX_CARS)
        pitch: Air pitch input [-1, 1]. Shape: (N, MAX_CARS)
        yaw: Air yaw input [-1, 1]. Shape: (N, MAX_CARS)
        roll: Air roll input [-1, 1]. Shape: (N, MAX_CARS)
        jump: Jump button pressed. Shape: (N, MAX_CARS) bool
        boost: Boost button held. Shape: (N, MAX_CARS) bool
        handbrake: Handbrake/powerslide held. Shape: (N, MAX_CARS) bool
    """
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
    
    This is the top-level state container, holding ball and car states
    for N_ENVS parallel environments.
    
    Attributes:
        ball: Ball state for all environments.
        cars: Car state for all cars in all environments.
        tick_count: Number of physics ticks elapsed. Shape: (N,)
        pad_is_active: Boost pad active state. Shape: (N, N_PADS_TOTAL)
        pad_timers: Boost pad cooldown timers. Shape: (N, N_PADS_TOTAL)
        blue_score: Blue team scored this tick. Shape: (N,)
        orange_score: Orange team scored this tick. Shape: (N,)
    """
    ball: BallState
    cars: CarState
    tick_count: jnp.ndarray      # (N,) - Physics tick counter
    pad_is_active: jnp.ndarray   # (N, N_PADS_TOTAL) - Boost pad active flags
    pad_timers: jnp.ndarray      # (N, N_PADS_TOTAL) - Cooldown timers
    blue_score: jnp.ndarray      # (N,) - Blue scored flag (bool)
    orange_score: jnp.ndarray    # (N,) - Orange scored flag (bool)
