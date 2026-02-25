"""
Flip / Double Jump Mechanics
=============================
Directional dodges (flips) and double jumps, including flip cancel
and Z-velocity damping during flips.
"""

from __future__ import annotations
import jax.numpy as jnp

from ..constants import (
    DT, CAR_MAX_SPEED,
    DOUBLEJUMP_MAX_DELAY, DODGE_DEADZONE,
    FLIP_INITIAL_VEL_SCALE, FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE,
    FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE, FLIP_BACKWARD_IMPULSE_SCALE_X,
    FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE,
    FLIP_TORQUE_TIME, FLIP_TORQUE_X, FLIP_TORQUE_Y,
    FLIP_Z_DAMP_START, FLIP_Z_DAMP_END, FLIP_Z_DAMP_120,
    FLIP_PITCHLOCK_TIME,
    JUMP_IMMEDIATE_FORCE,
)
from ..types import CarState, CarControls
from ..math_utils import quat_rotate_vector, get_car_forward_dir, get_car_up_dir


def handle_flip_or_double_jump(
    cars: CarState,
    controls: CarControls,
    forward_speed: jnp.ndarray,
    dt: float = DT
) -> tuple[CarState, jnp.ndarray, jnp.ndarray]:
    """
    Handle double jump / flip (dodge) mechanics.
    
    Logic (when airborne):
    - If jump pressed AND within time window AND haven't used it:
      - If stick input magnitude >= DODGE_DEADZONE: FLIP (directional dodge)
      - Else: DOUBLE JUMP (straight up impulse)
    
    Args:
        cars: Current car state
        controls: Control inputs
        forward_speed: Forward speed for each car (N, MAX_CARS)
        dt: Time step
        
    Returns:
        Updated car state, velocity impulse (N, MAX_CARS, 3), torque impulse (N, MAX_CARS, 3)
    """
    # Edge detection: only trigger on button PRESS
    jump_pressed = controls.jump & ~cars.last_jump_pressed
    
    is_on_ground = cars.is_on_ground
    has_jumped = cars.has_jumped
    has_flipped = cars.has_flipped
    has_double_jumped = cars.has_double_jumped
    is_flipping = cars.is_flipping
    is_jumping = cars.is_jumping
    flip_timer = cars.flip_timer
    flip_rel_torque = cars.flip_rel_torque
    air_time = cars.air_time
    air_time_since_jump = cars.air_time_since_jump
    
    forward_dir = get_car_forward_dir(cars.quat)
    up_dir = get_car_up_dir(cars.quat)
    
    forward_2d = forward_dir[..., :2]
    forward_2d_norm = forward_2d / (jnp.linalg.norm(forward_2d, axis=-1, keepdims=True) + 1e-8)
    right_2d = jnp.stack([-forward_2d_norm[..., 1], forward_2d_norm[..., 0]], axis=-1)
    
    # Ground reset - when car is on ground, reset flip/double jump ability
    has_double_jumped = jnp.where(is_on_ground, False, has_double_jumped)
    has_flipped = jnp.where(is_on_ground, False, has_flipped)
    air_time = jnp.where(is_on_ground, 0.0, air_time + dt)
    flip_timer = jnp.where(is_on_ground, 0.0, flip_timer)
    
    # Track air time since jump ended
    not_jumping_anymore = has_jumped & ~cars.is_jumping
    air_time_since_jump = jnp.where(
        is_on_ground, 0.0,
        jnp.where(not_jumping_anymore, air_time_since_jump + dt, 0.0)
    )
    
    # Check if can use double jump / flip
    is_airborne = ~is_on_ground
    within_time = air_time_since_jump < DOUBLEJUMP_MAX_DELAY
    can_use = is_airborne & within_time & ~has_flipped & ~has_double_jumped
    
    input_magnitude = jnp.abs(controls.yaw) + jnp.abs(controls.pitch) + jnp.abs(controls.roll)
    is_flip_input = input_magnitude >= DODGE_DEADZONE
    
    trigger = jump_pressed & can_use
    do_flip = trigger & is_flip_input
    do_double_jump = trigger & ~is_flip_input
    
    # Double jump
    double_jump_impulse = jnp.where(
        do_double_jump[..., None],
        up_dir * JUMP_IMMEDIATE_FORCE,
        jnp.zeros_like(up_dir)
    )
    has_double_jumped = jnp.where(do_double_jump, True, has_double_jumped)
    
    # Flip direction
    dodge_dir_x = -controls.pitch
    dodge_dir_y = controls.yaw + controls.roll
    
    both_small = (jnp.abs(dodge_dir_y) < 0.1) & (jnp.abs(dodge_dir_x) < 0.1)
    dodge_mag = jnp.sqrt(dodge_dir_x**2 + dodge_dir_y**2 + 1e-8)
    dodge_dir_x_norm = jnp.where(both_small, 0.0, dodge_dir_x / dodge_mag)
    dodge_dir_y_norm = jnp.where(both_small, 0.0, dodge_dir_y / dodge_mag)
    
    new_flip_rel_torque = jnp.stack([
        -dodge_dir_y_norm,
        dodge_dir_x_norm,
        jnp.zeros_like(dodge_dir_x)
    ], axis=-1)
    
    # Zero small components of dodge direction for velocity impulse
    dodge_dir_x_norm = jnp.where(jnp.abs(dodge_dir_x_norm) < 0.1, 0.0, dodge_dir_x_norm)
    dodge_dir_y_norm = jnp.where(jnp.abs(dodge_dir_y_norm) < 0.1, 0.0, dodge_dir_y_norm)
    
    has_dodge_input = (jnp.abs(dodge_dir_x_norm) > 0.01) | (jnp.abs(dodge_dir_y_norm) > 0.01)
    flip_rel_torque = jnp.where(do_flip[..., None], new_flip_rel_torque, flip_rel_torque)
    
    # Velocity impulse
    forward_speed_ratio = jnp.abs(forward_speed) / CAR_MAX_SPEED
    
    dodging_backwards = jnp.where(
        jnp.abs(forward_speed) < 100.0,
        dodge_dir_x_norm < 0.0,
        (dodge_dir_x_norm >= 0.0) != (forward_speed >= 0.0)
    )
    
    impulse_x = dodge_dir_x_norm * FLIP_INITIAL_VEL_SCALE
    impulse_y = dodge_dir_y_norm * FLIP_INITIAL_VEL_SCALE
    
    max_scale_x = jnp.where(dodging_backwards, FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE, FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE)
    impulse_x = impulse_x * ((max_scale_x - 1) * forward_speed_ratio + 1)
    impulse_y = impulse_y * ((FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE - 1) * forward_speed_ratio + 1)
    
    impulse_x = jnp.where(dodging_backwards, impulse_x * FLIP_BACKWARD_IMPULSE_SCALE_X, impulse_x)
    
    flip_vel_xy = impulse_x[..., None] * forward_2d_norm + impulse_y[..., None] * right_2d
    flip_vel_impulse = jnp.concatenate([flip_vel_xy, jnp.zeros_like(impulse_x[..., None])], axis=-1)
    
    flip_vel_impulse = jnp.where(
        (do_flip & has_dodge_input)[..., None],
        flip_vel_impulse,
        jnp.zeros_like(flip_vel_impulse)
    )
    
    has_flipped = jnp.where(do_flip, True, has_flipped)
    is_flipping = jnp.where(do_flip, True, is_flipping)
    flip_timer = jnp.where(do_flip, 0.0, flip_timer)
    
    # Ongoing flip torque
    is_flipping = has_flipped & (flip_timer < FLIP_TORQUE_TIME)
    
    flip_torque_local = flip_rel_torque * jnp.array([FLIP_TORQUE_X, FLIP_TORQUE_Y, 0.0])
    
    # === FLIP CANCEL (C++ pitch lock logic) ===
    in_pitchlock = flip_timer < FLIP_PITCHLOCK_TIME
    flip_pitch_dir = flip_rel_torque[..., 1]
    pitch_input = controls.pitch
    
    is_cancelling = (flip_pitch_dir * pitch_input) > 0
    cancel_scale = jnp.where(
        is_flipping & in_pitchlock & is_cancelling,
        1.0 - jnp.abs(pitch_input),
        1.0
    )
    
    flip_torque_local = flip_torque_local.at[..., 1].set(
        flip_torque_local[..., 1] * cancel_scale
    )
    
    flip_torque_world = quat_rotate_vector(cars.quat, flip_torque_local)
    
    flip_torque = jnp.where(
        is_flipping[..., None],
        flip_torque_world,
        jnp.zeros_like(flip_torque_world)
    )
    
    flip_timer = jnp.where(has_flipped, flip_timer + dt, flip_timer)
    
    total_vel_impulse = double_jump_impulse + flip_vel_impulse
    new_last_jump_pressed = controls.jump
    
    updated_cars = cars.replace(
        has_jumped=has_jumped,
        has_flipped=has_flipped,
        has_double_jumped=has_double_jumped,
        is_flipping=is_flipping,
        flip_timer=flip_timer,
        flip_rel_torque=flip_rel_torque,
        air_time=air_time,
        air_time_since_jump=air_time_since_jump,
        last_jump_pressed=new_last_jump_pressed,
    )
    
    return updated_cars, total_vel_impulse, flip_torque


def apply_flip_z_damping(
    vel: jnp.ndarray,
    is_flipping: jnp.ndarray,
    flip_timer: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    Apply Z velocity damping during flip.
    
    Args:
        vel: Velocity (N, MAX_CARS, 3)
        is_flipping: Flip state (N, MAX_CARS)
        flip_timer: Time since flip started (N, MAX_CARS)
        dt: Time step
        
    Returns:
        Velocity with Z damping applied
    """
    in_torque_time = flip_timer <= FLIP_TORQUE_TIME
    past_damp_start = flip_timer >= FLIP_Z_DAMP_START
    z_negative = vel[..., 2] < 0
    before_damp_end = flip_timer < FLIP_Z_DAMP_END
    
    should_damp = is_flipping & in_torque_time & past_damp_start & (z_negative | before_damp_end)
    
    damp_factor = jnp.power(1 - FLIP_Z_DAMP_120, dt / (1/120))
    
    vel_z_damped = vel[..., 2] * damp_factor
    vel_z = jnp.where(should_damp, vel_z_damped, vel[..., 2])
    
    return vel.at[..., 2].set(vel_z)
