"""
Jump, Flip, and Boost Mechanics
================================
Game mechanics specific to Rocket League: jumping, flipping/dodging, 
double jumps, and boost consumption/acceleration.
"""

from __future__ import annotations
import jax.numpy as jnp

from sim_constants import (
    DT, CAR_MAX_SPEED,
    JUMP_IMMEDIATE_FORCE, JUMP_ACCEL, JUMP_MIN_TIME, JUMP_MAX_TIME,
    JUMP_RESET_TIME_PAD,
    DOUBLEJUMP_MAX_DELAY, DODGE_DEADZONE,
    FLIP_INITIAL_VEL_SCALE, FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE,
    FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE, FLIP_BACKWARD_IMPULSE_SCALE_X,
    FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE,
    FLIP_TORQUE_TIME, FLIP_TORQUE_X, FLIP_TORQUE_Y,
    FLIP_Z_DAMP_START, FLIP_Z_DAMP_END, FLIP_Z_DAMP_120,
    FLIP_PITCHLOCK_TIME,
    BOOST_ACCEL_GROUND, BOOST_ACCEL_AIR, BOOST_USED_PER_SECOND, BOOST_MAX, BOOST_MIN_TIME,
    SUPERSONIC_START_SPEED, SUPERSONIC_MAINTAIN_MIN_SPEED,
    SUPERSONIC_MAINTAIN_MAX_TIME,
)
from sim_types import CarState, CarControls
from math_utils import quat_rotate_vector, get_car_forward_dir, get_car_up_dir, get_car_right_dir


# =============================================================================
# JUMP MECHANICS
# =============================================================================


def handle_jump(
    cars: CarState,
    controls: CarControls,
    dt: float = DT
) -> tuple[CarState, jnp.ndarray]:
    """
    Handle first jump mechanics (ground -> air).
    
    Logic:
    1. On ground + jump pressed -> Start jumping
    2. While jumping, apply upward force
    3. Jumping ends when: jump released OR max time exceeded
    4. Reset jump state when returning to ground
    
    Args:
        cars: Current car state
        controls: Control inputs
        dt: Time step
        
    Returns:
        Updated car state, jump impulse to apply (N, MAX_CARS, 3),
        z_vel_reset_mask (N, MAX_CARS) - True when Z velocity should be zeroed before applying impulse
    """
    jump_pressed = controls.jump
    is_on_ground = cars.is_on_ground
    is_jumping = cars.is_jumping
    has_jumped = cars.has_jumped
    jump_timer = cars.jump_timer
    
    up_dir = get_car_up_dir(cars.quat)
    
    # Ground reset logic
    can_reset = is_on_ground & ~is_jumping
    reset_allowed = (jump_timer >= JUMP_MIN_TIME + JUMP_RESET_TIME_PAD) | ~has_jumped
    do_reset = can_reset & reset_allowed
    
    has_jumped = jnp.where(do_reset, False, has_jumped)
    jump_timer = jnp.where(do_reset, 0.0, jump_timer)
    
    # Continue or end jumping
    can_continue = (jump_timer < JUMP_MIN_TIME) | (jump_pressed & (jump_timer < JUMP_MAX_TIME))
    is_jumping = jnp.where(is_jumping, can_continue, is_jumping)
    
    # Start new jump
    start_jump = is_on_ground & ~is_jumping & jump_pressed
    is_jumping = jnp.where(start_jump, True, is_jumping)
    jump_timer = jnp.where(start_jump, 0.0, jump_timer)
    
    # Initial jump impulse
    jump_impulse = jnp.where(
        start_jump[..., None],
        up_dir * JUMP_IMMEDIATE_FORCE,
        jnp.zeros_like(up_dir)
    )
    
    # Continuous jump force
    JUMP_PRE_MIN_ACCEL_SCALE = 0.62
    force_scale = jnp.where(jump_timer < JUMP_MIN_TIME, JUMP_PRE_MIN_ACCEL_SCALE, 1.0)
    
    jump_accel = jnp.where(
        is_jumping[..., None],
        up_dir * JUMP_ACCEL * force_scale[..., None] * dt,
        jnp.zeros_like(up_dir)
    )
    
    # Update timers
    has_jumped = jnp.where(is_jumping, True, has_jumped)
    jump_timer = jnp.where(
        is_jumping | has_jumped,
        jump_timer + dt,
        jump_timer
    )
    
    total_jump_vel_delta = jump_impulse + jump_accel
    
    updated_cars = cars.replace(
        is_jumping=is_jumping,
        has_jumped=has_jumped,
        jump_timer=jump_timer,
    )
    
    return updated_cars, total_jump_vel_delta, start_jump


# =============================================================================
# FLIP / DOUBLE JUMP MECHANICS
# =============================================================================


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
    # is_jumping comes from handle_jump_start, has_jumped means first jump was initiated
    not_jumping_anymore = has_jumped & ~cars.is_jumping
    air_time_since_jump = jnp.where(
        is_on_ground, 0.0,
        jnp.where(not_jumping_anymore, air_time_since_jump + dt, 0.0)
    )
    
    # Check if can use double jump / flip
    # C++ logic: jumpPressed && airTimeSinceJump < DOUBLEJUMP_MAX_DELAY && !hasFlipped && !hasDoubleJumped
    # Note: C++ does NOT require has_jumped. If the car is bumped airborne without jumping,
    # airTimeSinceJump stays at 0 (< DOUBLEJUMP_MAX_DELAY), so the car CAN flip/double-jump.
    # This is how flip resets after bumps work in RL.
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
    # C++ order: normalize first, compute flip torque, THEN zero small components
    dodge_dir_x = -controls.pitch
    dodge_dir_y = controls.yaw + controls.roll
    
    # C++: if both inputs < 0.1, zero out; otherwise normalize
    both_small = (jnp.abs(dodge_dir_y) < 0.1) & (jnp.abs(dodge_dir_x) < 0.1)
    dodge_mag = jnp.sqrt(dodge_dir_x**2 + dodge_dir_y**2 + 1e-8)
    dodge_dir_x_norm = jnp.where(both_small, 0.0, dodge_dir_x / dodge_mag)
    dodge_dir_y_norm = jnp.where(both_small, 0.0, dodge_dir_y / dodge_mag)
    
    # C++: flipRelTorque is set from the NORMALIZED direction BEFORE small-component zeroing
    new_flip_rel_torque = jnp.stack([
        -dodge_dir_y_norm,
        dodge_dir_x_norm,
        jnp.zeros_like(dodge_dir_x)
    ], axis=-1)
    
    # C++: THEN zero small components of dodge direction for velocity impulse
    dodge_dir_x_norm = jnp.where(jnp.abs(dodge_dir_x_norm) < 0.1, 0.0, dodge_dir_x_norm)
    dodge_dir_y_norm = jnp.where(jnp.abs(dodge_dir_y_norm) < 0.1, 0.0, dodge_dir_y_norm)
    
    # C++: !dodgeDir.fuzzyZero() — true if EITHER component is non-negligible
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
    # When flipping within FLIP_PITCHLOCK_TIME, pitch input can cancel the pitch
    # component of the flip torque. If the player's pitch input opposes the flip's
    # pitch direction, the pitch torque is reduced by |pitch_input|.
    in_pitchlock = flip_timer < FLIP_PITCHLOCK_TIME
    flip_pitch_dir = flip_rel_torque[..., 1]  # Y component = pitch direction
    pitch_input = controls.pitch
    
    # Cancel occurs when pitch input sign MATCHES the flip pitch direction
    # C++: if (RS_SGN(relDodgeTorque.y()) == RS_SGN(controls.pitch))
    # When they have the SAME sign, the player is cancelling the flip
    is_cancelling = (flip_pitch_dir * pitch_input) > 0
    cancel_scale = jnp.where(
        is_flipping & in_pitchlock & is_cancelling,
        1.0 - jnp.abs(pitch_input),
        1.0
    )
    
    # Apply cancel only to pitch (Y) component, leave roll (X) untouched
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


def apply_flip_z_damping(vel: jnp.ndarray, is_flipping: jnp.ndarray, flip_timer: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
    Apply Z velocity damping during flip.
    
    C++ logic:
    if (flipTime <= FLIP_TORQUE_TIME):
        if (flipTime >= FLIP_Z_DAMP_START AND (vel.z < 0 OR flipTime < FLIP_Z_DAMP_END)):
            vel.z *= pow(1 - FLIP_Z_DAMP_120, dt / (1/120))
    
    Args:
        vel: Velocity (N, MAX_CARS, 3)
        is_flipping: Flip state (N, MAX_CARS)
        flip_timer: Time since flip started (N, MAX_CARS)
        dt: Time step
        
    Returns:
        Velocity with Z damping applied
    """
    # C++: flipTime <= FLIP_TORQUE_TIME AND flipTime >= FLIP_Z_DAMP_START AND (vel.z < 0 OR flipTime < FLIP_Z_DAMP_END)
    in_torque_time = flip_timer <= FLIP_TORQUE_TIME
    past_damp_start = flip_timer >= FLIP_Z_DAMP_START
    z_negative = vel[..., 2] < 0
    before_damp_end = flip_timer < FLIP_Z_DAMP_END
    
    should_damp = is_flipping & in_torque_time & past_damp_start & (z_negative | before_damp_end)
    
    damp_factor = jnp.power(1 - FLIP_Z_DAMP_120, dt / (1/120))
    
    vel_z_damped = vel[..., 2] * damp_factor
    vel_z = jnp.where(should_damp, vel_z_damped, vel[..., 2])
    
    return vel.at[..., 2].set(vel_z)


# =============================================================================
# BOOST MECHANICS
# =============================================================================


def apply_boost(
    vel: jnp.ndarray,
    boost_amount: jnp.ndarray,
    quat: jnp.ndarray,
    is_on_ground: jnp.ndarray,
    boost_input: jnp.ndarray,
    active_mask: jnp.ndarray,
    is_boosting_prev: jnp.ndarray,
    boosting_time_prev: jnp.ndarray,
    dt: float = DT
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Apply boost acceleration and consume boost.
    
    C++ logic:
    - If currently boosting: keep boosting if (button held OR boostingTime < BOOST_MIN_TIME)
    - If not boosting: start if button pressed and has fuel
    - If boosting: boostingTime += dt, else boostingTime = 0
    
    Args:
        vel: Car velocities. Shape: (N, MAX_CARS, 3)
        boost_amount: Current boost. Shape: (N, MAX_CARS)
        quat: Car quaternions. Shape: (N, MAX_CARS, 4)
        is_on_ground: Ground contact flags. Shape: (N, MAX_CARS)
        boost_input: Boost button held. Shape: (N, MAX_CARS)
        active_mask: Non-demoed cars. Shape: (N, MAX_CARS)
        is_boosting_prev: Was boosting last tick. Shape: (N, MAX_CARS)
        boosting_time_prev: Time spent boosting. Shape: (N, MAX_CARS)
        dt: Time step
        
    Returns:
        Tuple of (new_vel, new_boost_amount, new_is_boosting, new_boosting_time)
    """
    has_fuel = boost_amount > 0.0
    
    # C++ boost state machine
    # If was boosting: keep if (button held OR within min time), stop otherwise
    keep_boosting = is_boosting_prev & has_fuel & (boost_input | (boosting_time_prev < BOOST_MIN_TIME))
    # If was not boosting: start if button pressed and has fuel
    start_boosting = ~is_boosting_prev & boost_input & has_fuel
    
    is_boosting = (keep_boosting | start_boosting) & active_mask
    
    # Update boosting time
    new_boosting_time = jnp.where(
        is_boosting,
        boosting_time_prev + dt,
        0.0
    )
    
    boost_accel = jnp.where(
        is_on_ground,
        BOOST_ACCEL_GROUND,
        BOOST_ACCEL_AIR
    )
    
    forward_dir = quat_rotate_vector(quat, jnp.array([1.0, 0.0, 0.0]))
    boost_vel_delta = forward_dir * (boost_accel * dt)[..., None]
    
    new_vel = vel + jnp.where(
        is_boosting[..., None],
        boost_vel_delta,
        0.0
    )
    
    boost_consumed = BOOST_USED_PER_SECOND * dt
    new_boost_amount = boost_amount - jnp.where(
        is_boosting,
        boost_consumed,
        0.0
    )
    
    new_boost_amount = jnp.clip(new_boost_amount, 0.0, BOOST_MAX)
    
    return new_vel, new_boost_amount, is_boosting, new_boosting_time


def update_supersonic_status(
    vel: jnp.ndarray,
    is_supersonic: jnp.ndarray,
    supersonic_timer: jnp.ndarray,
    dt: float = DT
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Update supersonic status based on current speed.
    
    C++ logic:
    - If supersonic AND timer < MAINTAIN_MAX_TIME: use lower threshold (MAINTAIN_MIN_SPEED)
    - Else: use higher threshold (START_SPEED)
    - If supersonic: timer += dt
    - If not supersonic: timer = 0
    
    Args:
        vel: Car velocities. Shape: (N, MAX_CARS, 3)
        is_supersonic: Current supersonic state. Shape: (N, MAX_CARS)
        supersonic_timer: Time spent supersonic (counts up). Shape: (N, MAX_CARS)
        dt: Time step
        
    Returns:
        Tuple of (new_is_supersonic, new_supersonic_timer)
    """
    speed_sq = jnp.sum(vel * vel, axis=-1)
    
    start_speed_sq = SUPERSONIC_START_SPEED * SUPERSONIC_START_SPEED
    maintain_speed_sq = SUPERSONIC_MAINTAIN_MIN_SPEED * SUPERSONIC_MAINTAIN_MIN_SPEED
    
    # C++: if (isSupersonic && supersonicTime < MAINTAIN_MAX_TIME) -> use maintain threshold
    #      else -> use start threshold
    in_grace = is_supersonic & (supersonic_timer < SUPERSONIC_MAINTAIN_MAX_TIME)
    new_is_supersonic = jnp.where(
        in_grace,
        speed_sq >= maintain_speed_sq,
        speed_sq >= start_speed_sq
    )
    
    # C++: if (isSupersonic) timer += dt; else timer = 0;
    new_supersonic_timer = jnp.where(
        new_is_supersonic,
        supersonic_timer + dt,
        0.0
    )
    
    return new_is_supersonic, new_supersonic_timer
