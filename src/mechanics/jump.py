"""
Jump Mechanics
==============
First jump: ground -> air impulse and continuous upward force.
"""

from __future__ import annotations
import jax.numpy as jnp

from ..constants import (
    DT,
    JUMP_IMMEDIATE_FORCE, JUMP_ACCEL, JUMP_MIN_TIME, JUMP_MAX_TIME,
    JUMP_RESET_TIME_PAD,
)
from ..types import CarState, CarControls
from ..math_utils import get_car_up_dir


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
