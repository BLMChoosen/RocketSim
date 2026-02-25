"""
Boost Mechanics
===============
Boost acceleration, fuel consumption, and supersonic status tracking.
"""

from __future__ import annotations
import jax.numpy as jnp

from ..constants import (
    DT,
    BOOST_ACCEL_GROUND, BOOST_ACCEL_AIR, BOOST_USED_PER_SECOND, BOOST_MAX, BOOST_MIN_TIME,
    SUPERSONIC_START_SPEED, SUPERSONIC_MAINTAIN_MIN_SPEED,
    SUPERSONIC_MAINTAIN_MAX_TIME,
)
from ..math_utils import quat_rotate_vector


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
    
    keep_boosting = is_boosting_prev & has_fuel & (boost_input | (boosting_time_prev < BOOST_MIN_TIME))
    start_boosting = ~is_boosting_prev & boost_input & has_fuel
    
    is_boosting = (keep_boosting | start_boosting) & active_mask
    
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
    
    Args:
        vel: Car velocities. Shape: (N, MAX_CARS, 3)
        is_supersonic: Current supersonic state. Shape: (N, MAX_CARS)
        supersonic_timer: Time spent supersonic. Shape: (N, MAX_CARS)
        dt: Time step
        
    Returns:
        Tuple of (new_is_supersonic, new_supersonic_timer)
    """
    speed_sq = jnp.sum(vel * vel, axis=-1)
    
    start_speed_sq = SUPERSONIC_START_SPEED * SUPERSONIC_START_SPEED
    maintain_speed_sq = SUPERSONIC_MAINTAIN_MIN_SPEED * SUPERSONIC_MAINTAIN_MIN_SPEED
    
    in_grace = is_supersonic & (supersonic_timer < SUPERSONIC_MAINTAIN_MAX_TIME)
    new_is_supersonic = jnp.where(
        in_grace,
        speed_sq >= maintain_speed_sq,
        speed_sq >= start_speed_sq
    )
    
    new_supersonic_timer = jnp.where(
        new_is_supersonic,
        supersonic_timer + dt,
        0.0
    )
    
    return new_is_supersonic, new_supersonic_timer
