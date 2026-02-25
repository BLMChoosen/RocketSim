"""
Boost Pad Logic
===============
Boost pad pickups and cooldown timer management.
"""

from __future__ import annotations
import jax.numpy as jnp

from ..constants import (
    DT, PAD_LOCATIONS, PAD_RADII, PAD_CYL_HEIGHT,
    PAD_BOOST_AMOUNTS, PAD_COOLDOWNS, BOOST_MAX,
)


def resolve_boost_pads(
    car_pos: jnp.ndarray,
    car_boost: jnp.ndarray,
    pad_is_active: jnp.ndarray,
    pad_timers: jnp.ndarray,
    dt: float = DT
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Handle boost pad pickups and cooldown timers.
    
    Args:
        car_pos: Car positions. Shape: (N, MAX_CARS, 3)
        car_boost: Car boost amounts. Shape: (N, MAX_CARS)
        pad_is_active: Pad active flags. Shape: (N, N_PADS_TOTAL)
        pad_timers: Pad cooldown timers. Shape: (N, N_PADS_TOTAL)
        dt: Time step
        
    Returns:
        Tuple of (new_car_boost, new_pad_is_active, new_pad_timers)
    """
    # Update pad cooldown timers
    new_pad_timers = jnp.maximum(pad_timers - dt, 0.0)
    new_pad_is_active = new_pad_timers <= 0.0
    
    # Check car-pad collisions
    car_pos_exp = car_pos[:, :, None, :]
    pad_locs_exp = PAD_LOCATIONS[None, None, :, :]
    
    diff = car_pos_exp - pad_locs_exp
    dist_xy_sq = diff[..., 0]**2 + diff[..., 1]**2
    dist_z = jnp.abs(diff[..., 2])
    
    pad_radii_sq = (PAD_RADII ** 2)[None, None, :]
    
    in_xy_range = dist_xy_sq < pad_radii_sq
    in_z_range = dist_z < PAD_CYL_HEIGHT
    touching = in_xy_range & in_z_range
    
    # Determine pickups
    pad_active_exp = new_pad_is_active[:, None, :]
    can_pickup = touching & pad_active_exp
    
    # Award boost
    boost_amounts_exp = PAD_BOOST_AMOUNTS[None, None, :]
    boost_gained = jnp.sum(
        jnp.where(can_pickup, boost_amounts_exp, 0.0),
        axis=-1
    )
    new_car_boost = jnp.minimum(car_boost + boost_gained, BOOST_MAX)
    
    # Deactivate picked pads
    pad_was_picked = jnp.any(can_pickup, axis=1)
    new_pad_is_active = jnp.where(pad_was_picked, False, new_pad_is_active)
    
    # Set cooldown
    cooldowns_exp = PAD_COOLDOWNS[None, :]
    new_pad_timers = jnp.where(pad_was_picked, cooldowns_exp, new_pad_timers)
    
    return new_car_boost, new_pad_is_active, new_pad_timers
