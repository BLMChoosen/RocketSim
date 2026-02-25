"""
Car-Car Collision Resolution
=============================
Handles bumps and demolitions between cars.
"""

from __future__ import annotations
import jax.numpy as jnp

from ..constants import (
    OCTANE_HITBOX_SIZE, OCTANE_HITBOX_OFFSET,
    DEMO_MIN_SPEED, DEMO_FORWARD_ANGLE_COS,
    BUMP_VEL_AMOUNT_GROUND_SPEEDS, BUMP_VEL_AMOUNT_GROUND_VALUES,
    BUMP_VEL_AMOUNT_AIR_SPEEDS, BUMP_VEL_AMOUNT_AIR_VALUES,
    BUMP_UPWARD_VEL_AMOUNT_SPEEDS, BUMP_UPWARD_VEL_AMOUNT_VALUES,
)
from ..math_utils import quat_rotate_vector


def resolve_car_car_collision(
    car_pos: jnp.ndarray,
    car_vel: jnp.ndarray,
    car_ang_vel: jnp.ndarray,
    car_quat: jnp.ndarray,
    car_is_on_ground: jnp.ndarray,
    car_is_supersonic: jnp.ndarray,
    hitbox_half_size: jnp.ndarray = OCTANE_HITBOX_SIZE / 2,
    hitbox_offset: jnp.ndarray = OCTANE_HITBOX_OFFSET,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Resolve collisions between cars (demos and bumps).
    Returns (new_car_vel, new_car_ang_vel, is_demoed_mask).
    """
    n_envs = car_pos.shape[0]
    max_cars = car_pos.shape[1]
    
    bounding_radius = jnp.sqrt(jnp.sum(hitbox_half_size ** 2)) + 10.0
    
    hitbox_center_offset = quat_rotate_vector(car_quat, hitbox_offset)
    hitbox_center = car_pos + hitbox_center_offset
    
    center_i = hitbox_center[:, :, None, :]
    center_j = hitbox_center[:, None, :, :]
    diff = center_i - center_j
    dist_sq = jnp.sum(diff ** 2, axis=-1)
    dist = jnp.sqrt(dist_sq + 1e-8)
    
    collision_dist = 2 * bounding_radius
    potentially_colliding = dist < collision_dist
    
    identity_mask = jnp.eye(max_cars, dtype=jnp.bool_)[None, :, :]
    potentially_colliding = potentially_colliding & ~identity_mask
    
    upper_tri_mask = jnp.triu(jnp.ones((max_cars, max_cars), dtype=jnp.bool_), k=1)[None, :, :]
    is_valid_pair = potentially_colliding & upper_tri_mask
    
    collision_normal = diff / (dist[..., None] + 1e-8)
    proj_half_size = jnp.sum(jnp.abs(collision_normal) * hitbox_half_size, axis=-1)
    penetration = 2 * proj_half_size - dist
    is_colliding = (penetration > 0) & is_valid_pair
    
    forward_local = jnp.array([1.0, 0.0, 0.0])
    car_forward = quat_rotate_vector(car_quat, forward_local)
    forward_speed = jnp.sum(car_vel * car_forward, axis=-1)
    forward_speed_i = forward_speed[:, :, None]
    forward_speed_j = forward_speed[:, None, :]
    
    i_is_bumper = forward_speed_i > forward_speed_j
    
    # Demo detection
    is_supersonic_i = car_is_supersonic[:, :, None]
    is_supersonic_j = car_is_supersonic[:, None, :]
    bumper_is_supersonic = jnp.where(i_is_bumper, is_supersonic_i, is_supersonic_j)
    
    speed_i = jnp.linalg.norm(car_vel, axis=-1)[:, :, None]
    speed_j = jnp.linalg.norm(car_vel, axis=-1)[:, None, :]
    bumper_total_speed = jnp.where(i_is_bumper, speed_i, speed_j)
    bumper_fast_enough = bumper_total_speed >= DEMO_MIN_SPEED
    
    hit_dir = jnp.where(i_is_bumper[..., None], -collision_normal, collision_normal)
    car_forward_i = car_forward[:, :, None, :]
    car_forward_j = car_forward[:, None, :, :]
    bumper_forward = jnp.where(i_is_bumper[..., None], car_forward_i, car_forward_j)
    
    impact_angle = jnp.sum(bumper_forward * hit_dir, axis=-1)
    is_front_hit = impact_angle > DEMO_FORWARD_ANGLE_COS
    
    is_demo = is_colliding & bumper_is_supersonic & bumper_fast_enough & is_front_hit
    
    i_is_victim = ~i_is_bumper
    is_demoed_i_by_j = is_demo & i_is_victim
    j_is_victim = i_is_bumper
    is_demoed_j_by_i = is_demo & j_is_victim
    
    demoed_by_higher_index = jnp.any(is_demoed_i_by_j, axis=2)
    demoed_by_lower_index = jnp.any(is_demoed_j_by_i, axis=1)
    is_demoed_mask = demoed_by_higher_index | demoed_by_lower_index
    
    # Bump impulse
    is_on_ground_i = car_is_on_ground[:, :, None]
    is_on_ground_j = car_is_on_ground[:, None, :]
    
    bumper_speed = jnp.where(i_is_bumper, forward_speed_i, forward_speed_j)
    bumper_speed = jnp.abs(bumper_speed)
    target_grounded = jnp.where(i_is_bumper, is_on_ground_j, is_on_ground_i)
    
    bump_vel_ground = jnp.interp(bumper_speed, BUMP_VEL_AMOUNT_GROUND_SPEEDS, BUMP_VEL_AMOUNT_GROUND_VALUES)
    bump_vel_air = jnp.interp(bumper_speed, BUMP_VEL_AMOUNT_AIR_SPEEDS, BUMP_VEL_AMOUNT_AIR_VALUES)
    bump_upward = jnp.interp(bumper_speed, BUMP_UPWARD_VEL_AMOUNT_SPEEDS, BUMP_UPWARD_VEL_AMOUNT_VALUES)
    
    bump_vel_magnitude = jnp.where(target_grounded, bump_vel_ground, bump_vel_air)
    
    bump_dir = jnp.where(i_is_bumper[..., None], collision_normal, -collision_normal)
    bump_dir_xy = bump_dir.at[..., 2].set(0.0)
    bump_dir_xy = bump_dir_xy / (jnp.linalg.norm(bump_dir_xy, axis=-1, keepdims=True) + 1e-8)
    
    bump_impulse = bump_dir_xy * bump_vel_magnitude[..., None] + jnp.array([0.0, 0.0, 1.0]) * bump_upward[..., None]
    bump_impulse = jnp.where(is_colliding[..., None] & ~is_demo[..., None], bump_impulse, 0.0)
    
    recoil_factor = 0.3
    impulse_on_i_from_pair = jnp.where(i_is_bumper[..., None], -bump_impulse * recoil_factor, bump_impulse)
    
    total_impulse_i = jnp.sum(impulse_on_i_from_pair, axis=2)
    impulse_on_j_from_pair = jnp.where(i_is_bumper[..., None], bump_impulse, -bump_impulse * recoil_factor)
    total_impulse_j = jnp.sum(impulse_on_j_from_pair, axis=1)
    
    total_impulse = total_impulse_i + total_impulse_j
    new_vel = car_vel + total_impulse
    
    separation_strength = 0.5
    separation = collision_normal * penetration[..., None] * separation_strength
    separation = jnp.where(is_colliding[..., None], separation, 0.0)
    
    push_i = jnp.sum(separation, axis=2)
    push_j = jnp.sum(-separation, axis=1)
    new_vel = new_vel + push_i + push_j
    
    new_ang_vel = car_ang_vel
    
    return new_vel, new_ang_vel, is_demoed_mask
