"""
Car-Arena Collision Resolution
==============================
Resolves car hitbox collisions with arena geometry using OBB corner checks.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

from ..constants import OCTANE_HITBOX_SIZE, OCTANE_HITBOX_OFFSET, CAR_INERTIA
from ..math_utils import quat_rotate_vector
from .arena_sdf import arena_sdf


def resolve_car_arena_collision(
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    ang_vel: jnp.ndarray,
    quat: jnp.ndarray,
    hitbox_half_size: jnp.ndarray = OCTANE_HITBOX_SIZE / 2,
    hitbox_offset: jnp.ndarray = OCTANE_HITBOX_OFFSET,
    restitution: float = 0.3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Resolve car-arena collisions by checking hitbox corners against arena SDF.
    """
    signs = jnp.array([
        [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
        [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
    ], dtype=jnp.float32)
    
    corners_local = signs * hitbox_half_size + hitbox_offset
    
    def rotate_corner(corner):
        return quat_rotate_vector(quat, corner)
    
    corners_world_offset = jax.vmap(rotate_corner)(corners_local)
    corners_world_offset = jnp.moveaxis(corners_world_offset, 0, -2)
    corners_world = corners_world_offset + pos[..., None, :]
    
    orig_shape = corners_world.shape
    corners_flat = corners_world.reshape(-1, 3)
    corners_for_sdf = corners_flat[None, :, :]
    
    dist_flat, normal_flat = arena_sdf(corners_for_sdf)
    dist_flat = dist_flat[0]
    normal_flat = normal_flat[0]
    
    n_envs = orig_shape[0]
    max_cars = orig_shape[1]
    corner_dist = dist_flat.reshape(n_envs, max_cars, 8)
    corner_normal = normal_flat.reshape(n_envs, max_cars, 8, 3)
    
    penetration = -corner_dist
    max_penetration = jnp.max(penetration, axis=-1)
    deepest_idx = jnp.argmax(penetration, axis=-1)
    
    batch_idx = jnp.arange(n_envs)[:, None]
    car_idx = jnp.arange(max_cars)[None, :]
    deepest_normal = corner_normal[batch_idx, car_idx, deepest_idx]
    deepest_corner_offset = corners_world_offset[batch_idx, car_idx, deepest_idx]
    
    any_penetrating = max_penetration > 0
    
    pos_correction = deepest_normal * max_penetration[..., None]
    new_pos = jnp.where(any_penetrating[..., None], pos + pos_correction, pos)
    
    r = deepest_corner_offset
    vel_at_contact = vel + jnp.cross(ang_vel, r)
    v_rel_n = jnp.sum(vel_at_contact * deepest_normal, axis=-1)
    
    should_respond = any_penetrating & (v_rel_n < 0)
    
    j = -(1.0 + restitution) * v_rel_n
    vel_impulse = j[..., None] * deepest_normal
    new_vel = jnp.where(should_respond[..., None], vel + vel_impulse, vel)
    
    inertia_avg = jnp.mean(jnp.array(CAR_INERTIA))
    torque_arm = jnp.cross(r, deepest_normal)
    ang_impulse = j[..., None] * torque_arm / inertia_avg
    new_ang_vel = jnp.where(should_respond[..., None], ang_vel + ang_impulse, ang_vel)
    
    return new_pos, new_vel, new_ang_vel
