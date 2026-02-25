"""
Car-Ball Collision Resolution
=============================
OBB-Sphere collision between car hitbox and ball,
including RL's "extra impulse" power hit mechanic.
"""

from __future__ import annotations
import jax.numpy as jnp

from ..constants import (
    BALL_RADIUS, BALL_MASS, BALL_MAX_ANG_SPEED, CAR_MASS, CAR_INERTIA,
    OCTANE_HITBOX_SIZE, OCTANE_HITBOX_OFFSET,
    CARBALL_COLLISION_FRICTION,
    BALL_CAR_EXTRA_IMPULSE_Z_SCALE, BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE,
    BALL_CAR_EXTRA_IMPULSE_MAX_DELTA_VEL,
    BALL_CAR_EXTRA_IMPULSE_FACTOR_SPEEDS, BALL_CAR_EXTRA_IMPULSE_FACTOR_VALUES,
)
from ..math_utils import quat_rotate_vector


def resolve_car_ball_collision(
    ball_pos: jnp.ndarray,
    ball_vel: jnp.ndarray,
    ball_ang_vel: jnp.ndarray,
    car_pos: jnp.ndarray,
    car_vel: jnp.ndarray,
    car_ang_vel: jnp.ndarray,
    car_quat: jnp.ndarray,
    ball_radius: float = BALL_RADIUS,
    hitbox_half_size: jnp.ndarray = OCTANE_HITBOX_SIZE / 2,
    hitbox_offset: jnp.ndarray = OCTANE_HITBOX_OFFSET,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Resolve collisions between ball and all cars.
    Returns (new_ball_vel, new_ball_ang_vel, new_car_vel, new_car_ang_vel, hit_mask).
    """
    n_envs = ball_pos.shape[0]
    max_cars = car_pos.shape[1]
    
    ball_pos_exp = ball_pos[:, None, :]
    ball_vel_exp = ball_vel[:, None, :]
    
    rel_pos_world = ball_pos_exp - car_pos
    car_quat_inv = car_quat * jnp.array([1.0, -1.0, -1.0, -1.0])
    
    local_ball_pos = quat_rotate_vector(car_quat_inv, rel_pos_world)
    local_ball_pos_hitbox = local_ball_pos - hitbox_offset
    
    closest_local = jnp.clip(local_ball_pos_hitbox, -hitbox_half_size, hitbox_half_size)
    dist_vec_local = local_ball_pos_hitbox - closest_local
    dist_sq = jnp.sum(dist_vec_local ** 2, axis=-1)
    
    is_colliding = dist_sq < (ball_radius ** 2)
    dist = jnp.sqrt(dist_sq + 1e-8)
    
    local_normal = dist_vec_local / (dist[..., None] + 1e-8)
    world_normal = quat_rotate_vector(car_quat, local_normal)
    
    car_contact_local = closest_local + hitbox_offset
    car_contact_world = quat_rotate_vector(car_quat, car_contact_local) + car_pos
    car_contact_offset = car_contact_world - car_pos
    
    car_vel_at_contact = car_vel + jnp.cross(car_ang_vel, car_contact_offset)
    rel_vel = ball_vel_exp - car_vel_at_contact
    rel_vel_normal = jnp.sum(rel_vel * world_normal, axis=-1)
    
    approaching = rel_vel_normal < 0
    inv_mass_sum = 1.0 / BALL_MASS + 1.0 / CAR_MASS
    restitution = 0.0
    impulse_mag = -(1 + restitution) * rel_vel_normal / inv_mass_sum
    
    impulse_mask = is_colliding & approaching
    impulse_mag = jnp.where(impulse_mask, impulse_mag, 0.0)
    impulse = impulse_mag[..., None] * world_normal
    
    # RL's "Extra Impulse" (power hit mechanic)
    rel_speed = jnp.linalg.norm(rel_vel, axis=-1)
    rel_speed_clamped = jnp.minimum(rel_speed, BALL_CAR_EXTRA_IMPULSE_MAX_DELTA_VEL)
    
    car_forward = quat_rotate_vector(car_quat, jnp.array([1.0, 0.0, 0.0]))
    
    hit_dir_raw = rel_pos_world * jnp.array([1.0, 1.0, BALL_CAR_EXTRA_IMPULSE_Z_SCALE])
    hit_dir = hit_dir_raw / (jnp.linalg.norm(hit_dir_raw, axis=-1, keepdims=True) + 1e-8)
    
    forward_component = jnp.sum(hit_dir * car_forward, axis=-1, keepdims=True)
    forward_adjustment = car_forward * forward_component * (1 - BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE)
    hit_dir = hit_dir - forward_adjustment
    hit_dir = hit_dir / (jnp.linalg.norm(hit_dir, axis=-1, keepdims=True) + 1e-8)
    
    extra_factor = jnp.interp(
        rel_speed_clamped,
        BALL_CAR_EXTRA_IMPULSE_FACTOR_SPEEDS,
        BALL_CAR_EXTRA_IMPULSE_FACTOR_VALUES
    )
    
    extra_vel = hit_dir * rel_speed_clamped[..., None] * extra_factor[..., None]
    extra_vel = jnp.where(impulse_mask[..., None], extra_vel, 0.0)
    
    ball_vel_delta_physics = jnp.sum(impulse / BALL_MASS, axis=1)
    ball_vel_delta_extra = jnp.sum(extra_vel, axis=1)
    new_ball_vel = ball_vel + ball_vel_delta_physics + ball_vel_delta_extra
    
    car_vel_delta = -impulse / CAR_MASS
    new_car_vel = car_vel + car_vel_delta
    
    car_impulse = -impulse
    torque_on_car = jnp.cross(car_contact_offset, car_impulse)
    inertia_avg = jnp.mean(CAR_INERTIA)
    ang_vel_delta = torque_on_car / inertia_avg
    new_car_ang_vel = car_ang_vel + ang_vel_delta
    
    # Ball spin from friction
    ball_contact_offset = -world_normal * ball_radius
    ball_ang_vel_exp = ball_ang_vel[:, None, :]
    ball_surface_vel = ball_vel_exp + jnp.cross(ball_ang_vel_exp, ball_contact_offset)
    
    rel_surface_vel = ball_surface_vel - car_vel_at_contact
    rel_surface_vel_normal = jnp.sum(rel_surface_vel * world_normal, axis=-1, keepdims=True)
    tangential_slip_vel = rel_surface_vel - rel_surface_vel_normal * world_normal
    
    tangential_speed = jnp.linalg.norm(tangential_slip_vel, axis=-1, keepdims=True)
    tangential_dir = tangential_slip_vel / (tangential_speed + 1e-8)
    
    friction_impulse_mag = jnp.abs(impulse_mag) * CARBALL_COLLISION_FRICTION * 0.1
    friction_impulse_mag = jnp.minimum(friction_impulse_mag, tangential_speed[..., 0] * BALL_MASS)
    friction_force = -tangential_dir * friction_impulse_mag[..., None]
    friction_force = jnp.where(impulse_mask[..., None], friction_force, 0.0)
    
    ball_friction_torque = jnp.cross(ball_contact_offset, friction_force)
    total_ball_torque = jnp.sum(ball_friction_torque, axis=1)
    
    ball_inertia = (2.0 / 5.0) * BALL_MASS * (ball_radius ** 2)
    ball_ang_accel = total_ball_torque / ball_inertia
    new_ball_ang_vel = ball_ang_vel + ball_ang_accel
    
    ball_ang_speed = jnp.linalg.norm(new_ball_ang_vel, axis=-1, keepdims=True)
    new_ball_ang_vel = jnp.where(
        ball_ang_speed > BALL_MAX_ANG_SPEED,
        new_ball_ang_vel * (BALL_MAX_ANG_SPEED / (ball_ang_speed + 1e-8)),
        new_ball_ang_vel
    )
    
    return new_ball_vel, new_ball_ang_vel, new_car_vel, new_car_ang_vel, is_colliding
