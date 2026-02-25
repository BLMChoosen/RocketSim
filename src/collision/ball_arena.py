"""
Ball-Arena Collision Resolution
===============================
Resolves ball collisions with arena boundaries (walls, floor, ceiling).
"""

from __future__ import annotations
import jax.numpy as jnp

from ..constants import (
    BALL_RADIUS, BALL_WALL_RESTITUTION, BALL_GROUND_RESTITUTION,
    BALL_SURFACE_FRICTION, BALL_MASS, BALL_MAX_ANG_SPEED,
    GRAVITY_Z, DT,
)
from .arena_sdf import arena_sdf


def resolve_ball_arena_collision(
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    ang_vel: jnp.ndarray,
    radius: float = BALL_RADIUS,
    restitution: float = BALL_WALL_RESTITUTION,
    friction: float = BALL_SURFACE_FRICTION
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Resolve ball collisions with arena boundaries, including rounded corners.
    Applies friction torque to make the ball roll properly.
    """
    dist, normal = arena_sdf(pos)
    
    normal_z = normal[..., 2]
    is_ground = normal_z > 0.9
    effective_restitution = jnp.where(is_ground, BALL_GROUND_RESTITUTION, BALL_WALL_RESTITUTION)
    
    penetration = radius - dist
    is_penetrating = penetration > 0
    contact_tolerance = 2.0
    is_in_contact = penetration > -contact_tolerance
    
    new_pos = jnp.where(
        is_penetrating[..., None],
        pos + normal * penetration[..., None],
        pos
    )
    
    contact_offset = -normal * radius
    surface_vel = vel + jnp.cross(ang_vel, contact_offset)
    
    v_dot_n = jnp.sum(surface_vel * normal, axis=-1, keepdims=True)
    v_normal = v_dot_n * normal
    v_tangent = surface_vel - v_normal
    
    should_bounce = is_penetrating & (v_dot_n[..., 0] < 0)
    
    ball_inertia = (2.0 / 5.0) * BALL_MASS * (radius ** 2)
    v_t_mag = jnp.linalg.norm(v_tangent, axis=-1, keepdims=True)
    v_t_dir = v_tangent / (v_t_mag + 1e-8)
    
    new_v_normal = jnp.where(
        should_bounce[..., None],
        -v_normal * effective_restitution[..., None],
        v_normal
    )
    
    v_n_mag = jnp.abs(v_dot_n)
    bounce_impulse = v_n_mag * (1 + effective_restitution[..., None]) * BALL_MASS
    gravity_normal_force = BALL_MASS * jnp.abs(GRAVITY_Z) * jnp.abs(normal[..., 2:3])
    
    effective_normal_impulse = jnp.where(
        should_bounce[..., None],
        bounce_impulse,
        gravity_normal_force * DT
    )
    
    max_friction_impulse = friction * effective_normal_impulse
    slip_stop_impulse = v_t_mag * BALL_MASS * (2.0 / 7.0)
    friction_impulse_mag = jnp.minimum(max_friction_impulse, slip_stop_impulse)
    friction_impulse = -v_t_dir * friction_impulse_mag
    
    friction_vel_change = friction_impulse / BALL_MASS
    friction_torque = jnp.cross(contact_offset, friction_impulse)
    ang_vel_change = friction_torque / ball_inertia
    
    new_vel = jnp.where(
        is_in_contact[..., None],
        new_v_normal + (vel - v_normal) + friction_vel_change,
        vel
    )
    
    new_ang_vel = jnp.where(
        is_in_contact[..., None],
        ang_vel + ang_vel_change,
        ang_vel
    )
    
    ang_speed = jnp.linalg.norm(new_ang_vel, axis=-1, keepdims=True)
    new_ang_vel = jnp.where(
        ang_speed > BALL_MAX_ANG_SPEED,
        new_ang_vel * (BALL_MAX_ANG_SPEED / (ang_speed + 1e-8)),
        new_ang_vel
    )
    
    return new_pos, new_vel, new_ang_vel
