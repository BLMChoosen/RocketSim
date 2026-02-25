"""
Basic Physics Integration
=========================
Gravity, drag, position and rotation integration.
"""

from __future__ import annotations
import jax.numpy as jnp

from ..constants import DT, GRAVITY_Z, BALL_DRAG
from ..math_utils import quat_from_angular_velocity, quat_multiply, quat_normalize


def apply_gravity(vel: jnp.ndarray, dt: float = DT) -> jnp.ndarray:
    """Apply gravitational acceleration to velocity."""
    gravity = jnp.array([0.0, 0.0, GRAVITY_Z])
    return vel + gravity * dt


def apply_ball_drag(vel: jnp.ndarray, drag: float = BALL_DRAG, dt: float = DT) -> jnp.ndarray:
    """Apply air drag to ball velocity (per-tick Bullet style)."""
    damping_factor = jnp.clip(1.0 - drag, 0.0, 1.0)
    return vel * damping_factor


def integrate_position(pos: jnp.ndarray, vel: jnp.ndarray, dt: float = DT) -> jnp.ndarray:
    """Semi-implicit Euler integration for position."""
    return pos + vel * dt


def integrate_rotation(quat: jnp.ndarray, ang_vel: jnp.ndarray, dt: float = DT) -> jnp.ndarray:
    """Integrate rotation quaternion given angular velocity."""
    delta_q = quat_from_angular_velocity(ang_vel, dt)
    new_quat = quat_multiply(quat, delta_q)
    return quat_normalize(new_quat)
