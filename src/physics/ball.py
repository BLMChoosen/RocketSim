"""
Ball Physics Step
=================
Complete ball physics integration per tick.
"""

from __future__ import annotations
import jax.numpy as jnp

from ..constants import DT, BALL_DRAG, BALL_MAX_SPEED, BALL_MAX_ANG_SPEED
from ..types import BallState
from ..math_utils import clamp_velocity, clamp_angular_velocity
from .integration import apply_gravity, apply_ball_drag, integrate_position
from ..collision.ball_arena import resolve_ball_arena_collision


def step_ball(ball: BallState, dt: float = DT) -> BallState:
    """Advance ball physics by one timestep."""
    vel = apply_gravity(ball.vel, dt)
    vel = apply_ball_drag(vel, BALL_DRAG, dt)
    vel = clamp_velocity(vel, BALL_MAX_SPEED)
    ang_vel = clamp_angular_velocity(ball.ang_vel, BALL_MAX_ANG_SPEED)
    
    pos = integrate_position(ball.pos, vel, dt)
    pos, vel, ang_vel = resolve_ball_arena_collision(pos, vel, ang_vel)
    
    return ball.replace(pos=pos, vel=vel, ang_vel=ang_vel)
