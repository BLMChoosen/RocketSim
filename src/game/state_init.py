"""
State Initialization
====================
Factory functions for creating initial physics state, ball, cars, and controls.
"""

from __future__ import annotations
import jax.numpy as jnp

from ..constants import (
    BALL_REST_Z, BOOST_SPAWN_AMOUNT, CAR_SPAWN_Z, N_PADS_TOTAL,
)
from ..types import BallState, CarState, CarControls, PhysicsState


def create_initial_ball_state(n_envs: int) -> BallState:
    """Create initial ball state for n_envs parallel environments."""
    return BallState(
        pos=jnp.tile(jnp.array([0.0, 0.0, BALL_REST_Z])[None, :], (n_envs, 1)),
        vel=jnp.zeros((n_envs, 3)),
        ang_vel=jnp.zeros((n_envs, 3)),
    )


def create_initial_car_state(n_envs: int, max_cars: int = 6) -> CarState:
    """Create initial car state for n_envs environments with max_cars per env."""
    spawn_positions = jnp.array([
        [-2048.0, -2560.0, CAR_SPAWN_Z],
        [0.0, -4608.0, CAR_SPAWN_Z],
        [2048.0, -2560.0, CAR_SPAWN_Z],
        [-2048.0, 2560.0, CAR_SPAWN_Z],
        [0.0, 4608.0, CAR_SPAWN_Z],
        [2048.0, 2560.0, CAR_SPAWN_Z],
    ])[:max_cars]
    
    if max_cars > 6:
        spawn_positions = jnp.concatenate([
            spawn_positions,
            jnp.zeros((max_cars - 6, 3))
        ], axis=0)
    
    identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    teams = jnp.array([0, 0, 0, 1, 1, 1][:max_cars], dtype=jnp.int32)
    
    return CarState(
        pos=jnp.tile(spawn_positions[None, :, :], (n_envs, 1, 1)),
        vel=jnp.zeros((n_envs, max_cars, 3)),
        ang_vel=jnp.zeros((n_envs, max_cars, 3)),
        quat=jnp.tile(identity_quat[None, None, :], (n_envs, max_cars, 1)),
        boost_amount=jnp.full((n_envs, max_cars), BOOST_SPAWN_AMOUNT),
        is_on_ground=jnp.ones((n_envs, max_cars), dtype=jnp.bool_),
        wheel_contacts=jnp.ones((n_envs, max_cars, 4), dtype=jnp.bool_),
        handbrake_val=jnp.zeros((n_envs, max_cars)),
        is_jumping=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        has_jumped=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        jump_timer=jnp.zeros((n_envs, max_cars)),
        has_flipped=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        has_double_jumped=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        is_flipping=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        flip_timer=jnp.zeros((n_envs, max_cars)),
        flip_rel_torque=jnp.zeros((n_envs, max_cars, 3)),
        air_time=jnp.zeros((n_envs, max_cars)),
        air_time_since_jump=jnp.zeros((n_envs, max_cars)),
        last_jump_pressed=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        is_demoed=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        demo_respawn_timer=jnp.zeros((n_envs, max_cars)),
        is_supersonic=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        supersonic_timer=jnp.zeros((n_envs, max_cars)),
        is_boosting=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        boosting_time=jnp.zeros((n_envs, max_cars)),
        team=jnp.tile(teams[None, :], (n_envs, 1)),
    )


def create_zero_controls(n_envs: int, max_cars: int = 6) -> CarControls:
    """Create zero-initialized control inputs."""
    return CarControls(
        throttle=jnp.zeros((n_envs, max_cars)),
        steer=jnp.zeros((n_envs, max_cars)),
        pitch=jnp.zeros((n_envs, max_cars)),
        yaw=jnp.zeros((n_envs, max_cars)),
        roll=jnp.zeros((n_envs, max_cars)),
        jump=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        boost=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        handbrake=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
    )


def create_initial_state(n_envs: int, max_cars: int = 6) -> PhysicsState:
    """Create complete initial physics state."""
    return PhysicsState(
        ball=create_initial_ball_state(n_envs),
        cars=create_initial_car_state(n_envs, max_cars),
        tick_count=jnp.zeros(n_envs, dtype=jnp.int32),
        pad_is_active=jnp.ones((n_envs, N_PADS_TOTAL), dtype=jnp.bool_),
        pad_timers=jnp.zeros((n_envs, N_PADS_TOTAL)),
        blue_score=jnp.zeros(n_envs, dtype=jnp.bool_),
        orange_score=jnp.zeros(n_envs, dtype=jnp.bool_),
    )
