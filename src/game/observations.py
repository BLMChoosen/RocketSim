"""
Observations
============
Extract normalized observation vectors for neural network input.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

from ..constants import (
    ARENA_EXTENT_X, CAR_MAX_SPEED, CAR_MAX_ANG_SPEED,
    BOOST_MAX, DOUBLEJUMP_MAX_DELAY,
)
from ..types import PhysicsState
from ..math_utils import get_forward_up_right


# Normalization constants
NORM_POS = ARENA_EXTENT_X
NORM_VEL = CAR_MAX_SPEED
NORM_ANG_VEL = CAR_MAX_ANG_SPEED
NORM_BOOST = BOOST_MAX


@jax.jit
def get_observations(
    state: PhysicsState,
    observer_car_idx: int = 0,
) -> jnp.ndarray:
    """
    Extract normalized observation vectors for neural network input.
    
    Args:
        state: Current physics state
        observer_car_idx: Which car's perspective to use (default: 0)
        
    Returns:
        Observation array. Shape: (N_ENVS, OBS_SIZE)
    """
    n_envs = state.ball.pos.shape[0]
    max_cars = state.cars.pos.shape[1]
    
    # Ball observation
    ball_pos_norm = state.ball.pos / NORM_POS
    ball_vel_norm = state.ball.vel / NORM_VEL
    ball_ang_vel_norm = state.ball.ang_vel / NORM_ANG_VEL
    
    ball_obs = jnp.concatenate([
        ball_pos_norm,
        ball_vel_norm,
        ball_ang_vel_norm,
    ], axis=-1)
    
    # Ball relative to observer
    observer_pos = state.cars.pos[:, observer_car_idx, :]
    observer_vel = state.cars.vel[:, observer_car_idx, :]
    
    ball_rel_pos_norm = (state.ball.pos - observer_pos) / NORM_POS
    ball_rel_vel_norm = (state.ball.vel - observer_vel) / NORM_VEL
    
    ball_relative_obs = jnp.concatenate([
        ball_rel_pos_norm,
        ball_rel_vel_norm,
    ], axis=-1)
    
    # All car observations
    all_pos_norm = state.cars.pos / NORM_POS
    all_vel_norm = state.cars.vel / NORM_VEL
    all_ang_vel_norm = state.cars.ang_vel / NORM_ANG_VEL
    
    quat_flat = state.cars.quat.reshape(-1, 4)
    forward_flat, up_flat, right_flat = get_forward_up_right(quat_flat)
    all_forward = forward_flat.reshape(n_envs, max_cars, 3)
    all_up = up_flat.reshape(n_envs, max_cars, 3)
    all_right = right_flat.reshape(n_envs, max_cars, 3)
    
    all_boost_norm = (state.cars.boost_amount / NORM_BOOST)[..., None]
    all_on_ground = state.cars.is_on_ground[..., None].astype(jnp.float32)
    
    all_has_flip = (
        ~state.cars.has_flipped & 
        ~state.cars.has_double_jumped &
        (state.cars.air_time_since_jump < DOUBLEJUMP_MAX_DELAY)
    )
    all_has_flip = all_has_flip[..., None].astype(jnp.float32)
    
    all_car_features = jnp.concatenate([
        all_pos_norm,
        all_vel_norm,
        all_ang_vel_norm,
        all_forward,
        all_up,
        all_right,
        all_boost_norm,
        all_on_ground,
        all_has_flip,
    ], axis=-1)
    
    # Reorder so observer is first
    before_observer = jnp.arange(observer_car_idx)
    after_observer = jnp.arange(observer_car_idx + 1, max_cars)
    reorder_indices = jnp.concatenate([
        jnp.array([observer_car_idx]),
        before_observer,
        after_observer
    ])
    
    reordered_features = all_car_features[:, reorder_indices, :]
    all_car_obs = reordered_features.reshape(n_envs, -1)
    
    observations = jnp.concatenate([
        ball_obs,
        ball_relative_obs,
        all_car_obs,
    ], axis=-1)
    
    return observations
