"""
Goal Detection
==============
Check if the ball has scored in either goal.
"""

from __future__ import annotations
import jax.numpy as jnp

from ..constants import GOAL_THRESHOLD_Y, BALL_RADIUS


def check_goal(ball_pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Check if the ball has scored in either goal.
    
    Args:
        ball_pos: Ball positions. Shape: (N, 3)
        
    Returns:
        Tuple of (blue_scored, orange_scored) boolean arrays. Shape: (N,)
    """
    ball_y = ball_pos[:, 1]
    goal_threshold_with_ball = GOAL_THRESHOLD_Y + BALL_RADIUS
    
    blue_scored = ball_y > goal_threshold_with_ball
    orange_scored = ball_y < -goal_threshold_with_ball
    
    return blue_scored, orange_scored
