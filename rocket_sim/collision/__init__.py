"""
Collision detection and resolution subpackage.
Split from the original 1408-line collision.py.
"""

from .arena_sdf import arena_sdf
from .ball_arena import resolve_ball_arena_collision
from .car_arena import resolve_car_arena_collision
from .car_ball import resolve_car_ball_collision
from .car_car import resolve_car_car_collision

__all__ = [
    "arena_sdf",
    "resolve_ball_arena_collision",
    "resolve_car_arena_collision",
    "resolve_car_ball_collision",
    "resolve_car_car_collision",
]
