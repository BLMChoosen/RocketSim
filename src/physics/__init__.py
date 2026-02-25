"""
Physics subpackage - Vehicle physics, suspension, tires, and integration.
Split from the original 934-line physics.py.
"""

from .integration import apply_gravity, apply_ball_drag, integrate_position, integrate_rotation
from .suspension import solve_suspension_and_tires
from .ball import step_ball

__all__ = [
    "apply_gravity",
    "apply_ball_drag",
    "integrate_position",
    "integrate_rotation",
    "solve_suspension_and_tires",
    "step_ball",
]
