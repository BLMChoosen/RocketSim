"""
Mechanics Subpackage
====================
Game mechanics: jumping, flipping/dodging, double jumps, and boost.
"""

from .jump import handle_jump
from .flip import handle_flip_or_double_jump, apply_flip_z_damping
from .boost import apply_boost, update_supersonic_status

__all__ = [
    "handle_jump",
    "handle_flip_or_double_jump",
    "apply_flip_z_damping",
    "apply_boost",
    "update_supersonic_status",
]
