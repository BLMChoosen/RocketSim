"""
Arena Signed Distance Field (SDF)
=================================
Computes signed distance from a point to the nearest arena surface.
Uses plane equations extracted from Rocket League collision meshes.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

from ..constants import ARENA_EXTENT_X, ARENA_EXTENT_Y, ARENA_HEIGHT


# ---- Mesh data globals (disabled in favor of analytical planes) ----
MESH_DATA_PATH = "mesh_data.npz"
HAS_MESH_DATA = False


def point_triangle_distance(p, v0, v1, v2):
    """Compute distance from point p to triangle (v0, v1, v2)."""
    edge0 = v1 - v0
    edge1 = v2 - v0
    diff = v0 - p
    
    a00 = jnp.dot(edge0, edge0)
    a01 = jnp.dot(edge0, edge1)
    a11 = jnp.dot(edge1, edge1)

    normal = jnp.cross(edge0, edge1)
    normal_len = jnp.linalg.norm(normal)
    unit_normal = normal / (normal_len + 1e-8)
    
    plane_dist = jnp.dot(p - v0, unit_normal)
    proj_p = p - plane_dist * unit_normal
    
    v2_ = proj_p - v0
    d20 = jnp.dot(v2_, edge0)
    d21 = jnp.dot(v2_, edge1)
    denom = a00 * a11 - a01 * a01 + 1e-8
    v = (a11 * d20 - a01 * d21) / denom
    w = (a00 * d21 - a01 * d20) / denom
    u = 1.0 - v - w
    
    is_inside = (u >= 0) & (v >= 0) & (w >= 0)
    closest_face = proj_p
    dist_sq_face = plane_dist * plane_dist
    
    def dist_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = jnp.dot(ap, ab) / (jnp.dot(ab, ab) + 1e-8)
        t = jnp.clip(t, 0.0, 1.0)
        closest = a + t * ab
        d = p - closest
        return jnp.dot(d, d), closest

    d_e0, c_e0 = dist_segment(p, v0, v1)
    d_e1, c_e1 = dist_segment(p, v1, v2)
    d_e2, c_e2 = dist_segment(p, v2, v0)
    
    min_edge_dist_sq = jnp.minimum(d_e0, jnp.minimum(d_e1, d_e2))
    closest_edge = jnp.where(
        d_e0 < d_e1,
        jnp.where(d_e0 < d_e2, c_e0, c_e2),
        jnp.where(d_e1 < d_e2, c_e1, c_e2)
    )
    
    dist_sq = jnp.where(is_inside, dist_sq_face, min_edge_dist_sq)
    closest = jnp.where(is_inside, closest_face, closest_edge)
    return dist_sq, closest


def simple_arena_sdf(pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simple AABB + Goal SDF (no corners)."""
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    
    dist_floor = z
    dist_ceiling = ARENA_HEIGHT - z
    dist_left = x + ARENA_EXTENT_X
    dist_right = ARENA_EXTENT_X - x
    dist_back = y + ARENA_EXTENT_Y
    dist_front = ARENA_EXTENT_Y - y
    
    GOAL_HALF_WIDTH = 892.755
    GOAL_HEIGHT = 642.775
    in_goal_x = jnp.abs(x) < GOAL_HALF_WIDTH
    in_goal_z = z < GOAL_HEIGHT
    in_goal_aperture = in_goal_x & in_goal_z
    
    dist_back = jnp.where(in_goal_aperture, 1e6, dist_back)
    dist_front = jnp.where(in_goal_aperture, 1e6, dist_front)
    
    distances = jnp.stack([dist_floor, dist_ceiling, dist_left, dist_right, dist_back, dist_front], axis=-1)
    
    n_floor = jnp.array([0.0, 0.0, 1.0])
    n_ceil = jnp.array([0.0, 0.0, -1.0])
    n_left = jnp.array([1.0, 0.0, 0.0])
    n_right = jnp.array([-1.0, 0.0, 0.0])
    n_back = jnp.array([0.0, 1.0, 0.0])
    n_front = jnp.array([0.0, -1.0, 0.0])
    
    ones = jnp.ones_like(x)[..., None]
    def b(n): return n * ones
    normals_stack = jnp.stack([b(n_floor), b(n_ceil), b(n_left), b(n_right), b(n_back), b(n_front)], axis=-2)
    
    min_idx = jnp.argmin(distances, axis=-1)
    min_dist = jnp.min(distances, axis=-1)
    min_idx_expanded = min_idx[..., None, None]
    normal = jnp.take_along_axis(normals_stack, min_idx_expanded, axis=-2)
    normal = normal[..., 0, :]
    return min_dist, normal


# =====================================================================
# Plane-based Arena SDF (extracted from RL collision meshes)
# =====================================================================

ARENA_PLANES = jnp.array([
    # Floor (z=0)
    [0.0, 0.0, 1.0, 0.0],
    # Ceiling (z=2044)
    [0.0, 0.0, -1.0, -2044.0],
    # Side Walls (x=+-4096)
    [-1.0, 0.0, 0.0, -4096.0],
    [1.0, 0.0, 0.0, -4096.0],
    # Back Walls (y=+-5120)
    [0.0, -1.0, 0.0, -5120.0],
    [0.0, 1.0, 0.0, -5120.0],
    # Corners (45 deg)
    [-0.7071, -0.7071, 0.0, -5700.0],
    [0.7071, -0.7071, 0.0, -5700.0],
    [-0.7071, 0.7071, 0.0, -5700.0],
    [0.7071, 0.7071, 0.0, -5700.0],
    # Ramps (Floor/Wall connections)
    [0.0, -0.879, 0.477, -4440.0],
    [0.0, 0.879, 0.477, -4440.0],
    [-0.879, 0.0, 0.477, -3505.0],
    [0.879, 0.0, 0.477, -3505.0],
    [0.0, -0.955, 0.297, -4850.0],
    [0.0, 0.955, 0.297, -4850.0],
    [0.0, -0.637, 0.771, -3195.0],
    [0.0, 0.637, 0.771, -3195.0],
    [0.0, -0.771, 0.637, -3885.0],
    [0.0, 0.771, 0.637, -3885.0],
    [0.0, -0.477, 0.879, -2385.0],
    [0.0, 0.477, 0.879, -2385.0],
])

GOAL_PLANES_POS = jnp.array([
    [0.0, -1.0, 0.0, -5995.0],
    [-1.0, 0.0, 0.0, -895.0],
    [1.0, 0.0, 0.0, -895.0],
    [0.0, 0.0, -1.0, -640.0],
    [0.0, 0.0, 1.0, 0.0],
])

GOAL_PLANES_NEG = jnp.array([
    [0.0, 1.0, 0.0, -5995.0],
    [-1.0, 0.0, 0.0, -895.0],
    [1.0, 0.0, 0.0, -895.0],
    [0.0, 0.0, -1.0, -640.0],
    [0.0, 0.0, 1.0, 0.0],
])


def arena_sdf(pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Exact Arena SDF using extracted plane equations.
    Returns POSITIVE distance when INSIDE the arena (safe).
    Returns NEGATIVE distance when PENETRATING a wall.
    """
    # Main Arena SDF
    n_main = ARENA_PLANES[:, :3]
    d_main = ARENA_PLANES[:, 3]
    dot_main = jnp.matmul(pos, n_main.T)
    dist_main_all = dot_main - d_main
    dist_main = jnp.min(dist_main_all, axis=-1)
    idx_main = jnp.argmin(dist_main_all, axis=-1)
    norm_main = n_main[idx_main]
    
    # Goal Pos SDF
    n_gp = GOAL_PLANES_POS[:, :3]
    d_gp = GOAL_PLANES_POS[:, 3]
    dot_gp = jnp.matmul(pos, n_gp.T)
    dist_gp_all = dot_gp - d_gp
    dist_gp = jnp.min(dist_gp_all, axis=-1)
    idx_gp = jnp.argmin(dist_gp_all, axis=-1)
    norm_gp = n_gp[idx_gp]
    
    # Goal Neg SDF
    n_gn = GOAL_PLANES_NEG[:, :3]
    d_gn = GOAL_PLANES_NEG[:, 3]
    dot_gn = jnp.matmul(pos, n_gn.T)
    dist_gn_all = dot_gn - d_gn
    dist_gn = jnp.min(dist_gn_all, axis=-1)
    idx_gn = jnp.argmin(dist_gn_all, axis=-1)
    norm_gn = n_gn[idx_gn]
    
    # Union: max(main, goal_pos, goal_neg)
    max_dist = jnp.maximum(dist_main, jnp.maximum(dist_gp, dist_gn))
    
    use_gp = (dist_gp > dist_main) & (dist_gp > dist_gn)
    use_gn = (dist_gn > dist_main) & (dist_gn > dist_gp)
    
    normal = jnp.where(
        use_gp[..., None], norm_gp,
        jnp.where(use_gn[..., None], norm_gn, norm_main)
    )
    
    return max_dist, normal
