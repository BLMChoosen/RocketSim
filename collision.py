"""
Arena SDFs and Collision Resolution
====================================
Signed Distance Field for arena geometry and collision resolution functions.
Uses branchless jnp.where() for GPU efficiency.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import os

from sim_constants import (
    ARENA_EXTENT_X, ARENA_EXTENT_Y, ARENA_HEIGHT,
    BALL_RADIUS, BALL_WALL_RESTITUTION, BALL_SURFACE_FRICTION, BALL_MASS,
    BALL_MAX_ANG_SPEED, BALL_CAR_EXTRA_IMPULSE_Z_SCALE,
    BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE, BALL_CAR_EXTRA_IMPULSE_MAX_DELTA_VEL,
    BALL_CAR_EXTRA_IMPULSE_FACTOR_SPEEDS, BALL_CAR_EXTRA_IMPULSE_FACTOR_VALUES,
    CAR_MASS, OCTANE_HITBOX_SIZE, OCTANE_HITBOX_OFFSET,
    CARBALL_COLLISION_FRICTION,
    BUMP_VEL_AMOUNT_GROUND_SPEEDS, BUMP_VEL_AMOUNT_GROUND_VALUES,
    BUMP_VEL_AMOUNT_AIR_SPEEDS, BUMP_VEL_AMOUNT_AIR_VALUES,
    BUMP_UPWARD_VEL_AMOUNT_SPEEDS, BUMP_UPWARD_VEL_AMOUNT_VALUES,
    GRAVITY_Z, DT,
)
from math_utils import quat_rotate_vector

# Load Mesh Data if available
MESH_DATA_PATH = "mesh_data.npz"
HAS_MESH_DATA = False
# Mesh loading disabled in favor of analytical planes
# if os.path.exists(MESH_DATA_PATH): ...


def point_triangle_distance(p, v0, v1, v2):
    """
    Compute distance from point p to triangle (v0, v1, v2).
    Returns (dist_sq, closest_point).
    """
    # Based on standard algorithm (e.g. Eberly)
    edge0 = v1 - v0
    edge1 = v2 - v0
    diff = v0 - p
    
    a00 = jnp.dot(edge0, edge0)
    a01 = jnp.dot(edge0, edge1)
    a11 = jnp.dot(edge1, edge1)
    b0 = jnp.dot(diff, edge0)
    b1 = jnp.dot(diff, edge1)
    c = jnp.dot(diff, diff)
    det = jnp.abs(a00 * a11 - a01 * a01)
    s = a01 * b1 - a11 * b0
    t = a01 * b0 - a00 * b1
    
    # Conditions
    cond0 = s + t <= det
    cond1 = s < 0
    cond2 = t < 0
    
    # Region 0 (Interior)
    # invDet = 1.0 / det
    # s *= invDet
    # t *= invDet
    
    # We need to handle all regions. This is complex to branchless-ize.
    # Simplified approach: Project point to plane, clamp to triangle.
    # Or use a library function if available? No.
    
    # Alternative: Check edges and face.
    # 1. Project to plane.
    # 2. Check barycentric coords.
    # 3. If inside, dist is plane dist.
    # 4. If outside, dist is dist to edges.
    
    # Plane projection
    normal = jnp.cross(edge0, edge1)
    normal_len = jnp.linalg.norm(normal)
    unit_normal = normal / (normal_len + 1e-8)
    
    plane_dist = jnp.dot(p - v0, unit_normal)
    proj_p = p - plane_dist * unit_normal
    
    # Barycentric
    # v2_ = proj_p - v0
    # d00 = dot(edge0, edge0)
    # d01 = dot(edge0, edge1)
    # d11 = dot(edge1, edge1)
    # d20 = dot(v2_, edge0)
    # d21 = dot(v2_, edge1)
    # denom = d00 * d11 - d01 * d01
    # v = (d11 * d20 - d01 * d21) / denom
    # w = (d00 * d21 - d01 * d20) / denom
    # u = 1.0 - v - w
    
    v2_ = proj_p - v0
    d00 = a00
    d01 = a01
    d11 = a11
    d20 = jnp.dot(v2_, edge0)
    d21 = jnp.dot(v2_, edge1)
    denom = d00 * d11 - d01 * d01 + 1e-8
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    # Check if inside
    is_inside = (u >= 0) & (v >= 0) & (w >= 0)
    
    closest_face = proj_p
    dist_sq_face = plane_dist * plane_dist
    
    # Edge distances
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
    
    # Select closest edge
    # We want min(d_e0, d_e1, d_e2)
    
    # If inside, use face. Else use min edge.
    
    min_edge_dist_sq = jnp.minimum(d_e0, jnp.minimum(d_e1, d_e2))
    
    # Find which edge is closest (for closest point)
    # This is a bit messy to select the point without branching
    # We can just compute all 3 points and select.
    
    closest_edge = jnp.where(
        d_e0 < d_e1,
        jnp.where(d_e0 < d_e2, c_e0, c_e2),
        jnp.where(d_e1 < d_e2, c_e1, c_e2)
    )
    
    dist_sq = jnp.where(is_inside, dist_sq_face, min_edge_dist_sq)
    closest = jnp.where(is_inside, closest_face, closest_edge)
    
    return dist_sq, closest

def mesh_sdf(pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    SDF using mesh grid.
    """
    if not HAS_MESH_DATA:
        return analytical_arena_sdf(pos)
        
    # Grid lookup
    # pos shape: (..., 3)
    
    # Normalize pos to grid coords
    grid_pos = (pos - MESH_MIN_BOUNDS) / MESH_CELL_SIZE
    grid_idx = jnp.floor(grid_pos).astype(jnp.int32)
    grid_idx = jnp.clip(grid_idx, 0, MESH_GRID_SHAPE - 1)
    
    # Fetch triangle indices
    # grid_idx shape: (..., 3)
    # MESH_GRID shape: (NX, NY, NZ, MAX_TRIS)
    # We need to index into MESH_GRID
    
    # Flatten batch dims
    batch_shape = pos.shape[:-1]
    flat_pos = pos.reshape(-1, 3)
    flat_grid_idx = grid_idx.reshape(-1, 3)
    
    # Gather
    # MESH_GRID[x, y, z]
    tri_indices = MESH_GRID[flat_grid_idx[:, 0], flat_grid_idx[:, 1], flat_grid_idx[:, 2]] # (N, MAX_TRIS)
    
    # Fetch verts and normals
    # MESH_TRI_VERTS shape: (TotalTris, 3, 3)
    # tri_indices has -1 for empty slots. We need to handle that.
    # We can clamp -1 to 0 and mask the result.
    
    valid_mask = tri_indices >= 0
    safe_indices = jnp.maximum(tri_indices, 0)
    
    batch_verts = MESH_TRI_VERTS[safe_indices] # (N, MAX_TRIS, 3, 3)
    batch_normals = MESH_TRI_NORMALS[safe_indices] # (N, MAX_TRIS, 3)
    
    # Compute distances
    # We map over MAX_TRIS dimension
    
    def check_tris(p, verts, normals, mask):
        # p: (3,)
        # verts: (MAX_TRIS, 3, 3)
        # normals: (MAX_TRIS, 3)
        # mask: (MAX_TRIS,)
        
        # Vectorize point_triangle_distance over verts
        vmap_dist = jax.vmap(point_triangle_distance, in_axes=(None, 0, 0, 0))
        dists_sq, closests = vmap_dist(p, verts[:, 0], verts[:, 1], verts[:, 2])
        
        # Signed distance
        # dist = sqrt(dist_sq)
        # sign = dot(p - closest, normal)
        # If dot < 0, we are "inside" (safe).
        # If dot > 0, we are "outside" (colliding).
        # We want SDF to be positive if colliding?
        # In analytical SDF, we returned dist to surface.
        # And checked `penetration = radius - dist`.
        # If dist is small (close to wall), penetration is positive.
        # If dist is negative (inside wall), penetration is large.
        # Wait, analytical SDF: `dist_floor = z`.
        # If z=10, dist=10. Radius=90. Pen=80. Colliding!
        # So analytical SDF returns POSITIVE distance when INSIDE the arena (safe).
        # And NEGATIVE distance when OUTSIDE (colliding).
        # So we want `dist` to be positive when safe.
        
        # Mesh normals point OUT (into wall).
        # `dot(p - closest, n)`:
        # If safe (inside room), p is "behind" the wall face.
        # n points "forward" (into wall).
        # So `dot < 0`.
        # So `signed_dist = -dot`.
        # But `dot` is only approximate distance.
        # Real distance is `sqrt(dist_sq)`.
        # Sign is `-sign(dot)`.
        # So `sdf = -sign(dot) * sqrt(dist_sq)`.
        # If safe, dot < 0 -> sign -1 -> sdf > 0. Correct.
        # If colliding, dot > 0 -> sign 1 -> sdf < 0. Correct.
        
        vec = p - closests
        dot = jnp.sum(vec * normals, axis=-1)
        
        dist = jnp.sqrt(dists_sq)
        sdf = jnp.sign(dot) * dist
        
        # Mask invalid triangles
        # Set SDF to infinity for invalid triangles so they are not picked as min
        # We want the CLOSEST surface.
        # If we are safe, we want the smallest positive SDF.
        # If we are colliding, we want the largest negative SDF (closest to 0) or most negative?
        # Usually we want the signed distance to the boundary.
        # `min(sdf)`?
        # If I am safe, all SDFs are positive (or far negative if behind other walls?).
        # Wait.
        # If I am in the room, I am "behind" all walls.
        # So all dots are negative. All SDFs are positive.
        # The closest wall has the smallest positive SDF.
        # So `min(sdf)` is correct.
        
        # If I am colliding with one wall, its dot is positive. Its SDF is negative.
        # Other walls are far away (positive SDF).
        # `min(sdf)` will pick the negative one. Correct.
        
        sdf = jnp.where(mask, sdf, 1e9)
        
        min_idx = jnp.argmin(sdf)
        min_sdf = sdf[min_idx]
        best_normal = normals[min_idx] # Use triangle normal
        
        # If we are colliding (sdf < 0), the normal points INTO the wall.
        # We want to push OUT.
        # Force direction = -normal.
        # In analytical: `dist, normal = arena_sdf(pos)`.
        # `penetration = radius - dist`.
        # `new_pos = pos + normal * penetration`.
        # If dist is negative (colliding), penetration is large positive.
        # We push along `normal`.
        # If `normal` points INTO wall, we push deeper!
        # Analytical normals: `n_floor = (0,0,1)`. Points UP (into room).
        # Mesh normals: Point OUT (into wall).
        # So we must FLIP mesh normals to match analytical convention.
        
        return min_sdf, best_normal

    # Vmap over batch
    vmap_check = jax.vmap(check_tris, in_axes=(0, 0, 0, 0))
    dists, normals = vmap_check(flat_pos, batch_verts, batch_normals, valid_mask)
    
    # Reshape back
    dists = dists.reshape(batch_shape)
    normals = normals.reshape(*batch_shape, 3)
    
    return dists, normals

def simple_arena_sdf(pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simple AABB + Goal SDF (no corners).
    """
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    
    # 1. AABB Distances
    dist_floor = z
    dist_ceiling = ARENA_HEIGHT - z
    dist_left = x + ARENA_EXTENT_X
    dist_right = ARENA_EXTENT_X - x
    dist_back = y + ARENA_EXTENT_Y
    dist_front = ARENA_EXTENT_Y - y
    
    # 2. Goal Openings
    GOAL_HALF_WIDTH = 892.755
    GOAL_HEIGHT = 642.775
    
    in_goal_x = jnp.abs(x) < GOAL_HALF_WIDTH
    in_goal_z = z < GOAL_HEIGHT
    in_goal_aperture = in_goal_x & in_goal_z
    
    # If in aperture, make back/front wall distance large
    dist_back = jnp.where(in_goal_aperture, 1e6, dist_back)
    dist_front = jnp.where(in_goal_aperture, 1e6, dist_front)
    
    # Stack distances
    distances = jnp.stack([
        dist_floor, dist_ceiling,
        dist_left, dist_right,
        dist_back, dist_front
    ], axis=-1)
    
    # Normals
    n_floor = jnp.array([0.0, 0.0, 1.0])
    n_ceil = jnp.array([0.0, 0.0, -1.0])
    n_left = jnp.array([1.0, 0.0, 0.0])
    n_right = jnp.array([-1.0, 0.0, 0.0])
    n_back = jnp.array([0.0, 1.0, 0.0])
    n_front = jnp.array([0.0, -1.0, 0.0])
    
    ones = jnp.ones_like(x)[..., None]
    def b(n): return n * ones
    
    normals_stack = jnp.stack([
        b(n_floor), b(n_ceil),
        b(n_left), b(n_right),
        b(n_back), b(n_front)
    ], axis=-2)
    
    min_idx = jnp.argmin(distances, axis=-1)
    min_dist = jnp.min(distances, axis=-1)
    
    min_idx_expanded = min_idx[..., None, None]
    normal = jnp.take_along_axis(normals_stack, min_idx_expanded, axis=-2)
    normal = normal[..., 0, :]
    
    return min_dist, normal

def analytical_arena_sdf_legacy(pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Signed Distance Field for the Rocket League arena.
    
    Computes the signed distance from a point to the nearest arena surface
    and the surface normal at that closest point.
    
    The arena is modeled as an AABB with corner ramps:
    - Floor at Z=0
    - Ceiling at Z=ARENA_HEIGHT
    - Walls at X=±ARENA_EXTENT_X, Y=±ARENA_EXTENT_Y
    - Goal openings on Y walls
    - 45° ceiling ramps and vertical corner chamfers
    
    Args:
        pos: Query point positions. Shape: (..., 3)
        
    Returns:
        distance: Signed distance to surface (negative = inside arena). Shape: (...)
        normal: Surface normal at closest point. Shape: (..., 3)
    """
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    
    # 1. AABB Distances
    dist_floor = z
    dist_ceiling = ARENA_HEIGHT - z
    dist_left = x + ARENA_EXTENT_X
    dist_right = ARENA_EXTENT_X - x
    dist_back = y + ARENA_EXTENT_Y
    dist_front = ARENA_EXTENT_Y - y
    
    # 2. Goal Openings
    GOAL_HALF_WIDTH = 892.755
    GOAL_HEIGHT = 642.775
    
    in_goal_x = jnp.abs(x) < GOAL_HALF_WIDTH
    in_goal_z = z < GOAL_HEIGHT
    in_goal_aperture = in_goal_x & in_goal_z
    
    # If in aperture, make back/front wall distance large
    dist_back = jnp.where(in_goal_aperture, 1e6, dist_back)
    dist_front = jnp.where(in_goal_aperture, 1e6, dist_front)
    
    # 3. Corner Ramps (Ceiling & Vertical)
    # Standard Rocket League Arena (Soccar) has curved corners, not chamfers.
    # We use Euclidean distance to a rounded corner for the vertical walls.
    
    CORNER_RADIUS = 1152.0
    # Centers of the 4 corner circles
    # Front-Right (Quad 1): (+X, -Y) -> Center at (EXTENT_X - R, -EXTENT_Y + R)
    # Back-Right (Quad 2): (+X, +Y) -> Center at (EXTENT_X - R, EXTENT_Y - R)
    # ... and so on.
    
    # We can exploit symmetry by working in the first quadrant of absolute coordinates
    abs_x = jnp.abs(x)
    abs_y = jnp.abs(y)
    
    corner_center_x = ARENA_EXTENT_X - CORNER_RADIUS
    corner_center_y = ARENA_EXTENT_Y - CORNER_RADIUS
    
    # Check if we are in the corner region (outside the central box)
    in_corner_region = (abs_x > corner_center_x) & (abs_y > corner_center_y)
    
    # Distance to the corner center
    dx = abs_x - corner_center_x
    dy = abs_y - corner_center_y
    dist_corner = jnp.sqrt(dx*dx + dy*dy) - CORNER_RADIUS
    
    # If in corner region, this distance replaces the wall distances
    # The wall distance is min(dist_right, dist_back) usually.
    # If we are in the corner, the wall is the curve.
    # Distance to wall = negative of distance to surface (SDF convention here is negative = inside?)
    # Wait, the function returns "Signed distance to surface (negative = inside arena)".
    # dist_right = ARENA_EXTENT_X - x. If x < EXTENT, dist is positive.
    # So positive = inside.
    # My dist_corner calculation: sqrt(...) - R.
    # If we are inside the curve (closer to center), sqrt < R, so dist_corner is negative.
    # So we want positive = inside.
    # So dist_corner_sdf = CORNER_RADIUS - sqrt(dx*dx + dy*dy)
    
    dist_corner_sdf = CORNER_RADIUS - jnp.sqrt(dx*dx + dy*dy)
    
    # Only apply this if we are actually in the corner zone
    # Otherwise we use the standard wall distances
    # We can combine them using smooth min or just min/max logic.
    # Since we want the intersection of volumes, we take the minimum distance to any boundary.
    # But here we are defining the boundary itself.
    
    # Let's refine the wall distances.
    # The "Side Wall" is valid only when |y| < corner_center_y
    # The "Back Wall" is valid only when |x| < corner_center_x
    # The "Corner" is valid when both are exceeded.
    
    # However, simpler SDF logic for a rounded box (2D):
    # d = length(max(abs(p) - b, 0.0)) - r
    # Here we want the interior distance.
    
    # Let's stick to the "in_corner_region" logic for simplicity and readability
    # If in corner region, use corner SDF. Else use min(dist_x, dist_y).
    
    # Update dist_right/left/back/front based on corner
    # Actually, we can just add the corner distance to the stack and let the minimum win?
    # No, because dist_right might be large (far from wall) but we are in the corner.
    # If we are in the corner, dist_right is not the correct distance to the surface.
    
    # Correct approach:
    # 1. Compute distance to the rounded rectangle boundary.
    # 2. But we have separate walls in the return values.
    # The physics engine likely uses the minimum of all these to resolve collision.
    # So we should output a "dist_corner" and ensure dist_right/back are ignored if we are in the corner?
    # Or better: modify dist_right/back/left/front to be "infinity" if we are in the corner region,
    # and add a specific "dist_corner" output.
    
    # But the return signature is fixed? No, it returns a stack of distances.
    # I can add more channels.
    
    # Let's modify the existing wall distances to account for corners.
    # This is tricky because we have 4 corners.
    
    # Alternative: Just add the 4 corner distances to the stack.
    # And make sure the linear wall distances are large when in the corner?
    # If I am in the corner, dist_right = (EXTENT - x).
    # dist_corner = (R - dist_from_center).
    # If I am in the corner, dist_corner < dist_right.
    # So if I take min(dist_right, dist_corner), it will pick the corner.
    # Wait.
    # Example: x = EXTENT - 10, y = EXTENT - 10. (Very close to corner).
    # dist_right = 10. dist_back = 10.
    # corner_center = EXTENT - 1152.
    # dx = 1142, dy = 1142.
    # dist_from_center = sqrt(1142^2 + 1142^2) = 1615.
    # dist_corner_sdf = 1152 - 1615 = -463. (Outside arena!)
    # dist_right = 10 (Inside arena).
    # The collision solver pushes out if dist < 0.
    # So if dist_corner is negative, it will push.
    # But dist_right is positive.
    # The solver usually takes the *minimum* positive distance (closest surface) or *maximum* negative distance (deepest penetration).
    # Actually, usually it checks "if dist < radius".
    # If dist_corner < radius, we collide.
    # So adding dist_corner to the list is sufficient, provided the solver checks all of them.
    
    # Let's check resolve_ball_arena_collision in jax_sim.py (or collision.py if moved).
    # Wait, resolve_ball_arena_collision in jax_sim.py (which I read) does NOT use arena_sdf!
    # It uses explicit planes:
    # hit_left = px < min_x
    # px = jnp.where(hit_left, min_x, px)
    
    # AHA! `resolve_ball_arena_collision` is hardcoded to AABB!
    # `arena_sdf` is used for CAR suspension raycasts.
    # So changing `arena_sdf` fixes the car suspension on corners (maybe), but NOT the ball collision.
    # AND NOT the car body collision (resolve_car_arena_collision).
    
    # I need to update:
    # 1. `arena_sdf` (for suspension/raycasts)
    # 2. `resolve_ball_arena_collision` (for ball physics)
    # 3. `resolve_car_arena_collision` (for car body physics)
    
    # Let's start with `arena_sdf` in this file.
    
    # Ceiling Ramps (45 deg) - These are actually chamfers in standard RL too? 
    # Or are they curved? Usually ceiling corners are chamfered or curved. 
    # Let's keep them as chamfers for now unless requested, user focused on "borda curva" (walls).
    
    RAMP_INSET = 820.0
    SQRT2 = jnp.sqrt(2.0)
    
    ramp_c_r = (dist_right + dist_ceiling - RAMP_INSET) / SQRT2
    ramp_c_l = (dist_left + dist_ceiling - RAMP_INSET) / SQRT2
    ramp_c_b = (dist_back + dist_ceiling - RAMP_INSET) / SQRT2
    ramp_c_f = (dist_front + dist_ceiling - RAMP_INSET) / SQRT2
    
    # Vertical Corners (Rounded)
    # We calculate distance to the 4 rounded corners
    # Corner centers:
    cx_r = ARENA_EXTENT_X - CORNER_RADIUS
    cx_l = -ARENA_EXTENT_X + CORNER_RADIUS
    cy_b = ARENA_EXTENT_Y - CORNER_RADIUS  # Back is +Y in this file? 
    # In jax_sim.py: min_y = -ARENA_EXTENT_Y. Back wall (-Y)? 
    # Let's check jax_sim.py again.
    # "Back wall (-Y)". "Front wall (+Y)".
    # In collision.py: dist_back = y + ARENA_EXTENT_Y. (y - (-EXTENT)). So Back is -Y.
    # dist_front = ARENA_EXTENT_Y - y. Front is +Y.
    
    cy_back = -ARENA_EXTENT_Y + CORNER_RADIUS
    cy_front = ARENA_EXTENT_Y - CORNER_RADIUS
    
    # Distances to corner axes (cylinders)
    # We want (CORNER_RADIUS - dist_to_axis)
    
    # RB: Right (+X), Back (-Y)
    d_rb = jnp.sqrt((x - cx_r)**2 + (y - cy_back)**2)
    ramp_w_rb = CORNER_RADIUS - d_rb
    # Only valid if x > cx_r and y < cy_back
    mask_rb = (x > cx_r) & (y < cy_back)
    ramp_w_rb = jnp.where(mask_rb, ramp_w_rb, 1e6)

    # RF: Right (+X), Front (+Y)
    d_rf = jnp.sqrt((x - cx_r)**2 + (y - cy_front)**2)
    ramp_w_rf = CORNER_RADIUS - d_rf
    mask_rf = (x > cx_r) & (y > cy_front)
    ramp_w_rf = jnp.where(mask_rf, ramp_w_rf, 1e6)

    # LB: Left (-X), Back (-Y)
    d_lb = jnp.sqrt((x - cx_l)**2 + (y - cy_back)**2)
    ramp_w_lb = CORNER_RADIUS - d_lb
    mask_lb = (x < cx_l) & (y < cy_back)
    ramp_w_lb = jnp.where(mask_lb, ramp_w_lb, 1e6)

    # LF: Left (-X), Front (+Y)
    d_lf = jnp.sqrt((x - cx_l)**2 + (y - cy_front)**2)
    ramp_w_lf = CORNER_RADIUS - d_lf
    mask_lf = (x < cx_l) & (y > cy_front)
    ramp_w_lf = jnp.where(mask_lf, ramp_w_lf, 1e6)
    
    # Stack all distances
    distances = jnp.stack([
        dist_floor, dist_ceiling,
        dist_left, dist_right,
        dist_back, dist_front,
        ramp_c_r, ramp_c_l, ramp_c_b, ramp_c_f,
        ramp_w_rb, ramp_w_rf, ramp_w_lb, ramp_w_lf
    ], axis=-1)
    
    # Normals
    n_floor = jnp.array([0.0, 0.0, 1.0])
    n_ceil = jnp.array([0.0, 0.0, -1.0])
    n_left = jnp.array([1.0, 0.0, 0.0])
    n_right = jnp.array([-1.0, 0.0, 0.0])
    n_back = jnp.array([0.0, 1.0, 0.0])
    n_front = jnp.array([0.0, -1.0, 0.0])
    
    # Ramp normals (Ceiling)
    n_c_r = (n_right + n_ceil) / SQRT2
    n_c_l = (n_left + n_ceil) / SQRT2
    n_c_b = (n_back + n_ceil) / SQRT2
    n_c_f = (n_front + n_ceil) / SQRT2
    
    # Dynamic normals for corners
    # We compute them for all points, but only use them if the corner is the closest surface
    # Normal points from wall to interior (towards center of curvature)
    # n = normalize(center - pos) (projected to XY)
    
    # RB: Center (cx_r, cy_back)
    v_rb = jnp.stack([cx_r - x, cy_back - y, jnp.zeros_like(z)], axis=-1)
    n_w_rb = v_rb / (jnp.linalg.norm(v_rb, axis=-1, keepdims=True) + 1e-6)
    
    # RF: Center (cx_r, cy_front)
    v_rf = jnp.stack([cx_r - x, cy_front - y, jnp.zeros_like(z)], axis=-1)
    n_w_rf = v_rf / (jnp.linalg.norm(v_rf, axis=-1, keepdims=True) + 1e-6)
    
    # LB: Center (cx_l, cy_back)
    v_lb = jnp.stack([cx_l - x, cy_back - y, jnp.zeros_like(z)], axis=-1)
    n_w_lb = v_lb / (jnp.linalg.norm(v_lb, axis=-1, keepdims=True) + 1e-6)
    
    # LF: Center (cx_l, cy_front)
    v_lf = jnp.stack([cx_l - x, cy_front - y, jnp.zeros_like(z)], axis=-1)
    n_w_lf = v_lf / (jnp.linalg.norm(v_lf, axis=-1, keepdims=True) + 1e-6)
    
    # Stack static normals (expand to match shape)
    # We need to broadcast static normals to (..., 3)
    ones = jnp.ones_like(x)[..., None]
    
    # Helper to broadcast
    def b(n): return n * ones
    
    normals_stack = jnp.stack([
        b(n_floor), b(n_ceil),
        b(n_left), b(n_right),
        b(n_back), b(n_front),
        b(n_c_r), b(n_c_l), b(n_c_b), b(n_c_f),
        n_w_rb, n_w_rf, n_w_lb, n_w_lf
    ], axis=-2) # Stack along the "surfaces" dimension (second to last)
    
    # Find closest surface
    min_idx = jnp.argmin(distances, axis=-1)
    min_dist = jnp.min(distances, axis=-1)
    
    # Get normal
    # min_idx shape: (...)
    # normals_stack shape: (..., 14, 3)
    # We need to gather
    
    # JAX gather/take
    # We can use one-hot multiplication or advanced indexing
    # Since min_idx is dynamic, we use take_along_axis
    
    min_idx_expanded = min_idx[..., None, None] # (..., 1, 1)
    normal = jnp.take_along_axis(normals_stack, min_idx_expanded, axis=-2)
    normal = normal[..., 0, :] # Remove the singleton dimension
    
    return min_dist, normal

# Extracted from Rocket League collision meshes
# Format: [nx, ny, nz, d]
# Units: Unreal Units (d is scaled by 50.0 from Bullet units)
# Normals point INWARD (towards the center of the field)
# Condition for inside: dot(n, p) > d  =>  d - dot(n, p) < 0
# SDF = max(d - dot(n, p))

ARENA_PLANES = jnp.array([
    # Floor (z=0)
    [0.0, 0.0, 1.0, 0.0],
    # Ceiling (z=2044)
    [0.0, 0.0, -1.0, -2044.0],
    # Side Walls (x=+-4096)
    [-1.0, 0.0, 0.0, -4096.0],
    [1.0, 0.0, 0.0, -4096.0],
    # Back Walls (y=+-5120) - Note: These have holes for goals!
    # We handle goals separately.
    [0.0, -1.0, 0.0, -5120.0],
    [0.0, 1.0, 0.0, -5120.0],
    
    # Corners (45 deg)
    # N=[-0.707, -0.707, 0], D=-5700
    [-0.7071, -0.7071, 0.0, -5700.0],
    [0.7071, -0.7071, 0.0, -5700.0],
    [-0.7071, 0.7071, 0.0, -5700.0],
    [0.7071, 0.7071, 0.0, -5700.0],
    
    # Ramps (Floor/Wall connections)
    # N=[0, -0.879, 0.477], D=-4440 (88.8 * 50)
    [0.0, -0.879, 0.477, -4440.0],
    [0.0, 0.879, 0.477, -4440.0],
    [-0.879, 0.0, 0.477, -3505.0], # D=-70.1 * 50 = 3505
    [0.879, 0.0, 0.477, -3505.0],
    
    # N=[0, -0.955, 0.297], D=-4850 (97.0 * 50)
    [0.0, -0.955, 0.297, -4850.0],
    [0.0, 0.955, 0.297, -4850.0],
    
    # N=[0, -0.637, 0.771], D=-3195 (63.9 * 50)
    [0.0, -0.637, 0.771, -3195.0],
    [0.0, 0.637, 0.771, -3195.0],
    
    # N=[0, -0.771, 0.637], D=-3885 (77.7 * 50)
    [0.0, -0.771, 0.637, -3885.0],
    [0.0, 0.771, 0.637, -3885.0],
    
    # N=[0, -0.477, 0.879], D=-2385 (47.7 * 50)
    [0.0, -0.477, 0.879, -2385.0],
    [0.0, 0.477, 0.879, -2385.0],
])

GOAL_PLANES_POS = jnp.array([
    # Back (y=6000 approx)
    [0.0, -1.0, 0.0, -5995.0], # D=-119.9 * 50 = 5995
    # Sides (x=+-895)
    [-1.0, 0.0, 0.0, -895.0], # D=-17.9 * 50 = 895
    [1.0, 0.0, 0.0, -895.0],
    # Top (z=640)
    [0.0, 0.0, -1.0, -640.0], # D=-12.8 * 50 = 640
    # Bottom (z=0)
    [0.0, 0.0, 1.0, 0.0],
])

GOAL_PLANES_NEG = jnp.array([
    # Back (y=-6000)
    [0.0, 1.0, 0.0, -5995.0],
    # Sides
    [-1.0, 0.0, 0.0, -895.0],
    [1.0, 0.0, 0.0, -895.0],
    # Top
    [0.0, 0.0, -1.0, -640.0],
    # Bottom
    [0.0, 0.0, 1.0, 0.0],
])

def arena_sdf(pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Exact Arena SDF using extracted plane equations.
    Returns POSITIVE distance when INSIDE the arena (safe).
    Returns NEGATIVE distance when PENETRATING a wall.
    """
    # Main Arena
    # dist = dot(n, p) - d
    # We are inside if dist > 0 for ALL planes.
    # So dist_vol = min(dist_planes)
    
    # Main Arena SDF
    n_main = ARENA_PLANES[:, :3]
    d_main = ARENA_PLANES[:, 3]
    dot_main = jnp.matmul(pos, n_main.T) # (B, N)
    dist_main_all = dot_main - d_main
    dist_main = jnp.min(dist_main_all, axis=-1) # (B,)
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
    # We want the union of the EMPTY spaces.
    # Inside main: dist_main > 0
    # Inside goal: dist_goal > 0
    # Union of inside regions -> max(dist)
    
    max_dist = jnp.maximum(dist_main, jnp.maximum(dist_gp, dist_gn))
    
    # Determine which volume we are in to pick the normal
    # We pick the normal from the volume that gives the BEST (largest) distance.
    # i.e. if we are deep inside the goal (dist_gp large) but outside main (dist_main small/neg),
    # we are safe because of the goal.
    
    use_gp = (dist_gp > dist_main) & (dist_gp > dist_gn)
    use_gn = (dist_gn > dist_main) & (dist_gn > dist_gp)
    
    normal = jnp.where(
        use_gp[..., None],
        norm_gp,
        jnp.where(
            use_gn[..., None],
            norm_gn,
            norm_main
        )
    )
    
    return max_dist, normal

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
    
    # Check penetration and contact
    penetration = radius - dist
    is_penetrating = penetration > 0
    
    # Use a contact tolerance for friction - ball is "in contact" if very close
    # This allows friction to work even when ball is resting on surface
    contact_tolerance = 2.0  # UU - about 2 units tolerance
    is_in_contact = penetration > -contact_tolerance  # True if ball within tolerance of surface
    
    # Resolve position (push out only if penetrating)
    new_pos = jnp.where(
        is_penetrating[..., None],
        pos + normal * penetration[..., None],
        pos
    )
    
    # Calculate velocity at contact point (ball surface)
    # Contact point is at: pos - normal * radius
    contact_offset = -normal * radius  # Vector from ball center to contact
    
    # Surface velocity = linear vel + angular vel × contact_offset
    surface_vel = vel + jnp.cross(ang_vel, contact_offset)
    
    # Decompose into normal and tangential components
    v_dot_n = jnp.sum(surface_vel * normal, axis=-1, keepdims=True)
    v_normal = v_dot_n * normal
    v_tangent = surface_vel - v_normal
    
    # Check if moving INTO the surface (v_dot_n < 0) for bounce
    should_bounce = is_penetrating & (v_dot_n[..., 0] < 0)
    
    # Ball moment of inertia: I = (2/5) * m * r^2
    ball_inertia = (2.0 / 5.0) * BALL_MASS * (radius ** 2)
    
    # Tangential slip velocity magnitude
    v_t_mag = jnp.linalg.norm(v_tangent, axis=-1, keepdims=True)
    v_t_dir = v_tangent / (v_t_mag + 1e-8)
    
    # === BOUNCE (Normal direction) ===
    # Apply normal velocity change (bounce) only if approaching surface
    new_v_normal = jnp.where(
        should_bounce[..., None],
        -v_normal * restitution,
        v_normal
    )
    
    # === FRICTION (Tangential direction) ===
    # Apply friction whenever in contact, not just when bouncing
    # This makes the ball roll properly on the ground
    
    # Coulomb friction limit based on normal force
    # When bouncing: impulse = (1+e) * m * |v_n|
    # When rolling: use gravity-based normal force
    v_n_mag = jnp.abs(v_dot_n)
    bounce_impulse = v_n_mag * (1 + restitution) * BALL_MASS
    
    # For rolling contact without bouncing, use weight as normal force
    # F_n = m * g * cos(angle), but simplified to m * g * |normal_z|
    gravity_normal_force = BALL_MASS * jnp.abs(GRAVITY_Z) * jnp.abs(normal[..., 2:3])
    
    # Use bounce impulse when bouncing, gravity-based when just rolling
    effective_normal_impulse = jnp.where(
        should_bounce[..., None],
        bounce_impulse,
        gravity_normal_force * DT  # Force * dt = impulse
    )
    
    max_friction_impulse = friction * effective_normal_impulse
    
    # Impulse needed to stop tangential slip
    # For sphere: J_t = m * v_t / (1 + m*r^2/I) = m * v_t * (2/7)
    slip_stop_impulse = v_t_mag * BALL_MASS * (2.0 / 7.0)
    
    # Actual friction impulse is the smaller of the two
    friction_impulse_mag = jnp.minimum(max_friction_impulse, slip_stop_impulse)
    friction_impulse = -v_t_dir * friction_impulse_mag
    
    # Apply friction impulse to linear velocity
    friction_vel_change = friction_impulse / BALL_MASS
    
    # Apply friction torque to angular velocity
    friction_torque = jnp.cross(contact_offset, friction_impulse)
    ang_vel_change = friction_torque / ball_inertia
    
    # New velocities - apply friction whenever in contact (using tolerance)
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
    
    # Clamp angular velocity
    ang_speed = jnp.linalg.norm(new_ang_vel, axis=-1, keepdims=True)
    new_ang_vel = jnp.where(
        ang_speed > BALL_MAX_ANG_SPEED,
        new_ang_vel * (BALL_MAX_ANG_SPEED / (ang_speed + 1e-8)),
        new_ang_vel
    )
    
    return new_pos, new_vel, new_ang_vel


def resolve_car_arena_collision(
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    ang_vel: jnp.ndarray,
    quat: jnp.ndarray,
    hitbox_half_size: jnp.ndarray = OCTANE_HITBOX_SIZE / 2,
    hitbox_offset: jnp.ndarray = OCTANE_HITBOX_OFFSET,
    restitution: float = 0.3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Resolve car-arena collisions by checking hitbox corners against arena SDF.
    
    This function checks the 8 corners of the car's hitbox (OBB) against the
    arena geometry. If any corner penetrates, we apply a position correction
    and velocity impulse based on physics.
    
    Args:
        pos: Car CoM position. Shape: (N, MAX_CARS, 3)
        vel: Car linear velocity. Shape: (N, MAX_CARS, 3)
        ang_vel: Car angular velocity. Shape: (N, MAX_CARS, 3)
        quat: Car rotation quaternion. Shape: (N, MAX_CARS, 4)
        hitbox_half_size: Half-extents of hitbox (3,)
        hitbox_offset: Offset from CoM to hitbox center (3,)
        restitution: Coefficient of restitution (0 = inelastic, 1 = elastic)
        
    Returns:
        Tuple of (new_pos, new_vel, new_ang_vel)
    """
    # Generate 8 corners of the hitbox in local space
    # Corners are combinations of +/- half_size
    signs = jnp.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ], dtype=jnp.float32)  # (8, 3)
    
    # Local corners relative to hitbox center
    corners_local = signs * hitbox_half_size  # (8, 3)
    # Offset to CoM
    corners_local = corners_local + hitbox_offset  # (8, 3)
    
    # Transform corners to world space
    # quat: (N, MAX_CARS, 4), corners_local: (8, 3)
    # Need to rotate each corner by each car's quaternion
    # Result: (N, MAX_CARS, 8, 3)
    
    # Expand dimensions for broadcasting
    # quat_expanded: (N, MAX_CARS, 1, 4)
    quat_expanded = quat[..., None, :]
    # corners_local_expanded: (1, 1, 8, 3)
    corners_local_expanded = corners_local[None, None, :, :]
    
    # Rotate corners - quat_rotate_vector expects (N, MAX_CARS, 4) and (N, MAX_CARS, 3)
    # We need to do this per-corner
    def rotate_corner(corner):
        """Rotate a single corner by all car quaternions."""
        # corner: (3,)
        # quat: (N, MAX_CARS, 4)
        return quat_rotate_vector(quat, corner)
    
    # Vectorize over corners
    corners_world_offset = jax.vmap(rotate_corner)(corners_local)  # (8, N, MAX_CARS, 3)
    corners_world_offset = jnp.moveaxis(corners_world_offset, 0, -2)  # (N, MAX_CARS, 8, 3)
    
    # Add car position
    corners_world = corners_world_offset + pos[..., None, :]  # (N, MAX_CARS, 8, 3)
    
    # Check each corner against arena SDF
    # arena_sdf expects (N, MAX_CARS, 3), so we need to reshape
    orig_shape = corners_world.shape  # (N, MAX_CARS, 8, 3)
    corners_flat = corners_world.reshape(-1, 3)  # (N*MAX_CARS*8, 3)
    # Add batch dimension for arena_sdf
    corners_for_sdf = corners_flat[None, :, :]  # (1, N*MAX_CARS*8, 3)
    
    dist_flat, normal_flat = arena_sdf(corners_for_sdf)  # (1, N*MAX_CARS*8), (1, N*MAX_CARS*8, 3)
    dist_flat = dist_flat[0]  # (N*MAX_CARS*8,)
    normal_flat = normal_flat[0]  # (N*MAX_CARS*8, 3)
    
    # Reshape back
    n_envs = orig_shape[0]
    max_cars = orig_shape[1]
    corner_dist = dist_flat.reshape(n_envs, max_cars, 8)  # (N, MAX_CARS, 8)
    corner_normal = normal_flat.reshape(n_envs, max_cars, 8, 3)  # (N, MAX_CARS, 8, 3)
    
    # Find penetration (negative distance = penetration)
    penetration = -corner_dist  # Positive when penetrating
    is_penetrating = penetration > 0  # (N, MAX_CARS, 8)
    
    # Find the deepest penetrating corner for each car
    # Use max to find worst penetration
    max_penetration = jnp.max(penetration, axis=-1)  # (N, MAX_CARS)
    deepest_idx = jnp.argmax(penetration, axis=-1)  # (N, MAX_CARS)
    
    # Get normal at deepest penetration point
    # Use gather to select the correct normal
    batch_idx = jnp.arange(n_envs)[:, None]  # (N, 1)
    car_idx = jnp.arange(max_cars)[None, :]  # (1, MAX_CARS)
    deepest_normal = corner_normal[batch_idx, car_idx, deepest_idx]  # (N, MAX_CARS, 3)
    deepest_corner_offset = corners_world_offset[batch_idx, car_idx, deepest_idx]  # (N, MAX_CARS, 3)
    
    # Check if any corner is penetrating
    any_penetrating = max_penetration > 0  # (N, MAX_CARS)
    
    # Position correction: push CoM out by penetration amount along normal
    pos_correction = deepest_normal * max_penetration[..., None]
    new_pos = jnp.where(
        any_penetrating[..., None],
        pos + pos_correction,
        pos
    )
    
    # Velocity at contact point: v_contact = v_com + omega x r
    r = deepest_corner_offset  # Vector from CoM to contact point
    vel_at_contact = vel + jnp.cross(ang_vel, r)
    
    # Relative velocity along normal (negative = approaching)
    v_rel_n = jnp.sum(vel_at_contact * deepest_normal, axis=-1)  # (N, MAX_CARS)
    
    # Only apply impulse if approaching the wall
    should_respond = any_penetrating & (v_rel_n < 0)
    
    # Simple impulse-based response
    # For a point contact: J = -(1 + e) * v_rel_n / (1/m + (r x n)·(I^-1 · (r x n)))
    # Simplified: assume infinite mass for arena, just reverse normal velocity component
    
    # Normal impulse magnitude (per unit mass for simplicity)
    # j = -(1 + e) * v_rel_n
    j = -(1.0 + restitution) * v_rel_n
    
    # Apply impulse to linear velocity
    # delta_v = j * n / m, but we compute j as v_rel_n which already includes m implicitly
    # So: delta_v = j * n (treating j as velocity change)
    vel_impulse = j[..., None] * deepest_normal
    
    new_vel = jnp.where(
        should_respond[..., None],
        vel + vel_impulse,
        vel
    )
    
    # Apply impulse to angular velocity
    # delta_omega = I^-1 · (r x J)
    # Simplified: use scalar inertia approximation
    from sim_constants import CAR_INERTIA
    inertia_avg = jnp.mean(jnp.array(CAR_INERTIA))
    
    # r x (j * n) = j * (r x n)
    torque_arm = jnp.cross(r, deepest_normal)  # (N, MAX_CARS, 3)
    ang_impulse = j[..., None] * torque_arm / inertia_avg
    
    new_ang_vel = jnp.where(
        should_respond[..., None],
        ang_vel + ang_impulse,
        ang_vel
    )
    
    return new_pos, new_vel, new_ang_vel


def resolve_car_ball_collision(
    ball_pos: jnp.ndarray,
    ball_vel: jnp.ndarray,
    ball_ang_vel: jnp.ndarray,
    car_pos: jnp.ndarray,
    car_vel: jnp.ndarray,
    car_ang_vel: jnp.ndarray,
    car_quat: jnp.ndarray,
    ball_radius: float = BALL_RADIUS,
    hitbox_half_size: jnp.ndarray = OCTANE_HITBOX_SIZE / 2,
    hitbox_offset: jnp.ndarray = OCTANE_HITBOX_OFFSET,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Resolve collisions between ball and all cars.
    
    Uses OBB-Sphere collision detection:
    1. Transform ball position to car's local space
    2. Find closest point on hitbox (AABB in local space)
    3. Check if distance < ball_radius
    4. Apply impulse based on relative velocity and RL's extra impulse mechanics
    5. Calculate friction torque to apply spin to the ball
    
    Args:
        ball_pos: Ball positions. Shape: (N, 3)
        ball_vel: Ball velocities. Shape: (N, 3)
        ball_ang_vel: Ball angular velocities. Shape: (N, 3)
        car_pos: Car positions. Shape: (N, MAX_CARS, 3)
        car_vel: Car velocities. Shape: (N, MAX_CARS, 3)
        car_ang_vel: Car angular velocities. Shape: (N, MAX_CARS, 3)
        car_quat: Car quaternions. Shape: (N, MAX_CARS, 4)
        ball_radius: Ball collision radius
        hitbox_half_size: Half-extents of car hitbox
        hitbox_offset: Offset of hitbox center from car origin
        
    Returns:
        Tuple of (new_ball_vel, new_ball_ang_vel, new_car_vel, new_car_ang_vel, hit_mask)
        hit_mask shape: (N, MAX_CARS) - True if car hit ball this frame
    """
    # Get batch dimensions
    n_envs = ball_pos.shape[0]
    max_cars = car_pos.shape[1]
    
    # Expand ball to match car dimensions
    ball_pos_exp = ball_pos[:, None, :]
    ball_vel_exp = ball_vel[:, None, :]
    
    # Transform ball to car's local space
    rel_pos_world = ball_pos_exp - car_pos
    
    # Get inverse quaternion (conjugate for unit quaternions)
    car_quat_inv = car_quat * jnp.array([1.0, -1.0, -1.0, -1.0])
    
    # Rotate relative position into car's local space
    local_ball_pos = quat_rotate_vector(car_quat_inv, rel_pos_world)
    local_ball_pos_hitbox = local_ball_pos - hitbox_offset
    
    # Find closest point on hitbox (AABB clamp)
    closest_local = jnp.clip(local_ball_pos_hitbox, -hitbox_half_size, hitbox_half_size)
    
    # Distance vector from closest point to ball center
    dist_vec_local = local_ball_pos_hitbox - closest_local
    dist_sq = jnp.sum(dist_vec_local ** 2, axis=-1)
    
    # Collision detection
    is_colliding = dist_sq < (ball_radius ** 2)
    dist = jnp.sqrt(dist_sq + 1e-8)
    penetration = jnp.maximum(ball_radius - dist, 0.0)
    
    # Calculate collision normal (in world space)
    local_normal = dist_vec_local / (dist[..., None] + 1e-8)
    world_normal = quat_rotate_vector(car_quat, local_normal)
    
    # Contact point on car surface (in world space)
    car_contact_local = closest_local + hitbox_offset  # In car local space
    car_contact_world = quat_rotate_vector(car_quat, car_contact_local) + car_pos  # World space
    
    # Contact point offset from car center (for torque calculation)
    car_contact_offset = car_contact_world - car_pos  # (N, MAX_CARS, 3)
    
    # Calculate relative velocity at contact point
    car_vel_at_contact = car_vel + jnp.cross(car_ang_vel, car_contact_offset)
    rel_vel = ball_vel_exp - car_vel_at_contact
    rel_vel_normal = jnp.sum(rel_vel * world_normal, axis=-1)
    
    # Only process if objects are approaching
    approaching = rel_vel_normal < 0
    
    # Calculate collision impulse using conservation of momentum
    # For ball-car collision with restitution e:
    # J = -(1+e) * v_rel_n / (1/m1 + 1/m2)
    # But C++ uses restitution = 0 for car-ball, making it inelastic
    inv_mass_sum = 1.0 / BALL_MASS + 1.0 / CAR_MASS
    restitution = 0.0  # C++: CARBALL_COLLISION_RESTITUTION = 0.0
    impulse_mag = -(1 + restitution) * rel_vel_normal / inv_mass_sum
    
    impulse_mask = is_colliding & approaching
    impulse_mag = jnp.where(impulse_mask, impulse_mag, 0.0)
    impulse = impulse_mag[..., None] * world_normal  # Impulse applied TO the ball
    
    # RL's "Extra Impulse" (power hit mechanic)
    # This is the main source of ball velocity after a hit
    rel_speed = jnp.linalg.norm(rel_vel, axis=-1)
    rel_speed_clamped = jnp.minimum(rel_speed, BALL_CAR_EXTRA_IMPULSE_MAX_DELTA_VEL)
    
    # Get car forward direction
    car_forward = quat_rotate_vector(car_quat, jnp.array([1.0, 0.0, 0.0]))
    
    # Hit direction: from car center to ball, with Z component scaled down
    hit_dir_raw = rel_pos_world * jnp.array([1.0, 1.0, BALL_CAR_EXTRA_IMPULSE_Z_SCALE])
    hit_dir = hit_dir_raw / (jnp.linalg.norm(hit_dir_raw, axis=-1, keepdims=True) + 1e-8)
    
    # Reduce forward component (makes hits go more "up" and less "forward")
    forward_component = jnp.sum(hit_dir * car_forward, axis=-1, keepdims=True)
    forward_adjustment = car_forward * forward_component * (1 - BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE)
    hit_dir = hit_dir - forward_adjustment
    hit_dir = hit_dir / (jnp.linalg.norm(hit_dir, axis=-1, keepdims=True) + 1e-8)
    
    # Interpolate extra impulse factor based on relative speed
    extra_factor = jnp.interp(
        rel_speed_clamped,
        BALL_CAR_EXTRA_IMPULSE_FACTOR_SPEEDS,
        BALL_CAR_EXTRA_IMPULSE_FACTOR_VALUES
    )
    
    # Extra velocity added to ball
    extra_vel = hit_dir * rel_speed_clamped[..., None] * extra_factor[..., None]
    extra_vel = jnp.where(impulse_mask[..., None], extra_vel, 0.0)
    
    # Apply impulses to velocities
    # Ball receives: +impulse/m_ball + extra_vel
    # Car receives: -impulse/m_car (Newton's 3rd law)
    ball_vel_delta_physics = jnp.sum(impulse / BALL_MASS, axis=1)
    ball_vel_delta_extra = jnp.sum(extra_vel, axis=1)
    new_ball_vel = ball_vel + ball_vel_delta_physics + ball_vel_delta_extra
    
    # Car receives opposite impulse (but NOT the extra_vel - that's game magic for ball only)
    car_vel_delta = -impulse / CAR_MASS  # Correct: ball pushes car back
    new_car_vel = car_vel + car_vel_delta
    
    # Car angular velocity change from impulse at contact point
    # torque = r x F, ang_accel = torque / I
    # For impulse: delta_ang_vel = (r x J) / I
    car_impulse = -impulse  # Impulse ON the car
    torque_on_car = jnp.cross(car_contact_offset, car_impulse)
    
    # Use proper inertia tensor (diagonal approximation)
    from sim_constants import CAR_INERTIA
    inertia_avg = jnp.mean(CAR_INERTIA)
    ang_vel_delta = torque_on_car / inertia_avg
    new_car_ang_vel = car_ang_vel + ang_vel_delta
    
    # Ball spin from friction
    ball_contact_offset = -world_normal * ball_radius
    ball_ang_vel_exp = ball_ang_vel[:, None, :]
    ball_surface_vel = ball_vel_exp + jnp.cross(ball_ang_vel_exp, ball_contact_offset)
    
    rel_surface_vel = ball_surface_vel - car_vel_at_contact
    rel_surface_vel_normal = jnp.sum(rel_surface_vel * world_normal, axis=-1, keepdims=True)
    tangential_slip_vel = rel_surface_vel - rel_surface_vel_normal * world_normal
    
    friction_coef = CARBALL_COLLISION_FRICTION
    tangential_speed = jnp.linalg.norm(tangential_slip_vel, axis=-1, keepdims=True)
    tangential_dir = tangential_slip_vel / (tangential_speed + 1e-8)
    
    friction_impulse_mag = jnp.abs(impulse_mag) * friction_coef * 0.1
    friction_impulse_mag = jnp.minimum(friction_impulse_mag, tangential_speed[..., 0] * BALL_MASS)
    friction_force = -tangential_dir * friction_impulse_mag[..., None]
    friction_force = jnp.where(impulse_mask[..., None], friction_force, 0.0)
    
    ball_friction_torque = jnp.cross(ball_contact_offset, friction_force)
    total_ball_torque = jnp.sum(ball_friction_torque, axis=1)
    
    ball_inertia = (2.0 / 5.0) * BALL_MASS * (ball_radius ** 2)
    ball_ang_accel = total_ball_torque / ball_inertia
    new_ball_ang_vel = ball_ang_vel + ball_ang_accel
    
    # Clamp ball angular velocity
    ball_ang_speed = jnp.linalg.norm(new_ball_ang_vel, axis=-1, keepdims=True)
    new_ball_ang_vel = jnp.where(
        ball_ang_speed > BALL_MAX_ANG_SPEED,
        new_ball_ang_vel * (BALL_MAX_ANG_SPEED / (ball_ang_speed + 1e-8)),
        new_ball_ang_vel
    )
    
    return new_ball_vel, new_ball_ang_vel, new_car_vel, new_car_ang_vel, is_colliding


def resolve_car_car_collision(
    car_pos: jnp.ndarray,
    car_vel: jnp.ndarray,
    car_ang_vel: jnp.ndarray,
    car_quat: jnp.ndarray,
    car_is_on_ground: jnp.ndarray,
    car_is_supersonic: jnp.ndarray,
    hitbox_half_size: jnp.ndarray = OCTANE_HITBOX_SIZE / 2,
    hitbox_offset: jnp.ndarray = OCTANE_HITBOX_OFFSET,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Resolve collisions between cars (demos and bumps).
    
    Uses simplified OBB-OBB collision with sphere approximation for efficiency.
    
    Args:
        car_pos: Car positions. Shape: (N, MAX_CARS, 3)
        car_vel: Car velocities. Shape: (N, MAX_CARS, 3)
        car_ang_vel: Car angular velocities. Shape: (N, MAX_CARS, 3)
        car_quat: Car quaternions. Shape: (N, MAX_CARS, 4)
        car_is_on_ground: Whether each car is grounded. Shape: (N, MAX_CARS)
        car_is_supersonic: Whether each car is supersonic. Shape: (N, MAX_CARS)
        hitbox_half_size: Half-extents of car hitbox
        hitbox_offset: Offset of hitbox center from car origin
        
    Returns:
        Tuple of (new_car_vel, new_car_ang_vel, is_demoed_mask)
    """
    n_envs = car_pos.shape[0]
    max_cars = car_pos.shape[1]
    
    # Bounding sphere radius
    bounding_radius = jnp.sqrt(jnp.sum(hitbox_half_size ** 2)) + 10.0
    
    # Compute hitbox centers in world space
    hitbox_center_offset = quat_rotate_vector(car_quat, hitbox_offset)
    hitbox_center = car_pos + hitbox_center_offset
    
    # Compute pairwise distances
    center_i = hitbox_center[:, :, None, :]
    center_j = hitbox_center[:, None, :, :]
    diff = center_i - center_j
    dist_sq = jnp.sum(diff ** 2, axis=-1)
    dist = jnp.sqrt(dist_sq + 1e-8)
    
    # Broad phase - sphere collision
    collision_dist = 2 * bounding_radius
    potentially_colliding = dist < collision_dist
    
    # Mask out self-collision
    identity_mask = jnp.eye(max_cars, dtype=jnp.bool_)[None, :, :]
    potentially_colliding = potentially_colliding & ~identity_mask
    
    # Only process upper triangle
    upper_tri_mask = jnp.triu(jnp.ones((max_cars, max_cars), dtype=jnp.bool_), k=1)[None, :, :]
    is_valid_pair = potentially_colliding & upper_tri_mask
    
    # Narrow phase
    collision_normal = diff / (dist[..., None] + 1e-8)
    proj_half_size = jnp.sum(jnp.abs(collision_normal) * hitbox_half_size, axis=-1)
    penetration = 2 * proj_half_size - dist
    is_colliding = (penetration > 0) & is_valid_pair
    
    # Determine bumper/bumped
    vel_i = car_vel[:, :, None, :]
    vel_j = car_vel[:, None, :, :]
    
    forward_local = jnp.array([1.0, 0.0, 0.0])
    car_forward = quat_rotate_vector(car_quat, forward_local)
    
    forward_speed = jnp.sum(car_vel * car_forward, axis=-1)
    forward_speed_i = forward_speed[:, :, None]
    forward_speed_j = forward_speed[:, None, :]
    
    i_is_bumper = forward_speed_i > forward_speed_j
    
    # Demo detection
    is_supersonic_i = car_is_supersonic[:, :, None]
    is_supersonic_j = car_is_supersonic[:, None, :]
    bumper_is_supersonic = jnp.where(i_is_bumper, is_supersonic_i, is_supersonic_j)
    
    hit_dir = jnp.where(i_is_bumper[..., None], -collision_normal, collision_normal)
    car_forward_i = car_forward[:, :, None, :]
    car_forward_j = car_forward[:, None, :, :]
    bumper_forward = jnp.where(i_is_bumper[..., None], car_forward_i, car_forward_j)
    
    impact_angle = jnp.sum(bumper_forward * hit_dir, axis=-1)
    is_front_hit = impact_angle > 0.707
    
    is_demo = is_colliding & bumper_is_supersonic & is_front_hit
    
    i_is_victim = ~i_is_bumper
    is_demoed_i_by_j = is_demo & i_is_victim
    j_is_victim = i_is_bumper
    is_demoed_j_by_i = is_demo & j_is_victim
    
    demoed_by_higher_index = jnp.any(is_demoed_i_by_j, axis=2)
    demoed_by_lower_index = jnp.any(is_demoed_j_by_i, axis=1)
    is_demoed_mask = demoed_by_higher_index | demoed_by_lower_index
    
    # Calculate bump impulse
    is_on_ground_i = car_is_on_ground[:, :, None]
    is_on_ground_j = car_is_on_ground[:, None, :]
    
    bumper_speed = jnp.where(i_is_bumper, forward_speed_i, forward_speed_j)
    bumper_speed = jnp.abs(bumper_speed)
    target_grounded = jnp.where(i_is_bumper, is_on_ground_j, is_on_ground_i)
    
    bump_vel_ground = jnp.interp(bumper_speed, BUMP_VEL_AMOUNT_GROUND_SPEEDS, BUMP_VEL_AMOUNT_GROUND_VALUES)
    bump_vel_air = jnp.interp(bumper_speed, BUMP_VEL_AMOUNT_AIR_SPEEDS, BUMP_VEL_AMOUNT_AIR_VALUES)
    bump_upward = jnp.interp(bumper_speed, BUMP_UPWARD_VEL_AMOUNT_SPEEDS, BUMP_UPWARD_VEL_AMOUNT_VALUES)
    
    bump_vel_magnitude = jnp.where(target_grounded, bump_vel_ground, bump_vel_air)
    
    # Calculate impulse direction
    bump_dir = jnp.where(i_is_bumper[..., None], collision_normal, -collision_normal)
    bump_dir_xy = bump_dir.at[..., 2].set(0.0)
    bump_dir_xy = bump_dir_xy / (jnp.linalg.norm(bump_dir_xy, axis=-1, keepdims=True) + 1e-8)
    
    bump_impulse = bump_dir_xy * bump_vel_magnitude[..., None] + jnp.array([0.0, 0.0, 1.0]) * bump_upward[..., None]
    bump_impulse = jnp.where(is_colliding[..., None] & ~is_demo[..., None], bump_impulse, 0.0)
    
    recoil_factor = 0.3
    impulse_on_i_from_pair = jnp.where(i_is_bumper[..., None], -bump_impulse * recoil_factor, bump_impulse)
    
    total_impulse_i = jnp.sum(impulse_on_i_from_pair, axis=2)
    impulse_on_j_from_pair = jnp.where(i_is_bumper[..., None], bump_impulse, -bump_impulse * recoil_factor)
    total_impulse_j = jnp.sum(impulse_on_j_from_pair, axis=1)
    
    total_impulse = total_impulse_i + total_impulse_j
    new_vel = car_vel + total_impulse
    
    # Push cars apart
    separation_strength = 0.5
    separation = collision_normal * penetration[..., None] * separation_strength
    separation = jnp.where(is_colliding[..., None], separation, 0.0)
    
    push_i = jnp.sum(separation, axis=2)
    push_j = jnp.sum(-separation, axis=1)
    new_vel = new_vel + push_i + push_j
    
    new_ang_vel = car_ang_vel
    
    return new_vel, new_ang_vel, is_demoed_mask
