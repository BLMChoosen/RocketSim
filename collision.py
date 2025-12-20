"""
Arena SDFs and Collision Resolution
====================================
Signed Distance Field for arena geometry and collision resolution functions.
Uses branchless jnp.where() for GPU efficiency.
"""

from __future__ import annotations
import jax.numpy as jnp

from .constants import (
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
)
from .math_utils import quat_rotate_vector


def arena_sdf(pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    RAMP_INSET = 820.0
    CORNER_INSET = 1152.0
    SQRT2 = jnp.sqrt(2.0)
    
    # Ceiling Ramps (45 deg)
    ramp_c_r = (dist_right + dist_ceiling - RAMP_INSET) / SQRT2
    ramp_c_l = (dist_left + dist_ceiling - RAMP_INSET) / SQRT2
    ramp_c_b = (dist_back + dist_ceiling - RAMP_INSET) / SQRT2
    ramp_c_f = (dist_front + dist_ceiling - RAMP_INSET) / SQRT2
    
    # Vertical Corners (45 deg)
    ramp_w_rb = (dist_right + dist_back - CORNER_INSET) / SQRT2
    ramp_w_rf = (dist_right + dist_front - CORNER_INSET) / SQRT2
    ramp_w_lb = (dist_left + dist_back - CORNER_INSET) / SQRT2
    ramp_w_lf = (dist_left + dist_front - CORNER_INSET) / SQRT2
    
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
    
    # Ramp normals
    n_c_r = (n_right + n_ceil) / SQRT2
    n_c_l = (n_left + n_ceil) / SQRT2
    n_c_b = (n_back + n_ceil) / SQRT2
    n_c_f = (n_front + n_ceil) / SQRT2
    
    n_w_rb = (n_right + n_back) / SQRT2
    n_w_rf = (n_right + n_front) / SQRT2
    n_w_lb = (n_left + n_back) / SQRT2
    n_w_lf = (n_left + n_front) / SQRT2
    
    normals = jnp.stack([
        n_floor, n_ceil,
        n_left, n_right,
        n_back, n_front,
        n_c_r, n_c_l, n_c_b, n_c_f,
        n_w_rb, n_w_rf, n_w_lb, n_w_lf
    ], axis=0)
    
    # Find closest surface
    min_idx = jnp.argmin(distances, axis=-1)
    min_dist = jnp.min(distances, axis=-1)
    
    # Get normal
    normal = normals[min_idx]
    
    return min_dist, normal


def resolve_ball_arena_collision(
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    ang_vel: jnp.ndarray,
    radius: float = BALL_RADIUS,
    restitution: float = BALL_WALL_RESTITUTION,
    friction: float = BALL_SURFACE_FRICTION
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Resolve ball collisions with arena boundaries.
    
    The arena is an axis-aligned bounding box (AABB):
    - X: [-ARENA_EXTENT_X, +ARENA_EXTENT_X]
    - Y: [-ARENA_EXTENT_Y, +ARENA_EXTENT_Y]  
    - Z: [0, ARENA_HEIGHT]
    
    Uses branchless jnp.where() for GPU efficiency.
    
    Args:
        pos: Ball position. Shape: (N, 3) or (3,)
        vel: Ball velocity. Shape: (N, 3) or (3,)
        ang_vel: Ball angular velocity. Shape: (N, 3) or (3,)
        radius: Ball collision radius
        restitution: Coefficient of restitution (0=sticky, 1=perfect bounce)
        friction: Tangential velocity retention (1=no friction, 0=full stop)
        
    Returns:
        Tuple of (new_pos, new_vel, new_ang_vel)
    """
    # Arena bounds (accounting for ball radius)
    min_x = -ARENA_EXTENT_X + radius
    max_x = ARENA_EXTENT_X - radius
    min_y = -ARENA_EXTENT_Y + radius
    max_y = ARENA_EXTENT_Y - radius
    min_z = radius  # Floor
    max_z = ARENA_HEIGHT - radius  # Ceiling
    
    # Extract components
    px, py, pz = pos[..., 0], pos[..., 1], pos[..., 2]
    vx, vy, vz = vel[..., 0], vel[..., 1], vel[..., 2]
    
    # Friction factor
    friction_factor = 1.0 - friction
    
    # X-AXIS WALLS
    hit_left = px < min_x
    px = jnp.where(hit_left, min_x, px)
    vx = jnp.where(hit_left & (vx < 0), -vx * restitution, vx)
    vy = jnp.where(hit_left, vy * friction_factor, vy)
    vz = jnp.where(hit_left, vz * friction_factor, vz)
    
    hit_right = px > max_x
    px = jnp.where(hit_right, max_x, px)
    vx = jnp.where(hit_right & (vx > 0), -vx * restitution, vx)
    vy = jnp.where(hit_right, vy * friction_factor, vy)
    vz = jnp.where(hit_right, vz * friction_factor, vz)
    
    # Y-AXIS WALLS
    hit_back = py < min_y
    py = jnp.where(hit_back, min_y, py)
    vy = jnp.where(hit_back & (vy < 0), -vy * restitution, vy)
    vx = jnp.where(hit_back, vx * friction_factor, vx)
    vz = jnp.where(hit_back, vz * friction_factor, vz)
    
    hit_front = py > max_y
    py = jnp.where(hit_front, max_y, py)
    vy = jnp.where(hit_front & (vy > 0), -vy * restitution, vy)
    vx = jnp.where(hit_front, vx * friction_factor, vx)
    vz = jnp.where(hit_front, vz * friction_factor, vz)
    
    # Z-AXIS (FLOOR AND CEILING)
    hit_floor = pz < min_z
    pz = jnp.where(hit_floor, min_z, pz)
    vz = jnp.where(hit_floor & (vz < 0), -vz * restitution, vz)
    vx = jnp.where(hit_floor, vx * friction_factor, vx)
    vy = jnp.where(hit_floor, vy * friction_factor, vy)
    
    hit_ceiling = pz > max_z
    pz = jnp.where(hit_ceiling, max_z, pz)
    vz = jnp.where(hit_ceiling & (vz > 0), -vz * restitution, vz)
    vx = jnp.where(hit_ceiling, vx * friction_factor, vx)
    vy = jnp.where(hit_ceiling, vy * friction_factor, vy)
    
    # Reconstruct vectors
    new_pos = jnp.stack([px, py, pz], axis=-1)
    new_vel = jnp.stack([vx, vy, vz], axis=-1)
    
    # Angular velocity is preserved
    new_ang_vel = ang_vel
    
    return new_pos, new_vel, new_ang_vel


def resolve_car_arena_collision(
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    margin: float = 50.0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Keep car center-of-mass inside arena bounds.
    
    This is a simple positional clamp with velocity kill.
    
    Args:
        pos: Car position. Shape: (N, MAX_CARS, 3)
        vel: Car velocity. Shape: (N, MAX_CARS, 3)
        margin: Distance from wall to keep car CoM
        
    Returns:
        Tuple of (new_pos, new_vel)
    """
    # Arena bounds with margin
    min_x = -ARENA_EXTENT_X + margin
    max_x = ARENA_EXTENT_X - margin
    min_y = -ARENA_EXTENT_Y + margin
    max_y = ARENA_EXTENT_Y - margin
    min_z = 0.0
    max_z = ARENA_HEIGHT - margin
    
    # Extract components
    px, py, pz = pos[..., 0], pos[..., 1], pos[..., 2]
    vx, vy, vz = vel[..., 0], vel[..., 1], vel[..., 2]
    
    # X-AXIS WALLS
    hit_left = px < min_x
    px = jnp.where(hit_left, min_x, px)
    vx = jnp.where(hit_left & (vx < 0), 0.0, vx)
    
    hit_right = px > max_x
    px = jnp.where(hit_right, max_x, px)
    vx = jnp.where(hit_right & (vx > 0), 0.0, vx)
    
    # Y-AXIS WALLS
    hit_back = py < min_y
    py = jnp.where(hit_back, min_y, py)
    vy = jnp.where(hit_back & (vy < 0), 0.0, vy)
    
    hit_front = py > max_y
    py = jnp.where(hit_front, max_y, py)
    vy = jnp.where(hit_front & (vy > 0), 0.0, vy)
    
    # Z-AXIS
    hit_floor = pz < min_z
    pz = jnp.where(hit_floor, min_z, pz)
    vz = jnp.where(hit_floor & (vz < 0), 0.0, vz)
    
    hit_ceiling = pz > max_z
    pz = jnp.where(hit_ceiling, max_z, pz)
    vz = jnp.where(hit_ceiling & (vz > 0), 0.0, vz)
    
    # Reconstruct vectors
    new_pos = jnp.stack([px, py, pz], axis=-1)
    new_vel = jnp.stack([vx, vy, vz], axis=-1)
    
    return new_pos, new_vel


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
    
    # Calculate relative velocity at contact point
    contact_offset = -world_normal * (ball_radius - penetration[..., None] / 2)
    car_vel_at_contact = car_vel + jnp.cross(car_ang_vel, contact_offset)
    rel_vel = ball_vel_exp - car_vel_at_contact
    rel_vel_normal = jnp.sum(rel_vel * world_normal, axis=-1)
    
    # Only process if objects are approaching
    approaching = rel_vel_normal < 0
    
    # Calculate collision impulse
    inv_mass_sum = 1.0 / BALL_MASS + 1.0 / CAR_MASS
    restitution = 0.6
    impulse_mag = -(1 + restitution) * rel_vel_normal / inv_mass_sum
    
    impulse_mask = is_colliding & approaching
    impulse_mag = jnp.where(impulse_mask, impulse_mag, 0.0)
    impulse = impulse_mag[..., None] * world_normal
    
    # RL's "Extra Impulse" (power hit mechanic)
    rel_speed = jnp.linalg.norm(rel_vel, axis=-1)
    rel_speed_clamped = jnp.minimum(rel_speed, BALL_CAR_EXTRA_IMPULSE_MAX_DELTA_VEL)
    
    # Get car forward direction
    car_forward = quat_rotate_vector(car_quat, jnp.array([1.0, 0.0, 0.0]))
    
    # Hit direction
    hit_dir_raw = rel_pos_world * jnp.array([1.0, 1.0, BALL_CAR_EXTRA_IMPULSE_Z_SCALE])
    hit_dir = hit_dir_raw / (jnp.linalg.norm(hit_dir_raw, axis=-1, keepdims=True) + 1e-8)
    
    # Reduce forward component
    forward_component = jnp.sum(hit_dir * car_forward, axis=-1, keepdims=True)
    forward_adjustment = car_forward * forward_component * (1 - BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE)
    hit_dir = hit_dir - forward_adjustment
    hit_dir = hit_dir / (jnp.linalg.norm(hit_dir, axis=-1, keepdims=True) + 1e-8)
    
    # Interpolate extra impulse factor
    extra_factor = jnp.interp(
        rel_speed_clamped,
        BALL_CAR_EXTRA_IMPULSE_FACTOR_SPEEDS,
        BALL_CAR_EXTRA_IMPULSE_FACTOR_VALUES
    )
    
    extra_vel = hit_dir * rel_speed_clamped[..., None] * extra_factor[..., None]
    extra_vel = jnp.where(impulse_mask[..., None], extra_vel, 0.0)
    
    # Apply impulses to velocities
    ball_vel_delta_physics = jnp.sum(impulse / BALL_MASS, axis=1)
    ball_vel_delta_extra = jnp.sum(extra_vel, axis=1)
    new_ball_vel = ball_vel + ball_vel_delta_physics + ball_vel_delta_extra
    
    car_vel_delta = -impulse / CAR_MASS
    new_car_vel = car_vel + car_vel_delta
    
    # Car angular velocity change
    torque = jnp.cross(contact_offset, -impulse)
    inertia_approx = 1000.0
    ang_vel_delta = torque / inertia_approx
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
