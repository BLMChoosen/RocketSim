"""
Suspension and Tire Physics
============================
Wheel positions, suspension raycasting, tire forces, and steering.
Matches C++ btVehicleRL behavior.
"""

from __future__ import annotations
import jax.numpy as jnp

from ..constants import (
    DT, CAR_MASS, CAR_INERTIA,
    WHEEL_LOCAL_OFFSETS, WHEEL_RADII, EFFECTIVE_REST_LENGTHS,
    SUSPENSION_STIFFNESS, SUSPENSION_FORCE_SCALES, MAX_SUSPENSION_TRAVEL,
    GROUND_Z, SUSPENSION_SUBTRACTION,
    WHEELS_DAMPING_COMPRESSION, WHEELS_DAMPING_RELAXATION,
    LATERAL_FRICTION_BASE, LATERAL_FRICTION_MIN, FRICTION_FORCE_SCALE,
    HANDBRAKE_LAT_FRICTION_FACTOR, HANDBRAKE_LONG_FRICTION_FACTOR,
    TIRE_DRIVE_FORCE, BRAKE_FORCE,
    STEER_ANGLE_CURVE_SPEEDS, STEER_ANGLE_CURVE_ANGLES,
    DRIVE_TORQUE_CURVE_SPEEDS, DRIVE_TORQUE_CURVE_FACTORS,
    FRONT_WHEEL_MASK, THROTTLE_TORQUE_AMOUNT, BRAKE_TORQUE_AMOUNT,
    UU_TO_BT, BT_TO_UU,
    NON_STICKY_FRICTION_CURVE_X, NON_STICKY_FRICTION_CURVE_Y,
    POWERSLIDE_STEER_CURVE_SPEEDS, POWERSLIDE_STEER_CURVE_ANGLES,
)
from ..types import CarState, CarControls
from ..math_utils import quat_rotate_vector, get_car_up_dir, get_car_forward_dir, get_car_right_dir
from ..collision.arena_sdf import arena_sdf


# =============================================================================
# WHEEL POSITION AND VELOCITY
# =============================================================================

def compute_wheel_world_positions(car_pos: jnp.ndarray, car_quat: jnp.ndarray) -> jnp.ndarray:
    """Compute world positions of all 4 wheel hardpoints. Returns (N, MAX_CARS, 4, 3)."""
    car_pos_expanded = car_pos[..., None, :]
    car_quat_expanded = car_quat[..., None, :]
    local_offsets = WHEEL_LOCAL_OFFSETS[None, None, :, :]
    world_offsets = quat_rotate_vector(car_quat_expanded, local_offsets)
    return car_pos_expanded + world_offsets


def compute_wheel_velocities(car_vel: jnp.ndarray, car_ang_vel: jnp.ndarray, car_quat: jnp.ndarray) -> jnp.ndarray:
    """Compute world-space velocities of wheel contact points. Returns (N, MAX_CARS, 4, 3)."""
    car_quat_expanded = car_quat[..., None, :]
    local_offsets = WHEEL_LOCAL_OFFSETS[None, None, :, :]
    world_offsets = quat_rotate_vector(car_quat_expanded, local_offsets)
    car_vel_expanded = car_vel[..., None, :]
    car_ang_vel_expanded = car_ang_vel[..., None, :]
    omega_cross_r = jnp.cross(car_ang_vel_expanded, world_offsets)
    return car_vel_expanded + omega_cross_r


# =============================================================================
# SUSPENSION PHYSICS
# =============================================================================

def raycast_suspension(
    wheel_world_pos: jnp.ndarray,
    car_quat: jnp.ndarray,
    wheel_radii: jnp.ndarray = WHEEL_RADII,
    sus_rest: jnp.ndarray = EFFECTIVE_REST_LENGTHS,
    max_travel: float = MAX_SUSPENSION_TRAVEL,
    ground_z: float = GROUND_Z,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Raycast from wheel hardpoints DOWN along car's local -Z axis.
    Uses arena SDF for collision detection.
    Returns (compression, is_contact, contact_normal, contact_surface_pos).
    """
    car_up = get_car_up_dir(car_quat)
    car_down = -car_up
    
    car_down_expanded = car_down[..., None, :]
    car_down_expanded = jnp.broadcast_to(car_down_expanded, wheel_world_pos.shape)
    
    radii = wheel_radii[None, None, :]
    rest = sus_rest[None, None, :]
    ray_length = rest + max_travel + radii - SUSPENSION_SUBTRACTION
    
    n_samples = 8
    sample_fractions = jnp.linspace(0.0, 1.0, n_samples)
    ray_end = wheel_world_pos + car_down_expanded * ray_length[..., None]
    
    wheel_pos_exp = wheel_world_pos[..., None, :]
    ray_end_exp = ray_end[..., None, :]
    fracs = sample_fractions[None, None, None, :, None]
    sample_positions = wheel_pos_exp + (ray_end_exp - wheel_pos_exp) * fracs
    
    orig_shape = sample_positions.shape
    n_envs = orig_shape[0]
    max_cars = orig_shape[1]
    
    sample_flat = sample_positions.reshape(-1, 3)
    sample_for_sdf = sample_flat[None, :, :]
    
    dist_flat, normal_flat = arena_sdf(sample_for_sdf)
    dist_flat = dist_flat[0]
    normal_flat = normal_flat[0]
    
    sdf_dist = dist_flat.reshape(n_envs, max_cars, 4, n_samples)
    sdf_normal = normal_flat.reshape(n_envs, max_cars, 4, n_samples, 3)
    
    radii_exp = radii[..., None]
    is_penetrating = sdf_dist < radii_exp
    
    sample_idx = jnp.arange(n_samples)[None, None, None, :]
    masked_idx = jnp.where(is_penetrating, sample_idx, n_samples * 2)
    first_contact_idx = jnp.argmin(masked_idx, axis=-1)
    any_contact = jnp.any(is_penetrating, axis=-1)
    
    batch_idx = jnp.arange(n_envs)[:, None, None]
    car_idx_arr = jnp.arange(max_cars)[None, :, None]
    wheel_idx_arr = jnp.arange(4)[None, None, :]
    contact_sdf = sdf_dist[batch_idx, car_idx_arr, wheel_idx_arr, first_contact_idx]
    contact_normal = sdf_normal[batch_idx, car_idx_arr, wheel_idx_arr, first_contact_idx]
    
    contact_fraction = sample_fractions[first_contact_idx]
    contact_fraction = jnp.where(any_contact, contact_fraction, 1.0)
    
    car_up_expanded_4 = car_up[..., None, :]
    car_up_expanded_4 = jnp.broadcast_to(car_up_expanded_4, contact_normal.shape)
    normal_dot_up = jnp.sum(contact_normal * car_up_expanded_4, axis=-1)
    normal_dot_up_safe = jnp.maximum(normal_dot_up, 0.1)
    
    wheel_trace_len = contact_fraction * ray_length[0, 0, :] + contact_sdf / normal_dot_up_safe
    sus_length = wheel_trace_len - radii[0, 0, :]
    
    min_sus = rest[0, 0, :] - max_travel
    max_sus = rest[0, 0, :] + max_travel
    sus_length = jnp.clip(sus_length, min_sus, max_sus)
    
    compression = rest[0, 0, :] - sus_length
    compression = jnp.where(any_contact, compression, 0.0)
    
    car_down_4 = car_down[..., None, :]
    car_down_4 = jnp.broadcast_to(car_down_4, wheel_world_pos.shape)
    contact_surface_pos = wheel_world_pos + car_down_4 * wheel_trace_len[..., None]
    
    car_up_fallback = car_up[..., None, :]
    car_up_fallback = jnp.broadcast_to(car_up_fallback, contact_normal.shape)
    contact_normal = jnp.where(any_contact[..., None], contact_normal, car_up_fallback)
    
    return compression, any_contact, contact_normal, contact_surface_pos


def compute_suspension_force(
    compression: jnp.ndarray,
    compression_vel: jnp.ndarray,
    is_contact: jnp.ndarray,
    inv_contact_dot: jnp.ndarray,
    stiffness: float = SUSPENSION_STIFFNESS,
    damping_comp: float = WHEELS_DAMPING_COMPRESSION,
    damping_relax: float = WHEELS_DAMPING_RELAXATION,
    force_scales: jnp.ndarray = SUSPENSION_FORCE_SCALES,
) -> jnp.ndarray:
    """Compute suspension force using spring-damper model. Returns (N, MAX_CARS, 4)."""
    spring_force = stiffness * compression * inv_contact_dot
    effective_vel = compression_vel * inv_contact_dot
    
    damping = jnp.where(effective_vel > 0, damping_comp, damping_relax)
    damper_force = -damping * effective_vel
    
    force_scales_expanded = force_scales[None, None, :]
    total_force = (spring_force + damper_force) * force_scales_expanded
    total_force = jnp.maximum(total_force, 0.0)
    total_force = jnp.where(is_contact, total_force, 0.0)
    return total_force


# =============================================================================
# TIRE PHYSICS
# =============================================================================

def compute_steering_angle(
    steer_input: jnp.ndarray,
    forward_speed: jnp.ndarray,
    handbrake_val: jnp.ndarray = None,
) -> jnp.ndarray:
    """Compute steering angle based on input, speed, and powerslide."""
    abs_speed = jnp.abs(forward_speed)
    base_angle = jnp.interp(abs_speed, STEER_ANGLE_CURVE_SPEEDS, STEER_ANGLE_CURVE_ANGLES)
    
    if handbrake_val is not None:
        powerslide_angle = jnp.interp(abs_speed, POWERSLIDE_STEER_CURVE_SPEEDS, POWERSLIDE_STEER_CURVE_ANGLES)
        base_angle = base_angle + (powerslide_angle - base_angle) * handbrake_val
    
    return steer_input * base_angle


def compute_tire_forces(
    car_quat: jnp.ndarray,
    car_vel: jnp.ndarray,
    car_ang_vel: jnp.ndarray,
    car_pos: jnp.ndarray,
    wheel_world_pos: jnp.ndarray,
    throttle: jnp.ndarray,
    steer: jnp.ndarray,
    handbrake_val: jnp.ndarray,
    handbrake_button: jnp.ndarray,
    is_contact: jnp.ndarray,
    contact_normal: jnp.ndarray,
    forward_speed: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute tire forces matching C++ btVehicleRL logic.
    Returns (impulses, wheel_rel_pos) for per-wheel forces.
    """
    friction_scale = CAR_MASS / 3.0
    steer_angle = compute_steering_angle(steer, forward_speed, handbrake_val)
    
    # Get car basis vectors
    forward_local = jnp.array([1.0, 0.0, 0.0])
    right_local = jnp.array([0.0, -1.0, 0.0])
    up_local = jnp.array([0.0, 0.0, 1.0])
    
    car_forward = quat_rotate_vector(car_quat, forward_local)
    car_right = quat_rotate_vector(car_quat, right_local)
    car_up = quat_rotate_vector(car_quat, up_local)
    car_left = -car_right
    
    # Apply steering to front wheels
    cos_steer = jnp.cos(steer_angle)[..., None]
    sin_steer = jnp.sin(steer_angle)[..., None]
    steered_left = -car_forward * sin_steer + car_left * cos_steer
    
    car_left_4 = car_left[..., None, :]
    steered_left_4 = steered_left[..., None, :]
    front_mask = FRONT_WHEEL_MASK[None, None, :, None]
    
    axle_dir = jnp.where(
        front_mask > 0.5,
        jnp.broadcast_to(steered_left_4, car_left_4.shape[:-2] + (4, 3)),
        jnp.broadcast_to(car_left_4, car_left_4.shape[:-2] + (4, 3))
    )
    
    # Project axle onto surface plane
    proj = jnp.sum(axle_dir * contact_normal, axis=-1, keepdims=True)
    axle_dir = axle_dir - contact_normal * proj
    axle_dir = axle_dir / jnp.maximum(jnp.linalg.norm(axle_dir, axis=-1, keepdims=True), 1e-8)
    
    forward_dir = jnp.cross(contact_normal, axle_dir)
    forward_dir = forward_dir / jnp.maximum(jnp.linalg.norm(forward_dir, axis=-1, keepdims=True), 1e-8)
    
    # Wheel velocities at contact point
    wheel_delta = wheel_world_pos - car_pos[..., None, :]
    car_ang_vel_4 = car_ang_vel[..., None, :]
    car_vel_4 = car_vel[..., None, :]
    cross_vec = jnp.cross(car_ang_vel_4, wheel_delta) + car_vel_4
    
    # Lateral and longitudinal velocities
    vel_lateral = jnp.sum(cross_vec * axle_dir, axis=-1)
    side_impulse = -vel_lateral
    vel_forward = jnp.sum(cross_vec * forward_dir, axis=-1)
    
    # Throttle/brake logic
    throttle_4 = throttle[..., None]
    forward_speed_4 = forward_speed[..., None]
    abs_forward_speed = jnp.abs(forward_speed_4)
    
    drive_speed_scale = jnp.interp(abs_forward_speed, DRIVE_TORQUE_CURVE_SPEEDS, DRIVE_TORQUE_CURVE_FACTORS)
    num_contacts = jnp.sum(is_contact.astype(jnp.float32), axis=-1, keepdims=True)
    drive_speed_scale = jnp.where(num_contacts < 3, drive_speed_scale / 4.0, drive_speed_scale)
    
    engine_throttle = throttle_4
    handbrake_button_4 = handbrake_button[..., None].astype(jnp.float32)
    is_handbraking = handbrake_button_4 > 0.5
    
    abs_throttle = jnp.abs(throttle_4)
    is_reversing = (abs_forward_speed > 25.0) & (jnp.sign(throttle_4) != jnp.sign(forward_speed_4)) & (abs_throttle > 0.001) & ~is_handbraking
    is_coasting = (abs_throttle < 0.001) & ~is_handbraking
    
    engine_throttle = jnp.where(is_reversing & (abs_forward_speed > 0.01), 0.0, engine_throttle)
    engine_throttle = jnp.where(is_coasting, 0.0, engine_throttle)
    
    brake_input = jnp.where(is_reversing, 1.0, 0.0)
    brake_input = jnp.where(is_coasting, jnp.where(abs_forward_speed < 25.0, 1.0, 0.15), brake_input)
    
    ROLLING_FRICTION_SCALE_MAGIC = 113.73963
    
    drive_engine_force = engine_throttle * (THROTTLE_TORQUE_AMOUNT * UU_TO_BT) * drive_speed_scale
    drive_brake_force = brake_input * (BRAKE_TORQUE_AMOUNT * UU_TO_BT)
    
    has_engine = jnp.abs(drive_engine_force) > 0.001
    
    vel_to_friction = CAR_MASS / jnp.maximum(friction_scale * DT * BT_TO_UU, 1e-8)
    max_stop_friction = jnp.abs(vel_forward) * vel_to_friction / 4.0
    effective_brake = jnp.minimum(drive_brake_force, max_stop_friction)
    rolling_friction_brake = jnp.clip(-vel_forward * ROLLING_FRICTION_SCALE_MAGIC, -effective_brake, effective_brake)
    rolling_friction_brake = jnp.where(drive_brake_force > 0, rolling_friction_brake, 0.0)
    
    rolling_friction_engine = -drive_engine_force / friction_scale
    rolling_friction = jnp.where(has_engine, rolling_friction_engine, rolling_friction_brake)
    
    # Friction curves
    base_friction = jnp.abs(vel_lateral)
    friction_curve_input = jnp.where(
        base_friction > 5,
        base_friction / (jnp.abs(vel_forward) + base_friction),
        0.0
    )
    
    lat_friction = 1.0 - 0.8 * friction_curve_input
    long_friction = jnp.ones_like(friction_curve_input)
    
    # Handbrake adjustments
    handbrake_4 = handbrake_val[..., None]
    lat_friction = lat_friction * ((HANDBRAKE_LAT_FRICTION_FACTOR - 1.0) * handbrake_4 + 1.0)
    
    handbrake_long_factor = 0.5 + 0.4 * friction_curve_input
    long_friction = long_friction * ((handbrake_long_factor - 1.0) * handbrake_4 + 1.0)
    long_friction = jnp.where(handbrake_4 > 0.001, long_friction, jnp.ones_like(long_friction))
    
    # Non-sticky friction
    is_contact_sticky = jnp.abs(throttle_4) > 0.001
    contact_normal_z = contact_normal[..., 2]
    non_sticky_scale = jnp.interp(contact_normal_z, NON_STICKY_FRICTION_CURVE_X, NON_STICKY_FRICTION_CURVE_Y)
    
    lat_friction = jnp.where(is_contact_sticky, lat_friction, lat_friction * non_sticky_scale)
    long_friction = jnp.where(is_contact_sticky, long_friction, long_friction * non_sticky_scale)
    
    # Final impulse
    total_friction_force = (
        forward_dir * (rolling_friction * long_friction)[..., None] +
        axle_dir * (side_impulse * lat_friction)[..., None]
    )
    
    wheel_impulse = total_friction_force * friction_scale
    is_contact_4 = is_contact[..., None]
    wheel_impulse = jnp.where(is_contact_4, wheel_impulse, 0.0)
    
    car_up_4 = car_up[..., None, :]
    contact_up_dot = jnp.sum(car_up_4 * wheel_delta, axis=-1, keepdims=True)
    wheel_rel_pos = wheel_delta - car_up_4 * contact_up_dot
    
    return wheel_impulse, wheel_rel_pos


# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve_suspension_and_tires(
    cars: CarState,
    controls: CarControls,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Main function to compute all wheel-related forces.
    Returns (sus_force, sus_torque, tire_impulse, wheel_rel_pos, is_contact, num_contacts, upwards_dir).
    """
    wheel_world_pos = compute_wheel_world_positions(cars.pos, cars.quat)
    wheel_vel = compute_wheel_velocities(cars.vel, cars.ang_vel, cars.quat)
    
    compression, is_contact, contact_normal, contact_surface_pos = raycast_suspension(wheel_world_pos, cars.quat)
    
    car_sdf, _ = arena_sdf(cars.pos)
    car_is_inside_arena = car_sdf > -10.0
    
    proj_vel = -jnp.sum(wheel_vel * contact_normal, axis=-1)
    
    car_up = get_car_up_dir(cars.quat)
    car_up_expanded = car_up[..., None, :]
    denominator = jnp.sum(contact_normal * car_up_expanded, axis=-1)
    
    denom_valid = denominator > 0.1
    inv_contact_dot = jnp.where(denom_valid, 1.0 / jnp.maximum(denominator, 0.1), 10.0)
    compression_vel = jnp.where(denom_valid, proj_vel, 0.0)
    
    is_contact_valid = is_contact & car_is_inside_arena[..., None]
    
    suspension_force = compute_suspension_force(compression, compression_vel, is_contact_valid, inv_contact_dot)
    
    forward_local = jnp.array([1.0, 0.0, 0.0])
    car_forward = quat_rotate_vector(cars.quat, forward_local)
    forward_speed = jnp.sum(cars.vel * car_forward, axis=-1)
    
    is_boosting = controls.boost & (cars.boost_amount > 0)
    real_throttle = jnp.where(is_boosting, 1.0, controls.throttle)
    
    tire_impulse, wheel_rel_pos = compute_tire_forces(
        car_quat=cars.quat, car_vel=cars.vel, car_ang_vel=cars.ang_vel,
        car_pos=cars.pos, wheel_world_pos=wheel_world_pos,
        throttle=real_throttle, steer=controls.steer,
        handbrake_val=cars.handbrake_val, handbrake_button=controls.handbrake,
        is_contact=is_contact_valid, contact_normal=contact_normal,
        forward_speed=forward_speed,
    )
    
    sus_force_expanded = suspension_force[..., None]
    sus_force_vec = contact_normal * sus_force_expanded * DT
    
    contact_offset = contact_surface_pos - cars.pos[..., None, :]
    total_sus_force = jnp.sum(sus_force_vec, axis=-2)
    
    sus_torque_per_wheel = jnp.cross(contact_offset, sus_force_vec)
    total_sus_torque = jnp.sum(sus_torque_per_wheel, axis=-2)
    
    num_contacts = jnp.sum(is_contact_valid.astype(jnp.float32), axis=-1)
    
    sum_contact_normals = jnp.sum(
        jnp.where(is_contact_valid[..., None], contact_normal, 0.0), axis=-2
    )
    upwards_dir_norm = jnp.linalg.norm(sum_contact_normals, axis=-1, keepdims=True)
    upwards_dir = jnp.where(upwards_dir_norm > 1e-8, sum_contact_normals / upwards_dir_norm, car_up)
    
    return total_sus_force, total_sus_torque, tire_impulse, wheel_rel_pos, is_contact_valid, num_contacts, upwards_dir
