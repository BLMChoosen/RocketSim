"""
Vehicle Physics and Integration
================================
Suspension, tire forces, gravity, drag, and physics integration.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

from sim_constants import (
    DT, GRAVITY_Z, BALL_DRAG, BALL_MAX_SPEED, BALL_MAX_ANG_SPEED,
    CAR_MASS, CAR_MAX_SPEED, CAR_MAX_ANG_SPEED, CAR_INERTIA, CAR_TORQUE_SCALE,
    CAR_AIR_CONTROL_TORQUE, CAR_AIR_CONTROL_DAMPING,
    WHEEL_LOCAL_OFFSETS, WHEEL_RADII, SUSPENSION_REST_LENGTHS,
    SUSPENSION_STIFFNESS, SUSPENSION_FORCE_SCALES, MAX_SUSPENSION_TRAVEL, GROUND_Z,
    WHEELS_DAMPING_COMPRESSION, WHEELS_DAMPING_RELAXATION,
    LATERAL_FRICTION_BASE, LATERAL_FRICTION_MIN, FRICTION_FORCE_SCALE,
    HANDBRAKE_LAT_FRICTION_FACTOR, HANDBRAKE_LONG_FRICTION_FACTOR,
    TIRE_DRIVE_FORCE, BRAKE_FORCE, ROLLING_RESISTANCE,
    STEER_ANGLE_CURVE_SPEEDS, STEER_ANGLE_CURVE_ANGLES,
    DRIVE_TORQUE_CURVE_SPEEDS, DRIVE_TORQUE_CURVE_FACTORS,
    FRONT_WHEEL_MASK, STICKY_FORCE_SCALE_BASE, STOPPING_FORWARD_VEL,
)
from sim_types import BallState, CarState, CarControls
from math_utils import (
    quat_rotate_vector, quat_multiply, quat_normalize, quat_from_angular_velocity,
    get_car_forward_dir, get_car_up_dir, get_car_right_dir,
    clamp_velocity, clamp_angular_velocity,
)
from collision import arena_sdf, resolve_ball_arena_collision, resolve_car_arena_collision


# =============================================================================
# BASIC PHYSICS INTEGRATION
# =============================================================================


def apply_gravity(vel: jnp.ndarray, dt: float = DT) -> jnp.ndarray:
    """
    Apply gravitational acceleration to velocity.
    
    Args:
        vel: Current velocity [..., 3]
        dt: Time step
        
    Returns:
        Updated velocity [..., 3]
    """
    gravity = jnp.array([0.0, 0.0, GRAVITY_Z])
    return vel + gravity * dt


def apply_ball_drag(vel: jnp.ndarray, drag: float = BALL_DRAG, dt: float = DT) -> jnp.ndarray:
    """
    Apply air drag to ball velocity.
    
    From Bullet Physics, linear damping is applied ONCE PER TICK as:
    vel *= clamp(1.0 - damping, 0, 1)
    
    This is NOT scaled by dt - Bullet applies damping per substep.
    
    Args:
        vel: Current velocity [..., 3]
        drag: Linear damping coefficient
        dt: Time step (unused, kept for API compatibility)
        
    Returns:
        Damped velocity [..., 3]
    """
    damping_factor = jnp.clip(1.0 - drag, 0.0, 1.0)
    return vel * damping_factor


def integrate_position(pos: jnp.ndarray, vel: jnp.ndarray, dt: float = DT) -> jnp.ndarray:
    """
    Semi-implicit Euler integration for position.
    
    pos(t+dt) = pos(t) + vel(t+dt) * dt
    
    Args:
        pos: Current position [..., 3]
        vel: Updated velocity [..., 3]
        dt: Time step
        
    Returns:
        New position [..., 3]
    """
    return pos + vel * dt


def integrate_rotation(
    quat: jnp.ndarray, 
    ang_vel: jnp.ndarray, 
    dt: float = DT
) -> jnp.ndarray:
    """
    Integrate rotation quaternion given angular velocity.
    
    q(t+dt) = q(t) * delta_q(ang_vel, dt)
    
    CRITICAL: Normalizes the result to prevent quaternion drift.
    
    Args:
        quat: Current rotation quaternion [..., 4] in [w, x, y, z]
        ang_vel: Angular velocity [..., 3] in rad/s
        dt: Time step
        
    Returns:
        New normalized quaternion [..., 4]
    """
    delta_q = quat_from_angular_velocity(ang_vel, dt)
    new_quat = quat_multiply(quat, delta_q)
    return quat_normalize(new_quat)


# =============================================================================
# WHEEL POSITION AND VELOCITY
# =============================================================================


def compute_wheel_world_positions(
    car_pos: jnp.ndarray,
    car_quat: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute world positions of all 4 wheel hardpoints.
    
    Args:
        car_pos: Car center position (N, MAX_CARS, 3)
        car_quat: Car rotation quaternion (N, MAX_CARS, 4)
        
    Returns:
        Wheel world positions (N, MAX_CARS, 4, 3)
    """
    car_pos_expanded = car_pos[..., None, :]
    car_quat_expanded = car_quat[..., None, :]
    local_offsets = WHEEL_LOCAL_OFFSETS[None, None, :, :]
    
    world_offsets = quat_rotate_vector(car_quat_expanded, local_offsets)
    wheel_world_pos = car_pos_expanded + world_offsets
    
    return wheel_world_pos


def compute_wheel_velocities(
    car_vel: jnp.ndarray,
    car_ang_vel: jnp.ndarray,
    car_quat: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute world-space velocities of wheel contact points.
    
    v_wheel = v_car + omega × r_wheel
    
    Args:
        car_vel: Car linear velocity (N, MAX_CARS, 3)
        car_ang_vel: Car angular velocity (N, MAX_CARS, 3)
        car_quat: Car rotation quaternion (N, MAX_CARS, 4)
        
    Returns:
        Wheel velocities (N, MAX_CARS, 4, 3)
    """
    car_quat_expanded = car_quat[..., None, :]
    local_offsets = WHEEL_LOCAL_OFFSETS[None, None, :, :]
    world_offsets = quat_rotate_vector(car_quat_expanded, local_offsets)
    
    car_vel_expanded = car_vel[..., None, :]
    car_ang_vel_expanded = car_ang_vel[..., None, :]
    
    omega_cross_r = jnp.cross(car_ang_vel_expanded, world_offsets)
    wheel_vel = car_vel_expanded + omega_cross_r
    
    return wheel_vel


# =============================================================================
# SUSPENSION PHYSICS
# =============================================================================


def raycast_suspension(
    wheel_world_pos: jnp.ndarray,
    car_quat: jnp.ndarray,
    wheel_radii: jnp.ndarray = WHEEL_RADII,
    sus_rest: jnp.ndarray = SUSPENSION_REST_LENGTHS,
    max_travel: float = MAX_SUSPENSION_TRAVEL,
    ground_z: float = GROUND_Z,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Raycast from wheel hardpoints DOWN along car's local -Z axis.
    
    The raycast determines how far the wheel can extend before hitting the ground.
    We simulate this by finding the intersection of a ray from the wheel hardpoint
    in the direction of the car's down vector (-Z local).
    
    For a flat ground plane at z=0, this is simply:
        hit_dist = (wheel_z - ground_z) / cos(angle)
    where angle is between ray direction and vertical.
    
    Suspension compression = ray_length - hit_dist
    where ray_length = rest_length + wheel_radius
    
    Args:
        wheel_world_pos: Wheel hardpoint positions (N, MAX_CARS, 4, 3)
        car_quat: Car rotation quaternions (N, MAX_CARS, 4)
        wheel_radii: Wheel radii (4,)
        sus_rest: Suspension rest lengths (4,)
        max_travel: Maximum suspension travel
        ground_z: Ground plane Z coordinate
        
    Returns:
        compression: Suspension compression (N, MAX_CARS, 4)
        is_contact: Boolean contact flags (N, MAX_CARS, 4)
        contact_normal: Normal at contact point (N, MAX_CARS, 4, 3)
    """
    # Get car's down direction (negative Z in local space)
    # We need the up direction first, then negate
    car_up = get_car_up_dir(car_quat)  # (N, MAX_CARS, 3)
    car_down = -car_up  # Ray direction (pointing down from wheel)
    
    # Expand for 4 wheels: (N, MAX_CARS, 4, 3)
    car_down_expanded = car_down[..., None, :]  # (N, MAX_CARS, 1, 3)
    car_down_expanded = jnp.broadcast_to(car_down_expanded, wheel_world_pos.shape)
    
    # Ground plane: z = ground_z, normal = [0, 0, 1]
    # Ray: P = wheel_pos + t * car_down
    # Intersection: P.z = ground_z
    # wheel_pos.z + t * car_down.z = ground_z
    # t = (ground_z - wheel_pos.z) / car_down.z
    
    wheel_z = wheel_world_pos[..., 2]  # (N, MAX_CARS, 4)
    ray_dir_z = car_down_expanded[..., 2]  # (N, MAX_CARS, 4)
    
    # Avoid division by zero when car is upside down
    ray_dir_z_safe = jnp.where(jnp.abs(ray_dir_z) < 1e-6, -1e-6, ray_dir_z)
    
    # t is the distance along the ray to hit ground_z
    t = (ground_z - wheel_z) / ray_dir_z_safe
    
    # t should be positive (ray goes down, hits ground below)
    # If t < 0, the ray points away from the ground (car is upside down or tilted)
    t = jnp.maximum(t, 0.0)
    
    # The "hit distance" is the ray parameter t
    # But we need to account for the wheel touching, not the ray origin
    # When t=0, wheel is at ground level or ray points up
    # hit_dist represents how far from wheel hardpoint to ground surface along ray
    hit_dist = t
    
    # Ray length is rest length + wheel radius
    radii = wheel_radii[None, None, :]  # (1, 1, 4)
    rest = sus_rest[None, None, :]  # (1, 1, 4)
    ray_length = rest + radii
    
    # Compression: how much the suspension is compressed from rest
    # If hit_dist < ray_length, wheel is touching and suspension compresses
    # If hit_dist >= ray_length, wheel is in air, no compression
    compression = ray_length - hit_dist
    compression = jnp.clip(compression, 0.0, max_travel)
    
    # Contact when compression > 0
    is_contact = compression > 0.0
    
    # Contact normal is the ground normal (assuming flat ground for now)
    # For walls/curved surfaces, we'd need actual SDF normals
    contact_normal = jnp.zeros_like(wheel_world_pos)
    contact_normal = contact_normal.at[..., 2].set(1.0)  # Normal pointing up
    
    return compression, is_contact, contact_normal


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
    """
    Compute suspension force using spring-damper model.
    
    F = (k * compression - c * velocity) * force_scale
    
    Args:
        compression: Suspension compression (N, MAX_CARS, 4)
        compression_vel: Velocity of compression (positive = compressing) (N, MAX_CARS, 4)
        is_contact: Contact flags (N, MAX_CARS, 4)
        inv_contact_dot: 1 / dot(contact_normal, car_up) (N, MAX_CARS, 4)
        stiffness: Spring constant (N/m)
        damping_comp: Compression damping coefficient
        damping_relax: Relaxation damping coefficient
        force_scales: Per-wheel force multipliers (4,)
        
    Returns:
        Suspension force magnitude (N, MAX_CARS, 4)
    """
    # C++: force = (rest - len) * stiffness * clippedInvContactDotSuspension
    spring_force = stiffness * compression * inv_contact_dot
    
    # C++: suspensionRelativeVelocity = projVel * clippedInvContactDotSuspension
    effective_vel = compression_vel * inv_contact_dot
    
    # Damping opposes velocity
    # If effective_vel > 0 (compressing), use compression damping
    # If effective_vel < 0 (extending), use relaxation damping
    damping = jnp.where(effective_vel > 0, damping_comp, damping_relax)
    damper_force = damping * effective_vel
    
    force_scales_expanded = force_scales[None, None, :]
    total_force = (spring_force + damper_force) * force_scales_expanded
    
    # RL never uses downwards suspension forces
    total_force = jnp.maximum(total_force, 0.0)
    
    total_force = jnp.where(is_contact, total_force, 0.0)
    
    return total_force


# =============================================================================
# TIRE PHYSICS
# =============================================================================


def compute_steering_angle(
    steer_input: jnp.ndarray,
    forward_speed: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute steering angle based on input and speed.
    
    Args:
        steer_input: Steering input [-1, 1] (N, MAX_CARS)
        forward_speed: Forward speed of car (N, MAX_CARS)
        
    Returns:
        Steering angle in radians (N, MAX_CARS)
    """
    max_angle = jnp.interp(
        jnp.abs(forward_speed),
        STEER_ANGLE_CURVE_SPEEDS,
        STEER_ANGLE_CURVE_ANGLES
    )
    return steer_input * max_angle


def compute_tire_basis_vectors(
    car_quat: jnp.ndarray,
    steer_angle: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute forward and right vectors for each tire, accounting for steering.
    
    Args:
        car_quat: Car rotation quaternion (N, MAX_CARS, 4)
        steer_angle: Steering angle in radians (N, MAX_CARS)
        
    Returns:
        tire_forward: Forward direction for each tire (N, MAX_CARS, 4, 3)
        tire_right: Right direction for each tire (N, MAX_CARS, 4, 3)
    """
    # Car's local coordinate system:
    # Forward = +X, Right = -Y, Up = +Z
    forward_local = jnp.array([1.0, 0.0, 0.0])
    right_local = jnp.array([0.0, -1.0, 0.0])
    
    car_forward = quat_rotate_vector(car_quat, forward_local)
    car_right = quat_rotate_vector(car_quat, right_local)
    
    # Steering rotates around Z axis
    cos_steer = jnp.cos(steer_angle)[..., None]
    sin_steer = jnp.sin(steer_angle)[..., None]
    
    # Steered forward: rotated around Z
    steered_forward = car_forward * cos_steer - car_right * sin_steer
    steered_right = car_forward * sin_steer + car_right * cos_steer
    
    car_forward_exp = car_forward[..., None, :]
    car_right_exp = car_right[..., None, :]
    steered_forward_exp = steered_forward[..., None, :]
    steered_right_exp = steered_right[..., None, :]
    
    front_mask = FRONT_WHEEL_MASK[None, None, :, None]
    
    tire_forward = jnp.where(
        front_mask > 0.5,
        jnp.broadcast_to(steered_forward_exp, car_forward_exp.shape[:-2] + (4, 3)),
        jnp.broadcast_to(car_forward_exp, car_forward_exp.shape[:-2] + (4, 3))
    )
    tire_right = jnp.where(
        front_mask > 0.5,
        jnp.broadcast_to(steered_right_exp, car_right_exp.shape[:-2] + (4, 3)),
        jnp.broadcast_to(car_right_exp, car_right_exp.shape[:-2] + (4, 3))
    )
    
    # Project to XY plane and normalize (tire forces are in ground plane)
    tire_forward = tire_forward.at[..., 2].set(0.0)
    tire_forward = tire_forward / jnp.maximum(
        jnp.linalg.norm(tire_forward, axis=-1, keepdims=True), 1e-8
    )
    
    tire_right = tire_right.at[..., 2].set(0.0)
    tire_right = tire_right / jnp.maximum(
        jnp.linalg.norm(tire_right, axis=-1, keepdims=True), 1e-8
    )
    
    return tire_forward, tire_right


def compute_tire_forces(
    car_quat: jnp.ndarray,
    car_vel: jnp.ndarray,
    wheel_vel: jnp.ndarray,
    throttle: jnp.ndarray,
    steer: jnp.ndarray,
    handbrake: jnp.ndarray,
    is_contact: jnp.ndarray,
    suspension_force: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute tire forces including steering, lateral friction, and longitudinal forces.
    
    Args:
        car_quat: Car rotation quaternion (N, MAX_CARS, 4)
        car_vel: Car linear velocity (N, MAX_CARS, 3)
        wheel_vel: Velocity at each wheel contact (N, MAX_CARS, 4, 3)
        throttle: Throttle input [-1, 1] (N, MAX_CARS)
        steer: Steering input [-1, 1] (N, MAX_CARS)
        handbrake: Handbrake flag (N, MAX_CARS) bool
        is_contact: Per-wheel ground contact (N, MAX_CARS, 4) bool
        suspension_force: Normal force at each wheel (N, MAX_CARS, 4)
        
    Returns:
        Tire force vectors in world frame (N, MAX_CARS, 4, 3)
    """
    forward_local = jnp.array([1.0, 0.0, 0.0])
    car_forward = quat_rotate_vector(car_quat, forward_local)
    
    # Use CAR velocity for forward speed, not wheel velocity
    # This is important because wheel_vel includes omega x r contribution
    # which shouldn't affect rolling resistance or drive forces
    forward_speed = jnp.sum(car_vel * car_forward, axis=-1)  # Car's forward speed
    
    steer_angle = compute_steering_angle(steer, forward_speed)
    tire_forward, tire_right = compute_tire_basis_vectors(car_quat, steer_angle)
    
    # For lateral friction, we DO want wheel velocity (to counter side slip from omega x r)
    vel_right = jnp.sum(wheel_vel * tire_right, axis=-1)
    
    # For longitudinal forces (drive, brake, rolling resistance), use CAR forward speed
    # expanded to match wheel dimensions
    forward_speed_expanded = forward_speed[..., None]  # (N, MAX_CARS, 1)
    
    abs_vel_forward = jnp.abs(forward_speed_expanded)
    abs_vel_right = jnp.abs(vel_right)
    slip_ratio = abs_vel_right / (abs_vel_forward + abs_vel_right + 1e-6)
    
    lat_friction_coef = LATERAL_FRICTION_BASE * (1.0 - slip_ratio) + LATERAL_FRICTION_MIN * slip_ratio
    
    # Coulomb Friction Model for Lateral Force
    # Max friction force = Normal Force * Friction Coefficient
    # Normal Force is suspension_force
    lat_stiffness = FRICTION_FORCE_SCALE * 10.0
    lat_force_viscous = -vel_right * lat_friction_coef * lat_stiffness
    
    # Coulomb limit
    max_lat_force = suspension_force * 2.0  # Mu ~ 2.0 for sticky tires
    
    # Additional: when car is nearly stationary (low car forward speed),
    # apply friction to FORWARD wheel velocity too (to prevent pitch-induced drift)
    # This simulates the wheel "sticking" to ground when not moving
    vel_forward_wheel = jnp.sum(wheel_vel * tire_forward, axis=-1)  # Wheel vel in tire forward dir
    
    is_nearly_stopped = jnp.abs(forward_speed_expanded) < 50.0  # Below 50 UU/s
    stopping_friction = jnp.where(
        is_nearly_stopped,
        -vel_forward_wheel * lat_stiffness * 1.5,  # Apply strong friction to wheel forward vel
        0.0
    )
    stopping_friction = jnp.clip(stopping_friction, -max_lat_force, max_lat_force)
    
    # The stopping friction acts in tire_forward direction (longitudinal)
    
    lat_force_mag = jnp.clip(lat_force_viscous, -max_lat_force, max_lat_force)
    
    handbrake_expanded = handbrake[..., None]
    handbrake_factor = jnp.where(
        handbrake_expanded > 0.5,
        HANDBRAKE_LAT_FRICTION_FACTOR,
        1.0
    )
    lat_force_mag = lat_force_mag * handbrake_factor
    
    throttle_expanded = throttle[..., None]
    
    # Use car forward speed (not wheel velocity) for drive torque curve
    drive_factor = jnp.interp(
        abs_vel_forward,  # Use car forward speed
        DRIVE_TORQUE_CURVE_SPEEDS,
        DRIVE_TORQUE_CURVE_FACTORS
    )
    throttle_force = throttle_expanded * TIRE_DRIVE_FORCE * drive_factor / 4.0
    
    is_braking = (throttle_expanded * forward_speed_expanded) < 0
    
    # Smooth braking at low speeds to prevent chatter (match C++ logic)
    # C++ uses ROLLING_FRICTION_SCALE_MAGIC = 113.73963
    # Use car forward speed for braking too
    braking_friction = -forward_speed_expanded * 113.74
    max_brake = jnp.abs(throttle_expanded) * BRAKE_FORCE / 4.0
    
    brake_force_mag = jnp.clip(braking_friction, -max_brake, max_brake)
    
    brake_force_mag = jnp.where(
        is_braking,
        brake_force_mag,
        0.0
    )
    
    is_coasting = jnp.abs(throttle_expanded) < 0.01
    rolling_resistance_force = jnp.where(
        is_coasting,
        -forward_speed_expanded * ROLLING_RESISTANCE * FRICTION_FORCE_SCALE,
        0.0
    )
    
    # Add stopping friction when nearly stationary (prevents pitch-induced drift)
    long_force_mag = jnp.where(
        is_braking,
        brake_force_mag,
        throttle_force + rolling_resistance_force + stopping_friction
    )
    
    long_handbrake_factor = jnp.where(
        handbrake_expanded > 0.5,
        HANDBRAKE_LONG_FRICTION_FACTOR,
        1.0
    )
    long_force_mag = long_force_mag * long_handbrake_factor
    
    lateral_force = tire_right * lat_force_mag[..., None]
    longitudinal_force = tire_forward * long_force_mag[..., None]
    
    total_tire_force = lateral_force + longitudinal_force
    
    is_contact_expanded = is_contact[..., None]
    total_tire_force = jnp.where(is_contact_expanded, total_tire_force, 0.0)
    
    return total_tire_force


def aggregate_wheel_forces(
    suspension_force: jnp.ndarray,
    tire_force: jnp.ndarray,
    wheel_world_pos: jnp.ndarray,
    car_pos: jnp.ndarray,
    contact_normal: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sum forces and torques from all wheels onto the car chassis.
    
    Args:
        suspension_force: Scalar suspension force per wheel (N, MAX_CARS, 4)
        tire_force: Tire force vectors (N, MAX_CARS, 4, 3)
        wheel_world_pos: Wheel positions (N, MAX_CARS, 4, 3)
        car_pos: Car center position (N, MAX_CARS, 3)
        contact_normal: Surface normal at contact (N, MAX_CARS, 4, 3)
        
    Returns:
        total_force: (N, MAX_CARS, 3)
        total_torque: (N, MAX_CARS, 3)
    """
    sus_force_expanded = suspension_force[..., None]
    sus_force_vec = contact_normal * sus_force_expanded
    
    force_per_wheel = sus_force_vec + tire_force
    total_force = jnp.sum(force_per_wheel, axis=-2)
    
    r = wheel_world_pos - car_pos[..., None, :]
    torque_per_wheel = jnp.cross(r, force_per_wheel)
    total_torque = jnp.sum(torque_per_wheel, axis=-2)
    
    return total_force, total_torque


def solve_suspension_and_tires(
    cars: CarState,
    controls: CarControls,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Main function to compute all wheel-related forces.
    
    Args:
        cars: Current car state
        controls: Car control inputs
        
    Returns:
        total_force: Net force on chassis (N, MAX_CARS, 3)
        total_torque: Net torque on chassis (N, MAX_CARS, 3)
        is_contact: Per-wheel contact flags (N, MAX_CARS, 4)
        num_contacts: Number of wheels in contact (N, MAX_CARS)
    """
    wheel_world_pos = compute_wheel_world_positions(cars.pos, cars.quat)
    wheel_vel = compute_wheel_velocities(cars.vel, cars.ang_vel, cars.quat)
    
    compression, is_contact, contact_normal = raycast_suspension(wheel_world_pos, cars.quat)
    
    # Calculate compression velocity (project wheel velocity onto contact normal)
    # compression_vel = -dot(wheel_vel, normal)
    # If wheel moves INTO wall (vel opposes normal), compression increases.
    compression_vel = -jnp.sum(wheel_vel * contact_normal, axis=-1)
    
    # Calculate inv_contact_dot for suspension scaling
    car_up = get_car_up_dir(cars.quat)  # (N, MAX_CARS, 3)
    car_up_expanded = car_up[..., None, :]  # (N, MAX_CARS, 1, 3)
    
    denominator = jnp.sum(contact_normal * car_up_expanded, axis=-1)
    
    # C++ logic: if denominator > 0.1, inv = 1/denom, else inv = 10
    inv_contact_dot = jnp.where(
        denominator > 0.1,
        1.0 / denominator,
        10.0
    )
    
    suspension_force = compute_suspension_force(
        compression, 
        compression_vel, 
        is_contact,
        inv_contact_dot
    )
    
    tire_force = compute_tire_forces(
        cars.quat, 
        cars.vel, 
        wheel_vel,
        controls.throttle, 
        controls.steer, 
        controls.handbrake,
        is_contact,
        suspension_force
    )
    
    total_force, total_torque = aggregate_wheel_forces(
        suspension_force, tire_force, wheel_world_pos, cars.pos, contact_normal
    )
    
    num_contacts = jnp.sum(is_contact.astype(jnp.float32), axis=-1)
    
    return total_force, total_torque, is_contact, num_contacts


# =============================================================================
# BALL PHYSICS STEP
# =============================================================================


def step_ball(ball: BallState, dt: float = DT) -> BallState:
    """
    Advance ball physics by one timestep.
    
    Args:
        ball: Current ball state
        dt: Time step
        
    Returns:
        Updated ball state
    """
    vel = apply_gravity(ball.vel, dt)
    vel = apply_ball_drag(vel, BALL_DRAG, dt)
    vel = clamp_velocity(vel, BALL_MAX_SPEED)
    ang_vel = clamp_angular_velocity(ball.ang_vel, BALL_MAX_ANG_SPEED)
    
    pos = integrate_position(ball.pos, vel, dt)
    pos, vel, ang_vel = resolve_ball_arena_collision(pos, vel, ang_vel)
    
    return ball.replace(
        pos=pos,
        vel=vel,
        ang_vel=ang_vel
    )
