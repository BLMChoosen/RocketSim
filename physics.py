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
    TIRE_DRIVE_FORCE, BRAKE_FORCE,
    STEER_ANGLE_CURVE_SPEEDS, STEER_ANGLE_CURVE_ANGLES,
    DRIVE_TORQUE_CURVE_SPEEDS, DRIVE_TORQUE_CURVE_FACTORS,
    FRONT_WHEEL_MASK, STICKY_FORCE_SCALE_BASE, STOPPING_FORWARD_VEL,
    THROTTLE_TORQUE_AMOUNT, BRAKE_TORQUE_AMOUNT, UU_TO_BT, BT_TO_UU,
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
    Uses arena SDF for proper collision detection with walls/ceiling/corners.
    
    For wall driving, we sample multiple points along the ray to find the
    closest intersection point with the arena surface.
    
    Args:
        wheel_world_pos: Wheel hardpoint positions (N, MAX_CARS, 4, 3)
        car_quat: Car rotation quaternions (N, MAX_CARS, 4)
        wheel_radii: Wheel radii (4,)
        sus_rest: Suspension rest lengths (4,)
        max_travel: Maximum suspension travel
        ground_z: Ground plane Z coordinate (fallback)
        
    Returns:
        compression: Suspension compression (N, MAX_CARS, 4)
        is_contact: Boolean contact flags (N, MAX_CARS, 4)
        contact_normal: Normal at contact point (N, MAX_CARS, 4, 3)
    """
    from collision import arena_sdf
    
    # Get car's down direction (negative Z in local space)
    car_up = get_car_up_dir(car_quat)  # (N, MAX_CARS, 3)
    car_down = -car_up  # Ray direction (pointing down from wheel)
    
    # Expand for 4 wheels: (N, MAX_CARS, 4, 3)
    car_down_expanded = car_down[..., None, :]  # (N, MAX_CARS, 1, 3)
    car_down_expanded = jnp.broadcast_to(car_down_expanded, wheel_world_pos.shape)
    
    # Ray length is rest length + wheel radius
    radii = wheel_radii[None, None, :]  # (1, 1, 4)
    rest = sus_rest[None, None, :]  # (1, 1, 4)
    ray_length = rest + radii
    
    # Sample multiple points along the ray to find intersection
    # This is more expensive but handles wall driving correctly
    n_samples = 8
    sample_fractions = jnp.linspace(0.0, 1.0, n_samples)  # (n_samples,)
    
    # Calculate sample positions along ray
    # wheel_world_pos: (N, MAX_CARS, 4, 3)
    # sample_fractions: (n_samples,)
    # We want: (N, MAX_CARS, 4, n_samples, 3)
    
    ray_end = wheel_world_pos + car_down_expanded * ray_length[..., None]
    
    # Expand for samples
    wheel_pos_exp = wheel_world_pos[..., None, :]  # (N, MAX_CARS, 4, 1, 3)
    ray_end_exp = ray_end[..., None, :]  # (N, MAX_CARS, 4, 1, 3)
    fracs = sample_fractions[None, None, None, :, None]  # (1, 1, 1, n_samples, 1)
    
    sample_positions = wheel_pos_exp + (ray_end_exp - wheel_pos_exp) * fracs  # (N, MAX_CARS, 4, n_samples, 3)
    
    # Query SDF at all sample points
    orig_shape = sample_positions.shape  # (N, MAX_CARS, 4, n_samples, 3)
    n_envs = orig_shape[0]
    max_cars = orig_shape[1]
    
    sample_flat = sample_positions.reshape(-1, 3)  # (N*MAX_CARS*4*n_samples, 3)
    sample_for_sdf = sample_flat[None, :, :]  # (1, N*MAX_CARS*4*n_samples, 3)
    
    dist_flat, normal_flat = arena_sdf(sample_for_sdf)
    dist_flat = dist_flat[0]  # (N*MAX_CARS*4*n_samples,)
    normal_flat = normal_flat[0]  # (N*MAX_CARS*4*n_samples, 3)
    
    sdf_dist = dist_flat.reshape(n_envs, max_cars, 4, n_samples)  # (N, MAX_CARS, 4, n_samples)
    sdf_normal = normal_flat.reshape(n_envs, max_cars, 4, n_samples, 3)  # (N, MAX_CARS, 4, n_samples, 3)
    
    # Find the first sample that penetrates (SDF < wheel_radius)
    # SDF is POSITIVE inside arena, so penetration is when SDF < wheel_radius
    # wheel_radii: (4,) -> (1, 1, 4, 1)
    radii_exp = radii[..., None]  # (1, 1, 4, 1)
    
    # Check which samples are "in contact" (surface within wheel radius)
    is_penetrating = sdf_dist < radii_exp  # (N, MAX_CARS, 4, n_samples)
    
    # Find the first penetrating sample (smallest index where penetrating)
    # Use a mask to find first True
    # Add a large value to non-penetrating samples so argmin finds first penetrating
    sample_idx = jnp.arange(n_samples)[None, None, None, :]  # (1, 1, 1, n_samples)
    masked_idx = jnp.where(is_penetrating, sample_idx, n_samples * 2)  # (N, MAX_CARS, 4, n_samples)
    first_contact_idx = jnp.argmin(masked_idx, axis=-1)  # (N, MAX_CARS, 4)
    
    # Check if ANY sample penetrates
    any_contact = jnp.any(is_penetrating, axis=-1)  # (N, MAX_CARS, 4)
    
    # Get the contact fraction
    contact_fraction = sample_fractions[first_contact_idx]  # (N, MAX_CARS, 4)
    contact_fraction = jnp.where(any_contact, contact_fraction, 1.0)
    
    # Calculate compression
    # If contact at fraction f, the ray traveled f * ray_length before hitting
    # Compression = (1 - f) * ray_length - wheel_radius (distance saved)
    # Actually: compression = ray_length * (1 - contact_fraction) but capped
    compression_raw = ray_length[0, 0, :] * (1.0 - contact_fraction)
    compression = jnp.clip(compression_raw, 0.0, max_travel)
    
    # Get contact normal at the first contact point
    batch_idx = jnp.arange(n_envs)[:, None, None]  # (N, 1, 1)
    car_idx = jnp.arange(max_cars)[None, :, None]  # (1, MAX_CARS, 1)
    wheel_idx = jnp.arange(4)[None, None, :]  # (1, 1, 4)
    contact_normal = sdf_normal[batch_idx, car_idx, wheel_idx, first_contact_idx]  # (N, MAX_CARS, 4, 3)
    
    # Use car_up as fallback normal when no contact
    car_up_exp = car_up[..., None, :]  # (N, MAX_CARS, 1, 3)
    car_up_exp = jnp.broadcast_to(car_up_exp, contact_normal.shape)
    contact_normal = jnp.where(any_contact[..., None], contact_normal, car_up_exp)
    
    return compression, any_contact, contact_normal


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
    car_ang_vel: jnp.ndarray,
    car_pos: jnp.ndarray,
    wheel_world_pos: jnp.ndarray,
    throttle: jnp.ndarray,
    steer: jnp.ndarray,
    handbrake: jnp.ndarray,
    is_contact: jnp.ndarray,
    contact_normal: jnp.ndarray,
    forward_speed: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute tire forces matching C++ btVehicleRL logic.
    
    C++ calcFrictionImpulses():
    1. Lateral: resolveSingleBilateral() - computes impulse to zero lateral slip
    2. Longitudinal: either engine force or rolling friction (ROLLING_FRICTION_SCALE_MAGIC)
    3. Both scaled by (mass/3) and friction curves
    
    Returns:
        impulses: Per-wheel impulse vectors (N, MAX_CARS, 4, 3) - applied at contact point
        wheel_rel_pos: Relative position for torque (N, MAX_CARS, 4, 3)
    """
    from sim_constants import CAR_MASS, DT, BT_TO_UU, UU_TO_BT
    
    # Friction scale from C++: frictionScale = mass / 3
    friction_scale = CAR_MASS / 3.0
    
    # Steering angle
    steer_angle = compute_steering_angle(steer, forward_speed)
    
    # Get car basis vectors
    forward_local = jnp.array([1.0, 0.0, 0.0])
    right_local = jnp.array([0.0, -1.0, 0.0])  # -Y is right in RL coords
    up_local = jnp.array([0.0, 0.0, 1.0])
    
    car_forward = quat_rotate_vector(car_quat, forward_local)
    car_right = quat_rotate_vector(car_quat, right_local)
    car_up = quat_rotate_vector(car_quat, up_local)
    
    # === COMPUTE WHEEL AXLE AND FORWARD DIRECTIONS ===
    # C++: axleDir = wheel.m_worldTransform.getBasis().getColumn(m_indexRightAxis)
    # Where m_indexRightAxis = 0, and column 0 stores -right (i.e., LEFT direction)
    # So axleDir points LEFT, not right!
    car_left = -car_right  # axleDir in C++ is the LEFT vector
    
    # Apply steering to front wheels
    # Steered left direction = rotate car_left around up axis by steer_angle
    cos_steer = jnp.cos(steer_angle)[..., None]  # (N, MAX_CARS, 1)
    sin_steer = jnp.sin(steer_angle)[..., None]
    
    # Steered left direction (axle direction)
    # When steer > 0 (right turn), the wheel points more forward-left
    steered_left = -car_forward * sin_steer + car_left * cos_steer
    
    # Expand for 4 wheels
    car_left_4 = car_left[..., None, :]  # (N, MAX_CARS, 1, 3)
    steered_left_4 = steered_left[..., None, :]
    
    front_mask = FRONT_WHEEL_MASK[None, None, :, None]  # (1, 1, 4, 1)
    
    # Axle direction per wheel (LEFT vector, as in C++)
    axle_dir = jnp.where(
        front_mask > 0.5,
        jnp.broadcast_to(steered_left_4, car_left_4.shape[:-2] + (4, 3)),
        jnp.broadcast_to(car_left_4, car_left_4.shape[:-2] + (4, 3))
    )
    
    # C++: Project axle onto surface plane
    # proj = axleDir.dot(surfNormalWS)
    # axleDir -= surfNormalWS * proj
    # axleDir = axleDir.safeNormalized()
    proj = jnp.sum(axle_dir * contact_normal, axis=-1, keepdims=True)
    axle_dir = axle_dir - contact_normal * proj
    axle_dir = axle_dir / jnp.maximum(jnp.linalg.norm(axle_dir, axis=-1, keepdims=True), 1e-8)
    
    # C++: forwardDir = surfNormalWS.cross(axleDir).safeNormalized()
    forward_dir = jnp.cross(contact_normal, axle_dir)
    forward_dir = forward_dir / jnp.maximum(jnp.linalg.norm(forward_dir, axis=-1, keepdims=True), 1e-8)
    
    # === COMPUTE WHEEL VELOCITIES AT CONTACT POINT ===
    # C++: crossVec = (angularVel.cross(wheelDelta) + vel) * BT_TO_UU
    wheel_delta = wheel_world_pos - car_pos[..., None, :]  # (N, MAX_CARS, 4, 3)
    car_ang_vel_4 = car_ang_vel[..., None, :]  # (N, MAX_CARS, 1, 3)
    car_vel_4 = car_vel[..., None, :]
    
    cross_vec = jnp.cross(car_ang_vel_4, wheel_delta) + car_vel_4  # Already in UU
    
    # === LATERAL FRICTION (C++ resolveSingleBilateral) ===
    # The bilateral constraint computes impulse to zero velocity along axis
    # sideImpulse = -vel_lateral (simplified)
    vel_lateral = jnp.sum(cross_vec * axle_dir, axis=-1)  # (N, MAX_CARS, 4)
    side_impulse = -vel_lateral
    
    # === LONGITUDINAL FRICTION ===
    vel_forward = jnp.sum(cross_vec * forward_dir, axis=-1)  # (N, MAX_CARS, 4)
    
    # C++ logic: 
    # if engineForce == 0:
    #   if brake: rollingFriction = clamp(-relVel * 113.73963, -brake, brake)
    #   else: rollingFriction = 0
    # else:
    #   rollingFriction = -engineForce / frictionScale
    
    throttle_4 = throttle[..., None]  # (N, MAX_CARS, 1)
    forward_speed_4 = forward_speed[..., None]
    abs_forward_speed = jnp.abs(forward_speed_4)
    
    # Drive speed scale curve
    drive_speed_scale = jnp.interp(
        abs_forward_speed,
        DRIVE_TORQUE_CURVE_SPEEDS,
        DRIVE_TORQUE_CURVE_FACTORS
    )
    
    # Check if fewer than 3 wheels in contact (C++ divides by 4)
    num_contacts = jnp.sum(is_contact.astype(jnp.float32), axis=-1, keepdims=True)  # (N, MAX_CARS, 1)
    drive_speed_scale = jnp.where(num_contacts < 3, drive_speed_scale / 4.0, drive_speed_scale)
    
    # Engine force (per wheel, C++ applies to all wheels)
    engine_throttle = throttle_4
    
    # C++ throttle/brake logic
    abs_throttle = jnp.abs(throttle_4)
    is_reversing = (abs_forward_speed > 25.0) & (jnp.sign(throttle_4) != jnp.sign(forward_speed_4)) & (abs_throttle > 0.001)
    is_coasting = abs_throttle < 0.001
    
    # When reversing, we brake (engine_throttle = 0 if speed > threshold)
    engine_throttle = jnp.where(
        is_reversing & (abs_forward_speed > 0.01),
        0.0,
        engine_throttle
    )
    
    # Coasting: no engine, apply brake factor
    engine_throttle = jnp.where(is_coasting, 0.0, engine_throttle)
    
    # Brake force
    brake_input = jnp.where(is_reversing, 1.0, 0.0)  # Full brake when reversing
    brake_input = jnp.where(
        is_coasting, 
        jnp.where(abs_forward_speed < 25.0, 1.0, 0.15),  # Coasting brake
        brake_input
    )
    
    # C++ constants
    ROLLING_FRICTION_SCALE_MAGIC = 113.73963
    
    # Engine force calculation (per wheel)
    drive_engine_force = engine_throttle * (THROTTLE_TORQUE_AMOUNT * UU_TO_BT) * drive_speed_scale
    drive_brake_force = brake_input * (BRAKE_TORQUE_AMOUNT * UU_TO_BT)
    
    # Rolling friction (C++ version)
    # When engine == 0 and brake > 0: rollingFriction = clamp(-relVel * MAGIC, -brake, brake)
    # When engine != 0: rollingFriction = -engineForce / frictionScale
    
    has_engine = jnp.abs(drive_engine_force) > 0.001
    
    # Brake rolling friction
    rolling_friction_brake = jnp.clip(
        -vel_forward * ROLLING_FRICTION_SCALE_MAGIC,
        -drive_brake_force,
        drive_brake_force
    )
    rolling_friction_brake = jnp.where(drive_brake_force > 0, rolling_friction_brake, 0.0)
    
    # Engine rolling friction (opposite of engine direction)
    rolling_friction_engine = -drive_engine_force / friction_scale
    
    rolling_friction = jnp.where(has_engine, rolling_friction_engine, rolling_friction_brake)
    
    # === FRICTION CURVES (from C++ RLConst.h) ===
    # frictionCurveInput = |vel_lat| / (|vel_fwd| + |vel_lat|) if |vel_lat| > 5 else 0
    base_friction = jnp.abs(vel_lateral)
    friction_curve_input = jnp.where(
        base_friction > 5,
        base_friction / (jnp.abs(vel_forward) + base_friction),
        0.0
    )
    
    # Lateral friction curve: {0: 1.0, 1: 0.2}
    lat_friction = 1.0 - 0.8 * friction_curve_input
    
    # Longitudinal friction: default 1.0 (curve is empty in C++)
    long_friction = jnp.ones_like(friction_curve_input)
    
    # Handbrake adjustments
    handbrake_4 = handbrake[..., None].astype(jnp.float32)  # (N, MAX_CARS, 1)
    
    # Handbrake lateral friction factor: 0.1
    lat_friction = lat_friction * (1.0 - handbrake_4 * 0.9)
    
    # Handbrake longitudinal friction factor curve: {0: 0.5, 1: 0.9}
    handbrake_long_factor = 0.5 + 0.4 * friction_curve_input
    long_friction = jnp.where(handbrake_4 > 0.5, long_friction * handbrake_long_factor, 1.0)
    
    # === STICKY FRICTION (non-sticky when no throttle) ===
    # C++ scales friction by NON_STICKY_FRICTION_FACTOR_CURVE based on contact_normal.z
    # when throttle == 0
    is_sticky = jnp.abs(throttle_4) > 0
    normal_z = contact_normal[..., 2]  # (N, MAX_CARS, 4)
    
    # Non-sticky curve: {0: 0.1, 0.7075: 0.5, 1: 1.0}
    non_sticky_scale = jnp.interp(
        normal_z,
        jnp.array([0.0, 0.7075, 1.0]),
        jnp.array([0.1, 0.5, 1.0])
    )
    
    lat_friction = jnp.where(is_sticky, lat_friction, lat_friction * non_sticky_scale)
    long_friction = jnp.where(is_sticky, long_friction, long_friction * non_sticky_scale)
    
    # === COMPUTE FINAL IMPULSE ===
    # C++: totalFrictionForce = (forwardDir * rollingFriction * longFriction) + (axleDir * sideImpulse * latFriction)
    # wheel.m_impulse = totalFrictionForce * frictionScale
    
    total_friction_force = (
        forward_dir * (rolling_friction * long_friction)[..., None] + 
        axle_dir * (side_impulse * lat_friction)[..., None]
    )
    
    wheel_impulse = total_friction_force * friction_scale
    
    # Zero out impulse for wheels not in contact
    is_contact_4 = is_contact[..., None]
    wheel_impulse = jnp.where(is_contact_4, wheel_impulse, 0.0)
    
    # === WHEEL RELATIVE POSITION FOR TORQUE ===
    # C++: wheelContactOffset = contactPointWS - chassisOrigin
    # float contactUpDot = upDir.dot(wheelContactOffset)
    # wheelRelPos = wheelContactOffset - upDir * contactUpDot
    
    car_up_4 = car_up[..., None, :]  # (N, MAX_CARS, 1, 3)
    contact_up_dot = jnp.sum(car_up_4 * wheel_delta, axis=-1, keepdims=True)
    wheel_rel_pos = wheel_delta - car_up_4 * contact_up_dot
    
    return wheel_impulse, wheel_rel_pos


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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Main function to compute all wheel-related forces.
    
    Matches C++ btVehicleRL flow:
    1. updateVehicleFirst: raycast, calcFrictionImpulses
    2. updateVehicleSecond: updateSuspension, applyFrictionImpulses
    
    Args:
        cars: Current car state
        controls: Car control inputs
        
    Returns:
        sus_force: Suspension force vectors (N, MAX_CARS, 3)
        sus_torque: Suspension torque vectors (N, MAX_CARS, 3)
        tire_impulse: Tire impulse vectors to apply (N, MAX_CARS, 4, 3)
        wheel_rel_pos: Relative position for tire torque (N, MAX_CARS, 4, 3)
        is_contact: Per-wheel contact flags (N, MAX_CARS, 4)
        num_contacts: Number of wheels in contact (N, MAX_CARS)
    """
    from sim_constants import DT
    from collision import arena_sdf
    
    wheel_world_pos = compute_wheel_world_positions(cars.pos, cars.quat)
    wheel_vel = compute_wheel_velocities(cars.vel, cars.ang_vel, cars.quat)
    
    compression, is_contact, contact_normal = raycast_suspension(wheel_world_pos, cars.quat)
    
    # Check if car body is penetrating arena (SDF < 0 at car center)
    # If so, disable suspension to avoid weird forces during collision resolution
    car_sdf, _ = arena_sdf(cars.pos)  # (N, MAX_CARS)
    car_is_inside_arena = car_sdf > -10.0  # Allow small penetration (10 UU)
    
    # Calculate compression velocity (project wheel velocity onto contact normal)
    proj_vel = -jnp.sum(wheel_vel * contact_normal, axis=-1)
    
    # Calculate inv_contact_dot for suspension scaling
    car_up = get_car_up_dir(cars.quat)
    car_up_expanded = car_up[..., None, :]
    
    denominator = jnp.sum(contact_normal * car_up_expanded, axis=-1)
    
    # C++ logic: if denominator > 0.1, inv = 1/denom, else inv = 10
    # Additionally, filter out contacts where normal points away from car's up
    # This prevents issues when car is penetrating walls
    denom_valid = denominator > 0.1
    
    # Only count as valid contact if:
    # 1. Normal roughly aligns with car's up (denom_valid)
    # 2. Car body is not deeply penetrating arena (car_is_inside_arena)
    is_contact_valid = is_contact & denom_valid & car_is_inside_arena[..., None]
    
    inv_contact_dot = jnp.where(
        denom_valid,
        1.0 / jnp.maximum(denominator, 0.1),
        10.0
    )
    compression_vel = jnp.where(denom_valid, proj_vel, 0.0)
    
    # Use filtered contact for suspension
    suspension_force = compute_suspension_force(
        compression, 
        compression_vel, 
        is_contact_valid,  # Use filtered contact
        inv_contact_dot
    )
    
    # Forward speed for tire forces
    forward_local = jnp.array([1.0, 0.0, 0.0])
    car_forward = quat_rotate_vector(cars.quat, forward_local)
    forward_speed = jnp.sum(cars.vel * car_forward, axis=-1)
    
    # Compute tire impulses (C++ style)
    # Use filtered contact to avoid applying tire forces on invalid contacts
    tire_impulse, wheel_rel_pos = compute_tire_forces(
        car_quat=cars.quat,
        car_vel=cars.vel,
        car_ang_vel=cars.ang_vel,
        car_pos=cars.pos,
        wheel_world_pos=wheel_world_pos,
        throttle=controls.throttle,
        steer=controls.steer,
        handbrake=controls.handbrake,
        is_contact=is_contact_valid,  # Use filtered contact
        contact_normal=contact_normal,
        forward_speed=forward_speed,
    )
    
    # === SUSPENSION FORCE APPLICATION ===
    # C++ applies suspension as impulse at contact point:
    # force = contactNormalWS * (suspensionForce * dt + extraPushback)
    # applyImpulse(force, contactPointOffset)
    
    sus_force_expanded = suspension_force[..., None]
    sus_force_vec = contact_normal * sus_force_expanded * DT  # Impulse = force * dt
    
    # Contact point offset from CoM
    contact_offset = wheel_world_pos - cars.pos[..., None, :]
    
    # Sum suspension forces
    total_sus_force = jnp.sum(sus_force_vec, axis=-2)
    
    # Suspension torque
    sus_torque_per_wheel = jnp.cross(contact_offset, sus_force_vec)
    total_sus_torque = jnp.sum(sus_torque_per_wheel, axis=-2)
    
    # Use filtered contacts for ground detection
    num_contacts = jnp.sum(is_contact_valid.astype(jnp.float32), axis=-1)
    
    return total_sus_force, total_sus_torque, tire_impulse, wheel_rel_pos, is_contact_valid, num_contacts


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
