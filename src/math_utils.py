"""
Quaternion and Linear Algebra Helpers
=====================================
Vectorized math operations for physics simulation.
Quaternions are stored as [w, x, y, z] (scalar-first convention).
"""

from __future__ import annotations
import jax.numpy as jnp


def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Hamilton product of two quaternions.
    
    Args:
        q1: First quaternion [..., 4] in [w, x, y, z] order
        q2: Second quaternion [..., 4] in [w, x, y, z] order
        
    Returns:
        Product quaternion [..., 4]
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    return jnp.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], axis=-1)


def quat_normalize(q: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize quaternion to unit length.
    
    CRITICAL: Must be called after integration to prevent quaternion drift.
    
    Args:
        q: Quaternion [..., 4]
        
    Returns:
        Normalized quaternion [..., 4]
    """
    norm = jnp.linalg.norm(q, axis=-1, keepdims=True)
    # Prevent division by zero (degenerate quaternion -> identity)
    norm = jnp.maximum(norm, 1e-8)
    return q / norm


def quat_from_angular_velocity(ang_vel: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
    Create a quaternion representing rotation by angular velocity over dt.
    
    Uses small-angle approximation for efficiency:
    q ≈ [1, wx*dt/2, wy*dt/2, wz*dt/2] then normalize
    
    For larger angles, this should use the exponential map, but for
    dt=1/120 and typical ang_vel < 6 rad/s, approximation is fine.
    
    Args:
        ang_vel: Angular velocity [..., 3] in rad/s
        dt: Time step
        
    Returns:
        Rotation quaternion [..., 4]
    """
    # Half-angle representation
    half_angle = ang_vel * (dt * 0.5)
    
    # More accurate version with proper quaternion exponential:
    angle = jnp.linalg.norm(ang_vel, axis=-1, keepdims=True) * dt
    half_angle_mag = angle * 0.5
    
    # For very small rotations, use linear approximation
    small_angle_mask = angle < 1e-6
    
    # Axis (normalized angular velocity direction)
    axis = ang_vel / jnp.maximum(jnp.linalg.norm(ang_vel, axis=-1, keepdims=True), 1e-8)
    
    # Quaternion components
    w = jnp.where(small_angle_mask[..., 0], 1.0, jnp.cos(half_angle_mag[..., 0]))
    xyz = jnp.where(small_angle_mask, half_angle, axis * jnp.sin(half_angle_mag))
    
    q = jnp.concatenate([w[..., None], xyz], axis=-1)
    return quat_normalize(q)


def quat_rotate_vector(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate a vector by a quaternion.
    
    Uses the formula: v' = q * v * q^(-1)
    Optimized to avoid full quaternion multiplications.
    
    Args:
        q: Rotation quaternion [..., 4] in [w, x, y, z] order
        v: Vector to rotate [..., 3]
        
    Returns:
        Rotated vector [..., 3]
    """
    # Extract quaternion components
    qw = q[..., 0:1]  # Keep dims for broadcasting
    qv = q[..., 1:4]  # xyz part
    
    # Cross products
    uv = jnp.cross(qv, v)
    uuv = jnp.cross(qv, uv)
    
    # v' = v + 2 * (qw * (qv × v) + qv × (qv × v))
    return v + 2.0 * (qw * uv + uuv)


def quat_to_rotation_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        q: Quaternion [..., 4] in [w, x, y, z] order
        
    Returns:
        Rotation matrix [..., 3, 3]
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Precompute products
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    # Build rotation matrix
    r00 = 1 - 2*(yy + zz)
    r01 = 2*(xy - wz)
    r02 = 2*(xz + wy)
    
    r10 = 2*(xy + wz)
    r11 = 1 - 2*(xx + zz)
    r12 = 2*(yz - wx)
    
    r20 = 2*(xz - wy)
    r21 = 2*(yz + wx)
    r22 = 1 - 2*(xx + yy)
    
    return jnp.stack([
        jnp.stack([r00, r01, r02], axis=-1),
        jnp.stack([r10, r11, r12], axis=-1),
        jnp.stack([r20, r21, r22], axis=-1),
    ], axis=-2)


def quat_from_yaw(yaw: jnp.ndarray) -> jnp.ndarray:
    """
    Create quaternion from yaw angle (rotation around Z axis).
    
    Args:
        yaw: Yaw angle(s) in radians. Shape: (...,)
        
    Returns:
        Quaternion(s) [w, x, y, z]. Shape: (..., 4)
    """
    half_yaw = yaw / 2.0
    cos_half = jnp.cos(half_yaw)
    sin_half = jnp.sin(half_yaw)
    
    # Rotation around Z: [cos(θ/2), 0, 0, sin(θ/2)]
    zeros = jnp.zeros_like(yaw)
    return jnp.stack([cos_half, zeros, zeros, sin_half], axis=-1)


def get_forward_up_right(quat: jnp.ndarray) -> tuple:
    """
    Get forward, up, and right vectors from quaternion.
    
    In Rocket League coordinate system:
    - Forward = local +X axis
    - Right = local +Y axis (but we use -Y for right)
    - Up = local +Z axis
    
    Args:
        quat: Quaternion [w, x, y, z]. Shape: (..., 4)
        
    Returns:
        Tuple of (forward, up, right) vectors, each shape: (..., 3)
    """
    rot_mat = quat_to_rotation_matrix(quat)
    
    forward = rot_mat[..., :, 0]  # First column = local X axis = forward
    right = -rot_mat[..., :, 1]   # Second column (negated) = local -Y = right
    up = rot_mat[..., :, 2]       # Third column = local Z axis = up
    
    return forward, up, right


def get_car_forward_dir(quat: jnp.ndarray) -> jnp.ndarray:
    """
    Get car forward direction from quaternion.
    Forward is +X in local space.
    
    Args:
        quat: Quaternion [w, x, y, z]. Shape: (..., 4)
        
    Returns:
        Forward direction in world space. Shape: (..., 3)
    """
    local_forward = jnp.array([1.0, 0.0, 0.0])
    return quat_rotate_vector(quat, local_forward)


def get_car_up_dir(quat: jnp.ndarray) -> jnp.ndarray:
    """
    Get car up direction from quaternion.
    Up is +Z in local space.
    
    Args:
        quat: Quaternion [w, x, y, z]. Shape: (..., 4)
        
    Returns:
        Up direction in world space. Shape: (..., 3)
    """
    local_up = jnp.array([0.0, 0.0, 1.0])
    return quat_rotate_vector(quat, local_up)


def get_car_right_dir(quat: jnp.ndarray) -> jnp.ndarray:
    """
    Get car right direction from quaternion.
    Right is -Y in local space (RocketSim convention).
    
    Args:
        quat: Quaternion [w, x, y, z]. Shape: (..., 4)
        
    Returns:
        Right direction in world space. Shape: (..., 3)
    """
    local_right = jnp.array([0.0, -1.0, 0.0])
    return quat_rotate_vector(quat, local_right)


def clamp_velocity(vel: jnp.ndarray, max_speed: float) -> jnp.ndarray:
    """
    Clamp velocity magnitude to maximum speed.
    
    Uses branchless implementation via jnp.where.
    
    Args:
        vel: Velocity [..., 3]
        max_speed: Maximum speed scalar
        
    Returns:
        Clamped velocity [..., 3]
    """
    speed_sq = jnp.sum(vel * vel, axis=-1, keepdims=True)
    max_speed_sq = max_speed * max_speed
    
    # Compute scale factor (1.0 if under limit, otherwise scale down)
    scale = jnp.where(
        speed_sq > max_speed_sq,
        max_speed / jnp.sqrt(jnp.maximum(speed_sq, 1e-8)),
        1.0
    )
    return vel * scale


def clamp_angular_velocity(ang_vel: jnp.ndarray, max_ang_speed: float) -> jnp.ndarray:
    """
    Clamp angular velocity magnitude.
    
    Args:
        ang_vel: Angular velocity [..., 3]
        max_ang_speed: Maximum angular speed (rad/s)
        
    Returns:
        Clamped angular velocity [..., 3]
    """
    ang_speed_sq = jnp.sum(ang_vel * ang_vel, axis=-1, keepdims=True)
    max_ang_speed_sq = max_ang_speed * max_ang_speed
    
    scale = jnp.where(
        ang_speed_sq > max_ang_speed_sq,
        max_ang_speed / jnp.sqrt(jnp.maximum(ang_speed_sq, 1e-8)),
        1.0
    )
    return ang_vel * scale
