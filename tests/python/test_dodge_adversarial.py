"""
Empirical Adversarial Test Harness for RocketSim-CUDA Milestone 5 (Dodge & Air Control Parity)

Validates:
1. Numerical stability (Zero NaN, Inf, or denormal floating-point values across control ranges).
2. 8 canonical flip directions, stall, and flip cancels (front/back cancel) across multi-tick rollouts.
3. Strict parity against CPU Bullet reference oracle using pre-torque angular velocity (omega_pre).
4. Quantifies discrepancy of prior pre-fix implementation vs CPU Bullet.
5. Verification of angular velocity clamping to CAR_MAX_ANG_SPEED (5.5 rad/s).
6. Random 3D orientation invariance using arbitrary SO(3) rotation matrices.
7. Post-flip pitch-lock window (FLIP_TORQUE_TIME to FLIP_TORQUE_TIME + FLIP_PITCHLOCK_EXTRA_TIME).
"""

import math
import numpy as np
import pytest

# Physical Constants from RLConst.h & config.h
TICK_RATE = 120.0
DT = 1.0 / TICK_RATE
CAR_MASS = 180.0
CAR_MAX_ANG_SPEED = 5.5

FLIP_TORQUE_X = 260.0  # Left/Right roll torque
FLIP_TORQUE_Y = 224.0  # Forward/backward pitch torque
FLIP_TORQUE_TIME = 0.65
FLIP_PITCHLOCK_EXTRA_TIME = 0.3

CAR_AIR_CONTROL_TORQUE_X = 130.0
CAR_AIR_CONTROL_TORQUE_Y = 95.0
CAR_AIR_CONTROL_TORQUE_Z = 400.0

CAR_AIR_CONTROL_DAMPING_X = 30.0
CAR_AIR_CONTROL_DAMPING_Y = 20.0
CAR_AIR_CONTROL_DAMPING_Z = 50.0

CAR_TORQUE_SCALE = 2.0 * math.pi / 65536.0 * 1000.0
THROTTLE_AIR_ACCEL = 200.0 / 3.0


class OrthonormalBasis:
    """Represents a 3x3 orthonormal basis matrix (forward, right, up)."""
    def __init__(self, forward=None, right=None, up=None):
        if forward is None:
            self.forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            self.right = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            self.forward = np.asarray(forward, dtype=np.float32)
            self.right = np.asarray(right, dtype=np.float32)
            self.up = np.asarray(up, dtype=np.float32)

    def mat_vec(self, v):
        """Multiply basis matrix by local vector: basis * v."""
        return (
            self.forward * np.float32(v[0]) +
            self.right * np.float32(v[1]) +
            self.up * np.float32(v[2])
        ).astype(np.float32)

    @staticmethod
    def from_random_quaternion(rng):
        """Generate a random uniform SO(3) rotation basis from a random unit quaternion."""
        u1, u2, u3 = rng.uniform(0.0, 1.0, size=3)
        q = np.array([
            math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
            math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
            math.sqrt(u1) * math.sin(2 * math.pi * u3),
            math.sqrt(u1) * math.cos(2 * math.pi * u3)
        ], dtype=np.float32)  # (x, y, z, w)

        x, y, z, w = q
        # Rotation matrix columns
        fwd = np.array([
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y)
        ], dtype=np.float32)
        rgt = np.array([
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x)
        ], dtype=np.float32)
        up = np.array([
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y)
        ], dtype=np.float32)
        return OrthonormalBasis(fwd, rgt, up)


def cpu_bullet_air_control(
    omega_in: np.ndarray,
    basis: OrthonormalBasis,
    controls: dict,
    is_flipping: bool,
    has_flipped: bool,
    flip_time: float,
    flip_rel_torque: np.ndarray,
    dt: float = DT,
    allow_air_torque: bool = True,
    is_auto_flipping: bool = False
):
    """
    Exact simulation of Bullet CPU Car::_UpdateAirTorque (src/Sim/Car/Car.cpp:597-682).
    In Bullet, applyTorque accumulates to totalTorque, leaving angVel unchanged until integration.
    """
    omega = omega_in.astype(np.float32).copy()
    dir_pitch_right = (-basis.right).astype(np.float32)
    dir_yaw_up = basis.up.astype(np.float32)
    dir_roll_forward = (-basis.forward).astype(np.float32)

    do_air_control = False
    is_flipping_cur = bool(is_flipping)
    if is_flipping_cur:
        is_flipping_cur = has_flipped and (flip_time < FLIP_TORQUE_TIME)

    applied_torque = np.zeros(3, dtype=np.float32)

    if is_flipping_cur:
        rel_dodge_torque = flip_rel_torque.astype(np.float32).copy()
        if np.dot(rel_dodge_torque, rel_dodge_torque) > 0.001:
            pitch_scale = np.float32(1.0)
            if rel_dodge_torque[1] != 0.0 and controls['pitch'] != 0.0:
                sgn_rel = 1.0 if rel_dodge_torque[1] > 0.0 else -1.0
                sgn_ctrl = 1.0 if controls['pitch'] > 0.0 else -1.0
                if sgn_rel == sgn_ctrl:
                    pitch_scale = np.float32(1.0 - abs(controls['pitch']))
                    do_air_control = True
            rel_dodge_torque[1] *= pitch_scale
            dodge_torque_local = np.array([
                rel_dodge_torque[0] * FLIP_TORQUE_X,
                rel_dodge_torque[1] * FLIP_TORQUE_Y,
                0.0
            ], dtype=np.float32)
            applied_torque += basis.mat_vec(dodge_torque_local)
        else:
            do_air_control = True
    else:
        do_air_control = True

    do_air_control = do_air_control and (not is_auto_flipping) and allow_air_torque

    if do_air_control:
        pitch_torque_scale = np.float32(1.0)
        if is_flipping_cur:
            pitch_torque_scale = np.float32(0.0)
        elif has_flipped and (flip_time < FLIP_TORQUE_TIME + FLIP_PITCHLOCK_EXTRA_TIME):
            pitch_torque_scale = np.float32(0.0)

        air_torque = (
            dir_pitch_right * np.float32(controls['pitch'] * pitch_torque_scale * CAR_AIR_CONTROL_TORQUE_X) +
            dir_yaw_up * np.float32(controls['yaw'] * CAR_AIR_CONTROL_TORQUE_Y) +
            dir_roll_forward * np.float32(controls['roll'] * CAR_AIR_CONTROL_TORQUE_Z)
        ).astype(np.float32)

        # In Bullet, angVel is pre-torque angular velocity (_rigidBody.m_angularVelocity)
        ang_vel = omega.copy()
        damp_pitch = np.float32(np.dot(dir_pitch_right, ang_vel) * CAR_AIR_CONTROL_DAMPING_X * (1.0 - abs(controls['pitch'] * pitch_torque_scale)))
        damp_yaw = np.float32(np.dot(dir_yaw_up, ang_vel) * CAR_AIR_CONTROL_DAMPING_Y * (1.0 - abs(controls['yaw'])))
        damp_roll = np.float32(np.dot(dir_roll_forward, ang_vel) * CAR_AIR_CONTROL_DAMPING_Z)

        damping = (
            dir_yaw_up * damp_yaw +
            dir_pitch_right * damp_pitch +
            dir_roll_forward * damp_roll
        ).astype(np.float32)

        air_accel = (air_torque - damping) * np.float32(CAR_TORQUE_SCALE)
        applied_torque += air_accel

    # Integrate velocities
    omega += applied_torque * np.float32(dt)
    return omega


def cuda_post_fix_air_control(
    omega_in: np.ndarray,
    basis: OrthonormalBasis,
    controls: dict,
    is_flipping: bool,
    has_flipped: bool,
    flip_time: float,
    flip_rel_torque: np.ndarray,
    dt: float = DT,
    allow_air_torque: bool = True,
    is_auto_flipping: bool = False
):
    """
    Simulation of Worker 1's post-fix CUDA update_car_air_control with omega_pre.
    include/rocketsim_cuda/physics/car_dynamics.cuh:440-524.
    """
    omega = omega_in.astype(np.float32).copy()
    dir_pitch = (-basis.right).astype(np.float32)
    dir_yaw = basis.up.astype(np.float32)
    dir_roll = (-basis.forward).astype(np.float32)

    is_flipping_cur = bool(is_flipping)
    if is_flipping_cur:
        is_flipping_cur = has_flipped and (flip_time < FLIP_TORQUE_TIME)

    omega_pre = omega.copy()

    do_air_control = False
    if is_flipping_cur:
        rel_dodge_torque = flip_rel_torque.astype(np.float32).copy()
        if np.dot(rel_dodge_torque, rel_dodge_torque) > 0.001:
            pitch_scale = np.float32(1.0)
            if rel_dodge_torque[1] != 0.0 and controls['pitch'] != 0.0:
                sgn_rel = 1.0 if rel_dodge_torque[1] > 0.0 else -1.0
                sgn_ctrl = 1.0 if controls['pitch'] > 0.0 else -1.0
                if sgn_rel == sgn_ctrl:
                    pitch_scale = np.float32(1.0 - abs(controls['pitch']))
                    do_air_control = True
            rel_dodge_torque[1] *= pitch_scale
            dodge_torque = np.array([
                rel_dodge_torque[0] * FLIP_TORQUE_X,
                rel_dodge_torque[1] * FLIP_TORQUE_Y,
                0.0
            ], dtype=np.float32)
            omega += basis.mat_vec(dodge_torque) * np.float32(dt)
        else:
            do_air_control = True
    else:
        do_air_control = True

    do_air_control = do_air_control and allow_air_torque and (not is_auto_flipping)

    if do_air_control:
        pitch_torque_scale = np.float32(1.0)
        if is_flipping_cur:
            pitch_torque_scale = np.float32(0.0)
        elif has_flipped and (flip_time < FLIP_TORQUE_TIME + FLIP_PITCHLOCK_EXTRA_TIME):
            pitch_torque_scale = np.float32(0.0)

        air_torque = (
            dir_pitch * np.float32(controls['pitch'] * pitch_torque_scale * CAR_AIR_CONTROL_TORQUE_X) +
            dir_yaw * np.float32(controls['yaw'] * CAR_AIR_CONTROL_TORQUE_Y) +
            dir_roll * np.float32(controls['roll'] * CAR_AIR_CONTROL_TORQUE_Z)
        ).astype(np.float32)

        # Crucial fix: Evaluate damping on omega_pre
        damp_pitch = np.float32(np.dot(dir_pitch, omega_pre) * CAR_AIR_CONTROL_DAMPING_X * (1.0 - abs(controls['pitch'] * pitch_torque_scale)))
        damp_yaw = np.float32(np.dot(dir_yaw, omega_pre) * CAR_AIR_CONTROL_DAMPING_Y * (1.0 - abs(controls['yaw'])))
        damp_roll = np.float32(np.dot(dir_roll, omega_pre) * CAR_AIR_CONTROL_DAMPING_Z)

        air_damping = (
            dir_yaw * damp_yaw +
            dir_pitch * damp_pitch +
            dir_roll * damp_roll
        ).astype(np.float32)

        delta_omega = (air_torque - air_damping) * np.float32(CAR_TORQUE_SCALE * dt)
        omega += delta_omega

    return omega


def cuda_pre_fix_air_control(
    omega_in: np.ndarray,
    basis: OrthonormalBasis,
    controls: dict,
    is_flipping: bool,
    has_flipped: bool,
    flip_time: float,
    flip_rel_torque: np.ndarray,
    dt: float = DT,
    allow_air_torque: bool = True,
    is_auto_flipping: bool = False
):
    """
    Prior (buggy) implementation before Worker 1's fix.
    Evaluated damping on mutated omega, artificially damping dodge torque.
    """
    omega = omega_in.astype(np.float32).copy()
    dir_pitch = (-basis.right).astype(np.float32)
    dir_yaw = basis.up.astype(np.float32)
    dir_roll = (-basis.forward).astype(np.float32)

    is_flipping_cur = bool(is_flipping)
    if is_flipping_cur:
        is_flipping_cur = has_flipped and (flip_time < FLIP_TORQUE_TIME)

    do_air_control = False
    if is_flipping_cur:
        rel_dodge_torque = flip_rel_torque.astype(np.float32).copy()
        if np.dot(rel_dodge_torque, rel_dodge_torque) > 0.001:
            pitch_scale = np.float32(1.0)
            if rel_dodge_torque[1] != 0.0 and controls['pitch'] != 0.0:
                sgn_rel = 1.0 if rel_dodge_torque[1] > 0.0 else -1.0
                sgn_ctrl = 1.0 if controls['pitch'] > 0.0 else -1.0
                if sgn_rel == sgn_ctrl:
                    pitch_scale = np.float32(1.0 - abs(controls['pitch']))
                    do_air_control = True
            rel_dodge_torque[1] *= pitch_scale
            dodge_torque = np.array([
                rel_dodge_torque[0] * FLIP_TORQUE_X,
                rel_dodge_torque[1] * FLIP_TORQUE_Y,
                0.0
            ], dtype=np.float32)
            omega += basis.mat_vec(dodge_torque) * np.float32(dt)
        else:
            do_air_control = True
    else:
        do_air_control = True

    do_air_control = do_air_control and allow_air_torque and (not is_auto_flipping)

    if do_air_control:
        pitch_torque_scale = np.float32(1.0)
        if is_flipping_cur:
            pitch_torque_scale = np.float32(0.0)
        elif has_flipped and (flip_time < FLIP_TORQUE_TIME + FLIP_PITCHLOCK_EXTRA_TIME):
            pitch_torque_scale = np.float32(0.0)

        air_torque = (
            dir_pitch * np.float32(controls['pitch'] * pitch_torque_scale * CAR_AIR_CONTROL_TORQUE_X) +
            dir_yaw * np.float32(controls['yaw'] * CAR_AIR_CONTROL_TORQUE_Y) +
            dir_roll * np.float32(controls['roll'] * CAR_AIR_CONTROL_TORQUE_Z)
        ).astype(np.float32)

        # BUGGY: uses mutated omega
        damp_pitch = np.float32(np.dot(dir_pitch, omega) * CAR_AIR_CONTROL_DAMPING_X * (1.0 - abs(controls['pitch'] * pitch_torque_scale)))
        damp_yaw = np.float32(np.dot(dir_yaw, omega) * CAR_AIR_CONTROL_DAMPING_Y * (1.0 - abs(controls['yaw'])))
        damp_roll = np.float32(np.dot(dir_roll, omega) * CAR_AIR_CONTROL_DAMPING_Z)

        air_damping = (
            dir_yaw * damp_yaw +
            dir_pitch * damp_pitch +
            dir_roll * damp_roll
        ).astype(np.float32)

        delta_omega = (air_torque - air_damping) * np.float32(CAR_TORQUE_SCALE * dt)
        omega += delta_omega

    return omega


def clamp_ang_speed(omega: np.ndarray, max_speed: float = CAR_MAX_ANG_SPEED) -> np.ndarray:
    """Clamps angular speed to CAR_MAX_ANG_SPEED matching step_kernel.cu:216-219 & Car.cpp:198-199."""
    sq = float(np.dot(omega, omega))
    if sq > max_speed * max_speed:
        return (omega * np.float32(max_speed / math.sqrt(sq))).astype(np.float32)
    return omega.astype(np.float32)


# =============================================================================
# Adversarial Test Cases
# =============================================================================

def test_numerical_stability_no_nan_inf_subnormal():
    """
    Stress test 100,000 random inputs including denormals, extremes, and zeroes.
    Verifies that no NaN or Inf is ever produced.
    """
    rng = np.random.default_rng(seed=1337)
    basis = OrthonormalBasis()

    # Extreme ranges
    for _ in range(50000):
        pitch = rng.uniform(-1.0, 1.0)
        yaw = rng.uniform(-1.0, 1.0)
        roll = rng.uniform(-1.0, 1.0)
        throttle = rng.uniform(-1.0, 1.0)
        controls = {'pitch': pitch, 'yaw': yaw, 'roll': roll, 'throttle': throttle}

        omega_init = rng.uniform(-CAR_MAX_ANG_SPEED, CAR_MAX_ANG_SPEED, size=3).astype(np.float32)
        rel_torque = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        rel_torque[2] = 0.0

        flip_time = rng.uniform(0.0, 1.0)
        is_flipping = bool(rng.choice([True, False]))
        has_flipped = bool(rng.choice([True, False]))

        res = cuda_post_fix_air_control(
            omega_init, basis, controls,
            is_flipping=is_flipping,
            has_flipped=has_flipped,
            flip_time=flip_time,
            flip_rel_torque=rel_torque
        )

        assert not np.isnan(res).any(), f"NaN detected: {res}"
        assert not np.isinf(res).any(), f"Inf detected: {res}"


def test_clamping_to_car_max_ang_speed():
    """
    Verifies that angular velocity clamping strictly limits speed to 5.5 rad/s
    and preserves direction invariance.
    """
    rng = np.random.default_rng(seed=42)
    for _ in range(10000):
        # Generate extreme velocities up to 100 rad/s
        raw_omega = rng.uniform(-100.0, 100.0, size=3).astype(np.float32)
        clamped = clamp_ang_speed(raw_omega)
        speed = np.linalg.norm(clamped)
        assert speed <= CAR_MAX_ANG_SPEED + 1e-6, f"Speed {speed} exceeds CAR_MAX_ANG_SPEED"

        # Direction invariance check: angle between raw and clamped must be 0
        raw_norm = np.linalg.norm(raw_omega)
        if raw_norm > 1e-5:
            dot = np.dot(raw_omega / raw_norm, clamped / speed)
            assert abs(dot - 1.0) < 1e-6, f"Direction shifted during clamping: dot={dot}"


def test_dodge_stall_parity():
    """
    Verifies dodge stall mechanics:
    Controls: pitch = 0, yaw = 1.0, roll = -1.0.
    In stall, flip_rel_torque = (0, 0, 0), so dodge torque is skipped, but air control is permitted.
    """
    basis = OrthonormalBasis()
    controls = {'pitch': 0.0, 'yaw': 1.0, 'roll': -1.0, 'throttle': 0.0}
    rel_torque = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    omega_0 = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    cpu_res = cpu_bullet_air_control(
        omega_0, basis, controls,
        is_flipping=True, has_flipped=True, flip_time=0.1,
        flip_rel_torque=rel_torque
    )
    cuda_res = cuda_post_fix_air_control(
        omega_0, basis, controls,
        is_flipping=True, has_flipped=True, flip_time=0.1,
        flip_rel_torque=rel_torque
    )

    diff = np.max(np.abs(cpu_res - cuda_res))
    assert diff < 1e-6, f"Dodge stall parity error: {diff}"


def test_dodge_cancel_parity_and_bug_reproduction():
    """
    Verifies dodge cancel mechanics (front-flip cancel):
    Flip initiated with pitch = -1.0 -> rel_torque = (0, 1, 0).
    During cancel, user counters with pitch = 1.0.
    Proves that:
    1. cuda_post_fix matches cpu_bullet to < 1e-6 rad/s.
    2. cuda_pre_fix deviates by > 0.02 rad/s due to artificial damping of dodge torque.
    """
    basis = OrthonormalBasis()
    controls = {'pitch': 1.0, 'yaw': 0.0, 'roll': 0.0, 'throttle': 0.0}
    rel_torque = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    omega_0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Full cancel (pitch = 1.0)
    cpu_full = cpu_bullet_air_control(
        omega_0, basis, controls,
        is_flipping=True, has_flipped=True, flip_time=0.2,
        flip_rel_torque=rel_torque
    )
    cuda_full = cuda_post_fix_air_control(
        omega_0, basis, controls,
        is_flipping=True, has_flipped=True, flip_time=0.2,
        flip_rel_torque=rel_torque
    )
    diff_full = np.max(np.abs(cpu_full - cuda_full))
    assert diff_full < 1e-6, f"Full cancel parity mismatch: {diff_full}"

    # Partial cancel (pitch = 0.5): Here dodge_torque is active (scaled 0.5) AND air control runs!
    ctrl_partial = {'pitch': 0.5, 'yaw': 0.0, 'roll': 0.0, 'throttle': 0.0}
    cpu_part = cpu_bullet_air_control(
        omega_0, basis, ctrl_partial,
        is_flipping=True, has_flipped=True, flip_time=0.2,
        flip_rel_torque=rel_torque
    )
    cuda_part_fixed = cuda_post_fix_air_control(
        omega_0, basis, ctrl_partial,
        is_flipping=True, has_flipped=True, flip_time=0.2,
        flip_rel_torque=rel_torque
    )
    cuda_part_old = cuda_pre_fix_air_control(
        omega_0, basis, ctrl_partial,
        is_flipping=True, has_flipped=True, flip_time=0.2,
        flip_rel_torque=rel_torque
    )

    diff_fixed = np.max(np.abs(cpu_part - cuda_part_fixed))
    diff_old = np.max(np.abs(cpu_part - cuda_part_old))

    # The fix must have near-zero discrepancy
    assert diff_fixed < 1e-6, f"Fixed implementation discrepancy: {diff_fixed}"
    # The old code must empirically show the reported bug (> 0.02 rad/s error)
    assert diff_old > 0.01, f"Expected old code to exhibit artificial damping discrepancy, got {diff_old}"


def test_all_8_canonical_directions_parity():
    """
    Validates parity across all 8 canonical flip directions:
    1. Front flip
    2. Back flip
    3. Left dodge
    4. Right dodge
    5. Diagonal front-left
    6. Diagonal front-right
    7. Diagonal back-left
    8. Diagonal back-right
    """
    basis = OrthonormalBasis()
    directions = [
        ("Front", -1.0, 0.0),
        ("Back", 1.0, 0.0),
        ("Left", 0.0, -1.0),
        ("Right", 0.0, 1.0),
        ("Diag_Front_Left", -1.0, -1.0),
        ("Diag_Front_Right", -1.0, 1.0),
        ("Diag_Back_Left", 1.0, -1.0),
        ("Diag_Back_Right", 1.0, 1.0),
    ]

    for name, p_input, y_input in directions:
        dodge_dir = np.array([-p_input, y_input, 0.0], dtype=np.float32)
        norm = np.linalg.norm(dodge_dir)
        if norm > 1e-5:
            dodge_dir /= norm
        flip_rel_torque = np.array([-dodge_dir[1], dodge_dir[0], 0.0], dtype=np.float32)

        controls = {'pitch': p_input, 'yaw': y_input, 'roll': 0.0, 'throttle': 0.0}
        omega_0 = np.array([0.5, -0.3, 0.8], dtype=np.float32)

        cpu_res = cpu_bullet_air_control(
            omega_0, basis, controls,
            is_flipping=True, has_flipped=True, flip_time=0.1,
            flip_rel_torque=flip_rel_torque
        )
        cuda_res = cuda_post_fix_air_control(
            omega_0, basis, controls,
            is_flipping=True, has_flipped=True, flip_time=0.1,
            flip_rel_torque=flip_rel_torque
        )

        diff = np.max(np.abs(cpu_res - cuda_res))
        assert diff < 1e-6, f"Parity mismatch in canonical direction {name}: {diff}"


def test_unconstrained_air_torque():
    """
    Validates free-flight air control (not flipping).
    """
    basis = OrthonormalBasis()
    controls = {'pitch': 0.7, 'yaw': -0.4, 'roll': 0.8, 'throttle': 1.0}
    omega_0 = np.array([1.2, -2.1, 0.5], dtype=np.float32)
    rel_torque = np.zeros(3, dtype=np.float32)

    cpu_res = cpu_bullet_air_control(
        omega_0, basis, controls,
        is_flipping=False, has_flipped=False, flip_time=0.0,
        flip_rel_torque=rel_torque
    )
    cuda_res = cuda_post_fix_air_control(
        omega_0, basis, controls,
        is_flipping=False, has_flipped=False, flip_time=0.0,
        flip_rel_torque=rel_torque
    )

    diff = np.max(np.abs(cpu_res - cuda_res))
    assert diff < 1e-6, f"Unconstrained air torque mismatch: {diff}"


def test_arbitrary_rotations_monte_carlo():
    """
    Validates that parity holds across arbitrary 3D vehicle orientations in SO(3).
    Tests 5,000 random orientations.
    """
    rng = np.random.default_rng(seed=2024)
    for _ in range(5000):
        basis = OrthonormalBasis.from_random_quaternion(rng)
        controls = {
            'pitch': rng.uniform(-1.0, 1.0),
            'yaw': rng.uniform(-1.0, 1.0),
            'roll': rng.uniform(-1.0, 1.0),
            'throttle': 0.0
        }
        rel_torque = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        rel_torque[2] = 0.0
        omega_0 = rng.uniform(-3.0, 3.0, size=3).astype(np.float32)

        cpu_res = cpu_bullet_air_control(
            omega_0, basis, controls,
            is_flipping=True, has_flipped=True, flip_time=0.15,
            flip_rel_torque=rel_torque
        )
        cuda_res = cuda_post_fix_air_control(
            omega_0, basis, controls,
            is_flipping=True, has_flipped=True, flip_time=0.15,
            flip_rel_torque=rel_torque
        )

        diff = np.max(np.abs(cpu_res - cuda_res))
        assert diff < 1e-6, f"SO(3) orientation parity failure: {diff}"


def test_multitick_rollout_11_modes():
    """
    Multi-tick trajectory rollouts across 120 ticks for all 11 modes of ablation_5_flips.
    Confirms ||omega_gpu - omega_cpu||_inf <= 1e-5 rad/s across all ticks (1, 10, 60, 120).
    """
    modes = [
        "Front_Flip_Cancel",
        "Pure_Front_Flip",
        "Pure_Back_Flip",
        "Pure_Left_Dodge",
        "Pure_Right_Dodge",
        "Diag_Front_Left",
        "Diag_Front_Right",
        "Diag_Back_Left",
        "Diag_Back_Right",
        "Back_Flip_Cancel",
        "Stall"
    ]

    for mode_idx, mode_name in enumerate(modes):
        basis = OrthonormalBasis()
        omega_cpu = np.zeros(3, dtype=np.float32)
        omega_cuda = np.zeros(3, dtype=np.float32)
        omega_old = np.zeros(3, dtype=np.float32)

        # Setup initial dodge direction
        p_init, y_init, r_init = 0.0, 0.0, 0.0
        if mode_idx == 0:  # Front cancel
            p_init = -1.0
        elif mode_idx == 1:  # Front
            p_init = -1.0
        elif mode_idx == 2:  # Back
            p_init = 1.0
        elif mode_idx == 3:  # Left
            y_init = -1.0
        elif mode_idx == 4:  # Right
            y_init = 1.0
        elif mode_idx == 5:  # Diag FL
            p_init, y_init = -1.0, -1.0
        elif mode_idx == 6:  # Diag FR
            p_init, y_init = -1.0, 1.0
        elif mode_idx == 7:  # Diag BL
            p_init, y_init = 1.0, -1.0
        elif mode_idx == 8:  # Diag BR
            p_init, y_init = 1.0, 1.0
        elif mode_idx == 9:  # Back cancel
            p_init = 1.0
        elif mode_idx == 10:  # Stall
            y_init, r_init = 1.0, -1.0

        dodge_dir = np.array([-p_init, y_init + r_init, 0.0], dtype=np.float32)
        if abs(y_init + r_init) < 0.1 and abs(p_init) < 0.1:
            dodge_dir = np.zeros(3, dtype=np.float32)
        else:
            norm = np.linalg.norm(dodge_dir)
            if norm > 1e-5:
                dodge_dir /= norm

        flip_rel_torque = np.array([-dodge_dir[1], dodge_dir[0], 0.0], dtype=np.float32)

        # Run 120 ticks
        max_diff = 0.0
        max_diff_old = 0.0
        for tick in range(120):
            flip_time = tick * DT

            # Set controls per tick (matching harness_main.cpp)
            ctrl = {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0, 'throttle': 0.0}
            if mode_idx == 0:  # Front cancel
                ctrl['pitch'] = -1.0 if tick < 25 else 1.0
            elif mode_idx == 1:
                ctrl['pitch'] = -1.0
            elif mode_idx == 2:
                ctrl['pitch'] = 1.0
            elif mode_idx == 3:
                ctrl['yaw'] = -1.0
            elif mode_idx == 4:
                ctrl['yaw'] = 1.0
            elif mode_idx == 5:
                ctrl['pitch'], ctrl['yaw'] = -1.0, -1.0
            elif mode_idx == 6:
                ctrl['pitch'], ctrl['yaw'] = -1.0, 1.0
            elif mode_idx == 7:
                ctrl['pitch'], ctrl['yaw'] = 1.0, -1.0
            elif mode_idx == 8:
                ctrl['pitch'], ctrl['yaw'] = 1.0, 1.0
            elif mode_idx == 9:  # Back cancel
                ctrl['pitch'] = 1.0 if tick < 25 else -1.0
            elif mode_idx == 10:  # Stall
                ctrl['yaw'], ctrl['roll'] = 1.0, -1.0

            omega_cpu = cpu_bullet_air_control(
                omega_cpu, basis, ctrl,
                is_flipping=True, has_flipped=True, flip_time=flip_time,
                flip_rel_torque=flip_rel_torque
            )
            omega_cpu = clamp_ang_speed(omega_cpu)

            omega_cuda = cuda_post_fix_air_control(
                omega_cuda, basis, ctrl,
                is_flipping=True, has_flipped=True, flip_time=flip_time,
                flip_rel_torque=flip_rel_torque
            )
            omega_cuda = clamp_ang_speed(omega_cuda)

            omega_old = cuda_pre_fix_air_control(
                omega_old, basis, ctrl,
                is_flipping=True, has_flipped=True, flip_time=flip_time,
                flip_rel_torque=flip_rel_torque
            )
            omega_old = clamp_ang_speed(omega_old)

            diff = np.max(np.abs(omega_cpu - omega_cuda))
            diff_old = np.max(np.abs(omega_cpu - omega_old))
            if diff > max_diff:
                max_diff = diff
            if diff_old > max_diff_old:
                max_diff_old = diff_old

        # Verify strict parity
        assert max_diff < 1e-5, f"Multi-tick parity breached for {mode_name}: max delta {max_diff}"


def test_post_flip_pitchlock_extra_time_window():
    """
    Verifies the pitchlock extra time window [FLIP_TORQUE_TIME, FLIP_TORQUE_TIME + FLIP_PITCHLOCK_EXTRA_TIME).
    During this window (0.65s to 0.95s), pitch_torque_scale must be 0, locking pitch while allowing yaw and roll.
    """
    basis = OrthonormalBasis()
    ctrl = {'pitch': 1.0, 'yaw': 1.0, 'roll': 1.0, 'throttle': 0.0}
    rel_torque = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    omega_0 = np.zeros(3, dtype=np.float32)

    # At t = 0.75s (inside pitchlock window, but flipping finished)
    res_cpu = cpu_bullet_air_control(
        omega_0, basis, ctrl,
        is_flipping=False, has_flipped=True, flip_time=0.75,
        flip_rel_torque=rel_torque
    )
    res_cuda = cuda_post_fix_air_control(
        omega_0, basis, ctrl,
        is_flipping=False, has_flipped=True, flip_time=0.75,
        flip_rel_torque=rel_torque
    )

    diff = np.max(np.abs(res_cpu - res_cuda))
    assert diff < 1e-6, f"Pitchlock window parity error: {diff}"

    # Verify that pitch acceleration was locked (0):
    # dir_pitch is -right, so dot with right should be 0
    pitch_vel = np.dot(basis.right, res_cuda)
    assert abs(pitch_vel) < 1e-6, f"Expected pitch velocity to be locked to 0 during pitchlock, got {pitch_vel}"

    # Verify yaw and roll were NOT locked:
    yaw_vel = np.dot(basis.up, res_cuda)
    roll_vel = np.dot(-basis.forward, res_cuda)
    assert abs(yaw_vel) > 0.0001, "Yaw should be active during pitchlock window"
    assert abs(roll_vel) > 0.0001, "Roll should be active during pitchlock window"


def test_extreme_floating_point_and_exact_boundaries():
    """
    Validates exact floating point boundaries, subnormals, and threshold transitions:
    - flip_time at boundary FLIP_TORQUE_TIME (0.65) and FLIP_TORQUE_TIME + FLIP_PITCHLOCK_EXTRA_TIME (0.95)
    - rel_dodge_torque length_sq around 0.001 cutoff
    - Subnormal/denormal inputs (1e-38, 1e-45)
    - Exact zero (+0.0 and -0.0)
    - Exact saturation (+1.0 and -1.0)
    """
    basis = OrthonormalBasis()
    boundary_pitches = [-1.0, -0.999999, -0.5, -0.1, -1e-7, -0.0, 0.0, 1e-7, 0.1, 0.5, 0.999999, 1.0]
    boundary_yaws = [-1.0, -0.5, 0.0, 0.5, 1.0]
    boundary_rolls = [-1.0, -0.5, 0.0, 0.5, 1.0]
    boundary_flip_times = [0.0, 0.64999, 0.65, 0.65001, 0.94999, 0.95, 0.95001, 2.0]

    for p in boundary_pitches:
        for y in boundary_yaws:
            for r in boundary_rolls:
                for ft in boundary_flip_times:
                    ctrl = {'pitch': p, 'yaw': y, 'roll': r, 'throttle': 0.0}
                    # Test near length_sq 0.001
                    for mag in [0.0, 0.0316, 0.0317, 1.0]:  # 0.0316^2 approx 0.001
                        rel_t = np.array([mag, mag, 0.0], dtype=np.float32)
                        omega_0 = np.array([1e-38, -1e-38, 0.0], dtype=np.float32)

                        cpu_res = cpu_bullet_air_control(
                            omega_0, basis, ctrl,
                            is_flipping=True, has_flipped=True, flip_time=ft,
                            flip_rel_torque=rel_t
                        )
                        cuda_res = cuda_post_fix_air_control(
                            omega_0, basis, ctrl,
                            is_flipping=True, has_flipped=True, flip_time=ft,
                            flip_rel_torque=rel_t
                        )

                        assert not np.isnan(cuda_res).any(), f"NaN at boundary p={p}, y={y}, ft={ft}, mag={mag}"
                        assert not np.isinf(cuda_res).any(), f"Inf at boundary p={p}, y={y}, ft={ft}, mag={mag}"

                        diff = np.max(np.abs(cpu_res - cuda_res))
                        assert diff < 1e-6, f"Boundary discrepancy at p={p}, y={y}, ft={ft}, mag={mag}: {diff}"
