"""
Adversarial Stress Test Suite for RocketSim-CUDA Ball Physics Kernel (Milestone 5.2 Challenger).

Empirically challenges:
1. Extreme Speeds (up to 6000 UU/s) against floor, ceiling, side walls, back walls, 45 deg chamfer.
2. Extreme Spins (up to 6.0 rad/s) and Coulomb friction / torque coupling stability.
3. Corner and Seam Singularities (chamfer seams, 3-way floor/chamfer and ceiling/chamfer corners, goalposts, crossbar).
4. Massive Parallel Monte Carlo Stress (1024 environments x 600 ticks = 614,400 simulation steps).
5. Numerical invariants: absence of NaNs, Infs, boundary tunneling, and kinetic energy explosion.
6. Empirical audit of goal cavity entry vs back wall boundary.
"""

import os
import sys
import math
import random
import pytest

# Ensure build and binding directories are available
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "bindings"))
python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
for d in [build_dir, src_dir, python_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

import rocketsim_cuda
from gym_env import RocketSimBatchedEnv

BALL_RADIUS = 91.25
ARENA_EXTENT_X = 4096.0
ARENA_EXTENT_Y = 5120.0
ARENA_HEIGHT = 2048.0
CORNER_SUM = 8064.0
GOAL_HALF_WIDTH = 892.8
GOAL_HEIGHT = 642.7


def _check_ball_invariants(ball_obs, env_idx=0, initial_max_speed=6500.0):
    """Assert strict numerical and geometric physical invariants for a ball."""
    px = float(ball_obs[env_idx, 0])
    py = float(ball_obs[env_idx, 1])
    pz = float(ball_obs[env_idx, 2])

    vx = float(ball_obs[env_idx, 3])
    vy = float(ball_obs[env_idx, 4])
    vz = float(ball_obs[env_idx, 5])

    qw = float(ball_obs[env_idx, 6])
    qx = float(ball_obs[env_idx, 7])
    qy = float(ball_obs[env_idx, 8])
    qz = float(ball_obs[env_idx, 9])

    wx = float(ball_obs[env_idx, 10])
    wy = float(ball_obs[env_idx, 11])
    wz = float(ball_obs[env_idx, 12])

    # 1. Check for NaN or Inf in all 13 components
    all_vals = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    for i, v in enumerate(all_vals):
        assert not math.isnan(v), f"Env {env_idx} component {i} is NaN!"
        assert not math.isinf(v), f"Env {env_idx} component {i} is Inf!"

    # 2. Check quaternion normalization
    q_len = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    assert abs(q_len - 1.0) < 1e-3, f"Env {env_idx} quaternion norm {q_len:.6f} drifted from 1.0!"

    # 3. Check for kinetic energy explosion
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    assert speed <= initial_max_speed * 1.05, (
        f"Env {env_idx} kinetic energy explosion: speed {speed:.1f} > {initial_max_speed}!"
    )

    # 4. Check tunneling / boundary containment
    # Tolerances allow minor penetration resolution during the discrete collision tick
    tol = 10.0
    assert pz >= -tol, f"Env {env_idx} tunneled below floor: Z={pz:.2f}"
    assert pz <= ARENA_HEIGHT + tol, f"Env {env_idx} tunneled above ceiling: Z={pz:.2f}"

    # Side walls
    assert abs(px) <= ARENA_EXTENT_X + tol, f"Env {env_idx} tunneled past side wall: X={px:.2f}"

    # Back wall (unless in goal cavity)
    in_goal_box = (abs(px) <= GOAL_HALF_WIDTH + tol and pz <= GOAL_HEIGHT + tol)
    if not in_goal_box:
        assert abs(py) <= ARENA_EXTENT_Y + tol, f"Env {env_idx} tunneled past back wall: Y={py:.2f}"
    else:
        assert abs(py) <= 6000.0 + tol, f"Env {env_idx} tunneled past goal back wall: Y={py:.2f}"

    # Corner chamfer
    corner_sum = abs(px) + abs(py)
    assert corner_sum <= CORNER_SUM + tol * 2.0, (
        f"Env {env_idx} tunneled past corner chamfer: |X|+|Y|={corner_sum:.2f} > {CORNER_SUM}"
    )

    return speed, math.sqrt(wx * wx + wy * wy + wz * wz)


def test_ball_extreme_linear_speeds():
    """
    Stress-test high speed impacts at 6000 UU/s (maximum physics speed)
    against floor, ceiling, side walls, back walls, and 45 deg chamfer.
    """
    num_envs = 6
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=1, tick_skip=1, use_torch=False)
    env.reset()
    ball = env.get_ball_observations()

    # Move cars far away to avoid interference
    car = env.get_car_observations()
    for e in range(num_envs):
        car[e, 0, 0] = -3800.0
        car[e, 0, 1] = -4800.0
        car[e, 0, 2] = 17.03

    # Env 0: Floor impact at 6000 UU/s straight down
    ball[0, 0] = 0.0;    ball[0, 1] = 0.0;    ball[0, 2] = 400.0
    ball[0, 3] = 0.0;    ball[0, 4] = 0.0;    ball[0, 5] = -6000.0

    # Env 1: Ceiling impact at 6000 UU/s straight up
    ball[1, 0] = 0.0;    ball[1, 1] = 0.0;    ball[1, 2] = 1600.0
    ball[1, 3] = 0.0;    ball[1, 4] = 0.0;    ball[1, 5] = 6000.0

    # Env 2: Side wall (+X) impact at 6000 UU/s
    ball[2, 0] = 3600.0; ball[2, 1] = 0.0;    ball[2, 2] = 500.0
    ball[2, 3] = 6000.0; ball[2, 4] = 0.0;    ball[2, 5] = 0.0

    # Env 3: Back wall (+Y) impact at 6000 UU/s
    ball[3, 0] = 2000.0; ball[3, 1] = 4600.0; ball[3, 2] = 500.0
    ball[3, 3] = 0.0;    ball[3, 4] = 6000.0; ball[3, 5] = 0.0

    # Env 4: 45 deg Corner Chamfer impact at 6000 UU/s (heading towards X+Y=8064)
    # Target chamfer center: (3520, 4544)
    v4 = 6000.0 / math.sqrt(2.0)
    ball[4, 0] = 3000.0; ball[4, 1] = 4024.0; ball[4, 2] = 500.0
    ball[4, 3] = v4;     ball[4, 4] = v4;     ball[4, 5] = 0.0

    # Env 5: Diagonal Floor-Wall impact at 6000 UU/s (Vx = 4242, Vz = -4242)
    v5 = 6000.0 / math.sqrt(2.0)
    ball[5, 0] = 3600.0; ball[5, 1] = 0.0;    ball[5, 2] = 400.0
    ball[5, 3] = v5;     ball[5, 4] = 0.0;    ball[5, 5] = -v5

    rebounded = [False] * num_envs
    actions = rocketsim_cuda.zeros([num_envs, 8], dtype="float32")

    for tick in range(120):
        env.step(actions)

        for e in range(num_envs):
            _check_ball_invariants(ball, e, initial_max_speed=6000.0)

            # Check rebound detection
            if e == 0 and float(ball[e, 5]) > 0.0: rebounded[e] = True
            if e == 1 and float(ball[e, 5]) < 0.0: rebounded[e] = True
            if e == 2 and float(ball[e, 3]) < 0.0: rebounded[e] = True
            if e == 3 and float(ball[e, 4]) < 0.0: rebounded[e] = True
            if e == 4 and float(ball[e, 3]) < 0.0 and float(ball[e, 4]) < 0.0: rebounded[e] = True
            if e == 5 and float(ball[e, 5]) > 0.0: rebounded[e] = True

    env.close()

    for e in range(num_envs):
        assert rebounded[e], f"Env {e} failed to rebound at 6000 UU/s!"


def test_ball_extreme_spins_and_torque_coupling():
    """
    Stress-test high spin rates up to 6.0 rad/s (the RocketSim spin clamp limit)
    combined with high linear velocities to stress Coulomb friction and torque coupling.
    """
    num_envs = 4
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=1, tick_skip=1, use_torch=False)
    env.reset()
    ball = env.get_ball_observations()

    # Env 0: Angled floor bounce with extreme Topspin (+Wy = 6.0 rad/s)
    ball[0, 0] = 0.0;    ball[0, 1] = 0.0;    ball[0, 2] = 300.0
    ball[0, 3] = 1000.0; ball[0, 4] = 0.0;    ball[0, 5] = -3000.0
    ball[0, 10] = 0.0;   ball[0, 11] = 6.0;   ball[0, 12] = 0.0

    # Env 1: Angled floor bounce with extreme Backspin (-Wy = -6.0 rad/s)
    ball[1, 0] = 0.0;    ball[1, 1] = 0.0;    ball[1, 2] = 300.0
    ball[1, 3] = 1000.0; ball[1, 4] = 0.0;    ball[1, 5] = -3000.0
    ball[1, 10] = 0.0;   ball[1, 11] = -6.0;  ball[1, 12] = 0.0

    # Env 2: Wall bounce with extreme Sidespin (+Wz = 6.0 rad/s)
    ball[2, 0] = 3600.0; ball[2, 1] = 0.0;    ball[2, 2] = 500.0
    ball[2, 3] = 4000.0; ball[2, 4] = 500.0;  ball[2, 5] = 0.0
    ball[2, 10] = 0.0;   ball[2, 11] = 0.0;   ball[2, 12] = 6.0

    # Env 3: Ceiling bounce with multi-axis extreme spin (Wx=Wy=Wz=3.46 rad/s, |W|=6.0)
    w_comp = 6.0 / math.sqrt(3.0)
    ball[3, 0] = 0.0;    ball[3, 1] = 0.0;    ball[3, 2] = 1700.0
    ball[3, 3] = 1500.0; ball[3, 4] = 1500.0; ball[3, 5] = 4000.0
    ball[3, 10] = w_comp; ball[3, 11] = w_comp; ball[3, 12] = w_comp

    actions = rocketsim_cuda.zeros([num_envs, 8], dtype="float32")

    for tick in range(120):
        env.step(actions)
        for e in range(num_envs):
            _check_ball_invariants(ball, e, initial_max_speed=5000.0)

    # Verify physical difference between topspin and backspin
    # Topspin on floor (+Wy) pushes contact point backward, propelling ball forward (+Vx)
    # Backspin on floor (-Wy) pushes contact point forward, decelerating Vx
    vx_topspin = float(ball[0, 3])
    vx_backspin = float(ball[1, 3])
    assert vx_topspin > vx_backspin, (
        f"Coulomb torque coupling failure: topspin Vx ({vx_topspin:.2f}) "
        f"should exceed backspin Vx ({vx_backspin:.2f})!"
    )

    env.close()


def test_ball_corner_and_seam_impacts():
    """
    Stress-test sharp geometric seams and corners at 5000-6000 UU/s:
    1. Seam between Chamfer and Side Wall: (4096, 3968)
    2. Seam between Chamfer and Back Wall: (2944, 5120)
    3. 3-way Floor + Chamfer corner
    4. 3-way Ceiling + Chamfer corner
    5. Goalpost rim: (892.8, 5120)
    6. Crossbar rim: (0, 5120, 642.7)
    """
    num_envs = 6
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=1, tick_skip=1, use_torch=False)
    env.reset()
    ball = env.get_ball_observations()

    # Move cars out of way
    car = env.get_car_observations()
    for e in range(num_envs):
        car[e, 0, 0] = -3800.0; car[e, 0, 1] = -4800.0; car[e, 0, 2] = 17.03

    # Env 0: Chamfer / Side Wall seam (4096, 3968)
    # Shoot straight towards the seam from (3700, 3572, 500)
    dx0 = 4096.0 - 3700.0; dy0 = 3968.0 - 3572.0
    l0 = math.sqrt(dx0 * dx0 + dy0 * dy0)
    ball[0, 0] = 3700.0; ball[0, 1] = 3572.0; ball[0, 2] = 500.0
    ball[0, 3] = (dx0 / l0) * 5500.0; ball[0, 4] = (dy0 / l0) * 5500.0; ball[0, 5] = 0.0

    # Env 1: Chamfer / Back Wall seam (2944, 5120)
    dx1 = 2944.0 - 2500.0; dy1 = 5120.0 - 4676.0
    l1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
    ball[1, 0] = 2500.0; ball[1, 1] = 4676.0; ball[1, 2] = 500.0
    ball[1, 3] = (dx1 / l1) * 5500.0; ball[1, 4] = (dy1 / l1) * 5500.0; ball[1, 5] = 0.0

    # Env 2: 3-way Floor + Chamfer corner: (3520, 4544, 0)
    v2_xy = 3500.0; v2_z = -3500.0
    ball[2, 0] = 3100.0; ball[2, 1] = 4124.0; ball[2, 2] = 400.0
    ball[2, 3] = v2_xy / math.sqrt(2.0); ball[2, 4] = v2_xy / math.sqrt(2.0); ball[2, 5] = v2_z

    # Env 3: 3-way Ceiling + Chamfer corner: (3520, 4544, 2048)
    v3_xy = 3500.0; v3_z = 3500.0
    ball[3, 0] = 3100.0; ball[3, 1] = 4124.0; ball[3, 2] = 1648.0
    ball[3, 3] = v3_xy / math.sqrt(2.0); ball[3, 4] = v3_xy / math.sqrt(2.0); ball[3, 5] = v3_z

    # Env 4: Vertical Goalpost rim: (892.8, 5120, 300)
    ball[4, 0] = 892.8;  ball[4, 1] = 4700.0; ball[4, 2] = 300.0
    ball[4, 3] = 0.0;    ball[4, 4] = 4500.0; ball[4, 5] = 0.0

    # Env 5: Crossbar rim: (0, 5120, 642.7)
    ball[5, 0] = 0.0;    ball[5, 1] = 4700.0; ball[5, 2] = 642.7
    ball[5, 3] = 0.0;    ball[5, 4] = 4500.0; ball[5, 5] = 0.0

    actions = rocketsim_cuda.zeros([num_envs, 8], dtype="float32")

    for tick in range(120):
        env.step(actions)
        for e in range(num_envs):
            _check_ball_invariants(ball, e, initial_max_speed=6000.0)

    env.close()


def test_ball_massive_monte_carlo_stress():
    """
    Massive parallel Monte Carlo stress harness:
    1024 concurrent GPU environments with random initial positions,
    extreme speeds (3000 to 6000 UU/s), and extreme spins (-6.0 to 6.0 rad/s)
    stepped for 600 ticks (over 600,000 physical sub-steps).
    Confirms zero NaNs, zero Infs, zero tunneling, and zero energy explosion.
    """
    num_envs = 1024
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=1, tick_skip=1, use_torch=False)
    env.reset()
    ball = env.get_ball_observations()

    # Move cars away to field corners
    car = env.get_car_observations()
    for e in range(num_envs):
        car[e, 0, 0] = -3800.0; car[e, 0, 1] = -4800.0; car[e, 0, 2] = 17.03

    random.seed(42)

    for e in range(num_envs):
        # Spawn within arena interior
        px = random.uniform(-3000.0, 3000.0)
        py = random.uniform(-4000.0, 4000.0)
        pz = random.uniform(200.0, 1800.0)

        # Random velocity with speed in [3000, 6000] UU/s
        theta = random.uniform(0.0, 2.0 * math.pi)
        phi = random.uniform(-0.4 * math.pi, 0.4 * math.pi)
        speed = random.uniform(3000.0, 6000.0)

        vx = speed * math.cos(phi) * math.cos(theta)
        vy = speed * math.cos(phi) * math.sin(theta)
        vz = speed * math.sin(phi)

        # Random spin in [-6.0, 6.0] rad/s
        wx = random.uniform(-6.0, 6.0)
        wy = random.uniform(-6.0, 6.0)
        wz = random.uniform(-6.0, 6.0)

        ball[e, 0] = px; ball[e, 1] = py; ball[e, 2] = pz
        ball[e, 3] = vx; ball[e, 4] = vy; ball[e, 5] = vz
        ball[e, 10] = wx; ball[e, 11] = wy; ball[e, 12] = wz

    actions = rocketsim_cuda.zeros([num_envs, 8], dtype="float32")

    # Step for 600 ticks (5.0 seconds at 120Hz)
    for tick in range(600):
        env.step(actions)

        # Sample check every 60 ticks across 32 environments for performance
        if tick % 60 == 0:
            for e in range(0, num_envs, 32):
                _check_ball_invariants(ball, e, initial_max_speed=6000.0)

    # Full audit of all 1024 environments at the end
    for e in range(num_envs):
        _check_ball_invariants(ball, e, initial_max_speed=6000.0)

    env.close()


def test_ball_goal_entrance_audit():
    """
    Audits the ball behavior when approaching the goal mouth from inside the field:
    Ball spawned at (0, 4800, 200) moving at +2000 UU/s along Y.
    Reveals that arena_sdf_2d_wall enforces a solid back wall at Y=5120 even within
    the goal width |X| <= 892.8, preventing the ball from entering the net naturally.
    """
    env = RocketSimBatchedEnv(num_envs=1, cars_per_env=1, tick_skip=1, use_torch=False)
    env.reset()
    ball = env.get_ball_observations()

    # Move car out of way
    car = env.get_car_observations()
    car[0, 0, 0] = -3800.0; car[0, 0, 1] = -4800.0; car[0, 0, 2] = 17.03

    ball[0, 0] = 0.0;    ball[0, 1] = 4800.0; ball[0, 2] = 200.0
    ball[0, 3] = 0.0;    ball[0, 4] = 2000.0; ball[0, 5] = 0.0

    actions = rocketsim_cuda.zeros([1, 8], dtype="float32")

    bounced = False
    max_y_reached = 4800.0
    for tick in range(60):
        env.step(actions)
        py = float(ball[0, 1])
        vy = float(ball[0, 4])
        if py > max_y_reached:
            max_y_reached = py
        if vy < 0.0:
            bounced = True
            break

    env.close()

    # The ball bounces off the back wall at Y ~ 5028.75 (5120 - BALL_RADIUS)
    # instead of passing through into the goal cavity (Y in [5120, 6000]).
    assert bounced, "Ball did not bounce!"
    assert max_y_reached < 5120.0, (
        f"Ball penetrated into goal cavity: maxY={max_y_reached:.2f}"
    )
