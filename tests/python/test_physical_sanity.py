"""
Absolute physical sanity tests for RocketSim-CUDA.
Validates analytical physical laws directly on GPU simulation without relying on CPU oracle:
(a) Idle car suspension rest height Z ~ 17.03 UU.
(b) Ball free-fall under gravity g = -650 UU/s^2.
(c) Maximum drive speeds: ~1410 UU/s (throttle only) and ~2300 UU/s (supersonic boost).
(d) High airborne ball at Z=1000 falls under gravity (no freeze/sleep).
"""

import pytest
import sys
import os
import math

# Ensure project bindings are in path
for d in ["src/bindings", "build", "python"]:
    p = os.path.abspath(d)
    if p not in sys.path:
        sys.path.insert(0, p)

import rocketsim_cuda
from gym_env import RocketSimBatchedEnv


def test_idle_rest_height():
    """Verify car idle suspension settles into analytical equilibrium at Z = 17.03 UU."""
    env = RocketSimBatchedEnv(num_envs=1, cars_per_env=1, tick_skip=1, use_torch=False)
    env.reset()
    car_obs = env.get_car_observations()

    # Drop from spawn height Z=35.95 onto flat floor away from ball at (1000, 1000)
    car_obs[0, 0, 0] = 1000.0
    car_obs[0, 0, 1] = 1000.0
    car_obs[0, 0, 2] = 35.9549
    car_obs[0, 0, 3] = 0.0
    car_obs[0, 0, 4] = 0.0
    car_obs[0, 0, 5] = 0.0

    actions = rocketsim_cuda.zeros([1, 8], dtype="float32")

    # Step for 120 ticks (1.0 s)
    for _ in range(120):
        env.step(actions)

    z_120 = float(car_obs[0, 0, 2])
    vz_120 = float(car_obs[0, 0, 5])
    assert abs(z_120 - 17.03) < 0.01, f"Idle car rest height Z={z_120:.4f} diverges from 17.03 UU!"
    assert abs(vz_120) < 0.01, f"Idle car vertical velocity VZ={vz_120:.4f} is not stationary!"

    # Step up to 600 ticks (5.0 s) to ensure permanent stable equilibrium
    for _ in range(480):
        env.step(actions)

    z_600 = float(car_obs[0, 0, 2])
    vz_600 = float(car_obs[0, 0, 5])
    assert abs(z_600 - 17.03) < 0.01, f"Idle car rest height Z={z_600:.4f} drifted at 600 ticks!"
    assert abs(vz_600) < 0.01, f"Idle car vertical velocity VZ={vz_600:.4f} is not stationary at 600 ticks!"
    env.close()


def test_ball_freefall_gravity():
    """Verify free-falling ball follows g = -650 UU/s^2."""
    env = RocketSimBatchedEnv(num_envs=1, cars_per_env=1, tick_skip=1, use_torch=False)
    env.reset()
    ball_obs = env.get_ball_observations()

    # Ball at high Z in mid-arena with zero initial velocity
    ball_obs[0, 0] = 0.0
    ball_obs[0, 1] = 0.0
    ball_obs[0, 2] = 1500.0
    ball_obs[0, 3] = 0.0
    ball_obs[0, 4] = 0.0
    ball_obs[0, 5] = 0.0

    actions = rocketsim_cuda.zeros([1, 8], dtype="float32")

    # Step 12 ticks (0.1 s)
    num_ticks = 12
    dt = 1.0 / 120.0
    for _ in range(num_ticks):
        env.step(actions)

    vz = float(ball_obs[0, 5])
    expected_vz = -650.0 * (num_ticks * dt)  # -65.0 UU/s before minor air drag
    assert abs(vz - expected_vz) < 1.0, f"Ball VZ={vz:.3f} deviates from expected g*t={expected_vz:.3f}"
    env.close()


def test_car_max_speeds():
    """Verify maximum ground drive speed: ~1410 UU/s throttle, ~2300 UU/s boost."""
    env = RocketSimBatchedEnv(num_envs=2, cars_per_env=1, tick_skip=1, use_torch=False)
    env.reset()
    car_obs = env.get_car_observations()

    # Set both cars stationary on ground at X=-3500 facing +X (identity quat)
    for e in range(2):
        car_obs[e, 0, 0] = -3500.0
        car_obs[e, 0, 1] = 0.0
        car_obs[e, 0, 2] = 17.03
        car_obs[e, 0, 3] = 0.0
        car_obs[e, 0, 4] = 0.0
        car_obs[e, 0, 5] = 0.0
        car_obs[e, 0, 13] = 100.0  # Full boost

    actions = rocketsim_cuda.zeros([2, 8], dtype="float32")
    # Env 0: Throttle only (no boost)
    actions[0, 0] = 1.0
    # Env 1: Throttle + Boost
    actions[1, 0] = 1.0
    actions[1, 6] = 1.0

    # Accelerate for 360 ticks (3.0 seconds) to reach terminal throttle speed
    for _ in range(360):
        # Keep boost full in env 1
        car_obs[1, 0, 13] = 100.0
        env.step(actions)

    vx_throttle = float(car_obs[0, 0, 3])
    vy_throttle = float(car_obs[0, 0, 4])
    speed_throttle = math.sqrt(vx_throttle * vx_throttle + vy_throttle * vy_throttle)

    vx_boost = float(car_obs[1, 0, 3])
    vy_boost = float(car_obs[1, 0, 4])
    speed_boost = math.sqrt(vx_boost * vx_boost + vy_boost * vy_boost)

    # Throttle terminal speed without boost is ~1410 UU/s
    assert 1400.0 <= speed_throttle <= 1420.0, f"Throttle terminal speed={speed_throttle:.1f} UU/s not in ~1410 range!"
    # Supersonic max speed with boost is clamped to 2300 UU/s
    assert 2295.0 <= speed_boost <= 2305.0, f"Boosted speed={speed_boost:.1f} UU/s not at 2300 UU/s limit!"
    env.close()


def test_ball_resting_at_height_falls():
    """Verify an airborne ball initialized at Z=1000 falls and does not get stuck/sleep."""
    env = RocketSimBatchedEnv(num_envs=1, cars_per_env=1, tick_skip=1, use_torch=False)
    env.reset()
    ball_obs = env.get_ball_observations()

    ball_obs[0, 0] = 0.0
    ball_obs[0, 1] = 0.0
    ball_obs[0, 2] = 1000.0
    ball_obs[0, 3] = 0.0
    ball_obs[0, 4] = 0.0
    ball_obs[0, 5] = 0.0

    actions = rocketsim_cuda.zeros([1, 8], dtype="float32")

    for _ in range(60):  # 0.5 seconds
        env.step(actions)

    z = float(ball_obs[0, 2])
    vz = float(ball_obs[0, 5])
    assert z < 950.0, f"Ball at Z=1000 failed to fall: Z={z:.2f}"
    assert vz < -250.0, f"Ball at Z=1000 has insufficient downward speed: VZ={vz:.2f}"
    env.close()
