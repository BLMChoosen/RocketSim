"""
Unit and physical parity test suite for GPU Car-Ball OBB vs Sphere Collision Resolution in RocketSim-CUDA (Milestone 4.3).

Validates:
1. Front bumper hit (ball accelerates forward, car decelerates).
2. Side hit (ball deflects laterally along contact normal).
3. Roof hit (ball bounces upward, reversing downward velocity).
4. High-speed hit (piecewise extra hit impulse applied, velocities clamped <= 6000 UU/s for ball, <= 2300 UU/s for car).
5. Hit state tracking (`ball_hit_is_valid` flag populated on impact, 0 when separated).
"""

import os
import sys
import math
import pytest

# Ensure build directory and bindings directory are in sys.path
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "bindings"))
python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
for d in [build_dir, src_dir, python_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

import rocketsim_cuda
from gym_env import RocketSimBatchedEnv

try:
    import torch
    _TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _TORCH_AVAILABLE = False


def _make_actions(env, num_envs=None, cars_per_env=None):
    """Helper to allocate a zeroed actions tensor on GPU."""
    ne = env.num_envs if num_envs is None else num_envs
    nc = env.cars_per_env if cars_per_env is None else cars_per_env
    if _TORCH_AVAILABLE:
        return torch.zeros((ne * nc, 8), device="cuda", dtype=torch.float32)
    else:
        return rocketsim_cuda.zeros([ne * nc, 8], dtype="float32")


def test_front_bumper_hit():
    """
    Test 1: Front bumper hit.
    Car at origin facing +X moving forward into stationary ball.
    Ball must accelerate forward in +X direction, and car must decelerate.
    """
    num_envs = 1
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    car_obs = env.get_car_observations()
    ball_obs = env.get_ball_observations()

    # Position car airborne at z=500.0, facing +X (identity quat), moving forward at 1000 UU/s
    car_obs[0, 0, 0] = 0.0       # pos_x
    car_obs[0, 0, 1] = 0.0       # pos_y
    car_obs[0, 0, 2] = 500.0     # pos_z
    car_obs[0, 0, 3] = 1000.0    # vel_x
    car_obs[0, 0, 4] = 0.0       # vel_y
    car_obs[0, 0, 5] = 0.0       # vel_z
    car_obs[0, 0, 6] = 1.0       # q_w
    car_obs[0, 0, 7] = 0.0       # q_x
    car_obs[0, 0, 8] = 0.0       # q_y
    car_obs[0, 0, 9] = 0.0       # q_z
    car_obs[0, 0, 10] = 0.0      # ang_vel_x
    car_obs[0, 0, 11] = 0.0      # ang_vel_y
    car_obs[0, 0, 12] = 0.0      # ang_vel_z

    # Position ball in front of car:
    # Octane front bumper is at hitbox_center.x + half_extents.x = 13.88 + 60.25 = 74.13
    # Ball radius = 91.25. Setting ball at X = 150.0 puts it 75.87 UU from front bumper (penetration ~15.38 UU)
    ball_obs[0, 0] = 150.0       # pos_x
    ball_obs[0, 1] = 0.0         # pos_y
    ball_obs[0, 2] = 520.755     # pos_z (aligned with hitbox center z = 500 + 20.755)
    ball_obs[0, 3] = 0.0         # vel_x
    ball_obs[0, 4] = 0.0         # vel_y
    ball_obs[0, 5] = 0.0         # vel_z
    ball_obs[0, 6] = 1.0         # q_w
    ball_obs[0, 7] = 0.0         # q_x
    ball_obs[0, 8] = 0.0         # q_y
    ball_obs[0, 9] = 0.0         # q_z
    ball_obs[0, 10] = 0.0        # ang_vel_x
    ball_obs[0, 11] = 0.0        # ang_vel_y
    ball_obs[0, 12] = 0.0        # ang_vel_z

    # Step simulation 1 tick
    env.sim.step(0)

    new_car_obs = env.get_car_observations()
    new_ball_obs = env.get_ball_observations()

    ball_vel_x = float(new_ball_obs[0, 3])
    car_vel_x = float(new_car_obs[0, 0, 3])

    # Ball must accelerate forward strongly (> 800 UU/s from bilateral impulse + extra hit impulse)
    assert ball_vel_x > 800.0, f"Ball vel_x did not accelerate forward! ball_vel_x = {ball_vel_x}"

    # Car must decelerate from 1000 UU/s (< 950 UU/s)
    assert car_vel_x < 950.0, f"Car vel_x did not decelerate! car_vel_x = {car_vel_x}"

    env.close()


def test_side_hit():
    """
    Test 2: Side hit.
    Car at origin facing +X moving laterally (+Y) into stationary ball on its right side.
    Ball must deflect laterally (+Y), and car must decelerate in +Y.
    """
    num_envs = 1
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    car_obs = env.get_car_observations()
    ball_obs = env.get_ball_observations()

    # Position car airborne at z=500.0, moving laterally (+Y) at 800 UU/s
    car_obs[0, 0, 0] = 0.0       # pos_x
    car_obs[0, 0, 1] = 0.0       # pos_y
    car_obs[0, 0, 2] = 500.0     # pos_z
    car_obs[0, 0, 3] = 0.0       # vel_x
    car_obs[0, 0, 4] = 800.0     # vel_y
    car_obs[0, 0, 5] = 0.0       # vel_z
    car_obs[0, 0, 6] = 1.0       # q_w
    car_obs[0, 0, 7] = 0.0
    car_obs[0, 0, 8] = 0.0
    car_obs[0, 0, 9] = 0.0

    # Right face of car hitbox is at Y = 43.35. Ball placed at Y = 120.0 (dist = 76.65 < 91.25)
    ball_obs[0, 0] = 13.8757     # pos_x
    ball_obs[0, 1] = 120.0       # pos_y
    ball_obs[0, 2] = 520.755     # pos_z
    ball_obs[0, 3] = 0.0
    ball_obs[0, 4] = 0.0
    ball_obs[0, 5] = 0.0

    # Step simulation 1 tick
    env.sim.step(0)

    new_car_obs = env.get_car_observations()
    new_ball_obs = env.get_ball_observations()

    ball_vel_y = float(new_ball_obs[0, 4])
    car_vel_y = float(new_car_obs[0, 0, 4])

    # Ball must deflect laterally in +Y direction (> 600 UU/s)
    assert ball_vel_y > 600.0, f"Ball vel_y did not deflect laterally! ball_vel_y = {ball_vel_y}"

    # Car lateral velocity must decelerate (< 750 UU/s)
    assert car_vel_y < 750.0, f"Car vel_y did not decelerate! car_vel_y = {car_vel_y}"

    env.close()


def test_roof_hit():
    """
    Test 3: Roof hit.
    Ball placed directly above car roof and moving downward (-Z) onto car.
    Ball must bounce upward (+Z), reversing downward velocity.
    """
    num_envs = 1
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    car_obs = env.get_car_observations()
    ball_obs = env.get_ball_observations()

    # Settle car on ground
    for _ in range(60):
        env.sim.step(0)

    # Position ball right above roof (hitbox top z = car_z + 20.755 + 19.33 ~ car_z + 40.1)
    car_z = float(car_obs[0, 0, 2])
    ball_obs[0, 0] = float(car_obs[0, 0, 0]) + 13.8757 # pos_x
    ball_obs[0, 1] = float(car_obs[0, 0, 1])            # pos_y
    ball_obs[0, 2] = car_z + 40.1 + 80.0                # pos_z (penetration ~11.25 UU)
    ball_obs[0, 3] = 0.0
    ball_obs[0, 4] = 0.0
    ball_obs[0, 5] = -800.0                             # moving down fast

    # Step simulation 1 tick
    env.sim.step(0)

    new_ball_obs = env.get_ball_observations()
    ball_vel_z = float(new_ball_obs[0, 5])

    # Ball vel_z must have reversed from negative to strongly positive (> 300 UU/s)
    assert ball_vel_z > 300.0, f"Ball vel_z did not bounce upward! ball_vel_z = {ball_vel_z}"

    env.close()


def test_high_speed_hit_and_velocity_clamping():
    """
    Test 4: High speed hit and velocity clamping.
    - Tests RocketSim extra hit impulse curve factor along relative speed.
    - Tests strict clamping: ball linear speed <= 6000 UU/s, car linear speed <= 2300 UU/s.
    """
    num_envs = 2
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    car_obs = env.get_car_observations()
    ball_obs = env.get_ball_observations()

    # Env 0: Car supersonic impact at 2200 UU/s into stationary ball
    car_obs[0, 0, 0] = 0.0
    car_obs[0, 0, 1] = 0.0
    car_obs[0, 0, 2] = 500.0
    car_obs[0, 0, 3] = 2200.0    # vel_x
    car_obs[0, 0, 4] = 0.0
    car_obs[0, 0, 5] = 0.0
    car_obs[0, 0, 6] = 1.0

    ball_obs[0, 0] = 150.0       # pos_x
    ball_obs[0, 1] = 0.0
    ball_obs[0, 2] = 520.755
    ball_obs[0, 3] = 0.0
    ball_obs[0, 4] = 0.0
    ball_obs[0, 5] = 0.0

    # Env 1: Extreme speed collision designed to test clamp caps (6000 UU/s and 2300 UU/s)
    car_obs[1, 0, 0] = 0.0
    car_obs[1, 0, 1] = 0.0
    car_obs[1, 0, 2] = 500.0
    car_obs[1, 0, 3] = 2300.0    # car max speed
    car_obs[1, 0, 4] = 0.0
    car_obs[1, 0, 5] = 0.0
    car_obs[1, 0, 6] = 1.0

    ball_obs[1, 0] = 150.0
    ball_obs[1, 1] = 0.0
    ball_obs[1, 2] = 520.755
    ball_obs[1, 3] = 5800.0      # ball near speed limit moving same direction
    ball_obs[1, 4] = 0.0
    ball_obs[1, 5] = 0.0

    # Step simulation 1 tick
    env.sim.step(0)

    new_car_obs = env.get_car_observations()
    new_ball_obs = env.get_ball_observations()

    # Env 0: High-speed hit produces large impulse + extra hit impulse
    ball_speed_0 = math.sqrt(
        float(new_ball_obs[0, 3])**2 +
        float(new_ball_obs[0, 4])**2 +
        float(new_ball_obs[0, 5])**2
    )
    assert ball_speed_0 > 2500.0, f"Extra hit impulse not applied! ball_speed_0 = {ball_speed_0}"
    assert ball_speed_0 <= 6000.0 + 1e-3, f"Ball speed exceeded 6000 UU/s! ball_speed_0 = {ball_speed_0}"

    # Env 1: Absolute speed clamping verification
    ball_speed_1 = math.sqrt(
        float(new_ball_obs[1, 3])**2 +
        float(new_ball_obs[1, 4])**2 +
        float(new_ball_obs[1, 5])**2
    )
    car_speed_1 = math.sqrt(
        float(new_car_obs[1, 0, 3])**2 +
        float(new_car_obs[1, 0, 4])**2 +
        float(new_car_obs[1, 0, 5])**2
    )

    assert ball_speed_1 <= 6000.0 + 1e-3, f"Ball speed exceeded BALL_MAX_SPEED: {ball_speed_1}"
    assert car_speed_1 <= 2300.0 + 1e-3, f"Car speed exceeded CAR_MAX_SPEED: {car_speed_1}"

    env.close()


def test_ball_hit_is_valid_flag():
    """
    Test 5: Hit state tracking.
    - When car and ball are far apart: ball_hit_is_valid must be 0.
    - When impact occurs: ball_hit_is_valid must transition to 1.
    """
    num_envs = 2
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    car_obs = env.get_car_observations()
    ball_obs = env.get_ball_observations()

    # Env 0: Far apart (no collision)
    car_obs[0, 0, 0] = -2000.0
    car_obs[0, 0, 1] = 0.0
    car_obs[0, 0, 2] = 500.0
    car_obs[0, 0, 3] = 0.0
    car_obs[0, 0, 4] = 0.0
    car_obs[0, 0, 5] = 0.0

    ball_obs[0, 0] = 2000.0
    ball_obs[0, 1] = 0.0
    ball_obs[0, 2] = 500.0
    ball_obs[0, 3] = 0.0
    ball_obs[0, 4] = 0.0
    ball_obs[0, 5] = 0.0

    # Env 1: In collision contact
    car_obs[1, 0, 0] = 0.0
    car_obs[1, 0, 1] = 0.0
    car_obs[1, 0, 2] = 500.0
    car_obs[1, 0, 3] = 500.0
    car_obs[1, 0, 4] = 0.0
    car_obs[1, 0, 5] = 0.0

    ball_obs[1, 0] = 150.0
    ball_obs[1, 1] = 0.0
    ball_obs[1, 2] = 520.755
    ball_obs[1, 3] = 0.0
    ball_obs[1, 4] = 0.0
    ball_obs[1, 5] = 0.0

    # Check flags before step
    hit_valid_pre = env.get_ball_hit_is_valid()
    assert int(hit_valid_pre[0, 0]) == 0, "Env 0 hit_valid should be 0 before step"
    assert int(hit_valid_pre[1, 0]) == 0, "Env 1 hit_valid should be 0 before step"

    # Step simulation 1 tick
    env.sim.step(0)

    # Check flags after step
    hit_valid_post = env.get_ball_hit_is_valid()
    assert int(hit_valid_post[0, 0]) == 0, "Env 0 (separated) hit_valid must remain 0"
    assert int(hit_valid_post[1, 0]) == 1, "Env 1 (impact) hit_valid must be populated to 1"

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
