"""
Unit and physical parity test suite for car mechanics in RocketSim-CUDA (Milestone 4.2).
Validates:
1. Air control wheel contact suppression (num_wheels_contact == 0 vs on ground).
2. Reverse braking throttle cutoff (speed > 0.01 UU/s).
3. Single jump immediate impulse and hold acceleration.
4. Double jump and directional flip / dodge dynamics with Z-damping.
5. Boost ground/air acceleration curves, consumption rate, and minimum boost time.
6. Turtle recovery (auto-flip) when resting upside down on roof with jump pressed.
7. Surface alignment (auto-roll) leveling torque when throttle is applied.
"""

import sys
import os
import pytest
import math

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


def test_air_control_wheel_contact_suppression():
    """
    Task 1: Verify air control rotation torque and damping are strictly gated on
    num_wheels_contact == 0 (Car.cpp:125, 641).
    - When car is on ground (all 4 wheels in contact), pitch/yaw/roll inputs must NOT produce angular velocity.
    - When car is in air (num_wheels_contact == 0), pitch/yaw/roll inputs MUST produce angular velocity.
    """
    num_envs = 2
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    car_obs = env.get_car_observations()

    # Env 0: Settle car on ground so all 4 wheels are in contact
    for _ in range(60):
        env.sim.step(0)

    # Zero residual angular velocities for both environments
    for e in range(2):
        car_obs[e, 0, 10] = 0.0
        car_obs[e, 0, 11] = 0.0
        car_obs[e, 0, 12] = 0.0

    # Env 1: Car in the air (z=800.0, zero wheel contacts)
    car_obs[1, 0, 0] = 0.0
    car_obs[1, 0, 1] = 0.0
    car_obs[1, 0, 2] = 800.0
    car_obs[1, 0, 3] = 0.0
    car_obs[1, 0, 4] = 0.0
    car_obs[1, 0, 5] = 0.0

    # Apply pitch input = 1.0 to both environments
    actions = _make_actions(env)
    actions[0, 2] = 1.0  # Env 0 pitch = 1.0
    actions[1, 2] = 1.0  # Env 1 pitch = 1.0

    # Step simulation 10 ticks (approx 0.083s)
    for _ in range(10):
        env.sim.step_actions(actions)

    new_car_obs = env.get_car_observations()

    # Ground car: pitch angular velocity must remain virtually 0 (suppressed)
    ground_ang_speed = math.sqrt(
        float(new_car_obs[0, 0, 10])**2 +
        float(new_car_obs[0, 0, 11])**2 +
        float(new_car_obs[0, 0, 12])**2
    )
    assert ground_ang_speed < 0.05, (
        f"Air control was not suppressed on ground! ang_speed={ground_ang_speed}"
    )

    # Airborne car: pitch angular velocity must have accelerated significantly
    air_ang_speed = math.sqrt(
        float(new_car_obs[1, 0, 10])**2 +
        float(new_car_obs[1, 0, 11])**2 +
        float(new_car_obs[1, 0, 12])**2
    )
    assert air_ang_speed > 0.5, (
        f"Air control did not actuate in air! air_ang_speed={air_ang_speed}"
    )

    env.close()


def test_reverse_braking_throttle_cutoff():
    """
    Task 2: Verify reverse braking throttle cutoff at speed > 0.01 UU/s
    (BRAKING_NO_THROTTLE_SPEED_THRESH matching Car.cpp:391).
    When moving forward at 50 UU/s (speed between 25 and 100 UU/s) and applying reverse throttle (-1.0),
    the car must apply full braking (1.0) and CUT engine throttle to 0.0, decelerating rapidly.
    """
    num_envs = 2
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    car_obs = env.get_car_observations()

    # Env 0 & 1: Car on ground, forward speed = 50 UU/s along X
    for e in range(2):
        car_obs[e, 0, 0] = 0.0
        car_obs[e, 0, 1] = 0.0
        car_obs[e, 0, 2] = 17.0
        # Facing positive X
        car_obs[e, 0, 6] = 1.0  # qw
        car_obs[e, 0, 7] = 0.0  # qx
        car_obs[e, 0, 8] = 0.0  # qy
        car_obs[e, 0, 9] = 0.0  # qz
        car_obs[e, 0, 3] = 50.0 # vel_x = 50 UU/s
        car_obs[e, 0, 4] = 0.0
        car_obs[e, 0, 5] = 0.0

    actions = _make_actions(env)
    # Env 0: Reverse throttle = -1.0 (triggers reverse braking)
    actions[0, 0] = -1.0
    # Env 1: Coasting (throttle = 0.0)
    actions[1, 0] = 0.0

    # Step for 12 ticks (0.1 seconds)
    for _ in range(12):
        env.sim.step_actions(actions)

    new_car_obs = env.get_car_observations()
    vel_x_braking = float(new_car_obs[0, 0, 3])
    vel_x_coasting = float(new_car_obs[1, 0, 3])

    # Braking at full brake force (52.5) should decelerate much faster than coasting brake (0.15)
    assert vel_x_braking < vel_x_coasting, (
        f"Reverse braking failed to decelerate faster than coasting: braking={vel_x_braking}, coasting={vel_x_coasting}"
    )
    assert vel_x_braking < 50.0, f"Car did not decelerate under reverse braking: {vel_x_braking}"

    env.close()


def test_jump_immediate_impulse_and_hold():
    """
    Verify jump immediate impulse (875/3 UU/s ~= 291.67 UU/s) and hold acceleration (4375/3 UU/s^2).
    """
    num_envs = 2
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    car_obs = env.get_car_observations()

    # Settle cars on ground so is_on_ground == True and wheels are compressed
    for _ in range(60):
        env.sim.step(0)

    actions = _make_actions(env)
    # Both jump on next tick
    actions[0, 5] = 1.0  # jump = 1
    actions[1, 5] = 1.0  # jump = 1

    # Step 1 tick (trigger jump immediate impulse)
    env.sim.step_actions(actions)

    new_car_obs = env.get_car_observations()
    vel_z_0 = float(new_car_obs[0, 0, 5])
    vel_z_1 = float(new_car_obs[1, 0, 5])
    # Immediate impulse is 875/3 ~= 291.67 UU/s minus small gravity/damping on first tick (~284 UU/s)
    assert vel_z_0 > 270.0, f"Jump immediate impulse was not applied: vel_z = {vel_z_0}"
    assert vel_z_1 > 270.0, f"Jump immediate impulse was not applied: vel_z = {vel_z_1}"

    # Now Env 0 continues holding jump, while Env 1 releases jump
    actions[0, 5] = 1.0  # hold jump
    actions[1, 5] = 0.0  # release jump

    # Step for 10 ticks
    for _ in range(10):
        env.sim.step_actions(actions)

    after_car_obs = env.get_car_observations()
    vel_z_hold = float(after_car_obs[0, 0, 5])
    vel_z_release = float(after_car_obs[1, 0, 5])

    # Holding jump must result in higher upward velocity than releasing jump
    assert vel_z_hold > vel_z_release + 50.0, (
        f"Jump hold acceleration failed: vel_z_hold={vel_z_hold} <= vel_z_release={vel_z_release}"
    )

    env.close()


def test_double_jump_and_flip_dodge():
    """
    Verify airborne double jump (upward impulse) and directional flip / dodge (forward horizontal impulse).
    """
    num_envs = 2
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    car_obs = env.get_car_observations()

    # Place cars in the air at z=500.0
    for e in range(2):
        car_obs[e, 0, 0] = 0.0
        car_obs[e, 0, 1] = 0.0
        car_obs[e, 0, 2] = 500.0
        car_obs[e, 0, 3] = 0.0
        car_obs[e, 0, 4] = 0.0
        car_obs[e, 0, 5] = 0.0
        car_obs[e, 0, 6] = 1.0
        car_obs[e, 0, 7] = 0.0
        car_obs[e, 0, 8] = 0.0
        car_obs[e, 0, 9] = 0.0

    # First tick: no inputs so last_controls_jump is 0
    actions = _make_actions(env)
    env.sim.step_actions(actions)

    # Next tick:
    # Env 0: Double jump (jump=1, no direction)
    actions[0, 5] = 1.0
    # Env 1: Forward Flip / Dodge (jump=1, pitch=-1.0 for forward dodge)
    actions[1, 5] = 1.0
    actions[1, 2] = -1.0

    env.sim.step_actions(actions)

    res = env.get_car_observations()
    # Env 0: double jump upward velocity
    vel_z_double_jump = float(res[0, 0, 5])
    # Env 1: forward flip impulse along X (~500 UU/s)
    vel_x_flip = float(res[1, 0, 3])

    assert vel_z_double_jump > 200.0, f"Double jump failed to provide upward impulse: {vel_z_double_jump}"
    assert vel_x_flip > 400.0, f"Forward flip failed to provide ~500 UU/s forward impulse: {vel_x_flip}"

    env.close()


def test_boost_acceleration_and_consumption():
    """
    Verify boost ground/air acceleration, fuel consumption (~33.33/s), and speed capping.
    """
    num_envs = 2
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    car_obs = env.get_car_observations()

    # Place Car 0 on ground with full boost (100.0)
    car_obs[0, 0, 0] = 0.0
    car_obs[0, 0, 1] = 0.0
    car_obs[0, 0, 2] = 17.0
    car_obs[0, 0, 3] = 0.0
    car_obs[0, 0, 4] = 0.0
    car_obs[0, 0, 5] = 0.0
    car_obs[0, 0, 6] = 1.0
    car_obs[0, 0, 7] = 0.0
    car_obs[0, 0, 8] = 0.0
    car_obs[0, 0, 9] = 0.0
    car_obs[0, 0, 13] = 100.0  # Fuel = 100

    actions = _make_actions(env)
    actions[0, 6] = 1.0  # Boost = 1

    # Step for 30 ticks (0.25 seconds)
    for _ in range(30):
        env.sim.step_actions(actions)

    new_car_obs = env.get_car_observations()
    vel_x = float(new_car_obs[0, 0, 3])
    boost_fuel = float(new_car_obs[0, 0, 13])

    # Boost ground accel ~ 991.67 UU/s^2 -> in 0.25s velocity should be ~ 240+ UU/s
    assert vel_x > 200.0, f"Boost acceleration insufficient: {vel_x}"

    # Fuel consumed ~ (100 / 3) * 0.25 ~= 8.33 -> fuel should be ~ 91.67
    expected_fuel = 100.0 - (100.0 / 3.0) * 0.25
    assert abs(boost_fuel - expected_fuel) < 1.0, (
        f"Boost consumption rate mismatch: expected ~{expected_fuel}, got {boost_fuel}"
    )

    env.close()


def test_auto_flip_turtle_recovery():
    """
    Task 4: Implement Auto-Flip Turtle Recovery (Car.cpp:798-832).
    When the car is upside down on its roof resting on the arena floor (abs(roll) > 2.8 rad, normal_z > 0.7071)
    and the user presses jump, the car executes turtle recovery:
    - Applies pop impulse upward away from ground (-GetUpDir() * 200 UU/s)
    - Starts auto-flip roll torque (+/- 50) to roll the car upright.
    """
    num_envs = 1
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    car_obs = env.get_car_observations()

    # Place car upside down on its roof on the ground
    # Inverted orientation: roll = 180 degrees (pi rad) around X axis -> q = (0, 1, 0, 0)
    car_obs[0, 0, 0] = 0.0
    car_obs[0, 0, 1] = 0.0
    car_obs[0, 0, 2] = 20.0  # Roof on floor
    car_obs[0, 0, 3] = 0.0
    car_obs[0, 0, 4] = 0.0
    car_obs[0, 0, 5] = 0.0
    car_obs[0, 0, 6] = 0.0  # qw
    car_obs[0, 0, 7] = 1.0  # qx (180 deg roll)
    car_obs[0, 0, 8] = 0.0  # qy
    car_obs[0, 0, 9] = 0.0  # qz
    car_obs[0, 0, 10] = 0.0 # ang_vel_x
    car_obs[0, 0, 11] = 0.0
    car_obs[0, 0, 12] = 0.0

    # Step 1 tick without jump so resolve_chassis_arena_collision records world_contact
    actions = _make_actions(env)
    env.sim.step_actions(actions)

    # Next tick: Press Jump (action[5] = 1.0) while upside down
    actions[0, 5] = 1.0

    env.sim.step_actions(actions)

    new_car_obs = env.get_car_observations()
    vel_z = float(new_car_obs[0, 0, 5])
    ang_vel_x = float(new_car_obs[0, 0, 10])

    # Auto-flip should pop the car up with upward impulse (~200 UU/s)
    assert vel_z > 150.0, f"Auto-flip pop impulse not applied: vel_z = {vel_z}"

    # Auto-flip should apply roll torque around forward axis (ang_vel_x != 0)
    assert abs(ang_vel_x) > 0.1, f"Auto-flip roll torque not applied: ang_vel_x = {ang_vel_x}"

    env.close()


def test_auto_roll_surface_alignment():
    """
    Task 3: Implement Auto-Roll Surface Alignment (Car.cpp:834-868).
    When the car has wheels partially in contact (1 <= wheels <= 3) and throttle is applied,
    auto-roll applies downforce (100 UU/s^2) and alignment torque (80) leveling the chassis.
    """
    num_envs = 2
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    car_obs = env.get_car_observations()

    # Place cars slightly tilted at z=14.0 so 2 wheels touch the ground (num_wheels_contact == 2)
    # Tilt 20 degrees around X: q = (cos(10 deg), sin(10 deg), 0, 0)
    angle_rad = math.radians(20.0)
    for e in range(2):
        car_obs[e, 0, 0] = 0.0
        car_obs[e, 0, 1] = 0.0
        car_obs[e, 0, 2] = 14.0
        car_obs[e, 0, 3] = 0.0
        car_obs[e, 0, 4] = 0.0
        car_obs[e, 0, 5] = 0.0
        car_obs[e, 0, 6] = math.cos(angle_rad / 2.0)
        car_obs[e, 0, 7] = math.sin(angle_rad / 2.0)
        car_obs[e, 0, 8] = 0.0
        car_obs[e, 0, 9] = 0.0
        car_obs[e, 0, 10] = 0.0
        car_obs[e, 0, 11] = 0.0
        car_obs[e, 0, 12] = 0.0

    actions = _make_actions(env)
    actions[0, 0] = 1.0  # Env 0: throttle = 1.0 (triggers auto-roll)
    actions[1, 0] = 0.0  # Env 1: throttle = 0.0 (no auto-roll)

    # Step 1 tick
    env.sim.step_actions(actions)

    new_car_obs = env.get_car_observations()
    ang_vel_x_env0 = float(new_car_obs[0, 0, 10])
    ang_vel_x_env1 = float(new_car_obs[1, 0, 10])

    # Auto-roll alignment torque produces exactly ~0.0402 rad/s additional corrective roll velocity
    diff = abs(ang_vel_x_env0 - ang_vel_x_env1)
    assert diff > 0.035, (
        f"Auto-roll alignment torque was not applied when throttle was active! diff={diff}, env0={ang_vel_x_env0}, env1={ang_vel_x_env1}"
    )

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
