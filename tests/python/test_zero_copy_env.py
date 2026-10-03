"""
Unit and integration tests for RocketSimBatchedEnv and zero-copy tensor fidelity.
Validates:
1. Pointer identity / zero-copy in-place mutation.
2. VRAM leak stability across 100,000 simulation steps (0-byte delta).
3. Selective asynchronous GPU resets (isolation per environment).
4. Physical consistency (gravity, throttle acceleration, boundary checks).
"""

import sys
import os
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

# Optional PyTorch import
try:
    import torch
except ImportError:
    torch = None


def test_pointer_identity_and_zero_copy():
    """
    Test 1 (Pointer Identity / Zero-Copy):
    Compare tensor data_ptr() with C++ raw pointer.
    Verify in-place write reflects immediately in C++ GPU VRAM in the same cycle.
    """
    num_envs = 32
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)

    car_obs = env.get_car_observations()
    ball_obs = env.get_ball_observations()

    # Raw C++ pointers from SimContext
    raw_car_ptr = env.sim.get_car_observations().data_ptr
    raw_ball_ptr = env.sim.get_ball_observations().data_ptr

    # Pointer Identity Verification
    car_ptr = car_obs.data_ptr() if callable(getattr(car_obs, "data_ptr", None)) else car_obs.data_ptr
    ball_ptr = ball_obs.data_ptr() if callable(getattr(ball_obs, "data_ptr", None)) else ball_obs.data_ptr

    assert car_ptr == raw_car_ptr, f"Car pointer mismatch: {car_ptr} != {raw_car_ptr}"
    assert ball_ptr == raw_ball_ptr, f"Ball pointer mismatch: {ball_ptr} != {raw_ball_ptr}"

    # In-place write test: mutate ball Z position of env 0 in-place
    target_z = 1234.5
    ball_obs[0, 2] = target_z

    # Verify that reading from a newly queried view from C++ reflects the mutation immediately
    new_ball_view = env.sim.get_ball_observations()
    assert abs(new_ball_view[0, 2] - target_z) < 1e-4, (
        f"In-place write failed to reflect in C++ VRAM: {new_ball_view[0, 2]} != {target_z}"
    )

    # In-place write on car observation
    car_target_x = 987.6
    car_obs[0, 0, 0] = car_target_x
    new_car_view = env.sim.get_car_observations()
    assert abs(new_car_view[0, 0, 0] - car_target_x) < 1e-4

    env.close()


def test_vram_leak_check_100k_steps():
    """
    Test 2 (VRAM Leak Check):
    Run continuous loop of 100,000 steps with actions on GPU.
    Verify torch.cuda.memory_allocated() / VRAM delta is EXACTLY 0 bytes (zero memory leak).
    """
    num_envs = 64
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)

    # Warm up 10 steps
    for _ in range(10):
        env.step()

    # Measure initial VRAM allocation
    if torch is not None and hasattr(torch, "cuda"):
        initial_vram = torch.cuda.memory_allocated()
    else:
        free_b, total_b = rocketsim_cuda.get_vram_info()
        initial_vram = total_b - free_b

    # Allocate a persistent GPU action tensor (zero-copy)
    total_cars = num_envs * cars_per_env
    actions = rocketsim_cuda.zeros([total_cars, 8], dtype="float32")
    # Set throttle = 1.0 for all cars
    for i in range(total_cars):
        actions[i, 0] = 1.0

    # Execute 100,000 continuous simulation steps entirely on GPU
    num_stress_steps = 100000
    batch_chunk = 1000
    for _ in range(num_stress_steps // batch_chunk):
        for _ in range(batch_chunk):
            env.sim.step_actions(actions)

    # Measure final VRAM allocation
    if torch is not None and hasattr(torch, "cuda"):
        final_vram = torch.cuda.memory_allocated()
    else:
        free_b, total_b = rocketsim_cuda.get_vram_info()
        final_vram = total_b - free_b

    vram_delta = final_vram - initial_vram
    assert vram_delta == 0, f"VRAM leak detected! Delta: {vram_delta} bytes over {num_stress_steps} steps"

    env.close()


def test_selective_reset_isolation():
    """
    Test 3 (Selective Reset):
    Verify resetting only environment K modifies K without affecting other environments.
    """
    num_envs = 16
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)

    # Reset all to default kickoff positions
    env.reset()

    # Apply throttle and step 60 ticks so all cars advance
    total_cars = num_envs * cars_per_env
    actions = rocketsim_cuda.zeros([total_cars, 8], dtype="float32")
    for i in range(total_cars):
        actions[i, 0] = 1.0  # throttle

    for _ in range(60):
        env.step(actions)

    car_obs = env.get_car_observations()

    # Record state of environment 0 (K) and environment 1 (J != K)
    k = 0
    j = 1
    k_pos_x_evolved = car_obs[k, 0, 0]
    j_pos_x_evolved = car_obs[j, 0, 0]

    # Verify both moved away from initial 0 along forward axis (+X)
    assert abs(k_pos_x_evolved) > 1.0, f"Car {k} did not move: {k_pos_x_evolved}"
    assert abs(j_pos_x_evolved) > 1.0, f"Car {j} did not move: {j_pos_x_evolved}"

    # Now selectively reset ONLY environment K (env 0)
    # Using a GPU mask or index list
    env.reset(env_ids=[k])

    # Check states after selective reset
    car_obs_after = env.get_car_observations()
    k_pos_x_reset = car_obs_after[k, 0, 0]
    j_pos_x_after = car_obs_after[j, 0, 0]

    # Environment K must be reset to default position (0.0 for car 0)
    assert abs(k_pos_x_reset) < 1e-2, (
        f"Environment {k} was not reset properly: {k_pos_x_reset}"
    )

    # Environment J must NOT be modified (strict isolation)
    assert abs(j_pos_x_after - j_pos_x_evolved) < 1e-4, (
        f"Environment {j} was mutated during reset of {k}! {j_pos_x_after} != {j_pos_x_evolved}"
    )

    env.close()


def test_physical_consistency():
    """
    Test 4 (Physical Consistency Check):
    Verify gravity acceleration (-650 UU/s^2), throttle acceleration, and goal bounds.
    """
    num_envs = 4
    cars_per_env = 1
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env)
    env.reset()

    ball_obs = env.get_ball_observations()

    # Drop ball from Z = 1000.0 with slight downward velocity to wake from sleep guard
    env_id = 0
    ball_obs[env_id, 2] = 1000.0  # pos_z
    ball_obs[env_id, 5] = -0.01   # vel_z (wakes dormant rigid body)

    # Step for 60 ticks (0.5 seconds at 120Hz)
    for _ in range(60):
        env.sim.step(0)

    # Vel_z should be negative due to gravity (g_z = -650 UU/s^2)
    # vel_z ~ -650 * 0.5 = -325 (minus air drag)
    ball_obs_after = env.get_ball_observations()
    vel_z = ball_obs_after[env_id, 5]
    pos_z = ball_obs_after[env_id, 2]

    assert vel_z < -200.0, f"Ball vel_z did not accelerate under gravity: {vel_z}"
    assert pos_z < 1000.0, f"Ball pos_z did not drop under gravity: {pos_z}"

    # Goal detection test: place ball inside Orange goal (Y > 5120)
    goal_env = 1
    ball_obs[goal_env, 0] = 0.0     # X center
    ball_obs[goal_env, 1] = 5150.0  # Y in Orange goal
    ball_obs[goal_env, 2] = 200.0   # Z below crossbar

    # Step 1 tick
    env.sim.step(0)

    is_goal = env.get_is_goal()
    scoring_team = env.get_scoring_team()
    terminated = env.get_terminated()

    assert is_goal[goal_env] == 1, "Goal was not triggered in Orange goal cavity"
    assert scoring_team[goal_env] == 0, f"Scoring team should be Blue (0), got {scoring_team[goal_env]}"
    assert terminated[goal_env] == 1, "Episode was not terminated on goal"

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
