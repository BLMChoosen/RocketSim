"""
Unit and integration tests for RocketSimBatchedEnv Gymnasium interface.
Tests gym step/reset lifecycle, sub-stepping (tick_skip), reward calculation,
and truncation/termination flags.
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


def test_env_init():
    """Verify initialization at various concurrency scales."""
    for n in [16, 64, 256]:
        env = RocketSimBatchedEnv(num_envs=n, cars_per_env=1, tick_skip=4)
        assert env.num_envs == n
        assert env.cars_per_env == 1
        assert env.total_cars == n
        assert env.tick_skip == 4
        assert env.sim.allocated_bytes > 0
        env.close()


def test_env_step_lifecycle():
    """Verify gym-standard step() returning (obs, rewards, terminated, truncated, info)."""
    num_envs = 32
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env, tick_skip=2)

    obs, info = env.reset(return_info=True)
    assert obs is not None
    assert isinstance(info, dict)
    assert "ball" in info
    assert "is_goal" in info

    # Step without actions (free physics evolution)
    obs, rewards, terminated, truncated, info = env.step()
    assert obs.shape == (num_envs, cars_per_env, 14)
    assert rewards.shape == (num_envs, cars_per_env)
    assert terminated.shape == (num_envs,)
    assert truncated.shape == (num_envs,)
    assert isinstance(info, dict)

    # Step with GPU actions tensor
    actions = rocketsim_cuda.zeros([num_envs * cars_per_env, 8], dtype="float32")
    obs, rewards, terminated, truncated, info = env.step(actions)
    assert obs is not None

    env.close()


def test_env_tick_skip():
    """Verify tick_skip executes multiple 120Hz sub-steps."""
    num_envs = 8
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=1, tick_skip=4)
    env.reset()

    initial_ticks = env.get_tick_count()[0]
    env.step()
    after_ticks = env.get_tick_count()[0]

    assert after_ticks == initial_ticks + 4, f"Tick count mismatch: {after_ticks} != {initial_ticks + 4}"
    env.close()


def test_env_reward_goal_signal():
    """Verify goal reward produces positive signal for scoring team."""
    num_envs = 4
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=2, tick_skip=1)
    env.reset()

    # Move ball into Orange goal (Blue scores)
    ball_obs = env.get_ball_observations()
    ball_obs[0, 0] = 0.0
    ball_obs[0, 1] = 5150.0  # inside goal
    ball_obs[0, 2] = 200.0

    obs, rewards, terminated, truncated, info = env.step()

    # Car 0 (Blue team) should receive positive reward (+1.0)
    # Car 1 (Orange team) should receive negative reward (-1.0)
    rew_blue = rewards[0, 0]
    rew_orange = rewards[0, 1]

    assert rew_blue > 0.5, f"Blue car did not receive positive goal reward: {rew_blue}"
    assert rew_orange < -0.5, f"Orange car did not receive negative goal reward: {rew_orange}"
    assert terminated[0] == 1, "Episode was not terminated on goal"

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
