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

    # Move ball into Orange goal (Y = 5220 > GOAL_SCORE_THRESHOLD_Y 5215.5)
    ball_obs = env.get_ball_observations()
    ball_obs[0, 0] = 0.0
    ball_obs[0, 1] = 5220.0  # inside goal past goal line (5215.5)
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


def test_env_goal_and_selective_reset_lifecycle():
    """
    Exhaustively verify M4.4 requirements:
    1. Ball in Orange goal -> Blue scores (scoring_team = 0, is_goal = 1, terminated = 1).
    2. Ball in Blue goal -> Orange scores (scoring_team = 1, is_goal = 1, terminated = 1).
    3. Ball in field -> No goal (is_goal = 0, terminated = 0).
    4. Auto/selective reset restores ball to (0, 0, 93) with zero velocity and cars to kickoff slots.
    """
    num_envs = 4
    cars_per_env = 2
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env, tick_skip=1)
    env.reset()

    ball_obs = env.get_ball_observations()
    # Env 0: Blue scores into Orange goal (Y = +5220, past threshold 5215.5)
    ball_obs[0, 0] = 0.0
    ball_obs[0, 1] = 5220.0
    ball_obs[0, 2] = 200.0

    # Env 1: Orange scores into Blue goal (Y = -5220, past threshold -5215.5)
    ball_obs[1, 0] = 0.0
    ball_obs[1, 1] = -5220.0
    ball_obs[1, 2] = 200.0

    # Env 2: Ball at Y = +5200 (inside goal cavity but before threshold 5215.5 -> no goal yet)
    ball_obs[2, 0] = 0.0
    ball_obs[2, 1] = 5200.0
    ball_obs[2, 2] = 200.0

    # Env 3: Ball at Y = -5200 (before threshold -5215.5 -> no goal yet)
    ball_obs[3, 0] = 0.0
    ball_obs[3, 1] = -5200.0
    ball_obs[3, 2] = 200.0

    obs, rewards, terminated, truncated, info = env.step()

    # Env 0 checks (Goal for Blue)
    assert info["is_goal"][0] == 1
    assert info["scoring_team"][0] == 0
    assert terminated[0] == 1
    assert rewards[0, 0] > 0.5  # Blue car rewarded
    assert rewards[0, 1] < -0.5 # Orange car penalized

    # Env 1 checks (Goal for Orange)
    assert info["is_goal"][1] == 1
    assert info["scoring_team"][1] == 1
    assert terminated[1] == 1
    assert rewards[1, 1] > 0.5  # Orange car rewarded
    assert rewards[1, 0] < -0.5 # Blue car penalized

    # Env 2 checks (Y = +5200 -> no goal)
    assert info["is_goal"][2] == 0
    assert terminated[2] == 0

    # Env 3 checks (Y = -5200 -> no goal)
    assert info["is_goal"][3] == 0
    assert terminated[3] == 0

    # Test selective reset on env 0 and 1
    env.reset(env_ids=[0, 1])

    # Env 0 and 1 ball must be at center resting height (93.15 UU) and zero velocity
    ball_after = env.get_ball_observations()
    assert abs(float(ball_after[0, 0])) < 1e-3
    assert abs(float(ball_after[0, 1])) < 1e-3
    assert abs(float(ball_after[0, 2]) - 93.15) < 0.5
    assert abs(float(ball_after[0, 3])) < 1e-3
    assert abs(float(ball_after[0, 4])) < 1e-3
    assert abs(float(ball_after[0, 5])) < 1e-3

    assert abs(float(ball_after[1, 0])) < 1e-3
    assert abs(float(ball_after[1, 1])) < 1e-3
    assert abs(float(ball_after[1, 2]) - 93.15) < 0.5

    # Arena state for reset envs should be cleared
    assert info["is_goal"][0] == 0
    assert info["is_goal"][1] == 0

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

