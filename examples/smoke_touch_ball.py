"""
RocketSim-CUDA Example: Smoke Touch Ball (RL Loop Verification)
Demonstrates a batched reinforcement learning environment directly on GPU VRAM.

Tests:
1. Negative Control: Ball placed at 10,000 UU -> 0 touches (guarantees no false positive flags).
2. Real Kickoff Touch Test: Standard Soccar kickoff spawn via env.reset(). Proportional steering
   towards ball with full throttle + boost. First touch verified to occur within the physical
   tick window [240, 330] ticks (approx 2.0s - 2.75s at 120 Hz).
3. Immediate Contact Sanity: Single-tick close-proximity contact impulse verification.
"""

import sys
import os
import time
import math

# Ensure project bindings and build directory are in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for d in ["src/bindings", "build", "python"]:
    path = os.path.join(project_root, d)
    if path not in sys.path:
        sys.path.insert(0, path)

import rocketsim_cuda
from gym_env import RocketSimBatchedEnv


def run_negative_control():
    """Negative Control: Ensure 0 false positive touches when ball is far away."""
    print("\n" + "=" * 75)
    print(" [1/3] NEGATIVE CONTROL: Ball at 10,000 UU (Expected: 0 touches)")
    print("=" * 75)

    num_envs = 64
    cars_per_env = 1
    tick_skip = 4
    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env, tick_skip=tick_skip)
    env.reset()

    # Move ball far away outside the arena bounds
    ball_obs = env.get_ball_observations()
    for e in range(num_envs):
        ball_obs[e, 0] = 0.0
        ball_obs[e, 1] = 10000.0
        ball_obs[e, 2] = 93.15

    actions = rocketsim_cuda.zeros([num_envs, 8], dtype="float32")
    for i in range(num_envs):
        actions[i, 0] = 1.0  # throttle
        actions[i, 6] = 1.0  # boost

    touches = 0
    for _ in range(25):  # 100 ticks
        obs, rew, term, trunc, info = env.step(actions)
        hits = info["ball_hit_is_valid"]
        for e in range(num_envs):
            if int(hits[e, 0]) > 0:
                touches += 1

    env.close()
    print(f"    - Negative Control Touches: {touches}")
    assert touches == 0, f"False positive touch detected! Touches: {touches}"
    print("    [+] PASS: Zero touches detected when ball is out of reach.")


def run_real_kickoff_smoke_test():
    """Real Kickoff: Standard Soccar spawn without overriding positions."""
    print("\n" + "=" * 75)
    print(" [2/3] REAL SOCCAR KICKOFF SMOKE TEST: Full Field Run to Ball")
    print("=" * 75)

    num_envs = 128
    cars_per_env = 1
    tick_skip = 1  # 1-tick resolution to measure exact first touch timing
    max_ticks = 400

    env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env, tick_skip=tick_skip)
    env.reset()

    # Inspect initial car-ball distance at kickoff
    car_obs = env.get_car_observations()
    ball_obs = env.get_ball_observations()
    init_dists = []
    for e in range(min(5, num_envs)):
        dx = float(car_obs[e, 0, 0]) - float(ball_obs[e, 0])
        dy = float(car_obs[e, 0, 1]) - float(ball_obs[e, 1])
        dz = float(car_obs[e, 0, 2]) - float(ball_obs[e, 2])
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        init_dists.append(dist)

    print(f"    - Environments:        {num_envs:,}")
    print(f"    - Initial Car Distances: {[f'{d:.1f} UU' for d in init_dists]} (Real Soccar spawns)")
    print(f"    - Expected Touch Window: 240 to 330 ticks (2.0s - 2.75s at 120 Hz)")

    actions = rocketsim_cuda.zeros([num_envs, 8], dtype="float32")
    first_touch_ticks = {}
    total_touches = 0

    t0 = time.perf_counter()

    for tick in range(1, max_ticks + 1):
        # Vectorized proportional steering policy heading directly towards ball (0, 0)
        for e in range(num_envs):
            cx = float(car_obs[e, 0, 0])
            cy = float(car_obs[e, 0, 1])
            qw = float(car_obs[e, 0, 6])
            qz = float(car_obs[e, 0, 9])
            car_yaw = 2.0 * math.atan2(qz, qw)
            angle_to_ball = math.atan2(-cy, -cx)
            steer = angle_to_ball - car_yaw
            while steer > math.pi:
                steer -= 2.0 * math.pi
            while steer < -math.pi:
                steer += 2.0 * math.pi

            actions[e, 0] = 1.0  # throttle
            actions[e, 1] = max(-1.0, min(1.0, steer * 3.0))  # steer towards ball
            actions[e, 6] = 1.0  # boost

        obs, rew, term, trunc, info = env.step(actions)
        hits = info["ball_hit_is_valid"]

        for e in range(num_envs):
            if int(hits[e, 0]) > 0:
                total_touches += 1
                if e not in first_touch_ticks:
                    first_touch_ticks[e] = tick

        if len(first_touch_ticks) == num_envs:
            print(f"    [+] All {num_envs} environments reached and touched the ball at tick {tick}!")
            break

    elapsed = time.perf_counter() - t0
    env.close()

    assert len(first_touch_ticks) > 0, "No cars touched the ball during kickoff run!"

    touch_ticks = list(first_touch_ticks.values())
    min_touch = min(touch_ticks)
    max_touch = max(touch_ticks)
    avg_touch = sum(touch_ticks) / len(touch_ticks)

    print(f"    - Completed Ticks:     {tick}")
    print(f"    - First Touch (Min):   Tick {min_touch} ({min_touch / 120.0:.3f} s)")
    print(f"    - First Touch (Max):   Tick {max_touch} ({max_touch / 120.0:.3f} s)")
    print(f"    - First Touch (Avg):   Tick {avg_touch:.1f} ({avg_touch / 120.0:.3f} s)")
    print(f"    - Total Touches:       {total_touches:,}")
    print(f"    - Total Time:          {elapsed:.3f} s ({num_envs * tick / elapsed:,.0f} SPS)")

    # Assert that first touch falls in the physically validated window [240, 330] ticks
    assert 240 <= min_touch <= 330, f"First touch tick {min_touch} is outside expected range [240, 330]!"
    print(f"    [+] PASS: Real kickoff first touch occurred within expected physical window [240, 330] ticks.")


def run_contact_sanity():
    """Immediate Contact Sanity: Single-tick close contact check."""
    print("\n" + "=" * 75)
    print(" [3/3] CONTACT SANITY: Immediate Proximity Contact Check")
    print("=" * 75)

    env = RocketSimBatchedEnv(num_envs=4, cars_per_env=1, tick_skip=1)
    env.reset()

    car_obs = env.get_car_observations()
    ball_obs = env.get_ball_observations()

    # Place car right in front of ball
    car_obs[0, 0, 0] = -120.0
    car_obs[0, 0, 1] = 0.0
    car_obs[0, 0, 2] = 17.0
    car_obs[0, 0, 3] = 1000.0  # moving forward fast
    car_obs[0, 0, 6] = 1.0

    ball_obs[0, 0] = 0.0
    ball_obs[0, 1] = 0.0
    ball_obs[0, 2] = 93.15

    actions = rocketsim_cuda.zeros([4, 8], dtype="float32")
    actions[0, 0] = 1.0

    obs, rew, term, trunc, info = env.step(actions)
    hits = info["ball_hit_is_valid"]
    hit = int(hits[0, 0])

    env.close()
    assert hit == 1, "Immediate contact was not registered!"
    print("    [+] PASS: Immediate contact flag registered correctly.")


def main():
    print("=" * 75)
    print("      ROCKETSIM-CUDA: COMPREHENSIVE RL SMOKE TOUCH BALL SUITE         ")
    print("=" * 75)

    run_negative_control()
    run_real_kickoff_smoke_test()
    run_contact_sanity()

    print("\n" + "=" * 75)
    print(" [SUCCESS] ALL SMOKE TESTS AND PHYSICAL GATES PASSED!                ")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
