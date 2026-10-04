"""
RocketSim-CUDA Example: Smoke Touch Ball
Demonstrates a batched reinforcement learning loop directly on GPU VRAM.
Runs a vectorized driving policy across 1,024 parallel environments,
detecting and aggregating car-ball contact impulses via GPU tensor flags.
"""

import sys
import os
import time

# Ensure project bindings and build directory are in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for d in ["src/bindings", "build", "python"]:
    path = os.path.join(project_root, d)
    if path not in sys.path:
        sys.path.insert(0, path)

import rocketsim_cuda
from gym_env import RocketSimBatchedEnv


def main():
    print("=" * 75)
    print("      ROCKETSIM-CUDA: ZERO-COPY GPU RL TOUCH BALL SMOKE TEST          ")
    print("=" * 75)

    num_envs = 1024
    cars_per_env = 1
    tick_skip = 4  # 30 Hz policy action rate (120Hz / 4)
    target_touches = 500
    max_steps = 100

    print(f"[*] Initializing RocketSimBatchedEnv...")
    print(f"    - Environments:    {num_envs:,}")
    print(f"    - Cars per Env:    {cars_per_env} (1v0)")
    print(f"    - Tick Skip:       {tick_skip} (30 Hz action frequency)")
    print(f"    - Target Touches:  {target_touches}")

    env = RocketSimBatchedEnv(
        num_envs=num_envs,
        cars_per_env=cars_per_env,
        tick_skip=tick_skip,
        use_torch=False
    )

    free_b, total_b = rocketsim_cuda.get_vram_info()
    pool_mb = env.sim.allocated_bytes / (1024.0 * 1024.0)
    print(f"[+] Monolithic VRAM Allocation: {pool_mb:.2f} MB")
    print(f"[+] Total GPU VRAM: {total_b / (1024**2):.1f} MB (Free: {free_b / (1024**2):.1f} MB)\n")

    # Reset environments to kickoff poses
    env.reset()

    # Position cars and balls in touch practice setup:
    # Cars at varying distances along -X, facing +X (identity quat), ball at X=0, Y=0, Z=93.15
    car_obs = env.get_car_observations()
    ball_obs = env.get_ball_observations()
    total_cars = num_envs * cars_per_env

    for e in range(num_envs):
        # Stagger initial distances from 150 UU up to 800 UU
        dist = 150.0 + (e % 16) * 40.0
        car_obs[e, 0, 0] = -dist      # pos_x
        car_obs[e, 0, 1] = 0.0        # pos_y
        car_obs[e, 0, 2] = 17.0       # pos_z
        car_obs[e, 0, 3] = 0.0        # vel_x
        car_obs[e, 0, 4] = 0.0        # vel_y
        car_obs[e, 0, 5] = 0.0        # vel_z
        car_obs[e, 0, 6] = 1.0        # q_w (facing +X directly at ball)
        car_obs[e, 0, 7] = 0.0        # q_x
        car_obs[e, 0, 8] = 0.0        # q_y
        car_obs[e, 0, 9] = 0.0        # q_z

        ball_obs[e, 0] = 0.0          # pos_x
        ball_obs[e, 1] = 0.0          # pos_y
        ball_obs[e, 2] = 93.15        # pos_z
        ball_obs[e, 3] = 0.0
        ball_obs[e, 4] = 0.0
        ball_obs[e, 5] = 0.0

    # Pre-allocate action tensor directly in GPU VRAM (shape: [num_envs * cars_per_env, 8])
    actions = rocketsim_cuda.zeros([total_cars, 8], dtype="float32")

    # Seed driving policy: full forward throttle + boost
    # Index format: 0: throttle, 1: steer, 2: pitch, 3: yaw, 4: roll, 5: jump, 6: boost, 7: handbrake
    for i in range(total_cars):
        actions[i, 0] = 1.0  # Full forward throttle
        actions[i, 6] = 1.0  # Boost

    total_touches = 0
    t0 = time.perf_counter()
    step_count = 0

    print(f"[*] Starting simulation loop...")
    print("-" * 75)
    print(f"{'Step':>6} | {'Touches (Step)':>14} | {'Total Touches':>14} | {'Step Time':>12} | {'Throughput (SPS)':>18}")
    print("-" * 75)

    while step_count < max_steps and total_touches < target_touches:
        step_t0 = time.perf_counter()

        # Step GPU physics with sub-stepping
        obs, rewards, terminated, truncated, info = env.step(actions)
        step_count += 1

        step_t1 = time.perf_counter()
        step_duration = step_t1 - step_t0

        # Read ball touch validity flags directly from GPU tensor
        ball_hits = info["ball_hit_is_valid"]
        hits_this_step = int(ball_hits.sum())
        total_touches += hits_this_step

        if step_count % 10 == 0 or total_touches >= target_touches:
            sps = (num_envs * tick_skip) / max(step_duration, 1e-6)
            print(f"{step_count:>6} | {hits_this_step:>14} | {total_touches:>14} | {step_duration*1000:>9.2f} ms | {sps:>18,.0f}")

    total_elapsed = time.perf_counter() - t0
    total_physical_ticks = step_count * tick_skip
    effective_sps = (num_envs * total_physical_ticks) / total_elapsed

    print("-" * 75)
    print(f"\n[+] Simulation Complete!")
    print(f"    - Total Steps:            {step_count}")
    print(f"    - Total Physical Ticks:   {total_physical_ticks:,} (120 Hz equivalent)")
    print(f"    - Total Ball Touches:     {total_touches:,}")
    print(f"    - Total Elapsed Time:     {total_elapsed:.3f} s")
    print(f"    - Effective SPS:          {effective_sps:,.0f} physical ticks/sec")

    env.close()

    assert total_touches > 0, "Smoke test failed: zero ball touches detected!"
    print("\n[SUCCESS] Smoke test verified: policy successfully touches ball on GPU!\n")


if __name__ == "__main__":
    main()
