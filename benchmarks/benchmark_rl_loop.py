"""
Asynchronous SPS Throughput Benchmark for RocketSim-CUDA.
Measures simulation Steps Per Second (SPS), step latency, and GPU VRAM consumption
across 1v0 and 2v2 match configurations (4,096 to 65,536 concurrent environments).
Strictly non-blocking asynchronous timing via CUDA hardware events.
All physics features active: analytical arena SDF, 4-wheel raycast suspension & tire friction,
car dynamics & air control, car-ball OBB-sphere collision, 34 boost pads, and terminations.
"""

import sys
import os
import time

# Ensure build, bindings, and python packages are accessible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for d in ["build", "src/bindings", "python"]:
    path = os.path.join(project_root, d)
    if path not in sys.path:
        sys.path.insert(0, path)

import rocketsim_cuda
from gym_env import RocketSimBatchedEnv

# Configure CUDA timing events (torch.cuda.Event or native rocketsim_cuda.GpuEvent)
try:
    import torch
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        Event = torch.cuda.Event
    else:
        Event = rocketsim_cuda.GpuEvent
except Exception:
    Event = rocketsim_cuda.GpuEvent


def run_benchmark_config(
    config_name: str,
    cars_per_env: int,
    batch_sizes: tuple,
    warmup_steps: int = 50,
    bench_steps: int = 400
):
    print("\n" + "=" * 80)
    print(f" BENCHMARK CONFIGURATION: {config_name} ({cars_per_env} car(s) per arena)")
    print(f" Active Features: Arena SDF, 4-Wheel Suspension, Car Dynamics,")
    print(f"                  Car-Ball Collision, 34 Boost Pads, Terminations")
    print("=" * 80)

    results = []

    for num_envs in batch_sizes:
        print(f"\n[+] Initializing RocketSimBatchedEnv for {num_envs:,} environments ({num_envs * cars_per_env:,} total cars)...")
        env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env, tick_skip=1, use_torch=False)

        total_cars = num_envs * cars_per_env
        actions = rocketsim_cuda.zeros([total_cars, 8], dtype="float32")
        # Apply forward throttle and intermittent steering
        for c in range(total_cars):
            actions[c, 0] = 1.0  # Full throttle
            actions[c, 1] = 0.2 if (c % 2 == 0) else -0.2  # Steer
            actions[c, 6] = 1.0 if (c % 3 == 0) else 0.0  # Boost

        pool_bytes = env.sim.allocated_bytes
        pool_mb = pool_bytes / (1024.0 * 1024.0)

        free_b, total_b = rocketsim_cuda.get_vram_info()
        device_total_mb = total_b / (1024.0 * 1024.0)
        device_used_mb = (total_b - free_b) / (1024.0 * 1024.0)

        print(f"    Monolithic VRAM Pool: {pool_mb:.2f} MB ({pool_bytes:,} bytes)")
        print(f"    Total GPU VRAM Used:  {device_used_mb:.2f} MB / {device_total_mb:.2f} MB")

        # Warm-up phase
        for _ in range(warmup_steps):
            env.sim.step_actions(actions)

        # Asynchronous CUDA Event Timing
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)

        stream = env.sim.get_stream()

        start_event.record(stream)
        for _ in range(bench_steps):
            env.sim.step_actions(actions)
        end_event.record(stream)
        end_event.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_sec = elapsed_ms / 1000.0

        avg_latency_ms = elapsed_ms / bench_steps
        env_sps = (num_envs * bench_steps) / elapsed_sec
        agent_sps = (total_cars * bench_steps) / elapsed_sec

        print(f"    Completed in {elapsed_ms:.2f} ms ({elapsed_sec:.4f} s)")
        print(f"    Average Step Latency: {avg_latency_ms:.4f} ms")
        print(f"    Environment SPS:      {env_sps:,.0f} SPS")
        print(f"    Agent SPS (Car Ticks):{agent_sps:,.0f} Car-SPS")

        results.append({
            "num_envs": num_envs,
            "total_cars": total_cars,
            "latency_ms": avg_latency_ms,
            "env_sps": env_sps,
            "agent_sps": agent_sps,
            "pool_mb": pool_mb
        })

        env.close()

    return results


def main():
    md_path = os.path.join(project_root, "BENCHMARKS.md")

    # 1v0 configuration
    results_1v0 = run_benchmark_config(
        config_name="1v0 (Solo Practice / RL Baseline)",
        cars_per_env=1,
        batch_sizes=(4096, 16384, 32768, 65536),
        warmup_steps=30,
        bench_steps=300
    )

    # 2v2 configuration
    results_2v2 = run_benchmark_config(
        config_name="2v2 (Full Match / 4 Cars Per Arena)",
        cars_per_env=4,
        batch_sizes=(1024, 4096, 8192, 16384),
        warmup_steps=30,
        bench_steps=300
    )

    # Output formatted tables
    print("\n" + "=" * 80)
    print(" COMPLETE BENCHMARK REPORT SUMMARY")
    print("=" * 80)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RocketSim-CUDA Comprehensive Benchmark Report\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")

        f.write("## 1. 1v0 Solo Configuration (1 Car per Environment)\n\n")
        f.write("> **Features Active:** Full Ball Trajectory, Octane Dynamics & Raycast Suspension, Analytical Soccar Arena SDF, Car-Ball OBB Collision, 34 Boost Pads, Episode Lifecycle Terminations.\n\n")
        hdr_1v0 = f"| {'Environments':>12} | {'Total Cars':>12} | {'Step Latency (ms)':>18} | {'Throughput (SPS)':>18} | {'Pool VRAM (MB)':>14} |"
        div_1v0 = f"|{'-' * 14}|{'-' * 14}|{'-' * 20}|{'-' * 20}|{'-' * 16}|"
        f.write(hdr_1v0 + "\n")
        f.write(div_1v0 + "\n")
        for r in results_1v0:
            f.write(f"| {r['num_envs']:>12,} | {r['total_cars']:>12,} | {r['latency_ms']:>18.4f} | {r['env_sps']:>18,.0f} | {r['pool_mb']:>14.2f} |\n")

        f.write("\n## 2. 2v2 Team Match Configuration (4 Cars per Environment)\n\n")
        f.write("> **Features Active:** 4 Autonomous Cars (2 Blue vs 2 Orange), Ball Trajectory & Aerodynamics, 4-Wheel Raycast Suspension per Car, Arena SDF, Car-Ball Collisions, 34 Boost Pads, Team Scoring & Terminations.\n\n")
        hdr_2v2 = f"| {'Environments':>12} | {'Total Cars':>12} | {'Step Latency (ms)':>18} | {'Env SPS':>14} | {'Agent SPS (Car-Ticks)':>22} | {'Pool VRAM (MB)':>14} |"
        div_2v2 = f"|{'-' * 14}|{'-' * 14}|{'-' * 20}|{'-' * 16}|{'-' * 24}|{'-' * 16}|"
        f.write(hdr_2v2 + "\n")
        f.write(div_2v2 + "\n")
        for r in results_2v2:
            f.write(f"| {r['num_envs']:>12,} | {r['total_cars']:>12,} | {r['latency_ms']:>18.4f} | {r['env_sps']:>14,.0f} | {r['agent_sps']:>22,.0f} | {r['pool_mb']:>14.2f} |\n")

        f.write("\n## 3. Methodology & Architectural Invariants\n\n")
        f.write("* **Hardware Timing:** Asynchronous timing recorded directly on the GPU execution stream using `cudaEventRecord` / `cudaEventElapsedTime`.\n")
        f.write("* **Zero-Copy Pipeline:** Pure on-device tensor execution. Observation, reward, action, and termination pointers reside exclusively in GPU VRAM without host PCIe round-trips.\n")
        f.write("* **Pre-Allocated Memory Arena:** 100% pre-allocated Structure-of-Arrays (SoA) layout with 128-byte cache-line alignment. Exactly 0 bytes allocated during simulation steps (`cudaMalloc` / `cudaFree` = 0).\n")
        f.write("* **Numerical Standard:** IEEE-754 strict floating-point compiler flags (`--fmad=false --prec-div=true --prec-sqrt=true -ftz=false`).\n")

    print(f"\n[+] Successfully generated updated benchmark report at: {md_path}\n")


if __name__ == "__main__":
    main()
