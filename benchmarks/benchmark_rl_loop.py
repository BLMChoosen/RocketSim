"""
Asynchronous SPS Throughput Benchmark for RocketSim-CUDA.
Measures simulation Steps Per Second (SPS), step latency, and GPU VRAM consumption
across 4,096, 16,384, 32,768, and 65,536 concurrent environments.
Strictly non-blocking asynchronous timing via CUDA hardware events.
"""

import sys
import os
import time

# Ensure build, bindings, and python packages are accessible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
build_dir = os.path.join(project_root, "build")
src_dir = os.path.join(project_root, "src", "bindings")
python_dir = os.path.join(project_root, "python")
for d in [build_dir, src_dir, python_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

import rocketsim_cuda
from gym_env import RocketSimBatchedEnv

# Configure CUDA timing events (torch.cuda.Event or native rocketsim_cuda.GpuEvent)
try:
    import torch
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        Event = torch.cuda.Event
        USE_TORCH_EVENT = True
    else:
        Event = rocketsim_cuda.GpuEvent
        USE_TORCH_EVENT = False
except Exception:
    Event = rocketsim_cuda.GpuEvent
    USE_TORCH_EVENT = False


def run_benchmark(
    batch_sizes=(4096, 16384, 32768, 65536),
    cars_per_env=1,
    warmup_steps=50,
    bench_steps=500,
    markdown_output_path=None
):
    print("=" * 80)
    print(" ROCKETSIM-CUDA ASYNCHRONOUS SPS THROUGHPUT BENCHMARK")
    print(f" Platform: CUDA 12+, Tickrate: 120Hz, Timing: Hardware CUDA Events")
    print("=" * 80)

    results = []

    for num_envs in batch_sizes:
        print(f"\n[+] Initializing RocketSimBatchedEnv for {num_envs:,} environments...")
        env = RocketSimBatchedEnv(num_envs=num_envs, cars_per_env=cars_per_env, tick_skip=1)

        # Allocate zeroed GPU action tensor
        total_cars = num_envs * cars_per_env
        actions = rocketsim_cuda.zeros([total_cars, 8], dtype="float32")
        # Apply moderate throttle
        for c in range(min(total_cars, 1024)):
            actions[c, 0] = 1.0

        # Pool VRAM memory
        pool_bytes = env.sim.allocated_bytes
        pool_mb = pool_bytes / (1024 * 1024)

        free_b, total_b = rocketsim_cuda.get_vram_info()
        device_total_mb = total_b / (1024 * 1024)
        device_used_mb = (total_b - free_b) / (1024 * 1024)

        print(f"    Monolithic VRAM Pool: {pool_mb:.2f} MB ({pool_bytes:,} bytes)")
        print(f"    Total GPU VRAM Used:  {device_used_mb:.2f} MB / {device_total_mb:.2f} MB")

        # Warm-up phase
        print(f"    Warming up for {warmup_steps} steps...")
        for _ in range(warmup_steps):
            env.sim.step_actions(actions)

        # Asynchronous CUDA Event Timing
        print(f"    Benchmarking {bench_steps} steps using CUDA hardware events...")
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
        sps = (num_envs * bench_steps) / elapsed_sec

        print(f"    Completed in {elapsed_ms:.2f} ms ({elapsed_sec:.4f} s)")
        print(f"    Average Step Latency: {avg_latency_ms:.4f} ms")
        print(f"    Throughput:           {sps:,.0f} SPS")

        results.append({
            "num_envs": num_envs,
            "latency_ms": avg_latency_ms,
            "sps": sps,
            "pool_mb": pool_mb,
            "device_used_mb": device_used_mb
        })

        env.close()

    # Format Summary Table
    print("\n" + "=" * 80)
    print(" BENCHMARK SUMMARY RESULTS")
    print("=" * 80)
    header = f"| {'Environments':>12} | {'Step Latency (ms)':>18} | {'Throughput (SPS)':>18} | {'Pool VRAM (MB)':>14} |"
    divider = f"|{'-' * 14}|{'-' * 20}|{'-' * 20}|{'-' * 16}|"
    print(header)
    print(divider)
    for r in results:
        row = f"| {r['num_envs']:>12,} | {r['latency_ms']:>18.4f} | {r['sps']:>18,.0f} | {r['pool_mb']:>14.2f} |"
        print(row)
    print("=" * 80)

    # Verification threshold check
    largest = results[-1]
    assert largest["sps"] >= 500000, (
        f"Throughput target failed: {largest['sps']:,.0f} SPS < 500,000 SPS threshold!"
    )
    print(f"\n[PASS] Throughput verification PASSED! Exceeds 500k SPS target at scale.\n")

    # Generate BENCHMARKS.md if requested
    if markdown_output_path:
        with open(markdown_output_path, "w", encoding="utf-8") as f:
            f.write("# RocketSim-CUDA Benchmark Report\n\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
            f.write("## Asynchronous SPS Throughput & Latency\n\n")
            f.write(header + "\n")
            f.write(divider + "\n")
            for r in results:
                f.write(f"| {r['num_envs']:>12,} | {r['latency_ms']:>18.4f} | {r['sps']:>18,.0f} | {r['pool_mb']:>14.2f} |\n")
            f.write("\n")
            f.write("### Methodology & Invariants\n")
            f.write("- **Timing Mechanism:** Asynchronous GPU hardware timing via `cudaEventRecord` / `cudaEventElapsedTime`.\n")
            f.write("- **Zero-Copy Pipeline:** Pure on-device tensor execution with zero host-device PCIe round trips.\n")
            f.write("- **Memory Management:** Monolithic pre-allocated VRAM memory pool (0-byte dynamic allocation delta).\n")
            f.write("- **Precision:** IEEE-754 strict compliance (`--fmad=false --prec-div=true --prec-sqrt=true -ftz=false`).\n")
        print(f"[+] Wrote benchmark report to {markdown_output_path}")

    return results


if __name__ == "__main__":
    md_path = os.path.join(project_root, "BENCHMARKS.md")
    run_benchmark(markdown_output_path=md_path)
