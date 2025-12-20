import time
import jax
import jax.numpy as jnp
import jax_sim

# Configuration
N_ENVS = 4096
MAX_CARS = 6
N_STEPS_LATENCY = 1000
N_STEPS_THROUGHPUT = 10000

def run_benchmark():
    print(f"JAX Backend: {jax.devices()[0].platform}")
    print(f"Devices: {jax.devices()}")
    
    # Initialize state
    print("Initializing state...")
    key = jax.random.PRNGKey(0)
    state = jax_sim.create_initial_state(N_ENVS, MAX_CARS)
    controls = jax_sim.create_zero_controls(N_ENVS, MAX_CARS)
    
    # JIT compile step function
    print("Compiling step_physics...")
    step_jit = jax.jit(jax_sim.step_physics)
    
    # Warmup
    print("Warmup step...")
    _ = step_jit(state, controls)
    
    # --- TEST A: LATENCY (Python Loop) ---
    print(f"\n--- TEST A: LATENCY (Python Loop, {N_STEPS_LATENCY} steps) ---")
    start_time = time.time()
    current_state = state
    for _ in range(N_STEPS_LATENCY):
        current_state = step_jit(current_state, controls)
    # Block on a leaf node to ensure computation is done
    jax.block_until_ready(current_state.ball.pos) 
    end_time = time.time()
    
    total_time_a = end_time - start_time
    sps_a = (N_ENVS * N_STEPS_LATENCY) / total_time_a
    print(f"Total Time: {total_time_a:.4f} s")
    print(f"Steps Per Second (SPS): {sps_a:,.0f}")
    print(f"Latency per step: {(total_time_a / N_STEPS_LATENCY) * 1000:.4f} ms")

    # --- TEST B: THROUGHPUT (XLA Fusion) ---
    print(f"\n--- TEST B: THROUGHPUT (XLA Fusion, {N_STEPS_THROUGHPUT} steps) ---")
    
    def rollout_scan(carry, _):
        state, controls = carry
        new_state = jax_sim.step_physics(state, controls)
        return (new_state, controls), None

    @jax.jit
    def run_rollout(state, controls):
        (final_state, _), _ = jax.lax.scan(rollout_scan, (state, controls), None, length=N_STEPS_THROUGHPUT)
        return final_state

    # Warmup
    print("Compiling rollout_scan...")
    _ = run_rollout(state, controls)
    
    print("Running benchmark...")
    start_time = time.time()
    final_state = run_rollout(state, controls)
    jax.block_until_ready(final_state.ball.pos)
    end_time = time.time()
    
    total_time_b = end_time - start_time
    sps_b = (N_ENVS * N_STEPS_THROUGHPUT) / total_time_b
    print(f"Total Time: {total_time_b:.4f} s")
    print(f"Steps Per Second (SPS): {sps_b:,.0f}")
    
    if sps_b < 1_000_000:
        print("\n[WARNING] SPS < 1,000,000. Performance investigation required.")
    else:
        print("\n[SUCCESS] SPS > 1,000,000. High performance achieved.")

    # --- TEST C: SUSTAINED STRESS TEST (60 Seconds) ---
    DURATION_SEC = 60
    print(f"\n--- TEST C: SUSTAINED STRESS TEST ({DURATION_SEC} Seconds) ---")
    print(f"Running continuous rollouts to analyze thermal/CPU stability...")
    
    start_stress = time.time()
    end_stress = start_stress + DURATION_SEC
    
    iterations = 0
    total_steps_c = 0
    
    current_state = final_state
    
    while time.time() < end_stress:
        current_state = run_rollout(current_state, controls)
        # Block to keep loop in sync with GPU execution
        jax.block_until_ready(current_state.ball.pos)
        
        iterations += 1
        total_steps_c += N_STEPS_THROUGHPUT
        
        elapsed = time.time() - start_stress
        print(f"  Time: {elapsed:.1f}s | Iterations: {iterations} | Total Steps: {total_steps_c * N_ENVS:,.0f}")

    total_time_c = time.time() - start_stress
    sps_c = (total_steps_c * N_ENVS) / total_time_c
    
    print(f"\nStress Test Complete.")
    print(f"Total Duration: {total_time_c:.4f} s")
    print(f"Average SPS: {sps_c:,.0f}")
    print("Check your CPU/GPU usage history now.")

if __name__ == "__main__":
    run_benchmark()
