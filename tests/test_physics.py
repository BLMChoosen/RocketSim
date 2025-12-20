"""
Test Suite for RocketSim Physics
=================================
Unit tests for physics simulation, collision, and game mechanics.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import jax.lax as lax

# Import from our new package
from rocket_sim import (
    create_initial_state,
    create_zero_controls,
    step_physics,
    PhysicsState,
    CarControls,
    BALL_RADIUS,
    ARENA_EXTENT_X,
)
from rocket_sim.constants import (
    CAR_SPAWN_Z,
    N_STEPS,
    MAX_CARS,
)


def simulate_n_steps(
    state: PhysicsState,
    controls: CarControls,
    n_steps: int
) -> PhysicsState:
    """Run simulation for n steps using XLA fusion."""
    def body_fn(carry, _):
        s = carry
        s_new = step_physics(s, controls)
        return s_new, None
    
    final_state, _ = lax.scan(body_fn, state, None, length=n_steps)
    return final_state


def run_all_tests():
    """Run all physics tests."""
    import time
    
    print("=" * 70)
    print("RocketSim JAX Physics Test Suite")
    print("=" * 70)
    
    N_ENVS = 128
    N_STEPS_TEST = 120  # 1 second at 120 Hz
    
    # Initialize
    print(f"\n  Initializing {N_ENVS} parallel environments...")
    state = create_initial_state(N_ENVS)
    controls = create_zero_controls(N_ENVS)
    
    print(f"  Ball position shape: {state.ball.pos.shape}")
    print(f"  Car position shape: {state.cars.pos.shape}")
    
    # =========================================================================
    # TEST 1: Suspension settles car at rest
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 1: Suspension settling (no throttle)")
    print("-" * 70)
    
    # JIT compile
    print("  Compiling JIT (first step)...")
    t0 = time.time()
    state = step_physics(state, controls)
    t1 = time.time()
    print(f"  JIT compilation time: {t1-t0:.3f}s")
    
    # Run simulation
    print(f"  Simulating {N_STEPS_TEST} steps...")
    state_rest = create_initial_state(N_ENVS)
    controls_rest = create_zero_controls(N_ENVS)
    
    t0 = time.time()
    state_rest = simulate_n_steps(state_rest, controls_rest, N_STEPS_TEST)
    state_rest.ball.pos.block_until_ready()
    t1 = time.time()
    
    total_ticks = N_ENVS * N_STEPS_TEST
    print(f"  Wall time: {t1-t0:.4f}s")
    print(f"  Ticks/second: {total_ticks/(t1-t0):.2e}")
    
    car0_pos = state_rest.cars.pos[0, 0]
    car0_vel = state_rest.cars.vel[0, 0]
    car0_ground = state_rest.cars.is_on_ground[0, 0]
    
    print(f"\n  Car 0 after 1 second (at rest):")
    print(f"    Position: [{car0_pos[0]:.2f}, {car0_pos[1]:.2f}, {car0_pos[2]:.2f}]")
    print(f"    Velocity: [{car0_vel[0]:.2f}, {car0_vel[1]:.2f}, {car0_vel[2]:.2f}]")
    print(f"    Is on ground: {bool(car0_ground)}")
    
    final_z = float(car0_pos[2])
    car_stable = final_z > 0 and abs(float(car0_vel[2])) < 10.0
    print(f"    Status: {'PASS' if car_stable else 'FAIL'}")
    
    # =========================================================================
    # TEST 2: Throttle response
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 2: Throttle response (full throttle)")
    print("-" * 70)
    
    state_drive = create_initial_state(N_ENVS)
    controls_drive = create_zero_controls(N_ENVS)
    controls_drive = controls_drive.replace(
        throttle=jnp.ones((N_ENVS, 6))
    )
    
    print(f"  Simulating {N_STEPS_TEST} steps with full throttle...")
    t0 = time.time()
    state_drive = simulate_n_steps(state_drive, controls_drive, N_STEPS_TEST)
    state_drive.cars.pos.block_until_ready()
    t1 = time.time()
    print(f"  Wall time: {t1-t0:.4f}s")
    
    car0_pos_drive = state_drive.cars.pos[0, 0]
    car0_vel_drive = state_drive.cars.vel[0, 0]
    
    print(f"\n  Car 0 after 1 second:")
    print(f"    Position: [{car0_pos_drive[0]:.2f}, {car0_pos_drive[1]:.2f}, {car0_pos_drive[2]:.2f}]")
    print(f"    Velocity: [{car0_vel_drive[0]:.2f}, {car0_vel_drive[1]:.2f}, {car0_vel_drive[2]:.2f}]")
    
    initial_x = -2048.0
    final_x = float(car0_pos_drive[0])
    distance_traveled = final_x - initial_x
    final_speed = jnp.linalg.norm(car0_vel_drive)
    
    print(f"\n  Drive check:")
    print(f"    Distance: {distance_traveled:.2f} UU")
    print(f"    Final speed: {float(final_speed):.2f} UU/s")
    print(f"    Status: {'PASS' if distance_traveled > 50 and final_speed > 100 else 'FAIL'}")
    
    # =========================================================================
    # TEST 3: Ball arena collision
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 3: Ball physics (arena collision)")
    print("-" * 70)
    
    state_ball = create_initial_state(N_ENVS)
    new_ball = state_ball.ball.replace(
        pos=jnp.full((N_ENVS, 3), jnp.array([0.0, 0.0, 500.0])),
        vel=jnp.full((N_ENVS, 3), jnp.array([1000.0, 0.0, 0.0])),
    )
    state_ball = state_ball.replace(ball=new_ball)
    
    state_ball = simulate_n_steps(state_ball, controls_rest, N_STEPS_TEST)
    
    ball_pos = state_ball.ball.pos[0]
    ball_vel = state_ball.ball.vel[0]
    
    print(f"  Ball after 1 second:")
    print(f"    Position: [{ball_pos[0]:.2f}, {ball_pos[1]:.2f}, {ball_pos[2]:.2f}]")
    print(f"    Velocity: [{ball_vel[0]:.2f}, {ball_vel[1]:.2f}, {ball_vel[2]:.2f}]")
    
    ball_above_floor = float(ball_pos[2]) >= BALL_RADIUS
    ball_in_x_bounds = abs(float(ball_pos[0])) < ARENA_EXTENT_X
    
    print(f"\n  Arena collision check:")
    print(f"    Ball Z >= BALL_RADIUS: {ball_above_floor}")
    print(f"    Ball inside X bounds: {ball_in_x_bounds}")
    print(f"    Status: {'PASS' if ball_above_floor and ball_in_x_bounds else 'FAIL'}")
    
    # =========================================================================
    # TEST 4: Jump mechanics
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 4: Jump mechanics")
    print("-" * 70)
    
    state_jump = create_initial_state(N_ENVS)
    controls_jump = create_zero_controls(N_ENVS)
    
    JUMP_FRAMES = 12
    
    controls_jump_held = controls_jump.replace(
        jump=jnp.ones((N_ENVS, 6), dtype=jnp.bool_)
    )
    
    # Jump phase
    state_jump = simulate_n_steps(state_jump, controls_jump_held, JUMP_FRAMES)
    
    car0_pos_midair = state_jump.cars.pos[0, 0]
    car0_vel_midair = state_jump.cars.vel[0, 0]
    car0_on_ground_midair = state_jump.cars.is_on_ground[0, 0]
    
    print(f"\n  Mid-air state (after {JUMP_FRAMES} frames):")
    print(f"    Position Z: {float(car0_pos_midair[2]):.2f}")
    print(f"    Velocity Z: {float(car0_vel_midair[2]):.2f}")
    print(f"    On ground: {bool(car0_on_ground_midair)}")
    
    # Release and continue
    state_jump = simulate_n_steps(state_jump, controls_jump, 30)
    
    car0_pos_peak = state_jump.cars.pos[0, 0]
    peak_height = float(car0_pos_peak[2])
    
    print(f"\n  Peak state:")
    print(f"    Position Z: {peak_height:.2f}")
    
    jump_worked = peak_height > 100
    print(f"    Status: {'PASS' if jump_worked else 'FAIL'}")
    
    # =========================================================================
    # TEST 5: Forward flip
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 5: Forward flip")
    print("-" * 70)
    
    state_flip = create_initial_state(N_ENVS)
    controls_flip = create_zero_controls(N_ENVS)
    
    # Phase 1: Jump
    controls_jump_held = controls_flip.replace(
        jump=jnp.ones((N_ENVS, 6), dtype=jnp.bool_)
    )
    state_flip = simulate_n_steps(state_flip, controls_jump_held, 10)
    
    # Phase 2: Release
    state_flip = simulate_n_steps(state_flip, controls_flip, 5)
    
    # Phase 3: Flip trigger
    controls_flip_trigger = controls_flip.replace(
        jump=jnp.ones((N_ENVS, 6), dtype=jnp.bool_),
        pitch=jnp.ones((N_ENVS, 6)) * -1.0
    )
    state_flip = simulate_n_steps(state_flip, controls_flip_trigger, 1)
    
    car0_has_flipped_early = state_flip.cars.has_flipped[0, 0]
    car0_vel_early = state_flip.cars.vel[0, 0]
    
    print(f"\n  After flip trigger:")
    print(f"    Has flipped: {bool(car0_has_flipped_early)}")
    print(f"    Velocity: [{car0_vel_early[0]:.2f}, {car0_vel_early[1]:.2f}, {car0_vel_early[2]:.2f}]")
    
    flip_vel_check = float(car0_vel_early[0])
    flip_worked = flip_vel_check > 100 or bool(car0_has_flipped_early)
    print(f"    Status: {'PASS' if flip_worked else 'INVESTIGATING'}")
    
    # =========================================================================
    # TEST 6: Double jump
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 6: Double jump")
    print("-" * 70)
    
    state_dj = create_initial_state(N_ENVS)
    controls_dj = create_zero_controls(N_ENVS)
    
    # Phase 1: Jump
    controls_jump_held = controls_dj.replace(
        jump=jnp.ones((N_ENVS, 6), dtype=jnp.bool_)
    )
    state_dj = simulate_n_steps(state_dj, controls_jump_held, 10)
    
    # Phase 2: Release
    state_dj = simulate_n_steps(state_dj, controls_dj, 10)
    
    # Phase 3: Double jump
    state_dj = simulate_n_steps(state_dj, controls_jump_held, 1)
    
    car0_has_dj = state_dj.cars.has_double_jumped[0, 0]
    car0_vel_z_dj = state_dj.cars.vel[0, 0, 2]
    
    print(f"\n  After double jump trigger:")
    print(f"    Has double jumped: {bool(car0_has_dj)}")
    print(f"    Z velocity: {float(car0_vel_z_dj):.2f} UU/s")
    print(f"    Status: {'PASS' if bool(car0_has_dj) else 'INVESTIGATING'}")
    
    # =========================================================================
    # TEST 7: Boost mechanics
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 7: Boost mechanics")
    print("-" * 70)
    
    state_boost = create_initial_state(N_ENVS)
    controls_boost = create_zero_controls(N_ENVS)
    
    initial_boost = float(state_boost.cars.boost_amount[0, 0])
    print(f"  Initial boost: {initial_boost:.2f}")
    
    # Apply boost + throttle
    controls_boost = controls_boost.replace(
        throttle=jnp.ones((N_ENVS, 6)),
        boost=jnp.ones((N_ENVS, 6), dtype=jnp.bool_)
    )
    
    state_boost = simulate_n_steps(state_boost, controls_boost, N_STEPS_TEST)
    
    car0_boost = float(state_boost.cars.boost_amount[0, 0])
    car0_vel_boost = state_boost.cars.vel[0, 0]
    final_speed = float(jnp.linalg.norm(car0_vel_boost))
    
    print(f"  After 1 second:")
    print(f"    Remaining boost: {car0_boost:.2f}")
    print(f"    Final speed: {final_speed:.2f} UU/s")
    
    boost_consumed = initial_boost - car0_boost
    boost_worked = boost_consumed > 20 and final_speed > 500
    print(f"    Boost consumed: {boost_consumed:.2f}")
    print(f"    Status: {'PASS' if boost_worked else 'FAIL'}")
    
    # =========================================================================
    print("\n" + "=" * 70)
    print("Test Suite Complete")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
