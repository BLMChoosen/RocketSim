"""
Test script to verify car stability (no drift without input)
"""
import jax
jax.config.update('jax_platform_name', 'cpu')

from game import create_initial_state, step_physics, create_zero_controls
import jax.numpy as jnp

print("=== Car Stability Test ===")
state = create_initial_state(1, 6)
controls = create_zero_controls(1, 6)

# Set the car at a stable height
new_pos = state.cars.pos.at[0, 0, 2].set(29.0)
state = state.replace(cars=state.cars.replace(pos=new_pos))

print('Initial position:', state.cars.pos[0, 0])

# Run for 240 steps (2 seconds)
for i in range(240):
    state = step_physics(state, controls)
    if (i+1) % 60 == 0:  # Print every 0.5 seconds
        v = state.cars.vel[0, 0]
        p = state.cars.pos[0, 0]
        print(f"t={((i+1)/120):.2f}s: pos=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}], vel=[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]")

# Final check
v = state.cars.vel[0, 0]
speed = jnp.sqrt(v[0]**2 + v[1]**2)  # Horizontal speed

print(f"\nFinal horizontal speed: {speed:.4f} UU/s")
if speed < 0.5:
    print("✅ SUCCESS: Car is stable (horizontal speed < 0.5)")
else:
    print("❌ FAIL: Car is still drifting")
