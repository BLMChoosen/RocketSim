"""
Detailed drift test - verify car drives forward without lateral drift
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
from rocket_sim import create_initial_state, create_zero_controls, step_physics
from rocket_sim.math_utils import get_car_forward_dir

state = create_initial_state(1, 6)
controls = create_zero_controls(1, 6)
controls = controls.replace(throttle=controls.throttle.at[0,0].set(1.0))

print('t speed fwd lat yawrate vx vy')
for i in range(600):
    state = step_physics(state, controls)
    if (i+1) % 120 == 0:
        vel = state.cars.vel[0,0]
        ang = state.cars.ang_vel[0,0]
        fwd = get_car_forward_dir(state.cars.quat)[0,0]
        right = jnp.array([-fwd[1], fwd[0], 0.0])
        speed = float(jnp.linalg.norm(vel[:2]))
        fwd_speed = float(jnp.dot(vel, fwd))
        lat = float(jnp.dot(vel, right))
        print(f"{(i+1)/120:.2f} {speed:.3f} {fwd_speed:.3f} {lat:.3f} {float(ang[2]):.4f} {float(vel[0]):.3f} {float(vel[1]):.3f}")
