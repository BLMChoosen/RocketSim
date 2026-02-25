"""
Step Functions
==============
Main physics step, car step, environment step, and round reset.
This is the central orchestration module that ties all subsystems together.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

from ..constants import (
    DT, BALL_RADIUS, BALL_MAX_SPEED, CAR_MAX_SPEED, CAR_MAX_ANG_SPEED,
    CAR_MASS, CAR_INERTIA, CAR_TORQUE_SCALE, CAR_AIR_CONTROL_TORQUE, CAR_AIR_CONTROL_DAMPING,
    GRAVITY_Z, GOAL_THRESHOLD_Y, DEMO_RESPAWN_TIME, CAR_SPAWN_Z,
    BOOST_MAX, BOOST_SPAWN_AMOUNT, N_PADS_TOTAL, ARENA_EXTENT_X,
    DOUBLEJUMP_MAX_DELAY, STICKY_FORCE_SCALE_BASE, STOPPING_FORWARD_VEL,
    KICKOFF_POSITIONS_BLUE, KICKOFF_POSITIONS_ORANGE, KICKOFF_YAW_BLUE, KICKOFF_YAW_ORANGE,
    BT_TO_UU, UU_TO_BT, THROTTLE_AIR_ACCEL, BALL_REST_Z,
    POWERSLIDE_RISE_RATE, POWERSLIDE_FALL_RATE,
    FLIP_TORQUE_TIME, FLIP_PITCHLOCK_EXTRA_TIME,
)
from ..types import BallState, CarState, CarControls, PhysicsState
from ..math_utils import (
    quat_rotate_vector, quat_from_yaw,
    get_car_forward_dir, get_car_up_dir, get_car_right_dir,
    clamp_velocity, clamp_angular_velocity,
)
from ..collision import (
    resolve_ball_arena_collision, resolve_car_arena_collision,
    resolve_car_ball_collision, resolve_car_car_collision,
)
from ..physics import (
    step_ball, solve_suspension_and_tires, integrate_position, integrate_rotation,
)
from ..mechanics import (
    handle_jump, handle_flip_or_double_jump, apply_flip_z_damping,
    apply_boost, update_supersonic_status,
)
from .boost_pads import resolve_boost_pads


# =============================================================================
# CAR PHYSICS STEP
# =============================================================================


def step_cars(
    cars: CarState, 
    controls: CarControls,
    dt: float = DT
) -> CarState:
    """
    Advance car physics by one timestep.
    
    Args:
        cars: Current car state
        controls: Control inputs
        dt: Time step
        
    Returns:
        Updated car state
    """
    active_mask = ~cars.is_demoed
    
    # C++ handbrake analog value: rises at POWERSLIDE_RISE_RATE, falls at POWERSLIDE_FALL_RATE
    handbrake_val = cars.handbrake_val
    handbrake_val = jnp.where(
        controls.handbrake,
        handbrake_val + POWERSLIDE_RISE_RATE * dt,
        handbrake_val - POWERSLIDE_FALL_RATE * dt,
    )
    handbrake_val = jnp.clip(handbrake_val, 0.0, 1.0)
    cars = cars.replace(handbrake_val=handbrake_val)
    
    # Calculate forward speed
    forward_dir = get_car_forward_dir(cars.quat)
    forward_speed = jnp.sum(cars.vel * forward_dir, axis=-1)
    
    # Handle jump mechanics
    cars, jump_vel_delta, jump_start_mask = handle_jump(cars, controls, dt)
    
    # Handle flip/double-jump mechanics
    cars, flip_vel_impulse, flip_torque = handle_flip_or_double_jump(
        cars, controls, forward_speed, dt
    )
    
    # Compute suspension and tire forces (C++ btVehicleRL style)
    sus_force, sus_torque, tire_impulse, wheel_rel_pos, wheel_contacts, num_contacts, upwards_dir = solve_suspension_and_tires(
        cars, controls
    )
    
    # FLIP RESET: When all 4 wheels touch something while airborne, reset flip ability
    is_airborne = ~cars.is_on_ground
    all_wheels_contact = num_contacts >= 4
    flip_reset = is_airborne & all_wheels_contact
    
    cars = cars.replace(
        has_flipped=jnp.where(flip_reset, False, cars.has_flipped),
        has_double_jumped=jnp.where(flip_reset, False, cars.has_double_jumped),
        air_time_since_jump=jnp.where(flip_reset, 0.0, cars.air_time_since_jump),
    )
    
    # Apply gravity
    gravity_vec = jnp.array([0.0, 0.0, GRAVITY_Z])
    gravity_force = gravity_vec * CAR_MASS
    
    # Disable suspension when jumping
    is_jumping_expanded = cars.is_jumping[..., None]
    sus_force_masked = jnp.where(is_jumping_expanded, 0.0, sus_force)
    sus_torque_masked = jnp.where(is_jumping_expanded, 0.0, sus_torque)
    
    # Sticky forces (C++ style)
    has_any_contact = num_contacts >= 1
    
    is_boosting_now = controls.boost & (cars.boost_amount > 0)
    real_throttle = jnp.where(is_boosting_now, 1.0, controls.throttle)
    
    abs_forward_speed = jnp.abs(forward_speed)
    throttle_active = jnp.abs(real_throttle) > 0.01
    full_stick = throttle_active | (abs_forward_speed > STOPPING_FORWARD_VEL)
    
    upward_z = upwards_dir[..., 2]
    extra_stick = jnp.where(full_stick, 1.0 - jnp.abs(upward_z), 0.0)
    sticky_force_scale = STICKY_FORCE_SCALE_BASE + extra_stick
    
    sticky_force = upwards_dir * sticky_force_scale[..., None] * GRAVITY_Z * CAR_MASS
    sticky_force = jnp.where(
        (has_any_contact & ~cars.is_jumping)[..., None],
        sticky_force,
        0.0
    )
    
    # Total force from suspension + gravity + sticky
    total_force = (sus_force_masked / dt) + gravity_force + sticky_force
    
    # Apply force to velocity
    accel = total_force / CAR_MASS
    vel = cars.vel + jnp.where(
        active_mask[..., None],
        accel * dt,
        0.0
    )
    
    # Apply tire impulses
    tire_impulse_total = jnp.sum(tire_impulse, axis=-2)  # (N, MAX_CARS, 3)
    tire_impulse_masked = jnp.where(cars.is_jumping[..., None], 0.0, tire_impulse_total)
    tire_vel_delta = tire_impulse_masked * dt / CAR_MASS * BT_TO_UU
    vel = vel + jnp.where(
        active_mask[..., None],
        tire_vel_delta,
        0.0
    )
    
    # Reset Z velocity before applying jump impulse
    vel_z = jnp.where(
        jump_start_mask & active_mask,
        jnp.maximum(vel[..., 2], 0.0),
        vel[..., 2]
    )
    vel = vel.at[..., 2].set(vel_z)
    
    vel = vel + jnp.where(active_mask[..., None], jump_vel_delta, 0.0)
    vel = vel + jnp.where(active_mask[..., None], flip_vel_impulse, 0.0)
    
    # Apply boost
    vel, boost_amount, is_boosting_new, boosting_time_new = apply_boost(
        vel=vel,
        boost_amount=cars.boost_amount,
        quat=cars.quat,
        is_on_ground=cars.is_on_ground,
        boost_input=controls.boost,
        active_mask=active_mask,
        is_boosting_prev=cars.is_boosting,
        boosting_time_prev=cars.boosting_time,
        dt=dt
    )
    
    # Apply torque to angular velocity
    tire_torque_per_wheel = jnp.cross(wheel_rel_pos, tire_impulse)
    tire_torque_total = jnp.sum(tire_torque_per_wheel, axis=-2)
    tire_torque_masked = jnp.where(cars.is_jumping[..., None], 0.0, tire_torque_total)
    
    sus_ang_delta = sus_torque_masked / CAR_INERTIA
    tire_ang_delta = tire_torque_masked * dt * UU_TO_BT / CAR_INERTIA
    flip_ang_delta = flip_torque * dt
    
    ang_vel = cars.ang_vel + jnp.where(
        active_mask[..., None],
        sus_ang_delta + tire_ang_delta + flip_ang_delta,
        0.0
    )
    
    # Air control
    is_airborne = ~cars.is_on_ground
    no_wheel_contact = num_contacts < 1
    up_dir = get_car_up_dir(cars.quat)
    right_dir = get_car_right_dir(cars.quat)
    
    # Pitch lock after flip
    pitch_locked = cars.has_flipped & (cars.flip_timer < (FLIP_TORQUE_TIME + FLIP_PITCHLOCK_EXTRA_TIME))
    pitch_locked_during_flip = cars.is_flipping
    pitch_scale = jnp.where(pitch_locked | pitch_locked_during_flip, 0.0, 1.0)
    
    air_torque_pitch = -right_dir * (controls.pitch[..., None] * pitch_scale[..., None] * CAR_AIR_CONTROL_TORQUE[0])
    air_torque_yaw = up_dir * (controls.yaw[..., None] * CAR_AIR_CONTROL_TORQUE[1])
    air_torque_roll = -forward_dir * (controls.roll[..., None] * CAR_AIR_CONTROL_TORQUE[2])
    
    ang_vel_pitch = jnp.sum(ang_vel * right_dir, axis=-1, keepdims=True)
    ang_vel_yaw = jnp.sum(ang_vel * up_dir, axis=-1, keepdims=True)
    ang_vel_roll = jnp.sum(ang_vel * forward_dir, axis=-1, keepdims=True)
    
    pitch_damp_factor = 1.0 - jnp.abs(controls.pitch[..., None] * pitch_scale[..., None])
    yaw_damp_factor = 1.0 - jnp.abs(controls.yaw[..., None])
    
    air_damp_pitch = -right_dir * ang_vel_pitch * CAR_AIR_CONTROL_DAMPING[0] * pitch_damp_factor
    air_damp_yaw = -up_dir * ang_vel_yaw * CAR_AIR_CONTROL_DAMPING[1] * yaw_damp_factor
    air_damp_roll = -forward_dir * ang_vel_roll * CAR_AIR_CONTROL_DAMPING[2]
    
    air_control_torque = (air_torque_pitch + air_torque_yaw + air_torque_roll +
                          air_damp_pitch + air_damp_yaw + air_damp_roll)
    air_ang_accel = air_control_torque * CAR_TORQUE_SCALE
    
    ang_vel = ang_vel + jnp.where(
        (is_airborne & no_wheel_contact & active_mask)[..., None],
        air_ang_accel * dt,
        0.0
    )
    
    # Throttle air acceleration when airborne
    has_throttle = jnp.abs(controls.throttle) > 0.001
    throttle_air_vel_delta = forward_dir * (controls.throttle[..., None] * THROTTLE_AIR_ACCEL * dt)
    vel = vel + jnp.where(
        (is_airborne & has_throttle & active_mask)[..., None],
        throttle_air_vel_delta,
        0.0
    )
    
    # Apply flip Z damping
    vel = apply_flip_z_damping(vel, cars.is_flipping, cars.flip_timer, dt)
    
    # Clamp velocities
    vel = clamp_velocity(vel, CAR_MAX_SPEED)
    ang_vel = clamp_angular_velocity(ang_vel, CAR_MAX_ANG_SPEED)
    
    # Integrate position
    pos = jnp.where(
        active_mask[..., None],
        integrate_position(cars.pos, vel, dt),
        cars.pos
    )
    
    # Integrate rotation
    quat = jnp.where(
        active_mask[..., None],
        integrate_rotation(cars.quat, ang_vel, dt),
        cars.quat
    )
    
    # Resolve arena collisions (car hitbox vs arena)
    pos, vel, ang_vel = resolve_car_arena_collision(
        pos=pos,
        vel=vel,
        ang_vel=ang_vel,
        quat=quat,
    )
    
    # Update ground contact state
    is_on_ground = num_contacts >= 3
    
    # When grounded, isFlipping is forced to false
    is_flipping_updated = jnp.where(is_on_ground, False, cars.is_flipping)
    
    # Update supersonic status
    is_supersonic, supersonic_timer = update_supersonic_status(
        vel=vel,
        is_supersonic=cars.is_supersonic,
        supersonic_timer=cars.supersonic_timer,
        dt=dt
    )
    
    return cars.replace(
        pos=pos,
        vel=vel,
        ang_vel=ang_vel,
        quat=quat,
        boost_amount=boost_amount,
        is_on_ground=is_on_ground,
        is_flipping=is_flipping_updated,
        wheel_contacts=wheel_contacts,
        is_supersonic=is_supersonic,
        supersonic_timer=supersonic_timer,
        is_boosting=is_boosting_new,
        boosting_time=boosting_time_new,
    )


# =============================================================================
# MAIN PHYSICS STEP
# =============================================================================


@jax.jit
def step_physics(
    state: PhysicsState, 
    controls: CarControls,
    dt: float = DT
) -> PhysicsState:
    """
    Main physics step function.
    
    PURE FUNCTION: No side effects, no mutations.
    state_new = step_physics(state_old, controls)
    
    Args:
        state: Current physics state
        controls: Car control inputs
        dt: Time step (default 1/120)
        
    Returns:
        New physics state after one tick
    """
    # Update ball
    new_ball = step_ball(state.ball, dt)
    
    # Update cars
    new_cars = step_cars(state.cars, controls, dt)
    
    # Resolve car-ball collisions
    new_ball_vel, new_ball_ang_vel, new_car_vel, new_car_ang_vel, hit_mask = resolve_car_ball_collision(
        new_ball.pos,
        new_ball.vel,
        new_ball.ang_vel,
        new_cars.pos,
        new_cars.vel,
        new_cars.ang_vel,
        new_cars.quat,
    )
    
    new_ball = new_ball.replace(vel=new_ball_vel, ang_vel=new_ball_ang_vel)
    new_cars = new_cars.replace(vel=new_car_vel, ang_vel=new_car_ang_vel)
    
    # Resolve car-car collisions
    car_vel_after, car_ang_vel_after, is_demoed_mask = resolve_car_car_collision(
        new_cars.pos,
        new_cars.vel,
        new_cars.ang_vel,
        new_cars.quat,
        new_cars.is_on_ground,
        new_cars.is_supersonic,
    )
    
    # Update demo state
    was_demoed = new_cars.is_demoed
    demo_timer = new_cars.demo_respawn_timer
    
    newly_demoed = is_demoed_mask & ~was_demoed
    new_demo_timer = jnp.where(newly_demoed, DEMO_RESPAWN_TIME, demo_timer - dt)
    new_demo_timer = jnp.maximum(new_demo_timer, 0.0)
    
    should_respawn = was_demoed & (new_demo_timer <= 0.0)
    is_demoed_final = (was_demoed | newly_demoed) & ~should_respawn
    
    car_vel_after = jnp.where(is_demoed_final[..., None], 0.0, car_vel_after)
    car_ang_vel_after = jnp.where(is_demoed_final[..., None], 0.0, car_ang_vel_after)
    
    new_cars = new_cars.replace(
        vel=car_vel_after,
        ang_vel=car_ang_vel_after,
        is_demoed=is_demoed_final,
        demo_respawn_timer=new_demo_timer
    )
    
    # Clamp velocities after collision
    new_ball = new_ball.replace(vel=clamp_velocity(new_ball.vel, BALL_MAX_SPEED))
    new_cars = new_cars.replace(
        vel=clamp_velocity(new_cars.vel, CAR_MAX_SPEED),
        ang_vel=clamp_angular_velocity(new_cars.ang_vel, CAR_MAX_ANG_SPEED),
    )
    
    # Resolve boost pad pickups
    new_car_boost, new_pad_is_active, new_pad_timers = resolve_boost_pads(
        new_cars.pos,
        new_cars.boost_amount,
        state.pad_is_active,
        state.pad_timers,
        dt
    )
    new_cars = new_cars.replace(boost_amount=new_car_boost)
    
    # Check for goals
    goal_threshold = GOAL_THRESHOLD_Y + BALL_RADIUS
    blue_scored = new_ball.pos[:, 1] > goal_threshold
    orange_scored = new_ball.pos[:, 1] < -goal_threshold
    
    new_tick_count = state.tick_count + 1
    
    return state.replace(
        ball=new_ball,
        cars=new_cars,
        tick_count=new_tick_count,
        pad_is_active=new_pad_is_active,
        pad_timers=new_pad_timers,
        blue_score=blue_scored,
        orange_score=orange_scored,
    )


# =============================================================================
# ROUND RESET
# =============================================================================


@jax.jit
def reset_round(
    state: PhysicsState,
    rng_key: jax.random.PRNGKey,
) -> PhysicsState:
    """
    Reset the round for kickoff after a goal.
    
    Args:
        state: Current physics state (used for shapes)
        rng_key: JAX random key for randomization
        
    Returns:
        New state configured for kickoff
    """
    n_envs = state.ball.pos.shape[0]
    max_cars = state.cars.pos.shape[1]
    
    key1, key2, key3 = jax.random.split(rng_key, 3)
    
    # Ball reset
    ball_pos = jnp.tile(
        jnp.array([0.0, 0.0, BALL_REST_Z])[None, :],
        (n_envs, 1)
    )
    ball_vel = jax.random.uniform(key1, (n_envs, 3), minval=-10.0, maxval=10.0)
    ball_vel = ball_vel.at[:, 2].set(0.0)
    ball_ang_vel = jnp.zeros((n_envs, 3))
    
    new_ball = BallState(
        pos=ball_pos,
        vel=ball_vel,
        ang_vel=ball_ang_vel,
    )
    
    # Car positions
    n_blue = 3
    n_orange = 3
    
    blue_indices = jax.random.randint(key2, (n_envs, n_blue), 0, 5)
    orange_indices = jax.random.randint(key3, (n_envs, n_orange), 0, 5)
    
    blue_positions = KICKOFF_POSITIONS_BLUE[blue_indices]
    orange_positions = KICKOFF_POSITIONS_ORANGE[orange_indices]
    
    car_positions = jnp.concatenate([blue_positions, orange_positions], axis=1)
    
    if max_cars < 6:
        car_positions = car_positions[:, :max_cars, :]
    elif max_cars > 6:
        extra = max_cars - 6
        extra_pos = jax.random.uniform(
            jax.random.fold_in(key2, 999),
            (n_envs, extra, 3),
            minval=-2000.0, maxval=2000.0
        )
        extra_pos = extra_pos.at[:, :, 2].set(17.0)
        car_positions = jnp.concatenate([car_positions, extra_pos], axis=1)
    
    # Car orientations
    blue_yaw = jnp.full((n_envs, n_blue), KICKOFF_YAW_BLUE)
    orange_yaw = jnp.full((n_envs, n_orange), KICKOFF_YAW_ORANGE)
    car_yaws = jnp.concatenate([blue_yaw, orange_yaw], axis=1)
    
    if max_cars < 6:
        car_yaws = car_yaws[:, :max_cars]
    elif max_cars > 6:
        extra_yaw = jnp.zeros((n_envs, max_cars - 6))
        car_yaws = jnp.concatenate([car_yaws, extra_yaw], axis=1)
    
    car_quats = quat_from_yaw(car_yaws)
    
    new_cars = CarState(
        pos=car_positions,
        vel=jnp.zeros((n_envs, max_cars, 3)),
        ang_vel=jnp.zeros((n_envs, max_cars, 3)),
        quat=car_quats,
        boost_amount=jnp.full((n_envs, max_cars), BOOST_SPAWN_AMOUNT),
        is_on_ground=jnp.ones((n_envs, max_cars), dtype=jnp.bool_),
        wheel_contacts=jnp.ones((n_envs, max_cars, 4), dtype=jnp.bool_),
        handbrake_val=jnp.zeros((n_envs, max_cars)),
        is_jumping=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        has_jumped=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        jump_timer=jnp.zeros((n_envs, max_cars)),
        has_flipped=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        has_double_jumped=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        is_flipping=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        flip_timer=jnp.zeros((n_envs, max_cars)),
        flip_rel_torque=jnp.zeros((n_envs, max_cars, 3)),
        air_time=jnp.zeros((n_envs, max_cars)),
        air_time_since_jump=jnp.zeros((n_envs, max_cars)),
        last_jump_pressed=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        is_demoed=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        demo_respawn_timer=jnp.zeros((n_envs, max_cars)),
        is_supersonic=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        supersonic_timer=jnp.zeros((n_envs, max_cars)),
        is_boosting=jnp.zeros((n_envs, max_cars), dtype=jnp.bool_),
        boosting_time=jnp.zeros((n_envs, max_cars)),
        team=state.cars.team,
    )
    
    new_pad_is_active = jnp.ones((n_envs, N_PADS_TOTAL), dtype=jnp.bool_)
    new_pad_timers = jnp.zeros((n_envs, N_PADS_TOTAL))
    
    return PhysicsState(
        ball=new_ball,
        cars=new_cars,
        tick_count=jnp.zeros(n_envs, dtype=jnp.int32),
        pad_is_active=new_pad_is_active,
        pad_timers=new_pad_timers,
        blue_score=jnp.zeros(n_envs, dtype=jnp.bool_),
        orange_score=jnp.zeros(n_envs, dtype=jnp.bool_),
    )


# =============================================================================
# RL ENVIRONMENT STEP
# =============================================================================


@jax.jit
def step_env(
    state: PhysicsState,
    controls: CarControls,
    rng_key: jax.random.PRNGKey,
) -> tuple:
    """
    Full RL environment step with physics, goal detection, and auto-reset.
    
    Args:
        state: Current physics state
        controls: Control inputs for all cars
        rng_key: JAX random key for reset randomization
        
    Returns:
        Tuple of (next_state, observations, rewards, dones)
    """
    from .observations import get_observations
    
    key, subkey = jax.random.split(rng_key)
    
    stepped_state = step_physics(state, controls)
    
    blue_scored = stepped_state.blue_score
    orange_scored = stepped_state.orange_score
    is_done = blue_scored | orange_scored
    
    n_envs = state.ball.pos.shape[0]
    max_cars = state.cars.pos.shape[1]
    
    blue_reward = blue_scored.astype(jnp.float32) - orange_scored.astype(jnp.float32)
    orange_reward = orange_scored.astype(jnp.float32) - blue_scored.astype(jnp.float32)
    
    is_blue_team = (stepped_state.cars.team == 0)
    rewards = jnp.where(
        is_blue_team,
        blue_reward[:, None],
        orange_reward[:, None]
    )
    
    reset_state = reset_round(stepped_state, subkey)
    
    def select_state(reset_val, step_val):
        if reset_val.ndim == 1:
            return jnp.where(is_done, reset_val, step_val)
        elif reset_val.ndim == 2:
            return jnp.where(is_done[:, None], reset_val, step_val)
        elif reset_val.ndim == 3:
            return jnp.where(is_done[:, None, None], reset_val, step_val)
        elif reset_val.ndim == 4:
            return jnp.where(is_done[:, None, None, None], reset_val, step_val)
        else:
            return reset_val
    
    next_state = jax.tree_util.tree_map(select_state, reset_state, stepped_state)
    
    observations = get_observations(next_state)
    
    return next_state, observations, rewards, is_done
