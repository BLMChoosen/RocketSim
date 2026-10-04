#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/sim_context.cuh"
#include "rocketsim_cuda/physics/arena_sdf.cuh"
#include "rocketsim_cuda/physics/integrator.cuh"
#include "rocketsim_cuda/physics/suspension.cuh"
#include "rocketsim_cuda/physics/contact_solver.cuh"
#include "rocketsim_cuda/physics/car_dynamics.cuh"

namespace rocketsim_cuda {

__device__ void StepBallDevice(
    uint32_t env_idx,
    BallStateSoA& ball_state,
    float dt)
{
    Vec3 pos(ball_state.pos_x[env_idx], ball_state.pos_y[env_idx], ball_state.pos_z[env_idx]);
    Vec3 vel(ball_state.vel_x[env_idx], ball_state.vel_y[env_idx], ball_state.vel_z[env_idx]);
    Vec3 ang_vel(ball_state.ang_vel_x[env_idx], ball_state.ang_vel_y[env_idx], ball_state.ang_vel_z[env_idx]);
    Quat quat(ball_state.q_w[env_idx], ball_state.q_x[env_idx], ball_state.q_y[env_idx], ball_state.q_z[env_idx]);

    // Check if sleeping (zero velocity on ground)
    if (vel.length_sq() == 0.0f && ang_vel.length_sq() == 0.0f && pos.z <= BALL_REST_Z + 0.05f) {
        return;
    }

    // Damping
    apply_rigid_body_damping(vel, ang_vel, BALL_DRAG, 0.0f, dt);

    // Gravity
    vel.z += GRAVITY_Z * dt;

    // Linear pos (in Bullet units for exact rounding parity)
    pos = (pos * 0.02f + vel * (0.02f * dt)) * 50.0f;

    // Arena / Ground collision
    resolve_ball_arena_collision(pos, vel, ang_vel);

    // Rotation integration
    quat = bullet_integrate_quaternion(quat, ang_vel, dt);

    // Write back coalesced
    ball_state.pos_x[env_idx] = pos.x;
    ball_state.pos_y[env_idx] = pos.y;
    ball_state.pos_z[env_idx] = pos.z;

    ball_state.vel_x[env_idx] = vel.x;
    ball_state.vel_y[env_idx] = vel.y;
    ball_state.vel_z[env_idx] = vel.z;

    ball_state.q_w[env_idx] = quat.w;
    ball_state.q_x[env_idx] = quat.x;
    ball_state.q_y[env_idx] = quat.y;
    ball_state.q_z[env_idx] = quat.z;

    ball_state.ang_vel_x[env_idx] = ang_vel.x;
    ball_state.ang_vel_y[env_idx] = ang_vel.y;
    ball_state.ang_vel_z[env_idx] = ang_vel.z;
}

__device__ void StepCarDevice(
    uint32_t car_idx,
    CarStateSoA& car_state,
    const CarControlsSoA& controls,
    const float* __restrict__ actions_tensor,
    float dt)
{
    // 1. Controls: Direct VRAM tensor consumption or fallback to CarControlsSoA
    CarControls ctrl;
    if (actions_tensor) {
        const float* a = actions_tensor + car_idx * 8;
        ctrl.throttle  = a[0];
        ctrl.steer     = a[1];
        ctrl.pitch     = a[2];
        ctrl.yaw       = a[3];
        ctrl.roll      = a[4];
        ctrl.jump      = (a[5] > 0.5f) ? 1 : 0;
        ctrl.boost     = (a[6] > 0.5f) ? 1 : 0;
        ctrl.handbrake = (a[7] > 0.5f) ? 1 : 0;
        ctrl.padding   = 0;
        ctrl.clamp_fix();
    } else {
        ctrl.throttle  = controls.throttle[car_idx];
        ctrl.steer     = controls.steer[car_idx];
        ctrl.pitch     = controls.pitch[car_idx];
        ctrl.yaw       = controls.yaw[car_idx];
        ctrl.roll      = controls.roll[car_idx];
        ctrl.boost     = controls.boost[car_idx];
        ctrl.jump      = controls.jump[car_idx];
        ctrl.handbrake = controls.handbrake[car_idx];
        ctrl.padding   = 0;
        ctrl.clamp_fix();
    }

    // 2. Load car state
    Vec3 pos(car_state.pos_x[car_idx], car_state.pos_y[car_idx], car_state.pos_z[car_idx]);
    Vec3 pos_bt = pos * 0.02f;
    Vec3 vel(car_state.vel_x[car_idx], car_state.vel_y[car_idx], car_state.vel_z[car_idx]);
    Vec3 omega(car_state.ang_vel_x[car_idx], car_state.ang_vel_y[car_idx], car_state.ang_vel_z[car_idx]);
    Quat quat(car_state.q_w[car_idx], car_state.q_x[car_idx], car_state.q_y[car_idx], car_state.q_z[car_idx]);

    Mat3 basis = Mat3::from_quat(quat);

    // 3. Wheel raycast query (btVehicleRL::updateVehicleFirst)
    uint8_t wheels_contact[4] = {0};
    float susp_lengths[4] = {0};
    WheelRaycastResult wheel_results[4];

    evaluate_car_wheels_raycast(
        pos, basis,
        wheels_contact, susp_lengths,
        wheel_results
    );

    int num_wheels_contact = wheels_contact[0] + wheels_contact[1] + wheels_contact[2] + wheels_contact[3];
    bool is_on_ground = (num_wheels_contact >= 3);

    // 4. Load previous tick's cached wheel dynamics
    float cached_engine_force = car_state.wheel_engine_force[car_idx];
    float cached_brake = car_state.wheel_brake[car_idx];
    float cached_steer_angle = car_state.wheel_steer_angle[car_idx];
    float cached_lat_frictions[4] = {
        car_state.wheel_lat_friction_0[car_idx],
        car_state.wheel_lat_friction_1[car_idx],
        car_state.wheel_lat_friction_2[car_idx],
        car_state.wheel_lat_friction_3[car_idx]
    };
    float cached_long_frictions[4] = {
        car_state.wheel_long_friction_0[car_idx],
        car_state.wheel_long_friction_1[car_idx],
        car_state.wheel_long_friction_2[car_idx],
        car_state.wheel_long_friction_3[car_idx]
    };

    // 5. Apply suspension & bilateral tire friction impulses
    Vec3 vel_bt = vel * 0.02f;
    apply_suspension_and_friction(
        pos, basis, wheel_results, dt,
        cached_engine_force, cached_brake, cached_steer_angle,
        cached_lat_frictions, cached_long_frictions,
        vel_bt, omega
    );
    vel = vel_bt * 50.0f;

    // 6. Update wheel dynamics (throttle, brake, steer, friction curves, sticky downforce) for NEXT tick
    Vec3 total_force(0.0f, 0.0f, 0.0f);
    Vec3 contact_normals[4] = {
        wheel_results[0].contact_normal,
        wheel_results[1].contact_normal,
        wheel_results[2].contact_normal,
        wheel_results[3].contact_normal
    };

    update_car_wheel_dynamics(
        car_idx, car_state, ctrl,
        num_wheels_contact, wheels_contact,
        contact_normals, basis,
        vel, omega, dt,
        total_force
    );

    // 7. Air control vs flipping reset
    float fwd_speed = vel.dot(basis.forward);
    if (num_wheels_contact < 3) {
        bool allow_air_torque = (num_wheels_contact == 0);
        update_car_air_control(car_idx, car_state, ctrl, basis, dt, omega, total_force, allow_air_torque);
    } else {
        car_state.is_flipping[car_idx] = 0;
    }

    // 8. Turtle recovery (auto-flip)
    update_car_auto_flip(car_idx, car_state, ctrl, basis, dt, vel, omega);

    // 9. Jump, double jump, flip/dodge
    update_car_jump(car_idx, car_state, ctrl, is_on_ground, basis, fwd_speed, dt, vel, total_force);

    // 10. Surface alignment (auto-roll)
    if (ctrl.throttle != 0.0f && ((num_wheels_contact > 0 && num_wheels_contact < 4) || car_state.world_contact_has_contact[car_idx])) {
        update_car_auto_roll(
            car_idx, car_state, num_wheels_contact, wheels_contact,
            contact_normals, basis, dt, total_force, omega
        );
    }

    // Clear world contact has contact flag after auto-roll / auto-flip have consumed it
    car_state.world_contact_has_contact[car_idx] = 0;

    // 11. Boost update (persists minimum boost time and fuel)
    update_car_boost(car_idx, car_state, ctrl, is_on_ground, basis, dt, total_force);

    // 12. Gravity
    total_force.z += GRAVITY_Z * CAR_MASS;

    // 13. Symplectic Euler linear integration (in Bullet units for exact rounding parity)
    vel = vel + total_force * ((1.0f / CAR_MASS) * dt);
    pos_bt = pos_bt + (vel * 0.02f) * dt;
    pos = pos_bt * 50.0f;

    // 14. Angular dynamics
    Vec3 total_torque(0.0f, 0.0f, 0.0f);
    bullet_angular_dynamics(omega, total_torque, get_octane_inv_inertia_local(), basis, dt);

    // 15. Chassis arena contact
    resolve_chassis_arena_collision(car_idx, car_state, pos, vel, omega, basis, dt);
    pos_bt = pos * 0.02f;

    // 14. Quaternion integration
    quat = bullet_integrate_quaternion(quat, omega, dt);

    // 15. Velocity limiting (clamping)
    float speed_sq = vel.length_sq();
    if (speed_sq > CAR_MAX_SPEED * CAR_MAX_SPEED) {
        vel = vel * (CAR_MAX_SPEED / sqrtf(speed_sq));
    }
    float ang_speed_sq = omega.length_sq();
    if (ang_speed_sq > CAR_MAX_ANG_SPEED * CAR_MAX_ANG_SPEED) {
        omega = omega * (CAR_MAX_ANG_SPEED / sqrtf(ang_speed_sq));
    }

    // 16. Supersonic status update
    update_car_supersonic(car_idx, car_state, vel, dt);

    // 17. Write back SoA
    car_state.pos_bt_x[car_idx] = pos_bt.x;
    car_state.pos_bt_y[car_idx] = pos_bt.y;
    car_state.pos_bt_z[car_idx] = pos_bt.z;

    car_state.pos_x[car_idx] = pos.x;
    car_state.pos_y[car_idx] = pos.y;
    car_state.pos_z[car_idx] = pos.z;

    car_state.vel_x[car_idx] = vel.x;
    car_state.vel_y[car_idx] = vel.y;
    car_state.vel_z[car_idx] = vel.z;

    car_state.q_w[car_idx] = quat.w;
    car_state.q_x[car_idx] = quat.x;
    car_state.q_y[car_idx] = quat.y;
    car_state.q_z[car_idx] = quat.z;

    car_state.ang_vel_x[car_idx] = omega.x;
    car_state.ang_vel_y[car_idx] = omega.y;
    car_state.ang_vel_z[car_idx] = omega.z;

    car_state.is_on_ground[car_idx] = is_on_ground ? 1 : 0;

    car_state.wheel_contact_0[car_idx] = wheels_contact[0];
    car_state.wheel_contact_1[car_idx] = wheels_contact[1];
    car_state.wheel_contact_2[car_idx] = wheels_contact[2];
    car_state.wheel_contact_3[car_idx] = wheels_contact[3];

    car_state.suspension_length_0[car_idx] = susp_lengths[0];
    car_state.suspension_length_1[car_idx] = susp_lengths[1];
    car_state.suspension_length_2[car_idx] = susp_lengths[2];
    car_state.suspension_length_3[car_idx] = susp_lengths[3];

    // Last Controls
    car_state.last_controls_throttle[car_idx]   = ctrl.throttle;
    car_state.last_controls_steer[car_idx]      = ctrl.steer;
    car_state.last_controls_pitch[car_idx]      = ctrl.pitch;
    car_state.last_controls_yaw[car_idx]        = ctrl.yaw;
    car_state.last_controls_roll[car_idx]       = ctrl.roll;
    car_state.last_controls_boost[car_idx]      = ctrl.boost;
    car_state.last_controls_jump[car_idx]       = ctrl.jump;
    car_state.last_controls_handbrake[car_idx]  = ctrl.handbrake;
}

__global__ void StepSimulationKernel(
    uint32_t num_envs,
    uint32_t cars_per_env,
    BallStateSoA ball_state,
    CarStateSoA car_state,
    ArenaStateSoA arena_state,
    CarControlsSoA controls,
    const float* __restrict__ actions_tensor,
    float dt)
{
    uint32_t env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx >= num_envs) return;

    // Step Ball
    StepBallDevice(env_idx, ball_state, dt);

    // Step Cars
    for (uint32_t c = 0; c < cars_per_env; ++c) {
        uint32_t car_idx = env_idx * cars_per_env + c;
        StepCarDevice(car_idx, car_state, controls, actions_tensor, dt);
    }

    // Increment Arena Tick Count
    if (arena_state.tick_count) {
        arena_state.tick_count[env_idx]++;
    }

    // Resolve Car-Ball Collisions
    for (uint32_t c = 0; c < cars_per_env; ++c) {
        uint32_t car_idx = env_idx * cars_per_env + c;
        if (car_state.is_demoed && car_state.is_demoed[car_idx]) continue;
        resolve_car_ball_collision(env_idx, car_idx, ball_state, car_state, arena_state, dt);
    }

    // Step Arena Termination & Boost Pads
    if (arena_state.is_goal) {
        float bx = ball_state.pos_x[env_idx];
        float by = ball_state.pos_y[env_idx];
        float bz = ball_state.pos_z[env_idx];
        float b_vy = ball_state.vel_y[env_idx];

        uint8_t goal_flag = 0;
        uint8_t score_team = 0;
        if (fabsf(bx) < GOAL_WIDTH * 0.5f && bz < GOAL_HEIGHT) {
            if (by > ARENA_EXTENT_Y) {
                goal_flag = 1;
                score_team = 0;
            } else if (by < -ARENA_EXTENT_Y) {
                goal_flag = 1;
                score_team = 1;
            }
        }
        arena_state.is_goal[env_idx] = goal_flag;
        arena_state.scoring_team[env_idx] = score_team;

        uint8_t oob_flag = 0;
        if (bz > ARENA_HEIGHT + 200.0f || fabsf(bx) > ARENA_EXTENT_X + 500.0f || fabsf(by) > ARENA_EXTENT_Y + 1200.0f) {
            oob_flag = 1;
        }
        arena_state.is_out_of_bounds[env_idx] = oob_flag;

        if (arena_state.terminated) {
            arena_state.terminated[env_idx] = (goal_flag || oob_flag) ? 1 : 0;
        }
        if (arena_state.truncated) {
            arena_state.truncated[env_idx] = 0;
        }

        if (arena_state.rewards) {
            for (uint32_t c = 0; c < cars_per_env; ++c) {
                uint32_t car_idx = env_idx * cars_per_env + c;
                uint8_t team = (c % 2 == 0) ? 0 : 1;
                float goal_dir = (team == 0) ? 1.0f : -1.0f;
                float rew = (b_vy * goal_dir) * (1.0f / 6000.0f);
                if (goal_flag) {
                    rew += (score_team == team) ? 1.0f : -1.0f;
                }
                arena_state.rewards[car_idx] = rew;
            }
        }

        if (arena_state.pad_cooldown) {
            for (uint32_t p = 0; p < MAX_BOOST_PADS; ++p) {
                uint32_t pad_idx = env_idx * MAX_BOOST_PADS + p;
                float cd = arena_state.pad_cooldown[pad_idx];
                if (cd > 0.0f) {
                    cd -= dt;
                    if (cd <= 0.0f) {
                        cd = 0.0f;
                        if (arena_state.pad_is_active) {
                            arena_state.pad_is_active[pad_idx] = 1;
                        }
                    }
                    arena_state.pad_cooldown[pad_idx] = cd;
                }
            }
        }

        // Proximity Boost Pickup for cars in this environment
        if (arena_state.pad_is_active) {
            for (uint32_t c = 0; c < cars_per_env; ++c) {
                uint32_t car_idx = env_idx * cars_per_env + c;
                if (car_state.is_demoed && car_state.is_demoed[car_idx]) continue;
                float current_boost = car_state.boost[car_idx];
                if (current_boost >= BOOST_MAX) continue;

                Vec3 car_pos(car_state.pos_x[car_idx], car_state.pos_y[car_idx], car_state.pos_z[car_idx]);
                for (uint32_t p = 0; p < MAX_BOOST_PADS; ++p) {
                    uint32_t pad_idx = env_idx * MAX_BOOST_PADS + p;
                    if (arena_state.pad_is_active[pad_idx]) {
                        const BoostPadDef& pad = SOCCAR_BOOST_PADS[p];
                        float dz = fabsf(car_pos.z - pad.z);
                        if (dz < 95.0f) {
                            float dx = car_pos.x - pad.x;
                            float dy = car_pos.y - pad.y;
                            if ((dx * dx + dy * dy) < pad.radius_sq) {
                                current_boost = fminf(current_boost + pad.boost_amount, BOOST_MAX);
                                car_state.boost[car_idx] = current_boost;
                                arena_state.pad_is_active[pad_idx] = 0;
                                if (arena_state.pad_cooldown) {
                                    arena_state.pad_cooldown[pad_idx] = pad.cooldown;
                                }
                                if (current_boost >= BOOST_MAX) {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void sim_step_batch(SimContext* ctx, uint32_t batch_size, const CarControlsSoA* controls, const float* actions_tensor) {
    if (!ctx) return;
    uint32_t num_envs = (batch_size > 0) ? batch_size : ctx->GetNumEnvs();
    uint32_t cars_per_env = ctx->GetCarsPerEnv();

    const CarControlsSoA& ctrl_soa = controls ? *controls : ctx->GetControls();
    cudaStream_t stream = ctx->GetStream();

    uint32_t block_size = 128;
    uint32_t grid_size = (num_envs + block_size - 1) / block_size;

    StepSimulationKernel<<<grid_size, block_size, 0, stream>>>(
        num_envs,
        cars_per_env,
        ctx->GetBallState(),
        ctx->GetCarState(),
        ctx->GetArenaState(),
        ctrl_soa,
        actions_tensor,
        DELTA_TIME
    );
}

void SimContext::Step(uint32_t batch_size, const float* actions_tensor) {
    sim_step_batch(this, batch_size, nullptr, actions_tensor);
}

} // namespace rocketsim_cuda
