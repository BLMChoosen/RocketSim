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

    // Check if sleeping (zero velocity)
    if (vel.length_sq() == 0.0f && ang_vel.length_sq() == 0.0f) {
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
    Vec3 pos(car_state.pos_x[car_idx], car_state.pos_y[car_idx], car_state.pos_z[car_idx]);
    Vec3 vel(car_state.vel_x[car_idx], car_state.vel_y[car_idx], car_state.vel_z[car_idx]);
    Vec3 omega(car_state.ang_vel_x[car_idx], car_state.ang_vel_y[car_idx], car_state.ang_vel_z[car_idx]);
    Quat quat(car_state.q_w[car_idx], car_state.q_x[car_idx], car_state.q_y[car_idx], car_state.q_z[car_idx]);
    float boost = car_state.boost[car_idx];

    Mat3 basis = Mat3::from_quat(quat);

    // Controls: Direct VRAM tensor consumption or fallback to CarControlsSoA
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
    }

    // Suspension
    uint8_t wheels_contact[4] = {0};
    float susp_lengths[4] = {0};
    Vec3 susp_impulse(0.0f, 0.0f, 0.0f);
    Vec3 susp_torque_impulse(0.0f, 0.0f, 0.0f);

    update_car_suspension(
        pos, vel, omega, basis, dt,
        wheels_contact, susp_lengths,
        susp_impulse, susp_torque_impulse
    );

    int num_wheels_contact = wheels_contact[0] + wheels_contact[1] + wheels_contact[2] + wheels_contact[3];
    bool is_on_ground = (num_wheels_contact >= 3);

    // Forces accumulation
    Vec3 total_force = susp_impulse * (1.0f / dt);
    Vec3 total_torque = susp_torque_impulse * (1.0f / dt);

    float fwd_speed = vel.dot(basis.forward);

    // Drive torque from throttle or Air Control
    if (is_on_ground) {
        float abs_fwd_speed = fabsf(fwd_speed);
        float drive_scale = (abs_fwd_speed < 1400.0f) ? (1.0f - (abs_fwd_speed / 1400.0f) * 0.9f) : 0.1f;
        float drive_force_mag = ctrl.throttle * (400.0f * CAR_MASS * 0.02f) * drive_scale * 50.0f;
        total_force = total_force + basis.forward * drive_force_mag;
    } else {
        update_car_air_control(car_idx, car_state, ctrl, basis, dt, omega, total_force);
    }

    // Jump & Flip mechanics
    update_car_jump(car_idx, car_state, ctrl, is_on_ground, basis, fwd_speed, dt, vel, total_force);

    // Boost
    if (ctrl.boost && boost > 0.0f) {
        float boost_accel = is_on_ground ? BOOST_ACCEL_GROUND : BOOST_ACCEL_AIR;
        total_force = total_force + basis.forward * (boost_accel * CAR_MASS);
        boost = fmaxf(0.0f, boost - BOOST_CONSUMPTION_RATE * dt);
    }

    // Gravity
    total_force.z += GRAVITY_Z * CAR_MASS;

    // Symplectic Euler Linear Integration (in Bullet units for exact rounding parity)
    vel = vel + total_force * ((1.0f / CAR_MASS) * dt);
    pos = (pos * 0.02f + vel * (0.02f * dt)) * 50.0f;

    // Angular Dynamics
    bullet_angular_dynamics(omega, total_torque, get_octane_inv_inertia_local(), basis, dt);

    // Chassis Arena Contact
    resolve_chassis_arena_collision(pos, vel, omega, basis, dt);

    // Quaternion Integration
    quat = bullet_integrate_quaternion(quat, omega, dt);

    // Write back SoA
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

    car_state.boost[car_idx] = boost;
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

    // Step Arena Termination & Boost Pads
    if (arena_state.tick_count) {
        arena_state.tick_count[env_idx]++;
    }

    if (arena_state.is_goal) {
        float bx = ball_state.pos_x[env_idx];
        float by = ball_state.pos_y[env_idx];
        float bz = ball_state.pos_z[env_idx];

        if (fabsf(bx) < GOAL_WIDTH * 0.5f && bz < GOAL_HEIGHT) {
            if (by > ARENA_EXTENT_Y) {
                arena_state.is_goal[env_idx] = 1;
                arena_state.scoring_team[env_idx] = 0;
            } else if (by < -ARENA_EXTENT_Y) {
                arena_state.is_goal[env_idx] = 1;
                arena_state.scoring_team[env_idx] = 1;
            }
        }

        if (bz > ARENA_HEIGHT + 200.0f || fabsf(bx) > ARENA_EXTENT_X + 500.0f || fabsf(by) > ARENA_EXTENT_Y + 1200.0f) {
            arena_state.is_out_of_bounds[env_idx] = 1;
        }

        if (arena_state.pad_cooldown) {
            for (uint32_t p = 0; p < MAX_BOOST_PADS; ++p) {
                uint32_t pad_idx = env_idx * MAX_BOOST_PADS + p;
                float cd = arena_state.pad_cooldown[pad_idx];
                if (cd > 0.0f) {
                    cd -= dt;
                    if (cd <= 0.0f) {
                        cd = 0.0f;
                        arena_state.pad_is_active[pad_idx] = 1;
                    }
                    arena_state.pad_cooldown[pad_idx] = cd;
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
