#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/quat.cuh"
#include "rocketsim_cuda/math/mat3.cuh"
#include "rocketsim_cuda/types/car_state.cuh"
#include "rocketsim_cuda/types/car_controls.cuh"

namespace rocketsim_cuda {

// Mechanics constants from RLConst
constexpr float JUMP_ACCEL                  = 4375.0f / 3.0f;
constexpr float JUMP_IMMEDIATE_FORCE        = 875.0f / 3.0f;
constexpr float JUMP_MIN_TIME               = 0.025f;
constexpr float JUMP_RESET_TIME_PAD         = 1.0f / 40.0f;
constexpr float JUMP_MAX_TIME               = 0.2f;
constexpr float DOUBLEJUMP_MAX_DELAY        = 1.25f;

constexpr float FLIP_Z_DAMP_120             = 0.35f;
constexpr float FLIP_Z_DAMP_START           = 0.15f;
constexpr float FLIP_Z_DAMP_END             = 0.21f;
constexpr float FLIP_TORQUE_TIME            = 0.65f;
constexpr float FLIP_TORQUE_MIN_TIME        = 0.41f;
constexpr float FLIP_PITCHLOCK_TIME         = 1.0f;
constexpr float FLIP_PITCHLOCK_EXTRA_TIME   = 0.3f;
constexpr float FLIP_INITIAL_VEL_SCALE      = 500.0f;
constexpr float FLIP_TORQUE_X               = 260.0f; // Left/Right
constexpr float FLIP_TORQUE_Y               = 224.0f; // Forward/backward
constexpr float FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE = 1.0f;
constexpr float FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE    = 1.9f;
constexpr float FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE= 2.5f;
constexpr float FLIP_BACKWARD_IMPULSE_SCALE_X        = 16.0f / 15.0f;

constexpr float CAR_AIR_CONTROL_TORQUE_X    = 130.0f;
constexpr float CAR_AIR_CONTROL_TORQUE_Y    = 95.0f;
constexpr float CAR_AIR_CONTROL_TORQUE_Z    = 400.0f;

constexpr float CAR_AIR_CONTROL_DAMPING_X   = 30.0f;
constexpr float CAR_AIR_CONTROL_DAMPING_Y   = 20.0f;
constexpr float CAR_AIR_CONTROL_DAMPING_Z   = 50.0f;

constexpr float CAR_TORQUE_SCALE            = 2.0f * 3.14159265358979323846f / 65536.0f * 1000.0f;
constexpr float THROTTLE_AIR_ACCEL          = 200.0f / 3.0f;

/**
 * @brief Updates car jump, double jump, and dodge/flip mechanics.
 */
__device__ __forceinline__ void update_car_jump(
    uint32_t car_idx,
    CarStateSoA& car_state,
    const CarControls& controls,
    bool is_on_ground,
    const Mat3& basis,
    float forward_speed,
    float dt,
    Vec3& vel,
    Vec3& total_force)
{
    bool jump_pressed = controls.jump && !(car_state.last_controls_jump[car_idx]);
    uint8_t has_jumped = car_state.has_jumped[car_idx];
    uint8_t is_jumping = car_state.is_jumping[car_idx];
    float jump_time = car_state.jump_time[car_idx];

    // Reset jump when settled on ground
    if (is_on_ground && !is_jumping) {
        if (has_jumped && jump_time < JUMP_MIN_TIME + JUMP_RESET_TIME_PAD) {
            // Keep state while leaving ground
        } else {
            has_jumped = 0;
            jump_time = 0.0f;
        }
    }

    if (is_jumping) {
        if (jump_time < JUMP_MIN_TIME || (controls.jump && jump_time < JUMP_MAX_TIME)) {
            is_jumping = 1;
        } else {
            is_jumping = 0;
        }
    } else if (is_on_ground && jump_pressed) {
        // Start jumping: instant impulse along chassis Up vector
        is_jumping = 1;
        jump_time = 0.0f;
        vel = vel + basis.up * JUMP_IMMEDIATE_FORCE;
    }

    if (is_jumping) {
        has_jumped = 1;
        float jump_accel = JUMP_ACCEL;
        if (jump_time < JUMP_MIN_TIME) {
            jump_accel *= 0.62f;
        }
        total_force = total_force + basis.up * (jump_accel * CAR_MASS);
    }

    if (is_jumping || has_jumped) {
        jump_time += dt;
    }

    car_state.is_jumping[car_idx] = is_jumping;
    car_state.has_jumped[car_idx] = has_jumped;
    car_state.jump_time[car_idx] = jump_time;

    // --- Double Jump or Flip ---
    uint8_t has_double_jumped = car_state.has_double_jumped[car_idx];
    uint8_t has_flipped = car_state.has_flipped[car_idx];
    uint8_t is_flipping = car_state.is_flipping[car_idx];
    float air_time = car_state.air_time[car_idx];
    float air_time_since_jump = car_state.air_time_since_jump[car_idx];
    float flip_time = car_state.flip_time[car_idx];

    if (is_on_ground) {
        has_double_jumped = 0;
        has_flipped = 0;
        air_time = 0.0f;
        air_time_since_jump = 0.0f;
        flip_time = 0.0f;
    } else {
        air_time += dt;
        if (has_jumped && !is_jumping) {
            air_time_since_jump += dt;
        } else {
            air_time_since_jump = 0.0f;
        }

        if (jump_pressed && air_time_since_jump < DOUBLEJUMP_MAX_DELAY) {
            float input_mag = fabsf(controls.yaw) + fabsf(controls.pitch) + fabsf(controls.roll);
            bool is_flip_input = (input_mag >= 0.5f);

            bool can_use = (!has_double_jumped && !has_flipped);
            if (can_use) {
                if (is_flip_input) {
                    flip_time = 0.0f;
                    has_flipped = 1;
                    is_flipping = 1;

                    float forward_ratio = fabsf(forward_speed) / CAR_MAX_SPEED;
                    Vec3 dodge_dir(-controls.pitch, controls.yaw + controls.roll, 0.0f);
                    if (fabsf(controls.yaw + controls.roll) < 0.1f && fabsf(controls.pitch) < 0.1f) {
                        dodge_dir = Vec3(0.0f, 0.0f, 0.0f);
                    } else {
                        dodge_dir = dodge_dir.normalized();
                    }

                    car_state.flip_rel_torque_x[car_idx] = -dodge_dir.y;
                    car_state.flip_rel_torque_y[car_idx] = dodge_dir.x;
                    car_state.flip_rel_torque_z[car_idx] = 0.0f;

                    if (fabsf(dodge_dir.x) < 0.1f) dodge_dir.x = 0.0f;
                    if (fabsf(dodge_dir.y) < 0.1f) dodge_dir.y = 0.0f;

                    if (dodge_dir.length_sq() > 0.001f) {
                        bool should_dodge_backward;
                        if (fabsf(forward_speed) < 100.0f) {
                            should_dodge_backward = (dodge_dir.x < 0.0f);
                        } else {
                            should_dodge_backward = (dodge_dir.x >= 0.0f) != (forward_speed >= 0.0f);
                        }

                        Vec3 init_dodge_vel = dodge_dir * FLIP_INITIAL_VEL_SCALE;
                        float max_speed_scale_x = should_dodge_backward ?
                            FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE : FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE;

                        init_dodge_vel.x *= ((max_speed_scale_x - 1.0f) * forward_ratio) + 1.0f;
                        init_dodge_vel.y *= ((FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE - 1.0f) * forward_ratio) + 1.0f;

                        if (should_dodge_backward) {
                            init_dodge_vel.x *= FLIP_BACKWARD_IMPULSE_SCALE_X;
                        }

                        // 2D horizontal projection of forward vector
                        Vec3 fwd2d(basis.forward.x, basis.forward.y, 0.0f);
                        if (fwd2d.length_sq() > 1e-6f) fwd2d = fwd2d.normalized();
                        Vec3 rgt2d(-fwd2d.y, fwd2d.x, 0.0f);

                        Vec3 delta_vel = fwd2d * init_dodge_vel.x + rgt2d * init_dodge_vel.y;
                        vel = vel + delta_vel;
                    }
                } else {
                    vel = vel + basis.up * JUMP_IMMEDIATE_FORCE;
                    has_double_jumped = 1;
                }
            }
        }
    }

    if (is_flipping) {
        flip_time += dt;
        if (flip_time <= FLIP_TORQUE_TIME) {
            if (flip_time >= FLIP_Z_DAMP_START && (vel.z < 0.0f || flip_time < FLIP_Z_DAMP_END)) {
                vel.z *= (1.0f - FLIP_Z_DAMP_120);
            }
        }
    } else if (has_flipped) {
        flip_time += dt;
    }

    car_state.has_double_jumped[car_idx] = has_double_jumped;
    car_state.has_flipped[car_idx] = has_flipped;
    car_state.is_flipping[car_idx] = is_flipping;
    car_state.air_time[car_idx] = air_time;
    car_state.air_time_since_jump[car_idx] = air_time_since_jump;
    car_state.flip_time[car_idx] = flip_time;
}

/**
 * @brief Updates air control torques and pitch-lock/damping.
 */
__device__ __forceinline__ void update_car_air_control(
    uint32_t car_idx,
    CarStateSoA& car_state,
    const CarControls& controls,
    const Mat3& basis,
    float dt,
    Vec3& omega,
    Vec3& total_force)
{
    // Air throttle
    if (controls.throttle != 0.0f) {
        total_force = total_force + basis.forward * (controls.throttle * THROTTLE_AIR_ACCEL * CAR_MASS);
    }

    Vec3 dir_pitch = basis.right * -1.0f;
    Vec3 dir_yaw = basis.up;
    Vec3 dir_roll = basis.forward * -1.0f;

    bool is_flipping = (car_state.is_flipping[car_idx] != 0);
    float flip_time = car_state.flip_time[car_idx];

    if (is_flipping) {
        is_flipping = (car_state.has_flipped[car_idx] && flip_time < FLIP_TORQUE_TIME);
        car_state.is_flipping[car_idx] = is_flipping ? 1 : 0;
    }

    bool do_air_control = false;
    if (is_flipping) {
        Vec3 rel_dodge_torque(
            car_state.flip_rel_torque_x[car_idx],
            car_state.flip_rel_torque_y[car_idx],
            0.0f
        );

        if (rel_dodge_torque.length_sq() > 0.001f) {
            float pitch_scale = 1.0f;
            if (rel_dodge_torque.y != 0.0f && controls.pitch != 0.0f) {
                float sgn_rel = (rel_dodge_torque.y > 0.0f) ? 1.0f : -1.0f;
                float sgn_ctrl = (controls.pitch > 0.0f) ? 1.0f : -1.0f;
                if (sgn_rel == sgn_ctrl) {
                    pitch_scale = 1.0f - fabsf(controls.pitch);
                    do_air_control = true;
                }
            }
            rel_dodge_torque.y *= pitch_scale;
            Vec3 dodge_torque(
                rel_dodge_torque.x * FLIP_TORQUE_X,
                rel_dodge_torque.y * FLIP_TORQUE_Y,
                0.0f
            );
            // Bullet cancels out inertia in flip torque: alpha = basis * dodgeTorque
            omega = omega + basis * dodge_torque * dt;
        } else {
            do_air_control = true;
        }
    } else {
        do_air_control = true;
    }

    if (do_air_control) {
        float pitch_torque_scale = 1.0f;
        if (is_flipping) {
            pitch_torque_scale = 0.0f;
        } else if (car_state.has_flipped[car_idx] && flip_time < FLIP_TORQUE_TIME + FLIP_PITCHLOCK_EXTRA_TIME) {
            pitch_torque_scale = 0.0f;
        }

        Vec3 air_torque = dir_pitch * (controls.pitch * pitch_torque_scale * CAR_AIR_CONTROL_TORQUE_X)
                        + dir_yaw * (controls.yaw * CAR_AIR_CONTROL_TORQUE_Y)
                        + dir_roll * (controls.roll * CAR_AIR_CONTROL_TORQUE_Z);

        float damp_pitch = dir_pitch.dot(omega) * CAR_AIR_CONTROL_DAMPING_X * (1.0f - fabsf(controls.pitch * pitch_torque_scale));
        float damp_yaw = dir_yaw.dot(omega) * CAR_AIR_CONTROL_DAMPING_Y * (1.0f - fabsf(controls.yaw));
        float damp_roll = dir_roll.dot(omega) * CAR_AIR_CONTROL_DAMPING_Z;

        Vec3 air_damping = dir_yaw * damp_yaw + dir_pitch * damp_pitch + dir_roll * damp_roll;
        Vec3 delta_omega = (air_torque - air_damping) * (CAR_TORQUE_SCALE * dt);
        omega = omega + delta_omega;
    }
}

} // namespace rocketsim_cuda
