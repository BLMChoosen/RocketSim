#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/quat.cuh"
#include "rocketsim_cuda/math/mat3.cuh"
#include "rocketsim_cuda/types/car_state.cuh"
#include "rocketsim_cuda/types/car_controls.cuh"
#include "rocketsim_cuda/physics/suspension.cuh"

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
constexpr float FLIP_PITCHLOCK_EXTRA_TIME   = 0.3f;
constexpr float FLIP_INITIAL_VEL_SCALE      = 500.0f;
constexpr float FLIP_TORQUE_X               = 260.0f; // Left/Right
constexpr float FLIP_TORQUE_Y               = 224.0f; // Forward/backward
constexpr float FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE  = 1.0f;
constexpr float FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE     = 1.9f;
constexpr float FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE = 2.5f;
constexpr float FLIP_BACKWARD_IMPULSE_SCALE_X         = 16.0f / 15.0f;

constexpr float CAR_AIR_CONTROL_TORQUE_X    = 130.0f;
constexpr float CAR_AIR_CONTROL_TORQUE_Y    = 95.0f;
constexpr float CAR_AIR_CONTROL_TORQUE_Z    = 400.0f;

constexpr float CAR_AIR_CONTROL_DAMPING_X   = 30.0f;
constexpr float CAR_AIR_CONTROL_DAMPING_Y   = 20.0f;
constexpr float CAR_AIR_CONTROL_DAMPING_Z   = 50.0f;

constexpr float CAR_TORQUE_SCALE            = 2.0f * 3.14159265358979323846f / 65536.0f * 1000.0f;
constexpr float THROTTLE_AIR_ACCEL          = 200.0f / 3.0f;

constexpr float BRAKING_NO_THROTTLE_SPEED_THRESH = 0.01f;

constexpr float CAR_AUTOROLL_FORCE          = 100.0f;
constexpr float CAR_AUTOROLL_TORQUE         = 80.0f;

constexpr float CAR_AUTOFLIP_IMPULSE        = 200.0f;
constexpr float CAR_AUTOFLIP_TORQUE         = 50.0f;
constexpr float CAR_AUTOFLIP_TIME           = 0.4f;
constexpr float CAR_AUTOFLIP_NORMZ_THRESH   = 0.7071067811865475f; // M_SQRT1_2
constexpr float CAR_AUTOFLIP_ROLL_THRESH    = 2.8f;

// Piecewise curve functions
__device__ __forceinline__ float get_drive_torque_factor(float abs_fwd_speed) {
    if (abs_fwd_speed <= 0.0f) return 1.0f;
    if (abs_fwd_speed < 1400.0f) return 1.0f - (abs_fwd_speed / 1400.0f) * 0.9f;
    if (abs_fwd_speed < 1410.0f) return 0.1f - ((abs_fwd_speed - 1400.0f) / 10.0f) * 0.1f;
    return 0.0f;
}

__device__ __forceinline__ float get_steer_angle(float speed) {
    if (speed <= 0.0f) return 0.53356f;
    if (speed < 500.0f) return 0.53356f + (speed / 500.0f) * (0.31930f - 0.53356f);
    if (speed < 1000.0f) return 0.31930f + ((speed - 500.0f) / 500.0f) * (0.18203f - 0.31930f);
    if (speed < 1500.0f) return 0.18203f + ((speed - 1000.0f) / 500.0f) * (0.10570f - 0.18203f);
    if (speed < 1750.0f) return 0.10570f + ((speed - 1500.0f) / 250.0f) * (0.08507f - 0.10570f);
    if (speed < 3000.0f) return 0.08507f + ((speed - 1750.0f) / 1250.0f) * (0.03454f - 0.08507f);
    return 0.03454f;
}

__device__ __forceinline__ float get_powerslide_steer_angle(float speed) {
    if (speed <= 0.0f) return 0.39235f;
    if (speed < 2500.0f) return 0.39235f + (speed / 2500.0f) * (0.12610f - 0.39235f);
    return 0.12610f;
}

__device__ __forceinline__ float get_non_sticky_friction_scale(float normal_z) {
    if (normal_z <= 0.0f) return 0.1f;
    if (normal_z < 0.7075f) return 0.1f + (normal_z / 0.7075f) * (0.5f - 0.1f);
    if (normal_z < 1.0f) return 0.5f + ((normal_z - 0.7075f) / (1.0f - 0.7075f)) * (1.0f - 0.5f);
    return 1.0f;
}

__device__ __forceinline__ float get_lat_friction(float slip) {
    if (slip <= 0.0f) return 1.0f;
    if (slip < 1.0f) return 1.0f - 0.8f * slip;
    return 0.2f;
}

__device__ __forceinline__ float get_handbrake_long_friction(float slip) {
    if (slip <= 0.0f) return 0.5f;
    if (slip < 1.0f) return 0.5f + 0.4f * slip;
    return 0.9f;
}

/**
 * @brief Faithful boost persistence and consumption matching Car::_UpdateBoost.
 */
__device__ __forceinline__ void update_car_boost(
    uint32_t car_idx,
    CarStateSoA& car_state,
    const CarControls& ctrl,
    bool is_on_ground,
    const Mat3& basis,
    float dt,
    Vec3& total_force)
{
    float boost = car_state.boost[car_idx];
    bool is_boosting = (car_state.is_boosting[car_idx] != 0);
    float boosting_time = car_state.boosting_time[car_idx];

    if (boost > 0.0f) {
        if (is_boosting) {
            is_boosting = (ctrl.boost != 0) || (boosting_time < BOOST_MIN_TIME);
        } else {
            is_boosting = (ctrl.boost != 0);
        }
    } else {
        is_boosting = false;
    }

    if (is_boosting) {
        boosting_time += dt;
        boost = fmaxf(0.0f, boost - (100.0f / 3.0f) * dt);
        float boost_accel = is_on_ground ? BOOST_ACCEL_GROUND : BOOST_ACCEL_AIR;
        total_force = total_force + basis.forward * (boost_accel * CAR_MASS);
        car_state.time_since_boosted[car_idx] = 0.0f;
    } else {
        boosting_time = 0.0f;
        car_state.time_since_boosted[car_idx] += dt;
    }

    car_state.boost[car_idx] = boost;
    car_state.is_boosting[car_idx] = is_boosting ? 1 : 0;
    car_state.boosting_time[car_idx] = boosting_time;
}

/**
 * @brief Updates handbrake, wheel engine/brake forces, steer angles, friction curves, and sticky downforce.
 */
__device__ __forceinline__ void update_car_wheel_dynamics(
    uint32_t car_idx,
    CarStateSoA& car_state,
    const CarControls& ctrl,
    int num_wheels_in_contact,
    const uint8_t* wheels_contact,
    const Vec3* contact_normals,
    const Mat3& basis,
    const Vec3& vel,
    const Vec3& omega,
    float dt,
    Vec3& total_force)
{
    float fwd_speed = vel.dot(basis.forward);
    float abs_fwd_speed = fabsf(fwd_speed);

    // 1. Handbrake value update
    float handbrake_val = car_state.handbrake_val[car_idx];
    if (ctrl.handbrake) {
        handbrake_val += 5.0f * dt;
    } else {
        handbrake_val -= 2.0f * dt;
    }
    handbrake_val = fminf(fmaxf(handbrake_val, 0.0f), 1.0f);
    car_state.handbrake_val[car_idx] = handbrake_val;

    // 2. Throttle & Brake forces
    float real_throttle = ctrl.throttle;
    float real_brake = 0.0f;
    if (ctrl.boost && car_state.boost[car_idx] > 0.0f) {
        real_throttle = 1.0f;
    }

    float drive_speed_scale = get_drive_torque_factor(abs_fwd_speed);
    float engine_throttle = real_throttle;

    if (!ctrl.handbrake) {
        float abs_throttle = fabsf(real_throttle);
        if (abs_throttle >= 0.001f) {
            if (abs_fwd_speed > 25.0f && ((real_throttle > 0.0f) != (fwd_speed > 0.0f))) {
                real_brake = 1.0f;
                if (abs_fwd_speed > BRAKING_NO_THROTTLE_SPEED_THRESH) {
                    engine_throttle = 0.0f;
                }
            }
        } else {
            engine_throttle = 0.0f;
            real_brake = (abs_fwd_speed < 25.0f) ? 1.0f : 0.15f;
        }
    }

    if (num_wheels_in_contact < 3) {
        drive_speed_scale /= 4.0f;
    }

    float drive_engine_force = engine_throttle * 1440.0f * drive_speed_scale; // 180 * 400 * 0.02 = 1440
    float drive_brake_force = real_brake * 52.5f;                             // 180 * 14.58333 * 0.02 = 52.5

    car_state.wheel_engine_force[car_idx] = drive_engine_force;
    car_state.wheel_brake[car_idx] = drive_brake_force;

    // 3. Steer angle
    float steer_angle = get_steer_angle(abs_fwd_speed);
    if (handbrake_val > 0.0f) {
        steer_angle += (get_powerslide_steer_angle(abs_fwd_speed) - steer_angle) * handbrake_val;
    }
    steer_angle *= ctrl.steer;
    car_state.wheel_steer_angle[car_idx] = steer_angle;

    // 4. Per-wheel friction coefficients
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (!wheels_contact[w]) continue;

        Vec3 hit_normal = contact_normals[w];
        float steer = (w < 2) ? steer_angle : 0.0f;
        Vec3 lat_dir = basis.right * cosf(steer) - basis.forward * sinf(steer);
        Vec3 long_dir = lat_dir.cross(hit_normal);

        Vec3 wheel_offset = get_octane_wheel_offset(w);
        Vec3 wheel_delta = basis * wheel_offset;
        Vec3 cross_vec = omega.cross(wheel_delta) + vel;

        float base_friction = fabsf(cross_vec.dot(lat_dir));
        float friction_curve_input = 0.0f;
        if (base_friction > 5.0f) {
            friction_curve_input = base_friction / (fabsf(cross_vec.dot(long_dir)) + base_friction);
        }

        float lat_fric = get_lat_friction(friction_curve_input);
        float long_fric = 1.0f;

        if (handbrake_val > 0.0f) {
            lat_fric *= (0.1f - 1.0f) * handbrake_val + 1.0f;
            long_fric *= (get_handbrake_long_friction(friction_curve_input) - 1.0f) * handbrake_val + 1.0f;
        }

        if (real_throttle == 0.0f) {
            float non_sticky = get_non_sticky_friction_scale(hit_normal.z);
            lat_fric *= non_sticky;
            long_fric *= non_sticky;
        }

        if (w == 0) car_state.wheel_lat_friction_0[car_idx] = lat_fric;
        else if (w == 1) car_state.wheel_lat_friction_1[car_idx] = lat_fric;
        else if (w == 2) car_state.wheel_lat_friction_2[car_idx] = lat_fric;
        else car_state.wheel_lat_friction_3[car_idx] = lat_fric;

        if (w == 0) car_state.wheel_long_friction_0[car_idx] = long_fric;
        else if (w == 1) car_state.wheel_long_friction_1[car_idx] = long_fric;
        else if (w == 2) car_state.wheel_long_friction_2[car_idx] = long_fric;
        else car_state.wheel_long_friction_3[car_idx] = long_fric;
    }

    // 5. Sticky Downforce (Car.cpp:485-497)
    if (num_wheels_in_contact > 0) {
        Vec3 sum_normals(0.0f, 0.0f, 0.0f);
        for (int w = 0; w < 4; ++w) {
            if (wheels_contact[w]) {
                sum_normals = sum_normals + contact_normals[w];
            }
        }
        Vec3 upwards_dir = (sum_normals.length_sq() > 1e-6f) ? sum_normals.normalized() : basis.up;
        bool full_stick = (real_throttle != 0.0f) || (abs_fwd_speed > 25.0f);
        float sticky_scale = 0.5f + (full_stick ? (1.0f - fabsf(upwards_dir.z)) : 0.0f);
        total_force = total_force + upwards_dir * (sticky_scale * GRAVITY_Z * CAR_MASS);
    }
}

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
            if (car_state.is_auto_flipping[car_idx]) {
                can_use = false;
            }
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
    Vec3& total_force,
    bool allow_air_torque = true)
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

    Vec3 omega_pre = omega;

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
            omega = omega + basis * dodge_torque * dt;
        } else {
            do_air_control = true;
        }
    } else {
        do_air_control = true;
    }

    do_air_control = do_air_control && allow_air_torque && (car_state.is_auto_flipping[car_idx] == 0);

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

        // Evaluate damping using pre-torque angular velocity, mirroring Bullet Car.cpp:665-677
        float damp_pitch = dir_pitch.dot(omega_pre) * CAR_AIR_CONTROL_DAMPING_X * (1.0f - fabsf(controls.pitch * pitch_torque_scale));
        float damp_yaw = dir_yaw.dot(omega_pre) * CAR_AIR_CONTROL_DAMPING_Y * (1.0f - fabsf(controls.yaw));
        float damp_roll = dir_roll.dot(omega_pre) * CAR_AIR_CONTROL_DAMPING_Z;

        Vec3 air_damping = dir_yaw * damp_yaw + dir_pitch * damp_pitch + dir_roll * damp_roll;
        Vec3 delta_omega = (air_torque - air_damping) * (CAR_TORQUE_SCALE * dt);
        omega = omega + delta_omega;
    }
}

/**
 * @brief Turtle recovery (auto-flip) matching Car::_UpdateAutoFlip.
 */
__device__ __forceinline__ void update_car_auto_flip(
    uint32_t car_idx,
    CarStateSoA& car_state,
    const CarControls& ctrl,
    const Mat3& basis,
    float dt,
    Vec3& vel,
    Vec3& omega)
{
    bool jump_pressed = ctrl.jump && !(car_state.last_controls_jump[car_idx]);

    if (jump_pressed &&
        car_state.world_contact_has_contact[car_idx] &&
        car_state.world_contact_normal_z[car_idx] > CAR_AUTOFLIP_NORMZ_THRESH)
    {
        // Extract roll angle matching Angle::FromRotMat
        float pitch = asinf(fmaxf(-1.0f, fminf(1.0f, -basis.forward.z)));
        float roll = atan2f(basis.right.z, basis.up.z);
        constexpr float HALF_PI = 1.5707963267948966f;
        constexpr float PI_VAL  = 3.14159265358979323846f;
        if (fabsf(pitch) >= HALF_PI - 1e-4f) {
            if (roll > 0.0f) roll -= PI_VAL;
            else roll += PI_VAL;
        }
        float angle_roll = -roll;

        float abs_roll = fabsf(angle_roll);
        if (abs_roll > CAR_AUTOFLIP_ROLL_THRESH) {
            car_state.auto_flip_timer[car_idx] = CAR_AUTOFLIP_TIME * (abs_roll / PI_VAL);
            car_state.auto_flip_torque_scale[car_idx] = (angle_roll > 0.0f) ? 1.0f : -1.0f;
            car_state.is_auto_flipping[car_idx] = 1;

            // Apply upward jump impulse away from ground
            vel = vel - basis.up * CAR_AUTOFLIP_IMPULSE;
        }
    }

    if (car_state.is_auto_flipping[car_idx]) {
        float timer = car_state.auto_flip_timer[car_idx];
        if (timer <= 0.0f) {
            car_state.is_auto_flipping[car_idx] = 0;
            car_state.auto_flip_timer[car_idx] = 0.0f;
        } else {
            omega = omega + basis.forward * (CAR_AUTOFLIP_TORQUE * car_state.auto_flip_torque_scale[car_idx] * dt);
            timer -= dt;
            car_state.auto_flip_timer[car_idx] = fmaxf(0.0f, timer);
            if (timer <= 0.0f) {
                car_state.is_auto_flipping[car_idx] = 0;
            }
        }
    }
}

/**
 * @brief Surface alignment (auto-roll) matching Car::_UpdateAutoRoll.
 */
__device__ __forceinline__ void update_car_auto_roll(
    uint32_t car_idx,
    const CarStateSoA& car_state,
    int num_wheels_in_contact,
    const uint8_t* wheels_contact,
    const Vec3* contact_normals,
    const Mat3& basis,
    float dt,
    Vec3& total_force,
    Vec3& omega)
{
    Vec3 ground_up_dir;
    if (num_wheels_in_contact > 0) {
        Vec3 sum_contact_dir(0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (wheels_contact[w]) {
                sum_contact_dir = sum_contact_dir + contact_normals[w];
            }
        }
        ground_up_dir = (sum_contact_dir.length_sq() > 1e-6f) ? sum_contact_dir.normalized() : basis.up;
    } else {
        ground_up_dir = Vec3(
            car_state.world_contact_normal_x[car_idx],
            car_state.world_contact_normal_y[car_idx],
            car_state.world_contact_normal_z[car_idx]
        );
        if (ground_up_dir.length_sq() > 1e-6f) {
            ground_up_dir = ground_up_dir.normalized();
        } else {
            ground_up_dir = basis.up;
        }
    }

    Vec3 ground_down_dir = ground_up_dir * -1.0f;
    Vec3 forward_dir = basis.forward;
    Vec3 right_dir   = basis.right;

    Vec3 cross_right_dir   = ground_up_dir.cross(forward_dir);
    Vec3 cross_forward_dir = ground_down_dir.cross(cross_right_dir);

    float right_torque_factor   = 1.0f - fminf(fmaxf(right_dir.dot(cross_right_dir), 0.0f), 1.0f);
    float forward_torque_factor = 1.0f - fminf(fmaxf(forward_dir.dot(cross_forward_dir), 0.0f), 1.0f);

    Vec3 torque_dir_right   = forward_dir * (right_dir.dot(ground_up_dir) >= 0.0f ? -1.0f : 1.0f);
    Vec3 torque_dir_forward = right_dir * (forward_dir.dot(ground_up_dir) >= 0.0f ? 1.0f : -1.0f);

    Vec3 torque_right   = torque_dir_right * right_torque_factor;
    Vec3 torque_forward = torque_dir_forward * forward_torque_factor;

    total_force = total_force + ground_down_dir * (CAR_AUTOROLL_FORCE * CAR_MASS);
    omega = omega + (torque_forward + torque_right) * (CAR_AUTOROLL_TORQUE * dt);
}

/**
 * @brief Updates supersonic status matching Car::_PostTickUpdate.
 */
__device__ __forceinline__ void update_car_supersonic(
    uint32_t car_idx,
    CarStateSoA& car_state,
    const Vec3& vel,
    float dt)
{
    float speed_sq = vel.length_sq();
    bool is_super = (car_state.is_supersonic[car_idx] != 0);
    float super_time = car_state.supersonic_time[car_idx];

    if (is_super && super_time < SUPERSONIC_MAINTAIN_MAX_TIME) {
        is_super = (speed_sq >= SUPERSONIC_MAINTAIN_MIN_SPEED * SUPERSONIC_MAINTAIN_MIN_SPEED);
    } else {
        is_super = (speed_sq >= SUPERSONIC_START_SPEED * SUPERSONIC_START_SPEED);
    }

    if (is_super) {
        super_time += dt;
    } else {
        super_time = 0.0f;
    }

    car_state.is_supersonic[car_idx] = is_super ? 1 : 0;
    car_state.supersonic_time[car_idx] = super_time;
}

} // namespace rocketsim_cuda
