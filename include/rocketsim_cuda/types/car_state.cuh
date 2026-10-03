#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "../math/vec3.cuh"
#include "../math/quat.cuh"
#include "car_controls.cuh"

namespace rocketsim_cuda {

// Host POD structure for Car (used in Golden Master & Interop)
struct CarStatePOD {
    Vec3 pos     = Vec3(0.0f, 0.0f, 17.0f);
    Vec3 vel     = Vec3(0.0f, 0.0f, 0.0f);
    Quat quat    = Quat::identity();
    Vec3 ang_vel = Vec3(0.0f, 0.0f, 0.0f);

    float boost = 33.33333f;
    uint8_t is_on_ground        = 1;
    uint8_t has_jumped          = 0;
    uint8_t has_double_jumped   = 0;
    uint8_t has_flipped         = 0;
    uint8_t is_demoed           = 0;
    uint8_t wheels_with_contact[4] = {1, 1, 1, 1};
    float suspension_lengths[4]    = {0.0f, 0.0f, 0.0f, 0.0f};
    CarControls last_controls;
};

// Device SoA structure for N cars (passed by value to CUDA kernels)
struct CarStateSoA {
    // --- 1. Rigid Body Transform & Velocities (Coalesced 128-byte transactions) ---
    float* __restrict__ pos_x = nullptr;
    float* __restrict__ pos_y = nullptr;
    float* __restrict__ pos_z = nullptr;

    float* __restrict__ vel_x = nullptr;
    float* __restrict__ vel_y = nullptr;
    float* __restrict__ vel_z = nullptr;

    float* __restrict__ q_w = nullptr;
    float* __restrict__ q_x = nullptr;
    float* __restrict__ q_y = nullptr;
    float* __restrict__ q_z = nullptr;

    float* __restrict__ ang_vel_x = nullptr;
    float* __restrict__ ang_vel_y = nullptr;
    float* __restrict__ ang_vel_z = nullptr;

    // --- 2. Boost Mechanics ---
    float* __restrict__ boost              = nullptr;
    float* __restrict__ time_since_boosted = nullptr;
    float* __restrict__ boosting_time      = nullptr;
    uint8_t* __restrict__ is_boosting      = nullptr;

    // --- 3. Wheel & Suspension States (4 Wheels) ---
    uint8_t* __restrict__ is_on_ground     = nullptr;
    uint8_t* __restrict__ wheel_contact_0  = nullptr;
    uint8_t* __restrict__ wheel_contact_1  = nullptr;
    uint8_t* __restrict__ wheel_contact_2  = nullptr;
    uint8_t* __restrict__ wheel_contact_3  = nullptr;

    float* __restrict__ suspension_length_0 = nullptr;
    float* __restrict__ suspension_length_1 = nullptr;
    float* __restrict__ suspension_length_2 = nullptr;
    float* __restrict__ suspension_length_3 = nullptr;

    // --- 4. Jump & Air Mechanics ---
    uint8_t* __restrict__ has_jumped          = nullptr;
    uint8_t* __restrict__ is_jumping          = nullptr;
    float* __restrict__ jump_time             = nullptr;
    uint8_t* __restrict__ has_double_jumped   = nullptr;
    float* __restrict__ air_time              = nullptr;
    float* __restrict__ air_time_since_jump   = nullptr;

    // --- 5. Flip & Dodge Mechanics ---
    uint8_t* __restrict__ has_flipped      = nullptr;
    uint8_t* __restrict__ is_flipping      = nullptr;
    float* __restrict__ flip_time          = nullptr;
    float* __restrict__ flip_rel_torque_x  = nullptr;
    float* __restrict__ flip_rel_torque_y  = nullptr;
    float* __restrict__ flip_rel_torque_z  = nullptr;

    // --- 6. Auto-Flip & Handbrake ---
    uint8_t* __restrict__ is_auto_flipping        = nullptr;
    float* __restrict__ auto_flip_timer           = nullptr;
    float* __restrict__ auto_flip_torque_scale    = nullptr;
    float* __restrict__ handbrake_val             = nullptr;

    // --- 7. Supersonic & Demo ---
    uint8_t* __restrict__ is_supersonic        = nullptr;
    float* __restrict__ supersonic_time        = nullptr;
    uint8_t* __restrict__ is_demoed            = nullptr;
    float* __restrict__ demo_respawn_timer     = nullptr;

    // --- 8. World & Car Contacts ---
    uint8_t* __restrict__ world_contact_has_contact  = nullptr;
    float* __restrict__ world_contact_normal_x       = nullptr;
    float* __restrict__ world_contact_normal_y       = nullptr;
    float* __restrict__ world_contact_normal_z       = nullptr;
    int32_t* __restrict__ car_contact_other_car_id   = nullptr;
    float* __restrict__ car_contact_cooldown_timer   = nullptr;

    // --- 9. Ball Hit Information ---
    uint8_t* __restrict__ ball_hit_is_valid           = nullptr;
    float* __restrict__ ball_hit_rel_pos_x            = nullptr;
    float* __restrict__ ball_hit_rel_pos_y            = nullptr;
    float* __restrict__ ball_hit_rel_pos_z            = nullptr;
    float* __restrict__ ball_hit_extra_hit_force_x    = nullptr;
    float* __restrict__ ball_hit_extra_hit_force_y    = nullptr;
    float* __restrict__ ball_hit_extra_hit_force_z    = nullptr;
    uint64_t* __restrict__ ball_hit_tick_count        = nullptr;

    // --- 10. Last Controls ---
    float* __restrict__ last_controls_throttle   = nullptr;
    float* __restrict__ last_controls_steer      = nullptr;
    float* __restrict__ last_controls_pitch      = nullptr;
    float* __restrict__ last_controls_yaw        = nullptr;
    float* __restrict__ last_controls_roll       = nullptr;
    uint8_t* __restrict__ last_controls_boost    = nullptr;
    uint8_t* __restrict__ last_controls_jump     = nullptr;
    uint8_t* __restrict__ last_controls_handbrake= nullptr;

    // Device helper methods
    __device__ inline Vec3 get_pos(uint32_t idx) const {
        return Vec3(pos_x[idx], pos_y[idx], pos_z[idx]);
    }
    __device__ inline void set_pos(uint32_t idx, const Vec3& p) {
        pos_x[idx] = p.x; pos_y[idx] = p.y; pos_z[idx] = p.z;
    }

    __device__ inline Vec3 get_vel(uint32_t idx) const {
        return Vec3(vel_x[idx], vel_y[idx], vel_z[idx]);
    }
    __device__ inline void set_vel(uint32_t idx, const Vec3& v) {
        vel_x[idx] = v.x; vel_y[idx] = v.y; vel_z[idx] = v.z;
    }

    __device__ inline Quat get_quat(uint32_t idx) const {
        return Quat(q_w[idx], q_x[idx], q_y[idx], q_z[idx]);
    }
    __device__ inline void set_quat(uint32_t idx, const Quat& q) {
        q_w[idx] = q.w; q_x[idx] = q.x; q_y[idx] = q.y; q_z[idx] = q.z;
    }

    __device__ inline Vec3 get_ang_vel(uint32_t idx) const {
        return Vec3(ang_vel_x[idx], ang_vel_y[idx], ang_vel_z[idx]);
    }
    __device__ inline void set_ang_vel(uint32_t idx, const Vec3& w) {
        ang_vel_x[idx] = w.x; ang_vel_y[idx] = w.y; ang_vel_z[idx] = w.z;
    }

    __device__ inline CarStatePOD get_pod(uint32_t idx) const {
        CarStatePOD pod;
        pod.pos     = get_pos(idx);
        pod.vel     = get_vel(idx);
        pod.quat    = get_quat(idx);
        pod.ang_vel = get_ang_vel(idx);

        pod.boost                  = boost[idx];
        pod.is_on_ground           = is_on_ground[idx];
        pod.has_jumped             = has_jumped[idx];
        pod.has_double_jumped      = has_double_jumped[idx];
        pod.has_flipped            = has_flipped[idx];
        pod.is_demoed              = is_demoed[idx];
        pod.wheels_with_contact[0] = wheel_contact_0[idx];
        pod.wheels_with_contact[1] = wheel_contact_1[idx];
        pod.wheels_with_contact[2] = wheel_contact_2[idx];
        pod.wheels_with_contact[3] = wheel_contact_3[idx];

        pod.suspension_lengths[0]  = suspension_length_0[idx];
        pod.suspension_lengths[1]  = suspension_length_1[idx];
        pod.suspension_lengths[2]  = suspension_length_2[idx];
        pod.suspension_lengths[3]  = suspension_length_3[idx];

        pod.last_controls.throttle  = last_controls_throttle[idx];
        pod.last_controls.steer     = last_controls_steer[idx];
        pod.last_controls.pitch     = last_controls_pitch[idx];
        pod.last_controls.yaw       = last_controls_yaw[idx];
        pod.last_controls.roll      = last_controls_roll[idx];
        pod.last_controls.boost     = last_controls_boost[idx];
        pod.last_controls.jump      = last_controls_jump[idx];
        pod.last_controls.handbrake = last_controls_handbrake[idx];
        return pod;
    }

    __device__ inline void set_pod(uint32_t idx, const CarStatePOD& pod) {
        set_pos(idx, pod.pos);
        set_vel(idx, pod.vel);
        set_quat(idx, pod.quat);
        set_ang_vel(idx, pod.ang_vel);

        boost[idx]                 = pod.boost;
        is_on_ground[idx]          = pod.is_on_ground;
        has_jumped[idx]            = pod.has_jumped;
        has_double_jumped[idx]     = pod.has_double_jumped;
        has_flipped[idx]           = pod.has_flipped;
        is_demoed[idx]             = pod.is_demoed;
        wheel_contact_0[idx]       = pod.wheels_with_contact[0];
        wheel_contact_1[idx]       = pod.wheels_with_contact[1];
        wheel_contact_2[idx]       = pod.wheels_with_contact[2];
        wheel_contact_3[idx]       = pod.wheels_with_contact[3];

        suspension_length_0[idx]   = pod.suspension_lengths[0];
        suspension_length_1[idx]   = pod.suspension_lengths[1];
        suspension_length_2[idx]   = pod.suspension_lengths[2];
        suspension_length_3[idx]   = pod.suspension_lengths[3];

        last_controls_throttle[idx]   = pod.last_controls.throttle;
        last_controls_steer[idx]      = pod.last_controls.steer;
        last_controls_pitch[idx]      = pod.last_controls.pitch;
        last_controls_yaw[idx]        = pod.last_controls.yaw;
        last_controls_roll[idx]       = pod.last_controls.roll;
        last_controls_boost[idx]      = pod.last_controls.boost;
        last_controls_jump[idx]       = pod.last_controls.jump;
        last_controls_handbrake[idx]  = pod.last_controls.handbrake;
    }
};

} // namespace rocketsim_cuda
