#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/quat.cuh"
#include "rocketsim_cuda/math/mat3.cuh"
#include "rocketsim_cuda/physics/arena_sdf.cuh"

namespace rocketsim_cuda {

// Suspension & Tire Tuning Constants (RLConst::BTVehicle)
constexpr float SUSP_STIFFNESS                  = 500.0f; // in Bullet units
constexpr float SUSP_DAMPING_COMPRESSION        = 25.0f;
constexpr float SUSP_DAMPING_RELAXATION         = 40.0f;
constexpr float SUSP_MAX_TRAVEL                 = 12.0f;  // UU
constexpr float SUSP_SUBTRACTION                = 2.5f;   // 0.05f BT = 2.5f UU
constexpr float SUSP_FORCE_SCALE_FRONT          = 35.75f; // 36 - 1/4
constexpr float SUSP_FORCE_SCALE_BACK           = 54.265f;// 54 + 1/4 + 1.5/100

// Octane Wheel Geometry (CarConfig.cpp)
__device__ __forceinline__ Vec3 get_octane_wheel_offset(int w) {
    switch (w) {
        case 0: return Vec3( 51.25f,  25.90f, 20.755f);
        case 1: return Vec3( 51.25f, -25.90f, 20.755f);
        case 2: return Vec3(-33.75f,  29.50f, 20.755f);
        default: return Vec3(-33.75f, -29.50f, 20.755f);
    }
}

__device__ __forceinline__ float get_octane_wheel_rad(int w) {
    return (w < 2) ? 12.50f : 15.00f;
}

__device__ __forceinline__ float get_octane_susp_rest(int w) {
    return (w < 2) ? 26.755f : 25.055f;
}

__device__ __forceinline__ float get_octane_force_scale(int w) {
    return (w < 2) ? SUSP_FORCE_SCALE_FRONT : SUSP_FORCE_SCALE_BACK;
}

// Octane Inverse Inertia Diag (Local coords in UU)
__device__ __forceinline__ Vec3 get_octane_inv_inertia_local() {
    return Vec3(
        1.0f / (54.06816f * 2500.0f),
        1.0f / (96.098775f * 2500.0f),
        1.0f / (132.232635f * 2500.0f)
    );
}

struct SuspensionQueryResult {
    bool in_contact;
    float suspension_length;     // Current length in UU
    float compression;           // Rest - length
    Vec3 contact_point;
    Vec3 contact_normal;
    Vec3 force_impulse;          // Linear impulse on chassis
    Vec3 torque_impulse;         // Angular impulse on chassis
};

/**
 * @brief Evaluates suspension query and force accumulation for 4 wheels of a car.
 */
__device__ __forceinline__ void update_car_suspension(
    const Vec3& car_pos,
    const Vec3& car_vel,
    const Vec3& car_omega,
    const Mat3& basis,
    float dt,
    uint8_t* __restrict__ wheels_in_contact,
    float* __restrict__ suspension_lengths,
    Vec3& out_total_impulse,
    Vec3& out_total_torque_impulse)
{
    out_total_impulse = Vec3(0.0f, 0.0f, 0.0f);
    out_total_torque_impulse = Vec3(0.0f, 0.0f, 0.0f);

    Vec3 up_dir = basis.up;
    Vec3 wheel_dir = up_dir * -1.0f;

    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        Vec3 hardpoint = car_pos + basis * get_octane_wheel_offset(w);
        float rest_len = get_octane_susp_rest(w);
        float radius = get_octane_wheel_rad(w);
        float real_ray_len = rest_len + SUSP_MAX_TRAVEL + radius - SUSP_SUBTRACTION;

        float hit_dist = 0.0f;
        Vec3 hit_normal = Vec3(0.0f, 0.0f, 1.0f);
        bool hit = raycast_arena_sdf(hardpoint, wheel_dir, real_ray_len, &hit_dist, &hit_normal);

        if (hit) {
            wheels_in_contact[w] = 1;
            float wheel_trace_len = hit_dist;
            float cur_susp_len = fminf(fmaxf(wheel_trace_len - radius, rest_len - SUSP_MAX_TRAVEL), rest_len + SUSP_MAX_TRAVEL);
            suspension_lengths[w] = rest_len - cur_susp_len; // Compression stored for state

            Vec3 contact_pt = hardpoint + wheel_dir * hit_dist;
            Vec3 rel_pos = contact_pt - car_pos;
            Vec3 vel_at_pt = car_vel + car_omega.cross(rel_pos);

            float denominator = hit_normal.dot(up_dir);
            float inv_dot = (denominator > 0.1f) ? (1.0f / denominator) : 10.0f;
            float proj_vel = hit_normal.dot(vel_at_pt);
            float v_rel = (denominator > 0.1f) ? (proj_vel * inv_dot) : 0.0f;

            // Spring & Damping force
            // stiffness in UU/s^2 equivalent: 500 * 0.02 = 10.0
            float spring_force = (rest_len - cur_susp_len) * 10.0f * inv_dot;
            float damping_scale = (v_rel < 0.0f) ? SUSP_DAMPING_COMPRESSION : SUSP_DAMPING_RELAXATION;
            float susp_force = spring_force - (damping_scale * (v_rel * 0.02f));
            susp_force *= get_octane_force_scale(w);
            if (susp_force < 0.0f) susp_force = 0.0f;

            // Extra Pushback
            float extra_pushback = 0.0f;
            float pushback_thresh = (rest_len + radius) - SUSP_SUBTRACTION;
            if (wheel_trace_len < pushback_thresh) {
                float dist_delta = wheel_trace_len - pushback_thresh;
                float pos_error = 0.2f * (-dist_delta) / dt;
                float vel_error = -proj_vel;
                float denom = (1.0f / CAR_MASS) + 0.0001f; // approximated effective compliance
                extra_pushback = fmaxf(0.0f, (pos_error + vel_error) / denom) * 0.25f;
            }

            // Normal impulse
            float base_scale = (susp_force * 50.0f * dt) + extra_pushback;
            Vec3 linear_imp = hit_normal * base_scale;
            Vec3 torque_imp = rel_pos.cross(linear_imp);

            out_total_impulse = out_total_impulse + linear_imp;
            out_total_torque_impulse = out_total_torque_impulse + torque_imp;
        } else {
            wheels_in_contact[w] = 0;
            suspension_lengths[w] = -SUSP_MAX_TRAVEL;
        }
    }
}

} // namespace rocketsim_cuda
