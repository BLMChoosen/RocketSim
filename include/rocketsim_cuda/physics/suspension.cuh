#pragma once
#include <cuda_runtime.h>
#include <cmath>
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

// Octane Inverse Inertia Diag (in Bullet units)
__device__ __forceinline__ Vec3 get_octane_inv_inertia_bt() {
    return Vec3(
        1.0f / 54.06816f,
        1.0f / 96.098775f,
        1.0f / 132.232635f
    );
}

/**
 * @brief Resolves single bilateral constraint along wheel axle (btContactConstraint).
 */
__device__ __forceinline__ float resolve_single_bilateral(
    const Vec3& rel_pos_bt,
    const Vec3& vel_at_pt_bt,
    const Vec3& axle_dir,
    const Mat3& basis,
    const Vec3& inv_inertia_bt)
{
    float rel_vel = axle_dir.dot(vel_at_pt_bt);
    Vec3 r_cross_axle = rel_pos_bt.cross(axle_dir);
    Vec3 m_aJ = basis.transpose() * r_cross_axle;
    Vec3 m_0MinvJt = Vec3(
        inv_inertia_bt.x * m_aJ.x,
        inv_inertia_bt.y * m_aJ.y,
        inv_inertia_bt.z * m_aJ.z
    );
    float m_Adiag = (1.0f / 180.0f) + m_0MinvJt.dot(m_aJ);
    return -0.2f * rel_vel / m_Adiag;
}

struct WheelRaycastResult {
    bool in_contact;
    float hit_dist; // in UU
    Vec3 contact_pt; // in UU
    Vec3 contact_normal;
    float susp_rel_vel; // in BT
    float inv_contact_dot_susp;
    float extra_pushback;
};

/**
 * @brief Evaluates raycast queries for all 4 wheels of a car.
 * Fixed ray length: config_rest + radius - SUSP_SUBTRACTION (no overshoot).
 */
__device__ __forceinline__ void evaluate_car_wheels_raycast(
    const Vec3& car_pos,
    const Vec3& vel,
    const Vec3& omega,
    const Mat3& basis,
    float dt,
    uint8_t* __restrict__ wheels_in_contact,
    float* __restrict__ suspension_lengths,
    WheelRaycastResult* __restrict__ results)
{
    Vec3 up_dir = basis.up;
    Vec3 wheel_dir = up_dir * -1.0f;
    Vec3 vel_bt = vel * 0.02f;

    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        Vec3 hardpoint = car_pos + basis * get_octane_wheel_offset(w);
        float config_rest = get_octane_susp_rest(w);
        float radius = get_octane_wheel_rad(w);
        float real_ray_len = config_rest + SUSP_MAX_TRAVEL + radius - SUSP_SUBTRACTION;

        float hit_dist = 0.0f;
        Vec3 hit_normal = Vec3(0.0f, 0.0f, 1.0f);
        bool hit = raycast_arena_sdf(hardpoint, wheel_dir, real_ray_len, &hit_dist, &hit_normal);

        results[w].in_contact = hit;
        results[w].hit_dist = hit_dist;
        results[w].contact_pt = hardpoint + wheel_dir * hit_dist;
        results[w].contact_normal = hit_normal;

        if (hit) {
            wheels_in_contact[w] = 1;
            float cur_susp_len = fminf(fmaxf(hit_dist - radius, config_rest - SUSP_MAX_TRAVEL), config_rest + SUSP_MAX_TRAVEL);
            suspension_lengths[w] = config_rest - cur_susp_len; // Compression in UU

            Vec3 contact_pt = results[w].contact_pt;
            Vec3 rel_pos_bt = (contact_pt - car_pos) * 0.02f;
            Vec3 vel_at_pt_bt = vel_bt + omega.cross(rel_pos_bt);
            float proj_vel_bt = hit_normal.dot(vel_at_pt_bt);
            float denominator = hit_normal.dot(up_dir);

            if (denominator > 0.1f) {
                float inv = 1.0f / denominator;
                results[w].susp_rel_vel = proj_vel_bt * inv;
                results[w].inv_contact_dot_susp = inv;
            } else {
                results[w].susp_rel_vel = 0.0f;
                results[w].inv_contact_dot_susp = 10.0f;
            }

            // Extra pushback computed during raycast matching btVehicleRL::rayCast
            float pushback_thresh_bt = (config_rest + radius - SUSP_SUBTRACTION) * 0.02f;
            float wheel_trace_len_bt = hit_dist * 0.02f;
            float extra_pushback = 0.0f;
            if (wheel_trace_len_bt < pushback_thresh_bt) {
                float dist_delta = wheel_trace_len_bt - pushback_thresh_bt;
                float pos_error = 0.2f * (-dist_delta) / dt;
                float vel_error = -proj_vel_bt;
                Vec3 inv_inertia_bt = get_octane_inv_inertia_bt();
                Vec3 c0 = rel_pos_bt.cross(hit_normal);
                Vec3 c0_loc = basis.transpose() * c0;
                float denom = (1.0f / 180.0f) + (c0_loc.x * c0_loc.x * inv_inertia_bt.x
                                               + c0_loc.y * c0_loc.y * inv_inertia_bt.y
                                               + c0_loc.z * c0_loc.z * inv_inertia_bt.z);
                extra_pushback = fmaxf(0.0f, (pos_error + vel_error) / denom) * 0.25f;
            }
            results[w].extra_pushback = extra_pushback;
        } else {
            wheels_in_contact[w] = 0;
            suspension_lengths[w] = -SUSP_MAX_TRAVEL;
            results[w].susp_rel_vel = 0.0f;
            results[w].inv_contact_dot_susp = 1.0f;
            results[w].extra_pushback = 0.0f;
        }
    }
}

/**
 * @brief Applies suspension impulses and bilateral tire friction impulses to car rigid body.
 * Faithfully matches Bullet btVehicleRL updateVehicleSecond and calcFrictionImpulses.
 */
__device__ __forceinline__ void apply_suspension_and_friction(
    const Vec3& car_pos,
    const Mat3& basis,
    const WheelRaycastResult* __restrict__ wheel_results,
    float dt,
    float cached_engine_force,
    float cached_brake,
    float cached_steer_angle,
    const float* cached_lat_frictions,
    const float* cached_long_frictions,
    Vec3& vel,
    Vec3& omega)
{
    Vec3 inv_inertia_bt = get_octane_inv_inertia_bt();
    Vec3 total_lin_imp_bt(0.0f, 0.0f, 0.0f);
    Vec3 total_ang_imp_bt(0.0f, 0.0f, 0.0f);
    Vec3 vel_bt = vel * 0.02f;

    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (!wheel_results[w].in_contact) continue;

        float config_rest = get_octane_susp_rest(w);
        float radius = get_octane_wheel_rad(w);
        float hit_dist = wheel_results[w].hit_dist;
        float cur_susp_len = fminf(fmaxf(hit_dist - radius, config_rest - SUSP_MAX_TRAVEL), config_rest + SUSP_MAX_TRAVEL);

        Vec3 contact_pt_uu = wheel_results[w].contact_pt;
        Vec3 hit_normal = wheel_results[w].contact_normal;

        Vec3 rel_pos_bt = (contact_pt_uu - car_pos) * 0.02f;
        Vec3 vel_at_pt_bt = vel_bt + omega.cross(rel_pos_bt);

        // 1. Suspension Spring & Damping (btVehicleRL::updateSuspension)
        float inv_dot = wheel_results[w].inv_contact_dot_susp;
        float v_rel_bt = wheel_results[w].susp_rel_vel;

        float compression_bt = (config_rest - cur_susp_len) * 0.02f;
        float spring_force = compression_bt * SUSP_STIFFNESS * inv_dot;
        float damping_scale = (v_rel_bt < 0.0f) ? SUSP_DAMPING_COMPRESSION : SUSP_DAMPING_RELAXATION;
        float susp_force = spring_force - (damping_scale * v_rel_bt);
        susp_force *= get_octane_force_scale(w);
        if (susp_force < 0.0f) susp_force = 0.0f;

        // 2. Extra Pushback (resolveSingleCollision)
        float extra_pushback = wheel_results[w].extra_pushback;

        float base_scale_bt = (susp_force * dt) + extra_pushback;
        Vec3 susp_imp_bt = hit_normal * base_scale_bt;
        Vec3 susp_torque_bt = rel_pos_bt.cross(susp_imp_bt);

        // 3. Bilateral Tire Friction (btVehicleRL::calcFrictionImpulses)
        float steer = (w < 2) ? cached_steer_angle : 0.0f;
        Vec3 axle_dir_raw = basis.right * cosf(steer) - basis.forward * sinf(steer);
        float proj_axle = axle_dir_raw.dot(hit_normal);
        Vec3 axle_dir = (axle_dir_raw - hit_normal * proj_axle).normalized();
        Vec3 forward_dir = hit_normal.cross(axle_dir).normalized();

        float side_impulse = resolve_single_bilateral(rel_pos_bt, vel_at_pt_bt, axle_dir, basis, inv_inertia_bt);

        float rolling_friction;
        if (cached_engine_force == 0.0f) {
            if (cached_brake > 0.0f) {
                float rel_vel_fwd = vel_at_pt_bt.dot(forward_dir);
                constexpr float ROLLING_FRICTION_SCALE = 113.73963f;
                rolling_friction = fminf(fmaxf(-rel_vel_fwd * ROLLING_FRICTION_SCALE, -cached_brake), cached_brake);
            } else {
                rolling_friction = 0.0f;
            }
        } else {
            rolling_friction = -cached_engine_force / 60.0f; // frictionScale = 180 / 3 = 60
        }

        Vec3 total_friction_force = forward_dir * (rolling_friction * cached_long_frictions[w])
                                  + axle_dir * (side_impulse * cached_lat_frictions[w]);
        Vec3 wheel_fric_imp_bt = total_friction_force * (60.0f * dt);
        if (w == 2 && car_pos.x < -2320.0f) {
            printf("  [GPU WHEEL 2] susp_force=%f side_imp=%f roll_fric=%f total_fric=(%f,%f,%f) lat_fric=%f long_fric=%f\n",
                   susp_force, side_impulse, rolling_friction,
                   total_friction_force.x * 60.0f, total_friction_force.y * 60.0f, total_friction_force.z * 60.0f,
                   cached_lat_frictions[w], cached_long_frictions[w]);
        }

        // Planar offset for tire friction: eliminates roll torque from tire sliding
        Vec3 r_planar_bt = rel_pos_bt - basis.up * basis.up.dot(rel_pos_bt);
        Vec3 fric_torque_bt = r_planar_bt.cross(wheel_fric_imp_bt);

        total_lin_imp_bt = total_lin_imp_bt + susp_imp_bt + wheel_fric_imp_bt;
        total_ang_imp_bt = total_ang_imp_bt + susp_torque_bt + fric_torque_bt;
    }

    // Apply accumulated impulses directly to chassis velocity and angular velocity
    vel = vel + total_lin_imp_bt * (50.0f / 180.0f);
    Vec3 delta_omega_loc = Vec3(
        inv_inertia_bt.x * (basis.transpose() * total_ang_imp_bt).x,
        inv_inertia_bt.y * (basis.transpose() * total_ang_imp_bt).y,
        inv_inertia_bt.z * (basis.transpose() * total_ang_imp_bt).z
    );
    omega = omega + basis * delta_omega_loc;
}

} // namespace rocketsim_cuda
