#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/mat3.cuh"
#include "rocketsim_cuda/physics/arena_sdf.cuh"
#include "rocketsim_cuda/types/car_state.cuh"

namespace rocketsim_cuda {

// Octane Hitbox Geometry
__device__ __forceinline__ Vec3 get_octane_hitbox_offset() {
    return Vec3(13.8757f, 0.0f, 20.755f);
}

__device__ __forceinline__ Vec3 get_octane_hitbox_half() {
    return Vec3(60.2535f, 43.3497f, 19.32955f);
}

/**
 * @brief Resolves ball collision against arena SDF and floor.
 */
__device__ __forceinline__ void resolve_ball_arena_collision(
    Vec3& pos,
    Vec3& vel,
    Vec3& ang_vel,
    float radius = BALL_RADIUS,
    float restitution = BALL_RESTITUTION,
    float friction = BALL_FRICTION)
{
    float dist = 0.0f;
    Vec3 normal(0.0f, 0.0f, 1.0f);
    arena_sdf_and_normal(pos, dist, normal);

    if (dist < radius) {
        float penetration = radius - dist;
        pos = pos + normal * penetration;

        float vn = normal.dot(vel);
        if (vn < 0.0f) {
            float impulse_n = -(1.0f + restitution) * vn;
            Vec3 normal_impulse = normal * impulse_n;

            Vec3 v_tan = vel - normal * vn;
            Vec3 tangent_impulse = v_tan * (-friction);

            vel = vel + normal_impulse + tangent_impulse;

            // Rolling friction rotational coupling
            Vec3 ang_impulse = normal.cross(v_tan) * (1.0f / radius);
            ang_vel = ang_vel + ang_impulse * friction;
        }
    }
}

/**
 * @brief Resolves chassis-arena and chassis-ground penetration and records world contact normals.
 */
__device__ __forceinline__ void resolve_chassis_arena_collision(
    uint32_t car_idx,
    CarStateSoA& car_state,
    Vec3& pos,
    Vec3& vel,
    Vec3& omega,
    const Mat3& basis,
    float dt)
{
    Vec3 hitbox_offset = get_octane_hitbox_offset();
    Vec3 hitbox_half = get_octane_hitbox_half();

    bool has_contact = false;
    Vec3 sum_normal(0.0f, 0.0f, 0.0f);

    // Check 8 corner vertices of oriented hitbox
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float sx = (i & 1) ? 1.0f : -1.0f;
        float sy = (i & 2) ? 1.0f : -1.0f;
        float sz = (i & 4) ? 1.0f : -1.0f;

        Vec3 local_corner = hitbox_offset + Vec3(
            sx * hitbox_half.x,
            sy * hitbox_half.y,
            sz * hitbox_half.z
        );

        Vec3 world_corner = pos + basis * local_corner;
        float dist = 0.0f;
        Vec3 normal(0.0f, 0.0f, 1.0f);
        arena_sdf_and_normal(world_corner, dist, normal);

        if (dist < 0.0f) {
            has_contact = true;
            sum_normal = sum_normal + normal;

            float depth = -dist;
            pos = pos + normal * (depth * 0.125f); // Distributed position correction

            Vec3 rel_pos = world_corner - pos;
            Vec3 pt_vel = vel + omega.cross(rel_pos);
            float vn = normal.dot(pt_vel);

            if (vn < 0.0f) {
                float impulse_mag = -(1.1f * vn) + (0.2f * depth / dt);
                Vec3 impulse = normal * (impulse_mag * CAR_MASS * 0.125f);
                vel = vel + impulse * (1.0f / CAR_MASS);
                omega = omega + rel_pos.cross(impulse) * (1.0f / (CAR_MASS * 1000.0f));
            }
        }
    }

    if (has_contact) {
        Vec3 avg_normal = (sum_normal.length_sq() > 1e-6f) ? sum_normal.normalized() : Vec3(0.0f, 0.0f, 1.0f);
        car_state.world_contact_has_contact[car_idx] = 1;
        car_state.world_contact_normal_x[car_idx]    = avg_normal.x;
        car_state.world_contact_normal_y[car_idx]    = avg_normal.y;
        car_state.world_contact_normal_z[car_idx]    = avg_normal.z;
    } else {
        car_state.world_contact_has_contact[car_idx] = 0;
        car_state.world_contact_normal_x[car_idx]    = 0.0f;
        car_state.world_contact_normal_y[car_idx]    = 0.0f;
        car_state.world_contact_normal_z[car_idx]    = 0.0f;
    }
}

} // namespace rocketsim_cuda
