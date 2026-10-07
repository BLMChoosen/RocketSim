#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/quat.cuh"
#include "rocketsim_cuda/math/mat3.cuh"
#include "rocketsim_cuda/physics/arena_sdf.cuh"
#include "rocketsim_cuda/types/car_state.cuh"
#include "rocketsim_cuda/types/ball_state.cuh"

namespace rocketsim_cuda {

// ============================================================================
// Suspension & Tire Tuning Constants (RLConst::BTVehicle)
// CPU Reference: src/Sim/RLConst.h:35-45, 110-145
// ============================================================================
constexpr float SUSP_STIFFNESS                  = 500.0f; // in Bullet units
constexpr float SUSP_DAMPING_COMPRESSION        = 25.0f;
constexpr float SUSP_DAMPING_RELAXATION         = 40.0f;
constexpr float SUSP_MAX_TRAVEL                 = 12.0f;  // UU
constexpr float SUSP_SUBTRACTION                = 2.5f;   // 0.05f BT = 2.5f UU
constexpr float SUSP_FORCE_SCALE_FRONT          = 35.75f; // 36 - 1/4
constexpr float SUSP_FORCE_SCALE_BACK           = 54.265f;// 54 + 1/4 + 1.5/100

// ============================================================================
// Ball & Car Physical Mass and Inertia Constants
// CPU Reference: src/Sim/RLConst.h:80-95, src/Sim/Ball/Ball.cpp:80-95
// ============================================================================
constexpr float BALL_COLLISION_RADIUS_DEFAULT   = 91.25f; // RLConst::BALL_COLLISION_RADIUS
constexpr float BALL_MASS_BT_DEFAULT            = 30.0f;  // RLConst::BALL_MASS_BT
constexpr float CAR_MASS_BT_DEFAULT             = 180.0f; // RLConst::CAR_MASS_BT
// Solid sphere inertia: I = 2/5 * M * R^2. For Ball: 0.4 * 30.0 * (1.825^2) = 39.9675 BT
constexpr float BALL_INERTIA_BT_DEFAULT         = 39.9675f;
constexpr float INV_BALL_INERTIA_BT_DEFAULT     = 1.0f / BALL_INERTIA_BT_DEFAULT;
constexpr float INV_BALL_MASS_BT_DEFAULT        = 1.0f / BALL_MASS_BT_DEFAULT;
constexpr float INV_CAR_MASS_BT_DEFAULT         = 1.0f / CAR_MASS_BT_DEFAULT;

// ============================================================================
// Octane Wheel & Hitbox Geometry (CarConfig.cpp)
// CPU Reference: src/Sim/Car/CarConfig/CarConfig.cpp:20-60
// ============================================================================
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

// Octane Default Hitbox geometry (RLConst.h, CarConfig.cpp:20,31)
__device__ __forceinline__ Vec3 get_octane_hitbox_offset_susp() {
    return Vec3(13.8757f, 0.0f, 20.755f);
}

__device__ __forceinline__ Vec3 get_octane_hitbox_half_susp() {
    return Vec3(60.2535f, 43.3497f, 19.32955f);
}

// ============================================================================
// Types & Result Enums
// CPU Reference: src/Sim/btVehicleRL/btVehicleRL.cpp:140-150
// ============================================================================

/**
 * @brief Classification of objects hit by wheel raycasts.
 * Mirrors Bullet collision object types in btDefaultVehicleRaycaster (btCollisionObject).
 */
enum HitObjectType : uint8_t {
    HIT_OBJECT_NONE  = 0,
    HIT_OBJECT_WORLD = 1, // Static arena geometry (SDF ground / walls / ramps)
    HIT_OBJECT_BALL  = 2, // Ball sphere
    HIT_OBJECT_CAR   = 3  // Other car OBB
};

/**
 * @brief Stores reaction impulse to be applied to a hit dynamic body (Newton's 3rd Law).
 */
struct BodyReactionImpulse {
    Vec3 lin_impulse_bt = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 ang_impulse_bt = Vec3(0.0f, 0.0f, 0.0f);

    __device__ __forceinline__ void reset() {
        lin_impulse_bt = Vec3(0.0f, 0.0f, 0.0f);
        ang_impulse_bt = Vec3(0.0f, 0.0f, 0.0f);
    }
};

/**
 * @brief Result of a single wheel raycast query.
 */
struct WheelRaycastResult {
    bool in_contact = false;
    float hit_dist = 0.0f; // in UU
    Vec3 contact_pt = Vec3(0.0f, 0.0f, 0.0f); // in UU
    Vec3 contact_normal = Vec3(0.0f, 0.0f, 1.0f); // Outward normal from hit body towards wheel
    uint8_t hit_object_type = HIT_OBJECT_NONE; // HitObjectType
    int hit_car_index = -1; // Index of hit car, or -1 if not a car
};

// ============================================================================
// Raycast Helpers: Sphere (Ball) and OBB (Other Cars)
// CPU Reference: libsrc/bullet3-3.24/BulletCollision/CollisionDispatch/btCollisionWorld.cpp:267-340
// ============================================================================

/**
 * @brief Closed-form raycast against a sphere (Ball geometry).
 * Tests ray segment from ray_origin along ray_dir up to max_dist.
 * Matches Bullet btSubsimplexConvexCast against btSphereShape (src/Sim/Ball/Ball.cpp:79).
 */
__device__ __forceinline__ bool raycast_sphere(
    const Vec3& ray_origin,
    const Vec3& ray_dir, // normalized unit vector
    float max_dist,
    const Vec3& sphere_center,
    float sphere_radius,
    float* __restrict__ hit_dist,
    Vec3* __restrict__ hit_normal)
{
    Vec3 m = ray_origin - sphere_center;
    float b = m.dot(ray_dir);
    float c = m.dot(m) - (sphere_radius * sphere_radius);

    // Ray origin outside sphere and pointing away
    if (c > 0.0f && b > 0.0f) {
        return false;
    }

    float discr = b * b - c;
    if (discr < 0.0f) {
        return false;
    }

    float s = sqrtf(fmaxf(0.0f, discr));
    float t = -b - s;

    // If ray starts inside the sphere (c <= 0), contact is immediate at origin
    if (t < 0.0f) {
        t = 0.0f;
    }

    if (t > max_dist) {
        return false;
    }

    *hit_dist = t;
    Vec3 contact_pt = ray_origin + ray_dir * t;
    Vec3 norm = contact_pt - sphere_center;
    float norm_len_sq = norm.length_sq();
    if (norm_len_sq > 1e-8f) {
        *hit_normal = norm * (1.0f / sqrtf(norm_len_sq));
    } else {
        *hit_normal = ray_dir * -1.0f;
    }
    return true;
}

/**
 * @brief Closed-form raycast against an Oriented Bounding Box (Car hitbox geometry).
 * Transforms ray to box local coordinates and applies Kay-Kajiya slab method.
 * Matches Bullet btCollisionWorld::rayTestSingle against compound btBoxShape (src/Sim/Car/Car.cpp:210-216).
 */
__device__ __forceinline__ bool raycast_obb(
    const Vec3& ray_origin,
    const Vec3& ray_dir, // normalized unit vector
    float max_dist,
    const Vec3& box_pos,
    const Mat3& box_basis,
    const Vec3& box_offset,
    const Vec3& box_half_extents,
    float* __restrict__ hit_dist,
    Vec3* __restrict__ hit_normal)
{
    Vec3 box_center = box_pos + box_basis * box_offset;
    Vec3 r0 = box_basis.transpose() * (ray_origin - box_center);
    Vec3 d = box_basis.transpose() * ray_dir;

    float t_min = 0.0f;
    float t_max = max_dist;
    Vec3 entry_normal_loc(0.0f, 0.0f, 0.0f);

    // X slab
    if (fabsf(d.x) > 1e-7f) {
        float inv_dx = 1.0f / d.x;
        float t1 = (-box_half_extents.x - r0.x) * inv_dx;
        float t2 = ( box_half_extents.x - r0.x) * inv_dx;
        float s = -1.0f;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; s = 1.0f; }
        if (t1 > t_min) {
            t_min = t1;
            entry_normal_loc = Vec3(s, 0.0f, 0.0f);
        }
        if (t2 < t_max) t_max = t2;
        if (t_min > t_max) return false;
    } else {
        if (r0.x < -box_half_extents.x || r0.x > box_half_extents.x) return false;
    }

    // Y slab
    if (fabsf(d.y) > 1e-7f) {
        float inv_dy = 1.0f / d.y;
        float t1 = (-box_half_extents.y - r0.y) * inv_dy;
        float t2 = ( box_half_extents.y - r0.y) * inv_dy;
        float s = -1.0f;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; s = 1.0f; }
        if (t1 > t_min) {
            t_min = t1;
            entry_normal_loc = Vec3(0.0f, s, 0.0f);
        }
        if (t2 < t_max) t_max = t2;
        if (t_min > t_max) return false;
    } else {
        if (r0.y < -box_half_extents.y || r0.y > box_half_extents.y) return false;
    }

    // Z slab
    if (fabsf(d.z) > 1e-7f) {
        float inv_dz = 1.0f / d.z;
        float t1 = (-box_half_extents.z - r0.z) * inv_dz;
        float t2 = ( box_half_extents.z - r0.z) * inv_dz;
        float s = -1.0f;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; s = 1.0f; }
        if (t1 > t_min) {
            t_min = t1;
            entry_normal_loc = Vec3(0.0f, 0.0f, s);
        }
        if (t2 < t_max) t_max = t2;
        if (t_min > t_max) return false;
    } else {
        if (r0.z < -box_half_extents.z || r0.z > box_half_extents.z) return false;
    }

    if (t_min > max_dist) return false;

    // Ray origin is inside box; find closest face normal
    if (entry_normal_loc.length_sq() < 0.5f) {
        float dx_pos = box_half_extents.x - r0.x;
        float dx_neg = r0.x - (-box_half_extents.x);
        float dy_pos = box_half_extents.y - r0.y;
        float dy_neg = r0.y - (-box_half_extents.y);
        float dz_pos = box_half_extents.z - r0.z;
        float dz_neg = r0.z - (-box_half_extents.z);
        float min_d = dx_pos;
        entry_normal_loc = Vec3(1.0f, 0.0f, 0.0f);
        if (dx_neg < min_d) { min_d = dx_neg; entry_normal_loc = Vec3(-1.0f, 0.0f, 0.0f); }
        if (dy_pos < min_d) { min_d = dy_pos; entry_normal_loc = Vec3(0.0f, 1.0f, 0.0f); }
        if (dy_neg < min_d) { min_d = dy_neg; entry_normal_loc = Vec3(0.0f, -1.0f, 0.0f); }
        if (dz_pos < min_d) { min_d = dz_pos; entry_normal_loc = Vec3(0.0f, 0.0f, 1.0f); }
        if (dz_neg < min_d) { min_d = dz_neg; entry_normal_loc = Vec3(0.0f, 0.0f, -1.0f); }
    }

    *hit_dist = t_min;
    Vec3 world_norm = box_basis * entry_normal_loc;
    float norm_len_sq = world_norm.length_sq();
    if (norm_len_sq > 1e-8f) {
        *hit_normal = world_norm * (1.0f / sqrtf(norm_len_sq));
    } else {
        *hit_normal = ray_dir * -1.0f;
    }
    return true;
}

// ============================================================================
// Multi-Body Wheel Raycast Query
// CPU Reference: src/Sim/btVehicleRL/btVehicleRL.cpp:120-180
// ============================================================================

/**
 * @brief Evaluates raycast queries for all 4 wheels of a car against Arena SDF, Ball, and other cars.
 * Selects the closest hit across all candidates, faithfully replicating Bullet btDefaultVehicleRaycaster.
 * CPU Reference: src/Sim/btVehicleRL/btVehicleRL.cpp:120-180 and src/Sim/Car/Car.cpp:103-120.
 */
__device__ __forceinline__ void evaluate_car_wheels_raycast_multibody(
    const Vec3& car_pos,
    const Mat3& basis,
    bool check_ball,
    const Vec3& ball_pos,
    float ball_radius,
    uint32_t num_other_cars,
    uint32_t current_car_idx,
    const Vec3* __restrict__ other_cars_pos,
    const Mat3* __restrict__ other_cars_basis,
    const uint8_t* __restrict__ other_cars_is_demoed,
    uint8_t* __restrict__ wheels_in_contact,
    float* __restrict__ suspension_lengths,
    WheelRaycastResult* __restrict__ results)
{
    Vec3 up_dir = basis.up;
    Vec3 wheel_dir = up_dir * -1.0f;

    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        Vec3 hardpoint = car_pos + basis * get_octane_wheel_offset(w);
        float config_rest = get_octane_susp_rest(w);
        float radius = get_octane_wheel_rad(w);
        float real_ray_len = config_rest + SUSP_MAX_TRAVEL + radius - SUSP_SUBTRACTION;

        float closest_dist = real_ray_len + 1.0f;
        Vec3 closest_normal = up_dir;
        uint8_t closest_type = HIT_OBJECT_NONE;
        int closest_car_idx = -1;

        // 1. Raycast against analytical Arena SDF (Static geometry)
        float sdf_hit_dist = 0.0f;
        Vec3 sdf_hit_normal = Vec3(0.0f, 0.0f, 1.0f);
        if (raycast_arena_sdf(hardpoint, wheel_dir, real_ray_len, &sdf_hit_dist, &sdf_hit_normal)) {
            if (sdf_hit_dist < closest_dist) {
                closest_dist = sdf_hit_dist;
                closest_normal = sdf_hit_normal;
                closest_type = HIT_OBJECT_WORLD;
                closest_car_idx = -1;
            }
        }

        // 2. Raycast against Ball (Sphere geometry)
        if (check_ball) {
            float ball_hit_dist = 0.0f;
            Vec3 ball_hit_normal = Vec3(0.0f, 0.0f, 1.0f);
            if (raycast_sphere(hardpoint, wheel_dir, real_ray_len, ball_pos, ball_radius, &ball_hit_dist, &ball_hit_normal)) {
                if (ball_hit_dist < closest_dist) {
                    closest_dist = ball_hit_dist;
                    closest_normal = ball_hit_normal;
                    closest_type = HIT_OBJECT_BALL;
                    closest_car_idx = -1;
                }
            }
        }

        // 3. Raycast against other cars (OBB geometry)
        if (other_cars_pos && other_cars_basis && num_other_cars > 0) {
            Vec3 hitbox_offset = get_octane_hitbox_offset_susp();
            Vec3 hitbox_half = get_octane_hitbox_half_susp();

            for (uint32_t c = 0; c < num_other_cars; ++c) {
                if (c == current_car_idx) continue;
                if (other_cars_is_demoed && other_cars_is_demoed[c]) continue;

                float car_hit_dist = 0.0f;
                Vec3 car_hit_normal = Vec3(0.0f, 0.0f, 1.0f);
                if (raycast_obb(hardpoint, wheel_dir, real_ray_len,
                                other_cars_pos[c], other_cars_basis[c],
                                hitbox_offset, hitbox_half,
                                &car_hit_dist, &car_hit_normal)) {
                    if (car_hit_dist < closest_dist) {
                        closest_dist = car_hit_dist;
                        closest_normal = car_hit_normal;
                        closest_type = HIT_OBJECT_CAR;
                        closest_car_idx = static_cast<int>(c);
                    }
                }
            }
        }

        // Record closest hit result
        if (closest_type != HIT_OBJECT_NONE && closest_dist <= real_ray_len) {
            results[w].in_contact = true;
            results[w].hit_dist = closest_dist;
            results[w].contact_pt = hardpoint + wheel_dir * closest_dist;
            results[w].contact_normal = closest_normal;
            results[w].hit_object_type = closest_type;
            results[w].hit_car_index = closest_car_idx;

            wheels_in_contact[w] = 1;
            float cur_susp_len = fminf(fmaxf(closest_dist - radius, config_rest - SUSP_MAX_TRAVEL), config_rest + SUSP_MAX_TRAVEL);
            suspension_lengths[w] = config_rest - cur_susp_len; // Compression in UU
        } else {
            results[w].in_contact = false;
            results[w].hit_dist = real_ray_len;
            results[w].contact_pt = hardpoint + wheel_dir * real_ray_len;
            results[w].contact_normal = up_dir;
            results[w].hit_object_type = HIT_OBJECT_NONE;
            results[w].hit_car_index = -1;

            wheels_in_contact[w] = 0;
            suspension_lengths[w] = -SUSP_MAX_TRAVEL;
        }
    }
}

/**
 * @brief Legacy single-car evaluate_car_wheels_raycast query against Arena SDF.
 * Preserves 100% backward compatibility for existing callers.
 */
__device__ __forceinline__ void evaluate_car_wheels_raycast(
    const Vec3& car_pos,
    const Mat3& basis,
    uint8_t* __restrict__ wheels_in_contact,
    float* __restrict__ suspension_lengths,
    WheelRaycastResult* __restrict__ results)
{
    evaluate_car_wheels_raycast_multibody(
        car_pos, basis,
        false, Vec3(0,0,0), 0.0f,
        0, 0, nullptr, nullptr, nullptr,
        wheels_in_contact, suspension_lengths, results
    );
}

// ============================================================================
// Support Condition & Jump/Flip Restoration
// CPU Reference: src/Sim/Car/Car.cpp:117-128, 550-559, 689-695
// ============================================================================

/**
 * @brief Evaluates whether car is on ground based on wheel contacts.
 * Condition: >= 3 wheels in contact define isOnGround = true.
 * CPU Reference: src/Sim/Car/Car.cpp:117-119.
 */
__device__ __forceinline__ bool is_on_ground_from_wheels(int num_wheels_contact) {
    return num_wheels_contact >= 3;
}

/**
 * @brief Updates car ground state and resets jump/flip timers upon ground contact (flip reset).
 * Replicates Car::_UpdateJump (src/Sim/Car/Car.cpp:550-559) and Car::_UpdateDoubleJumpOrFlip (src/Sim/Car/Car.cpp:689-695).
 */
__device__ __forceinline__ void update_car_ground_support(
    int num_wheels_contact,
    uint8_t& is_on_ground,
    uint8_t is_jumping,
    float jump_time,
    uint8_t& has_jumped,
    uint8_t& has_double_jumped,
    uint8_t& has_flipped,
    uint8_t& is_flipping,
    float& air_time,
    float& air_time_since_jump,
    float& flip_time)
{
    is_on_ground = (num_wheels_contact >= 3) ? 1 : 0;
    if (is_on_ground) {
        // Reset jump if not currently executing an active jump
        // (RLConst::JUMP_MIN_TIME = 0.025f, RLConst::JUMP_RESET_TIME_PAD = 1/120f)
        if (!is_jumping) {
            constexpr float JUMP_RESET_THRESHOLD = 0.025f + (1.0f / 120.0f);
            if (!has_jumped || jump_time >= JUMP_RESET_THRESHOLD) {
                has_jumped = 0;
            }
        }
        has_double_jumped = 0;
        has_flipped = 0;
        is_flipping = 0;
        air_time = 0.0f;
        air_time_since_jump = 0.0f;
        flip_time = 0.0f;
    }
}

/**
 * @brief SoA helper to apply wheel contact support and flip reset directly to CarStateSoA.
 */
__device__ __forceinline__ void update_car_ground_support_soa(
    uint32_t car_idx,
    CarStateSoA& car_state,
    int num_wheels_contact)
{
    uint8_t is_on_ground = 0;
    uint8_t is_jumping = car_state.is_jumping ? car_state.is_jumping[car_idx] : 0;
    float jump_time = car_state.jump_time ? car_state.jump_time[car_idx] : 0.0f;
    uint8_t has_jumped = car_state.has_jumped ? car_state.has_jumped[car_idx] : 0;
    uint8_t has_double_jumped = car_state.has_double_jumped ? car_state.has_double_jumped[car_idx] : 0;
    uint8_t has_flipped = car_state.has_flipped ? car_state.has_flipped[car_idx] : 0;
    uint8_t is_flipping = car_state.is_flipping ? car_state.is_flipping[car_idx] : 0;
    float air_time = car_state.air_time ? car_state.air_time[car_idx] : 0.0f;
    float air_time_since_jump = car_state.air_time_since_jump ? car_state.air_time_since_jump[car_idx] : 0.0f;
    float flip_time = car_state.flip_time ? car_state.flip_time[car_idx] : 0.0f;

    update_car_ground_support(
        num_wheels_contact,
        is_on_ground,
        is_jumping,
        jump_time,
        has_jumped,
        has_double_jumped,
        has_flipped,
        is_flipping,
        air_time,
        air_time_since_jump,
        flip_time
    );

    if (car_state.is_on_ground) car_state.is_on_ground[car_idx] = is_on_ground;
    if (car_state.has_jumped) car_state.has_jumped[car_idx] = has_jumped;
    if (car_state.has_double_jumped) car_state.has_double_jumped[car_idx] = has_double_jumped;
    if (car_state.has_flipped) car_state.has_flipped[car_idx] = has_flipped;
    if (car_state.is_flipping) car_state.is_flipping[car_idx] = is_flipping;
    if (car_state.air_time) car_state.air_time[car_idx] = air_time;
    if (car_state.air_time_since_jump) car_state.air_time_since_jump[car_idx] = air_time_since_jump;
    if (car_state.flip_time) car_state.flip_time[car_idx] = flip_time;
}

// ============================================================================
// Bilateral Constraint & Newton's 3rd Law Reaction Applicators
// CPU Reference: libsrc/bullet3-3.24/BulletDynamics/ConstraintSolver/btContactConstraint.cpp:108-150
// ============================================================================

/**
 * @brief Resolves single bilateral constraint along wheel axle (btContactConstraint).
 * Supports both static world contacts and dynamic bodies (Ball, other Car).
 * CPU Reference: libsrc/bullet3-3.24/BulletDynamics/ConstraintSolver/btContactConstraint.cpp:108-150.
 */
__device__ __forceinline__ float resolve_single_bilateral(
    const Vec3& rel_pos_bt,
    const Vec3& vel_at_pt_bt,
    const Vec3& axle_dir,
    const Mat3& basis,
    const Vec3& inv_inertia_bt,
    float other_body_inv_mass_bt = 0.0f,
    float other_body_ang_term = 0.0f,
    const Vec3& other_body_vel_at_pt_bt = Vec3(0.0f, 0.0f, 0.0f))
{
    Vec3 rel_vel_vec = vel_at_pt_bt - other_body_vel_at_pt_bt;
    float rel_vel = axle_dir.dot(rel_vel_vec);
    Vec3 r_cross_axle = rel_pos_bt.cross(axle_dir);
    Vec3 m_aJ = basis.transpose() * r_cross_axle;
    Vec3 m_0MinvJt = Vec3(
        inv_inertia_bt.x * m_aJ.x,
        inv_inertia_bt.y * m_aJ.y,
        inv_inertia_bt.z * m_aJ.z
    );
    float m_Adiag = INV_CAR_MASS_BT_DEFAULT + m_0MinvJt.dot(m_aJ);
    float m_total_diag = m_Adiag + other_body_inv_mass_bt + other_body_ang_term;
    return -0.2f * rel_vel / m_total_diag;
}

/**
 * @brief Applies reaction impulse to Ball rigid body from wheel contact (Newton's 3rd Law).
 * Ball mass = 30.0f BT, Moment of inertia = 2/5 * M * R^2 = 39.9675f BT.
 * CPU Reference: src/Sim/Ball/Ball.cpp:80-95, src/Sim/btVehicleRL/btVehicleRL.cpp:294-302.
 */
__device__ __forceinline__ void apply_wheel_reaction_to_ball(
    const BodyReactionImpulse& reaction,
    Vec3& ball_vel_bt,
    Vec3& ball_omega)
{
    ball_vel_bt = ball_vel_bt + reaction.lin_impulse_bt * INV_BALL_MASS_BT_DEFAULT;
    ball_omega = ball_omega + reaction.ang_impulse_bt * INV_BALL_INERTIA_BT_DEFAULT;
}

/**
 * @brief Applies reaction impulse to target Car rigid body from wheel contact (Newton's 3rd Law).
 * Car mass = 180.0f BT, Inv inertia = get_octane_inv_inertia_bt().
 * CPU Reference: src/Sim/Car/Car.cpp:218-228, src/Sim/btVehicleRL/btVehicleRL.cpp:294-302.
 */
__device__ __forceinline__ void apply_wheel_reaction_to_car(
    const BodyReactionImpulse& reaction,
    const Mat3& target_basis,
    Vec3& target_vel_bt,
    Vec3& target_omega)
{
    Vec3 inv_inertia_bt = get_octane_inv_inertia_bt();
    target_vel_bt = target_vel_bt + reaction.lin_impulse_bt * INV_CAR_MASS_BT_DEFAULT;
    Vec3 delta_omega_loc = Vec3(
        inv_inertia_bt.x * (target_basis.transpose() * reaction.ang_impulse_bt).x,
        inv_inertia_bt.y * (target_basis.transpose() * reaction.ang_impulse_bt).y,
        inv_inertia_bt.z * (target_basis.transpose() * reaction.ang_impulse_bt).z
    );
    target_omega = target_omega + target_basis * delta_omega_loc;
}

// ============================================================================
// Multi-Body Suspension & Tire Friction Solver
// CPU Reference: src/Sim/btVehicleRL/btVehicleRL.cpp:270-380
// ============================================================================

/**
 * @brief Full multi-body suspension & bilateral tire friction solver.
 * Handles contact against static world, ball, and other cars with Newton's 3rd law reaction accumulation.
 * CPU Reference: src/Sim/btVehicleRL/btVehicleRL.cpp:270-380.
 */
__device__ __forceinline__ void apply_suspension_and_friction_multibody(
    const Vec3& car_pos,
    const Mat3& basis,
    const WheelRaycastResult* __restrict__ wheel_results,
    float dt,
    float cached_engine_force,
    float cached_brake,
    float cached_steer_angle,
    const float* cached_lat_frictions,
    const float* cached_long_frictions,
    Vec3& vel_bt,
    Vec3& omega,
    // Multi-body target kinematics
    bool has_ball = false,
    const Vec3& ball_pos_uu = Vec3(0,0,0),
    const Vec3& ball_vel_bt = Vec3(0,0,0),
    const Vec3& ball_omega = Vec3(0,0,0),
    uint32_t num_other_cars = 0,
    const Vec3* other_cars_pos_uu = nullptr,
    const Mat3* other_cars_basis = nullptr,
    const Vec3* other_cars_vel_bt = nullptr,
    const Vec3* other_cars_omega = nullptr,
    // Output reaction accumulators (Newton's 3rd law)
    BodyReactionImpulse* ball_reaction = nullptr,
    BodyReactionImpulse* other_cars_reactions = nullptr)
{
    Vec3 inv_inertia_bt = get_octane_inv_inertia_bt();
    Vec3 total_lin_imp_bt(0.0f, 0.0f, 0.0f);
    Vec3 total_ang_imp_bt(0.0f, 0.0f, 0.0f);

    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (!wheel_results[w].in_contact) continue;

        float config_rest = get_octane_susp_rest(w);
        float radius = get_octane_wheel_rad(w);
        float hit_dist = wheel_results[w].hit_dist;
        float cur_susp_len = fminf(fmaxf(hit_dist - radius, config_rest - SUSP_MAX_TRAVEL), config_rest + SUSP_MAX_TRAVEL);

        Vec3 contact_pt_uu = wheel_results[w].contact_pt;
        Vec3 hit_normal = wheel_results[w].contact_normal;
        uint8_t hit_type = wheel_results[w].hit_object_type;
        int hit_car_idx = wheel_results[w].hit_car_index;

        Vec3 rel_pos_bt = (contact_pt_uu - car_pos) * 0.02f;
        Vec3 vel_at_pt_bt = vel_bt + omega.cross(rel_pos_bt);

        // 1. Suspension Spring & Damping (btVehicleRL::updateSuspension)
        float denominator = hit_normal.dot(basis.up);
        float inv_dot = (denominator > 0.1f) ? (1.0f / denominator) : 10.0f;
        float proj_vel_bt = hit_normal.dot(vel_at_pt_bt);
        float v_rel_bt = (denominator > 0.1f) ? (proj_vel_bt * inv_dot) : 0.0f;

        float compression_bt = (config_rest - cur_susp_len) * 0.02f;
        float spring_force = compression_bt * SUSP_STIFFNESS * inv_dot;
        float damping_scale = (v_rel_bt < 0.0f) ? SUSP_DAMPING_COMPRESSION : SUSP_DAMPING_RELAXATION;
        float susp_force = spring_force - (damping_scale * v_rel_bt);
        susp_force *= get_octane_force_scale(w);
        if (susp_force < 0.0f) susp_force = 0.0f;

        // 2. Extra Pushback (resolveSingleCollision)
        // CPU Reference btVehicleRL.cpp:181: ONLY computed for static objects!
        float extra_pushback = 0.0f;
        if (hit_type == HIT_OBJECT_WORLD) {
            float pushback_thresh_bt = (config_rest + radius - SUSP_SUBTRACTION) * 0.02f;
            float wheel_trace_len_bt = hit_dist * 0.02f;
            if (wheel_trace_len_bt < pushback_thresh_bt) {
                float dist_delta = wheel_trace_len_bt - pushback_thresh_bt;
                float pos_error = 0.2f * (-dist_delta) / dt;
                float vel_error = -proj_vel_bt;
                Vec3 c0 = rel_pos_bt.cross(hit_normal);
                Vec3 c0_loc = basis.transpose() * c0;
                float denom = INV_CAR_MASS_BT_DEFAULT + (c0_loc.x * c0_loc.x * inv_inertia_bt.x
                                                       + c0_loc.y * c0_loc.y * inv_inertia_bt.y
                                                       + c0_loc.z * c0_loc.z * inv_inertia_bt.z);
                extra_pushback = fmaxf(0.0f, (pos_error + vel_error) / denom) * 0.25f;
            }
        }

        Vec3 susp_imp_bt(0.0f, 0.0f, 0.0f);
        Vec3 susp_torque_bt(0.0f, 0.0f, 0.0f);
        if (susp_force > 0.0f || extra_pushback > 0.0f) {
            float base_scale_bt = (susp_force * dt) + extra_pushback;
            susp_imp_bt = hit_normal * base_scale_bt;
            susp_torque_bt = rel_pos_bt.cross(susp_imp_bt);
        }

        // 3. Bilateral Tire Friction (btVehicleRL::calcFrictionImpulses)
        float steer = (w < 2) ? cached_steer_angle : 0.0f;
        Vec3 axle_dir_raw = basis.right * cosf(steer) - basis.forward * sinf(steer);
        float proj_axle = axle_dir_raw.dot(hit_normal);
        Vec3 axle_dir = (axle_dir_raw - hit_normal * proj_axle).normalized();
        Vec3 forward_dir = hit_normal.cross(axle_dir).normalized();

        // Target body kinematics and Jacobian terms for two-body bilateral constraint
        float other_inv_mass_bt = 0.0f;
        float other_ang_term = 0.0f;
        Vec3 target_vel_at_pt_bt(0.0f, 0.0f, 0.0f);
        Vec3 target_rel_pos_bt(0.0f, 0.0f, 0.0f);

        if (hit_type == HIT_OBJECT_BALL && has_ball) {
            other_inv_mass_bt = INV_BALL_MASS_BT_DEFAULT;
            target_rel_pos_bt = (contact_pt_uu - ball_pos_uu) * 0.02f;
            target_vel_at_pt_bt = ball_vel_bt + ball_omega.cross(target_rel_pos_bt);
            Vec3 r_cross_axle = target_rel_pos_bt.cross(axle_dir);
            other_ang_term = INV_BALL_INERTIA_BT_DEFAULT * r_cross_axle.length_sq();
        } else if (hit_type == HIT_OBJECT_CAR && hit_car_idx >= 0 && other_cars_pos_uu && other_cars_basis && other_cars_vel_bt && other_cars_omega) {
            other_inv_mass_bt = INV_CAR_MASS_BT_DEFAULT;
            target_rel_pos_bt = (contact_pt_uu - other_cars_pos_uu[hit_car_idx]) * 0.02f;
            target_vel_at_pt_bt = other_cars_vel_bt[hit_car_idx] + other_cars_omega[hit_car_idx].cross(target_rel_pos_bt);
            Vec3 r_cross_axle = target_rel_pos_bt.cross(axle_dir);
            Vec3 m_bJ = other_cars_basis[hit_car_idx].transpose() * r_cross_axle;
            other_ang_term = (inv_inertia_bt.x * m_bJ.x * m_bJ.x
                            + inv_inertia_bt.y * m_bJ.y * m_bJ.y
                            + inv_inertia_bt.z * m_bJ.z * m_bJ.z);
        }

        float side_impulse = resolve_single_bilateral(
            rel_pos_bt, vel_at_pt_bt, axle_dir, basis, inv_inertia_bt,
            other_inv_mass_bt, other_ang_term, target_vel_at_pt_bt
        );

        float rolling_friction;
        if (cached_engine_force == 0.0f) {
            if (cached_brake > 0.0f) {
                Vec3 rel_vel_contact = vel_at_pt_bt - target_vel_at_pt_bt;
                float rel_vel_fwd = rel_vel_contact.dot(forward_dir);
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

        // Planar offset for tire friction: eliminates roll torque from tire sliding
        Vec3 r_planar_bt = rel_pos_bt - basis.up * basis.up.dot(rel_pos_bt);
        Vec3 fric_torque_bt = r_planar_bt.cross(wheel_fric_imp_bt);

        Vec3 wheel_total_imp_bt = susp_imp_bt + wheel_fric_imp_bt;
        total_lin_imp_bt = total_lin_imp_bt + wheel_total_imp_bt;
        total_ang_imp_bt = total_ang_imp_bt + susp_torque_bt + fric_torque_bt;

        // 4. Reaction Impulses on Hit Bodies (Newton's 3rd Law: -F on target body at contact point)
        if (hit_type == HIT_OBJECT_BALL && ball_reaction) {
            Vec3 react_lin_bt = wheel_total_imp_bt * -1.0f;
            Vec3 react_ang_bt = target_rel_pos_bt.cross(react_lin_bt);
            ball_reaction->lin_impulse_bt = ball_reaction->lin_impulse_bt + react_lin_bt;
            ball_reaction->ang_impulse_bt = ball_reaction->ang_impulse_bt + react_ang_bt;
        } else if (hit_type == HIT_OBJECT_CAR && hit_car_idx >= 0 && other_cars_reactions) {
            Vec3 react_lin_bt = wheel_total_imp_bt * -1.0f;
            Vec3 react_ang_bt = target_rel_pos_bt.cross(react_lin_bt);
            other_cars_reactions[hit_car_idx].lin_impulse_bt = other_cars_reactions[hit_car_idx].lin_impulse_bt + react_lin_bt;
            other_cars_reactions[hit_car_idx].ang_impulse_bt = other_cars_reactions[hit_car_idx].ang_impulse_bt + react_ang_bt;
        }
    }

    // Apply accumulated impulses directly to chassis velocity and angular velocity
    vel_bt = vel_bt + total_lin_imp_bt * INV_CAR_MASS_BT_DEFAULT;
    Vec3 delta_omega_loc = Vec3(
        inv_inertia_bt.x * (basis.transpose() * total_ang_imp_bt).x,
        inv_inertia_bt.y * (basis.transpose() * total_ang_imp_bt).y,
        inv_inertia_bt.z * (basis.transpose() * total_ang_imp_bt).z
    );
    omega = omega + basis * delta_omega_loc;
}

/**
 * @brief Legacy apply_suspension_and_friction for car chassis.
 * Preserves exact signature and behavior for existing callers.
 * Gates extra pushback to static world hits (extraPushback = 0 for ball / car hits per btVehicleRL.cpp:181).
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
    Vec3& vel_bt,
    Vec3& omega)
{
    apply_suspension_and_friction_multibody(
        car_pos, basis, wheel_results, dt,
        cached_engine_force, cached_brake, cached_steer_angle,
        cached_lat_frictions, cached_long_frictions,
        vel_bt, omega,
        false, Vec3(0,0,0), Vec3(0,0,0), Vec3(0,0,0),
        0, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr
    );
}

} // namespace rocketsim_cuda
