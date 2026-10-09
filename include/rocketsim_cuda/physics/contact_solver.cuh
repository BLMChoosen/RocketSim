#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/mat3.cuh"
#include "rocketsim_cuda/physics/arena_sdf.cuh"
#include "rocketsim_cuda/types/car_state.cuh"
#include "rocketsim_cuda/types/ball_state.cuh"
#include "rocketsim_cuda/types/arena_state.cuh"
#include "rocketsim_cuda/types/car_config.cuh"
#include "rocketsim_cuda/types/arena_config.cuh"
#include "rocketsim_cuda/physics/suspension.cuh"

namespace rocketsim_cuda {

// Octane Hitbox Geometry
__device__ __forceinline__ Vec3 get_octane_hitbox_offset() {
    return Vec3(13.8757f, 0.0f, 20.755f);
}

__device__ __forceinline__ Vec3 get_octane_hitbox_half() {
    return Vec3(60.2535f, 43.3497f, 19.32955f);
}

/**
 * @brief Evaluates RocketSim BALL_CAR_EXTRA_IMPULSE_FACTOR_CURVE (RLConst.h:496-503).
 * Piecewise curve points: (0, 0.65), (500, 0.65), (2300, 0.55), (4600, 0.30).
 */
__device__ __forceinline__ float evaluate_ball_car_extra_impulse_factor(float rel_speed) {
    if (rel_speed <= 500.0f) {
        return 0.65f;
    } else if (rel_speed <= 2300.0f) {
        float t = (rel_speed - 500.0f) / 1800.0f;
        return 0.65f - 0.10f * t;
    } else if (rel_speed <= 4600.0f) {
        float t = (rel_speed - 2300.0f) / 2300.0f;
        return 0.55f - 0.25f * t;
    } else {
        return 0.30f;
    }
}

/**
 * @brief Tests collision between an Octane oriented bounding box (OBB) and the ball sphere.
 * 
 * @param car_pos Center of mass position of the car in world space.
 * @param car_basis 3x3 rotation matrix of the car (columns: forward, right, up).
 * @param ball_pos Center position of the ball in world space.
 * @param ball_radius Radius of the ball sphere (default BALL_RADIUS = 91.25f).
 * @param out_normal_world Contact normal pointing from the car toward the ball in world space.
 * @param out_contact_pt_world Contact point on the car surface in world space.
 * @param out_penetration Penetration depth (positive when intersecting).
 * @return true if car and ball are colliding, false otherwise.
 */
__device__ __forceinline__ bool test_car_ball_collision(
    const Vec3& car_pos,
    const Mat3& car_basis,
    const Vec3& hitbox_offset,
    const Vec3& hitbox_half,
    const Vec3& ball_pos,
    float ball_radius,
    Vec3& out_normal_world,
    Vec3& out_contact_pt_world,
    float& out_penetration)
{
    Vec3 hitbox_center = car_pos + car_basis * hitbox_offset;

    constexpr float BOX_MARGIN = 2.0f; // Bullet CONVEX_DISTANCE_MARGIN = 0.04 BT = 2.0 UU
    Vec3 inner_half = hitbox_half - Vec3(BOX_MARGIN, BOX_MARGIN, BOX_MARGIN);

    // 1. Transform ball into car hitbox local frame
    Vec3 d_world = ball_pos - hitbox_center;
    Vec3 p_local = car_basis.transpose() * d_world;

    // 2. Clamp to inner box half-extents (matching btSphereBoxCollisionAlgorithm)
    Vec3 q_inner(
        fmaxf(-inner_half.x, fminf(inner_half.x, p_local.x)),
        fmaxf(-inner_half.y, fminf(inner_half.y, p_local.y)),
        fmaxf(-inner_half.z, fminf(inner_half.z, p_local.z))
    );

    Vec3 diff_local = p_local - q_inner;
    float dist_sq = diff_local.length_sq();
    float total_radius = ball_radius + BOX_MARGIN;

    if (dist_sq > total_radius * total_radius) {
        return false;
    }

    Vec3 normal_local;
    Vec3 q_box;
    if (dist_sq > 1e-8f) {
        float dist = sqrtf(dist_sq);
        normal_local = diff_local * (1.0f / dist);
        out_penetration = total_radius - dist;
        q_box = q_inner + normal_local * BOX_MARGIN;
    } else {
        // Center is inside inner box: project to closest face (mirroring btSphereBoxCollisionAlgorithm::getSpherePenetration)
        float min_dist = inner_half.x - p_local.x;
        normal_local = Vec3(1.0f, 0.0f, 0.0f);
        Vec3 closest_pt = p_local;
        closest_pt.x = inner_half.x;

        float face_dist = inner_half.x + p_local.x;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_pt = p_local;
            closest_pt.x = -inner_half.x;
            normal_local = Vec3(-1.0f, 0.0f, 0.0f);
        }

        face_dist = inner_half.y - p_local.y;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_pt = p_local;
            closest_pt.y = inner_half.y;
            normal_local = Vec3(0.0f, 1.0f, 0.0f);
        }

        face_dist = inner_half.y + p_local.y;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_pt = p_local;
            closest_pt.y = -inner_half.y;
            normal_local = Vec3(0.0f, -1.0f, 0.0f);
        }

        face_dist = inner_half.z - p_local.z;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_pt = p_local;
            closest_pt.z = inner_half.z;
            normal_local = Vec3(0.0f, 0.0f, 1.0f);
        }

        face_dist = inner_half.z + p_local.z;
        if (face_dist < min_dist) {
            min_dist = face_dist;
            closest_pt = p_local;
            closest_pt.z = -inner_half.z;
            normal_local = Vec3(0.0f, 0.0f, -1.0f);
        }

        out_penetration = total_radius + min_dist;
        q_box = closest_pt + normal_local * BOX_MARGIN;
    }

    out_normal_world = car_basis * normal_local;
    out_contact_pt_world = hitbox_center + car_basis * q_box;
    return true;
}

__device__ __forceinline__ bool test_car_ball_collision(
    const Vec3& car_pos,
    const Mat3& car_basis,
    const Vec3& ball_pos,
    float ball_radius,
    Vec3& out_normal_world,
    Vec3& out_contact_pt_world,
    float& out_penetration)
{
    return test_car_ball_collision(
        car_pos, car_basis,
        get_octane_hitbox_offset(), get_octane_hitbox_half(),
        ball_pos, ball_radius,
        out_normal_world, out_contact_pt_world, out_penetration
    );
}

/**
 * @brief Resolves analytical Car-Ball collision with bilateral impulse exchange,
 * RocketSim extra hit impulse curve, velocity limits, and hit tracking state updates.
 */
__device__ __forceinline__ bool resolve_car_ball_collision(
    uint32_t env_idx,
    uint32_t car_idx,
    BallStateSoA& ball_state,
    CarStateSoA& car_state,
    ArenaStateSoA& arena_state,
    float dt,
    const MutatorConfig& mut_cfg,
    const CarConfig& car_cfg)
{
    // Load car pose & velocities
    Vec3 car_pos(car_state.pos_x[car_idx], car_state.pos_y[car_idx], car_state.pos_z[car_idx]);
    Vec3 car_vel(car_state.vel_x[car_idx], car_state.vel_y[car_idx], car_state.vel_z[car_idx]);
    Vec3 car_omega(car_state.ang_vel_x[car_idx], car_state.ang_vel_y[car_idx], car_state.ang_vel_z[car_idx]);
    Quat car_quat(car_state.q_w[car_idx], car_state.q_x[car_idx], car_state.q_y[car_idx], car_state.q_z[car_idx]);
    Mat3 car_basis = Mat3::from_quat(car_quat);

    // Load ball pose & velocities
    Vec3 ball_pos(ball_state.pos_x[env_idx], ball_state.pos_y[env_idx], ball_state.pos_z[env_idx]);
    Vec3 ball_vel(ball_state.vel_x[env_idx], ball_state.vel_y[env_idx], ball_state.vel_z[env_idx]);
    Vec3 ball_omega(ball_state.ang_vel_x[env_idx], ball_state.ang_vel_y[env_idx], ball_state.ang_vel_z[env_idx]);

    Vec3 normal_world;
    Vec3 contact_pt_world;
    float penetration = 0.0f;

    float ball_radius = mut_cfg.ball_radius;
    float ball_mass = mut_cfg.ball_mass;
    float car_mass = mut_cfg.car_mass;
    Vec3 hitbox_offset = car_cfg.hitbox_pos_offset;
    Vec3 hitbox_half = car_cfg.get_hitbox_half();

    if (!test_car_ball_collision(car_pos, car_basis, hitbox_offset, hitbox_half, ball_pos, ball_radius, normal_world, contact_pt_world, penetration)) {
        return false;
    }

    // Hit detected: determine non-consecutive hit for extra hit impulse
    uint64_t current_tick = arena_state.tick_count ? arena_state.tick_count[env_idx] : 0;
    bool is_first_hit = (car_state.ball_hit_is_valid[car_idx] == 0);
    uint64_t last_hit_tick = car_state.ball_hit_tick_count[car_idx];
    bool is_non_consecutive = is_first_hit || (current_tick > last_hit_tick + 1) || (last_hit_tick > current_tick);

    // Compute RocketSim Piecewise Extra Hit Impulse (RLConst.h:135-139, 496-503, Ball.cpp:261-285)
    Vec3 car_forward = car_basis.forward;
    Vec3 rel_pos = ball_pos - car_pos;
    Vec3 rel_vel = ball_vel - car_vel;
    float rel_speed = fminf(rel_vel.length(), 4600.0f); // BALL_CAR_EXTRA_IMPULSE_MAXDELTAVEL_UU

    Vec3 added_vel(0.0f, 0.0f, 0.0f);
    if (is_non_consecutive && rel_speed > 0.0f) {
        float z_scale = 0.35f; // BALL_CAR_EXTRA_IMPULSE_Z_SCALE
        Vec3 scaled_rel_pos(rel_pos.x, rel_pos.y, rel_pos.z * z_scale);
        float scaled_len = scaled_rel_pos.length();
        Vec3 hit_dir = (scaled_len > 1e-6f) ? (scaled_rel_pos * (1.0f / scaled_len)) : Vec3(0.0f, 0.0f, 1.0f);

        // BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE = 0.65f -> (1 - 0.65f) = 0.35f
        Vec3 fwd_adj = car_forward * (hit_dir.dot(car_forward) * (1.0f - 0.65f));
        Vec3 adj_hit_dir = hit_dir - fwd_adj;
        float adj_len = adj_hit_dir.length();
        hit_dir = (adj_len > 1e-6f) ? (adj_hit_dir * (1.0f / adj_len)) : hit_dir;

        float factor = evaluate_ball_car_extra_impulse_factor(rel_speed);
        added_vel = hit_dir * (rel_speed * factor * mut_cfg.ball_hit_extra_force_scale);
    }

    // Contact point lever arms
    Vec3 r_c = contact_pt_world - car_pos;
    Vec3 r_b = normal_world * (-ball_radius);

    // Contact point velocities
    Vec3 v_c_pt = car_vel + car_omega.cross(r_c);
    Vec3 v_b_pt = ball_vel + ball_omega.cross(r_b);
    Vec3 v_rel_pt = v_b_pt - v_c_pt;
    float vn = v_rel_pt.dot(normal_world);

    // Inertia and mass parameters
    float inv_m_c = 1.0f / car_mass;
    float inv_m_b = 1.0f / ball_mass;
    Vec3 inv_I_c_local = car_cfg.calculate_inv_inertia(car_mass);
    float inv_I_b = 1.0f / (0.4f * ball_mass * ball_radius * ball_radius);

    // Bilateral impulse exchange (restitution e = 0.0f, friction mu = 2.0f)
    Vec3 J_total(0.0f, 0.0f, 0.0f);
    if (vn < 0.0f) {
        // Car rotational term for normal direction
        Vec3 u_c_n = r_c.cross(normal_world);
        Vec3 u_c_n_local = car_basis.transpose() * u_c_n;
        Vec3 w_c_n_local(u_c_n_local.x * inv_I_c_local.x, u_c_n_local.y * inv_I_c_local.y, u_c_n_local.z * inv_I_c_local.z);
        float rot_term_c = u_c_n_local.dot(w_c_n_local);

        // Ball rotational term for normal direction
        Vec3 u_b_n = r_b.cross(normal_world);
        float rot_term_b = u_b_n.length_sq() * inv_I_b;

        float Kn = inv_m_c + inv_m_b + rot_term_c + rot_term_b;
        float Jn = (Kn > 1e-8f) ? (-vn / Kn) : 0.0f; // e = 0.0f
        Vec3 normal_impulse = normal_world * Jn;

        // Tangential friction (mu = 2.0f)
        Vec3 v_t = v_rel_pt - normal_world * vn;
        float vt_mag = v_t.length();
        Vec3 tangent_impulse(0.0f, 0.0f, 0.0f);
        if (vt_mag > 1e-5f) {
            Vec3 t = v_t * (1.0f / vt_mag);

            Vec3 u_c_t = r_c.cross(t);
            Vec3 u_c_t_local = car_basis.transpose() * u_c_t;
            Vec3 w_c_t_local(u_c_t_local.x * inv_I_c_local.x, u_c_t_local.y * inv_I_c_local.y, u_c_t_local.z * inv_I_c_local.z);
            float rot_t_c = u_c_t_local.dot(w_c_t_local);

            Vec3 u_b_t = r_b.cross(t);
            float rot_t_b = u_b_t.length_sq() * inv_I_b;

            float Kt = inv_m_c + inv_m_b + rot_t_c + rot_t_b;
            float Jt_desired = (Kt > 1e-8f) ? (vt_mag / Kt) : 0.0f;
            float Jt = fminf(Jt_desired, 2.0f * Jn); // mu = 2.0f
            tangent_impulse = t * (-Jt);
        }

        J_total = normal_impulse + tangent_impulse;
    }

    // Apply impulse to ball
    ball_vel = ball_vel + J_total * inv_m_b;
    Vec3 tau_ball = r_b.cross(J_total);
    ball_omega = ball_omega + tau_ball * inv_I_b;

    // Apply impulse to car (equal and opposite)
    car_vel = car_vel - J_total * inv_m_c;
    Vec3 tau_car = r_c.cross(J_total * (-1.0f));
    Vec3 tau_car_local = car_basis.transpose() * tau_car;
    Vec3 delta_omega_car_local(tau_car_local.x * inv_I_c_local.x, tau_car_local.y * inv_I_c_local.y, tau_car_local.z * inv_I_c_local.z);
    car_omega = car_omega + car_basis * delta_omega_car_local;

    // Apply RocketSim extra hit impulse to ball
    ball_vel = ball_vel + added_vel;

    // Resolve interpenetration: distribute split-impulse penetration push
    if (penetration > 0.0f) {
        float p_push = penetration * 0.8f; // erp2 = 0.8f
        float mass_sum = car_mass + ball_mass;
        float ball_frac = car_mass / mass_sum;
        float car_frac  = ball_mass / mass_sum;

        ball_pos = ball_pos + normal_world * (p_push * ball_frac);
        car_pos  = car_pos  - normal_world * (p_push * car_frac);
    }

    // Velocity Clamping
    float b_speed = ball_vel.length();
    if (b_speed > mut_cfg.ball_max_speed) {
        ball_vel = ball_vel * (mut_cfg.ball_max_speed / b_speed);
    }
    float b_ang_speed = ball_omega.length();
    if (b_ang_speed > BALL_MAX_ANG_SPEED) {
        ball_omega = ball_omega * (BALL_MAX_ANG_SPEED / b_ang_speed);
    }
    float c_speed = car_vel.length();
    if (c_speed > CAR_MAX_SPEED) {
        car_vel = car_vel * (CAR_MAX_SPEED / c_speed);
    }
    float c_ang_speed = car_omega.length();
    if (c_ang_speed > CAR_MAX_ANG_SPEED) {
        car_omega = car_omega * (CAR_MAX_ANG_SPEED / c_ang_speed);
    }

    // Relative position on ball surface
    Vec3 rel_pos_on_ball = normal_world * (-ball_radius);

    // Populate Hit State Tracking
    car_state.ball_hit_is_valid[car_idx] = 1;
    car_state.ball_hit_rel_pos_x[car_idx] = rel_pos_on_ball.x;
    car_state.ball_hit_rel_pos_y[car_idx] = rel_pos_on_ball.y;
    car_state.ball_hit_rel_pos_z[car_idx] = rel_pos_on_ball.z;
    car_state.ball_hit_extra_hit_force_x[car_idx] = added_vel.x;
    car_state.ball_hit_extra_hit_force_y[car_idx] = added_vel.y;
    car_state.ball_hit_extra_hit_force_z[car_idx] = added_vel.z;
    car_state.ball_hit_tick_count[car_idx] = current_tick;

    // Write back updated ball state
    ball_state.pos_x[env_idx] = ball_pos.x;
    ball_state.pos_y[env_idx] = ball_pos.y;
    ball_state.pos_z[env_idx] = ball_pos.z;

    ball_state.vel_x[env_idx] = ball_vel.x;
    ball_state.vel_y[env_idx] = ball_vel.y;
    ball_state.vel_z[env_idx] = ball_vel.z;

    ball_state.ang_vel_x[env_idx] = ball_omega.x;
    ball_state.ang_vel_y[env_idx] = ball_omega.y;
    ball_state.ang_vel_z[env_idx] = ball_omega.z;

    // Write back updated car state
    car_state.pos_x[car_idx] = car_pos.x;
    car_state.pos_y[car_idx] = car_pos.y;
    car_state.pos_z[car_idx] = car_pos.z;

    car_state.vel_x[car_idx] = car_vel.x;
    car_state.vel_y[car_idx] = car_vel.y;
    car_state.vel_z[car_idx] = car_vel.z;

    car_state.ang_vel_x[car_idx] = car_omega.x;
    car_state.ang_vel_y[car_idx] = car_omega.y;
    car_state.ang_vel_z[car_idx] = car_omega.z;

    return true;
}

__device__ __forceinline__ bool resolve_car_ball_collision(
    uint32_t env_idx,
    uint32_t car_idx,
    BallStateSoA& ball_state,
    CarStateSoA& car_state,
    ArenaStateSoA& arena_state,
    float dt)
{
    uint8_t ht = car_state.hitbox_type ? car_state.hitbox_type[car_idx] : 0;
    return resolve_car_ball_collision(
        env_idx, car_idx, ball_state, car_state, arena_state, dt,
        MutatorConfig(), get_car_config(ht)
    );
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
    float friction = BALL_FRICTION,
    float gravity_z = GRAVITY_Z,
    float dt = 1.0f / 120.0f)
{
    float dist = 0.0f;
    Vec3 normal(0.0f, 0.0f, 1.0f);
    arena_sdf_and_normal(pos, dist, normal);

    if (dist < radius) {
        float penetration = radius - dist;
        pos = pos + normal * penetration;

        float vn = normal.dot(vel);
        if (vn < 0.0f) {
            // Low-speed restitution threshold matching Bullet (0.2 BT units = 10.0 UU/s)
            float e = (fabsf(vn) < 10.0f) ? 0.0f : restitution;
            float delta_vn = -(1.0f + e) * vn;
            Vec3 normal_impulse = normal * delta_vn;

            // Surface contact slip velocity (contact point r = -radius * normal)
            // v_contact = vel + ang_vel x r = vel - (ang_vel x normal) * radius
            Vec3 v_contact = vel - ang_vel.cross(normal) * radius;
            Vec3 v_slip = v_contact - normal * normal.dot(v_contact);
            float slip_speed = v_slip.length();

            Vec3 tangent_impulse(0.0f, 0.0f, 0.0f);
            if (slip_speed > 1e-4f) {
                // Effective tangential mass ratio for rolling solid sphere: 2/7
                float j_slip_per_m = (2.0f / 7.0f) * slip_speed;
                float j_coulomb_max_per_m = friction * delta_vn;

                float j_fric_mag = fminf(j_slip_per_m, j_coulomb_max_per_m);
                Vec3 t_dir = v_slip * (1.0f / slip_speed);
                tangent_impulse = t_dir * (-j_fric_mag);

                // Rotational torque coupling:
                // Delta omega = (r x J_t) / I = (-radius * normal x J_t) / (0.4 * M * radius^2)
                //             = (2.5 / radius) * (tangent_impulse x normal)
                Vec3 delta_omega = tangent_impulse.cross(normal) * (2.5f / radius);
                ang_vel = ang_vel + delta_omega;
            }

            vel = vel + normal_impulse + tangent_impulse;
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
    float dt,
    const Vec3& hitbox_offset = get_octane_hitbox_offset(),
    const Vec3& hitbox_half = get_octane_hitbox_half(),
    float car_mass = CAR_MASS,
    const Vec3& inv_inertia_bt = Vec3(1.0f / 54.0f, 1.0f / 96.0f, 1.0f / 132.0f))
{
    bool has_contact = false;
    Vec3 sum_normal(0.0f, 0.0f, 0.0f);
    int num_contacts = 0;
    int contact_indices[8];
    float contact_dists[8];
    Vec3 contact_normals[8];
    Vec3 contact_corners[8];

    constexpr float CHASSIS_MARGIN_UU = 0.04f * 50.0f; // 2.0f UU (CONVEX_DISTANCE_MARGIN)

    // 1. Identify penetrating corners against Arena SDF
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

        if (dist <= CHASSIS_MARGIN_UU) {
            has_contact = true;
            sum_normal = sum_normal + normal;
            contact_indices[num_contacts] = i;
            contact_dists[num_contacts] = dist;
            contact_normals[num_contacts] = normal;
            contact_corners[num_contacts] = world_corner;
            num_contacts++;
        }
    }

    // 2. Resolve contact impulses matching Bullet resolveSingleCollision
    if (num_contacts > 0) {
        float contact_scale = 1.0f / static_cast<float>(num_contacts);
        Vec3 vel_bt = vel * 0.02f;

        for (int k = 0; k < num_contacts; ++k) {
            float dist = contact_dists[k];
            Vec3 normal = contact_normals[k];
            Vec3 world_corner = contact_corners[k];

            float dist_bt = dist * 0.02f;
            float penetration_bt = 0.04f - dist_bt;
            if (penetration_bt > 0.0f) {
                pos = pos + normal * (penetration_bt * 50.0f * contact_scale);
            }

            Vec3 rel_pos_bt = (world_corner - pos) * 0.02f;
            Vec3 pt_vel_bt = vel_bt + omega.cross(rel_pos_bt);
            float rel_vel = normal.dot(pt_vel_bt);

            if (rel_vel < 0.0f || penetration_bt > 0.0f) {
                float pos_error = (0.2f * penetration_bt) / dt;
                float vel_error = -(1.0f + 0.3f) * rel_vel; // CARWORLD_COLLISION_RESTITUTION = 0.3f

                Vec3 c0 = rel_pos_bt.cross(normal);
                Vec3 c0_loc = basis.transpose() * c0;
                float denom = (1.0f / car_mass) + (c0_loc.x * c0_loc.x * inv_inertia_bt.x
                                                + c0_loc.y * c0_loc.y * inv_inertia_bt.y
                                                + c0_loc.z * c0_loc.z * inv_inertia_bt.z);

                float normal_impulse_bt = fmaxf(0.0f, (pos_error + vel_error) / denom) * contact_scale;
                Vec3 imp_bt = normal * normal_impulse_bt;

                // Coulomb friction: CARWORLD_COLLISION_FRICTION = 0.3f
                Vec3 tangent_vel = pt_vel_bt - normal * rel_vel;
                float tangent_speed = tangent_vel.length();
                if (tangent_speed > 1e-4f && normal_impulse_bt > 0.0f) {
                    Vec3 tangent_dir = tangent_vel * (1.0f / tangent_speed);
                    float friction_impulse_bt = fminf(tangent_speed * car_mass, normal_impulse_bt * 0.3f);
                    imp_bt = imp_bt - tangent_dir * friction_impulse_bt;
                }

                vel_bt = vel_bt + imp_bt * (1.0f / car_mass);
                Vec3 ang_imp_loc = basis.transpose() * (rel_pos_bt.cross(imp_bt));
                Vec3 d_omega_loc(
                    ang_imp_loc.x * inv_inertia_bt.x,
                    ang_imp_loc.y * inv_inertia_bt.y,
                    ang_imp_loc.z * inv_inertia_bt.z
                );
                omega = omega + basis * d_omega_loc;
            }
        }
        vel = vel_bt * 50.0f;
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
