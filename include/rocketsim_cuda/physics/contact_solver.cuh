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
    const Vec3& ball_pos,
    float ball_radius,
    Vec3& out_normal_world,
    Vec3& out_contact_pt_world,
    float& out_penetration)
{
    Vec3 hitbox_offset = get_octane_hitbox_offset();
    Vec3 hitbox_half = get_octane_hitbox_half();
    Vec3 hitbox_center = car_pos + car_basis * hitbox_offset;

    // 1. Transform ball into car hitbox local frame
    Vec3 d_world = ball_pos - hitbox_center;
    Vec3 p_local = car_basis.transpose() * d_world;

    // 2. Clamp to box half-extents to find closest point on OBB
    Vec3 q_local(
        fmaxf(-hitbox_half.x, fminf(hitbox_half.x, p_local.x)),
        fmaxf(-hitbox_half.y, fminf(hitbox_half.y, p_local.y)),
        fmaxf(-hitbox_half.z, fminf(hitbox_half.z, p_local.z))
    );

    Vec3 diff_local = p_local - q_local;
    float dist_sq = diff_local.length_sq();

    if (dist_sq > ball_radius * ball_radius) {
        return false;
    }

    Vec3 normal_local;
    if (dist_sq > 1e-8f) {
        float dist = sqrtf(dist_sq);
        normal_local = diff_local * (1.0f / dist);
        out_penetration = ball_radius - dist;
    } else {
        // Center is inside the box: project to the closest face
        float dx = hitbox_half.x - fabsf(p_local.x);
        float dy = hitbox_half.y - fabsf(p_local.y);
        float dz = hitbox_half.z - fabsf(p_local.z);

        if (dx <= dy && dx <= dz) {
            normal_local = Vec3((p_local.x >= 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f);
            out_penetration = ball_radius + dx;
            q_local.x = (p_local.x >= 0.0f) ? hitbox_half.x : -hitbox_half.x;
        } else if (dy <= dz) {
            normal_local = Vec3(0.0f, (p_local.y >= 0.0f) ? 1.0f : -1.0f, 0.0f);
            out_penetration = ball_radius + dy;
            q_local.y = (p_local.y >= 0.0f) ? hitbox_half.y : -hitbox_half.y;
        } else {
            normal_local = Vec3(0.0f, 0.0f, (p_local.z >= 0.0f) ? 1.0f : -1.0f);
            out_penetration = ball_radius + dz;
            q_local.z = (p_local.z >= 0.0f) ? hitbox_half.z : -hitbox_half.z;
        }
    }

    out_normal_world = car_basis * normal_local;
    out_contact_pt_world = hitbox_center + car_basis * q_local;
    return true;
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
    float dt)
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

    if (!test_car_ball_collision(car_pos, car_basis, ball_pos, BALL_RADIUS, normal_world, contact_pt_world, penetration)) {
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
        added_vel = hit_dir * (rel_speed * factor);
    }

    // Contact point lever arms
    Vec3 r_c = contact_pt_world - car_pos;
    Vec3 r_b = contact_pt_world - ball_pos;

    // Contact point velocities
    Vec3 v_c_pt = car_vel + car_omega.cross(r_c);
    Vec3 v_b_pt = ball_vel + ball_omega.cross(r_b);
    Vec3 v_rel_pt = v_b_pt - v_c_pt;
    float vn = v_rel_pt.dot(normal_world);

    // Inertia and mass parameters
    float inv_m_c = 1.0f / CAR_MASS;
    float inv_m_b = 1.0f / BALL_MASS;
    Vec3 inv_I_c_local = get_octane_inv_inertia_local();
    float inv_I_b = 1.0f / (0.4f * BALL_MASS * BALL_RADIUS * BALL_RADIUS);

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

    // Resolve interpenetration: push ball along contact normal
    if (penetration > 0.0f) {
        ball_pos = ball_pos + normal_world * penetration;
    }

    // Velocity Clamping
    // Clamp ball linear speed to 6000.0f UU/s
    float b_speed = ball_vel.length();
    if (b_speed > BALL_MAX_SPEED) {
        ball_vel = ball_vel * (BALL_MAX_SPEED / b_speed);
    }
    // Clamp ball angular speed to 6.0f rad/s
    float b_ang_speed = ball_omega.length();
    if (b_ang_speed > BALL_MAX_ANG_SPEED) {
        ball_omega = ball_omega * (BALL_MAX_ANG_SPEED / b_ang_speed);
    }
    // Clamp car linear speed to 2300.0f UU/s
    float c_speed = car_vel.length();
    if (c_speed > CAR_MAX_SPEED) {
        car_vel = car_vel * (CAR_MAX_SPEED / c_speed);
    }
    // Clamp car angular speed to 5.5f rad/s
    float c_ang_speed = car_omega.length();
    if (c_ang_speed > CAR_MAX_ANG_SPEED) {
        car_omega = car_omega * (CAR_MAX_ANG_SPEED / c_ang_speed);
    }

    // Relative position on ball surface (in ball local coordinates matching Bullet manifoldPoint.m_localPointA)
    Vec3 rel_pos_on_ball = normal_world * (-BALL_RADIUS);

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
    car_state.vel_x[car_idx] = car_vel.x;
    car_state.vel_y[car_idx] = car_vel.y;
    car_state.vel_z[car_idx] = car_vel.z;

    car_state.ang_vel_x[car_idx] = car_omega.x;
    car_state.ang_vel_y[car_idx] = car_omega.y;
    car_state.ang_vel_z[car_idx] = car_omega.z;

    return true;
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
 * Implements Bullet's btSequentialImpulseConstraintSolver for contact and friction resolution.
 */
__device__ __forceinline__ void resolve_chassis_arena_collision(
    uint32_t car_idx,
    CarStateSoA& car_state,
    const Vec3& pos,
    const Vec3& vel_pre,
    const Vec3& omega_pre,
    Vec3& vel,
    Vec3& omega,
    const Mat3& basis,
    float dt)
{
    Vec3 hitbox_offset = get_octane_hitbox_offset();
    Vec3 hitbox_half = get_octane_hitbox_half();
    constexpr float BOX_SAFE_MARGIN_INSET = 0.067045f; // (0.04f - 0.0386591f) * 50.0f UU
    Vec3 hitbox_half_eff = hitbox_half - Vec3(BOX_SAFE_MARGIN_INSET, BOX_SAFE_MARGIN_INSET, BOX_SAFE_MARGIN_INSET);
    constexpr float CHASSIS_MARGIN_UU = 1.932955f; // Margin of btBoxShape in UU

    float min_dist = 1e9f;
    int best_corner = -1;
    Vec3 best_normal(0.0f, 0.0f, 1.0f);
    Vec3 best_world_corner(0.0f, 0.0f, 0.0f);

    // Check 8 corner vertices of oriented hitbox
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float sx = (i & 1) ? 1.0f : -1.0f;
        float sy = (i & 2) ? 1.0f : -1.0f;
        float sz = (i & 4) ? 1.0f : -1.0f;

        Vec3 local_corner = hitbox_offset + Vec3(
            sx * hitbox_half_eff.x,
            sy * hitbox_half_eff.y,
            sz * hitbox_half_eff.z
        );

        Vec3 world_corner = pos + basis * local_corner;
        float dist = 0.0f;
        Vec3 normal(0.0f, 0.0f, 1.0f);
        arena_sdf_and_normal(world_corner, dist, normal);

        if (dist < min_dist) {
            min_dist = dist;
            best_corner = i;
            best_normal = normal;
            best_world_corner = world_corner;
        }
    }

    if (min_dist <= CHASSIS_MARGIN_UU && best_corner >= 0) {
        Vec3 rel_pos_bt = (best_world_corner - pos) * 0.02f;
        Vec3 vel_bt = vel * 0.02f;
        Vec3 pt_vel_bt = vel_bt + omega.cross(rel_pos_bt);
        float vn_bt = best_normal.dot(pt_vel_bt);

        if (vn_bt < 0.0f) {
            car_state.world_contact_has_contact[car_idx] = 1;
            car_state.world_contact_normal_x[car_idx]    = best_normal.x;
            car_state.world_contact_normal_y[car_idx]    = best_normal.y;
            car_state.world_contact_normal_z[car_idx]    = best_normal.z;

            // Bullet sequential impulse constraint solver for static contact
            Vec3 inv_I = get_octane_inv_inertia_bt();
            constexpr float inv_m = 1.0f / CAR_MASS;

            Vec3 c_n = rel_pos_bt.cross(best_normal);
            Vec3 c_n_loc = basis.transpose() * c_n;
            float denom_n = inv_m + (c_n_loc.x * c_n_loc.x * inv_I.x +
                                     c_n_loc.y * c_n_loc.y * inv_I.y +
                                     c_n_loc.z * c_n_loc.z * inv_I.z);
            float jac_n = 1.0f / denom_n;
            Vec3 ang_comp_n = basis * Vec3(c_n_loc.x * inv_I.x, c_n_loc.y * inv_I.y, c_n_loc.z * inv_I.z);

            // Positional error if penetrated past margin core
            float penetration_bt = min_dist * 0.02f;
            float pos_err = (penetration_bt < 0.0f) ? (-penetration_bt * 0.8f / dt) : 0.0f;
            Vec3 pt_vel_pre_bt = vel_pre * 0.02f + omega_pre.cross(rel_pos_bt);
            float vn_pre_bt = best_normal.dot(pt_vel_pre_bt);
            float rest = (vn_pre_bt < -0.2f) ? (0.3f * (-vn_pre_bt)) : 0.0f;
            float rhs_n = (rest - vn_bt + pos_err) * jac_n;

            // Friction setup (Coulomb friction mu = 0.3)
            Vec3 v_tan_bt = pt_vel_bt - best_normal * vn_bt;
            float v_tan_len = v_tan_bt.length();
            Vec3 lat_dir = (v_tan_len > 1e-4f) ? (v_tan_bt * (1.0f / v_tan_len)) : Vec3(0.0f, 0.0f, 0.0f);
            float vt_bt = v_tan_len;

            Vec3 c_t = rel_pos_bt.cross(lat_dir);
            Vec3 c_t_loc = basis.transpose() * c_t;
            float denom_t = inv_m + (c_t_loc.x * c_t_loc.x * inv_I.x +
                                     c_t_loc.y * c_t_loc.y * inv_I.y +
                                     c_t_loc.z * c_t_loc.z * inv_I.z);
            float jac_t = (denom_t > 1e-6f) ? (1.0f / denom_t) : 0.0f;
            Vec3 ang_comp_t = basis * Vec3(c_t_loc.x * inv_I.x, c_t_loc.y * inv_I.y, c_t_loc.z * inv_I.z);
            float rhs_t = -vt_bt * jac_t;

            // 10 Gauss-Seidel sequential impulse iterations matching Bullet
            float applied_n = 0.0f;
            float applied_t = 0.0f;
            Vec3 delta_lin_bt(0.0f, 0.0f, 0.0f);
            Vec3 delta_ang(0.0f, 0.0f, 0.0f);

            #pragma unroll
            for (int iter = 0; iter < 10; ++iter) {
                // Normal constraint row
                float dv_n = best_normal.dot(delta_lin_bt) + c_n.dot(delta_ang);
                float delta_n = rhs_n - dv_n * jac_n;
                float new_n = fmaxf(0.0f, applied_n + delta_n);
                delta_n = new_n - applied_n;
                applied_n = new_n;
                delta_lin_bt = delta_lin_bt + best_normal * (delta_n * inv_m);
                delta_ang = delta_ang + ang_comp_n * delta_n;

                // Friction constraint row
                if (v_tan_len > 1e-4f) {
                    float max_fric = 0.3f * applied_n;
                    float dv_t = lat_dir.dot(delta_lin_bt) + c_t.dot(delta_ang);
                    float delta_t = rhs_t - dv_t * jac_t;
                    float new_t = fminf(fmaxf(applied_t + delta_t, -max_fric), max_fric);
                    delta_t = new_t - applied_t;
                    applied_t = new_t;
                    delta_lin_bt = delta_lin_bt + lat_dir * (delta_t * inv_m);
                    delta_ang = delta_ang + ang_comp_t * delta_t;
                }
            }

            vel = vel + delta_lin_bt * 50.0f;
            omega = omega + delta_ang;
            if (car_idx == 0) {
                printf("  [GPU CHASSIS SOLVER] vn=%f rhs_n=%f jac_n=%f applied_n=%f applied_t=%f rel_pos=(%f,%f,%f) pt_vel=(%f,%f,%f)\n",
                       vn_bt, rhs_n, jac_n, applied_n, applied_t,
                       rel_pos_bt.x, rel_pos_bt.y, rel_pos_bt.z,
                       pt_vel_bt.x, pt_vel_bt.y, pt_vel_bt.z);
            }
            return;
        }
    }

    car_state.world_contact_has_contact[car_idx] = 0;
    car_state.world_contact_normal_x[car_idx]    = 0.0f;
    car_state.world_contact_normal_y[car_idx]    = 0.0f;
    car_state.world_contact_normal_z[car_idx]    = 0.0f;
}

} // namespace rocketsim_cuda
