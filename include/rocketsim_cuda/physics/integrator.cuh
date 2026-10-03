#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/quat.cuh"
#include "rocketsim_cuda/math/mat3.cuh"

namespace rocketsim_cuda {

/**
 * @brief Exponential map quaternion rotation integration matching Bullet 3.24 btTransformUtil::integrateTransform.
 * 
 * Ensures 1:1 numerical parity with Bullet Physics without orientation drift.
 * 
 * @param cur_q Current normalized orientation quaternion (w, x, y, z)
 * @param angvel Angular velocity vector in world space (rad/s)
 * @param dt Timestep duration in seconds (1/120 s)
 * @return Integrated and normalized orientation quaternion
 */
__device__ __forceinline__ Quat bullet_integrate_quaternion(const Quat& cur_q, const Vec3& angvel, float dt) {
    float fAngle2 = angvel.length_sq();
    float fAngle = 0.0f;
    if (fAngle2 > 1e-8f) {
        fAngle = sqrtf(fAngle2);
    }

    constexpr float ANGULAR_MOTION_THRESHOLD = 0.5f * 3.14159265358979323846f;
    if (fAngle * dt > ANGULAR_MOTION_THRESHOLD) {
        fAngle = ANGULAR_MOTION_THRESHOLD / dt;
    }

    Vec3 axis;
    if (fAngle < 0.001f) {
        // Taylor expansion of sinc function
        axis = angvel * (0.5f * dt - (dt * dt * dt) * 0.020833333333333333f * fAngle * fAngle);
    } else {
        axis = angvel * (sinf(0.5f * fAngle * dt) / fAngle);
    }

    Quat dorn(cosf(fAngle * dt * 0.5f), axis.x, axis.y, axis.z);
    Quat predictedOrn = dorn * cur_q;
    return predictedOrn.normalized();
}

/**
 * @brief Symplectic Euler linear integration step.
 * 
 * v_{t+1} = v_t + (g + f_ext / m) * dt
 * p_{t+1} = p_t + v_{t+1} * dt
 */
__device__ __forceinline__ void symplectic_euler_linear(
    Vec3& pos,
    Vec3& vel,
    const Vec3& ext_force,
    float mass,
    float dt,
    const Vec3& gravity = Vec3(0.0f, 0.0f, GRAVITY_Z))
{
    float inv_mass = (mass > 0.0f) ? (1.0f / mass) : 0.0f;
    vel = vel + (gravity + ext_force * inv_mass) * dt;
    pos = pos + vel * dt;
}

/**
 * @brief Rotational dynamics update using world-space inverse inertia tensor.
 * 
 * Matches Bullet Physics rigid body without gyroscopic force (m_rigidbodyFlags = 0).
 * omega_{t+1} = omega_t + (basis * inv_inertia_local * basis^T) * torque * dt
 */
__device__ __forceinline__ void bullet_angular_dynamics(
    Vec3& omega,
    const Vec3& ext_torque,
    const Vec3& inv_inertia_local,
    const Mat3& basis,
    float dt)
{
    // Local to world transformation of diagonal inverse inertia:
    // I_world^-1 = basis * diag(inv_inertia_local) * basis^T
    // First, v_loc = basis^T * torque
    Vec3 t_local = basis.transpose() * ext_torque;
    // Apply local diagonal inverse inertia
    Vec3 alpha_local(
        t_local.x * inv_inertia_local.x,
        t_local.y * inv_inertia_local.y,
        t_local.z * inv_inertia_local.z
    );
    // Transform angular acceleration back to world space: alpha_world = basis * alpha_local
    Vec3 alpha_world = basis * alpha_local;
    omega = omega + alpha_world * dt;
}

/**
 * @brief Applies Bullet rigid body exponential damping (applyDamping).
 */
__device__ __forceinline__ void apply_rigid_body_damping(
    Vec3& lin_vel,
    Vec3& ang_vel,
    float lin_damping,
    float ang_damping,
    float dt)
{
    if (lin_damping > 0.0f) {
        lin_vel = lin_vel * powf(1.0f - lin_damping, dt);
    }
    if (ang_damping > 0.0f) {
        ang_vel = ang_vel * powf(1.0f - ang_damping, dt);
    }
}

} // namespace rocketsim_cuda
