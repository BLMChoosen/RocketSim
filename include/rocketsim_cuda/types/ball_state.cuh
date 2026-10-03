#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "../math/vec3.cuh"
#include "../math/quat.cuh"

namespace rocketsim_cuda {

// Host POD structure for Ball (used in Golden Master & Interop)
struct BallStatePOD {
    Vec3 pos     = Vec3(0.0f, 0.0f, 93.15f);
    Vec3 vel     = Vec3(0.0f, 0.0f, 0.0f);
    Quat quat    = Quat::identity();
    Vec3 ang_vel = Vec3(0.0f, 0.0f, 0.0f);
};

// Device SoA structure for N balls (passed by value to CUDA kernels)
struct BallStateSoA {
    // Rigid Body Transform & Velocities (Coalesced 128-byte transactions)
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

    // Gamemode extensions (Heatseeker / Dropshot)
    float* __restrict__ hs_y_target_dir       = nullptr;
    float* __restrict__ hs_cur_target_speed   = nullptr;
    float* __restrict__ hs_time_since_hit     = nullptr;

    int32_t* __restrict__ ds_charge_level          = nullptr;
    float* __restrict__ ds_accumulated_hit_force   = nullptr;
    float* __restrict__ ds_y_target_dir            = nullptr;
    uint8_t* __restrict__ ds_has_damaged           = nullptr;
    uint64_t* __restrict__ ds_last_damage_tick     = nullptr;

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

    __device__ inline BallStatePOD get_pod(uint32_t idx) const {
        BallStatePOD pod;
        pod.pos     = get_pos(idx);
        pod.vel     = get_vel(idx);
        pod.quat    = get_quat(idx);
        pod.ang_vel = get_ang_vel(idx);
        return pod;
    }

    __device__ inline void set_pod(uint32_t idx, const BallStatePOD& pod) {
        set_pos(idx, pod.pos);
        set_vel(idx, pod.vel);
        set_quat(idx, pod.quat);
        set_ang_vel(idx, pod.ang_vel);
    }
};

} // namespace rocketsim_cuda
