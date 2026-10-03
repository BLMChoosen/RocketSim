#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "vec3.cuh"

namespace rocketsim_cuda {

struct alignas(4) Quat {
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    __host__ __device__ constexpr Quat() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ constexpr Quat(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}

    static __host__ __device__ constexpr Quat identity() { return Quat(1.0f, 0.0f, 0.0f, 0.0f); }

    __host__ __device__ constexpr Quat operator+(const Quat& o) const {
        return Quat(w + o.w, x + o.x, y + o.y, z + o.z);
    }

    __host__ __device__ constexpr Quat operator-(const Quat& o) const {
        return Quat(w - o.w, x - o.x, y - o.y, z - o.z);
    }

    __host__ __device__ constexpr Quat operator*(float s) const {
        return Quat(w * s, x * s, y * s, z * s);
    }

    __host__ __device__ constexpr Quat operator*(const Quat& o) const {
        return Quat(
            w * o.w - x * o.x - y * o.y - z * o.z,
            w * o.x + x * o.w + y * o.z - z * o.y,
            w * o.y - x * o.z + y * o.w + z * o.x,
            w * o.z + x * o.y - y * o.x + z * o.w
        );
    }

    __host__ __device__ constexpr Quat conjugate() const { return Quat(w, -x, -y, -z); }

    __host__ __device__ float length_sq() const {
        return w * w + x * x + y * y + z * z;
    }

    __host__ __device__ float length() const {
        return sqrtf(length_sq());
    }

    __host__ __device__ Quat normalized(float eps = 1e-8f) const {
        float lsq = length_sq();
        if (lsq > eps) {
            float inv = 1.0f / sqrtf(lsq);
            return Quat(w * inv, x * inv, y * inv, z * inv);
        }
        return identity();
    }

    // Efficient Rodrigues vector rotation: v' = v + 2 * q_xyz x (q_xyz x v + w * v)
    __host__ __device__ Vec3 rotate(const Vec3& v) const {
        Vec3 qv(x, y, z);
        Vec3 t = 2.0f * qv.cross(v);
        return v + w * t + qv.cross(t);
    }

    // Rocket League coordinate conventions: X=Forward, Y=Right, Z=Up
    __host__ __device__ Vec3 forward() const { return rotate(Vec3(1.0f, 0.0f, 0.0f)); }
    __host__ __device__ Vec3 right() const { return rotate(Vec3(0.0f, 1.0f, 0.0f)); }
    __host__ __device__ Vec3 up() const { return rotate(Vec3(0.0f, 0.0f, 1.0f)); }

    // Antipodal Chebyshev Distance (GEMINI.md Section 3.1):
    // Accounts for q and -q representing identical physical rotations in SO(3)
    __host__ __device__ float chebyshev_dist(const Quat& o) const {
        float d_pos = fmaxf(fmaxf(fabsf(w - o.w), fabsf(x - o.x)), fmaxf(fabsf(y - o.y), fabsf(z - o.z)));
        float d_neg = fmaxf(fmaxf(fabsf(w + o.w), fabsf(x + o.x)), fmaxf(fabsf(y + o.y), fabsf(z + o.z)));
        return fminf(d_pos, d_neg);
    }
};

__host__ __device__ inline constexpr Quat operator*(float s, const Quat& q) {
    return Quat(s * q.w, s * q.x, s * q.y, s * q.z);
}

inline std::ostream& operator<<(std::ostream& os, const Quat& q) {
    os << "Quat(" << q.w << ", " << q.x << ", " << q.y << ", " << q.z << ")";
    return os;
}

} // namespace rocketsim_cuda
