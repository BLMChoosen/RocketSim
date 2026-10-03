#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "vec3.cuh"
#include "quat.cuh"

namespace rocketsim_cuda {

// 3x3 Column-major rotation matrix matching RocketSim RotMat (forward, right, up)
struct Mat3 {
    Vec3 forward = Vec3(1.0f, 0.0f, 0.0f); // Column 0: X axis
    Vec3 right   = Vec3(0.0f, 1.0f, 0.0f); // Column 1: Y axis
    Vec3 up      = Vec3(0.0f, 0.0f, 1.0f); // Column 2: Z axis

    __host__ __device__ constexpr Mat3()
        : forward(1.0f, 0.0f, 0.0f), right(0.0f, 1.0f, 0.0f), up(0.0f, 0.0f, 1.0f) {}

    __host__ __device__ constexpr Mat3(const Vec3& f, const Vec3& r, const Vec3& u)
        : forward(f), right(r), up(u) {}

    static __host__ __device__ constexpr Mat3 identity() {
        return Mat3(Vec3(1.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f));
    }

    __host__ __device__ Vec3 operator*(const Vec3& v) const {
        return Vec3(
            forward.x * v.x + right.x * v.y + up.x * v.z,
            forward.y * v.x + right.y * v.y + up.y * v.z,
            forward.z * v.x + right.z * v.y + up.z * v.z
        );
    }

    __host__ __device__ Mat3 operator*(const Mat3& o) const {
        return Mat3(
            *this * o.forward,
            *this * o.right,
            *this * o.up
        );
    }

    __host__ __device__ Mat3 transpose() const {
        return Mat3(
            Vec3(forward.x, right.x, up.x),
            Vec3(forward.y, right.y, up.y),
            Vec3(forward.z, right.z, up.z)
        );
    }

    __host__ __device__ static Mat3 from_quat(const Quat& q) {
        float s = 2.0f;
        float xs = q.x * s, ys = q.y * s, zs = q.z * s;
        float wx = q.w * xs, wy = q.w * ys, wz = q.w * zs;
        float xx = q.x * xs, xy = q.x * ys, xz = q.x * zs;
        float yy = q.y * ys, yz = q.y * zs, zz = q.z * zs;

        return Mat3(
            Vec3(1.0f - (yy + zz), xy + wz, xz - wy),
            Vec3(xy - wz, 1.0f - (xx + zz), yz + wx),
            Vec3(xz + wy, yz - wx, 1.0f - (xx + yy))
        );
    }

    __host__ __device__ Quat to_quat() const {
        float trace = forward.x + right.y + up.z;
        if (trace > 0.0f) {
            float s = 0.5f / sqrtf(trace + 1.0f);
            return Quat(0.25f / s, (right.z - up.y) * s, (up.x - forward.z) * s, (forward.y - right.x) * s);
        } else if (forward.x > right.y && forward.x > up.z) {
            float s = 2.0f * sqrtf(1.0f + forward.x - right.y - up.z);
            return Quat((right.z - up.y) / s, 0.25f * s, (forward.y + right.x) / s, (forward.z + up.x) / s);
        } else if (right.y > up.z) {
            float s = 2.0f * sqrtf(1.0f + right.y - forward.x - up.z);
            return Quat((up.x - forward.z) / s, (forward.y + right.x) / s, 0.25f * s, (right.z + up.y) / s);
        } else {
            float s = 2.0f * sqrtf(1.0f + up.z - forward.x - right.y);
            return Quat((forward.y - right.x) / s, (forward.z + up.x) / s, (right.z + up.y) / s, 0.25f * s);
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, const Mat3& m) {
    os << "Mat3[fwd=" << m.forward << ", rgt=" << m.right << ", up=" << m.up << "]";
    return os;
}

} // namespace rocketsim_cuda
