#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace rocketsim_cuda {

struct alignas(4) Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    __host__ __device__ constexpr Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ constexpr Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ constexpr Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    __host__ __device__ constexpr Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    __host__ __device__ constexpr Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __host__ __device__ constexpr Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }
    __host__ __device__ constexpr Vec3 operator-() const { return Vec3(-x, -y, -z); }

    __host__ __device__ constexpr Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    __host__ __device__ constexpr Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    __host__ __device__ constexpr Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    __host__ __device__ constexpr Vec3& operator/=(float s) { x /= s; y /= s; z /= s; return *this; }

    __host__ __device__ constexpr bool operator==(const Vec3& o) const { return x == o.x && y == o.y && z == o.z; }
    __host__ __device__ constexpr bool operator!=(const Vec3& o) const { return !(*this == o); }

    __host__ __device__ constexpr float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ constexpr Vec3 cross(const Vec3& o) const {
        return Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
    }

    __host__ __device__ float length_sq() const { return dot(*this); }
    __host__ __device__ float length() const { return sqrtf(length_sq()); }

    __host__ __device__ Vec3 normalized(float eps = 1e-8f) const {
        float lsq = length_sq();
        if (lsq > eps) {
            float inv = 1.0f / sqrtf(lsq);
            return Vec3(x * inv, y * inv, z * inv);
        }
        return Vec3(0.0f, 0.0f, 0.0f);
    }

    __host__ __device__ float dist(const Vec3& o) const { return (*this - o).length(); }

    __host__ __device__ float chebyshev_dist(const Vec3& o) const {
        float dx = fabsf(x - o.x);
        float dy = fabsf(y - o.y);
        float dz = fabsf(z - o.z);
        return fmaxf(dx, fmaxf(dy, dz));
    }
};

__host__ __device__ inline constexpr Vec3 operator*(float s, const Vec3& v) {
    return Vec3(s * v.x, s * v.y, s * v.z);
}

inline std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

} // namespace rocketsim_cuda
