#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"

namespace rocketsim_cuda {

// Soccar Arena Analytical Constants
constexpr float SDF_ARENA_EXTENT_X = 4096.0f;
constexpr float SDF_ARENA_EXTENT_Y = 5120.0f;
constexpr float SDF_ARENA_HEIGHT   = 2048.0f;
constexpr float SDF_RAMP_RADIUS    = 260.0f;

// Corner Chamfer: connects (2944, 5120) to (4096, 3968) with X + Y = 8064
constexpr float SDF_CORNER_X0      = 2944.0f;
constexpr float SDF_CORNER_Y0      = 5120.0f;
constexpr float SDF_CORNER_X1      = 4096.0f;
constexpr float SDF_CORNER_Y1      = 3968.0f;
constexpr float SDF_CORNER_SUM     = 8064.0f; // 2944 + 5120 = 4096 + 3968
constexpr float SDF_INV_SQRT2      = 0.7071067811865475f;

// Goal Box Dimensions
constexpr float SDF_GOAL_HALF_WIDTH = 892.8f;
constexpr float SDF_GOAL_HEIGHT     = 642.7f;
constexpr float SDF_GOAL_DEPTH      = 880.0f;
constexpr float SDF_GOAL_Y_BACK     = 6000.0f; // 5120 + 880

/**
 * @brief Computes 2D distance to arena horizontal boundaries in the first quadrant.
 * 
 * @param x Absolute X coordinate (>= 0)
 * @param y Absolute Y coordinate (>= 0)
 * @param out_nx Inward normal X in first quadrant
 * @param out_ny Inward normal Y in first quadrant
 * @return Signed distance in 2D (positive = inside, negative = outside)
 */
__device__ __forceinline__ float arena_sdf_2d_wall(
    float x, float y,
    float* __restrict__ out_nx, float* __restrict__ out_ny)
{
    float d_side = SDF_ARENA_EXTENT_X - x;
    float d_back = SDF_ARENA_EXTENT_Y - y;
    float d_chamfer = (SDF_CORNER_SUM - (x + y)) * SDF_INV_SQRT2;

    if (d_side <= d_back && d_side <= d_chamfer) {
        *out_nx = -1.0f;
        *out_ny = 0.0f;
        return d_side;
    } else if (d_back <= d_chamfer) {
        *out_nx = 0.0f;
        *out_ny = -1.0f;
        return d_back;
    } else {
        *out_nx = -SDF_INV_SQRT2;
        *out_ny = -SDF_INV_SQRT2;
        return d_chamfer;
    }
}

/**
 * @brief Computes analytical Signed Distance and Surface Normal for Soccar Arena.
 * 
 * @param p 3D query point in world space
 * @param out_dist Output signed distance (> 0 inside, < 0 outside)
 * @param out_normal Output unit inward surface normal pointing into arena
 */
__device__ __forceinline__ void arena_sdf_and_normal(
    const Vec3& p,
    float& out_dist,
    Vec3& out_normal)
{
    float sx = copysignf(1.0f, p.x);
    float sy = copysignf(1.0f, p.y);
    float x_abs = fabsf(p.x);
    float y_abs = fabsf(p.y);
    float z = p.z;

    // 1. Goal Cavity Interior Check (y_abs >= 5120, x_abs <= 892.8, z <= 642.7)
    if (y_abs >= SDF_ARENA_EXTENT_Y && x_abs <= SDF_GOAL_HALF_WIDTH && z <= SDF_GOAL_HEIGHT) {
        float d_floor = z;
        float d_ceil  = SDF_GOAL_HEIGHT - z;
        float d_side  = SDF_GOAL_HALF_WIDTH - x_abs;
        float d_back  = SDF_GOAL_Y_BACK - y_abs;

        float d_min = fminf(fminf(d_floor, d_ceil), fminf(d_side, d_back));
        out_dist = d_min;

        if (d_min == d_floor) {
            out_normal = Vec3(0.0f, 0.0f, 1.0f);
        } else if (d_min == d_ceil) {
            out_normal = Vec3(0.0f, 0.0f, -1.0f);
        } else if (d_min == d_side) {
            out_normal = Vec3(-sx, 0.0f, 0.0f);
        } else {
            out_normal = Vec3(0.0f, -sy, 0.0f);
        }
        return;
    }

    // 2. Goalpost / Crossbar External Corner Proximity
    if (y_abs > SDF_ARENA_EXTENT_Y) {
        if (x_abs > SDF_GOAL_HALF_WIDTH && z <= SDF_GOAL_HEIGHT) {
            // Vertical Goalpost Rim at (892.8, 5120)
            float dx = x_abs - SDF_GOAL_HALF_WIDTH;
            float dy = y_abs - SDF_ARENA_EXTENT_Y;
            float r = sqrtf(dx * dx + dy * dy);
            out_dist = -r;
            float inv = (r > 1e-6f) ? (1.0f / r) : 1.0f;
            out_normal = Vec3(-sx * dx * inv, -sy * dy * inv, 0.0f);
            return;
        } else if (z > SDF_GOAL_HEIGHT && x_abs <= SDF_GOAL_HALF_WIDTH) {
            // Horizontal Crossbar Rim at (y = 5120, z = 642.7)
            float dy = y_abs - SDF_ARENA_EXTENT_Y;
            float dz = z - SDF_GOAL_HEIGHT;
            float r = sqrtf(dy * dy + dz * dz);
            out_dist = -r;
            float inv = (r > 1e-6f) ? (1.0f / r) : 1.0f;
            out_normal = Vec3(0.0f, -sy * dy * inv, -dz * inv);
            return;
        }
    }

    // 3. 2D Wall Distance and Normal in First Quadrant
    float nx_quad = 0.0f;
    float ny_quad = 0.0f;
    float d_wall = arena_sdf_2d_wall(x_abs, y_abs, &nx_quad, &ny_quad);

    // 4. Vertical Z Splitting: Floor Ramp, Ceiling Ramp, or Open Height
    float R = SDF_RAMP_RADIUS;
    float H = SDF_ARENA_HEIGHT;

    if (z < R) {
        // Floor transition zone
        if (d_wall >= R) {
            // Pure flat floor
            out_dist = z;
            out_normal = Vec3(0.0f, 0.0f, 1.0f);
        } else {
            // Bottom cylindrical fillet
            float delta_h = R - d_wall;
            float delta_z = R - z;
            float rho = sqrtf(delta_h * delta_h + delta_z * delta_z);
            out_dist = R - rho;

            float inv_rho = (rho > 1e-6f) ? (1.0f / rho) : 1.0f;
            float scale_h = delta_h * inv_rho;
            float scale_z = delta_z * inv_rho;

            out_normal = Vec3(
                scale_h * nx_quad * sx,
                scale_h * ny_quad * sy,
                scale_z
            );
        }
    } else if (z > H - R) {
        // Ceiling transition zone
        float z_from_top = H - z;
        if (d_wall >= R) {
            // Pure flat ceiling
            out_dist = z_from_top;
            out_normal = Vec3(0.0f, 0.0f, -1.0f);
        } else {
            // Top cylindrical fillet
            float delta_h = R - d_wall;
            float delta_z = R - z_from_top;
            float rho = sqrtf(delta_h * delta_h + delta_z * delta_z);
            out_dist = R - rho;

            float inv_rho = (rho > 1e-6f) ? (1.0f / rho) : 1.0f;
            float scale_h = delta_h * inv_rho;
            float scale_z = delta_z * inv_rho;

            out_normal = Vec3(
                scale_h * nx_quad * sx,
                scale_h * ny_quad * sy,
                -scale_z
            );
        }
    } else {
        // Central height: pure vertical walls or top/bottom bounds
        float z_ceil = H - z;
        float d_min = fminf(d_wall, fminf(z, z_ceil));
        out_dist = d_min;

        if (d_min == z) {
            out_normal = Vec3(0.0f, 0.0f, 1.0f);
        } else if (d_min == z_ceil) {
            out_normal = Vec3(0.0f, 0.0f, -1.0f);
        } else {
            out_normal = Vec3(nx_quad * sx, ny_quad * sy, 0.0f);
        }
    }
}

/**
 * @brief Computes analytical Signed Distance for Soccar Arena.
 */
__device__ __forceinline__ float arena_sdf(const Vec3& p) {
    float dist;
    Vec3 normal;
    arena_sdf_and_normal(p, dist, normal);
    return dist;
}

/**
 * @brief Computes analytical Inward Unit Normal for Soccar Arena.
 */
__device__ __forceinline__ Vec3 arena_normal(const Vec3& p) {
    float dist;
    Vec3 normal;
    arena_sdf_and_normal(p, dist, normal);
    return normal;
}

/**
 * @brief Performs analytical sphere tracing raycast against Soccar Arena SDF.
 * 
 * Used for vehicle suspension wheel queries (casting down from wheel hardpoints).
 * 
 * @param origin Ray start point in world coordinates
 * @param dir Normalized ray direction
 * @param max_dist Maximum raycast distance
 * @param hit_dist Output hit distance along ray
 * @param hit_normal Output surface normal at hit point
 * @return true if ray intersects arena surface within max_dist
 */
__device__ __forceinline__ bool raycast_arena_sdf(
    const Vec3& origin,
    const Vec3& dir,
    float max_dist,
    float* __restrict__ hit_dist,
    Vec3* __restrict__ hit_normal)
{
    // Fast-path for flat floor (standard case when driving on field away from ramps)
    if (origin.z < 80.0f && dir.z < -0.2f &&
        fabsf(origin.x) < (SDF_ARENA_EXTENT_X - 300.0f) &&
        fabsf(origin.y) < (SDF_ARENA_EXTENT_Y - 300.0f))
    {
        float t_floor = -origin.z / dir.z;
        if (t_floor >= 0.0f && t_floor <= max_dist) {
            *hit_dist = t_floor;
            *hit_normal = Vec3(0.0f, 0.0f, 1.0f);
            return true;
        }
    }

    // Sphere tracing for general surfaces (ramps, chamfers, walls, goalposts)
    float t = 0.0f;
    #pragma unroll
    for (int iter = 0; iter < 10; ++iter) {
        Vec3 p = origin + dir * t;
        float dist;
        Vec3 normal;
        arena_sdf_and_normal(p, dist, normal);

        if (dist <= 0.05f) {
            float final_t = t + dist;
            if (final_t <= max_dist) {
                *hit_dist = final_t;
                *hit_normal = arena_normal(origin + dir * final_t);
                return true;
            }
            return false;
        }

        t += dist;
        if (t > max_dist) {
            break;
        }
    }

    return false;
}

} // namespace rocketsim_cuda
