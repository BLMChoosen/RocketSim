#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/mat3.cuh"
#include "rocketsim_cuda/types/car_state.cuh"
#include "rocketsim_cuda/types/arena_state.cuh"
#include "rocketsim_cuda/types/car_config.cuh"

namespace rocketsim_cuda {

// ============================================================================
// Physical Constants for Car-Car Collision & Bump (src/RLConst.h)
// ============================================================================

// Restitution & Friction for car-car contact (RLConst.h:40-41)
constexpr float CARCAR_COLLISION_FRICTION = 0.09f;
constexpr float CARCAR_COLLISION_RESTITUTION = 0.1f;

// Bump and Demo Thresholds (RLConst.h:144-146)
constexpr float BUMP_COOLDOWN_TIME = 0.25f;
constexpr float BUMP_MIN_FORWARD_DIST = 64.5f; // Unreal Units along car forward axis
constexpr float DEMO_RESPAWN_TIME = 3.0f;

// Restitution velocity threshold (Bullet BT units 0.2 = 10.0 UU/s)
constexpr float CARCAR_RESTITUTION_THRESHOLD = 10.0f;

// Default Octane Hitbox Extents & Offsets (CarConfig.cpp)
__device__ __host__ __forceinline__ Vec3 get_car_hitbox_offset_default() {
    return Vec3(13.8757f, 0.0f, 20.755f);
}

__device__ __host__ __forceinline__ Vec3 get_car_hitbox_half_default() {
    return Vec3(60.2535f, 43.3497f, 19.32955f);
}

__device__ __host__ __forceinline__ Vec3 get_car_inv_inertia_local_default() {
    return Vec3(1.0f / 135169.6f, 1.0f / 240248.8f, 1.0f / 330582.4f);
}

// ============================================================================
// Bump Velocity Response Curves (RLConst.h:505-527 & Math.cpp:5-35)
// ============================================================================

/**
 * @brief Evaluates BUMP_VEL_AMOUNT_GROUND_CURVE (RLConst.h:505-511).
 * Points: (0, 5/6), (1400, 1100), (2200, 1530).
 */
__device__ __host__ __forceinline__ float evaluate_bump_vel_ground(float speed) {
    constexpr float Y0 = 5.0f / 6.0f;
    constexpr float Y1 = 1100.0f;
    constexpr float Y2 = 1530.0f;

    if (speed <= 0.0f) {
        return Y0;
    } else if (speed <= 1400.0f) {
        float t = speed / 1400.0f;
        return Y0 + t * (Y1 - Y0);
    } else if (speed <= 2200.0f) {
        float t = (speed - 1400.0f) / 800.0f;
        return Y1 + t * (Y2 - Y1);
    } else {
        return Y2;
    }
}

/**
 * @brief Evaluates BUMP_VEL_AMOUNT_AIR_CURVE (RLConst.h:513-519).
 * Points: (0, 5/6), (1400, 1390), (2200, 1945).
 */
__device__ __host__ __forceinline__ float evaluate_bump_vel_air(float speed) {
    constexpr float Y0 = 5.0f / 6.0f;
    constexpr float Y1 = 1390.0f;
    constexpr float Y2 = 1945.0f;

    if (speed <= 0.0f) {
        return Y0;
    } else if (speed <= 1400.0f) {
        float t = speed / 1400.0f;
        return Y0 + t * (Y1 - Y0);
    } else if (speed <= 2200.0f) {
        float t = (speed - 1400.0f) / 800.0f;
        return Y1 + t * (Y2 - Y1);
    } else {
        return Y2;
    }
}

/**
 * @brief Evaluates BUMP_UPWARD_VEL_AMOUNT_CURVE (RLConst.h:521-527).
 * Points: (0, 2/6), (1400, 278), (2200, 417).
 */
__device__ __host__ __forceinline__ float evaluate_bump_upward_vel(float speed) {
    constexpr float Y0 = 2.0f / 6.0f;
    constexpr float Y1 = 278.0f;
    constexpr float Y2 = 417.0f;

    if (speed <= 0.0f) {
        return Y0;
    } else if (speed <= 1400.0f) {
        float t = speed / 1400.0f;
        return Y0 + t * (Y1 - Y0);
    } else if (speed <= 2200.0f) {
        float t = (speed - 1400.0f) / 800.0f;
        return Y1 + t * (Y2 - Y1);
    } else {
        return Y2;
    }
}

// ============================================================================
// Contact Manifold Structures (Fixed Size, Zero Dynamic Allocation)
// ============================================================================

struct CarContactPoint {
    Vec3 point_world;    // World position of contact point on Car B
    Vec3 point_local_a;  // Contact point in Car A chassis frame (UU)
    Vec3 point_local_b;  // Contact point in Car B chassis frame (UU)
    float depth = 0.0f;  // Penetration depth along normal (> 0 when intersecting)
};

struct CarContactManifold {
    Vec3 normal_world = Vec3(0.0f, 0.0f, 1.0f); // Normal pointing from B towards A
    CarContactPoint points[4];                   // Up to 4 contact points
    int num_points = 0;                          // Number of valid contact points
};

// ============================================================================
// Box-Box Detector Geometry Helpers (Port of ODE / Bullet btBoxBoxDetector.cpp)
// ============================================================================

/**
 * @brief Finds closest approach between two infinite 3D lines.
 * Mirrors dLineClosestApproach (btBoxBoxDetector.cpp:80-107).
 */
__device__ __host__ __forceinline__ void d_line_closest_approach(
    const Vec3& pa, const Vec3& ua,
    const Vec3& pb, const Vec3& ub,
    float& alpha, float& beta)
{
    Vec3 p = pb - pa;
    float uaub = ua.dot(ub);
    float q1 = ua.dot(p);
    float q2 = -ub.dot(p);
    float d = 1.0f - uaub * uaub;
    if (d <= 0.0001f) {
        alpha = 0.0f;
        beta = 0.0f;
    } else {
        float inv_d = 1.0f / d;
        alpha = (q1 + uaub * q2) * inv_d;
        beta = (uaub * q1 + q2) * inv_d;
    }
}

/**
 * @brief Sutherland-Hodgman 2D rectangle vs quadrilateral clipping.
 * Mirrors intersectRectQuad2 (btBoxBoxDetector.cpp:117-175).
 */
__device__ __host__ __forceinline__ int d_intersect_rect_quad2(
    const float h[2], const float p[8], float ret[16])
{
    int nq = 4, nr = 0;
    float buffer[16];
    const float* q = p;
    float* r = ret;

    for (int dir = 0; dir <= 1; dir++) {
        for (int sign = -1; sign <= 1; sign += 2) {
            const float* pq = q;
            float* pr = r;
            nr = 0;
            for (int i = nq; i > 0; i--) {
                if (sign * pq[dir] < h[dir]) {
                    pr[0] = pq[0];
                    pr[1] = pq[1];
                    pr += 2;
                    nr++;
                    if (nr & 8) {
                        q = r;
                        goto done;
                    }
                }
                const float* nextq = (i > 1) ? (pq + 2) : q;
                if ((sign * pq[dir] < h[dir]) ^ (sign * nextq[dir] < h[dir])) {
                    pr[1 - dir] = pq[1 - dir] + (nextq[1 - dir] - pq[1 - dir]) /
                                  (nextq[dir] - pq[dir]) * (sign * h[dir] - pq[dir]);
                    pr[dir] = (float)sign * h[dir];
                    pr += 2;
                    nr++;
                    if (nr & 8) {
                        q = r;
                        goto done;
                    }
                }
                pq += 2;
            }
            q = r;
            r = (q == ret) ? buffer : ret;
            nq = nr;
        }
    }

done:
    if (q != ret) {
        for (int i = 0; i < nr * 2; ++i) {
            ret[i] = q[i];
        }
    }
    return nr;
}

/**
 * @brief Polygon point reduction to m representative points.
 * Mirrors cullPoints2 (btBoxBoxDetector.cpp:188-265).
 */
__device__ __host__ __forceinline__ void d_cull_points2(
    int n, const float p[], int m, int i0, int iret[])
{
    float cx = 0.0f, cy = 0.0f;
    if (n == 1) {
        cx = p[0];
        cy = p[1];
    } else if (n == 2) {
        cx = 0.5f * (p[0] + p[2]);
        cy = 0.5f * (p[1] + p[3]);
    } else {
        float a = 0.0f;
        for (int i = 0; i < (n - 1); i++) {
            float q = p[i * 2] * p[i * 2 + 3] - p[i * 2 + 2] * p[i * 2 + 1];
            a += q;
            cx += q * (p[i * 2] + p[i * 2 + 2]);
            cy += q * (p[i * 2 + 1] + p[i * 2 + 3]);
        }
        float q = p[n * 2 - 2] * p[1] - p[0] * p[n * 2 - 1];
        if (fabsf(a + q) > 1e-6f) {
            a = 1.0f / (3.0f * (a + q));
        } else {
            a = 1e30f;
        }
        cx = a * (cx + q * (p[n * 2 - 2] + p[0]));
        cy = a * (cy + q * (p[n * 2 - 1] + p[1]));
    }

    float angles[8];
    for (int i = 0; i < n; i++) {
        angles[i] = atan2f(p[i * 2 + 1] - cy, p[i * 2] - cx);
    }

    int avail[8];
    for (int i = 0; i < n; i++) avail[i] = 1;
    avail[i0] = 0;
    iret[0] = i0;

    constexpr float PI_CONST = 3.14159265358979323846f;
    for (int j = 1; j < m; j++) {
        float target_angle = (float)j * (2.0f * PI_CONST / (float)m) + angles[i0];
        if (target_angle > PI_CONST) target_angle -= 2.0f * PI_CONST;
        float max_diff = 1e9f;
        int best_idx = i0;

        for (int i = 0; i < n; i++) {
            if (avail[i]) {
                float diff = fabsf(angles[i] - target_angle);
                if (diff > PI_CONST) diff = 2.0f * PI_CONST - diff;
                if (diff < max_diff) {
                    max_diff = diff;
                    best_idx = i;
                }
            }
        }
        iret[j] = best_idx;
        avail[best_idx] = 0;
    }
}

// ============================================================================
// Box-Box Narrowphase Detector (Mirroring dBoxBox2 / btBoxBoxDetector.cpp)
// ============================================================================

/**
 * @brief Evaluates exact OBB-OBB separating axis test and manifold generation.
 * 
 * @param p1 Center of Box 1 in world coordinates.
 * @param basis1 3x3 orientation matrix of Box 1 (columns: forward, right, up).
 * @param half1 Half-extents of Box 1 (A0, A1, A2).
 * @param p2 Center of Box 2 in world coordinates.
 * @param basis2 3x3 orientation matrix of Box 2 (columns: forward, right, up).
 * @param half2 Half-extents of Box 2 (B0, B1, B2).
 * @param out_manifold Output contact manifold populated with contacts and normal.
 * @return true if boxes intersect, false otherwise.
 */
__device__ __host__ inline bool test_box_box_collision(
    const Vec3& p1,
    const Mat3& basis1,
    const Vec3& half1,
    const Vec3& p2,
    const Mat3& basis2,
    const Vec3& half2,
    CarContactManifold& out_manifold)
{
    out_manifold.num_points = 0;

    const Vec3 u[3] = { basis1.forward, basis1.right, basis1.up };
    const Vec3 v[3] = { basis2.forward, basis2.right, basis2.up };

    // Vector between centers
    Vec3 p = p2 - p1;

    // Vector p in Box 1 local frame: pp = basis1.transpose() * p
    float pp[3] = {
        p.dot(u[0]),
        p.dot(u[1]),
        p.dot(u[2])
    };

    // Half extents
    const float A[3] = { half1.x, half1.y, half1.z };
    const float B[3] = { half2.x, half2.y, half2.z };

    // Relative rotation matrix R_ij = u_i . v_j
    float R[3][3];
    float Q[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R[i][j] = u[i].dot(v[j]);
            Q[i][j] = fabsf(R[i][j]);
        }
    }

    float s = -1e30f;
    int invert_normal = 0;
    int code = 0;
    const Vec3* normal_face = nullptr;
    Vec3 normal_cross(0.0f, 0.0f, 0.0f);

    // Separating axes 1..3: u0, u1, u2 (Face normals of Box 1)
    for (int i = 0; i < 3; ++i) {
        float expr1 = pp[i];
        float expr2 = A[i] + B[0] * Q[i][0] + B[1] * Q[i][1] + B[2] * Q[i][2];
        float s2 = fabsf(expr1) - expr2;
        if (s2 > 0.0f) return false;
        if (s2 > s) {
            s = s2;
            normal_face = &u[i];
            invert_normal = (expr1 < 0.0f);
            code = i + 1;
        }
    }

    // Separating axes 4..6: v0, v1, v2 (Face normals of Box 2)
    for (int j = 0; j < 3; ++j) {
        float expr1 = p.dot(v[j]);
        float expr2 = A[0] * Q[0][j] + A[1] * Q[1][j] + A[2] * Q[2][j] + B[j];
        float s2 = fabsf(expr1) - expr2;
        if (s2 > 0.0f) return false;
        if (s2 > s) {
            s = s2;
            normal_face = &v[j];
            invert_normal = (expr1 < 0.0f);
            code = j + 4;
        }
    }

    // Add small epsilon fudge to Q before cross products (btBoxBoxDetector.cpp:375-388)
    constexpr float FUDGE2 = 1.0e-5f;
    constexpr float FUDGE_FACTOR = 1.05f;
    constexpr float SIMD_EPSILON = 1.0e-6f;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Q[i][j] += FUDGE2;
        }
    }

    // Lambda helper for testing edge-edge cross product axes
    auto test_cross_axis = [&](float expr1, float expr2, float n1, float n2, float n3, int cc) -> bool {
        float s2 = fabsf(expr1) - expr2;
        if (s2 > SIMD_EPSILON) return false; // Separated!
        float l = sqrtf(n1 * n1 + n2 * n2 + n3 * n3);
        if (l > SIMD_EPSILON) {
            s2 /= l;
            if (s2 * FUDGE_FACTOR > s) {
                s = s2;
                normal_face = nullptr;
                normal_cross = Vec3(n1 / l, n2 / l, n3 / l);
                invert_normal = (expr1 < 0.0f);
                code = cc;
            }
        }
        return true;
    };

    // Separating axis 7: u0 x v0
    if (!test_cross_axis(pp[2] * R[1][0] - pp[1] * R[2][0],
        A[1] * Q[2][0] + A[2] * Q[1][0] + B[1] * Q[0][2] + B[2] * Q[0][1],
        0.0f, -R[2][0], R[1][0], 7)) return false;

    // Separating axis 8: u0 x v1
    if (!test_cross_axis(pp[2] * R[1][1] - pp[1] * R[2][1],
        A[1] * Q[2][1] + A[2] * Q[1][1] + B[0] * Q[0][2] + B[2] * Q[0][0],
        0.0f, -R[2][1], R[1][1], 8)) return false;

    // Separating axis 9: u0 x v2
    if (!test_cross_axis(pp[2] * R[1][2] - pp[1] * R[2][2],
        A[1] * Q[2][2] + A[2] * Q[1][2] + B[0] * Q[0][1] + B[1] * Q[0][0],
        0.0f, -R[2][2], R[1][2], 9)) return false;

    // Separating axis 10: u1 x v0
    if (!test_cross_axis(pp[0] * R[2][0] - pp[2] * R[0][0],
        A[0] * Q[2][0] + A[2] * Q[0][0] + B[1] * Q[1][2] + B[2] * Q[1][1],
        R[2][0], 0.0f, -R[0][0], 10)) return false;

    // Separating axis 11: u1 x v1
    if (!test_cross_axis(pp[0] * R[2][1] - pp[2] * R[0][1],
        A[0] * Q[2][1] + A[2] * Q[0][1] + B[0] * Q[1][2] + B[2] * Q[1][0],
        R[2][1], 0.0f, -R[0][1], 11)) return false;

    // Separating axis 12: u1 x v2
    if (!test_cross_axis(pp[0] * R[2][2] - pp[2] * R[0][2],
        A[0] * Q[2][2] + A[2] * Q[0][2] + B[0] * Q[1][1] + B[1] * Q[1][0],
        R[2][2], 0.0f, -R[0][2], 12)) return false;

    // Separating axis 13: u2 x v0
    if (!test_cross_axis(pp[1] * R[0][0] - pp[0] * R[1][0],
        A[0] * Q[1][0] + A[1] * Q[0][0] + B[1] * Q[2][2] + B[2] * Q[2][1],
        -R[1][0], R[0][0], 0.0f, 13)) return false;

    // Separating axis 14: u2 x v1
    if (!test_cross_axis(pp[1] * R[0][1] - pp[0] * R[1][1],
        A[0] * Q[1][1] + A[1] * Q[0][1] + B[0] * Q[2][2] + B[2] * Q[2][0],
        -R[1][1], R[0][1], 0.0f, 14)) return false;

    // Separating axis 15: u2 x v2
    if (!test_cross_axis(pp[1] * R[0][2] - pp[0] * R[1][2],
        A[0] * Q[1][2] + A[1] * Q[0][2] + B[0] * Q[2][1] + B[1] * Q[2][0],
        -R[1][2], R[0][2], 0.0f, 15)) return false;

    if (code == 0) return false;

    // Compute normal in world coordinates (points from Box 1 towards Box 2)
    Vec3 normal;
    if (normal_face) {
        normal = *normal_face;
    } else {
        normal = u[0] * normal_cross.x + u[1] * normal_cross.y + u[2] * normal_cross.z;
    }

    if (invert_normal) {
        normal = normal * (-1.0f);
    }

    float depth = -s;

    // Output normal points from Box 2 toward Box 1 (matching Bullet normalOnBInWorld)
    out_manifold.normal_world = normal * (-1.0f);

    // ========================================================================
    // Contact Point Generation
    // ========================================================================

    if (code > 6) {
        // Edge-Edge Intersection (btBoxBoxDetector.cpp:430-477)
        Vec3 pa = p1;
        for (int j = 0; j < 3; j++) {
            float sign = (normal.dot(u[j]) > 0.0f) ? 1.0f : -1.0f;
            pa = pa + u[j] * (sign * A[j]);
        }

        Vec3 pb = p2;
        for (int j = 0; j < 3; j++) {
            float sign = (normal.dot(v[j]) > 0.0f) ? -1.0f : 1.0f;
            pb = pb + v[j] * (sign * B[j]);
        }

        Vec3 ua = u[(code - 7) / 3];
        Vec3 ub = v[(code - 7) % 3];

        float alpha, beta;
        d_line_closest_approach(pa, ua, pb, ub, alpha, beta);
        pa = pa + ua * alpha;
        pb = pb + ub * beta;

        CarContactPoint& cp = out_manifold.points[0];
        cp.point_world = pb;
        cp.point_local_a = basis1.transpose() * (pb + normal * depth - p1);
        cp.point_local_b = basis2.transpose() * (pb - p2);
        cp.depth = depth;

        out_manifold.num_points = 1;
        return true;
    }

    // Face-Something Intersection (btBoxBoxDetector.cpp:484-686)
    const Vec3* Ra = (code <= 3) ? u : v;
    const Vec3* Rb = (code <= 3) ? v : u;
    Vec3 pa = (code <= 3) ? p1 : p2;
    Vec3 pb = (code <= 3) ? p2 : p1;
    const float* Sa = (code <= 3) ? A : B;
    const float* Sb = (code <= 3) ? B : A;

    Vec3 normal2 = (code <= 3) ? normal : (normal * (-1.0f));

    // Normal vector of reference face dotted with axes of incident box
    float nr[3] = {
        normal2.dot(Rb[0]),
        normal2.dot(Rb[1]),
        normal2.dot(Rb[2])
    };
    float anr[3] = { fabsf(nr[0]), fabsf(nr[1]), fabsf(nr[2]) };

    int lanr, a1, a2;
    if (anr[1] > anr[0]) {
        if (anr[1] > anr[2]) {
            a1 = 0; lanr = 1; a2 = 2;
        } else {
            a1 = 0; a2 = 1; lanr = 2;
        }
    } else {
        if (anr[0] > anr[2]) {
            lanr = 0; a1 = 1; a2 = 2;
        } else {
            a1 = 0; a2 = 1; lanr = 2;
        }
    }

    // Center point of incident face in reference coordinates
    Vec3 center = pb - pa + Rb[lanr] * ((nr[lanr] < 0.0f ? 1.0f : -1.0f) * Sb[lanr]);

    int codeN = (code <= 3) ? (code - 1) : (code - 4);
    int code1, code2;
    if (codeN == 0) {
        code1 = 1; code2 = 2;
    } else if (codeN == 1) {
        code1 = 0; code2 = 2;
    } else {
        code1 = 0; code2 = 1;
    }

    // Project incident face 4 corners to 2D
    float c1 = center.dot(Ra[code1]);
    float c2 = center.dot(Ra[code2]);
    float m11 = Ra[code1].dot(Rb[a1]);
    float m12 = Ra[code1].dot(Rb[a2]);
    float m21 = Ra[code2].dot(Rb[a1]);
    float m22 = Ra[code2].dot(Rb[a2]);

    float k1 = m11 * Sb[a1];
    float k2 = m21 * Sb[a1];
    float k3 = m12 * Sb[a2];
    float k4 = m22 * Sb[a2];

    float quad[8];
    quad[0] = c1 - k1 - k3; quad[1] = c2 - k2 - k4;
    quad[2] = c1 - k1 + k3; quad[3] = c2 - k2 + k4;
    quad[4] = c1 + k1 + k3; quad[5] = c2 + k2 + k4;
    quad[6] = c1 + k1 - k3; quad[7] = c2 + k2 - k4;

    float rect[2] = { Sa[code1], Sa[code2] };
    float ret[16];
    int n = d_intersect_rect_quad2(rect, quad, ret);
    if (n < 1) return false;

    float det = m11 * m22 - m12 * m21;
    if (fabsf(det) < 1e-8f) return false;
    float det1 = 1.0f / det;
    m11 *= det1; m12 *= det1; m21 *= det1; m22 *= det1;

    float point[24];
    float dep[8];
    int cnum = 0;

    for (int j = 0; j < n; j++) {
        float unp_k1 = m22 * (ret[j * 2] - c1) - m12 * (ret[j * 2 + 1] - c2);
        float unp_k2 = -m21 * (ret[j * 2] - c1) + m11 * (ret[j * 2 + 1] - c2);
        Vec3 pt = center + Rb[a1] * unp_k1 + Rb[a2] * unp_k2;
        float d_val = Sa[codeN] - normal2.dot(pt);
        if (d_val >= 0.0f) {
            point[cnum * 3 + 0] = pt.x;
            point[cnum * 3 + 1] = pt.y;
            point[cnum * 3 + 2] = pt.z;
            dep[cnum] = d_val;
            ret[cnum * 2 + 0] = ret[j * 2 + 0];
            ret[cnum * 2 + 1] = ret[j * 2 + 1];
            cnum++;
        }
    }

    if (cnum < 1) return false;

    constexpr int MAXC = 4;
    int target_contacts = (cnum > MAXC) ? MAXC : cnum;
    int iret[8];

    if (cnum <= MAXC) {
        for (int j = 0; j < cnum; j++) iret[j] = j;
    } else {
        int i1 = 0;
        float max_depth = dep[0];
        for (int i = 1; i < cnum; i++) {
            if (dep[i] > max_depth) {
                max_depth = dep[i];
                i1 = i;
            }
        }
        d_cull_points2(cnum, ret, target_contacts, i1, iret);
    }

    for (int j = 0; j < target_contacts; j++) {
        int idx = iret[j];
        Vec3 pt(point[idx * 3 + 0], point[idx * 3 + 1], point[idx * 3 + 2]);
        Vec3 pos_in_world = pt + pa;
        if (code >= 4) {
            pos_in_world = pos_in_world - normal * dep[idx];
        }

        CarContactPoint& cp = out_manifold.points[j];
        cp.point_world = pos_in_world;
        cp.depth = dep[idx];

        Vec3 pos_on_a = pos_in_world + normal * dep[idx];
        Vec3 pos_on_b = pos_in_world;

        cp.point_local_a = basis1.transpose() * (pos_on_a - p1);
        cp.point_local_b = basis2.transpose() * (pos_on_b - p2);
    }

    out_manifold.num_points = target_contacts;
    return true;
}

/**
 * @brief High-level Car-Car OBB collision detector.
 * Accounts for car center of mass vs hitbox offsets.
 */
__device__ __host__ inline bool test_car_car_collision_obb(
    const Vec3& pos_a, const Mat3& basis_a, const Vec3& hitbox_offset_a, const Vec3& hitbox_half_a,
    const Vec3& pos_b, const Mat3& basis_b, const Vec3& hitbox_offset_b, const Vec3& hitbox_half_b,
    CarContactManifold& out_manifold)
{
    Vec3 center_a = pos_a + basis_a * hitbox_offset_a;
    Vec3 center_b = pos_b + basis_b * hitbox_offset_b;

    return test_box_box_collision(
        center_a, basis_a, hitbox_half_a,
        center_b, basis_b, hitbox_half_b,
        out_manifold
    );
}

// Default overload using Octane hitbox
__device__ __host__ inline bool test_car_car_collision_obb(
    const Vec3& pos_a, const Mat3& basis_a,
    const Vec3& pos_b, const Mat3& basis_b,
    CarContactManifold& out_manifold)
{
    Vec3 offset = get_car_hitbox_offset_default();
    Vec3 half = get_car_hitbox_half_default();
    return test_car_car_collision_obb(
        pos_a, basis_a, offset, half,
        pos_b, basis_b, offset, half,
        out_manifold
    );
}

// ============================================================================
// Bump Callback Logic (src/Sim/Arena/Arena.cpp:323-405)
// ============================================================================

enum class CarBumpType {
    NONE = 0,
    BUMP = 1,
    DEMO = 2
};

struct BumpEvaluation {
    CarBumpType type = CarBumpType::NONE;
    Vec3 impulse = Vec3(0.0f, 0.0f, 0.0f); // Velocity impulse to apply to target car
};

/**
 * @brief Evaluates whether car1 bumps or demolishes car2.
 * Mirrors Arena::_BtCallback_OnCarCarCollision (Arena.cpp:323-405).
 */
__device__ __host__ inline BumpEvaluation evaluate_single_car_bump(
    const Vec3& pos1, const Vec3& vel1, const Mat3& basis1,
    uint8_t is_supersonic1, uint8_t team1, int32_t last_other_id1, float cooldown_timer1,
    const Vec3& pos2, const Vec3& vel2, const Mat3& basis2,
    uint8_t target_is_on_ground, uint8_t team2, int32_t id2,
    const Vec3& local_contact_point1,
    int demo_mode = 0, // 0 = NORMAL, 1 = ON_CONTACT, 2 = DISABLED
    bool enable_team_demos = false,
    float bump_force_scale = 1.0f)
{
    BumpEvaluation eval;

    // Cooldown check (Arena.cpp:344-345)
    if ((last_other_id1 == id2) && (cooldown_timer1 > 0.0f)) {
        return eval;
    }

    // Moving towards other car? (Arena.cpp:348)
    Vec3 delta_pos = pos2 - pos1;
    if (vel1.dot(delta_pos) <= 0.0f) {
        return eval;
    }

    float vel1_speed = vel1.length();
    float delta_dist = delta_pos.length();
    if (vel1_speed < 1e-4f || delta_dist < 1e-4f) {
        return eval;
    }

    Vec3 vel_dir = vel1 * (1.0f / vel1_speed);
    Vec3 dir_to_other = delta_pos * (1.0f / delta_dist);

    float speed_towards = vel1.dot(dir_to_other);
    float other_away_speed = vel2.dot(vel_dir);

    // Going towards other car faster than they are going away? (Arena.cpp:356)
    if (speed_towards <= other_away_speed) {
        return eval;
    }

    // Hit with front bumper? (Arena.cpp:359-360)
    bool hit_with_bumper = (local_contact_point1.x > BUMP_MIN_FORWARD_DIST);
    if (!hit_with_bumper) {
        return eval;
    }

    // Determine Demo vs Bump (Arena.cpp:362-376)
    bool is_demo = false;
    switch (demo_mode) {
    case 1: // ON_CONTACT
        is_demo = true;
        break;
    case 2: // DISABLED
        is_demo = false;
        break;
    default: // NORMAL
        is_demo = (is_supersonic1 != 0);
        break;
    }

    if (is_demo && !enable_team_demos) {
        is_demo = (team1 != team2);
    }

    if (is_demo) {
        eval.type = CarBumpType::DEMO;
        return eval;
    }

    // Non-demo bump impulse calculation (Arena.cpp:380-394)
    eval.type = CarBumpType::BUMP;

    float base_scale = target_is_on_ground ?
        evaluate_bump_vel_ground(speed_towards) :
        evaluate_bump_vel_air(speed_towards);

    Vec3 hit_up_dir = target_is_on_ground ? basis2.up : Vec3(0.0f, 0.0f, 1.0f);
    float up_scale = evaluate_bump_upward_vel(speed_towards) * bump_force_scale;

    eval.impulse = vel_dir * base_scale + hit_up_dir * up_scale;
    return eval;
}

// ============================================================================
// Contact Constraint Impulse & Friction Solver (btSequentialImpulseConstraintSolver)
// ============================================================================

/**
 * @brief Resolves bilateral contact impulses, friction, and penetration push.
 * Follows Bullet Sequential Impulse resolution order with strict IEEE-754 math.
 */
__device__ __host__ inline void resolve_car_car_contact(
    Vec3& pos_a, Vec3& vel_a, Vec3& omega_a, const Mat3& basis_a, const Vec3& inv_inertia_a,
    Vec3& pos_b, Vec3& vel_b, Vec3& omega_b, const Mat3& basis_b, const Vec3& inv_inertia_b,
    const CarContactManifold& manifold)
{
    if (manifold.num_points <= 0) return;

    constexpr float INV_M_A = 1.0f / CAR_MASS; // 1.0f / 180.0f
    constexpr float INV_M_B = 1.0f / CAR_MASS; // 1.0f / 180.0f

    // normal_world points from B toward A
    const Vec3& normal = manifold.normal_world;

    for (int p = 0; p < manifold.num_points; ++p) {
        const CarContactPoint& cp = manifold.points[p];

        // Lever arms from centers of mass
        Vec3 r_a = (cp.point_world + normal * cp.depth) - pos_a;
        Vec3 r_b = cp.point_world - pos_b;

        // Relative contact velocity (v_a - v_b)
        Vec3 v_pt_a = vel_a + omega_a.cross(r_a);
        Vec3 v_pt_b = vel_b + omega_b.cross(r_b);
        Vec3 v_rel = v_pt_a - v_pt_b;
        float vn = v_rel.dot(normal);

        // Effective normal mass inverse Kn
        Vec3 u_a_n = r_a.cross(normal);
        Vec3 u_a_n_local = basis_a.transpose() * u_a_n;
        Vec3 w_a_n_local(u_a_n_local.x * inv_inertia_a.x, u_a_n_local.y * inv_inertia_a.y, u_a_n_local.z * inv_inertia_a.z);
        float rot_n_a = u_a_n_local.dot(w_a_n_local);

        Vec3 u_b_n = r_b.cross(normal);
        Vec3 u_b_n_local = basis_b.transpose() * u_b_n;
        Vec3 w_b_n_local(u_b_n_local.x * inv_inertia_b.x, u_b_n_local.y * inv_inertia_b.y, u_b_n_local.z * inv_inertia_b.z);
        float rot_n_b = u_b_n_local.dot(w_b_n_local);

        float Kn = INV_M_A + INV_M_B + rot_n_a + rot_n_b;
        if (Kn < 1e-8f) continue;

        // Normal impulse (with restitution)
        float Jn = 0.0f;
        if (vn < 0.0f) {
            float e = (vn < -CARCAR_RESTITUTION_THRESHOLD) ? CARCAR_COLLISION_RESTITUTION : 0.0f;
            Jn = -(1.0f + e) * vn / Kn;
            Jn = fmaxf(Jn, 0.0f);
        }

        Vec3 normal_impulse = normal * Jn;

        // Tangential Coulomb friction (mu = 0.09f)
        Vec3 v_t = v_rel - normal * vn;
        float vt_mag = v_t.length();
        Vec3 tangent_impulse(0.0f, 0.0f, 0.0f);

        if (vt_mag > 1e-5f) {
            Vec3 t = v_t * (1.0f / vt_mag);

            Vec3 u_a_t = r_a.cross(t);
            Vec3 u_a_t_local = basis_a.transpose() * u_a_t;
            Vec3 w_a_t_local(u_a_t_local.x * inv_inertia_a.x, u_a_t_local.y * inv_inertia_a.y, u_a_t_local.z * inv_inertia_a.z);
            float rot_t_a = u_a_t_local.dot(w_a_t_local);

            Vec3 u_b_t = r_b.cross(t);
            Vec3 u_b_t_local = basis_b.transpose() * u_b_t;
            Vec3 w_b_t_local(u_b_t_local.x * inv_inertia_b.x, u_b_t_local.y * inv_inertia_b.y, u_b_t_local.z * inv_inertia_b.z);
            float rot_t_b = u_b_t_local.dot(w_b_t_local);

            float Kt = INV_M_A + INV_M_B + rot_t_a + rot_t_b;
            if (Kt > 1e-8f) {
                float Jt_desired = vt_mag / Kt;
                float Jt = fminf(Jt_desired, CARCAR_COLLISION_FRICTION * Jn);
                tangent_impulse = t * (-Jt);
            }
        }

        Vec3 J_total = normal_impulse + tangent_impulse;

        // Apply impulse to Car A (+J)
        vel_a = vel_a + J_total * INV_M_A;
        Vec3 tau_a = r_a.cross(J_total);
        Vec3 tau_a_local = basis_a.transpose() * tau_a;
        Vec3 delta_omega_a_local(tau_a_local.x * inv_inertia_a.x, tau_a_local.y * inv_inertia_a.y, tau_a_local.z * inv_inertia_a.z);
        omega_a = omega_a + basis_a * delta_omega_a_local;

        // Apply equal and opposite impulse to Car B (-J)
        vel_b = vel_b - J_total * INV_M_B;
        Vec3 tau_b = r_b.cross(J_total * (-1.0f));
        Vec3 tau_b_local = basis_b.transpose() * tau_b;
        Vec3 delta_omega_b_local(tau_b_local.x * inv_inertia_b.x, tau_b_local.y * inv_inertia_b.y, tau_b_local.z * inv_inertia_b.z);
        omega_b = omega_b + basis_b * delta_omega_b_local;

        // Split impulse penetration displacement push (erp2 = 0.8f, 50% split)
        if (cp.depth > 0.0f) {
            float p_push = cp.depth * 0.4f; // 0.8f * 0.5f
            pos_a = pos_a + normal * p_push;
            pos_b = pos_b - normal * p_push;
        }
    }

    // Velocity Clamping to canonical maximums
    float speed_a = vel_a.length();
    if (speed_a > CAR_MAX_SPEED) {
        vel_a = vel_a * (CAR_MAX_SPEED / speed_a);
    }
    float speed_b = vel_b.length();
    if (speed_b > CAR_MAX_SPEED) {
        vel_b = vel_b * (CAR_MAX_SPEED / speed_b);
    }

    float ang_speed_a = omega_a.length();
    if (ang_speed_a > CAR_MAX_ANG_SPEED) {
        omega_a = omega_a * (CAR_MAX_ANG_SPEED / ang_speed_a);
    }
    float ang_speed_b = omega_b.length();
    if (ang_speed_b > CAR_MAX_ANG_SPEED) {
        omega_b = omega_b * (CAR_MAX_ANG_SPEED / ang_speed_b);
    }
}

// ============================================================================
// Complete Pairwise Collision & Bump Execution (Device Step Function)
// ============================================================================

/**
 * @brief Performs complete OBB collision detection, mutual bump testing,
 * and contact impulse resolution between two cars.
 */
__device__ inline bool resolve_car_pair_collision_device(
    uint32_t car_idx_a, uint32_t car_idx_b,
    CarStateSoA& car_state,
    float dt,
    int demo_mode = 0,
    bool enable_team_demos = false,
    float bump_force_scale = 1.0f,
    float respawn_delay = DEMO_RESPAWN_TIME)
{
    // Ignore if either car is demolished (Arena.cpp:341)
    if (car_state.is_demoed && (car_state.is_demoed[car_idx_a] || car_state.is_demoed[car_idx_b])) {
        return false;
    }

    // Load states for Car A
    Vec3 pos_a(car_state.pos_x[car_idx_a], car_state.pos_y[car_idx_a], car_state.pos_z[car_idx_a]);
    Vec3 vel_a(car_state.vel_x[car_idx_a], car_state.vel_y[car_idx_a], car_state.vel_z[car_idx_a]);
    Vec3 omega_a(car_state.ang_vel_x[car_idx_a], car_state.ang_vel_y[car_idx_a], car_state.ang_vel_z[car_idx_a]);
    Quat quat_a(car_state.q_w[car_idx_a], car_state.q_x[car_idx_a], car_state.q_y[car_idx_a], car_state.q_z[car_idx_a]);
    Mat3 basis_a = Mat3::from_quat(quat_a);

    // Load states for Car B
    Vec3 pos_b(car_state.pos_x[car_idx_b], car_state.pos_y[car_idx_b], car_state.pos_z[car_idx_b]);
    Vec3 vel_b(car_state.vel_x[car_idx_b], car_state.vel_y[car_idx_b], car_state.vel_z[car_idx_b]);
    Vec3 omega_b(car_state.ang_vel_x[car_idx_b], car_state.ang_vel_y[car_idx_b], car_state.ang_vel_z[car_idx_b]);
    Quat quat_b(car_state.q_w[car_idx_b], car_state.q_x[car_idx_b], car_state.q_y[car_idx_b], car_state.q_z[car_idx_b]);
    Mat3 basis_b = Mat3::from_quat(quat_b);

    // Hitbox configs for Car A and Car B (R5)
    uint8_t type_a = car_state.hitbox_type ? car_state.hitbox_type[car_idx_a] : 0;
    uint8_t type_b = car_state.hitbox_type ? car_state.hitbox_type[car_idx_b] : 0;
    Vec3 offset_a = get_hitbox_offset(type_a);
    Vec3 half_a = get_hitbox_half(type_a);
    Vec3 offset_b = get_hitbox_offset(type_b);
    Vec3 half_b = get_hitbox_half(type_b);
    Vec3 inv_inertia_a = get_inv_inertia(type_a);
    Vec3 inv_inertia_b = get_inv_inertia(type_b);

    // Run OBB-OBB narrowphase collision detection
    CarContactManifold manifold;
    bool is_colliding = test_car_car_collision_obb(
        pos_a, basis_a, offset_a, half_a,
        pos_b, basis_b, offset_b, half_b,
        manifold
    );
    if (!is_colliding || manifold.num_points <= 0) {
        return false;
    }

    // Extract attributes for bump evaluation
    uint8_t supersonic_a = car_state.is_supersonic ? car_state.is_supersonic[car_idx_a] : 0;
    uint8_t supersonic_b = car_state.is_supersonic ? car_state.is_supersonic[car_idx_b] : 0;
    uint8_t team_a = car_state.team ? car_state.team[car_idx_a] : 0;
    uint8_t team_b = car_state.team ? car_state.team[car_idx_b] : 1;
    uint8_t on_ground_a = car_state.is_on_ground ? car_state.is_on_ground[car_idx_a] : 1;
    uint8_t on_ground_b = car_state.is_on_ground ? car_state.is_on_ground[car_idx_b] : 1;

    int32_t last_other_a = car_state.car_contact_other_car_id ? car_state.car_contact_other_car_id[car_idx_a] : -1;
    int32_t last_other_b = car_state.car_contact_other_car_id ? car_state.car_contact_other_car_id[car_idx_b] : -1;
    float cooldown_a = car_state.car_contact_cooldown_timer ? car_state.car_contact_cooldown_timer[car_idx_a] : 0.0f;
    float cooldown_b = car_state.car_contact_cooldown_timer ? car_state.car_contact_cooldown_timer[car_idx_b] : 0.0f;

    // Contact points in local chassis frames
    Vec3 local_pt_a = manifold.points[0].point_local_a;
    Vec3 local_pt_b = manifold.points[0].point_local_b;

    // Test bump both ways (Arena.cpp:331-404)
    // 1. Car A attacking Car B
    BumpEvaluation bump_a_to_b = evaluate_single_car_bump(
        pos_a, vel_a, basis_a, supersonic_a, team_a, last_other_a, cooldown_a,
        pos_b, vel_b, basis_b, on_ground_b, team_b, (int32_t)car_idx_b,
        local_pt_a, demo_mode, enable_team_demos, bump_force_scale
    );

    // 2. Car B attacking Car A
    BumpEvaluation bump_b_to_a = evaluate_single_car_bump(
        pos_b, vel_b, basis_b, supersonic_b, team_b, last_other_b, cooldown_b,
        pos_a, vel_a, basis_a, on_ground_a, team_a, (int32_t)car_idx_a,
        local_pt_b, demo_mode, enable_team_demos, bump_force_scale
    );

    bool a_demoed = false;
    bool b_demoed = false;

    // Apply Bump A -> B
    if (bump_a_to_b.type == CarBumpType::DEMO) {
        b_demoed = true;
        if (car_state.is_demoed) car_state.is_demoed[car_idx_b] = 1;
        if (car_state.demo_respawn_timer) car_state.demo_respawn_timer[car_idx_b] = respawn_delay;
        if (car_state.is_supersonic) car_state.is_supersonic[car_idx_b] = 0;
        if (car_state.supersonic_time) car_state.supersonic_time[car_idx_b] = 0.0f;
        vel_b = Vec3(0.0f, 0.0f, 0.0f);
        omega_b = Vec3(0.0f, 0.0f, 0.0f);
        if (car_state.vel_bt_x) {
            car_state.vel_bt_x[car_idx_b] = 0.0f;
            car_state.vel_bt_y[car_idx_b] = 0.0f;
            car_state.vel_bt_z[car_idx_b] = 0.0f;
        }
        if (car_state.car_contact_other_car_id) car_state.car_contact_other_car_id[car_idx_a] = (int32_t)car_idx_b;
        if (car_state.car_contact_cooldown_timer) car_state.car_contact_cooldown_timer[car_idx_a] = BUMP_COOLDOWN_TIME;
    } else if (bump_a_to_b.type == CarBumpType::BUMP) {
        vel_b = vel_b + bump_a_to_b.impulse;
        if (car_state.car_contact_other_car_id) car_state.car_contact_other_car_id[car_idx_a] = (int32_t)car_idx_b;
        if (car_state.car_contact_cooldown_timer) car_state.car_contact_cooldown_timer[car_idx_a] = BUMP_COOLDOWN_TIME;
    }

    // Apply Bump B -> A (only if Car B was not demolished first)
    if (!b_demoed) {
        if (bump_b_to_a.type == CarBumpType::DEMO) {
            a_demoed = true;
            if (car_state.is_demoed) car_state.is_demoed[car_idx_a] = 1;
            if (car_state.demo_respawn_timer) car_state.demo_respawn_timer[car_idx_a] = respawn_delay;
            if (car_state.is_supersonic) car_state.is_supersonic[car_idx_a] = 0;
            if (car_state.supersonic_time) car_state.supersonic_time[car_idx_a] = 0.0f;
            vel_a = Vec3(0.0f, 0.0f, 0.0f);
            omega_a = Vec3(0.0f, 0.0f, 0.0f);
            if (car_state.vel_bt_x) {
                car_state.vel_bt_x[car_idx_a] = 0.0f;
                car_state.vel_bt_y[car_idx_a] = 0.0f;
                car_state.vel_bt_z[car_idx_a] = 0.0f;
            }
            if (car_state.car_contact_other_car_id) car_state.car_contact_other_car_id[car_idx_b] = (int32_t)car_idx_a;
            if (car_state.car_contact_cooldown_timer) car_state.car_contact_cooldown_timer[car_idx_b] = BUMP_COOLDOWN_TIME;
        } else if (bump_b_to_a.type == CarBumpType::BUMP) {
            vel_a = vel_a + bump_b_to_a.impulse;
            if (car_state.car_contact_other_car_id) car_state.car_contact_other_car_id[car_idx_b] = (int32_t)car_idx_a;
            if (car_state.car_contact_cooldown_timer) car_state.car_contact_cooldown_timer[car_idx_b] = BUMP_COOLDOWN_TIME;
        }
    }

    // If neither was demoed, resolve contact constraint restitution, friction and pushback
    if (!a_demoed && !b_demoed) {
        resolve_car_car_contact(
            pos_a, vel_a, omega_a, basis_a, inv_inertia_a,
            pos_b, vel_b, omega_b, basis_b, inv_inertia_b,
            manifold
        );
    }

    // Write back updated states to SoA (Coalesced 128-byte transactions)
    car_state.pos_x[car_idx_a] = pos_a.x;
    car_state.pos_y[car_idx_a] = pos_a.y;
    car_state.pos_z[car_idx_a] = pos_a.z;
    car_state.vel_x[car_idx_a] = vel_a.x;
    car_state.vel_y[car_idx_a] = vel_a.y;
    car_state.vel_z[car_idx_a] = vel_a.z;
    car_state.ang_vel_x[car_idx_a] = omega_a.x;
    car_state.ang_vel_y[car_idx_a] = omega_a.y;
    car_state.ang_vel_z[car_idx_a] = omega_a.z;

    car_state.pos_x[car_idx_b] = pos_b.x;
    car_state.pos_y[car_idx_b] = pos_b.y;
    car_state.pos_z[car_idx_b] = pos_b.z;
    car_state.vel_x[car_idx_b] = vel_b.x;
    car_state.vel_y[car_idx_b] = vel_b.y;
    car_state.vel_z[car_idx_b] = vel_b.z;
    car_state.ang_vel_x[car_idx_b] = omega_b.x;
    car_state.ang_vel_y[car_idx_b] = omega_b.y;
    car_state.ang_vel_z[car_idx_b] = omega_b.z;

    return true;
}

// ============================================================================
// All-Pairs Car-Car Collision Resolution per Arena (up to 6 cars <= 15 pairs)
// ============================================================================

/**
 * @brief Resolves all pairs of cars in a single environment.
 * For N cars (N <= 6), executes N*(N-1)/2 pairs (<= 15 pairs).
 * 
 * @param env_idx Environment index.
 * @param cars_per_env Number of cars in this environment (up to 6).
 * @param car_state CarStateSoA storage.
 * @param dt Simulation delta-T (default DELTA_TIME = 1/120s).
 */
__device__ inline void resolve_all_car_car_collisions(
    uint32_t env_idx,
    uint32_t cars_per_env,
    CarStateSoA& car_state,
    float dt,
    int demo_mode = 0,
    bool enable_team_demos = false,
    float bump_force_scale = 1.0f,
    float respawn_delay = DEMO_RESPAWN_TIME)
{
    if (cars_per_env <= 1) return;

    for (uint32_t i = 0; i < cars_per_env; ++i) {
        uint32_t car_idx_a = env_idx * cars_per_env + i;

        for (uint32_t j = i + 1; j < cars_per_env; ++j) {
            uint32_t car_idx_b = env_idx * cars_per_env + j;

            resolve_car_pair_collision_device(
                car_idx_a, car_idx_b,
                car_state, dt,
                demo_mode, enable_team_demos, bump_force_scale,
                respawn_delay
            );
        }
    }
}

/**
 * @brief Updates car contact cooldown timer for a single car.
 * Mirrors Car::_PostTickUpdate (Car.cpp:172-173).
 */
__device__ __host__ inline void update_car_contact_cooldown(
    uint32_t car_idx,
    CarStateSoA& car_state,
    float dt)
{
    if (car_state.car_contact_cooldown_timer) {
        float cd = car_state.car_contact_cooldown_timer[car_idx];
        if (cd > 0.0f) {
            cd = fmaxf(cd - dt, 0.0f);
            car_state.car_contact_cooldown_timer[car_idx] = cd;
        }
    }
}

} // namespace rocketsim_cuda
