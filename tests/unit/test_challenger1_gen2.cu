#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/quat.cuh"
#include "rocketsim_cuda/math/mat3.cuh"

using namespace rocketsim_cuda;

#define STRESS_CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "[-] [STRESS ASSERTION FAILED] " << msg << "\n" \
                  << "    Location: " << __FILE__ << ":" << __LINE__ << "\n"; \
        return false; \
    } \
} while(0)

// ============================================================================
// CUDA Device Kernel for Math Verification
// ============================================================================
__global__ void k_device_stress_math(
    const Quat* d_quats,
    float* d_max_quat_err,
    float* d_max_basis_err,
    float* d_max_vel_err,
    int count,
    int* out_pass
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Quat q = d_quats[idx];
    Mat3 m = Mat3::from_quat(q);
    Quat q_rec = m.to_quat();

    // 1. Quat Chebyshev error
    float quat_err = q.chebyshev_dist(q_rec);
    if (quat_err > 1e-4f) {
        atomicExch(out_pass, 0);
    }
    // Track max quat error
    // Simple atomic max for float
    atomicMax((int*)d_max_quat_err, __float_as_int(quat_err));

    // 2. Basis vectors: M * e_i vs q.rotate(e_i)
    Vec3 ex(1.0f, 0.0f, 0.0f);
    Vec3 ey(0.0f, 1.0f, 0.0f);
    Vec3 ez(0.0f, 0.0f, 1.0f);

    float err_x = (m * ex).chebyshev_dist(q.rotate(ex));
    float err_y = (m * ey).chebyshev_dist(q.rotate(ey));
    float err_z = (m * ez).chebyshev_dist(q.rotate(ez));

    float max_b = fmaxf(err_x, fmaxf(err_y, err_z));
    if (max_b > 1e-4f) {
        atomicExch(out_pass, 0);
    }
    atomicMax((int*)d_max_basis_err, __float_as_int(max_b));

    // 3. High Velocity (6000 UU/s supersonic pinch)
    Vec3 v_pinch(6000.0f, -4000.0f, 2048.0f);
    Vec3 mv = m * v_pinch;
    Vec3 qv = q.rotate(v_pinch);
    float err_pinch = mv.chebyshev_dist(qv);
    // Relative error on 6000 UU/s
    float rel_err = err_pinch / 7496.0f;
    if (rel_err > 1e-5f) {
        atomicExch(out_pass, 0);
    }
    atomicMax((int*)d_max_vel_err, __float_as_int(err_pinch));
}

// ============================================================================
// Test 1: Extreme Velocity & Supersonic / Pinch Vectors
// ============================================================================
bool TestExtremeVelocities() {
    std::cout << ">>> Running Test 1: Extreme Velocity Vectors (Supersonic & Pinches)...\n";

    float test_speeds[] = {
        500.0f,      // Subsonic car speed
        1400.0f,     // Ground speed
        2300.0f,     // Supersonic threshold
        2900.0f,     // Max boosted speed
        4000.0f,     // Fast pinch
        6000.0f,     // Extreme ball pinch
        10000.0f,    // Super-pinch
        50000.0f,    // Outlier stress
        100000.0f    // Glitch boundary
    };

    for (float s : test_speeds) {
        Vec3 v(s, -s * 0.75f, s * 0.5f);
        float expected_len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
        float len = v.length();
        float len_err = fabsf(len - expected_len);
        STRESS_CHECK(len_err <= expected_len * 1e-5f, "Vec3 length calculation error at extreme speed");

        Vec3 n = v.normalized();
        float n_len = n.length();
        STRESS_CHECK(fabsf(n_len - 1.0f) <= 1e-5f, "Vec3 normalization failed to produce unit length");

        // Direction preservation: n . v == ||v||
        float dot_nv = n.dot(v);
        STRESS_CHECK(fabsf(dot_nv - len) <= len * 1e-5f, "Vec3 normalized direction deviated from original vector");

        // Test with arbitrary rotation: Rodrigues vs Mat3 rotation on high-speed vector
        Quat q_rot(0.6f, 0.48f, 0.36f, 0.52915f); // Unit quaternion
        q_rot = q_rot.normalized();
        Mat3 m_rot = Mat3::from_quat(q_rot);

        Vec3 v_rot_q = q_rot.rotate(v);
        Vec3 v_rot_m = m_rot * v;
        float diff = v_rot_q.chebyshev_dist(v_rot_m);
        float rel_err = diff / len;
        STRESS_CHECK(rel_err <= 1e-5f, "Relative error between Quat and Mat3 rotation exceeded float32 ULP bound");
        STRESS_CHECK(fabsf(v_rot_q.length() - len) <= len * 1e-5f, "Quat rotation failed to preserve vector magnitude");
        STRESS_CHECK(fabsf(v_rot_m.length() - len) <= len * 1e-5f, "Mat3 rotation failed to preserve vector magnitude");
    }

    std::cout << "    [PASS] All velocity scales from 500 to 100,000 UU/s preserved magnitude and rotation parity.\n";
    return true;
}

// ============================================================================
// Test 2: Degenerate, Zero & Small Epsilon Vectors & Quaternions
// ============================================================================
bool TestDegenerateAndBoundaries() {
    std::cout << ">>> Running Test 2: Degenerate, Zero, and Small-Epsilon Boundaries...\n";

    // 2.1 Zero vector normalization
    Vec3 v_zero(0.0f, 0.0f, 0.0f);
    Vec3 n_zero = v_zero.normalized();
    STRESS_CHECK(n_zero == Vec3(0.0f, 0.0f, 0.0f), "Zero vector normalized() must return zero vector");

    // 2.2 Sub-epsilon vector normalization
    Vec3 v_tiny(1e-9f, -1e-9f, 1e-9f);
    Vec3 n_tiny = v_tiny.normalized();
    STRESS_CHECK(n_tiny == Vec3(0.0f, 0.0f, 0.0f), "Sub-epsilon vector normalized() must return zero vector");

    // 2.3 Identity quaternion
    Quat q_id = Quat::identity();
    Vec3 v_test(123.4f, -567.8f, 910.11f);
    Vec3 v_id = q_id.rotate(v_test);
    STRESS_CHECK(v_id.chebyshev_dist(v_test) < 1e-6f, "Identity quaternion rotate() must leave vector unchanged");

    Mat3 m_id = Mat3::from_quat(q_id);
    STRESS_CHECK((m_id * v_test).chebyshev_dist(v_test) < 1e-6f, "Identity matrix must leave vector unchanged");

    Quat q_id_rec = m_id.to_quat();
    STRESS_CHECK(q_id.chebyshev_dist(q_id_rec) < 1e-6f, "to_quat() on identity matrix must return identity quaternion");

    // 2.4 Zero quaternion normalization
    Quat q_zero(0.0f, 0.0f, 0.0f, 0.0f);
    Quat q_zero_norm = q_zero.normalized();
    STRESS_CHECK(q_zero_norm.chebyshev_dist(Quat::identity()) < 1e-6f, "Zero quat normalized() must return identity");

    // 2.5 Nearly identity rotations (tiny angles)
    float tiny_angles[] = {1e-7f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f};
    Vec3 axis(0.57735f, 0.57735f, 0.57735f);
    axis = axis.normalized();

    for (float ang : tiny_angles) {
        float half = ang * 0.5f;
        Quat q_tiny(cosf(half), axis.x * sinf(half), axis.y * sinf(half), axis.z * sinf(half));
        q_tiny = q_tiny.normalized();

        Mat3 m_tiny = Mat3::from_quat(q_tiny);
        Quat q_tiny_rec = m_tiny.to_quat();

        float err = q_tiny.chebyshev_dist(q_tiny_rec);
        STRESS_CHECK(err <= 1e-5f, "Nearly-identity rotation recovery exceeded 1e-5 tolerance");

        // Basis vectors
        STRESS_CHECK(m_tiny.forward.chebyshev_dist(q_tiny.forward()) <= 1e-5f, "Nearly-identity forward mismatch");
        STRESS_CHECK(m_tiny.right.chebyshev_dist(q_tiny.right()) <= 1e-5f, "Nearly-identity right mismatch");
        STRESS_CHECK(m_tiny.up.chebyshev_dist(q_tiny.up()) <= 1e-5f, "Nearly-identity up mismatch");
    }

    std::cout << "    [PASS] Zero vectors, zero quats, identity, and tiny angles verified safely.\n";
    return true;
}

// ============================================================================
// Test 3: Gimbal-Lock and 180-Degree Antipodal Axis Rotations (All 4 Branches)
// ============================================================================
bool Test180DegAndGimbalLock() {
    std::cout << ">>> Running Test 3: 180-Degree Rotations & Gimbal-Lock Boundaries...\n";

    // 3.1 Canonical 180-degree rotations around X, Y, Z
    struct AxisTest {
        const char* name;
        Quat q;
        int expected_branch;
    };

    AxisTest tests180[] = {
        // 180 deg around X: w=0, x=1, y=0, z=0 (Branch 2: forward.x dominant)
        {"180 deg X (Branch 2)", Quat(0.0f, 1.0f, 0.0f, 0.0f), 2},
        // 180 deg around Y: w=0, x=0, y=1, z=0 (Branch 3: right.y dominant)
        {"180 deg Y (Branch 3)", Quat(0.0f, 0.0f, 1.0f, 0.0f), 3},
        // 180 deg around Z: w=0, x=0, y=0, z=1 (Branch 4: up.z dominant)
        {"180 deg Z (Branch 4)", Quat(0.0f, 0.0f, 0.0f, 1.0f), 4},
        // 0 deg: w=1, x=0, y=0, z=0 (Branch 1: trace dominant)
        {"Identity (Branch 1)", Quat(1.0f, 0.0f, 0.0f, 0.0f), 1}
    };

    for (const auto& t : tests180) {
        Mat3 m = Mat3::from_quat(t.q);
        Quat q_rec = m.to_quat();

        float quat_err = t.q.chebyshev_dist(q_rec);
        STRESS_CHECK(quat_err <= 1e-5f, "180 deg axis rotation failed to recover quaternion");

        // Verify rotation of basis
        STRESS_CHECK(m.forward.chebyshev_dist(t.q.forward()) <= 1e-5f, "180 deg forward basis error");
        STRESS_CHECK(m.right.chebyshev_dist(t.q.right()) <= 1e-5f, "180 deg right basis error");
        STRESS_CHECK(m.up.chebyshev_dist(t.q.up()) <= 1e-5f, "180 deg up basis error");

        // Verify roundtrip matrix
        Mat3 m_roundtrip = Mat3::from_quat(q_rec);
        STRESS_CHECK(m.forward.chebyshev_dist(m_roundtrip.forward) <= 1e-5f, "180 deg matrix forward roundtrip error");
        STRESS_CHECK(m.right.chebyshev_dist(m_roundtrip.right) <= 1e-5f, "180 deg matrix right roundtrip error");
        STRESS_CHECK(m.up.chebyshev_dist(m_roundtrip.up) <= 1e-5f, "180 deg matrix up roundtrip error");
    }

    // 3.2 Diagonal 180-degree rotations (where 2 or 3 diagonal entries are equal)
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    float inv_sqrt3 = 1.0f / sqrtf(3.0f);

    Vec3 diagonal_axes[] = {
        Vec3(inv_sqrt2, inv_sqrt2, 0.0f),         // XY diagonal
        Vec3(inv_sqrt2, 0.0f, inv_sqrt2),         // XZ diagonal
        Vec3(0.0f, inv_sqrt2, inv_sqrt2),         // YZ diagonal
        Vec3(inv_sqrt2, -inv_sqrt2, 0.0f),        // X-Y diagonal
        Vec3(inv_sqrt3, inv_sqrt3, inv_sqrt3),    // XYZ diagonal (all 3 equal!)
        Vec3(-inv_sqrt3, inv_sqrt3, inv_sqrt3),   // -XYZ diagonal
        Vec3(inv_sqrt3, -inv_sqrt3, inv_sqrt3),   // X-YZ diagonal
        Vec3(inv_sqrt3, inv_sqrt3, -inv_sqrt3)    // XY-Z diagonal
    };

    for (const auto& axis : diagonal_axes) {
        // 180 degree rotation: w = cos(pi/2) = 0, xyz = sin(pi/2) * axis = axis
        Quat q_diag(0.0f, axis.x, axis.y, axis.z);
        q_diag = q_diag.normalized();

        Mat3 m = Mat3::from_quat(q_diag);
        Quat q_rec = m.to_quat();

        float quat_err = q_diag.chebyshev_dist(q_rec);
        STRESS_CHECK(quat_err <= 1e-4f, "Diagonal 180-deg rotation quaternion error > 1e-4");
        STRESS_CHECK(quat_err <= 1e-5f, "Diagonal 180-deg rotation quaternion error > 1e-5");

        // Verify basis rotation
        STRESS_CHECK(m.forward.chebyshev_dist(q_diag.forward()) <= 1e-5f, "Diagonal forward basis error");
        STRESS_CHECK(m.right.chebyshev_dist(q_diag.right()) <= 1e-5f, "Diagonal right basis error");
        STRESS_CHECK(m.up.chebyshev_dist(q_diag.up()) <= 1e-5f, "Diagonal up basis error");
    }

    // 3.3 Gimbal-Lock: 90 degree pitch (pitch = +/- pi/2)
    // At pitch = 90 deg, car forward points straight up into +Z
    {
        // 90 deg pitch around Y axis
        Quat q_pitch90(cosf(3.14159265f * 0.25f), 0.0f, sinf(3.14159265f * 0.25f), 0.0f);
        q_pitch90 = q_pitch90.normalized();

        Mat3 m_pitch90 = Mat3::from_quat(q_pitch90);
        Quat q_pitch90_rec = m_pitch90.to_quat();

        STRESS_CHECK(q_pitch90.chebyshev_dist(q_pitch90_rec) <= 1e-5f, "90 deg pitch recovery error > 1e-5");
        // Forward vector should point to -Z or +Z depending on pitch direction
        Vec3 fwd = q_pitch90.forward();
        STRESS_CHECK(fabsf(fwd.y) <= 1e-5f, "90 deg pitch forward.y should be 0");
        STRESS_CHECK(fabsf(fwd.z - (-1.0f)) <= 1e-5f || fabsf(fwd.z - 1.0f) <= 1e-5f, "90 deg pitch forward.z should be +/- 1");
    }

    // 3.4 Trace near zero transitions
    // Trace = 4*w^2 - 1. When w = 0.5, Trace = 0.
    // Angle = 2 * arccos(0.5) = 120 degrees (2*pi/3).
    float angles_near_120[] = {
        2.0943951f - 1e-4f, // 119.994 deg (trace slightly > 0)
        2.0943951f,         // 120.000 deg (trace exactly ~ 0)
        2.0943951f + 1e-4f  // 120.006 deg (trace slightly < 0)
    };

    for (float ang : angles_near_120) {
        for (const auto& axis : diagonal_axes) {
            float half = ang * 0.5f;
            Quat q(cosf(half), axis.x * sinf(half), axis.y * sinf(half), axis.z * sinf(half));
            q = q.normalized();

            Mat3 m = Mat3::from_quat(q);
            Quat q_rec = m.to_quat();

            float quat_err = q.chebyshev_dist(q_rec);
            STRESS_CHECK(quat_err <= 1e-5f, "Near-zero trace boundary transition quaternion error > 1e-5");

            STRESS_CHECK(m.forward.chebyshev_dist(q.forward()) <= 1e-5f, "Near-zero trace forward basis error");
            STRESS_CHECK(m.right.chebyshev_dist(q.right()) <= 1e-5f, "Near-zero trace right basis error");
            STRESS_CHECK(m.up.chebyshev_dist(q.up()) <= 1e-5f, "Near-zero trace up basis error");
        }
    }

    std::cout << "    [PASS] 180-deg rotations, diagonal axes, gimbal-lock, and zero-trace transitions verified.\n";
    return true;
}

// ============================================================================
// Test 4: Monte Carlo SO(3) 100,000 Uniform Rotations Stress Test
// ============================================================================
bool TestMonteCarloSO3(int count, float* out_max_quat_err, float* out_max_basis_err, float* out_max_vel_err) {
    std::cout << ">>> Running Test 4: Monte Carlo SO(3) (" << count << " Uniform Orientations)...\n";

    std::mt19937 rng(1337);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);

    float max_quat_err = 0.0f;
    float max_basis_err = 0.0f;
    float max_vel_err = 0.0f;

    Vec3 pinch_vector(6000.0f, -4000.0f, 2048.0f);
    float pinch_len = pinch_vector.length();

    int branch_counts[4] = {0, 0, 0, 0};

    for (int i = 0; i < count; i++) {
        // Uniform quaternion on S^3 using 4 Gaussian variables
        Quat q(norm_dist(rng), norm_dist(rng), norm_dist(rng), norm_dist(rng));
        q = q.normalized();

        Mat3 m = Mat3::from_quat(q);

        // Classify branch
        float tr = m.forward.x + m.right.y + m.up.z;
        if (tr > 0.0f) {
            branch_counts[0]++;
        } else if (m.forward.x > m.right.y && m.forward.x > m.up.z) {
            branch_counts[1]++;
        } else if (m.right.y > m.up.z) {
            branch_counts[2]++;
        } else {
            branch_counts[3]++;
        }

        Quat q_rec = m.to_quat();

        // 1. Quat Chebyshev error (Chebyshev norm <= 1e-5 per GEMINI.md)
        float q_err = q.chebyshev_dist(q_rec);
        if (q_err > max_quat_err) max_quat_err = q_err;
        STRESS_CHECK(q_err <= 1e-5f, "Monte Carlo orientation exceeded 1e-5 quaternion error bound");

        // 2. Basis vectors Chebyshev error: M * e_i vs q.rotate(e_i)
        float err_fwd = m.forward.chebyshev_dist(q.forward());
        float err_rgt = m.right.chebyshev_dist(q.right());
        float err_up  = m.up.chebyshev_dist(q.up());
        float b_err = fmaxf(err_fwd, fmaxf(err_rgt, err_up));
        if (b_err > max_basis_err) max_basis_err = b_err;
        STRESS_CHECK(b_err <= 1e-5f, "Monte Carlo basis vector rotation exceeded 1e-5 error bound");

        // 3. Supersonic pinch vector (6000 UU/s)
        Vec3 mv = m * pinch_vector;
        Vec3 qv = q.rotate(pinch_vector);
        float v_err = mv.chebyshev_dist(qv);
        if (v_err > max_vel_err) max_vel_err = v_err;
        // Relative error must be <= 1e-5
        STRESS_CHECK(v_err / pinch_len <= 1e-5f, "Monte Carlo high-velocity pinch relative error > 1e-5");
    }

    std::cout << "    Branch distribution: Branch 1=" << branch_counts[0]
              << ", Branch 2=" << branch_counts[1]
              << ", Branch 3=" << branch_counts[2]
              << ", Branch 4=" << branch_counts[3] << "\n";
    std::cout << "    Maximum Quat Chebyshev Error:   " << std::scientific << std::setprecision(5) << max_quat_err << "\n";
    std::cout << "    Maximum Basis Chebyshev Error:  " << std::scientific << std::setprecision(5) << max_basis_err << "\n";
    std::cout << "    Maximum 6000 UU/s Vector Error: " << std::fixed << std::setprecision(6) << max_vel_err
              << " UU/s (Rel: " << std::scientific << (max_vel_err / pinch_len) << ")\n";

    *out_max_quat_err = max_quat_err;
    *out_max_basis_err = max_basis_err;
    *out_max_vel_err = max_vel_err;

    STRESS_CHECK(max_quat_err <= 1e-5f, "Max quat Chebyshev error exceeded 1e-5");
    STRESS_CHECK(max_basis_err <= 1e-5f, "Max basis Chebyshev error exceeded 1e-5");

    std::cout << "    [PASS] All " << count << " Monte Carlo rotations passed strict GEMINI.md tolerances.\n";
    return true;
}

// ============================================================================
// Test 5: CUDA GPU Device Kernel Execution
// ============================================================================
bool TestCudaDeviceStress(int count) {
    std::cout << ">>> Running Test 5: CUDA Device Kernel Execution (" << count << " Parallel Orientations)...\n";

    std::vector<Quat> h_quats(count);
    std::mt19937 rng(9999);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);

    for (int i = 0; i < count; i++) {
        Quat q(norm_dist(rng), norm_dist(rng), norm_dist(rng), norm_dist(rng));
        h_quats[i] = q.normalized();
    }

    Quat* d_quats = nullptr;
    float* d_max_quat = nullptr;
    float* d_max_basis = nullptr;
    float* d_max_vel = nullptr;
    int* d_pass = nullptr;

    cudaMalloc(&d_quats, count * sizeof(Quat));
    cudaMalloc(&d_max_quat, sizeof(float));
    cudaMalloc(&d_max_basis, sizeof(float));
    cudaMalloc(&d_max_vel, sizeof(float));
    cudaMalloc(&d_pass, sizeof(int));

    float zero_f = 0.0f;
    int one_i = 1;

    cudaMemcpy(d_quats, h_quats.data(), count * sizeof(Quat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_quat, &zero_f, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_basis, &zero_f, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_vel, &zero_f, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pass, &one_i, sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    k_device_stress_math<<<blocks, threads>>>(d_quats, d_max_quat, d_max_basis, d_max_vel, count, d_pass);
    cudaDeviceSynchronize();

    int h_pass = 0;
    float h_max_quat = 0.0f;
    float h_max_basis = 0.0f;
    float h_max_vel = 0.0f;

    cudaMemcpy(&h_pass, d_pass, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_quat, d_max_quat, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_basis, d_max_basis, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_vel, d_max_vel, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_quats);
    cudaFree(d_max_quat);
    cudaFree(d_max_basis);
    cudaFree(d_max_vel);
    cudaFree(d_pass);

    std::cout << "    Device Execution Result: Pass=" << h_pass << "\n";
    std::cout << "    Device Max Quat Error:   " << std::scientific << std::setprecision(5) << h_max_quat << "\n";
    std::cout << "    Device Max Basis Error:  " << std::scientific << std::setprecision(5) << h_max_basis << "\n";
    std::cout << "    Device Max 6000 UU/s Err:" << std::fixed << std::setprecision(6) << h_max_vel << " UU/s\n";

    STRESS_CHECK(h_pass == 1, "CUDA device kernel execution encountered errors");
    STRESS_CHECK(h_max_quat <= 1e-5f, "CUDA device max quat error exceeded 1e-5");
    STRESS_CHECK(h_max_basis <= 1e-5f, "CUDA device max basis error exceeded 1e-5");

    std::cout << "    [PASS] Device execution passed with 100% parity against IEEE-754 invariants.\n";
    return true;
}

// ============================================================================
// Main Runner
// ============================================================================
int main() {
    std::cout << "======================================================================\n"
              << "       CHALLENGER 1 (GEN 2): EMPIRICAL MATH STRESS HARNESS            \n"
              << "======================================================================\n\n";

    bool all_passed = true;

    all_passed &= TestExtremeVelocities();
    std::cout << "\n";

    all_passed &= TestDegenerateAndBoundaries();
    std::cout << "\n";

    all_passed &= Test180DegAndGimbalLock();
    std::cout << "\n";

    float max_q = 0.0f, max_b = 0.0f, max_v = 0.0f;
    all_passed &= TestMonteCarloSO3(100000, &max_q, &max_b, &max_v);
    std::cout << "\n";

    all_passed &= TestCudaDeviceStress(10000);
    std::cout << "\n";

    std::cout << "======================================================================\n";
    if (all_passed) {
        std::cout << "  ALL EMPIRICAL CHALLENGES PASSED! VERDICT: APPROVE                  \n";
    } else {
        std::cout << "  EMPIRICAL CHALLENGES FAILED! VERDICT: REQUEST_CHANGES              \n";
    }
    std::cout << "======================================================================\n";

    return all_passed ? 0 : 1;
}
