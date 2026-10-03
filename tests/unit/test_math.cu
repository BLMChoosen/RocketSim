#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/quat.cuh"
#include "rocketsim_cuda/math/mat3.cuh"

using namespace rocketsim_cuda;

__global__ void k_test_math_device(int* out_pass) {
    int tid = threadIdx.x;
    if (tid != 0) return;

    // Test Vec3 on device
    Vec3 a(1.0f, 2.0f, 3.0f);
    Vec3 b(4.0f, 5.0f, 6.0f);
    Vec3 c = a + b;
    if (fabsf(c.x - 5.0f) > 1e-6f || fabsf(c.y - 7.0f) > 1e-6f || fabsf(c.z - 9.0f) > 1e-6f) {
        *out_pass = 0;
        return;
    }

    // Test dot and cross
    float d = a.dot(b); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    if (fabsf(d - 32.0f) > 1e-6f) {
        *out_pass = 0;
        return;
    }

    Vec3 cr = a.cross(b); // (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) = (-3, 6, -3)
    if (fabsf(cr.x - (-3.0f)) > 1e-6f || fabsf(cr.y - 6.0f) > 1e-6f || fabsf(cr.z - (-3.0f)) > 1e-6f) {
        *out_pass = 0;
        return;
    }

    // Test Quat rotation on device
    Quat q = Quat::identity();
    Vec3 fwd = q.forward();
    if (fabsf(fwd.x - 1.0f) > 1e-6f || fabsf(fwd.y) > 1e-6f || fabsf(fwd.z) > 1e-6f) {
        *out_pass = 0;
        return;
    }

    // 90 degree yaw (rotation around Z axis)
    // q = [cos(pi/4), 0, 0, sin(pi/4)] = [0.70710678, 0, 0, 0.70710678]
    Quat q_yaw(0.70710678f, 0.0f, 0.0f, 0.70710678f);
    Vec3 rot_fwd = q_yaw.forward(); // should point in +Y direction
    if (fabsf(rot_fwd.x) > 1e-5f || fabsf(rot_fwd.y - 1.0f) > 1e-5f || fabsf(rot_fwd.z) > 1e-5f) {
        *out_pass = 0;
        return;
    }

    // Antipodal metric: q and -q represent same rotation
    Quat q_neg(-q_yaw.w, -q_yaw.x, -q_yaw.y, -q_yaw.z);
    float antipodal_dist = q_yaw.chebyshev_dist(q_neg);
    if (antipodal_dist > 1e-6f) {
        *out_pass = 0;
        return;
    }

    // Test Mat3 from quat
    Mat3 m = Mat3::from_quat(q_yaw);
    Vec3 m_fwd = m.forward;
    if (fabsf(m_fwd.x) > 1e-5f || fabsf(m_fwd.y - 1.0f) > 1e-5f || fabsf(m_fwd.z) > 1e-5f) {
        *out_pass = 0;
        return;
    }

    // Test device to_quat recovery (Branch 1: trace > 0)
    Quat q_yaw_rec = m.to_quat();
    if (q_yaw.chebyshev_dist(q_yaw_rec) > 1e-5f) {
        *out_pass = 0;
        return;
    }

    // Test 90-deg roll (X-axis) and pitch (Y-axis) on device
    Quat q_roll(0.70710678f, 0.70710678f, 0.0f, 0.0f);
    Mat3 m_roll = Mat3::from_quat(q_roll);
    if (q_roll.chebyshev_dist(m_roll.to_quat()) > 1e-5f) {
        *out_pass = 0;
        return;
    }

    Quat q_pitch(0.70710678f, 0.0f, 0.70710678f, 0.0f);
    Mat3 m_pitch = Mat3::from_quat(q_pitch);
    if (q_pitch.chebyshev_dist(m_pitch.to_quat()) > 1e-5f) {
        *out_pass = 0;
        return;
    }

    // Test 180-deg rotations on device to exercise branches 2, 3, 4 (trace <= 0)
    Quat q_180x(0.0f, 1.0f, 0.0f, 0.0f);
    if (q_180x.chebyshev_dist(Mat3::from_quat(q_180x).to_quat()) > 1e-5f) {
        *out_pass = 0;
        return;
    }

    Quat q_180y(0.0f, 0.0f, 1.0f, 0.0f);
    if (q_180y.chebyshev_dist(Mat3::from_quat(q_180y).to_quat()) > 1e-5f) {
        *out_pass = 0;
        return;
    }

    Quat q_180z(0.0f, 0.0f, 0.0f, 1.0f);
    if (q_180z.chebyshev_dist(Mat3::from_quat(q_180z).to_quat()) > 1e-5f) {
        *out_pass = 0;
        return;
    }

    // Test arbitrary rotation on device: M * v == q.rotate(v)
    Quat q_arb(0.5f, 0.5f, 0.5f, 0.5f);
    Mat3 m_arb = Mat3::from_quat(q_arb);
    if (q_arb.chebyshev_dist(m_arb.to_quat()) > 1e-5f) {
        *out_pass = 0;
        return;
    }
    Vec3 v_test(1.0f, 2.0f, 3.0f);
    Vec3 v_m = m_arb * v_test;
    Vec3 v_q = q_arb.rotate(v_test);
    if (v_m.chebyshev_dist(v_q) > 1e-5f) {
        *out_pass = 0;
        return;
    }

    *out_pass = 1;
}

#define MATH_CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "[-] MATH ASSERTION FAILED: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
        return 1; \
    } \
} while (0)

int main() {
    std::cout << "[Unit Test: Math] Starting device and host math verification...\n";

    // 1. Host Vec3 tests
    Vec3 v1(1.0f, 0.0f, 0.0f);
    Vec3 v2(0.0f, 1.0f, 0.0f);
    Vec3 v3 = v1.cross(v2);
    MATH_CHECK(fabs(v3.x) < 1e-6f && fabs(v3.y) < 1e-6f && fabs(v3.z - 1.0f) < 1e-6f);
    MATH_CHECK(fabs(v1.length() - 1.0f) < 1e-6f);
    std::cout << "  [PASS] Host Vec3 operations\n";

    // 2. Host Quat antipodal distance tests
    Quat q1(1.0f, 0.0f, 0.0f, 0.0f);
    Quat q2(-1.0f, 0.0f, 0.0f, 0.0f);
    float d = q1.chebyshev_dist(q2);
    MATH_CHECK(d < 1e-6f);
    std::cout << "  [PASS] Host Quat antipodal distance\n";

    // 3. Host Mat3 round-trip (identity)
    Mat3 id = Mat3::identity();
    Quat q_id = id.to_quat();
    MATH_CHECK(q1.chebyshev_dist(q_id) < 1e-5f);
    std::cout << "  [PASS] Host Mat3 to Quat round-trip (identity)\n";

    // 4. Host 90-degree axis rotations (X, Y, Z)
    {
        // 90 deg around X (Roll)
        Quat q_90x(0.70710678f, 0.70710678f, 0.0f, 0.0f);
        Mat3 m_90x = Mat3::from_quat(q_90x);
        Quat q_90x_rec = m_90x.to_quat();
        MATH_CHECK(q_90x.chebyshev_dist(q_90x_rec) < 1e-5f);
        Vec3 vx(0.0f, 1.0f, 0.0f);
        MATH_CHECK((m_90x * vx).chebyshev_dist(q_90x.rotate(vx)) < 1e-5f);
        MATH_CHECK((m_90x * vx).chebyshev_dist(Vec3(0.0f, 0.0f, 1.0f)) < 1e-5f);

        // 90 deg around Y (Pitch)
        Quat q_90y(0.70710678f, 0.0f, 0.70710678f, 0.0f);
        Mat3 m_90y = Mat3::from_quat(q_90y);
        Quat q_90y_rec = m_90y.to_quat();
        MATH_CHECK(q_90y.chebyshev_dist(q_90y_rec) < 1e-5f);
        Vec3 vy(1.0f, 0.0f, 0.0f);
        MATH_CHECK((m_90y * vy).chebyshev_dist(q_90y.rotate(vy)) < 1e-5f);
        MATH_CHECK((m_90y * vy).chebyshev_dist(Vec3(0.0f, 0.0f, -1.0f)) < 1e-5f);

        // 90 deg around Z (Yaw)
        Quat q_90z(0.70710678f, 0.0f, 0.0f, 0.70710678f);
        Mat3 m_90z = Mat3::from_quat(q_90z);
        Quat q_90z_rec = m_90z.to_quat();
        MATH_CHECK(q_90z.chebyshev_dist(q_90z_rec) < 1e-5f);
        Vec3 vz(1.0f, 0.0f, 0.0f);
        MATH_CHECK((m_90z * vz).chebyshev_dist(q_90z.rotate(vz)) < 1e-5f);
        MATH_CHECK((m_90z * vz).chebyshev_dist(Vec3(0.0f, 1.0f, 0.0f)) < 1e-5f);

        std::cout << "  [PASS] Host 90-degree axis rotations (X, Y, Z)\n";
    }

    // 5. Host 180-degree axis rotations exercising branches 2, 3, 4 (trace <= 0)
    {
        // 180 deg around X (Branch 2: forward.x > right.y && forward.x > up.z)
        Quat q_180x(0.0f, 1.0f, 0.0f, 0.0f);
        Mat3 m_180x = Mat3::from_quat(q_180x);
        Quat q_180x_rec = m_180x.to_quat();
        MATH_CHECK(q_180x.chebyshev_dist(q_180x_rec) < 1e-5f);
        MATH_CHECK((m_180x * Vec3(1.0f, 2.0f, 3.0f)).chebyshev_dist(q_180x.rotate(Vec3(1.0f, 2.0f, 3.0f))) < 1e-5f);

        // 180 deg around Y (Branch 3: right.y > up.z)
        Quat q_180y(0.0f, 0.0f, 1.0f, 0.0f);
        Mat3 m_180y = Mat3::from_quat(q_180y);
        Quat q_180y_rec = m_180y.to_quat();
        MATH_CHECK(q_180y.chebyshev_dist(q_180y_rec) < 1e-5f);
        MATH_CHECK((m_180y * Vec3(1.0f, 2.0f, 3.0f)).chebyshev_dist(q_180y.rotate(Vec3(1.0f, 2.0f, 3.0f))) < 1e-5f);

        // 180 deg around Z (Branch 4: up.z largest)
        Quat q_180z(0.0f, 0.0f, 0.0f, 1.0f);
        Mat3 m_180z = Mat3::from_quat(q_180z);
        Quat q_180z_rec = m_180z.to_quat();
        MATH_CHECK(q_180z.chebyshev_dist(q_180z_rec) < 1e-5f);
        MATH_CHECK((m_180z * Vec3(1.0f, 2.0f, 3.0f)).chebyshev_dist(q_180z.rotate(Vec3(1.0f, 2.0f, 3.0f))) < 1e-5f);

        std::cout << "  [PASS] Host 180-degree axis rotations (Branches 2, 3, 4)\n";
    }

    // 6. Host Pseudo-random / Arbitrary 3D Rotations (Monte Carlo Verification)
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        constexpr int NUM_RANDOM_ROTS = 1000;
        float max_quat_err = 0.0f;
        float max_vec_err = 0.0f;

        Vec3 test_vectors[] = {
            Vec3(1.0f, 2.0f, 3.0f),
            Vec3(-2300.0f, 1500.0f, 60.0f),
            Vec3(6000.0f, -4000.0f, 2048.0f),
            Vec3(0.0f, 0.0f, 0.0f)
        };

        for (int i = 0; i < NUM_RANDOM_ROTS; i++) {
            Quat q_rnd(dist(rng), dist(rng), dist(rng), dist(rng));
            q_rnd = q_rnd.normalized();

            Mat3 m_rnd = Mat3::from_quat(q_rnd);
            Quat q_recovered = m_rnd.to_quat();

            float quat_err = q_rnd.chebyshev_dist(q_recovered);
            if (quat_err > max_quat_err) max_quat_err = quat_err;
            MATH_CHECK(quat_err <= 1e-4f);

            for (const Vec3& v : test_vectors) {
                Vec3 v_mat = m_rnd * v;
                Vec3 v_quat = q_rnd.rotate(v);
                Vec3 v_rec = q_recovered.rotate(v);

                float tol = 1e-4f * v.length() + 1e-4f;
                float err_mat_quat = v_mat.chebyshev_dist(v_quat);
                float err_mat_rec = v_mat.chebyshev_dist(v_rec);

                if (err_mat_quat > max_vec_err) max_vec_err = err_mat_quat;
                if (err_mat_rec > max_vec_err) max_vec_err = err_mat_rec;

                MATH_CHECK(err_mat_quat <= tol);
                MATH_CHECK(err_mat_rec <= tol);
            }
        }
        std::cout << "  [PASS] Host 1000 pseudo-random 3D rotations (max quat err = "
                  << max_quat_err << ", max vec err = " << max_vec_err << ")\n";
    }

    // 7. Device CUDA kernel math tests
    int* d_pass = nullptr;
    cudaMalloc(&d_pass, sizeof(int));
    int init_val = 0;
    cudaMemcpy(d_pass, &init_val, sizeof(int), cudaMemcpyHostToDevice);

    k_test_math_device<<<1, 32>>>(d_pass);
    cudaDeviceSynchronize();

    int h_pass = 0;
    cudaMemcpy(&h_pass, d_pass, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_pass);

    if (h_pass != 1) {
        std::cerr << "[-] FAIL: Device CUDA math kernel assertion failed!\n";
        return 1;
    }
    std::cout << "  [PASS] Device CUDA math kernel execution\n";

    std::cout << "[Unit Test: Math] ALL MATH TESTS PASSED CLEANLY!\n";
    return 0;
}
