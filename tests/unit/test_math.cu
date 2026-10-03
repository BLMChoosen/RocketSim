#include <iostream>
#include <cassert>
#include <cmath>
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

    *out_pass = 1;
}

int main() {
    std::cout << "[Unit Test: Math] Starting device and host math verification...\n";

    // 1. Host Vec3 tests
    Vec3 v1(1.0f, 0.0f, 0.0f);
    Vec3 v2(0.0f, 1.0f, 0.0f);
    Vec3 v3 = v1.cross(v2);
    assert(fabs(v3.x) < 1e-6f && fabs(v3.y) < 1e-6f && fabs(v3.z - 1.0f) < 1e-6f);
    assert(fabs(v1.length() - 1.0f) < 1e-6f);
    std::cout << "  [PASS] Host Vec3 operations\n";

    // 2. Host Quat antipodal distance tests
    Quat q1(1.0f, 0.0f, 0.0f, 0.0f);
    Quat q2(-1.0f, 0.0f, 0.0f, 0.0f);
    float d = q1.chebyshev_dist(q2);
    assert(d < 1e-6f);
    std::cout << "  [PASS] Host Quat antipodal distance\n";

    // 3. Host Mat3 round-trip
    Mat3 id = Mat3::identity();
    Quat q_id = id.to_quat();
    assert(q1.chebyshev_dist(q_id) < 1e-5f);
    std::cout << "  [PASS] Host Mat3 to Quat round-trip\n";

    // 4. Device CUDA kernel math tests
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
