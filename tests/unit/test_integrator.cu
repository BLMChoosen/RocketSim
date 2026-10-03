#include <iostream>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/quat.cuh"
#include "rocketsim_cuda/physics/integrator.cuh"

using namespace rocketsim_cuda;

struct IntegratorTestResult {
    Vec3 freefall_pos;
    Vec3 freefall_vel;

    Quat rotated_quat;
    float damped_vel;
};

__global__ void TestIntegratorKernel(IntegratorTestResult* res) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // 1. Linear freefall
    Vec3 p(0.0f, 0.0f, 1000.0f);
    Vec3 v(0.0f, 0.0f, 0.0f);
    symplectic_euler_linear(p, v, Vec3(0.0f, 0.0f, 0.0f), CAR_MASS, DELTA_TIME);
    res->freefall_pos = p;
    res->freefall_vel = v;

    // 2. Exponential Map Quaternion Integration: rotate 1.0 rad/s around Z for 120 steps = 1.0 second
    Quat q = Quat::identity();
    Vec3 omega(0.0f, 0.0f, 1.0f);
    for (int i = 0; i < 120; ++i) {
        q = bullet_integrate_quaternion(q, omega, DELTA_TIME);
    }
    res->rotated_quat = q;

    // 3. Damping
    Vec3 ball_v(1000.0f, 0.0f, 0.0f);
    Vec3 ball_w(0.0f, 0.0f, 0.0f);
    apply_rigid_body_damping(ball_v, ball_w, BALL_DRAG, 0.0f, DELTA_TIME);
    res->damped_vel = ball_v.x;
}

int main() {
    std::cout << "[TestIntegrator] Running CUDA Symplectic Euler & Exponential Map unit tests...\n";

    IntegratorTestResult* d_res = nullptr;
    cudaMalloc(&d_res, sizeof(IntegratorTestResult));

    TestIntegratorKernel<<<1, 1>>>(d_res);
    cudaDeviceSynchronize();

    IntegratorTestResult h_res;
    cudaMemcpy(&h_res, d_res, sizeof(IntegratorTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_res);

    // 1. Check freefall
    float expected_v = GRAVITY_Z * DELTA_TIME;
    float expected_p = 1000.0f + expected_v * DELTA_TIME;
    std::cout << "  Freefall vel: " << h_res.freefall_vel.z << " (expected: " << expected_v << ")\n";
    std::cout << "  Freefall pos: " << h_res.freefall_pos.z << " (expected: " << expected_p << ")\n";
    assert(std::fabs(h_res.freefall_vel.z - expected_v) < 1e-5f);
    assert(std::fabs(h_res.freefall_pos.z - expected_p) < 1e-5f);

    // 2. Check 1.0 rad rotation around Z
    float expected_w = std::cos(0.5f); // ~0.87758256
    float expected_z = std::sin(0.5f); // ~0.47942554
    std::cout << "  Rotated quat: w = " << h_res.rotated_quat.w << ", z = " << h_res.rotated_quat.z 
              << " (expected: w = " << expected_w << ", z = " << expected_z << ")\n";
    assert(std::fabs(h_res.rotated_quat.w - expected_w) < 1e-4f);
    assert(std::fabs(h_res.rotated_quat.z - expected_z) < 1e-4f);

    // 3. Check damping
    float expected_damped = 1000.0f * std::pow(1.0f - BALL_DRAG, DELTA_TIME);
    std::cout << "  Damped vel: " << h_res.damped_vel << " (expected: " << expected_damped << ")\n";
    assert(std::fabs(h_res.damped_vel - expected_damped) < 1e-4f);

    std::cout << "[+] All Integrator tests passed successfully!\n";
    return 0;
}
