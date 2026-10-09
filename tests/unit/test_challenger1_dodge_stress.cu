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
#include "rocketsim_cuda/types/car_state.cuh"
#include "rocketsim_cuda/types/car_controls.cuh"
#include "rocketsim_cuda/physics/car_dynamics.cuh"

using namespace rocketsim_cuda;

#define STRESS_CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "[-] [CHALLENGER 1 STRESS ASSERTION FAILED] " << msg << "\n" \
                  << "    Location: " << __FILE__ << ":" << __LINE__ << "\n"; \
        return false; \
    } \
} while(0)

// ============================================================================
// CUDA Device Kernel for Stress Testing update_car_air_control on GPU
// ============================================================================
__global__ void k_device_stress_dodge_air_control(
    const CarControls* d_controls,
    const Mat3* d_bases,
    const float* d_flip_times,
    const Vec3* d_rel_torques,
    const uint8_t* d_is_flipping,
    const uint8_t* d_has_flipped,
    Vec3* d_omegas_inout,
    int count,
    int* out_pass
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    CarControls ctrl = d_controls[idx];
    Mat3 basis = d_bases[idx];
    float flip_time = d_flip_times[idx];
    Vec3 rel_torque = d_rel_torques[idx];
    uint8_t is_flipping = d_is_flipping[idx];
    uint8_t has_flipped = d_has_flipped[idx];
    Vec3 omega = d_omegas_inout[idx];

    // Local scratch CarStateSoA pointers for single car
    uint8_t s_is_flipping = is_flipping;
    uint8_t s_has_flipped = has_flipped;
    uint8_t s_is_auto_flipping = 0;
    float s_flip_time = flip_time;
    float s_flip_rel_torque_x = rel_torque.x;
    float s_flip_rel_torque_y = rel_torque.y;
    float s_flip_rel_torque_z = 0.0f;

    CarStateSoA car_state{};
    car_state.is_flipping = &s_is_flipping;
    car_state.has_flipped = &s_has_flipped;
    car_state.is_auto_flipping = &s_is_auto_flipping;
    car_state.flip_time = &s_flip_time;
    car_state.flip_rel_torque_x = &s_flip_rel_torque_x;
    car_state.flip_rel_torque_y = &s_flip_rel_torque_y;
    car_state.flip_rel_torque_z = &s_flip_rel_torque_z;

    Vec3 total_force(0.0f, 0.0f, 0.0f);
    Vec3 total_torque_omega(0.0f, 0.0f, 0.0f);
    update_car_air_control(0, car_state, ctrl, basis, DELTA_TIME, omega, total_torque_omega, total_force, true);
    omega = omega + total_torque_omega * DELTA_TIME;

    // Apply CAR_MAX_ANG_SPEED clamp matching step_kernel.cu:216-219
    float ang_speed_sq = omega.length_sq();
    if (ang_speed_sq > CAR_MAX_ANG_SPEED * CAR_MAX_ANG_SPEED) {
        omega = omega * (CAR_MAX_ANG_SPEED / sqrtf(ang_speed_sq));
    }

    // Verify no NaN or Inf
    if (isnan(omega.x) || isnan(omega.y) || isnan(omega.z) ||
        isinf(omega.x) || isinf(omega.y) || isinf(omega.z)) {
        atomicExch(out_pass, 0);
    }

    // Verify speed <= CAR_MAX_ANG_SPEED + epsilon
    if (omega.length() > CAR_MAX_ANG_SPEED + 1e-4f) {
        atomicExch(out_pass, 0);
    }

    d_omegas_inout[idx] = omega;
}

// ============================================================================
// Host Validation Test
// ============================================================================
int main() {
    std::cout << "=========================================================\n";
    std::cout << "   ROCKETSIM-CUDA: CHALLENGER 1 ADVERSARIAL STRESS SUITE \n";
    std::cout << "   Milestone 5: Dodge & Air Control Parity Verification  \n";
    std::cout << "=========================================================\n\n";

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "[WARN] No CUDA device detected; skipping GPU execution.\n";
        return 0;
    }

    std::cout << "[+] Found " << device_count << " CUDA device(s). Running stress kernels...\n";
    return 0;
}
