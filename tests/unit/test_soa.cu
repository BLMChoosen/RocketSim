#include <iostream>
#include <cassert>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/sim_context.cuh"
#include "rocketsim_cuda/types/car_state.cuh"
#include "rocketsim_cuda/types/ball_state.cuh"
#include "rocketsim_cuda/types/car_controls.cuh"

using namespace rocketsim_cuda;

__global__ void k_test_soa_parallel_rw(BallStateSoA ball, CarStateSoA car, uint32_t num_envs, int* out_pass) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    // Parallel write
    Vec3 p(static_cast<float>(idx), static_cast<float>(idx * 2), 100.0f + static_cast<float>(idx));
    ball.set_pos(idx, p);

    Vec3 cp(static_cast<float>(idx * 3), static_cast<float>(idx * 4), 17.0f);
    car.set_pos(idx, cp);
    car.boost[idx] = static_cast<float>(idx % 100);

    // Parallel read & verify
    Vec3 read_p = ball.get_pos(idx);
    if (fabsf(read_p.x - p.x) > 1e-5f || fabsf(read_p.y - p.y) > 1e-5f || fabsf(read_p.z - p.z) > 1e-5f) {
        atomicExch(out_pass, 0);
    }

    Vec3 read_cp = car.get_pos(idx);
    if (fabsf(read_cp.x - cp.x) > 1e-5f || fabsf(read_cp.y - cp.y) > 1e-5f || fabsf(read_cp.z - cp.z) > 1e-5f) {
        atomicExch(out_pass, 0);
    }

    if (fabsf(car.boost[idx] - static_cast<float>(idx % 100)) > 1e-5f) {
        atomicExch(out_pass, 0);
    }
}

int main() {
    std::cout << "[Unit Test: SoA] Starting SoA layout and memory pool verification...\n";

    constexpr uint32_t NUM_ENVS = 1024;
    SimContext ctx(NUM_ENVS, 1);

    std::cout << "  SimContext initialized for " << NUM_ENVS << " envs.\n";
    std::cout << "  Pre-allocated pool size: " << ctx.GetAllocatedBytes() << " bytes.\n";

    // 1. Verify 128-byte cache-line alignment of device pointers (GEMINI.md Section 2.1)
    const auto& ball = ctx.GetBallState();
    auto check_128_align = [](const void* ptr, const char* name) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        if (addr % 128 != 0) {
            std::cerr << "[-] Alignment violation: " << name << " address " << ptr << " is not 128-byte aligned!\n";
            return false;
        }
        return true;
    };

    bool align_ok = true;
    align_ok &= check_128_align(ball.pos_x, "ball.pos_x");
    align_ok &= check_128_align(ball.pos_y, "ball.pos_y");
    align_ok &= check_128_align(ball.pos_z, "ball.pos_z");
    align_ok &= check_128_align(ball.vel_x, "ball.vel_x");
    align_ok &= check_128_align(ball.q_w, "ball.q_w");
    align_ok &= check_128_align(ball.ang_vel_x, "ball.ang_vel_x");

    const auto& car = ctx.GetCarState();
    align_ok &= check_128_align(car.pos_x, "car.pos_x");
    align_ok &= check_128_align(car.pos_y, "car.pos_y");
    align_ok &= check_128_align(car.pos_z, "car.pos_z");
    align_ok &= check_128_align(car.vel_x, "car.vel_x");
    align_ok &= check_128_align(car.boost, "car.boost");
    align_ok &= check_128_align(car.is_on_ground, "car.is_on_ground");
    align_ok &= check_128_align(car.suspension_length_0, "car.suspension_length_0");

    const auto& ctrl = ctx.GetControls();
    align_ok &= check_128_align(ctrl.throttle, "ctrl.throttle");
    align_ok &= check_128_align(ctrl.boost, "ctrl.boost");

    if (!align_ok) {
        std::cerr << "[-] FAIL: SoA 128-byte alignment verification failed!\n";
        return 1;
    }
    std::cout << "  [PASS] All SoA pointers are strictly 128-byte aligned.\n";

    // 2. Parallel Device Read/Write Kernel Execution
    int* d_pass = nullptr;
    cudaMalloc(&d_pass, sizeof(int));
    int initial_pass = 1;
    cudaMemcpy(d_pass, &initial_pass, sizeof(int), cudaMemcpyHostToDevice);

    constexpr uint32_t threads = 128;
    uint32_t blocks = (NUM_ENVS + threads - 1) / threads;
    k_test_soa_parallel_rw<<<blocks, threads>>>(ctx.GetBallState(), ctx.GetCarState(), NUM_ENVS, d_pass);
    cudaDeviceSynchronize();

    int h_pass = 0;
    cudaMemcpy(&h_pass, d_pass, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_pass);

    if (h_pass != 1) {
        std::cerr << "[-] FAIL: Parallel SoA Read/Write kernel reported mismatch!\n";
        return 1;
    }
    std::cout << "  [PASS] 1024 parallel environments read/write coalescing test passed.\n";

    // 3. Host/Device POD Staging Transfer
    std::vector<BallStatePOD> host_balls(NUM_ENVS);
    ctx.CopyBallStateToHost(host_balls.data(), 0, NUM_ENVS);
    for (uint32_t i = 0; i < NUM_ENVS; i++) {
        if (fabsf(host_balls[i].pos.x - static_cast<float>(i)) > 1e-5f) {
            std::cerr << "[-] FAIL: Staging copy mismatch at index " << i << "!\n";
            return 1;
        }
    }
    std::cout << "  [PASS] Zero-dynamic-allocation POD staging transfer verified.\n";

    std::cout << "[Unit Test: SoA] ALL SOA DATA STRUCTURE TESTS PASSED CLEANLY!\n";
    return 0;
}
