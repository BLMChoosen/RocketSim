#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <cuda_runtime.h>

#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/sim_context.cuh"
#include "rocketsim_cuda/math/vec3.cuh"
#include "rocketsim_cuda/math/quat.cuh"
#include "rocketsim_cuda/math/mat3.cuh"
#include "rocketsim_cuda/types/car_state.cuh"
#include "rocketsim_cuda/types/ball_state.cuh"
#include "rocketsim_cuda/types/car_controls.cuh"

using namespace rocketsim_cuda;

static inline size_t align_128(size_t size) {
    return (size + 127) & ~size_t(127);
}

// Device Kernel for Stress Testing Math Primitives on GPU
__global__ void k_device_math_stress(int* out_pass) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) return;

    // High Velocity Vec3 Tests
    float high_speeds[] = {2300.0f, 6000.0f, 20000.0f, 100000.0f, 1000000.0f};
    for (float speed : high_speeds) {
        Vec3 v(speed, -speed, speed * 0.5f);
        float expected_len = sqrtf(speed * speed + speed * speed + speed * speed * 0.25f);
        if (fabsf(v.length() - expected_len) > expected_len * 1e-5f) {
            *out_pass = 0;
            return;
        }

        Vec3 n = v.normalized();
        if (fabsf(n.length() - 1.0f) > 1e-5f) {
            *out_pass = 0;
            return;
        }
    }

    // Near-Zero and Zero Vec3 Normalization
    Vec3 v_zero(0.0f, 0.0f, 0.0f);
    Vec3 n_zero = v_zero.normalized();
    if (n_zero.x != 0.0f || n_zero.y != 0.0f || n_zero.z != 0.0f) {
        *out_pass = 0;
        return;
    }

    // Antipodal Quat & Rodrigues Rotation Consistency
    float angles[] = {0.0f, 1e-6f, 0.785398f, 1.570796f, 3.14159265f};
    Vec3 axes[] = {
        Vec3(1.0f, 0.0f, 0.0f),
        Vec3(0.0f, 1.0f, 0.0f),
        Vec3(0.0f, 0.0f, 1.0f),
        Vec3(0.57735f, 0.57735f, 0.57735f)
    };

    Vec3 test_vectors[] = {
        Vec3(100.0f, 200.0f, 300.0f),
        Vec3(-2300.0f, 1500.0f, 60.0f),
        Vec3(6000.0f, -4000.0f, 2048.0f)
    };

    for (Vec3 axis : axes) {
        for (float ang : angles) {
            float half = ang * 0.5f;
            Quat q(cosf(half), axis.x * sinf(half), axis.y * sinf(half), axis.z * sinf(half));
            Quat q_antipodal(-q.w, -q.x, -q.y, -q.z);

            if (q.chebyshev_dist(q_antipodal) > 1e-6f) {
                *out_pass = 0;
                return;
            }

            for (Vec3 tv : test_vectors) {
                Vec3 r1 = q.rotate(tv);
                Vec3 r2 = q_antipodal.rotate(tv);
                if (r1.chebyshev_dist(r2) > 1e-4f) {
                    *out_pass = 0;
                    return;
                }
                if (fabsf(r1.length() - tv.length()) > tv.length() * 1e-5f) {
                    *out_pass = 0;
                    return;
                }
            }
        }
    }

    *out_pass = 1;
}

// Parallel Coalesced Read/Write Kernel across arbitrary Env counts
__global__ void k_soa_stress_rw(BallStateSoA ball, CarStateSoA car, CarControlsSoA ctrl,
                                uint32_t num_envs, uint32_t total_cars, int* out_pass) {
    uint32_t env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx < num_envs) {
        ball.pos_x[env_idx] = static_cast<float>(env_idx) * 1.5f;
        ball.pos_y[env_idx] = -static_cast<float>(env_idx) * 2.5f;
        ball.pos_z[env_idx] = 93.15f + static_cast<float>(env_idx);
        ball.vel_x[env_idx] = 100.0f;
        ball.q_w[env_idx] = 1.0f;
    }

    uint32_t car_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (car_idx < total_cars) {
        car.pos_x[car_idx] = static_cast<float>(car_idx) * 3.0f;
        car.pos_y[car_idx] = static_cast<float>(car_idx) * 4.0f;
        car.pos_z[car_idx] = 17.0f;
        car.boost[car_idx] = static_cast<float>(car_idx % 100);
        car.is_on_ground[car_idx] = (car_idx % 2 == 0) ? 1 : 0;
        car.suspension_length_0[car_idx] = 12.34f;
        car.has_jumped[car_idx] = (car_idx % 3 == 0) ? 1 : 0;
        car.has_double_jumped[car_idx] = (car_idx % 5 == 0) ? 1 : 0;
        car.ball_hit_tick_count[car_idx] = 1000ULL + car_idx;

        ctrl.throttle[car_idx] = 0.75f;
        ctrl.boost[car_idx] = 1;
    }
}

__global__ void k_soa_stress_verify(BallStateSoA ball, CarStateSoA car, CarControlsSoA ctrl,
                                    uint32_t num_envs, uint32_t total_cars, int* out_pass) {
    uint32_t env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx < num_envs) {
        float expected_px = static_cast<float>(env_idx) * 1.5f;
        float expected_py = -static_cast<float>(env_idx) * 2.5f;
        float expected_pz = 93.15f + static_cast<float>(env_idx);
        if (fabsf(ball.pos_x[env_idx] - expected_px) > 1e-4f ||
            fabsf(ball.pos_y[env_idx] - expected_py) > 1e-4f ||
            fabsf(ball.pos_z[env_idx] - expected_pz) > 1e-4f ||
            fabsf(ball.vel_x[env_idx] - 100.0f) > 1e-4f ||
            fabsf(ball.q_w[env_idx] - 1.0f) > 1e-4f) {
            atomicExch(out_pass, 0);
        }
    }

    uint32_t car_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (car_idx < total_cars) {
        float expected_cx = static_cast<float>(car_idx) * 3.0f;
        float expected_cy = static_cast<float>(car_idx) * 4.0f;
        float expected_boost = static_cast<float>(car_idx % 100);
        uint8_t expected_ground = (car_idx % 2 == 0) ? 1 : 0;
        uint8_t expected_jump = (car_idx % 3 == 0) ? 1 : 0;
        uint8_t expected_double_jump = (car_idx % 5 == 0) ? 1 : 0;

        if (fabsf(car.pos_x[car_idx] - expected_cx) > 1e-4f ||
            fabsf(car.pos_y[car_idx] - expected_cy) > 1e-4f ||
            fabsf(car.boost[car_idx] - expected_boost) > 1e-4f ||
            car.is_on_ground[car_idx] != expected_ground ||
            car.has_jumped[car_idx] != expected_jump ||
            car.has_double_jumped[car_idx] != expected_double_jump ||
            car.ball_hit_tick_count[car_idx] != (1000ULL + car_idx) ||
            fabsf(ctrl.throttle[car_idx] - 0.75f) > 1e-4f ||
            ctrl.boost[car_idx] != 1) {
            atomicExch(out_pass, 0);
        }
    }
}

bool verify_sim_context_alignment_and_bounds(SimContext& ctx, uint32_t num_envs, uint32_t cars_per_env) {
    uint32_t total_cars = num_envs * cars_per_env;

    auto is_128_aligned = [](const void* ptr, const char* name) -> bool {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        if (addr % 128 != 0) {
            std::cerr << "[-] ALIGNMENT VIOLATION: " << name << " at " << ptr
                      << " (offset " << (addr % 128) << " != 0)\n";
            return false;
        }
        return true;
    };

    bool ok = true;
    const auto& b = ctx.GetBallState();
    ok &= is_128_aligned(b.pos_x, "b.pos_x");
    ok &= is_128_aligned(b.pos_y, "b.pos_y");
    ok &= is_128_aligned(b.pos_z, "b.pos_z");
    ok &= is_128_aligned(b.vel_x, "b.vel_x");
    ok &= is_128_aligned(b.vel_y, "b.vel_y");
    ok &= is_128_aligned(b.vel_z, "b.vel_z");
    ok &= is_128_aligned(b.q_w, "b.q_w");
    ok &= is_128_aligned(b.q_x, "b.q_x");
    ok &= is_128_aligned(b.q_y, "b.q_y");
    ok &= is_128_aligned(b.q_z, "b.q_z");
    ok &= is_128_aligned(b.ang_vel_x, "b.ang_vel_x");
    ok &= is_128_aligned(b.ang_vel_y, "b.ang_vel_y");
    ok &= is_128_aligned(b.ang_vel_z, "b.ang_vel_z");
    ok &= is_128_aligned(b.hs_y_target_dir, "b.hs_y_target_dir");
    ok &= is_128_aligned(b.hs_cur_target_speed, "b.hs_cur_target_speed");
    ok &= is_128_aligned(b.hs_time_since_hit, "b.hs_time_since_hit");
    ok &= is_128_aligned(b.ds_charge_level, "b.ds_charge_level");
    ok &= is_128_aligned(b.ds_accumulated_hit_force, "b.ds_accumulated_hit_force");
    ok &= is_128_aligned(b.ds_y_target_dir, "b.ds_y_target_dir");
    ok &= is_128_aligned(b.ds_has_damaged, "b.ds_has_damaged");
    ok &= is_128_aligned(b.ds_last_damage_tick, "b.ds_last_damage_tick");

    const auto& c = ctx.GetCarState();
    ok &= is_128_aligned(c.pos_x, "c.pos_x");
    ok &= is_128_aligned(c.pos_y, "c.pos_y");
    ok &= is_128_aligned(c.pos_z, "c.pos_z");
    ok &= is_128_aligned(c.vel_x, "c.vel_x");
    ok &= is_128_aligned(c.vel_y, "c.vel_y");
    ok &= is_128_aligned(c.vel_z, "c.vel_z");
    ok &= is_128_aligned(c.q_w, "c.q_w");
    ok &= is_128_aligned(c.q_x, "c.q_x");
    ok &= is_128_aligned(c.q_y, "c.q_y");
    ok &= is_128_aligned(c.q_z, "c.q_z");
    ok &= is_128_aligned(c.ang_vel_x, "c.ang_vel_x");
    ok &= is_128_aligned(c.ang_vel_y, "c.ang_vel_y");
    ok &= is_128_aligned(c.ang_vel_z, "c.ang_vel_z");

    ok &= is_128_aligned(c.boost, "c.boost");
    ok &= is_128_aligned(c.time_since_boosted, "c.time_since_boosted");
    ok &= is_128_aligned(c.boosting_time, "c.boosting_time");
    ok &= is_128_aligned(c.is_boosting, "c.is_boosting");

    ok &= is_128_aligned(c.is_on_ground, "c.is_on_ground");
    ok &= is_128_aligned(c.wheel_contact_0, "c.wheel_contact_0");
    ok &= is_128_aligned(c.wheel_contact_1, "c.wheel_contact_1");
    ok &= is_128_aligned(c.wheel_contact_2, "c.wheel_contact_2");
    ok &= is_128_aligned(c.wheel_contact_3, "c.wheel_contact_3");
    ok &= is_128_aligned(c.suspension_length_0, "c.suspension_length_0");
    ok &= is_128_aligned(c.suspension_length_1, "c.suspension_length_1");
    ok &= is_128_aligned(c.suspension_length_2, "c.suspension_length_2");
    ok &= is_128_aligned(c.suspension_length_3, "c.suspension_length_3");

    ok &= is_128_aligned(c.has_jumped, "c.has_jumped");
    ok &= is_128_aligned(c.is_jumping, "c.is_jumping");
    ok &= is_128_aligned(c.jump_time, "c.jump_time");
    ok &= is_128_aligned(c.has_double_jumped, "c.has_double_jumped");
    ok &= is_128_aligned(c.air_time, "c.air_time");
    ok &= is_128_aligned(c.air_time_since_jump, "c.air_time_since_jump");

    ok &= is_128_aligned(c.has_flipped, "c.has_flipped");
    ok &= is_128_aligned(c.is_flipping, "c.is_flipping");
    ok &= is_128_aligned(c.flip_time, "c.flip_time");
    ok &= is_128_aligned(c.flip_rel_torque_x, "c.flip_rel_torque_x");
    ok &= is_128_aligned(c.flip_rel_torque_y, "c.flip_rel_torque_y");
    ok &= is_128_aligned(c.flip_rel_torque_z, "c.flip_rel_torque_z");

    ok &= is_128_aligned(c.is_auto_flipping, "c.is_auto_flipping");
    ok &= is_128_aligned(c.auto_flip_timer, "c.auto_flip_timer");
    ok &= is_128_aligned(c.auto_flip_torque_scale, "c.auto_flip_torque_scale");
    ok &= is_128_aligned(c.handbrake_val, "c.handbrake_val");

    ok &= is_128_aligned(c.is_supersonic, "c.is_supersonic");
    ok &= is_128_aligned(c.supersonic_time, "c.supersonic_time");
    ok &= is_128_aligned(c.is_demoed, "c.is_demoed");
    ok &= is_128_aligned(c.demo_respawn_timer, "c.demo_respawn_timer");

    ok &= is_128_aligned(c.world_contact_has_contact, "c.world_contact_has_contact");
    ok &= is_128_aligned(c.world_contact_normal_x, "c.world_contact_normal_x");
    ok &= is_128_aligned(c.world_contact_normal_y, "c.world_contact_normal_y");
    ok &= is_128_aligned(c.world_contact_normal_z, "c.world_contact_normal_z");
    ok &= is_128_aligned(c.car_contact_other_car_id, "c.car_contact_other_car_id");
    ok &= is_128_aligned(c.car_contact_cooldown_timer, "c.car_contact_cooldown_timer");

    ok &= is_128_aligned(c.ball_hit_is_valid, "c.ball_hit_is_valid");
    ok &= is_128_aligned(c.ball_hit_rel_pos_x, "c.ball_hit_rel_pos_x");
    ok &= is_128_aligned(c.ball_hit_rel_pos_y, "c.ball_hit_rel_pos_y");
    ok &= is_128_aligned(c.ball_hit_rel_pos_z, "c.ball_hit_rel_pos_z");
    ok &= is_128_aligned(c.ball_hit_extra_hit_force_x, "c.ball_hit_extra_hit_force_x");
    ok &= is_128_aligned(c.ball_hit_extra_hit_force_y, "c.ball_hit_extra_hit_force_y");
    ok &= is_128_aligned(c.ball_hit_extra_hit_force_z, "c.ball_hit_extra_hit_force_z");
    ok &= is_128_aligned(c.ball_hit_tick_count, "c.ball_hit_tick_count");

    ok &= is_128_aligned(c.last_controls_throttle, "c.last_controls_throttle");
    ok &= is_128_aligned(c.last_controls_steer, "c.last_controls_steer");
    ok &= is_128_aligned(c.last_controls_pitch, "c.last_controls_pitch");
    ok &= is_128_aligned(c.last_controls_yaw, "c.last_controls_yaw");
    ok &= is_128_aligned(c.last_controls_roll, "c.last_controls_roll");
    ok &= is_128_aligned(c.last_controls_boost, "c.last_controls_boost");
    ok &= is_128_aligned(c.last_controls_jump, "c.last_controls_jump");
    ok &= is_128_aligned(c.last_controls_handbrake, "c.last_controls_handbrake");

    const auto& ctrl = ctx.GetControls();
    ok &= is_128_aligned(ctrl.throttle, "ctrl.throttle");
    ok &= is_128_aligned(ctrl.steer, "ctrl.steer");
    ok &= is_128_aligned(ctrl.pitch, "ctrl.pitch");
    ok &= is_128_aligned(ctrl.yaw, "ctrl.yaw");
    ok &= is_128_aligned(ctrl.roll, "ctrl.roll");
    ok &= is_128_aligned(ctrl.boost, "ctrl.boost");
    ok &= is_128_aligned(ctrl.jump, "ctrl.jump");
    ok &= is_128_aligned(ctrl.handbrake, "ctrl.handbrake");

    uintptr_t last_ctrl_slice_end = reinterpret_cast<uintptr_t>(ctrl.handbrake) + align_128(total_cars * sizeof(uint8_t));
    uintptr_t pool_start = reinterpret_cast<uintptr_t>(b.pos_x);
    uintptr_t pool_end = pool_start + ctx.GetAllocatedBytes();

    size_t staging_bytes = align_128(num_envs * sizeof(BallStatePOD))
                         + align_128(total_cars * sizeof(CarStatePOD))
                         + align_128(total_cars * sizeof(CarControls));
    uintptr_t staging_start = pool_end - staging_bytes;

    if (last_ctrl_slice_end > staging_start) {
        std::cerr << "[-] OVERLAP VIOLATION: last SoA slice ends at " << last_ctrl_slice_end
                  << " but staging begins at " << staging_start << "\n";
        return false;
    }

    if (staging_start % 128 != 0) {
        std::cerr << "[-] STAGING ALIGNMENT VIOLATION: staging_start " << staging_start << " is not 128 aligned!\n";
        return false;
    }

    return ok;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "        CHALLENGER 2: EMPIRICAL STRESS & INVARIANT VERIFICATION       \n";
    std::cout << "======================================================================\n\n";

    bool all_passed = true;

    // ------------------------------------------------------------------------
    // Part 1: Verify 128-byte alignment across edge environment counts
    // ------------------------------------------------------------------------
    std::cout << "[Part 1] Stress-Testing 128-Byte Alignment & Memory Bounds...\n";
    struct EnvConfig { uint32_t envs; uint32_t cars; };
    EnvConfig test_configs[] = {
        {1, 1},
        {33, 1},
        {33, 8},
        {1024, 1},
        {1024, 8},
        {32768, 1},
        {65536, 1}
    };

    int* d_pass = nullptr;
    cudaMalloc(&d_pass, sizeof(int));

    for (const auto& cfg : test_configs) {
        std::cout << "  Testing SimContext(envs=" << cfg.envs << ", cars_per_env=" << cfg.cars << "): ";
        try {
            SimContext ctx(cfg.envs, cfg.cars);
            size_t bytes = ctx.GetAllocatedBytes();
            bool align_ok = verify_sim_context_alignment_and_bounds(ctx, cfg.envs, cfg.cars);

            if (!align_ok) {
                std::cout << "[FAIL - Alignment / Bounds]\n";
                all_passed = false;
                continue;
            }

            int h_init = 1;
            cudaMemcpy(d_pass, &h_init, sizeof(int), cudaMemcpyHostToDevice);

            constexpr uint32_t threads = 128;
            uint32_t max_items = (cfg.envs > cfg.envs * cfg.cars) ? cfg.envs : (cfg.envs * cfg.cars);
            uint32_t blocks = (max_items + threads - 1) / threads;

            k_soa_stress_rw<<<blocks, threads>>>(ctx.GetBallState(), ctx.GetCarState(), ctx.GetControls(),
                                                 cfg.envs, cfg.envs * cfg.cars, d_pass);
            cudaDeviceSynchronize();

            k_soa_stress_verify<<<blocks, threads>>>(ctx.GetBallState(), ctx.GetCarState(), ctx.GetControls(),
                                                     cfg.envs, cfg.envs * cfg.cars, d_pass);
            cudaDeviceSynchronize();

            int h_res = 0;
            cudaMemcpy(&h_res, d_pass, sizeof(int), cudaMemcpyDeviceToHost);

            if (h_res == 1) {
                std::cout << "[PASS] (" << std::fixed << std::setprecision(2)
                          << (bytes / 1024.0f) << " KB, 73 pointers + staging strictly 128-byte aligned, zero overlap)\n";
            } else {
                std::cout << "[FAIL - R/W Corruption]\n";
                all_passed = false;
            }
        } catch (const std::exception& e) {
            std::cout << "[EXCEPTION: " << e.what() << "]\n";
            all_passed = false;
        }
    }

    // ------------------------------------------------------------------------
    // Part 2: Verify Zero Dynamic Memory Allocation in Simulation Loops
    // ------------------------------------------------------------------------
    std::cout << "\n[Part 2] Verifying Zero Dynamic Device Allocation in Simulation Loops...\n";
    {
        SimContext ctx(1024, 1);
        std::vector<BallStatePOD> ball_host(1024);
        std::vector<CarStatePOD> car_host(1024);
        std::vector<CarControls> ctrl_host(1024);

        for (int i = 0; i < 100; i++) {
            ctx.CopyBallStateToDevice(ball_host.data(), 0, 1024);
            ctx.CopyCarStateToDevice(car_host.data(), 0, 1024);
            ctx.CopyControlsToDevice(ctrl_host.data(), 0, 1024);
            ctx.CopyBallStateToHost(ball_host.data(), 0, 1024);
            ctx.CopyCarStateToHost(car_host.data(), 0, 1024);
        }

        size_t free_before = 0, total_mem = 0;
        cudaMemGetInfo(&free_before, &total_mem);

        constexpr int NUM_ITERATIONS = 10000;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            ctx.CopyBallStateToDevice(ball_host.data(), 0, 1024);
            ctx.CopyCarStateToDevice(car_host.data(), 0, 1024);
            ctx.CopyControlsToDevice(ctrl_host.data(), 0, 1024);
            ctx.CopyBallStateToHost(ball_host.data(), 0, 1024);
            ctx.CopyCarStateToHost(car_host.data(), 0, 1024);
        }

        size_t free_after = 0;
        cudaMemGetInfo(&free_after, &total_mem);

        if (free_before == free_after) {
            std::cout << "  [PASS] 10,000 step loops executed with exactly 0 bytes device allocation delta!\n";
        } else {
            std::cout << "  [FAIL] Memory allocation occurred during loop: delta = "
                      << (free_before - free_after) << " bytes\n";
            all_passed = false;
        }
    }

    // ------------------------------------------------------------------------
    // Part 3: Stress-Testing Math Primitives
    // ------------------------------------------------------------------------
    std::cout << "\n[Part 3] Stress-Testing Math Primitives (Extreme Ranges, Antipodal & Degenerate)...\n";
    {
        Quat q_base(0.70710678f, 0.70710678f, 0.0f, 0.0f);
        Quat q_antipodal(-q_base.w, -q_base.x, -q_base.y, -q_base.z);
        float d_antipodal = q_base.chebyshev_dist(q_antipodal);
        assert(d_antipodal < 1e-6f);
        std::cout << "  [PASS] Host Antipodal Chebyshev metric (distance = " << d_antipodal << ")\n";

        Vec3 pinch_vel(6000.0f, 6000.0f, -6000.0f);
        Vec3 rot1 = q_base.rotate(pinch_vel);
        Vec3 rot2 = q_antipodal.rotate(pinch_vel);
        float rot_diff = rot1.chebyshev_dist(rot2);
        assert(rot_diff < 1e-4f);
        std::cout << "  [PASS] Rodrigues rotation on 10,000+ UU/s pinch velocity (delta = " << rot_diff << ")\n";

        // Monte Carlo Oracle: Random Rotations Round-Trip Test with Worker's to_quat()
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        float max_roundtrip_err = 0.0f;
        int failed_rotations = 0;
        constexpr int NUM_RANDOM_ROTS = 1000;

        for (int i = 0; i < NUM_RANDOM_ROTS; i++) {
            Quat q_rnd(dist(rng), dist(rng), dist(rng), dist(rng));
            q_rnd = q_rnd.normalized();

            Mat3 m_rnd = Mat3::from_quat(q_rnd);
            Quat q_recovered = m_rnd.to_quat();

            float err = q_rnd.chebyshev_dist(q_recovered);
            if (err > max_roundtrip_err) max_roundtrip_err = err;

            if (err > 1e-4f) {
                failed_rotations++;
            }
        }

        if (failed_rotations > 0) {
            std::cerr << "  [FAIL - BUG CONFIRMED] Mat3::to_quat failed " << failed_rotations
                      << " of " << NUM_RANDOM_ROTS << " random rotations! Max Error = "
                      << max_roundtrip_err << " (Threshold <= 1e-4)\n"
                      << "         Root Cause: Off-diagonal subtraction signs in Mat3::to_quat() are inverted\n"
                      << "         (e.g. up.y - right.z produces -4wx instead of +4wx, generating conjugate/inverse quaternions).\n";
            all_passed = false;
        } else {
            std::cout << "  [PASS] Random orientations round-trip: Max Error = "
                      << max_roundtrip_err << "\n";
        }

        int h_pass = 0;
        cudaMemcpy(d_pass, &h_pass, sizeof(int), cudaMemcpyHostToDevice);
        k_device_math_stress<<<1, 32>>>(d_pass);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_pass, d_pass, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_pass == 1) {
            std::cout << "  [PASS] Device CUDA math stress kernel (high velocity, antipodal, Rodrigues)\n";
        } else {
            std::cout << "  [FAIL] Device CUDA math stress kernel reported failure!\n";
            all_passed = false;
        }
    }

    cudaFree(d_pass);

    std::cout << "\n======================================================================\n";
    if (all_passed) {
        std::cout << "  ALL EMPIRICAL CHALLENGES PASSED CLEANLY! VERDICT: APPROVE           \n";
    } else {
        std::cout << "  CHALLENGES FAILED! VERDICT: REQUEST_CHANGES                         \n";
    }
    std::cout << "======================================================================\n";

    return all_passed ? 0 : 1;
}
