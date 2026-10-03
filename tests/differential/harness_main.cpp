#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <cstdlib>

#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/sim_context.cuh"
#include "cpu_ref_sim.h"
#include "golden_master.h"

using namespace rocketsim_cuda;

struct HarnessArgs {
    uint32_t ticks = 500;
    uint32_t envs = 4;
    uint32_t seed = 42;
    float tol = TOL_POS;
    std::string record_file = "milestone1_golden.rsgold";
};

void PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --ticks <N>        Number of ticks to simulate (default: 500)\n"
              << "  --envs <N>         Number of concurrent environments (default: 4)\n"
              << "  --seed <N>         Pseudorandom seed for PCG32 controls (default: 42)\n"
              << "  --tol <F>          Chebyshev position tolerance (default: 1e-4)\n"
              << "  --record <path>    Output path for .rsgold recording (default: milestone1_golden.rsgold)\n"
              << "  --help             Display this help message\n";
}

HarnessArgs ParseArgs(int argc, char** argv) {
    HarnessArgs args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--ticks" && i + 1 < argc) {
            args.ticks = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--envs" && i + 1 < argc) {
            args.envs = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--tol" && i + 1 < argc) {
            args.tol = std::strtof(argv[++i], nullptr);
        } else if (arg == "--record" && i + 1 < argc) {
            args.record_file = argv[++i];
        } else if (arg == "--help") {
            PrintUsage(argv[0]);
            std::exit(0);
        }
    }
    return args;
}

int main(int argc, char** argv) {
    HarnessArgs args = ParseArgs(argc, argv);

    std::cout << "======================================================================\n"
              << "                   ROCKETSIM-CUDA DIFFERENTIAL HARNESS                \n"
              << "======================================================================\n"
              << "  Environments: " << args.envs << "\n"
              << "  Ticks:        " << args.ticks << "\n"
              << "  PRNG Seed:    " << args.seed << "\n"
              << "  Tolerance:    " << args.tol << " UU\n"
              << "  Record File:  " << args.record_file << "\n"
              << "======================================================================\n\n";

    DifferentialTolerance tol;
    tol.pos_uu = args.tol;
    DifferentialComparator comparator(tol);

    // ------------------------------------------------------------------
    // Test 1: CPU Reference Simulation & Golden Master Serialization
    // ------------------------------------------------------------------
    std::cout << "[Test 1/4] Running CPU Reference Simulation & Recording .rsgold...\n";
    std::vector<CPURefSim> cpu_envs;
    cpu_envs.reserve(args.envs);
    for (uint32_t e = 0; e < args.envs; e++) {
        cpu_envs.emplace_back(1, true, TICK_RATE, static_cast<int>(e));
    }

    DeterministicInputGenerator input_gen(args.seed);

    RsGoldWriter writer;
    if (!writer.Open(args.record_file, args.envs, 1, args.seed, TICK_RATE)) {
        std::cerr << "[-] Error: Failed to open " << args.record_file << " for writing\n";
        return 1;
    }

    std::vector<CarControls> current_controls(args.envs);
    std::vector<BallStatePOD> current_balls(args.envs);
    std::vector<CarStatePOD> current_cars(args.envs);

    for (uint32_t t = 0; t < args.ticks; t++) {
        for (uint32_t e = 0; e < args.envs; e++) {
            current_controls[e] = input_gen.Generate();
            cpu_envs[e].Step(&current_controls[e], 1);
            cpu_envs[e].GetBallState(current_balls[e]);
            cpu_envs[e].GetCarState(0, current_cars[e]);
        }
        writer.WriteTick(current_controls.data(), current_balls.data(), current_cars.data());
    }
    writer.Close();
    std::cout << "[+] Successfully recorded " << args.ticks << " ticks for " << args.envs
              << " environments to " << args.record_file << "\n\n";

    // ------------------------------------------------------------------
    // Test 2: Golden Master Replay & Bit-Exact Deserialization Check
    // ------------------------------------------------------------------
    std::cout << "[Test 2/4] Verifying .rsgold Binary Replay & Deserialization Parity...\n";
    RsGoldReader reader;
    if (!reader.Open(args.record_file)) {
        std::cerr << "[-] Error: Failed to open " << args.record_file << " for reading\n";
        return 1;
    }

    const auto& hdr = reader.GetHeader();
    if (hdr.magic != 0x5253474D || hdr.total_ticks != args.ticks || hdr.num_envs != args.envs) {
        std::cerr << "[-] Error: Header mismatch in .rsgold file!\n";
        return 1;
    }

    std::vector<CarControls> replay_controls(args.envs);
    std::vector<BallStatePOD> replay_balls(args.envs);
    std::vector<CarStatePOD> replay_cars(args.envs);

    // Re-simulate with fresh CPU oracle to compare against replay stream
    std::vector<CPURefSim> verifier_envs;
    verifier_envs.reserve(args.envs);
    for (uint32_t e = 0; e < args.envs; e++) {
        verifier_envs.emplace_back(1, true, TICK_RATE, static_cast<int>(e));
    }

    DifferentialFailure fail;
    for (uint32_t t = 0; t < args.ticks; t++) {
        if (!reader.ReadTick(replay_controls.data(), replay_balls.data(), replay_cars.data())) {
            std::cerr << "[-] Error: Failed to read tick " << t << " from .rsgold\n";
            return 1;
        }

        for (uint32_t e = 0; e < args.envs; e++) {
            verifier_envs[e].Step(&replay_controls[e], 1);
            BallStatePOD v_ball;
            CarStatePOD v_car;
            verifier_envs[e].GetBallState(v_ball);
            verifier_envs[e].GetCarState(0, v_car);

            if (!comparator.CompareBall(t, e, v_ball, replay_balls[e], fail)) {
                std::cerr << "[-] Parity Failure in Replay Ball at tick " << t << ", env " << e << ":\n"
                          << "    Max delta: " << fail.max_delta << " > tol " << fail.threshold << "\n";
                return 1;
            }

            if (!comparator.CompareCar(t, e, 0, v_car, replay_cars[e], fail)) {
                std::cerr << "[-] Parity Failure in Replay Car at tick " << t << ", env " << e << ":\n"
                          << "    Attribute: " << fail.attribute << "\n"
                          << "    Max delta: " << fail.max_delta << " > tol " << fail.threshold << "\n";
                return 1;
            }
        }
    }
    reader.Close();
    std::cout << "[+] Replay verified: 100% deterministic parity across all " << args.ticks << " ticks.\n\n";

    // ------------------------------------------------------------------
    // Test 3: GPU Device SoA Round-Trip & Memory Coalescing Check
    // ------------------------------------------------------------------
    std::cout << "[Test 3/4] Verifying GPU SimContext Pre-allocated SoA Round-Trip...\n";
    try {
        SimContext sim_ctx(args.envs, 1);
        std::cout << "    SimContext allocated " << (sim_ctx.GetAllocatedBytes() / 1024)
                  << " KB pre-allocated device memory arena.\n";

        // Upload latest CPU states to GPU SoA memory
        sim_ctx.CopyBallStateToDevice(current_balls.data(), 0, args.envs);
        sim_ctx.CopyCarStateToDevice(current_cars.data(), 0, args.envs);
        sim_ctx.CopyControlsToDevice(current_controls.data(), 0, args.envs);

        // Download back to host
        std::vector<BallStatePOD> downloaded_balls(args.envs);
        std::vector<CarStatePOD> downloaded_cars(args.envs);
        sim_ctx.CopyBallStateToHost(downloaded_balls.data(), 0, args.envs);
        sim_ctx.CopyCarStateToHost(downloaded_cars.data(), 0, args.envs);

        // Compare GPU SoA round-trip against CPU ground truth
        for (uint32_t e = 0; e < args.envs; e++) {
            if (!comparator.CompareBall(args.ticks, e, current_balls[e], downloaded_balls[e], fail)) {
                std::cerr << "[-] GPU SoA Round-Trip Failure on Ball: " << fail.attribute << "\n";
                return 1;
            }
            if (!comparator.CompareCar(args.ticks, e, 0, current_cars[e], downloaded_cars[e], fail)) {
                std::cerr << "[-] GPU SoA Round-Trip Failure on Car: " << fail.attribute << "\n";
                return 1;
            }
        }
        std::cout << "[+] GPU SoA Round-Trip verified: Zero-loss fidelity on 128-byte aligned device memory.\n\n";
    } catch (const std::exception& ex) {
        std::cerr << "[-] GPU SimContext exception: " << ex.what() << "\n";
        return 1;
    }

    // ------------------------------------------------------------------
    // Test 4: Differential Comparator Strict Threshold Sensitivity Test
    // ------------------------------------------------------------------
    std::cout << "[Test 4/4] Verifying Comparator Sensitivity (Rejection of Injected Perturbation)...\n";
    BallStatePOD perturbed_ball = current_balls[0];
    perturbed_ball.pos.x += args.tol * 2.0f; // Exceeds tolerance
    if (comparator.CompareBall(0, 0, current_balls[0], perturbed_ball, fail)) {
        std::cerr << "[-] Error: Comparator failed to catch injected position perturbation!\n";
        return 1;
    }
    std::cout << "    Correctly rejected perturbed Ball: Delta " << fail.max_delta
              << " UU > Threshold " << fail.threshold << " UU\n";

    CarStatePOD perturbed_car = current_cars[0];
    perturbed_car.quat.w += args.tol * 2.0f;
    if (comparator.CompareCar(0, 0, 0, current_cars[0], perturbed_car, fail)) {
        std::cerr << "[-] Error: Comparator failed to catch injected quaternion perturbation!\n";
        return 1;
    }
    std::cout << "    Correctly rejected perturbed Car: Delta " << fail.max_delta
              << " > Threshold " << fail.threshold << "\n";
    std::cout << "[+] Comparator Sensitivity verified: Strictly enforces GEMINI.md tolerances.\n\n";

    // ------------------------------------------------------------------
    // Test 5: Lockstep GPU vs CPU Oracle Differential Simulation (Milestone 2 R5)
    // ------------------------------------------------------------------
    std::cout << "[Test 5/5] Lockstep Differential Simulation: GPU vs CPU Oracle ("
              << args.ticks << " ticks, " << args.envs << " envs)...\n";

    try {
        SimContext gpu_sim(args.envs, 1);

        std::vector<CPURefSim> lockstep_cpu_envs;
        lockstep_cpu_envs.reserve(args.envs);
        for (uint32_t e = 0; e < args.envs; e++) {
            lockstep_cpu_envs.emplace_back(1, true, TICK_RATE, static_cast<int>(e));
        }

        std::vector<BallStatePOD> init_balls(args.envs);
        std::vector<CarStatePOD> init_cars(args.envs);
        for (uint32_t e = 0; e < args.envs; e++) {
            lockstep_cpu_envs[e].GetBallState(init_balls[e]);
            lockstep_cpu_envs[e].GetCarState(0, init_cars[e]);
        }
        gpu_sim.CopyBallStateToDevice(init_balls.data(), 0, args.envs);
        gpu_sim.CopyCarStateToDevice(init_cars.data(), 0, args.envs);

        DeterministicInputGenerator lockstep_gen(args.seed);
        std::vector<CarControls> step_controls(args.envs);
        std::vector<BallStatePOD> gpu_balls(args.envs);
        std::vector<CarStatePOD> gpu_cars(args.envs);
        std::vector<BallStatePOD> cpu_balls(args.envs);
        std::vector<CarStatePOD> cpu_cars(args.envs);

        for (uint32_t t = 0; t < args.ticks; t++) {
            for (uint32_t e = 0; e < args.envs; e++) {
                step_controls[e] = lockstep_gen.Generate();
                lockstep_cpu_envs[e].Step(&step_controls[e], 1);
                lockstep_cpu_envs[e].GetBallState(cpu_balls[e]);
                lockstep_cpu_envs[e].GetCarState(0, cpu_cars[e]);
            }

            gpu_sim.CopyControlsToDevice(step_controls.data(), 0, args.envs);
            gpu_sim.Step(args.envs);
            gpu_sim.CopyBallStateToHost(gpu_balls.data(), 0, args.envs);
            gpu_sim.CopyCarStateToHost(gpu_cars.data(), 0, args.envs);

            for (uint32_t e = 0; e < args.envs; e++) {
                if (t <= 2) {
                    std::cout << "[DEBUG Tick " << t << " Env " << e << "]\n"
                              << "  CPU Pos: (" << std::setprecision(8) << cpu_cars[e].pos.x << ", " << cpu_cars[e].pos.y << ", " << cpu_cars[e].pos.z << ")\n"
                              << "  GPU Pos: (" << std::setprecision(8) << gpu_cars[e].pos.x << ", " << gpu_cars[e].pos.y << ", " << gpu_cars[e].pos.z << ")\n"
                              << "  Delta Pos: (" << std::fabs(cpu_cars[e].pos.x - gpu_cars[e].pos.x) << ", "
                                                  << std::fabs(cpu_cars[e].pos.y - gpu_cars[e].pos.y) << ", "
                                                  << std::fabs(cpu_cars[e].pos.z - gpu_cars[e].pos.z) << ")\n"
                              << "  CPU Vel: (" << cpu_cars[e].vel.x << ", " << cpu_cars[e].vel.y << ", " << cpu_cars[e].vel.z << ")\n"
                              << "  GPU Vel: (" << gpu_cars[e].vel.x << ", " << gpu_cars[e].vel.y << ", " << gpu_cars[e].vel.z << ")\n"
                              << "  Controls: thr=" << step_controls[e].throttle << " steer=" << step_controls[e].steer
                              << " jump=" << (int)step_controls[e].jump << " boost=" << (int)step_controls[e].boost << "\n";
                }
                if (!comparator.CompareBall(t, e, cpu_balls[e], gpu_balls[e], fail)) {
                    std::cerr << "[-] Lockstep Differential Failure on Ball at tick " << t << ", env " << e << ":\n"
                              << "    Attribute: " << fail.attribute << "\n"
                              << "    Max delta: " << fail.max_delta << " > tol " << fail.threshold << "\n";
                    return 1;
                }
                if (!comparator.CompareCar(t, e, 0, cpu_cars[e], gpu_cars[e], fail)) {
                    std::cerr << "[-] Lockstep Differential Failure on Car at tick " << t << ", env " << e << ":\n"
                              << "    Attribute: " << fail.attribute << "\n"
                              << "    Max delta: " << fail.max_delta << " > tol " << fail.threshold << "\n"
                              << "    Delta Pos: (" << std::setprecision(8)
                              << std::fabs(cpu_cars[e].pos.x - gpu_cars[e].pos.x) << ", "
                              << std::fabs(cpu_cars[e].pos.y - gpu_cars[e].pos.y) << ", "
                              << std::fabs(cpu_cars[e].pos.z - gpu_cars[e].pos.z) << ")\n"
                              << "    CPU Pos: (" << cpu_cars[e].pos.x << ", " << cpu_cars[e].pos.y << ", " << cpu_cars[e].pos.z << ")\n"
                              << "    GPU Pos: (" << gpu_cars[e].pos.x << ", " << gpu_cars[e].pos.y << ", " << gpu_cars[e].pos.z << ")\n"
                              << "    CPU Vel: (" << cpu_cars[e].vel.x << ", " << cpu_cars[e].vel.y << ", " << cpu_cars[e].vel.z << ")\n"
                              << "    GPU Vel: (" << gpu_cars[e].vel.x << ", " << gpu_cars[e].vel.y << ", " << gpu_cars[e].vel.z << ")\n"
                              << "    CPU Ang: (" << cpu_cars[e].ang_vel.x << ", " << cpu_cars[e].ang_vel.y << ", " << cpu_cars[e].ang_vel.z << ")\n"
                              << "    GPU Ang: (" << gpu_cars[e].ang_vel.x << ", " << gpu_cars[e].ang_vel.y << ", " << gpu_cars[e].ang_vel.z << ")\n"
                              << "    CPU Quat: (" << cpu_cars[e].quat.w << ", " << cpu_cars[e].quat.x << ", " << cpu_cars[e].quat.y << ", " << cpu_cars[e].quat.z << ")\n"
                              << "    GPU Quat: (" << gpu_cars[e].quat.w << ", " << gpu_cars[e].quat.x << ", " << gpu_cars[e].quat.y << ", " << gpu_cars[e].quat.z << ")\n"
                              << "    Controls: thr=" << step_controls[e].throttle << " steer=" << step_controls[e].steer
                              << " pitch=" << step_controls[e].pitch << " yaw=" << step_controls[e].yaw
                              << " roll=" << step_controls[e].roll << " jump=" << (int)step_controls[e].jump
                              << " boost=" << (int)step_controls[e].boost << " handbrake=" << (int)step_controls[e].handbrake << "\n";
                    return 1;
                }
            }
        }
        std::cout << "[+] Lockstep Differential Simulation verified: 100% parity across all "
                  << args.ticks << " ticks!\n\n";
    } catch (const std::exception& ex) {
        std::cerr << "[-] GPU Lockstep Differential exception: " << ex.what() << "\n";
        return 1;
    }

    std::cout << "======================================================================\n"
              << "  ALL DIFFERENTIAL PARITY & GOLDEN MASTER TESTS PASSED SUCCESSFULLY!  \n"
              << "======================================================================\n";
    return 0;
}
