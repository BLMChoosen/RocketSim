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
    std::string scenario = "random";
    bool report_mode = false;
    std::string out_report = "";
};

void PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --ticks <N>        Number of ticks to simulate (default: 500)\n"
              << "  --envs <N>         Number of concurrent environments (default: 4)\n"
              << "  --seed <N>         Pseudorandom seed for PCG32 controls (default: 42)\n"
              << "  --tol <F>          Chebyshev position tolerance (default: 1e-4)\n"
              << "  --record <path>    Output path for .rsgold recording (default: milestone1_golden.rsgold)\n"
              << "  --scenario <name>  Scenario to run: random, idle, freefall, throttle, boost, jump_flip, ball_flight, car_ball_hit, all (default: random)\n"
              << "  --report           Enable windowed differential report mode (no early abort, records windows 1, 10, 120, 600, 10k)\n"
              << "  --out-report <path>Save report table in Markdown format to file\n"
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
        } else if (arg == "--scenario" && i + 1 < argc) {
            args.scenario = argv[++i];
        } else if (arg == "--report") {
            args.report_mode = true;
        } else if (arg == "--out-report" && i + 1 < argc) {
            args.out_report = argv[++i];
        } else if (arg == "--help") {
            PrintUsage(argv[0]);
            std::exit(0);
        }
    }
    return args;
}

CarControls GetScenarioControl(const std::string& scenario, uint32_t tick, uint32_t env, DeterministicInputGenerator& gen) {
    CarControls c{};
    if (scenario == "idle" || scenario == "freefall" || scenario == "ball_flight") {
        return c;
    } else if (scenario == "throttle") {
        c.throttle = 1.0f;
        return c;
    } else if (scenario == "boost") {
        c.throttle = 1.0f;
        c.boost = 1;
        return c;
    } else if (scenario == "jump_flip") {
        c.throttle = 1.0f;
        if (tick >= 10 && tick < 15) {
            c.jump = 1;
        } else if (tick >= 25 && tick < 30) {
            c.jump = 1;
            c.pitch = -1.0f;
        } else if (tick >= 30) {
            c.pitch = 1.0f;
            c.roll = 0.5f;
        }
        return c;
    } else if (scenario == "car_ball_hit") {
        c.throttle = 1.0f;
        c.boost = 1;
        return c;
    }
    return gen.Generate();
}

void ApplyScenarioInitialState(const std::string& scenario, CPURefSim& env, uint32_t env_idx) {
    if (scenario == "freefall") {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(1000.0f, 0.0f, 1500.0f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(-1000.0f, 0.0f, 1500.0f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    } else if (scenario == "ball_flight") {
        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 200.0f);
        b.vel = Vec3(1500.0f, 2000.0f, 1000.0f);
        b.ang_vel = Vec3(2.0f, 3.0f, -1.0f);
        env.SetBallState(b);
    } else if (scenario == "car_ball_hit") {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, -1000.0f, 17.0f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 93.15f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        env.SetBallState(b);
    }
}

struct WindowMetrics {
    float max_pos_uu = 0.0f;
    float max_vel_uus = 0.0f;
    float max_quat = 0.0f;
    float max_ang_vel_rads = 0.0f;
    float max_susp_uu = 0.0f;
    float max_boost = 0.0f;
    bool passed = true;

    void Update(float pos, float vel, float quat, float ang, float susp, float boost, bool ok) {
        max_pos_uu = std::max(max_pos_uu, pos);
        max_vel_uus = std::max(max_vel_uus, vel);
        max_quat = std::max(max_quat, quat);
        max_ang_vel_rads = std::max(max_ang_vel_rads, ang);
        max_susp_uu = std::max(max_susp_uu, susp);
        max_boost = std::max(max_boost, boost);
        if (!ok) passed = false;
    }
};

struct ScenarioReport {
    std::string name;
    uint32_t ticks_simulated = 0;
    int first_breach_tick = -1;
    std::string first_breach_attr = "None";
    float first_breach_delta = 0.0f;
    float first_breach_threshold = 0.0f;

    WindowMetrics w1;
    WindowMetrics w10;
    WindowMetrics w120;
    WindowMetrics w600;
    WindowMetrics w10000;
};

bool RunScenarioDifferential(
    const std::string& scenario_name,
    const HarnessArgs& args,
    const DifferentialTolerance& tol,
    DifferentialComparator& comparator,
    ScenarioReport& report,
    bool fail_fast) {

    report.name = scenario_name;
    report.ticks_simulated = args.ticks;

    SimContext gpu_sim(args.envs, 1);

    std::vector<CPURefSim> lockstep_cpu_envs;
    lockstep_cpu_envs.reserve(args.envs);
    for (uint32_t e = 0; e < args.envs; e++) {
        lockstep_cpu_envs.emplace_back(1, true, TICK_RATE, static_cast<int>(e));
        ApplyScenarioInitialState(scenario_name, lockstep_cpu_envs[e], e);
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

    DifferentialFailure fail;

    for (uint32_t t = 0; t < args.ticks; t++) {
        for (uint32_t e = 0; e < args.envs; e++) {
            step_controls[e] = GetScenarioControl(scenario_name, t, e, lockstep_gen);
            lockstep_cpu_envs[e].Step(&step_controls[e], 1);
            lockstep_cpu_envs[e].GetBallState(cpu_balls[e]);
            lockstep_cpu_envs[e].GetCarState(0, cpu_cars[e]);
        }

        gpu_sim.CopyControlsToDevice(step_controls.data(), 0, args.envs);
        gpu_sim.Step(args.envs);
        gpu_sim.CopyBallStateToHost(gpu_balls.data(), 0, args.envs);
        gpu_sim.CopyCarStateToHost(gpu_cars.data(), 0, args.envs);

        for (uint32_t e = 0; e < args.envs; e++) {
            if (e == 0 && t <= 2) {
                std::cout << "    [DEBUG " << scenario_name << " t=" << t << "]\n"
                          << "      CPU Car Pos: " << cpu_cars[e].pos << " Vel: " << cpu_cars[e].vel << "\n"
                          << "      GPU Car Pos: " << gpu_cars[e].pos << " Vel: " << gpu_cars[e].vel << "\n"
                          << "      CPU Ball Pos: " << cpu_balls[e].pos << " Vel: " << cpu_balls[e].vel << "\n"
                          << "      GPU Ball Pos: " << gpu_balls[e].pos << " Vel: " << gpu_balls[e].vel << "\n";
            }

            float d_c_pos = cpu_cars[e].pos.chebyshev_dist(gpu_cars[e].pos);
            float d_b_pos = cpu_balls[e].pos.chebyshev_dist(gpu_balls[e].pos);
            float max_d_pos = std::max(d_c_pos, d_b_pos);

            float d_c_vel = cpu_cars[e].vel.chebyshev_dist(gpu_cars[e].vel);
            float d_b_vel = cpu_balls[e].vel.chebyshev_dist(gpu_balls[e].vel);
            float max_d_vel = std::max(d_c_vel, d_b_vel);

            float d_c_quat = cpu_cars[e].quat.chebyshev_dist(gpu_cars[e].quat);
            float d_b_quat = cpu_balls[e].quat.chebyshev_dist(gpu_balls[e].quat);
            float max_d_quat = std::max(d_c_quat, d_b_quat);

            float d_c_ang = cpu_cars[e].ang_vel.chebyshev_dist(gpu_cars[e].ang_vel);
            float d_b_ang = cpu_balls[e].ang_vel.chebyshev_dist(gpu_balls[e].ang_vel);
            float max_d_ang = std::max(d_c_ang, d_b_ang);

            float d_susp = 0.0f;
            for (int w = 0; w < 4; w++) {
                d_susp = std::max(d_susp, std::fabs(cpu_cars[e].suspension_lengths[w] - gpu_cars[e].suspension_lengths[w]));
            }
            float d_boost = std::fabs(cpu_cars[e].boost - gpu_cars[e].boost);

            bool ball_ok = comparator.CompareBall(t, e, cpu_balls[e], gpu_balls[e], fail);
            DifferentialFailure car_fail;
            bool car_ok = comparator.CompareCar(t, e, 0, cpu_cars[e], gpu_cars[e], car_fail);
            bool step_ok = ball_ok && car_ok;

            if (t < 1) report.w1.Update(max_d_pos, max_d_vel, max_d_quat, max_d_ang, d_susp, d_boost, step_ok);
            if (t < 10) report.w10.Update(max_d_pos, max_d_vel, max_d_quat, max_d_ang, d_susp, d_boost, step_ok);
            if (t < 120) report.w120.Update(max_d_pos, max_d_vel, max_d_quat, max_d_ang, d_susp, d_boost, step_ok);
            if (t < 600) report.w600.Update(max_d_pos, max_d_vel, max_d_quat, max_d_ang, d_susp, d_boost, step_ok);
            if (t < 10000) report.w10000.Update(max_d_pos, max_d_vel, max_d_quat, max_d_ang, d_susp, d_boost, step_ok);

            if (!step_ok && report.first_breach_tick == -1) {
                report.first_breach_tick = static_cast<int>(t);
                if (!ball_ok) {
                    report.first_breach_attr = fail.attribute;
                    report.first_breach_delta = fail.max_delta;
                    report.first_breach_threshold = fail.threshold;
                } else {
                    report.first_breach_attr = car_fail.attribute;
                    report.first_breach_delta = car_fail.max_delta;
                    report.first_breach_threshold = car_fail.threshold;
                }
            }

            if (!step_ok && fail_fast) {
                std::cerr << "[-] Lockstep Differential Failure in scenario '" << scenario_name
                          << "' at tick " << t << ", env " << e << ":\n";
                if (!ball_ok) {
                    std::cerr << "    Ball: " << fail.attribute << " | Max delta: " << fail.max_delta
                              << " > tol " << fail.threshold << "\n";
                } else {
                    std::cerr << "    Car:  " << car_fail.attribute << " | Max delta: " << car_fail.max_delta
                              << " > tol " << car_fail.threshold << "\n";
                }
                return false;
            }
        }
    }
    return (report.first_breach_tick == -1);
}

void PrintScenarioReportTable(const std::vector<ScenarioReport>& reports, std::ostream& os) {
    os << "\n| Scenario | Window (Ticks) | Max Pos Delta (UU) | Max Vel Delta (UU/s) | Max Quat Delta | Max AngVel (rad/s) | First Breach Tick | Status |\n";
    os << "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n";
    for (const auto& rep : reports) {
        auto print_row = [&](const std::string& win_name, const WindowMetrics& m) {
            os << "| " << std::setw(14) << std::left << rep.name
               << " | " << std::setw(14) << win_name
               << " | " << std::scientific << std::setprecision(3) << m.max_pos_uu
               << " | " << std::scientific << std::setprecision(3) << m.max_vel_uus
               << " | " << std::scientific << std::setprecision(3) << m.max_quat
               << " | " << std::scientific << std::setprecision(3) << m.max_ang_vel_rads
               << " | " << (rep.first_breach_tick >= 0 ? std::to_string(rep.first_breach_tick) : "None")
               << " | " << (m.passed ? "PASS" : "DRIFT") << " |\n";
        };
        print_row("1 (0.008s)", rep.w1);
        print_row("10 (0.083s)", rep.w10);
        if (rep.ticks_simulated >= 120) print_row("120 (1.0s)", rep.w120);
        if (rep.ticks_simulated >= 600) print_row("600 (5.0s)", rep.w600);
        if (rep.ticks_simulated >= 10000) print_row("10000 (83s)", rep.w10000);
    }
    os << "\n";
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
              << "  Scenario:     " << args.scenario << "\n"
              << "  Report Mode:  " << (args.report_mode ? "ENABLED (Windowed Stats)" : "DISABLED (Fail-Fast)") << "\n"
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
        ApplyScenarioInitialState(args.scenario == "all" ? "random" : args.scenario, cpu_envs[e], e);
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
            current_controls[e] = GetScenarioControl(args.scenario == "all" ? "random" : args.scenario, t, e, input_gen);
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

    std::vector<CPURefSim> verifier_envs;
    verifier_envs.reserve(args.envs);
    for (uint32_t e = 0; e < args.envs; e++) {
        verifier_envs.emplace_back(1, true, TICK_RATE, static_cast<int>(e));
        ApplyScenarioInitialState(args.scenario == "all" ? "random" : args.scenario, verifier_envs[e], e);
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

        sim_ctx.CopyBallStateToDevice(current_balls.data(), 0, args.envs);
        sim_ctx.CopyCarStateToDevice(current_cars.data(), 0, args.envs);
        sim_ctx.CopyControlsToDevice(current_controls.data(), 0, args.envs);

        std::vector<BallStatePOD> downloaded_balls(args.envs);
        std::vector<CarStatePOD> downloaded_cars(args.envs);
        sim_ctx.CopyBallStateToHost(downloaded_balls.data(), 0, args.envs);
        sim_ctx.CopyCarStateToHost(downloaded_cars.data(), 0, args.envs);

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
    perturbed_ball.pos.x += args.tol * 2.0f;
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
    // Test 5: Lockstep Differential Simulation
    // ------------------------------------------------------------------
    std::cout << "[Test 5/5] Lockstep Differential Simulation: GPU vs CPU Oracle ("
              << args.ticks << " ticks, " << args.envs << " envs)...\n";

    std::vector<std::string> scenarios_to_run;
    if (args.scenario == "all") {
        scenarios_to_run = {"idle", "freefall", "throttle", "boost", "jump_flip", "ball_flight", "car_ball_hit", "random"};
    } else {
        scenarios_to_run = {args.scenario};
    }

    std::vector<ScenarioReport> reports;
    bool all_passed = true;

    for (const auto& scn : scenarios_to_run) {
        std::cout << "  --> Running scenario: " << scn << " (" << args.ticks << " ticks)...\n";
        ScenarioReport rep;
        bool scn_ok = RunScenarioDifferential(scn, args, tol, comparator, rep, !args.report_mode);
        reports.push_back(rep);
        if (!scn_ok) {
            all_passed = false;
            if (!args.report_mode) {
                return 1;
            }
        }
    }

    if (args.report_mode) {
        PrintScenarioReportTable(reports, std::cout);
        if (!args.out_report.empty()) {
            std::ofstream out_f(args.out_report);
            if (out_f.is_open()) {
                PrintScenarioReportTable(reports, out_f);
                std::cout << "[+] Report successfully written to: " << args.out_report << "\n";
            }
        }
    }

    if (all_passed) {
        std::cout << "======================================================================\n"
                  << "  ALL DIFFERENTIAL PARITY & GOLDEN MASTER TESTS PASSED SUCCESSFULLY!  \n"
                  << "======================================================================\n";
        return 0;
    } else if (args.report_mode) {
        std::cout << "======================================================================\n"
                  << "  DIFFERENTIAL PARITY REPORT COMPLETED ACROSS ALL REQUESTED WINDOWS!  \n"
                  << "======================================================================\n";
        return 0;
    } else {
        std::cerr << "[-] Differential Parity Failed\n";
        return 1;
    }
}
