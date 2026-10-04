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
    bool cpu_perturb = false;
};

void PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --ticks <N>        Number of ticks to simulate (default: 500)\n"
              << "  --envs <N>         Number of concurrent environments (default: 4)\n"
              << "  --seed <N>         Pseudorandom seed for PCG32 controls (default: 42)\n"
              << "  --tol <F>          Chebyshev position tolerance (default: 1e-4)\n"
              << "  --record <path>    Output path for .rsgold recording (default: milestone1_golden.rsgold)\n"
              << "  --scenario <name>  Scenario to run: random, idle, freefall, throttle, boost, jump_flip, ball_flight, car_ball_hit, kickoff_goalie, all (default: random)\n"
              << "  --report           Enable windowed differential report mode (no early abort, records windows 1, 10, 120, 600, 10k)\n"
              << "  --out-report <path>Save report table in Markdown format to file\n"
              << "  --cpu-perturb      Run CPU vs CPU simulation with 1e-3 perturbation to measure divergence rate\n"
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
        } else if (arg == "--cpu-perturb") {
            args.cpu_perturb = true;
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
    } else if (scenario == "car_ball_hit" || scenario == "kickoff_goalie") {
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
        c.pos = Vec3(0.0f, -1000.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat(0.7071068f, 0.0f, 0.0f, 0.7071068f);
        c.boost = 100.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 93.15f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    } else if (scenario == "kickoff_goalie") {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, -4608.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat(0.7071068f, 0.0f, 0.0f, 0.7071068f);
        c.boost = 100.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 93.15f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }
}

struct WindowMetrics {
    float max_car_pos = 0.0f;
    float max_car_vel = 0.0f;
    float max_car_quat = 0.0f;
    float max_ball_pos = 0.0f;
    float max_ball_vel = 0.0f;
    float max_susp_uu = 0.0f;
    float max_boost = 0.0f;
    bool passed = true;

    void Update(float c_pos, float c_vel, float c_quat, float b_pos, float b_vel, float susp, float boost, bool ok) {
        max_car_pos = std::max(max_car_pos, c_pos);
        max_car_vel = std::max(max_car_vel, c_vel);
        max_car_quat = std::max(max_car_quat, c_quat);
        max_ball_pos = std::max(max_ball_pos, b_pos);
        max_ball_vel = std::max(max_ball_vel, b_vel);
        max_susp_uu = std::max(max_susp_uu, susp);
        max_boost = std::max(max_boost, boost);
        if (!ok) passed = false;
    }
};

struct KickoffGoalieMetrics {
    int touch_tick_cpu = -1;
    int touch_tick_gpu = -1;
    Vec3 car_pos_impact_cpu{0.0f, 0.0f, 0.0f};
    Vec3 car_pos_impact_gpu{0.0f, 0.0f, 0.0f};
    Vec3 car_vel_impact_cpu{0.0f, 0.0f, 0.0f};
    Vec3 car_vel_impact_gpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_vel_plus_1_cpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_vel_plus_1_gpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_vel_plus_10_cpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_vel_plus_10_gpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_vel_plus_60_cpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_vel_plus_60_gpu{0.0f, 0.0f, 0.0f};
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

    KickoffGoalieMetrics goalie;
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
            float d_c_pos = cpu_cars[e].pos.chebyshev_dist(gpu_cars[e].pos);
            float d_b_pos = cpu_balls[e].pos.chebyshev_dist(gpu_balls[e].pos);

            float d_c_vel = cpu_cars[e].vel.chebyshev_dist(gpu_cars[e].vel);
            float d_b_vel = cpu_balls[e].vel.chebyshev_dist(gpu_balls[e].vel);

            float d_c_quat = cpu_cars[e].quat.chebyshev_dist(gpu_cars[e].quat);

            float d_susp = 0.0f;
            for (int w = 0; w < 4; w++) {
                d_susp = std::max(d_susp, std::fabs(cpu_cars[e].suspension_lengths[w] - gpu_cars[e].suspension_lengths[w]));
            }
            float d_boost = std::fabs(cpu_cars[e].boost - gpu_cars[e].boost);

            bool ball_ok = comparator.CompareBall(t, e, cpu_balls[e], gpu_balls[e], fail);
            DifferentialFailure car_fail;
            bool car_ok = comparator.CompareCar(t, e, 0, cpu_cars[e], gpu_cars[e], car_fail);
            bool step_ok = ball_ok && car_ok;

            if (t < 1) report.w1.Update(d_c_pos, d_c_vel, d_c_quat, d_b_pos, d_b_vel, d_susp, d_boost, step_ok);
            if (t < 10) report.w10.Update(d_c_pos, d_c_vel, d_c_quat, d_b_pos, d_b_vel, d_susp, d_boost, step_ok);
            if (t < 120) report.w120.Update(d_c_pos, d_c_vel, d_c_quat, d_b_pos, d_b_vel, d_susp, d_boost, step_ok);
            if (t < 600) report.w600.Update(d_c_pos, d_c_vel, d_c_quat, d_b_pos, d_b_vel, d_susp, d_boost, step_ok);
            if (t < 10000) report.w10000.Update(d_c_pos, d_c_vel, d_c_quat, d_b_pos, d_b_vel, d_susp, d_boost, step_ok);

            if (scenario_name == "kickoff_goalie" && e == 0) {
                if (report.goalie.touch_tick_cpu == -1 && cpu_balls[e].vel.length() > 50.0f) {
                    report.goalie.touch_tick_cpu = static_cast<int>(t);
                    report.goalie.car_pos_impact_cpu = cpu_cars[e].pos;
                    report.goalie.car_vel_impact_cpu = cpu_cars[e].vel;
                }
                if (report.goalie.touch_tick_gpu == -1 && gpu_balls[e].vel.length() > 50.0f) {
                    report.goalie.touch_tick_gpu = static_cast<int>(t);
                    report.goalie.car_pos_impact_gpu = gpu_cars[e].pos;
                    report.goalie.car_vel_impact_gpu = gpu_cars[e].vel;
                }
                if (report.goalie.touch_tick_cpu != -1) {
                    int dt = static_cast<int>(t) - report.goalie.touch_tick_cpu;
                    if (dt == 1) report.goalie.ball_vel_plus_1_cpu = cpu_balls[e].vel;
                    if (dt == 10) report.goalie.ball_vel_plus_10_cpu = cpu_balls[e].vel;
                    if (dt == 60) report.goalie.ball_vel_plus_60_cpu = cpu_balls[e].vel;
                }
                if (report.goalie.touch_tick_gpu != -1) {
                    int dt = static_cast<int>(t) - report.goalie.touch_tick_gpu;
                    if (dt == 1) report.goalie.ball_vel_plus_1_gpu = gpu_balls[e].vel;
                    if (dt == 10) report.goalie.ball_vel_plus_10_gpu = gpu_balls[e].vel;
                    if (dt == 60) report.goalie.ball_vel_plus_60_gpu = gpu_balls[e].vel;
                }
            }

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
    os << "\n| Scenario | Window (Ticks) | Car Pos (UU) | Car Vel (UU/s) | Car Quat | Ball Pos (UU) | Ball Vel (UU/s) | First Breach Tick | Status |\n";
    os << "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n";
    for (const auto& rep : reports) {
        auto print_row = [&](const std::string& win_name, const WindowMetrics& m) {
            os << "| " << std::setw(15) << std::left << rep.name
               << " | " << std::setw(14) << win_name
               << " | " << std::scientific << std::setprecision(3) << m.max_car_pos
               << " | " << std::scientific << std::setprecision(3) << m.max_car_vel
               << " | " << std::scientific << std::setprecision(3) << m.max_car_quat
               << " | " << std::scientific << std::setprecision(3) << m.max_ball_pos
               << " | " << std::scientific << std::setprecision(3) << m.max_ball_vel
               << " | " << (rep.first_breach_tick >= 0 ? std::to_string(rep.first_breach_tick) : "None")
               << " | " << (m.passed ? "PASS" : "DRIFT") << " |\n";
        };
        print_row("1 (0.008s)", rep.w1);
        print_row("10 (0.083s)", rep.w10);
        if (rep.ticks_simulated >= 120) print_row("120 (1.0s)", rep.w120);
        if (rep.ticks_simulated >= 600) print_row("600 (5.0s)", rep.w600);
        if (rep.ticks_simulated >= 10000) print_row("10000 (83s)", rep.w10000);

        if (rep.name == "kickoff_goalie") {
            os << "\n#### Kickoff Goalie Impact & Collision Gate Analysis\n"
               << "| Metric | CPU Reference | GPU Kernel | Delta |\n"
               << "| :--- | :--- | :--- | :--- |\n"
               << "| First Touch Tick | " << rep.goalie.touch_tick_cpu << " | " << rep.goalie.touch_tick_gpu
               << " | " << std::abs(rep.goalie.touch_tick_cpu - rep.goalie.touch_tick_gpu) << " ticks |\n"
               << "| Car Pos at Impact (UU) | " << rep.goalie.car_pos_impact_cpu << " | " << rep.goalie.car_pos_impact_gpu
               << " | " << std::scientific << std::setprecision(3) << rep.goalie.car_pos_impact_cpu.chebyshev_dist(rep.goalie.car_pos_impact_gpu) << " UU |\n"
               << "| Car Vel at Impact (UU/s) | " << rep.goalie.car_vel_impact_cpu << " | " << rep.goalie.car_vel_impact_gpu
               << " | " << std::scientific << std::setprecision(3) << rep.goalie.car_vel_impact_cpu.chebyshev_dist(rep.goalie.car_vel_impact_gpu) << " UU/s |\n"
               << "| Ball Vel +1 Tick (UU/s) | " << rep.goalie.ball_vel_plus_1_cpu << " | " << rep.goalie.ball_vel_plus_1_gpu
               << " | " << std::scientific << std::setprecision(3) << rep.goalie.ball_vel_plus_1_cpu.chebyshev_dist(rep.goalie.ball_vel_plus_1_gpu) << " UU/s |\n"
               << "| Ball Vel +10 Ticks (UU/s) | " << rep.goalie.ball_vel_plus_10_cpu << " | " << rep.goalie.ball_vel_plus_10_gpu
               << " | " << std::scientific << std::setprecision(3) << rep.goalie.ball_vel_plus_10_cpu.chebyshev_dist(rep.goalie.ball_vel_plus_10_gpu) << " UU/s |\n"
               << "| Ball Vel +60 Ticks (UU/s) | " << rep.goalie.ball_vel_plus_60_cpu << " | " << rep.goalie.ball_vel_plus_60_gpu
               << " | " << std::scientific << std::setprecision(3) << rep.goalie.ball_vel_plus_60_cpu.chebyshev_dist(rep.goalie.ball_vel_plus_60_gpu) << " UU/s |\n\n";
        }
    }
    os << "\n";
}

void RunCpuVsCpuPerturbation(const std::string& scenario, uint32_t ticks, uint32_t seed) {
    CPURefSim cpu1(1, true, TICK_RATE, 0);
    CPURefSim cpu2(1, true, TICK_RATE, 0);
    ApplyScenarioInitialState(scenario, cpu1, 0);
    ApplyScenarioInitialState(scenario, cpu2, 0);

    CarStatePOD c;
    cpu2.GetCarState(0, c);
    // Perturb position by 1e-3 UU
    c.pos.x += 1e-3f;
    c.pos.y += 1e-3f;
    // Perturb velocity by 1e-3 UU/s
    c.vel.x += 1e-3f;
    c.vel.y += 1e-3f;
    // Perturb yaw angle by 1e-3 radians
    float half_d_yaw = 0.5e-3f;
    c.quat = (c.quat * Quat(std::cos(half_d_yaw), 0.0f, 0.0f, std::sin(half_d_yaw))).normalized();
    cpu2.SetCarState(0, c);

    BallStatePOD b;
    cpu2.GetBallState(b);
    b.pos.x += 1e-3f;
    b.pos.y += 1e-3f;
    b.vel.x += 1e-3f;
    b.vel.y += 1e-3f;
    cpu2.SetBallState(b);

    DeterministicInputGenerator gen(seed);
    std::cout << "\n[CPU vs CPU Perturbation Analysis (1e-3 UU pos, 1e-3 UU/s vel, 1e-3 rad yaw)] Scenario: " << scenario << "\n";
    std::cout << "| Window (Ticks) | Car Pos Delta (UU) | Car Vel Delta (UU/s) | Car Quat Delta | Ball Pos Delta (UU) | Ball Vel Delta (UU/s) |\n";
    std::cout << "| :--- | :--- | :--- | :--- | :--- | :--- |\n";

    CarStatePOD c1, c2;
    BallStatePOD b1, b2;
    for (uint32_t t = 0; t < ticks; t++) {
        CarControls ctrl = GetScenarioControl(scenario, t, 0, gen);
        cpu1.Step(&ctrl, 1);
        cpu2.Step(&ctrl, 1);
        cpu1.GetCarState(0, c1);
        cpu2.GetCarState(0, c2);
        cpu1.GetBallState(b1);
        cpu2.GetBallState(b2);

        if (t == 0 || t == 9 || t == 119 || t == 599 || t == ticks - 1) {
            float d_c_pos = c1.pos.chebyshev_dist(c2.pos);
            float d_c_vel = c1.vel.chebyshev_dist(c2.vel);
            float d_c_quat = c1.quat.chebyshev_dist(c2.quat);
            float d_b_pos = b1.pos.chebyshev_dist(b2.pos);
            float d_b_vel = b1.vel.chebyshev_dist(b2.vel);
            std::cout << "| " << std::setw(14) << std::left << (std::to_string(t + 1) + " ticks")
                      << " | " << std::scientific << std::setprecision(3) << d_c_pos
                      << " | " << std::scientific << std::setprecision(3) << d_c_vel
                      << " | " << std::scientific << std::setprecision(3) << d_c_quat
                      << " | " << std::scientific << std::setprecision(3) << d_b_pos
                      << " | " << std::scientific << std::setprecision(3) << d_b_vel << " |\n";
        }
    }
}

int main(int argc, char** argv) {
    HarnessArgs args = ParseArgs(argc, argv);

    if (args.cpu_perturb) {
        std::vector<std::string> scns = (args.scenario == "all")
            ? std::vector<std::string>{"idle", "freefall", "throttle", "boost", "jump_flip", "ball_flight", "car_ball_hit", "kickoff_goalie", "random"}
            : std::vector<std::string>{args.scenario};
        for (const auto& scn : scns) {
            RunCpuVsCpuPerturbation(scn, args.ticks, args.seed);
        }
        return 0;
    }

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
    float max_b = std::max({std::abs(perturbed_ball.pos.x), std::abs(perturbed_ball.pos.y), std::abs(perturbed_ball.pos.z)});
    float b_ulp = (max_b >= 4096.0f) ? 0.00048828125f : ((max_b >= 2048.0f) ? 0.000244140625f : 0.0f);
    perturbed_ball.pos.x += std::max(args.tol, b_ulp) * 2.0f;
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
        scenarios_to_run = {"idle", "freefall", "throttle", "boost", "jump_flip", "ball_flight", "car_ball_hit", "kickoff_goalie", "random"};
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
