#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <algorithm>
#include <cmath>
#include <map>
#include <fstream>
#include <sstream>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/sim_context.cuh"
#include "cpu_ref_sim.h"
#include "golden_master.h"
#include "scenarios/scenario_registry.h"

namespace rocketsim_cuda {
    bool GetCPURefSimBoostPadState(const CPURefSim* sim, int padIdx, bool& isActive, float& cooldown);
}

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
    bool baseline_mode = false;
    bool check_mode = false;
    std::string check_file = "docs/parity_thresholds.json";
    uint32_t cars = 1;
};

void PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --ticks <N>        Number of ticks to simulate (default: 500)\n"
              << "  --envs <N>         Number of concurrent environments (default: 4)\n"
              << "  --cars <N>         Number of cars per environment (default: 1, up to 6 for 1v1, 2v2, 3v3)\n"
              << "  --seed <N>         Pseudorandom seed for PCG32 controls (default: 42)\n"
              << "  --tol <F>          Chebyshev position tolerance (default: 1e-4)\n"
              << "  --record <path>    Output path for .rsgold recording (default: milestone1_golden.rsgold)\n"
              << "  --scenario <name>  Scenario to run: random, idle, freefall, throttle, boost, jump_flip, ablation_5_flips, ball_flight, car_ball_hit, kickoff_goalie, boost_pad_pickup, all (default: random)\n"
              << "  --report           Enable windowed differential report mode (no early abort, records windows 1, 10, 60, 120, 600, 10k)\n"
              << "  --baseline         Run baseline mode directly (bypasses serialization, computes component-wise Median & P95)\n"
              << "  --out-report <path>Save report table in Markdown format to file\n"
              << "  --cpu-perturb      Run CPU vs CPU simulation with 1e-3 perturbation to measure divergence rate\n"
              << "  --check [path]     Validate run metrics against docs/parity_thresholds.json (exit 1 on failure)\n"
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
        } else if (arg == "--cars" && i + 1 < argc) {
            args.cars = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
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
        } else if (arg == "--baseline") {
            args.baseline_mode = true;
            args.report_mode = true;
        } else if (arg == "--out-report" && i + 1 < argc) {
            args.out_report = argv[++i];
        } else if (arg == "--cpu-perturb") {
            args.cpu_perturb = true;
        } else if (arg == "--check") {
            args.check_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.check_file = argv[++i];
            }
        } else if (arg == "--help") {
            PrintUsage(argv[0]);
            std::exit(0);
        }
    }
    return args;
}

inline CarControls GetScenarioControl(const std::string& scenario, uint32_t tick, uint32_t env, DeterministicInputGenerator& gen) {
    auto scn = ScenarioRegistry::Instance().Get(scenario);
    if (scn) return scn->GetControl(tick, env, 0, gen);
    return gen.Generate();
}

inline CarControls GetScenarioControl(const std::string& scenario, uint32_t tick, uint32_t env, uint32_t car_idx, DeterministicInputGenerator& gen) {
    auto scn = ScenarioRegistry::Instance().Get(scenario);
    if (scn) return scn->GetControl(tick, env, car_idx, gen);
    return gen.Generate();
}

inline void ApplyScenarioInitialState(const std::string& scenario, CPURefSim& env, uint32_t env_idx) {
    auto scn = ScenarioRegistry::Instance().Get(scenario);
    if (scn) scn->ApplyInitialState(env, env_idx);
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

struct BounceMetrics {
    int bounce_tick_cpu = -1;
    int bounce_tick_gpu = -1;
    Vec3 ball_vel_plus_1_cpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_vel_plus_1_gpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_vel_plus_5_cpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_vel_plus_5_gpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_vel_plus_30_cpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_vel_plus_30_gpu{0.0f, 0.0f, 0.0f};

    Vec3 ball_ang_vel_plus_1_cpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_ang_vel_plus_1_gpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_ang_vel_plus_5_cpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_ang_vel_plus_5_gpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_ang_vel_plus_30_cpu{0.0f, 0.0f, 0.0f};
    Vec3 ball_ang_vel_plus_30_gpu{0.0f, 0.0f, 0.0f};
};

struct BoostPadMetrics {
    int target_pad_idx = 0;
    bool is_big = true;
    int pickup_tick_cpu = -1;
    int pickup_tick_gpu = -1;
    float boost_before_cpu = 0.0f;
    float boost_before_gpu = 0.0f;
    float boost_after_cpu = 0.0f;
    float boost_after_gpu = 0.0f;
    bool pad_active_after_pickup_cpu = true;
    bool pad_active_after_pickup_gpu = true;
    float cooldown_after_pickup_cpu = 0.0f;
    float cooldown_after_pickup_gpu = 0.0f;
    int respawn_tick_cpu = -1;
    int respawn_tick_gpu = -1;
};

enum ComponentId {
    CAR_POS_X = 0, CAR_POS_Y, CAR_POS_Z,
    CAR_VEL_X, CAR_VEL_Y, CAR_VEL_Z,
    CAR_QUAT_W, CAR_QUAT_X, CAR_QUAT_Y, CAR_QUAT_Z,
    BALL_POS_X, BALL_POS_Y, BALL_POS_Z,
    BALL_VEL_X, BALL_VEL_Y, BALL_VEL_Z,
    BALL_ANG_VEL_X, BALL_ANG_VEL_Y, BALL_ANG_VEL_Z,
    NUM_COMPONENTS = 19
};

inline const char* GetComponentName(uint32_t c) {
    static const char* const names[NUM_COMPONENTS] = {
        "Car Pos X", "Car Pos Y", "Car Pos Z",
        "Car Vel X", "Car Vel Y", "Car Vel Z",
        "Car Quat W", "Car Quat X", "Car Quat Y", "Car Quat Z",
        "Ball Pos X", "Ball Pos Y", "Ball Pos Z",
        "Ball Vel X", "Ball Vel Y", "Ball Vel Z",
        "Ball AngVel X", "Ball AngVel Y", "Ball AngVel Z"
    };
    return (c < NUM_COMPONENTS) ? names[c] : "Unknown";
}

constexpr uint32_t SNAPSHOT_TICKS[] = {1, 10, 60, 120, 600};
constexpr size_t NUM_SNAPSHOTS = sizeof(SNAPSHOT_TICKS) / sizeof(SNAPSHOT_TICKS[0]);

struct ComponentStat {
    float median_abs = 0.0f;
    float p95_abs = 0.0f;
    float median_rel = 0.0f;
    float p95_rel = 0.0f;
};

struct SnapshotReport {
    uint32_t tick = 0;
    ComponentStat stats[NUM_COMPONENTS];
};

struct BaselineMetrics {
    std::vector<std::vector<std::vector<float>>> abs_errors;
    std::vector<std::vector<std::vector<float>>> rel_errors;
    std::vector<SnapshotReport> snapshots;

    void Init(uint32_t num_envs) {
        abs_errors.assign(NUM_SNAPSHOTS, std::vector<std::vector<float>>(NUM_COMPONENTS, std::vector<float>(num_envs, 0.0f)));
        rel_errors.assign(NUM_SNAPSHOTS, std::vector<std::vector<float>>(NUM_COMPONENTS, std::vector<float>(num_envs, 0.0f)));
    }

    void RecordSnapshot(size_t snap_idx, uint32_t env, uint32_t comp, float abs_err, float rel_err) {
        if (snap_idx < NUM_SNAPSHOTS && comp < NUM_COMPONENTS && env < abs_errors[snap_idx][comp].size()) {
            abs_errors[snap_idx][comp][env] = abs_err;
            rel_errors[snap_idx][comp][env] = rel_err;
        }
    }

    static float ComputePercentile(std::vector<float>& samples, float p) {
        if (samples.empty()) return 0.0f;
        size_t idx = static_cast<size_t>(std::clamp(p * (samples.size() - 1), 0.0f, static_cast<float>(samples.size() - 1)));
        std::nth_element(samples.begin(), samples.begin() + idx, samples.end());
        return samples[idx];
    }

    void Finalize(uint32_t max_ticks) {
        snapshots.clear();
        for (size_t s = 0; s < NUM_SNAPSHOTS; ++s) {
            if (SNAPSHOT_TICKS[s] > max_ticks) continue;
            SnapshotReport snap;
            snap.tick = SNAPSHOT_TICKS[s];
            for (uint32_t c = 0; c < NUM_COMPONENTS; ++c) {
                std::vector<float>& abs_vec = abs_errors[s][c];
                std::vector<float>& rel_vec = rel_errors[s][c];

                snap.stats[c].median_abs = ComputePercentile(abs_vec, 0.50f);
                snap.stats[c].p95_abs    = ComputePercentile(abs_vec, 0.95f);
                snap.stats[c].median_rel = ComputePercentile(rel_vec, 0.50f);
                snap.stats[c].p95_rel    = ComputePercentile(rel_vec, 0.95f);
            }
            snapshots.push_back(snap);
        }
    }
};

class ThreadPool {
public:
    explicit ThreadPool(size_t threads) : stop_(false), active_tasks_(0) {
        workers_.reserve(threads);
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        active_tasks_--;
                        if (active_tasks_ == 0 && tasks_.empty()) {
                            finished_cv_.notify_all();
                        }
                    }
                }
            });
        }
    }

    void ParallelFor(uint32_t start, uint32_t end, const std::function<void(uint32_t, uint32_t)>& func) {
        if (start >= end) return;
        uint32_t total = end - start;
        uint32_t num_workers = static_cast<uint32_t>(workers_.size());
        if (num_workers <= 1) {
            func(start, end);
            return;
        }
        uint32_t chunk = (total + num_workers - 1) / num_workers;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            for (uint32_t i = 0; i < num_workers; ++i) {
                uint32_t c_start = start + i * chunk;
                if (c_start >= end) break;
                uint32_t c_end = (std::min)(c_start + chunk, end);
                active_tasks_++;
                tasks_.push([func, c_start, c_end] {
                    func(c_start, c_end);
                });
            }
        }
        cv_.notify_all();

        std::unique_lock<std::mutex> lock(queue_mutex_);
        finished_cv_.wait(lock, [this] { return active_tasks_ == 0 && tasks_.empty(); });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::condition_variable finished_cv_;
    size_t active_tasks_ = 0;
    bool stop_ = false;
};

struct ScenarioReport {
    std::string name;
    uint32_t ticks_simulated = 0;
    uint32_t num_envs = 0;
    int first_breach_tick = -1;
    std::string first_breach_attr = "None";
    float first_breach_delta = 0.0f;
    float first_breach_threshold = 0.0f;

    WindowMetrics w1;
    WindowMetrics w10;
    WindowMetrics w60;
    WindowMetrics w120;
    WindowMetrics w600;
    WindowMetrics w10000;

    KickoffGoalieMetrics goalie;
    BounceMetrics bounce;
    BaselineMetrics baseline;
    BoostPadMetrics boost_pad;
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
    report.num_envs = args.envs;
    report.baseline.Init(args.envs);

    uint32_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 4;
    uint32_t num_threads = std::clamp(hw_threads, 1u, 16u);
    if (args.envs < num_threads) num_threads = args.envs;
    ThreadPool thread_pool(num_threads);

    auto scn_def = ScenarioRegistry::Instance().Get(scenario_name);
    uint32_t cars_per_env = (args.cars > 1) ? args.cars : (scn_def ? scn_def->GetDefaultCars() : 1);
    SimContext gpu_sim(args.envs, cars_per_env);

    std::vector<CPURefSim> lockstep_cpu_envs;
    lockstep_cpu_envs.reserve(args.envs);
    for (uint32_t e = 0; e < args.envs; e++) {
        lockstep_cpu_envs.emplace_back(cars_per_env, true, TICK_RATE, static_cast<int>(e));
        ApplyScenarioInitialState(scenario_name, lockstep_cpu_envs[e], e);
    }

    std::vector<BallStatePOD> init_balls(args.envs);
    std::vector<CarStatePOD> init_cars(args.envs * cars_per_env);
    for (uint32_t e = 0; e < args.envs; e++) {
        lockstep_cpu_envs[e].GetBallState(init_balls[e]);
        for (uint32_t c = 0; c < cars_per_env; c++) {
            lockstep_cpu_envs[e].GetCarState(c, init_cars[e * cars_per_env + c]);
        }
    }
    gpu_sim.CopyBallStateToDevice(init_balls.data(), 0, args.envs);
    gpu_sim.CopyCarStateToDevice(init_cars.data(), 0, args.envs * cars_per_env);

    if (scenario_name == "boost_pad_pickup") {
        report.boost_pad.target_pad_idx = 0;
        report.boost_pad.is_big = true;
        report.boost_pad.boost_before_cpu = init_cars[0].boost;
        report.boost_pad.boost_before_gpu = init_cars[0].boost;
    }

    DeterministicInputGenerator lockstep_gen(args.seed);
    std::vector<CarControls> step_controls(args.envs * cars_per_env);
    std::vector<BallStatePOD> gpu_balls(args.envs);
    std::vector<CarStatePOD> gpu_cars(args.envs * cars_per_env);
    std::vector<BallStatePOD> cpu_balls(args.envs);
    std::vector<CarStatePOD> cpu_cars(args.envs * cars_per_env);
    std::vector<Vec3> prev_cpu_ball_vel(args.envs);
    std::vector<Vec3> prev_gpu_ball_vel(args.envs);
    for (uint32_t e = 0; e < args.envs; e++) {
        prev_cpu_ball_vel[e] = init_balls[e].vel;
        prev_gpu_ball_vel[e] = init_balls[e].vel;
    }

    DifferentialFailure fail;

    for (uint32_t t = 0; t < args.ticks; t++) {
        for (uint32_t e = 0; e < args.envs; e++) {
            for (uint32_t c = 0; c < cars_per_env; c++) {
                step_controls[e * cars_per_env + c] = GetScenarioControl(scenario_name, t, e + c * 100, lockstep_gen);
            }
        }

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < static_cast<int>(args.envs); ++e) {
            lockstep_cpu_envs[e].Step(&step_controls[e * cars_per_env], cars_per_env);
            lockstep_cpu_envs[e].GetBallState(cpu_balls[e]);
            for (uint32_t c = 0; c < cars_per_env; c++) {
                lockstep_cpu_envs[e].GetCarState(c, cpu_cars[e * cars_per_env + c]);
            }
        }
#else
        thread_pool.ParallelFor(0, args.envs, [&](uint32_t start_e, uint32_t end_e) {
            for (uint32_t e = start_e; e < end_e; ++e) {
                lockstep_cpu_envs[e].Step(&step_controls[e * cars_per_env], cars_per_env);
                lockstep_cpu_envs[e].GetBallState(cpu_balls[e]);
                for (uint32_t c = 0; c < cars_per_env; c++) {
                    lockstep_cpu_envs[e].GetCarState(c, cpu_cars[e * cars_per_env + c]);
                }
            }
        });
#endif

        gpu_sim.CopyControlsToDevice(step_controls.data(), 0, args.envs * cars_per_env);
        gpu_sim.Step(args.envs);
        gpu_sim.CopyBallStateToHost(gpu_balls.data(), 0, args.envs);
        gpu_sim.CopyCarStateToHost(gpu_cars.data(), 0, args.envs * cars_per_env);

        uint32_t current_tick = t + 1;
        int snap_idx = -1;
        for (size_t s = 0; s < NUM_SNAPSHOTS; ++s) {
            if (SNAPSHOT_TICKS[s] == current_tick) {
                snap_idx = static_cast<int>(s);
                break;
            }
        }

        for (uint32_t e = 0; e < args.envs; e++) {
            float d_c_pos = 0.0f;
            float d_c_vel = 0.0f;
            float d_c_quat = 0.0f;
            float d_susp = 0.0f;
            float d_boost = 0.0f;
            bool cars_ok = true;
            DifferentialFailure car_fail;

            for (uint32_t c = 0; c < cars_per_env; c++) {
                uint32_t c_idx = e * cars_per_env + c;
                d_c_pos = std::max(d_c_pos, cpu_cars[c_idx].pos.chebyshev_dist(gpu_cars[c_idx].pos));
                d_c_vel = std::max(d_c_vel, cpu_cars[c_idx].vel.chebyshev_dist(gpu_cars[c_idx].vel));
                d_c_quat = std::max(d_c_quat, cpu_cars[c_idx].quat.chebyshev_dist(gpu_cars[c_idx].quat));
                for (int w = 0; w < 4; w++) {
                    d_susp = std::max(d_susp, std::fabs(cpu_cars[c_idx].suspension_lengths[w] - gpu_cars[c_idx].suspension_lengths[w]));
                }
                d_boost = std::max(d_boost, std::fabs(cpu_cars[c_idx].boost - gpu_cars[c_idx].boost));

                DifferentialFailure this_car_fail;
                if (!comparator.CompareCar(t, e, c, cpu_cars[c_idx], gpu_cars[c_idx], this_car_fail)) {
                    cars_ok = false;
                    car_fail = this_car_fail;
                    fail = this_car_fail;
                }
            }

            float d_b_pos = cpu_balls[e].pos.chebyshev_dist(gpu_balls[e].pos);
            float d_b_vel = cpu_balls[e].vel.chebyshev_dist(gpu_balls[e].vel);
            bool ball_ok = comparator.CompareBall(t, e, cpu_balls[e], gpu_balls[e], fail);
            bool step_ok = ball_ok && cars_ok;

            if (t < 1) report.w1.Update(d_c_pos, d_c_vel, d_c_quat, d_b_pos, d_b_vel, d_susp, d_boost, step_ok);
            if (t < 10) report.w10.Update(d_c_pos, d_c_vel, d_c_quat, d_b_pos, d_b_vel, d_susp, d_boost, step_ok);
            if (t < 60) report.w60.Update(d_c_pos, d_c_vel, d_c_quat, d_b_pos, d_b_vel, d_susp, d_boost, step_ok);
            if (t < 120) report.w120.Update(d_c_pos, d_c_vel, d_c_quat, d_b_pos, d_b_vel, d_susp, d_boost, step_ok);
            if (t < 600) report.w600.Update(d_c_pos, d_c_vel, d_c_quat, d_b_pos, d_b_vel, d_susp, d_boost, step_ok);
            if (t < 10000) report.w10000.Update(d_c_pos, d_c_vel, d_c_quat, d_b_pos, d_b_vel, d_susp, d_boost, step_ok);

            if (snap_idx >= 0) {
                const auto& c_cpu = cpu_cars[e];
                const auto& c_gpu = gpu_cars[e];
                const auto& b_cpu = cpu_balls[e];
                const auto& b_gpu = gpu_balls[e];

                Quat q_gpu_adj = c_gpu.quat;
                float q_dot = c_cpu.quat.w * c_gpu.quat.w +
                              c_cpu.quat.x * c_gpu.quat.x +
                              c_cpu.quat.y * c_gpu.quat.y +
                              c_cpu.quat.z * c_gpu.quat.z;
                if (q_dot < 0.0f) {
                    q_gpu_adj.w = -q_gpu_adj.w;
                    q_gpu_adj.x = -q_gpu_adj.x;
                    q_gpu_adj.y = -q_gpu_adj.y;
                    q_gpu_adj.z = -q_gpu_adj.z;
                }

                float cpu_vals[NUM_COMPONENTS] = {
                    c_cpu.pos.x, c_cpu.pos.y, c_cpu.pos.z,
                    c_cpu.vel.x, c_cpu.vel.y, c_cpu.vel.z,
                    c_cpu.quat.w, c_cpu.quat.x, c_cpu.quat.y, c_cpu.quat.z,
                    b_cpu.pos.x, b_cpu.pos.y, b_cpu.pos.z,
                    b_cpu.vel.x, b_cpu.vel.y, b_cpu.vel.z,
                    b_cpu.ang_vel.x, b_cpu.ang_vel.y, b_cpu.ang_vel.z
                };

                float gpu_vals[NUM_COMPONENTS] = {
                    c_gpu.pos.x, c_gpu.pos.y, c_gpu.pos.z,
                    c_gpu.vel.x, c_gpu.vel.y, c_gpu.vel.z,
                    q_gpu_adj.w, q_gpu_adj.x, q_gpu_adj.y, q_gpu_adj.z,
                    b_gpu.pos.x, b_gpu.pos.y, b_gpu.pos.z,
                    b_gpu.vel.x, b_gpu.vel.y, b_gpu.vel.z,
                    b_gpu.ang_vel.x, b_gpu.ang_vel.y, b_gpu.ang_vel.z
                };

                for (uint32_t c = 0; c < NUM_COMPONENTS; ++c) {
                    float abs_err = std::fabs(cpu_vals[c] - gpu_vals[c]);
                    float rel_err = abs_err / (std::fabs(cpu_vals[c]) + 1.0f);
                    report.baseline.RecordSnapshot(static_cast<size_t>(snap_idx), e, c, abs_err, rel_err);
                }
            }

            if ((scenario_name == "kickoff_goalie" || scenario_name == "car_ball_hit") && e == 0) {
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

            if (scenario_name.rfind("ball_", 0) == 0 && scenario_name != "ball_flight" && e == 0) {
                Vec3 g_dt(0.0f, 0.0f, GRAVITY_Z * DELTA_TIME);
                if (t > 0 && report.bounce.bounce_tick_cpu == -1) {
                    Vec3 diff = (cpu_balls[e].vel - prev_cpu_ball_vel[e]) - g_dt;
                    if (diff.length() > 50.0f) {
                        report.bounce.bounce_tick_cpu = static_cast<int>(t);
                    }
                }
                if (t > 0 && report.bounce.bounce_tick_gpu == -1) {
                    Vec3 diff = (gpu_balls[e].vel - prev_gpu_ball_vel[e]) - g_dt;
                    if (diff.length() > 50.0f) {
                        report.bounce.bounce_tick_gpu = static_cast<int>(t);
                    }
                }
                if (report.bounce.bounce_tick_cpu != -1) {
                    int dt = static_cast<int>(t) - report.bounce.bounce_tick_cpu;
                    if (dt == 1) {
                        report.bounce.ball_vel_plus_1_cpu = cpu_balls[e].vel;
                        report.bounce.ball_ang_vel_plus_1_cpu = cpu_balls[e].ang_vel;
                    } else if (dt == 5) {
                        report.bounce.ball_vel_plus_5_cpu = cpu_balls[e].vel;
                        report.bounce.ball_ang_vel_plus_5_cpu = cpu_balls[e].ang_vel;
                    } else if (dt == 30) {
                        report.bounce.ball_vel_plus_30_cpu = cpu_balls[e].vel;
                        report.bounce.ball_ang_vel_plus_30_cpu = cpu_balls[e].ang_vel;
                    }
                }
                if (report.bounce.bounce_tick_gpu != -1) {
                    int dt = static_cast<int>(t) - report.bounce.bounce_tick_gpu;
                    if (dt == 1) {
                        report.bounce.ball_vel_plus_1_gpu = gpu_balls[e].vel;
                        report.bounce.ball_ang_vel_plus_1_gpu = gpu_balls[e].ang_vel;
                    } else if (dt == 5) {
                        report.bounce.ball_vel_plus_5_gpu = gpu_balls[e].vel;
                        report.bounce.ball_ang_vel_plus_5_gpu = gpu_balls[e].ang_vel;
                    } else if (dt == 30) {
                        report.bounce.ball_vel_plus_30_gpu = gpu_balls[e].vel;
                        report.bounce.ball_ang_vel_plus_30_gpu = gpu_balls[e].ang_vel;
                    }
                }
            }

            if (scenario_name == "boost_pad_pickup" && e == 0) {
                int target_pad = (e % 2 == 0) ? 0 : 19;

                bool cpu_pad_active = false;
                float cpu_pad_cd = 0.0f;
                GetCPURefSimBoostPadState(&lockstep_cpu_envs[e], target_pad, cpu_pad_active, cpu_pad_cd);

                uint8_t gpu_pad_active_byte = 0;
                float gpu_pad_cd = 0.0f;
                cudaMemcpy(&gpu_pad_active_byte, &gpu_sim.GetArenaState().pad_is_active[e * MAX_BOOST_PADS + target_pad], sizeof(uint8_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(&gpu_pad_cd, &gpu_sim.GetArenaState().pad_cooldown[e * MAX_BOOST_PADS + target_pad], sizeof(float), cudaMemcpyDeviceToHost);
                bool gpu_pad_active = (gpu_pad_active_byte != 0);

                if (report.boost_pad.pickup_tick_cpu == -1 && cpu_cars[e].boost > 0.0f) {
                    report.boost_pad.pickup_tick_cpu = static_cast<int>(t + 1);
                    report.boost_pad.boost_after_cpu = cpu_cars[e].boost;
                    report.boost_pad.pad_active_after_pickup_cpu = cpu_pad_active;
                    report.boost_pad.cooldown_after_pickup_cpu = cpu_pad_cd;
                }
                if (report.boost_pad.pickup_tick_gpu == -1 && gpu_cars[e].boost > 0.0f) {
                    report.boost_pad.pickup_tick_gpu = static_cast<int>(t + 1);
                    report.boost_pad.boost_after_gpu = gpu_cars[e].boost;
                    report.boost_pad.pad_active_after_pickup_gpu = gpu_pad_active;
                    report.boost_pad.cooldown_after_pickup_gpu = gpu_pad_cd;
                }

                if (report.boost_pad.pickup_tick_cpu != -1 && report.boost_pad.respawn_tick_cpu == -1 && static_cast<int>(t + 1) > report.boost_pad.pickup_tick_cpu) {
                    if (cpu_pad_active) {
                        report.boost_pad.respawn_tick_cpu = static_cast<int>(t + 1);
                    }
                }
                if (report.boost_pad.pickup_tick_gpu != -1 && report.boost_pad.respawn_tick_gpu == -1 && static_cast<int>(t + 1) > report.boost_pad.pickup_tick_gpu) {
                    if (gpu_pad_active) {
                        report.boost_pad.respawn_tick_gpu = static_cast<int>(t + 1);
                    }
                }
            }

            prev_cpu_ball_vel[e] = cpu_balls[e].vel;
            prev_gpu_ball_vel[e] = gpu_balls[e].vel;

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

    if (scenario_name == "boost_pad_pickup") {
        if (report.boost_pad.pickup_tick_cpu == -1 || report.boost_pad.pickup_tick_gpu == -1 ||
            report.boost_pad.pickup_tick_cpu != report.boost_pad.pickup_tick_gpu) {
            report.first_breach_tick = (report.boost_pad.pickup_tick_cpu != -1) ? report.boost_pad.pickup_tick_cpu : 0;
            report.first_breach_attr = "Boost Pad Pickup Mismatch";
        }
        int expected_respawn_delta = report.boost_pad.is_big ? 1201 : 480;
        if (args.ticks >= static_cast<uint32_t>(expected_respawn_delta + 2)) {
            if (report.boost_pad.respawn_tick_cpu != report.boost_pad.respawn_tick_gpu || report.boost_pad.respawn_tick_cpu == -1) {
                report.first_breach_tick = (report.boost_pad.respawn_tick_cpu != -1) ? report.boost_pad.respawn_tick_cpu : expected_respawn_delta;
                report.first_breach_attr = "Boost Pad Respawn Mismatch";
            }
        }
    }

    report.baseline.Finalize(args.ticks);
    return (report.first_breach_tick == -1);
}

void PrintScenarioReportTable(const std::vector<ScenarioReport>& reports, std::ostream& os) {
    os << "\n| Scenario | Window (Ticks) | Car Pos (UU) | Car Vel (UU/s) | Car Quat | Ball Pos (UU) | Ball Vel (UU/s) | First Breach Tick | Status |\n";
    os << "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n";
    for (const auto& rep : reports) {
        auto print_row = [&](const std::string& win_name, const WindowMetrics& m) {
            os << "| " << std::setw(17) << std::left << rep.name
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
        if (rep.ticks_simulated >= 60) print_row("60 (0.5s)", rep.w60);
        if (rep.ticks_simulated >= 120) print_row("120 (1.0s)", rep.w120);
        if (rep.ticks_simulated >= 600) print_row("600 (5.0s)", rep.w600);
        if (rep.ticks_simulated >= 10000) print_row("10000 (83s)", rep.w10000);

        if (!rep.baseline.snapshots.empty()) {
            os << "\n### Milestone 5 Phase 1 Component-Wise Parity Baseline (" << rep.name
               << ", " << rep.num_envs << " environments)\n\n";

            os << "#### Absolute Error (|Δ|) — Median and P95 across Snapshot Ticks\n\n";
            os << "| Component |";
            for (const auto& snap : rep.baseline.snapshots) {
                os << " Tick " << snap.tick << " (Med) | Tick " << snap.tick << " (P95) |";
            }
            os << "\n| :--- |";
            for (size_t s = 0; s < rep.baseline.snapshots.size(); ++s) {
                os << " :--- | :--- |";
            }
            os << "\n";

            for (uint32_t c = 0; c < NUM_COMPONENTS; ++c) {
                os << "| " << std::setw(17) << std::left << GetComponentName(c) << " |";
                for (const auto& snap : rep.baseline.snapshots) {
                    os << " " << std::scientific << std::setprecision(3) << snap.stats[c].median_abs
                       << " | " << std::scientific << std::setprecision(3) << snap.stats[c].p95_abs << " |";
                }
                os << "\n";
            }
            os << "\n";

            os << "#### Relative Error (|Δ| / (|v_cpu| + 1.0)) — Median and P95 across Snapshot Ticks\n\n";
            os << "| Component |";
            for (const auto& snap : rep.baseline.snapshots) {
                os << " Tick " << snap.tick << " (Med) | Tick " << snap.tick << " (P95) |";
            }
            os << "\n| :--- |";
            for (size_t s = 0; s < rep.baseline.snapshots.size(); ++s) {
                os << " :--- | :--- |";
            }
            os << "\n";

            for (uint32_t c = 0; c < NUM_COMPONENTS; ++c) {
                os << "| " << std::setw(17) << std::left << GetComponentName(c) << " |";
                for (const auto& snap : rep.baseline.snapshots) {
                    os << " " << std::scientific << std::setprecision(3) << snap.stats[c].median_rel
                       << " | " << std::scientific << std::setprecision(3) << snap.stats[c].p95_rel << " |";
                }
                os << "\n";
            }
            os << "\n";

            os << "#### Snapshot Detail (Per-Tick Breakdown)\n\n";
            for (const auto& snap : rep.baseline.snapshots) {
                os << "**Tick " << snap.tick << "** (" << rep.num_envs << " environments):\n\n"
                   << "| Component | Median Abs Error | P95 Abs Error | Median Rel Error | P95 Rel Error |\n"
                   << "| :--- | :--- | :--- | :--- | :--- |\n";
                for (uint32_t c = 0; c < NUM_COMPONENTS; ++c) {
                    os << "| " << std::setw(17) << std::left << GetComponentName(c)
                       << " | " << std::scientific << std::setprecision(3) << snap.stats[c].median_abs
                       << " | " << std::scientific << std::setprecision(3) << snap.stats[c].p95_abs
                       << " | " << std::scientific << std::setprecision(3) << snap.stats[c].median_rel
                       << " | " << std::scientific << std::setprecision(3) << snap.stats[c].p95_rel
                       << " |\n";
                }
                os << "\n";
            }
        }

        auto compute_angle = [](const Vec3& v, float& yaw_deg, float& pitch_deg) {
            yaw_deg = std::atan2(v.y, v.x) * 57.29577951308232f;
            float horiz = std::sqrt(v.x * v.x + v.y * v.y);
            pitch_deg = std::atan2(v.z, horiz) * 57.29577951308232f;
        };

        if (rep.name == "kickoff_goalie" || rep.name == "car_ball_hit") {
            float yaw_cpu_1, pitch_cpu_1, yaw_gpu_1, pitch_gpu_1;
            compute_angle(rep.goalie.ball_vel_plus_1_cpu, yaw_cpu_1, pitch_cpu_1);
            compute_angle(rep.goalie.ball_vel_plus_1_gpu, yaw_gpu_1, pitch_gpu_1);

            float yaw_cpu_10, pitch_cpu_10, yaw_gpu_10, pitch_gpu_10;
            compute_angle(rep.goalie.ball_vel_plus_10_cpu, yaw_cpu_10, pitch_cpu_10);
            compute_angle(rep.goalie.ball_vel_plus_10_gpu, yaw_gpu_10, pitch_gpu_10);

            float yaw_cpu_60, pitch_cpu_60, yaw_gpu_60, pitch_gpu_60;
            compute_angle(rep.goalie.ball_vel_plus_60_cpu, yaw_cpu_60, pitch_cpu_60);
            compute_angle(rep.goalie.ball_vel_plus_60_gpu, yaw_gpu_60, pitch_gpu_60);

            os << "\n#### " << (rep.name == "kickoff_goalie" ? "Kickoff Goalie" : "Car-Ball Hit") << " Impact & Collision Gate Analysis\n"
               << "| Metric | CPU Reference | GPU Kernel | Delta |\n"
               << "| :--- | :--- | :--- | :--- |\n"
               << "| First Touch Tick | " << rep.goalie.touch_tick_cpu << " | " << rep.goalie.touch_tick_gpu
               << " | " << std::abs(rep.goalie.touch_tick_cpu - rep.goalie.touch_tick_gpu) << " ticks |\n"
               << "| Car Pos at Impact (UU) | " << rep.goalie.car_pos_impact_cpu << " | " << rep.goalie.car_pos_impact_gpu
               << " | " << std::scientific << std::setprecision(3) << rep.goalie.car_pos_impact_cpu.chebyshev_dist(rep.goalie.car_pos_impact_gpu) << " UU |\n"
               << "| Car Vel at Impact (UU/s) | " << rep.goalie.car_vel_impact_cpu << " | " << rep.goalie.car_vel_impact_gpu
               << " | " << std::scientific << std::setprecision(3) << rep.goalie.car_vel_impact_cpu.chebyshev_dist(rep.goalie.car_vel_impact_gpu) << " UU/s |\n"
               << "| Ball Vel +1 Tick (vx, vy, vz) | (" << std::fixed << std::setprecision(1) << rep.goalie.ball_vel_plus_1_cpu.x << ", " << rep.goalie.ball_vel_plus_1_cpu.y << ", " << rep.goalie.ball_vel_plus_1_cpu.z << ") | ("
               << rep.goalie.ball_vel_plus_1_gpu.x << ", " << rep.goalie.ball_vel_plus_1_gpu.y << ", " << rep.goalie.ball_vel_plus_1_gpu.z << ") | Δ=("
               << std::fabs(rep.goalie.ball_vel_plus_1_cpu.x - rep.goalie.ball_vel_plus_1_gpu.x) << ", "
               << std::fabs(rep.goalie.ball_vel_plus_1_cpu.y - rep.goalie.ball_vel_plus_1_gpu.y) << ", "
               << std::fabs(rep.goalie.ball_vel_plus_1_cpu.z - rep.goalie.ball_vel_plus_1_gpu.z) << ") UU/s |\n"
               << "| Ball Exit Angle +1 Tick (yaw, pitch) | (" << std::fixed << std::setprecision(2) << yaw_cpu_1 << "°, " << pitch_cpu_1 << "°) | ("
               << yaw_gpu_1 << "°, " << pitch_gpu_1 << "°) | Δ=("
               << std::fabs(yaw_cpu_1 - yaw_gpu_1) << "°, " << std::fabs(pitch_cpu_1 - pitch_gpu_1) << "°) |\n"
               << "| Ball Vel +10 Ticks (vx, vy, vz) | (" << std::fixed << std::setprecision(1) << rep.goalie.ball_vel_plus_10_cpu.x << ", " << rep.goalie.ball_vel_plus_10_cpu.y << ", " << rep.goalie.ball_vel_plus_10_cpu.z << ") | ("
               << rep.goalie.ball_vel_plus_10_gpu.x << ", " << rep.goalie.ball_vel_plus_10_gpu.y << ", " << rep.goalie.ball_vel_plus_10_gpu.z << ") | Δ=("
               << std::fabs(rep.goalie.ball_vel_plus_10_cpu.x - rep.goalie.ball_vel_plus_10_gpu.x) << ", "
               << std::fabs(rep.goalie.ball_vel_plus_10_cpu.y - rep.goalie.ball_vel_plus_10_gpu.y) << ", "
               << std::fabs(rep.goalie.ball_vel_plus_10_cpu.z - rep.goalie.ball_vel_plus_10_gpu.z) << ") UU/s |\n"
               << "| Ball Exit Angle +10 Ticks (yaw, pitch) | (" << std::fixed << std::setprecision(2) << yaw_cpu_10 << "°, " << pitch_cpu_10 << "°) | ("
               << yaw_gpu_10 << "°, " << pitch_gpu_10 << "°) | Δ=("
               << std::fabs(yaw_cpu_10 - yaw_gpu_10) << "°, " << std::fabs(pitch_cpu_10 - pitch_gpu_10) << "°) |\n"
               << "| Ball Vel +60 Ticks (vx, vy, vz) | (" << std::fixed << std::setprecision(1) << rep.goalie.ball_vel_plus_60_cpu.x << ", " << rep.goalie.ball_vel_plus_60_cpu.y << ", " << rep.goalie.ball_vel_plus_60_cpu.z << ") | ("
               << rep.goalie.ball_vel_plus_60_gpu.x << ", " << rep.goalie.ball_vel_plus_60_gpu.y << ", " << rep.goalie.ball_vel_plus_60_gpu.z << ") | Δ=("
               << std::fabs(rep.goalie.ball_vel_plus_60_cpu.x - rep.goalie.ball_vel_plus_60_gpu.x) << ", "
               << std::fabs(rep.goalie.ball_vel_plus_60_cpu.y - rep.goalie.ball_vel_plus_60_gpu.y) << ", "
               << std::fabs(rep.goalie.ball_vel_plus_60_cpu.z - rep.goalie.ball_vel_plus_60_gpu.z) << ") UU/s |\n"
               << "| Ball Exit Angle +60 Ticks (yaw, pitch) | (" << std::fixed << std::setprecision(2) << yaw_cpu_60 << "°, " << pitch_cpu_60 << "°) | ("
               << yaw_gpu_60 << "°, " << pitch_gpu_60 << "°) | Δ=("
               << std::fabs(yaw_cpu_60 - yaw_gpu_60) << "°, " << std::fabs(pitch_cpu_60 - pitch_gpu_60) << "°) |\n\n";
        }

        if (rep.name.rfind("ball_", 0) == 0 && rep.name != "ball_flight") {
            os << "\n#### Single-Bounce Impact Telemetry: " << rep.name << "\n"
               << "| Metric | CPU Reference | GPU Kernel | Delta (Abs) |\n"
               << "| :--- | :--- | :--- | :--- |\n"
               << "| Bounce Tick | " << rep.bounce.bounce_tick_cpu << " | " << rep.bounce.bounce_tick_gpu
               << " | " << std::abs(rep.bounce.bounce_tick_cpu - rep.bounce.bounce_tick_gpu) << " ticks |\n"
               << "| Vel +1 Tick (vx, vy, vz) | (" << std::fixed << std::setprecision(2) << rep.bounce.ball_vel_plus_1_cpu.x << ", " << rep.bounce.ball_vel_plus_1_cpu.y << ", " << rep.bounce.ball_vel_plus_1_cpu.z << ") | ("
               << rep.bounce.ball_vel_plus_1_gpu.x << ", " << rep.bounce.ball_vel_plus_1_gpu.y << ", " << rep.bounce.ball_vel_plus_1_gpu.z << ") | Δ=("
               << std::fabs(rep.bounce.ball_vel_plus_1_cpu.x - rep.bounce.ball_vel_plus_1_gpu.x) << ", "
               << std::fabs(rep.bounce.ball_vel_plus_1_cpu.y - rep.bounce.ball_vel_plus_1_gpu.y) << ", "
               << std::fabs(rep.bounce.ball_vel_plus_1_cpu.z - rep.bounce.ball_vel_plus_1_gpu.z) << ") UU/s |\n"
               << "| Spin +1 Tick (wx, wy, wz) | (" << rep.bounce.ball_ang_vel_plus_1_cpu.x << ", " << rep.bounce.ball_ang_vel_plus_1_cpu.y << ", " << rep.bounce.ball_ang_vel_plus_1_cpu.z << ") | ("
               << rep.bounce.ball_ang_vel_plus_1_gpu.x << ", " << rep.bounce.ball_ang_vel_plus_1_gpu.y << ", " << rep.bounce.ball_ang_vel_plus_1_gpu.z << ") | Δ=("
               << std::fabs(rep.bounce.ball_ang_vel_plus_1_cpu.x - rep.bounce.ball_ang_vel_plus_1_gpu.x) << ", "
               << std::fabs(rep.bounce.ball_ang_vel_plus_1_cpu.y - rep.bounce.ball_ang_vel_plus_1_gpu.y) << ", "
               << std::fabs(rep.bounce.ball_ang_vel_plus_1_cpu.z - rep.bounce.ball_ang_vel_plus_1_gpu.z) << ") rad/s |\n"
               << "| Vel +5 Ticks (vx, vy, vz) | (" << rep.bounce.ball_vel_plus_5_cpu.x << ", " << rep.bounce.ball_vel_plus_5_cpu.y << ", " << rep.bounce.ball_vel_plus_5_cpu.z << ") | ("
               << rep.bounce.ball_vel_plus_5_gpu.x << ", " << rep.bounce.ball_vel_plus_5_gpu.y << ", " << rep.bounce.ball_vel_plus_5_gpu.z << ") | Δ=("
               << std::fabs(rep.bounce.ball_vel_plus_5_cpu.x - rep.bounce.ball_vel_plus_5_gpu.x) << ", "
               << std::fabs(rep.bounce.ball_vel_plus_5_cpu.y - rep.bounce.ball_vel_plus_5_gpu.y) << ", "
               << std::fabs(rep.bounce.ball_vel_plus_5_cpu.z - rep.bounce.ball_vel_plus_5_gpu.z) << ") UU/s |\n"
               << "| Spin +5 Ticks (wx, wy, wz) | (" << rep.bounce.ball_ang_vel_plus_5_cpu.x << ", " << rep.bounce.ball_ang_vel_plus_5_cpu.y << ", " << rep.bounce.ball_ang_vel_plus_5_cpu.z << ") | ("
               << rep.bounce.ball_ang_vel_plus_5_gpu.x << ", " << rep.bounce.ball_ang_vel_plus_5_gpu.y << ", " << rep.bounce.ball_ang_vel_plus_5_gpu.z << ") | Δ=("
               << std::fabs(rep.bounce.ball_ang_vel_plus_5_cpu.x - rep.bounce.ball_ang_vel_plus_5_gpu.x) << ", "
               << std::fabs(rep.bounce.ball_ang_vel_plus_5_cpu.y - rep.bounce.ball_ang_vel_plus_5_gpu.y) << ", "
               << std::fabs(rep.bounce.ball_ang_vel_plus_5_cpu.z - rep.bounce.ball_ang_vel_plus_5_gpu.z) << ") rad/s |\n"
               << "| Vel +30 Ticks (vx, vy, vz) | (" << rep.bounce.ball_vel_plus_30_cpu.x << ", " << rep.bounce.ball_vel_plus_30_cpu.y << ", " << rep.bounce.ball_vel_plus_30_cpu.z << ") | ("
               << rep.bounce.ball_vel_plus_30_gpu.x << ", " << rep.bounce.ball_vel_plus_30_gpu.y << ", " << rep.bounce.ball_vel_plus_30_gpu.z << ") | Δ=("
               << std::fabs(rep.bounce.ball_vel_plus_30_cpu.x - rep.bounce.ball_vel_plus_30_gpu.x) << ", "
               << std::fabs(rep.bounce.ball_vel_plus_30_cpu.y - rep.bounce.ball_vel_plus_30_gpu.y) << ", "
               << std::fabs(rep.bounce.ball_vel_plus_30_cpu.z - rep.bounce.ball_vel_plus_30_gpu.z) << ") UU/s |\n"
               << "| Spin +30 Ticks (wx, wy, wz) | (" << rep.bounce.ball_ang_vel_plus_30_cpu.x << ", " << rep.bounce.ball_ang_vel_plus_30_cpu.y << ", " << rep.bounce.ball_ang_vel_plus_30_cpu.z << ") | ("
               << rep.bounce.ball_ang_vel_plus_30_gpu.x << ", " << rep.bounce.ball_ang_vel_plus_30_gpu.y << ", " << rep.bounce.ball_ang_vel_plus_30_gpu.z << ") | Δ=("
               << std::fabs(rep.bounce.ball_ang_vel_plus_30_cpu.x - rep.bounce.ball_ang_vel_plus_30_gpu.x) << ", "
               << std::fabs(rep.bounce.ball_ang_vel_plus_30_cpu.y - rep.bounce.ball_ang_vel_plus_30_gpu.y) << ", "
                << std::fabs(rep.bounce.ball_ang_vel_plus_30_cpu.z - rep.bounce.ball_ang_vel_plus_30_gpu.z) << ") rad/s |\n\n";
        }

        if (rep.name == "boost_pad_pickup") {
            int expected_respawn_delta = rep.boost_pad.is_big ? 1201 : 480;
            float expected_boost = rep.boost_pad.is_big ? 100.0f : 12.0f;
            int respawn_delta_cpu = (rep.boost_pad.pickup_tick_cpu != -1 && rep.boost_pad.respawn_tick_cpu != -1)
                ? (rep.boost_pad.respawn_tick_cpu - rep.boost_pad.pickup_tick_cpu) : -1;
            int respawn_delta_gpu = (rep.boost_pad.pickup_tick_gpu != -1 && rep.boost_pad.respawn_tick_gpu != -1)
                ? (rep.boost_pad.respawn_tick_gpu - rep.boost_pad.pickup_tick_gpu) : -1;

            os << "\n#### Boost Pad Pickup & Respawn Telemetry: Pad " << rep.boost_pad.target_pad_idx
               << " (" << (rep.boost_pad.is_big ? "Big Pad, +100 boost" : "Small Pad, +12 boost") << ")\n"
               << "| Metric | CPU Reference | GPU Kernel | Delta / Parity |\n"
               << "| :--- | :--- | :--- | :--- |\n"
               << "| Initial Boost | " << std::fixed << std::setprecision(1) << rep.boost_pad.boost_before_cpu << " | " << rep.boost_pad.boost_before_gpu
               << " | Δ=" << std::fabs(rep.boost_pad.boost_before_cpu - rep.boost_pad.boost_before_gpu) << " (Bit-exact) |\n"
               << "| Pickup Tick | " << rep.boost_pad.pickup_tick_cpu << " | " << rep.boost_pad.pickup_tick_gpu
               << " | " << (rep.boost_pad.pickup_tick_cpu == rep.boost_pad.pickup_tick_gpu ? "MATCH" : "MISMATCH") << " |\n"
               << "| Post-Pickup Boost | " << rep.boost_pad.boost_after_cpu << " | " << rep.boost_pad.boost_after_gpu
               << " | Δ=" << std::fabs(rep.boost_pad.boost_after_cpu - rep.boost_pad.boost_after_gpu) << " (Bit-exact, Expected " << expected_boost << ") |\n"
               << "| Pad Active Post-Pickup | " << (rep.boost_pad.pad_active_after_pickup_cpu ? "true" : "false")
               << " | " << (rep.boost_pad.pad_active_after_pickup_gpu ? "true" : "false")
               << " | " << (rep.boost_pad.pad_active_after_pickup_cpu == rep.boost_pad.pad_active_after_pickup_gpu ? "MATCH (Deactivated)" : "MISMATCH") << " |\n"
               << "| Cooldown Assigned | " << std::fixed << std::setprecision(1) << rep.boost_pad.cooldown_after_pickup_cpu << " s"
               << " | " << rep.boost_pad.cooldown_after_pickup_gpu << " s"
               << " | Δ=" << std::scientific << std::setprecision(3) << std::fabs(rep.boost_pad.cooldown_after_pickup_cpu - rep.boost_pad.cooldown_after_pickup_gpu) << " s |\n"
               << "| Respawn Tick | " << rep.boost_pad.respawn_tick_cpu << " | " << rep.boost_pad.respawn_tick_gpu
               << " | " << (rep.boost_pad.respawn_tick_cpu == rep.boost_pad.respawn_tick_gpu ? "MATCH" : "MISMATCH") << " |\n"
               << "| Cooldown Duration (Ticks) | " << respawn_delta_cpu << " ticks | " << respawn_delta_gpu << " ticks"
               << " | Exact (Expected " << expected_respawn_delta << " ticks) |\n\n";
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
    c.vel.x += 1e-3f;
    c.vel.y += 1e-3f;
    float half_d_yaw = 0.5e-3f;
    c.quat = (c.quat * Quat(std::cos(half_d_yaw), 0.0f, 0.0f, std::sin(half_d_yaw))).normalized();
    cpu2.SetCarState(0, c);

    BallStatePOD b;
    cpu2.GetBallState(b);
    // Perturb ball position by 1e-3 UU in each component
    b.pos.x += 1e-3f;
    b.pos.y += 1e-3f;
    b.pos.z += 1e-3f;
    cpu2.SetBallState(b);

    DeterministicInputGenerator gen(seed);
    std::cout << "\n#### CPU vs CPU Perturbation Analysis (1e-3 UU ball pos perturbation) - Scenario: " << scenario << "\n"
              << "| Window | Ball ΔPos (x, y, z) UU | Ball ΔVel (vx, vy, vz) UU/s | Ball ΔSpin (wx, wy, wz) rad/s |\n"
              << "| :--- | :--- | :--- | :--- |\n";

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

        if (t == 119 || t == 239 || t == 359 || t == 479 || t == 599 || t == ticks - 1) {
            Vec3 d_b_pos(std::fabs(b1.pos.x - b2.pos.x), std::fabs(b1.pos.y - b2.pos.y), std::fabs(b1.pos.z - b2.pos.z));
            Vec3 d_b_vel(std::fabs(b1.vel.x - b2.vel.x), std::fabs(b1.vel.y - b2.vel.y), std::fabs(b1.vel.z - b2.vel.z));
            Vec3 d_b_ang(std::fabs(b1.ang_vel.x - b2.ang_vel.x), std::fabs(b1.ang_vel.y - b2.ang_vel.y), std::fabs(b1.ang_vel.z - b2.ang_vel.z));

            std::cout << "| " << std::setw(14) << std::left << (std::to_string(t + 1) + " ticks")
                      << " | (" << std::scientific << std::setprecision(2) << d_b_pos.x << ", " << d_b_pos.y << ", " << d_b_pos.z << ")"
                      << " | (" << d_b_vel.x << ", " << d_b_vel.y << ", " << d_b_vel.z << ")"
                      << " | (" << d_b_ang.x << ", " << d_b_ang.y << ", " << d_b_ang.z << ") |\n";
        }
    }
}

struct ThresholdLimits {
    float max_car_pos = 1e9f;
    float max_car_vel = 1e9f;
    float max_car_quat = 1e9f;
    float max_ball_pos = 1e9f;
    float max_ball_vel = 1e9f;
};

inline bool LoadParityThresholds(const std::string& path, std::map<std::string, std::map<int, ThresholdLimits>>& out_thresholds) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[-] Error: Unable to open parity thresholds file: " << path << "\n";
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    size_t i = 0;
    auto skip_whitespace = [&]() {
        while (i < json.size() && (json[i] == ' ' || json[i] == '\t' || json[i] == '\r' || json[i] == '\n')) {
            i++;
        }
    };

    auto parse_string = [&]() -> std::string {
        skip_whitespace();
        if (i >= json.size() || json[i] != '"') return "";
        i++; // skip opening quote
        size_t start = i;
        while (i < json.size() && json[i] != '"') {
            if (json[i] == '\\' && i + 1 < json.size()) i += 2;
            else i++;
        }
        std::string s = json.substr(start, i - start);
        if (i < json.size() && json[i] == '"') i++; // skip closing quote
        return s;
    };

    size_t scn_pos = json.find("\"scenarios\"");
    if (scn_pos == std::string::npos) {
        std::cerr << "[-] Error: 'scenarios' key not found in " << path << "\n";
        return false;
    }
    i = scn_pos + 11;
    skip_whitespace();
    if (i < json.size() && json[i] == ':') i++;
    skip_whitespace();
    if (i >= json.size() || json[i] != '{') return false;
    i++; // enter scenarios object

    while (i < json.size()) {
        skip_whitespace();
        if (i < json.size() && json[i] == '}') { i++; break; }
        std::string scn_name = parse_string();
        if (scn_name.empty()) { i++; continue; }
        skip_whitespace();
        if (i < json.size() && json[i] == ':') i++;
        skip_whitespace();
        if (i >= json.size() || json[i] != '{') break;
        i++; // enter scenario definition

        while (i < json.size()) {
            skip_whitespace();
            if (i < json.size() && json[i] == '}') { i++; break; }
            std::string key = parse_string();
            skip_whitespace();
            if (i < json.size() && json[i] == ':') i++;
            skip_whitespace();
            if (key == "windows") {
                if (i < json.size() && json[i] == '{') i++;
                while (i < json.size()) {
                    skip_whitespace();
                    if (i < json.size() && json[i] == '}') { i++; break; }
                    std::string win_tick_str = parse_string();
                    if (win_tick_str.empty()) { i++; continue; }
                    int win_tick = std::stoi(win_tick_str);
                    skip_whitespace();
                    if (i < json.size() && json[i] == ':') i++;
                    skip_whitespace();
                    if (i < json.size() && json[i] == '{') i++;
                    ThresholdLimits limits;
                    while (i < json.size()) {
                        skip_whitespace();
                        if (i < json.size() && json[i] == '}') { i++; break; }
                        std::string metric_name = parse_string();
                        skip_whitespace();
                        if (i < json.size() && json[i] == ':') i++;
                        skip_whitespace();
                        size_t val_start = i;
                        while (i < json.size() && json[i] != ',' && json[i] != '}' && json[i] != ' ' && json[i] != '\n' && json[i] != '\r') {
                            i++;
                        }
                        std::string val_str = json.substr(val_start, i - val_start);
                        float val = std::strtof(val_str.c_str(), nullptr);
                        if (metric_name == "max_car_pos") limits.max_car_pos = val;
                        else if (metric_name == "max_car_vel") limits.max_car_vel = val;
                        else if (metric_name == "max_car_quat") limits.max_car_quat = val;
                        else if (metric_name == "max_ball_pos") limits.max_ball_pos = val;
                        else if (metric_name == "max_ball_vel") limits.max_ball_vel = val;
                        skip_whitespace();
                        if (i < json.size() && json[i] == ',') i++;
                    }
                    out_thresholds[scn_name][win_tick] = limits;
                    skip_whitespace();
                    if (i < json.size() && json[i] == ',') i++;
                }
            } else {
                int depth = 0;
                while (i < json.size()) {
                    if (json[i] == '{' || json[i] == '[') depth++;
                    else if (json[i] == '}' || json[i] == ']') {
                        if (depth == 0) break;
                        depth--;
                    } else if (json[i] == ',' && depth == 0) {
                        break;
                    }
                    i++;
                }
            }
            skip_whitespace();
            if (i < json.size() && json[i] == ',') i++;
        }
        skip_whitespace();
        if (i < json.size() && json[i] == ',') i++;
    }

    return true;
}

inline bool ValidateScenarioThresholds(
    const ScenarioReport& rep,
    const std::map<std::string, std::map<int, ThresholdLimits>>& all_thresholds,
    std::ostream& os)
{
    auto it = all_thresholds.find(rep.name);
    if (it == all_thresholds.end()) {
        for (const auto& kv : all_thresholds) {
            if (kv.first.find(rep.name) != std::string::npos || rep.name.find(kv.first) != std::string::npos) {
                it = all_thresholds.find(kv.first);
                break;
            }
        }
    }

    if (it == all_thresholds.end()) {
        os << "[Warning] No regression thresholds defined for scenario '" << rep.name << "'\n";
        return true;
    }

    bool passed = true;
    const auto& win_thresholds = it->second;

    auto check_window = [&](int win_tick, const WindowMetrics& m, const std::string& win_label) {
        auto w_it = win_thresholds.find(win_tick);
        if (w_it == win_thresholds.end()) return;
        const auto& lim = w_it->second;

        auto check_metric = [&](const char* name, float actual, float limit) {
            if (actual > limit) {
                os << "[-] Regression Check FAILED for '" << rep.name << "' at Window " << win_label
                   << " (" << win_tick << " ticks): " << name << " " << std::scientific << std::setprecision(3)
                   << actual << " > threshold " << limit << "\n";
                passed = false;
            }
        };

        check_metric("Car Pos", m.max_car_pos, lim.max_car_pos);
        check_metric("Car Vel", m.max_car_vel, lim.max_car_vel);
        check_metric("Car Quat", m.max_car_quat, lim.max_car_quat);
        check_metric("Ball Pos", m.max_ball_pos, lim.max_ball_pos);
        check_metric("Ball Vel", m.max_ball_vel, lim.max_ball_vel);
    };

    check_window(1, rep.w1, "1");
    check_window(10, rep.w10, "10");
    if (rep.ticks_simulated >= 60) check_window(60, rep.w60, "60");
    if (rep.ticks_simulated >= 120) check_window(120, rep.w120, "120");
    if (rep.ticks_simulated >= 600) check_window(600, rep.w600, "600");

    if (passed) {
        os << "[+] Parity thresholds check PASSED for scenario '" << rep.name << "'\n";
    }
    return passed;
}

int main(int argc, char** argv) {
    RegisterAllScenarios();
    HarnessArgs args = ParseArgs(argc, argv);

    if (args.cpu_perturb) {
        std::vector<std::string> scns;
        if (args.scenario == "all") {
            scns = ScenarioRegistry::Instance().GetAllNames();
        } else if (args.scenario == "ball_suite") {
            for (const auto& scn : ScenarioRegistry::Instance().GetAll()) {
                if (scn->IsBallBounce() || scn->GetName() == "ball_flight") {
                    scns.push_back(scn->GetName());
                }
            }
        } else {
            scns = {args.scenario};
        }
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

    bool skip_serialization = args.baseline_mode || (args.report_mode && args.envs >= 256);
    if (skip_serialization) {
        std::cout << "[Info] Large batch or baseline mode detected (envs=" << args.envs
                  << ", report=" << (args.report_mode ? "true" : "false")
                  << ", baseline=" << (args.baseline_mode ? "true" : "false")
                  << "). Bypassing .rsgold serialization (Tests 1-4) and proceeding directly to Test 5.\n\n";
    } else {
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
    }

    // ------------------------------------------------------------------
    // Test 5: Lockstep Differential Simulation
    // ------------------------------------------------------------------
    std::cout << "[Test 5/5] Lockstep Differential Simulation: GPU vs CPU Oracle ("
              << args.ticks << " ticks, " << args.envs << " envs)...\n";

    std::vector<std::string> scenarios_to_run;
    if (args.scenario == "all") {
        scenarios_to_run = ScenarioRegistry::Instance().GetAllNames();
    } else if (args.scenario == "ball_suite") {
        for (const auto& scn : ScenarioRegistry::Instance().GetAll()) {
            if (scn->IsBallBounce() || scn->GetName() == "ball_flight") {
                scenarios_to_run.push_back(scn->GetName());
            }
        }
    } else {
        if (!ScenarioRegistry::Instance().Has(args.scenario)) {
            std::cerr << "[-] Error: Unknown scenario '" << args.scenario << "'\n";
            return 1;
        }
        scenarios_to_run = {args.scenario};
    }

    std::vector<ScenarioReport> reports;
    bool all_passed = true;

    for (const auto& scn : scenarios_to_run) {
        std::cout << "  --> Running scenario: " << scn << " (" << args.ticks << " ticks)...\n";
        ScenarioReport rep;
        bool fail_fast = !args.report_mode && !args.check_mode;
        bool scn_ok = RunScenarioDifferential(scn, args, tol, comparator, rep, fail_fast);
        reports.push_back(rep);
        if (!scn_ok && fail_fast) {
            all_passed = false;
            return 1;
        } else if (!scn_ok) {
            all_passed = false;
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

    if (args.check_mode) {
        std::cout << "\n[Regression Guard] Validating parity thresholds against: " << args.check_file << "\n";
        std::map<std::string, std::map<int, ThresholdLimits>> thresholds;
        if (!LoadParityThresholds(args.check_file, thresholds)) {
            std::cerr << "[-] Error: Failed to load parity thresholds from " << args.check_file << "\n";
            return 1;
        }

        bool thresholds_ok = true;
        for (const auto& rep : reports) {
            if (!ValidateScenarioThresholds(rep, thresholds, std::cout)) {
                thresholds_ok = false;
            }
        }

        if (!thresholds_ok) {
            std::cerr << "\n[-] FATAL: Parity threshold limit exceeded! Regression guard failed.\n";
            return 1;
        }
        std::cout << "\n[+] SUCCESS: All parity thresholds verified within limits (+25% buffer)!\n";
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
    } else if (args.check_mode) {
        std::cout << "======================================================================\n"
                  << "  PARITY THRESHOLDS CHECK PASSED SUCCESSFULLY ACROSS ALL SCENARIOS!   \n"
                  << "======================================================================\n";
        return 0;
    } else {
        std::cerr << "[-] Differential Parity Failed\n";
        return 1;
    }
}
