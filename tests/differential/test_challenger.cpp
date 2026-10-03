#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <memory>
#include <cstring>

#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/sim_context.cuh"
#include "golden_master.h"
#include "cpu_ref_sim.h"

using namespace rocketsim_cuda;

// Helper assertion macro
#define CHALLENGER_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] [CHALLENGER FAILURE] " << msg \
                      << " (Line " << __LINE__ << ")\n"; \
            return false; \
        } \
    } while (0)

#define CHALLENGER_TEST_START(name) \
    std::cout << ">>> Running " << name << " ...\n"

#define CHALLENGER_TEST_PASS(name) \
    std::cout << "[+] PASSED: " << name << "\n\n"

// ---------------------------------------------------------------------------
// Test 1: Full-Spectrum Differential Comparator Sensitivity & Boundaries
// ---------------------------------------------------------------------------
bool TestComparatorSensitivity() {
    CHALLENGER_TEST_START("Test 1: Full-Spectrum Differential Comparator Sensitivity & Boundaries");

    DifferentialTolerance tol;
    DifferentialComparator comp(tol);
    DifferentialFailure fail;

    // Use base position at origin to avoid ULP quantisation masking of 1e-4 thresholds
    BallStatePOD base_ball;
    base_ball.pos = Vec3(0.0f, 0.0f, 0.0f);
    base_ball.vel = Vec3(0.0f, 0.0f, 0.0f);
    base_ball.quat = Quat(1.0f, 0.0f, 0.0f, 0.0f);
    base_ball.ang_vel = Vec3(0.0f, 0.0f, 0.0f);

    // 1.1 Baseline exact match must pass
    CHALLENGER_ASSERT(comp.CompareBall(0, 0, base_ball, base_ball, fail), "Exact ball match failed");

    // 1.2 Ball Position: Pos X near origin (ULP around 1.0 is ~1.19e-7, near 0 is tiny)
    {
        BallStatePOD p = base_ball;
        p.pos.x = tol.pos_uu * 0.95f;
        CHALLENGER_ASSERT(comp.CompareBall(0, 0, base_ball, p, fail), "Sub-threshold ball pos.x rejected");

        p.pos.x = tol.pos_uu * 1.05f;
        CHALLENGER_ASSERT(!comp.CompareBall(0, 0, base_ball, p, fail), "Super-threshold ball pos.x accepted");
        CHALLENGER_ASSERT(fail.attribute.find("Position") != std::string::npos, "Wrong attribute for ball pos");
    }

    // 1.3 Ball Position: Pos Y & Pos Z
    {
        BallStatePOD p = base_ball;
        p.pos.y = -tol.pos_uu * 1.05f;
        CHALLENGER_ASSERT(!comp.CompareBall(0, 0, base_ball, p, fail), "Super-threshold ball pos.y accepted");

        p = base_ball;
        p.pos.z = tol.pos_uu * 1.05f;
        CHALLENGER_ASSERT(!comp.CompareBall(0, 0, base_ball, p, fail), "Super-threshold ball pos.z accepted");
    }

    // 1.4 Ball Velocity: Vel X, Y, Z
    {
        BallStatePOD p = base_ball;
        p.vel.x = tol.vel_uus * 0.95f;
        CHALLENGER_ASSERT(comp.CompareBall(0, 0, base_ball, p, fail), "Sub-threshold ball vel rejected");

        p.vel.x = tol.vel_uus * 1.05f;
        CHALLENGER_ASSERT(!comp.CompareBall(0, 0, base_ball, p, fail), "Super-threshold ball vel.x accepted");
        CHALLENGER_ASSERT(fail.attribute.find("Velocity") != std::string::npos, "Wrong attribute for ball vel");
    }

    {
        BallStatePOD p = base_ball;
        p.ang_vel.y = tol.ang_vel_rads * 1.05f;
        CHALLENGER_ASSERT(!comp.CompareBall(0, 0, base_ball, p, fail), "Super-threshold ball ang_vel accepted");
        CHALLENGER_ASSERT(fail.attribute.find("Angular Velocity") != std::string::npos, "Wrong attribute for ball ang_vel");
    }

    // 1.6 Large Coordinate ULP Resilience Check (e.g. Car at arena wall X = 4096.0f)
    // At 4096.0f, ULP = 2^(12 - 23) = 2^(-11) = 0.000488 UU!
    // A perturbation must be larger than ULP to be representable as distinct in float.
    {
        BallStatePOD far_ball;
        far_ball.pos = Vec3(4000.0f, 5000.0f, 1000.0f);
        BallStatePOD perturbed_far = far_ball;
        perturbed_far.pos.x += 0.002f; // Distinct representation above 1e-4 threshold
        CHALLENGER_ASSERT(!comp.CompareBall(0, 0, far_ball, perturbed_far, fail), "Super-threshold far ball accepted");
    }

    // 1.7 Car State: Pos, Vel, Quat, AngVel, Suspension, Boost
    CarStatePOD base_car;
    base_car.pos = Vec3(0.0f, 0.0f, 17.0f);
    base_car.vel = Vec3(0.0f, 0.0f, 0.0f);
    base_car.quat = Quat::identity();
    base_car.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
    base_car.boost = 33.3f;
    for (int w = 0; w < 4; w++) base_car.suspension_lengths[w] = 12.0f;

    CHALLENGER_ASSERT(comp.CompareCar(0, 0, 0, base_car, base_car, fail), "Exact car match failed");

    // Suspension per wheel
    for (int w = 0; w < 4; w++) {
        CarStatePOD c = base_car;
        c.suspension_lengths[w] += tol.suspension_uu * 0.95f;
        CHALLENGER_ASSERT(comp.CompareCar(0, 0, 0, base_car, c, fail), "Sub-threshold car susp rejected");

        c.suspension_lengths[w] = base_car.suspension_lengths[w] + tol.suspension_uu * 1.5f;
        CHALLENGER_ASSERT(!comp.CompareCar(0, 0, 0, base_car, c, fail), "Super-threshold car susp accepted");
        CHALLENGER_ASSERT(fail.attribute.find("Wheel " + std::to_string(w)) != std::string::npos, "Wrong wheel reported");
    }

    // Boost: bit-exact check when tol.boost == 0.0f
    // Even a single ULP modification (the smallest possible bit alteration) must be detected
    {
        CarStatePOD c = base_car;
        c.boost = std::nextafterf(base_car.boost, 100.0f);
        CHALLENGER_ASSERT(c.boost != base_car.boost, "nextafterf failed to alter float representation");
        CHALLENGER_ASSERT(!comp.CompareCar(0, 0, 0, base_car, c, fail), "1-ULP boost delta accepted when tol.boost is 0.0");
        CHALLENGER_ASSERT(fail.attribute.find("Boost") != std::string::npos, "Wrong attribute for boost");
    }

    CHALLENGER_TEST_PASS("Test 1: Full-Spectrum Differential Comparator Sensitivity & Boundaries");
    return true;
}

// ---------------------------------------------------------------------------
// Test 2: Quaternion Antipodal Invariance and Rotational Perturbation
// ---------------------------------------------------------------------------
bool TestQuaternionAntipodalInvariance() {
    CHALLENGER_TEST_START("Test 2: Quaternion Antipodal Invariance & Rotational Perturbation");

    DifferentialTolerance tol;
    DifferentialComparator comp(tol);
    DifferentialFailure fail;

    BallStatePOD ball;
    ball.pos = Vec3(0, 0, 0);
    ball.vel = Vec3(0, 0, 0);
    ball.ang_vel = Vec3(0, 0, 0);
    ball.quat = Quat(0.5f, 0.5f, 0.5f, 0.5f); // Normalized rotation

    // 2.1 Identical physical rotation represented by antipodal quaternion -q
    BallStatePOD antipodal_ball = ball;
    antipodal_ball.quat = Quat(-ball.quat.w, -ball.quat.x, -ball.quat.y, -ball.quat.z);
    CHALLENGER_ASSERT(comp.CompareBall(0, 0, ball, antipodal_ball, fail), "Antipodal quaternion -q rejected!");

    // 2.2 Perturbation around -q: sub-threshold must pass
    antipodal_ball.quat.x = -ball.quat.x + tol.quat * 0.9f;
    CHALLENGER_ASSERT(comp.CompareBall(0, 0, ball, antipodal_ball, fail), "Sub-threshold antipodal perturbation rejected");

    // 2.3 Perturbation around -q: super-threshold must fail
    antipodal_ball.quat.x = -ball.quat.x + tol.quat * 1.5f;
    CHALLENGER_ASSERT(!comp.CompareBall(0, 0, ball, antipodal_ball, fail), "Super-threshold antipodal perturbation accepted");
    CHALLENGER_ASSERT(fail.attribute.find("Quaternion") != std::string::npos, "Wrong attribute for quat failure");

    CHALLENGER_TEST_PASS("Test 2: Quaternion Antipodal Invariance & Rotational Perturbation");
    return true;
}

// ---------------------------------------------------------------------------
// Test 3: .rsgold File Format Integrity, Binary Serialization & Corruption Resistance
// ---------------------------------------------------------------------------
bool TestRsGoldFormatIntegrity() {
    CHALLENGER_TEST_START("Test 3: .rsgold File Format Integrity, Binary Serialization & Corruption Resistance");

    const std::string test_file = "challenger_format_test.rsgold";
    const uint32_t ENVS = 4;
    const uint32_t CARS_PER_ENV = 1;
    const uint32_t TICKS = 50;
    const uint32_t SEED = 777;

    // 3.1 Write file
    {
        RsGoldWriter writer;
        CHALLENGER_ASSERT(writer.Open(test_file, ENVS, CARS_PER_ENV, SEED), "Failed to open RsGoldWriter");

        std::vector<CarControls> controls(ENVS);
        std::vector<BallStatePOD> balls(ENVS);
        std::vector<CarStatePOD> cars(ENVS);

        for (uint32_t t = 0; t < TICKS; t++) {
            for (uint32_t e = 0; e < ENVS; e++) {
                controls[e].throttle = static_cast<float>(t) * 0.01f;
                balls[e].pos = Vec3(static_cast<float>(t), static_cast<float>(e), 100.0f);
                cars[e].pos = Vec3(static_cast<float>(t * 2), static_cast<float>(e * 2), 17.0f);
            }
            writer.WriteTick(controls.data(), balls.data(), cars.data());
        }
        writer.Close();
    }

    // 3.2 Check file size on disk matches exact analytical formula
    size_t expected_size = sizeof(RsGoldHeader) + TICKS * ENVS * (sizeof(RsGoldCarControls) + sizeof(RsGoldBallRecord) + sizeof(RsGoldCarRecord));
    CHALLENGER_ASSERT(sizeof(RsGoldHeader) == 64, "RsGoldHeader size is not 64 bytes");
    CHALLENGER_ASSERT(sizeof(RsGoldCarControls) == 24, "RsGoldCarControls size is not 24 bytes");
    CHALLENGER_ASSERT(sizeof(RsGoldBallRecord) == 52, "RsGoldBallRecord size is not 52 bytes");
    CHALLENGER_ASSERT(sizeof(RsGoldCarRecord) == 84, "RsGoldCarRecord size is not 84 bytes");

    std::ifstream in(test_file, std::ios::binary | std::ios::ate);
    CHALLENGER_ASSERT(in.is_open(), "Cannot open generated .rsgold file");
    size_t actual_size = static_cast<size_t>(in.tellg());
    in.close();
    CHALLENGER_ASSERT(actual_size == expected_size, "File size (" + std::to_string(actual_size) + ") != expected (" + std::to_string(expected_size) + ")");

    // 3.3 Verify Bit-Exact Deserialization
    {
        RsGoldReader reader;
        CHALLENGER_ASSERT(reader.Open(test_file), "Failed to open RsGoldReader on valid file");
        const auto& hdr = reader.GetHeader();
        CHALLENGER_ASSERT(hdr.magic == 0x5253474D, "Magic mismatch");
        CHALLENGER_ASSERT(hdr.version == 1, "Version mismatch");
        CHALLENGER_ASSERT(hdr.num_envs == ENVS, "Num envs mismatch");
        CHALLENGER_ASSERT(hdr.total_ticks == TICKS, "Total ticks mismatch");
        CHALLENGER_ASSERT(hdr.seed == SEED, "Seed mismatch");

        std::vector<CarControls> read_controls(ENVS);
        std::vector<BallStatePOD> read_balls(ENVS);
        std::vector<CarStatePOD> read_cars(ENVS);

        for (uint32_t t = 0; t < TICKS; t++) {
            CHALLENGER_ASSERT(reader.ReadTick(read_controls.data(), read_balls.data(), read_cars.data()), "ReadTick failed at tick " + std::to_string(t));
            for (uint32_t e = 0; e < ENVS; e++) {
                CHALLENGER_ASSERT(read_balls[e].pos.x == static_cast<float>(t), "Bit-exact mismatch in ball pos.x");
                CHALLENGER_ASSERT(read_cars[e].pos.x == static_cast<float>(t * 2), "Bit-exact mismatch in car pos.x");
            }
        }
        // Reading past EOF must fail cleanly
        CHALLENGER_ASSERT(!reader.ReadTick(read_controls.data(), read_balls.data(), read_cars.data()), "ReadTick succeeded past EOF");
        reader.Close();
    }

    // 3.4 Adversarial Corruption 1: Invalid Magic
    {
        std::string corrupt_magic = "corrupt_magic.rsgold";
        std::ifstream src(test_file, std::ios::binary);
        std::ofstream dst(corrupt_magic, std::ios::binary);
        dst << src.rdbuf();
        src.close();
        dst.close();

        // Overwrite magic with 0xDEADBEEF
        std::fstream f(corrupt_magic, std::ios::in | std::ios::out | std::ios::binary);
        uint32_t bad_magic = 0xDEADBEEF;
        f.write(reinterpret_cast<const char*>(&bad_magic), 4);
        f.close();

        RsGoldReader bad_reader;
        CHALLENGER_ASSERT(!bad_reader.Open(corrupt_magic), "RsGoldReader accepted invalid magic!");
    }

    // 3.5 Adversarial Corruption 2: Truncated Header
    {
        std::string trunc_hdr = "corrupt_trunc_hdr.rsgold";
        std::ofstream dst(trunc_hdr, std::ios::binary);
        char partial[30] = {0};
        dst.write(partial, sizeof(partial));
        dst.close();

        RsGoldReader bad_reader;
        CHALLENGER_ASSERT(!bad_reader.Open(trunc_hdr), "RsGoldReader accepted truncated header!");
    }

    // 3.6 Adversarial Corruption 3: Truncated Stream
    {
        std::string trunc_stream = "corrupt_trunc_stream.rsgold";
        std::ifstream src(test_file, std::ios::binary);
        std::vector<char> buffer(64 + 10 * ENVS * 160); // only 10 ticks instead of 50
        src.read(buffer.data(), buffer.size());
        src.close();

        std::ofstream dst(trunc_stream, std::ios::binary);
        dst.write(buffer.data(), buffer.size());
        dst.close();

        RsGoldReader trunc_reader;
        CHALLENGER_ASSERT(trunc_reader.Open(trunc_stream), "Failed to open truncated stream file");
        std::vector<CarControls> read_controls(ENVS);
        std::vector<BallStatePOD> read_balls(ENVS);
        std::vector<CarStatePOD> read_cars(ENVS);

        bool failed_as_expected = false;
        for (uint32_t t = 0; t < TICKS; t++) {
            if (!trunc_reader.ReadTick(read_controls.data(), read_balls.data(), read_cars.data())) {
                failed_as_expected = true;
                CHALLENGER_ASSERT(t == 10, "Truncated read failed at unexpected tick: " + std::to_string(t));
                break;
            }
        }
        CHALLENGER_ASSERT(failed_as_expected, "Truncated stream was not detected!");
    }

    CHALLENGER_TEST_PASS("Test 3: .rsgold File Format Integrity, Binary Serialization & Corruption Resistance");
    return true;
}

// ---------------------------------------------------------------------------
// Test 4: Determinism & Seed Sensitivity Across CPU Simulation Replays
// ---------------------------------------------------------------------------
bool TestSimulationDeterminismAndSeedEntropy() {
    CHALLENGER_TEST_START("Test 4: Determinism & Seed Sensitivity Across CPU Simulation Replays");

    const uint32_t TICKS = 300;
    const uint32_t ENVS = 2;

    // Run 1 with Seed 1234
    std::vector<BallStatePOD> run1_balls(TICKS * ENVS);
    std::vector<CarStatePOD> run1_cars(TICKS * ENVS);
    {
        std::vector<CPURefSim> sim_envs;
        for (uint32_t e = 0; e < ENVS; e++) sim_envs.emplace_back(1, true, TICK_RATE, static_cast<int>(e));
        DeterministicInputGenerator gen(1234);

        for (uint32_t t = 0; t < TICKS; t++) {
            for (uint32_t e = 0; e < ENVS; e++) {
                CarControls ctrl = gen.Generate();
                sim_envs[e].Step(&ctrl, 1);
                sim_envs[e].GetBallState(run1_balls[t * ENVS + e]);
                sim_envs[e].GetCarState(0, run1_cars[t * ENVS + e]);
            }
        }
    }

    // Run 2 with same Seed 1234 (must be 100% bit-exact across all ticks)
    {
        std::vector<CPURefSim> sim_envs;
        for (uint32_t e = 0; e < ENVS; e++) sim_envs.emplace_back(1, true, TICK_RATE, static_cast<int>(e));
        DeterministicInputGenerator gen(1234);

        for (uint32_t t = 0; t < TICKS; t++) {
            for (uint32_t e = 0; e < ENVS; e++) {
                CarControls ctrl = gen.Generate();
                sim_envs[e].Step(&ctrl, 1);
                BallStatePOD b;
                CarStatePOD c;
                sim_envs[e].GetBallState(b);
                sim_envs[e].GetCarState(0, c);

                CHALLENGER_ASSERT(std::memcmp(&b, &run1_balls[t * ENVS + e], sizeof(BallStatePOD)) == 0,
                    "Bit-level non-determinism detected in ball at tick " + std::to_string(t));
                CHALLENGER_ASSERT(std::memcmp(&c, &run1_cars[t * ENVS + e], sizeof(CarStatePOD)) == 0,
                    "Bit-level non-determinism detected in car at tick " + std::to_string(t));
            }
        }
    }
    std::cout << "  [+] 100% Bit-exact determinism confirmed across independent runs with identical seed.\n";

    // Run 3 with Seed 5678 (must diverge rapidly from Run 1)
    {
        std::vector<CPURefSim> sim_envs;
        for (uint32_t e = 0; e < ENVS; e++) sim_envs.emplace_back(1, true, TICK_RATE, static_cast<int>(e));
        DeterministicInputGenerator gen(5678);

        bool diverged = false;
        for (uint32_t t = 0; t < TICKS; t++) {
            for (uint32_t e = 0; e < ENVS; e++) {
                CarControls ctrl = gen.Generate();
                sim_envs[e].Step(&ctrl, 1);
                CarStatePOD c;
                sim_envs[e].GetCarState(0, c);
                if (c.pos.chebyshev_dist(run1_cars[t * ENVS + e].pos) > 1.0f) {
                    diverged = true;
                    break;
                }
            }
            if (diverged) break;
        }
        CHALLENGER_ASSERT(diverged, "Seed 5678 failed to produce diverging simulation trajectory from Seed 1234!");
    }
    std::cout << "  [+] Seed sensitivity & trajectory divergence confirmed.\n";

    CHALLENGER_TEST_PASS("Test 4: Determinism & Seed Sensitivity Across CPU Simulation Replays");
    return true;
}

// ---------------------------------------------------------------------------
// Test 5: SimContext SoA Memory Layout, Boundary Alignments & Scale Stress
// ---------------------------------------------------------------------------
bool TestSimContextMemoryInvariants() {
    CHALLENGER_TEST_START("Test 5: SimContext SoA Memory Layout, Boundary Alignments & Scale Stress");

    const std::vector<uint32_t> test_env_counts = {1, 2, 7, 16, 128, 1024, 8192, 16384, 32768, 65536};

    for (uint32_t envs : test_env_counts) {
        SimContext ctx(envs, 1);
        CHALLENGER_ASSERT(ctx.GetNumEnvs() == envs, "Mismatched env count in SimContext");
        CHALLENGER_ASSERT(ctx.GetCarsPerEnv() == 1, "Mismatched cars per env in SimContext");

        const auto& b = ctx.GetBallState();
        const auto& c = ctx.GetCarState();
        const auto& ctrl = ctx.GetControls();

        auto check_align = [](const void* ptr) -> bool {
            return (reinterpret_cast<uintptr_t>(ptr) % 128) == 0;
        };

        CHALLENGER_ASSERT(check_align(b.pos_x), "b.pos_x not 128-byte aligned at scale " + std::to_string(envs));
        CHALLENGER_ASSERT(check_align(b.pos_y), "b.pos_y not 128-byte aligned");
        CHALLENGER_ASSERT(check_align(b.pos_z), "b.pos_z not 128-byte aligned");
        CHALLENGER_ASSERT(check_align(b.vel_x), "b.vel_x not 128-byte aligned");
        CHALLENGER_ASSERT(check_align(b.q_w), "b.q_w not 128-byte aligned");
        CHALLENGER_ASSERT(check_align(b.ang_vel_x), "b.ang_vel_x not 128-byte aligned");

        CHALLENGER_ASSERT(check_align(c.pos_x), "c.pos_x not 128-byte aligned");
        CHALLENGER_ASSERT(check_align(c.vel_x), "c.vel_x not 128-byte aligned");
        CHALLENGER_ASSERT(check_align(c.q_w), "c.q_w not 128-byte aligned");
        CHALLENGER_ASSERT(check_align(c.boost), "c.boost not 128-byte aligned");
        CHALLENGER_ASSERT(check_align(c.is_on_ground), "c.is_on_ground not 128-byte aligned");
        CHALLENGER_ASSERT(check_align(c.suspension_length_0), "c.suspension_length_0 not 128-byte aligned");

        CHALLENGER_ASSERT(check_align(ctrl.throttle), "ctrl.throttle not 128-byte aligned");
        CHALLENGER_ASSERT(check_align(ctrl.boost), "ctrl.boost not 128-byte aligned");

        // Verify bounds: pos_y must be at or after pos_x + aligned(envs * sizeof(float))
        size_t slice_size = (envs * sizeof(float) + 127) & ~size_t(127);
        CHALLENGER_ASSERT(reinterpret_cast<uintptr_t>(b.pos_y) >= reinterpret_cast<uintptr_t>(b.pos_x) + slice_size,
            "b.pos_y overlaps with b.pos_x!");
    }
    std::cout << "  [+] Verified 128-byte cache-line alignment and slice non-overlap for scales 1 up to 65536 environments.\n";

    CHALLENGER_TEST_PASS("Test 5: SimContext SoA Memory Layout, Boundary Alignments & Scale Stress");
    return true;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "              CHALLENGER 1: EMPIRICAL ADVERSARIAL TEST SUITE          \n";
    std::cout << "======================================================================\n\n";

    bool all_ok = true;
    all_ok &= TestComparatorSensitivity();
    all_ok &= TestQuaternionAntipodalInvariance();
    all_ok &= TestRsGoldFormatIntegrity();
    all_ok &= TestSimulationDeterminismAndSeedEntropy();
    all_ok &= TestSimContextMemoryInvariants();

    std::cout << "======================================================================\n";
    if (all_ok) {
        std::cout << "  [SUCCESS] ALL ADVERSARIAL CHALLENGER TESTS PASSED WITHOUT EXCEPTION!\n";
    } else {
        std::cout << "  [FAILURE] ONE OR MORE ADVERSARIAL CHALLENGER TESTS FAILED!\n";
    }
    std::cout << "======================================================================\n";

    return all_ok ? 0 : 1;
}
