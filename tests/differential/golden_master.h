#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/types/car_controls.cuh"
#include "rocketsim_cuda/types/ball_state.cuh"
#include "rocketsim_cuda/types/car_state.cuh"

namespace rocketsim_cuda {

#pragma pack(push, 1)

// 64-byte Header for .rsgold binary replay files (GEMINI.md Section 3.3)
struct RsGoldHeader {
    uint32_t magic = 0x5253474D; // "RSGM"
    uint32_t version = 1;
    uint32_t num_envs = 1;
    uint32_t cars_per_env = 1;
    uint32_t total_ticks = 0;
    float tick_rate = TICK_RATE;
    uint32_t seed = 42;
    uint32_t scenario_id = 0;
    uint8_t reserved[32] = {0};
};

// 24-byte record per car control input
struct RsGoldCarControls {
    float throttle = 0.0f;
    float steer = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    float roll = 0.0f;
    uint8_t boost = 0;
    uint8_t jump = 0;
    uint8_t handbrake = 0;
    uint8_t padding = 0;
};

// 52-byte record per ball state
struct RsGoldBallRecord {
    float pos[3] = {0.0f, 0.0f, 0.0f};
    float vel[3] = {0.0f, 0.0f, 0.0f};
    float quat[4] = {1.0f, 0.0f, 0.0f, 0.0f}; // w, x, y, z
    float ang_vel[3] = {0.0f, 0.0f, 0.0f};
};

// 84-byte record per car state
struct RsGoldCarRecord {
    float pos[3] = {0.0f, 0.0f, 0.0f};
    float vel[3] = {0.0f, 0.0f, 0.0f};
    float quat[4] = {1.0f, 0.0f, 0.0f, 0.0f}; // w, x, y, z
    float ang_vel[3] = {0.0f, 0.0f, 0.0f};
    float boost = 0.0f;
    uint8_t is_on_ground = 0;
    uint8_t has_jumped = 0;
    uint8_t has_double_jumped = 0;
    uint8_t has_flipped = 0;
    uint8_t is_demoed = 0;
    uint8_t team = 0;
    uint8_t wheels_with_contact[4] = {0, 0, 0, 0};
    uint8_t hitbox_type = 0;
    uint8_t padding = 0;
    float suspension_lengths[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

#pragma pack(pop)

// Deterministic PCG32 pseudo-random input generator
class DeterministicInputGenerator {
public:
    explicit DeterministicInputGenerator(uint64_t seed = 42, uint64_t seq = 1);
    void Seed(uint64_t seed, uint64_t seq = 1);
    CarControls Generate();

private:
    uint64_t m_state = 0;
    uint64_t m_inc = 1;

    uint32_t NextU32();
    float NextFloatSigned();
};

// Differential tolerances matching GEMINI.md Section 3.1
struct DifferentialTolerance {
    float pos_uu = TOL_POS;                 // <= 1e-4 UU
    float vel_uus = TOL_VEL;                // <= 1e-3 UU/s
    float quat = TOL_QUAT;                  // <= 1e-5
    float ang_vel_rads = TOL_ANG_VEL;       // <= 1e-4 rad/s
    float suspension_uu = TOL_SUSPENSION;   // <= 1e-4 UU
    float boost = TOL_BOOST;                // 0.0f (bit-exact)
};

struct DifferentialFailure {
    uint32_t tick = 0;
    uint32_t env_idx = 0;
    int32_t car_idx = -1; // -1 for ball
    std::string attribute;
    float max_delta = 0.0f;
    float threshold = 0.0f;
    std::string cpu_val;
    std::string test_val;
};

class DifferentialComparator {
public:
    explicit DifferentialComparator(const DifferentialTolerance& tol = DifferentialTolerance{});

    bool CompareBall(uint32_t tick, uint32_t env_idx, const BallStatePOD& cpu, const BallStatePOD& test, DifferentialFailure& failure);
    bool CompareCar(uint32_t tick, uint32_t env_idx, uint32_t car_idx, const CarStatePOD& cpu, const CarStatePOD& test, DifferentialFailure& failure);

    const DifferentialTolerance& GetTolerance() const { return m_tol; }
    void SetTolerance(const DifferentialTolerance& tol) { m_tol = tol; }

private:
    DifferentialTolerance m_tol;
};

// Replay file serializer
class RsGoldWriter {
public:
    RsGoldWriter() = default;
    ~RsGoldWriter();

    bool Open(const std::string& path, uint32_t num_envs, uint32_t cars_per_env, uint32_t seed, float tick_rate = TICK_RATE);
    void WriteTick(const CarControls* controls, const BallStatePOD* balls, const CarStatePOD* cars);
    void Close();

private:
    std::ofstream m_file;
    RsGoldHeader m_header;
    bool m_open = false;
};

// Replay file deserializer
class RsGoldReader {
public:
    RsGoldReader() = default;
    ~RsGoldReader();

    bool Open(const std::string& path);
    const RsGoldHeader& GetHeader() const { return m_header; }
    bool ReadTick(CarControls* controls, BallStatePOD* balls, CarStatePOD* cars);
    void Close();

private:
    std::ifstream m_file;
    RsGoldHeader m_header;
    bool m_open = false;
};

} // namespace rocketsim_cuda
