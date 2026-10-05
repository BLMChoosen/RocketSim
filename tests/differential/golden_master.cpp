#include "golden_master.h"
#include <sstream>
#include <iomanip>
#include <cmath>

namespace rocketsim_cuda {

DeterministicInputGenerator::DeterministicInputGenerator(uint64_t seed, uint64_t seq) {
    Seed(seed, seq);
}

void DeterministicInputGenerator::Seed(uint64_t seed, uint64_t seq) {
    m_state = 0;
    m_inc = (seq << 1u) | 1u;
    NextU32();
    m_state += seed;
    NextU32();
}

uint32_t DeterministicInputGenerator::NextU32() {
    uint64_t oldstate = m_state;
    m_state = oldstate * 6364136223846793005ULL + (m_inc | 1ULL);
    uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-static_cast<int32_t>(rot)) & 31));
}

float DeterministicInputGenerator::NextFloatSigned() {
    return (static_cast<float>(NextU32()) / 2147483648.0f) - 1.0f;
}

CarControls DeterministicInputGenerator::Generate() {
    CarControls c;
    c.throttle  = NextFloatSigned();
    c.steer     = NextFloatSigned();
    c.pitch     = NextFloatSigned();
    c.yaw       = NextFloatSigned();
    c.roll      = NextFloatSigned();
    c.boost     = (NextU32() % 100 < 30) ? 1 : 0;
    c.jump      = (NextU32() % 100 < 20) ? 1 : 0;
    c.handbrake = (NextU32() % 100 < 10) ? 1 : 0;
    c.clamp_fix();
    return c;
}

DifferentialComparator::DifferentialComparator(const DifferentialTolerance& tol)
    : m_tol(tol) {}

bool DifferentialComparator::CompareBall(
    uint32_t tick, uint32_t env_idx,
    const BallStatePOD& cpu, const BallStatePOD& test,
    DifferentialFailure& failure) {

    // 1. Position Chebyshev ||Delta_pos||_inf
    float d_pos = cpu.pos.chebyshev_dist(test.pos);
    float max_c = std::max({std::abs(cpu.pos.x), std::abs(cpu.pos.y), std::abs(cpu.pos.z)});
    float ulp_floor = (max_c >= 4096.0f) ? 0.00048828125f : ((max_c >= 2048.0f) ? 0.000244140625f : 0.0f);
    float tol_pos = std::max(m_tol.pos_uu, ulp_floor);
    if (d_pos > tol_pos) {
        failure.tick = tick;
        failure.env_idx = env_idx;
        failure.car_idx = -1;
        failure.attribute = "Ball Position (UU)";
        failure.max_delta = d_pos;
        failure.threshold = tol_pos;
        std::ostringstream ss_cpu, ss_test;
        ss_cpu << cpu.pos;
        ss_test << test.pos;
        failure.cpu_val = ss_cpu.str();
        failure.test_val = ss_test.str();
        return false;
    }

    // 2. Velocity Chebyshev ||Delta_vel||_inf
    float d_vel = cpu.vel.chebyshev_dist(test.vel);
    if (d_vel > m_tol.vel_uus) {
        failure.tick = tick;
        failure.env_idx = env_idx;
        failure.car_idx = -1;
        failure.attribute = "Ball Velocity (UU/s)";
        failure.max_delta = d_vel;
        failure.threshold = m_tol.vel_uus;
        std::ostringstream ss_cpu, ss_test;
        ss_cpu << cpu.vel;
        ss_test << test.vel;
        failure.cpu_val = ss_cpu.str();
        failure.test_val = ss_test.str();
        return false;
    }

    // 3. Quaternion antipodal Chebyshev ||Delta_q||_inf
    float d_quat = cpu.quat.chebyshev_dist(test.quat);
    if (d_quat > m_tol.quat) {
        failure.tick = tick;
        failure.env_idx = env_idx;
        failure.car_idx = -1;
        failure.attribute = "Ball Quaternion";
        failure.max_delta = d_quat;
        failure.threshold = m_tol.quat;
        std::ostringstream ss_cpu, ss_test;
        ss_cpu << cpu.quat;
        ss_test << test.quat;
        failure.cpu_val = ss_cpu.str();
        failure.test_val = ss_test.str();
        return false;
    }

    // 4. Angular velocity Chebyshev ||Delta_angvel||_inf
    float d_ang_vel = cpu.ang_vel.chebyshev_dist(test.ang_vel);
    if (d_ang_vel > m_tol.ang_vel_rads) {
        failure.tick = tick;
        failure.env_idx = env_idx;
        failure.car_idx = -1;
        failure.attribute = "Ball Angular Velocity (rad/s)";
        failure.max_delta = d_ang_vel;
        failure.threshold = m_tol.ang_vel_rads;
        std::ostringstream ss_cpu, ss_test;
        ss_cpu << cpu.ang_vel;
        ss_test << test.ang_vel;
        failure.cpu_val = ss_cpu.str();
        failure.test_val = ss_test.str();
        return false;
    }

    return true;
}

bool DifferentialComparator::CompareCar(
    uint32_t tick, uint32_t env_idx, uint32_t car_idx,
    const CarStatePOD& cpu, const CarStatePOD& test,
    DifferentialFailure& failure) {

    // 1. Position Chebyshev ||Delta_pos||_inf
    float d_pos = cpu.pos.chebyshev_dist(test.pos);
    float max_c = std::max({std::abs(cpu.pos.x), std::abs(cpu.pos.y), std::abs(cpu.pos.z)});
    float ulp_floor = (max_c >= 4096.0f) ? 0.00048828125f : ((max_c >= 2048.0f) ? 0.000244140625f : 0.0f);
    float tol_pos = std::max(m_tol.pos_uu, ulp_floor);
    if (d_pos > tol_pos) {
        failure.tick = tick;
        failure.env_idx = env_idx;
        failure.car_idx = static_cast<int32_t>(car_idx);
        failure.attribute = "Car Position (UU)";
        failure.max_delta = d_pos;
        failure.threshold = tol_pos;
        std::ostringstream ss_cpu, ss_test;
        ss_cpu << cpu.pos;
        ss_test << test.pos;
        failure.cpu_val = ss_cpu.str();
        failure.test_val = ss_test.str();
        return false;
    }

    // 2. Velocity Chebyshev ||Delta_vel||_inf
    float d_vel = cpu.vel.chebyshev_dist(test.vel);
    if (d_vel > m_tol.vel_uus) {
        failure.tick = tick;
        failure.env_idx = env_idx;
        failure.car_idx = static_cast<int32_t>(car_idx);
        failure.attribute = "Car Velocity (UU/s)";
        failure.max_delta = d_vel;
        failure.threshold = m_tol.vel_uus;
        std::ostringstream ss_cpu, ss_test;
        ss_cpu << cpu.vel;
        ss_test << test.vel;
        failure.cpu_val = ss_cpu.str();
        failure.test_val = ss_test.str();
        return false;
    }

    // 3. Quaternion antipodal Chebyshev ||Delta_q||_inf
    float d_quat = cpu.quat.chebyshev_dist(test.quat);
    if (d_quat > m_tol.quat) {
        failure.tick = tick;
        failure.env_idx = env_idx;
        failure.car_idx = static_cast<int32_t>(car_idx);
        failure.attribute = "Car Quaternion";
        failure.max_delta = d_quat;
        failure.threshold = m_tol.quat;
        std::ostringstream ss_cpu, ss_test;
        ss_cpu << cpu.quat;
        ss_test << test.quat;
        failure.cpu_val = ss_cpu.str();
        failure.test_val = ss_test.str();
        return false;
    }

    // 4. Angular velocity Chebyshev ||Delta_angvel||_inf
    float d_ang_vel = cpu.ang_vel.chebyshev_dist(test.ang_vel);
    if (d_ang_vel > m_tol.ang_vel_rads) {
        failure.tick = tick;
        failure.env_idx = env_idx;
        failure.car_idx = static_cast<int32_t>(car_idx);
        failure.attribute = "Car Angular Velocity (rad/s)";
        failure.max_delta = d_ang_vel;
        failure.threshold = m_tol.ang_vel_rads;
        std::ostringstream ss_cpu, ss_test;
        ss_cpu << cpu.ang_vel;
        ss_test << test.ang_vel;
        failure.cpu_val = ss_cpu.str();
        failure.test_val = ss_test.str();
        return false;
    }

    // 5. Suspension compression (per wheel)
    for (int w = 0; w < 4; w++) {
        float d_susp = std::fabs(cpu.suspension_lengths[w] - test.suspension_lengths[w]);
        if (d_susp > m_tol.suspension_uu) {
            failure.tick = tick;
            failure.env_idx = env_idx;
            failure.car_idx = static_cast<int32_t>(car_idx);
            failure.attribute = "Car Suspension Compression Wheel " + std::to_string(w) + " (UU)";
            failure.max_delta = d_susp;
            failure.threshold = m_tol.suspension_uu;
            failure.cpu_val = std::to_string(cpu.suspension_lengths[w]);
            failure.test_val = std::to_string(test.suspension_lengths[w]);
            return false;
        }
    }

    // 6. Boost amount (bit-exact or delta <= tol)
    float d_boost = std::fabs(cpu.boost - test.boost);
    if (d_boost > m_tol.boost) {
        failure.tick = tick;
        failure.env_idx = env_idx;
        failure.car_idx = static_cast<int32_t>(car_idx);
        failure.attribute = "Car Boost";
        failure.max_delta = d_boost;
        failure.threshold = m_tol.boost;
        failure.cpu_val = std::to_string(cpu.boost);
        failure.test_val = std::to_string(test.boost);
        return false;
    }

    return true;
}

RsGoldWriter::~RsGoldWriter() {
    Close();
}

bool RsGoldWriter::Open(const std::string& path, uint32_t num_envs, uint32_t cars_per_env, uint32_t seed, float tick_rate) {
    m_file.open(path, std::ios::binary | std::ios::trunc);
    if (!m_file.is_open()) return false;

    m_header.magic = 0x5253474D;
    m_header.version = 1;
    m_header.num_envs = num_envs;
    m_header.cars_per_env = cars_per_env;
    m_header.total_ticks = 0;
    m_header.tick_rate = tick_rate;
    m_header.seed = seed;
    m_header.scenario_id = 0;

    m_file.write(reinterpret_cast<const char*>(&m_header), sizeof(RsGoldHeader));
    m_open = true;
    return true;
}

void RsGoldWriter::WriteTick(const CarControls* controls, const BallStatePOD* balls, const CarStatePOD* cars) {
    if (!m_open) return;

    uint32_t total_cars = m_header.num_envs * m_header.cars_per_env;

    // 1. Controls (total_cars * 24 bytes)
    for (uint32_t i = 0; i < total_cars; i++) {
        RsGoldCarControls c;
        if (controls) {
            c.throttle  = controls[i].throttle;
            c.steer     = controls[i].steer;
            c.pitch     = controls[i].pitch;
            c.yaw       = controls[i].yaw;
            c.roll      = controls[i].roll;
            c.boost     = controls[i].boost;
            c.jump      = controls[i].jump;
            c.handbrake = controls[i].handbrake;
            c.padding   = 0;
        }
        m_file.write(reinterpret_cast<const char*>(&c), sizeof(RsGoldCarControls));
    }

    // 2. Balls (num_envs * 52 bytes)
    for (uint32_t e = 0; e < m_header.num_envs; e++) {
        RsGoldBallRecord b;
        if (balls) {
            b.pos[0] = balls[e].pos.x; b.pos[1] = balls[e].pos.y; b.pos[2] = balls[e].pos.z;
            b.vel[0] = balls[e].vel.x; b.vel[1] = balls[e].vel.y; b.vel[2] = balls[e].vel.z;
            b.quat[0] = balls[e].quat.w; b.quat[1] = balls[e].quat.x; b.quat[2] = balls[e].quat.y; b.quat[3] = balls[e].quat.z;
            b.ang_vel[0] = balls[e].ang_vel.x; b.ang_vel[1] = balls[e].ang_vel.y; b.ang_vel[2] = balls[e].ang_vel.z;
        }
        m_file.write(reinterpret_cast<const char*>(&b), sizeof(RsGoldBallRecord));
    }

    // 3. Cars (total_cars * 84 bytes)
    for (uint32_t i = 0; i < total_cars; i++) {
        RsGoldCarRecord cr;
        if (cars) {
            cr.pos[0] = cars[i].pos.x; cr.pos[1] = cars[i].pos.y; cr.pos[2] = cars[i].pos.z;
            cr.vel[0] = cars[i].vel.x; cr.vel[1] = cars[i].vel.y; cr.vel[2] = cars[i].vel.z;
            cr.quat[0] = cars[i].quat.w; cr.quat[1] = cars[i].quat.x; cr.quat[2] = cars[i].quat.y; cr.quat[3] = cars[i].quat.z;
            cr.ang_vel[0] = cars[i].ang_vel.x; cr.ang_vel[1] = cars[i].ang_vel.y; cr.ang_vel[2] = cars[i].ang_vel.z;
            cr.boost = cars[i].boost;
            cr.is_on_ground = cars[i].is_on_ground;
            cr.has_jumped = cars[i].has_jumped;
            cr.has_double_jumped = cars[i].has_double_jumped;
            cr.has_flipped = cars[i].has_flipped;
            cr.is_demoed = cars[i].is_demoed;
            cr.team = cars[i].team;
            for (int w = 0; w < 4; w++) {
                cr.wheels_with_contact[w] = cars[i].wheels_with_contact[w];
                cr.suspension_lengths[w] = cars[i].suspension_lengths[w];
            }
        }
        m_file.write(reinterpret_cast<const char*>(&cr), sizeof(RsGoldCarRecord));
    }

    m_header.total_ticks++;
}

void RsGoldWriter::Close() {
    if (m_open) {
        // Rewrite header with updated total_ticks count
        m_file.seekp(0, std::ios::beg);
        m_file.write(reinterpret_cast<const char*>(&m_header), sizeof(RsGoldHeader));
        m_file.close();
        m_open = false;
    }
}

RsGoldReader::~RsGoldReader() {
    Close();
}

bool RsGoldReader::Open(const std::string& path) {
    m_file.open(path, std::ios::binary);
    if (!m_file.is_open()) return false;

    m_file.read(reinterpret_cast<char*>(&m_header), sizeof(RsGoldHeader));
    if (!m_file.good()) {
        Close();
        return false;
    }

    if (m_header.magic != 0x5253474D) {
        Close();
        return false;
    }

    m_open = true;
    return true;
}

bool RsGoldReader::ReadTick(CarControls* controls, BallStatePOD* balls, CarStatePOD* cars) {
    if (!m_open || !m_file.good()) return false;

    uint32_t total_cars = m_header.num_envs * m_header.cars_per_env;

    // 1. Controls
    for (uint32_t i = 0; i < total_cars; i++) {
        RsGoldCarControls c;
        m_file.read(reinterpret_cast<char*>(&c), sizeof(RsGoldCarControls));
        if (!m_file.good()) return false;
        if (controls) {
            controls[i].throttle  = c.throttle;
            controls[i].steer     = c.steer;
            controls[i].pitch     = c.pitch;
            controls[i].yaw       = c.yaw;
            controls[i].roll      = c.roll;
            controls[i].boost     = c.boost;
            controls[i].jump      = c.jump;
            controls[i].handbrake = c.handbrake;
            controls[i].padding   = 0;
        }
    }

    // 2. Balls
    for (uint32_t e = 0; e < m_header.num_envs; e++) {
        RsGoldBallRecord b;
        m_file.read(reinterpret_cast<char*>(&b), sizeof(RsGoldBallRecord));
        if (!m_file.good()) return false;
        if (balls) {
            balls[e].pos = Vec3(b.pos[0], b.pos[1], b.pos[2]);
            balls[e].vel = Vec3(b.vel[0], b.vel[1], b.vel[2]);
            balls[e].quat = Quat(b.quat[0], b.quat[1], b.quat[2], b.quat[3]);
            balls[e].ang_vel = Vec3(b.ang_vel[0], b.ang_vel[1], b.ang_vel[2]);
        }
    }

    // 3. Cars
    for (uint32_t i = 0; i < total_cars; i++) {
        RsGoldCarRecord cr;
        m_file.read(reinterpret_cast<char*>(&cr), sizeof(RsGoldCarRecord));
        if (!m_file.good()) return false;
        if (cars) {
            cars[i].pos = Vec3(cr.pos[0], cr.pos[1], cr.pos[2]);
            cars[i].vel = Vec3(cr.vel[0], cr.vel[1], cr.vel[2]);
            cars[i].quat = Quat(cr.quat[0], cr.quat[1], cr.quat[2], cr.quat[3]);
            cars[i].ang_vel = Vec3(cr.ang_vel[0], cr.ang_vel[1], cr.ang_vel[2]);
            cars[i].boost = cr.boost;
            cars[i].is_on_ground = cr.is_on_ground;
            cars[i].has_jumped = cr.has_jumped;
            cars[i].has_double_jumped = cr.has_double_jumped;
            cars[i].has_flipped = cr.has_flipped;
            cars[i].is_demoed = cr.is_demoed;
            cars[i].team = cr.team;
            for (int w = 0; w < 4; w++) {
                cars[i].wheels_with_contact[w] = cr.wheels_with_contact[w];
                cars[i].suspension_lengths[w] = cr.suspension_lengths[w];
            }
        }
    }

    return true;
}

void RsGoldReader::Close() {
    if (m_open) {
        m_file.close();
        m_open = false;
    }
}

} // namespace rocketsim_cuda
