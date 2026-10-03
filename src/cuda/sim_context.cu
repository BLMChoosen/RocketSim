#include "rocketsim_cuda/sim_context.cuh"
#include <stdexcept>
#include <iostream>

namespace rocketsim_cuda {

namespace {

inline size_t align_128(size_t size) {
    return (size + 127) & ~size_t(127);
}

__device__ inline void init_single_ball(BallStateSoA& ball_state, uint32_t idx) {
    ball_state.pos_x[idx] = 0.0f;
    ball_state.pos_y[idx] = 0.0f;
    ball_state.pos_z[idx] = BALL_REST_Z;

    ball_state.vel_x[idx] = 0.0f;
    ball_state.vel_y[idx] = 0.0f;
    ball_state.vel_z[idx] = 0.0f;

    ball_state.q_w[idx] = 1.0f;
    ball_state.q_x[idx] = 0.0f;
    ball_state.q_y[idx] = 0.0f;
    ball_state.q_z[idx] = 0.0f;

    ball_state.ang_vel_x[idx] = 0.0f;
    ball_state.ang_vel_y[idx] = 0.0f;
    ball_state.ang_vel_z[idx] = 0.0f;
}

__device__ inline void init_single_car(CarStateSoA& car_state, uint32_t idx) {
    car_state.pos_x[idx] = 0.0f;
    car_state.pos_y[idx] = 0.0f;
    car_state.pos_z[idx] = 17.0f;

    car_state.vel_x[idx] = 0.0f;
    car_state.vel_y[idx] = 0.0f;
    car_state.vel_z[idx] = 0.0f;

    car_state.q_w[idx] = 1.0f;
    car_state.q_x[idx] = 0.0f;
    car_state.q_y[idx] = 0.0f;
    car_state.q_z[idx] = 0.0f;

    car_state.ang_vel_x[idx] = 0.0f;
    car_state.ang_vel_y[idx] = 0.0f;
    car_state.ang_vel_z[idx] = 0.0f;

    car_state.boost[idx] = BOOST_SPAWN;
    car_state.time_since_boosted[idx] = 0.0f;
    car_state.boosting_time[idx] = 0.0f;
    car_state.is_boosting[idx] = 0;

    car_state.is_on_ground[idx] = 1;
    car_state.wheel_contact_0[idx] = 1;
    car_state.wheel_contact_1[idx] = 1;
    car_state.wheel_contact_2[idx] = 1;
    car_state.wheel_contact_3[idx] = 1;

    car_state.suspension_length_0[idx] = 0.0f;
    car_state.suspension_length_1[idx] = 0.0f;
    car_state.suspension_length_2[idx] = 0.0f;
    car_state.suspension_length_3[idx] = 0.0f;

    car_state.has_jumped[idx] = 0;
    car_state.is_jumping[idx] = 0;
    car_state.jump_time[idx] = 0.0f;
    car_state.has_double_jumped[idx] = 0;
    car_state.air_time[idx] = 0.0f;
    car_state.air_time_since_jump[idx] = 0.0f;

    car_state.has_flipped[idx] = 0;
    car_state.is_flipping[idx] = 0;
    car_state.flip_time[idx] = 0.0f;
    car_state.flip_rel_torque_x[idx] = 0.0f;
    car_state.flip_rel_torque_y[idx] = 0.0f;
    car_state.flip_rel_torque_z[idx] = 0.0f;

    car_state.is_auto_flipping[idx] = 0;
    car_state.auto_flip_timer[idx] = 0.0f;
    car_state.auto_flip_torque_scale[idx] = 0.0f;
    car_state.handbrake_val[idx] = 0.0f;

    car_state.is_supersonic[idx] = 0;
    car_state.supersonic_time[idx] = 0.0f;
    car_state.is_demoed[idx] = 0;
    car_state.demo_respawn_timer[idx] = 0.0f;

    car_state.world_contact_has_contact[idx] = 0;
    car_state.world_contact_normal_x[idx] = 0.0f;
    car_state.world_contact_normal_y[idx] = 0.0f;
    car_state.world_contact_normal_z[idx] = 0.0f;
    car_state.car_contact_other_car_id[idx] = -1;
    car_state.car_contact_cooldown_timer[idx] = 0.0f;

    car_state.ball_hit_is_valid[idx] = 0;
    car_state.ball_hit_rel_pos_x[idx] = 0.0f;
    car_state.ball_hit_rel_pos_y[idx] = 0.0f;
    car_state.ball_hit_rel_pos_z[idx] = 0.0f;
    car_state.ball_hit_extra_hit_force_x[idx] = 0.0f;
    car_state.ball_hit_extra_hit_force_y[idx] = 0.0f;
    car_state.ball_hit_extra_hit_force_z[idx] = 0.0f;
    car_state.ball_hit_tick_count[idx] = 0;

    car_state.last_controls_throttle[idx] = 0.0f;
    car_state.last_controls_steer[idx] = 0.0f;
    car_state.last_controls_pitch[idx] = 0.0f;
    car_state.last_controls_yaw[idx] = 0.0f;
    car_state.last_controls_roll[idx] = 0.0f;
    car_state.last_controls_boost[idx] = 0;
    car_state.last_controls_jump[idx] = 0;
    car_state.last_controls_handbrake[idx] = 0;

    car_state.wheel_engine_force[idx] = 0.0f;
    car_state.wheel_brake[idx] = 0.0f;
    car_state.wheel_steer_angle[idx] = 0.0f;
    car_state.wheel_lat_friction_0[idx] = 0.0f;
    car_state.wheel_lat_friction_1[idx] = 0.0f;
    car_state.wheel_lat_friction_2[idx] = 0.0f;
    car_state.wheel_lat_friction_3[idx] = 0.0f;
    car_state.wheel_long_friction_0[idx] = 0.0f;
    car_state.wheel_long_friction_1[idx] = 0.0f;
    car_state.wheel_long_friction_2[idx] = 0.0f;
    car_state.wheel_long_friction_3[idx] = 0.0f;

    car_state.pos_bt_x[idx] = car_state.pos_x[idx] * 0.02f;
    car_state.pos_bt_y[idx] = car_state.pos_y[idx] * 0.02f;
    car_state.pos_bt_z[idx] = car_state.pos_z[idx] * 0.02f;
    car_state.vel_bt_x[idx] = car_state.vel_x[idx] * 0.02f;
    car_state.vel_bt_y[idx] = car_state.vel_y[idx] * 0.02f;
    car_state.vel_bt_z[idx] = car_state.vel_z[idx] * 0.02f;
}

__device__ inline void init_single_arena(ArenaStateSoA& arena_state, uint32_t env_idx, uint32_t cars_per_env = 1) {
    if (arena_state.is_goal) {
        arena_state.is_goal[env_idx] = 0;
        arena_state.scoring_team[env_idx] = 0;
        arena_state.is_out_of_bounds[env_idx] = 0;
        arena_state.tick_count[env_idx] = 0;
        if (arena_state.terminated) arena_state.terminated[env_idx] = 0;
        if (arena_state.truncated) arena_state.truncated[env_idx] = 0;
        if (arena_state.rewards) {
            for (uint32_t c = 0; c < cars_per_env; ++c) {
                arena_state.rewards[env_idx * cars_per_env + c] = 0.0f;
            }
        }
        for (uint32_t p = 0; p < MAX_BOOST_PADS; ++p) {
            arena_state.pad_is_active[env_idx * MAX_BOOST_PADS + p] = 1;
            arena_state.pad_cooldown[env_idx * MAX_BOOST_PADS + p] = 0.0f;
        }
    }
}

__global__ void k_init_ball_state(BallStateSoA ball_state, uint32_t count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    init_single_ball(ball_state, idx);
}

__global__ void k_init_car_state(CarStateSoA car_state, uint32_t count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    init_single_car(car_state, idx);
}

__global__ void k_init_arena_state(ArenaStateSoA arena_state, uint32_t count, uint32_t cars_per_env) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    init_single_arena(arena_state, idx, cars_per_env);
}

template <typename TIndex>
__global__ void k_reset_environments_indexed(
    const TIndex* __restrict__ env_indices,
    uint32_t num_resets,
    uint32_t cars_per_env,
    BallStateSoA ball_state,
    CarStateSoA car_state,
    ArenaStateSoA arena_state)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_resets) return;

    int64_t env_idx = env_indices[i];
    if (env_idx < 0) return;

    uint32_t e = static_cast<uint32_t>(env_idx);
    init_single_ball(ball_state, e);
    init_single_arena(arena_state, e, cars_per_env);

    for (uint32_t c = 0; c < cars_per_env; ++c) {
        uint32_t car_idx = e * cars_per_env + c;
        init_single_car(car_state, car_idx);
    }
}

__global__ void k_reset_environments_masked(
    const uint8_t* __restrict__ reset_mask,
    uint32_t num_envs,
    uint32_t cars_per_env,
    BallStateSoA ball_state,
    CarStateSoA car_state,
    ArenaStateSoA arena_state)
{
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_envs) return;
    if (!reset_mask[e]) return;

    init_single_ball(ball_state, e);
    init_single_arena(arena_state, e, cars_per_env);

    for (uint32_t c = 0; c < cars_per_env; ++c) {
        uint32_t car_idx = e * cars_per_env + c;
        init_single_car(car_state, car_idx);
    }
}

__global__ void k_export_ball_pod(BallStateSoA src, BallStatePOD* dst, uint32_t count, uint32_t start_env) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst[i] = src.get_pod(start_env + i);
}

__global__ void k_import_ball_pod(BallStateSoA dst, const BallStatePOD* src, uint32_t count, uint32_t start_env) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst.set_pod(start_env + i, src[i]);
}

__global__ void k_export_car_pod(CarStateSoA src, CarStatePOD* dst, uint32_t count, uint32_t start_car) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst[i] = src.get_pod(start_car + i);
}

__global__ void k_import_car_pod(CarStateSoA dst, const CarStatePOD* src, uint32_t count, uint32_t start_car) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst.set_pod(start_car + i, src[i]);
}

__global__ void k_import_controls(CarControlsSoA dst, const CarControls* src, uint32_t count, uint32_t start_car) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst.set(start_car + i, src[i]);
}

} // namespace

SimContext::SimContext(uint32_t num_envs, uint32_t cars_per_env, cudaStream_t stream)
    : m_num_envs(num_envs), m_cars_per_env(cars_per_env), m_stream(stream) {
    m_total_cars = m_num_envs * m_cars_per_env;
    AllocateArena();
    ResetToDefault();
}

SimContext::~SimContext() {
    FreeArena();
}

SimContext::SimContext(SimContext&& o) noexcept
    : m_num_envs(o.m_num_envs),
      m_cars_per_env(o.m_cars_per_env),
      m_total_cars(o.m_total_cars),
      m_allocated_bytes(o.m_allocated_bytes),
      m_d_pool(o.m_d_pool),
      m_stream(o.m_stream),
      m_ball_state(o.m_ball_state),
      m_car_state(o.m_car_state),
      m_arena_state(o.m_arena_state),
      m_controls(o.m_controls) {
    o.m_d_pool = nullptr;
    o.m_allocated_bytes = 0;
    o.m_num_envs = 0;
    o.m_cars_per_env = 0;
    o.m_total_cars = 0;
    o.m_arena_state = ArenaStateSoA{};
}

SimContext& SimContext::operator=(SimContext&& o) noexcept {
    if (this != &o) {
        FreeArena();
        m_num_envs = o.m_num_envs;
        m_cars_per_env = o.m_cars_per_env;
        m_total_cars = o.m_total_cars;
        m_allocated_bytes = o.m_allocated_bytes;
        m_d_pool = o.m_d_pool;
        m_stream = o.m_stream;
        m_ball_state = o.m_ball_state;
        m_car_state = o.m_car_state;
        m_arena_state = o.m_arena_state;
        m_controls = o.m_controls;

        o.m_d_pool = nullptr;
        o.m_allocated_bytes = 0;
        o.m_num_envs = 0;
        o.m_cars_per_env = 0;
        o.m_total_cars = 0;
        o.m_arena_state = ArenaStateSoA{};
    }
    return *this;
}

size_t SimContext::GetBallPitchFloats() const {
    return align_128(m_num_envs * sizeof(float)) / sizeof(float);
}

size_t SimContext::GetCarPitchFloats() const {
    size_t car_count = (m_total_cars > 0) ? m_total_cars : 1;
    return align_128(car_count * sizeof(float)) / sizeof(float);
}

void SimContext::AllocateArena() {
    if (m_num_envs == 0) return;

    size_t env_count = m_num_envs;
    size_t car_count = (m_total_cars > 0) ? m_total_cars : 1;

    // Helper to calculate aligned bytes
    auto calc_slice = [](size_t count, size_t elem_size) -> size_t {
        return align_128(count * elem_size);
    };

    // Calculate total pool bytes across Ball, Car, Controls, and Staging buffers
    size_t total = 0;

    // --- Ball SoA slices (length = env_count) ---
    // 13 float arrays: pos(3), vel(3), quat(4), ang_vel(3)
    total += calc_slice(env_count, sizeof(float)) * 13;
    // Gamemode extensions: hs(3 floats), ds(1 int32, 2 floats, 1 uint8, 1 uint64)
    total += calc_slice(env_count, sizeof(float)) * 3;
    total += calc_slice(env_count, sizeof(int32_t));
    total += calc_slice(env_count, sizeof(float)) * 2;
    total += calc_slice(env_count, sizeof(uint8_t));
    total += calc_slice(env_count, sizeof(uint64_t));

    // --- Car SoA slices (length = car_count) ---
    // Rigid body: 13 float arrays (pos, vel, quat, ang_vel)
    total += calc_slice(car_count, sizeof(float)) * 13;
    // Boost: 3 floats, 1 uint8
    total += calc_slice(car_count, sizeof(float)) * 3;
    total += calc_slice(car_count, sizeof(uint8_t));
    // Wheels: 5 uint8 (ground + 4 contacts), 4 floats (suspension lengths)
    total += calc_slice(car_count, sizeof(uint8_t)) * 5;
    total += calc_slice(car_count, sizeof(float)) * 4;
    // Jump: 2 uint8, 4 floats
    total += calc_slice(car_count, sizeof(uint8_t)) * 2;
    total += calc_slice(car_count, sizeof(float)) * 4;
    // Flip: 2 uint8, 4 floats
    total += calc_slice(car_count, sizeof(uint8_t)) * 2;
    total += calc_slice(car_count, sizeof(float)) * 4;
    // Auto-flip & handbrake: 1 uint8, 3 floats
    total += calc_slice(car_count, sizeof(uint8_t));
    total += calc_slice(car_count, sizeof(float)) * 3;
    // Supersonic & demo: 2 uint8, 2 floats
    total += calc_slice(car_count, sizeof(uint8_t)) * 2;
    total += calc_slice(car_count, sizeof(float)) * 2;
    // Contacts: 1 uint8, 3 floats, 1 int32, 1 float
    total += calc_slice(car_count, sizeof(uint8_t));
    total += calc_slice(car_count, sizeof(float)) * 3;
    total += calc_slice(car_count, sizeof(int32_t));
    total += calc_slice(car_count, sizeof(float));
    // Ball hit: 1 uint8, 6 floats, 1 uint64
    total += calc_slice(car_count, sizeof(uint8_t));
    total += calc_slice(car_count, sizeof(float)) * 6;
    total += calc_slice(car_count, sizeof(uint64_t));
    // Last controls: 5 floats, 3 uint8
    total += calc_slice(car_count, sizeof(float)) * 5;
    total += calc_slice(car_count, sizeof(uint8_t)) * 3;
    // Persistent wheel & Bullet unit dynamics: 17 floats
    total += calc_slice(car_count, sizeof(float)) * 17;

    // --- Controls SoA slices (length = car_count) ---
    total += calc_slice(car_count, sizeof(float)) * 5;
    total += calc_slice(car_count, sizeof(uint8_t)) * 3;

    // --- Arena SoA slices (length = env_count) ---
    total += calc_slice(env_count, sizeof(uint8_t));                  // is_goal
    total += calc_slice(env_count, sizeof(uint8_t));                  // scoring_team
    total += calc_slice(env_count, sizeof(uint8_t));                  // is_out_of_bounds
    total += calc_slice(env_count, sizeof(uint32_t));                 // tick_count
    total += calc_slice(env_count, sizeof(uint8_t));                  // terminated
    total += calc_slice(env_count, sizeof(uint8_t));                  // truncated
    total += calc_slice(car_count, sizeof(float));                    // rewards
    total += calc_slice(env_count * MAX_BOOST_PADS, sizeof(uint8_t)); // pad_is_active
    total += calc_slice(env_count * MAX_BOOST_PADS, sizeof(float));   // pad_cooldown

    // --- Staging POD buffers (pre-allocated to eliminate runtime allocations) ---
    total += calc_slice(env_count, sizeof(BallStatePOD));
    total += calc_slice(car_count, sizeof(CarStatePOD));
    total += calc_slice(car_count, sizeof(CarControls));

    m_allocated_bytes = total;

    // Single pre-allocated device memory arena (GEMINI.md Section 2.2)
    cudaError_t err = cudaMalloc(&m_d_pool, m_allocated_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed in SimContext: ") + cudaGetErrorString(err));
    }

    // Zero entire memory pool
    cudaMemsetAsync(m_d_pool, 0, m_allocated_bytes, m_stream);

    // Carve out 128-byte aligned slices
    size_t offset = 0;
    auto assign_slice = [&](size_t count, size_t elem_size) -> void* {
        void* ptr = static_cast<char*>(m_d_pool) + offset;
        offset += calc_slice(count, elem_size);
        return ptr;
    };

    // Assign Ball pointers
    m_ball_state.pos_x = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.pos_y = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.pos_z = static_cast<float*>(assign_slice(env_count, sizeof(float)));

    m_ball_state.vel_x = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.vel_y = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.vel_z = static_cast<float*>(assign_slice(env_count, sizeof(float)));

    m_ball_state.q_w   = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.q_x   = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.q_y   = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.q_z   = static_cast<float*>(assign_slice(env_count, sizeof(float)));

    m_ball_state.ang_vel_x = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.ang_vel_y = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.ang_vel_z = static_cast<float*>(assign_slice(env_count, sizeof(float)));

    m_ball_state.hs_y_target_dir     = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.hs_cur_target_speed = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.hs_time_since_hit   = static_cast<float*>(assign_slice(env_count, sizeof(float)));

    m_ball_state.ds_charge_level        = static_cast<int32_t*>(assign_slice(env_count, sizeof(int32_t)));
    m_ball_state.ds_accumulated_hit_force = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.ds_y_target_dir        = static_cast<float*>(assign_slice(env_count, sizeof(float)));
    m_ball_state.ds_has_damaged         = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_ball_state.ds_last_damage_tick    = static_cast<uint64_t*>(assign_slice(env_count, sizeof(uint64_t)));

    // Assign Car pointers
    m_car_state.pos_x = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.pos_y = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.pos_z = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.vel_x = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.vel_y = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.vel_z = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.q_w   = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.q_x   = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.q_y   = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.q_z   = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.ang_vel_x = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ang_vel_y = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ang_vel_z = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.boost              = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.time_since_boosted = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.boosting_time      = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.is_boosting        = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));

    m_car_state.is_on_ground     = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.wheel_contact_0  = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.wheel_contact_1  = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.wheel_contact_2  = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.wheel_contact_3  = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));

    m_car_state.suspension_length_0 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.suspension_length_1 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.suspension_length_2 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.suspension_length_3 = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.has_jumped        = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.is_jumping        = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.jump_time         = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.has_double_jumped = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.air_time          = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.air_time_since_jump = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.has_flipped       = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.is_flipping       = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.flip_time         = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.flip_rel_torque_x = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.flip_rel_torque_y = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.flip_rel_torque_z = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.is_auto_flipping     = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.auto_flip_timer        = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.auto_flip_torque_scale = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.handbrake_val          = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.is_supersonic    = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.supersonic_time    = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.is_demoed        = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.demo_respawn_timer = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.world_contact_has_contact = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.world_contact_normal_x    = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.world_contact_normal_y    = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.world_contact_normal_z    = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.car_contact_other_car_id  = static_cast<int32_t*>(assign_slice(car_count, sizeof(int32_t)));
    m_car_state.car_contact_cooldown_timer= static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.ball_hit_is_valid         = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.ball_hit_rel_pos_x        = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ball_hit_rel_pos_y        = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ball_hit_rel_pos_z        = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ball_hit_extra_hit_force_x= static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ball_hit_extra_hit_force_y= static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ball_hit_extra_hit_force_z= static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.ball_hit_tick_count       = static_cast<uint64_t*>(assign_slice(car_count, sizeof(uint64_t)));

    m_car_state.last_controls_throttle    = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.last_controls_steer       = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.last_controls_pitch       = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.last_controls_yaw         = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.last_controls_roll        = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.last_controls_boost       = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.last_controls_jump        = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_car_state.last_controls_handbrake   = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));

    m_car_state.wheel_engine_force        = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_brake               = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_steer_angle         = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_lat_friction_0      = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_lat_friction_1      = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_lat_friction_2      = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_lat_friction_3      = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_long_friction_0     = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_long_friction_1     = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_long_friction_2     = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.wheel_long_friction_3     = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    m_car_state.pos_bt_x                 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.pos_bt_y                 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.pos_bt_z                 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.vel_bt_x                 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.vel_bt_y                 = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_car_state.vel_bt_z                 = static_cast<float*>(assign_slice(car_count, sizeof(float)));

    // Assign Controls pointers
    m_controls.throttle  = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_controls.steer     = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_controls.pitch     = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_controls.yaw       = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_controls.roll      = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_controls.boost     = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_controls.jump      = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));
    m_controls.handbrake = static_cast<uint8_t*>(assign_slice(car_count, sizeof(uint8_t)));

    // Assign Arena pointers
    m_arena_state.is_goal          = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_arena_state.scoring_team     = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_arena_state.is_out_of_bounds = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_arena_state.tick_count       = static_cast<uint32_t*>(assign_slice(env_count, sizeof(uint32_t)));
    m_arena_state.terminated       = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_arena_state.truncated        = static_cast<uint8_t*>(assign_slice(env_count, sizeof(uint8_t)));
    m_arena_state.rewards          = static_cast<float*>(assign_slice(car_count, sizeof(float)));
    m_arena_state.pad_is_active    = static_cast<uint8_t*>(assign_slice(env_count * MAX_BOOST_PADS, sizeof(uint8_t)));
    m_arena_state.pad_cooldown     = static_cast<float*>(assign_slice(env_count * MAX_BOOST_PADS, sizeof(float)));
}

void SimContext::FreeArena() {
    if (m_d_pool) {
        cudaFree(m_d_pool);
        m_d_pool = nullptr;
    }
    m_allocated_bytes = 0;
}

void SimContext::ResetToDefault() {
    if (m_num_envs == 0) return;

    constexpr uint32_t threads = 128;
    uint32_t ball_blocks = (m_num_envs + threads - 1) / threads;
    k_init_ball_state<<<ball_blocks, threads, 0, m_stream>>>(m_ball_state, m_num_envs);

    if (m_total_cars > 0) {
        uint32_t car_blocks = (m_total_cars + threads - 1) / threads;
        k_init_car_state<<<car_blocks, threads, 0, m_stream>>>(m_car_state, m_total_cars);
    }

    uint32_t arena_blocks = (m_num_envs + threads - 1) / threads;
    k_init_arena_state<<<arena_blocks, threads, 0, m_stream>>>(m_arena_state, m_num_envs, m_cars_per_env);

    if (m_stream) {
        cudaStreamSynchronize(m_stream);
    } else {
        cudaDeviceSynchronize();
    }
}

void SimContext::ResetEnvironmentsIndexed(const int32_t* d_env_indices, uint32_t num_resets) {
    if (!d_env_indices || num_resets == 0) return;
    constexpr uint32_t threads = 128;
    uint32_t blocks = (num_resets + threads - 1) / threads;
    k_reset_environments_indexed<<<blocks, threads, 0, m_stream>>>(
        d_env_indices, num_resets, m_cars_per_env, m_ball_state, m_car_state, m_arena_state
    );
    // Asynchronous execution on m_stream with NO host synchronization barriers
}

void SimContext::ResetEnvironmentsIndexed(const int64_t* d_env_indices, uint32_t num_resets) {
    if (!d_env_indices || num_resets == 0) return;
    constexpr uint32_t threads = 128;
    uint32_t blocks = (num_resets + threads - 1) / threads;
    k_reset_environments_indexed<<<blocks, threads, 0, m_stream>>>(
        d_env_indices, num_resets, m_cars_per_env, m_ball_state, m_car_state, m_arena_state
    );
    // Asynchronous execution on m_stream with NO host synchronization barriers
}

void SimContext::ResetEnvironmentsMasked(const uint8_t* d_reset_mask) {
    if (!d_reset_mask || m_num_envs == 0) return;
    constexpr uint32_t threads = 128;
    uint32_t blocks = (m_num_envs + threads - 1) / threads;
    k_reset_environments_masked<<<blocks, threads, 0, m_stream>>>(
        d_reset_mask, m_num_envs, m_cars_per_env, m_ball_state, m_car_state, m_arena_state
    );
    // Asynchronous execution on m_stream with NO host synchronization barriers
}

void SimContext::CopyBallStateToHost(BallStatePOD* host_out, uint32_t env_start, uint32_t count) {
    if (count == 0) count = m_num_envs - env_start;
    if (count == 0 || host_out == nullptr) return;

    // Use pre-allocated staging buffer at end of pool
    char* staging_base = static_cast<char*>(m_d_pool) + m_allocated_bytes
        - align_128(m_num_envs * sizeof(BallStatePOD))
        - align_128(m_total_cars * sizeof(CarStatePOD))
        - align_128(m_total_cars * sizeof(CarControls));
    BallStatePOD* d_staging = reinterpret_cast<BallStatePOD*>(staging_base);

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_export_ball_pod<<<blocks, threads, 0, m_stream>>>(m_ball_state, d_staging, count, env_start);

    cudaMemcpyAsync(host_out, d_staging, count * sizeof(BallStatePOD), cudaMemcpyDeviceToHost, m_stream);
    if (m_stream) cudaStreamSynchronize(m_stream); else cudaDeviceSynchronize();
}

void SimContext::CopyCarStateToHost(CarStatePOD* host_out, uint32_t car_start, uint32_t count) {
    if (count == 0) count = m_total_cars - car_start;
    if (count == 0 || host_out == nullptr) return;

    char* staging_base = static_cast<char*>(m_d_pool) + m_allocated_bytes
        - align_128(m_total_cars * sizeof(CarStatePOD))
        - align_128(m_total_cars * sizeof(CarControls));
    CarStatePOD* d_staging = reinterpret_cast<CarStatePOD*>(staging_base);

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_export_car_pod<<<blocks, threads, 0, m_stream>>>(m_car_state, d_staging, count, car_start);

    cudaMemcpyAsync(host_out, d_staging, count * sizeof(CarStatePOD), cudaMemcpyDeviceToHost, m_stream);
    if (m_stream) cudaStreamSynchronize(m_stream); else cudaDeviceSynchronize();
}

void SimContext::CopyBallStateToDevice(const BallStatePOD* host_in, uint32_t env_start, uint32_t count) {
    if (count == 0) count = m_num_envs - env_start;
    if (count == 0 || host_in == nullptr) return;

    char* staging_base = static_cast<char*>(m_d_pool) + m_allocated_bytes
        - align_128(m_num_envs * sizeof(BallStatePOD))
        - align_128(m_total_cars * sizeof(CarStatePOD))
        - align_128(m_total_cars * sizeof(CarControls));
    BallStatePOD* d_staging = reinterpret_cast<BallStatePOD*>(staging_base);

    cudaMemcpyAsync(d_staging, host_in, count * sizeof(BallStatePOD), cudaMemcpyHostToDevice, m_stream);

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_import_ball_pod<<<blocks, threads, 0, m_stream>>>(m_ball_state, d_staging, count, env_start);

    if (m_stream) cudaStreamSynchronize(m_stream); else cudaDeviceSynchronize();
}

void SimContext::CopyCarStateToDevice(const CarStatePOD* host_in, uint32_t car_start, uint32_t count) {
    if (count == 0) count = m_total_cars - car_start;
    if (count == 0 || host_in == nullptr) return;

    char* staging_base = static_cast<char*>(m_d_pool) + m_allocated_bytes
        - align_128(m_total_cars * sizeof(CarStatePOD))
        - align_128(m_total_cars * sizeof(CarControls));
    CarStatePOD* d_staging = reinterpret_cast<CarStatePOD*>(staging_base);

    cudaMemcpyAsync(d_staging, host_in, count * sizeof(CarStatePOD), cudaMemcpyHostToDevice, m_stream);

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_import_car_pod<<<blocks, threads, 0, m_stream>>>(m_car_state, d_staging, count, car_start);

    if (m_stream) cudaStreamSynchronize(m_stream); else cudaDeviceSynchronize();
}

void SimContext::CopyControlsToDevice(const CarControls* host_in, uint32_t car_start, uint32_t count) {
    if (count == 0) count = m_total_cars - car_start;
    if (count == 0 || host_in == nullptr) return;

    char* staging_base = static_cast<char*>(m_d_pool) + m_allocated_bytes
        - align_128(m_total_cars * sizeof(CarControls));
    CarControls* d_staging = reinterpret_cast<CarControls*>(staging_base);

    cudaMemcpyAsync(d_staging, host_in, count * sizeof(CarControls), cudaMemcpyHostToDevice, m_stream);

    constexpr uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;
    k_import_controls<<<blocks, threads, 0, m_stream>>>(m_controls, d_staging, count, car_start);

    if (m_stream) cudaStreamSynchronize(m_stream); else cudaDeviceSynchronize();
}

} // namespace rocketsim_cuda
