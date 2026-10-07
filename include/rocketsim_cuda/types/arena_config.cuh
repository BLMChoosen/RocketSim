#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "../math/vec3.cuh"
#include "../config.h"

// ============================================================================
// RocketSim-CUDA: Arena & Mutator Configurations (R6)
//
// Mirrored CPU Oracle Sources:
// - src/Sim/Arena/ArenaConfig/ArenaConfig.h:10-50 (Arena settings, memory modes)
// - src/Sim/MutatorConfig/MutatorConfig.h:10-79 (Mutator definitions)
// - src/Sim/MutatorConfig/MutatorConfig.cpp:5-41 (GameMode preset defaults)
// - src/RLConst.h:12-74, 94-95, 117-121, 144-147, 258-267 (Physical constants)
// ============================================================================

namespace rocketsim_cuda {

// Game Modes supported by RocketSim CPU
enum class GameMode : uint8_t {
    SOCCAR     = 0,
    HOOPS      = 1,
    DROPSHOT   = 2,
    SNOWDAY    = 3,
    THE_VOID   = 4,
    HEATSEEKER = 5
};

// Demolition Modes mirrored from src/Sim/MutatorConfig/MutatorConfig.h:10-14
enum class DemoMode : uint8_t {
    NORMAL     = 0,
    ON_CONTACT = 1,
    DISABLED   = 2
};

// Arena Memory Weight Modes mirrored from src/Sim/Arena/ArenaConfig/ArenaConfig.h:12-16
enum class ArenaMemWeightMode : uint8_t {
    HEAVY = 0, // High-performance / full cache
    LIGHT = 1  // Compact footprint
};

// ============================================================================
// MutatorConfig POD Struct (Consumable by Host & Device)
// Mirrored from src/Sim/MutatorConfig/MutatorConfig.h:16-79
// ============================================================================
struct alignas(16) MutatorConfig {
    // 1. Gravity & Rigid Body Dynamics
    Vec3 gravity                 = Vec3(0.0f, 0.0f, GRAVITY_Z); // Default: {0, 0, -650}
    float car_mass               = CAR_MASS;                    // 180.0f BT
    float car_world_friction     = 0.3f;                        // CARWORLD_COLLISION_FRICTION
    float car_world_restitution  = 0.3f;                        // CARWORLD_COLLISION_RESTITUTION

    // 2. Ball Physical Properties
    float ball_mass              = BALL_MASS;                   // 30.0f BT (180 / 6)
    float ball_max_speed         = BALL_MAX_SPEED;              // 6000.0f UU/s
    float ball_drag              = BALL_DRAG;                   // 0.03f
    float ball_world_friction    = BALL_FRICTION;               // 0.35f
    float ball_world_restitution = BALL_RESTITUTION;            // 0.6f
    float ball_radius            = BALL_RADIUS;                 // 91.25f UU (Soccar)

    // 3. Jump & Air Mechanics
    float jump_accel             = 4375.0f / 3.0f;              // JUMP_ACCEL (~1458.3333f)
    float jump_immediate_force   = 875.0f / 3.0f;               // JUMP_IMMEDIATE_FORCE (~291.6667f)
    bool unlimited_flips         = false;
    bool unlimited_double_jumps  = false;

    // 4. Boost Mechanics
    float boost_accel_ground     = BOOST_ACCEL_GROUND;          // 2975.0f / 3.0f (~991.6667f)
    float boost_accel_air        = BOOST_ACCEL_AIR;             // 3175.0f / 3.0f (~1058.3333f)
    float boost_used_per_second  = BOOST_CONSUMPTION_RATE;      // 100.0f / 3.0f (~33.33333f)
    float car_spawn_boost_amount = BOOST_SPAWN;                 // 100.0f / 3.0f (~33.33333f)
    bool recharge_boost_enabled  = false;
    float recharge_boost_per_sec = 10.0f;                       // RECHARGE_BOOST_PER_SECOND
    float recharge_boost_delay   = 0.25f;                       // RECHARGE_BOOST_DELAY

    // 5. Demolition, Respawn & Bumps
    float respawn_delay          = 3.0f;                        // DEMO_RESPAWN_TIME
    float bump_cooldown_time     = 0.25f;                       // BUMP_COOLDOWN_TIME
    DemoMode demo_mode           = DemoMode::NORMAL;
    bool enable_team_demos       = false;

    // 6. Boost Pad Cooldown Timers
    float boost_pad_cooldown_big   = 10.0f;                     // BoostPads::COOLDOWN_BIG
    float boost_pad_cooldown_small = 4.0f;                      // BoostPads::COOLDOWN_SMALL

    // 7. Impact Scaling & Score Triggers
    float ball_hit_extra_force_scale = 1.0f;
    float bump_force_scale           = 1.0f;
    float goal_base_threshold_y      = SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y; // 5124.25f

    // Default Constructor: Standard Soccar
    __host__ __device__ constexpr MutatorConfig() = default;

    // GameMode Constructor mirroring src/Sim/MutatorConfig/MutatorConfig.cpp:5-41
    __host__ __device__ constexpr explicit MutatorConfig(GameMode game_mode) {
        switch (game_mode) {
            case GameMode::HOOPS:
                ball_radius = 96.3831f; // BALL_COLLISION_RADIUS_HOOPS
                break;
            case GameMode::SNOWDAY:
                ball_radius = 114.25f;  // Snowday::PUCK_RADIUS
                break;
            case GameMode::DROPSHOT:
                ball_radius = 100.2565f;// BALL_COLLISION_RADIUS_DROPSHOT
                break;
            default:
                ball_radius = BALL_RADIUS; // 91.25f (Soccar)
                break;
        }

        if (game_mode == GameMode::SNOWDAY) {
            ball_world_friction    = 0.1f; // Snowday::PUCK_FRICTION
            ball_world_restitution = 0.3f; // Snowday::PUCK_RESTITUTION
            ball_mass              = 50.0f;// Snowday::PUCK_MASS_BT
        } else {
            ball_world_friction    = BALL_FRICTION;    // 0.35f
            ball_world_restitution = BALL_RESTITUTION; // 0.6f
            ball_mass              = BALL_MASS;        // 30.0f
        }

        if (game_mode == GameMode::HEATSEEKER) {
            car_spawn_boost_amount = 100.0f;
            boost_used_per_second  = 0.0f;
        } else if (game_mode == GameMode::DROPSHOT) {
            car_spawn_boost_amount = 100.0f;
            recharge_boost_enabled = true;
        }
    }
};

// ============================================================================
// ArenaConfig POD Struct (Consumable by Host & Device)
// Mirrored from src/Sim/Arena/ArenaConfig/ArenaConfig.h:18-50
// ============================================================================
struct alignas(16) ArenaConfig {
    ArenaMemWeightMode mem_weight_mode = ArenaMemWeightMode::HEAVY;

    // Minimum and maximum positions for all physics objects in arena
    Vec3 min_pos = Vec3(-5600.0f, -6000.0f, 0.0f);
    Vec3 max_pos = Vec3( 5600.0f,  6000.0f, 2200.0f);

    // Maximum diagonal length of any object AABB
    float max_aabb_len = 370.0f;

    // Ball rotation performance flag (disabled in Snowday)
    bool no_ball_rot = true;

    // Broadphase optimization flag
    bool use_custom_broadphase = true;

    // Maximum number of active rigid objects
    int32_t max_objects = 512;

    // Fixed capacity boost pad configuration
    bool use_custom_boost_pads = false;
    uint32_t num_boost_pads    = MAX_BOOST_PADS; // Default: 34 Soccar pads
};

// ============================================================================
// Structure of Arrays (SoA) for Mutator Configurations (GEMINI.md Invariant 2.1)
// Enables coalesced 128-byte transactions for multi-arena domain randomization
// ============================================================================
struct MutatorConfigSoA {
    // 1. Gravity & Car Rigid Body
    float* __restrict__ gravity_x = nullptr;
    float* __restrict__ gravity_y = nullptr;
    float* __restrict__ gravity_z = nullptr;
    float* __restrict__ car_mass  = nullptr;
    float* __restrict__ car_world_friction    = nullptr;
    float* __restrict__ car_world_restitution = nullptr;

    // 2. Ball Properties
    float* __restrict__ ball_mass              = nullptr;
    float* __restrict__ ball_radius            = nullptr;
    float* __restrict__ ball_max_speed         = nullptr;
    float* __restrict__ ball_drag              = nullptr;
    float* __restrict__ ball_world_friction    = nullptr;
    float* __restrict__ ball_world_restitution = nullptr;

    // 3. Jump & Air Mechanics
    float* __restrict__ jump_accel             = nullptr;
    float* __restrict__ jump_immediate_force   = nullptr;
    uint8_t* __restrict__ unlimited_flips      = nullptr;
    uint8_t* __restrict__ unlimited_double_jumps = nullptr;

    // 4. Boost Mechanics
    float* __restrict__ boost_accel_ground     = nullptr;
    float* __restrict__ boost_accel_air        = nullptr;
    float* __restrict__ boost_used_per_second  = nullptr;
    float* __restrict__ car_spawn_boost_amount = nullptr;
    uint8_t* __restrict__ recharge_boost_enabled = nullptr;
    float* __restrict__ recharge_boost_per_sec = nullptr;
    float* __restrict__ recharge_boost_delay   = nullptr;

    // 5. Demolitions & Bumps
    float* __restrict__ respawn_delay          = nullptr;
    float* __restrict__ bump_cooldown_time     = nullptr;
    uint8_t* __restrict__ demo_mode            = nullptr;
    uint8_t* __restrict__ enable_team_demos    = nullptr;

    // 6. Boost Pads & Scoring
    float* __restrict__ boost_pad_cooldown_big   = nullptr;
    float* __restrict__ boost_pad_cooldown_small = nullptr;
    float* __restrict__ ball_hit_extra_force_scale = nullptr;
    float* __restrict__ bump_force_scale       = nullptr;
    float* __restrict__ goal_base_threshold_y  = nullptr;

    // Reads single-environment MutatorConfig into thread local registers
    __device__ inline MutatorConfig get(uint32_t env_idx) const {
        MutatorConfig cfg;
        if (gravity_x) {
            cfg.gravity = Vec3(gravity_x[env_idx], gravity_y[env_idx], gravity_z[env_idx]);
        }
        if (car_mass)               cfg.car_mass               = car_mass[env_idx];
        if (car_world_friction)     cfg.car_world_friction     = car_world_friction[env_idx];
        if (car_world_restitution)  cfg.car_world_restitution  = car_world_restitution[env_idx];
        if (ball_mass)              cfg.ball_mass              = ball_mass[env_idx];
        if (ball_radius)            cfg.ball_radius            = ball_radius[env_idx];
        if (ball_max_speed)         cfg.ball_max_speed         = ball_max_speed[env_idx];
        if (ball_drag)              cfg.ball_drag              = ball_drag[env_idx];
        if (ball_world_friction)    cfg.ball_world_friction    = ball_world_friction[env_idx];
        if (ball_world_restitution) cfg.ball_world_restitution = ball_world_restitution[env_idx];
        if (jump_accel)             cfg.jump_accel             = jump_accel[env_idx];
        if (jump_immediate_force)   cfg.jump_immediate_force   = jump_immediate_force[env_idx];
        if (unlimited_flips)        cfg.unlimited_flips        = (unlimited_flips[env_idx] != 0);
        if (unlimited_double_jumps) cfg.unlimited_double_jumps = (unlimited_double_jumps[env_idx] != 0);
        if (boost_accel_ground)     cfg.boost_accel_ground     = boost_accel_ground[env_idx];
        if (boost_accel_air)        cfg.boost_accel_air        = boost_accel_air[env_idx];
        if (boost_used_per_second)  cfg.boost_used_per_second  = boost_used_per_second[env_idx];
        if (car_spawn_boost_amount) cfg.car_spawn_boost_amount = car_spawn_boost_amount[env_idx];
        if (recharge_boost_enabled) cfg.recharge_boost_enabled = (recharge_boost_enabled[env_idx] != 0);
        if (recharge_boost_per_sec) cfg.recharge_boost_per_sec = recharge_boost_per_sec[env_idx];
        if (recharge_boost_delay)   cfg.recharge_boost_delay   = recharge_boost_delay[env_idx];
        if (respawn_delay)          cfg.respawn_delay          = respawn_delay[env_idx];
        if (bump_cooldown_time)     cfg.bump_cooldown_time     = bump_cooldown_time[env_idx];
        if (demo_mode)              cfg.demo_mode              = static_cast<DemoMode>(demo_mode[env_idx]);
        if (enable_team_demos)      cfg.enable_team_demos      = (enable_team_demos[env_idx] != 0);
        if (boost_pad_cooldown_big)   cfg.boost_pad_cooldown_big   = boost_pad_cooldown_big[env_idx];
        if (boost_pad_cooldown_small) cfg.boost_pad_cooldown_small = boost_pad_cooldown_small[env_idx];
        if (ball_hit_extra_force_scale) cfg.ball_hit_extra_force_scale = ball_hit_extra_force_scale[env_idx];
        if (bump_force_scale)       cfg.bump_force_scale       = bump_force_scale[env_idx];
        if (goal_base_threshold_y)  cfg.goal_base_threshold_y  = goal_base_threshold_y[env_idx];
        return cfg;
    }
};

// ============================================================================
// Structure of Arrays (SoA) for Arena Configurations (GEMINI.md Invariant 2.1)
// ============================================================================
struct ArenaConfigSoA {
    uint8_t* __restrict__ mem_weight_mode = nullptr;

    float* __restrict__ min_pos_x = nullptr;
    float* __restrict__ min_pos_y = nullptr;
    float* __restrict__ min_pos_z = nullptr;

    float* __restrict__ max_pos_x = nullptr;
    float* __restrict__ max_pos_y = nullptr;
    float* __restrict__ max_pos_z = nullptr;

    float* __restrict__ max_aabb_len = nullptr;
    uint8_t* __restrict__ no_ball_rot = nullptr;
    uint8_t* __restrict__ use_custom_broadphase = nullptr;
    int32_t* __restrict__ max_objects = nullptr;
    uint8_t* __restrict__ use_custom_boost_pads = nullptr;
    uint32_t* __restrict__ num_boost_pads = nullptr;

    __device__ inline ArenaConfig get(uint32_t env_idx) const {
        ArenaConfig cfg;
        if (mem_weight_mode) cfg.mem_weight_mode = static_cast<ArenaMemWeightMode>(mem_weight_mode[env_idx]);
        if (min_pos_x) cfg.min_pos = Vec3(min_pos_x[env_idx], min_pos_y[env_idx], min_pos_z[env_idx]);
        if (max_pos_x) cfg.max_pos = Vec3(max_pos_x[env_idx], max_pos_y[env_idx], max_pos_z[env_idx]);
        if (max_aabb_len) cfg.max_aabb_len = max_aabb_len[env_idx];
        if (no_ball_rot) cfg.no_ball_rot = (no_ball_rot[env_idx] != 0);
        if (use_custom_broadphase) cfg.use_custom_broadphase = (use_custom_broadphase[env_idx] != 0);
        if (max_objects) cfg.max_objects = max_objects[env_idx];
        if (use_custom_boost_pads) cfg.use_custom_boost_pads = (use_custom_boost_pads[env_idx] != 0);
        if (num_boost_pads) cfg.num_boost_pads = num_boost_pads[env_idx];
        return cfg;
    }
};

} // namespace rocketsim_cuda
