#pragma once
#include <cstdint>
#include <cstddef>

namespace rocketsim_cuda {

// Simulation Timing
constexpr float TICK_RATE = 120.0f;
constexpr float DELTA_TIME = 1.0f / TICK_RATE; // 0.008333333333333333f

// Concurrency Limits
constexpr uint32_t DEFAULT_MAX_ENVS = 65536;
constexpr uint32_t MAX_CARS_PER_ENV = 8;
constexpr uint32_t MAX_BOOST_PADS = 34;

// Physical Constants (matching RocketSim CPU RLConst)
constexpr float GRAVITY_Z = -650.0f;
constexpr float CAR_MASS = 180.0f;
constexpr float BALL_MASS = CAR_MASS / 6.0f; // 30.0f
constexpr float CAR_MAX_SPEED = 2300.0f;
constexpr float BALL_MAX_SPEED = 6000.0f;
constexpr float CAR_MAX_ANG_SPEED = 5.5f;
constexpr float BALL_MAX_ANG_SPEED = 6.0f;
constexpr float BALL_RADIUS = 91.25f;
constexpr float BALL_REST_Z = 93.15f;
constexpr float BALL_DRAG = 0.03f;
constexpr float BALL_FRICTION = 0.35f;
constexpr float BALL_RESTITUTION = 0.6f;

// Boost Constants
constexpr float BOOST_MAX = 100.0f;
constexpr float BOOST_SPAWN = BOOST_MAX / 3.0f;
constexpr float BOOST_CONSUMPTION_RATE = BOOST_MAX / 3.0f; // 33.33333f/s
constexpr float BOOST_MIN_TIME = 0.1f;
constexpr float BOOST_ACCEL_GROUND = 2975.0f / 3.0f;
constexpr float BOOST_ACCEL_AIR = 3175.0f / 3.0f;

// Supersonic Constants
constexpr float SUPERSONIC_START_SPEED = 2200.0f;
constexpr float SUPERSONIC_MAINTAIN_MIN_SPEED = 2100.0f;
constexpr float SUPERSONIC_MAINTAIN_MAX_TIME = 1.0f;

// Arena Dimensions (Standard Soccar)
constexpr float ARENA_EXTENT_X = 4096.0f;
constexpr float ARENA_EXTENT_Y = 5120.0f;
constexpr float ARENA_HEIGHT = 2048.0f;
constexpr float ARENA_RAMP_RADIUS = 260.0f;
constexpr float GOAL_WIDTH = 1785.6f;
constexpr float GOAL_HEIGHT = 642.7f;
constexpr float GOAL_DEPTH = 880.0f;
constexpr float SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y = 5124.25f;
constexpr float GOAL_SCORE_THRESHOLD_Y = SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y + BALL_RADIUS; // 5215.5f

// Differential Parity Tolerances (Chebyshev Norm ||Delta||_inf per GEMINI.md Section 3.1)
constexpr float TOL_POS = 1e-4f;          // <= 10^-4 UU per tick
constexpr float TOL_VEL = 1e-3f;          // <= 10^-3 UU/s per tick
constexpr float TOL_QUAT = 1e-5f;         // <= 10^-5 per tick
constexpr float TOL_ANG_VEL = 1e-4f;      // <= 10^-4 rad/s per tick
constexpr float TOL_SUSPENSION = 1e-4f;   // <= 10^-4 UU per tick
constexpr float TOL_BOOST = 0.0f;         // 0.0 bit-exact float per tick

} // namespace rocketsim_cuda
