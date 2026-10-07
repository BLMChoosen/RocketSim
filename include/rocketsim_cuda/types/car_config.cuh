#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "../math/vec3.cuh"
#include "../config.h"

// ============================================================================
// RocketSim-CUDA: Official Car Hitbox Presets & Configurations (R5)
//
// Mirrored CPU Oracle Sources:
// - src/Sim/Car/CarConfig/CarConfig.h:6-50 (Hitbox & Wheel Pair definitions)
// - src/Sim/Car/CarConfig/CarConfig.cpp:20-101 (Official RL Hitbox Tables:
//   Octane, Dominus, Plank, Breakout, Hybrid, Merc, Psyclops)
// - src/Sim/Car/Car.cpp:210-220, 261-295 (Hitbox half-extents, inertia,
//   wheel hardpoint symmetry, effective suspension rest length)
// - src/RLConst.h:28, 84-85, 158-165 (Vehicle tuning constants & mass)
// ============================================================================

namespace rocketsim_cuda {

// Canonical Car Hitbox Presets enumeration matching RocketSim CPU
enum class CarHitboxType : uint8_t {
    OCTANE   = 0,
    DOMINUS  = 1,
    PLANK    = 2, // Batmobile preset
    BREAKOUT = 3,
    HYBRID   = 4,
    MERC     = 5,
    PSYCLOPS = 6, // 3-wheel preset
    COUNT    = 7
};

// Numeric constants for direct indexing and tensor bindings
constexpr uint8_t CAR_HITBOX_OCTANE   = 0;
constexpr uint8_t CAR_HITBOX_DOMINUS  = 1;
constexpr uint8_t CAR_HITBOX_PLANK    = 2;
constexpr uint8_t CAR_HITBOX_BREAKOUT = 3;
constexpr uint8_t CAR_HITBOX_HYBRID   = 4;
constexpr uint8_t CAR_HITBOX_MERC     = 5;
constexpr uint8_t CAR_HITBOX_PSYCLOPS = 6;
constexpr uint8_t CAR_HITBOX_COUNT    = 7;

// Configuration for a pair of wheels (front or back)
// Mirrored from src/Sim/Car/CarConfig/CarConfig.h:6-16
struct alignas(16) WheelPairConfig {
    float wheel_radius;            // Wheel radius in UU
    float suspension_rest_length;  // Raw suspension rest length in UU
    Vec3 connection_point_offset;  // Suspension hardpoint start position (Y > 0)
};

// Full Car Configuration (Hitbox, Wheels, Inertia, Dodge Deadzone)
// Mirrored from src/Sim/Car/CarConfig/CarConfig.h:21-38
struct alignas(16) CarConfig {
    Vec3 hitbox_size;          // Full size (extent * 2) in UU: X=length, Y=width, Z=height
    Vec3 hitbox_pos_offset;     // Hitbox center offset from car origin in UU
    WheelPairConfig front_wheels;
    WheelPairConfig back_wheels;
    bool three_wheels = false; // Psyclops 3-wheel behavior flag
    float dodge_deadzone = 0.5f;

    // Returns half-extents of the collision box
    __host__ __device__ inline constexpr Vec3 get_hitbox_half() const {
        return Vec3(hitbox_size.x * 0.5f, hitbox_size.y * 0.5f, hitbox_size.z * 0.5f);
    }

    // Returns wheel radius for wheel index w in [0..3]
    __host__ __device__ inline constexpr float get_wheel_radius(int w) const {
        return (w < 2) ? front_wheels.wheel_radius : back_wheels.wheel_radius;
    }

    // Returns effective suspension rest length used in Bullet simulation
    // Mirrored from src/Sim/Car/Car.cpp:277-280:
    // suspensionRestLength -= RLConst::BTVehicle::MAX_SUSPENSION_TRAVEL (12.0 UU)
    __host__ __device__ inline constexpr float get_susp_rest_effective(int w) const {
        float raw = (w < 2) ? front_wheels.suspension_rest_length : back_wheels.suspension_rest_length;
        return raw - 12.0f;
    }

    // Returns suspension connection point offset in car-local frame
    // Mirrored from src/Sim/Car/Car.cpp:271-276:
    // Wheel 0: Front-Right (+Y), Wheel 1: Front-Left (-Y)
    // Wheel 2: Back-Right  (+Y), Wheel 3: Back-Left  (-Y)
    __host__ __device__ inline constexpr Vec3 get_wheel_connection_offset(int w) const {
        const WheelPairConfig& pair = (w < 2) ? front_wheels : back_wheels;
        float y_sign = (w % 2 != 0) ? -1.0f : 1.0f;
        return Vec3(pair.connection_point_offset.x, pair.connection_point_offset.y * y_sign, pair.connection_point_offset.z);
    }

    // Computes box moment of inertia in UU units (matching Bullet calculateLocalInertia)
    // Formula: I_BT = (mass / 12) * (l_a^2 + l_b^2); I_UU = I_BT * 2500.0f
    __host__ __device__ inline constexpr Vec3 calculate_inertia(float mass = CAR_MASS) const {
        float lx = hitbox_size.x * 0.02f;
        float ly = hitbox_size.y * 0.02f;
        float lz = hitbox_size.z * 0.02f;
        float m_factor = (mass / 12.0f) * 2500.0f;
        return Vec3(
            m_factor * (ly * ly + lz * lz),
            m_factor * (lx * lx + lz * lz),
            m_factor * (lx * lx + ly * ly)
        );
    }

    // Computes diagonal inverse moment of inertia in UU units
    __host__ __device__ inline constexpr Vec3 calculate_inv_inertia(float mass = CAR_MASS) const {
        Vec3 I = calculate_inertia(mass);
        return Vec3(1.0f / I.x, 1.0f / I.y, 1.0f / I.z);
    }
};

// ============================================================================
// Official Hitbox Preset Definitions (CarConfig.cpp:20-101)
// ============================================================================

// 1. OCTANE PRESET (Index 0)
// Hitbox Size:   {120.507f, 86.6994f, 38.6591f}
// Hitbox Offset: {13.8757f, 0.0f, 20.755f}
// Wheels Front:  r=12.50f, rest=38.755f, offset={51.25f, 25.90f, 20.755f}
// Wheels Back:   r=15.00f, rest=37.055f, offset={-33.75f, 29.50f, 20.755f}
inline constexpr CarConfig CAR_CONFIG_OCTANE = {
    Vec3(120.507f, 86.6994f, 38.6591f),
    Vec3(13.8757f, 0.0f, 20.755f),
    { 12.50f, 38.755f, Vec3(51.25f, 25.90f, 20.755f) },
    { 15.00f, 37.055f, Vec3(-33.75f, 29.50f, 20.755f) },
    false,
    0.5f
};

// 2. DOMINUS PRESET (Index 1)
// Hitbox Size:   {130.427f, 85.7799f, 33.8f}
// Hitbox Offset: {9.0f, 0.0f, 15.75f}
// Wheels Front:  r=12.00f, rest=33.95f, offset={50.30f, 31.10f, 15.75f}
// Wheels Back:   r=13.50f, rest=33.85f, offset={-34.75f, 33.00f, 15.75f}
inline constexpr CarConfig CAR_CONFIG_DOMINUS = {
    Vec3(130.427f, 85.7799f, 33.8f),
    Vec3(9.0f, 0.0f, 15.75f),
    { 12.00f, 33.95f, Vec3(50.30f, 31.10f, 15.75f) },
    { 13.50f, 33.85f, Vec3(-34.75f, 33.00f, 15.75f) },
    false,
    0.5f
};

// 3. PLANK PRESET (Index 2 - Batmobile)
// Hitbox Size:   {131.32f, 87.1704f, 31.8944f}
// Hitbox Offset: {9.00857f, 0.0f, 12.0942f}
// Wheels Front:  r=12.50f, rest=31.9242f, offset={49.97f, 27.80f, 12.0942f}
// Wheels Back:   r=17.00f, rest=27.9242f, offset={-35.43f, 20.28f, 12.0942f}
inline constexpr CarConfig CAR_CONFIG_PLANK = {
    Vec3(131.32f, 87.1704f, 31.8944f),
    Vec3(9.00857f, 0.0f, 12.0942f),
    { 12.50f, 31.9242f, Vec3(49.97f, 27.80f, 12.0942f) },
    { 17.00f, 27.9242f, Vec3(-35.43f, 20.28f, 12.0942f) },
    false,
    0.5f
};

// 4. BREAKOUT PRESET (Index 3)
// Hitbox Size:   {133.992f, 83.021f, 32.8f}
// Hitbox Offset: {12.5f, 0.0f, 11.75f}
// Wheels Front:  r=13.50f, rest=29.7f, offset={51.50f, 26.67f, 11.75f}
// Wheels Back:   r=15.00f, rest=29.666f, offset={-35.75f, 35.00f, 11.75f}
inline constexpr CarConfig CAR_CONFIG_BREAKOUT = {
    Vec3(133.992f, 83.021f, 32.8f),
    Vec3(12.5f, 0.0f, 11.75f),
    { 13.50f, 29.7f, Vec3(51.50f, 26.67f, 11.75f) },
    { 15.00f, 29.666f, Vec3(-35.75f, 35.00f, 11.75f) },
    false,
    0.5f
};

// 5. HYBRID PRESET (Index 4)
// Hitbox Size:   {129.519f, 84.6879f, 36.6591f}
// Hitbox Offset: {13.8757f, 0.0f, 20.755f}
// Wheels Front:  r=12.50f, rest=38.755f, offset={51.25f, 25.90f, 20.755f}
// Wheels Back:   r=15.00f, rest=37.055f, offset={-34.00f, 29.50f, 20.755f}
inline constexpr CarConfig CAR_CONFIG_HYBRID = {
    Vec3(129.519f, 84.6879f, 36.6591f),
    Vec3(13.8757f, 0.0f, 20.755f),
    { 12.50f, 38.755f, Vec3(51.25f, 25.90f, 20.755f) },
    { 15.00f, 37.055f, Vec3(-34.00f, 29.50f, 20.755f) },
    false,
    0.5f
};

// 6. MERC PRESET (Index 5)
// Hitbox Size:   {123.22f, 79.2103f, 44.1591f}
// Hitbox Offset: {11.3757f, 0.0f, 21.505f}
// Wheels Front:  r=15.00f, rest=39.505f, offset={51.25f, 25.90f, 21.505f}
// Wheels Back:   r=15.00f, rest=39.105f, offset={-33.75f, 29.50f, 21.505f}
inline constexpr CarConfig CAR_CONFIG_MERC = {
    Vec3(123.22f, 79.2103f, 44.1591f),
    Vec3(11.3757f, 0.0f, 21.505f),
    { 15.00f, 39.505f, Vec3(51.25f, 25.90f, 21.505f) },
    { 15.00f, 39.105f, Vec3(-33.75f, 29.50f, 21.505f) },
    false,
    0.5f
};

// 7. PSYCLOPS PRESET (Index 6 - 3-Wheel Experimental)
// Hitbox Size:   {120.641f, 86.8334f, 38.7931f} (Octane + 0.134f)
// Hitbox Offset: {13.8757f, 0.0f, 15.0f}
// Wheels Front:  r=12.50f, rest=33.0f, offset={51.25f, 5.0f, 15.0f}
// Wheels Back:   r=15.00f, rest=31.3f, offset={-33.75f, 29.50f, 15.0f}
inline constexpr CarConfig CAR_CONFIG_PSYCLOPS = {
    Vec3(120.507f + 0.134f, 86.6994f + 0.134f, 38.6591f + 0.134f),
    Vec3(13.8757f, 0.0f, 15.0f),
    { 12.50f, 33.0f, Vec3(51.25f, 5.0f, 15.0f) },
    { 15.00f, 31.3f, Vec3(-33.75f, 29.50f, 15.0f) },
    true,
    0.5f
};

// ============================================================================
// Constant Presets Table & Fast Lookup Helpers
// ============================================================================

#if defined(__CUDA_ARCH__)
__device__ static constexpr CarConfig CAR_CONFIG_PRESETS[CAR_HITBOX_COUNT] = {
#else
static constexpr CarConfig CAR_CONFIG_PRESETS[CAR_HITBOX_COUNT] = {
#endif
    CAR_CONFIG_OCTANE,
    CAR_CONFIG_DOMINUS,
    CAR_CONFIG_PLANK,
    CAR_CONFIG_BREAKOUT,
    CAR_CONFIG_HYBRID,
    CAR_CONFIG_MERC,
    CAR_CONFIG_PSYCLOPS
};

// Direct index lookup for full CarConfig (bounds-safe)
__host__ __device__ inline constexpr const CarConfig& get_car_config(uint8_t index) {
    if (index >= CAR_HITBOX_COUNT) {
        index = 0;
    }
    return CAR_CONFIG_PRESETS[index];
}

// Fast branchless/switch hitbox size lookup
__host__ __device__ inline constexpr Vec3 get_hitbox_size(uint8_t index) {
    switch (index) {
        case CAR_HITBOX_DOMINUS:  return CAR_CONFIG_DOMINUS.hitbox_size;
        case CAR_HITBOX_PLANK:    return CAR_CONFIG_PLANK.hitbox_size;
        case CAR_HITBOX_BREAKOUT: return CAR_CONFIG_BREAKOUT.hitbox_size;
        case CAR_HITBOX_HYBRID:   return CAR_CONFIG_HYBRID.hitbox_size;
        case CAR_HITBOX_MERC:     return CAR_CONFIG_MERC.hitbox_size;
        case CAR_HITBOX_PSYCLOPS: return CAR_CONFIG_PSYCLOPS.hitbox_size;
        case CAR_HITBOX_OCTANE:
        default:                  return CAR_CONFIG_OCTANE.hitbox_size;
    }
}

// Fast branchless/switch hitbox center offset lookup
__host__ __device__ inline constexpr Vec3 get_hitbox_offset(uint8_t index) {
    switch (index) {
        case CAR_HITBOX_DOMINUS:  return CAR_CONFIG_DOMINUS.hitbox_pos_offset;
        case CAR_HITBOX_PLANK:    return CAR_CONFIG_PLANK.hitbox_pos_offset;
        case CAR_HITBOX_BREAKOUT: return CAR_CONFIG_BREAKOUT.hitbox_pos_offset;
        case CAR_HITBOX_HYBRID:   return CAR_CONFIG_HYBRID.hitbox_pos_offset;
        case CAR_HITBOX_MERC:     return CAR_CONFIG_MERC.hitbox_pos_offset;
        case CAR_HITBOX_PSYCLOPS: return CAR_CONFIG_PSYCLOPS.hitbox_pos_offset;
        case CAR_HITBOX_OCTANE:
        default:                  return CAR_CONFIG_OCTANE.hitbox_pos_offset;
    }
}

// Fast branchless/switch hitbox half-extents lookup
__host__ __device__ inline constexpr Vec3 get_hitbox_half(uint8_t index) {
    Vec3 size = get_hitbox_size(index);
    return Vec3(size.x * 0.5f, size.y * 0.5f, size.z * 0.5f);
}

// Fast branchless/switch wheel suspension hardpoint offset lookup
__host__ __device__ inline constexpr Vec3 get_wheel_connection_offset(uint8_t index, int wheel_idx) {
    const CarConfig& cfg = get_car_config(index);
    return cfg.get_wheel_connection_offset(wheel_idx);
}

// Fast wheel radius lookup
__host__ __device__ inline constexpr float get_wheel_radius(uint8_t index, int wheel_idx) {
    const CarConfig& cfg = get_car_config(index);
    return cfg.get_wheel_radius(wheel_idx);
}

// Fast effective suspension rest length lookup
__host__ __device__ inline constexpr float get_susp_rest_effective(uint8_t index, int wheel_idx) {
    const CarConfig& cfg = get_car_config(index);
    return cfg.get_susp_rest_effective(wheel_idx);
}

// Fast diagonal inverse inertia lookup
__host__ __device__ inline constexpr Vec3 get_inv_inertia(uint8_t index, float mass = CAR_MASS) {
    const CarConfig& cfg = get_car_config(index);
    return cfg.calculate_inv_inertia(mass);
}

// ============================================================================
// Structure of Arrays (SoA) for Car Configurations (GEMINI.md Invariant 2.1)
// ============================================================================
struct CarConfigSoA {
    // Hitbox type enum for each car: 0=Octane, 1=Dominus, ..., 6=Psyclops
    uint8_t* __restrict__ hitbox_type = nullptr;

    // Optional precomputed per-car geometry buffers for 128-byte coalescing
    float* __restrict__ hitbox_half_x = nullptr;
    float* __restrict__ hitbox_half_y = nullptr;
    float* __restrict__ hitbox_half_z = nullptr;

    float* __restrict__ hitbox_offset_x = nullptr;
    float* __restrict__ hitbox_offset_y = nullptr;
    float* __restrict__ hitbox_offset_z = nullptr;

    float* __restrict__ inv_inertia_x = nullptr;
    float* __restrict__ inv_inertia_y = nullptr;
    float* __restrict__ inv_inertia_z = nullptr;

    float* __restrict__ front_wheel_radius = nullptr;
    float* __restrict__ back_wheel_radius  = nullptr;
    float* __restrict__ front_susp_rest    = nullptr;
    float* __restrict__ back_susp_rest     = nullptr;

    float* __restrict__ front_wheel_offset_x = nullptr;
    float* __restrict__ front_wheel_offset_y = nullptr;
    float* __restrict__ front_wheel_offset_z = nullptr;

    float* __restrict__ back_wheel_offset_x = nullptr;
    float* __restrict__ back_wheel_offset_y = nullptr;
    float* __restrict__ back_wheel_offset_z = nullptr;

    __device__ inline Vec3 get_hitbox_half(uint32_t car_idx) const {
        if (hitbox_half_x) {
            return Vec3(hitbox_half_x[car_idx], hitbox_half_y[car_idx], hitbox_half_z[car_idx]);
        }
        uint8_t type = hitbox_type ? hitbox_type[car_idx] : 0;
        return rocketsim_cuda::get_hitbox_half(type);
    }

    __device__ inline Vec3 get_hitbox_offset(uint32_t car_idx) const {
        if (hitbox_offset_x) {
            return Vec3(hitbox_offset_x[car_idx], hitbox_offset_y[car_idx], hitbox_offset_z[car_idx]);
        }
        uint8_t type = hitbox_type ? hitbox_type[car_idx] : 0;
        return rocketsim_cuda::get_hitbox_offset(type);
    }

    __device__ inline Vec3 get_inv_inertia(uint32_t car_idx) const {
        if (inv_inertia_x) {
            return Vec3(inv_inertia_x[car_idx], inv_inertia_y[car_idx], inv_inertia_z[car_idx]);
        }
        uint8_t type = hitbox_type ? hitbox_type[car_idx] : 0;
        return rocketsim_cuda::get_inv_inertia(type);
    }

    __device__ inline Vec3 get_wheel_connection_offset(uint32_t car_idx, int wheel_idx) const {
        uint8_t type = hitbox_type ? hitbox_type[car_idx] : 0;
        return rocketsim_cuda::get_wheel_connection_offset(type, wheel_idx);
    }

    __device__ inline float get_wheel_radius(uint32_t car_idx, int wheel_idx) const {
        if (wheel_idx < 2 && front_wheel_radius) return front_wheel_radius[car_idx];
        if (wheel_idx >= 2 && back_wheel_radius) return back_wheel_radius[car_idx];
        uint8_t type = hitbox_type ? hitbox_type[car_idx] : 0;
        return rocketsim_cuda::get_wheel_radius(type, wheel_idx);
    }

    __device__ inline float get_susp_rest_effective(uint32_t car_idx, int wheel_idx) const {
        if (wheel_idx < 2 && front_susp_rest) return front_susp_rest[car_idx];
        if (wheel_idx >= 2 && back_susp_rest) return back_susp_rest[car_idx];
        uint8_t type = hitbox_type ? hitbox_type[car_idx] : 0;
        return rocketsim_cuda::get_susp_rest_effective(type, wheel_idx);
    }
};

} // namespace rocketsim_cuda
