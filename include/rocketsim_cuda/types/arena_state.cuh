#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "../config.h"

namespace rocketsim_cuda {

// Boost pad definition (matching RLConst.h)
struct BoostPadDef {
    float x;
    float y;
    float z;
    float radius;
    float radius_sq;
    float boost_amount;
    float cooldown;
    bool is_big;
};

// 34 Standard Soccar Boost Pads Table (6 big pads, 28 small pads matching RLConst.h)
#if defined(__CUDA_ARCH__)
__device__ static constexpr BoostPadDef SOCCAR_BOOST_PADS[MAX_BOOST_PADS] = {
#else
static constexpr BoostPadDef SOCCAR_BOOST_PADS[MAX_BOOST_PADS] = {
#endif
    // 6 Big Pads (indices 0..5): 100 boost, 10s cooldown, 208 radius (RLConst.h:305-312)
    { -3584.0f,     0.0f, 73.0f, 208.0f, 43264.0f, 100.0f, 10.0f, true },
    {  3584.0f,     0.0f, 73.0f, 208.0f, 43264.0f, 100.0f, 10.0f, true },
    { -3072.0f,  4096.0f, 73.0f, 208.0f, 43264.0f, 100.0f, 10.0f, true },
    {  3072.0f,  4096.0f, 73.0f, 208.0f, 43264.0f, 100.0f, 10.0f, true },
    { -3072.0f, -4096.0f, 73.0f, 208.0f, 43264.0f, 100.0f, 10.0f, true },
    {  3072.0f, -4096.0f, 73.0f, 208.0f, 43264.0f, 100.0f, 10.0f, true },

    // 28 Small Pads (indices 6..33): 12 boost, 4s cooldown, 144 radius (RLConst.h:274-303)
    {     0.0f, -4240.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    { -1792.0f, -4184.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {  1792.0f, -4184.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {  -940.0f, -3308.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {   940.0f, -3308.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {     0.0f, -2816.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    { -3584.0f, -2484.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {  3584.0f, -2484.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    { -1788.0f, -2300.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {  1788.0f, -2300.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    { -2048.0f, -1036.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {     0.0f, -1024.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {  2048.0f, -1036.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    { -1024.0f,     0.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {  1024.0f,     0.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    { -2048.0f,  1036.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {     0.0f,  1024.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {  2048.0f,  1036.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    { -1788.0f,  2300.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {  1788.0f,  2300.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    { -3584.0f,  2484.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {  3584.0f,  2484.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {     0.0f,  2816.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {  -940.0f,  3308.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {   940.0f,  3308.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    { -1792.0f,  4184.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {  1792.0f,  4184.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false },
    {     0.0f,  4240.0f, 70.0f, 144.0f, 20736.0f,  12.0f,  4.0f, false }
};

// Host POD structure for Arena episode state
struct ArenaStatePOD {
    uint8_t is_goal = 0;
    uint8_t scoring_team = 0;
    uint8_t is_out_of_bounds = 0;
    uint32_t tick_count = 0;
    uint8_t terminated = 0;
    uint8_t truncated = 0;
    float rewards[MAX_CARS_PER_ENV] = {0.0f};
    uint8_t pad_is_active[MAX_BOOST_PADS] = {0};
    float pad_cooldown[MAX_BOOST_PADS] = {0.0f};
};

// Device SoA structure for Arena & Boost pad states across N environments
struct ArenaStateSoA {
    // Episode / Termination flags (length = num_envs, aligned to 128 bytes)
    uint8_t* __restrict__ is_goal = nullptr;
    uint8_t* __restrict__ scoring_team = nullptr;
    uint8_t* __restrict__ is_out_of_bounds = nullptr;
    uint32_t* __restrict__ tick_count = nullptr;
    uint8_t* __restrict__ terminated = nullptr;
    uint8_t* __restrict__ truncated = nullptr;

    // Episode rewards (length = num_envs * cars_per_env, aligned to 128 bytes)
    float* __restrict__ rewards = nullptr;

    // Boost pad states (total elements = num_envs * MAX_BOOST_PADS, aligned to 128 bytes)
    uint8_t* __restrict__ pad_is_active = nullptr;
    float* __restrict__ pad_cooldown = nullptr;

    __device__ inline uint8_t get_pad_active(uint32_t env_idx, uint32_t pad_idx) const {
        return pad_is_active[env_idx * MAX_BOOST_PADS + pad_idx];
    }

    __device__ inline void set_pad_active(uint32_t env_idx, uint32_t pad_idx, uint8_t active) {
        pad_is_active[env_idx * MAX_BOOST_PADS + pad_idx] = active;
    }

    __device__ inline float get_pad_cooldown(uint32_t env_idx, uint32_t pad_idx) const {
        return pad_cooldown[env_idx * MAX_BOOST_PADS + pad_idx];
    }

    __device__ inline void set_pad_cooldown(uint32_t env_idx, uint32_t pad_idx, float cd) {
        pad_cooldown[env_idx * MAX_BOOST_PADS + pad_idx] = cd;
    }

    __device__ inline ArenaStatePOD get_pod(uint32_t env_idx) const {
        ArenaStatePOD pod;
        pod.is_goal = is_goal[env_idx];
        pod.scoring_team = scoring_team[env_idx];
        pod.is_out_of_bounds = is_out_of_bounds[env_idx];
        pod.tick_count = tick_count[env_idx];
        pod.terminated = terminated ? terminated[env_idx] : 0;
        pod.truncated = truncated ? truncated[env_idx] : 0;
        for (uint32_t i = 0; i < MAX_BOOST_PADS; ++i) {
            pod.pad_is_active[i] = pad_is_active[env_idx * MAX_BOOST_PADS + i];
            pod.pad_cooldown[i] = pad_cooldown[env_idx * MAX_BOOST_PADS + i];
        }
        return pod;
    }

    __device__ inline void set_pod(uint32_t env_idx, const ArenaStatePOD& pod) {
        is_goal[env_idx] = pod.is_goal;
        scoring_team[env_idx] = pod.scoring_team;
        is_out_of_bounds[env_idx] = pod.is_out_of_bounds;
        tick_count[env_idx] = pod.tick_count;
        if (terminated) terminated[env_idx] = pod.terminated;
        if (truncated) truncated[env_idx] = pod.truncated;
        for (uint32_t i = 0; i < MAX_BOOST_PADS; ++i) {
            pad_is_active[env_idx * MAX_BOOST_PADS + i] = pod.pad_is_active[i];
            pad_cooldown[env_idx * MAX_BOOST_PADS + i] = pod.pad_cooldown[i];
        }
    }
};

} // namespace rocketsim_cuda
