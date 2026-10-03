#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "../config.h"

namespace rocketsim_cuda {

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
