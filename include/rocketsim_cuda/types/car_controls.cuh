#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace rocketsim_cuda {

// Host/Device single car input representation (24 bytes)
struct alignas(4) CarControls {
    float throttle = 0.0f; // [-1.0, 1.0]
    float steer    = 0.0f; // [-1.0, 1.0]
    float pitch    = 0.0f; // [-1.0, 1.0]
    float yaw      = 0.0f; // [-1.0, 1.0]
    float roll     = 0.0f; // [-1.0, 1.0]
    uint8_t boost     = 0;
    uint8_t jump      = 0;
    uint8_t handbrake = 0;
    uint8_t padding   = 0;

    __host__ __device__ constexpr void clamp_fix() {
        throttle  = (throttle > 1.0f) ? 1.0f : ((throttle < -1.0f) ? -1.0f : throttle);
        steer     = (steer > 1.0f) ? 1.0f : ((steer < -1.0f) ? -1.0f : steer);
        pitch     = (pitch > 1.0f) ? 1.0f : ((pitch < -1.0f) ? -1.0f : pitch);
        yaw       = (yaw > 1.0f) ? 1.0f : ((yaw < -1.0f) ? -1.0f : yaw);
        roll      = (roll > 1.0f) ? 1.0f : ((roll < -1.0f) ? -1.0f : roll);
    }
};

// Device SoA view for N concurrent cars
struct CarControlsSoA {
    float* __restrict__ throttle = nullptr;
    float* __restrict__ steer    = nullptr;
    float* __restrict__ pitch    = nullptr;
    float* __restrict__ yaw      = nullptr;
    float* __restrict__ roll     = nullptr;
    uint8_t* __restrict__ boost     = nullptr;
    uint8_t* __restrict__ jump      = nullptr;
    uint8_t* __restrict__ handbrake = nullptr;

    __device__ inline CarControls get(uint32_t idx) const {
        CarControls c;
        c.throttle  = throttle[idx];
        c.steer     = steer[idx];
        c.pitch     = pitch[idx];
        c.yaw       = yaw[idx];
        c.roll      = roll[idx];
        c.boost     = boost[idx];
        c.jump      = jump[idx];
        c.handbrake = handbrake[idx];
        c.padding   = 0;
        return c;
    }

    __device__ inline void set(uint32_t idx, const CarControls& c) {
        throttle[idx]  = c.throttle;
        steer[idx]     = c.steer;
        pitch[idx]     = c.pitch;
        yaw[idx]       = c.yaw;
        roll[idx]      = c.roll;
        boost[idx]     = c.boost;
        jump[idx]      = c.jump;
        handbrake[idx] = c.handbrake;
    }
};

} // namespace rocketsim_cuda
