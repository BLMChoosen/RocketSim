#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include "config.h"
#include "types/car_state.cuh"
#include "types/ball_state.cuh"
#include "types/car_controls.cuh"
#include "types/arena_state.cuh"

namespace rocketsim_cuda {

class SimContext {
public:
    explicit SimContext(uint32_t num_envs = 1, uint32_t cars_per_env = 1, cudaStream_t stream = nullptr);
    ~SimContext();

    // Disable copy, allow move
    SimContext(const SimContext&) = delete;
    SimContext& operator=(const SimContext&) = delete;
    SimContext(SimContext&& o) noexcept;
    SimContext& operator=(SimContext&& o) noexcept;

    uint32_t GetNumEnvs() const { return m_num_envs; }
    uint32_t GetCarsPerEnv() const { return m_cars_per_env; }
    uint32_t GetTotalCars() const { return m_total_cars; }
    size_t GetAllocatedBytes() const { return m_allocated_bytes; }
    cudaStream_t GetStream() const { return m_stream; }

    size_t GetBallPitchFloats() const;
    size_t GetCarPitchFloats() const;

    const BallStateSoA& GetBallState() const { return m_ball_state; }
    BallStateSoA& GetBallState() { return m_ball_state; }

    const CarStateSoA& GetCarState() const { return m_car_state; }
    CarStateSoA& GetCarState() { return m_car_state; }

    const ArenaStateSoA& GetArenaState() const { return m_arena_state; }
    ArenaStateSoA& GetArenaState() { return m_arena_state; }

    const CarControlsSoA& GetControls() const { return m_controls; }
    CarControlsSoA& GetControls() { return m_controls; }

    // State initialization kernels
    void ResetToDefault();

    // Asynchronous GPU selective resets (no host synchronization barriers)
    void ResetEnvironmentsIndexed(const int32_t* d_env_indices, uint32_t num_resets);
    void ResetEnvironmentsIndexed(const int64_t* d_env_indices, uint32_t num_resets);
    void ResetEnvironmentsMasked(const uint8_t* d_reset_mask);

    // State transfer helpers for differential testing & serialization
    void CopyBallStateToHost(BallStatePOD* host_out, uint32_t env_start = 0, uint32_t count = 0);
    void CopyCarStateToHost(CarStatePOD* host_out, uint32_t car_start = 0, uint32_t count = 0);
    void CopyBallStateToDevice(const BallStatePOD* host_in, uint32_t env_start = 0, uint32_t count = 0);
    void CopyCarStateToDevice(const CarStatePOD* host_in, uint32_t car_start = 0, uint32_t count = 0);
    void CopyControlsToDevice(const CarControls* host_in, uint32_t car_start = 0, uint32_t count = 0);

    // Simulation Step
    void Step(uint32_t batch_size = 0, const float* actions_tensor = nullptr);

private:
    void AllocateArena();
    void FreeArena();

    uint32_t m_num_envs = 0;
    uint32_t m_cars_per_env = 0;
    uint32_t m_total_cars = 0;
    size_t m_allocated_bytes = 0;
    void* m_d_pool = nullptr;
    cudaStream_t m_stream = nullptr;

    BallStateSoA m_ball_state;
    CarStateSoA m_car_state;
    ArenaStateSoA m_arena_state;
    CarControlsSoA m_controls;
};

// Global Simulation Step Entry Point (GEMINI.md Section 6.2)
void sim_step_batch(SimContext* ctx, uint32_t batch_size = 0, const CarControlsSoA* controls = nullptr, const float* actions_tensor = nullptr);

} // namespace rocketsim_cuda
