#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include "config.h"
#include "types/car_state.cuh"
#include "types/ball_state.cuh"
#include "types/car_controls.cuh"

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

    const BallStateSoA& GetBallState() const { return m_ball_state; }
    BallStateSoA& GetBallState() { return m_ball_state; }

    const CarStateSoA& GetCarState() const { return m_car_state; }
    CarStateSoA& GetCarState() { return m_car_state; }

    const CarControlsSoA& GetControls() const { return m_controls; }
    CarControlsSoA& GetControls() { return m_controls; }

    // State initialization kernels
    void ResetToDefault();

    // State transfer helpers for differential testing & serialization
    void CopyBallStateToHost(BallStatePOD* host_out, uint32_t env_start = 0, uint32_t count = 0);
    void CopyCarStateToHost(CarStatePOD* host_out, uint32_t car_start = 0, uint32_t count = 0);
    void CopyBallStateToDevice(const BallStatePOD* host_in, uint32_t env_start = 0, uint32_t count = 0);
    void CopyCarStateToDevice(const CarStatePOD* host_in, uint32_t car_start = 0, uint32_t count = 0);
    void CopyControlsToDevice(const CarControls* host_in, uint32_t car_start = 0, uint32_t count = 0);

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
    CarControlsSoA m_controls;
};

} // namespace rocketsim_cuda
