#ifndef NOMINMAX
#define NOMINMAX
#endif

#pragma once
#include <vector>
#include <cstdint>
#include "rocketsim_cuda/types/car_controls.cuh"
#include "rocketsim_cuda/types/ball_state.cuh"
#include "rocketsim_cuda/types/car_state.cuh"

namespace RocketSim {
    class Arena;
    class Car;
}

namespace rocketsim_cuda {

class CPURefSim {
public:
    explicit CPURefSim(int numCars = 1, bool addFloor = true, float tickRate = 120.0f, int spawnSeed = 0);
    ~CPURefSim();

    // Disable copy
    CPURefSim(const CPURefSim&) = delete;
    CPURefSim& operator=(const CPURefSim&) = delete;

    // Enable move
    CPURefSim(CPURefSim&& other) noexcept;
    CPURefSim& operator=(CPURefSim&& other) noexcept;

    void Step(const CarControls* controls = nullptr, int numCars = 1);
    void GetBallState(BallStatePOD& out) const;
    void GetCarState(int carIdx, CarStatePOD& out) const;
    void SetBallState(const BallStatePOD& in);
    void SetCarState(int carIdx, const CarStatePOD& in);
    void Reset();

    int GetNumCars() const { return m_numCars; }
    uint64_t GetTickCount() const;

private:
    void InitArena();
    void CleanupArena();

    RocketSim::Arena* m_arena = nullptr;
    std::vector<RocketSim::Car*> m_cars;
    int m_numCars = 1;
    bool m_addFloor = true;
    float m_tickRate = 120.0f;
    int m_spawnSeed = 0;
};

} // namespace rocketsim_cuda
