#pragma once
#include <string>
#include <vector>
#include <memory>

#include "rocketsim_cuda/config.h"
#include "cpu_ref_sim.h"
#include "golden_master.h"

namespace rocketsim_cuda {

class IScenario {
public:
    virtual ~IScenario() = default;

    virtual std::string GetName() const = 0;
    virtual std::string GetDescription() const { return ""; }
    virtual uint32_t GetDefaultTicks() const { return 500; }
    virtual uint32_t GetDefaultCars() const { return 1; }
    virtual bool IsBallBounce() const { return false; }
    virtual bool IsGoalieHit() const { return false; }
    virtual bool IsBoostPad() const { return false; }

    virtual void ApplyInitialState(CPURefSim& env, uint32_t env_idx) const = 0;
    virtual CarControls GetControl(uint32_t tick, uint32_t env, uint32_t car_idx, DeterministicInputGenerator& gen) const = 0;
};

} // namespace rocketsim_cuda
