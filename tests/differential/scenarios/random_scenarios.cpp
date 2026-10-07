#include "scenario_registry.h"

namespace rocketsim_cuda {

namespace {

class RandomScenario : public IScenario {
public:
    std::string GetName() const override { return "random"; }
    std::string GetDescription() const override { return "Full multi-environment simulation under pseudo-random PCG32 controls"; }

    void ApplyInitialState(CPURefSim& /*env*/, uint32_t /*env_idx*/) const override {
        // Default spawn
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& gen) const override {
        return gen.Generate();
    }
};

} // namespace

void RegisterRandomScenarios() {
    auto& reg = ScenarioRegistry::Instance();
    reg.Register(std::make_shared<RandomScenario>());
}

} // namespace rocketsim_cuda
