#include "scenario_registry.h"

namespace rocketsim_cuda {

namespace {

class KickoffMultiCarScenario : public IScenario {
public:
    std::string GetName() const override { return "kickoff_multicar"; }
    std::string GetDescription() const override { return "Full 6-car canonical kickoff with blue and mirrored orange teams"; }
    uint32_t GetDefaultCars() const override { return 6; }

    void ApplyInitialState(CPURefSim& env, uint32_t env_idx) const override {
        env.ResetToRandomKickoff(static_cast<int>(env_idx));
    }

    CarControls GetControl(uint32_t tick, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = 1.0f;
        if (tick >= 10 && tick < 40) {
            c.boost = 1;
        }
        return c;
    }
};

} // namespace

void RegisterMultiCarScenarios() {
    auto& reg = ScenarioRegistry::Instance();
    reg.Register(std::make_shared<KickoffMultiCarScenario>());
}

} // namespace rocketsim_cuda
