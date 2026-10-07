#include "scenario_registry.h"

namespace rocketsim_cuda {

namespace {

class Ablation5FlipsScenario : public IScenario {
public:
    std::string GetName() const override { return "ablation_5_flips"; }
    std::string GetDescription() const override { return "Canonical flip directions (8-way, stall, cancel) across environments"; }
    uint32_t GetDefaultTicks() const override { return 120; }

    void ApplyInitialState(CPURefSim& /*env*/, uint32_t /*env_idx*/) const override {
        // Default spawn
    }

    CarControls GetControl(uint32_t tick, uint32_t env, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = 1.0f;
        uint32_t mode = env % 11;
        if (tick >= 10 && tick < 15) {
            c.jump = 1;
        } else if (tick >= 25 && tick < 30) {
            c.jump = 1;
            switch (mode) {
                case 0: // Front flip cancel (initiate front flip)
                case 1: // Front flip (pure)
                    c.pitch = -1.0f;
                    break;
                case 2: // Back flip (pure)
                case 9: // Back flip cancel (initiate back flip)
                    c.pitch = 1.0f;
                    break;
                case 3: // Left dodge
                    c.yaw = -1.0f;
                    break;
                case 4: // Right dodge
                    c.yaw = 1.0f;
                    break;
                case 5: // Diagonal front-left
                    c.pitch = -1.0f;
                    c.yaw = -1.0f;
                    break;
                case 6: // Diagonal front-right
                    c.pitch = -1.0f;
                    c.yaw = 1.0f;
                    break;
                case 7: // Diagonal back-left
                    c.pitch = 1.0f;
                    c.yaw = -1.0f;
                    break;
                case 8: // Diagonal back-right
                    c.pitch = 1.0f;
                    c.yaw = 1.0f;
                    break;
                case 10: // Stall: equal and opposite yaw and roll
                    c.pitch = 0.0f;
                    c.yaw = 1.0f;
                    c.roll = -1.0f;
                    break;
            }
        } else if (tick >= 30) {
            if (mode == 0) {
                // Front flip cancel: counter-pitch
                c.pitch = 1.0f;
                c.roll = 0.5f;
            } else if (mode == 9) {
                // Back flip cancel: counter-pitch
                c.pitch = -1.0f;
                c.roll = 0.5f;
            } else if (mode == 10) {
                // Maintain stall roll/yaw
                c.yaw = 1.0f;
                c.roll = -1.0f;
            }
        }
        return c;
    }
};

} // namespace

void RegisterAblationScenarios() {
    auto& reg = ScenarioRegistry::Instance();
    reg.Register(std::make_shared<Ablation5FlipsScenario>());
    reg.RegisterAlias("ablation_1_idle", "idle");
    reg.RegisterAlias("ablation_1_freefall", "freefall");
    reg.RegisterAlias("ablation_2_ball_bounces", "ball_floor_drop");
    reg.RegisterAlias("ablation_3_throttle", "throttle");
    reg.RegisterAlias("ablation_3_boost", "boost");
    reg.RegisterAlias("ablation_4_car_ball_hit", "car_ball_hit");
    reg.RegisterAlias("ablation_4_kickoff_goalie", "kickoff_goalie");
}

} // namespace rocketsim_cuda
