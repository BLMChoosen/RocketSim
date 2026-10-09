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

class Ablation1DiscreteThrottleSteer : public IScenario {
public:
    std::string GetName() const override { return "ablation_1_discrete_throttle_steer"; }
    std::string GetDescription() const override { return "Cumulative 1: Discrete throttle and steer (-1, 0, 1)"; }
    uint32_t GetDefaultCars() const override { return 1; }
    uint32_t GetDefaultTicks() const override { return 60; }
    void ApplyInitialState(CPURefSim&, uint32_t) const override {}
    CarControls GetControl(uint32_t, uint32_t, uint32_t, DeterministicInputGenerator& gen) const override {
        CarControls c{};
        float th = gen.NextFloatSigned();
        c.throttle = (th > 0.33f) ? 1.0f : ((th < -0.33f) ? -1.0f : 0.0f);
        float st = gen.NextFloatSigned();
        c.steer = (st > 0.33f) ? 1.0f : ((st < -0.33f) ? -1.0f : 0.0f);
        return c;
    }
};

class Ablation2AnalogThrottleSteer : public IScenario {
public:
    std::string GetName() const override { return "ablation_2_analog_throttle_steer"; }
    std::string GetDescription() const override { return "Cumulative 2: Analog throttle and steer in [-1, 1]"; }
    uint32_t GetDefaultCars() const override { return 1; }
    uint32_t GetDefaultTicks() const override { return 60; }
    void ApplyInitialState(CPURefSim&, uint32_t) const override {}
    CarControls GetControl(uint32_t, uint32_t, uint32_t, DeterministicInputGenerator& gen) const override {
        CarControls c{};
        c.throttle = gen.NextFloatSigned();
        c.steer = gen.NextFloatSigned();
        return c;
    }
};

class Ablation3PlusBoost : public IScenario {
public:
    std::string GetName() const override { return "ablation_3_plus_boost"; }
    std::string GetDescription() const override { return "Cumulative 3: Analog throttle/steer + boost"; }
    uint32_t GetDefaultCars() const override { return 1; }
    uint32_t GetDefaultTicks() const override { return 60; }
    void ApplyInitialState(CPURefSim&, uint32_t) const override {}
    CarControls GetControl(uint32_t, uint32_t, uint32_t, DeterministicInputGenerator& gen) const override {
        CarControls c{};
        c.throttle = gen.NextFloatSigned();
        c.steer = gen.NextFloatSigned();
        c.boost = (gen.NextU32() % 100 < 30) ? 1 : 0;
        return c;
    }
};

class Ablation4PlusHandbrake : public IScenario {
public:
    std::string GetName() const override { return "ablation_4_plus_handbrake"; }
    std::string GetDescription() const override { return "Cumulative 4: + handbrake"; }
    uint32_t GetDefaultCars() const override { return 1; }
    uint32_t GetDefaultTicks() const override { return 60; }
    void ApplyInitialState(CPURefSim&, uint32_t) const override {}
    CarControls GetControl(uint32_t, uint32_t, uint32_t, DeterministicInputGenerator& gen) const override {
        CarControls c{};
        c.throttle = gen.NextFloatSigned();
        c.steer = gen.NextFloatSigned();
        c.boost = (gen.NextU32() % 100 < 30) ? 1 : 0;
        c.handbrake = (gen.NextU32() % 100 < 10) ? 1 : 0;
        return c;
    }
};

class Ablation5PlusSingleJump : public IScenario {
public:
    std::string GetName() const override { return "ablation_5_plus_single_jump"; }
    std::string GetDescription() const override { return "Cumulative 5: + single jump"; }
    uint32_t GetDefaultCars() const override { return 1; }
    uint32_t GetDefaultTicks() const override { return 60; }
    void ApplyInitialState(CPURefSim&, uint32_t) const override {}
    CarControls GetControl(uint32_t tick, uint32_t, uint32_t, DeterministicInputGenerator& gen) const override {
        CarControls c{};
        c.throttle = gen.NextFloatSigned();
        c.steer = gen.NextFloatSigned();
        c.boost = (gen.NextU32() % 100 < 30) ? 1 : 0;
        c.handbrake = (gen.NextU32() % 100 < 10) ? 1 : 0;
        c.jump = (tick >= 10 && tick < 20) ? 1 : 0;
        return c;
    }
};

class Ablation6PlusDoubleJump : public IScenario {
public:
    std::string GetName() const override { return "ablation_6_plus_double_jump"; }
    std::string GetDescription() const override { return "Cumulative 6: + pure vertical double jump"; }
    uint32_t GetDefaultCars() const override { return 1; }
    uint32_t GetDefaultTicks() const override { return 60; }
    void ApplyInitialState(CPURefSim&, uint32_t) const override {}
    CarControls GetControl(uint32_t tick, uint32_t, uint32_t, DeterministicInputGenerator& gen) const override {
        CarControls c{};
        c.throttle = gen.NextFloatSigned();
        c.steer = gen.NextFloatSigned();
        c.boost = (gen.NextU32() % 100 < 30) ? 1 : 0;
        c.handbrake = (gen.NextU32() % 100 < 10) ? 1 : 0;
        c.jump = ((tick >= 10 && tick < 15) || (tick >= 25 && tick < 30)) ? 1 : 0;
        return c;
    }
};

class Ablation7PlusDodgeFlip : public IScenario {
public:
    std::string GetName() const override { return "ablation_7_plus_dodge_flip"; }
    std::string GetDescription() const override { return "Cumulative 7: + dodge/flip with partial directional inputs"; }
    uint32_t GetDefaultCars() const override { return 1; }
    uint32_t GetDefaultTicks() const override { return 60; }
    void ApplyInitialState(CPURefSim&, uint32_t) const override {}
    CarControls GetControl(uint32_t tick, uint32_t, uint32_t, DeterministicInputGenerator& gen) const override {
        CarControls c{};
        c.throttle = gen.NextFloatSigned();
        c.steer = gen.NextFloatSigned();
        c.boost = (gen.NextU32() % 100 < 30) ? 1 : 0;
        c.handbrake = (gen.NextU32() % 100 < 10) ? 1 : 0;
        bool jump1 = (tick >= 10 && tick < 15);
        bool jump2 = (tick >= 25 && tick < 30);
        c.jump = (jump1 || jump2) ? 1 : 0;
        if (jump2) {
            c.pitch = gen.NextFloatSigned();
            c.yaw = gen.NextFloatSigned();
        }
        return c;
    }
};

class Ablation8PlusAirControl : public IScenario {
public:
    std::string GetName() const override { return "ablation_8_plus_air_control"; }
    std::string GetDescription() const override { return "Cumulative 8: + continuous air control (pitch/yaw/roll)"; }
    uint32_t GetDefaultCars() const override { return 1; }
    uint32_t GetDefaultTicks() const override { return 60; }
    void ApplyInitialState(CPURefSim&, uint32_t) const override {}
    CarControls GetControl(uint32_t tick, uint32_t, uint32_t, DeterministicInputGenerator& gen) const override {
        CarControls c{};
        c.throttle = gen.NextFloatSigned();
        c.steer = gen.NextFloatSigned();
        c.boost = (gen.NextU32() % 100 < 30) ? 1 : 0;
        c.handbrake = (gen.NextU32() % 100 < 10) ? 1 : 0;
        bool jump1 = (tick >= 10 && tick < 15);
        bool jump2 = (tick >= 25 && tick < 30);
        c.jump = (jump1 || jump2) ? 1 : 0;
        if (jump2) {
            c.pitch = gen.NextFloatSigned();
            c.yaw = gen.NextFloatSigned();
        }
        if (tick >= 30) {
            c.pitch = gen.NextFloatSigned();
            c.yaw = gen.NextFloatSigned();
            c.roll = gen.NextFloatSigned();
        }
        return c;
    }
};

class Ablation9PlusLandings : public IScenario {
public:
    std::string GetName() const override { return "ablation_9_plus_landings"; }
    std::string GetDescription() const override { return "Cumulative 9: Airborne car landing and settling on suspension"; }
    uint32_t GetDefaultCars() const override { return 1; }
    uint32_t GetDefaultTicks() const override { return 60; }
    void ApplyInitialState(CPURefSim& env, uint32_t env_idx) const override {
        CarStatePOD c{};
        c.pos = Vec3(0.0f, 2000.0f, 150.0f);
        float speed_x = (static_cast<float>(env_idx % 100) - 50.0f) * 10.0f;
        c.vel = Vec3(speed_x, 0.0f, -100.0f);
        c.quat = Quat::identity();
        c.is_on_ground = 0;
        env.SetCarState(0, c);
    }
    CarControls GetControl(uint32_t, uint32_t, uint32_t, DeterministicInputGenerator& gen) const override {
        CarControls c{};
        c.throttle = gen.NextFloatSigned();
        c.steer = gen.NextFloatSigned();
        c.boost = (gen.NextU32() % 100 < 30) ? 1 : 0;
        c.handbrake = (gen.NextU32() % 100 < 10) ? 1 : 0;
        return c;
    }
};

class Ablation10RandomFull : public IScenario {
public:
    std::string GetName() const override { return "ablation_10_random_full"; }
    std::string GetDescription() const override { return "Cumulative 10: Full random PCG32 control generator"; }
    uint32_t GetDefaultCars() const override { return 1; }
    uint32_t GetDefaultTicks() const override { return 60; }
    void ApplyInitialState(CPURefSim&, uint32_t) const override {}
    CarControls GetControl(uint32_t, uint32_t, uint32_t, DeterministicInputGenerator& gen) const override {
        return gen.Generate();
    }
};

} // namespace

void RegisterAblationScenarios() {
    auto& reg = ScenarioRegistry::Instance();
    reg.Register(std::make_shared<Ablation5FlipsScenario>());
    reg.Register(std::make_shared<Ablation1DiscreteThrottleSteer>());
    reg.Register(std::make_shared<Ablation2AnalogThrottleSteer>());
    reg.Register(std::make_shared<Ablation3PlusBoost>());
    reg.Register(std::make_shared<Ablation4PlusHandbrake>());
    reg.Register(std::make_shared<Ablation5PlusSingleJump>());
    reg.Register(std::make_shared<Ablation6PlusDoubleJump>());
    reg.Register(std::make_shared<Ablation7PlusDodgeFlip>());
    reg.Register(std::make_shared<Ablation8PlusAirControl>());
    reg.Register(std::make_shared<Ablation9PlusLandings>());
    reg.Register(std::make_shared<Ablation10RandomFull>());
    reg.RegisterAlias("ablation_1_idle", "idle");
    reg.RegisterAlias("ablation_1_freefall", "freefall");
    reg.RegisterAlias("ablation_2_ball_bounces", "ball_floor_drop");
    reg.RegisterAlias("ablation_3_throttle", "throttle");
    reg.RegisterAlias("ablation_3_boost", "boost");
    reg.RegisterAlias("ablation_4_car_ball_hit", "car_ball_hit");
    reg.RegisterAlias("ablation_4_kickoff_goalie", "kickoff_goalie");
}

} // namespace rocketsim_cuda
