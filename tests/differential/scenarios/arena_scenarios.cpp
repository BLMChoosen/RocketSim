#include "scenario_registry.h"

namespace rocketsim_cuda {

namespace {

class CarSideWallScenario : public IScenario {
public:
    std::string GetName() const override { return "car_side_wall"; }
    std::string GetDescription() const override { return "Car driving at supersonic speed into arena side wall"; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(3500.0f, 0.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        c.boost = 100.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 93.15f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = 1.0f;
        c.boost = 1;
        return c;
    }
};

class CarCornerRampScenario : public IScenario {
public:
    std::string GetName() const override { return "car_corner_ramp"; }
    std::string GetDescription() const override { return "Car accelerating into 45-degree corner ramp and bevel"; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(2500.0f, 3500.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat(0.9238795f, 0.0f, 0.0f, 0.3826834f);
        c.boost = 100.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 93.15f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = 1.0f;
        c.boost = 1;
        return c;
    }
};

class CarCeilingScenario : public IScenario {
public:
    std::string GetName() const override { return "car_ceiling"; }
    std::string GetDescription() const override { return "Car airborne ascent and collision against arena ceiling"; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, 0.0f, 1500.0f);
        c.vel = Vec3(0.0f, 0.0f, 1000.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat(0.7071068f, 0.0f, -0.7071068f, 0.0f);
        c.boost = 0.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 93.15f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class CarJumpWallLandScenario : public IScenario {
public:
    std::string GetName() const override { return "car_jump_wall_land"; }
    std::string GetDescription() const override { return "Car takeoff jump, wall contact, and landing recovery"; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(3000.0f, 0.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        c.boost = 100.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 93.15f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t tick, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = 1.0f;
        if (tick >= 10 && tick < 15) {
            c.jump = 1;
        }
        if (tick >= 20 && tick < 50) {
            c.boost = 1;
        }
        return c;
    }
};

class CarFloorWallTransitionScenario : public IScenario {
public:
    std::string GetName() const override { return "car_floor_wall_transition"; }
    std::string GetDescription() const override { return "Car driving through curved floor-to-wall transition fillet"; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(3500.0f, 1000.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        c.boost = 100.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 93.15f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = 1.0f;
        c.boost = 1;
        return c;
    }
};

class ConfigLowGravityScenario : public IScenario {
public:
    std::string GetName() const override { return "config_low_gravity"; }
    std::string GetDescription() const override { return "Physics simulation under half-gravity mutator setting"; }
    uint32_t GetDefaultTicks() const override { return 120; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c{};
        c.pos = Vec3(0.0f, 0.0f, 500.0f);
        c.vel = Vec3(0.0f, 0.0f, 200.0f);
        c.quat = Quat::identity();
        c.is_on_ground = 0;
        env.SetCarState(0, c);

        BallStatePOD b{};
        b.pos = Vec3(500.0f, 0.0f, 500.0f);
        b.vel = Vec3(0.0f, 0.0f, 200.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class ConfigHeavyBallScenario : public IScenario {
public:
    std::string GetName() const override { return "config_heavy_ball"; }
    std::string GetDescription() const override { return "Car bouncing heavy ball under custom mutator settings"; }
    uint32_t GetDefaultTicks() const override { return 120; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c{};
        c.pos = Vec3(0.0f, -300.0f, 17.0f);
        c.vel = Vec3(0.0f, 1000.0f, 0.0f);
        c.quat = Quat(0.7071068f, 0.0f, 0.0f, 0.7071068f);
        c.is_on_ground = 1;
        env.SetCarState(0, c);

        BallStatePOD b{};
        b.pos = Vec3(0.0f, 0.0f, 93.15f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = 1.0f;
        return c;
    }
};

} // namespace

void RegisterArenaScenarios() {
    auto& reg = ScenarioRegistry::Instance();
    reg.Register(std::make_shared<CarSideWallScenario>());
    reg.Register(std::make_shared<CarCornerRampScenario>());
    reg.Register(std::make_shared<CarCeilingScenario>());
    reg.Register(std::make_shared<CarJumpWallLandScenario>());
    reg.Register(std::make_shared<CarFloorWallTransitionScenario>());
    reg.Register(std::make_shared<ConfigLowGravityScenario>());
    reg.Register(std::make_shared<ConfigHeavyBallScenario>());
}

} // namespace rocketsim_cuda
