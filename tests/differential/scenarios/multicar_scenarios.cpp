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

class CarCarFrontScenario : public IScenario {
public:
    std::string GetName() const override { return "car_car_front"; }
    std::string GetDescription() const override { return "Head-on car-car collision along Y axis with throttle and boost"; }
    uint32_t GetDefaultCars() const override { return 2; }
    uint32_t GetDefaultTicks() const override { return 120; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c0{};
        c0.pos = Vec3(0.0f, -1200.0f, 17.0f);
        c0.quat = Quat(0.7071068f, 0.0f, 0.0f, 0.7071068f); // Facing +Y (Blue)
        c0.vel = Vec3(0.0f, 0.0f, 0.0f);
        c0.is_on_ground = 1;
        c0.team = 0;
        env.SetCarState(0, c0);

        CarStatePOD c1{};
        c1.pos = Vec3(0.0f, 1200.0f, 17.0f);
        c1.quat = Quat(0.7071068f, 0.0f, 0.0f, -0.7071068f); // Facing -Y (Orange)
        c1.vel = Vec3(0.0f, 0.0f, 0.0f);
        c1.is_on_ground = 1;
        c1.team = 1;
        env.SetCarState(1, c1);
    }

    CarControls GetControl(uint32_t tick, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = 1.0f;
        c.boost = 1;
        return c;
    }
};

class CarCarSideScenario : public IScenario {
public:
    std::string GetName() const override { return "car_car_side"; }
    std::string GetDescription() const override { return "Perpendicular T-bone car-car collision (X and Y trajectories)"; }
    uint32_t GetDefaultCars() const override { return 2; }
    uint32_t GetDefaultTicks() const override { return 120; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c0{};
        c0.pos = Vec3(0.0f, -400.0f, 17.0f);
        c0.vel = Vec3(0.0f, 1000.0f, 0.0f);
        c0.quat = Quat(0.7071068f, 0.0f, 0.0f, 0.7071068f); // Facing +Y
        c0.is_on_ground = 1;
        c0.team = 0;
        env.SetCarState(0, c0);

        CarStatePOD c1{};
        c1.pos = Vec3(-1000.0f, 0.0f, 17.0f);
        c1.vel = Vec3(1400.0f, 0.0f, 0.0f);
        c1.quat = Quat::identity(); // Facing +X
        c1.is_on_ground = 1;
        c1.team = 1;
        env.SetCarState(1, c1);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = 1.0f;
        return c;
    }
};

class CarCarRearScenario : public IScenario {
public:
    std::string GetName() const override { return "car_car_rear"; }
    std::string GetDescription() const override { return "Rear-end car collision (fast car bumping slower car from behind)"; }
    uint32_t GetDefaultCars() const override { return 2; }
    uint32_t GetDefaultTicks() const override { return 120; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c0{};
        c0.pos = Vec3(0.0f, -500.0f, 17.0f);
        c0.vel = Vec3(0.0f, 1400.0f, 0.0f);
        c0.quat = Quat(0.7071068f, 0.0f, 0.0f, 0.7071068f); // Facing +Y
        c0.is_on_ground = 1;
        c0.team = 0;
        env.SetCarState(0, c0);

        CarStatePOD c1{};
        c1.pos = Vec3(0.0f, 0.0f, 17.0f);
        c1.vel = Vec3(0.0f, 400.0f, 0.0f);
        c1.quat = Quat(0.7071068f, 0.0f, 0.0f, 0.7071068f); // Facing +Y
        c1.is_on_ground = 1;
        c1.team = 1;
        env.SetCarState(1, c1);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t car_idx, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = (car_idx == 0) ? 1.0f : 0.5f;
        c.boost = (car_idx == 0) ? 1 : 0;
        return c;
    }
};

class CarCarAirScenario : public IScenario {
public:
    std::string GetName() const override { return "car_car_air"; }
    std::string GetDescription() const override { return "Mid-air airborne collision between two cars"; }
    uint32_t GetDefaultCars() const override { return 2; }
    uint32_t GetDefaultTicks() const override { return 120; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c0{};
        c0.pos = Vec3(-600.0f, 0.0f, 600.0f);
        c0.vel = Vec3(1000.0f, 0.0f, 0.0f);
        c0.quat = Quat::identity(); // Facing +X
        c0.is_on_ground = 0;
        c0.team = 0;
        env.SetCarState(0, c0);

        CarStatePOD c1{};
        c1.pos = Vec3(600.0f, 0.0f, 600.0f);
        c1.vel = Vec3(-1000.0f, 0.0f, 0.0f);
        c1.quat = Quat(0.0f, 0.0f, 0.0f, 1.0f); // Facing -X (180 deg yaw)
        c1.is_on_ground = 0;
        c1.team = 1;
        env.SetCarState(1, c1);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class CarOnCarScenario : public IScenario {
public:
    std::string GetName() const override { return "car_on_car"; }
    std::string GetDescription() const override { return "Car dropped vertically onto the roof of another resting car"; }
    uint32_t GetDefaultCars() const override { return 2; }
    uint32_t GetDefaultTicks() const override { return 120; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c0{};
        c0.pos = Vec3(0.0f, 0.0f, 17.0f);
        c0.vel = Vec3(0.0f, 0.0f, 0.0f);
        c0.quat = Quat::identity();
        c0.is_on_ground = 1;
        c0.team = 0;
        env.SetCarState(0, c0);

        CarStatePOD c1{};
        c1.pos = Vec3(0.0f, 0.0f, 120.0f);
        c1.vel = Vec3(0.0f, 0.0f, -200.0f);
        c1.quat = Quat::identity();
        c1.is_on_ground = 0;
        c1.team = 1;
        env.SetCarState(1, c1);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class WheelsOnBallScenario : public IScenario {
public:
    std::string GetName() const override { return "wheels_on_ball"; }
    std::string GetDescription() const override { return "Car dropped onto ball with 4 wheels raycasting and contacting the ball"; }
    uint32_t GetDefaultCars() const override { return 1; }
    uint32_t GetDefaultTicks() const override { return 120; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        BallStatePOD b{};
        b.pos = Vec3(0.0f, 0.0f, 93.15f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);

        CarStatePOD c{};
        c.pos = Vec3(0.0f, 0.0f, 150.0f);
        c.vel = Vec3(0.0f, 0.0f, -50.0f);
        c.quat = Quat::identity();
        c.is_on_ground = 0;
        env.SetCarState(0, c);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class WheelsOnCarScenario : public IScenario {
public:
    std::string GetName() const override { return "wheels_on_car"; }
    std::string GetDescription() const override { return "Car positioned above another resting car with wheels raycasting onto the chassis roof"; }
    uint32_t GetDefaultCars() const override { return 2; }
    uint32_t GetDefaultTicks() const override { return 120; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c0{};
        c0.pos = Vec3(0.0f, 0.0f, 17.0f);
        c0.vel = Vec3(0.0f, 0.0f, 0.0f);
        c0.quat = Quat::identity();
        c0.is_on_ground = 1;
        c0.team = 0;
        env.SetCarState(0, c0);

        CarStatePOD c1{};
        c1.pos = Vec3(0.0f, 0.0f, 65.0f);
        c1.vel = Vec3(0.0f, 0.0f, -20.0f);
        c1.quat = Quat::identity();
        c1.is_on_ground = 0;
        c1.team = 1;
        env.SetCarState(1, c1);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

} // namespace

void RegisterMultiCarScenarios() {
    auto& reg = ScenarioRegistry::Instance();
    reg.Register(std::make_shared<KickoffMultiCarScenario>());
    reg.Register(std::make_shared<CarCarFrontScenario>());
    reg.Register(std::make_shared<CarCarSideScenario>());
    reg.Register(std::make_shared<CarCarRearScenario>());
    reg.Register(std::make_shared<CarCarAirScenario>());
    reg.Register(std::make_shared<CarOnCarScenario>());
    reg.Register(std::make_shared<WheelsOnBallScenario>());
    reg.Register(std::make_shared<WheelsOnCarScenario>());
}

} // namespace rocketsim_cuda
