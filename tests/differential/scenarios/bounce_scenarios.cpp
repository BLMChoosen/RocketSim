#include "scenario_registry.h"

namespace rocketsim_cuda {

namespace {

class BallFloorDropScenario : public IScenario {
public:
    std::string GetName() const override { return "ball_floor_drop"; }
    std::string GetDescription() const override { return "Ball freefall drop onto arena floor"; }
    bool IsBallBounce() const override { return true; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, -4608.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        c.boost = 0.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 250.0f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class BallFloorAngledScenario : public IScenario {
public:
    std::string GetName() const override { return "ball_floor_angled"; }
    std::string GetDescription() const override { return "Ball impact on floor with tangential velocity and angular spin"; }
    bool IsBallBounce() const override { return true; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, -4608.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        c.boost = 0.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 250.0f);
        b.vel = Vec3(500.0f, 0.0f, -300.0f);
        b.ang_vel = Vec3(0.0f, 3.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class BallSideWallScenario : public IScenario {
public:
    std::string GetName() const override { return "ball_side_wall"; }
    std::string GetDescription() const override { return "High-speed ball rebound off arena side wall"; }
    bool IsBallBounce() const override { return true; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, -4608.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        c.boost = 0.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(3500.0f, 0.0f, 500.0f);
        b.vel = Vec3(1500.0f, 0.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class BallBackWallScenario : public IScenario {
public:
    std::string GetName() const override { return "ball_back_wall"; }
    std::string GetDescription() const override { return "High-speed ball rebound off arena back wall"; }
    bool IsBallBounce() const override { return true; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, -4608.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        c.boost = 0.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(2000.0f, 4500.0f, 500.0f);
        b.vel = Vec3(0.0f, 1500.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class BallCeilingScenario : public IScenario {
public:
    std::string GetName() const override { return "ball_ceiling"; }
    std::string GetDescription() const override { return "Vertical upward ball impact against arena ceiling"; }
    bool IsBallBounce() const override { return true; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, -4608.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        c.boost = 0.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 1600.0f);
        b.vel = Vec3(0.0f, 0.0f, 1200.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class BallCornerRampScenario : public IScenario {
public:
    std::string GetName() const override { return "ball_corner_ramp"; }
    std::string GetDescription() const override { return "Ball trajectory hitting corner chamfer/fillet ramp"; }
    bool IsBallBounce() const override { return true; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, -4608.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        c.boost = 0.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(3450.0f, 4250.0f, 200.0f);
        b.vel = Vec3(600.0f, 600.0f, -200.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class BallGoalPostScenario : public IScenario {
public:
    std::string GetName() const override { return "ball_goal_post"; }
    std::string GetDescription() const override { return "Ball collision against vertical cylindrical goalpost"; }
    bool IsBallBounce() const override { return true; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, -4608.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        c.boost = 0.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(892.8f, 4500.0f, 300.0f);
        b.vel = Vec3(0.0f, 1500.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class BallCrossbarScenario : public IScenario {
public:
    std::string GetName() const override { return "ball_crossbar"; }
    std::string GetDescription() const override { return "Ball collision against horizontal goal crossbar"; }
    bool IsBallBounce() const override { return true; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, -4608.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        c.boost = 0.0f;
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 4500.0f, 642.7f);
        b.vel = Vec3(0.0f, 1500.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class BallFlightScenario : public IScenario {
public:
    std::string GetName() const override { return "ball_flight"; }
    std::string GetDescription() const override { return "Long-range parabolic ball trajectory in free flight"; }
    bool IsBallBounce() const override { return false; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(0.0f, 0.0f, 200.0f);
        b.vel = Vec3(1500.0f, 2000.0f, 1000.0f);
        b.ang_vel = Vec3(2.0f, 3.0f, -1.0f);
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

} // namespace

void RegisterBounceScenarios() {
    auto& reg = ScenarioRegistry::Instance();
    reg.Register(std::make_shared<BallFlightScenario>());
    reg.Register(std::make_shared<BallFloorDropScenario>());
    reg.Register(std::make_shared<BallFloorAngledScenario>());
    reg.Register(std::make_shared<BallSideWallScenario>());
    reg.Register(std::make_shared<BallBackWallScenario>());
    reg.Register(std::make_shared<BallCeilingScenario>());
    reg.Register(std::make_shared<BallCornerRampScenario>());
    reg.Register(std::make_shared<BallGoalPostScenario>());
    reg.Register(std::make_shared<BallCrossbarScenario>());
}

} // namespace rocketsim_cuda
