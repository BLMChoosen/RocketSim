#include "scenario_registry.h"

namespace rocketsim_cuda {

namespace {

class IdleScenario : public IScenario {
public:
    std::string GetName() const override { return "idle"; }
    std::string GetDescription() const override { return "Car resting idly on flat ground without inputs"; }

    void ApplyInitialState(CPURefSim& /*env*/, uint32_t /*env_idx*/) const override {
        // Defaults from CPURefSim constructor (spawn on ground, zero velocity)
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class FreefallScenario : public IScenario {
public:
    std::string GetName() const override { return "freefall"; }
    std::string GetDescription() const override { return "Car and ball in pure mid-air gravitational freefall"; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(1000.0f, 0.0f, 1500.0f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
        env.SetCarState(0, c);

        BallStatePOD b;
        env.GetBallState(b);
        b.pos = Vec3(-1000.0f, 0.0f, 1500.0f);
        b.vel = Vec3(0.0f, 0.0f, 0.0f);
        b.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        b.quat = Quat::identity();
        env.SetBallState(b);
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        return CarControls{};
    }
};

class ThrottleScenario : public IScenario {
public:
    std::string GetName() const override { return "throttle"; }
    std::string GetDescription() const override { return "Continuous ground acceleration under forward throttle"; }

    void ApplyInitialState(CPURefSim& /*env*/, uint32_t /*env_idx*/) const override {
        // Default spawn
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = 1.0f;
        return c;
    }
};

class BoostScenario : public IScenario {
public:
    std::string GetName() const override { return "boost"; }
    std::string GetDescription() const override { return "Continuous forward ground boost acceleration"; }

    void ApplyInitialState(CPURefSim& /*env*/, uint32_t /*env_idx*/) const override {
        // Default spawn
    }

    CarControls GetControl(uint32_t /*tick*/, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = 1.0f;
        c.boost = 1;
        return c;
    }
};

class JumpFlipScenario : public IScenario {
public:
    std::string GetName() const override { return "jump_flip"; }
    std::string GetDescription() const override { return "Takeoff jump, front flip initiation, counter-pitch flip cancel and roll"; }

    void ApplyInitialState(CPURefSim& /*env*/, uint32_t /*env_idx*/) const override {
        // Default spawn
    }

    CarControls GetControl(uint32_t tick, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        c.throttle = 1.0f;
        if (tick >= 10 && tick < 15) {
            c.jump = 1;
        } else if (tick >= 25 && tick < 30) {
            c.jump = 1;
            c.pitch = -1.0f;
        } else if (tick >= 30) {
            c.pitch = 1.0f;
            c.roll = 0.5f;
        }
        return c;
    }
};

class CarBallHitScenario : public IScenario {
public:
    std::string GetName() const override { return "car_ball_hit"; }
    std::string GetDescription() const override { return "Straight forward high-speed impact against resting ball"; }
    bool IsGoalieHit() const override { return true; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, -1000.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat(0.7071068f, 0.0f, 0.0f, 0.7071068f);
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

class KickoffGoalieScenario : public IScenario {
public:
    std::string GetName() const override { return "kickoff_goalie"; }
    std::string GetDescription() const override { return "Full-field kickoff acceleration from goalie spawn into center ball"; }
    bool IsGoalieHit() const override { return true; }

    void ApplyInitialState(CPURefSim& env, uint32_t /*env_idx*/) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        c.pos = Vec3(0.0f, -4608.0f, 17.03f);
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat(0.7071068f, 0.0f, 0.0f, 0.7071068f);
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

class BoostPadPickupScenario : public IScenario {
public:
    std::string GetName() const override { return "boost_pad_pickup"; }
    std::string GetDescription() const override { return "Driving over big and small boost pads verifying pickup and cooldown respawn"; }
    bool IsBoostPad() const override { return true; }

    void ApplyInitialState(CPURefSim& env, uint32_t env_idx) const override {
        CarStatePOD c;
        env.GetCarState(0, c);
        if (env_idx % 2 == 0) {
            // Big Pad 0 (Midfield Left: X=-3584, Y=0, Z=73, Rad=208, +100 boost, 10s cooldown / 1201 ticks)
            c.pos = Vec3(-3644.0f, 0.0f, 17.03f);
        } else {
            // Small Pad 19 (Midfield Inner Left: X=-1024, Y=0, Z=70, Rad=144, +12 boost, 4s cooldown / 480 ticks)
            c.pos = Vec3(-1084.0f, 0.0f, 17.03f);
        }
        c.vel = Vec3(0.0f, 0.0f, 0.0f);
        c.ang_vel = Vec3(0.0f, 0.0f, 0.0f);
        c.quat = Quat::identity();
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

    CarControls GetControl(uint32_t tick, uint32_t /*env*/, uint32_t /*car_idx*/, DeterministicInputGenerator& /*gen*/) const override {
        CarControls c{};
        if (tick < 50) {
            c.throttle = 1.0f;
        }
        return c;
    }
};

} // namespace

void RegisterCarScenarios() {
    auto& reg = ScenarioRegistry::Instance();
    reg.Register(std::make_shared<IdleScenario>());
    reg.Register(std::make_shared<FreefallScenario>());
    reg.Register(std::make_shared<ThrottleScenario>());
    reg.Register(std::make_shared<BoostScenario>());
    reg.Register(std::make_shared<JumpFlipScenario>());
    reg.Register(std::make_shared<CarBallHitScenario>());
    reg.Register(std::make_shared<KickoffGoalieScenario>());
    reg.Register(std::make_shared<BoostPadPickupScenario>());
}

} // namespace rocketsim_cuda
