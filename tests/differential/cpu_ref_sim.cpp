#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "cpu_ref_sim.h"

// RocketSim CPU & Bullet headers
#include "RocketSim.h"
#include "Sim/Arena/Arena.h"
#include "Sim/Car/Car.h"
#include "Sim/Ball/Ball.h"
#include "RLConst.h"
#include "BulletLink.h"
#include "BulletCollision/CollisionShapes/btStaticPlaneShape.h"
#include "LinearMath/btQuaternion.h"
#include "LinearMath/btMatrix3x3.h"
#include "Sim/BoostPad/BoostPad.h"

#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <mutex>

namespace rocketsim_cuda {

namespace {
    static bool s_rocketSimInitialized = false;

    void EnsureRocketSimInit() {
        if (!s_rocketSimInitialized) {
            RocketSim::Init("", true);
            s_rocketSimInitialized = true;
        }
    }

    static std::unordered_map<const CPURefSim*, RocketSim::Arena*> s_simArenaMap;
    static std::mutex s_simArenaMapMutex;
}

bool GetCPURefSimBoostPadState(const CPURefSim* sim, int padIdx, bool& isActive, float& cooldown) {
    std::lock_guard<std::mutex> lock(s_simArenaMapMutex);
    auto it = s_simArenaMap.find(sim);
    if (it != s_simArenaMap.end() && it->second) {
        auto arena = it->second;
        if (padIdx >= 0 && padIdx < static_cast<int>(arena->_boostPads.size())) {
            auto state = arena->_boostPads[padIdx]->GetState();
            isActive = state.isActive;
            cooldown = state.cooldown;
            return true;
        }
    }
    return false;
}

CPURefSim::CPURefSim(int numCars, bool addFloor, float tickRate, int spawnSeed, int hitboxType)
    : m_numCars(numCars), m_addFloor(addFloor), m_tickRate(tickRate), m_spawnSeed(spawnSeed), m_hitboxType(hitboxType) {
    EnsureRocketSimInit();
    InitArena();
}

CPURefSim::~CPURefSim() {
    CleanupArena();
}

CPURefSim::CPURefSim(CPURefSim&& other) noexcept
    : m_arena(other.m_arena),
      m_cars(std::move(other.m_cars)),
      m_numCars(other.m_numCars),
      m_addFloor(other.m_addFloor),
      m_tickRate(other.m_tickRate),
      m_spawnSeed(other.m_spawnSeed),
      m_hitboxType(other.m_hitboxType) {
    {
        std::lock_guard<std::mutex> lock(s_simArenaMapMutex);
        s_simArenaMap.erase(&other);
        s_simArenaMap[this] = m_arena;
    }
    other.m_arena = nullptr;
    other.m_cars.clear();
}

CPURefSim& CPURefSim::operator=(CPURefSim&& other) noexcept {
    if (this != &other) {
        CleanupArena();
        m_arena = other.m_arena;
        m_cars = std::move(other.m_cars);
        m_numCars = other.m_numCars;
        m_addFloor = other.m_addFloor;
        m_tickRate = other.m_tickRate;
        m_spawnSeed = other.m_spawnSeed;
        m_hitboxType = other.m_hitboxType;
        {
            std::lock_guard<std::mutex> lock(s_simArenaMapMutex);
            s_simArenaMap.erase(&other);
            s_simArenaMap[this] = m_arena;
        }
        other.m_arena = nullptr;
        other.m_cars.clear();
    }
    return *this;
}

void CPURefSim::InitArena() {
    // GameMode::THE_VOID bypasses mesh loading entirely
    m_arena = RocketSim::Arena::Create(RocketSim::GameMode::THE_VOID, {}, m_tickRate);
    if (!m_arena) {
        throw std::runtime_error("Failed to create RocketSim CPU Arena in GameMode::THE_VOID");
    }

    if (m_addFloor) {
        auto fnAddStaticPlane = [&](const btVector3& normal, const btVector3& posBT) {
            btCollisionShape* planeShape = new btStaticPlaneShape(normal, 0);
            btRigidBody* rb = m_arena->_AddStaticCollisionShape(planeShape, posBT);
            if (rb) {
                rb->setRestitution(RocketSim::RLConst::ARENA_COLLISION_BASE_RESTITUTION);
                rb->setFriction(RocketSim::RLConst::ARENA_COLLISION_BASE_FRICTION);
                rb->setRollingFriction(0.0f);
            }
        };

        // Floor (z = 0)
        fnAddStaticPlane(btVector3(0, 0, 1), btVector3(0, 0, 0));

        // Ceiling (z = 2048 UU)
        fnAddStaticPlane(btVector3(0, 0, -1), btVector3(0, 0, RocketSim::RLConst::ARENA_HEIGHT * UU_TO_BT));

        // Side walls (x = +/- 4096 UU)
        fnAddStaticPlane(btVector3(1, 0, 0), btVector3(-RocketSim::RLConst::ARENA_EXTENT_X * UU_TO_BT, 0, 0));
        fnAddStaticPlane(btVector3(-1, 0, 0), btVector3(RocketSim::RLConst::ARENA_EXTENT_X * UU_TO_BT, 0, 0));

        // Back walls (y = +/- 5120 UU)
        fnAddStaticPlane(btVector3(0, 1, 0), btVector3(0, -RocketSim::RLConst::ARENA_EXTENT_Y * UU_TO_BT, 0));
        fnAddStaticPlane(btVector3(0, -1, 0), btVector3(0, RocketSim::RLConst::ARENA_EXTENT_Y * UU_TO_BT, 0));

        // Corner chamfers (x + y = 8064 UU)
        constexpr float invSqrt2 = 0.7071067811865475f;
        fnAddStaticPlane(btVector3(-invSqrt2, -invSqrt2, 0), btVector3(3520.0f * UU_TO_BT, 4544.0f * UU_TO_BT, 0));
        fnAddStaticPlane(btVector3(invSqrt2, -invSqrt2, 0), btVector3(-3520.0f * UU_TO_BT, 4544.0f * UU_TO_BT, 0));
        fnAddStaticPlane(btVector3(-invSqrt2, invSqrt2, 0), btVector3(3520.0f * UU_TO_BT, -4544.0f * UU_TO_BT, 0));
        fnAddStaticPlane(btVector3(invSqrt2, invSqrt2, 0), btVector3(-3520.0f * UU_TO_BT, -4544.0f * UU_TO_BT, 0));
    }

    // Enable arena boost pad simulation without loading collision meshes
    m_arena->gameMode = RocketSim::GameMode::SOCCAR;

    // Populate standard 34 Soccar boost pads
    using namespace RocketSim::RLConst::BoostPads;
    m_arena->_boostPads.reserve(LOCS_AMOUNT_BIG + LOCS_AMOUNT_SMALL_SOCCAR);

    for (int i = 0; i < (LOCS_AMOUNT_BIG + LOCS_AMOUNT_SMALL_SOCCAR); i++) {
        RocketSim::BoostPadConfig padConfig;
        padConfig.isBig = (i < LOCS_AMOUNT_BIG);
        padConfig.pos = padConfig.isBig
            ? LOCS_BIG_SOCCAR[i]
            : LOCS_SMALL_SOCCAR[i - LOCS_AMOUNT_BIG];

        RocketSim::BoostPad* pad = RocketSim::BoostPad::_AllocBoostPad();
        pad->_Setup(padConfig);

        m_arena->_boostPads.push_back(pad);
        m_arena->_boostPadGrid.Add(pad);
    }

    {
        std::lock_guard<std::mutex> lock(s_simArenaMapMutex);
        s_simArenaMap[this] = m_arena;
    }

    m_cars.clear();
    const RocketSim::CarConfig* carCfg = &RocketSim::CAR_CONFIG_OCTANE;
    switch (m_hitboxType) {
        case 1: carCfg = &RocketSim::CAR_CONFIG_DOMINUS; break;
        case 2: carCfg = &RocketSim::CAR_CONFIG_PLANK; break;
        case 3: carCfg = &RocketSim::CAR_CONFIG_BREAKOUT; break;
        case 4: carCfg = &RocketSim::CAR_CONFIG_HYBRID; break;
        case 5: carCfg = &RocketSim::CAR_CONFIG_MERC; break;
        case 6: carCfg = &RocketSim::CAR_CONFIG_PSYCLOPS; break;
        default: carCfg = &RocketSim::CAR_CONFIG_OCTANE; break;
    }

    for (int i = 0; i < m_numCars; i++) {
        RocketSim::Team team = (m_numCars > 1 && (i % 2 != 0)) ? RocketSim::Team::ORANGE : RocketSim::Team::BLUE;
        RocketSim::Car* car = m_arena->AddCar(team, *carCfg);
        if (!car) {
            throw std::runtime_error("Failed to add car to RocketSim CPU Arena");
        }
        // Enforce deterministic initial spawn position and boost
        car->Respawn(RocketSim::GameMode::THE_VOID, m_spawnSeed + i, RocketSim::RLConst::BOOST_SPAWN_AMOUNT);
        m_cars.push_back(car);
    }
}

void CPURefSim::CleanupArena() {
    {
        std::lock_guard<std::mutex> lock(s_simArenaMapMutex);
        s_simArenaMap.erase(this);
    }
    if (m_arena) {
        delete m_arena;
        m_arena = nullptr;
    }
    m_cars.clear();
}

void CPURefSim::Reset() {
    CleanupArena();
    InitArena();
}

void CPURefSim::SetHitboxType(int ht) {
    if (m_hitboxType == ht) return;
    m_hitboxType = ht;
    CleanupArena();
    InitArena();
}

void CPURefSim::ResetToRandomKickoff(int seed) {
    if (m_arena) {
        m_arena->ResetToRandomKickoff(seed);
    }
}

uint64_t CPURefSim::GetTickCount() const {
    return m_arena ? m_arena->tickCount : 0;
}

void CPURefSim::Step(const CarControls* controls, int numCars) {
    if (!m_arena) return;

    if (controls) {
        int applyCount = (std::min)(numCars, static_cast<int>(m_cars.size()));
        for (int i = 0; i < applyCount; i++) {
            m_cars[i]->controls.throttle  = controls[i].throttle;
            m_cars[i]->controls.steer     = controls[i].steer;
            m_cars[i]->controls.pitch     = controls[i].pitch;
            m_cars[i]->controls.yaw       = controls[i].yaw;
            m_cars[i]->controls.roll      = controls[i].roll;
            m_cars[i]->controls.boost     = (controls[i].boost != 0);
            m_cars[i]->controls.jump      = (controls[i].jump != 0);
            m_cars[i]->controls.handbrake = (controls[i].handbrake != 0);
            m_cars[i]->controls.ClampFix();
        }
    }

    m_arena->Step(1);
}

void CPURefSim::GetBallState(BallStatePOD& out) const {
    if (!m_arena || !m_arena->ball) return;

    RocketSim::BallState bs = m_arena->ball->GetState();
    out.pos = Vec3(bs.pos.x, bs.pos.y, bs.pos.z);
    out.vel = Vec3(bs.vel.x, bs.vel.y, bs.vel.z);
    out.ang_vel = Vec3(bs.angVel.x, bs.angVel.y, bs.angVel.z);

    btQuaternion bq = m_arena->ball->_rigidBody.getWorldTransform().getRotation();
    out.quat = Quat(bq.w(), bq.x(), bq.y(), bq.z());
}

void CPURefSim::GetCarState(int carIdx, CarStatePOD& out) const {
    if (carIdx < 0 || carIdx >= static_cast<int>(m_cars.size())) return;
    RocketSim::Car* car = m_cars[carIdx];
    RocketSim::CarState cs = car->GetState();

    out.pos = Vec3(cs.pos.x, cs.pos.y, cs.pos.z);
    out.vel = Vec3(cs.vel.x, cs.vel.y, cs.vel.z);
    out.ang_vel = Vec3(cs.angVel.x, cs.angVel.y, cs.angVel.z);

    btQuaternion cq = car->_rigidBody.getWorldTransform().getRotation();
    out.quat = Quat(cq.w(), cq.x(), cq.y(), cq.z());

    out.boost = cs.boost;
    out.is_on_ground = cs.isOnGround ? 1 : 0;
    out.has_jumped = cs.hasJumped ? 1 : 0;
    out.has_double_jumped = cs.hasDoubleJumped ? 1 : 0;
    out.has_flipped = cs.hasFlipped ? 1 : 0;
    out.is_demoed = cs.isDemoed ? 1 : 0;
    out.team = (car->team == RocketSim::Team::BLUE) ? 0 : 1;
    out.hitbox_type = static_cast<uint8_t>(m_hitboxType);

    for (int w = 0; w < 4; w++) {
        out.wheels_with_contact[w] = cs.wheelsWithContact[w] ? 1 : 0;
        float restLen = car->_bulletVehicle.m_wheelInfo[w].getSuspensionRestLength();
        float curLen = car->_bulletVehicle.m_wheelInfo[w].m_raycastInfo.m_suspensionLength;
        out.suspension_lengths[w] = (restLen - curLen) * BT_TO_UU;
    }

    out.last_controls.throttle  = cs.lastControls.throttle;
    out.last_controls.steer     = cs.lastControls.steer;
    out.last_controls.pitch     = cs.lastControls.pitch;
    out.last_controls.yaw       = cs.lastControls.yaw;
    out.last_controls.roll      = cs.lastControls.roll;
    out.last_controls.boost     = cs.lastControls.boost ? 1 : 0;
    out.last_controls.jump      = cs.lastControls.jump ? 1 : 0;
    out.last_controls.handbrake = cs.lastControls.handbrake ? 1 : 0;
}

void CPURefSim::SetBallState(const BallStatePOD& in) {
    if (!m_arena || !m_arena->ball) return;

    RocketSim::BallState bs;
    bs.pos = RocketSim::Vec(in.pos.x, in.pos.y, in.pos.z);
    bs.vel = RocketSim::Vec(in.vel.x, in.vel.y, in.vel.z);
    bs.angVel = RocketSim::Vec(in.ang_vel.x, in.ang_vel.y, in.ang_vel.z);
    btMatrix3x3 basis(btQuaternion(in.quat.x, in.quat.y, in.quat.z, in.quat.w));
    bs.rotMat = RocketSim::RotMat(basis);

    // If ball is spawned in mid-air with zero velocity, impart microscopic vel so Arena::Step won't force ISLAND_SLEEPING
    if (bs.pos.z > 95.0f && bs.vel.LengthSq() == 0.0f && bs.angVel.LengthSq() == 0.0f) {
        bs.vel.z = -1e-6f;
    }

    m_arena->ball->SetState(bs);
    m_arena->ball->_rigidBody.activate(true);
    m_arena->ball->_rigidBody.setActivationState(ACTIVE_TAG);
}

void CPURefSim::SetCarState(int carIdx, const CarStatePOD& in) {
    if (carIdx < 0 || carIdx >= static_cast<int>(m_cars.size())) return;
    RocketSim::Car* car = m_cars[carIdx];
    RocketSim::CarState cs = car->GetState();

    cs.pos = RocketSim::Vec(in.pos.x, in.pos.y, in.pos.z);
    cs.vel = RocketSim::Vec(in.vel.x, in.vel.y, in.vel.z);
    cs.angVel = RocketSim::Vec(in.ang_vel.x, in.ang_vel.y, in.ang_vel.z);
    btMatrix3x3 basis(btQuaternion(in.quat.x, in.quat.y, in.quat.z, in.quat.w));
    cs.rotMat = RocketSim::RotMat(basis);
    cs.boost = in.boost;
    cs.isOnGround = (in.is_on_ground != 0);
    car->team = (in.team == 0) ? RocketSim::Team::BLUE : RocketSim::Team::ORANGE;

    car->SetState(cs);
    car->_rigidBody.activate(true);
    car->_rigidBody.setActivationState(ACTIVE_TAG);
}

} // namespace rocketsim_cuda
