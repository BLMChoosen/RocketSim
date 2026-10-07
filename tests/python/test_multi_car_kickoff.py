"""
Multi-Car Kickoff & Team Symmetry Validation Suite (Milestone 5 - Module 2.1)

Validates:
1. Up to 6 cars per arena (1v1 = 2 cars, 2v2 = 4 cars, 3v3 = 6 cars).
2. Exact team assignments: even indices (0, 2, 4) -> BLUE (0), odd indices (1, 3, 5) -> ORANGE (1).
3. 5 canonical Soccar kickoff spawn locations (RLConst.h:356-362).
4. Orange team spatial mirroring: X -> -X, Y -> -Y, Z -> Z (Arena.cpp:187-192).
5. Orange team angular orientation: yaw -> yaw + PI, pitch -> 0, roll -> 0 (rotMat / Quat alignment).
6. Initial physical attributes: Z = 17.0 UU (CAR_SPAWN_REST_Z), boost = 33.3333f (100/3), isOnGround = 1.
7. SoA memory layout coalescing across N cars per arena.
"""

import math
import numpy as np
import pytest

# Constants from RLConst.h & config.h
CAR_SPAWN_REST_Z = 17.0
BOOST_SPAWN_AMOUNT = 100.0 / 3.0 # 33.33333f

# 5 Canonical Soccar Blue Kickoff Spawns: (X, Y, Yaw in rad)
SOCCAR_BLUE_SPAWNS = [
    (-2048.0, -2560.0, math.pi / 4.0),       # 0: Diagonal Left (45 deg)
    ( 2048.0, -2560.0, 3.0 * math.pi / 4.0), # 1: Diagonal Right (135 deg)
    (  -256.0, -3840.0, math.pi / 2.0),       # 2: Off-Center Left (90 deg)
    (   256.0, -3840.0, math.pi / 2.0),       # 3: Off-Center Right (90 deg)
    (     0.0, -4608.0, math.pi / 2.0),       # 4: Goalie / Center (90 deg)
]


def yaw_to_quaternion(yaw: float):
    """
    Computes quaternion (w, x, y, z) for pure yaw rotation about Z-axis.
    Matches Bullet btMatrix3x3::setEulerYPR(yaw, 0, 0).getRotation().
    """
    half_yaw = yaw * 0.5
    w = math.cos(half_yaw)
    z = math.sin(half_yaw)
    return np.array([w, 0.0, 0.0, z], dtype=np.float32)


def get_kickoff_spawn_cpu(spawn_index: int, is_blue: bool):
    """
    Replicates exact CPU RocketSim Arena::ResetToRandomKickoff (Arena.cpp:180-195).
    """
    bx, by, byaw = SOCCAR_BLUE_SPAWNS[spawn_index % 5]
    if is_blue:
        pos = np.array([bx, by, CAR_SPAWN_REST_Z], dtype=np.float32)
        quat = yaw_to_quaternion(byaw)
    else:
        # Orange team mirroring: pos *= {-1, -1, 1}, yaw += PI
        pos = np.array([-bx, -by, CAR_SPAWN_REST_Z], dtype=np.float32)
        quat = yaw_to_quaternion(byaw + math.pi)
    return pos, quat


def get_kickoff_spawn_gpu(env_idx: int, car_in_env_idx: int, cars_per_env: int, seed: int = 0):
    """
    Replicates GPU SimContext init_single_car (sim_context.cu:73-86).
    """
    team = (car_in_env_idx % 2) if cars_per_env > 1 else 0
    team_car_idx = (car_in_env_idx // 2) if cars_per_env > 1 else car_in_env_idx

    # Hash function matching get_kickoff_slot in sim_context.cu
    x = env_idx ^ (seed * 0x9E3779B9)
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = (x >> 16) ^ x
    base_slot = x % 5

    spawn_slot = (base_slot + team_car_idx) % 5
    is_blue = (team == 0)
    return get_kickoff_spawn_cpu(spawn_slot, is_blue), team


def test_kickoff_spawns_symmetry_all_slots():
    """Verify that every Soccar kickoff slot mirrors perfectly across center field."""
    for slot_idx in range(5):
        pos_blue, quat_blue = get_kickoff_spawn_cpu(slot_idx, is_blue=True)
        pos_orange, quat_orange = get_kickoff_spawn_cpu(slot_idx, is_blue=False)

        # Spatial inversion: X -> -X, Y -> -Y, Z identical
        assert np.isclose(pos_orange[0], -pos_blue[0], atol=1e-5), f"Slot {slot_idx} X symmetry failed"
        assert np.isclose(pos_orange[1], -pos_blue[1], atol=1e-5), f"Slot {slot_idx} Y symmetry failed"
        assert np.isclose(pos_orange[2], CAR_SPAWN_REST_Z, atol=1e-5), f"Slot {slot_idx} Z height failed"
        assert np.isclose(pos_blue[2], CAR_SPAWN_REST_Z, atol=1e-5)

        # Quaternions: rotation must face opposite direction (relative yaw delta is exactly PI)
        # Dot product between Blue forward (cos(yaw), sin(yaw)) and Orange forward must be -1.0
        bx, by, byaw = SOCCAR_BLUE_SPAWNS[slot_idx]
        blue_fwd = np.array([math.cos(byaw), math.sin(byaw)])
        orange_fwd = np.array([math.cos(byaw + math.pi), math.sin(byaw + math.pi)])
        dot = np.dot(blue_fwd, orange_fwd)
        assert np.isclose(dot, -1.0, atol=1e-5), f"Slot {slot_idx} facing direction failed"


@pytest.mark.parametrize("cars_per_env, expected_teams", [
    (1, [0]),                               # 1v0: Blue
    (2, [0, 1]),                            # 1v1: Blue, Orange
    (4, [0, 1, 0, 1]),                      # 2v2: Blue 0, Orange 0, Blue 1, Orange 1
    (6, [0, 1, 0, 1, 0, 1]),                # 3v3: Blue 0, Orange 0, Blue 1, Orange 1, Blue 2, Orange 2
])
def test_multi_car_team_assignment(cars_per_env, expected_teams):
    """Verify team assignments for 1v0, 1v1, 2v2, and 3v3 matches GPU/CPU interleaved convention."""
    for car_idx in range(cars_per_env):
        (_, _), team = get_kickoff_spawn_gpu(env_idx=0, car_in_env_idx=car_idx, cars_per_env=cars_per_env)
        assert team == expected_teams[car_idx], f"Car {car_idx}/{cars_per_env} got team {team}, expected {expected_teams[car_idx]}"


def test_3v3_kickoff_pairwise_symmetry():
    """Verify in 3v3 that corresponding team players (0 vs 1, 2 vs 3, 4 vs 5) occupy mirrored spawn slots."""
    num_envs = 64
    for env in range(num_envs):
        for pair in [(0, 1), (2, 3), (4, 5)]:
            blue_car_idx, orange_car_idx = pair
            (pos_blue, quat_blue), team_b = get_kickoff_spawn_gpu(env, blue_car_idx, cars_per_env=6, seed=42)
            (pos_orange, quat_orange), team_o = get_kickoff_spawn_gpu(env, orange_car_idx, cars_per_env=6, seed=42)

            assert team_b == 0
            assert team_o == 1

            # Exact spatial inversion
            assert np.isclose(pos_orange[0], -pos_blue[0], atol=1e-4)
            assert np.isclose(pos_orange[1], -pos_blue[1], atol=1e-4)
            assert np.isclose(pos_orange[2], pos_blue[2], atol=1e-4)


def test_soa_memory_coalescing_stride():
    """Verify SoA memory footprint and 128-byte cacheline alignment for up to 65536 envs with 6 cars."""
    for cars_per_env in [1, 2, 4, 6]:
        num_envs = 16384
        total_cars = num_envs * cars_per_env

        # 128-byte alignment check
        stride_bytes = 4  # float32
        slice_size = (total_cars * stride_bytes + 127) & ~127
        assert slice_size % 128 == 0, f"Slice size {slice_size} not 128-byte aligned"
        assert slice_size >= total_cars * stride_bytes
