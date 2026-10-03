"""
Unit and integration tests for rocketsim_cuda Nanobind Python bindings
and DLPack zero-copy tensor views.
"""

import sys
import os
import pytest

# Ensure build output directory is in sys.path
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import rocketsim_cuda


def test_import_module():
    """Verify rocketsim_cuda module imports correctly and exposes expected types."""
    assert hasattr(rocketsim_cuda, "SimContext")
    assert hasattr(rocketsim_cuda, "GpuTensorView")


def test_sim_context_properties():
    """Verify SimContext initial properties and memory allocations."""
    num_envs = 16
    cars_per_env = 2
    sim = rocketsim_cuda.SimContext(num_envs=num_envs, cars_per_env=cars_per_env)

    assert sim.num_envs == num_envs
    assert sim.cars_per_env == cars_per_env
    assert sim.total_cars == num_envs * cars_per_env
    assert sim.allocated_bytes > 0
    assert sim.ball_pitch_floats >= num_envs
    assert sim.car_pitch_floats >= (num_envs * cars_per_env)


def test_ball_observations_dlpack():
    """Verify get_ball_observations DLPack view shape, strides, and capsule protocol."""
    num_envs = 8
    cars_per_env = 1
    sim = rocketsim_cuda.SimContext(num_envs=num_envs, cars_per_env=cars_per_env)

    ball_obs = sim.get_ball_observations()
    assert isinstance(ball_obs, rocketsim_cuda.GpuTensorView)
    assert ball_obs.shape == (num_envs, 13)
    assert ball_obs.strides == (1, sim.ball_pitch_floats)
    assert ball_obs.dtype == "float32"
    assert ball_obs.device == "cuda:0"
    assert ball_obs.data_ptr != 0

    # Test alias
    ball_state = sim.get_ball_state_tensor()
    assert ball_state.shape == (num_envs, 13)
    assert ball_state.strides == (1, sim.ball_pitch_floats)
    assert ball_state.data_ptr == ball_obs.data_ptr

    # DLPack protocol verification
    dev_type, dev_id = ball_obs.__dlpack_device__()
    assert dev_type == 2  # kDLCUDA
    assert dev_id == 0

    capsule = ball_obs.__dlpack__()
    assert capsule is not None


def test_car_observations_dlpack():
    """Verify get_car_observations DLPack view shape, strides, and pointer fidelity."""
    num_envs = 8
    cars_per_env = 2
    sim = rocketsim_cuda.SimContext(num_envs=num_envs, cars_per_env=cars_per_env)

    car_obs = sim.get_car_observations()
    assert isinstance(car_obs, rocketsim_cuda.GpuTensorView)
    assert car_obs.shape == (num_envs, cars_per_env, 14)
    assert car_obs.strides == (cars_per_env, 1, sim.car_pitch_floats)
    assert car_obs.dtype == "float32"
    assert car_obs.device == "cuda:0"
    assert car_obs.data_ptr != 0

    # Test alias
    car_state = sim.get_car_state_tensor()
    assert car_state.shape == (num_envs, cars_per_env, 14)
    assert car_state.strides == (cars_per_env, 1, sim.car_pitch_floats)
    assert car_state.data_ptr == car_obs.data_ptr

    # DLPack protocol verification
    dev_type, dev_id = car_obs.__dlpack_device__()
    assert dev_type == 2
    assert dev_id == 0

    capsule = car_obs.__dlpack__()
    assert capsule is not None


def test_arena_state_views():
    """Verify arena boost pad and termination flag DLPack views."""
    num_envs = 4
    sim = rocketsim_cuda.SimContext(num_envs=num_envs, cars_per_env=1)

    # Boost pads: active flags and cooldowns
    pad_active = sim.get_pad_is_active()
    assert pad_active.shape == (num_envs, 34)
    assert pad_active.strides == (34, 1)
    assert pad_active.dtype == "uint8"
    assert pad_active.data_ptr != 0

    pad_cd = sim.get_pad_cooldown()
    assert pad_cd.shape == (num_envs, 34)
    assert pad_cd.strides == (34, 1)
    assert pad_cd.dtype == "float32"
    assert pad_cd.data_ptr != 0

    # Termination flags
    is_goal = sim.get_is_goal()
    assert is_goal.shape == (num_envs,)
    assert is_goal.strides == (1,)
    assert is_goal.dtype == "uint8"

    scoring_team = sim.get_scoring_team()
    assert scoring_team.shape == (num_envs,)
    assert scoring_team.strides == (1,)
    assert scoring_team.dtype == "uint8"

    is_oob = sim.get_is_out_of_bounds()
    assert is_oob.shape == (num_envs,)
    assert is_oob.strides == (1,)
    assert is_oob.dtype == "uint8"

    tick_count = sim.get_tick_count()
    assert tick_count.shape == (num_envs,)
    assert tick_count.strides == (1,)
    assert tick_count.dtype == "uint32"


def test_zero_copy_pointer_immutability():
    """Verify zero-copy invariant: repeated accessor calls return identical GPU VRAM pointers."""
    sim = rocketsim_cuda.SimContext(num_envs=8, cars_per_env=2)

    ptr_b1 = sim.get_ball_observations().data_ptr
    ptr_b2 = sim.get_ball_observations().data_ptr
    ptr_b3 = sim.get_ball_state_tensor().data_ptr
    assert ptr_b1 == ptr_b2 == ptr_b3

    ptr_c1 = sim.get_car_observations().data_ptr
    ptr_c2 = sim.get_car_observations().data_ptr
    ptr_c3 = sim.get_car_state_tensor().data_ptr
    assert ptr_c1 == ptr_c2 == ptr_c3

    ptr_pad1 = sim.get_pad_is_active().data_ptr
    ptr_pad2 = sim.get_pad_is_active().data_ptr
    assert ptr_pad1 == ptr_pad2


def test_sim_step():
    """Verify sim.step() execution with and without batch_size."""
    sim = rocketsim_cuda.SimContext(num_envs=4, cars_per_env=1)
    sim.step()
    sim.step(4)
    sim.step(2)


def test_step_actions():
    """Verify sim.step_actions() consuming GPU action tensor view directly."""
    num_envs = 4
    cars_per_env = 2
    sim = rocketsim_cuda.SimContext(num_envs=num_envs, cars_per_env=cars_per_env)

    # Use a dummy tensor view on GPU (e.g., car_obs buffer or zeroed slice)
    car_obs = sim.get_car_observations()
    # step_actions consumes a GPU view directly without host transfer
    sim.step_actions(car_obs)
    sim.step(car_obs)


def test_selective_resets():
    """Verify selective reset methods: reset_to_default, reset_batch, and reset_masked."""
    num_envs = 8
    sim = rocketsim_cuda.SimContext(num_envs=num_envs, cars_per_env=1)

    # Full reset
    sim.reset_to_default()

    # Reset with Python list
    sim.reset_batch([0, 2, 5])

    # Reset with None (equivalent to reset_to_default)
    sim.reset_batch(None)

    # Reset masked with GPU tensor view
    is_goal = sim.get_is_goal()
    sim.reset_masked(is_goal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
