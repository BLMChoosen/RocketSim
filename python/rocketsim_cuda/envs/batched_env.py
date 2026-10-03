"""
Vectorized Gymnasium-compatible environment for RocketSim-CUDA.
Enables massive parallel RL rollouts (16k - 64k envs) in GPU device memory
with zero-copy PyTorch/DLPack tensors and zero host-device transfers.
"""

from typing import Optional, Tuple, Union, Dict, Any
import os
import sys

# Try importing torch
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


class RocketSimBatchedEnv:
    """
    Ultra-parallel vectorized RL environment running entirely in GPU VRAM.

    Invariants:
    - Zero PCIe transfers during step() and reset()
    - Action ingestion from GPU VRAM directly into CUDA simulation kernel
    - Observations, rewards, terminations exposed as pure GPU tensors
    - Asynchronous selective resets without CPU synchronization barriers
    """

    def __init__(
        self,
        num_envs: int,
        cars_per_env: int = 1,
        team_size: int = 1,
        tick_skip: int = 1,
        use_torch: bool = True,
    ):
        """
        Initialize RocketSimBatchedEnv.

        Args:
            num_envs: Number of concurrent environments in GPU memory (e.g. 16384 to 65536).
            cars_per_env: Number of cars simulated per arena.
            team_size: Number of cars per team.
            tick_skip: Number of 120Hz physics sub-steps per environment step.
            use_torch: If True and torch.cuda is available, wrap views in torch.Tensor via DLPack.
        """
        self.num_envs = int(num_envs)
        self.cars_per_env = int(cars_per_env)
        self.team_size = int(team_size)
        self.tick_skip = int(tick_skip)
        self.total_cars = self.num_envs * self.cars_per_env
        self.use_torch = bool(use_torch)

        # Import SimContext from native module
        import rocketsim_cuda
        self._native = rocketsim_cuda

        # Allocate monolithic GPU VRAM arena
        self.sim = rocketsim_cuda.SimContext(
            num_envs=self.num_envs,
            cars_per_env=self.cars_per_env
        )

        # Pre-cache zero-copy device tensor views
        self._car_obs_view = self.sim.get_car_observations()
        self._ball_obs_view = self.sim.get_ball_observations()
        self._rewards_view = self.sim.get_rewards()
        self._terminated_view = self.sim.get_terminated()
        self._truncated_view = self.sim.get_truncated()
        self._is_goal_view = self.sim.get_is_goal()
        self._scoring_team_view = self.sim.get_scoring_team()
        self._is_oob_view = self.sim.get_is_out_of_bounds()
        self._tick_count_view = self.sim.get_tick_count()
        self._pad_active_view = self.sim.get_pad_is_active()
        self._pad_cd_view = self.sim.get_pad_cooldown()

        # Check if torch with CUDA support is active
        self._torch_cuda_ready = (
            self.use_torch
            and _TORCH_AVAILABLE
            and hasattr(torch, "cuda")
            and torch.cuda.is_available()
        )

    def _wrap(self, view: Any) -> Any:
        """Wrap GpuTensorView as PyTorch CUDA tensor if PyTorch CUDA is available, else return view."""
        if self._torch_cuda_ready:
            try:
                return torch.from_dlpack(view)
            except Exception:
                return view
        return view

    def get_car_observations(self) -> Any:
        """
        Get zero-copy car observations in GPU VRAM.
        Shape: [num_envs, cars_per_env, 14]
        Order: [pos_x..z, vel_x..z, q_w..z, ang_vel_x..z, boost]
        """
        return self._wrap(self._car_obs_view)

    def get_ball_observations(self) -> Any:
        """
        Get zero-copy ball observations in GPU VRAM.
        Shape: [num_envs, 13]
        Order: [pos_x..z, vel_x..z, q_w..z, ang_vel_x..z]
        """
        return self._wrap(self._ball_obs_view)

    def get_rewards(self) -> Any:
        """
        Get zero-copy reward tensor in GPU VRAM.
        Shape: [num_envs, cars_per_env]
        """
        return self._wrap(self._rewards_view)

    def get_terminated(self) -> Any:
        """
        Get zero-copy terminated flags in GPU VRAM (1 if goal or out_of_bounds, else 0).
        Shape: [num_envs]
        """
        return self._wrap(self._terminated_view)

    def get_truncated(self) -> Any:
        """
        Get zero-copy truncated flags in GPU VRAM.
        Shape: [num_envs]
        """
        return self._wrap(self._truncated_view)

    def get_is_goal(self) -> Any:
        """Get zero-copy goal flags in GPU VRAM. Shape: [num_envs]"""
        return self._wrap(self._is_goal_view)

    def get_scoring_team(self) -> Any:
        """Get zero-copy scoring team flags in GPU VRAM (0=Blue, 1=Orange). Shape: [num_envs]"""
        return self._wrap(self._scoring_team_view)

    def get_is_out_of_bounds(self) -> Any:
        """Get zero-copy out-of-bounds flags in GPU VRAM. Shape: [num_envs]"""
        return self._wrap(self._is_oob_view)

    def get_tick_count(self) -> Any:
        """Get zero-copy tick count tensor in GPU VRAM. Shape: [num_envs]"""
        return self._wrap(self._tick_count_view)

    def get_pad_is_active(self) -> Any:
        """Get zero-copy boost pad active flags in GPU VRAM. Shape: [num_envs, 34]"""
        return self._wrap(self._pad_active_view)

    def get_pad_cooldown(self) -> Any:
        """Get zero-copy boost pad cooldown timers in GPU VRAM. Shape: [num_envs, 34]"""
        return self._wrap(self._pad_cd_view)

    @property
    def observations(self) -> Dict[str, Any]:
        """Dictionary of car and ball observations in GPU VRAM."""
        return {
            "cars": self.get_car_observations(),
            "ball": self.get_ball_observations(),
        }

    def get_info(self) -> Dict[str, Any]:
        """Info dictionary containing supplementary zero-copy GPU tensors."""
        return {
            "ball": self.get_ball_observations(),
            "is_goal": self.get_is_goal(),
            "scoring_team": self.get_scoring_team(),
            "is_out_of_bounds": self.get_is_out_of_bounds(),
            "tick_count": self.get_tick_count(),
            "pad_is_active": self.get_pad_is_active(),
            "pad_cooldown": self.get_pad_cooldown(),
        }

    def step(
        self,
        actions: Optional[Any] = None
    ) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
        """
        Execute sub-stepping physics simulation at 120Hz for `tick_skip` ticks.

        Pure Zero-Copy Pipeline:
        - Ingests `actions` directly from GPU VRAM without host copies.
        - Updates physics state in-place in device memory.
        - Computes rewards and terminations on device.
        - Returns (obs, rewards, terminated, truncated, info) as pure GPU tensors.
        """
        for _ in range(self.tick_skip):
            if actions is not None:
                self.sim.step_actions(actions)
            else:
                self.sim.step(0)

        obs = self.get_car_observations()
        rewards = self.get_rewards()
        terminated = self.get_terminated()
        truncated = self.get_truncated()
        info = self.get_info()

        return obs, rewards, terminated, truncated, info

    def reset(
        self,
        env_indices: Optional[Any] = None,
        return_info: bool = False
    ) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
        """
        Asynchronously reset environments on GPU.

        Args:
            env_indices: If None, reset all environments to default kickoff positions.
                         If a GPU tensor/list of indices is provided, reset only those environments.
            return_info: If True, return (obs, info) tuple (Gymnasium standard).

        Returns:
            obs or (obs, info)
        """
        if env_indices is None:
            self.sim.reset_to_default()
        else:
            self.sim.reset_batch(env_indices)

        obs = self.get_car_observations()
        if return_info:
            return obs, self.get_info()
        return obs

    def reset_masked(self, mask: Any) -> Any:
        """
        Selectively reset environments where mask is non-zero directly on GPU.

        Args:
            mask: GPU uint8 tensor of shape [num_envs].
        """
        self.sim.reset_masked(mask)
        return self.get_car_observations()

    def close(self):
        """Release environment resources."""
        self._car_obs_view = None
        self._ball_obs_view = None
        self._rewards_view = None
        self._terminated_view = None
        self._truncated_view = None
        self.sim = None
