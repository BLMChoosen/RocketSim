import jax
import jax.numpy as jnp
import numpy as np
from jax_sim import create_initial_state, step_physics, create_zero_controls
from visualizer.states import GameState, CarState, PhysState
from pyrr import Vector3, Quaternion, Matrix33

class SimWrapper:
    def __init__(self, n_envs=1, max_cars=6):
        self.n_envs = n_envs
        self.max_cars = max_cars
        
        print(f"[SimWrapper] Initializing JAX sim with {n_envs} envs, {max_cars} cars...")
        self.state = create_initial_state(n_envs, max_cars)
        self.controls = create_zero_controls(n_envs, max_cars)
        
        # JIT the step function for performance
        print("[SimWrapper] JIT compiling step function...")
        self.step_fn = jax.jit(step_physics)
        # Trigger compilation
        self.step_fn(self.state, self.controls)
        print("[SimWrapper] JIT compilation complete.")
        
        # We only visualize the first environment
        self.env_idx = 0
        
        # Initialize visualizer GameState
        self.game_state = GameState()
        self.init_game_state()

    def init_game_state(self):
        # Initialize cars in GameState
        self.game_state.car_states = []
        for i in range(self.max_cars):
            self.game_state.car_states.append(CarState())
        
        # Initialize boost pads
        # We need to populate boost_pad_locations from JAX sim constants if possible,
        # but GameState already has default locations. We'll assume they match for now.
        # JAX sim uses PAD_LOCATIONS from RLConst.h, which should match.
        
        self.update_visualizer_state()

    def reset(self):
        self.state = create_initial_state(self.n_envs, self.max_cars)
        self.update_visualizer_state()

    def step(self, user_controls=None):
        # Update controls if provided
        # user_controls: list of (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
        # or similar structure.
        
        if user_controls is not None:
            # Apply controls to the first car of the first env
            # user_controls is expected to be a dict or object with attributes
            # We need to update self.controls (CarControls)
            
            # Create a new CarControls object with updated values for the specific car
            # Since JAX arrays are immutable, we need to update the specific indices
            
            # This is a bit tricky with JAX structures. 
            # self.controls is a CarControls object with arrays of shape (N, MAX_CARS)
            
            # Let's assume user_controls is for car 0
            car_idx = 0
            
            # Helper to update a specific index in a JAX array
            def update_array(arr, val):
                return arr.at[self.env_idx, car_idx].set(val)
            
            new_throttle = update_array(self.controls.throttle, user_controls.throttle)
            new_steer = update_array(self.controls.steer, user_controls.steer)
            new_pitch = update_array(self.controls.pitch, user_controls.pitch)
            new_yaw = update_array(self.controls.yaw, user_controls.yaw)
            new_roll = update_array(self.controls.roll, user_controls.roll)
            new_jump = update_array(self.controls.jump, user_controls.jump)
            new_boost = update_array(self.controls.boost, user_controls.boost)
            new_handbrake = update_array(self.controls.handbrake, user_controls.handbrake)
            
            from jax_sim import CarControls
            self.controls = CarControls(
                throttle=new_throttle,
                steer=new_steer,
                pitch=new_pitch,
                yaw=new_yaw,
                roll=new_roll,
                jump=new_jump,
                boost=new_boost,
                handbrake=new_handbrake
            )
            
        # Step physics
        self.state = self.step_fn(self.state, self.controls)
        
        # Update GameState
        self.update_visualizer_state()
        
    def update_visualizer_state(self):
        # Convert JAX state to numpy for the visualizer
        # We take the slice for env_idx
        
        # Ball
        ball_pos = np.array(self.state.ball.pos[self.env_idx])
        ball_vel = np.array(self.state.ball.vel[self.env_idx])
        ball_ang_vel = np.array(self.state.ball.ang_vel[self.env_idx])
        
        self.update_phys_state(self.game_state.ball_state, ball_pos, ball_vel, ball_ang_vel)
        
        # Cars
        for i in range(self.max_cars):
            car_pos = np.array(self.state.cars.pos[self.env_idx, i])
            car_vel = np.array(self.state.cars.vel[self.env_idx, i])
            car_ang_vel = np.array(self.state.cars.ang_vel[self.env_idx, i])
            car_quat = np.array(self.state.cars.quat[self.env_idx, i]) # w, x, y, z
            
            # JAX quat is [w, x, y, z]
            # pyrr Quaternion is [x, y, z, w]
            quat_xyzw = np.array([car_quat[1], car_quat[2], car_quat[3], car_quat[0]])
            
            cs = self.game_state.car_states[i]
            self.update_phys_state(cs.phys, car_pos, car_vel, car_ang_vel, quat_xyzw)
            
            cs.team_num = int(self.state.cars.team[self.env_idx, i])
            cs.boost_amount = float(self.state.cars.boost_amount[self.env_idx, i])
            cs.is_demoed = bool(self.state.cars.is_demoed[self.env_idx, i])
            cs.on_ground = bool(self.state.cars.is_on_ground[self.env_idx, i])
            
            # Update boost pads
            if self.game_state.boost_pad_states is None:
                 self.game_state.boost_pad_states = [False] * len(self.game_state.boost_pad_locations)
            
            # JAX sim has pad_is_active
            pad_active = np.array(self.state.pad_is_active[self.env_idx])
            self.game_state.boost_pad_states = pad_active.astype(bool).tolist()

    def update_phys_state(self, phys_state: PhysState, pos, vel, ang_vel, quat=None):
        # Shift prev to next
        phys_state.prev_pos = phys_state.next_pos
        phys_state.prev_vel = phys_state.next_vel
        phys_state.prev_forward = phys_state.next_forward
        phys_state.prev_up = phys_state.next_up
        
        # Update next
        phys_state.next_pos = Vector3(pos)
        phys_state.next_vel = Vector3(vel)
        phys_state.ang_vel = Vector3(ang_vel)
        
        if quat is not None:
            phys_state.has_rot = True
            q = Quaternion(quat)
            mat = Matrix33.from_quaternion(q)
            
            # Extract forward (X) and up (Z) from rotation matrix
            # Matrix33 in pyrr is column-major?
            # mat[0] is column 0 (X axis), mat[1] is column 1 (Y axis), mat[2] is column 2 (Z axis)
            phys_state.next_forward = Vector3((mat[0][0], mat[1][0], mat[2][0]))
            phys_state.next_up = Vector3((mat[0][2], mat[1][2], mat[2][2]))
