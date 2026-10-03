# Car Mechanics Implementation & Parity Status (M4.2)

> **RocketSim-CUDA Engine Governance**  
> **Reference Oracle:** RocketSim CPU (Bullet Physics 3.24)  
> **Target Tolerance:** Chebyshev $\Vert{}\Delta{}\Vert{}_\infty \le 10^{-4}\text{ UU}$ per tick  
> **Milestone:** M4.2 (Car Mechanics Completeness & Parity)  
> **Status:** Full Parity Achieved across all 8 Core Mechanics  
> **Last Updated:** 2026-10-03  

---

## 1. Overview & Mechanics Parity Matrix

This document tracks the completeness, physical accuracy, and tick-by-tick parity of Rocket League car mechanics in **RocketSim-CUDA** against the authoritative CPU reference implementation (**RocketSim / Bullet 3.24**).

All car dynamics execute purely on-device within GPU VRAM using Structure of Arrays (SoA) layout with zero dynamic memory allocation and strict IEEE-754 floating point arithmetic (`--fmad=false`).

### Parity Tracking Table

| Mechanic | Status | CPU Reference File:Line | CUDA Implementation File:Line | `CarStateSoA` Fields | Control Inputs | Physical Notes & Implementation Details |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **Jump: Immediate Impulse** | ✅ Full | `Car.cpp:573`, `RLConst.h:95` | `car_dynamics.cuh:304` | `is_jumping`, `has_jumped`, `jump_time` | `jump`, `last_controls_jump` | Instant upward impulse $875/3\text{ UU/s}$ ($\approx 291.67\text{ UU/s}$) along chassis Up vector. Bit-exact match. |
| **Jump: Hold Acceleration** | ✅ Full | `Car.cpp:581-589`, `RLConst.h:94` | `car_dynamics.cuh:309-314` | `is_jumping`, `jump_time` | `jump` | Continuous acceleration $4375/3\text{ UU/s}^2$ ($0.62\times$ for $t < 0.025$s, active while held up to $0.2$s). |
| **Jump: Ground Reset Pad** | ✅ Full | `Car.cpp:550-558`, `RLConst.h:97` | `car_dynamics.cuh:285-292` | `has_jumped`, `jump_time`, `is_on_ground` | None | Time pad $0.025\text{s} + 0.025\text{s} = 0.05\text{s}$ before ground settling resets jump state. |
| **Double Jump** | ✅ Full | `Car.cpp:773-779`, `RLConst.h:99` | `car_dynamics.cuh:399-402` | `has_double_jumped`, `air_time_since_jump` | `jump`, `pitch`, `yaw`, `roll` | Instant upward impulse $875/3\text{ UU/s}$ within $1.25$s window. Preserves flip reset when $has\_jumped == 0$. Auto-flip suppresses double jump. |
| **Flip / Dodge: 8-Way Impulses** | ✅ Full | `Car.cpp:728-771`, `RLConst.h:109-115` | `car_dynamics.cuh:357-397` | `has_flipped`, `is_flipping`, `flip_time` | `jump`, `pitch`, `yaw`, `roll` | 2D horizontal impulse ($500\text{ UU/s}$ base, scaled by speed ratio: backward $2.5\times$, side $1.9\times$, back factor $16/15$). $v_z$ impulse is 0. |
| **Flip: Z-Damping** | ✅ Full | `Car.cpp:784-790`, `RLConst.h:102-104` | `car_dynamics.cuh:406-412` | `is_flipping`, `flip_time`, `vel_z` | None | $35\%$ Z-damping between $0.15$s and $0.21$s unconditionally; active up to $0.65$s when $v_z < 0$. Matches at 120Hz. |
| **Flip: Cancel Mechanics** | ✅ Full | `Car.cpp:614-631`, `RLConst.h:110-111` | `car_dynamics.cuh:463-477` | `flip_rel_torque_x..z`, `is_flipping` | `pitch` | Pitch scale $1 - \|\text{pitch}\|$ reduces flip torque and restores air control if pitch opposes dodge torque. |
| **Flip: Pitch Lock** | ✅ Full | `Car.cpp:649-656`, `RLConst.h:107-108` | `car_dynamics.cuh:487-492` | `has_flipped`, `flip_time` | `pitch` | Pitch torque locked to 0 for $0.65\text{s} + 0.3\text{s} = 0.95\text{s}$. Damping remains active. |
| **Boost: Consumption & Force** | ✅ Full | `Car.cpp:502-546`, `RLConst.h:56-61` | `car_dynamics.cuh:92-129` | `boost`, `is_boosting`, `boosting_time` | `boost` | Consumption $33.33$/s, min time $0.1$s, ground accel $991.67\text{ UU/s}^2$, air accel $1058.33\text{ UU/s}^2$. |
| **Boost: Throttle Override** | ✅ Full | `Car.cpp:373-375` | `car_dynamics.cuh:163-165` | `boost` | `boost` | Forces wheel throttle $1.0$ while boosting with boost $> 0$. |
| **Supersonic State Transition** | ✅ Full | `Car.cpp:153-169`, `RLConst.h:69-76` | `car_dynamics.cuh:511-535` | `is_supersonic`, `supersonic_time` | None | Triggers at speed $\ge 2200\text{ UU/s}$; maintained at $\ge 2100\text{ UU/s}$ for up to $1.0$s. |
| **Air Control: Torques & Damping** | ✅ Full | `Car.cpp:642-678`, `RLConst.h:149-152` | `car_dynamics.cuh:486-505`, `step_kernel.cu:162` | `ang_vel_x..z`, `q_w..z` | `pitch`, `yaw`, `roll` | Strictly gated on `num_wheels_contact == 0` and `!is_auto_flipping`. Pitch (130), Yaw (95), Roll (400) torques with input-scaled damping. |
| **Air Control: Air Throttle** | ✅ Full | `Car.cpp:680-682`, `RLConst.h:92` | `car_dynamics.cuh:452-454` | None | `throttle` | $200/3\text{ UU/s}^2$ ($\approx 66.67\text{ UU/s}^2$) forward acceleration in air. Active whenever $num\_wheels < 3$. |
| **Ground Stabilization: Auto-Roll** | ✅ Full | `Car.cpp:834-868`, `RLConst.h:132-133` | `car_dynamics.cuh:581-638`, `step_kernel.cu:174-179` | `world_contact_*` | `throttle` | Downforce $100\text{ UU}$ and torque $80$ aligning chassis with surface normal when $1 \le wheels \le 3$ or roof world contact. |
| **Turtle Recovery: Auto-Flip** | ✅ Full | `Car.cpp:798-832`, `RLConst.h:126-130` | `car_dynamics.cuh:523-575`, `step_kernel.cu:168` | `is_auto_flipping`, `auto_flip_timer`, `auto_flip_torque_scale`, `world_contact_*` | `jump` | Pop impulse $200\text{ UU/s}$ and roll torque $50$ when upside down on roof with jump pressed. Suppresses air control and double jump. |
| **Handbrake: Slew Rate** | ✅ Full | `Car.cpp:361-368`, `RLConst.h:81-82` | `car_dynamics.cuh:150-159` | `handbrake_val` | `handbrake` | Rise rate $+5.0$/s, fall rate $-2.0$/s. Range $[0, 1]$. |
| **Handbrake: Steer Expansion** | ✅ Full | `Car.cpp:420-425`, `RLConst.h:438-443` | `car_dynamics.cuh:196-201` | `wheel_steer_angle` | `steer`, `handbrake` | Linearly blends max steer angle towards powerslide curve by `handbrake_val`. |
| **Handbrake: Friction Reduction** | ✅ Full | `Car.cpp:459-466`, `RLConst.h:483-494` | `car_dynamics.cuh:228-232` | `wheel_lat_friction_*`, `wheel_long_friction_*` | `handbrake` | Lateral friction reduced to $10\%$ at full handbrake ($1 - 0.9 \cdot \text{val}$). Longitudinal scaled by curve. |
| **Drive Curves: Drive Torque** | ✅ Full | `Car.cpp:377-414`, `RLConst.h:447-453` | `car_dynamics.cuh:47-52, 167-192` | `wheel_engine_force`, `wheel_brake` | `throttle` | Torque factor curve: $1.0 \to 0.1 \to 0.0$ at $1400/1410\text{ UU/s}$. $0.25\times$ scale if $< 3$ wheels in contact. |
| **Drive Curves: Reverse Braking** | ✅ Full | `Car.cpp:386-395`, `RLConst.h:89` | `car_dynamics.cuh:186` | `wheel_engine_force`, `wheel_brake` | `throttle` | Full braking ($1.0$) when reversing forward motion; engine throttle cut to $0.0$ when speed $> 0.01\text{ UU/s}$ (`BRAKING_NO_THROTTLE_SPEED_THRESH`). |
| **Drive Curves: Steer Angle** | ✅ Full | `Car.cpp:417-428`, `RLConst.h:418-427` | `car_dynamics.cuh:54-62, 196` | `wheel_steer_angle` | `steer` | 6-point piecewise curvature matching speed. |
| **Sticky Downforce** | ✅ Full | `Car.cpp:485-497` | `car_dynamics.cuh:250-262` | None | None | Force along contact normal scaled by $0.5 + (1 - \|n_z\|)$ when throttle non-zero or speed $> 25\text{ UU/s}$. |
| **Speed & Angular Clamping** | ✅ Full | `Car.cpp:191-203`, `RLConst.h:53, 66` | `step_kernel.cu:208-214` | `vel_x..z`, `ang_vel_x..z` | None | Hard clamps at $2300\text{ UU/s}$ and $5.5\text{ rad/s}$. |
| **World Contact Normal Recording** | ✅ Full | `Arena.cpp:408-409` | `contact_solver.cuh:99-111`, `step_kernel.cu:200` | `world_contact_has_contact`, `world_contact_normal_x..z` | None | Analytical SDF chassis corner collisions populate contact flag and surface normals in `CarStateSoA` every tick. |

---

## 2. Structure of Arrays (SoA) State Variable Inventory

All car state attributes are laid out in contiguous, 128-byte aligned slices in `CarStateSoA` (`include/rocketsim_cuda/types/car_state.cuh`) pre-allocated within `SimContext::AllocateArena`:

| Category | Field Name | Data Type | Allocated in VRAM | Used in Kernel | Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Transform & Velocity** | `pos_x, pos_y, pos_z` | `float*` | ✅ | ✅ | World coordinates in Unreal Units (UU) |
| | `vel_x, vel_y, vel_z` | `float*` | ✅ | ✅ | Linear velocity in UU/s |
| | `q_w, q_x, q_y, q_z` | `float*` | ✅ | ✅ | Normalized orientation quaternion |
| | `ang_vel_x, ang_vel_y, ang_vel_z` | `float*` | ✅ | ✅ | Angular velocity in rad/s |
| | `pos_bt_x..z, vel_bt_x..z` | `float*` | ✅ | ✅ | Bullet unit representations for IEEE-754 bit parity |
| **Boost** | `boost` | `float*` | ✅ | ✅ | Fuel amount $[0.0, 100.0]$ |
| | `time_since_boosted` | `float*` | ✅ | ✅ | Time elapsed since last boost activation |
| | `boosting_time` | `float*` | ✅ | ✅ | Current boost duration (enforcing min $0.1$s) |
| | `is_boosting` | `uint8_t*` | ✅ | ✅ | Boolean flag active while boosting |
| **Suspension & Wheels** | `is_on_ground` | `uint8_t*` | ✅ | ✅ | True if $\ge 3$ wheels have contact |
| | `wheel_contact_0..3` | `uint8_t*` | ✅ | ✅ | Per-wheel SDF raycast contact flags |
| | `suspension_length_0..3` | `float*` | ✅ | ✅ | Per-wheel suspension compression lengths in UU |
| | `wheel_engine_force` | `float*` | ✅ | ✅ | Cached drive force applied to wheels |
| | `wheel_brake` | `float*` | ✅ | ✅ | Cached brake force applied to wheels |
| | `wheel_steer_angle` | `float*` | ✅ | ✅ | Steer angle in radians |
| | `wheel_lat_friction_0..3` | `float*` | ✅ | ✅ | Per-wheel lateral friction coefficients |
| | `wheel_long_friction_0..3` | `float*` | ✅ | ✅ | Per-wheel longitudinal friction coefficients |
| **Jump** | `has_jumped` | `uint8_t*` | ✅ | ✅ | Flag indicating initial jump was performed |
| | `is_jumping` | `uint8_t*` | ✅ | ✅ | Flag active during continuous jump hold |
| | `jump_time` | `float*` | ✅ | ✅ | Duration since jump initiation |
| | `has_double_jumped` | `uint8_t*` | ✅ | ✅ | Flag indicating double jump occurred |
| | `air_time` | `float*` | ✅ | ✅ | Total time spent in air |
| | `air_time_since_jump` | `float*` | ✅ | ✅ | Time since jump release ($1.25$s window) |
| **Flip / Dodge** | `has_flipped` | `uint8_t*` | ✅ | ✅ | Flag indicating flip has occurred |
| | `is_flipping` | `uint8_t*` | ✅ | ✅ | Flag active during flip torque phase ($0.65$s) |
| | `flip_time` | `float*` | ✅ | ✅ | Timer tracking flip duration |
| | `flip_rel_torque_x..z` | `float*` | ✅ | ✅ | Relative torque directions for pitch/yaw/roll |
| **Handbrake** | `handbrake_val` | `float*` | ✅ | ✅ | Powerslide accumulator in $[0.0, 1.0]$ |
| **Auto-Flip** | `is_auto_flipping` | `uint8_t*` | ✅ | ✅ | Turtle recovery active flag |
| | `auto_flip_timer` | `float*` | ✅ | ✅ | Countdown timer for auto-flip roll torque |
| | `auto_flip_torque_scale` | `float*` | ✅ | ✅ | Directional roll torque scale ($\pm 1$) |
| **Supersonic** | `is_supersonic` | `uint8_t*` | ✅ | ✅ | Supersonic state flag |
| | `supersonic_time` | `float*` | ✅ | ✅ | Time spent in supersonic state |
| **Contacts** | `world_contact_has_contact` | `uint8_t*` | ✅ | ✅ | Chassis contact with arena SDF |
| | `world_contact_normal_x..z` | `float*` | ✅ | ✅ | Chassis world contact normal vector |
| **Controls History** | `last_controls_*` (8 vars) | `float*/uint8_t*` | ✅ | ✅ | Cached inputs from preceding tick |

---

## 3. Detailed Milestone 4.2 Remediation Deliverables

### 3.1 Air Control Wheel Contact Suppression
- **Reference**: `Car.cpp:125, 640-642`
- **Location**: `src/cuda/step_kernel.cu:162`, `include/rocketsim_cuda/physics/car_dynamics.cuh:449, 499`
- **Implementation**:
  - `allow_air_torque = (num_wheels_contact == 0)` passed to `update_car_air_control`.
  - Air rotational torques (pitch, yaw, roll) and input-scaled angular damping are strictly disabled when any wheel touches a surface ($1 \le wheels \le 2$).
  - Air throttle ($66.67\text{ UU/s}^2$) remains active whenever airborne ($wheels < 3$).
  - Active auto-flipping (`is_auto_flipping != 0`) also suppresses air control.

### 3.2 Reverse Braking Throttle Cutoff
- **Reference**: `Car.cpp:391`, `RLConst.h:89`
- **Location**: `include/rocketsim_cuda/physics/car_dynamics.cuh:47, 186`
- **Implementation**:
  - Defined `BRAKING_NO_THROTTLE_SPEED_THRESH = 0.01f`.
  - Lowered speed threshold from $100.0\text{ UU/s}$ to $0.01\text{ UU/s}$.
  - When reversing against forward motion ($v_{\text{fwd}} > 25\text{ UU/s}$ and throttle $< 0$), engine drive force is cut to $0.0$ and full brake ($1.0$) is applied.

### 3.3 Auto-Roll Surface Alignment
- **Reference**: `Car.cpp:834-868`, `RLConst.h:132-133`
- **Location**: `include/rocketsim_cuda/physics/car_dynamics.cuh:581-638`, `src/cuda/step_kernel.cu:174-179`
- **Implementation**:
  - Implemented `update_car_auto_roll(...)`.
  - Computes ground normal from average wheel contact normals, or world contact normal if airborne with chassis contact.
  - Applies ground-directed downforce ($100\text{ UU/s}^2$) and alignment torque ($80$) rolling and pitching the chassis flush with the surface when throttle is active and wheels are partially off ground ($1 \le wheels \le 3$) or roof touches world.

### 3.4 Auto-Flip Turtle Recovery
- **Reference**: `Car.cpp:798-832`, `RLConst.h:126-130`
- **Location**: `include/rocketsim_cuda/physics/car_dynamics.cuh:523-575`, `src/cuda/step_kernel.cu:168`
- **Implementation**:
  - Implemented `update_car_auto_flip(...)`.
  - Extracts roll angle matching Bullet's `Angle::FromRotMat`.
  - When upside down on roof ($|\text{roll}| > 2.8\text{ rad}$) with roof resting on ground ($n_z > 0.7071$) and jump button pressed:
    - Applies immediate upward impulse ($200\text{ UU/s}$) along $-basis.up$.
    - Sets `is_auto_flipping = 1` and runs directional roll torque ($50$) for $0.4 \times (|\text{roll}| / \pi)$ seconds.
    - Suppresses air control and double jump during auto-flip execution.

### 3.5 World Contact Normal Recording
- **Reference**: `Arena.cpp:408-409`
- **Location**: `include/rocketsim_cuda/physics/contact_solver.cuh:99-111`, `src/cuda/step_kernel.cu:181, 200`
- **Implementation**:
  - `resolve_chassis_arena_collision` records contact occurrence and averaged surface normal into `car_state.world_contact_has_contact` and `world_contact_normal_x/y/z`.
  - Enables auto-roll and auto-flip to consume chassis contacts on subsequent simulation ticks.
