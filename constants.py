"""
Physics Constants from src/RLConst.h
====================================
These are the EXACT values extracted from the C++ codebase.
Units are in Unreal Units (UU) unless otherwise specified.
"""

from __future__ import annotations
import jax.numpy as jnp


# =============================================================================
# Core Physics
# =============================================================================
GRAVITY_Z = -650.0                      # Gravity in UU/s^2 (negative = down)
DT = 1.0 / 120.0                        # Physics tick rate: 120 Hz

# Unit conversion (Bullet Physics uses meters, RL uses Unreal Units)
# 1 Bullet Unit = 50 Unreal Units
BT_TO_UU = 50.0
UU_TO_BT = 1.0 / 50.0


# =============================================================================
# Arena Dimensions
# =============================================================================
ARENA_EXTENT_X = 4096.0                 # Half-width of arena
ARENA_EXTENT_Y = 5120.0                 # Half-length (not including goal)
ARENA_HEIGHT = 2048.0                   # Arena ceiling height


# =============================================================================
# Ball Constants
# =============================================================================
BALL_MASS = 30.0                        # CAR_MASS_BT / 6 = 180/6 = 30
BALL_RADIUS = 91.25                     # BALL_COLLISION_RADIUS_SOCCAR
BALL_MAX_SPEED = 6000.0                 # Maximum ball velocity
BALL_MAX_ANG_SPEED = 6.0                # Maximum angular velocity (rad/s)
BALL_DRAG = 0.03                        # Linear damping coefficient

# Ball collision properties
BALL_FRICTION = 0.35
BALL_RESTITUTION = 0.6                  # Bounce factor (coefficient of restitution)
BALL_REST_Z = 93.15                     # Ball rest position Z
BALL_WALL_RESTITUTION = 0.6             # Wall bounce factor
BALL_GROUND_RESTITUTION = 0.6           # Ground bounce factor
BALL_SURFACE_FRICTION = 0.35            # Tangential velocity reduction on collision


# =============================================================================
# Car Constants
# =============================================================================
CAR_MASS = 180.0                        # CAR_MASS_BT
CAR_MAX_SPEED = 2300.0                  # Maximum car velocity
CAR_MAX_ANG_SPEED = 5.5                 # Maximum angular velocity (rad/s)

# Boost constants
BOOST_MAX = 100.0                       # Maximum boost amount
BOOST_USED_PER_SECOND = BOOST_MAX / 3   # 33.33 boost/s consumption
BOOST_ACCEL_GROUND = 2975.0 / 3.0       # ~991.67 UU/s^2 on ground
BOOST_ACCEL_AIR = 3175.0 / 3.0          # ~1058.33 UU/s^2 in air
BOOST_SPAWN_AMOUNT = BOOST_MAX / 3      # Initial boost on spawn

# Throttle / Brake
THROTTLE_TORQUE_AMOUNT = CAR_MASS * 400.0    # 72000 UU
BRAKE_TORQUE_AMOUNT = CAR_MASS * (14.25 + 1/3)  # ~2565 UU
THROTTLE_AIR_ACCEL = 200.0 / 3.0        # ~66.67 UU/s^2 in air

# Jump constants
JUMP_ACCEL = 4375.0 / 3.0               # ~1458.33 UU/s^2
JUMP_IMMEDIATE_FORCE = 875.0 / 3.0      # ~291.67 UU/s
JUMP_MIN_TIME = 0.025                   # 25ms minimum jump
JUMP_MAX_TIME = 0.2                     # 200ms maximum jump
DOUBLEJUMP_MAX_DELAY = 1.25             # Seconds after jump ends

# Flip constants (from RLConst.h)
FLIP_TORQUE_TIME = 0.65                 # Duration of flip torque application
FLIP_TORQUE_MIN_TIME = 0.41             # Minimum time for full flip torque
FLIP_INITIAL_VEL_SCALE = 500.0          # Initial velocity impulse scale
FLIP_TORQUE_X = 260.0                   # Side flip torque (roll)
FLIP_TORQUE_Y = 224.0                   # Forward/back flip torque (pitch)
FLIP_Z_DAMP_120 = 0.35                  # Z velocity damping factor at 120fps
FLIP_Z_DAMP_START = 0.15                # Time to start Z damping
FLIP_Z_DAMP_END = 0.21                  # Time to stop Z damping
FLIP_PITCHLOCK_TIME = 1.0               # Pitch lock duration
FLIP_PITCHLOCK_EXTRA_TIME = 0.3         # Extra time for pitch lock after flip
FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE = 1.0
FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE = 1.9
FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE = 2.5
FLIP_BACKWARD_IMPULSE_SCALE_X = 16.0 / 15.0

# Dodge input deadzone
DODGE_DEADZONE = 0.5                    # Minimum stick input for flip vs double jump

# Jump timing reset
JUMP_RESET_TIME_PAD = 1.0 / 40.0        # Pad to avoid premature jump reset
JUMP_PRE_MIN_ACCEL_SCALE = 0.62         # Scale before JUMP_MIN_TIME

# Car torque scale (from RLConst.h - converts torque to usable angular accel)
# C++: CAR_TORQUE_SCALE = 2 * M_PI / (1 << 16) * 1000 = ~0.0958
CAR_TORQUE_SCALE = 2.0 * jnp.pi / 65536.0 * 1000.0  # ~0.0958

# Supersonic thresholds
SUPERSONIC_START_SPEED = 2200.0
SUPERSONIC_MAINTAIN_MIN_SPEED = 2100.0
SUPERSONIC_MAINTAIN_MAX_TIME = 1.0

# Air control (PYR = Pitch, Yaw, Roll order)
CAR_AIR_CONTROL_TORQUE = jnp.array([130.0, 95.0, 400.0])  # PYR
CAR_AIR_CONTROL_DAMPING = jnp.array([30.0, 20.0, 50.0])   # PYR

# Collision properties
CAR_COLLISION_FRICTION = 0.3
CAR_COLLISION_RESTITUTION = 0.1
CARCAR_COLLISION_FRICTION = 0.09        # From RLConst.h
CARCAR_COLLISION_RESTITUTION = 0.1      # From RLConst.h
CARBALL_COLLISION_FRICTION = 2.0
CARBALL_COLLISION_RESTITUTION = 0.0

# Bump mechanics (from RLConst.h)
BUMP_COOLDOWN_TIME = 0.25               # Minimum time between bumps
BUMP_MIN_FORWARD_DIST = 64.5            # Minimum forward distance for bump
DEMO_RESPAWN_TIME = 3.0                 # Respawn time after demo

# Bump velocity curves (from RLConst.h)
# Input: Forward speed of bumping car
# Output: Velocity impulse applied to bumped car
BUMP_VEL_AMOUNT_GROUND_SPEEDS = jnp.array([0.0, 1400.0, 2200.0])
BUMP_VEL_AMOUNT_GROUND_VALUES = jnp.array([5.0 / 6.0, 1100.0, 1530.0])

BUMP_VEL_AMOUNT_AIR_SPEEDS = jnp.array([0.0, 1400.0, 2200.0])
BUMP_VEL_AMOUNT_AIR_VALUES = jnp.array([5.0 / 6.0, 1390.0, 1945.0])

BUMP_UPWARD_VEL_AMOUNT_SPEEDS = jnp.array([0.0, 1400.0, 2200.0])
BUMP_UPWARD_VEL_AMOUNT_VALUES = jnp.array([2.0 / 6.0, 278.0, 417.0])

# Car-Ball Extra Impulse (RL's "power hit" mechanic)
# This extra impulse is applied on top of physics collision
BALL_CAR_EXTRA_IMPULSE_Z_SCALE = 0.35  # Reduces Z component of hit direction
BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE = 0.65  # Reduces forward component
BALL_CAR_EXTRA_IMPULSE_MAX_DELTA_VEL = 4600.0  # Max relative velocity for impulse

# Sticky forces (keeps car grounded when not moving/throttling on slopes)
# Applied as downward force along contact normal when grounded
STICKY_FORCE_SCALE_BASE = 0.5           # Base sticky force when stationary
STOPPING_FORWARD_VEL = 25.0             # Speed below which car is "stopping"

# Non-sticky friction factor curve (reduces friction when coasting on slopes)
# Input: contact normal Z component (1.0 = flat ground, 0 = vertical wall)
# Output: friction multiplier
NON_STICKY_FRICTION_CURVE_X = jnp.array([0.0, 0.7075, 1.0])  # Normal Z values
NON_STICKY_FRICTION_CURVE_Y = jnp.array([0.1, 0.5, 1.0])     # Friction multipliers

# Extra impulse factor curve: interpolates based on relative speed
# At 0-500 UU/s: 0.65, at 2300 UU/s: 0.55, at 4600 UU/s: 0.30
BALL_CAR_EXTRA_IMPULSE_FACTOR_SPEEDS = jnp.array([0.0, 500.0, 2300.0, 4600.0])
BALL_CAR_EXTRA_IMPULSE_FACTOR_VALUES = jnp.array([0.65, 0.65, 0.55, 0.30])


# =============================================================================
# Boost Pads (from RLConst.h BoostPads namespace)
# =============================================================================
# Pad geometry
PAD_CYL_HEIGHT = 95.0                   # Cylinder height for pickup detection
PAD_CYL_RAD_BIG = 208.0                 # Large pad cylinder radius
PAD_CYL_RAD_SMALL = 144.0               # Small pad cylinder radius

# Pad cooldowns and boost amounts
PAD_COOLDOWN_BIG = 10.0                 # Large pad respawn time (seconds)
PAD_COOLDOWN_SMALL = 4.0                # Small pad respawn time (seconds)
PAD_BOOST_AMOUNT_BIG = 100.0            # Boost from large pad
PAD_BOOST_AMOUNT_SMALL = 12.0           # Boost from small pad

# Number of pads (standard soccar)
N_PADS_SMALL = 28
N_PADS_BIG = 6
N_PADS_TOTAL = N_PADS_SMALL + N_PADS_BIG  # 34 total pads

# Small pad locations (28 pads) - from RLConst.h LOCS_SMALL_SOCCAR
PAD_LOCS_SMALL = jnp.array([
    [0.0, -4240.0, 70.0],
    [-1792.0, -4184.0, 70.0],
    [1792.0, -4184.0, 70.0],
    [-940.0, -3308.0, 70.0],
    [940.0, -3308.0, 70.0],
    [0.0, -2816.0, 70.0],
    [-3584.0, -2484.0, 70.0],
    [3584.0, -2484.0, 70.0],
    [-1788.0, -2300.0, 70.0],
    [1788.0, -2300.0, 70.0],
    [-2048.0, -1036.0, 70.0],
    [0.0, -1024.0, 70.0],
    [2048.0, -1036.0, 70.0],
    [-1024.0, 0.0, 70.0],
    [1024.0, 0.0, 70.0],
    [-2048.0, 1036.0, 70.0],
    [0.0, 1024.0, 70.0],
    [2048.0, 1036.0, 70.0],
    [-1788.0, 2300.0, 70.0],
    [1788.0, 2300.0, 70.0],
    [-3584.0, 2484.0, 70.0],
    [3584.0, 2484.0, 70.0],
    [0.0, 2816.0, 70.0],
    [-940.0, 3308.0, 70.0],
    [940.0, 3308.0, 70.0],
    [-1792.0, 4184.0, 70.0],
    [1792.0, 4184.0, 70.0],
    [0.0, 4240.0, 70.0],
])  # Shape: (28, 3)

# Large pad locations (6 pads) - from RLConst.h LOCS_BIG_SOCCAR
PAD_LOCS_BIG = jnp.array([
    [-3584.0, 0.0, 73.0],
    [3584.0, 0.0, 73.0],
    [-3072.0, 4096.0, 73.0],
    [3072.0, 4096.0, 73.0],
    [-3072.0, -4096.0, 73.0],
    [3072.0, -4096.0, 73.0],
])  # Shape: (6, 3)

# Combined pad locations: small pads first (0-27), then big pads (28-33)
PAD_LOCATIONS = jnp.concatenate([PAD_LOCS_SMALL, PAD_LOCS_BIG], axis=0)  # (34, 3)

# Pad radii (for each pad index)
PAD_RADII = jnp.concatenate([
    jnp.full((N_PADS_SMALL,), PAD_CYL_RAD_SMALL),
    jnp.full((N_PADS_BIG,), PAD_CYL_RAD_BIG),
])  # (34,)

# Pad boost amounts
PAD_BOOST_AMOUNTS = jnp.concatenate([
    jnp.full((N_PADS_SMALL,), PAD_BOOST_AMOUNT_SMALL),
    jnp.full((N_PADS_BIG,), PAD_BOOST_AMOUNT_BIG),
])  # (34,)

# Pad cooldowns
PAD_COOLDOWNS = jnp.concatenate([
    jnp.full((N_PADS_SMALL,), PAD_COOLDOWN_SMALL),
    jnp.full((N_PADS_BIG,), PAD_COOLDOWN_BIG),
])  # (34,)

# Pad type flags (True = big pad)
PAD_IS_BIG = jnp.concatenate([
    jnp.zeros((N_PADS_SMALL,), dtype=jnp.bool_),
    jnp.ones((N_PADS_BIG,), dtype=jnp.bool_),
])  # (34,)


# =============================================================================
# Goal Detection (from RLConst.h)
# =============================================================================
GOAL_THRESHOLD_Y = 5124.25              # Ball Y beyond this = goal scored
GOAL_BLUE_Y = -GOAL_THRESHOLD_Y         # Blue goal is at negative Y
GOAL_ORANGE_Y = GOAL_THRESHOLD_Y        # Orange goal is at positive Y


# =============================================================================
# Suspension / Vehicle (BTVehicle)
# =============================================================================
# NOTE: Values below are scaled for JAX simulation stability at 120Hz.
# Original C++ values work with Bullet's internal substeps.
SUSPENSION_STIFFNESS = 16000.0          # Scaled for stability (C++ BTVehicle: 500.0)
WHEELS_DAMPING_COMPRESSION = 800.0      # Scaled for stability (C++ BTVehicle: 25.0)
WHEELS_DAMPING_RELAXATION = 1280.0      # Scaled for stability (C++ BTVehicle: 40.0)
MAX_SUSPENSION_TRAVEL = 12.0            # In UU (unchanged)
SUSPENSION_SUBTRACTION = 0.05

SUSPENSION_FORCE_SCALE_FRONT = 36.0 - 0.25
SUSPENSION_FORCE_SCALE_BACK = 54.0 + 0.25 + 0.015


# =============================================================================
# Octane Car Config (Default)
# =============================================================================
OCTANE_HITBOX_SIZE = jnp.array([120.507, 86.6994, 38.6591])
OCTANE_HITBOX_OFFSET = jnp.array([13.8757, 0.0, 20.755])

# Wheel radii
OCTANE_FRONT_WHEEL_RADIUS = 12.50
OCTANE_BACK_WHEEL_RADIUS = 15.00

# Suspension rest lengths
OCTANE_FRONT_SUS_REST = 38.755
OCTANE_BACK_SUS_REST = 37.055

# Wheel connection offsets (X, Y, Z) - Y is positive, negated for right side
OCTANE_FRONT_WHEEL_OFFSET = jnp.array([51.25, 25.90, 20.755])
OCTANE_BACK_WHEEL_OFFSET = jnp.array([-33.75, 29.50, 20.755])


# =============================================================================
# Wheel Configuration (Octane - 4 wheels)
# =============================================================================
# Wheel order: FL, FR, BL, BR (Front-Left, Front-Right, Back-Left, Back-Right)
WHEEL_LOCAL_OFFSETS = jnp.array([
    [51.25, 25.90, 20.755],    # FL
    [51.25, -25.90, 20.755],   # FR (Y negated)
    [-33.75, 29.50, 20.755],   # BL
    [-33.75, -29.50, 20.755],  # BR (Y negated)
])

# Wheel radii for each wheel (front/back differ)
WHEEL_RADII = jnp.array([12.50, 12.50, 15.00, 15.00])  # FL, FR, BL, BR

# Suspension rest lengths for each wheel
SUSPENSION_REST_LENGTHS = jnp.array([38.755, 38.755, 37.055, 37.055])  # FL, FR, BL, BR

# Suspension force scale (front/back differ)
SUSPENSION_FORCE_SCALES = jnp.array([
    SUSPENSION_FORCE_SCALE_FRONT,  # FL
    SUSPENSION_FORCE_SCALE_FRONT,  # FR
    SUSPENSION_FORCE_SCALE_BACK,   # BL
    SUSPENSION_FORCE_SCALE_BACK,   # BR
])

# Drive wheels mask (which wheels receive throttle force)
DRIVE_WHEEL_MASK = jnp.array([1.0, 1.0, 1.0, 1.0])

# Ground plane Z (simplified - flat ground)
GROUND_Z = 0.0

# Car spawn Z (wheels just touching ground)
CAR_SPAWN_Z = 17.0

# Inertia tensor approximation (box inertia for Octane hitbox)
_hitbox = OCTANE_HITBOX_SIZE
CAR_INERTIA = jnp.array([
    (1/12) * CAR_MASS * (_hitbox[1]**2 + _hitbox[2]**2),  # Ixx (roll)
    (1/12) * CAR_MASS * (_hitbox[0]**2 + _hitbox[2]**2),  # Iyy (pitch)
    (1/12) * CAR_MASS * (_hitbox[0]**2 + _hitbox[1]**2),  # Izz (yaw)
])


# =============================================================================
# Tire Force Constants
# =============================================================================
TIRE_DRIVE_FORCE = THROTTLE_TORQUE_AMOUNT  # 72000 UU - full engine torque
TIRE_FRICTION_COEF = 1.0    # Simplified tire friction coefficient

# Drive force curve (reduces engine power at high speed)
DRIVE_TORQUE_CURVE_SPEEDS = jnp.array([0.0, 1400.0, 1410.0])
DRIVE_TORQUE_CURVE_FACTORS = jnp.array([1.0, 0.1, 0.0])


# =============================================================================
# Steering Constants
# =============================================================================
# Full curve from C++: speed -> max steer angle (radians)
STEER_ANGLE_CURVE_SPEEDS = jnp.array([0.0, 500.0, 1000.0, 1500.0, 1750.0, 3000.0])
STEER_ANGLE_CURVE_ANGLES = jnp.array([0.53356, 0.31930, 0.18203, 0.10570, 0.08507, 0.03454])

# Legacy simplified values (kept for reference)
MAX_STEER_ANGLE = 0.53356  # Max steering angle at low speed (radians)
MIN_STEER_ANGLE = 0.03454  # Min steering angle at high speed (radians)
STEER_SPEED_THRESHOLD = 1500.0  # Speed at which steering becomes restricted

# Front wheel indices (which wheels steer)
FRONT_WHEEL_MASK = jnp.array([1.0, 1.0, 0.0, 0.0])  # FL, FR steer; BL, BR don't


# =============================================================================
# Tire Friction Constants
# =============================================================================
LATERAL_FRICTION_BASE = 1.0       # Base lateral friction coefficient
LATERAL_FRICTION_MIN = 0.2        # Minimum lateral friction at full slip
LONGITUDINAL_FRICTION = 1.0       # Longitudinal friction coefficient
FRICTION_FORCE_SCALE = CAR_MASS / 3.0  # ~60 per wheel

# Handbrake friction reduction
HANDBRAKE_LAT_FRICTION_FACTOR = 0.1   # Multiply lateral friction by this when drifting
HANDBRAKE_LONG_FRICTION_FACTOR = 0.5  # Longitudinal friction factor during handbrake

# Rolling resistance (small braking when coasting)
ROLLING_RESISTANCE = 0.02  # Small velocity damping when not throttling

# Brake force
BRAKE_FORCE = BRAKE_TORQUE_AMOUNT  # ~2565 UU braking force


# =============================================================================
# RL Environment Constants
# =============================================================================
NORM_POS = ARENA_EXTENT_X               # Position normalizer (~4096)
NORM_VEL = CAR_MAX_SPEED                # Velocity normalizer (2300)
NORM_ANG_VEL = CAR_MAX_ANG_SPEED        # Angular velocity normalizer (5.5)
NORM_BOOST = BOOST_MAX                  # Boost normalizer (100)

# Observation sizes
OBS_SIZE_BALL = 9
OBS_SIZE_CAR = 21
OBS_SIZE_BALL_RELATIVE = 6

# Kickoff spawn positions
KICKOFF_BLUE_DIAGONAL_LEFT = jnp.array([-2048.0, -2560.0, 17.0])
KICKOFF_BLUE_DIAGONAL_RIGHT = jnp.array([2048.0, -2560.0, 17.0])
KICKOFF_BLUE_OFFCENTER_LEFT = jnp.array([-256.0, -3840.0, 17.0])
KICKOFF_BLUE_OFFCENTER_RIGHT = jnp.array([256.0, -3840.0, 17.0])
KICKOFF_BLUE_GOALIE = jnp.array([0.0, -4608.0, 17.0])

KICKOFF_ORANGE_DIAGONAL_LEFT = jnp.array([2048.0, 2560.0, 17.0])
KICKOFF_ORANGE_DIAGONAL_RIGHT = jnp.array([-2048.0, 2560.0, 17.0])
KICKOFF_ORANGE_OFFCENTER_LEFT = jnp.array([256.0, 3840.0, 17.0])
KICKOFF_ORANGE_OFFCENTER_RIGHT = jnp.array([-256.0, 3840.0, 17.0])
KICKOFF_ORANGE_GOALIE = jnp.array([0.0, 4608.0, 17.0])

KICKOFF_POSITIONS_BLUE = jnp.stack([
    KICKOFF_BLUE_DIAGONAL_LEFT,
    KICKOFF_BLUE_DIAGONAL_RIGHT,
    KICKOFF_BLUE_OFFCENTER_LEFT,
    KICKOFF_BLUE_OFFCENTER_RIGHT,
    KICKOFF_BLUE_GOALIE,
], axis=0)

KICKOFF_POSITIONS_ORANGE = jnp.stack([
    KICKOFF_ORANGE_DIAGONAL_LEFT,
    KICKOFF_ORANGE_DIAGONAL_RIGHT,
    KICKOFF_ORANGE_OFFCENTER_LEFT,
    KICKOFF_ORANGE_OFFCENTER_RIGHT,
    KICKOFF_ORANGE_GOALIE,
], axis=0)

# Kickoff facing angles (yaw in radians)
KICKOFF_YAW_BLUE = jnp.pi / 2           # 90 degrees (facing +Y)
KICKOFF_YAW_ORANGE = -jnp.pi / 2        # -90 degrees (facing -Y)
