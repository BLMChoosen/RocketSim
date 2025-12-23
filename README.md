**A Python library with JAX for simulating Rocket League games at maximum efficiency**

RocketSim-GPU is a complete simulation of Rocket League's gameplay logic and physics that is completely standalone.
RocketSim-GPU supports the game modes: Soccar, Hoops, Heatseeker, and Snowday.

# Speed
RocketSim-GPU is designed to run extremely fast, even when complex collisions and suspension calculations are happening every tick.
On an average PC running a single thread of RocketSim with two cars, RocketSim can simulate around 20 minutes of game time every second.
This means that with 12 threads running RocketSim, you can simulate around 10 days of game time every minute!

# Accuracy
RocketSim is not a perfectly accurate replication of Rocket League, but is close enough for most applications (such as training ML bots).
RocketSim is accurate enough to:
- *Train machine learning bots to SSL level (and probably beyond)*
- *Simulate different shots on the ball at different angles to find the best input combination*
- *Simulate air control to find the optimal orientation input*
- *Simulate pinches*

However, the tiny errors will accumulate over time, so RocketSim is best suited for simulation with consistent feedback.

## Installation
- Clone this repo and build it
- Use https://github.com/ZealanL/RLArenaCollisionDumper to dump all of Rocket League's arena collision meshes
- Move those assets into RocketSim's executing directory

## Documentation
Documentation is not avaliable yet.

## Performance Details
RocketSim-GPU already heavily outperforms the speed of RocketSim physics tick step with optimization.

Version performance comparison:
```
OS: Linux Ubuntu 24.04 LTS
CPU: 48vcpu
GPU: RTX PRO 6000 Blackwell
=================================
Arena: Default (Soccar)
Cars: 1 on each team (1v1)
=================================
Task: Train a model (same checkpoint, same RAM, same CPU and GPU)

RocketSim:

Multi-thread performance (calculated using average steps on the RocketSim threads) (1 min simulated):
v2.1.0 = 14.153 steps/s

RocketSim-GPU:
Using only GPU:
v1.0.0 = 14.78m steps/s
```

## Issues & PRs
Feel free to make issues and pull requests if you encounter any issues!

You can also contact me on Discord if you have questions: `Davi3320`

## Legal Notice
RocketSim-GPU was written to replicate Rocket League's game logic, but does not actually contain any code from the game.
To Epic Games/Psyonix: If any of you guys have an issue with this, let me know on Discord and we can resolve it.
