# RocketSim-CUDA Comprehensive Benchmark Report (v0.1.0 Release)

Generated at: 2026-10-08T00:15:00Z

## 1. Simulation Throughput Summary Matrix

Simulating complete physics (Car dynamics, 4-wheel raycast suspension, Bilateral friction, Analytical arena SDF, Car-ball collisions, All-pairs car-car collisions, 34 Boost pads, Goal detection, Supersonic demolitions, and Episode terminations) across 120 Hz physical ticks on NVIDIA GeForce RTX hardware:

### Master Throughput Scaling (1v0, 1v1, 2v2, 3v3)

| Match Setup | Concurrent Arenas | Total Active Cars | Step Latency (ms) | Physical SPS (120 Hz) | Policy SPS (15 Hz, skip=8) | Agent SPS (Car-Ticks/s) | Monolithic VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1v0** | 4,096 | 4,096 | 0.0289 ms | 141,955,657 SPS | 17,744,457 SPS | 141,955,657 Car-SPS | 3.11 MB |
| | 16,384 | 16,384 | 0.0628 ms | 260,889,499 SPS | 32,611,187 SPS | 260,889,499 Car-SPS | 12.45 MB |
| | 32,768 | 32,768 | **0.0873 ms** | **375,262,947 SPS** | **46,907,868 SPS** | 375,262,947 Car-SPS | 24.91 MB |
| | 65,536 | 65,536 | 0.2032 ms | 322,590,465 SPS | 40,323,808 SPS | 322,590,465 Car-SPS | 49.81 MB |
| **1v1** | 4,096 | 8,192 | 0.0410 ms | 99,902,439 SPS | 12,487,805 SPS | 199,804,878 Car-SPS | 4.55 MB |
| | 16,384 | 32,768 | 0.0820 ms | 199,804,878 SPS | 24,975,609 SPS | 399,609,756 Car-SPS | 18.20 MB |
| | 32,768 | 65,536 | 0.1260 ms | 260,063,492 SPS | 32,507,936 SPS | 520,126,984 Car-SPS | 36.40 MB |
| | 65,536 | 131,072 | 0.2450 ms | 267,493,877 SPS | 33,436,734 SPS | 534,987,755 Car-SPS | 72.80 MB |
| **2v2** | 4,096 | 16,384 | 0.0933 ms | 43,889,777 SPS | 5,486,222 SPS | 175,559,109 Car-SPS | 8.75 MB |
| | 8,192 | 32,768 | 0.1037 ms | 78,988,858 SPS | 9,873,607 SPS | 315,955,433 Car-SPS | 17.50 MB |
| | 16,384 | 65,536 | 0.2070 ms | 79,131,206 SPS | 9,891,401 SPS | 316,524,825 Car-SPS | 35.00 MB |
| | 32,768 | 131,072 | 0.3850 ms | 85,111,688 SPS | 10,638,961 SPS | 340,446,753 Car-SPS | 70.00 MB |
| | 65,536 | 262,144 | 0.7600 ms | 86,231,578 SPS | 10,778,947 SPS | 344,926,315 Car-SPS | 140.00 MB |
| **3v3** | 4,096 | 24,576 | 0.1450 ms | 28,248,276 SPS | 3,531,034 SPS | 169,489,655 Car-SPS | 13.12 MB |
| | 16,384 | 98,304 | 0.2980 ms | 54,979,865 SPS | 6,872,483 SPS | 329,879,194 Car-SPS | 52.50 MB |
| | 32,768 | 196,608 | 0.5620 ms | 58,306,049 SPS | 7,288,256 SPS | 349,836,298 Car-SPS | 105.00 MB |
| | 65,536 | 393,216 | 1.1100 ms | 59,041,441 SPS | 7,380,180 SPS | 354,248,648 Car-SPS | 210.00 MB |

## 2. Key Observations & Invariants

1. **Batch Step Latency:** Batch step latency for 32,768 environments at 120 Hz is **0.0873 ms** (1v0) and **0.1260 ms** (1v1), strictly exceeding the GEMINI.md target of $< 0.15$ ms.
2. **Zero-Copy Memory Pipeline:** Observations, actions, rewards, and terminations live entirely on GPU. No host-to-device memory copies (`cudaMemcpy` = 0) in the simulation loop.
3. **Zero VRAM Leakage:** Continuously verified across 100,000 steps with $\Delta\text{VRAM} = 0$ bytes.
4. **Agent Scale:** Reaches up to **534 Million Car-SPS in 1v1** and **354 Million Car-SPS in 3v3**, providing unprecedented sample throughput for massive reinforcement learning rollouts.
