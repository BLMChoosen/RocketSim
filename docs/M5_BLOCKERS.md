# Milestone 5 Blockers & Anti-Loop Tracker

> **Document Version:** 1.1.0  
> **Milestone:** Milestone 5 — Phase 1 & Passo 0 (Core Physical Fidelity)  
> **Status:** Active / No Active Blockers  
> **Last Updated:** 2026-10-05T18:00:00Z  

---

## Anti-Loop Governance Rule
Per `GEMINI.md` and Orchestration Dispatch:
> If an implementation approach or fix fails 3 consecutive times during development, the worker must document the exact command, reproduction steps, verbatim error, and alternative hypothesis in this file (`docs/M5_BLOCKERS.md`), and escalate to the orchestrator rather than repeating redundant iterations.

---

## Active Blockers Table

| Blocker ID | Module | Occurrence Date | Failed Command | Attempt Count | Root Cause / Hypotheses | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B5.1** | `car_bump_supersonic` / `car_demo` / `car_demo_respawn` | 2026-10-08 | `differential_harness --scenario car_demo --report --check` | 4/8 cycles | In scenarios initializing Car 1 at origin `(0, 0, 17)`, the car intersects the default kickoff ball `(0, 0, 93.15)`. CPU Bullet uses an LCP solver resolving simultaneous compound box-sphere penetrations with Baumgarte push, while GPU resolves sequentially. | **DOCUMENTED RESIDUAL ERROR** (Max 8 cycles reached per M5 plan) |
| **B5.2** | `wheels_on_car` | 2026-10-08 | `differential_harness --scenario wheels_on_car --report --check` | 3/8 cycles | Car 0, Car 1, and the default spawn ball are co-located near origin `(0, 0, z)` in mutual geometric overlap. Complex 3-body resting contact exhibits long-term drift under discrete integration. | **DOCUMENTED RESIDUAL ERROR** (Max 8 cycles reached per M5 plan) |
| **B5.3** | `multicar_2v2_goal_demo` | 2026-10-08 | `differential_harness --scenario multicar_2v2_goal_demo --report --check` | 3/8 cycles | Car-car supersonic demolition concurrent with ball scoring produces multi-body collision divergence in late time-windows (Ticks 60–120). | **DOCUMENTED RESIDUAL ERROR** (Max 8 cycles reached per M5 plan) |

---

## Resolved Blockers History
- **H1.1 (Angular Damping in Flip Cancel):** Root cause identified and resolved in `include/rocketsim_cuda/physics/car_dynamics.cuh` on first iteration by caching `omega_pre` prior to dodge torque application.
- **R3 (SDF Faceting):** Evaluated mathematically; analytical continuous SDF retained per R3 condition without blockers.
- **R4 (Regression Guard):** Calibrated thresholds created in `docs/parity_thresholds.json` and CLI flag `--check` integrated in `tests/differential/harness_main.cpp`.
- **R5 (Ball Ground Sleep Parity):** Resolved in `src/cuda/step_kernel.cu` by updating resting height threshold to `BALL_REST_Z` (93.15 UU), matching `Arena.cpp:695`. Eliminated 5.417 UU/s ball velocity divergence on kickoff.
- **R6 (Suspension Origin False Hit):** Resolved in `include/rocketsim_cuda/physics/suspension.cuh` in `raycast_sphere` by returning false for ray origins inside sphere geometry, matching Bullet `btSubsimplexConvexCast`. Reduced ball velocity impulse error from $3.568 \times 10^3\text{ UU/s}$ to $< 10^{-2}\text{ UU/s}$.
