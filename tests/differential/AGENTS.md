# Differential Harness Protocol & Golden Master Tests

> **Scope:** Applies to differential and unit tests in `tests/differential/` and `tests/unit/`.

---

## 1. CPU Oracle as Single Source of Truth

* **Immutability:** The CPU oracle (`src/Sim`, `libsrc/bullet3-3.24`) MUST NEVER be modified. It is the final judge.
* **No Loosening:** Never loosen tolerance thresholds to force kernels to pass.
* **Mandatory Citation:** All ported physics code must reference original CPU source in `file:line` format in `docs/M5_NOTES.md`.
* **Compile-Measure Cycle Cap:** Maximum 8 cycles per feature. If divergence persists beyond 8 cycles, commit the best state and record residual error in `docs/M5_BLOCKERS.md`.

---

## 2. Per-Tick Tolerance Thresholds ($\Vert \Delta \Vert_\infty$)

For any step $t \in [0, T]$ from identical state $S_0$ and inputs:

| State Attribute | Unit | Max Allowed Delta |
| :--- | :--- | :--- |
| **Linear Position (Ball & Car)** | Unreal Units (UU) | $\le 10^{-4}\text{ UU}$ per tick |
| **Linear Velocity** | UU / s | $\le 10^{-3}\text{ UU/s}$ per tick |
| **Rotation (Quaternion)** | Normalized $w, x, y, z$ | $\le 10^{-5}$ per tick |
| **Angular Velocity** | rad / s | $\le 10^{-4}\text{ rad/s}$ per tick |
| **Suspension Compression** | UU | $\le 10^{-4}\text{ UU}$ per tick |
| **Boost Amount** | $[0, 100]$ | $0.0$ (Bit-exact float) |

---

## 3. Challenger & Auditor Directives

* **Challenger:** May only create harness scenarios comparing against the CPU oracle (max 3 per feature). Prohibited from creating pytest suites with fabricated assertions without oracle.
* **Auditor / Success Auditor:** Must re-run official commands (`scripts/build_and_test.ps1` or `differential_harness --check`) and paste raw stdout. Test counts and hashes are only valid when generated on the fly.
* **Replay File Format (`.rsgold`):**
  - Header: Magic `0x5253474D` ("RSGM"), Version, Number of Environments, Total Ticks.
  - Per Tick: `CarInput` array, followed by CPU ground-truth `BallState` and `CarState` structs.
