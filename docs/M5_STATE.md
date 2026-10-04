# Milestone 5 State Tracker — Phase 1: Core Physical Fidelity

> **Document Version:** 1.1.0  
> **Last Updated:** 2026-10-04T17:10:00Z  
> **Active Worker:** Worker M5.2 (Próximo: Quiques da Bola / Ball Bounces)  
> **Parent Orchestrator:** Orchestrator M5  
> **Plan Reference:** [M5_PLAN.md](file:///C:/Users/Choosen/Documents/Estudo-Executor/RocketSim/docs/M5_PLAN.md)  
> **Session State:** Resumido com sucesso após interrupção. Working tree limpo, 35/35 testes verdes.

---

## 1. Status Overview

| Module | Requirement | Status | Commit Hash | Key Metrics / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **M5.1** | Harness & Parity Baseline (R1) | **COMPLETED** | `e883ba9` | 2048-env baseline recorded across 19 components and 5 snapshot ticks (1, 10, 60, 120, 600); all 35 Python tests green. |
| **M5.2** | Ball Bounces Fidelity (R2) | **COMPLETED** | `1201094` | 8 canonical surfaces validated in differential_harness; floor/walls/ceiling/crossbar/goalpost match rebound tick (0 tick delta); angled floor spin v_x delta dropped from 109.96 to 0.09 UU/s; test_sdf passes 8/8; pytests 35/35 green. |
| **M5.3** | Tire Friction & Contact Solver (R3) | Pending | - | Gauss-Seidel solver mirror, target $\le 0.1\%$ vel, $\le 1$ UU pos at 120 ticks. |
| **M5.4** | Car-Ball Collision Fidelity (R4) | Pending | - | Target $\le 0.5\%$ vel, $\le 0.5^\circ$ deflection angle; resolve car Z height at impact. |
| **M5.5** | Jump & Flip Mechanics Validation (R5) | Pending | - | 8-way directional flips, cancels, stalls against Bullet. |
| **M5.6** | Boost Pads Mechanics Validation (R6) | Pending | - | Pickup detection, respawn, boost gain parity. |
| **M5.7** | Phase 1 Completion & "Depois" Report (R7) | Pending | - | Before/after table comparison, Victory audit report. |

---

## 2. Standard Build & Verification Commands

### Native C++/CUDA Build (MSVC)
```cmd
cmd.exe /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build build --config Release -j"
```

### Full Differential Parity Baseline Execution (2048 envs, 600 ticks)
```powershell
.\build\differential_harness.exe --scenario random --ticks 600 --envs 2048 --report --out-report docs/PARITY_BASELINE_ANTES.md
```

### Fast Smoke Verification (4 envs, 10 ticks)
```powershell
.\build\differential_harness.exe --scenario random --ticks 10 --envs 4 --report
```

### Python Zero-Copy & Physics Unit Test Suite
```powershell
pytest tests/python/ -v
```

---

## 3. Current Parity Metrics Summary (M5.1 Baseline "Antes")

Evaluated on 2,048 parallel environments running `random` scenario across 600 ticks:

| Metric | Tick 1 (Med / P95) | Tick 10 (Med / P95) | Tick 60 (Med / P95) | Tick 120 (Med / P95) | Tick 600 (Med / P95) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Car Pos X (UU)** | 0.000 / 0.000 | 7.324e-4 / 1.929e-2 | 13.37 / 37.75 | 49.22 / 124.9 | 846.5 / 2946.0 |
| **Car Pos Y (UU)** | 0.000 / 0.000 | 1.465e-3 / 9.277e-3 | 29.38 / 66.95 | 79.35 / 261.1 | 1010.0 / 4052.0 |
| **Car Pos Z (UU)** | 0.000 / 0.000 | 1.984e-4 / 6.927e-3 | 3.497 / 23.01 | 9.712 / 51.34 | 35.14 / 211.0 |
| **Car Vel X (UU/s)** | 4.60e-8 / 5.34e-5 | 3.059e-2 / 1.357 | 55.75 / 130.5 | 76.26 / 387.3 | 384.8 / 1175.0 |
| **Car Vel Y (UU/s)** | 2.98e-8 / 6.10e-5 | 9.766e-4 / 0.601 | 113.1 / 301.0 | 103.3 / 441.5 | 405.9 / 1382.0 |
| **Car Quat (W,X,Y,Z)** | $\le 6.0\text{e}-8$ | $\le 5.4\text{e}-4$ / $1.7\text{e}-3$ | $\le 3.8\text{e}-2$ / $3.2\text{e}-1$ | $\le 6.3\text{e}-2$ / $6.2\text{e}-1$ | $\le 0.34$ / $1.00$ |
| **Ball Pos (X,Y,Z)** | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| **Ball Vel (X,Y,Z)** | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

---

## 4. Broken Items / Regressions
- **None.** All 35/35 python tests pass cleanly. `differential_harness` compiles without error and operates across thousands of environments.

---

## 5. Next Immediate Steps (Phase 1 Sequential Roadmap)
1. **M5.1 Concluído e Commitado:** Hashes `e883ba9` e `0865c0b`.
2. **M5.2 Concluído (Ball Bounces Fidelity - R2):**
   - 8 superfícies canônicas validadas em `differential_harness` com quiques de 1 impacto.
   - Quique tick idêntico (0 tick delta) em chão, paredes laterais, paredes de fundo, teto, travessão e traves.
   - Erro de velocidade tangencial com rotação reduzido em 1200x (de $109.96$ para $0.09\text{ UU/s}$).
   - Teste de perturbação CPU vs CPU comprova estabilidade física não-caótica.
3. **Próximo Módulo (M5.3):** Atrito Longitudinal & Solver de Contato do Carro (R3)
   - Espelhar `btVehicleRL` e a ordem do `btSequentialImpulseConstraintSolver`.
   - Meta: velocidade $\le 0.1\%$ e posição $\le 1$ UU em 120 ticks (throttle e boost).
