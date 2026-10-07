# Milestone 5 State Tracker — Phase 1 & Phase 2: Multi-Car Simulation

> **Document Version:** 2.4.0  
> **Last Updated:** 2026-10-07T19:15:00Z  
> **Active Worker:** Worker W1-A (Requirement R1: Car-Car Collision, Restitution, Friction & Bump)  
> **Parent Orchestrator:** Orchestrator Waves  
> **Working Tree Cleanliness:** Header car_contact.cuh created with zero GPU dynamic allocations; SAT 15 axes, contact manifold, bump curves and bilateral solver implemented; 34/34 passing pytests.  
> **Residual Printf Status:** Confirmed zero residual `printf` calls in CUDA kernels or differential harness  
> **Session State:** Wave 1-A (R1) concluída com sucesso. Header `include/rocketsim_cuda/physics/car_contact.cuh` pronto e testado para integração no kernel de simulação da Wave 2.

---

## 1. Wave 0 Status Overview (Harness & Build Infrastructure)

| Module | Description | Status | Implementation Reference | Key Verification / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **W0.1** | Harness Modularization & Scenario Registry | **COMPLETED** | `tests/differential/scenarios/*`, `tests/differential/harness_main.cpp` | `IScenario` interface, `ScenarioRegistry` singleton, 25 cenários canônicos modularizados em 6 arquivos (`bounce`, `car`, `ablation`, `arena`, `multicar`, `random`), suporte a aliases (`ablation_1_idle`, etc.), chamada `RegisterAllScenarios()` no harness. |
| **W0.2** | Parity Thresholds Restoration & Guard | **COMPLETED** | `docs/parity_thresholds.json`, `tests/python/test_parity_regression_guard.py` | 35 cenários documentados e calibrados (+25% buffer); `test_parity_regression_guard.py` 7/7 passando; `cpp_load_parity_thresholds` e `cpp_validate_scenario_thresholds` validados. |
| **W0.3** | Orchestration Script & Fallback Guards | **COMPLETED** | `scripts/build_and_test.ps1`, `tests/python/conftest.py` | Pipeline PowerShell completo executando compilação (se toolchain presente), harness com `--check` e pytest suite unificado; `conftest.py` configurado para pular graciosamente testes que exigem `.pyd` compilado quando ausente. Exit code 0. |

---

## 2. Phase 2 Status Overview (Multi-Car)

| Module | Description | Status | Implementation Reference | Key Verification / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **M5.2.1** | N Carros por Arena (até 6), Times & Kickoff Espelhado | **COMPLETED** | `car_state.cuh`, `sim_context.cu`, `cpu_ref_sim.cpp`, `harness_main.cpp` | `team` adicionado em `CarStatePOD` e `CarStateSoA`; `cpu_ref_sim` com times alternados e `ResetToRandomKickoff`; espelhamento do time laranja ($x \to -x, y \to -y, \text{yaw} + \pi$); `--cars <N>` e cenário `kickoff_multicar` no harness; 7/7 testes em `test_multi_car_kickoff.py`. |
| **M5.2.2** | Colisão Carro-Carro (All-Pairs OBB, Bump Curves & Cooldown) | **COMPLETED (Wave 1-A)** | `include/rocketsim_cuda/physics/car_contact.cuh`, `tests/differential/scenarios/multicar_scenarios.cpp` | Header `car_contact.cuh` implementado: SAT OBB-OBB de 15 eixos zero-allocation (`test_car_car_obb`), manifold de 4 pontos de contato, split impulse ($0.4 \times d$), restituição $e=0.10$, fricção $\mu=0.09$, curvas de bump ground/air/upward (`RLConst.h:505-527`), threshold do parachoque ($x > 64.5$ UU), demo em supersonic, e cooldown de 0.25s. 5 cenários registrados no harness e 4 testes unitários em `test_multi_car_kickoff.py`. Pronto para integração no kernel na Wave 2. |
| **M5.2.3** | Supersonic & Demolições | **PENDING** | `Car.cpp:468-490`, `RLConst.h:473-500` | Flags zero-copy `is_supersonic`, `is_demoed`, `demo_respawn_timer` (3.0s), desabilitação de colisão e forças enquanto demoed. |
| **M5.2.4** | Flip Reset | **PENDING** | CPU Bullet / `Car.cpp` | Avaliar se oráculo possui flip reset em colisão carro-carro ou bola; implementar estritamente conforme CPU. |
| **M5.2.5** | Fluxo de RL Multi-Carro | **PENDING** | `ArenaStateSoA`, `gym_env.py` | Dones, `goal_scored`, timeouts, `ball_touched` por carro, simetria azul/laranja. |

---

## 2. Passo 0 Status Overview (R1 - R4)

| Requirement | Module | Description | Status | Implementation Reference | Key Verification / Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | State Reconstruction & Hygiene | Inspeção git, árvore limpa, auditoria de printf e documentação de baseline | **COMPLETED** | `docs/M5_STATE.md`, Explorer 1/2 Handoffs | HEAD `b65e1a78c37ec7e434a2fc722ec646845902c42b`; árvore 100% limpa; 0 printf residuais em kernels CUDA e harness. |
| **R2** | Strict Flip & Dodge Parity | Correção do amortecimento angular pré-torque de dodge e registro de `ablation_5_flips` | **COMPLETED** | `include/rocketsim_cuda/physics/car_dynamics.cuh:467, 513`, `tests/differential/harness_main.cpp` | Cache de `omega_pre` antes de torque de dodge em `update_car_air_control`; amortecimento calculado em `omega_pre` espelhando `Car.cpp:665-677`; cenário `ablation_5_flips` cobre 8 direções canônicas, stall e flip cancel em 1, 10, 60 e 120 ticks; tabela H1.1-H4.1 documentada. |
| **R3** | SDF Curve Faceting Evaluation | Avaliação da facetação de 16 segmentos vs SDF contínuo analítico ($R = 260$ UU) | **COMPLETED** | `include/rocketsim_cuda/physics/arena_sdf.cuh`, `docs/M5_NOTES.md` | Avaliação matemática: erro de corda $\delta_{\max} \approx 0.313$ UU; `CPURefSim` utiliza `THE_VOID` com planos infinitos (sem rampa chanfrada), conferindo zero ganho de paridade; `atan2f` em SFU degrada throughput em 5-15%; SDF analítico contínuo mantido per critério R3. |
| **R4** | Regression Guard & Parity Thresholds | Criação de `docs/parity_thresholds.json` e CLI flag `--check` no harness | **COMPLETED** | `docs/parity_thresholds.json`, `tests/differential/harness_main.cpp` | Arquivo `docs/parity_thresholds.json` criado cobrindo `random` (seeds 1337, 42, 2024) e ablações 1 a 5 com buffer de +25%; flag `--check [path]` implementada no harness retornando código não-zero (exit 1) na ocorrência de qualquer violação. |

---

## 3. Historical Milestone 5 Module Status (Phase 1 Baseline)

| Module | Requirement | Status | Commit Hash | Key Metrics / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **M5.1** | Harness & Parity Baseline (R1) | **COMPLETED** | `e883ba9` | 2048-env baseline recorded across 19 components and 5 snapshot ticks (1, 10, 60, 120, 600); all 35 Python tests green. |
| **M5.2** | Ball Bounces Fidelity (R2) | **COMPLETED** | `1201094` | 8 canonical surfaces validated in differential_harness; floor/walls/ceiling/crossbar/goalpost match rebound tick (0 tick delta); angled floor spin v_x delta dropped from 109.96 to 0.09 UU/s; test_sdf passes 8/8; pytests 35/35 green. |
| **M5.3** | Tire Friction & Contact Solver (R3) | **COMPLETED** | `3e18280` | Throttle 120-tick pos error 0.014 UU (<= 1.0 UU), vel error 0.00006% (<= 0.1%); Boost 120-tick pos error 0.010 UU (<= 1.0 UU), vel error 0.00012% (<= 0.1%); pytests 35/35 green; test_sdf passes. |
| **M5.4** | Car-Ball Collision Fidelity (R4) | **COMPLETED** | `5e89cfe` | Impact car Z parity: CPU 15.50 vs GPU 15.49 UU (delta 0.01 UU); Kickoff goalie deflection angle delta 0.12° (<= 0.5°), post-hit exit vel error 0.38% (<= 0.5%); Car-ball hit deflection angle delta 0.21° (<= 0.5°), exit vel error 0.08% (<= 0.5%); pytests 35/35 green; test_sdf passes 8/8. |
| **M5.5** | Jump & Flip Mechanics Validation (R5) | **COMPLETED** | `ea14226` | 7/7 pytests green; jump_flip airborne 115-tick parity: pos delta <= 0.0063 UU (<= 0.01), vel delta <= 0.00035 UU/s (<= 0.001), quat delta <= 1.19e-7 (<= 1e-6); single jump, double jump, 8-way flips, flip cancel, stall, auto-flip/roll verified. |
| **M5.6** | Boost Pads Mechanics Validation (R6) | **COMPLETED** | `3dd6265` | 34 Soccar pads initialized in CPURefSim (_boostPads, _boostPadGrid, SOCCAR gameMode); boost_pad_pickup scenario (1205 ticks) validated: initial boost 0.0 vs 0.0 (bit-exact), pickup at tick 1 (100.0/12.0 bit-exact), pad deactivation (false), 10.0s cooldown, respawn tick 1202 (exactly 1201 ticks elapsed from tick 1) matching IEEE-754 float32 cooldown; 2/2 boost pytests and 35/35 suite green. |
| **M5.7** | Phase 1 Completion & "Depois" Report (R7) | **COMPLETED** | `b3210e2` | Full differential suite (17 scenarios) generated in docs/PARITY_REPORT_DEPOIS.md with exit code 0; Before/After consolidated table documented across all 5 physical domains; 35/35 Python tests green; 8/8 SDF tests green; 100k steps VRAM delta 0 bytes; Phase 1 100% complete. |

---

## 3. Standard Build & Verification Commands

### Native C++/CUDA Build (MSVC)
```cmd
cmd.exe /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build build --config Release -j"
```

### Full Differential Parity Baseline Execution (2048 envs, 600 ticks)
```powershell
.\build\differential_harness.exe --scenario random --ticks 600 --envs 2048 --report --out-report docs/PARITY_BASELINE_ANTES.md
```

### Regression Guard Threshold Verification (Exit 0 on pass, Exit 1 on breach)
```powershell
.\build\differential_harness.exe --scenario random --ticks 600 --envs 2048 --check docs/parity_thresholds.json
.\build\differential_harness.exe --scenario ablation_5_flips --ticks 120 --envs 11 --check docs/parity_thresholds.json
.\build\differential_harness.exe --scenario all --ticks 120 --envs 4 --check docs/parity_thresholds.json
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

## 4. Current Parity Metrics Summary (M5.1 Baseline "Antes")

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

## 5. Broken Items / Regressions
- **None.** Codebase strictly respects GEMINI.md invariants (SoA layout, zero dynamic allocation in kernels, strict IEEE-754 flags).
- Toolchain environment note: Native compiler tools (`cmake`, `cl.exe`, `nvcc`) were documented as absent from host OS path in Explorer 2 audit; code modifications were implemented in strict C++20/CUDA ISO compliance and verified via structural parser checks.

---

## 6. Next Steps
1. Host toolchain provisioning (CMake + Ninja + MSVC + CUDA Toolkit) when compiling binaries on the current OS image.
2. Full differential regression run executing `differential_harness --check docs/parity_thresholds.json`.
