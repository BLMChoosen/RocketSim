# Milestone 5 State Tracker — Phase 1: Core Physical Fidelity

> **Document Version:** 1.3.0  
> **Last Updated:** 2026-10-04T18:55:00Z  
> **Active Worker:** Worker M5.6 (Finalizando: Validação de Boost Pads)  
> **Parent Orchestrator:** Orchestrator M5  
> **Plan Reference:** [M5_PLAN.md](file:///C:/Users/Choosen/Documents/Estudo-Executor/RocketSim/docs/M5_PLAN.md)  
> **Session State:** M5.6 validado com sucesso. Working tree pronto para commit, 35/35 testes verdes.

---

## 1. Status Overview

| Module | Requirement | Status | Commit Hash | Key Metrics / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **M5.1** | Harness & Parity Baseline (R1) | **COMPLETED** | `e883ba9` | 2048-env baseline recorded across 19 components and 5 snapshot ticks (1, 10, 60, 120, 600); all 35 Python tests green. |
| **M5.2** | Ball Bounces Fidelity (R2) | **COMPLETED** | `1201094` | 8 canonical surfaces validated in differential_harness; floor/walls/ceiling/crossbar/goalpost match rebound tick (0 tick delta); angled floor spin v_x delta dropped from 109.96 to 0.09 UU/s; test_sdf passes 8/8; pytests 35/35 green. |
| **M5.3** | Tire Friction & Contact Solver (R3) | **COMPLETED** | `3e18280` | Throttle 120-tick pos error 0.014 UU (<= 1.0 UU), vel error 0.00006% (<= 0.1%); Boost 120-tick pos error 0.010 UU (<= 1.0 UU), vel error 0.00012% (<= 0.1%); pytests 35/35 green; test_sdf passes. |
| **M5.4** | Car-Ball Collision Fidelity (R4) | **COMPLETED** | `5e89cfe` | Impact car Z parity: CPU 15.50 vs GPU 15.49 UU (delta 0.01 UU); Kickoff goalie deflection angle delta 0.12° (<= 0.5°), post-hit exit vel error 0.38% (<= 0.5%); Car-ball hit deflection angle delta 0.21° (<= 0.5°), exit vel error 0.08% (<= 0.5%); pytests 35/35 green; test_sdf passes 8/8. |
| **M5.5** | Jump & Flip Mechanics Validation (R5) | **COMPLETED** | `ea14226` | 7/7 pytests green; jump_flip airborne 115-tick parity: pos delta <= 0.0063 UU (<= 0.01), vel delta <= 0.00035 UU/s (<= 0.001), quat delta <= 1.19e-7 (<= 1e-6); single jump, double jump, 8-way flips, flip cancel, stall, auto-flip/roll verified. |
| **M5.6** | Boost Pads Mechanics Validation (R6) | **COMPLETED** | `3dd6265` | 34 Soccar pads initialized in CPURefSim (_boostPads, _boostPadGrid, SOCCAR gameMode); boost_pad_pickup scenario (1205 ticks) validated: initial boost 0.0 vs 0.0 (bit-exact), pickup at tick 1 (100.0/12.0 bit-exact), pad deactivation (false), 10.0s cooldown, respawn tick 1202 (exactly 1201 ticks elapsed from tick 1) matching IEEE-754 float32 cooldown; 2/2 boost pytests and 35/35 suite green. |
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
3. **M5.3 Concluído (Tire Friction & Contact Solver - R3):**
   - Reordenação do ciclo de execução em `StepCarsDevice`: raycasts -> dynamics com velocidades pré-impulso -> aplicação de impulsos de suspensão e atrito -> integração simplética.
   - Gating estrito de `extra_pushback` para `susp_force > 0.0f` no kernel de suspensão.
   - Frame unprojected `lat_dir` para cálculo de `base_friction` e `long_dir = lat_dir.cross(hit_normal)`.
   - Resultados a 120 ticks: Throttle pos error $0.014\text{ UU} \le 1.0\text{ UU}$, vel error $0.00006\% \le 0.1\%$; Boost pos error $0.010\text{ UU} \le 1.0\text{ UU}$, vel error $0.00012\% \le 0.1\%$.
4. **M5.4 Concluído (Colisão Carro-Bola - R4):**
   - Implementado Bullet `CONVEX_DISTANCE_MARGIN = 2.0f` (`inner_half = hitbox_half - 2.0f`, offset de ponto de contato).
   - Braço de alavanca da bola na superfície estrita da esfera $\mathbf{r}_b = -\mathbf{n}_{world} R_{ball}$.
   - Distribuição de push do split impulse ($80\%$ penetração: $6/7$ para bola, $1/7$ para carro) com escrita de posição do carro.
   - Aplicação de deslocamento de velocidade pós-solve ($\Delta Z_{vel} = v_z \Delta t$) ao carro na colisão em `StepSimulationKernel`.
   - Resultados empíricos:
     - `kickoff_goalie`: Carro Z no impacto CPU 15.50 vs GPU 15.49 UU ($\Delta = 0.01$ UU); ângulo de saída $\Delta = 0.12^\circ \le 0.5^\circ$; erro de velocidade de saída $0.38\% \le 0.5\%$.
     - `car_ball_hit`: Carro Z no impacto CPU 16.36 vs GPU 16.37 UU ($\Delta = 0.01$ UU); ângulo de saída $\Delta = 0.21^\circ \le 0.5^\circ$; erro de velocidade de saída $0.08\% \le 0.5\%$.
   - Testes unitários SDF (8/8) e Python (35/35) verdes.
5. **M5.5 Concluído (Jump & Flip Mechanics Validation - R5):**
   - Fórmulas exatas do oráculo espelhadas: impulso inicial ($875/3\text{ UU/s}$), aceleração de hold ($4375/3 \times 0.62$ e $1.0$), double jump (delay $1.25\text{ s}$), flips 8 direções com scaling de velocidade ($1.0, 2.5, 1.9, 16/15$), Z-damping ($0.35$), flip cancel ($1 - |\text{pitch}|$), stall, auto-roll e auto-flip.
   - Paridade aérea a 115 ticks: pos delta $\le 0.006348\text{ UU} \le 0.01\text{ UU}$, vel delta $\le 0.000355\text{ UU/s} \le 0.001\text{ UU/s}$, quat delta $\le 1.192 \times 10^{-7} \le 10^{-6}$.
6. **M5.6 Concluído (Boost Pads Mechanics Validation - R6):**
   - 34 boost pads de Soccar instanciados em `CPURefSim` (`_boostPads`, `_boostPadGrid`) com ativação do `GameMode::SOCCAR` no `Arena::Step`.
   - Adicionado cenário `boost_pad_pickup` (1205 ticks) no `differential_harness`:
     - Initial Boost: 0.0 vs 0.0 (Bit-exact, $\Delta = 0.0$).
     - Pickup Tick: 1 vs 1 (MATCH).
     - Post-Pickup Boost: 100.0 vs 100.0 (Bit-exact).
     - Pad Active Post-Pickup: false vs false (MATCH Deactivated).
     - Cooldown Atribuído: 10.0s vs 10.0s ($\Delta = 0.000\text{s}$).
     - Respawn Tick: 1202 vs 1202 (MATCH, exatamente 1201 ticks decorridos desde tick 1).
     - Duração de Cooldown: Exatamente 1201 ticks em float32 IEEE-754.
   - Testes Python de boost: 2/2 verdes (`pytest tests/python/ -k boost -v`).
   - Suíte Python completa: 35/35 testes verdes.
7. **Próximo Módulo (M5.7):** Fase 1 Completion & "Depois" Report (R7)
   - Execução do baseline completo de 2048 ambientes por 600 ticks (`--scenario random`).
   - Geração da tabela de comparação "Antes vs Depois" e relatório de auditoria de vitória.

