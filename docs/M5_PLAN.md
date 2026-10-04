# Milestone 5 Plan — RocketSim-CUDA Parity & Multi-Car Simulation

> **Objetivo:** Máxima semelhança com o RocketSim CPU (Bullet 3.24) para bots em 1v1 e 2v2 (3v3 se couber).

---

## Regras de Governança & Execução
- **Oráculo Canônico:** O oráculo é o código CPU do repositório (`src/Sim/*`, `RLConst.h`, `Car.cpp`, `Arena.cpp`, `btVehicleRL.cpp`, `libsrc/bullet3-3.24`): ler, copiar lógica, constantes e ORDEM das operações; nunca inventar valores.
- **Rastreabilidade:** Registrar em `docs/M5_NOTES.md` o `arquivo:linha` do CPU espelhado e o plano de cada módulo. Se o CPU não implementa algo (ex.: flip reset), documentar "inexistente no oráculo" e não implementar.
- **Atualização de Estado:** Manter `docs/M5_STATE.md` atualizado ao fim de cada módulo (feito + hash, quebrado, próximo passo, comandos de build e teste).
- **Anti-Loop:** A mesma hipótese falhando 3 vezes → parar, registrar em `docs/M5_BLOCKERS.md` (comando + erro exato) e seguir. Paridade por isolamento: 1 env, 1 tick, sem input, estágio a estágio.
- **Tolerâncias:** PROIBIDO alterar testes, tolerâncias ou harness para caber no kernel; o CPU decide. Se um teste codificava comportamento errado, corrigir para o valor do CPU e mostrar o diff.
- **Invariantes do GEMINI.md:** Zero-copy, SoA, sem alocação dinâmica na GPU, sem `cudaMemcpy` H<->D no step, FP determinístico (`-fmad=false`, etc.). Manter testes passando, pointer identity e 100k steps com $\Delta\text{VRAM} = 0$. Commits atômicos. Economizar quota: grep e leitura por faixa, relatórios curtos.

---

## FASE 1: Fidelidade do Núcleo
- **1.1 Harness & Linha de Base:** Erro por componente + cenário `random` (mediana e P95 em 1, 10, 60, 120, 600 ticks); registrar a linha de base "antes".
- **1.2 Quiques da Bola:** Quiques de UM quique (chão, chão angulado com spin, parede lateral, fundo, teto, rampa do canto, trave/travessão) vs CPU; CPU vs CPU com perturbação 1e-3 para separar caos de erro sistemático.
- **1.3 Atrito Longitudinal & Solver de Contato do Carro:** Espelhar `btVehicleRL` e a ordem do `btSequentialImpulseConstraintSolver`; meta: velocidade $\le 0.1\%$ e posição $\le 1$ UU em 120 ticks (throttle e boost).
- **1.4 Colisão Carro-Bola:** Erro por componente do vetor e do ângulo de saída em +1, +10, +60 ticks; investigar altura do carro no impacto (CPU 15.5 vs GPU 17.0); meta $\le 0.5\%$ e $\le 0.5^\circ$.
- **1.5 Flips & Manobras:** Jump, double jump, direções, cancel/stall se existir no CPU.
- **1.6 Boost Pads no Harness:** Coleta, cooldown, respawn.
- **→ PARAR e reportar ao fim da Fase 1.**

---

## FASE 2: Multi-Carro
- **2.1 N Carros por Arena:** Até 6 carros, times, kickoff com spawns idênticos ao CPU e espelhamento do time laranja.
- **2.2 Colisão Carro-Carro:** Como o CPU/Bullet (all-pairs por arena, bump callbacks).
- **2.3 Supersonic & Demolições:** Limiares e timers do `RLConst.h`, `DemoMode`, respawn; flags zero-copy `is_supersonic`, `is_demoed`, `demo_respawn_timer`.
- **2.4 Flip Reset:** Só se existir no CPU.
- **2.5 Fluxo de RL Multi-Carro:** Dones, `goal_scored`, timeouts, `ball_touched` por carro, simetria azul/laranja.
- **→ PARAR e reportar ao fim da Fase 2.**

---

## FASE 3: Polimento, Hitboxes & Auditoria Final (se a quota permitir)
- **3.1 Hitboxes:** Dominus, Plank, Breakout, Hybrid, Merc.
- **3.2 Auditoria Final:** Auditoria com cenário `random` (mediana e P95).
- **3.3 Benchmarks Honestos:** 1v1, 2v2, 3v3 (ticks/s e steps de política com `tick_skip=8`).
- **3.4 Documentação:** README e `PARITY_REPORT` sem alegações de "1:1" não comprovadas.
