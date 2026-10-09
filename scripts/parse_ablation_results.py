import re
import os

scenarios = [
    (1, "ablation_1_discrete_throttle_steer", "(1) throttle e steer discretos"),
    (2, "ablation_2_analog_throttle_steer", "(2) throttle e steer analógicos"),
    (3, "ablation_3_plus_boost", "(3) + boost"),
    (4, "ablation_4_plus_handbrake", "(4) + handbrake"),
    (5, "ablation_5_plus_single_jump", "(5) + jump simples"),
    (6, "ablation_6_plus_double_jump", "(6) + double jump"),
    (7, "ablation_7_plus_dodge_flip", "(7) + dodge/flip"),
    (8, "ablation_8_plus_air_control", "(8) + controle aéreo"),
    (9, "ablation_9_plus_landings", "(9) + pouso e transições"),
    (10, "ablation_10_random_full", "(10) random completo"),
]

def parse_log(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        raw = f.read()
    if b"\x00" in raw[:100]:
        text = raw.decode("utf-16", errors="ignore")
    else:
        text = raw.decode("utf-8", errors="ignore")

    # Find **Tick 60** section
    idx = text.find("**Tick 60**")
    if idx == -1:
        return None
    sub = text[idx:idx+4000]

    # Pattern for table rows:
    # | Car Pos X         | 2.773e-01 | 8.660e-01 | ...
    metrics = {}
    for line in sub.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 3:
            comp = parts[0]
            try:
                med = float(parts[1])
                p95 = float(parts[2])
                metrics[comp] = (med, p95)
            except ValueError:
                continue

    return metrics

print("| Configuração | Pos Med (UU) | Pos P95 (UU) | Vel Med (UU/s) | Vel P95 (UU/s) | Quat Med | Quat P95 | Culpada (>1 UU)? |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

for num, scn, desc in scenarios:
    log_path = f"logs/verify/ablation_{num}.txt"
    m = parse_log(log_path)
    if not m:
        print(f"| {desc} | N/A | N/A | N/A | N/A | N/A | N/A | Pendente |")
        continue

    # Worst pos across X, Y, Z
    pos_med = max(m.get("Car Pos X", (0,0))[0], m.get("Car Pos Y", (0,0))[0], m.get("Car Pos Z", (0,0))[0])
    pos_p95 = max(m.get("Car Pos X", (0,0))[1], m.get("Car Pos Y", (0,0))[1], m.get("Car Pos Z", (0,0))[1])

    # Worst vel across X, Y, Z
    vel_med = max(m.get("Car Vel X", (0,0))[0], m.get("Car Vel Y", (0,0))[0], m.get("Car Vel Z", (0,0))[0])
    vel_p95 = max(m.get("Car Vel X", (0,0))[1], m.get("Car Vel Y", (0,0))[1], m.get("Car Vel Z", (0,0))[1])

    # Worst quat across W, X, Y, Z
    quat_med = max(m.get("Car Quat W", (0,0))[0], m.get("Car Quat X", (0,0))[0], m.get("Car Quat Y", (0,0))[0], m.get("Car Quat Z", (0,0))[0])
    quat_p95 = max(m.get("Car Quat W", (0,0))[1], m.get("Car Quat X", (0,0))[1], m.get("Car Quat Y", (0,0))[1], m.get("Car Quat Z", (0,0))[1])

    culprit = "**SIM (> 1 UU)**" if pos_med > 1.0 else "Não (<= 1 UU)"
    print(f"| {desc} | {pos_med:.4f} | {pos_p95:.4f} | {vel_med:.4f} | {vel_p95:.4f} | {quat_med:.4e} | {quat_p95:.4e} | {culprit} |")
