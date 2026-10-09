import re
import csv

input_file = "logs/verify/check.txt"
output_file = "logs/verify/failures.csv"

pattern = re.compile(r"\[-\] Regression Check FAILED for '([^']+)' at Window (\d+) \(\d+ ticks\): ([^>]+) > threshold ([0-9.e+-]+)")

failures = []

try:
    with open(input_file, "r", encoding="utf-16", errors="ignore") as f:
        content = f.read()
    if "Regression Check" not in content:
        with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
except Exception:
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

for line in content.splitlines():
        match = pattern.search(line)
        if match:
            scenario = match.group(1)
            window = int(match.group(2))
            comp_val_str = match.group(3).strip()
            threshold_str = match.group(4).strip()
            
            # comp_val_str is like "Car Pos 2.025e-02"
            parts = comp_val_str.rsplit(" ", 1)
            if len(parts) == 2:
                comp = parts[0]
                val = float(parts[1])
            else:
                comp = comp_val_str
                val = 0.0
            
            thresh = float(threshold_str)
            ratio = val / thresh if thresh > 0 else 0.0
            
            classification = "(c) FÍSICA"
            notes = ""
            if window == 1:
                classification = "(a) SETUP"
                notes = "Tick 1 initial contact / setup mismatch"
            elif scenario.startswith("ablation_"):
                classification = "(b) LIMITE NÃO MEDIDO"
                notes = "Threshold unmeasured (< 1 UU target)"
            elif scenario in ["car_demo", "car_demo_respawn", "car_bump_supersonic"] and window == 60:
                classification = "(c) FÍSICA"
                notes = "Post-collision multi-body resolution drift"
            elif scenario == "config_heavy_ball":
                classification = "(b) LIMITE NÃO MEDIDO"
                notes = "Extreme mutator unmeasured threshold"
            
            failures.append({
                "scenario": scenario,
                "window": window,
                "worst_component": comp,
                "value": f"{val:.3e}",
                "threshold": f"{thresh:.3e}",
                "ratio": f"{ratio:.2f}",
                "classification": classification,
                "notes": notes
            })

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["scenario", "window", "worst_component", "value", "threshold", "ratio", "classification", "notes"])
    writer.writeheader()
    for row in failures:
        writer.writerow(row)

print(f"Generated {output_file} with {len(failures)} failures.")
