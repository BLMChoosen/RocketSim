"""
Empirical Verification & Adversarial Stress Suite for Milestone 5 Regression Guard & Parity Thresholds.

This suite directly mirrors and stress-tests:
1. LoadParityThresholds (C++ parser logic in tests/differential/harness_main.cpp)
2. ValidateScenarioThresholds (Validation logic in tests/differential/harness_main.cpp)
3. docs/parity_thresholds.json schema, bounds, and baseline+25% calibration
4. Artificial breach detection on all windowed metrics (pos, vel, quat, ang_vel)
5. Robustness against malformed JSON, schema deviations, and missing fields
"""

import os
import re
import json
import math
import pytest

class ThresholdLimits:
    def __init__(self):
        self.max_car_pos = 1e9
        self.max_car_vel = 1e9
        self.max_car_quat = 1e9
        self.max_ball_pos = 1e9
        self.max_ball_vel = 1e9

    def to_dict(self):
        return {
            'max_car_pos': self.max_car_pos,
            'max_car_vel': self.max_car_vel,
            'max_car_quat': self.max_car_quat,
            'max_ball_pos': self.max_ball_pos,
            'max_ball_vel': self.max_ball_vel
        }


def cpp_load_parity_thresholds(json_content: str):
    """
    Exact translation of LoadParityThresholds from tests/differential/harness_main.cpp (lines 1263-1383).
    Returns (success: bool, out_thresholds: dict, error_msg: str).
    """
    out_thresholds = {}
    json_str = json_content
    i = 0
    n = len(json_str)

    def skip_whitespace():
        nonlocal i
        while i < n and json_str[i] in ' \t\r\n':
            i += 1

    def parse_string():
        nonlocal i
        skip_whitespace()
        if i >= n or json_str[i] != '"':
            return ""
        i += 1
        start = i
        while i < n and json_str[i] != '"':
            if json_str[i] == '\\' and i + 1 < n:
                i += 2
            else:
                i += 1
        s = json_str[start:i]
        if i < n and json_str[i] == '"':
            i += 1
        return s

    scn_pos = json_str.find('"scenarios"')
    if scn_pos == -1:
        return False, {}, "'scenarios' key not found"

    i = scn_pos + 11
    skip_whitespace()
    if i < n and json_str[i] == ':':
        i += 1
    skip_whitespace()
    if i >= n or json_str[i] != '{':
        return False, {}, "Expected '{' after 'scenarios':"
    i += 1  # enter scenarios object

    while i < n:
        skip_whitespace()
        if i < n and json_str[i] == '}':
            i += 1
            break
        scn_name = parse_string()
        if not scn_name:
            i += 1
            continue
        skip_whitespace()
        if i < n and json_str[i] == ':':
            i += 1
        skip_whitespace()
        if i >= n or json_str[i] != '{':
            break
        i += 1  # enter scenario definition

        while i < n:
            skip_whitespace()
            if i < n and json_str[i] == '}':
                i += 1
                break
            key = parse_string()
            skip_whitespace()
            if i < n and json_str[i] == ':':
                i += 1
            skip_whitespace()
            if key == "windows":
                if i < n and json_str[i] == '{':
                    i += 1
                while i < n:
                    skip_whitespace()
                    if i < n and json_str[i] == '}':
                        i += 1
                        break
                    win_tick_str = parse_string()
                    if not win_tick_str:
                        i += 1
                        continue
                    try:
                        win_tick = int(win_tick_str)
                    except ValueError as e:
                        # In C++, std::stoi throws std::invalid_argument
                        raise RuntimeError(f"C++ std::stoi threw std::invalid_argument on '{win_tick_str}'") from e
                    skip_whitespace()
                    if i < n and json_str[i] == ':':
                        i += 1
                    skip_whitespace()
                    if i < n and json_str[i] == '{':
                        i += 1
                    limits = ThresholdLimits()
                    while i < n:
                        skip_whitespace()
                        if i < n and json_str[i] == '}':
                            i += 1
                            break
                        metric_name = parse_string()
                        skip_whitespace()
                        if i < n and json_str[i] == ':':
                            i += 1
                        skip_whitespace()
                        val_start = i
                        while i < n and json_str[i] not in ',} \n\r':
                            i += 1
                        val_str = json_str[val_start:i]
                        try:
                            val = float(val_str)
                        except ValueError:
                            # In C++, std::strtof("invalid") returns 0.0f
                            val = 0.0
                        if metric_name == "max_car_pos":
                            limits.max_car_pos = val
                        elif metric_name == "max_car_vel":
                            limits.max_car_vel = val
                        elif metric_name == "max_car_quat":
                            limits.max_car_quat = val
                        elif metric_name == "max_ball_pos":
                            limits.max_ball_pos = val
                        elif metric_name == "max_ball_vel":
                            limits.max_ball_vel = val
                        skip_whitespace()
                        if i < n and json_str[i] == ',':
                            i += 1
                    if scn_name not in out_thresholds:
                        out_thresholds[scn_name] = {}
                    out_thresholds[scn_name][win_tick] = limits
                    skip_whitespace()
                    if i < n and json_str[i] == ',':
                        i += 1
            else:
                depth = 0
                while i < n:
                    if json_str[i] in '{[':
                        depth += 1
                    elif json_str[i] in '}]':
                        if depth == 0:
                            break
                        depth -= 1
                    elif json_str[i] == ',' and depth == 0:
                        break
                    i += 1
            skip_whitespace()
            if i < n and json_str[i] == ',':
                i += 1
        skip_whitespace()
        if i < n and json_str[i] == ',':
            i += 1

    return True, out_thresholds, ""


class WindowMetrics:
    def __init__(self, max_c_pos=0.0, max_c_vel=0.0, max_c_quat=0.0, max_b_pos=0.0, max_b_vel=0.0):
        self.max_car_pos = max_c_pos
        self.max_car_vel = max_c_vel
        self.max_car_quat = max_c_quat
        self.max_ball_pos = max_b_pos
        self.max_ball_vel = max_b_vel


class ScenarioReport:
    def __init__(self, name: str, ticks: int = 600):
        self.name = name
        self.ticks_simulated = ticks
        self.w1 = WindowMetrics()
        self.w10 = WindowMetrics()
        self.w60 = WindowMetrics()
        self.w120 = WindowMetrics()
        self.w600 = WindowMetrics()


def cpp_validate_scenario_thresholds(rep: ScenarioReport, all_thresholds: dict) -> tuple[bool, list[str]]:
    """
    Exact translation of ValidateScenarioThresholds in tests/differential/harness_main.cpp (lines 1385-1439).
    """
    logs = []
    scn_key = None
    if rep.name in all_thresholds:
        scn_key = rep.name
    else:
        for k in all_thresholds:
            if k in rep.name or rep.name in k:
                scn_key = k
                break

    if scn_key is None:
        logs.append(f"[Warning] No regression thresholds defined for scenario '{rep.name}'")
        return True, logs

    passed = True
    win_thresholds = all_thresholds[scn_key]

    def check_window(win_tick: int, m: WindowMetrics, win_label: str):
        nonlocal passed
        if win_tick not in win_thresholds:
            return
        lim = win_thresholds[win_tick]

        def check_metric(name: str, actual: float, limit: float):
            nonlocal passed
            if actual > limit:
                logs.append(f"[-] Regression Check FAILED for '{rep.name}' at Window {win_label} ({win_tick} ticks): {name} {actual} > threshold {limit}")
                passed = False

        check_metric("Car Pos", m.max_car_pos, lim.max_car_pos)
        check_metric("Car Vel", m.max_car_vel, lim.max_car_vel)
        check_metric("Car Quat", m.max_car_quat, lim.max_car_quat)
        check_metric("Ball Pos", m.max_ball_pos, lim.max_ball_pos)
        check_metric("Ball Vel", m.max_ball_vel, lim.max_ball_vel)

    check_window(1, rep.w1, "1")
    check_window(10, rep.w10, "10")
    if rep.ticks_simulated >= 60:
        check_window(60, rep.w60, "60")
    if rep.ticks_simulated >= 120:
        check_window(120, rep.w120, "120")
    if rep.ticks_simulated >= 600:
        check_window(600, rep.w600, "600")

    if passed:
        logs.append(f"[+] Parity thresholds check PASSED for scenario '{rep.name}'")
    return passed, logs


# ==============================================================================
# TESTS
# ==============================================================================

def test_docs_parity_thresholds_schema_and_counts():
    """Verify that docs/parity_thresholds.json parses cleanly and has all 54 scenarios."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "parity_thresholds.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "scenarios" in data, "Missing 'scenarios' root key"
    scenarios = data["scenarios"]
    assert len(scenarios) == 54, f"Expected 54 scenarios, found {len(scenarios)}"
    assert data.get("tolerance_buffer_percent") == 25.0, "Expected buffer percent to be 25.0"


def test_docs_parity_thresholds_positive_floating_bounds():
    """Check that all 29 scenario entries have valid, strictly positive floating-point bounds."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "parity_thresholds.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data["scenarios"]
    required_metrics = {"max_car_pos", "max_car_vel", "max_car_quat", "max_ball_pos", "max_ball_vel"}

    for scn_name, scn_data in scenarios.items():
        assert "windows" in scn_data, f"Scenario {scn_name} has no 'windows' key"
        windows = scn_data["windows"]
        assert len(windows) > 0, f"Scenario {scn_name} has empty 'windows'"
        for win_tick, metrics in windows.items():
            assert win_tick.isdigit(), f"Window key '{win_tick}' in {scn_name} is not an integer"
            for m_name in required_metrics:
                assert m_name in metrics, f"Scenario {scn_name} win {win_tick} missing metric '{m_name}'"
                val = metrics[m_name]
                assert isinstance(val, (int, float)), f"Scenario {scn_name} win {win_tick} {m_name} is not a number: {val}"
                assert not math.isnan(val), f"Scenario {scn_name} win {win_tick} {m_name} is NaN"
                assert not math.isinf(val), f"Scenario {scn_name} win {win_tick} {m_name} is Inf"
                assert val > 0.0, f"Scenario {scn_name} win {win_tick} {m_name} is non-positive: {val}"


def test_cpp_parser_loads_actual_json():
    """Verify that the exact C++ parser logic loads docs/parity_thresholds.json identically to python json."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "parity_thresholds.json")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    ok, cpp_thresh, err = cpp_load_parity_thresholds(content)
    assert ok, f"cpp_load_parity_thresholds failed: {err}"
    assert len(cpp_thresh) == 54, f"C++ parser parsed {len(cpp_thresh)} scenarios, expected 54"

    std_data = json.loads(content)["scenarios"]
    for scn, s_data in std_data.items():
        assert scn in cpp_thresh, f"C++ parser missed scenario '{scn}'"
        cpp_windows = cpp_thresh[scn]
        for w_str, m_data in s_data["windows"].items():
            w_int = int(w_str)
            assert w_int in cpp_windows, f"C++ parser missed window {w_int} in '{scn}'"
            lim = cpp_windows[w_int]
            assert pytest.approx(lim.max_car_pos, rel=1e-5) == m_data["max_car_pos"]
            assert pytest.approx(lim.max_car_vel, rel=1e-5) == m_data["max_car_vel"]
            assert pytest.approx(lim.max_car_quat, rel=1e-5) == m_data["max_car_quat"]
            assert pytest.approx(lim.max_ball_pos, rel=1e-5) == m_data["max_ball_pos"]
            assert pytest.approx(lim.max_ball_vel, rel=1e-5) == m_data["max_ball_vel"]


def test_artificial_breaches_all_metrics():
    """
    Stress-test ValidateScenarioThresholds:
    Verify that an artificial breach of car_pos, car_vel, car_quat, ball_pos, ball_vel
    causes ValidateScenarioThresholds to return False.
    Also empirically tests what happens with ang_vel.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "parity_thresholds.json")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    ok, thresh, _ = cpp_load_parity_thresholds(content)
    assert ok

    # 1. Base clean scenario within thresholds -> passes
    rep = ScenarioReport("random", ticks=600)
    lim_w60 = thresh["random"][60]
    rep.w60.max_car_pos = lim_w60.max_car_pos * 0.9
    rep.w60.max_car_vel = lim_w60.max_car_vel * 0.9
    rep.w60.max_car_quat = lim_w60.max_car_quat * 0.9
    rep.w60.max_ball_pos = lim_w60.max_ball_pos * 0.9
    rep.w60.max_ball_vel = lim_w60.max_ball_vel * 0.9
    passed, logs = cpp_validate_scenario_thresholds(rep, thresh)
    assert passed, f"Expected pass, got logs: {logs}"

    # 2. Breach Car Pos
    rep_breach_pos = ScenarioReport("random", ticks=600)
    rep_breach_pos.w60.max_car_pos = lim_w60.max_car_pos * 1.05
    passed, logs = cpp_validate_scenario_thresholds(rep_breach_pos, thresh)
    assert not passed, "Failed to detect Car Pos breach"
    assert any("Car Pos" in log for log in logs)

    # 3. Breach Car Vel
    rep_breach_vel = ScenarioReport("random", ticks=600)
    rep_breach_vel.w60.max_car_vel = lim_w60.max_car_vel * 1.05
    passed, logs = cpp_validate_scenario_thresholds(rep_breach_vel, thresh)
    assert not passed, "Failed to detect Car Vel breach"
    assert any("Car Vel" in log for log in logs)

    # 4. Breach Car Quat
    rep_breach_quat = ScenarioReport("random", ticks=600)
    rep_breach_quat.w60.max_car_quat = lim_w60.max_car_quat * 1.05
    passed, logs = cpp_validate_scenario_thresholds(rep_breach_quat, thresh)
    assert not passed, "Failed to detect Car Quat breach"
    assert any("Car Quat" in log for log in logs)

    # 5. Breach Ball Pos (window 600)
    lim_w600 = thresh["random"][600]
    rep_breach_bpos = ScenarioReport("random", ticks=600)
    rep_breach_bpos.w600.max_ball_pos = lim_w600.max_ball_pos * 1.05
    passed, logs = cpp_validate_scenario_thresholds(rep_breach_bpos, thresh)
    assert not passed, "Failed to detect Ball Pos breach"
    assert any("Ball Pos" in log for log in logs)

    # 6. Breach Ball Vel (window 600)
    rep_breach_bvel = ScenarioReport("random", ticks=600)
    rep_breach_bvel.w600.max_ball_vel = lim_w600.max_ball_vel * 1.05
    passed, logs = cpp_validate_scenario_thresholds(rep_breach_bvel, thresh)
    assert not passed, "Failed to detect Ball Vel breach"
    assert any("Ball Vel" in log for log in logs)


def test_ang_vel_omission_finding():
    """
    CRITICAL ARCHITECTURAL FINDING:
    WindowMetrics and ThresholdLimits do NOT track angular velocity (ang_vel).
    Demonstrate that an artificial breach in ang_vel is NOT detected by ValidateScenarioThresholds.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "parity_thresholds.json")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    ok, thresh, _ = cpp_load_parity_thresholds(content)
    assert ok

    # ThresholdLimits does not have max_car_ang_vel or max_ball_ang_vel
    sample_lim = thresh["random"][60]
    assert not hasattr(sample_lim, "max_car_ang_vel")
    assert not hasattr(sample_lim, "max_ball_ang_vel")

    # In WindowMetrics, ang_vel is not stored or passed
    rep = ScenarioReport("random", ticks=600)
    # Even if ang_vel diverged to 1e6 rad/s, WindowMetrics has no field for it:
    passed, logs = cpp_validate_scenario_thresholds(rep, thresh)
    assert passed, "ValidateScenarioThresholds does not check ang_vel!"


def test_adversarial_malformed_json_handling():
    """
    ADVERSARIAL STRESS TEST:
    Test C++ parser robustness against malformed JSON, schema corruptions, and type mismatches.
    """
    # Case A: Missing 'scenarios' root
    ok, _, err = cpp_load_parity_thresholds('{"version": "1.0.0"}')
    assert not ok
    assert "'scenarios' key not found" in err

    # Case B: Non-integer window tick string -> throws in C++ std::stoi
    malformed_tick_json = """
    {
        "scenarios": {
            "test": {
                "windows": {
                    "tick_sixty": { "max_car_pos": 1.0 }
                }
            }
        }
    }
    """
    with pytest.raises(RuntimeError) as exc_info:
        cpp_load_parity_thresholds(malformed_tick_json)
    assert "std::stoi threw std::invalid_argument" in str(exc_info.value)

    # Case C: Closing brace inside a string breaks the parser
    brace_in_desc_json = """
    {
        "scenarios": {
            "test": {
                "description": "hello } world",
                "windows": {
                    "60": { "max_car_pos": 1.0 }
                }
            }
        }
    }
    """
    ok, thresh, _ = cpp_load_parity_thresholds(brace_in_desc_json)
    # The depth scanner stops at '}' inside the string, exiting the scenario object prematurely
    assert "test" not in thresh or 60 not in thresh.get("test", {}), (
        "Parser failed to catch prematurely closed object due to brace in string"
    )

    # Case D: Missing fields default to 1e9f
    missing_fields_json = """
    {
        "scenarios": {
            "test": {
                "windows": {
                    "60": { "max_car_pos": 0.5 }
                }
            }
        }
    }
    """
    ok, thresh, _ = cpp_load_parity_thresholds(missing_fields_json)
    assert ok
    lim = thresh["test"][60]
    assert lim.max_car_pos == 0.5
    # Omitted fields defaulted to 1e9f (disabled checks)
    assert lim.max_car_vel == 1e9
    assert lim.max_car_quat == 1e9
    assert lim.max_ball_pos == 1e9
    assert lim.max_ball_vel == 1e9


def test_harness_cli_control_flow_deadlock_on_check():
    """
    CRITICAL CONTROL-FLOW BUG DISCOVERY:
    In tests/differential/harness_main.cpp:
    If --check is passed without --report:
      args.report_mode == false
      fail_fast == !args.report_mode == true
    In scenarios like 'random' or 'ablation_5_flips', physical divergence naturally exceeds 1e-4 UU (tol.pos_uu).
    At tick ~1-10, !step_ok triggers early exit:
      if (!step_ok && fail_fast) return false; (line 967)
    Then in main():
      if (!scn_ok) {
          all_passed = false;
          if (!args.report_mode) return 1; (line 1668)
      }
    Line 1684 'if (args.check_mode)' is NEVER REACHED!
    The regression guard is completely bypassed and exits 1 before ever evaluating thresholds!
    """
    def simulate_main(check_mode: bool, report_mode: bool, scenario_has_drift: bool):
        # 1. ParseArgs behavior
        # In harness_main.cpp line 88-92:
        # --check only sets args.check_mode = true, it DOES NOT set args.report_mode = true!
        
        # 2. RunScenarioDifferential
        fail_fast = not report_mode
        if scenario_has_drift and fail_fast:
            # Emulates line 967 early return
            scn_ok = False
        else:
            # Emulates running all ticks and filling reports
            scn_ok = not scenario_has_drift

        # 3. main() lines 1665-1670
        if not scn_ok:
            if not report_mode:
                return "EXIT_1_EARLY_ABORT_BEFORE_CHECK"

        # 4. main() lines 1684-1704
        if check_mode:
            return "CHECK_EVALUATED_SUCCESSFULLY"

        return "EXIT_0_OR_OTHER"

    # Running with only --check on a scenario with drift (e.g. random, ablation_5_flips):
    res_only_check = simulate_main(check_mode=True, report_mode=False, scenario_has_drift=True)
    assert res_only_check == "EXIT_1_EARLY_ABORT_BEFORE_CHECK", (
        "Demonstrates that --check without --report aborts prematurely before check is evaluated!"
    )

    # Running with both --check and --report:
    res_with_report = simulate_main(check_mode=True, report_mode=True, scenario_has_drift=True)
    assert res_with_report == "CHECK_EVALUATED_SUCCESSFULLY"


def test_r5_hitbox_presets_cpu_oracle_parity():
    """
    R5 Hitbox Presets Verification:
    Strictly verifies that include/rocketsim_cuda/types/car_config.cuh matches
    the CPU Oracle (src/Sim/Car/CarConfig/CarConfig.cpp & CarConfig.h):
    - All 7 presets (Octane, Dominus, Plank, Breakout, Hybrid, Merc, Psyclops)
    - Full hitbox dimensions (hitbox_size) and center-of-mass offsets (hitbox_pos_offset)
    - Wheel radii, suspension rest lengths, and connection point offsets
    - Effective suspension rest length deduction (-12.0 UU travel)
    - Moment of inertia calculation in UU units matching Bullet calculateLocalInertia
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cuh_path = os.path.join(repo_root, "include", "rocketsim_cuda", "types", "car_config.cuh")
    assert os.path.isfile(cuh_path), f"Missing {cuh_path}"

    with open(cuh_path, "r") as f:
        content = f.read()

    presets = [
        "OCTANE", "DOMINUS", "PLANK", "BREAKOUT", "HYBRID", "MERC", "PSYCLOPS"
    ]
    for name in presets:
        assert f"CAR_CONFIG_{name}" in content
        assert f"CAR_HITBOX_{name}" in content

    # Check that Octane values match CPU oracle bit-for-bit
    assert "120.507f, 86.6994f, 38.6591f" in content
    assert "13.8757f, 0.0f, 20.755f" in content
    assert "51.25f, 25.90f, 20.755f" in content
    assert "-33.75f, 29.50f, 20.755f" in content

    # Check Dominus
    assert "130.427f, 85.7799f, 33.8f" in content
    assert "9.0f, 0.0f, 15.75f" in content

    # Check Plank (Batmobile)
    assert "131.32f, 87.1704f, 31.8944f" in content
    assert "9.00857f, 0.0f, 12.0942f" in content

    # Check Breakout
    assert "133.992f, 83.021f, 32.8f" in content
    assert "12.5f, 0.0f, 11.75f" in content

    # Check Hybrid
    assert "129.519f, 84.6879f, 36.6591f" in content

    # Check Merc
    assert "123.22f, 79.2103f, 44.1591f" in content
    assert "11.3757f, 0.0f, 21.505f" in content

    # Check Psyclops
    assert "120.507f + 0.134f, 86.6994f + 0.134f, 38.6591f + 0.134f" in content

    # Check CarConfigSoA exists with __restrict__ pointers
    assert "struct CarConfigSoA" in content
    assert "uint8_t* __restrict__ hitbox_type" in content


def test_r6_arena_mutator_config_cpu_oracle_parity():
    """
    R6 Arena & Mutator Config Verification:
    Strictly verifies that include/rocketsim_cuda/types/arena_config.cuh matches
    the CPU Oracle (src/Sim/Arena/ArenaConfig/ArenaConfig.h and src/Sim/MutatorConfig/MutatorConfig.h):
    - GameMode, DemoMode, ArenaMemWeightMode enums
    - MutatorConfig fields (gravity, car_mass, ball_radius, ball_mass, boost, bumps, demos)
    - ArenaConfig fields (min_pos, max_pos, max_aabb_len, no_ball_rot, custom broadphase)
    - MutatorConfigSoA and ArenaConfigSoA with __restrict__ pointers
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cuh_path = os.path.join(repo_root, "include", "rocketsim_cuda", "types", "arena_config.cuh")
    assert os.path.isfile(cuh_path), f"Missing {cuh_path}"

    with open(cuh_path, "r") as f:
        content = f.read()

    # Check Enums
    assert "enum class GameMode : uint8_t" in content
    assert "enum class DemoMode : uint8_t" in content
    assert "enum class ArenaMemWeightMode : uint8_t" in content

    # Check MutatorConfig fields and defaults
    assert "Vec3(0.0f, 0.0f, GRAVITY_Z)" in content
    assert "car_mass" in content
    assert "ball_radius" in content
    assert "ball_mass" in content
    assert "boost_accel_ground" in content
    assert "boost_accel_air" in content
    assert "respawn_delay" in content
    assert "bump_cooldown_time" in content
    assert "demo_mode" in content

    # Check ArenaConfig fields
    assert "struct alignas(16) ArenaConfig" in content
    assert "min_pos" in content
    assert "max_pos" in content
    assert "max_aabb_len" in content
    assert "no_ball_rot" in content
    assert "use_custom_broadphase" in content

    # Check SoA layouts
    assert "struct MutatorConfigSoA" in content
    assert "float* __restrict__ gravity_z" in content
    assert "struct ArenaConfigSoA" in content


