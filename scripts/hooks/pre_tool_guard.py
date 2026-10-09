#!/usr/bin/env python3
"""
RocketSim-CUDA - Lifecycle Hook: PreToolUse Token Guard
Intercepts tool calls to prevent massive file dumps and guides subagents
toward focused semantic queries via MCP or line-range slicing.
"""

import sys
import os
import json
from pathlib import Path

MAX_UNCONSTRAINED_LINES = 150
MAX_SLICE_LINES = 250
INSPECTABLE_EXTENSIONS = {".cu", ".cuh", ".cpp", ".h", ".hpp", ".py", ".md", ".txt"}

def evaluate_tool_call(payload: dict) -> dict:
    tool_call = payload.get("toolCall", {})
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {})

    if tool_name != "view_file":
        return {"decision": "allow"}

    file_path_str = args.get("AbsolutePath", "")
    if not file_path_str:
        return {"decision": "allow"}

    path = Path(file_path_str)
    if not path.is_file() or path.suffix.lower() not in INSPECTABLE_EXTENSIONS:
        return {"decision": "allow"}

    # Line count check
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        return {"decision": "allow"}

    start_line = args.get("StartLine")
    end_line = args.get("EndLine")

    # Case 1: Unconstrained read on large file
    if (start_line is None or end_line is None) and total_lines > MAX_UNCONSTRAINED_LINES:
        return {
            "decision": "deny",
            "reason": (
                f"[Token Guard] Massive file read blocked ({path.name} has {total_lines} lines). "
                f"To conserve token budget in /teamwork-preview, use the MCP 'code-search' tool "
                f"(search_symbols or get_file_outline) to locate target code, or invoke view_file "
                f"with StartLine and EndLine (max slice: {MAX_SLICE_LINES} lines)."
            )
        }

    # Case 2: Slice exceeds allowed window
    if start_line is not None and end_line is not None:
        slice_size = end_line - start_line + 1
        if slice_size > MAX_SLICE_LINES:
            return {
                "decision": "deny",
                "reason": (
                    f"[Token Guard] Requested slice of {slice_size} lines exceeds maximum allowed {MAX_SLICE_LINES} lines. "
                    f"Please reduce the line range in view_file to protect context window budget."
                )
            }

    return {"decision": "allow"}

def run_tests():
    print("=== Testing PreToolUse Token Guard ===")
    
    # 1. Large file without StartLine/EndLine (must deny)
    big_file = str(Path(__file__).resolve().parent.parent.parent / "include/rocketsim_cuda/types/car_state.cuh")
    test_req_deny = {
        "toolCall": {
            "name": "view_file",
            "args": {
                "AbsolutePath": big_file
            }
        }
    }
    res_deny = evaluate_tool_call(test_req_deny)
    print("Test 1 (Unconstrained read on large file):", res_deny["decision"])
    assert res_deny["decision"] == "deny"

    # 2. Large file with slice (must allow)
    test_req_allow = {
        "toolCall": {
            "name": "view_file",
            "args": {
                "AbsolutePath": big_file,
                "StartLine": 40,
                "EndLine": 85
            }
        }
    }
    res_allow = evaluate_tool_call(test_req_allow)
    print("Test 2 (Targeted 45-line slice):", res_allow["decision"])
    assert res_allow["decision"] == "allow"

    # 3. Non-inspected tool (must allow)
    test_req_other = {
        "toolCall": {
            "name": "run_command",
            "args": {
                "CommandLine": "pytest"
            }
        }
    }
    res_other = evaluate_tool_call(test_req_other)
    print("Test 3 (Other tool):", res_other["decision"])
    assert res_other["decision"] == "allow"

    print("=== All Token Guard tests passed successfully! ===")

def main():
    if "--test" in sys.argv:
        run_tests()
        return

    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            sys.stdout.write(json.dumps({"decision": "allow"}) + "\n")
            return
        payload = json.loads(raw_input)
        response = evaluate_tool_call(payload)
    except Exception as e:
        response = {"decision": "allow"}

    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
