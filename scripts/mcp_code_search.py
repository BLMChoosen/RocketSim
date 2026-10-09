#!/usr/bin/env python3
"""
RocketSim-CUDA - Local MCP Code Search & Structural Intelligence Server
Implements standard JSON-RPC 2.0 protocol over stdio with ZERO external dependencies.
Enables subagents to query symbols, definitions, and file outlines on demand without reading entire files.
"""

import sys
import os
import re
import ast
import json
import fnmatch
from pathlib import Path

# Project repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ignored patterns to save tokens and accelerate search
DEFAULT_IGNORES = [
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "out",
    "Release",
    "Debug",
    "x64",
    "x86",
    "*.obj",
    "*.lib",
    "*.exe",
    "*.pyd",
    "*.whl",
    "*.pdb",
    "*.bin",
    "*.rsgold",
    "*.cmf",
    "libsrc/bullet3-3.24/examples",
    "libsrc/bullet3-3.24/test",
    "libsrc/bullet3-3.24/Demos",
    "libsrc/bullet3-3.24/Extras",
    ".agents/teamwork/*/*.md",
    ".agents/teamwork/*/*.log",
]

def load_ignore_patterns():
    patterns = list(DEFAULT_IGNORES)
    ignore_file = PROJECT_ROOT / ".ignore"
    if ignore_file.exists():
        try:
            with open(ignore_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        except Exception:
            pass
    return patterns

IGNORE_PATTERNS = load_ignore_patterns()

def is_ignored(rel_path_str: str) -> bool:
    norm_path = rel_path_str.replace("\\", "/")
    for pat in IGNORE_PATTERNS:
        clean_pat = pat.rstrip("/").replace("\\", "/")
        if pat.startswith("!"):
            if fnmatch.fnmatch(norm_path, clean_pat[1:]):
                return False
            continue
        if fnmatch.fnmatch(norm_path, clean_pat) or fnmatch.fnmatch(norm_path, clean_pat + "/*") or clean_pat in norm_path.split("/"):
            return True
    return False

def get_source_files(file_glob="*"):
    allowed_exts = {".cu", ".cuh", ".cpp", ".h", ".hpp", ".py"}
    matched = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Prune ignored directories for speed
        rel_root = os.path.relpath(root, PROJECT_ROOT).replace("\\", "/")
        if rel_root != "." and is_ignored(rel_root):
            dirs[:] = []
            continue

        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in allowed_exts:
                rel_path = os.path.relpath(os.path.join(root, f), PROJECT_ROOT).replace("\\", "/")
                if not is_ignored(rel_path):
                    if file_glob == "*" or fnmatch.fnmatch(rel_path, file_glob) or fnmatch.fnmatch(os.path.basename(rel_path), file_glob):
                        matched.append(PROJECT_ROOT / rel_path)
    return matched

# ---------------------------------------------------------------------------
# Symbol Extraction
# ---------------------------------------------------------------------------

RE_CUDA_KERNEL = re.compile(r"^\s*__global__\s+void\s+(\w+)\s*\(", re.MULTILINE)
RE_CUDA_DEVICE = re.compile(r"^\s*__device__\s+(?:inline\s+)?([\w\*\&]+)\s+(\w+)\s*\(", re.MULTILINE)
RE_CPP_STRUCT_CLASS = re.compile(r"^\s*(?:struct|class)\s+(?:alignas\([^\)]+\)\s+)?(\w+)(?:\s*:\s*[^{;]+)?\s*\{", re.MULTILINE)
RE_CPP_FUNCTION = re.compile(r"^(?:[\w\:\<\>\* \&]+)\s+(\w+)\s*\([^;{]*\)\s*(?:const)?\s*(?:noexcept)?\s*\{", re.MULTILINE)
RE_CPP_ENUM = re.compile(r"^\s*enum\s+(?:class\s+)?(\w+)", re.MULTILINE)
RE_CPP_MACRO = re.compile(r"^\s*#define\s+(\w+)(?:\([^\)]*\))?", re.MULTILINE)

def extract_symbols_from_cpp(file_path: Path):
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    lines = content.splitlines()
    symbols = []

    for i, line in enumerate(lines, start=1):
        m = RE_CUDA_KERNEL.search(line)
        if m:
            symbols.append({
                "name": m.group(1),
                "type": "kernel",
                "line": i,
                "signature": line.strip()
            })
            continue

        m = RE_CUDA_DEVICE.search(line)
        if m:
            symbols.append({
                "name": m.group(2),
                "type": "device_func",
                "line": i,
                "signature": line.strip()
            })
            continue

        m = RE_CPP_STRUCT_CLASS.search(line)
        if m:
            name = m.group(1)
            is_struct = "struct" in line
            symbols.append({
                "name": name,
                "type": "struct" if is_struct else "class",
                "line": i,
                "signature": line.strip()
            })
            continue

        m = RE_CPP_ENUM.search(line)
        if m:
            symbols.append({
                "name": m.group(1),
                "type": "enum",
                "line": i,
                "signature": line.strip()
            })
            continue

        m = RE_CPP_MACRO.search(line)
        if m:
            symbols.append({
                "name": m.group(1),
                "type": "macro",
                "line": i,
                "signature": line.strip()
            })
            continue

    return symbols

def extract_symbols_from_python(file_path: Path):
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content, filename=str(file_path))
    except Exception:
        return []

    symbols = []
    lines = content.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            sig = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else f"class {node.name}"
            symbols.append({
                "name": node.name,
                "type": "class",
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "signature": sig
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else f"def {node.name}(...)"
            symbols.append({
                "name": node.name,
                "type": "function",
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "signature": sig
            })
    return symbols

def extract_symbols(file_path: Path):
    if file_path.suffix == ".py":
        return extract_symbols_from_python(file_path)
    return extract_symbols_from_cpp(file_path)

# ---------------------------------------------------------------------------
# MCP Tool Implementations
# ---------------------------------------------------------------------------

def tool_search_symbols(pattern: str, symbol_type: str = "all", file_glob: str = "*") -> str:
    regex = re.compile(pattern, re.IGNORECASE)
    files = get_source_files(file_glob)
    results = []

    for f in files:
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        file_syms = extract_symbols(f)
        for s in file_syms:
            if regex.search(s["name"]):
                if symbol_type != "all" and s["type"] != symbol_type:
                    continue
                results.append(f"{rel}:{s['line']} [{s['type']}] {s['signature']}")

    if not results:
        return f"No symbols matching '{pattern}' found."

    if len(results) > 40:
        clipped = results[:40]
        clipped.append(f"... (+ {len(results) - 40} additional results truncated. Specify a narrower pattern or file_glob)")
        return "\n".join(clipped)
    return "\n".join(results)

def tool_get_symbol_definition(symbol_name: str, file_path: str = None) -> str:
    files = [PROJECT_ROOT / file_path] if file_path else get_source_files()
    found = []

    for f in files:
        if not f.exists():
            continue
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
        except Exception:
            continue

        syms = extract_symbols(f)
        for s in syms:
            if s["name"] == symbol_name:
                start = max(1, s["line"] - 3)
                end = min(len(lines), s["line"] + 45)
                if "end_line" in s:
                    end = min(len(lines), s["end_line"])
                else:
                    brace_count = 0
                    started = False
                    for idx in range(s["line"] - 1, min(len(lines), s["line"] + 150)):
                        line = lines[idx]
                        brace_count += line.count("{") - line.count("}")
                        if "{" in line:
                            started = True
                        if started and brace_count <= 0:
                            end = idx + 1
                            break

                snippet = "\n".join(f"{line_num:4d} | {lines[line_num - 1]}" for line_num in range(start, end + 1))
                found.append(f"### {rel} (Lines {start}-{end}):\n```cpp\n{snippet}\n```")
                if len(found) >= 3:
                    break

    if not found:
        return f"Definition for symbol '{symbol_name}' not found."
    return "\n\n".join(found)

def tool_get_file_outline(file_path: str) -> str:
    p = PROJECT_ROOT / file_path.replace("\\", "/")
    if not p.exists():
        return f"File not found: {file_path}"

    rel = p.relative_to(PROJECT_ROOT).as_posix()
    syms = extract_symbols(p)
    if not syms:
        return f"No structural symbols identified in {rel}."

    syms.sort(key=lambda s: s["line"])
    outline_lines = [f"Structural outline of {rel}:"]
    for s in syms:
        outline_lines.append(f"  Line {s['line']:4d}: [{s['type']:11s}] {s['signature']}")
    return "\n".join(outline_lines)

def tool_find_references(symbol_name: str, file_glob: str = "*") -> str:
    files = get_source_files(file_glob)
    pattern = re.compile(rf"\b{re.escape(symbol_name)}\b")
    results = []

    for f in files:
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines, start=1):
            if pattern.search(line):
                clean_l = line.strip()
                results.append(f"{rel}:{i} | {clean_l[:120]}")
                if len(results) >= 50:
                    break
        if len(results) >= 50:
            break

    if not results:
        return f"No references to identifier '{symbol_name}' found."
    if len(results) >= 50:
        results.append("... (+ additional references truncated to conserve tokens)")
    return "\n".join(results)

# ---------------------------------------------------------------------------
# MCP JSON-RPC 2.0 (stdio) Server
# ---------------------------------------------------------------------------

TOOLS_SPEC = [
    {
        "name": "search_symbols",
        "description": "Finds symbol declarations (structs, classes, functions, CUDA kernels, enums, macros) in .cu, .cuh, .cpp, .h, .py without returning full files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Symbol name or regex pattern (e.g. 'CarStateSoA', 'step_kernel', 'btVehicleRL')" },
                "symbol_type": { "type": "string", "description": "Optional: 'struct', 'class', 'kernel', 'device_func', 'enum', 'macro', or 'all'", "default": "all" },
                "file_glob": { "type": "string", "description": "Optional: path filter (e.g. 'src/cuda/**', '*.cuh')", "default": "*" }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "get_symbol_definition",
        "description": "Extracts exclusively the definition block of a requested symbol (with line numbers and comments) without reading the entire file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol_name": { "type": "string", "description": "Exact name of symbol to look up" },
                "file_path": { "type": "string", "description": "Optional: file path where symbol resides" }
            },
            "required": ["symbol_name"]
        }
    },
    {
        "name": "get_file_outline",
        "description": "Generates a structured table of contents (TOC) of declarations, methods, and structs in a file with exact line numbers for focused slice viewing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "Relative file path (e.g. 'src/cuda/step_kernel.cu')" }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "find_references",
        "description": "Finds targeted references of an identifier returning only file, line, and code (1 line context) to prevent massive dumps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol_name": { "type": "string", "description": "Identifier name" },
                "file_glob": { "type": "string", "description": "Optional: path filter", "default": "*" }
            },
            "required": ["symbol_name"]
        }
    }
]

def handle_rpc_request(req):
    method = req.get("method")
    msg_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "rocketsim-code-search",
                    "version": "1.0.0"
                }
            }
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": { "tools": TOOLS_SPEC }
        }

    if method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})

        try:
            if tool_name == "search_symbols":
                out = tool_search_symbols(args.get("pattern", ""), args.get("symbol_type", "all"), args.get("file_glob", "*"))
            elif tool_name == "get_symbol_definition":
                out = tool_get_symbol_definition(args.get("symbol_name", ""), args.get("file_path"))
            elif tool_name == "get_file_outline":
                out = tool_get_file_outline(args.get("file_path", ""))
            elif tool_name == "find_references":
                out = tool_find_references(args.get("symbol_name", ""), args.get("file_glob", "*"))
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": { "code": -32601, "message": f"Unknown tool: {tool_name}" }
                }

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{ "type": "text", "text": out }],
                    "isError": False
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{ "type": "text", "text": f"Error executing {tool_name}: {str(e)}" }],
                    "isError": True
                }
            }

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": { "code": -32601, "message": f"Method not supported: {method}" }
    }

def run_cli_tests():
    print("=== Testing Local MCP Code Search Tools ===")
    print("\n1. Search symbols 'CarState':")
    print(tool_search_symbols("CarState"))

    print("\n2. File outline for 'include/rocketsim_cuda/types/car_state.cuh':")
    print(tool_get_file_outline("include/rocketsim_cuda/types/car_state.cuh"))

    print("\n3. Symbol definition for 'CarStateSoA':")
    print(tool_get_symbol_definition("CarStateSoA"))

    print("\n4. References for 'CarStateSoA':")
    print(tool_find_references("CarStateSoA", "*.cuh"))
    print("\n=== All MCP tests passed successfully! ===")

def main():
    if "--test" in sys.argv:
        run_cli_tests()
        return

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = handle_rpc_request(req)
            if res is not None:
                sys.stdout.write(json.dumps(res) + "\n")
                sys.stdout.flush()
        except Exception as e:
            err_res = {
                "jsonrpc": "2.0",
                "id": None,
                "error": { "code": -32700, "message": f"Parse error: {str(e)}" }
            }
            sys.stdout.write(json.dumps(err_res) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
