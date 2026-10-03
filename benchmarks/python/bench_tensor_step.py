"""
Benchmark runner invoking benchmarks/benchmark_rl_loop.py.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(project_root, "benchmarks"))

from benchmark_rl_loop import run_benchmark

if __name__ == "__main__":
    md_path = os.path.join(project_root, "BENCHMARKS.md")
    run_benchmark(markdown_output_path=md_path)
