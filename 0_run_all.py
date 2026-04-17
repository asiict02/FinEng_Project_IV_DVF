"""
run_all.py — Execute the full IV surface pipeline in sequence.
Run this file to reproduce all results from scratch.
"""
import subprocess, sys

steps = [
    "1_data_collection.py",
    "2_implied_vol.py",
    "5_estimation.py",
    "6_evaluation.py",
    "7_visualization.py",
]

for step in steps:
    print(f"\n{'='*60}\nRunning {step}\n{'='*60}")
    result = subprocess.run([sys.executable, step], check=True)