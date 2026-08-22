"""CLI entrypoint for the evaluation harness

Run:
    PYTHONPATH=src python eval/run_eval.py

Writes a machine-readable result to eval/results/latest.json and prints a
human-readable report. Requires OPENAI_API_KEY in the environment (never
printed). Uses the current retrieval + generation layers, temperature 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.harness import DEFAULT_RESULTS_PATH, print_report, run_evaluation, save_results


def main() -> None:
    payload = run_evaluation()
    print_report(payload)
    path = save_results(payload, DEFAULT_RESULTS_PATH)
    print(f"\nWrote machine-readable results to: {path}")


if __name__ == "__main__":
    main()
