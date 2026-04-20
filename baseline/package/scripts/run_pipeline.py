#!/usr/bin/env python3
"""
End-to-end pipeline runner.

Runs all preprocessing steps in sequence.  Retrieval (step 2) and
evaluation (step 3) are interactive/notebook-driven and are not included here.

Usage
-----
    python scripts/run_pipeline.py \\
        --html-dir    data/raw \\
        --minimal-dir data/processed/minimal \\
        --clean-dir   data/processed/clean \\
        --valid-dir   data/processed/valid \\
        --pdf         data/raw/BenchmarkQuestionsAnswers.pdf \\
        --qa-out      data/benchmark/benchmark_qa.json
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Step failed (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description="Run all preprocessing steps.")
    p.add_argument("--html-dir",    required=True)
    p.add_argument("--minimal-dir", required=True)
    p.add_argument("--clean-dir",   required=True)
    p.add_argument("--valid-dir",   required=True)
    p.add_argument("--pdf",         required=True)
    p.add_argument("--qa-out",      required=True)
    p.add_argument("--threshold",   type=int, default=5)
    args = p.parse_args()

    # 1a: HTML parsing
    run(["python", "-m", "rag.preprocessing.html_parser",
         args.html_dir, args.minimal_dir])

    # 1b: Cleaning and metadata
    run(["python", "-m", "rag.preprocessing.cleaner",
         args.minimal_dir, args.clean_dir,
         "--threshold", str(args.threshold)])

    # 1c: Validation
    run(["python", "-m", "rag.preprocessing.validator",
         args.clean_dir, args.valid_dir])

    # 1d: Benchmark PDF extraction
    run(["python", "-m", "rag.preprocessing.benchmark",
         args.pdf, args.qa_out])

    print("\n✓ Preprocessing complete.")
    print("  Next: open notebooks/02_retrieval.ipynb to build retrieval agents.")


if __name__ == "__main__":
    main()
