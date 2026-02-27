"""Entry point for scheduling ILP search.

Usage:
  python -m staffsim.schedule --run results/<run_id> --mode run1
  python -m staffsim.schedule --mode run2
"""

from __future__ import annotations

import argparse

from staffsim.scheduling.io import ensure_run_inputs, read_n0_from_summary, read_required_matrix, resolve_run_dir
from staffsim.scheduling.search import find_min_n


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CP-SAT scheduling search.")
    parser.add_argument("--run", type=str, default=None, help="Path to results/<run_id> folder. If omitted, latest run is used.")
    parser.add_argument("--mode", type=str, choices=["run1", "run2"], required=True, help="Scheduling mode.")
    parser.add_argument("--coverage-target", type=float, default=0.90, help="Coverage target threshold.")
    parser.add_argument("--time-limit", type=float, default=30.0, help="Per-solve time limit in seconds.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = resolve_run_dir(args.run, base_results_dir="results")
    ensure_run_inputs(run_dir)
    required = read_required_matrix(run_dir)
    n0 = read_n0_from_summary(run_dir)

    result = find_min_n(
        run_dir=run_dir,
        required=required,
        mode=args.mode,
        n0=n0,
        coverage_target=float(args.coverage_target),
        time_limit_sec=float(args.time_limit),
    )

    print(f"N0 (HC_gross_ceil): {result.n0}")
    print(f"N final minimum: {result.n_final}")
    print(f"Coverage final: {result.coverage_final:.4f}")
    print(f"Saved outputs to: {result.final_output_dir.as_posix()}/")


if __name__ == "__main__":
    main()

