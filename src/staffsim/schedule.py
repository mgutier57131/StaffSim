"""Entry point for scheduling ILP search.

Usage:
  python -m staffsim.schedule --run results/<run_id> --mode run1
  python -m staffsim.schedule --mode run2
"""

from __future__ import annotations

import argparse

from staffsim.scheduling.io import (
    ensure_run_inputs,
    read_headcount_refs,
    read_n0_from_summary,
    read_required_matrix,
    resolve_run_dir,
    write_unified_summary_table,
)
from staffsim.scheduling.search import find_min_n


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CP-SAT scheduling search.")
    parser.add_argument("--run", type=str, default=None, help="Path to results/<run_id> folder. If omitted, latest run is used.")
    parser.add_argument("--mode", type=str, choices=["run1", "run2", "both"], required=True, help="Scheduling mode.")
    parser.add_argument("--coverage-target", type=float, default=0.90, help="Coverage target threshold.")
    parser.add_argument("--time-limit", type=float, default=30.0, help="Per-solve time limit in seconds (run1 default).")
    parser.add_argument("--time-limit-run2", type=float, default=90.0, help="Per-solve time limit in seconds for run2.")
    parser.add_argument("--workers", type=int, default=8, help="CP-SAT num_search_workers.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = resolve_run_dir(args.run, base_results_dir="results")
    print(f"Loading run: {run_dir.as_posix()}")
    ensure_run_inputs(run_dir)
    required = read_required_matrix(run_dir)
    n0 = read_n0_from_summary(run_dir)
    hc_refs = read_headcount_refs(run_dir)
    print(f"N0 (HC_gross_ceil): {n0}")
    if "HC_teorico_ceil" in hc_refs:
        print(f"HC from demand (HC_teorico_ceil): {hc_refs['HC_teorico_ceil']}")

    modes = ["run1", "run2"] if args.mode == "both" else [args.mode]
    failures: list[str] = []
    for mode in modes:
        try:
            time_limit_sec = float(args.time_limit_run2) if mode == "run2" else float(args.time_limit)
            result = find_min_n(
                run_dir=run_dir,
                required=required,
                mode=mode,  # type: ignore[arg-type]
                n0=n0,
                coverage_target=float(args.coverage_target),
                time_limit_sec=time_limit_sec,
                num_workers=int(args.workers),
                hc_refs=hc_refs,
            )
            print(f"{mode}: N final minimum: {result.n_final}")
            print(f"{mode}: Coverage final: {result.coverage_final:.4f}")
            print(f"{mode}: Saved outputs to: {result.final_output_dir.as_posix()}/")
        except Exception as exc:
            msg = f"{mode}: failed -> {exc}"
            print(msg)
            failures.append(msg)

    # Keep a unified run-level summary table (demand + scheduling) for easier review.
    if not failures:
        summary_out = write_unified_summary_table(run_dir)
        print(f"Unified summary updated: {summary_out.as_posix()}")

    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
