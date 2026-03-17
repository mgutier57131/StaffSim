"""CLI entry point for scenario orchestration.

Usage:
  python -m staffsim.orchestrate --out results_sim --parallel 4 --cp-sat-workers 1
"""

from __future__ import annotations

import argparse

from staffsim.orchestrator.engine import OrchestratorConfig, orchestrate


def _str_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run scenario orchestrator for demand + scheduling.")
    p.add_argument("--out", type=str, default="results_sim", help="Output folder (default: results_sim)")
    p.add_argument("--parallel", type=int, default=4, help="Parallel external runs (processes)")
    p.add_argument("--cp-sat-workers", type=int, default=1, help="CP-SAT internal workers per solve")
    p.add_argument("--base-seed", type=int, default=12345, help="Base seed for deterministic scenario seeds")
    p.add_argument("--run1-time-limit", type=float, default=120.0, help="Run1 per-solve time limit in seconds")
    p.add_argument("--run2-time-limit", type=float, default=600.0, help="Run2 per-solve time limit in seconds")
    p.add_argument("--retries-demand", type=int, default=2, help="Retries for demand stage")
    p.add_argument("--retries-sched", type=int, default=1, help="Retries for scheduling stage")
    p.add_argument("--regen-grid", type=_str_bool, default=False, help="Regenerate scenarios.csv true/false")
    p.add_argument("--coverage-target", type=float, default=0.90, help="Coverage target threshold")
    p.add_argument("--max-expand", type=int, default=256, help="Maximum upward N expansions when target is not met")
    p.add_argument("--stage", type=str, choices=["demand", "schedule", "both"], default="both", help="Execution stage")
    p.add_argument(
        "--scheduler",
        type=str,
        choices=["cp_sat", "hexaly", "both"],
        default="cp_sat",
        help="Scheduling backend(s) to run in schedule stage",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = OrchestratorConfig(
        out_dir=args.out,
        parallel_runs=int(args.parallel),
        cp_sat_workers=int(args.cp_sat_workers),
        base_seed=int(args.base_seed),
        run1_time_limit=float(args.run1_time_limit),
        run2_time_limit=float(args.run2_time_limit),
        retries_demand=int(args.retries_demand),
        retries_sched=int(args.retries_sched),
        regen_grid=bool(args.regen_grid),
        coverage_target=float(args.coverage_target),
        max_expand=int(args.max_expand),
        stage=str(args.stage),
        scheduler=str(args.scheduler),
    )
    orchestrate(cfg)


if __name__ == "__main__":
    main()
