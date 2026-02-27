"""Search utilities for minimum N meeting coverage target."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from staffsim.scheduling.io import make_output_dir, write_ilp_summary, write_matrix_csv
from staffsim.scheduling.metrics import compute_coverage, compute_under_over_delta
from staffsim.scheduling.plotting import plot_required_vs_planned
from staffsim.scheduling.run1_model import solve_run1
from staffsim.scheduling.run2_model import solve_run2

Mode = Literal["run1", "run2"]


@dataclass(frozen=True)
class SearchResult:
    mode: Mode
    n0: int
    n_final: int
    coverage_final: float
    final_output_dir: Path


def _solve_and_export(
    *,
    mode: Mode,
    required: np.ndarray,
    run_dir: Path,
    n_agents: int,
    time_limit_sec: float,
) -> tuple[bool, float, Path, np.ndarray]:
    if mode == "run1":
        solved = solve_run1(required, n_agents=n_agents, time_limit_sec=time_limit_sec)
    else:
        solved = solve_run2(required, n_agents=n_agents, time_limit_sec=time_limit_sec)

    planned = solved.planned_matrix
    under, over, delta = compute_under_over_delta(required, planned)
    coverage = compute_coverage(required, under)
    status_ok = solved.solver_status in {"OPTIMAL", "FEASIBLE"}

    out_dir = make_output_dir(run_dir, mode, n_agents)
    write_matrix_csv(out_dir / "required_matrix.csv", required)
    write_matrix_csv(out_dir / "planned_matrix.csv", planned)
    write_matrix_csv(out_dir / "under_matrix.csv", under)
    write_matrix_csv(out_dir / "over_matrix.csv", over)
    write_matrix_csv(out_dir / "delta_matrix.csv", delta)
    write_ilp_summary(
        out_dir / "ilp_summary.csv",
        mode=mode,
        n_agents=n_agents,
        solver_status=solved.solver_status,
        objective_value=solved.objective_value,
        coverage=coverage,
        sum_required=float(required.sum()),
        sum_under=float(under.sum()),
        sum_over=float(over.sum()),
        runtime_sec=solved.runtime_sec,
    )

    return status_ok, coverage, out_dir, planned


def find_min_n(
    *,
    run_dir: Path,
    required: np.ndarray,
    mode: Mode,
    n0: int,
    coverage_target: float = 0.90,
    time_limit_sec: float = 30.0,
    max_expand: int = 200,
) -> SearchResult:
    ok0, cov0, dir0, planned0 = _solve_and_export(
        mode=mode,
        required=required,
        run_dir=run_dir,
        n_agents=n0,
        time_limit_sec=time_limit_sec,
    )

    best_n = n0
    best_cov = cov0 if ok0 else 0.0
    best_dir = dir0
    best_planned = planned0

    if ok0 and cov0 >= coverage_target:
        n = n0 - 1
        while n >= 1:
            ok, cov, out_dir, planned = _solve_and_export(
                mode=mode,
                required=required,
                run_dir=run_dir,
                n_agents=n,
                time_limit_sec=time_limit_sec,
            )
            if ok and cov >= coverage_target:
                best_n, best_cov, best_dir, best_planned = n, cov, out_dir, planned
                n -= 1
            else:
                break
    else:
        found = False
        for step in range(1, max_expand + 1):
            n = n0 + step
            ok, cov, out_dir, planned = _solve_and_export(
                mode=mode,
                required=required,
                run_dir=run_dir,
                n_agents=n,
                time_limit_sec=time_limit_sec,
            )
            if ok and cov >= coverage_target:
                best_n, best_cov, best_dir, best_planned = n, cov, out_dir, planned
                found = True
                break
        if not found:
            raise RuntimeError(f"Coverage target {coverage_target:.2f} not reached up to N={n0 + max_expand}.")

    plot_required_vs_planned(required, best_planned, best_dir / "schedule_curve.png")

    return SearchResult(
        mode=mode,
        n0=n0,
        n_final=best_n,
        coverage_final=best_cov,
        final_output_dir=best_dir,
    )
