"""Search utilities for minimum N meeting coverage target."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from staffsim.scheduling.io import (
    make_final_output_dir,
    write_ilp_summary,
    write_matrix_csv,
    write_schedule_detail,
    write_search_log,
)
from staffsim.scheduling.metrics import compute_coverage, compute_under_over_delta
from staffsim.scheduling.plotting import plot_required_vs_planned
from staffsim.scheduling.run1_model import ModelSolveResult as Run1SolveResult
from staffsim.scheduling.run1_model import solve_run1
from staffsim.scheduling.run2_model import ModelSolveResult as Run2SolveResult
from staffsim.scheduling.run2_model import solve_run2

Mode = Literal["run1", "run2"]


@dataclass(frozen=True)
class TrialResult:
    n_agents: int
    solver_status: str
    objective_value: float
    coverage: float
    sum_under: float
    sum_over: float
    planned: np.ndarray
    schedule_detail: object
    runtime_sec: float


@dataclass(frozen=True)
class SearchResult:
    mode: Mode
    n0: int
    n_final: int
    coverage_final: float
    final_output_dir: Path
    trials: list[TrialResult]


def _solve_single(
    *,
    mode: Mode,
    required: np.ndarray,
    n_agents: int,
    time_limit_sec: float,
    num_workers: int | None,
) -> TrialResult:
    solved: Run1SolveResult | Run2SolveResult
    if mode == "run1":
        solved = solve_run1(required, n_agents=n_agents, time_limit_sec=time_limit_sec, num_workers=num_workers)
    else:
        solved = solve_run2(required, n_agents=n_agents, time_limit_sec=time_limit_sec, num_workers=num_workers)

    planned = solved.planned_matrix
    under, over, _ = compute_under_over_delta(required, planned)
    coverage = compute_coverage(required, under)
    return TrialResult(
        n_agents=n_agents,
        solver_status=solved.solver_status,
        objective_value=solved.objective_value,
        coverage=coverage,
        sum_under=float(under.sum()),
        sum_over=float(over.sum()),
        planned=planned,
        schedule_detail=solved.schedule_detail,
        runtime_sec=solved.runtime_sec,
    )


def _trial_ok(trial: TrialResult, target: float) -> bool:
    return trial.solver_status in {"OPTIMAL", "FEASIBLE"} and trial.coverage >= target


def _log_trial(prefix: str, trial: TrialResult, target: float, log_lines: list[str]) -> None:
    ok_txt = "OK" if _trial_ok(trial, target) else "FAIL"
    line = (
        f"{prefix} try N={trial.n_agents}, status={trial.solver_status}, "
        f"coverage={trial.coverage:.4f}, sum_under={trial.sum_under:.2f}, "
        f"objective={trial.objective_value:.2f} => {ok_txt}"
    )
    print(line)
    log_lines.append(line)


def _binary_search_min_ok(
    *,
    mode: Mode,
    required: np.ndarray,
    low_fail: int,
    high_ok: int,
    coverage_target: float,
    time_limit_sec: float,
    num_workers: int | None,
    log_lines: list[str],
    trials: dict[int, TrialResult],
) -> TrialResult:
    best = trials[high_ok]
    lo, hi = low_fail, high_ok
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid not in trials:
            trials[mid] = _solve_single(
                mode=mode,
                required=required,
                n_agents=mid,
                time_limit_sec=time_limit_sec,
                num_workers=num_workers,
            )
            _log_trial(mode, trials[mid], coverage_target, log_lines)
        if _trial_ok(trials[mid], coverage_target):
            best = trials[mid]
            hi = mid
        else:
            lo = mid
    return best


def _export_final(
    *,
    run_dir: Path,
    mode: Mode,
    required: np.ndarray,
    best: TrialResult,
    search_log_lines: list[str],
) -> Path:
    out_dir = make_final_output_dir(run_dir, mode)
    under, over, delta = compute_under_over_delta(required, best.planned)
    write_matrix_csv(out_dir / "required_matrix.csv", required)
    write_matrix_csv(out_dir / "planned_matrix.csv", best.planned)
    write_matrix_csv(out_dir / "under_matrix.csv", under)
    write_matrix_csv(out_dir / "over_matrix.csv", over)
    write_matrix_csv(out_dir / "delta_matrix.csv", delta)
    write_schedule_detail(out_dir / "schedule_detail.csv", best.schedule_detail)  # type: ignore[arg-type]
    write_ilp_summary(
        out_dir / "ilp_summary.csv",
        mode=mode,
        n_agents=best.n_agents,
        solver_status=best.solver_status,
        objective_value=best.objective_value,
        coverage=best.coverage,
        sum_required=float(required.sum()),
        sum_under=best.sum_under,
        sum_over=best.sum_over,
        runtime_sec=best.runtime_sec,
    )
    plot_required_vs_planned(required, best.planned, out_dir / "schedule_curve.png")
    write_search_log(out_dir / "search_log.txt", search_log_lines)
    return out_dir


def find_min_n(
    *,
    run_dir: Path,
    required: np.ndarray,
    mode: Mode,
    n0: int,
    coverage_target: float = 0.90,
    time_limit_sec: float = 30.0,
    num_workers: int | None = None,
    max_expand: int = 256,
) -> SearchResult:
    log_lines: list[str] = [f"{mode}: starting at N0={n0}"]
    print(log_lines[0])
    trials: dict[int, TrialResult] = {}

    trials[n0] = _solve_single(
        mode=mode,
        required=required,
        n_agents=n0,
        time_limit_sec=time_limit_sec,
        num_workers=num_workers,
    )
    _log_trial(mode, trials[n0], coverage_target, log_lines)

    if _trial_ok(trials[n0], coverage_target):
        high_ok = n0
        step = 1
        low_fail = 0
        while True:
            cand = max(1, high_ok - step)
            if cand in trials:
                trial = trials[cand]
            else:
                trial = _solve_single(
                    mode=mode,
                    required=required,
                    n_agents=cand,
                    time_limit_sec=time_limit_sec,
                    num_workers=num_workers,
                )
                trials[cand] = trial
                _log_trial(mode, trial, coverage_target, log_lines)
            if _trial_ok(trial, coverage_target):
                high_ok = cand
                if cand == 1:
                    low_fail = 0
                    break
                step *= 2
            else:
                low_fail = cand
                break
        best = _binary_search_min_ok(
            mode=mode,
            required=required,
            low_fail=low_fail,
            high_ok=high_ok,
            coverage_target=coverage_target,
            time_limit_sec=time_limit_sec,
            num_workers=num_workers,
            log_lines=log_lines,
            trials=trials,
        )
    else:
        low_fail = n0
        high_ok = None
        step = 1
        while step <= max_expand:
            cand = n0 + step
            trial = _solve_single(
                mode=mode,
                required=required,
                n_agents=cand,
                time_limit_sec=time_limit_sec,
                num_workers=num_workers,
            )
            trials[cand] = trial
            _log_trial(mode, trial, coverage_target, log_lines)
            if _trial_ok(trial, coverage_target):
                high_ok = cand
                break
            low_fail = cand
            step *= 2
        if high_ok is None:
            raise RuntimeError(f"{mode}: coverage target {coverage_target:.2f} not reached up to N={n0 + max_expand}.")
        best = _binary_search_min_ok(
            mode=mode,
            required=required,
            low_fail=low_fail,
            high_ok=high_ok,
            coverage_target=coverage_target,
            time_limit_sec=time_limit_sec,
            num_workers=num_workers,
            log_lines=log_lines,
            trials=trials,
        )

    out_dir = _export_final(
        run_dir=run_dir,
        mode=mode,
        required=required,
        best=best,
        search_log_lines=log_lines,
    )
    print(f"{mode}: final N={best.n_agents} coverage={best.coverage:.4f}")
    return SearchResult(
        mode=mode,
        n0=n0,
        n_final=best.n_agents,
        coverage_final=best.coverage,
        final_output_dir=out_dir,
        trials=sorted(trials.values(), key=lambda t: t.n_agents),
    )

