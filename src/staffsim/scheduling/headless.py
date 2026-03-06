"""Headless scheduling search for orchestrator usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from staffsim.scheduling.metrics import compute_coverage, compute_under_over_delta
from staffsim.scheduling.run1_model import solve_run1
from staffsim.scheduling.run2_model import solve_run2

Mode = Literal["run1", "run2"]
Solver = Literal["cp_sat", "hexaly"]


@dataclass(frozen=True)
class SchedulingHeadlessResult:
    solver: Solver
    mode: Mode
    n0: int
    n_final: int
    solver_status: str
    objective_value: float
    coverage: float
    coverage_fail: bool
    sum_required: float
    sum_under: float
    sum_over: float
    runtime_sec: float
    planned_matrix: np.ndarray
    under_matrix: np.ndarray
    over_matrix: np.ndarray
    delta_matrix: np.ndarray
    schedule_detail: pd.DataFrame
    search_log_lines: list[str]


@dataclass(frozen=True)
class _Trial:
    n: int
    solver_status: str
    objective_value: float
    coverage: float
    sum_under: float
    sum_over: float
    runtime_sec: float
    planned: np.ndarray
    detail: pd.DataFrame


def _trial_ok(trial: _Trial, target: float) -> bool:
    return trial.solver_status in {"OPTIMAL", "FEASIBLE"} and trial.coverage >= target


def _solve_trial(
    *,
    solver: Solver,
    mode: Mode,
    required: np.ndarray,
    n_agents: int,
    time_limit_sec: float,
    workers: int,
) -> _Trial:
    if solver == "cp_sat":
        if mode == "run1":
            solved = solve_run1(required, n_agents=n_agents, time_limit_sec=time_limit_sec, num_workers=workers)
        else:
            solved = solve_run2(required, n_agents=n_agents, time_limit_sec=time_limit_sec, num_workers=workers)
    else:
        from staffsim.scheduling.hexaly_models import solve_run1_hexaly, solve_run2_hexaly

        if mode == "run1":
            solved = solve_run1_hexaly(required, n_agents=n_agents, time_limit_sec=time_limit_sec, num_workers=workers)
        else:
            solved = solve_run2_hexaly(required, n_agents=n_agents, time_limit_sec=time_limit_sec, num_workers=workers)

    if solved.solver_status == "UNKNOWN":
        retry_limit = max(time_limit_sec * 2.0, time_limit_sec + 30.0)
        if solver == "cp_sat":
            if mode == "run1":
                solved = solve_run1(required, n_agents=n_agents, time_limit_sec=retry_limit, num_workers=workers)
            else:
                solved = solve_run2(required, n_agents=n_agents, time_limit_sec=retry_limit, num_workers=workers)
        else:
            from staffsim.scheduling.hexaly_models import solve_run1_hexaly, solve_run2_hexaly

            if mode == "run1":
                solved = solve_run1_hexaly(required, n_agents=n_agents, time_limit_sec=retry_limit, num_workers=workers)
            else:
                solved = solve_run2_hexaly(required, n_agents=n_agents, time_limit_sec=retry_limit, num_workers=workers)

    under, over, _ = compute_under_over_delta(required, solved.planned_matrix)
    coverage = compute_coverage(required, under)
    return _Trial(
        n=n_agents,
        solver_status=solved.solver_status,
        objective_value=solved.objective_value,
        coverage=coverage,
        sum_under=float(under.sum()),
        sum_over=float(over.sum()),
        runtime_sec=solved.runtime_sec,
        planned=solved.planned_matrix,
        detail=solved.schedule_detail,
    )


def run_headless(
    *,
    solver: Solver,
    required: np.ndarray,
    mode: Mode,
    n0: int,
    coverage_target: float = 0.90,
    time_limit_sec: float = 120.0,
    workers: int = 1,
    max_expand: int = 256,
) -> SchedulingHeadlessResult:
    """Find minimum N for a mode without writing files."""
    trials: dict[int, _Trial] = {}
    log_lines: list[str] = [f"{mode}: starting at N0={n0}"]

    def _log(trial: _Trial) -> None:
        ok_txt = "OK" if _trial_ok(trial, coverage_target) else "FAIL"
        log_lines.append(
            f"{mode} try N={trial.n}, status={trial.solver_status}, coverage={trial.coverage:.4f}, "
            f"sum_under={trial.sum_under:.2f}, objective={trial.objective_value:.2f} => {ok_txt}"
        )

    trials[n0] = _solve_trial(
        solver=solver,
        mode=mode,
        required=required,
        n_agents=n0,
        time_limit_sec=time_limit_sec,
        workers=workers,
    )
    _log(trials[n0])

    if _trial_ok(trials[n0], coverage_target):
        best = trials[n0]
        n = n0 - 1
        while n >= 1:
            trial = _solve_trial(
                solver=solver,
                mode=mode,
                required=required,
                n_agents=n,
                time_limit_sec=time_limit_sec,
                workers=workers,
            )
            trials[n] = trial
            _log(trial)
            if _trial_ok(trial, coverage_target):
                best = trial
                n -= 1
            else:
                break
    else:
        best = None
        for step in range(1, max_expand + 1):
            n = n0 + step
            trial = _solve_trial(
                solver=solver,
                mode=mode,
                required=required,
                n_agents=n,
                time_limit_sec=time_limit_sec,
                workers=workers,
            )
            trials[n] = trial
            _log(trial)
            if _trial_ok(trial, coverage_target):
                best = trial
                break
        if best is None:
            # Keep best effort result for analysis even when target is not reached.
            best = max(trials.values(), key=lambda t: t.coverage)

    under, over, delta = compute_under_over_delta(required, best.planned)
    coverage_fail = best.coverage < coverage_target
    log_lines.append(f"{mode}: final N={best.n} coverage={best.coverage:.4f}")

    return SchedulingHeadlessResult(
        solver=solver,
        mode=mode,
        n0=n0,
        n_final=best.n,
        solver_status=best.solver_status,
        objective_value=best.objective_value,
        coverage=best.coverage,
        coverage_fail=coverage_fail,
        sum_required=float(required.sum()),
        sum_under=best.sum_under,
        sum_over=best.sum_over,
        runtime_sec=best.runtime_sec,
        planned_matrix=best.planned,
        under_matrix=under,
        over_matrix=over,
        delta_matrix=delta,
        schedule_detail=best.detail,
        search_log_lines=log_lines,
    )
