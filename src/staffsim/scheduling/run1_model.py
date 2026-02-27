"""Run 1 CP-SAT model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

SCALE = 100


@dataclass(frozen=True)
class ModelSolveResult:
    mode: str
    n_agents: int
    solver_status: str
    objective_value: float
    runtime_sec: float
    planned_matrix: np.ndarray
    schedule_detail: pd.DataFrame


def _status_name(status: int) -> str:
    if status == cp_model.OPTIMAL:
        return "OPTIMAL"
    if status == cp_model.FEASIBLE:
        return "FEASIBLE"
    if status == cp_model.INFEASIBLE:
        return "INFEASIBLE"
    if status == cp_model.MODEL_INVALID:
        return "MODEL_INVALID"
    return "UNKNOWN"


def _fmt_slot(start_interval: int, length: int) -> str:
    start_minutes = start_interval * 30
    end_minutes = (start_interval + length) * 30
    sh, sm = divmod(start_minutes, 60)
    eh, em = divmod(end_minutes, 60)
    return f"{int(sh):02d}:{int(sm):02d}-{int(eh):02d}:{int(em):02d}"


def solve_run1(
    required_matrix: np.ndarray,
    n_agents: int,
    time_limit_sec: float = 30.0,
    num_workers: int | None = None,
) -> ModelSolveResult:
    days, intervals = required_matrix.shape
    if (days, intervals) != (7, 48):
        raise ValueError(f"Run1 expects 7x48 matrix, got {required_matrix.shape}")

    model = cp_model.CpModel()
    starts = range(35)  # 0..34 to fit 14 intervals
    r_int = np.rint(required_matrix * SCALE).astype(int)
    max_r = int(r_int.max())

    y: dict[tuple[int, int], cp_model.IntVar] = {}
    s_start: dict[tuple[int, int], cp_model.IntVar] = {}
    z: dict[tuple[int, int, int], cp_model.IntVar] = {}

    for k in range(n_agents):
        for d in range(days):
            y[(k, d)] = model.NewBoolVar(f"y_k{k}_d{d}")
        for r in starts:
            s_start[(k, r)] = model.NewBoolVar(f"s_k{k}_r{r}")

        model.Add(sum(y[(k, d)] for d in range(days)) == 6)
        model.Add(sum(s_start[(k, r)] for r in starts) == 1)

        # Feasible hint: work Mon-Sat, off Sun; fixed weekly start by employee.
        hinted_start = (3 * k) % 35
        for d in range(days):
            model.AddHint(y[(k, d)], 1 if d < 6 else 0)
        for r in starts:
            model.AddHint(s_start[(k, r)], 1 if r == hinted_start else 0)

    starts_covering: dict[int, list[int]] = {}
    for j in range(intervals):
        starts_covering[j] = [r for r in starts if r <= j <= r + 13]

    for k in range(n_agents):
        for d in range(days):
            for r in starts:
                z[(k, d, r)] = model.NewBoolVar(f"z_k{k}_d{d}_r{r}")
                model.Add(z[(k, d, r)] <= y[(k, d)])
                model.Add(z[(k, d, r)] <= s_start[(k, r)])
                model.Add(z[(k, d, r)] >= y[(k, d)] + s_start[(k, r)] - 1)
            model.Add(sum(z[(k, d, r)] for r in starts) == y[(k, d)])

    under: dict[tuple[int, int], cp_model.IntVar] = {}
    objective_terms: list[cp_model.IntVar] = []
    upper_under = max_r + SCALE * n_agents
    for d in range(days):
        for j in range(intervals):
            under[(d, j)] = model.NewIntVar(0, upper_under, f"under_d{d}_j{j}")
            c_expr = sum(z[(k, d, r)] for k in range(n_agents) for r in starts_covering[j])
            model.Add(under[(d, j)] >= int(r_int[d, j]) - SCALE * c_expr)
            objective_terms.append(under[(d, j)])

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    if num_workers is not None and num_workers > 0:
        solver.parameters.num_search_workers = int(num_workers)
    status = solver.Solve(model)
    status_name = _status_name(status)

    planned = np.zeros((days, intervals), dtype=float)
    schedule_rows: list[dict[str, str | int]] = []
    objective_value = float("inf")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        objective_value = float(solver.ObjectiveValue()) / SCALE
        for d in range(days):
            for j in range(intervals):
                planned[d, j] = float(sum(solver.Value(z[(k, d, r)]) for k in range(n_agents) for r in starts_covering[j]))
        day_cols = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for k in range(n_agents):
            chosen_start = next(r for r in starts if solver.Value(s_start[(k, r)]) == 1)
            row: dict[str, str | int] = {"employee": k + 1}
            slot = _fmt_slot(chosen_start, 14)
            for d_idx, day_name in enumerate(day_cols):
                row[day_name] = slot if solver.Value(y[(k, d_idx)]) == 1 else "OFF"
            schedule_rows.append(row)

    schedule_detail = pd.DataFrame(schedule_rows) if schedule_rows else pd.DataFrame(columns=["employee", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    return ModelSolveResult(
        mode="run1",
        n_agents=n_agents,
        solver_status=status_name,
        objective_value=objective_value,
        runtime_sec=float(solver.WallTime()),
        planned_matrix=planned,
        schedule_detail=schedule_detail,
    )
