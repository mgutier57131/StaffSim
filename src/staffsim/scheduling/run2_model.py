"""Run 2 CP-SAT model.

Rules (updated):
- 1 or 2 day offs => worked days in {5, 6}
- Start is constant by employee across the whole week
- Daily shift length is one of {8, 12, 16, 20} intervals (4, 6, 8, 10 hours)
- Weekly total length = 84 intervals
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

SCALE = 100
LENGTH_OPTIONS = [8, 12, 16, 20]


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


def _feasible_fallback(
    *,
    n_agents: int,
    required_matrix: np.ndarray,
) -> tuple[str, np.ndarray, pd.DataFrame, float]:
    """Deterministic fallback that respects Run2 rules."""
    days, intervals = required_matrix.shape
    planned = np.zeros((days, intervals), dtype=float)
    residual = required_matrix.astype(float).copy()

    # 5 worked days with fixed lengths summing 84 intervals.
    length_plan = [20, 16, 16, 16, 16]
    day_cols = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    schedule_rows: list[dict[str, str | int]] = []

    for k in range(n_agents):
        # Pick one constant start per employee using weekly residual demand for 20-interval window.
        best_start = 0
        best_score = float("-inf")
        for start in range(0, intervals - 20 + 1):
            score = float(np.sum(residual[:, start : start + 20]))
            if score > best_score:
                best_score = score
                best_start = start

        # Choose two off-days with lowest residual around this start.
        day_scores = []
        for d in range(days):
            day_scores.append((float(np.sum(residual[d, best_start : best_start + 20])), d))
        day_scores.sort(key=lambda x: x[0])
        off_days = {day_scores[0][1], day_scores[1][1]}
        work_days = [d for d in range(days) if d not in off_days]

        row: dict[str, str | int] = {"employee": k + 1}
        for d in range(days):
            if d in off_days:
                row[day_cols[d]] = "OFF"
                continue
            # Assign one of the pre-defined lengths in ranked-demand order of work days.
            # Highest demand gets length 20, others 16.
            row[day_cols[d]] = "OFF"
        ranked_work = sorted(work_days, key=lambda d: float(np.sum(residual[d, best_start : best_start + 20])), reverse=True)
        for idx, d in enumerate(ranked_work):
            length = length_plan[idx]
            planned[d, best_start : best_start + length] += 1.0
            residual[d, best_start : best_start + length] -= 1.0
            row[day_cols[d]] = _fmt_slot(best_start, length)
        schedule_rows.append(row)

    under = np.maximum(0.0, required_matrix - planned)
    objective = float(under.sum())
    schedule_detail = pd.DataFrame(schedule_rows)
    return "FEASIBLE_FALLBACK", planned, schedule_detail, objective


def solve_run2(
    required_matrix: np.ndarray,
    n_agents: int,
    time_limit_sec: float = 30.0,
    num_workers: int | None = None,
) -> ModelSolveResult:
    days, intervals = required_matrix.shape
    if (days, intervals) != (7, 48):
        raise ValueError(f"Run2 expects 7x48 matrix, got {required_matrix.shape}")

    model = cp_model.CpModel()
    r_int = np.rint(required_matrix * SCALE).astype(int)
    max_r = int(r_int.max())
    starts = range(48)

    # Variables:
    y: dict[tuple[int, int], cp_model.IntVar] = {}
    s_start: dict[tuple[int, int], cp_model.IntVar] = {}
    l_sel: dict[tuple[int, int, int], cp_model.IntVar] = {}
    z: dict[tuple[int, int, int, int], cp_model.IntVar] = {}

    for k in range(n_agents):
        for d in range(days):
            y[(k, d)] = model.NewBoolVar(f"y_k{k}_d{d}")
        for r in starts:
            s_start[(k, r)] = model.NewBoolVar(f"s_k{k}_r{r}")
        model.Add(sum(s_start[(k, r)] for r in starts) == 1)
        model.Add(sum(y[(k, d)] for d in range(days)) >= 5)
        model.Add(sum(y[(k, d)] for d in range(days)) <= 6)

        weekly_len_terms = []
        for d in range(days):
            # Exactly one chosen length if worked, none if off.
            for length in LENGTH_OPTIONS:
                l_sel[(k, d, length)] = model.NewBoolVar(f"l_k{k}_d{d}_len{length}")
            model.Add(sum(l_sel[(k, d, length)] for length in LENGTH_OPTIONS) == y[(k, d)])
            weekly_len_terms.extend(length * l_sel[(k, d, length)] for length in LENGTH_OPTIONS)

            # Link constant start and chosen length (AND) with feasibility start+length<=48.
            for r in starts:
                for length in LENGTH_OPTIONS:
                    z[(k, d, r, length)] = model.NewBoolVar(f"z_k{k}_d{d}_r{r}_l{length}")
                    model.Add(z[(k, d, r, length)] <= s_start[(k, r)])
                    model.Add(z[(k, d, r, length)] <= l_sel[(k, d, length)])
                    model.Add(z[(k, d, r, length)] >= s_start[(k, r)] + l_sel[(k, d, length)] - 1)
                    if r + length > 48:
                        model.Add(z[(k, d, r, length)] == 0)

            # One active (r,length) pair if worked day, otherwise zero.
            model.Add(sum(z[(k, d, r, length)] for r in starts for length in LENGTH_OPTIONS) == y[(k, d)])

        # Weekly total exactly 84 intervals.
        model.Add(sum(weekly_len_terms) == 84)

    # Precompute which (r,length) covers each interval j.
    covers: dict[int, list[tuple[int, int]]] = {j: [] for j in range(intervals)}
    for r in starts:
        for length in LENGTH_OPTIONS:
            end = r + length - 1
            if end >= intervals:
                continue
            for j in range(r, end + 1):
                covers[j].append((r, length))

    under: dict[tuple[int, int], cp_model.IntVar] = {}
    objective_terms: list[cp_model.IntVar] = []
    upper_under = max_r + SCALE * n_agents
    for d in range(days):
        for j in range(intervals):
            under[(d, j)] = model.NewIntVar(0, upper_under, f"under_d{d}_j{j}")
            c_expr = sum(z[(k, d, r, length)] for k in range(n_agents) for (r, length) in covers[j])
            model.Add(under[(d, j)] >= int(r_int[d, j]) - SCALE * c_expr)
            objective_terms.append(under[(d, j)])

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = min(time_limit_sec, 20.0)
    if num_workers is not None and num_workers > 0:
        solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.symmetry_level = 2
    solver.parameters.cp_model_presolve = True
    solver.parameters.random_seed = 12345
    status = solver.Solve(model)
    status_name = _status_name(status)

    planned = np.zeros((days, intervals), dtype=float)
    objective_value = float("inf")
    schedule_rows: list[dict[str, str | int]] = []
    day_cols = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        objective_value = float(solver.ObjectiveValue()) / SCALE
        for d in range(days):
            for j in range(intervals):
                planned[d, j] = float(sum(solver.Value(z[(k, d, r, length)]) for k in range(n_agents) for (r, length) in covers[j]))

        for k in range(n_agents):
            chosen_start = next(r for r in starts if solver.Value(s_start[(k, r)]) == 1)
            row: dict[str, str | int] = {"employee": k + 1}
            for d in range(days):
                if solver.Value(y[(k, d)]) == 0:
                    row[day_cols[d]] = "OFF"
                else:
                    chosen_len = next(length for length in LENGTH_OPTIONS if solver.Value(l_sel[(k, d, length)]) == 1)
                    row[day_cols[d]] = _fmt_slot(chosen_start, chosen_len)
            schedule_rows.append(row)

        schedule_detail = pd.DataFrame(schedule_rows)
        return ModelSolveResult(
            mode="run2",
            n_agents=n_agents,
            solver_status=status_name,
            objective_value=objective_value,
            runtime_sec=float(solver.WallTime()),
            planned_matrix=planned,
            schedule_detail=schedule_detail,
        )

    # Robust fallback that still respects the same run2 constraints.
    fb_status, fb_planned, fb_detail, fb_objective = _feasible_fallback(
        n_agents=n_agents,
        required_matrix=required_matrix,
    )
    return ModelSolveResult(
        mode="run2",
        n_agents=n_agents,
        solver_status=fb_status,
        objective_value=fb_objective,
        runtime_sec=float(solver.WallTime()),
        planned_matrix=fb_planned,
        schedule_detail=fb_detail,
    )

