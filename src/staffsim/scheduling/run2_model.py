"""Run 2 CP-SAT model."""

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


def _build_options(intervals: int = 48) -> tuple[list[tuple[int, int]], dict[int, list[int]]]:
    options: list[tuple[int, int]] = []
    for length in range(8, 21):  # 4..10h in 30-min intervals
        for start in range(0, intervals - length + 1):
            options.append((start, length))

    covers: dict[int, list[int]] = {j: [] for j in range(intervals)}
    for o_idx, (start, length) in enumerate(options):
        end = start + length - 1
        for j in range(start, end + 1):
            covers[j].append(o_idx)
    return options, covers


def _fmt_slot(start_interval: int, length: int) -> str:
    start_minutes = start_interval * 30
    end_minutes = (start_interval + length) * 30
    sh, sm = divmod(start_minutes, 60)
    eh, em = divmod(end_minutes, 60)
    return f"{int(sh):02d}:{int(sm):02d}-{int(eh):02d}:{int(em):02d}"


def _build_feasible_fallback(
    *,
    n_agents: int,
    days: int,
    intervals: int,
    options: list[tuple[int, int]],
    time_limit_sec: float,
    num_workers: int | None,
) -> tuple[str, np.ndarray, pd.DataFrame, float]:
    model = cp_model.CpModel()
    option_count = len(options)
    covers: dict[int, list[int]] = {j: [] for j in range(intervals)}
    for o_idx, (start, length) in enumerate(options):
        for j in range(start, start + length):
            covers[j].append(o_idx)
    option_index = {opt: idx for idx, opt in enumerate(options)}

    y: dict[tuple[int, int], cp_model.IntVar] = {}
    s_opt: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for k in range(n_agents):
        for d in range(days):
            y[(k, d)] = model.NewBoolVar(f"fy_k{k}_d{d}")
            for o in range(option_count):
                s_opt[(k, d, o)] = model.NewBoolVar(f"fs_k{k}_d{d}_o{o}")
            model.Add(sum(s_opt[(k, d, o)] for o in range(option_count)) == y[(k, d)])

        model.Add(sum(y[(k, d)] for d in range(days)) >= 5)
        model.Add(sum(y[(k, d)] for d in range(days)) <= 6)
        model.Add(sum(options[o][1] * s_opt[(k, d, o)] for d in range(days) for o in range(option_count)) == 84)
        for d in range(days):
            is_work_day = d < 6
            model.AddHint(y[(k, d)], 1 if is_work_day else 0)
            hinted_start = (2 * d + 3 * k) % (48 - 14 + 1)
            hinted_opt = option_index[(hinted_start, 14)] if is_work_day else None
            for o in range(option_count):
                model.AddHint(s_opt[(k, d, o)], 1 if (hinted_opt is not None and o == hinted_opt) else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(5.0, min(20.0, time_limit_sec))
    if num_workers is not None and num_workers > 0:
        solver.parameters.num_search_workers = int(num_workers)
    status = solver.Solve(model)

    planned = np.zeros((days, intervals), dtype=float)
    schedule_rows: list[dict[str, str | int]] = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        day_cols = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for d in range(days):
            for j in range(intervals):
                planned[d, j] = float(sum(solver.Value(s_opt[(k, d, o)]) for k in range(n_agents) for o in covers[j]))
        for k in range(n_agents):
            row: dict[str, str | int] = {"employee": k + 1}
            for d_idx, day_name in enumerate(day_cols):
                chosen = [o for o in range(option_count) if solver.Value(s_opt[(k, d_idx, o)]) == 1]
                if not chosen:
                    row[day_name] = "OFF"
                else:
                    start, length = options[chosen[0]]
                    row[day_name] = _fmt_slot(start, length)
            schedule_rows.append(row)
        return "FEASIBLE_FALLBACK", planned, pd.DataFrame(schedule_rows), float(solver.WallTime())

    return "UNKNOWN", planned, pd.DataFrame(columns=["employee", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]), float(solver.WallTime())


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

    options, covers = _build_options(intervals=intervals)
    option_count = len(options)
    option_index = {opt: idx for idx, opt in enumerate(options)}

    y: dict[tuple[int, int], cp_model.IntVar] = {}
    s_opt: dict[tuple[int, int, int], cp_model.IntVar] = {}

    for k in range(n_agents):
        for d in range(days):
            y[(k, d)] = model.NewBoolVar(f"y_k{k}_d{d}")
            for o in range(option_count):
                s_opt[(k, d, o)] = model.NewBoolVar(f"s_k{k}_d{d}_o{o}")

            model.Add(sum(s_opt[(k, d, o)] for o in range(option_count)) == y[(k, d)])

        model.Add(sum(y[(k, d)] for d in range(days)) >= 5)
        model.Add(sum(y[(k, d)] for d in range(days)) <= 6)
        model.Add(
            sum(options[o][1] * s_opt[(k, d, o)] for d in range(days) for o in range(option_count))
            == 84
        )

        # Feasible hint: work Mon-Sat with 14 intervals/day, Sunday OFF.
        # This satisfies 84 weekly intervals exactly.
        for d in range(days):
            is_work_day = d < 6
            model.AddHint(y[(k, d)], 1 if is_work_day else 0)
            hinted_start = (2 * d + 3 * k) % (48 - 14 + 1)
            hinted_opt = option_index[(hinted_start, 14)] if is_work_day else None
            for o in range(option_count):
                model.AddHint(s_opt[(k, d, o)], 1 if (hinted_opt is not None and o == hinted_opt) else 0)

    # Symmetry breaking: keep employees sorted by worked days.
    for k in range(n_agents - 1):
        model.Add(sum(y[(k, d)] for d in range(days)) >= sum(y[(k + 1, d)] for d in range(days)))

    under: dict[tuple[int, int], cp_model.IntVar] = {}
    objective_terms: list[cp_model.IntVar] = []
    upper_under = max_r + SCALE * n_agents
    for d in range(days):
        for j in range(intervals):
            under[(d, j)] = model.NewIntVar(0, upper_under, f"under_d{d}_j{j}")
            c_expr = sum(s_opt[(k, d, o)] for k in range(n_agents) for o in covers[j])
            model.Add(under[(d, j)] >= int(r_int[d, j]) - SCALE * c_expr)
            objective_terms.append(under[(d, j)])

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    if num_workers is not None and num_workers > 0:
        solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.symmetry_level = 2
    solver.parameters.cp_model_presolve = True
    solver.parameters.random_seed = 12345
    status = solver.Solve(model)
    status_name = _status_name(status)

    planned = np.zeros((days, intervals), dtype=float)
    schedule_rows: list[dict[str, str | int]] = []
    objective_value = float("inf")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        objective_value = float(solver.ObjectiveValue()) / SCALE
        for d in range(days):
            for j in range(intervals):
                planned[d, j] = float(sum(solver.Value(s_opt[(k, d, o)]) for k in range(n_agents) for o in covers[j]))
        day_cols = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for k in range(n_agents):
            row: dict[str, str | int] = {"employee": k + 1}
            for d_idx, day_name in enumerate(day_cols):
                chosen = [o for o in range(option_count) if solver.Value(s_opt[(k, d_idx, o)]) == 1]
                if not chosen:
                    row[day_name] = "OFF"
                else:
                    start, length = options[chosen[0]]
                    row[day_name] = _fmt_slot(start, length)
            schedule_rows.append(row)

    schedule_detail = pd.DataFrame(schedule_rows) if schedule_rows else pd.DataFrame(columns=["employee", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    if status_name == "UNKNOWN":
        fb_status, fb_planned, fb_detail, fb_runtime = _build_feasible_fallback(
            n_agents=n_agents,
            days=days,
            intervals=intervals,
            options=options,
            time_limit_sec=time_limit_sec,
            num_workers=num_workers,
        )
        if fb_status == "FEASIBLE_FALLBACK":
            status_name = fb_status
            planned = fb_planned
            schedule_detail = fb_detail
            objective_value = float("nan")
            runtime_sec = float(solver.WallTime()) + fb_runtime
            return ModelSolveResult(
                mode="run2",
                n_agents=n_agents,
                solver_status=status_name,
                objective_value=objective_value,
                runtime_sec=runtime_sec,
                planned_matrix=planned,
                schedule_detail=schedule_detail,
            )

    return ModelSolveResult(
        mode="run2",
        n_agents=n_agents,
        solver_status=status_name,
        objective_value=objective_value,
        runtime_sec=float(solver.WallTime()),
        planned_matrix=planned,
        schedule_detail=schedule_detail,
    )
