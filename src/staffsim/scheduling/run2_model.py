"""Run 2 CP-SAT model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


def solve_run2(required_matrix: np.ndarray, n_agents: int, time_limit_sec: float = 30.0) -> ModelSolveResult:
    days, intervals = required_matrix.shape
    if (days, intervals) != (7, 48):
        raise ValueError(f"Run2 expects 7x48 matrix, got {required_matrix.shape}")

    model = cp_model.CpModel()
    r_int = np.rint(required_matrix * SCALE).astype(int)
    max_r = int(r_int.max())

    options, covers = _build_options(intervals=intervals)
    option_count = len(options)

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
    status = solver.Solve(model)
    status_name = _status_name(status)

    planned = np.zeros((days, intervals), dtype=float)
    objective_value = float("inf")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        objective_value = float(solver.ObjectiveValue()) / SCALE
        for d in range(days):
            for j in range(intervals):
                planned[d, j] = float(sum(solver.Value(s_opt[(k, d, o)]) for k in range(n_agents) for o in covers[j]))

    return ModelSolveResult(
        mode="run2",
        n_agents=n_agents,
        solver_status=status_name,
        objective_value=objective_value,
        runtime_sec=float(solver.WallTime()),
        planned_matrix=planned,
    )

