"""Run 1 CP-SAT model."""

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


def solve_run1(required_matrix: np.ndarray, n_agents: int, time_limit_sec: float = 30.0) -> ModelSolveResult:
    days, intervals = required_matrix.shape
    if (days, intervals) != (7, 48):
        raise ValueError(f"Run1 expects 7x48 matrix, got {required_matrix.shape}")

    model = cp_model.CpModel()
    starts = range(35)  # 0..34 to fit 14 intervals
    r_int = np.rint(required_matrix * SCALE).astype(int)
    max_r = int(r_int.max())

    y: dict[tuple[int, int], cp_model.IntVar] = {}
    s_start: dict[tuple[int, int], cp_model.IntVar] = {}
    x: dict[tuple[int, int, int], cp_model.IntVar] = {}

    for k in range(n_agents):
        for d in range(days):
            y[(k, d)] = model.NewBoolVar(f"y_k{k}_d{d}")
        for r in starts:
            s_start[(k, r)] = model.NewBoolVar(f"s_k{k}_r{r}")

        model.Add(sum(y[(k, d)] for d in range(days)) == 6)
        model.Add(sum(s_start[(k, r)] for r in starts) == 1)

    starts_covering: dict[int, list[int]] = {}
    for j in range(intervals):
        starts_covering[j] = [r for r in starts if r <= j <= r + 13]

    for k in range(n_agents):
        for d in range(days):
            for j in range(intervals):
                x[(k, d, j)] = model.NewBoolVar(f"x_k{k}_d{d}_j{j}")
                cover_expr = sum(s_start[(k, r)] for r in starts_covering[j])  # 0 or 1 (single start)
                model.Add(x[(k, d, j)] <= y[(k, d)])
                model.Add(x[(k, d, j)] <= cover_expr)
                model.Add(x[(k, d, j)] >= y[(k, d)] + cover_expr - 1)
            model.Add(sum(x[(k, d, j)] for j in range(intervals)) == 14 * y[(k, d)])

    under: dict[tuple[int, int], cp_model.IntVar] = {}
    objective_terms: list[cp_model.IntVar] = []
    upper_under = max_r + SCALE * n_agents
    for d in range(days):
        for j in range(intervals):
            under[(d, j)] = model.NewIntVar(0, upper_under, f"under_d{d}_j{j}")
            c_expr = sum(x[(k, d, j)] for k in range(n_agents))
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
                planned[d, j] = float(sum(solver.Value(x[(k, d, j)]) for k in range(n_agents)))

    return ModelSolveResult(
        mode="run1",
        n_agents=n_agents,
        solver_status=status_name,
        objective_value=objective_value,
        runtime_sec=float(solver.WallTime()),
        planned_matrix=planned,
    )

