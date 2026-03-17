"""Hexaly models for run1 and run2 scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hexaly.optimizer import HexalyOptimizer

SCALE = 100
LENGTH_OPTIONS = [12, 14, 16, 18, 20]  # 6h, 7h, 8h, 9h, 10h


@dataclass(frozen=True)
class ModelSolveResult:
    mode: str
    n_agents: int
    solver_status: str
    objective_value: float
    runtime_sec: float
    planned_matrix: np.ndarray
    schedule_detail: pd.DataFrame


def _fmt_slot(start_interval: int, length: int) -> str:
    start_minutes = start_interval * 30
    end_minutes = (start_interval + length) * 30
    sh, sm = divmod(start_minutes, 60)
    eh, em = divmod(end_minutes, 60)
    return f"{int(sh):02d}:{int(sm):02d}-{int(eh):02d}:{int(em):02d}"


def solve_run1_hexaly(
    required_matrix: np.ndarray,
    n_agents: int,
    time_limit_sec: float = 30.0,
    num_workers: int | None = None,
) -> ModelSolveResult:
    days, intervals = required_matrix.shape
    if (days, intervals) != (7, 48):
        raise ValueError(f"Run1 expects 7x48 matrix, got {required_matrix.shape}")

    starts = list(range(35))  # 0..34 for 14 intervals
    r_int = np.rint(required_matrix * SCALE).astype(int)
    max_r = int(r_int.max())

    with HexalyOptimizer() as optimizer:
        model = optimizer.model

        y = [[model.bool() for _ in range(days)] for _ in range(n_agents)]
        s_start = [[model.bool() for _ in starts] for _ in range(n_agents)]
        z = [[[model.bool() for _ in starts] for _ in range(days)] for _ in range(n_agents)]

        for k in range(n_agents):
            model.constraint(model.sum(y[k][d] for d in range(days)) == 6)
            model.constraint(model.sum(s_start[k][r] for r in starts) == 1)
            for d in range(days):
                for r in starts:
                    model.constraint(z[k][d][r] <= y[k][d])
                    model.constraint(z[k][d][r] <= s_start[k][r])
                    model.constraint(z[k][d][r] >= y[k][d] + s_start[k][r] - 1)
                model.constraint(model.sum(z[k][d][r] for r in starts) == y[k][d])

        starts_covering: dict[int, list[int]] = {}
        for j in range(intervals):
            starts_covering[j] = [r for r in starts if r <= j <= r + 13]

        under = [[model.int(0, max_r + SCALE * n_agents) for _ in range(intervals)] for _ in range(days)]
        for d in range(days):
            for j in range(intervals):
                c_expr = model.sum(z[k][d][r] for k in range(n_agents) for r in starts_covering[j])
                model.constraint(under[d][j] >= int(r_int[d, j]) - SCALE * c_expr)
                model.constraint(under[d][j] >= 0)

        total_under = model.sum(under[d][j] for d in range(days) for j in range(intervals))
        model.minimize(total_under)
        model.close()

        optimizer.param.time_limit = int(max(1, round(time_limit_sec)))
        if num_workers is not None and num_workers > 0:
            optimizer.param.nb_threads = int(num_workers)
        optimizer.param.verbosity = 0
        optimizer.solve()

        status_name = optimizer.solution.status.name
        planned = np.zeros((days, intervals), dtype=float)
        schedule_rows: list[dict[str, str | int]] = []
        objective_value = float("inf")

        if status_name in {"OPTIMAL", "FEASIBLE"}:
            objective_value = float(total_under.value) / SCALE
            for d in range(days):
                for j in range(intervals):
                    planned[d, j] = float(sum(z[k][d][r].value for k in range(n_agents) for r in starts_covering[j]))

            day_cols = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            for k in range(n_agents):
                chosen_start = next(r for r in starts if int(round(s_start[k][r].value)) == 1)
                row: dict[str, str | int] = {"employee": k + 1}
                slot = _fmt_slot(chosen_start, 14)
                for d_idx, day_name in enumerate(day_cols):
                    row[day_name] = slot if int(round(y[k][d_idx].value)) == 1 else "OFF"
                schedule_rows.append(row)

        schedule_detail = (
            pd.DataFrame(schedule_rows)
            if schedule_rows
            else pd.DataFrame(columns=["employee", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        )

        return ModelSolveResult(
            mode="run1",
            n_agents=n_agents,
            solver_status=status_name,
            objective_value=objective_value,
            runtime_sec=float(optimizer.statistics.running_time),
            planned_matrix=planned,
            schedule_detail=schedule_detail,
        )


def solve_run2_hexaly(
    required_matrix: np.ndarray,
    n_agents: int,
    time_limit_sec: float = 30.0,
    num_workers: int | None = None,
) -> ModelSolveResult:
    days, intervals = required_matrix.shape
    if (days, intervals) != (7, 48):
        raise ValueError(f"Run2 expects 7x48 matrix, got {required_matrix.shape}")

    starts = list(range(intervals))  # 0..47
    feasible_pairs = [(r, length) for r in starts for length in LENGTH_OPTIONS if r + length <= intervals]
    r_int = np.rint(required_matrix * SCALE).astype(int)
    max_r = int(r_int.max())

    with HexalyOptimizer() as optimizer:
        model = optimizer.model

        y = [[model.bool() for _ in range(days)] for _ in range(n_agents)]
        s_start = [[model.bool() for _ in starts] for _ in range(n_agents)]
        l_sel = [[[model.bool() for _ in LENGTH_OPTIONS] for _ in range(days)] for _ in range(n_agents)]
        z = [
            [
                {(r, length): model.bool() for (r, length) in feasible_pairs}
                for _ in range(days)
            ]
            for _ in range(n_agents)
        ]

        for k in range(n_agents):
            model.constraint(model.sum(s_start[k][r] for r in starts) == 1)
            worked_days = model.sum(y[k][d] for d in range(days))
            model.constraint(worked_days >= 5)
            model.constraint(worked_days <= 6)

            weekly_len_terms = []
            for d in range(days):
                model.constraint(model.sum(l_sel[k][d][i] for i in range(len(LENGTH_OPTIONS))) == y[k][d])
                weekly_len_terms.extend(
                    LENGTH_OPTIONS[i] * l_sel[k][d][i] for i in range(len(LENGTH_OPTIONS))
                )

                for (r, length) in feasible_pairs:
                    li = LENGTH_OPTIONS.index(length)
                    model.constraint(z[k][d][(r, length)] <= s_start[k][r])
                    model.constraint(z[k][d][(r, length)] <= l_sel[k][d][li])
                    model.constraint(z[k][d][(r, length)] >= s_start[k][r] + l_sel[k][d][li] - 1)

                model.constraint(model.sum(z[k][d][(r, length)] for (r, length) in feasible_pairs) == y[k][d])

            model.constraint(model.sum(weekly_len_terms) == 84)

        for k in range(n_agents - 1):
            model.constraint(model.sum(y[k][d] for d in range(days)) >= model.sum(y[k + 1][d] for d in range(days)))

        covers: dict[int, list[tuple[int, int]]] = {j: [] for j in range(intervals)}
        for r, length in feasible_pairs:
            for j in range(r, r + length):
                covers[j].append((r, length))

        under = [[model.int(0, max_r + SCALE * n_agents) for _ in range(intervals)] for _ in range(days)]
        for d in range(days):
            for j in range(intervals):
                c_expr = model.sum(z[k][d][pair] for k in range(n_agents) for pair in covers[j])
                model.constraint(under[d][j] >= int(r_int[d, j]) - SCALE * c_expr)
                model.constraint(under[d][j] >= 0)

        total_under = model.sum(under[d][j] for d in range(days) for j in range(intervals))
        model.minimize(total_under)
        model.close()

        optimizer.param.time_limit = int(max(1, round(time_limit_sec)))
        if num_workers is not None and num_workers > 0:
            optimizer.param.nb_threads = int(num_workers)
        optimizer.param.verbosity = 0
        optimizer.solve()

        status_name = optimizer.solution.status.name
        planned = np.zeros((days, intervals), dtype=float)
        objective_value = float("inf")
        schedule_rows: list[dict[str, str | int]] = []
        day_cols = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        if status_name in {"OPTIMAL", "FEASIBLE"}:
            objective_value = float(total_under.value) / SCALE
            for d in range(days):
                for j in range(intervals):
                    planned[d, j] = float(sum(z[k][d][pair].value for k in range(n_agents) for pair in covers[j]))

            for k in range(n_agents):
                chosen_start = next(r for r in starts if int(round(s_start[k][r].value)) == 1)
                row: dict[str, str | int] = {"employee": k + 1}
                for d in range(days):
                    if int(round(y[k][d].value)) == 0:
                        row[day_cols[d]] = "OFF"
                    else:
                        chosen_len = next(
                            length
                            for i, length in enumerate(LENGTH_OPTIONS)
                            if int(round(l_sel[k][d][i].value)) == 1
                        )
                        row[day_cols[d]] = _fmt_slot(chosen_start, chosen_len)
                schedule_rows.append(row)

        schedule_detail = (
            pd.DataFrame(schedule_rows)
            if schedule_rows
            else pd.DataFrame(columns=["employee", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        )

        return ModelSolveResult(
            mode="run2",
            n_agents=n_agents,
            solver_status=status_name,
            objective_value=objective_value,
            runtime_sec=float(optimizer.statistics.running_time),
            planned_matrix=planned,
            schedule_detail=schedule_detail,
        )
