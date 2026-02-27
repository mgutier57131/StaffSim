"""I/O helpers for scheduling runs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def resolve_run_dir(run_arg: str | None, base_results_dir: str | Path = "results") -> Path:
    if run_arg:
        run_dir = Path(run_arg)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")
        return run_dir

    base = Path(base_results_dir)
    if not base.exists():
        raise FileNotFoundError(f"Results folder not found: {base}")

    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs found in: {base}")
    return sorted(candidates)[-1]


def _read_matrix_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "day" not in df.columns:
        raise ValueError(f"CSV missing 'day' column: {path}")
    cols = [f"t{idx:02d}" for idx in range(48)]
    if any(c not in df.columns for c in cols):
        raise ValueError(f"CSV does not contain 48 interval columns: {path}")
    matrix = df[cols].to_numpy(dtype=float)
    if matrix.shape != (7, 48):
        raise ValueError(f"Expected 7x48 matrix in {path}, got {matrix.shape}")
    return matrix


def read_required_matrix(run_dir: Path) -> np.ndarray:
    path = run_dir / "fte_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return _read_matrix_csv(path)


def read_summary_metrics(run_dir: Path) -> dict[str, str]:
    path = run_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    df = pd.read_csv(path)
    if "metric" not in df.columns or "value" not in df.columns:
        raise ValueError(f"summary.csv must contain metric,value columns: {path}")
    return {str(k): str(v) for k, v in zip(df["metric"], df["value"], strict=False)}


def read_n0_from_summary(run_dir: Path) -> int:
    metrics = read_summary_metrics(run_dir)
    raw = metrics.get("HC_gross_ceil")
    if raw is None:
        raise ValueError("summary.csv missing HC_gross_ceil.")
    return int(float(raw))


def ensure_run_inputs(run_dir: Path) -> None:
    for name in ("fte_matrix.csv", "summary.csv"):
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"Missing required input: {run_dir / name}")


def make_output_dir(run_dir: Path, mode: str, n_agents: int) -> Path:
    out = run_dir / "schedule" / mode / f"N_{n_agents}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["day"] + [f"t{idx:02d}" for idx in range(48)]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for day_idx, row in enumerate(matrix):
            formatted = [round(float(v), 2) for v in row.tolist()]
            writer.writerow([DAY_LABELS[day_idx], *formatted])


def write_ilp_summary(
    path: Path,
    *,
    mode: str,
    n_agents: int,
    solver_status: str,
    objective_value: float,
    coverage: float,
    sum_required: float,
    sum_under: float,
    sum_over: float,
    runtime_sec: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("mode", mode),
        ("N", n_agents),
        ("solver_status", solver_status),
        ("objective_value(sum_under)", round(objective_value, 2)),
        ("coverage", round(coverage, 4)),
        ("sum_required", round(sum_required, 2)),
        ("sum_under", round(sum_under, 2)),
        ("sum_over", round(sum_over, 2)),
        ("runtime_sec", round(runtime_sec, 3)),
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

