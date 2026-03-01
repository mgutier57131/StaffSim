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


def _read_metric_file(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    # Legacy format: metric,value rows
    if "metric" in df.columns and "value" in df.columns:
        return {str(k): str(v) for k, v in zip(df["metric"], df["value"], strict=False)}
    # Wide format: 1-row table with metric columns
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {str(k): str(v) for k, v in row.items()}


def read_required_matrix(run_dir: Path) -> np.ndarray:
    path = run_dir / "fte_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return _read_matrix_csv(path)


def read_summary_metrics(run_dir: Path) -> dict[str, str]:
    path = run_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    metrics = _read_metric_file(path)
    if not metrics:
        raise ValueError(f"summary.csv is empty or invalid: {path}")
    return metrics


def read_n0_from_summary(run_dir: Path) -> int:
    metrics = read_summary_metrics(run_dir)
    raw = metrics.get("HC_gross_ceil")
    if raw is None:
        raise ValueError("summary.csv missing HC_gross_ceil.")
    return int(float(raw))


def read_headcount_refs(run_dir: Path) -> dict[str, float | int]:
    """Read comparable HC references from summary.csv."""
    metrics = read_summary_metrics(run_dir)
    out: dict[str, float | int] = {}
    if "HC_gross" in metrics:
        out["HC_gross"] = round(float(metrics["HC_gross"]), 3)
    if "HC_gross_ceil" in metrics:
        out["HC_gross_ceil"] = int(float(metrics["HC_gross_ceil"]))
    if "HC_teorico" in metrics:
        out["HC_teorico"] = round(float(metrics["HC_teorico"]), 3)
    if "HC_teorico_ceil" in metrics:
        out["HC_teorico_ceil"] = int(float(metrics["HC_teorico_ceil"]))
    return out


def ensure_run_inputs(run_dir: Path) -> None:
    for name in ("fte_matrix.csv", "summary.csv"):
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"Missing required input: {run_dir / name}")


def make_final_output_dir(run_dir: Path, mode: str) -> Path:
    out = run_dir / "schedule" / mode
    out.mkdir(parents=True, exist_ok=True)
    # Best-effort cleanup of prior outputs to keep a single folder per mode.
    for name in [
        "required_matrix.csv",
        "planned_matrix.csv",
        "under_matrix.csv",
        "over_matrix.csv",
        "delta_matrix.csv",
        "schedule_detail.csv",
        "ilp_summary.csv",
        "schedule_curve.png",
        "search_log.txt",
    ]:
        p = out / name
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
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


def write_schedule_detail(path: Path, detail_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(path, index=False)


def write_search_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    extra_metrics: dict[str, float | int | str] | None = None,
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
    if extra_metrics:
        for k, v in extra_metrics.items():
            rows.append((k, v))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def write_unified_summary_table(run_dir: Path) -> Path:
    """
    Build a unified 1-row summary table in run_dir/summary.csv:
    - demand metrics from summary.csv
    - run1 metrics from schedule/run1/ilp_summary.csv (prefixed run1_)
    - run2 metrics from schedule/run2/ilp_summary.csv (prefixed run2_)
    """
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv in {run_dir}")

    demand = _read_metric_file(summary_path)
    if not demand:
        raise ValueError(f"Could not parse demand summary at {summary_path}")

    def _prefixed_ilp(mode: str) -> dict[str, str]:
        ilp_path = run_dir / "schedule" / mode / "ilp_summary.csv"
        if not ilp_path.exists():
            return {}
        raw = _read_metric_file(ilp_path)
        return {f"{mode}_{k}": v for k, v in raw.items()}

    run1 = _prefixed_ilp("run1")
    run2 = _prefixed_ilp("run2")

    merged: dict[str, str] = {}
    merged.update(demand)
    merged.update(run1)
    merged.update(run2)

    df = pd.DataFrame([merged])
    df.to_csv(summary_path, index=False)
    return summary_path
