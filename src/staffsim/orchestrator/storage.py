"""Storage helpers for orchestrator outputs in results_sim/."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from staffsim.orchestrator.grid import build_scenarios_df


def ensure_layout(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "staging" / "demand").mkdir(parents=True, exist_ok=True)
    (out_dir / "staging" / "scheduling").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "demand").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "schedule" / "run1").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "schedule" / "run2").mkdir(parents=True, exist_ok=True)


def _require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Parquet output requires 'pyarrow'. Install with: py -m pip install pyarrow"
        ) from exc


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    _require_parquet_engine()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    _require_parquet_engine()
    return pd.read_parquet(path)


def append_parquet(path: Path, new_df: pd.DataFrame, dedupe_keys: list[str] | None = None) -> None:
    if new_df.empty:
        return
    if path.exists():
        old = read_parquet(path)
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df.copy()
    if dedupe_keys:
        out = out.drop_duplicates(subset=dedupe_keys, keep="last")
    write_parquet(path, out)


def read_existing_ids(path: Path, col: str) -> set[str]:
    if not path.exists():
        return set()
    df = read_parquet(path)
    if col not in df.columns:
        return set()
    return set(df[col].astype(str).tolist())


def load_or_create_scenarios(out_dir: Path, regen_grid: bool) -> pd.DataFrame:
    scenarios_path = out_dir / "scenarios.csv"
    if regen_grid or not scenarios_path.exists():
        df = build_scenarios_df()
        df.to_csv(scenarios_path, index=False)
        return df
    return pd.read_csv(scenarios_path)


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
            .strip()
        )
    except Exception:
        return ""


def write_run_config(out_dir: Path, config: dict[str, Any]) -> Path:
    path = out_dir / "run_config.json"
    merged = dict(config)
    merged.setdefault("generated_at", datetime.now().isoformat(timespec="seconds"))
    merged.setdefault("git_commit", _git_commit_hash())
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def append_orchestration_log(
    out_dir: Path,
    *,
    scenario_id: str,
    base_id: str,
    stage: str,
    attempt: int,
    status: str,
    duration_sec: float,
    error_message: str = "",
) -> None:
    path = out_dir / "orchestration_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(
                [
                    "scenario_id",
                    "base_id",
                    "stage",
                    "attempt",
                    "status",
                    "duration_sec",
                    "error_message",
                    "timestamp",
                ]
            )
        writer.writerow(
            [
                scenario_id,
                base_id,
                stage,
                int(attempt),
                status,
                round(float(duration_sec), 3),
                (error_message or "")[:400],
                datetime.now().isoformat(timespec="seconds"),
            ]
        )


def cleanup_staging_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
