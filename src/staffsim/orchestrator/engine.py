"""Orchestrator engine: staged scenario execution with safe parquet merges."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ortools import __version__ as ORTOOLS_VERSION

from staffsim.demand.headless import run_headless as run_demand_headless
from staffsim.orchestrator.storage import (
    append_orchestration_log,
    append_parquet,
    cleanup_staging_dir,
    ensure_layout,
    load_or_create_scenarios,
    read_parquet,
    write_parquet,
    write_run_config,
)
from staffsim.scheduling.headless import run_headless as run_sched_headless
from staffsim.scheduling.plotting import plot_required_vs_planned

Stage = Literal["demand", "schedule", "both"]
Scheduler = Literal["cp_sat", "hexaly", "both"]
RunMode = Literal["run1", "run2"]
SolverName = Literal["cp_sat", "hexaly"]


@dataclass(frozen=True)
class OrchestratorConfig:
    out_dir: str
    parallel_runs: int = 4
    cp_sat_workers: int = 1
    base_seed: int = 12345
    run1_time_limit: float = 120.0
    run2_time_limit: float = 600.0
    retries_demand: int = 2
    retries_sched: int = 1
    regen_grid: bool = False
    coverage_target: float = 0.90
    max_expand: int = 256
    stage: Stage = "both"
    scheduler: Scheduler = "cp_sat"


def _hexaly_version_or_empty() -> str:
    try:
        import hexaly  # type: ignore

        return getattr(hexaly, "__version__", "") or "installed"
    except Exception:
        return ""


def _stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def _seed_for_base(base_seed: int, base_id: str) -> int:
    return int(base_seed + (_stable_int(base_id) % 1_000_000_000))


def _plain_row_dict(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if pd.isna(v):
            out[str(k)] = None
        elif isinstance(v, np.generic):
            out[str(k)] = v.item()
        else:
            out[str(k)] = v
    return out


def _matrix_to_long(base: dict[str, Any], matrix: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(matrix, dtype=float).reshape(-1)
    out = pd.DataFrame({"idx": np.arange(arr.size, dtype=int), "value": arr})
    for k, v in base.items():
        out[k] = v
    cols = list(base.keys()) + ["idx", "value"]
    return out[cols]


def _save_demand_image(path: Path, calls_matrix: np.ndarray) -> None:
    y = calls_matrix.reshape(-1)
    x = np.arange(y.size)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(x, y, linewidth=1.5)
    for d in range(1, 7):
        ax.axvline(48 * d, color="gray", linewidth=0.8, alpha=0.35)
    ax.set_title("Demand Calls Curve (Weekly)")
    ax.set_xlabel("Time (30-min intervals across week)")
    ax.set_ylabel("Calls")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _parse_slot(slot: str) -> tuple[str, str, int]:
    start, end = slot.split("-", maxsplit=1)
    sh, sm = start.split(":")
    eh, em = end.split(":")
    start_min = int(sh) * 60 + int(sm)
    end_min = int(eh) * 60 + int(em)
    length = max(0, int(round((end_min - start_min) / 30)))
    return start, end, length


def _detail_wide_to_long(
    detail_df: pd.DataFrame,
    *,
    scenario_id: str,
    base_id: str,
    schedule_case: str,
    solver: str,
) -> pd.DataFrame:
    cols = ["employee", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if detail_df.empty:
        return pd.DataFrame(
            columns=[
                "scenario_id",
                "base_id",
                "schedule_case",
                "solver",
                "employee_id",
                "day",
                "start",
                "end",
                "length",
                "off_flag",
            ]
        )
    for col in cols:
        if col not in detail_df.columns:
            raise ValueError(f"schedule_detail missing column: {col}")

    rows: list[dict[str, Any]] = []
    for _, r in detail_df.iterrows():
        employee = int(r["employee"])
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            val = str(r[day])
            if val == "OFF":
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "base_id": base_id,
                        "schedule_case": schedule_case,
                        "solver": solver,
                        "employee_id": employee,
                        "day": day,
                        "start": None,
                        "end": None,
                        "length": 0,
                        "off_flag": True,
                    }
                )
            else:
                start, end, length = _parse_slot(val)
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "base_id": base_id,
                        "schedule_case": schedule_case,
                        "solver": solver,
                        "employee_id": employee,
                        "day": day,
                        "start": start,
                        "end": end,
                        "length": length,
                        "off_flag": False,
                    }
                )
    return pd.DataFrame(rows)


def _selected_solvers(scheduler: Scheduler) -> list[SolverName]:
    if scheduler == "both":
        return ["cp_sat", "hexaly"]
    return [scheduler]  # type: ignore[list-item]


def _ensure_solver_dependencies(config: OrchestratorConfig) -> None:
    if config.stage not in {"schedule", "both"}:
        return
    needed = _selected_solvers(config.scheduler)
    if "hexaly" in needed:
        try:
            import hexaly  # type: ignore  # noqa: F401
            from hexaly.optimizer import HexalyOptimizer  # type: ignore

            # Fail fast if license/token is not available right now.
            with HexalyOptimizer():
                pass
        except Exception as exc:
            raise RuntimeError(
                "Hexaly solver selected but it is not ready in this Python environment "
                "(missing package or no free license token). "
                "Install in active interpreter and ensure token is free: "
                ".\\.venv\\Scripts\\python.exe -m pip install -i https://pip.hexaly.com hexaly"
            ) from exc


def _existing_base_ids(out_dir: Path) -> set[str]:
    p = out_dir / "demand_kpi.parquet"
    if not p.exists():
        return set()
    df = read_parquet(p)
    if "base_id" not in df.columns:
        return set()
    return set(df["base_id"].astype(str).tolist())


def _existing_sched_keys(out_dir: Path) -> set[tuple[str, str]]:
    p = out_dir / "sched_kpi.parquet"
    if not p.exists():
        return set()
    df = read_parquet(p)
    if not {"scenario_id", "solver"}.issubset(df.columns):
        return set()
    return set((str(r["scenario_id"]), str(r["solver"])) for _, r in df.iterrows())


def _build_summary_long(out_dir: Path) -> None:
    scenarios_path = out_dir / "scenarios.csv"
    demand_path = out_dir / "demand_kpi.parquet"
    summary_path = out_dir / "summary.csv"
    if not demand_path.exists() or not scenarios_path.exists():
        return
    scenarios = pd.read_csv(scenarios_path).copy()
    demand = read_parquet(demand_path).copy()

    # Always expose one row per scenario + solver to keep summary shape stable.
    solver_df = pd.DataFrame({"solver": ["cp_sat", "hexaly"]})
    scenarios["__k"] = 1
    solver_df["__k"] = 1
    base = scenarios.merge(solver_df, on="__k", how="inner").drop(columns=["__k"])

    merged = base.merge(demand, on="base_id", how="left", suffixes=("", "_demand"))

    sched_path = out_dir / "sched_kpi.parquet"
    if sched_path.exists():
        sched = read_parquet(sched_path).copy()
        merged = merged.merge(
            sched,
            on=["scenario_id", "base_id", "schedule_case", "solver"],
            how="left",
            suffixes=("", "_sched"),
        )

    if "day_weights" in merged.columns:
        def _fmt_weights(v: Any) -> Any:
            if pd.isna(v):
                return v
            try:
                arr = json.loads(v) if isinstance(v, str) else v
                if isinstance(arr, list):
                    return json.dumps([round(float(x), 3) for x in arr])
            except Exception:
                return v
            return v

        merged["day_weights"] = merged["day_weights"].apply(_fmt_weights)

    num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        merged[num_cols] = merged[num_cols].round(3)
    merged.to_csv(summary_path, index=False, float_format="%.3f")


def _demand_worker(task: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    base_id = str(task["base_id"])
    params = dict(task["params"])
    out_dir = Path(task["out_dir"])
    retries = int(task["retries"])
    base_seed = int(task["base_seed"])
    seed_scenario = _seed_for_base(base_seed, base_id)

    last_error = ""
    for attempt in range(1, retries + 2):
        try:
            res = run_demand_headless(params, seed=seed_scenario)
            stage_dir = out_dir / "staging" / "demand" / base_id
            stage_dir.mkdir(parents=True, exist_ok=True)

            key = {"base_id": base_id}
            calls_long = _matrix_to_long(key, res.calls_matrix)
            expected_long = _matrix_to_long(key, res.expected_matrix)
            fte_long = _matrix_to_long(key, res.fte_matrix)

            kpi_row = {
                "base_id": base_id,
                "base_seed": base_seed,
                "seed_scenario": seed_scenario,
                **params,
                **res.kpis,
                "day_weights": json.dumps([round(float(x), 6) for x in res.day_weights.tolist()]),
            }
            kpi_df = pd.DataFrame([kpi_row])

            write_parquet(stage_dir / "demand_calls.parquet", calls_long)
            write_parquet(stage_dir / "demand_expected.parquet", expected_long)
            write_parquet(stage_dir / "demand_fte.parquet", fte_long)
            write_parquet(stage_dir / "demand_kpi.parquet", kpi_df)

            _save_demand_image(out_dir / "images" / "demand" / f"{base_id}.png", res.calls_matrix)
            return {
                "ok": True,
                "base_id": base_id,
                "attempt": attempt,
                "duration_sec": time.time() - t0,
                "stage_dir": stage_dir.as_posix(),
                "error": "",
            }
        except Exception as exc:  # pragma: no cover - worker runtime
            last_error = str(exc)
    return {
        "ok": False,
        "base_id": base_id,
        "attempt": retries + 1,
        "duration_sec": time.time() - t0,
        "stage_dir": "",
        "error": last_error[:400],
    }


def _sched_worker(task: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    scenario_id = str(task["scenario_id"])
    base_id = str(task["base_id"])
    mode = str(task["schedule_case"])
    solver = str(task["solver"])
    out_dir = Path(task["out_dir"])
    retries = int(task["retries"])
    required = np.asarray(task["required"]).reshape(7, 48)
    n0 = int(task["n0"])
    coverage_target = float(task["coverage_target"])
    workers = int(task["workers"])
    max_expand = int(task["max_expand"])
    time_limit = float(task["time_limit"])
    base_seed = int(task["base_seed"])
    seed_scenario = _seed_for_base(base_seed, base_id)

    last_error = ""
    for attempt in range(1, retries + 2):
        try:
            sched = run_sched_headless(
                solver=solver,  # type: ignore[arg-type]
                required=required,
                mode=mode,  # type: ignore[arg-type]
                n0=n0,
                coverage_target=coverage_target,
                time_limit_sec=time_limit,
                workers=workers,
                max_expand=max_expand,
            )
            stage_dir = out_dir / "staging" / "scheduling" / f"{scenario_id}__{solver}"
            stage_dir.mkdir(parents=True, exist_ok=True)

            key = {
                "scenario_id": scenario_id,
                "base_id": base_id,
                "schedule_case": mode,
                "solver": solver,
            }
            write_parquet(stage_dir / "planned.parquet", _matrix_to_long(key, sched.planned_matrix))
            write_parquet(stage_dir / "under.parquet", _matrix_to_long(key, sched.under_matrix))
            write_parquet(stage_dir / "over.parquet", _matrix_to_long(key, sched.over_matrix))
            write_parquet(stage_dir / "delta.parquet", _matrix_to_long(key, sched.delta_matrix))

            kpi_df = pd.DataFrame(
                [
                    {
                        "scenario_id": scenario_id,
                        "base_id": base_id,
                        "schedule_case": mode,
                        "solver": solver,
                        "base_seed": base_seed,
                        "seed_scenario": seed_scenario,
                        "workers": workers,
                        "ortools_version": ORTOOLS_VERSION,
                        "hexaly_version": _hexaly_version_or_empty(),
                        "n0": n0,
                        "N_final": sched.n_final,
                        "solver_status": sched.solver_status,
                        "objective": round(float(sched.objective_value), 4),
                        "coverage": round(float(sched.coverage), 6),
                        "coverage_target": coverage_target,
                        "coverage_fail": bool(sched.coverage_fail),
                        "sum_required": round(float(sched.sum_required), 4),
                        "sum_under": round(float(sched.sum_under), 4),
                        "sum_over": round(float(sched.sum_over), 4),
                        "runtime_sec": round(float(sched.runtime_sec), 4),
                    }
                ]
            )
            write_parquet(stage_dir / "kpi.parquet", kpi_df)

            detail_long = _detail_wide_to_long(
                sched.schedule_detail,
                scenario_id=scenario_id,
                base_id=base_id,
                schedule_case=mode,
                solver=solver,
            )
            write_parquet(stage_dir / "detail.parquet", detail_long)

            (stage_dir / "search_log.txt").write_text("\n".join(sched.search_log_lines) + "\n", encoding="utf-8")
            plot_required_vs_planned(
                required,
                sched.planned_matrix,
                out_dir / "images" / "schedule" / mode / f"{scenario_id}_{solver}.png",
            )
            return {
                "ok": True,
                "scenario_id": scenario_id,
                "base_id": base_id,
                "mode": mode,
                "solver": solver,
                "attempt": attempt,
                "duration_sec": time.time() - t0,
                "stage_dir": stage_dir.as_posix(),
                "error": "",
            }
        except Exception as exc:  # pragma: no cover - worker runtime
            last_error = str(exc)

    return {
        "ok": False,
        "scenario_id": scenario_id,
        "base_id": base_id,
        "mode": mode,
        "solver": solver,
        "attempt": retries + 1,
        "duration_sec": time.time() - t0,
        "stage_dir": "",
        "error": last_error[:400],
    }


def _merge_demand_stage(out_dir: Path, stage_dir: Path) -> None:
    append_parquet(out_dir / "demand_calls.parquet", read_parquet(stage_dir / "demand_calls.parquet"), ["base_id", "idx"])
    append_parquet(
        out_dir / "demand_expected.parquet",
        read_parquet(stage_dir / "demand_expected.parquet"),
        ["base_id", "idx"],
    )
    append_parquet(out_dir / "demand_fte.parquet", read_parquet(stage_dir / "demand_fte.parquet"), ["base_id", "idx"])
    append_parquet(out_dir / "demand_kpi.parquet", read_parquet(stage_dir / "demand_kpi.parquet"), ["base_id"])


def _merge_sched_stage(out_dir: Path, stage_dir: Path) -> None:
    append_parquet(
        out_dir / "sched_planned.parquet",
        read_parquet(stage_dir / "planned.parquet"),
        ["scenario_id", "schedule_case", "solver", "idx"],
    )
    append_parquet(
        out_dir / "sched_under.parquet",
        read_parquet(stage_dir / "under.parquet"),
        ["scenario_id", "schedule_case", "solver", "idx"],
    )
    append_parquet(
        out_dir / "sched_over.parquet",
        read_parquet(stage_dir / "over.parquet"),
        ["scenario_id", "schedule_case", "solver", "idx"],
    )
    append_parquet(
        out_dir / "sched_delta.parquet",
        read_parquet(stage_dir / "delta.parquet"),
        ["scenario_id", "schedule_case", "solver", "idx"],
    )
    append_parquet(
        out_dir / "sched_kpi.parquet",
        read_parquet(stage_dir / "kpi.parquet"),
        ["scenario_id", "schedule_case", "solver"],
    )
    append_parquet(
        out_dir / "sched_detail.parquet",
        read_parquet(stage_dir / "detail.parquet"),
        ["scenario_id", "schedule_case", "solver", "employee_id", "day"],
    )


def _run_demand_phase(config: OrchestratorConfig, out_dir: Path, scenarios: pd.DataFrame) -> None:
    base_rows = scenarios.drop_duplicates(subset=["base_id"]).copy()
    existing_base_ids = _existing_base_ids(out_dir)

    demand_tasks: list[dict[str, Any]] = []
    for _, row in base_rows.iterrows():
        base_id = str(row["base_id"])
        if base_id in existing_base_ids:
            append_orchestration_log(
                out_dir,
                scenario_id=base_id,
                base_id=base_id,
                stage="demand",
                attempt=0,
                status="SKIP",
                duration_sec=0.0,
            )
            continue
        params = {
            "V": row["V"],
            "AHT": row["AHT"],
            "OCC": row["OCC"],
            "SHK": row["SHK"],
            "Hg": row["Hg"],
            "T": row["T"],
            "week_pattern": row["week_pattern"],
            "p_weekdays": row["p_weekdays"],
            "weekday_split": row["weekday_split"],
            "weekday_step": row["weekday_step"],
            "K": row["K"],
            "ratio_target": row["ratio_target"],
            "pos1": row["pos1"],
            "pos2": row["pos2"],
            "width1": row["width1"],
            "width2": row["width2"],
            "width_assignment_rule": row["width_assignment_rule"],
            "peak_amplitude_rule": row["peak_amplitude_rule"],
            "peak_amplitude_ratio": row["peak_amplitude_ratio"],
        }
        demand_tasks.append(
            {
                "base_id": base_id,
                "params": params,
                "out_dir": out_dir.as_posix(),
                "retries": int(config.retries_demand),
                "base_seed": int(config.base_seed),
            }
        )

    if not demand_tasks:
        return

    done = 0
    total = len(demand_tasks)
    with ProcessPoolExecutor(max_workers=int(config.parallel_runs)) as ex:
        futures = [ex.submit(_demand_worker, t) for t in demand_tasks]
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            base_id = str(res["base_id"])
            if res["ok"]:
                stage_dir = Path(str(res["stage_dir"]))
                _merge_demand_stage(out_dir, stage_dir)
                cleanup_staging_dir(stage_dir)
                append_orchestration_log(
                    out_dir,
                    scenario_id=base_id,
                    base_id=base_id,
                    stage="demand",
                    attempt=int(res["attempt"]),
                    status="OK",
                    duration_sec=float(res["duration_sec"]),
                )
            else:
                append_orchestration_log(
                    out_dir,
                    scenario_id=base_id,
                    base_id=base_id,
                    stage="demand",
                    attempt=int(res["attempt"]),
                    status="FAIL",
                    duration_sec=float(res["duration_sec"]),
                    error_message=str(res["error"]),
                )
                print(f"ERROR demand {base_id}: {res['error']}")
            if done % 10 == 0 or done == total:
                print(f"Demand done {done}/{total}")


def _load_required_maps(out_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if not (out_dir / "demand_fte.parquet").exists() or not (out_dir / "demand_kpi.parquet").exists():
        raise RuntimeError("Demand parquet files are required before schedule stage.")

    fte_df = read_parquet(out_dir / "demand_fte.parquet")
    kpi_df = read_parquet(out_dir / "demand_kpi.parquet")

    fte_map: dict[str, np.ndarray] = {}
    for base_id, g in fte_df.groupby("base_id"):
        ordered = g.sort_values("idx")
        vals = ordered["value"].to_numpy(dtype=float)
        if vals.size != 336:
            raise ValueError(f"base_id {base_id} has {vals.size} fte points; expected 336")
        fte_map[str(base_id)] = vals.reshape(7, 48)

    n0_map: dict[str, int] = {}
    for _, row in kpi_df.iterrows():
        bid = str(row["base_id"])
        if "HC_gross_ceil" not in row:
            raise ValueError("demand_kpi.parquet missing HC_gross_ceil")
        n0_map[bid] = int(float(row["HC_gross_ceil"]))
    return fte_map, n0_map


def _run_schedule_phase(config: OrchestratorConfig, out_dir: Path, scenarios: pd.DataFrame) -> None:
    fte_map, n0_map = _load_required_maps(out_dir)
    existing = _existing_sched_keys(out_dir)
    solvers = _selected_solvers(config.scheduler)

    tasks: list[dict[str, Any]] = []
    for _, row in scenarios.iterrows():
        plain = _plain_row_dict(row)
        scenario_id = str(plain["scenario_id"])
        base_id = str(plain["base_id"])
        mode = str(plain["schedule_case"])

        if base_id not in fte_map:
            append_orchestration_log(
                out_dir,
                scenario_id=scenario_id,
                base_id=base_id,
                stage=f"{mode}:missing_demand",
                attempt=1,
                status="FAIL",
                duration_sec=0.0,
                error_message="Missing demand_fte for base_id",
            )
            continue

        for solver in solvers:
            if (scenario_id, solver) in existing:
                append_orchestration_log(
                    out_dir,
                    scenario_id=scenario_id,
                    base_id=base_id,
                    stage=f"{mode}:{solver}",
                    attempt=0,
                    status="SKIP",
                    duration_sec=0.0,
                )
                continue
            tasks.append(
                {
                    "scenario_id": scenario_id,
                    "base_id": base_id,
                    "schedule_case": mode,
                    "solver": solver,
                    "required": fte_map[base_id].reshape(-1).tolist(),
                    "n0": int(n0_map[base_id]),
                    "coverage_target": float(config.coverage_target),
                    "workers": int(config.cp_sat_workers),
                    "max_expand": int(config.max_expand),
                    "time_limit": float(config.run1_time_limit if mode == "run1" else config.run2_time_limit),
                    "out_dir": out_dir.as_posix(),
                    "retries": int(config.retries_sched),
                    "base_seed": int(config.base_seed),
                }
            )

    if not tasks:
        return

    done = 0
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=int(config.parallel_runs)) as ex:
        futures = [ex.submit(_sched_worker, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            scenario_id = str(res["scenario_id"])
            base_id = str(res["base_id"])
            mode = str(res["mode"])
            solver = str(res["solver"])
            stage_name = f"{mode}:{solver}"
            if res["ok"]:
                stage_dir = Path(str(res["stage_dir"]))
                _merge_sched_stage(out_dir, stage_dir)
                search_log_dir = out_dir / "search_logs" / mode / solver
                search_log_dir.mkdir(parents=True, exist_ok=True)
                src_log = stage_dir / "search_log.txt"
                if src_log.exists():
                    (search_log_dir / f"{scenario_id}.txt").write_text(src_log.read_text(encoding="utf-8"), encoding="utf-8")
                cleanup_staging_dir(stage_dir)
                append_orchestration_log(
                    out_dir,
                    scenario_id=scenario_id,
                    base_id=base_id,
                    stage=stage_name,
                    attempt=int(res["attempt"]),
                    status="OK",
                    duration_sec=float(res["duration_sec"]),
                )
            else:
                append_orchestration_log(
                    out_dir,
                    scenario_id=scenario_id,
                    base_id=base_id,
                    stage=stage_name,
                    attempt=int(res["attempt"]),
                    status="FAIL",
                    duration_sec=float(res["duration_sec"]),
                    error_message=str(res["error"]),
                )
                print(f"ERROR {stage_name} {scenario_id}: {res['error']}")

            if done % 10 == 0 or done == total:
                print(f"Scheduling done {done}/{total}")


def orchestrate(config: OrchestratorConfig) -> None:
    out_dir = Path(config.out_dir)
    ensure_layout(out_dir)
    _ensure_solver_dependencies(config)

    scenarios = load_or_create_scenarios(out_dir, regen_grid=config.regen_grid)
    run_config = {
        **asdict(config),
        "parallel_runs": int(config.parallel_runs),
        "cp_sat_workers": int(config.cp_sat_workers),
        "ortools_version": ORTOOLS_VERSION,
        "hexaly_version": _hexaly_version_or_empty(),
        "scenario_count": int(len(scenarios)),
        "base_count": int(scenarios["base_id"].nunique()),
    }
    write_run_config(out_dir, run_config)

    if config.stage in {"demand", "both"}:
        _run_demand_phase(config, out_dir, scenarios)

    if config.stage in {"schedule", "both"}:
        _run_schedule_phase(config, out_dir, scenarios)

    _build_summary_long(out_dir)
