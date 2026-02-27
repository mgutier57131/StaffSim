"""Unified review app for demand KPIs and scheduling outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DAY_COLS = [f"t{i:02d}" for i in range(48)]


def _list_runs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def _read_metric_csv(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    if "metric" not in df.columns or "value" not in df.columns:
        return {}
    return {str(k): str(v) for k, v in zip(df["metric"], df["value"], strict=False)}


def _read_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "day" in df.columns:
        df = df.set_index("day")
    return df


def _matrix_intervals_x_days(df: pd.DataFrame) -> pd.DataFrame:
    out = df.T.copy()
    out.index.name = "interval"
    return out


def _run_sched_summary(run_dir: Path, mode: str) -> dict[str, str]:
    summary_path = run_dir / "schedule" / mode / "ilp_summary.csv"
    if not summary_path.exists():
        return {}
    return _read_metric_csv(summary_path)


def main() -> None:
    st.set_page_config(page_title="StaffSim Review", layout="wide")
    st.title("StaffSim Review")

    runs = _list_runs(Path("results"))
    if not runs:
        st.error("No runs found in ./results/")
        return

    run_names = [r.name for r in runs]
    default_idx = len(run_names) - 1
    selected_name = st.sidebar.selectbox("Run ID", run_names, index=default_idx)
    run_dir = Path("results") / selected_name
    st.sidebar.caption(f"Selected: {run_dir.as_posix()}")

    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        st.error(f"Missing summary.csv in {run_dir.as_posix()}")
        return
    demand = _read_metric_csv(summary_path)
    run1 = _run_sched_summary(run_dir, "run1")
    run2 = _run_sched_summary(run_dir, "run2")

    tab_kpi, tab_r1_m, tab_r1_h, tab_r2_m, tab_r2_h = st.tabs(
        ["KPIs", "Run1 Matrices", "Run1 Schedules", "Run2 Matrices", "Run2 Schedules"]
    )

    with tab_kpi:
        st.subheader("Demand KPIs")
        keys = [
            "V",
            "AHT",
            "OCC",
            "SHK",
            "Hg",
            "H_talk",
            "H_prod",
            "H_paid",
            "HC_teorico",
            "HC_gross",
            "HC_gross_ceil",
        ]
        ddf = pd.DataFrame([{"metric": k, "value": demand.get(k, "")} for k in keys])
        st.dataframe(ddf, hide_index=True, use_container_width=True)

        st.subheader("Scheduling Comparison")
        comp_rows = []
        for mode, src in [("run1", run1), ("run2", run2)]:
            if src:
                comp_rows.append(
                    {
                        "mode": mode,
                        "N_final": src.get("N", ""),
                        "coverage": src.get("coverage", ""),
                        "sum_under": src.get("sum_under", ""),
                        "sum_over": src.get("sum_over", ""),
                        "objective_value(sum_under)": src.get("objective_value(sum_under)", ""),
                        "solver_status": src.get("solver_status", ""),
                    }
                )
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
        else:
            st.info("No scheduling final outputs found yet for run1/run2.")

    with tab_r1_m:
        base = run_dir / "schedule" / "run1"
        if not base.exists():
            st.info("Run1 final folder not found.")
        else:
            for name in ["required_matrix.csv", "planned_matrix.csv", "under_matrix.csv", "over_matrix.csv", "delta_matrix.csv"]:
                p = base / name
                if p.exists():
                    st.markdown(f"**{name}**")
                    st.dataframe(_matrix_intervals_x_days(_read_matrix(p)), use_container_width=True)

    with tab_r1_h:
        p = run_dir / "schedule" / "run1" / "schedule_detail.csv"
        if p.exists():
            st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
        else:
            st.info("Run1 schedule_detail.csv not found.")

    with tab_r2_m:
        base = run_dir / "schedule" / "run2"
        if not base.exists():
            st.info("Run2 final folder not found.")
        else:
            for name in ["required_matrix.csv", "planned_matrix.csv", "under_matrix.csv", "over_matrix.csv", "delta_matrix.csv"]:
                p = base / name
                if p.exists():
                    st.markdown(f"**{name}**")
                    st.dataframe(_matrix_intervals_x_days(_read_matrix(p)), use_container_width=True)

    with tab_r2_h:
        p = run_dir / "schedule" / "run2" / "schedule_detail.csv"
        if p.exists():
            st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
        else:
            st.info("Run2 schedule_detail.csv not found.")


if __name__ == "__main__":
    main()
