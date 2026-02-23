"""Streamlit GUI for real-time weekly curve calibration."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from staffsim.curves.simulator_core import (
    MIN_WEEKDAY_SHARE,
    T_INTERVAL_DEFAULT,
    WEEKDAY_STEP_DEFAULT,
    run_simulation,
)
from staffsim.io.export import DAY_LABELS, export_results
from staffsim.workload.baseline import compute_baseline_summary

BASELINE_DEFAULTS: dict[str, Any] = {
    "v_week": 10000,
    "aht": 300.0,
    "occ_pct": 70.0,
    "shk_pct": 20.0,
    "hg": 42.0,
    "week_mode": "W1",
    "p": 0.82,
    "weekday_split": "uniform",
    "num_peaks": 1,
    "pos1": 24.0,
    "width1": 10.0,
    "pos2": 36.0,
    "width2": 10.0,
    "peak_ratio_mode": "equal",
    "peak_ratio": 1.4,
    "ratio_target": 1.0,
}


def _init_state() -> None:
    for key, value in BASELINE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_baseline() -> None:
    for key, value in BASELINE_DEFAULTS.items():
        st.session_state[key] = value


def _build_line_figure(y: np.ndarray, title: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3.6))
    x = np.arange(336)
    ax.plot(x, y, linewidth=1.5)
    for day_idx in range(1, 7):
        ax.axvline(48 * day_idx, color="gray", linewidth=0.8, alpha=0.35)
    # Major ticks every 12 intervals (6 hours) for a denser and clearer timeline.
    tick_positions = np.arange(0, 336, 12)
    tick_labels: list[str] = []
    for t in tick_positions:
        day = t // 48
        half_hours = (t % 48)
        hour = half_hours // 2
        minute = "30" if (half_hours % 2) else "00"
        tick_labels.append(f"{DAY_LABELS[int(day)]} {int(hour):02d}:{minute}")

    ax.set_title(title)
    ax.set_xlabel("Time (30-min intervals across week)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
    ax.grid(True, which="major", axis="both", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    return fig


def _build_params_text(params: dict[str, Any]) -> str:
    lines = ["StaffSim GUI Parameters", "====================="]
    for key, value in params.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def main() -> None:
    st.set_page_config(page_title="StaffSim GUI", layout="wide")
    st.title("StaffSim - Weekly Curve Simulator")
    _init_state()

    with st.sidebar:
        st.header("A) Volume and WFM Parameters")
        st.number_input(
            "Weekly Volume (calls)",
            min_value=1000,
            max_value=200000,
            step=100,
            key="v_week",
            help="Total weekly volume in calls.",
        )
        st.number_input(
            "AHT (seconds)",
            min_value=1.0,
            max_value=3600.0,
            step=5.0,
            key="aht",
            help="Average handle time in seconds per call.",
        )
        st.number_input(
            "OCC (%)",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key="occ_pct",
            help="Occupancy percentage (0 to 100).",
        )
        st.number_input(
            "SHK (%)",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key="shk_pct",
            help="Shrinkage percentage (0 to 100).",
        )
        st.number_input(
            "Paid Hours per Agent (weekly)",
            min_value=1.0,
            max_value=100.0,
            step=1.0,
            key="hg",
            help="Paid hours per agent per week.",
        )

        st.divider()
        st.header("B) Weekly Distribution")
        st.selectbox(
            "Week Mode",
            options=["W1", "W2"],
            key="week_mode",
            help="W1: uniform by day. W2: weekdays versus weekend split.",
        )
        if st.session_state["week_mode"] == "W2":
            st.number_input(
                "Weekday Share p (Mon-Fri)",
                min_value=float(MIN_WEEKDAY_SHARE),
                max_value=0.999,
                step=0.001,
                format="%.2f",
                key="p",
                help="Share of weekly volume assigned to weekdays (Mon-Fri).",
            )
            st.selectbox(
                "Weekday Split",
                options=["uniform", "increasing-to-friday", "decreasing-to-friday"],
                key="weekday_split",
                help="How weekday volume is distributed from Monday to Friday.",
            )
        else:
            st.session_state["weekday_split"] = "uniform"

        st.caption(f"Weekday step is fixed at {WEEKDAY_STEP_DEFAULT:.2f} (2%).")

        st.divider()
        st.header("C) Intraday Shape")
        st.selectbox(
            "Number of Peaks",
            options=[1, 2],
            key="num_peaks",
            help="Choose one or two peaks for the intraday pattern.",
        )
        if st.session_state["num_peaks"] == 2:
            pos1_now = float(st.session_state["pos1"])
            pos2_now = float(st.session_state["pos2"])
            if pos1_now >= 47.5:
                st.session_state["pos1"] = 47.0
                pos1_now = 47.0
            if pos2_now <= pos1_now:
                st.session_state["pos2"] = min(47.5, pos1_now + 0.5)

        st.number_input(
            "Peak Position 1",
            min_value=0.0,
            max_value=47.5,
            step=0.5,
            format="%.2f",
            key="pos1",
            help="Peak center in intervals 0..48 (0=00:00, 24=12:00).",
        )
        st.number_input(
            "Peak Width 1",
            min_value=1.0,
            max_value=24.0,
            step=0.5,
            format="%.2f",
            key="width1",
            help="Peak width in intervals. Sigma is width/2.",
        )
        if st.session_state["num_peaks"] == 2:
            st.number_input(
                "Peak Position 2",
                min_value=0.0,
                max_value=47.5,
                step=0.5,
                format="%.2f",
                key="pos2",
                help="Second peak center. Must be greater than Peak Position 1.",
            )
            st.number_input(
                "Peak Width 2",
                min_value=1.0,
                max_value=24.0,
                step=0.5,
                format="%.2f",
                key="width2",
                help="Second peak width in intervals. Sigma is width/2.",
            )
            st.selectbox(
                "Peak Height Mode",
                options=["equal", "peak1-higher", "peak2-higher"],
                key="peak_ratio_mode",
                help="Relative height between Peak 1 and Peak 2.",
            )
            if st.session_state["peak_ratio_mode"] != "equal":
                st.number_input(
                    "Peak Height Ratio",
                    min_value=1.0,
                    step=0.05,
                    format="%.2f",
                    key="peak_ratio",
                    help="Height ratio used in peak1-higher or peak2-higher mode.",
                )
        st.number_input(
            "Peak to Valley Ratio",
            min_value=1.0,
            step=0.05,
            format="%.2f",
            key="ratio_target",
            help="Ratio of peak to valley in intraday pattern. 1=flat, 2=double, 3=triple.",
        )

        st.divider()
        st.button("Reset to Flat Curve", on_click=_reset_baseline)

    if st.session_state["num_peaks"] == 2 and float(st.session_state["pos2"]) <= float(st.session_state["pos1"]):
        st.error("Validation error: for two peaks, Peak Position 2 must be greater than Peak Position 1.")
        return

    try:
        sim = run_simulation(
            v_week=int(st.session_state["v_week"]),
            aht=float(st.session_state["aht"]),
            occ=float(st.session_state["occ_pct"]) / 100.0,
            week_mode=str(st.session_state["week_mode"]),
            p=float(st.session_state["p"]),
            weekday_split=str(st.session_state["weekday_split"]),
            num_peaks=int(st.session_state["num_peaks"]),
            pos1=float(st.session_state["pos1"]),
            width1=float(st.session_state["width1"]),
            ratio_target=float(st.session_state["ratio_target"]),
            pos2=float(st.session_state["pos2"]) if int(st.session_state["num_peaks"]) == 2 else None,
            width2=float(st.session_state["width2"]) if int(st.session_state["num_peaks"]) == 2 else None,
            peak_ratio_mode=str(st.session_state["peak_ratio_mode"]) if int(st.session_state["num_peaks"]) == 2 else "equal",
            peak_ratio=float(st.session_state["peak_ratio"]) if int(st.session_state["num_peaks"]) == 2 else 1.4,
            weekday_step=WEEKDAY_STEP_DEFAULT,
            t_interval=T_INTERVAL_DEFAULT,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    baseline = compute_baseline_summary(
        v_week=int(st.session_state["v_week"]),
        aht=float(st.session_state["aht"]),
        occ=float(st.session_state["occ_pct"]) / 100.0,
        shk=float(st.session_state["shk_pct"]) / 100.0,
        hg=float(st.session_state["hg"]),
        t_interval=T_INTERVAL_DEFAULT,
    )

    calls_expected_week = sim.expected_matrix.reshape(-1)
    calls_week = sim.calls_matrix.reshape(-1)

    st.info(
        f"Ratio target: {float(st.session_state['ratio_target']):.2f} | "
        f"Ratio achieved: {sim.ratio_real:.2f} | Lambda: {sim.lmbda:.2f}"
    )

    st.subheader("Expected Calls (smooth)")
    expected_figure = _build_line_figure(calls_expected_week, "Expected Calls (smooth)", "Calls")
    st.pyplot(expected_figure, clear_figure=False, use_container_width=True)

    st.subheader("Final Integer Calls")
    calls_figure = _build_line_figure(calls_week, "Final Integer Calls", "Calls")
    st.pyplot(calls_figure, clear_figure=False, use_container_width=True)

    st.subheader("Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Target ratio", f"{float(st.session_state['ratio_target']):.2f}")
    m2.metric("Achieved ratio", f"{sim.ratio_real:.2f}")
    m3.metric("Lambda", f"{sim.lmbda:.2f}")
    m4, m5, m6 = st.columns(3)
    m4.metric("Pattern min / max", f"{sim.intraday_pattern.min():.2f} / {sim.intraday_pattern.max():.2f}")
    m5.metric("Calls sum", f"{int(sim.calls_matrix.sum())}")
    m6.metric("Calls min / max", f"{int(sim.calls_matrix.min())} / {int(sim.calls_matrix.max())}")
    m7, m8 = st.columns(2)
    m7.metric("FTE min / max", f"{float(sim.fte_matrix.min()):.2f} / {float(sim.fte_matrix.max()):.2f}")
    m8.metric("HC theoretical", f"{baseline.hc_teorico:.2f}")
    if sim.ratio_capped:
        st.warning("Requested ratio is above max reachable for this shape. Using lambda = 1.")

    st.subheader("Weekly Day Weights")
    weights_df = pd.DataFrame(
        {
            "Day": DAY_LABELS,
            "Weight": np.round(sim.day_weights, 6),
            "Percent": np.round(sim.day_weights * 100, 3),
        }
    )
    st.dataframe(weights_df, hide_index=True, use_container_width=True)

    params = {
        "Weekly Volume": int(st.session_state["v_week"]),
        "AHT Seconds": float(st.session_state["aht"]),
        "OCC Percent": float(st.session_state["occ_pct"]),
        "SHK Percent": float(st.session_state["shk_pct"]),
        "OCC": float(st.session_state["occ_pct"]) / 100.0,
        "SHK": float(st.session_state["shk_pct"]) / 100.0,
        "Paid Hours Weekly": float(st.session_state["hg"]),
        "T Interval Hours": T_INTERVAL_DEFAULT,
        "Week Mode": str(st.session_state["week_mode"]),
        "Weekday Share p": float(st.session_state["p"]) if str(st.session_state["week_mode"]) == "W2" else "",
        "Weekday Split": str(st.session_state["weekday_split"]) if str(st.session_state["week_mode"]) == "W2" else "",
        "Weekday Step": WEEKDAY_STEP_DEFAULT,
        "Number of Peaks": int(st.session_state["num_peaks"]),
        "Peak Position 1": float(st.session_state["pos1"]),
        "Peak Width 1": float(st.session_state["width1"]),
        "Peak Position 2": float(st.session_state["pos2"]) if int(st.session_state["num_peaks"]) == 2 else "",
        "Peak Width 2": float(st.session_state["width2"]) if int(st.session_state["num_peaks"]) == 2 else "",
        "Peak Height Mode": str(st.session_state["peak_ratio_mode"]) if int(st.session_state["num_peaks"]) == 2 else "",
        "Peak Height Ratio": float(st.session_state["peak_ratio"]) if int(st.session_state["num_peaks"]) == 2 else "",
        "Target Ratio": float(st.session_state["ratio_target"]),
        "Achieved Ratio": sim.ratio_real,
        "Lambda": sim.lmbda,
        "Ratio Capped": sim.ratio_capped,
        "Calls Generation": "largest remainder",
        "Day Weights Mon-Sun": [float(x) for x in sim.day_weights.tolist()],
    }
    params_text = _build_params_text(params)

    if st.button("Export Run", type="primary"):
        output_dir = export_results(
            calls_matrix=sim.calls_matrix,
            calls_expected_matrix=sim.expected_matrix,
            fte_matrix=sim.fte_matrix,
            params=params,
            summary=baseline,
            params_text=params_text,
            extra_metrics={
                "target_ratio": float(st.session_state["ratio_target"]),
                "achieved_ratio": float(sim.ratio_real),
                "lambda": float(sim.lmbda),
                "intraday_p_min": float(sim.intraday_pattern.min()),
                "intraday_p_max": float(sim.intraday_pattern.max()),
                "calls_sum": float(sim.calls_matrix.sum()),
                "calls_min": float(sim.calls_matrix.min()),
                "calls_max": float(sim.calls_matrix.max()),
                "fte_min": float(sim.fte_matrix.min()),
                "fte_max": float(sim.fte_matrix.max()),
            },
            figure=calls_figure,
            base_dir="results",
        )
        st.success(f"Saved results to: {output_dir.as_posix()}/")


if __name__ == "__main__":
    main()
