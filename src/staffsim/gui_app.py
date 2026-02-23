"""Streamlit GUI for real-time weekly curve calibration."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from staffsim.curves.simulator_core import T_INTERVAL_DEFAULT, WEEKDAY_STEP_DEFAULT, run_simulation
from staffsim.io.export import DAY_LABELS, export_results
from staffsim.workload.baseline import compute_baseline_summary

BASELINE_DEFAULTS: dict[str, Any] = {
    "v_week": 10000,
    "aht": 300.0,
    "occ": 0.70,
    "shk": 0.20,
    "hg": 42.0,
    "week_mode": "W1",
    "p": 0.82,
    "weekday_split": "uniform",
    "num_peaks": 1,
    "pos1": 24.0,
    "width1": 10.0,
    "pos2": 36.0,
    "width2": 10.0,
    "ratio_target": 1.0,
}


def _init_state() -> None:
    for k, v in BASELINE_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_baseline() -> None:
    for k, v in BASELINE_DEFAULTS.items():
        st.session_state[k] = v


def _build_line_figure(y: np.ndarray, title: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3.6))
    x = np.arange(336)
    ax.plot(x, y, linewidth=1.5)
    for d in range(1, 7):
        ax.axvline(48 * d, color="gray", linewidth=0.8, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel("Time (30-min intervals across week)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([48 * d for d in range(7)], [f"{DAY_LABELS[d]} 00:00" for d in range(7)], rotation=30)
    fig.tight_layout()
    return fig


def _build_params_text(params: dict[str, Any]) -> str:
    lines = ["StaffSim GUI Parameters", "====================="]
    for k, v in params.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines) + "\n"


def main() -> None:
    st.set_page_config(page_title="StaffSim GUI", layout="wide")
    st.title("StaffSim - Simulador de Curvas")
    _init_state()

    with st.sidebar:
        st.header("A) Volumen y parámetros WFM")
        st.number_input(
            "V (calls/semana)",
            min_value=1000,
            max_value=200000,
            step=100,
            key="v_week",
            help="Volumen total semanal (calls).",
        )
        st.number_input(
            "AHT (segundos)",
            min_value=1.0,
            max_value=3600.0,
            step=5.0,
            key="aht",
            help="Average Handle Time en segundos por call.",
        )
        st.slider("OCC", min_value=0.50, max_value=0.95, step=0.01, key="occ", help="Ocupación objetivo.")
        st.slider("SHK", min_value=0.00, max_value=0.40, step=0.01, key="shk", help="Shrinkage.")
        st.number_input("Hg", min_value=1.0, max_value=100.0, step=1.0, key="hg", help="Horas pagas por agente a la semana.")
        st.button("Reset a Curva Plana", on_click=_reset_baseline)

        st.divider()
        st.header("B) Semana")
        st.selectbox(
            "week_mode",
            options=["W1", "W2"],
            key="week_mode",
            help="W1: distribución uniforme. W2: L-V vs fin de semana.",
        )
        if st.session_state["week_mode"] == "W2":
            st.slider(
                "p (share L-V)",
                min_value=0.74,
                max_value=0.95,
                step=0.01,
                key="p",
                help="Porcentaje del volumen semanal asignado a L-V.",
            )
            st.selectbox(
                "weekday_split",
                options=["uniform", "increasing-to-friday", "decreasing-to-friday"],
                key="weekday_split",
                help="Cómo se reparte el volumen entre Mon..Fri.",
            )
        else:
            st.session_state["weekday_split"] = "uniform"

        st.caption(f"weekday_step fijo: {WEEKDAY_STEP_DEFAULT:.2f}")

        st.divider()
        st.header("C) Intradía")
        st.selectbox(
            "num_peaks",
            options=[1, 2],
            key="num_peaks",
            help="Cantidad de picos del patrón intradía.",
        )
        st.slider(
            "pos1",
            min_value=0.0,
            max_value=47.5,
            step=0.5,
            key="pos1",
            help="Centro del pico en intervalos 0..48 (0=00:00, 24=12:00).",
        )
        st.slider(
            "width1",
            min_value=1.0,
            max_value=24.0,
            step=0.5,
            key="width1",
            help="Ancho del pico (intervalos). Controla sigma = width/2.",
        )
        if st.session_state["num_peaks"] == 2:
            st.slider(
                "pos2",
                min_value=0.0,
                max_value=47.5,
                step=0.5,
                key="pos2",
                help="Centro del segundo pico (debe ser mayor a pos1).",
            )
            st.slider(
                "width2",
                min_value=1.0,
                max_value=24.0,
                step=0.5,
                key="width2",
                help="Ancho del segundo pico (intervalos).",
            )
        st.slider(
            "ratio_target",
            min_value=1.0,
            max_value=5.0,
            step=0.05,
            key="ratio_target",
            help="Relación pico/valle del patrón intradía. 1=plano, 2=doble, 3=triple.",
        )

    if st.session_state["num_peaks"] == 2 and float(st.session_state["pos2"]) <= float(st.session_state["pos1"]):
        st.error("Validación: para 2 picos, pos2 debe ser mayor que pos1.")
        return

    try:
        sim = run_simulation(
            v_week=int(st.session_state["v_week"]),
            aht=float(st.session_state["aht"]),
            occ=float(st.session_state["occ"]),
            week_mode=str(st.session_state["week_mode"]),
            p=float(st.session_state["p"]),
            weekday_split=str(st.session_state["weekday_split"]),
            num_peaks=int(st.session_state["num_peaks"]),
            pos1=float(st.session_state["pos1"]),
            width1=float(st.session_state["width1"]),
            ratio_target=float(st.session_state["ratio_target"]),
            pos2=float(st.session_state["pos2"]) if int(st.session_state["num_peaks"]) == 2 else None,
            width2=float(st.session_state["width2"]) if int(st.session_state["num_peaks"]) == 2 else None,
            weekday_step=WEEKDAY_STEP_DEFAULT,
            t_interval=T_INTERVAL_DEFAULT,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    baseline = compute_baseline_summary(
        v_week=int(st.session_state["v_week"]),
        aht=float(st.session_state["aht"]),
        occ=float(st.session_state["occ"]),
        shk=float(st.session_state["shk"]),
        hg=float(st.session_state["hg"]),
        t_interval=T_INTERVAL_DEFAULT,
    )

    calls_expected_week = sim.expected_matrix.reshape(-1)
    calls_week = sim.calls_matrix.reshape(-1)

    left_col, right_col = st.columns([1, 2.4])

    with left_col:
        st.subheader("Métricas")
        st.metric("ratio logrado", f"{sim.ratio_real:.3f}")
        st.metric("lambda", f"{sim.lmbda:.4f}")
        st.metric("p_min / p_max", f"{sim.intraday_pattern.min():.6f} / {sim.intraday_pattern.max():.6f}")
        st.metric("sum calls", f"{int(sim.calls_matrix.sum())}")
        st.metric("calls min / max", f"{int(sim.calls_matrix.min())} / {int(sim.calls_matrix.max())}")
        st.metric("fte min / max", f"{float(sim.fte_matrix.min()):.3f} / {float(sim.fte_matrix.max()):.3f}")
        if sim.ratio_capped:
            st.warning("ratio máximo alcanzable con esta forma: se usó lambda=1.")

        st.subheader("Pesos semanales")
        w_df = pd.DataFrame({"Dia": DAY_LABELS, "weight": np.round(sim.day_weights, 6), "%": np.round(sim.day_weights * 100, 3)})
        st.dataframe(w_df, hide_index=True, use_container_width=True)

    with right_col:
        st.subheader("Calls esperadas (smooth)")
        exp_fig = _build_line_figure(calls_expected_week, "Calls esperadas (smooth)", "Calls")
        st.pyplot(exp_fig, clear_figure=False, use_container_width=True)

        st.subheader("Calls enteras finales")
        calls_fig = _build_line_figure(calls_week, "Calls enteras finales", "Calls")
        st.pyplot(calls_fig, clear_figure=False, use_container_width=True)

    params = {
        "V": int(st.session_state["v_week"]),
        "AHT": float(st.session_state["aht"]),
        "OCC": float(st.session_state["occ"]),
        "SHK": float(st.session_state["shk"]),
        "Hg": float(st.session_state["hg"]),
        "T": T_INTERVAL_DEFAULT,
        "week_mode": str(st.session_state["week_mode"]),
        "p": float(st.session_state["p"]) if str(st.session_state["week_mode"]) == "W2" else "",
        "weekday_split": str(st.session_state["weekday_split"]) if str(st.session_state["week_mode"]) == "W2" else "",
        "weekday_step": WEEKDAY_STEP_DEFAULT,
        "num_peaks": int(st.session_state["num_peaks"]),
        "pos1": float(st.session_state["pos1"]),
        "width1": float(st.session_state["width1"]),
        "pos2": float(st.session_state["pos2"]) if int(st.session_state["num_peaks"]) == 2 else "",
        "width2": float(st.session_state["width2"]) if int(st.session_state["num_peaks"]) == 2 else "",
        "ratio_target": float(st.session_state["ratio_target"]),
        "ratio_real": sim.ratio_real,
        "lambda": sim.lmbda,
        "ratio_capped": sim.ratio_capped,
        "calls_generation": "largest_remainder",
        "day_weights_mon_sun": [float(x) for x in sim.day_weights.tolist()],
    }
    params_text = _build_params_text(params)

    if st.button("Exportar corrida", type="primary"):
        out_dir = export_results(
            calls_matrix=sim.calls_matrix,
            calls_expected_matrix=sim.expected_matrix,
            fte_matrix=sim.fte_matrix,
            params=params,
            summary=baseline,
            params_text=params_text,
            extra_metrics={
                "ratio_target": float(st.session_state["ratio_target"]),
                "ratio_real": float(sim.ratio_real),
                "lambda": float(sim.lmbda),
                "intraday_p_min": float(sim.intraday_pattern.min()),
                "intraday_p_max": float(sim.intraday_pattern.max()),
                "calls_sum": float(sim.calls_matrix.sum()),
                "calls_min": float(sim.calls_matrix.min()),
                "calls_max": float(sim.calls_matrix.max()),
                "fte_min": float(sim.fte_matrix.min()),
                "fte_max": float(sim.fte_matrix.max()),
            },
            figure=calls_fig,
            base_dir="results",
        )
        st.success(f"Saved results to: {out_dir.as_posix()}/")


if __name__ == "__main__":
    main()

