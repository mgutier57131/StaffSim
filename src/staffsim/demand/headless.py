"""Headless demand runner for orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from staffsim.curves.simulator_core import WEEKDAY_STEP_DEFAULT, run_simulation
from staffsim.workload.baseline import BaselineSummary, compute_baseline_summary


@dataclass(frozen=True)
class DemandHeadlessResult:
    calls_matrix: np.ndarray
    expected_matrix: np.ndarray
    fte_matrix: np.ndarray
    day_weights: np.ndarray
    intraday_pattern: np.ndarray
    baseline: BaselineSummary
    kpis: dict[str, float | int | str]


def _to_peak_ratio_cfg(peak_amplitude_rule: str | None, amplitude_ratio: float) -> tuple[str, float]:
    if not peak_amplitude_rule or peak_amplitude_rule == "equal":
        return "equal", 1.0
    if peak_amplitude_rule == "different_1_gt_2":
        return "peak1-higher", float(amplitude_ratio)
    if peak_amplitude_rule == "different_1_lt_2":
        return "peak2-higher", float(amplitude_ratio)
    raise ValueError(
        "peak_amplitude_rule must be one of equal/different_1_gt_2/different_1_lt_2 when provided."
    )


def _none_if_nan(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    return value


def run_headless(params: dict[str, Any], seed: int | None = None) -> DemandHeadlessResult:
    """
    Execute demand simulation without GUI/CSV side effects.

    Expected keys in params are aligned with orchestrator scenarios.csv columns.
    """
    if seed is not None:
        # Keep deterministic behavior in case stochastic components are introduced later.
        np.random.seed(int(seed))

    v_week = int(params["V"])
    aht = float(params.get("AHT", 300.0))
    occ = float(params.get("OCC", 0.70))
    shk = float(params.get("SHK", 0.20))
    hg = float(params.get("Hg", 42.0))
    t_interval = float(params.get("T", 0.5))

    week_pattern = str(params.get("week_pattern", "W1"))
    if week_pattern == "W1":
        p = 0.82
        weekday_split = "uniform"
        weekday_step = WEEKDAY_STEP_DEFAULT
    else:
        p = float(params.get("p_weekdays"))
        weekday_split = str(params.get("weekday_split"))
        weekday_step = float(params.get("weekday_step", WEEKDAY_STEP_DEFAULT))

    k = int(params.get("K", 1))
    pos1 = float(params.get("pos1"))
    width1 = float(params.get("width1"))
    ratio_target = float(params.get("ratio_target", 1.0))

    pos2 = _none_if_nan(params.get("pos2"))
    width2 = _none_if_nan(params.get("width2"))
    pos2_val = float(pos2) if pos2 is not None and pos2 == pos2 else None
    width2_val = float(width2) if width2 is not None and width2 == width2 else None

    peak_rule_raw = _none_if_nan(params.get("peak_amplitude_rule"))
    ratio_raw = _none_if_nan(params.get("peak_amplitude_ratio"))
    peak_ratio_mode, peak_ratio = _to_peak_ratio_cfg(
        str(peak_rule_raw) if peak_rule_raw is not None else None,
        float(ratio_raw if ratio_raw is not None else 2.0),
    )

    sim = run_simulation(
        v_week=v_week,
        aht=aht,
        occ=occ,
        week_mode=week_pattern,
        p=p,
        weekday_split=weekday_split,
        num_peaks=k,
        pos1=pos1,
        width1=width1,
        ratio_target=ratio_target,
        pos2=pos2_val,
        width2=width2_val,
        peak_ratio_mode=peak_ratio_mode,
        peak_ratio=peak_ratio,
        weekday_step=weekday_step,
        t_interval=t_interval,
    )

    baseline = compute_baseline_summary(
        v_week=v_week,
        aht=aht,
        occ=occ,
        shk=shk,
        hg=hg,
        t_interval=t_interval,
    )
    h_prod_noshk = baseline.h_prod
    hc_gross = h_prod_noshk / baseline.hg

    kpis: dict[str, float | int | str] = {
        "V": v_week,
        "AHT": aht,
        "OCC": occ,
        "SHK": shk,
        "Hg": hg,
        "T": t_interval,
        "H_talk": baseline.h_talk,
        "H_prod": baseline.h_prod,
        "H_paid": baseline.h_paid,
        "HC_teorico": baseline.hc_teorico,
        "HC_teorico_ceil": int(np.ceil(baseline.hc_teorico)),
        "H_prod_noSHK": h_prod_noshk,
        "HC_gross": hc_gross,
        "HC_gross_ceil": int(np.ceil(hc_gross)),
        "ratio_target": ratio_target,
        "ratio_real": sim.ratio_real,
        "lambda": sim.lmbda,
        "ratio_capped": str(sim.ratio_capped),
        "calls_sum": int(sim.calls_matrix.sum()),
        "calls_min": int(sim.calls_matrix.min()),
        "calls_max": int(sim.calls_matrix.max()),
        "fte_min": float(sim.fte_matrix.min()),
        "fte_max": float(sim.fte_matrix.max()),
    }

    return DemandHeadlessResult(
        calls_matrix=sim.calls_matrix,
        expected_matrix=sim.expected_matrix,
        fte_matrix=sim.fte_matrix,
        day_weights=sim.day_weights,
        intraday_pattern=sim.intraday_pattern,
        baseline=baseline,
        kpis=kpis,
    )
