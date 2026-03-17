"""Scenario grid generation for orchestration runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

FIXED_DEFAULTS: dict[str, float | int] = {
    "V": 7500,
    "AHT": 300,
    "OCC": 0.70,
    "SHK": 0.20,
    "Hg": 42,
    "T": 0.5,
}

WEEKDAY_STEP = 0.02
RATIO_TARGETS = [6, 4, 2]

WIDTHS_K1 = [16, 20, 24]
POS1_K1 = [14, 25, 36]

WIDTHS_ALLOWED_K2: dict[tuple[int, int], list[int]] = {
    (10, 28): [25, 20, 16],
    (12, 38): [17, 15, 13],
    (22, 40): [17, 15, 13],
}

WIDTH_ASSIGNMENT_RULES = ["equal", "extremes_high_low", "extremes_low_high"]
PEAK_AMPLITUDE_RULES = ["equal", "different_1_gt_2", "different_1_lt_2"]
AMPLITUDE_RATIO = 2
SCHEDULE_CASES = ["run1", "run2"]


def _base_id_for(payload: dict[str, Any]) -> str:
    key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "D_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _normalize_week_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "week_pattern": "W1",
            "p_weekdays": None,
            "weekday_split": None,
            "weekday_step": None,
        }
    )
    for p in [0.95, 0.85]:
        for split in ["uniform", "increasing-to-friday", "decreasing-to-friday"]:
            rows.append(
                {
                    "week_pattern": "W2",
                    "p_weekdays": p,
                    "weekday_split": split,
                    "weekday_step": WEEKDAY_STEP,
                }
            )
    return rows


def _intraday_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for ratio_target in RATIO_TARGETS:
        for pos1 in POS1_K1:
            for width in WIDTHS_K1:
                rows.append(
                    {
                        "K": 1,
                        "ratio_target": ratio_target,
                        "pos1": float(pos1),
                        "pos2": None,
                        "width1": float(width),
                        "width2": None,
                        "width_assignment_rule": None,
                        "peak_amplitude_rule": None,
                        "peak_amplitude_ratio": None,
                    }
                )

    for ratio_target in RATIO_TARGETS:
        for (pos1, pos2), allowed in WIDTHS_ALLOWED_K2.items():
            width_min = float(min(allowed))
            width_mid = float(sorted(allowed)[1])
            width_max = float(max(allowed))

            width_pairs: list[tuple[float, float, str]] = [
                (width_min, width_min, "equal"),
                (width_mid, width_mid, "equal"),
                (width_max, width_max, "equal"),
                (width_max, width_min, "extremes_high_low"),
                (width_min, width_max, "extremes_low_high"),
            ]

            for width1, width2, width_rule in width_pairs:
                for peak_rule in PEAK_AMPLITUDE_RULES:
                    rows.append(
                        {
                            "K": 2,
                            "ratio_target": ratio_target,
                            "pos1": float(pos1),
                            "pos2": float(pos2),
                            "width1": width1,
                            "width2": width2,
                            "width_assignment_rule": width_rule,
                            "peak_amplitude_rule": peak_rule,
                            "peak_amplitude_ratio": AMPLITUDE_RATIO,
                        }
                    )
    return rows


def _demand_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    week_rows = _normalize_week_rows()
    intraday_rows = _intraday_rows()
    for week in week_rows:
        for intra in intraday_rows:
            row: dict[str, Any] = {}
            row.update(FIXED_DEFAULTS)
            row.update(week)
            row.update(intra)
            demand_key_payload = {
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
            row["base_id"] = _base_id_for(demand_key_payload)
            rows.append(row)
    return rows


def build_scenarios_df() -> pd.DataFrame:
    demand_rows = _demand_rows()
    scenario_rows: list[dict[str, Any]] = []
    for row in demand_rows:
        for case in SCHEDULE_CASES:
            out = dict(row)
            out["schedule_case"] = case
            out["scenario_id"] = f"{row['base_id']}_{case}"
            scenario_rows.append(out)

    df = pd.DataFrame(scenario_rows)
    cols = [
        "base_id",
        "scenario_id",
        "schedule_case",
        "V",
        "AHT",
        "OCC",
        "SHK",
        "Hg",
        "T",
        "week_pattern",
        "p_weekdays",
        "weekday_split",
        "weekday_step",
        "K",
        "ratio_target",
        "pos1",
        "pos2",
        "width1",
        "width2",
        "width_assignment_rule",
        "peak_amplitude_rule",
        "peak_amplitude_ratio",
    ]
    df = df[cols].sort_values(["base_id", "schedule_case"]).reset_index(drop=True)
    return df
