"""Interval-level workload calculations."""

from __future__ import annotations

import numpy as np


def calls_to_fte(calls_matrix: np.ndarray, aht: float, occ: float, t_interval: float = 0.5) -> np.ndarray:
    return (calls_matrix.astype(float) * aht) / (3600.0 * t_interval * occ)


def apply_fte_floor(fte_matrix: np.ndarray, fte_floor: float = 0.5) -> np.ndarray:
    if fte_floor <= 0:
        return fte_matrix.copy()
    return np.maximum(fte_matrix, fte_floor)


def floor_metrics(fte_matrix_with_floor: np.ndarray, t_interval: float, shk: float, hg: float) -> dict[str, float]:
    h_prod_with_floor = float(fte_matrix_with_floor.sum() * t_interval)
    h_paid_with_floor = h_prod_with_floor / (1.0 - shk)
    hc_floor_implied = h_paid_with_floor / hg
    return {
        "H_prod_with_floor": h_prod_with_floor,
        "H_paid_with_floor": h_paid_with_floor,
        "HC_floor_implied": hc_floor_implied,
    }

