"""Deterministic calls generation from weekly shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

AmplitudePreset = Literal["min", "mid", "max"]
AMPLITUDE_MULTIPLIER = {"min": 0.25, "mid": 0.50, "max": 1.00}


@dataclass(frozen=True)
class CallsGenerationResult:
    calls_matrix: np.ndarray
    expected_matrix: np.ndarray
    probabilities: np.ndarray
    z_week: np.ndarray
    delta_max: float
    delta_used: float
    k_calls_per_fte_interval: float
    v_min_needed: float


def calls_from_shape(
    v_week: int,
    weekly_shape_matrix: np.ndarray,
    *,
    aht: float,
    occ: float,
    t_interval: float = 0.5,
    fte_min: float = 1.0,
    amplitude_preset: AmplitudePreset = "mid",
    amplitude_scale: float | None = None,
    mode: str = "largest_remainder",
    seed: int = 12345,
) -> CallsGenerationResult:
    """
    Convert weekly shape -> expected calls -> deterministic integer calls.

    seed is kept for future stochastic modes but unused in deterministic mode.
    """
    _ = seed
    if mode != "largest_remainder":
        raise ValueError("Only mode='largest_remainder' is supported.")
    if v_week <= 0:
        raise ValueError("V must be > 0.")
    if weekly_shape_matrix.shape != (7, 48):
        raise ValueError("weekly_shape_matrix must have shape (7,48).")
    if aht <= 0 or occ <= 0 or t_interval <= 0:
        raise ValueError("AHT, OCC and T must be > 0.")
    if amplitude_preset not in AMPLITUDE_MULTIPLIER:
        raise ValueError(f"Invalid amplitude_preset: {amplitude_preset}")

    weekly_shape = weekly_shape_matrix.reshape(-1).astype(float)
    w_min = float(weekly_shape.min())
    w_max = float(weekly_shape.max())
    if w_max <= w_min:
        raise ValueError("weekly_shape has no variation (max==min).")
    z_week = (weekly_shape - w_min) / (w_max - w_min)
    z_week = np.maximum(z_week, 0.0)

    n = 336
    k_calls_per_fte_interval = (3600.0 * t_interval * occ) / aht
    v_min_needed = n * fte_min * k_calls_per_fte_interval
    if not (v_week / k_calls_per_fte_interval > n * fte_min):
        raise ValueError(
            f"Insufficient V for FTE_min={fte_min}. Need V > {v_min_needed:.6f}, got V={v_week}."
        )

    z_sum = float(z_week.sum())
    if z_sum <= 0:
        raise ValueError("z_week sum must be > 0.")
    delta_max = (v_week / k_calls_per_fte_interval - n * fte_min) / z_sum
    if amplitude_scale is None:
        delta_used = AMPLITUDE_MULTIPLIER[amplitude_preset] * delta_max
    else:
        if amplitude_scale < 0:
            raise ValueError("amplitude_scale must be >= 0.")
        delta_used = amplitude_scale * delta_max

    fte_target = fte_min + delta_used * z_week
    calls_expected = fte_target * k_calls_per_fte_interval
    calls_expected *= float(v_week) / float(calls_expected.sum())

    base = np.floor(calls_expected).astype(int)
    missing = int(v_week - base.sum())
    if missing > 0:
        frac = calls_expected - np.floor(calls_expected)
        order = np.argsort(-frac, kind="stable")
        base[order[:missing]] += 1

    p = weekly_shape / weekly_shape.sum()
    return CallsGenerationResult(
        calls_matrix=base.reshape(7, 48).astype(int),
        expected_matrix=calls_expected.reshape(7, 48),
        probabilities=p,
        z_week=z_week,
        delta_max=float(delta_max),
        delta_used=float(delta_used),
        k_calls_per_fte_interval=float(k_calls_per_fte_interval),
        v_min_needed=float(v_min_needed),
    )
