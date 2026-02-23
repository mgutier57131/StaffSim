"""Weekly day-weight helpers for GUI/CLI usage."""

from __future__ import annotations

import numpy as np

from staffsim.curves.generator import build_week_weights

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def uniform_weights() -> np.ndarray:
    return np.ones(7, dtype=float) / 7.0


def weekday_weekend_weights(
    p: float,
    submode: str,
    weekday_step: float = 0.02,
    peak_day: str = "Wed",
) -> np.ndarray:
    return build_week_weights(
        week_mode="W2",
        w2_p=p,
        weekday_submode=submode,  # type: ignore[arg-type]
        weekday_peak_day=peak_day,  # type: ignore[arg-type]
        weekday_step=weekday_step,
    )


def manual_weights(raw: np.ndarray) -> np.ndarray:
    w = np.asarray(raw, dtype=float)
    if w.shape != (7,):
        raise ValueError("Manual weights must have length 7.")
    if np.any(w < 0):
        raise ValueError("Manual weights must be non-negative.")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Manual weights sum must be > 0.")
    return w / s

