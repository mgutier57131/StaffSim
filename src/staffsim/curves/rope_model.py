"""Intraday shape model with smooth finite-support bumps."""

from __future__ import annotations

import numpy as np

from staffsim.curves.generator import build_day_probabilities, build_day_signal


def intraday_probabilities(
    mode: str,
    *,
    pos1: float = 24.0,
    length1: float = 10.0,
    amp1: float = 0.0,
    pos2: float = 36.0,
    length2: float = 10.0,
    amp2: float = 0.0,
    d_min: float = 8.0,
    base_level: float = 1.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    if mode == "Plana":
        return np.ones(48, dtype=float) / 48.0
    intraday_mode = "D1" if mode == "1 pico" else "D2"
    day_signal = build_day_signal(
        intraday_mode=intraday_mode,  # type: ignore[arg-type]
        pos1=pos1,
        length1=length1,
        amp1=amp1,
        pos2=pos2,
        length2=length2,
        amp2=amp2,
        d_min=d_min,
        base_level=base_level,
        epsilon=epsilon,
    )
    return build_day_probabilities(day_signal)

