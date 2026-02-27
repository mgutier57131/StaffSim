"""Coverage and delta metrics for scheduling outputs."""

from __future__ import annotations

import numpy as np


def compute_under_over_delta(required: np.ndarray, planned: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = planned - required
    under = np.maximum(0.0, required - planned)
    over = np.maximum(0.0, planned - required)
    return under, over, delta


def compute_coverage(required: np.ndarray, under: np.ndarray) -> float:
    total_required = float(required.sum())
    if total_required <= 0:
        return 1.0
    return 1.0 - float(under.sum()) / total_required

