"""Deterministic rounding helpers."""

from __future__ import annotations

import numpy as np


def largest_remainder(values: np.ndarray, total: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    base = np.floor(x).astype(int)
    r = int(total - base.sum())
    if r > 0:
        frac = x - np.floor(x)
        idx = np.argsort(-frac, kind="stable")
        base[idx[:r]] += 1
    return base

