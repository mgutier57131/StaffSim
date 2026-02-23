"""Boundary smoothing utilities for weekly shapes."""

from __future__ import annotations

import numpy as np


def apply_local_boundary_smoothing(shape_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth day boundaries on a 7x48 shape matrix using local cubic interpolation.

    Returns:
    - smoothed_shape_matrix
    - jump_before (size 6)
    - jump_after (size 6)
    """
    if shape_matrix.shape != (7, 48):
        raise ValueError("shape_matrix must have shape (7, 48).")
    raw = shape_matrix.copy()
    s = shape_matrix.reshape(-1).copy()
    total_before = float(s.sum())

    boundaries = [48, 96, 144, 192, 240, 288]
    jump_before = np.array([abs(s[k - 1] - s[k]) for k in boundaries], dtype=float)

    t = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=float)
    h = 3.0 * t**2 - 2.0 * t**3
    for k in boundaries:
        i0, i1, i2, i3 = k - 2, k - 1, k, k + 1
        y_l = s[i0]
        y_r = s[i3]
        s[i0] = y_l + (y_r - y_l) * h[0]
        s[i1] = y_l + (y_r - y_l) * h[1]
        s[i2] = y_l + (y_r - y_l) * h[2]
        s[i3] = y_l + (y_r - y_l) * h[3]

    s = np.maximum(s, 1e-12)
    total_after = float(s.sum())
    if total_after <= 0:
        raise ValueError("Smoothed shape has non-positive total mass.")
    s *= total_before / total_after
    if abs(float(s.sum()) - total_before) >= 1e-9:
        raise ValueError("Boundary smoothing mass conservation failed.")

    jump_after = np.array([abs(s[k - 1] - s[k]) for k in boundaries], dtype=float)
    return s.reshape(7, 48), jump_before, jump_after


def apply_blend(
    shape_matrix: np.ndarray,
    k: int = 4,
    method: str = "raised-cosine",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper. New implementation uses local cubic smoothing."""
    _ = (k, method)
    return apply_local_boundary_smoothing(shape_matrix)
