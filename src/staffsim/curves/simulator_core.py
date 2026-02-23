"""Core functions for weekly curve simulation (ratio + convex mix)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WEEKDAY_STEP_DEFAULT = 0.02
T_INTERVAL_DEFAULT = 0.5


def build_day_weights(
    week_mode: str,
    *,
    p: float = 0.82,
    weekday_split: str = "uniform",
    weekday_step: float = WEEKDAY_STEP_DEFAULT,
) -> np.ndarray:
    """Build Mon..Sun weights summing exactly 1."""
    if week_mode == "W1":
        return np.ones(7, dtype=float) / 7.0
    if week_mode != "W2":
        raise ValueError("week_mode must be W1 or W2.")
    if p < 0.74 or p >= 1.0:
        raise ValueError("For W2, p must satisfy 0.74 <= p < 1.")

    if weekday_split == "uniform":
        m = np.array([1, 1, 1, 1, 1], dtype=float)
    elif weekday_split == "increasing-to-friday":
        m = np.array([1.0 + i * weekday_step for i in range(5)], dtype=float)
    elif weekday_split == "decreasing-to-friday":
        m = np.array([1.0 + i * weekday_step for i in range(5)], dtype=float)[::-1]
    else:
        raise ValueError("weekday_split must be uniform/increasing-to-friday/decreasing-to-friday.")

    weekdays = p * (m / m.sum())
    weekend = np.array([(1.0 - p) / 2.0, (1.0 - p) / 2.0], dtype=float)
    w = np.concatenate([weekdays, weekend])
    return w / float(w.sum())


def build_peak_shape_f(
    *,
    num_peaks: int,
    pos1: float,
    width1: float,
    pos2: float | None = None,
    width2: float | None = None,
    n_intervals: int = 48,
) -> np.ndarray:
    """Build smooth peak-only shape f_j (sum=1) using Gaussian bumps."""
    if num_peaks not in {1, 2}:
        raise ValueError("num_peaks must be 1 or 2.")
    if width1 <= 0:
        raise ValueError("width1 must be > 0.")
    if pos1 < 0 or pos1 >= n_intervals:
        raise ValueError(f"pos1 must be in [0,{n_intervals}).")
    if num_peaks == 2:
        if pos2 is None or width2 is None:
            raise ValueError("num_peaks=2 requires pos2 and width2.")
        if width2 <= 0:
            raise ValueError("width2 must be > 0.")
        if pos2 < 0 or pos2 >= n_intervals:
            raise ValueError(f"pos2 must be in [0,{n_intervals}).")
        if pos2 <= pos1:
            raise ValueError("For 2 peaks, pos2 must be > pos1.")

    j = np.arange(n_intervals, dtype=float)

    def _gaussian(center: float, width: float) -> np.ndarray:
        sigma = width / 2.0
        g = np.exp(-((j - center) ** 2) / (2.0 * sigma**2))
        return g / float(g.sum())

    g1 = _gaussian(pos1, width1)
    if num_peaks == 1:
        f = g1
    else:
        g2 = _gaussian(float(pos2), float(width2))
        f = 0.5 * g1 + 0.5 * g2
        f = f / float(f.sum())

    return f


def _ratio_for_lambda(lmbda: float, u: np.ndarray, f: np.ndarray) -> float:
    p = (1.0 - lmbda) * u + lmbda * f
    return float(np.max(p) / np.min(p))


def solve_lambda_for_ratio(
    *,
    f: np.ndarray,
    ratio_target: float,
    n_intervals: int = 48,
    tol: float = 1e-4,
    max_iter: int = 80,
) -> tuple[float, float, bool]:
    """
    Solve lambda in [0,1] so ratio(max/min) of p_j hits ratio_target.
    Returns: (lambda, ratio_real, capped_to_max_reachable)
    """
    if ratio_target < 1.0:
        raise ValueError("ratio_target must be >= 1.0.")

    u = np.ones(n_intervals, dtype=float) / n_intervals
    ratio_max = _ratio_for_lambda(1.0, u, f)
    if abs(ratio_target - 1.0) <= tol:
        return 0.0, 1.0, False
    if ratio_target >= ratio_max:
        return 1.0, ratio_max, True

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ratio_mid = _ratio_for_lambda(mid, u, f)
        if abs(ratio_mid - ratio_target) <= tol:
            return mid, ratio_mid, False
        if ratio_mid < ratio_target:
            lo = mid
        else:
            hi = mid
    lmbda = 0.5 * (lo + hi)
    return lmbda, _ratio_for_lambda(lmbda, u, f), False


def build_intraday_pattern_pj(
    *,
    num_peaks: int,
    pos1: float,
    width1: float,
    ratio_target: float,
    pos2: float | None = None,
    width2: float | None = None,
    n_intervals: int = 48,
) -> tuple[np.ndarray, float, float, bool]:
    """
    Build p_j from convex mix between uniform base and peak shape f.
    Returns: (p_j, lambda, ratio_real, capped_flag)
    """
    u = np.ones(n_intervals, dtype=float) / n_intervals
    f = build_peak_shape_f(
        num_peaks=num_peaks,
        pos1=pos1,
        width1=width1,
        pos2=pos2,
        width2=width2,
        n_intervals=n_intervals,
    )
    lmbda, ratio_real, capped = solve_lambda_for_ratio(f=f, ratio_target=ratio_target, n_intervals=n_intervals)
    p = (1.0 - lmbda) * u + lmbda * f
    p = p / float(p.sum())
    return p, lmbda, ratio_real, capped


def build_week_expected_matrix(v_week: int, day_weights: np.ndarray, intraday_pattern: np.ndarray) -> np.ndarray:
    """X_ij_expected = V * w_i * p_j."""
    if v_week <= 0:
        raise ValueError("V must be > 0.")
    w = np.asarray(day_weights, dtype=float)
    p = np.asarray(intraday_pattern, dtype=float)
    if w.shape != (7,):
        raise ValueError("day_weights must have shape (7,).")
    if p.shape != (48,):
        raise ValueError("intraday_pattern must have shape (48,).")
    if np.any(w < 0) or np.any(p < 0):
        raise ValueError("Weights/pattern must be non-negative.")
    if float(w.sum()) <= 0 or float(p.sum()) <= 0:
        raise ValueError("Weights/pattern sum must be > 0.")
    w = w / float(w.sum())
    p = p / float(p.sum())
    x = v_week * (w[:, None] * p[None, :])
    return x


def deterministic_rounding_largest_remainder(values: np.ndarray, total: int) -> np.ndarray:
    """Deterministic integer rounding preserving exact total."""
    x = np.asarray(values, dtype=float)
    base = np.floor(x).astype(int)
    r = int(total - base.sum())
    if r < 0:
        raise ValueError("Rounding produced base sum above total.")
    if r > 0:
        frac = x - np.floor(x)
        idx = np.argsort(-frac, kind="stable")
        base[idx[:r]] += 1
    return base


def compute_fte_matrix(
    calls_matrix: np.ndarray,
    *,
    aht: float,
    occ: float,
    t_interval: float = T_INTERVAL_DEFAULT,
) -> np.ndarray:
    return (calls_matrix.astype(float) * aht) / (3600.0 * t_interval * occ)


@dataclass(frozen=True)
class SimulationResult:
    day_weights: np.ndarray
    intraday_pattern: np.ndarray
    expected_matrix: np.ndarray
    calls_matrix: np.ndarray
    fte_matrix: np.ndarray
    ratio_target: float
    ratio_real: float
    lmbda: float
    ratio_capped: bool


def run_simulation(
    *,
    v_week: int,
    aht: float,
    occ: float,
    week_mode: str,
    p: float,
    weekday_split: str,
    num_peaks: int,
    pos1: float,
    width1: float,
    ratio_target: float,
    pos2: float | None = None,
    width2: float | None = None,
    weekday_step: float = WEEKDAY_STEP_DEFAULT,
    t_interval: float = T_INTERVAL_DEFAULT,
) -> SimulationResult:
    day_weights = build_day_weights(
        week_mode,
        p=p,
        weekday_split=weekday_split,
        weekday_step=weekday_step,
    )
    intraday_pattern, lmbda, ratio_real, ratio_capped = build_intraday_pattern_pj(
        num_peaks=num_peaks,
        pos1=pos1,
        width1=width1,
        pos2=pos2,
        width2=width2,
        ratio_target=ratio_target,
    )
    expected_matrix = build_week_expected_matrix(v_week, day_weights, intraday_pattern)
    calls_week = deterministic_rounding_largest_remainder(expected_matrix.reshape(-1), total=v_week)
    calls_matrix = calls_week.reshape(7, 48)
    fte_matrix = compute_fte_matrix(calls_matrix, aht=aht, occ=occ, t_interval=t_interval)
    return SimulationResult(
        day_weights=day_weights,
        intraday_pattern=intraday_pattern,
        expected_matrix=expected_matrix,
        calls_matrix=calls_matrix,
        fte_matrix=fte_matrix,
        ratio_target=ratio_target,
        ratio_real=ratio_real,
        lmbda=lmbda,
        ratio_capped=ratio_capped,
    )

