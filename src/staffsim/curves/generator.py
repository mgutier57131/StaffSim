"""Weekly curve generation with a rope-and-clamp model (7x48)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

WEEK_MODE = Literal["W1", "W2"]
WEEKDAY_SUBMODE = Literal[
    "uniform",
    "increasing-to-friday",
    "decreasing-to-friday",
    "midweek-peak",
]
WEEKDAY_PEAK_DAY = Literal["Tue", "Wed", "Thu"]
INTRADAY_MODE = Literal["D1", "D2"]
L_PRESET = Literal["min", "mid", "max"]
AMP_PRESET = Literal["min", "mid", "max"]
D1_POSITION = Literal["inicio", "medio", "final"]
D2_POSITION = Literal["extremos", "ambos_inicio", "ambos_final"]
HEIGHT_REL = Literal["equal", "peak1_higher", "peak2_higher"]

WEEKDAY_STEP = 0.02
D_MIN_DEFAULT = 8.0
EPS_DEFAULT = 1e-6
BASE_LEVEL_DEFAULT = 1.0
SEED_DEFAULT = 12345
L_PRESET_TO_LEN = {"min": 2.0, "mid": 5.0, "max": 8.0}
AMP_PRESET_TO_A = {"min": 0.5, "mid": 1.0, "max": 1.8}
AMP_PRESET_TO_DELTA_FRAC = {"min": 0.25, "mid": 0.50, "max": 1.00}
D1_POSITION_TO_POS = {"inicio": 8.0, "medio": 24.0, "final": 40.0}
D2_POSITION_TO_POS = {
    "extremos": (10.0, 38.0),
    "ambos_inicio": (8.0, 18.0),
    "ambos_final": (30.0, 40.0),
}
RATIO_DEFAULT = 1.40


@dataclass(frozen=True)
class CurveConfig:
    v_week: int
    week_mode: WEEK_MODE
    intraday_mode: INTRADAY_MODE
    l_preset: L_PRESET
    amp_preset: AMP_PRESET
    w2_p: float | None = None
    weekday_submode: WEEKDAY_SUBMODE | None = None
    weekday_peak_day: WEEKDAY_PEAK_DAY | None = None
    d1_position: D1_POSITION | None = None
    d2_position: D2_POSITION | None = None
    d2_height_rel: HEIGHT_REL | None = None


@dataclass(frozen=True)
class CurveGenerationResult:
    calls_matrix: np.ndarray
    expected_matrix: np.ndarray
    weekly_shape_raw: np.ndarray
    weekly_shape_smooth: np.ndarray
    jump_before: np.ndarray
    jump_after: np.ndarray
    v_min_needed: float
    delta_max: float
    delta_used: float
    k_calls_per_fte_interval: float


def largest_remainder_round(expected: np.ndarray, total_int: int) -> np.ndarray:
    base = np.floor(expected).astype(int)
    missing = int(total_int - base.sum())
    if missing > 0:
        frac = expected - np.floor(expected)
        idx = np.argsort(-frac, kind="stable")
        base[idx[:missing]] += 1
    return base


def build_week_weights(
    week_mode: WEEK_MODE,
    w2_p: float | None = None,
    weekday_submode: WEEKDAY_SUBMODE | None = None,
    weekday_peak_day: WEEKDAY_PEAK_DAY | None = None,
    weekday_step: float = WEEKDAY_STEP,
) -> np.ndarray:
    if week_mode == "W1":
        return np.ones(7, dtype=float) / 7.0
    if w2_p is None or not (0.0 < w2_p < 1.0):
        raise ValueError("For W2, p must satisfy 0<p<1.")
    if w2_p < 0.74:
        raise ValueError("For W2, p must be >= 0.74.")
    if weekday_submode is None:
        raise ValueError("For W2, weekday_submode is required.")

    if weekday_submode == "uniform":
        m = np.array([1, 1, 1, 1, 1], dtype=float)
    elif weekday_submode == "increasing-to-friday":
        m = np.array([1.00, 1.02, 1.04, 1.06, 1.08], dtype=float)
    elif weekday_submode == "decreasing-to-friday":
        m = np.array([1.08, 1.06, 1.04, 1.02, 1.00], dtype=float)
    elif weekday_submode == "midweek-peak":
        peak = weekday_peak_day or "Wed"
        idx = {"Tue": 1, "Wed": 2, "Thu": 3}[peak]
        m = np.array([1.00, 1.00, 1.00, 1.00, 1.00], dtype=float)
        m[idx] += 2 * weekday_step
        if idx - 1 >= 0:
            m[idx - 1] += weekday_step
        if idx + 1 <= 4:
            m[idx + 1] += weekday_step
    else:
        raise ValueError(f"Unsupported weekday_submode: {weekday_submode}")

    w_weekdays = w2_p * (m / m.sum())
    return np.array([*w_weekdays, (1.0 - w2_p) / 2.0, (1.0 - w2_p) / 2.0], dtype=float)


def _phi_raised_cosine(t: np.ndarray, pos: float, length: float) -> np.ndarray:
    if length <= 1.0:
        raise ValueError("length must be > 1 interval.")
    h = length / 2.0
    u = (t - pos) / h
    return np.where(np.abs(u) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * u)), 0.0)


def build_day_signal(
    intraday_mode: INTRADAY_MODE,
    *,
    pos1: float,
    length1: float,
    amp1: float,
    pos2: float = 38.0,
    length2: float = 5.0,
    amp2: float = 1.0,
    d_min: float = D_MIN_DEFAULT,
    base_level: float = BASE_LEVEL_DEFAULT,
    epsilon: float = EPS_DEFAULT,
) -> np.ndarray:
    t = np.arange(48, dtype=float)
    y = np.full(48, base_level, dtype=float)
    y += amp1 * _phi_raised_cosine(t, pos1, length1)
    if intraday_mode == "D2":
        if abs(pos1 - pos2) < d_min:
            raise ValueError(f"D2 requires |pos1-pos2| >= {d_min}.")
        y += amp2 * _phi_raised_cosine(t, pos2, length2)

    bump_part = y - base_level
    delta = float(np.mean(bump_part))
    centered = bump_part - delta

    # Keep area conservation and avoid shape distortion from hard clamping:
    # if amplitude is too high, scale centered bump uniformly so min stays >= epsilon.
    neg = centered < 0.0
    alpha = 1.0
    if np.any(neg):
        alpha_limit = float(np.min((base_level - epsilon) / (-centered[neg])))
        alpha = max(0.0, min(1.0, alpha_limit))

    y_tilde = base_level + alpha * centered
    if float(np.min(y_tilde)) < epsilon:
        y_tilde = np.maximum(y_tilde, epsilon)
        y_tilde *= (48.0 * base_level) / float(y_tilde.sum())

    return y_tilde


def build_day_probabilities(day_signal: np.ndarray) -> np.ndarray:
    p_day = day_signal / float(day_signal.sum())
    return p_day


def smooth_weekly_shape_local_cubic(weekly_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = weekly_shape.copy().astype(float)
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
    s *= total_before / float(s.sum())
    if abs(float(s.sum()) - total_before) >= 1e-9:
        raise ValueError("Local smoothing failed mass conservation.")
    jump_after = np.array([abs(s[k - 1] - s[k]) for k in boundaries], dtype=float)
    return s, jump_before, jump_after


def generate_shape_matrix(
    week_mode: WEEK_MODE,
    intraday_mode: INTRADAY_MODE,
    *,
    p: float = 0.82,
    weekday_submode: WEEKDAY_SUBMODE = "uniform",
    weekday_peak_day: WEEKDAY_PEAK_DAY = "Wed",
    weekday_step: float = WEEKDAY_STEP,
    pos1: float = 24.0,
    pos2: float = 38.0,
    length1: float = 5.0,
    length2: float = 5.0,
    amp1: float = 1.0,
    amp2: float = 1.0,
    ratio_mode: HEIGHT_REL = "equal",
    ratio: float = RATIO_DEFAULT,
    d_min: float = D_MIN_DEFAULT,
    base_level: float = BASE_LEVEL_DEFAULT,
    epsilon: float = EPS_DEFAULT,
    day_weights_override: np.ndarray | None = None,
) -> np.ndarray:
    if intraday_mode == "D2":
        if ratio_mode == "peak1_higher":
            amp1, amp2 = amp1 * ratio, amp2
        elif ratio_mode == "peak2_higher":
            amp1, amp2 = amp1, amp2 * ratio

    if day_weights_override is not None:
        day_weights = np.asarray(day_weights_override, dtype=float)
        if day_weights.shape != (7,):
            raise ValueError("day_weights_override must have length 7.")
        if np.any(day_weights < 0):
            raise ValueError("day_weights_override must be non-negative.")
        s = float(day_weights.sum())
        if s <= 0:
            raise ValueError("day_weights_override sum must be > 0.")
        day_weights = day_weights / s
    else:
        day_weights = build_week_weights(
            week_mode=week_mode,
            w2_p=p if week_mode == "W2" else None,
            weekday_submode=weekday_submode if week_mode == "W2" else None,
            weekday_peak_day=weekday_peak_day if week_mode == "W2" else None,
            weekday_step=weekday_step,
        )
    day_signal = build_day_signal(
        intraday_mode,
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
    p_day = build_day_probabilities(day_signal)
    return day_weights[:, None] * p_day[None, :]


def generate_calls_matrix(
    config: CurveConfig,
    *,
    aht: float = 300.0,
    occ: float = 0.70,
    t_interval: float = 0.5,
    fte_min: float = 1.0,
    seed: int = SEED_DEFAULT,
) -> CurveGenerationResult:
    _ = seed
    if config.v_week <= 0:
        raise ValueError("V must be > 0.")

    length = L_PRESET_TO_LEN[config.l_preset]
    amp_base = AMP_PRESET_TO_A[config.amp_preset]
    if config.intraday_mode == "D1":
        if config.d1_position is None:
            raise ValueError("D1 requires d1_position.")
        pos1 = D1_POSITION_TO_POS[config.d1_position]
        pos2 = 38.0
        amp1, amp2 = amp_base, amp_base
    else:
        if config.d2_position is None or config.d2_height_rel is None:
            raise ValueError("D2 requires d2_position and d2_height_rel.")
        pos1, pos2 = D2_POSITION_TO_POS[config.d2_position]
        if config.d2_height_rel == "equal":
            amp1, amp2 = amp_base, amp_base
        elif config.d2_height_rel == "peak1_higher":
            amp1, amp2 = amp_base * RATIO_DEFAULT, amp_base
        else:
            amp1, amp2 = amp_base, amp_base * RATIO_DEFAULT

    shape_matrix = generate_shape_matrix(
        week_mode=config.week_mode,
        intraday_mode=config.intraday_mode,
        p=config.w2_p or 0.82,
        weekday_submode=config.weekday_submode or "uniform",
        weekday_peak_day=config.weekday_peak_day or "Wed",
        weekday_step=WEEKDAY_STEP,
        pos1=pos1,
        pos2=pos2,
        length1=length,
        length2=length,
        amp1=amp1,
        amp2=amp2,
        d_min=D_MIN_DEFAULT,
        base_level=BASE_LEVEL_DEFAULT,
        epsilon=EPS_DEFAULT,
    )

    day_weights = shape_matrix.sum(axis=1)
    weekly_shape_raw = shape_matrix.reshape(-1)
    weekly_shape_smooth, jump_before, jump_after = smooth_weekly_shape_local_cubic(weekly_shape_raw)

    k = (3600.0 * t_interval * occ) / aht
    v_min_needed = 336.0 * fte_min * k
    if config.v_week < v_min_needed:
        raise ValueError(f"Insufficient V. Need at least V_min_needed={v_min_needed:.6f}, got {config.v_week}.")

    # Rebuild expected calls by day: first distribute V by day, then intraday shape.
    weekly_shape_matrix_smooth = weekly_shape_smooth.reshape(7, 48)
    p_day_smooth = weekly_shape_matrix_smooth / weekly_shape_matrix_smooth.sum(axis=1, keepdims=True)
    v_day = config.v_week * day_weights
    calls_expected_matrix = v_day[:, None] * p_day_smooth
    calls_expected_week = calls_expected_matrix.reshape(-1)
    calls_expected_week *= config.v_week / float(calls_expected_week.sum())

    calls_int_week = largest_remainder_round(calls_expected_week, total_int=config.v_week)
    calls_matrix = calls_int_week.reshape(7, 48).astype(int)

    z_week = (weekly_shape_smooth - float(weekly_shape_smooth.min())) / (
        float(weekly_shape_smooth.max()) - float(weekly_shape_smooth.min())
    )
    z_sum = float(z_week.sum())
    delta_max = (config.v_week / k - 336.0 * fte_min) / z_sum if z_sum > 0 else 0.0
    delta_used = AMP_PRESET_TO_DELTA_FRAC[config.amp_preset] * delta_max

    return CurveGenerationResult(
        calls_matrix=calls_matrix,
        expected_matrix=calls_expected_matrix,
        weekly_shape_raw=weekly_shape_raw,
        weekly_shape_smooth=weekly_shape_smooth,
        jump_before=jump_before,
        jump_after=jump_after,
        v_min_needed=v_min_needed,
        delta_max=delta_max,
        delta_used=delta_used,
        k_calls_per_fte_interval=k,
    )
