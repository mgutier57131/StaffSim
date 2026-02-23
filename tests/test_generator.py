import numpy as np
import pytest

from staffsim.curves.simulator_core import (
    build_day_weights,
    build_intraday_pattern_pj,
    build_peak_shape_f,
    build_week_expected_matrix,
    deterministic_rounding_largest_remainder,
)


def test_day_weights_w1_sum_one() -> None:
    w = build_day_weights("W1")
    assert w.shape == (7,)
    assert np.isclose(float(w.sum()), 1.0)


def test_day_weights_w2_weekday_share() -> None:
    p = 0.82
    w = build_day_weights("W2", p=p, weekday_split="increasing-to-friday")
    assert np.isclose(float(w.sum()), 1.0)
    assert np.isclose(float(w[:5].sum()), p)


def test_intraday_ratio_one_is_flat() -> None:
    p, lmbda, ratio_real, capped = build_intraday_pattern_pj(
        num_peaks=1,
        pos1=24.0,
        width1=10.0,
        ratio_target=1.0,
    )
    assert np.allclose(p, np.ones(48) / 48.0, atol=1e-9)
    assert np.isclose(lmbda, 0.0)
    assert np.isclose(ratio_real, 1.0)
    assert not capped


def test_peak_shape_two_peaks_requires_order() -> None:
    with pytest.raises(ValueError, match="pos2 must be > pos1"):
        build_peak_shape_f(num_peaks=2, pos1=30.0, width1=10.0, pos2=20.0, width2=10.0)


def test_expected_and_rounding_sum_exact() -> None:
    v = 12345
    w = build_day_weights("W2", p=0.82, weekday_split="uniform")
    p_j, _, _, _ = build_intraday_pattern_pj(num_peaks=2, pos1=10.0, width1=9.0, pos2=38.0, width2=9.0, ratio_target=2.2)
    x = build_week_expected_matrix(v, w, p_j)
    calls = deterministic_rounding_largest_remainder(x.reshape(-1), total=v)
    assert int(calls.sum()) == v
    assert np.all(calls >= 0)


def test_two_peak_crest_normalization_equal_and_ratio() -> None:
    pos1, pos2 = 10.0, 36.0
    width1, width2 = 6.0, 14.0

    f_equal = build_peak_shape_f(
        num_peaks=2,
        pos1=pos1,
        width1=width1,
        pos2=pos2,
        width2=width2,
        peak_ratio_mode="equal",
        peak_ratio=1.4,
    )
    f_ratio = build_peak_shape_f(
        num_peaks=2,
        pos1=pos1,
        width1=width1,
        pos2=pos2,
        width2=width2,
        peak_ratio_mode="peak1-higher",
        peak_ratio=1.4,
    )

    i1, i2 = int(round(pos1)), int(round(pos2))
    equal_ratio = float(f_equal[i1] / f_equal[i2])
    ratio_mode_ratio = float(f_ratio[i1] / f_ratio[i2])

    assert np.isclose(equal_ratio, 1.0, atol=0.05)
    assert np.isclose(ratio_mode_ratio, 1.4, atol=0.08)
