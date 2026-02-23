import numpy as np

from staffsim.curves.simulator_core import run_simulation


def test_run_simulation_default_flat_curve() -> None:
    res = run_simulation(
        v_week=10000,
        aht=300.0,
        occ=0.7,
        week_mode="W1",
        p=0.82,
        weekday_split="uniform",
        num_peaks=1,
        pos1=24.0,
        width1=10.0,
        ratio_target=1.0,
    )
    assert res.expected_matrix.shape == (7, 48)
    assert res.calls_matrix.shape == (7, 48)
    assert int(res.calls_matrix.sum()) == 10000
    assert np.allclose(res.intraday_pattern, np.ones(48) / 48.0, atol=1e-9)


def test_run_simulation_w2_and_two_peaks() -> None:
    res = run_simulation(
        v_week=20000,
        aht=300.0,
        occ=0.7,
        week_mode="W2",
        p=0.82,
        weekday_split="decreasing-to-friday",
        num_peaks=2,
        pos1=12.0,
        width1=8.0,
        ratio_target=3.0,
        pos2=36.0,
        width2=7.0,
    )
    assert int(res.calls_matrix.sum()) == 20000
    assert np.isclose(float(res.day_weights[:5].sum()), 0.82, atol=1e-12)
    assert res.ratio_real >= 1.0
    assert np.min(res.fte_matrix) >= 0.0
