import numpy as np
import importlib.util
import pytest

from staffsim.demand.headless import run_headless as run_demand_headless
from staffsim.orchestrator.grid import build_scenarios_df
from staffsim.scheduling.headless import run_headless as run_sched_headless


def test_scenarios_grid_has_two_schedule_rows_per_base() -> None:
    df = build_scenarios_df()
    counts = df.groupby("base_id")["schedule_case"].nunique()
    assert int(counts.min()) == 2
    assert int(counts.max()) == 2


def test_demand_headless_shapes() -> None:
    params = {
        "V": 7500,
        "AHT": 300,
        "OCC": 0.70,
        "SHK": 0.20,
        "Hg": 42,
        "T": 0.5,
        "week_pattern": "W1",
        "p_weekdays": None,
        "weekday_split": None,
        "weekday_step": None,
        "K": 1,
        "ratio_target": 2,
        "pos1": 25,
        "pos2": None,
        "width1": 16,
        "width2": None,
        "width_assignment_rule": None,
        "peak_amplitude_rule": None,
        "peak_amplitude_ratio": None,
    }
    out = run_demand_headless(params, seed=12345)
    assert out.calls_matrix.shape == (7, 48)
    assert out.expected_matrix.shape == (7, 48)
    assert out.fte_matrix.shape == (7, 48)
    assert int(out.calls_matrix.sum()) == 7500


def test_sched_headless_run1_run2_zero_required_fast() -> None:
    required = np.zeros((7, 48), dtype=float)
    r1 = run_sched_headless(solver="cp_sat", required=required, mode="run1", n0=1, time_limit_sec=5, workers=1)
    r2 = run_sched_headless(solver="cp_sat", required=required, mode="run2", n0=1, time_limit_sec=5, workers=1)
    assert r1.solver == "cp_sat"
    assert r2.solver == "cp_sat"
    assert r1.planned_matrix.shape == (7, 48)
    assert r2.planned_matrix.shape == (7, 48)
    assert r1.coverage >= 1.0
    assert r2.coverage >= 1.0


@pytest.mark.skipif(not importlib.util.find_spec("hexaly"), reason="hexaly not installed")
def test_sched_headless_hexaly_run1_zero_required_fast() -> None:
    required = np.zeros((7, 48), dtype=float)
    r = run_sched_headless(solver="hexaly", required=required, mode="run1", n0=1, time_limit_sec=2, workers=1)
    assert r.solver == "hexaly"
    assert r.planned_matrix.shape == (7, 48)
