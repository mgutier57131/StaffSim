import csv

import numpy as np

from staffsim.io.export import export_all
from staffsim.workload.baseline import compute_baseline_summary


def test_export_includes_hc_gross_metrics(tmp_path) -> None:
    calls = np.ones((7, 48), dtype=int)
    expected = np.ones((7, 48), dtype=float)
    fte = np.ones((7, 48), dtype=float)
    summary = compute_baseline_summary(v_week=6000, aht=300.0, occ=0.7, shk=0.2, hg=42.0, t_interval=0.5)

    export_all(
        output_dir=tmp_path,
        calls_matrix=calls,
        calls_expected_matrix=expected,
        fte_matrix=fte,
        params={"V": 6000},
        summary=summary,
        params_text="StaffSim GUI Parameters\n",
    )

    with (tmp_path / "summary.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    metrics = {row[0]: row[1] for row in rows[1:]}
    assert "H_prod_noSHK" in metrics
    assert "HC_gross" in metrics
    assert "HC_gross_ceil" in metrics
    assert "HC_gross_round" in metrics

    params_txt = (tmp_path / "params.txt").read_text(encoding="utf-8")
    assert "H_prod_noSHK" in params_txt
    assert "HC_gross" in params_txt
    assert "HC_gross_ceil" in params_txt
    assert "HC_gross_round" in params_txt

