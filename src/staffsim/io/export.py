"""CSV export helpers for calls/FTE matrices and summary."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from staffsim.workload.baseline import BaselineSummary

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _matrix_header() -> list[str]:
    return ["day"] + [f"t{idx:02d}" for idx in range(48)]


def export_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_matrix_header())
        for day_idx, row in enumerate(matrix):
            formatted_row: list[int | float] = []
            for value in row.tolist():
                if isinstance(value, (float, np.floating)):
                    formatted_row.append(round(float(value), 2))
                else:
                    formatted_row.append(int(value))
            writer.writerow([DAY_LABELS[day_idx], *formatted_row])


def export_summary_csv(
    path: Path,
    params: dict[str, str | int | float],
    summary: BaselineSummary,
    extra_metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h_prod_noshk = summary.h_prod
    hc_gross = h_prod_noshk / summary.hg
    rows: list[tuple[str, str | int | float]] = []
    rows.extend(params.items())
    rows.extend(
        [
            ("H_talk", summary.h_talk),
            ("H_prod", summary.h_prod),
            ("H_paid", summary.h_paid),
            ("HC_teorico", summary.hc_teorico),
            ("HC_teorico_ceil", math.ceil(summary.hc_teorico)),
            ("HC_teorico_round", round(summary.hc_teorico, 3)),
            ("H_prod_noSHK", h_prod_noshk),
            ("HC_gross", hc_gross),
            ("HC_gross_ceil", math.ceil(hc_gross)),
            ("HC_gross_round", round(hc_gross, 3)),
        ]
    )
    if extra_metrics:
        rows.extend(extra_metrics.items())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        formatted_rows: list[tuple[str, str | int | float]] = []
        for key, value in rows:
            if isinstance(value, (float, np.floating)):
                formatted_rows.append((key, round(float(value), 2)))
            else:
                formatted_rows.append((key, value))
        writer.writerows(formatted_rows)


def export_params_txt(path: Path, params_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(params_text, encoding="utf-8")


def export_all(
    output_dir: str | Path,
    calls_matrix: np.ndarray,
    calls_expected_matrix: np.ndarray | None,
    fte_matrix: np.ndarray,
    params: dict[str, str | int | float],
    summary: BaselineSummary,
    params_text: str,
    extra_metrics: dict[str, float] | None = None,
) -> None:
    out = Path(output_dir)
    h_prod_noshk = summary.h_prod
    hc_gross = h_prod_noshk / summary.hg
    params_text_with_gross = (
        params_text
        + f"H_prod_noSHK: {round(h_prod_noshk, 3)}\n"
        + f"HC_gross: {round(hc_gross, 3)}\n"
        + f"HC_gross_ceil: {math.ceil(hc_gross)}\n"
        + f"HC_gross_round: {round(hc_gross, 3)}\n"
    )
    if calls_expected_matrix is not None:
        export_matrix_csv(out / "expected_curve.csv", calls_expected_matrix)
    export_matrix_csv(out / "calls_matrix.csv", calls_matrix)
    export_matrix_csv(out / "fte_matrix.csv", fte_matrix)
    export_summary_csv(out / "summary.csv", params=params, summary=summary, extra_metrics=extra_metrics)
    export_params_txt(out / "params.txt", params_text=params_text_with_gross)


def export_results(
    *,
    calls_matrix: np.ndarray,
    calls_expected_matrix: np.ndarray | None,
    fte_matrix: np.ndarray,
    params: dict[str, str | int | float],
    summary: BaselineSummary,
    params_text: str,
    extra_metrics: dict[str, float] | None = None,
    figure=None,
    base_dir: str | Path = "results",
) -> Path:
    run_folder = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(base_dir) / run_folder
    export_all(
        output_dir=out,
        calls_matrix=calls_matrix,
        calls_expected_matrix=calls_expected_matrix,
        fte_matrix=fte_matrix,
        params=params,
        summary=summary,
        params_text=params_text,
        extra_metrics=extra_metrics,
    )
    if figure is not None:
        out.mkdir(parents=True, exist_ok=True)
        figure.savefig(out / "curve.png", dpi=150)
    h_prod_noshk = summary.h_prod
    hc_gross = h_prod_noshk / summary.hg
    print(f"HC_gross: {hc_gross:.3f}")
    print(f"HC_gross_ceil: {math.ceil(hc_gross)}")
    return out
