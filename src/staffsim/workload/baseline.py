"""Deterministic workload and staffing formulas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaselineSummary:
    v_week: int
    aht: float
    occ: float
    shk: float
    hg: float
    t_interval: float
    h_talk: float
    h_prod: float
    h_paid: float
    hc_teorico: float


def compute_baseline_summary(
    v_week: int,
    aht: float,
    occ: float,
    shk: float,
    hg: float,
    t_interval: float = 0.5,
) -> BaselineSummary:
    """Compute aggregate workload metrics from mandatory formulas."""
    h_talk = (v_week * aht) / 3600.0
    h_prod = h_talk / occ
    h_paid = h_prod / (1.0 - shk)
    hc_teorico = h_paid / hg
    return BaselineSummary(
        v_week=v_week,
        aht=aht,
        occ=occ,
        shk=shk,
        hg=hg,
        t_interval=t_interval,
        h_talk=h_talk,
        h_prod=h_prod,
        h_paid=h_paid,
        hc_teorico=hc_teorico,
    )


def calls_to_fte_matrix(
    calls_matrix: np.ndarray,
    aht: float,
    occ: float,
    t_interval: float = 0.5,
) -> np.ndarray:
    """
    Convert calls per interval to FTE per interval:
    FTE_i = (v_i * AHT) / (3600 * T * OCC)
    """
    return (calls_matrix * aht) / (3600.0 * t_interval * occ)

