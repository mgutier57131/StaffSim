"""Plot helpers for scheduling outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_required_vs_planned(required: np.ndarray, planned: np.ndarray, out_path: Path) -> None:
    x = np.arange(336)
    req = required.reshape(-1)
    pln = planned.reshape(-1)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(x, req, label="Required FTE", linewidth=1.8)
    ax.plot(x, pln, label="Planned FTE", linewidth=1.8)
    for d in range(1, 7):
        ax.axvline(48 * d, color="gray", linewidth=0.7, alpha=0.3)
    ax.set_title("Required vs Planned Coverage (Weekly)")
    ax.set_xlabel("Time (30-min intervals across week)")
    ax.set_ylabel("FTE")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

