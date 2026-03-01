"""Plot helpers for scheduling outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from staffsim.io.export import DAY_LABELS


def plot_required_vs_planned(required: np.ndarray, planned: np.ndarray, out_path: Path) -> None:
    x = np.arange(336)
    req = required.reshape(-1)
    pln = planned.reshape(-1)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(x, req, label="Required FTE", linewidth=1.5)
    ax.plot(x, pln, label="Planned FTE", linewidth=1.5)
    for d in range(1, 7):
        ax.axvline(48 * d, color="gray", linewidth=0.8, alpha=0.35)

    tick_positions = np.arange(0, 336, 12)
    tick_labels: list[str] = []
    for t in tick_positions:
        day = t // 48
        half_hours = t % 48
        hour = half_hours // 2
        minute = "30" if (half_hours % 2) else "00"
        tick_labels.append(f"{DAY_LABELS[int(day)]} {int(hour):02d}:{minute}")

    ax.set_title("Required vs Planned Coverage (Weekly)")
    ax.set_xlabel("Time (30-min intervals across week)")
    ax.set_ylabel("FTE")
    ax.set_ylim(bottom=0)
    ax.set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
    ax.grid(True, which="major", axis="both", alpha=0.25, linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
