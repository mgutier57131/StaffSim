"""Grafica la distribucion de M_obs."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def _find_root() -> Path:
    c = Path.cwd()
    for _ in range(6):
        if (c / "Resultados Final").exists(): return c
        c = c.parent
    raise FileNotFoundError

ROOT   = _find_root()
CSV_IN = ROOT / "Resultados Final" / "summary_valor_cp_sat.csv"
PNG_OUT = ROOT / "Resultados Final" / "mobs_distribucion.png"

df = pd.read_csv(CSV_IN)
df = df[df["solver_status"].isin(["FEASIBLE","OPTIMAL"])].copy()
df.rename(columns={"N_final": "HC_gross_sch"}, inplace=True)
df["HC_real"] = df["HC_gross_sch"] / 0.80
df["M_obs"]   = df["HC_real"] / df["HC_gross_ceil"]

m      = df["M_obs"].values
p33    = np.percentile(m, 33)
p66    = np.percentile(m, 66)
media  = m.mean()
mediana= np.median(m)
std    = m.std()

# Valores unicos y frecuencias
vals, cnts = np.unique(m, return_counts=True)
pcts = cnts / len(m) * 100

from matplotlib.patches import Patch

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "Factor de ajuste M observado en 2 268 escenarios simulados\n"
    r"$M = HC_{real}\ /\ HC_{requerido}$",
    fontsize=13, fontweight="bold", y=1.02,
)

# -----------------------------------------------------------
# Panel 1 — Frecuencia por valor de M
# -----------------------------------------------------------
ax = axes[0]

color_map = []
for v in vals:
    if v < p33:    color_map.append("#4CAF50")
    elif v <= p66: color_map.append("#FFC107")
    else:          color_map.append("#F44336")

bars = ax.bar(range(len(vals)), cnts, color=color_map, edgecolor="white", linewidth=1.2, width=0.6)

for bar, cnt, pct in zip(bars, cnts, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 6,
            f"{cnt}\n{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="500")

ax.set_xticks(range(len(vals)))
ax.set_xticklabels([f"{v:.4f}" for v in vals], rotation=30, ha="right", fontsize=9)
ax.set_xlabel("Valor de M", fontsize=11, labelpad=8)
ax.set_ylabel("Número de escenarios", fontsize=11)
ax.set_title("¿Con qué frecuencia aparece cada valor de M?", fontsize=11, pad=10)
ax.set_ylim(0, max(cnts) * 1.28)
ax.grid(True, axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(handles=[
    Patch(color="#4CAF50", label=f"Complejidad baja   (M < {p33:.4f})"),
    Patch(color="#FFC107", label=f"Complejidad media  ({p33:.4f} – {p66:.4f})"),
    Patch(color="#F44336", label=f"Complejidad alta   (M > {p66:.4f})"),
], fontsize=9, framealpha=0.9, loc="upper right")

# -----------------------------------------------------------
# Panel 2 — ¿Es normal? Histograma vs curva normal
# -----------------------------------------------------------
ax = axes[1]

x_range      = np.linspace(m.min() - 0.05, m.max() + 0.08, 300)
normal_curve = stats.norm.pdf(x_range, media, std)
stat_sw, p_sw = stats.shapiro(np.random.choice(m, 500, replace=False))

ax.hist(m, bins=28, density=True, color="#90CAF9", edgecolor="white",
        linewidth=0.6, alpha=0.85, label="Distribución real de M")
ax.plot(x_range, normal_curve, color="#E53935", linewidth=2.5,
        linestyle="--", label="Curva normal teórica")

ax.axvline(p33, color="#4CAF50", linewidth=1.8, linestyle=":", label=f"p33 = {p33:.4f}  →  límite Baja/Media")
ax.axvline(p66, color="#F44336", linewidth=1.8, linestyle=":", label=f"p66 = {p66:.4f}  →  límite Media/Alta")

ax.set_xlabel("Valor de M", fontsize=11, labelpad=8)
ax.set_ylabel("Densidad", fontsize=11)
ax.set_title("¿Sigue M una distribución normal?", fontsize=11, pad=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, axis="y", alpha=0.25)

# Caja con conclusión
conclusion = (
    f"Asimetría = {pd.Series(m).skew():.2f}  (normal = 0)\n"
    f"Shapiro-Wilk  p = {p_sw:.1e}\n"
    f"→ M no es normal: usar percentiles"
)
ax.text(0.97, 0.97, conclusion,
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", edgecolor="#F9A825", alpha=0.95))

fig.tight_layout()
fig.savefig(PNG_OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Grafica exportada: {PNG_OUT}")
