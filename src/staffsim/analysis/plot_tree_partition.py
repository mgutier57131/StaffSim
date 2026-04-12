"""
Partición del espacio de decisión — árbol M_obs
================================================
Muestra cómo el árbol divide el espacio (hora del pico × variación pico/valle)
para cada estrategia de turno (run1 / run2).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()


def _find_root() -> Path:
    candidate = _HERE
    for _ in range(8):
        if (candidate / "Resultados Final").exists():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError("No se encontró 'Resultados Final/'")


ROOT   = _find_root()
CSV_IN = ROOT / "Resultados Final" / "summary_valor_cp_sat.csv"
OUT    = ROOT / "Resultados Final" / "decision_tree_partition.png"

SHK  = 0.20
SEED = 42

FEATURE_COLS = [
    "week_pattern", "p_weekdays", "weekday_step", "K",
    "pos1", "pos2", "width1", "width2",
    "peak_amplitude_rule", "ratio_target", "schedule_case",
]
CAT_COLS = ["week_pattern", "peak_amplitude_rule", "schedule_case"]

COLORES = {"baja": "#66bb6a", "media": "#ffa726", "alta": "#ef5350"}
LABELS  = {"baja": "Baja",    "media": "Media",    "alta": "Alta"}


def intervalo_a_hora(n: float) -> str:
    h, m = divmod(int(round(n)) * 30, 60)
    return f"{h:02d}:{'30' if m else '00'}"


def clasificar_m(m: float, p33: float, p66: float) -> str:
    if m < p33:
        return "baja"
    elif m <= p66:
        return "media"
    return "alta"


def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Cargar y entrenar
    # -----------------------------------------------------------------------
    df = pd.read_csv(CSV_IN)
    df = df[df["solver_status"].isin(["FEASIBLE", "OPTIMAL"])].copy()
    df.rename(columns={"N_final": "HC_gross_sch"}, inplace=True)
    df["HC_real"] = df["HC_gross_sch"] / (1 - SHK)
    df["M_obs"]   = df["HC_real"] / df["HC_teorico"]

    X = df[FEATURE_COLS].copy()
    num_cols = [c for c in FEATURE_COLS if c not in CAT_COLS]
    X[num_cols] = X[num_cols].fillna(-1.0)

    encoders: dict[str, LabelEncoder] = {}
    for col in CAT_COLS:
        X[col] = X[col].fillna("N/A").astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    tree = DecisionTreeRegressor(max_depth=6, random_state=SEED)
    tree.fit(X, df["M_obs"])

    p33 = float(np.percentile(df["M_obs"].values, 33))
    p66 = float(np.percentile(df["M_obs"].values, 66))

    df["complejidad"] = df["M_obs"].apply(lambda m: clasificar_m(m, p33, p66))

    # -----------------------------------------------------------------------
    # 2. Ejes: pos1 × ratio_target
    # -----------------------------------------------------------------------
    pos1_vals  = [14, 25, 36]   # K=1: 07:00, 12:30, 18:00
    ratio_vals = [2, 4, 6]

    x_min, x_max = 10.0, 40.0
    y_min, y_max =  0.8,  7.2

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500),
    )

    cmap = plt.matplotlib.colors.ListedColormap(["#c8e6c9", "#ffe0b2", "#ffcdd2"])
    nivel_num = {"baja": 0, "media": 1, "alta": 2}

    # -----------------------------------------------------------------------
    # 3. Figura: 2 subplots (run1 | run2)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, schedule_case in zip(axes, ["run1", "run2"]):
        sc_enc = int(encoders["schedule_case"].transform([schedule_case])[0])
        wp_enc = int(encoders["week_pattern"].transform(["W1"])[0])
        par_enc = int(encoders["peak_amplitude_rule"].transform(["N/A"])[0])

        # Malla de predicciones con K=1, W1, p_weekdays=-1, run1/run2
        fixed = {
            "week_pattern"       : wp_enc,
            "p_weekdays"         : -1.0,   # W1: no se usa
            "weekday_step"       : 0.02,
            "K"                  : 1.0,
            "pos2"               : -1.0,
            "width1"             : 20.0,
            "width2"             : -1.0,
            "peak_amplitude_rule": par_enc,
            "schedule_case"      : sc_enc,
        }

        rows = []
        for xi, yi in zip(xx.ravel(), yy.ravel()):
            row = {**fixed, "pos1": xi, "ratio_target": yi}
            rows.append([row[c] for c in FEATURE_COLS])

        grid_df = pd.DataFrame(rows, columns=FEATURE_COLS)
        m_grid  = tree.predict(grid_df)
        comp_g  = np.array([clasificar_m(m, p33, p66) for m in m_grid])
        Z = np.array([nivel_num[c] for c in comp_g]).reshape(xx.shape)

        # Regiones de color
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                    cmap=cmap, alpha=0.50)
        ax.contour(xx, yy, Z, levels=[0.5, 1.5],
                   colors="white", linewidths=1.6, linestyles="--")

        # Etiquetas M en cada región
        for nivel, num in nivel_num.items():
            mask = comp_g == nivel
            if not mask.any():
                continue
            cx = float(xx.ravel()[mask].mean())
            cy = float(yy.ravel()[mask].mean())
            m_mean = float(m_grid[mask].mean())
            ax.text(cx, cy, f"M ≈ {m_mean:.3f}",
                    ha="center", va="center", fontsize=10.5, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=COLORES[nivel],
                              edgecolor="none", alpha=0.80))

        # Puntos reales (solo K=1 de ese schedule_case)
        sub = df[(df["K"] == 1) & (df["schedule_case"] == schedule_case)].copy()
        rng = np.random.default_rng(99)
        jx  = rng.uniform(-0.5, 0.5, size=len(sub))
        jy  = rng.uniform(-0.12, 0.12, size=len(sub))
        for nivel in ["baja", "media", "alta"]:
            mask = sub["complejidad"] == nivel
            ax.scatter(
                sub.loc[mask, "pos1"].values + jx[mask.values],
                sub.loc[mask, "ratio_target"].values + jy[mask.values],
                c=COLORES[nivel], s=30, edgecolors="white",
                linewidths=0.4, alpha=0.88, zorder=3,
            )

        # Formato del eje
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(pos1_vals)
        ax.set_xticklabels([intervalo_a_hora(v) for v in pos1_vals], fontsize=10)
        ax.set_yticks(ratio_vals)
        ax.set_yticklabels([f"{v}:1" for v in ratio_vals], fontsize=10)
        ax.set_xlabel("Hora del pico principal", fontsize=11)
        ax.set_title(
            f"Turnos {'fijos 7 h' if schedule_case == 'run1' else 'variables 6–10 h'}"
            f"  ({schedule_case})",
            fontsize=11, fontweight="bold", pad=8,
        )
        ax.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.35)

        n_pts = len(sub)
        ax.text(0.97, 0.03, f"n = {n_pts}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#666666")

    axes[0].set_ylabel("Variación pico / valle", fontsize=11)

    # -----------------------------------------------------------------------
    # 4. Leyenda y título general
    # -----------------------------------------------------------------------
    parches = [
        mpatches.Patch(facecolor=COLORES[n], edgecolor="#999", label=f"Complejidad {LABELS[n]}")
        for n in ["baja", "media", "alta"]
    ]
    fig.legend(handles=parches, loc="upper center", ncol=3,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(
        "Partición del espacio de decisión — Árbol M$_{obs}$ (profundidad 6)",
        fontsize=13, fontweight="bold", y=1.06,
    )

    fig.text(
        0.5, -0.03,
        "Escenarios con un solo pico (K=1), patrón semanal uniforme (W1). "
        "Las regiones muestran la complejidad predicha variando hora del pico y nivel de variación intradiaria.",
        ha="center", fontsize=8.5, color="#555555",
    )

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Exportado: {OUT}")
    plt.show()


if __name__ == "__main__":
    main()
