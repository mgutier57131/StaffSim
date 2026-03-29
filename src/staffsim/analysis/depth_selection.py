"""
Seleccion de profundidad optima para el arbol de decision
=========================================================
Para cada profundidad de 1 a MAX_DEPTH calcula:
  - R² train
  - R² validacion (20% hold-out)
  - Numero de hojas
  - Importancia de cada variable
  - Cuantas variables aportan (importancia > 0) y cuantas no

Exporta:
  - depth_selection_metrics.csv   tabla resumen
  - depth_selection_importance.csv tabla importancia por profundidad
  - depth_selection.png           grafica con 4 paneles
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_CWD = Path.cwd()

def _find_root() -> Path:
    candidate = _CWD
    for _ in range(6):
        if (candidate / "Resultados Final").exists():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError("No se encontro 'Resultados Final/' desde CWD.")

ROOT    = _find_root()
CSV_IN  = ROOT / "Resultados Final" / "summary_valor_cp_sat.csv"
OUT_DIR = ROOT / "Resultados Final"

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SHK       = 0.20
MAX_DEPTH = 11
TEST_SIZE = 0.20
SEED      = 42

FEATURE_COLS = [
    "week_pattern",
    "p_weekdays",
    "weekday_step",
    "K",
    "pos1",
    "pos2",
    "width1",
    "width2",
    "peak_amplitude_rule",
    "ratio_target",
    "schedule_case",
]
CAT_COLS     = ["week_pattern", "peak_amplitude_rule", "schedule_case"]
NAN_SENTINEL = -1.0

# Etiquetas legibles para la grafica
FEAT_LABELS = {
    "week_pattern"        : "modo_semanal",
    "p_weekdays"          : "p",
    "weekday_step"        : "gradiente",
    "K"                   : "K",
    "pos1"                : "pos1",
    "pos2"                : "pos2",
    "width1"              : "width1",
    "width2"              : "width2",
    "peak_amplitude_rule" : "modo_altura",
    "ratio_target"        : "R",
    "schedule_case"       : "run",
}


# ---------------------------------------------------------------------------
# Carga y preparacion
# ---------------------------------------------------------------------------
def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(CSV_IN)
    df = df[df["solver_status"].isin(["FEASIBLE", "OPTIMAL"])].copy()

    df.rename(columns={"N_final": "HC_gross_sch"}, inplace=True)
    df["HC_real"] = df["HC_gross_sch"] / (1 - SHK)
    df["M_obs"]   = df["HC_real"] / df["HC_gross_ceil"]

    X = df[FEATURE_COLS].copy()
    num_cols = [c for c in FEATURE_COLS if c not in CAT_COLS]
    X[num_cols] = X[num_cols].fillna(NAN_SENTINEL)

    encoders: dict[str, LabelEncoder] = {}
    for col in CAT_COLS:
        X[col] = X[col].fillna("N/A").astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    y = df["M_obs"]
    return df, X, y


# ---------------------------------------------------------------------------
# Analisis por profundidad
# ---------------------------------------------------------------------------
def run_depth_analysis(X: pd.DataFrame,
                       y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    print(f"Train: {len(X_train)} escenarios | Validacion: {len(X_val)} escenarios")
    print(f"\n{'Depth':>5}  {'R2_train':>9}  {'R2_val':>9}  {'Hojas':>6}  "
          f"{'Vars_aportan':>12}  {'Vars_no_aportan':>15}")
    print("-" * 65)

    metrics_rows    = []
    importance_rows = []

    for depth in range(1, MAX_DEPTH + 1):
        tree = DecisionTreeRegressor(max_depth=depth, random_state=SEED)
        tree.fit(X_train, y_train)

        r2_train = tree.score(X_train, y_train)
        r2_val   = tree.score(X_val,   y_val)
        n_leaves = tree.get_n_leaves()

        imp = pd.Series(tree.feature_importances_, index=FEATURE_COLS)
        n_aportan   = (imp > 0).sum()
        n_no_aportan = (imp == 0).sum()

        print(f"{depth:>5}  {r2_train:>9.4f}  {r2_val:>9.4f}  {n_leaves:>6}  "
              f"{n_aportan:>12}  {n_no_aportan:>15}")

        metrics_rows.append({
            "depth"          : depth,
            "r2_train"       : round(r2_train, 4),
            "r2_val"         : round(r2_val, 4),
            "n_leaves"       : n_leaves,
            "vars_aportan"   : int(n_aportan),
            "vars_no_aportan": int(n_no_aportan),
        })

        for feat in FEATURE_COLS:
            importance_rows.append({
                "depth"      : depth,
                "variable"   : feat,
                "label"      : FEAT_LABELS[feat],
                "importancia": round(float(imp[feat]), 4),
            })

    metrics_df    = pd.DataFrame(metrics_rows)
    importance_df = pd.DataFrame(importance_rows)
    return metrics_df, importance_df


# ---------------------------------------------------------------------------
# Grafica
# ---------------------------------------------------------------------------
def plot_results(metrics_df: pd.DataFrame,
                 importance_df: pd.DataFrame) -> None:

    depths = metrics_df["depth"].values
    best_depth = metrics_df.loc[metrics_df["r2_val"].idxmax(), "depth"]

    OPT_COLOR  = "#2E7D32"
    FONT_BASE  = 11

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Seleccion de la profundidad optima del arbol de decision",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # Etiquetas de subpanel
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, lbl in zip(axes.flat, panel_labels):
        ax.text(-0.08, 1.04, lbl, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    # --- (a) R² train vs validacion ---
    ax = axes[0, 0]
    ax.plot(depths, metrics_df["r2_train"], "o-", color="#2196F3",
            linewidth=2, markersize=5, label="R² entrenamiento")
    ax.plot(depths, metrics_df["r2_val"],   "s-", color="#FF5722",
            linewidth=2, markersize=5, label="R² validacion")
    ax.axvline(best_depth, color=OPT_COLOR, linestyle="--", linewidth=1.5,
               label=f"Optimo: depth = {best_depth}")
    ax.set_xlabel("Profundidad", fontsize=FONT_BASE)
    ax.set_ylabel("R²", fontsize=FONT_BASE)
    ax.set_title("Capacidad predictiva", fontsize=FONT_BASE, pad=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(depths)
    ax.spines[["top", "right"]].set_visible(False)

    # --- (b) Numero de hojas ---
    ax = axes[0, 1]
    bars = ax.bar(depths, metrics_df["n_leaves"], color="#9C27B0", alpha=0.75, width=0.6)
    ax.axvline(best_depth, color=OPT_COLOR, linestyle="--", linewidth=1.5,
               label=f"Optimo: depth = {best_depth}")
    for bar, val in zip(bars, metrics_df["n_leaves"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                str(val), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Profundidad", fontsize=FONT_BASE)
    ax.set_ylabel("Numero de hojas", fontsize=FONT_BASE)
    ax.set_title("Complejidad del arbol", fontsize=FONT_BASE, pad=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_xticks(depths)
    ax.spines[["top", "right"]].set_visible(False)

    # --- (c) Variables activas ---
    ax = axes[1, 0]
    ax.plot(depths, metrics_df["vars_aportan"],    "o-", color="#4CAF50",
            linewidth=2, markersize=5, label="Con importancia > 0")
    ax.plot(depths, metrics_df["vars_no_aportan"], "s-", color="#F44336",
            linewidth=2, markersize=5, label="Sin importancia")
    ax.axvline(best_depth, color=OPT_COLOR, linestyle="--", linewidth=1.5,
               label=f"Optimo: depth = {best_depth}")
    ax.set_xlabel("Profundidad", fontsize=FONT_BASE)
    ax.set_ylabel("Variables", fontsize=FONT_BASE)
    ax.set_title("Variables con aporte al modelo", fontsize=FONT_BASE, pad=8)
    ax.set_yticks(range(0, len(FEATURE_COLS) + 1))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(depths)
    ax.spines[["top", "right"]].set_visible(False)

    # --- (d) Heatmap importancia ---
    ax = axes[1, 1]
    labels = [FEAT_LABELS[f] for f in FEATURE_COLS]
    imp_matrix = importance_df.pivot(
        index="label", columns="depth", values="importancia"
    ).reindex(labels)

    im = ax.imshow(imp_matrix.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=imp_matrix.values.max())
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Profundidad", fontsize=FONT_BASE)
    ax.set_title("Importancia de cada variable", fontsize=FONT_BASE, pad=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(labels)):
        for j in range(len(depths)):
            val = imp_matrix.values[i, j]
            if val > 0:
                txt = f"{val:.2f}" if val >= 0.01 else f"{val:.3f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7,
                        color="white" if val > 0.35 else "black")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7, color="#CCCCCC")

    fig.tight_layout()
    out_png = OUT_DIR / "depth_selection.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGrafica exportada: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Seleccion de profundidad optima ===\n")
    _, X, y = load_and_prepare()

    metrics_df, importance_df = run_depth_analysis(X, y)

    # Exportar CSVs
    metrics_df.to_csv(OUT_DIR / "depth_selection_metrics.csv", index=False)
    importance_df.to_csv(OUT_DIR / "depth_selection_importance.csv", index=False)

    best = metrics_df.loc[metrics_df["r2_val"].idxmax()]
    print(f"\n>>> Profundidad optima: {int(best['depth'])}  "
          f"(R² val={best['r2_val']:.4f}, "
          f"hojas={int(best['n_leaves'])}, "
          f"vars activas={int(best['vars_aportan'])})")

    plot_results(metrics_df, importance_df)
    print("\nListo.")


if __name__ == "__main__":
    main()
