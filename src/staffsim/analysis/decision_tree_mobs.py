"""
Decision Tree Analysis for M_obs recalculado
=============================================
Flujo de calculo:
  HC_gross_sch = N_final  (agentes que devuelve el scheduler, sin SHK)
  HC_real      = HC_gross_sch / (1 - SHK)   — agentes reales con ausentismo
  HC_req       = HC_gross_ceil               — headcount bruto teorico requerido
  M_obs        = HC_real / HC_req            — factor multiplicador observado

El arbol predice M_obs a partir de los inputs puros del grid.
El usuario luego estima: HC_real ≈ HC_teorico × M_predicho

Variables de entrada (inputs del grid)
---------------------------------------
modo_semanal  -> week_pattern        (cat)
p             -> p_weekdays          (num)
gradiente     -> weekday_step        (num)
K             -> K                   (num)
pos1          -> pos1                (num)
pos2          -> pos2                (num, NaN si K=1)
width1        -> width1              (num)
width2        -> width2              (num, NaN si K=1)
modo_altura   -> peak_amplitude_rule (cat, NaN si K=1)
R             -> ratio_target        (num)
run           -> schedule_case       (cat)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree

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
    raise FileNotFoundError(
        "No se encontro 'Resultados Final/' subiendo desde CWD. "
        "Ejecuta desde la raiz del repo (Simulador/)."
    )

ROOT    = _find_root()
CSV_IN  = ROOT / "Resultados Final" / "summary_valor_cp_sat.csv"
OUT_DIR = ROOT / "Resultados Final"
CSV_OUT = OUT_DIR / "decision_tree_lookup.csv"
PNG_OUT = OUT_DIR / "decision_tree_mobs.png"

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SHK = 0.20   # Shrinkage fijo de todos los escenarios

FEATURE_COLS = [
    "week_pattern",        # modo_semanal  (cat)
    "p_weekdays",          # p
    "weekday_step",        # gradiente
    "K",                   # K
    "pos1",                # pos1
    "pos2",                # pos2 (NaN si K=1)
    "width1",              # width1
    "width2",              # width2 (NaN si K=1)
    "peak_amplitude_rule", # modo_altura   (cat, NaN si K=1)
    "ratio_target",        # R
    "schedule_case",       # run           (cat)
]

CAT_COLS      = ["week_pattern", "peak_amplitude_rule", "schedule_case"]
NAN_SENTINEL  = -1.0


# ---------------------------------------------------------------------------
# 1. Carga, limpieza y calculo de M_obs
# ---------------------------------------------------------------------------
def load_and_compute(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_total = len(df)

    # Excluir solver fallido (INFEASIBLE / UNKNOWN)
    valid_mask = df["solver_status"].isin(["FEASIBLE", "OPTIMAL"])
    n_excl = (~valid_mask).sum()
    df = df[valid_mask].copy()

    # --- Renombrar y recalcular ---
    # N_final es lo que devuelve el scheduler sin SHK → HC_gross_sch
    df.rename(columns={"N_final": "HC_gross_sch"}, inplace=True)

    # HC_real: agentes reales a contratar (con SHK aplicado)
    df["HC_real"] = df["HC_gross_sch"] / (1 - SHK)

    # M_obs: agentes reales (scheduler+SHK) sobre headcount teorico (workload+SHK)
    # Ambos en la misma base: agentes CON SHK
    df["M_obs"] = df["HC_real"] / df["HC_teorico"]

    print(f"[1] Escenarios cargados      : {n_total}")
    print(f"    Excluidos (solver failed) : {n_excl}")
    print(f"    Para analisis             : {len(df)}")
    print(f"    solver_status             : {df['solver_status'].value_counts().to_dict()}")
    print(f"\n    HC_gross_sch  — min:{df['HC_gross_sch'].min():.1f}  "
          f"med:{df['HC_gross_sch'].median():.1f}  max:{df['HC_gross_sch'].max():.1f}")
    print(f"    HC_real       — min:{df['HC_real'].min():.2f}  "
          f"med:{df['HC_real'].median():.2f}  max:{df['HC_real'].max():.2f}")
    print(f"    HC_req (ceil) — min:{df['HC_gross_ceil'].min():.1f}  "
          f"med:{df['HC_gross_ceil'].median():.1f}  max:{df['HC_gross_ceil'].max():.1f}")
    print(f"    M_obs         — min:{df['M_obs'].min():.4f}  "
          f"med:{df['M_obs'].median():.4f}  max:{df['M_obs'].max():.4f}")

    return df


# ---------------------------------------------------------------------------
# 2. Preparar features con encoding
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    X = df[FEATURE_COLS].copy()

    # NaN numericos → centinela
    num_cols = [c for c in FEATURE_COLS if c not in CAT_COLS]
    X[num_cols] = X[num_cols].fillna(NAN_SENTINEL)

    # NaN categoricos → "N/A"
    encoders: dict[str, LabelEncoder] = {}
    for col in CAT_COLS:
        X[col] = X[col].fillna("N/A").astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    return X, encoders


# ---------------------------------------------------------------------------
# 3. Entrenar arbol
# ---------------------------------------------------------------------------
def train_tree(X: pd.DataFrame, y: pd.Series) -> DecisionTreeRegressor:
    tree = DecisionTreeRegressor(max_depth=6, random_state=42)
    tree.fit(X, y)

    print(f"\n[2] Arbol entrenado — hojas: {tree.get_n_leaves()}, "
          f"R2 train: {tree.score(X, y):.4f}")

    imp = pd.Series(tree.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n    Importancia de variables (M_obs ~ inputs del grid):")
    for feat, val in imp.items():
        bar = "#" * int(val * 40)
        print(f"    {feat:<25} {val:.4f}  {bar}")

    return tree


# ---------------------------------------------------------------------------
# 4. Extraer condiciones de cada hoja (DFS)
# ---------------------------------------------------------------------------
def _leaf_conditions(tree: DecisionTreeRegressor,
                     encoders: dict[str, LabelEncoder]) -> dict[int, str]:
    t   = tree.tree_
    cl  = t.children_left
    cr  = t.children_right
    feat = t.feature
    thr  = t.threshold

    result: dict[int, str] = {}
    stack = [(0, [])]

    while stack:
        node, conds = stack.pop()

        if cl[node] == cr[node]:          # hoja
            result[node] = " AND ".join(conds) if conds else "ROOT"
            continue

        fname = FEATURE_COLS[feat[node]]
        th    = thr[node]

        if fname in encoders:
            le = encoders[fname]
            left_lbl  = [le.classes_[i] for i in range(len(le.classes_)) if i <= th]
            right_lbl = [le.classes_[i] for i in range(len(le.classes_)) if i >  th]
            lc = f"{fname} in {{{','.join(left_lbl)}}}"
            rc = f"{fname} in {{{','.join(right_lbl)}}}"
        else:
            lc = f"{fname} <= {th:.4g}"
            rc = f"{fname} > {th:.4g}"

        stack.append((cl[node], conds + [lc]))
        stack.append((cr[node], conds + [rc]))

    return result


# ---------------------------------------------------------------------------
# 5. Tabla de consulta por hoja
# ---------------------------------------------------------------------------
def build_lookup(df: pd.DataFrame,
                 X: pd.DataFrame,
                 tree: DecisionTreeRegressor,
                 encoders: dict[str, LabelEncoder]) -> pd.DataFrame:

    mobs_all = df["M_obs"].values
    over_all = df["sum_over"].values

    leaf_ids = tree.apply(X)
    df_work  = df[["M_obs", "HC_real", "HC_gross_sch", "sum_over"]].copy()
    df_work["leaf_id"] = leaf_ids

    # Percentiles globales de M_obs → clasificacion de complejidad
    p33_global = np.percentile(mobs_all, 33)
    p66_global = np.percentile(mobs_all, 66)
    print(f"\n[4] Percentiles globales M_obs: p33={p33_global:.4f}, p66={p66_global:.4f}")

    cond_map = _leaf_conditions(tree, encoders)

    rows = []
    for leaf_id, grp in df_work.groupby("leaf_id"):
        m_vals   = grp["M_obs"].values
        hcr_vals = grp["HC_real"].values
        n        = len(grp)

        # M recomendado = mediana de M_obs en la hoja
        m_rec  = float(np.median(m_vals))
        hcr_rec = float(np.median(hcr_vals))

        # Complejidad segun mediana vs percentiles globales
        if m_rec < p33_global:
            complejidad = "baja"
        elif m_rec <= p66_global:
            complejidad = "media"
        else:
            complejidad = "alta"

        rows.append({
            "leaf_id"          : int(leaf_id),
            "condiciones"      : cond_map.get(int(leaf_id), ""),
            "n_escenarios"     : n,
            "M_obs_min"        : round(float(m_vals.min()), 4),
            "M_obs_median"     : round(float(np.median(m_vals)), 4),
            "M_obs_max"        : round(float(m_vals.max()), 4),
            "HC_real_median"   : round(float(np.median(hcr_vals)), 2),
            "M_recomendado"    : round(m_rec, 4),
            "HC_real_rec"      : round(hcr_rec, 2),
            "complejidad"      : complejidad,
        })

    lookup = pd.DataFrame(rows).sort_values("leaf_id").reset_index(drop=True)

    print(f"\n[3+4] Hojas procesadas: {len(lookup)}")
    print(lookup[["leaf_id", "n_escenarios", "M_obs_median",
                  "M_recomendado", "complejidad"]].to_string(index=False))
    return lookup


# ---------------------------------------------------------------------------
# 6. Exportar PNG
# ---------------------------------------------------------------------------
def export_png(tree: DecisionTreeRegressor) -> None:
    # --- Arbol completo a alta resolucion ---
    fig, ax = plt.subplots(figsize=(80, 30))
    plot_tree(
        tree,
        feature_names=FEATURE_COLS,
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
        impurity=False,
        precision=3,
        proportion=False,
    )
    ax.set_title("Decision Tree — M_obs = HC_real / HC_req  (depth=6)",
                 fontsize=18, pad=16)
    fig.savefig(PNG_OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[6a] Arbol completo exportado: {PNG_OUT}")

    # --- Sub-arbol izquierdo (rama p_weekdays <= umbral) ---
    _export_subtree(tree, max_depth=3, title="Sub-arbol izquierdo (primeras 3 capas)",
                    out=OUT_DIR / "decision_tree_left.png")

    # --- Niveles 1-3 del arbol completo como resumen ejecutivo ---
    tree_shallow = DecisionTreeRegressor(max_depth=3, random_state=42)
    # Re-fit no disponible desde el arbol ya entrenado, usamos el arbol original
    # y truncamos visualmente con max_depth en plot_tree
    fig, ax = plt.subplots(figsize=(28, 12))
    plot_tree(
        tree,
        feature_names=FEATURE_COLS,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
        impurity=False,
        precision=3,
        max_depth=3,          # muestra solo los 3 primeros niveles
    )
    ax.set_title("Decision Tree — Primeros 3 niveles (resumen ejecutivo)",
                 fontsize=14, pad=12)
    fig.tight_layout()
    out_top = OUT_DIR / "decision_tree_top3.png"
    fig.savefig(out_top, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[6b] Arbol top-3 niveles exportado: {out_top}")


def _export_subtree(tree, max_depth, title, out):
    fig, ax = plt.subplots(figsize=(40, 18))
    plot_tree(
        tree,
        feature_names=FEATURE_COLS,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
        impurity=False,
        precision=3,
        max_depth=max_depth,
    )
    ax.set_title(title, fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"       Exportado: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_and_compute(CSV_IN)
    X, encoders = prepare_features(df)
    y = df["M_obs"]

    tree   = train_tree(X, y)
    lookup = build_lookup(df, X, tree, encoders)

    lookup.to_csv(CSV_OUT, index=False)
    print(f"\n[5] Tabla exportada: {CSV_OUT}")

    export_png(tree)

    # Resumen de complejidad
    comp = lookup.groupby("complejidad")["n_escenarios"].sum()
    total = comp.sum()
    print("\n=== Resumen de complejidad ===")
    for nivel in ["baja", "media", "alta"]:
        n = comp.get(nivel, 0)
        pct = n / total * 100
        m_med = lookup[lookup["complejidad"] == nivel]["M_recomendado"].mean()
        print(f"  {nivel:<6}: {n:>4} escenarios ({pct:5.1f}%)  M_recomendado_prom={m_med:.4f}")
    print("\nListo.")


if __name__ == "__main__":
    main()
