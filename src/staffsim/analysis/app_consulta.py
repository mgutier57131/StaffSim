"""
Formulario de consulta — Estimador de HC_real
=============================================
Para correr:
    streamlit run src/staffsim/analysis/app_consulta.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

from staffsim.analysis._theme import inject_css, nav_footer, page_config

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

ROOT   = _find_root()
CSV_IN = ROOT / "Resultados Final" / "summary_valor_cp_sat.csv"

SHK          = 0.20
DEPTH        = 6
SEED         = 42
NAN_SENTINEL = -1.0

FEATURE_COLS = [
    "week_pattern", "p_weekdays", "weekday_step", "K",
    "pos1", "pos2", "width1", "width2",
    "peak_amplitude_rule", "ratio_target", "schedule_case",
]
CAT_COLS = ["week_pattern", "peak_amplitude_rule", "schedule_case"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def intervalo_a_hora(n: float) -> str:
    n = int(n)
    h, m = divmod(n * 30, 60)
    return f"{h:02d}:{'30' if m else '00'}"


# ---------------------------------------------------------------------------
# Modelo (se carga una sola vez)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
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

    tree = DecisionTreeRegressor(max_depth=DEPTH, random_state=SEED)
    tree.fit(X, df["M_obs"])

    p33 = float(np.percentile(df["M_obs"].values, 33))
    p66 = float(np.percentile(df["M_obs"].values, 66))
    return tree, encoders, p33, p66


def predecir(vector: dict, tree, encoders, p33, p66) -> dict:
    row = {}
    for col in FEATURE_COLS:
        val = vector.get(col, np.nan)
        if col in CAT_COLS:
            le  = encoders[col]
            s   = str(val) if val is not None else "N/A"
            s   = s if s in le.classes_ else "N/A"
            row[col] = int(le.transform([s])[0])
        else:
            row[col] = float(val) if val is not None else NAN_SENTINEL

    m = float(tree.predict(pd.DataFrame([row])[FEATURE_COLS])[0])

    if m < p33:
        nivel, icono = "Baja",  "🟢"
    elif m <= p66:
        nivel, icono = "Media", "🟡"
    else:
        nivel, icono = "Alta",  "🔴"

    return {"M": m, "nivel": nivel, "icono": icono}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    page_config("Estimador de Headcount", layout="centered")
    inject_css()

    st.title("👥 ¿Cuántos agentes necesito realmente?")
    st.markdown(
        "Este formulario toma las características de tu operación, "
        "consulta el árbol de decisión entrenado con **2 268 simulaciones** "
        "y te devuelve el **factor M** y el **headcount real estimado**."
    )

    tree, encoders, p33, p66 = cargar_modelo()

    # -----------------------------------------------------------------------
    # BLOQUE 1 — Distribución semanal
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("📅 ¿Cómo se distribuye el trabajo en la semana?")

    week_pattern = st.radio(
        "Patrón de carga semanal",
        options=["W1", "W2"],
        format_func=lambda x: (
            "Uniforme — todos los días reciben una carga similar"
            if x == "W1" else
            "Concentrado — los días laborales (lun–vie) tienen más volumen"
        ),
        label_visibility="collapsed",
    )

    p_weekdays = NAN_SENTINEL
    if week_pattern == "W2":
        p_weekdays = st.radio(
            "¿Qué porcentaje del volumen semanal cae en días laborales?",
            options=[0.85, 0.95],
            format_func=lambda x: f"{int(x*100)}% en días laborales",
            horizontal=True,
        )

    # -----------------------------------------------------------------------
    # BLOQUE 2 — Forma de la curva intradiaria
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("📈 ¿Cómo es la curva de demanda durante el día?")

    K = st.radio(
        "¿Cuántos picos de llamadas tiene el día?",
        options=[1, 2],
        format_func=lambda x: (
            "Un solo pico — la demanda sube, llega a su máximo y baja"
            if x == 1 else
            "Dos picos — hay una subida en la mañana y otra en la tarde"
        ),
        label_visibility="collapsed",
    )

    st.markdown("**Pico principal**")
    col1, col2 = st.columns(2)

    pos1_opts  = [14, 25, 36]       if K == 1 else [10, 12, 22]
    width1_opts = [16, 20, 24]      if K == 1 else [13, 15, 17, 25]

    with col1:
        pos1 = st.selectbox(
            "¿A qué hora es el momento de mayor demanda?",
            options=pos1_opts,
            format_func=intervalo_a_hora,
        )
    with col2:
        width1 = st.selectbox(
            "¿Qué tan prolongado es ese pico?",
            options=width1_opts,
            format_func=lambda x: f"~{x//2} horas ({x} intervalos de 30 min)",
        )

    pos2   = NAN_SENTINEL
    width2 = NAN_SENTINEL
    peak_amplitude_rule = "N/A"

    if K == 2:
        st.markdown("**Segundo pico**")
        col3, col4 = st.columns(2)
        with col3:
            pos2 = st.selectbox(
                "¿A qué hora es el segundo pico?",
                options=[28, 38, 40],
                format_func=intervalo_a_hora,
            )
        with col4:
            width2 = st.selectbox(
                "¿Qué tan prolongado es el segundo pico?",
                options=[13, 15, 16, 17, 20, 25],
                format_func=lambda x: f"~{x//2} horas ({x} intervalos de 30 min)",
            )

        peak_amplitude_rule = st.radio(
            "¿Cuál de los dos picos concentra más llamadas?",
            options=["equal", "different_1_gt_2", "different_1_lt_2"],
            format_func=lambda x: {
                "equal"           : "Los dos picos tienen la misma intensidad",
                "different_1_gt_2": "El pico de la mañana es más alto",
                "different_1_lt_2": "El pico de la tarde es más alto",
            }[x],
        )

    # -----------------------------------------------------------------------
    # BLOQUE 3 — Intensidad de la variación
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("📊 ¿Qué tan marcada es la diferencia entre horas pico y horas valle?")

    ratio_target = st.radio(
        "Relación entre el intervalo más ocupado y el menos ocupado del día",
        options=[2, 4, 6],
        format_func=lambda x: {
            2: "2:1 — Poca variación (casi siempre hay carga pareja)",
            4: "4:1 — Variación moderada (hay picos notables pero no extremos)",
            6: "6:1 — Variación alta (las horas pico son muy superiores al valle)",
        }[x],
        label_visibility="collapsed",
    )

    # -----------------------------------------------------------------------
    # BLOQUE 4 — Estructura de turnos
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("🕐 ¿Cómo están estructurados los turnos?")

    duracion_turno = st.radio(
        "¿Los turnos tienen una duración fija o puede variar?",
        options=["fija", "variable"],
        format_func=lambda x: (
            "Fija — todos los agentes trabajan el mismo número de horas por turno"
            if x == "fija" else
            "Variable — los turnos pueden durar 6, 7, 8, 9 o 10 horas según la necesidad"
        ),
        label_visibility="collapsed",
    )

    inicio_turno = st.radio(
        "¿Los horarios de inicio de turno son fijos o flexibles?",
        options=["fijo", "flexible"],
        format_func=lambda x: (
            "Fijos — cada agente siempre entra a la misma hora"
            if x == "fijo" else
            "Flexibles — el sistema puede asignar cualquier hora de inicio"
        ),
        label_visibility="collapsed",
        horizontal=True,
    )

    # Inferir schedule_case
    # run1: turno fijo de 7h, start flexible dentro de rango limitado
    # run2: duraciones variables (6-10h), start en cualquier intervalo
    if duracion_turno == "variable" or inicio_turno == "flexible":
        schedule_case = "run2"
        st.caption("🔧 Estrategia de scheduling inferida: **run2** (turnos flexibles con duraciones variables)")
    else:
        schedule_case = "run1"
        st.caption("🔧 Estrategia de scheduling inferida: **run1** (turno estándar de 7 horas)")

    # -----------------------------------------------------------------------
    # BOTÓN — Calcular
    # -----------------------------------------------------------------------
    st.divider()
    calcular = st.button("⚡ Calcular factor M y headcount", type="primary", use_container_width=True)

    if calcular:
        vector = {
            "week_pattern"        : week_pattern,
            "p_weekdays"          : p_weekdays,
            "weekday_step"        : 0.02,
            "K"                   : K,
            "pos1"                : pos1,
            "pos2"                : pos2,
            "width1"              : width1,
            "width2"              : width2,
            "peak_amplitude_rule" : peak_amplitude_rule,
            "ratio_target"        : ratio_target,
            "schedule_case"       : schedule_case,
        }
        resultado = predecir(vector, tree, encoders, p33, p66)
        st.session_state["resultado"] = resultado
        st.session_state["vector"]    = vector

    # -----------------------------------------------------------------------
    # RESULTADO — persiste aunque el usuario cambie el HC teórico
    # -----------------------------------------------------------------------
    if "resultado" in st.session_state:
        res = st.session_state["resultado"]
        M   = res["M"]

        st.success("✅ Cálculo completado")

        c1, c2 = st.columns(2)
        c1.metric("Factor M recomendado", f"{M:.4f}")
        c2.metric("Complejidad operativa", f"{res['icono']} {res['nivel']}")

        with st.expander("¿Qué significa el nivel de complejidad?"):
            st.markdown(f"""
            | Nivel | Rango de M | Interpretación |
            |---|---|---|
            | 🟢 Baja | M < {p33:.4f} | La demanda es pareja y el scheduling logra ajustarse bien al teórico |
            | 🟡 Media | {p33:.4f} – {p66:.4f} | Hay picos que obligan a contratar más de lo teórico |
            | 🔴 Alta | M > {p66:.4f} | La forma de la demanda fuerza un headcount significativamente mayor |

            **Tu resultado: M = {M:.4f}** → complejidad **{res['nivel']}**
            """)

        st.divider()
        st.subheader("🧮 ¿Cuántos agentes necesito en mi operación?")
        st.markdown(
            "Ingresa el **HC teórico** de tu operación "
            "(el que devuelve el modelo de demanda como headcount bruto requerido) "
            "para calcular cuántos agentes reales vas a necesitar:"
        )

        hc_teo = st.number_input(
            "HC teórico (agentes sin shrinkage)",
            min_value=1.0,
            value=22.0,
            step=1.0,
            format="%.1f",
            key="hc_teo_input",
        )

        hc_real = hc_teo * M
        diferencia = hc_real - hc_teo

        ca, cb, cc = st.columns(3)
        ca.metric("HC teórico", f"{hc_teo:.1f}")
        cb.metric("Factor M", f"{M:.4f}")
        cc.metric("HC real estimado", f"{hc_real:.1f}", delta=f"+{diferencia:.1f}")

        st.latex(r"HC_{real} = HC_{teórico} \times M_{recomendado}")
        st.latex(rf"HC_{{real}} = {hc_teo:.1f} \times {M:.4f} = {hc_real:.1f} \text{{ agentes}}")

        with st.expander("Ver vector de entrada enviado al árbol"):
            st.json(st.session_state["vector"])

    nav_footer()


if __name__ == "__main__":
    main()
