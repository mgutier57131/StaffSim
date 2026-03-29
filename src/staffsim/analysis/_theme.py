"""
Tema visual compartido para todas las apps de StaffSim.
Importar e invocar inject_css() al inicio de cada app.
"""
from __future__ import annotations
import streamlit as st

# Paleta de colores
COLOR_PRIMARY   = "#1565C0"   # azul oscuro
COLOR_SUCCESS   = "#2E7D32"   # verde
COLOR_WARNING   = "#F9A825"   # amarillo
COLOR_DANGER    = "#C62828"   # rojo
COLOR_NEUTRAL   = "#546E7A"   # gris azulado
COLOR_BG_CARD   = "#F8F9FA"

COMPLEXITY_COLORS = {
    "baja" : ("#2E7D32", "🟢"),
    "media": ("#F9A825", "🟡"),
    "alta" : ("#C62828", "🔴"),
}

APP_ICON  = "👥"
APP_NAME  = "StaffSim"


def page_config(title: str, layout: str = "wide") -> None:
    """Llama set_page_config con configuracion unificada."""
    st.set_page_config(
        page_title=f"{title} — {APP_NAME}",
        page_icon=APP_ICON,
        layout=layout,
    )


def inject_css() -> None:
    """Inyecta CSS global para unificar tipografia, colores y espaciado."""
    st.markdown("""
    <style>
    /* Fuente base */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Titulo principal */
    h1 { color: #1565C0 !important; border-bottom: 2px solid #1565C0; padding-bottom: 8px; }

    /* Subtitulos */
    h2 { color: #1976D2 !important; }
    h3 { color: #37474F !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F0F4F8;
        border-right: 1px solid #CFD8DC;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1565C0 !important;
        border-bottom: none;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Metricas */
    [data-testid="stMetricValue"] {
        color: #1565C0 !important;
        font-weight: 700;
    }

    /* Boton primario */
    .stButton > button[kind="primary"] {
        background-color: #1565C0 !important;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0D47A1 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        color: #546E7A;
    }
    .stTabs [aria-selected="true"] {
        color: #1565C0 !important;
        border-bottom: 2px solid #1565C0 !important;
    }

    /* Separador */
    hr { border-color: #CFD8DC; }

    /* Info/warning/error box */
    .stAlert { border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)


def app_header(title: str, subtitle: str = "") -> None:
    """Encabezado estandar con titulo y subtitulo opcional."""
    st.title(f"{APP_ICON} {title}")
    if subtitle:
        st.markdown(f"<p style='color:{COLOR_NEUTRAL}; font-size:1.05rem; margin-top:-12px'>{subtitle}</p>",
                    unsafe_allow_html=True)
    st.divider()


def nav_footer() -> None:
    """Pie de pagina con navegacion entre apps."""
    st.divider()
    st.markdown(
        f"<p style='text-align:center; color:{COLOR_NEUTRAL}; font-size:0.85rem'>"
        f"{APP_ICON} <b>StaffSim</b> · "
        "Simulador de Headcount para Centros de Contacto"
        "</p>",
        unsafe_allow_html=True,
    )
