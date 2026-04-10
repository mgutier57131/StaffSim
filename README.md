# StaffSim

Simulador de headcount para centros de contacto.  
Cuantifica cuántos agentes reales se necesitan dado una curva de demanda intradiaria y una estructura de turnos, a través de un **factor multiplicador M** estimado sobre 2 268 escenarios simulados.

> Trabajo de grado — Universidad EAN

---

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

---

## Herramientas

| Herramienta | Comando |
|---|---|
| Simulador de curva de demanda | `streamlit run src/staffsim/gui_app.py` |
| Revisión de resultados | `streamlit run src/staffsim/review_app.py` |
| Estimador de headcount (árbol de decisión) | `streamlit run src/staffsim/analysis/app_consulta.py` |
| Orquestador de escenarios | `python -m staffsim.orchestrate --help` |

---

## Requisitos

- Python 3.10+
- Dependencias declaradas en `pyproject.toml` (pandas, numpy, scikit-learn, streamlit, matplotlib, ortools)

---

## Documentación

Ver [MANUAL.md](MANUAL.md) para descripción completa del modelo, fórmulas, parámetros y flujo de uso.
