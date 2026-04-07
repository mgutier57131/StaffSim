# Manual de uso — StaffSim

---

## Tabla de contenido

1. [¿Qué es StaffSim?](#1-qué-es-staffsim)
2. [Instalación y requisitos](#2-instalación-y-requisitos)
3. [Estructura del proyecto](#3-estructura-del-proyecto)
4. [Fórmulas y conceptos clave](#4-fórmulas-y-conceptos-clave)
5. [Herramienta 1 — Simulador de Curva de Demanda](#5-herramienta-1--simulador-de-curva-de-demanda-gui_apppy)
6. [Herramienta 2 — Orquestador (línea de comando)](#6-herramienta-2--orquestador-línea-de-comando)
7. [Herramienta 3 — Revisión de Resultados](#7-herramienta-3--revisión-de-resultados-review_apppy)
8. [Herramienta 4 — Estimador de Headcount](#8-herramienta-4--estimador-de-headcount-app_consultapy)
9. [Scripts de análisis](#9-scripts-de-análisis)
10. [Archivos generados](#10-archivos-generados)
11. [Flujo completo de uso](#11-flujo-completo-de-uso)

---

## 1. ¿Qué es StaffSim?

StaffSim es un simulador que responde la pregunta central del dimensionamiento de centros de contacto:

> *¿Cuántos agentes reales necesita una operación, dado una curva de demanda y una estructura de turnos?*

La teoría de workload entrega un headcount teórico, pero en la práctica siempre se necesitan más agentes porque la curva de demanda intradiaria no es plana y los turnos tienen restricciones estructurales. StaffSim cuantifica esa diferencia a través de un **factor multiplicador M**, estimado a partir de 2 268 escenarios simulados y un árbol de decisión entrenado sobre ellos.

**Fórmula central:**

```
HC_real = HC_teórico × M_recomendado
```

---

## 2. Instalación y requisitos

### Requisitos
- Python 3.10 o superior
- Las dependencias están declaradas en `pyproject.toml`

### Instalación

```bash
# Desde la carpeta raíz del proyecto (Simulador/)
pip install -e .
```

Esto instala el paquete `staffsim` en modo editable junto con todas sus dependencias (pandas, numpy, scikit-learn, streamlit, matplotlib, ortools, etc.).

### Verificar instalación

```bash
python -c "import staffsim; print('OK')"
```

---

## 3. Estructura del proyecto

```
Simulador/
├── src/staffsim/
│   ├── gui_app.py                      # Herramienta 1: Simulador de Curva (Streamlit)
│   ├── review_app.py                   # Herramienta 3: Revisión de Resultados (Streamlit)
│   ├── orchestrate.py                  # Herramienta 2: Orquestador (línea de comando)
│   │
│   ├── analysis/
│   │   ├── app_consulta.py             # Herramienta 4: Estimador de HC (Streamlit)
│   │   ├── decision_tree_mobs.py       # Script: entrena el árbol de decisión
│   │   ├── depth_selection.py          # Script: selección de profundidad óptima
│   │   ├── plot_mobs_dist.py           # Script: distribución de M_obs
│   │   └── _theme.py                   # Estilo visual compartido entre apps
│   │
│   ├── curves/                         # Generación de curvas de demanda
│   │   ├── simulator_core.py           # Núcleo: construye patrones y matrices
│   │   ├── generator.py
│   │   ├── calls.py
│   │   └── ...
│   │
│   ├── scheduling/                     # Modelos de optimización de turnos
│   │   ├── run1_model.py               # Turno fijo 7h, CP-SAT
│   │   ├── run2_model.py               # Turnos variables 6-10h, CP-SAT
│   │   ├── headless.py                 # Runner sin interfaz
│   │   └── metrics.py                  # Cálculo de cobertura y over/under
│   │
│   ├── orchestrator/                   # Motor del orquestador
│   │   ├── engine.py                   # Lógica principal de ejecución
│   │   ├── grid.py                     # Generación del grid de escenarios
│   │   └── storage.py                  # Lectura/escritura de resultados
│   │
│   ├── demand/
│   │   └── headless.py                 # Runner de demanda sin interfaz
│   │
│   └── workload/
│       └── baseline.py                 # Cálculo de H_talk, HC_teórico, etc.
│
├── Resultados Final/                   # Resultados de la simulación masiva
│   ├── summary_valor_cp_sat.csv        # 2 268 escenarios completos
│   ├── decision_tree_lookup.csv        # Tabla de consulta: 59 hojas con M recomendado
│   ├── decision_tree_mobs.png          # Árbol completo (depth=6)
│   ├── decision_tree_top3.png          # Árbol primeros 3 niveles (para tesis)
│   ├── depth_selection.png             # Gráfica de selección de profundidad
│   ├── depth_selection_metrics.csv     # R², hojas, vars activas por profundidad
│   ├── depth_selection_importance.csv  # Importancia por variable y profundidad
│   └── mobs_distribucion.png           # Distribución estadística de M_obs
│
├── results/                            # Corridas individuales (generadas por gui_app)
│   └── <run_id>/
│       ├── summary.csv
│       ├── params.txt
│       ├── schedule/run1/
│       └── schedule/run2/
│
├── tests/                              # Pruebas automatizadas
├── pyproject.toml                      # Configuración del paquete
└── MANUAL.md                           # Este archivo
```

---

## 4. Fórmulas y conceptos clave

### Cadena de cálculo de headcount

| Variable | Fórmula | Descripción |
|---|---|---|
| `H_talk` | `V × AHT / 3600` | Horas totales de conversación en la semana |
| `H_prod` | `H_talk / OCC` | Horas productivas (incluye tiempo entre llamadas) |
| `H_paid` | `H_prod / (1 − SHK)` | Horas pagadas, descontando ausentismo |
| `HC_teórico` | `H_paid / Hg` | Agentes teóricos necesarios |
| `HC_req` | `ceil(HC_gross)` | Headcount bruto requerido (entero) |
| `HC_gross_sch` | `N_final` del scheduler | Agentes que resuelven el scheduling (sin SHK) |
| `HC_real` | `HC_gross_sch / (1 − SHK)` | Agentes reales con ausentismo aplicado |
| `M_obs` | `HC_real / HC_req` | Factor multiplicador observado en cada simulación |

### Parámetros del modelo
| Parámetro | Símbolo | Valor en simulaciones |
|---|---|---|
| Volumen semanal | V | 7 500 llamadas |
| Tiempo medio de atención | AHT | 300 s (5 min) |
| Ocupación | OCC | 70% |
| Shrinkage | SHK | 20% |
| Horas de jornada semanal | Hg | 42 h |
| Intervalo de análisis | T | 30 min (0.5 h) |

### Niveles de complejidad operativa
Basados en los percentiles globales de M_obs sobre los 2 268 escenarios:

| Nivel | Rango | Interpretación |
|---|---|---|
| 🟢 Baja | M < 1.1932 | La demanda es pareja, el scheduling se ajusta bien |
| 🟡 Media | 1.1932 ≤ M ≤ 1.2500 | Picos que obligan a contratar por encima del teórico |
| 🔴 Alta | M > 1.2500 | La forma de la demanda fuerza un HC significativamente mayor |

### Estrategias de scheduling
| Estrategia | Duración de turno | Inicio de turno |
|---|---|---|
| **run1** | Fija (7 horas = 14 intervalos) | Flexible dentro de rango limitado |
| **run2** | Variable (6, 7, 8, 9 o 10 horas) | Cualquier intervalo del día |

---

## 5. Herramienta 1 — Simulador de Curva de Demanda (`gui_app.py`)

### ¿Qué hace?
Permite calibrar en tiempo real el patrón de llamadas semanal e intradiario.
Es la herramienta de exploración: ver cómo cambia la curva al mover los parámetros.

### Cómo correrla
```bash
streamlit run src/staffsim/gui_app.py
```
Se abre en el navegador en `http://localhost:8501`.

### Parámetros — Panel lateral

**A) Volumen y WFM**

| Campo | Descripción | Rango |
|---|---|---|
| Weekly Volume | Total de llamadas por semana | 1 000 – 200 000 |
| AHT (segundos) | Tiempo promedio de atención | 1 – 3 600 s |
| OCC (%) | Ocupación del agente (acepta 70 o 0.70) | 0 – 100% |
| SHK (%) | Shrinkage — ausentismo (acepta 20 o 0.20) | 0 – 100% |
| Paid Hours | Horas pagadas por agente por semana | 1 – 100 h |

**B) Distribución semanal**

| Campo | Descripción | Opciones |
|---|---|---|
| Week Mode | Cómo se distribuye la carga en la semana | W1: uniforme / W2: más en laborales |
| Weekday Share p | % del volumen en lunes–viernes (solo W2) | 0.71 – 0.999 |
| Weekday Split | Cómo se distribuye entre lun–vie | uniform / increasing-to-friday / decreasing-to-friday |

**C) Forma intradiaria**

| Campo | Descripción |
|---|---|
| Number of Peaks | 1 o 2 picos durante el día |
| Peak Position 1 | Centro del pico en intervalos (0 = 00:00, 24 = 12:00, 47 = 23:30) |
| Peak Width 1 | Ancho del pico en intervalos de 30 min |
| Peak to Valley Ratio | Relación max/min del patrón (1 = plano, 6 = muy marcado) |
| Peak Position 2 | Centro del segundo pico (solo si 2 picos, debe ser > pos1) |
| Peak Width 2 | Ancho del segundo pico |
| Peak Height Mode | Cuál pico es más alto: equal / peak1-higher / peak2-higher |
| Peak Height Ratio | Relación de alturas entre picos (solo si no son iguales) |

### Salidas en pantalla
- Gráfica de **llamadas esperadas** (curva suavizada, 7 días × 48 intervalos)
- Gráfica de **llamadas finales** (enteros, suma exacta = V)
- Métricas: ratio logrado, lambda (λ), HC teórico, FTE min/max
- Tabla de **pesos diarios** (distribución lunes–domingo)

### Exportar
El botón **"Exportar resultados"** guarda en `results/<fecha_hora>/`:
- `calls_matrix.csv` — matriz de llamadas
- `expected_matrix.csv` — matriz esperada
- `fte_matrix.csv` — matriz FTE requerida
- `params.txt` — todos los parámetros usados
- `summary.csv` — métricas de la corrida
- `figure.png` — gráfica de llamadas

---

## 6. Herramienta 2 — Orquestador (línea de comando)

### ¿Qué hace?
Corre el experimento masivo: genera el grid de escenarios, ejecuta la demanda y el scheduling
para cada uno en paralelo, y consolida todo en un archivo de resultados.

### Cómo correrlo
```bash
# Desde la carpeta raíz del proyecto
python -m staffsim.orchestrate --out "Resultados Final/" --parallel 4
```

### Parámetros principales

| Parámetro | Descripción | Default |
|---|---|---|
| `--out` | Directorio de salida | `results/` |
| `--parallel` | Procesos en paralelo | 4 |
| `--cp-sat-workers` | Workers internos del solver CP-SAT | 1 |
| `--stage` | Qué fases correr: `demand`, `schedule` o `both` | `both` |
| `--scheduler` | Solver a usar: `cp_sat`, `hexaly` o `both` | `cp_sat` |
| `--coverage-target` | Cobertura mínima requerida | 0.90 (90%) |
| `--run1-time-limit` | Tiempo máximo por escenario en run1 | 120 s |
| `--run2-time-limit` | Tiempo máximo por escenario en run2 | 600 s |

### Fases de ejecución

**Fase 1 — Demanda:** para cada combinación única de parámetros de curva genera:
- Matriz de llamadas (7 × 48)
- Matriz FTE requerida
- KPIs de demanda (`HC_gross`, `ratio_real`, etc.)

**Fase 2 — Scheduling:** para cada escenario (combinación demanda + estrategia de turno):
- Optimiza la asignación de turnos con CP-SAT
- Busca el mínimo de agentes que alcance ≥ 90% de cobertura
- Registra `N_final`, `coverage`, `sum_under`, `sum_over`

**Consolidación:** fusiona todo en `summary.csv` con todas las variables de entrada y salida.

### Grid de escenarios simulados

El orquestador genera automáticamente todas las combinaciones de:

| Variable | Valores |
|---|---|
| `week_pattern` | W1, W2 |
| `p_weekdays` | 0.85, 0.95 |
| `K` | 1, 2 |
| `pos1` (K=1) | 14, 25, 36 |
| `pos1` (K=2) | 10, 12, 22 |
| `pos2` (K=2) | 28, 38, 40 |
| `width1` (K=1) | 16, 20, 24 |
| `width1` (K=2) | 13, 15, 17, 25 |
| `width2` (K=2) | 13, 15, 16, 17, 20, 25 |
| `peak_amplitude_rule` (K=2) | equal, different_1_gt_2, different_1_lt_2 |
| `ratio_target` | 2, 4, 6 |
| `schedule_case` | run1, run2 |

**Total: 2 268 escenarios** (378 con K=1, 1 890 con K=2).

---

## 7. Herramienta 3 — Revisión de Resultados (`review_app.py`)

### ¿Qué hace?
Permite explorar en detalle los resultados de una corrida individual del Simulador de Curva:
KPIs de demanda, matrices de cobertura y horarios por agente.

### Cómo correrla
```bash
streamlit run src/staffsim/review_app.py
```

### Uso
1. En el panel lateral, selecciona la corrida que quieres revisar (lista de `results/`)
2. Navega por las pestañas:

| Pestaña | Contenido |
|---|---|
| **KPIs** | Parámetros base, HC calculados, comparación run1 vs run2 |
| **Run1 Matrices** | Matrices Required, Planned, Under, Over, Delta (48 × 7) |
| **Run1 Schedules** | Horario detallado por agente en run1 |
| **Run2 Matrices** | Igual para run2 |
| **Run2 Schedules** | Horario detallado por agente en run2 |

### Columnas clave en KPIs

| Columna | Significado |
|---|---|
| `N_final` | Agentes que encontró el scheduler (sin shrinkage) |
| `coverage` | Cobertura lograda (objetivo ≥ 90%) |
| `sum_under` | FTE faltante total en la semana |
| `sum_over` | FTE sobrante total en la semana |
| `solver_status` | OPTIMAL = solución exacta / FEASIBLE = buena solución no óptima |

---

## 8. Herramienta 4 — Estimador de Headcount (`app_consulta.py`)

### ¿Qué hace?
Es la herramienta de consulta final. Recibe las características de una operación,
consulta el árbol de decisión entrenado con los 2 268 escenarios y devuelve:
- El **factor M recomendado**
- El **nivel de complejidad** operativa
- El **HC real estimado** = HC teórico × M

### Cómo correrla
```bash
streamlit run src/staffsim/analysis/app_consulta.py
```

### Preguntas del formulario

| # | Pregunta | Opciones válidas |
|---|---|---|
| 1 | ¿Cómo se distribuye la carga en la semana? | W1: uniforme / W2: concentrada en laborales |
| 2 | % carga en días laborales (solo si W2) | 85% / 95% |
| 3 | ¿Cuántos picos tiene el día? | 1 pico / 2 picos |
| 4 | ¿A qué hora es el pico principal? | K=1: 07:00, 12:30, 18:00 / K=2: 05:00, 06:00, 11:00 |
| 5 | ¿Qué tan prolongado es el pico principal? | K=1: 16, 20, 24 intervalos / K=2: 13, 15, 17, 25 intervalos |
| 6 | ¿A qué hora es el segundo pico? (solo K=2) | 14:00, 19:00, 20:00 |
| 7 | ¿Qué tan prolongado es el segundo pico? (solo K=2) | 13, 15, 16, 17, 20, 25 intervalos |
| 8 | ¿Cuál pico es más alto? (solo K=2) | Iguales / Pico 1 mayor / Pico 2 mayor |
| 9 | ¿Qué tan marcada es la variación pico/valle? | Ratio 2:1 / 4:1 / 6:1 |
| 10 | ¿Los turnos tienen duración fija o variable? | Fija / Variable (6–10 h) |
| 11 | ¿Los horarios de inicio son fijos o flexibles? | Fijos / Flexibles |

> Las preguntas 10 y 11 determinan la estrategia de scheduling:
> duración variable **o** inicio flexible → **run2** / ambos fijos → **run1**

### Cómo interpretar el resultado

1. **Presiona "Calcular factor M y headcount"** — el M y la complejidad quedan fijos
2. **Ingresa el HC teórico** de tu operación (puede cambiarse libremente sin recalcular)
3. La app muestra: `HC_real = HC_teórico × M`

> **Ejemplo:** operación con W2, 95% en laborales, dos picos (06:00 y 19:00),
> ratio 4:1, turnos flexibles → M = 1.3068.
> Si HC teórico = 22 agentes → HC real estimado = 22 × 1.3068 = **29 agentes**.

### Limitación importante
Las opciones del formulario corresponden exactamente a los valores simulados.
Un valor fuera del rango (por ejemplo 92% en laborales) no genera error,
pero la predicción no tiene respaldo empírico directo — el árbol lo asigna
al balde más cercano sin advertencia. El modelo es válido **dentro** del espacio simulado.

---

## 9. Scripts de análisis

Estos scripts se corren desde la línea de comando y regeneran los archivos de análisis en `Resultados Final/`.

### `decision_tree_mobs.py` — Árbol de decisión principal
```bash
python -m staffsim.analysis.decision_tree_mobs
```
**Qué hace:**
- Carga `summary_valor_cp_sat.csv` y calcula M_obs
- Entrena un Decision Tree Regressor con depth=6
- Muestra importancia de variables
- Genera la tabla de lookup con M recomendado y complejidad por hoja
- Exporta `decision_tree_lookup.csv`, `decision_tree_mobs.png`, `decision_tree_top3.png`

### `depth_selection.py` — Selección de profundidad óptima
```bash
python -m staffsim.analysis.depth_selection
```
**Qué hace:**
- Entrena el árbol para profundidades 1 a 11
- Divide los datos 80% train / 20% validación
- Registra R² train, R² validación, número de hojas y variables activas
- Exporta `depth_selection_metrics.csv`, `depth_selection_importance.csv`, `depth_selection.png`

### `plot_mobs_dist.py` — Distribución de M_obs
```bash
python -m staffsim.analysis.plot_mobs_dist
```
**Qué hace:**
- Calcula M_obs para todos los escenarios
- Genera gráfica de 3 paneles: frecuencias reales, comparación con normal, CDF empírica vs teórica
- Incluye test de Shapiro-Wilk para verificar normalidad
- Exporta `mobs_distribucion.png`

---

## 10. Archivos generados

### `Resultados Final/summary_valor_cp_sat.csv`
Archivo principal con los 2 268 escenarios. Columnas clave:

| Columna | Tipo | Descripción |
|---|---|---|
| `scenario_id` | Input | Identificador único del escenario |
| `schedule_case` | Input | run1 o run2 |
| `week_pattern` | Input | W1 o W2 |
| `p_weekdays` | Input | 0.85 o 0.95 |
| `K` | Input | 1 o 2 picos |
| `pos1`, `pos2` | Input | Posición de los picos |
| `width1`, `width2` | Input | Ancho de los picos |
| `peak_amplitude_rule` | Input | Relación entre alturas de picos |
| `ratio_target` | Input | Ratio max/min objetivo |
| `HC_gross_ceil` | Calculado | Headcount bruto requerido (= HC_req) |
| `ratio_real` | Calculado | Ratio max/min real alcanzado |
| `N_final` | Calculado | Agentes del scheduler (= HC_gross_sch) |
| `coverage` | Calculado | Cobertura lograda |
| `sum_under` | Calculado | FTE faltante total |
| `sum_over` | Calculado | FTE sobrante total |
| `solver_status` | Calculado | OPTIMAL / FEASIBLE |

### `Resultados Final/decision_tree_lookup.csv`
Tabla de consulta con las 59 hojas del árbol. Columnas:

| Columna | Descripción |
|---|---|
| `leaf_id` | Identificador de la hoja |
| `condiciones` | Ruta de decisiones que lleva a esta hoja |
| `n_escenarios` | Escenarios que cayeron en esta hoja |
| `M_obs_min/median/max` | Distribución de M_obs en el grupo |
| `sum_over_medio` | Over promedio del grupo |
| `percentil_usado` | 60, 75 o 90 según nivel de over |
| `M_recomendado` | Percentil de M_obs aplicado al grupo |
| `complejidad` | baja / media / alta |

---

## 11. Flujo completo de uso

```
PASO 1 — Explorar el patrón de demanda
  streamlit run src/staffsim/gui_app.py
  → Calibrar los parámetros de la curva
  → Verificar que el patrón representa la operación real
  → Exportar si se desea guardar la corrida

PASO 2 — Correr la simulación masiva (una sola vez, ya está hecha)
  python -m staffsim.orchestrate --out "Resultados Final/" --parallel 4
  → Genera los 2 268 escenarios
  → Tarda varias horas en una máquina estándar
  → Resultado: summary_valor_cp_sat.csv

PASO 3 — Analizar resultados (ya están generados)
  python -m staffsim.analysis.decision_tree_mobs    → árbol y lookup
  python -m staffsim.analysis.depth_selection       → selección de profundidad
  python -m staffsim.analysis.plot_mobs_dist        → distribución de M_obs

PASO 4 — Revisar una corrida individual
  streamlit run src/staffsim/review_app.py
  → Seleccionar la corrida en el panel lateral
  → Explorar KPIs y matrices

PASO 5 — Estimar HC para una operación real
  streamlit run src/staffsim/analysis/app_consulta.py
  → Responder las preguntas del formulario
  → Obtener M recomendado y HC real estimado
```

---

*StaffSim — Trabajo de grado, Universidad EAN, 2026*
