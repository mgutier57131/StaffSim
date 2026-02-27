# StaffSim - Simulador de Curvas 7x48

Simulador para calibrar curvas de calls por intervalo (7 dias x 48 intervalos) y convertirlas a FTE.
Incluye modulo de scheduling ILP (OR-Tools CP-SAT) para planificar cobertura sobre `fte_matrix.csv`.

## Instalacion

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
```

## GUI (principal)

```powershell
python -m streamlit run src/staffsim/gui_app.py
```

Al abrir, inicia en **Curva plana**:
- `week_mode=W1`
- `num_peaks=1`
- `ratio_target=1.0`

## Modelo implementado

### 1) Distribucion semanal

- `W1`: uniforme, `w_i = 1/7`
- `W2`: L-V vs fin de semana
  - `p` (share L-V), con validacion `p >= 5/7` (`0.714286`)
  - `weekday_split`:
    - `uniform`
    - `increasing-to-friday` (step 2%)
    - `decreasing-to-friday` (step 2%)

### 2) Patron intradia comun `p_j`

Se construye con mezcla convexa:
- base uniforme: `u_j = 1/48`
- forma de picos suave `f_j` (gaussianas discretas, `sigma = width/2`)
- mezcla: `p_j = (1-lambda) * u_j + lambda * f_j`

`ratio_target = pico/valle` controla `lambda`:
- `ratio=1` => plana (`lambda=0`)
- `ratio=2` => pico doble
- `ratio=3` => pico triple

`lambda` se resuelve por biseccion en `[0,1]`.

### 3) Calls esperadas y enteras

- `X_ij_expected = V * w_i * p_j`
- Redondeo deterministico (`largest remainder`) sobre 336 celdas:
  - `base = floor(x)`
  - `R = V - sum(base)`
  - `+1` a los `R` mayores decimales

Garantia: `sum(calls_matrix) = V` exacto.

### 4) FTE por intervalo

- `fte_matrix = calls_matrix * AHT / (3600 * T * OCC)`
- `T = 0.5h`

## Export

Boton: **Exportar corrida**

Genera carpeta:

`results/YYYY-MM-DD_HHMMSS/`

con:
- `expected_curve.csv`
- `calls_matrix.csv`
- `fte_matrix.csv`
- `summary.csv`
- `params.txt`
- `curve.png`

## Scheduling ILP (CP-SAT)

Entrada:
- `results/<run_id>/fte_matrix.csv` (requerido R, 7x48)
- `results/<run_id>/summary.csv` (`HC_gross_ceil` como `N0`)

Comandos:

```powershell
python -m staffsim.schedule --run results/<run_id> --mode run1
python -m staffsim.schedule --run results/<run_id> --mode run2
python -m staffsim.schedule --run results/<run_id> --mode both
```

Si `--run` no se pasa, usa el ultimo run en `./results/`.
Se puede ajustar solver:

```powershell
python -m staffsim.schedule --mode both --time-limit 30 --workers 8
```

Outputs finales por run (solo N minimo exitoso):

`results/<run_id>/schedule/<mode>/`

- `required_matrix.csv`
- `planned_matrix.csv`
- `under_matrix.csv`
- `over_matrix.csv`
- `delta_matrix.csv`
- `schedule_detail.csv`
- `ilp_summary.csv`
- `search_log.txt`
- `schedule_curve.png`

## Review App

App unificada para comparar KPIs de demanda + Run1/Run2:

```powershell
python -m streamlit run src/staffsim/review_app.py
```

## Tests

```powershell
python -m pytest -q
```
