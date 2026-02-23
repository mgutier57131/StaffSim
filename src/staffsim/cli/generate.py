"""Interactive CLI for generating weekly calls and FTE curves (rope + clamp model)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from staffsim.curves.generator import (
    D1_POSITION_TO_POS,
    D2_POSITION_TO_POS,
    L_PRESET_TO_LEN,
    SEED_DEFAULT,
    WEEKDAY_STEP,
    CurveConfig,
    generate_calls_matrix,
)
from staffsim.io.export import DAY_LABELS, export_all
from staffsim.workload.baseline import calls_to_fte_matrix, compute_baseline_summary

T_INTERVAL_DEFAULT = 0.5
FTE_MIN_DEFAULT = 1.0
D_MIN_DEFAULT = 8.0
EPS_DEFAULT = 1e-6
BASE_LEVEL_DEFAULT = 1.0


def _input_with_default(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def _ask_float(prompt: str, default: float | None = None, condition=None, err_msg: str = "Valor invalido.") -> float:
    while True:
        raw = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
        if raw == "" and default is not None:
            value = float(default)
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Ingrese un numero valido.")
                continue
        if condition is None or condition(value):
            return value
        print(err_msg)


def _ask_int(prompt: str, condition=None, err_msg: str = "Valor invalido.") -> int:
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("Ingrese un numero entero valido.")
            continue
        if condition is None or condition(value):
            return value
        print(err_msg)


def _ask_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    choice_set = {c.lower(): c for c in choices}
    while True:
        shown = "/".join(choices)
        raw = _input_with_default(f"{prompt} ({shown})", default or choices[0]).lower()
        if raw in choice_set:
            return choice_set[raw]
        print(f"Opcion invalida. Use una de: {shown}")


def _ask_yes_no(prompt: str, default: str = "y") -> bool:
    return _ask_choice(prompt, ["y", "n"], default=default) == "y"


def _ask_letter_option(title: str, options: dict[str, tuple[str, str]], default_letter: str) -> str:
    print(title)
    for letter, (value, description) in options.items():
        print(f"  {letter}) {value}: {description}")
    while True:
        raw = _input_with_default("Seleccione opcion", default_letter).upper()
        if raw in options:
            return options[raw][0]
        print(f"Opcion invalida. Use una de: {'/'.join(options.keys())}")


def _build_weekly_calls_figure(calls_matrix: np.ndarray) -> plt.Figure:
    fig = plt.figure(figsize=(12, 5))
    x = np.arange(336)
    y = calls_matrix.reshape(-1)
    plt.plot(x, y, linewidth=1.5)
    for day_idx in range(1, 7):
        plt.axvline(48 * day_idx, color="gray", linewidth=0.8, alpha=0.4)
    plt.title("Weekly Calls Curve (Mon-Sun), 30-min intervals")
    plt.xlabel("Time (30-min intervals across week)")
    plt.ylabel("Calls per interval")
    plt.xticks([48 * d for d in range(7)], [f"{DAY_LABELS[d]} 00:00" for d in range(7)], rotation=30)
    plt.tight_layout()
    return fig


def _build_params_text(
    *,
    v_week: int,
    aht: float,
    occ: float,
    shk: float,
    hg: float,
    week_mode: str,
    w2_p: float | None,
    weekday_submode: str | None,
    weekday_peak_day: str | None,
    intraday_mode: str,
    l_preset: str,
    amp_preset: str,
    d1_position: str | None,
    d2_position: str | None,
    d2_height_rel: str | None,
    delta_max: float,
    delta_used: float,
    v_min_needed: float,
    fte_min_obs: float,
    fte_max_obs: float,
) -> str:
    l_value = float(L_PRESET_TO_LEN[l_preset])

    pos_info = "N/A"
    if intraday_mode == "D1" and d1_position is not None:
        c1 = D1_POSITION_TO_POS[d1_position]
        pos_info = f"D1: c1={c1}, L1={l_value}"
    if intraday_mode == "D2" and d2_position is not None:
        c1, c2 = D2_POSITION_TO_POS[d2_position]
        pos_info = f"D2: c1={c1}, c2={c2}, L1={l_value}, L2={l_value}, d={abs(c2-c1)}"

    lines = [
        "StaffSim Parameters",
        "===================",
        f"V: {v_week}",
        f"AHT: {aht}",
        f"OCC: {occ}",
        f"SHK: {shk}",
        f"Hg: {hg}",
        f"T: {T_INTERVAL_DEFAULT}",
        "",
        f"week_mode: {week_mode}",
        f"p: {'' if w2_p is None else w2_p}",
        f"weekday_submode: {'' if weekday_submode is None else weekday_submode}",
        f"weekday_peak_day: {'' if weekday_peak_day is None else weekday_peak_day}",
        f"weekday_step: {WEEKDAY_STEP}",
        "",
        f"intraday_mode: {intraday_mode}",
        f"L_preset: {l_preset}",
        f"amplitude_preset: {amp_preset}",
        f"positions/lengths: {pos_info}",
        f"height_relation (D2): {'' if d2_height_rel is None else d2_height_rel}",
        f"d_min: {D_MIN_DEFAULT}",
        f"base_level b: {BASE_LEVEL_DEFAULT}",
        f"epsilon: {EPS_DEFAULT}",
        "",
        "calls_generation: largest_remainder",
        "boundary_smoothing: local_cubic_4pt",
        f"seed: {SEED_DEFAULT} (reservada)",
        f"FTE_min: {FTE_MIN_DEFAULT}",
        f"delta_max: {delta_max}",
        f"delta_used: {delta_used}",
        f"V_min_needed: {v_min_needed}",
        f"fte_min_observed: {fte_min_obs}",
        f"fte_max_observed: {fte_max_obs}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    print("Simulador 7x48 - modelo cuerda + pinza")
    print("Calls enteros determinísticos por largest remainder (sin multinomial).")

    v_week = _ask_int("1) V (calls/semana)", condition=lambda x: x > 0, err_msg="V debe ser mayor a 0.")

    use_default_operatives = _ask_yes_no("2) Usar defaults de AHT/OCC/SHK/Hg?", default="y")
    if use_default_operatives:
        aht, occ, shk, hg = 300.0, 0.70, 0.20, 42.0
        print("   Usando: AHT=300, OCC=0.70, SHK=0.20, Hg=42")
    else:
        aht = _ask_float("   AHT", default=300.0, condition=lambda x: x > 0)
        occ = _ask_float("   OCC", default=0.70, condition=lambda x: 0 < x <= 1)
        shk = _ask_float("   SHK", default=0.20, condition=lambda x: 0 <= x < 1)
        hg = _ask_float("   Hg", default=42.0, condition=lambda x: x > 0)

    week_mode = _ask_letter_option(
        "3) week_mode:",
        {"A": ("W1", "dias iguales"), "B": ("W2", "L-V suman p, S-D suman 1-p")},
        default_letter="A",
    )
    w2_p = None
    weekday_submode = None
    weekday_peak_day = None
    if week_mode == "W2":
        w2_p = _ask_float("   p (0<p<1 y p>=0.74)", condition=lambda x: 0.74 <= x < 1)
        weekday_submode = _ask_letter_option(
            "   weekday_submode:",
            {
                "A": ("uniform", "L-V iguales"),
                "B": ("increasing-to-friday", "sube a viernes"),
                "C": ("decreasing-to-friday", "baja a viernes"),
                "D": ("midweek-peak", "pico Tue/Wed/Thu"),
            },
            default_letter="A",
        )
        if weekday_submode == "midweek-peak":
            weekday_peak_day = _ask_letter_option(
                "   peak_day:",
                {"A": ("Tue", "martes"), "B": ("Wed", "miercoles"), "C": ("Thu", "jueves")},
                default_letter="B",
            )

    intraday_mode = _ask_letter_option(
        "4) intraday_mode:",
        {"A": ("D1", "1 pinza"), "B": ("D2", "2 pinzas")},
        default_letter="A",
    )
    l_preset = _ask_letter_option(
        "   L preset:",
        {"A": ("min", "L=2"), "B": ("mid", "L=5"), "C": ("max", "L=8")},
        default_letter="B",
    )
    amp_preset = _ask_letter_option(
        "   amplitude preset:",
        {"A": ("min", "0.25*delta_max"), "B": ("mid", "0.50*delta_max"), "C": ("max", "1.00*delta_max")},
        default_letter="B",
    )

    d1_position = None
    d2_position = None
    d2_height_rel = None
    if intraday_mode == "D1":
        d1_position = _ask_letter_option(
            "   posicion pico:",
            {"A": ("inicio", "c=8"), "B": ("medio", "c=24"), "C": ("final", "c=40")},
            default_letter="B",
        )
    else:
        d2_position = _ask_letter_option(
            "   posiciones 2 picos:",
            {"A": ("extremos", "(10,38)"), "B": ("ambos_inicio", "(8,18)"), "C": ("ambos_final", "(30,40)")},
            default_letter="A",
        )
        d2_height_rel = _ask_letter_option(
            "   relacion alturas:",
            {"A": ("equal", "iguales"), "B": ("peak1_higher", "pico1 mayor"), "C": ("peak2_higher", "pico2 mayor")},
            default_letter="A",
        )

    config = CurveConfig(
        v_week=v_week,
        week_mode=week_mode,  # type: ignore[arg-type]
        intraday_mode=intraday_mode,  # type: ignore[arg-type]
        l_preset=l_preset,  # type: ignore[arg-type]
        amp_preset=amp_preset,  # type: ignore[arg-type]
        w2_p=w2_p,
        weekday_submode=weekday_submode,  # type: ignore[arg-type]
        weekday_peak_day=weekday_peak_day,  # type: ignore[arg-type]
        d1_position=d1_position,  # type: ignore[arg-type]
        d2_position=d2_position,  # type: ignore[arg-type]
        d2_height_rel=d2_height_rel,  # type: ignore[arg-type]
    )

    gen = generate_calls_matrix(
        config,
        aht=aht,
        occ=occ,
        t_interval=T_INTERVAL_DEFAULT,
        fte_min=FTE_MIN_DEFAULT,
        seed=SEED_DEFAULT,
    )

    calls_matrix = gen.calls_matrix
    calls_expected_matrix = gen.expected_matrix
    fte_matrix = calls_to_fte_matrix(calls_matrix, aht=aht, occ=occ, t_interval=T_INTERVAL_DEFAULT)
    summary = compute_baseline_summary(v_week=v_week, aht=aht, occ=occ, shk=shk, hg=hg, t_interval=T_INTERVAL_DEFAULT)

    print(f"\nsum(calls)={int(calls_matrix.sum())} (objetivo={v_week})")
    print(f"V_min_needed={gen.v_min_needed:.6f}")
    print(f"delta_max={gen.delta_max:.6f}, delta_used={gen.delta_used:.6f}")
    print("Saltos medianoche (shape smooth):")
    for d in range(6):
        print(f"  k={(d+1)*48}: before={gen.jump_before[d]:.8f}, after={gen.jump_after[d]:.8f}")

    fig = _build_weekly_calls_figure(calls_matrix)
    plt.show()

    if _ask_choice("Desea exportar a CSV?", ["y", "n"], default="y") == "y":
        result_dir = Path("results") / datetime.now().strftime("%Y-%m-%d_%H%M%S")

        params = {
            "V": v_week,
            "AHT": aht,
            "OCC": occ,
            "SHK": shk,
            "Hg": hg,
            "T": T_INTERVAL_DEFAULT,
            "week_mode": week_mode,
            "p": "" if w2_p is None else w2_p,
            "weekday_submode": "" if weekday_submode is None else weekday_submode,
            "weekday_step": WEEKDAY_STEP,
            "weekday_peak_day": "" if weekday_peak_day is None else weekday_peak_day,
            "intraday_mode": intraday_mode,
            "L_preset": l_preset,
            "amplitude_preset": amp_preset,
            "d1_position": "" if d1_position is None else d1_position,
            "d2_position": "" if d2_position is None else d2_position,
            "d2_height_rel": "" if d2_height_rel is None else d2_height_rel,
            "d_min": D_MIN_DEFAULT,
            "base_level": BASE_LEVEL_DEFAULT,
            "epsilon": EPS_DEFAULT,
            "FTE_min": FTE_MIN_DEFAULT,
            "delta_max": gen.delta_max,
            "delta_used": gen.delta_used,
            "v_min_needed": gen.v_min_needed,
            "k_calls_per_fte_interval": gen.k_calls_per_fte_interval,
            "calls_generation": "largest_remainder",
        }

        params_text = _build_params_text(
            v_week=v_week,
            aht=aht,
            occ=occ,
            shk=shk,
            hg=hg,
            week_mode=week_mode,
            w2_p=w2_p,
            weekday_submode=weekday_submode,
            weekday_peak_day=weekday_peak_day,
            intraday_mode=intraday_mode,
            l_preset=l_preset,
            amp_preset=amp_preset,
            d1_position=d1_position,
            d2_position=d2_position,
            d2_height_rel=d2_height_rel,
            delta_max=gen.delta_max,
            delta_used=gen.delta_used,
            v_min_needed=gen.v_min_needed,
            fte_min_obs=float(fte_matrix.min()),
            fte_max_obs=float(fte_matrix.max()),
        )

        export_all(
            output_dir=result_dir,
            calls_matrix=calls_matrix,
            calls_expected_matrix=calls_expected_matrix,
            fte_matrix=fte_matrix,
            params=params,
            summary=summary,
            params_text=params_text,
            extra_metrics={
                "fte_min_observed": float(fte_matrix.min()),
                "fte_max_observed": float(fte_matrix.max()),
            },
        )
        fig.savefig(result_dir / "curve.png", dpi=150)
        print(f"Saved results to: {result_dir.as_posix()}/")


if __name__ == "__main__":
    main()
