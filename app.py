"""Streamlit interface for the Ion Plume Geometric Expansion Model.

Two tabs share one plume propagation model:
  * Direct Density: the V1 workflow; the user enters n0.
  * Current-Derived Density: n0 = I0 / (q v0 A0) from beam current, accelerating
    voltage and individual ion mass, plus an ideal centered collector at x = l.

Each tab keeps its own inputs, errors, result, profile, example flag and result
snapshot under a key prefix ("d_" or "c_").

Run with:  python -m streamlit run app.py --server.address 127.0.0.1
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import streamlit as st

from constants import AMU_KG, MODELED_CHARGE_C
from current_derived import calculate_current_derived
from formatting import (
    format_area,
    format_current,
    format_current_eng,
    format_length,
    format_percent,
    format_sig,
    format_velocity,
    format_voltage,
)
from plume_model import PlumeModelError, PlumeRangeError, calculate_plume, plume_profile
from units import (
    AREA_UNITS,
    CURRENT_UNITS,
    DIRECT_PHYSICAL_FIELDS,
    LENGTH_UNITS,
    VOLTAGE_UNITS,
    CurrentInputs,
    InputParseError,
    area_to_m2,
    area_unit_for_length,
    length_to_m,
    length_unit_for_area,
    m2_to_area,
    m_to_length,
    parse_real,
    validate_current_inputs,
    validate_direct_inputs,
)
from visualization import geometry_figure, profile_figure

TITLE = "Ion Plume Geometric Expansion Model"
SUBTITLE = "Estimate particle-density reduction from geometric expansion under constant axial velocity."
MODEL_SENTENCE = (
    "Model: the particle density is uniform across each cross-section, the tracked particle flow "
    "is conserved, and the mean axial velocity is equal at both planes, so n₀A₀ = n₂A₂."
)
CURRENT_HELP = (
    "The entered current is modeled as one representative ion population with fixed elementary charge and "
    "the entered individual particle mass. Velocity is estimated from accelerating voltage and held constant "
    "downstream. Density and collection assume a uniform plume."
)
STALE_RTOL = 1e-12
D, C = "d_", "c_"  # session-state prefixes for the two tabs
VIEW_AUTO, VIEW_SCHEMATIC = "Auto (to scale when readable)", "Schematic"

DIRECT_EXAMPLE = {
    "d_n0_text": "1.0e20",
    "d_distance_text": "0.10",
    "d_distance_unit": "m",
    "d_theta_text": "10",
    "d_source_mode": "radius",
    "d_source_text": "1.0",
    "d_source_radius_unit": "mm",
}
CURRENT_EXAMPLE = {
    "c_current_text": "1.0",
    "c_current_unit": "µA",
    "c_voltage_text": "1.0",
    "c_voltage_unit": "kV",
    "c_mass_text": "100",
    "c_distance_text": "0.10",
    "c_distance_unit": "m",
    "c_theta_text": "10",
    "c_source_mode": "radius",
    "c_source_text": "1.0",
    "c_source_radius_unit": "mm",
    "c_collector_mode": "radius",
    "c_collector_text": "1.0",
    "c_collector_radius_unit": "mm",
}
CIRCLES = (D + "source", C + "source", C + "collector")  # independent radius/area fields

ss = st.session_state


def _init_state() -> None:
    defaults = {D + "n0_text": "", C + "current_text": "", C + "current_unit": "µA",
                C + "voltage_text": "", C + "voltage_unit": "kV", C + "mass_text": ""}
    for p in (D, C):
        defaults.update({
            p + "distance_text": "", p + "distance_unit": "m", p + "theta_text": "",
            p + "field_errors": {}, p + "range_error": None, p + "result": None, p + "profile": None,
            p + "result_inputs": None, p + "result_is_example": False,
            p + "density_scale": "Logarithmic", p + "diagram_view": VIEW_AUTO,
        })
    for base in CIRCLES:
        defaults.update({base + "_mode": "radius", base + "_text": "", base + "_radius_unit": "mm",
                         base + "_area_unit": "mm²", base + "_converted": None, base + "_note": None})
    for key, value in defaults.items():
        if key not in ss:
            ss[key] = value


_init_state()


# ---------------------------------------------------------------- radius/area fields
def _circle_unit(base: str) -> str:
    return ss[base + "_radius_unit"] if ss[base + "_mode"] == "radius" else ss[base + "_area_unit"]


def _on_circle_mode_change(base: str, noun: str) -> None:
    """Convert one radius-or-area value so its physical size is preserved.

    The exact radius behind a converted, unedited value is reused so repeated
    switches do not accumulate display rounding. Each field keeps its own state.
    """
    ss[base + "_note"] = None
    text = str(ss[base + "_text"]).strip()
    if not text:
        return
    to_area = ss[base + "_mode"] == "area"  # the radio already holds the new mode
    carried = ss[base + "_converted"]
    old_unit = ss[base + "_radius_unit"] if to_area else ss[base + "_area_unit"]
    try:
        if carried and carried[:2] == (text, old_unit):
            radius_m = carried[2]
        else:
            value = parse_real(text)
            if value <= 0.0:
                raise InputParseError("nonpositive")
            if to_area:
                radius_m = length_to_m(value, ss[base + "_radius_unit"])
            else:
                radius_m = math.sqrt(area_to_m2(value, ss[base + "_area_unit"]) / math.pi)
        if to_area:
            unit = area_unit_for_length(ss[base + "_radius_unit"])
            converted = m2_to_area(math.pi * radius_m * radius_m, unit)
            ss[base + "_area_unit"] = unit
        else:
            unit = length_unit_for_area(ss[base + "_area_unit"])
            converted = m_to_length(radius_m, unit)
            ss[base + "_radius_unit"] = unit
        if not (math.isfinite(converted) and converted > 0.0 and radius_m > 0.0):
            raise InputParseError("nonrepresentable")
        ss[base + "_text"] = f"{converted:.15g}"
        ss[base + "_converted"] = (ss[base + "_text"], unit, radius_m)
    except (InputParseError, PlumeModelError):
        ss[base + "_text"] = ""
        ss[base + "_note"] = f"The previous {noun} value was not a valid positive number, so it was cleared."


def _clear_circle(base: str) -> None:
    ss[base + "_text"] = ""
    ss[base + "_note"] = None
    ss[base + "_converted"] = None


# ---------------------------------------------------------------- validation and calculation
def _direct_inputs():
    return validate_direct_inputs(
        ss.d_n0_text, ss.d_distance_text, ss.d_distance_unit, ss.d_theta_text,
        ss.d_source_mode, ss.d_source_text, _circle_unit(D + "source"),
    )


def _current_inputs():
    return validate_current_inputs(
        current_text=ss.c_current_text, current_unit=ss.c_current_unit,
        voltage_text=ss.c_voltage_text, voltage_unit=ss.c_voltage_unit,
        mass_text=ss.c_mass_text,
        distance_text=ss.c_distance_text, distance_unit=ss.c_distance_unit,
        theta_text=ss.c_theta_text,
        geometry_mode=ss.c_source_mode, geometry_text=ss.c_source_text, geometry_unit=_circle_unit(C + "source"),
        collector_mode=ss.c_collector_mode, collector_text=ss.c_collector_text,
        collector_unit=_circle_unit(C + "collector"),
        charge_c=MODELED_CHARGE_C,
    )


def _run(p: str, validate, compute) -> None:
    """Validate one tab, compute, and store result + profile + snapshot together only on success."""
    normalized, errors = validate()
    ss[p + "field_errors"] = errors
    ss[p + "range_error"] = None
    if normalized is None:
        return
    try:
        result, profile = compute(normalized)
    except PlumeRangeError as exc:
        ss[p + "range_error"] = f"Numerical range error ({exc.quantity}): {exc}"
        return
    except PlumeModelError as exc:
        ss[p + "range_error"] = str(exc)
        return
    ss[p + "result"] = result
    ss[p + "profile"] = profile
    ss[p + "result_inputs"] = normalized
    ss[p + "result_is_example"] = False


def _compute_direct(n):
    result = calculate_plume(n.n0, n.r0_m, n.distance_m, n.theta_rad)
    return result, plume_profile(result)


def _compute_current(n: CurrentInputs):
    result = calculate_current_derived(n.current_a, n.voltage_v, n.mass_kg, n.r0_m, n.distance_m, n.theta_rad,
                                       n.collector_radius_m, n.charge_c)
    return result, plume_profile(result.plume)


def _calculate_direct() -> None:
    _run(D, _direct_inputs, _compute_direct)


def _calculate_current() -> None:
    _run(C, _current_inputs, _compute_current)


def _load_example(p: str) -> None:
    example, calculate = (DIRECT_EXAMPLE, _calculate_direct) if p == D else (CURRENT_EXAMPLE, _calculate_current)
    for key, value in example.items():
        ss[key] = value
    for base in CIRCLES:
        if base.startswith(p):
            ss[base + "_note"] = None
            ss[base + "_converted"] = None
    calculate()
    ss[p + "result_is_example"] = ss[p + "result"] is not None


def _reset(p: str) -> None:
    """Clear this tab's result, plots, errors, example state and required geometry only."""
    for key, value in (("result", None), ("profile", None), ("result_inputs", None),
                       ("result_is_example", False), ("field_errors", {}), ("range_error", None)):
        ss[p + key] = value
    for base in CIRCLES:
        if base.startswith(p):
            _clear_circle(base)


def _is_stale(p: str) -> bool:
    """True when the current physical inputs no longer describe the displayed result."""
    normalized, _ = (_direct_inputs if p == D else _current_inputs)()
    old = ss[p + "result_inputs"]
    if normalized is None or old is None:
        return True
    fields = DIRECT_PHYSICAL_FIELDS if p == D else CurrentInputs.PHYSICAL_FIELDS
    return not all(
        math.isclose(getattr(normalized, f), getattr(old, f), rel_tol=STALE_RTOL, abs_tol=0.0) for f in fields
    )


# ---------------------------------------------------------------- input widgets
def _field_error(p: str, field: str) -> None:
    message = ss[p + "field_errors"].get(field)
    if message:
        st.error(message, icon="⚠️")


def _render_distance_theta(p: str) -> None:
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    c1.text_input("Axial distance l from P₀ to P₂", key=p + "distance_text", placeholder="e.g. 0.10")
    c2.selectbox("Distance unit", LENGTH_UNITS, key=p + "distance_unit")
    _field_error(p, "distance")

    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    c1.text_input("Plume half-angle θ", key=p + "theta_text", placeholder="e.g. 10")
    c2.markdown("degrees (half-angle)")
    _field_error(p, "theta")


def _render_definitions() -> None:
    st.caption(
        "**Reference plane:** P₀ is the center of the finite source cross-section A₀ at x = 0; "
        "P₂ is the center of the downstream plane at x = l, measured along the plume axis (not the slant). "
        "**Half-angle:** θ is the angle between the plume boundary and a line parallel to the axis. "
        "If you have a full opening angle, divide it by two before entry."
    )


def _render_circle(p: str, base: str, noun: str, radius_label: str, area_label: str, field: str) -> None:
    symbol_r, symbol_a = ("r₀", "A₀") if noun == "source" else ("r_coll", "A_coll")
    st.radio(f"{noun.capitalize()} geometry (enter one)", ["radius", "area"], key=base + "_mode", horizontal=True,
             format_func=lambda m: f"Radius {symbol_r}" if m == "radius" else f"Area {symbol_a}",
             on_change=_on_circle_mode_change, args=(base, noun))
    radius_mode = ss[base + "_mode"] == "radius"
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    c1.text_input(radius_label if radius_mode else area_label, key=base + "_text",
                  placeholder="required; no default is assumed")
    if radius_mode:
        c2.selectbox(f"{noun.capitalize()} radius unit", LENGTH_UNITS, key=base + "_radius_unit")
    else:
        c2.selectbox(f"{noun.capitalize()} area unit", AREA_UNITS, key=base + "_area_unit")
    _field_error(p, field)
    if ss[base + "_note"]:
        st.warning(ss[base + "_note"])


def _signed_entry_note(text, symbol: str, unit: str) -> None:
    try:
        value = parse_real(text)
    except InputParseError:
        return
    if value < 0.0:
        st.caption(f"Signed entry interpreted by magnitude: |{symbol}| = {format_sig(abs(value))} {unit}. "
                   "The sign does not configure polarity.")


def _render_buttons(p: str, missing_message: str | None) -> None:
    b1, b2, b3 = st.columns(3)
    calculate = _calculate_direct if p == D else _calculate_current
    b1.button("Calculate", key=p + "calculate", type="primary", on_click=calculate,
              disabled=missing_message is not None, width="stretch")
    b2.button("Load example", key=p + "load_example", on_click=_load_example, args=(p,), width="stretch")
    b3.button("Reset", key=p + "reset", on_click=_reset, args=(p,), width="stretch")
    if missing_message:
        st.caption(missing_message)


# ---------------------------------------------------------------- shared result display
def _stale_banner(p: str, example_text: str) -> tuple[bool, bool, str]:
    stale = _is_stale(p)
    illustrative = ss[p + "result_is_example"] and not stale
    if stale:
        st.warning(
            "OUT OF DATE: the inputs have changed since this result was calculated. The values and plots "
            "below belong to the previous inputs. Press Calculate to update them.",
            icon="⚠️",
        )
    if illustrative:
        st.info(example_text)
    return stale, illustrative, "[Out of date] " if stale else ""


def _render_density_metric(result, prefix: str) -> None:
    st.metric(f"{prefix}Downstream density n₂ at P₂ (model estimate)", f"{format_sig(result.n2)} particles/m³")
    st.caption("n₂ is the uniform cross-sectional model estimate, not a measured density or a detector count rate.")


def _render_geometry_outputs(result) -> None:
    st.markdown(
        "| Quantity | Value |\n|---|---|\n"
        f"| Source radius r₀ | {format_length(result.r0_m)} |\n"
        f"| Downstream beam radius r₂ | {format_length(result.r2_m)} |\n"
        f"| Source area A₀ | {format_area(result.area0_m2)} |\n"
        f"| Full downstream beam area A₂ | {format_area(result.area2_m2)} |\n"
        f"| Area expansion E = A₂/A₀ | {format_sig(result.expansion)} |\n"
        f"| Remaining fraction F = A₀/A₂ = n₂/n₀ | {format_sig(result.remaining_fraction)} |\n"
    )
    m1, m2 = st.columns(2)
    m1.metric("Remaining density (% of n₀)", format_percent(result.remaining_fraction, result.reduction_fraction))
    m2.metric("Reduction (% decrease from n₀)", format_percent(result.reduction_fraction, result.remaining_fraction))
    if result.high_angle_sensitivity:
        st.warning(
            "θ ≥ 85°: tan θ grows rapidly, so the result is highly sensitive to small changes in the "
            "half-angle. This is a numerical-sensitivity note, not a physical validity boundary."
        )


def _geometry_detail_rows(result) -> str:
    d = lambda v: format_sig(v, 10)  # noqa: E731
    return (
        f"| r₀ | {d(result.r0_m)} m |\n"
        f"| l | {d(result.distance_m)} m |\n"
        f"| θ | {d(result.theta_deg)}° = {d(result.theta_rad)} rad |\n"
        f"| Δr = l tan θ | {d(result.delta_r_m)} m |\n"
        f"| r₂ = r₀ + Δr | {d(result.r2_m)} m |\n"
        f"| A₀ = πr₀² | {d(result.area0_m2)} m² |\n"
        f"| A₂ = πr₂² | {d(result.area2_m2)} m² |\n"
        f"| F = (r₀/r₂)² | {d(result.remaining_fraction)} |\n"
        f"| E = (r₂/r₀)² | {d(result.expansion)} |\n"
        f"| n₂ = n₀F | {d(result.n2)} particles/m³ |\n"
        f"| Remaining 100F | {d(result.remaining_percent)} % |\n"
        f"| Reduction 100(1 − F) | {d(result.reduction_percent)} % |\n"
    )


def _render_plots(p: str, result, profile, illustrative: bool, prefix: str,
                  collector_radius_m: float | None = None) -> None:
    st.subheader(f"{prefix}Plume geometry")
    st.radio("Diagram view", [VIEW_AUTO, VIEW_SCHEMATIC], key=p + "diagram_view", horizontal=True)
    fig, schematic = geometry_figure(
        result, view="schematic" if ss[p + "diagram_view"] == VIEW_SCHEMATIC else "auto",
        illustrative=illustrative, collector_radius_m=collector_radius_m,
    )
    st.pyplot(fig)
    plt.close(fig)
    if schematic:
        st.caption("Schematic, not to scale: proportions are adjusted for readability; the labels show the "
                   "true values.")

    st.subheader(f"{prefix}Density vs axial distance")
    st.radio("Density axis scale", ["Logarithmic", "Linear"], key=p + "density_scale", horizontal=True)
    x_m, density = profile
    fig = profile_figure(result, x_m, density, log_y=ss[p + "density_scale"] == "Logarithmic",
                         illustrative=illustrative)
    st.pyplot(fig)
    plt.close(fig)
    if result.distance_m == 0.0:
        st.caption("l = 0: P₀ and P₂ coincide, so the profile is a single point with n₂ = n₀.")


# ---------------------------------------------------------------- tabs
def _render_direct_tab() -> None:
    inputs_col, results_col = st.columns([2, 3], gap="large")
    with inputs_col:
        st.subheader("Inputs")
        c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
        c1.text_input("Source density n₀ at P₀", key=D + "n0_text", placeholder="e.g. 1e20",
                      help="Uniform number density at the reference plane. Scientific notation is accepted.")
        c2.markdown("particles/m³")
        _field_error(D, "n0")
        _render_distance_theta(D)
        _render_definitions()
        _render_circle(D, D + "source", "source", "Source radius r₀ at P₀",
                       "Source area A₀ at P₀ (full plume cross-section)", "geometry")
        st.caption("The source size must be known: n₀, l and θ alone do not determine n₂. "
                   "An assumed source radius changes the normalization of the prediction.")
        missing = None if str(ss.d_source_text).strip() else "Enter a source radius or area to enable Calculate."
        _render_buttons(D, missing)

    with results_col:
        st.subheader("Results")
        if ss.d_range_error:
            st.error(ss.d_range_error)
        result = ss.d_result
        if result is None:
            st.info("No result yet. Enter n₀, l, θ and a known source radius or area, then press Calculate.")
            return
        stale, illustrative, prefix = _stale_banner(
            D, "Illustrative example values (PRD benchmark case), not a measured lab configuration.")
        _render_density_metric(result, prefix)
        _render_geometry_outputs(result)
        with st.expander("Calculation details (10 significant figures)"):
            st.markdown("| Quantity | Value (SI) |\n|---|---|\n"
                        f"| n₀ (entered) | {format_sig(result.n0, 10)} particles/m³ |\n"
                        + _geometry_detail_rows(result))
        _render_plots(D, result, ss.d_profile, illustrative, prefix)


def _render_current_tab() -> None:
    st.info(CURRENT_HELP)
    inputs_col, results_col = st.columns([2, 3], gap="large")
    with inputs_col:
        st.subheader("Inputs")
        c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
        c1.text_input("Beam current magnitude |I₀|", key=C + "current_text", placeholder="e.g. 1.0",
                      help="Total current assigned to the modeled population through A₀.")
        c2.selectbox("Current unit", CURRENT_UNITS, key=C + "current_unit")
        _signed_entry_note(ss.c_current_text, "I₀", ss.c_current_unit)
        _field_error(C, "current")

        c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
        c1.text_input("Accelerating voltage magnitude |V₀|", key=C + "voltage_text", placeholder="e.g. 1.0",
                      help="Potential difference used to estimate the modeled particle energy qV₀.")
        c2.selectbox("Voltage unit", VOLTAGE_UNITS, key=C + "voltage_unit")
        _signed_entry_note(ss.c_voltage_text, "V₀", ss.c_voltage_unit)
        _field_error(C, "voltage")
        st.caption("Current and voltage are magnitudes: a signed entry is interpreted by its magnitude and does "
                   "not configure polarity.")

        c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
        c1.text_input("Individual ion/particle mass mᵢ", key=C + "mass_text", placeholder="e.g. 100",
                      help="Mass of one modeled particle, not the beam mass. Converted to kg internally.")
        c2.markdown("amu")
        _field_error(C, "mass")
        st.caption("Modeled charge q = e ≈ 1.6 × 10⁻¹⁹ C (one elementary charge; fixed in this version).")

        _render_distance_theta(C)
        _render_definitions()
        _render_circle(C, C + "source", "source", "Source radius r₀ at P₀",
                       "Source area A₀ at P₀ (full plume cross-section)", "geometry")
        _render_circle(C, C + "collector", "collector", "Collector radius r_coll at x = l",
                       "Collector area A_coll at x = l", "collector")
        st.caption("The collector is circular, centered on the axis and perpendicular to it at x = l. "
                   "Its size is separate from the full beam footprint A₂.")
        missing = [name for name, base in (("source", C + "source"), ("collector", C + "collector"))
                   if not str(ss[base + "_text"]).strip()]
        message = (f"Enter the {' and '.join(missing)} radius or area to enable Calculate." if missing else None)
        _render_buttons(C, message)

    with results_col:
        st.subheader("Results")
        if ss.c_range_error:
            st.error(ss.c_range_error)
        result = ss.c_result
        if result is None:
            st.info("No result yet. Enter |I₀|, |V₀|, mᵢ, l, θ and known source and collector sizes, "
                    "then press Calculate.")
            return
        stale, illustrative, prefix = _stale_banner(
            C, "Illustrative example values (software-verification case), not a measured lab configuration.")
        source, plume, collection = result.source, result.plume, result.collection
        inputs: CurrentInputs = ss.c_result_inputs

        _render_density_metric(plume, prefix)

        with st.container(border=True):
            st.metric(f"{prefix}Calculated source density n₀ (derived from I₀, V₀, mᵢ, r₀; not entered)",
                      f"{format_sig(source.n0)} particles/m³")
            current_note = " (entered with a negative sign; magnitude used)" if inputs.current_sign_negative else ""
            voltage_note = " (entered with a negative sign; magnitude used)" if inputs.voltage_sign_negative else ""
            st.markdown(
                "| Source quantity | Value |\n|---|---|\n"
                f"| Beam current magnitude \\|I₀\\| | {format_current(source.current_a)}{current_note} |\n"
                f"| Accelerating voltage magnitude \\|V₀\\| | {format_voltage(source.voltage_v)}{voltage_note} |\n"
                f"| Individual ion mass mᵢ | {format_sig(inputs.mass_amu)} amu |\n"
                f"| Idealized characteristic velocity v₀ | {format_velocity(source.velocity_m_s)} |\n"
            )
            st.caption("v₀ = √(2qV₀/mᵢ) is an idealized characteristic velocity, not a measured plume-average "
                       "velocity. It is the constant axial flux velocity in this model.")
            if source.relativistic_notice:
                st.warning(
                    f"v₀ is {format_sig(100 * source.speed_fraction_of_c)} % of the speed of light. The "
                    "nonrelativistic relation ½mᵢv₀² = qV₀ becomes approximate; relativistic corrections grow as "
                    "(v₀/c)². This is a notice, not a validity boundary."
                )

        _render_geometry_outputs(plume)

        with st.container(border=True):
            st.markdown(f"**{prefix}Centered collector at x = l** (circular, perpendicular to the axis)")
            st.markdown(
                "| Collector quantity | Value |\n|---|---|\n"
                f"| Collector radius r_coll | {format_length(collection.collector_radius_m)} |\n"
                f"| Collector area A_coll (not A₂) | {format_area(collection.collector_area_m2)} |\n"
                f"| Sampled current fraction f_coll | "
                f"{'1 (exactly)' if collection.saturated else format_sig(collection.fraction)} |\n"
            )
            k1, k2 = st.columns(2)
            k1.metric(f"{prefix}Collected current I_coll (ideal)", format_current_eng(collection.collected_current_a))
            k2.metric("Collected percentage (100 f_coll)",
                      format_percent(collection.fraction, collection.uncollected_fraction))
            if collection.saturated:
                st.info("r_coll ≥ r₂: the collector covers the entire modeled beam, so f_coll = 1 exactly and "
                        "I_coll = I₀. A larger collector collects no extra current outside the beam.")
            st.caption("I_coll is an ideal primary collected-current magnitude, not a calibrated detector reading "
                       "or a secondary-ion current. f_coll is separate from the density fraction F.")

        with st.expander("Calculation details (10 significant figures)"):
            d = lambda v: format_sig(v, 10)  # noqa: E731
            st.markdown(
                "| Quantity | Value (SI) |\n|---|---|\n"
                f"| \\|I₀\\| | {d(source.current_a)} A |\n"
                f"| \\|V₀\\| | {d(source.voltage_v)} V |\n"
                f"| mᵢ | {d(inputs.mass_amu)} amu = {d(source.mass_kg)} kg |\n"
                f"| amu conversion (fixed) | 1 amu = {format_sig(AMU_KG, 12)} kg |\n"
                f"| q (fixed modeled charge, e) | {d(source.charge_c)} C |\n"
                f"| qV₀ | {d(source.kinetic_energy_j)} J |\n"
                f"| v₀ = √(2qV₀/mᵢ) | {d(source.velocity_m_s)} m/s |\n"
                f"| A₀ = πr₀² | {d(source.area0_m2)} m² |\n"
                f"| n₀ = I₀/(q v₀ A₀) | {d(source.n0)} particles/m³ |\n"
                + _geometry_detail_rows(plume)
                + f"| Check: n₂ = I₀/(q v₀ A₂) | {d(result.n2_from_current)} particles/m³ |\n"
                f"| Full beam current at x = l: n₂ q v₀ A₂ | {d(result.downstream_beam_current_a)} A (= I₀) |\n"
                f"| r_coll | {d(collection.collector_radius_m)} m |\n"
                f"| A_coll = πr_coll² | {d(collection.collector_area_m2)} m² |\n"
                f"| f_coll = min(1, A_coll/A₂) | {d(collection.fraction)} |\n"
                f"| Collected 100 f_coll | {d(collection.percent)} % |\n"
                f"| I_coll = I₀ f_coll | {d(collection.collected_current_a)} A |\n"
            )
        _render_plots(C, plume, ss.c_profile, illustrative, prefix,
                      collector_radius_m=collection.collector_radius_m)


# ---------------------------------------------------------------- layout
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)
st.markdown(f"**{SUBTITLE}**")
st.info(MODEL_SENTENCE)

tab_direct, tab_current = st.tabs(["Direct Density", "Current-Derived Density"])
with tab_direct:
    _render_direct_tab()
with tab_current:
    _render_current_tab()

with st.expander("How the model works"):
    st.markdown(
        "**Idea.** Radial expansion spreads a conserved particle flow over a larger area. At constant mean axial "
        "velocity, n₀A₀ = n₂A₂. The formula gives a uniform density estimate; it does not provide an "
        "experimental radial profile. Both tabs use this same propagation law; the Current-Derived tab only "
        "changes how n₀ is obtained."
    )
    st.markdown("**Governing law** (steady flow of a conserved, tracked population, bounded by streamlines):")
    st.latex(r"\dot N = \int_A n\,\mathbf{v}\cdot\hat{\mathbf{x}}\,dA \;\rightarrow\; n A v_x")
    st.latex(r"\rho = nm,\qquad \dot m = \rho A v_x = n m A v_x")
    st.latex(r"n_0 A_0 v_{x,0} = n_2 A_2 v_{x,2},\qquad v_{x,0} = v_{x,2} > 0")
    st.markdown(
        "The velocity in the planar flux is its **axial component**. Equal total particle speed alone is not "
        "enough if the distribution of velocity directions changes. For one species of unchanged mass m, m "
        "cancels. Particle-number conservation is assumed explicitly; total mass conservation alone would not "
        "conserve the number of ions if the population fragmented, recombined or changed species."
    )
    st.markdown("**Geometry** (circular conical frustum with a finite source radius r₀):")
    st.latex(r"r_0 = \sqrt{A_0/\pi},\qquad r_2 = r_0 + l\tan\theta,\qquad A_2 = \pi r_2^2")
    st.latex(r"n_2 = n_0\frac{A_0}{A_2} = n_0\left(\frac{r_0}{r_0 + l\tan\theta}\right)^2")
    st.latex(r"F = \frac{A_0}{A_2} = \frac{n_2}{n_0},\qquad E = \frac{A_2}{A_0} = \frac{1}{F}")
    st.latex(r"r(x) = r_0 + x\tan\theta,\qquad n(x) = n_0\left(\frac{r_0}{r_0 + x\tan\theta}\right)^2")
    st.markdown(
        "- A₂ is the full plume footprint at the downstream plane, **not** the instrument aperture or collector "
        "area. Do not substitute the detector opening into the dilution equation.\n"
        "- The common height h of P₀ and P₂ only sets placement; it is not a calculation input.\n"
        "- θ is the **half**-angle; divide a full opening angle by two. l is axial, not a slant distance.\n"
        "- In Direct Density mode no particle mass, charge, energy or absolute velocity is needed, because the "
        "same mass and equal nonzero axial velocity cancel.\n"
        "- The density uses the radius ratio (r₀/r₂)², and 1 − F is evaluated as (Δr/r₂)(1 + r₀/r₂) to "
        "avoid rounding error when F is close to 1."
    )
    st.markdown("**Current-Derived Density mode**")
    st.markdown(CURRENT_HELP)
    st.latex(r"\tfrac12 m_i v_0^2 = qV_0,\qquad v_0 = \sqrt{\frac{2qV_0}{m_i}}")
    st.latex(r"I_0 = n_0 q v_0 A_0,\qquad n_0 = \frac{I_0}{q v_0 A_0} = \frac{I_0}{q\pi r_0^2}\sqrt{\frac{m_i}{2qV_0}}")
    st.latex(r"n_2 = \frac{I_0}{q v_0 A_2}\quad\text{(the full beam current stays } I_0 \text{ at every cross-section)}")
    st.markdown(
        f"- q is fixed at one elementary charge, e = {format_sig(MODELED_CHARGE_C, 10)} C. mᵢ is entered in amu "
        f"and converted with the agreed fixed factor 1 amu = {format_sig(AMU_KG, 12)} kg.\n"
        "- mᵢ is the mass of one modeled particle. Assigning the total measured current to one representative "
        "population with that mass is an explicit approximation; a real electrospray plume contains several "
        "species and a broad energy distribution.\n"
        "- V₀ is the potential difference used to estimate the particle energy. The energy relation assumes "
        "negligible initial kinetic energy and a nonrelativistic speed; v₀ is an idealized characteristic "
        "velocity, not a measured plume-average velocity. Inputs giving v₀ ≥ c are rejected.\n"
        "- The axial velocity is taken as v₀ and held constant downstream. θ controls footprint growth only; "
        "velocity is not multiplied by cos θ. Changing I₀ does not change v₀.\n"
        "- Current and voltage entries are magnitudes; a negative sign does not configure polarity. Zero current "
        "or voltage is rejected."
    )
    st.markdown("**Centered collector** (circular, centered on the axis, perpendicular to it at x = l):")
    st.latex(r"A_{coll} = \pi r_{coll}^2,\qquad f_{coll} = \min\!\left(1, \frac{A_{coll}}{A_2}\right),\qquad "
             r"I_{coll} = I_0 f_{coll}")
    st.markdown(
        "- Assumes uniform axial current density over the full footprint A₂ and ideal collection of every "
        "particle crossing the overlap. If r_coll ≥ r₂ the collector covers the whole beam: f_coll = 1 exactly "
        "and I_coll = I₀. A larger collector collects nothing outside the beam.\n"
        "- f_coll and F are separate quantities. They are equal only when the collector and source have the "
        "same area.\n"
        "- I_coll is an ideal primary collected-current magnitude, not a calibrated detector reading, count "
        "rate or secondary-ion current."
    )
    st.markdown("**Far-field limit** (explanation only; the app always uses the full frustum equation):")
    st.latex(r"l\tan\theta \gg r_0 \;\Rightarrow\; n_2 \approx \frac{n_0 r_0^2}{l^2\tan^2\theta}")
    st.markdown(
        "This inverse-square behavior applies only for θ > 0 and l tan θ ≫ r₀. A point-source model with finite "
        "particle flow would need an independent flow normalization or a finite reference measurement. With "
        "fixed n₀, shrinking A₀ toward zero also shrinks the particle flow."
    )
    st.markdown(
        "**Assumptions**\n"
        "- Emission is steady; tracked particles do not accumulate between the planes.\n"
        "- The plume is axisymmetric, with circular cross-sections, a constant half-angle and straight frustum "
        "boundaries.\n"
        "- Number density and axial current density are uniform across each cross-section.\n"
        "- The tracked particle population is conserved: no sources, losses, neutralization, fragmentation, "
        "deposition or escape across the modeled boundary.\n"
        "- Mean axial velocity is equal at both planes; acceleration and changes in the angular velocity "
        "distribution are excluded.\n"
        "- P₀ is a finite reference cross-section, even if it looks like a point in a schematic."
    )
    st.markdown(
        "**What n₂ represents.** In this uniform model the value at the centerline point P₂ equals the "
        "cross-sectional average. Area growth alone does not determine the local density of a real, nonuniform "
        "plume. Predicting an instrument signal requires its sampling geometry, acceptance, efficiency, particle "
        "properties and a radial density or flux distribution. n₂ is a model estimate, not a measured density or "
        "detector count rate, and geometric calculations do not establish the instrument's measured response."
    )
    st.markdown(
        "**Excluded physics:** changes in velocity along the plume; electric-field forces; Coulomb interactions "
        "and space charge; collisions; fragmentation or neutralization; evolution of velocity directions; "
        "nonuniform (for example Gaussian) radial distributions; multi-species mixtures; particle creation or "
        "loss; detector efficiency, secondary emission and instrument response. These exclusions describe the "
        "approximation. They do not justify assigning zero loss or uniform density to experimental data "
        "without validation."
    )
    st.markdown(
        "**Numerical notes.** Calculations use 64-bit floating point in SI units. θ ≥ 90° is rejected, and θ ≥ 85° "
        "triggers a sensitivity note (this is not a physical validity boundary). Values that overflow, underflow "
        "or round to zero are rejected with a message naming the quantity, and are never shown as exact results."
    )
