"""Streamlit interface for the Ion Plume Geometric Expansion Model.

Run with:  python -m streamlit run app.py --server.address 127.0.0.1
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import streamlit as st

from formatting import format_area, format_length, format_percent, format_sig
from plume_model import PlumeModelError, PlumeRangeError, calculate_plume, plume_profile
from visualization import geometry_figure, profile_figure
from units import (
    AREA_UNITS,
    LENGTH_UNITS,
    InputParseError,
    area_to_m2,
    area_unit_for_length,
    length_to_m,
    length_unit_for_area,
    m2_to_area,
    m_to_length,
    parse_real,
    validate_inputs,
)

TITLE = "Ion Plume Geometric Expansion Model"
SUBTITLE = "Estimate particle-density reduction from geometric expansion under constant axial velocity."
MODEL_SENTENCE = (
    "Model: the particle density is uniform across each cross-section, the tracked particle flow "
    "is conserved, and the mean axial velocity is equal at both planes, so n₁A₁ = n₂A₂."
)
EXAMPLE = {
    "n1_text": "1.0e20",
    "distance_text": "0.10",
    "distance_unit": "m",
    "theta_text": "10",
    "geometry_mode": "radius",
    "geometry_text": "1.0",
    "radius_unit": "mm",
}
STALE_RTOL = 1e-12

ss = st.session_state
_DEFAULTS = {
    "n1_text": "",
    "distance_text": "",
    "distance_unit": "m",
    "theta_text": "",
    "geometry_mode": "radius",
    "geometry_text": "",
    "radius_unit": "mm",
    "area_unit": "mm²",
    "field_errors": {},
    "range_error": None,
    "mode_note": None,
    "result": None,
    "result_inputs": None,
    "result_is_example": False,
    "density_scale": "Logarithmic",
    "diagram_view": "Auto (to scale when readable)",
}
for _key, _value in _DEFAULTS.items():
    if _key not in ss:
        ss[_key] = _value


def _geometry_unit() -> str:
    return ss.radius_unit if ss.geometry_mode == "radius" else ss.area_unit


def _current_inputs():
    return validate_inputs(
        ss.n1_text, ss.distance_text, ss.distance_unit, ss.theta_text,
        ss.geometry_mode, ss.geometry_text, _geometry_unit(),
    )


def _on_mode_change() -> None:
    """Convert the single source-geometry value so the physical size is preserved."""
    ss.mode_note = None
    text = str(ss.geometry_text).strip()
    if not text:
        return
    to_area = ss.geometry_mode == "area"  # the radio already holds the new mode
    # Reuse the exact radius behind a previously converted, unedited value so that
    # repeated mode switches do not accumulate display rounding.
    carried = ss.get("converted_geometry")
    try:
        if carried and carried[:2] == (text, ss.radius_unit if to_area else ss.area_unit):
            r1_m = carried[2]
        else:
            value = parse_real(text)
            if value <= 0.0:
                raise InputParseError("nonpositive")
            if to_area:
                r1_m = length_to_m(value, ss.radius_unit)
            else:
                r1_m = math.sqrt(area_to_m2(value, ss.area_unit) / math.pi)
        if to_area:
            unit = area_unit_for_length(ss.radius_unit)
            converted = m2_to_area(math.pi * r1_m * r1_m, unit)
            ss.area_unit = unit
        else:
            unit = length_unit_for_area(ss.area_unit)
            converted = m_to_length(r1_m, unit)
            ss.radius_unit = unit
        if not (math.isfinite(converted) and converted > 0.0 and r1_m > 0.0):
            raise InputParseError("nonrepresentable")
        ss.geometry_text = f"{converted:.15g}"
        ss.converted_geometry = (ss.geometry_text, unit, r1_m)
    except (InputParseError, PlumeModelError):
        ss.geometry_text = ""
        ss.mode_note = "The previous source value was not a valid positive number, so it was cleared."


def _calculate() -> None:
    normalized, errors = _current_inputs()
    ss.field_errors = errors
    ss.range_error = None
    if normalized is None:
        return
    try:
        result = calculate_plume(normalized.n1, normalized.r1_m, normalized.distance_m, normalized.theta_rad)
        profile = plume_profile(result)
    except PlumeRangeError as exc:
        ss.range_error = f"Numerical range error ({exc.quantity}): {exc}"
        return
    except PlumeModelError as exc:
        ss.range_error = str(exc)
        return
    ss.result = result
    ss.profile = profile
    ss.result_inputs = normalized
    ss.result_is_example = False


def _load_example() -> None:
    for key, value in EXAMPLE.items():
        ss[key] = value
    ss.mode_note = None
    _calculate()
    ss.result_is_example = ss.result is not None


def _reset() -> None:
    ss.geometry_text = ""
    ss.result = None
    ss.result_inputs = None
    ss.result_is_example = False
    ss.field_errors = {}
    ss.range_error = None
    ss.mode_note = None


def _is_stale() -> bool:
    """True when the current inputs no longer describe the displayed result."""
    normalized, _ = _current_inputs()
    if normalized is None or ss.result_inputs is None:
        return True
    old = ss.result_inputs
    return not all(
        math.isclose(getattr(normalized, f), getattr(old, f), rel_tol=STALE_RTOL, abs_tol=0.0)
        for f in ("n1", "distance_m", "theta_rad", "r1_m")
    )


def _field_error(field: str) -> None:
    message = ss.field_errors.get(field)
    if message:
        st.error(message, icon="⚠️")


# ---------------------------------------------------------------- layout
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)
st.markdown(f"**{SUBTITLE}**")
st.info(MODEL_SENTENCE)

inputs_col, results_col = st.columns([2, 3], gap="large")

with inputs_col:
    st.subheader("Inputs")

    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    c1.text_input("Initial density n₁ at P₁", key="n1_text", placeholder="e.g. 1e20",
                  help="Uniform number density at the reference plane. Scientific notation is accepted.")
    c2.markdown("particles/m³")
    _field_error("n1")

    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    c1.text_input("Axial distance l from P₁ to P₂", key="distance_text", placeholder="e.g. 0.10")
    c2.selectbox("Distance unit", LENGTH_UNITS, key="distance_unit")
    _field_error("distance")

    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    c1.text_input("Plume half-angle θ", key="theta_text", placeholder="e.g. 10")
    c2.markdown("degrees (half-angle)")
    _field_error("theta")

    st.caption(
        "**Reference plane:** P₁ is the center of the finite source cross-section A₁ at x = 0; "
        "P₂ is the center of the downstream plane at x = l, measured along the plume axis (not the slant). "
        "**Half-angle:** θ is the angle between the plume boundary and a line parallel to the axis. "
        "If you have a full opening angle, divide it by two before entry."
    )

    st.radio("Source geometry (enter one)", ["radius", "area"], key="geometry_mode", horizontal=True,
             format_func=lambda m: "Radius r₁" if m == "radius" else "Area A₁", on_change=_on_mode_change)
    radius_mode = ss.geometry_mode == "radius"
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    c1.text_input("Source radius r₁ at P₁" if radius_mode else "Source area A₁ at P₁ (full plume cross-section)",
                  key="geometry_text", placeholder="required; no default is assumed")
    if radius_mode:
        c2.selectbox("Radius unit", LENGTH_UNITS, key="radius_unit")
    else:
        c2.selectbox("Area unit", AREA_UNITS, key="area_unit")
    _field_error("geometry")
    if ss.mode_note:
        st.warning(ss.mode_note)
    st.caption(
        "The source size must be known: n₁, l and θ alone do not determine n₂. "
        "An assumed source radius changes the normalization of the prediction."
    )

    geometry_missing = not str(ss.geometry_text).strip()
    b1, b2, b3 = st.columns(3)
    b1.button("Calculate", key="calculate", type="primary", on_click=_calculate, disabled=geometry_missing,
              width="stretch")
    b2.button("Load example", key="load_example", on_click=_load_example, width="stretch")
    b3.button("Reset", key="reset", on_click=_reset, width="stretch")
    if geometry_missing:
        st.caption("Enter a source radius or area to enable Calculate.")

with results_col:
    st.subheader("Results")
    if ss.range_error:
        st.error(ss.range_error)

    result = ss.result
    if result is None:
        st.info("No result yet. Enter n₁, l, θ and a known source radius or area, then press Calculate.")
    else:
        stale = _is_stale()
        illustrative = ss.result_is_example and not stale
        if stale:
            st.warning(
                "OUT OF DATE: the inputs have changed since this result was calculated. The values and plots "
                "below belong to the previous inputs. Press Calculate to update them.",
                icon="⚠️",
            )
        if illustrative:
            st.info("Illustrative example values (PRD benchmark case), not a measured lab configuration.")

        prefix = "[Out of date] " if stale else ""
        st.metric(f"{prefix}Downstream density n₂ at P₂ (model estimate)",
                  f"{format_sig(result.n2)} particles/m³")
        st.caption(
            "n₂ is the uniform cross-sectional model estimate, not a measured density or a detector count rate."
        )
        remaining = format_percent(result.remaining_fraction, result.reduction_fraction)
        reduction = format_percent(result.reduction_fraction, result.remaining_fraction)
        st.markdown(
            "| Quantity | Value |\n|---|---|\n"
            f"| Source radius r₁ | {format_length(result.r1_m)} |\n"
            f"| Downstream radius r₂ | {format_length(result.r2_m)} |\n"
            f"| Source area A₁ | {format_area(result.area1_m2)} |\n"
            f"| Downstream plume area A₂ | {format_area(result.area2_m2)} |\n"
            f"| Area expansion E = A₂/A₁ | {format_sig(result.expansion)} |\n"
            f"| Remaining fraction F = A₁/A₂ = n₂/n₁ | {format_sig(result.remaining_fraction)} |\n"
        )
        p1, p2 = st.columns(2)
        p1.metric("Remaining density (% of n₁)", remaining)
        p2.metric("Reduction (% decrease from n₁)", reduction)
        if result.high_angle_sensitivity:
            st.warning(
                "θ ≥ 85°: tan θ grows rapidly, so the result is highly sensitive to small changes in the "
                "half-angle. This is a numerical-sensitivity note, not a physical validity boundary."
            )

        with st.expander("Calculation details (10 significant figures)"):
            d = lambda v: format_sig(v, 10)  # noqa: E731
            st.markdown(
                "| Quantity | Value (SI) |\n|---|---|\n"
                f"| n₁ | {d(result.n1)} particles/m³ |\n"
                f"| r₁ | {d(result.r1_m)} m |\n"
                f"| l | {d(result.distance_m)} m |\n"
                f"| θ | {d(result.theta_deg)}° = {d(result.theta_rad)} rad |\n"
                f"| Δr = l tan θ | {d(result.delta_r_m)} m |\n"
                f"| r₂ = r₁ + Δr | {d(result.r2_m)} m |\n"
                f"| A₁ = πr₁² | {d(result.area1_m2)} m² |\n"
                f"| A₂ = πr₂² | {d(result.area2_m2)} m² |\n"
                f"| F = (r₁/r₂)² | {d(result.remaining_fraction)} |\n"
                f"| E = (r₂/r₁)² | {d(result.expansion)} |\n"
                f"| n₂ = n₁F | {d(result.n2)} particles/m³ |\n"
                f"| Remaining 100F | {d(result.remaining_percent)} % |\n"
                f"| Reduction 100(1 − F) | {d(result.reduction_percent)} % |\n"
            )

        st.subheader(f"{prefix}Plume geometry")
        st.radio("Diagram view", ["Auto (to scale when readable)", "Schematic"], key="diagram_view",
                 horizontal=True)
        fig, schematic = geometry_figure(
            result, view="schematic" if ss.diagram_view == "Schematic" else "auto", illustrative=illustrative
        )
        st.pyplot(fig)
        plt.close(fig)
        if schematic:
            st.caption("Schematic, not to scale: proportions are adjusted for readability; the labels show the "
                       "true values.")

        st.subheader(f"{prefix}Density vs axial distance")
        st.radio("Density axis scale", ["Logarithmic", "Linear"], key="density_scale", horizontal=True)
        x_m, density = ss.profile
        fig = profile_figure(result, x_m, density, log_y=ss.density_scale == "Logarithmic",
                             illustrative=illustrative)
        st.pyplot(fig)
        plt.close(fig)
        if result.distance_m == 0.0:
            st.caption("l = 0: P₁ and P₂ coincide, so the profile is a single point with n₂ = n₁.")

with st.expander("How the model works"):
    st.markdown(
        "**Idea.** Radial expansion spreads a conserved particle flow over a larger area. At constant mean axial "
        "velocity, n₁A₁ = n₂A₂. The formula gives a uniform density estimate; it does not provide an "
        "experimental radial profile."
    )
    st.markdown("**Governing law** (steady flow of a conserved, tracked population, bounded by streamlines):")
    st.latex(r"\dot N = \int_A n\,\mathbf{v}\cdot\hat{\mathbf{x}}\,dA \;\rightarrow\; n A v_x")
    st.latex(r"\rho = nm,\qquad \dot m = \rho A v_x = n m A v_x")
    st.latex(r"n_1 A_1 v_{x,1} = n_2 A_2 v_{x,2},\qquad v_{x,1} = v_{x,2} > 0")
    st.markdown(
        "The velocity in the planar flux is its **axial component**. Equal total particle speed alone is not "
        "enough if the distribution of velocity directions changes. For one species of unchanged mass m, m "
        "cancels. Particle-number conservation is assumed explicitly; total mass conservation alone would not "
        "conserve the number of ions if the population fragmented, recombined or changed species."
    )
    st.markdown("**Geometry** (circular conical frustum with a finite source radius r₁):")
    st.latex(r"r_1 = \sqrt{A_1/\pi},\qquad r_2 = r_1 + l\tan\theta,\qquad A_2 = \pi r_2^2")
    st.latex(r"n_2 = n_1\frac{A_1}{A_2} = n_1\left(\frac{r_1}{r_1 + l\tan\theta}\right)^2")
    st.latex(r"F = \frac{A_1}{A_2} = \frac{n_2}{n_1},\qquad E = \frac{A_2}{A_1} = \frac{1}{F}")
    st.latex(r"r(x) = r_1 + x\tan\theta,\qquad n(x) = n_1\left(\frac{r_1}{r_1 + x\tan\theta}\right)^2")
    st.markdown(
        "- A₂ is the full plume footprint at the downstream plane, **not** the instrument aperture. Do not "
        "substitute the detector opening into the dilution equation.\n"
        "- The common height h of P₁ and P₂ only sets placement; it is not a calculation input.\n"
        "- θ is the **half**-angle; divide a full opening angle by two. l is axial, not a slant distance.\n"
        "- No particle mass, charge, energy or absolute velocity is needed, because the same mass and equal "
        "nonzero axial velocity cancel.\n"
        "- The density uses the radius ratio (r₁/r₂)², and 1 − F is evaluated as (Δr/r₂)(1 + r₁/r₂) to "
        "avoid rounding error when F is close to 1."
    )
    st.markdown("**Far-field limit** (explanation only; the app always uses the full frustum equation):")
    st.latex(r"l\tan\theta \gg r_1 \;\Rightarrow\; n_2 \approx \frac{n_1 r_1^2}{l^2\tan^2\theta}")
    st.markdown(
        "This inverse-square behavior applies only for θ > 0 and l tan θ ≫ r₁. A point-source model with finite "
        "particle flow would need an independent flow normalization or a finite reference measurement. With "
        "fixed n₁, shrinking A₁ toward zero also shrinks the particle flow."
    )
    st.markdown(
        "**Assumptions**\n"
        "- Emission is steady; tracked particles do not accumulate between the planes.\n"
        "- The plume is axisymmetric, with circular cross-sections, a constant half-angle and straight frustum "
        "boundaries.\n"
        "- Number density is uniform across each cross-section; n₁ and n₂ are those uniform model values.\n"
        "- The tracked particle population is conserved: no sources, losses, neutralization, deposition or "
        "escape across the modeled boundary.\n"
        "- Mean axial velocity is equal at both planes; acceleration and changes in the angular velocity "
        "distribution are excluded.\n"
        "- P₁ is a finite reference cross-section, even if it looks like a point in a schematic."
    )
    st.markdown(
        "**What n₂ represents.** In this uniform model the value at the centerline point P₂ equals the "
        "cross-sectional average. Area growth alone does not determine the local density of a real, nonuniform "
        "plume. Predicting an instrument signal requires its sampling geometry, acceptance, efficiency, particle "
        "properties and a radial density or flux distribution. n₂ is a model estimate, not a measured density or "
        "detector count rate, and geometric calculations do not establish the instrument's measured response."
    )
    st.markdown(
        "**Excluded physics:** electric-field forces; Coulomb interactions and space charge; collisions; "
        "acceleration; evolution of velocity directions; nonuniform radial distributions; particle creation or "
        "loss; instrument response. These exclusions describe the approximation. They do not justify assigning "
        "zero loss or uniform density to experimental data without validation."
    )
    st.markdown(
        "**Numerical notes.** Calculations use 64-bit floating point in SI units. θ ≥ 90° is rejected, and θ ≥ 85° "
        "triggers a sensitivity note (this is not a physical validity boundary). Values that overflow, underflow "
        "or round to zero are rejected with a message naming the quantity, and are never shown as exact results."
    )
