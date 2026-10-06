"""Display strings for model values (presentation only; no physics, no UI imports)."""

from __future__ import annotations

import math

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

# Readable engineering units: (threshold in SI, unit label, factor to SI).
_LENGTH_DISPLAY = [(1.0, "m", 1.0), (1e-2, "cm", 1e-2), (1e-3, "mm", 1e-3), (0.0, "µm", 1e-6)]
_AREA_DISPLAY = [(1.0, "m²", 1.0), (1e-4, "cm²", 1e-4), (1e-6, "mm²", 1e-6), (0.0, "µm²", 1e-12)]
_CURRENT_DISPLAY = [(1.0, "A", 1.0), (1e-3, "mA", 1e-3), (1e-6, "µA", 1e-6), (1e-9, "nA", 1e-9), (0.0, "pA", 1e-12)]
_VOLTAGE_DISPLAY = [(1e3, "kV", 1e3), (0.0, "V", 1.0)]
_VELOCITY_DISPLAY = [(1e3, "km/s", 1e3), (0.0, "m/s", 1.0)]

# A percentage near 100 carries about 14 fractional digits in float64; beyond this many
# decimals the fixed-point form is replaced by an explicit "100 % − gap" form.
_MAX_PERCENT_DECIMALS = 12


def format_sig(value: float, sig: int = 3) -> str:
    """Format with `sig` significant figures.

    Uses fixed notation for 1e-3 <= |value| < 1e4 and "m × 10ⁿ" otherwise.
    """
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    if value == 0.0:
        return "0"
    mantissa, exp_text = f"{value:.{sig - 1}e}".split("e")
    exponent = int(exp_text)
    if -3 <= exponent < 4:
        return f"{value:.{max(sig - 1 - exponent, 0)}f}"
    return f"{mantissa} × 10{str(exponent).translate(_SUPERSCRIPT)}"


def _with_display_unit(value_si: float, table, base_unit: str, sig: int) -> str:
    base = f"{format_sig(value_si, sig)} {base_unit}"
    for threshold, unit, factor in table:
        if abs(value_si) >= threshold:
            break
    if unit == base_unit:
        return base
    return f"{base} ({format_sig(value_si / factor, sig)} {unit})"


def format_length(value_m: float, sig: int = 3) -> str:
    """Length in m plus a readable engineering unit, e.g. '0.0186 m (1.86 cm)'."""
    return _with_display_unit(value_m, _LENGTH_DISPLAY, "m", sig)


def format_area(value_m2: float, sig: int = 3) -> str:
    """Area in m² plus a readable unit, e.g. '3.14 × 10⁻⁶ m² (3.14 mm²)'."""
    return _with_display_unit(value_m2, _AREA_DISPLAY, "m²", sig)


def format_current(value_a: float, sig: int = 3) -> str:
    """Current in A plus a readable unit, e.g. '2.88 × 10⁻⁹ A (2.88 nA)'."""
    return _with_display_unit(value_a, _CURRENT_DISPLAY, "A", sig)


def format_current_eng(value_a: float, sig: int = 3) -> str:
    """Current in the most readable unit only, e.g. '2.88 nA'."""
    for threshold, unit, factor in _CURRENT_DISPLAY:
        if abs(value_a) >= threshold:
            return f"{format_sig(value_a / factor, sig)} {unit}"
    return f"{format_sig(value_a, sig)} A"


def format_voltage(value_v: float, sig: int = 3) -> str:
    """Voltage in V plus kV when large, e.g. '1000 V (1.00 kV)'."""
    return _with_display_unit(value_v, _VOLTAGE_DISPLAY, "V", sig)


def format_velocity(value_m_s: float, sig: int = 3) -> str:
    """Velocity in m/s plus km/s when large, e.g. '4.39 × 10⁴ m/s (43.9 km/s)'."""
    return _with_display_unit(value_m_s, _VELOCITY_DISPLAY, "m/s", sig)


def format_percent(fraction: float, complement: float, sig: int = 3) -> str:
    """Format 100*fraction as a percentage that is never falsely 0 % or 100 %.

    `complement` must be the accurately computed 1 - fraction. Small values use
    significant figures (scientific notation when fixed decimals would show 0).
    Values near 100 % get enough decimals to show the gap from 100 %, and an
    explicit "100 % − gap" form when float64 decimals cannot show it.
    """
    if fraction <= 0.5:
        return f"{format_sig(100.0 * fraction, sig)} %"
    gap = 100.0 * complement
    if gap == 0.0:
        return "100 %"
    decimals = max(sig - 2, -math.floor(math.log10(gap)))
    if decimals > _MAX_PERCENT_DECIMALS:
        return f"100 % − {format_sig(gap, sig)} %"
    return f"{100.0 - gap:.{decimals}f} %"
