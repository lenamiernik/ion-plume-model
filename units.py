"""Explicit unit conversions and text-input validation (no UI dependencies).

Length factors to metres: m 1, cm 1e-2, mm 1e-3, um 1e-6.
Area factors to square metres are the squares: 1, 1e-4, 1e-6, 1e-12.
Angles: theta_rad = theta_deg * pi / 180.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from plume_model import PlumeRangeError

LENGTH_FACTORS_M = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "µm": 1e-6}
AREA_FACTORS_M2 = {"m²": 1.0, "cm²": 1e-4, "mm²": 1e-6, "µm²": 1e-12}
LENGTH_UNITS = tuple(LENGTH_FACTORS_M)
AREA_UNITS = tuple(AREA_FACTORS_M2)

_LENGTH_ALIASES = {"um": "µm", "μm": "µm", "micron": "µm"}  # "μ" is Greek mu, "µ" micro sign
_AREA_ALIASES = {
    "m2": "m²", "cm2": "cm²", "mm2": "mm²", "um2": "µm²",
    "um²": "µm²", "μm²": "µm²", "μm2": "µm²", "µm2": "µm²",
}


class InputParseError(ValueError):
    """Raised when text cannot be read as a finite real number."""


def _length_unit(unit: str) -> str:
    unit = _LENGTH_ALIASES.get(unit, unit)
    if unit not in LENGTH_FACTORS_M:
        raise ValueError(f"Unsupported length unit {unit!r}; use one of {', '.join(LENGTH_UNITS)}.")
    return unit


def _area_unit(unit: str) -> str:
    unit = _AREA_ALIASES.get(unit, unit)
    if unit not in AREA_FACTORS_M2:
        raise ValueError(f"Unsupported area unit {unit!r}; use one of {', '.join(AREA_UNITS)}.")
    return unit


def _checked(value: float, converted: float, what: str, quantity: str) -> float:
    if not math.isfinite(converted):
        raise PlumeRangeError(
            f"{what} overflows 64-bit floating point after unit conversion; enter a less extreme value.",
            quantity,
        )
    if value != 0.0 and converted == 0.0:
        raise PlumeRangeError(
            f"{what} rounds to zero after unit conversion; enter a less extreme value.", quantity
        )
    return converted


def length_to_m(value: float, unit: str) -> float:
    """Convert a length in `unit` (m, cm, mm, µm) to metres."""
    return _checked(value, value * LENGTH_FACTORS_M[_length_unit(unit)], "Length", "length_m")


def m_to_length(value_m: float, unit: str) -> float:
    """Convert a length in metres to `unit` (m, cm, mm, µm)."""
    return _checked(value_m, value_m / LENGTH_FACTORS_M[_length_unit(unit)], "Length", "length")


def area_to_m2(value: float, unit: str) -> float:
    """Convert an area in `unit` (m², cm², mm², µm²) to square metres."""
    return _checked(value, value * AREA_FACTORS_M2[_area_unit(unit)], "Area", "area_m2")


def m2_to_area(value_m2: float, unit: str) -> float:
    """Convert an area in square metres to `unit` (m², cm², mm², µm²)."""
    return _checked(value_m2, value_m2 / AREA_FACTORS_M2[_area_unit(unit)], "Area", "area")


def area_unit_for_length(unit: str) -> str:
    """Square unit matching a length unit, e.g. 'mm' -> 'mm²'."""
    return _length_unit(unit) + "²"


def length_unit_for_area(unit: str) -> str:
    """Length unit matching a square unit, e.g. 'mm²' -> 'mm'."""
    return _area_unit(unit)[:-1]


def deg_to_rad(theta_deg: float) -> float:
    """Degrees to radians: theta_deg * pi / 180."""
    return theta_deg * math.pi / 180.0


def rad_to_deg(theta_rad: float) -> float:
    """Radians to degrees: theta_rad * 180 / pi."""
    return theta_rad * 180.0 / math.pi


def parse_real(text) -> float:
    """Parse text such as '1e20', ' 1.0E+20 ' or '0.10' into a finite float.

    Raises InputParseError for blank, nonnumeric, NaN or infinite input.
    """
    if text is None or not str(text).strip():
        raise InputParseError("blank")
    cleaned = str(text).strip().replace("−", "-")  # accept a typographic minus sign
    try:
        value = float(cleaned)
    except ValueError:
        raise InputParseError("nonnumeric") from None
    if not math.isfinite(value):
        raise InputParseError("nonfinite")
    return value


@dataclass(frozen=True)
class NormalizedInputs:
    """Validated calculation inputs in SI units (theta in radians)."""

    n1: float
    distance_m: float
    theta_rad: float
    r1_m: float


def _parse_field(text, label: str, example: str, errors: dict, field: str) -> float | None:
    try:
        return parse_real(text)
    except InputParseError as exc:
        reason = str(exc)
        if reason == "blank":
            errors[field] = f"Enter {label}."
        elif reason == "nonfinite":
            errors[field] = f"{label[0].upper()}{label[1:]} must be a finite number (not NaN or infinity)."
        else:
            errors[field] = (
                f"{label[0].upper()}{label[1:]} must be a number, for example {example} "
                "(scientific notation is accepted)."
            )
        return None


def validate_inputs(
    n1_text,
    distance_text,
    distance_unit: str,
    theta_text,
    geometry_mode: str,
    geometry_text,
    geometry_unit: str,
) -> tuple[NormalizedInputs | None, dict[str, str]]:
    """Validate all text inputs together and convert them to SI.

    geometry_mode is "radius" (geometry_unit a length unit) or "area"
    (geometry_unit an area unit; r1 = sqrt(A1/pi)).

    Returns (NormalizedInputs, {}) on success, otherwise (None, errors) where
    errors maps "n1", "distance", "theta" and/or "geometry" to a message.
    Never raises for user input.
    """
    errors: dict[str, str] = {}
    radius_mode = geometry_mode == "radius"
    geom_word = "radius" if radius_mode else "area"

    n1 = _parse_field(n1_text, "the initial density n₁", "1e20", errors, "n1")
    if n1 is not None and n1 <= 0.0:
        errors["n1"] = "Initial density n₁ must be greater than zero."

    distance_m = None
    distance = _parse_field(distance_text, "the axial distance l", "0.10", errors, "distance")
    if distance is not None:
        if distance < 0.0:
            errors["distance"] = "Axial distance must be zero or greater."
        else:
            try:
                distance_m = length_to_m(distance, distance_unit)
            except PlumeRangeError:
                errors["distance"] = (
                    f"Axial distance {distance:g} {distance_unit} cannot be represented in metres "
                    "(it rounds to zero or overflows); enter a less extreme value."
                )

    theta_rad = None
    theta_deg = _parse_field(theta_text, "the half-angle θ", "10", errors, "theta")
    if theta_deg is not None:
        if not (0.0 <= theta_deg < 90.0):
            errors["theta"] = "Half-angle must be at least 0° and less than 90°."
        else:
            theta_rad = deg_to_rad(theta_deg)

    r1_m = None
    if geometry_text is None or not str(geometry_text).strip():
        errors["geometry"] = f"Enter a positive source {geom_word}."
    else:
        g = _parse_field(geometry_text, f"the source {geom_word}", "1.0", errors, "geometry")
        if g is not None:
            if g <= 0.0:
                errors["geometry"] = f"Enter a positive source {geom_word}."
            else:
                try:
                    if radius_mode:
                        r1_m = length_to_m(g, geometry_unit)
                    else:
                        area_m2 = area_to_m2(g, geometry_unit)
                        r1_m = math.sqrt(area_m2 / math.pi)
                        if r1_m == 0.0 or not math.isfinite(r1_m):
                            raise PlumeRangeError("Source radius from area is not representable.", "r1")
                except PlumeRangeError:
                    errors["geometry"] = (
                        f"Source {geom_word} {g:g} {geometry_unit} cannot be represented in SI units "
                        "(it rounds to zero or overflows); enter a less extreme value."
                    )

    if errors:
        return None, errors
    return NormalizedInputs(n1=n1, distance_m=distance_m, theta_rad=theta_rad, r1_m=r1_m), {}
