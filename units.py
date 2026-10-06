"""Explicit unit conversions and text-input validation (no UI dependencies).

Length factors to metres: m 1, cm 1e-2, mm 1e-3, um 1e-6.
Area factors to square metres are the squares: 1, 1e-4, 1e-6, 1e-12.
Current factors to amperes: A 1, mA 1e-3, uA 1e-6, nA 1e-9.
Voltage factors to volts: V 1, kV 1e3.
Mass: m_kg = m_amu * 1.66053906660e-27 (constants.AMU_KG).
Angles: theta_rad = theta_deg * pi / 180.

The calculation core only ever receives the SI values produced here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from constants import AMU_KG, MODELED_CHARGE_C
from plume_model import MIN_NORMAL, PlumeRangeError

LENGTH_FACTORS_M = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "µm": 1e-6}
AREA_FACTORS_M2 = {"m²": 1.0, "cm²": 1e-4, "mm²": 1e-6, "µm²": 1e-12}
CURRENT_FACTORS_A = {"A": 1.0, "mA": 1e-3, "µA": 1e-6, "nA": 1e-9}
VOLTAGE_FACTORS_V = {"V": 1.0, "kV": 1e3}
LENGTH_UNITS = tuple(LENGTH_FACTORS_M)
AREA_UNITS = tuple(AREA_FACTORS_M2)
CURRENT_UNITS = tuple(CURRENT_FACTORS_A)
VOLTAGE_UNITS = tuple(VOLTAGE_FACTORS_V)

# "μ" (U+03BC Greek mu) and "u" are accepted as the micro sign "µ" (U+00B5).
_LENGTH_ALIASES = {"um": "µm", "μm": "µm", "micron": "µm"}
_AREA_ALIASES = {
    "m2": "m²", "cm2": "cm²", "mm2": "mm²", "um2": "µm²",
    "um²": "µm²", "μm²": "µm²", "μm2": "µm²", "µm2": "µm²",
}
# Explicit aliases only: case-folding would silently read "MA" (megaampere) as mA.
_CURRENT_ALIASES = {"uA": "µA", "μA": "µA", "amp": "A", "amps": "A"}
_VOLTAGE_ALIASES = {"kv": "kV", "volt": "V", "volts": "V"}


class InputParseError(ValueError):
    """Raised when text cannot be read as a finite real number."""


def _unit(unit: str, factors: dict, aliases: dict, kind: str) -> str:
    unit = aliases.get(unit, unit)
    if unit not in factors:
        raise ValueError(f"Unsupported {kind} unit {unit!r}; use one of {', '.join(factors)}.")
    return unit


def _length_unit(unit: str) -> str:
    return _unit(unit, LENGTH_FACTORS_M, _LENGTH_ALIASES, "length")


def _area_unit(unit: str) -> str:
    return _unit(unit, AREA_FACTORS_M2, _AREA_ALIASES, "area")


def _checked(value: float, converted: float, what: str, quantity: str, require_normal: bool = False) -> float:
    if not math.isfinite(converted):
        raise PlumeRangeError(
            f"{what} overflows 64-bit floating point after unit conversion; enter a less extreme value.",
            quantity,
        )
    if value != 0.0 and (converted == 0.0 or (require_normal and abs(converted) < MIN_NORMAL)):
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


def current_to_a(value: float, unit: str) -> float:
    """Convert a current in `unit` (A, mA, µA, nA) to amperes; positive results must be normal."""
    factor = CURRENT_FACTORS_A[_unit(unit, CURRENT_FACTORS_A, _CURRENT_ALIASES, "current")]
    return _checked(value, value * factor, "Current", "I0", require_normal=True)


def voltage_to_v(value: float, unit: str) -> float:
    """Convert a voltage in `unit` (V, kV) to volts; positive results must be normal."""
    factor = VOLTAGE_FACTORS_V[_unit(unit, VOLTAGE_FACTORS_V, _VOLTAGE_ALIASES, "voltage")]
    return _checked(value, value * factor, "Voltage", "V0", require_normal=True)


def amu_to_kg(mass_amu: float) -> float:
    """Convert an individual particle mass in amu to kg with the fixed AMU_KG factor."""
    return _checked(mass_amu, mass_amu * AMU_KG, "Mass", "m_i", require_normal=True)


def kg_to_amu(mass_kg: float) -> float:
    """Convert a particle mass in kg to amu with the fixed AMU_KG factor."""
    return _checked(mass_kg, mass_kg / AMU_KG, "Mass", "m_i")


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
    """Validated direct-density inputs in SI units (theta in radians)."""

    n0: float
    distance_m: float
    theta_rad: float
    r0_m: float


@dataclass(frozen=True)
class CurrentInputs:
    """Validated current-derived inputs in SI units (theta in radians).

    current_a and voltage_v are magnitudes; current_sign_negative and
    voltage_sign_negative record that a signed entry was given (display only;
    polarity is not configured). mass_amu is kept for display; calculations use
    mass_kg. charge_c is the fixed modeled charge.
    """

    current_a: float
    voltage_v: float
    mass_amu: float
    mass_kg: float
    r0_m: float
    distance_m: float
    theta_rad: float
    collector_radius_m: float
    charge_c: float
    current_sign_negative: bool = False
    voltage_sign_negative: bool = False

    # Physical quantities that define a result; sign flags are presentation only.
    PHYSICAL_FIELDS = (
        "current_a", "voltage_v", "mass_kg", "r0_m", "distance_m", "theta_rad", "collector_radius_m", "charge_c",
    )


DIRECT_PHYSICAL_FIELDS = ("n0", "distance_m", "theta_rad", "r0_m")


def _cap(label: str) -> str:
    return label[0].upper() + label[1:]


def _parse_field(text, label: str, example: str, errors: dict, field: str) -> float | None:
    try:
        return parse_real(text)
    except InputParseError as exc:
        reason = str(exc)
        if reason == "blank":
            errors[field] = f"Enter {label}."
        elif reason == "nonfinite":
            errors[field] = f"{_cap(label)} must be a finite number (not NaN or infinity)."
        else:
            errors[field] = (
                f"{_cap(label)} must be a number, for example {example} (scientific notation is accepted)."
            )
        return None


def _validate_distance(text, unit: str, errors: dict) -> float | None:
    distance = _parse_field(text, "the axial distance l", "0.10", errors, "distance")
    if distance is None:
        return None
    if distance < 0.0:
        errors["distance"] = "Axial distance must be zero or greater."
        return None
    try:
        return length_to_m(distance, unit)
    except PlumeRangeError:
        errors["distance"] = (
            f"Axial distance {distance:g} {unit} cannot be represented in metres "
            "(it rounds to zero or overflows); enter a less extreme value."
        )
        return None


def _validate_theta(text, errors: dict) -> float | None:
    theta_deg = _parse_field(text, "the half-angle θ", "10", errors, "theta")
    if theta_deg is None:
        return None
    if not (0.0 <= theta_deg < 90.0):
        errors["theta"] = "Half-angle must be at least 0° and less than 90°."
        return None
    return deg_to_rad(theta_deg)


def _validate_circle(text, mode: str, unit: str, errors: dict, field: str, noun: str) -> float | None:
    """Validate one active radius-or-area field and return the radius in metres."""
    radius_mode = mode == "radius"
    word = f"{noun} {'radius' if radius_mode else 'area'}"
    if text is None or not str(text).strip():
        errors[field] = f"Enter a positive {word}."
        return None
    value = _parse_field(text, f"the {word}", "1.0", errors, field)
    if value is None:
        return None
    if value <= 0.0:
        errors[field] = f"Enter a positive {word}."
        return None
    try:
        if radius_mode:
            return length_to_m(value, unit)
        radius_m = math.sqrt(area_to_m2(value, unit) / math.pi)
        if radius_m == 0.0 or not math.isfinite(radius_m):
            raise PlumeRangeError("Radius from area is not representable.", field)
        return radius_m
    except PlumeRangeError:
        errors[field] = (
            f"{_cap(word)} {value:g} {unit} cannot be represented in SI units "
            "(it rounds to zero or overflows); enter a less extreme value."
        )
        return None


def _validate_signed_magnitude(text, unit: str, convert, label: str, example: str, zero_message: str,
                               errors: dict, field: str) -> tuple[float | None, bool]:
    """Parse a signed current/voltage entry and return (SI magnitude, was_negative)."""
    value = _parse_field(text, label, example, errors, field)
    if value is None:
        return None, False
    if value == 0.0:
        errors[field] = zero_message
        return None, False
    try:
        return convert(abs(value), unit), value < 0.0
    except PlumeRangeError:
        errors[field] = (
            f"{_cap(label)} {value:g} {unit} cannot be represented in SI units "
            "(it rounds to zero or overflows); enter a less extreme value."
        )
        return None, False


def validate_direct_inputs(
    n0_text,
    distance_text,
    distance_unit: str,
    theta_text,
    geometry_mode: str,
    geometry_text,
    geometry_unit: str,
) -> tuple[NormalizedInputs | None, dict[str, str]]:
    """Validate all direct-density text inputs together and convert them to SI.

    geometry_mode is "radius" (geometry_unit a length unit) or "area"
    (geometry_unit an area unit; r0 = sqrt(A0/pi)).

    Returns (NormalizedInputs, {}) on success, otherwise (None, errors) where
    errors maps "n0", "distance", "theta" and/or "geometry" to a message.
    Never raises for user input.
    """
    errors: dict[str, str] = {}
    n0 = _parse_field(n0_text, "the source density n₀", "1e20", errors, "n0")
    if n0 is not None and n0 <= 0.0:
        errors["n0"] = "Source density n₀ must be greater than zero."
    distance_m = _validate_distance(distance_text, distance_unit, errors)
    theta_rad = _validate_theta(theta_text, errors)
    r0_m = _validate_circle(geometry_text, geometry_mode, geometry_unit, errors, "geometry", "source")
    if errors:
        return None, errors
    return NormalizedInputs(n0=n0, distance_m=distance_m, theta_rad=theta_rad, r0_m=r0_m), {}


# V1 name for the direct-density validator.
validate_inputs = validate_direct_inputs


def validate_current_inputs(
    *,
    current_text,
    current_unit: str,
    voltage_text,
    voltage_unit: str,
    mass_text,
    distance_text,
    distance_unit: str,
    theta_text,
    geometry_mode: str,
    geometry_text,
    geometry_unit: str,
    collector_mode: str,
    collector_text,
    collector_unit: str,
    charge_c: float = MODELED_CHARGE_C,
) -> tuple[CurrentInputs | None, dict[str, str]]:
    """Validate all current-derived text inputs together and convert them to SI.

    Current and voltage entries may be signed; their magnitudes are used and zero
    is rejected. Mass is entered in amu and must be positive. Error keys:
    "current", "voltage", "mass", "distance", "theta", "geometry", "collector".
    Never raises for user input.
    """
    errors: dict[str, str] = {}
    current_a, current_negative = _validate_signed_magnitude(
        current_text, current_unit, current_to_a, "the beam current magnitude |I₀|", "1.0",
        "Beam current magnitude must be nonzero; zero current is not supported.", errors, "current",
    )
    voltage_v, voltage_negative = _validate_signed_magnitude(
        voltage_text, voltage_unit, voltage_to_v, "the accelerating voltage magnitude |V₀|", "1.0",
        "Accelerating voltage magnitude must be nonzero; zero voltage is not supported.", errors, "voltage",
    )
    mass_amu = _parse_field(mass_text, "the individual ion mass mᵢ", "100", errors, "mass")
    mass_kg = None
    if mass_amu is not None:
        if mass_amu <= 0.0:
            errors["mass"] = "Enter a positive individual ion mass in amu."
        else:
            try:
                mass_kg = amu_to_kg(mass_amu)
            except PlumeRangeError:
                errors["mass"] = (
                    f"Ion mass {mass_amu:g} amu cannot be represented in kg "
                    "(it rounds to zero or overflows); enter a less extreme value."
                )
    distance_m = _validate_distance(distance_text, distance_unit, errors)
    theta_rad = _validate_theta(theta_text, errors)
    r0_m = _validate_circle(geometry_text, geometry_mode, geometry_unit, errors, "geometry", "source")
    collector_m = _validate_circle(collector_text, collector_mode, collector_unit, errors, "collector", "collector")
    if errors:
        return None, errors
    return CurrentInputs(
        current_a=current_a,
        voltage_v=voltage_v,
        mass_amu=mass_amu,
        mass_kg=mass_kg,
        r0_m=r0_m,
        distance_m=distance_m,
        theta_rad=theta_rad,
        collector_radius_m=collector_m,
        charge_c=charge_c,
        current_sign_negative=current_negative,
        voltage_sign_negative=voltage_negative,
    ), {}
