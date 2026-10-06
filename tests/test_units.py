"""Unit conversion and text-input validation tests."""

import math

import pytest

from plume_model import PlumeRangeError, calculate_plume
from units import (
    AREA_FACTORS_M2,
    LENGTH_FACTORS_M,
    InputParseError,
    area_to_m2,
    area_unit_for_length,
    deg_to_rad,
    length_to_m,
    length_unit_for_area,
    m2_to_area,
    m_to_length,
    parse_real,
    rad_to_deg,
    validate_inputs,
)


def test_length_factors():
    assert LENGTH_FACTORS_M == {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "µm": 1e-6}
    assert length_to_m(1.0, "mm") == 0.001
    assert math.isclose(length_to_m(1000.0, "µm"), 0.001, rel_tol=1e-15)
    assert math.isclose(m_to_length(0.001, "µm"), 1000.0, rel_tol=1e-15)
    assert length_to_m(1.0, "um") == length_to_m(1.0, "µm") == length_to_m(1.0, "μm")
    assert length_to_m(10.0, "cm") == 0.1


def test_area_factors_are_squared_length_factors():
    for (lu, lf), (au, af) in zip(LENGTH_FACTORS_M.items(), AREA_FACTORS_M2.items()):
        assert au == lu + "²"
        assert math.isclose(af, lf * lf, rel_tol=1e-15)
    assert area_to_m2(1.0, "mm²") == 1e-6
    assert math.isclose(m2_to_area(1e-6, "mm²"), 1.0, rel_tol=1e-15)
    assert area_to_m2(1.0, "mm2") == 1e-6


def test_degree_conversion():
    assert deg_to_rad(180.0) == math.pi
    assert deg_to_rad(90.0) == math.pi / 2
    assert math.isclose(deg_to_rad(10.0), 0.174532925199, rel_tol=1e-11)
    assert math.isclose(rad_to_deg(deg_to_rad(37.5)), 37.5, rel_tol=1e-15)


def test_unit_pairs():
    assert area_unit_for_length("mm") == "mm²"
    assert length_unit_for_area("µm²") == "µm"
    with pytest.raises(ValueError):
        length_to_m(1.0, "in")
    with pytest.raises(ValueError):
        area_to_m2(1.0, "ft²")


def test_conversion_rounding_to_zero_or_overflow_is_rejected():
    with pytest.raises(PlumeRangeError):
        length_to_m(1e-320, "µm")
    with pytest.raises(PlumeRangeError):
        area_to_m2(1e-320, "µm²")
    with pytest.raises(PlumeRangeError):
        m_to_length(1e305, "µm")
    assert length_to_m(0.0, "mm") == 0.0


@pytest.mark.parametrize("text,value", [("1e20", 1e20), (" 1.0E+20 ", 1e20), ("0.10", 0.1), ("−5", -5.0), ("7", 7.0)])
def test_parse_real_accepts(text, value):
    assert parse_real(text) == value


@pytest.mark.parametrize("text", ["", "   ", None, "abc", "1e", "1,5", "nan", "NaN", "inf", "-Infinity", "1e999"])
def test_parse_real_rejects(text):
    with pytest.raises(InputParseError):
        parse_real(text)


def _validate(n0="1e20", d="0.1", du="m", t="10", mode="radius", g="1.0", gu="mm"):
    return validate_inputs(n0, d, du, t, mode, g, gu)


def test_validate_inputs_success():
    normalized, errors = _validate()
    assert errors == {}
    assert normalized.n0 == 1e20
    assert normalized.distance_m == 0.1
    assert normalized.r0_m == 0.001
    assert normalized.theta_rad == deg_to_rad(10.0)


def test_validate_area_mode_matches_radius_mode():
    a, _ = _validate()
    b, errors = _validate(mode="area", g=repr(math.pi), gu="mm²")
    assert errors == {}
    assert math.isclose(a.r0_m, b.r0_m, rel_tol=1e-15)
    ra = calculate_plume(a.n0, a.r0_m, a.distance_m, a.theta_rad)
    rb = calculate_plume(b.n0, b.r0_m, b.distance_m, b.theta_rad)
    assert math.isclose(ra.n2, rb.n2, rel_tol=1e-10)


def test_equivalent_units_validate_to_same_si():
    a, _ = _validate(d="0.1", du="m", g="1", gu="mm")
    b, _ = _validate(d="100000", du="µm", g="0.1", gu="cm")
    assert math.isclose(a.distance_m, b.distance_m, rel_tol=1e-15)
    assert math.isclose(a.r0_m, b.r0_m, rel_tol=1e-15)


@pytest.mark.parametrize(
    "kwargs,field,message",
    [
        ({"d": "-0.1"}, "distance", "Axial distance must be zero or greater."),
        ({"t": "90"}, "theta", "Half-angle must be at least 0° and less than 90°."),
        ({"t": "-1"}, "theta", "Half-angle must be at least 0° and less than 90°."),
        ({"t": "120"}, "theta", "Half-angle must be at least 0° and less than 90°."),
        ({"g": ""}, "geometry", "Enter a positive source radius."),
        ({"g": "0"}, "geometry", "Enter a positive source radius."),
        ({"g": "-2"}, "geometry", "Enter a positive source radius."),
        ({"mode": "area", "g": "0", "gu": "mm²"}, "geometry", "Enter a positive source area."),
        ({"n0": "0"}, "n0", "Source density n₀ must be greater than zero."),
        ({"n0": "-1e20"}, "n0", "Source density n₀ must be greater than zero."),
    ],
)
def test_validate_inputs_messages(kwargs, field, message):
    normalized, errors = _validate(**kwargs)
    assert normalized is None
    assert errors[field] == message


@pytest.mark.parametrize("field,kwargs", [("n0", {"n0": "lots"}), ("n0", {"n0": "nan"}), ("distance", {"d": "inf"}),
                                          ("theta", {"t": ""}), ("geometry", {"g": "1 mm"}),
                                          ("distance", {"d": "1e-320", "du": "µm"}),
                                          ("geometry", {"g": "1e-320", "gu": "µm"})])
def test_validate_inputs_bad_text(field, kwargs):
    normalized, errors = _validate(**kwargs)
    assert normalized is None
    assert field in errors and errors[field]


def test_all_errors_reported_together():
    normalized, errors = validate_inputs("x", "-1", "m", "95", "radius", "", "mm")
    assert normalized is None
    assert set(errors) == {"n0", "distance", "theta", "geometry"}


def test_zero_distance_is_valid():
    normalized, errors = _validate(d="0")
    assert errors == {} and normalized.distance_m == 0.0


# ---------------------------------------------------------------- V2: current-derived inputs
from units import (  # noqa: E402
    AMU_KG,
    CURRENT_FACTORS_A,
    VOLTAGE_FACTORS_V,
    amu_to_kg,
    current_to_a,
    kg_to_amu,
    validate_current_inputs,
    validate_direct_inputs,
    voltage_to_v,
)


def test_current_voltage_factors_and_aliases():
    assert CURRENT_FACTORS_A == {"A": 1.0, "mA": 1e-3, "µA": 1e-6, "nA": 1e-9}
    assert VOLTAGE_FACTORS_V == {"V": 1.0, "kV": 1e3}
    assert current_to_a(1.0, "µA") == 1e-6
    assert current_to_a(1.0, "uA") == current_to_a(1.0, "μA") == current_to_a(1.0, "µA")
    assert math.isclose(current_to_a(1000.0, "nA"), 1e-6, rel_tol=1e-15)
    assert math.isclose(current_to_a(0.001, "mA"), 1e-6, rel_tol=1e-15)
    assert voltage_to_v(1.0, "kV") == 1000.0
    assert voltage_to_v(1.0, "kv") == 1000.0
    with pytest.raises(ValueError):
        current_to_a(1.0, "pA")
    with pytest.raises(ValueError):
        voltage_to_v(1.0, "MV")


def test_amu_conversion_uses_agreed_constant():
    assert AMU_KG == 1.66053906660e-27
    assert math.isclose(amu_to_kg(100.0), 1.66053906660e-25, rel_tol=1e-15)
    assert math.isclose(kg_to_amu(amu_to_kg(123.4)), 123.4, rel_tol=1e-15)


def test_v2_conversions_reject_values_that_round_away():
    with pytest.raises(PlumeRangeError):
        current_to_a(1e-310, "nA")  # positive input that rounds to zero/subnormal
    with pytest.raises(PlumeRangeError):
        amu_to_kg(1e-290)  # ~1.7e-317 kg is subnormal
    with pytest.raises(PlumeRangeError):
        voltage_to_v(1e306, "kV")  # overflows


def _cvalidate(**overrides):
    fields = dict(current_text="1.0", current_unit="µA", voltage_text="1.0", voltage_unit="kV", mass_text="100",
                  distance_text="0.10", distance_unit="m", theta_text="10", geometry_mode="radius",
                  geometry_text="1.0", geometry_unit="mm", collector_mode="radius", collector_text="1.0",
                  collector_unit="mm")
    fields.update(overrides)
    return validate_current_inputs(**fields)


def test_validate_current_inputs_success():
    inputs, errors = _cvalidate()
    assert errors == {}
    assert inputs.current_a == 1e-6 and inputs.voltage_v == 1000.0
    assert inputs.mass_amu == 100.0 and math.isclose(inputs.mass_kg, 1.66053906660e-25, rel_tol=1e-15)
    assert inputs.r0_m == 1e-3 and inputs.collector_radius_m == 1e-3
    assert inputs.distance_m == 0.1 and inputs.theta_rad == deg_to_rad(10.0)
    assert inputs.charge_c == 1.602176634e-19
    assert not inputs.current_sign_negative and not inputs.voltage_sign_negative


def test_signed_current_and_voltage_use_magnitude():
    positive, _ = _cvalidate()
    signed, errors = _cvalidate(current_text="-1.0", voltage_text="−1.0")
    assert errors == {}
    assert signed.current_sign_negative and signed.voltage_sign_negative
    for field in signed.PHYSICAL_FIELDS:
        assert getattr(signed, field) == getattr(positive, field)


def test_collector_area_mode():
    inputs, errors = _cvalidate(collector_mode="area", collector_text=repr(math.pi), collector_unit="mm²")
    assert errors == {}
    assert math.isclose(inputs.collector_radius_m, 1e-3, rel_tol=1e-15)


@pytest.mark.parametrize(
    "overrides,field,message",
    [
        ({"current_text": "0"}, "current", "Beam current magnitude must be nonzero; zero current is not supported."),
        ({"current_text": "-0"}, "current", "Beam current magnitude must be nonzero; zero current is not supported."),
        ({"voltage_text": "0"}, "voltage",
         "Accelerating voltage magnitude must be nonzero; zero voltage is not supported."),
        ({"mass_text": "0"}, "mass", "Enter a positive individual ion mass in amu."),
        ({"mass_text": "-100"}, "mass", "Enter a positive individual ion mass in amu."),
        ({"collector_text": ""}, "collector", "Enter a positive collector radius."),
        ({"collector_text": "-1"}, "collector", "Enter a positive collector radius."),
        ({"collector_mode": "area", "collector_text": "0", "collector_unit": "mm²"}, "collector",
         "Enter a positive collector area."),
        ({"geometry_text": ""}, "geometry", "Enter a positive source radius."),
        ({"distance_text": "-1"}, "distance", "Axial distance must be zero or greater."),
        ({"theta_text": "90"}, "theta", "Half-angle must be at least 0° and less than 90°."),
    ],
)
def test_validate_current_inputs_messages(overrides, field, message):
    inputs, errors = _cvalidate(**overrides)
    assert inputs is None
    assert errors[field] == message


@pytest.mark.parametrize("overrides,field", [
    ({"current_text": "abc"}, "current"),
    ({"current_text": "nan"}, "current"),
    ({"voltage_text": "inf"}, "voltage"),
    ({"mass_text": ""}, "mass"),
    ({"mass_text": "1e-290"}, "mass"),
    ({"collector_text": "1e-320", "collector_unit": "µm"}, "collector"),
    ({"current_text": "1e-310", "current_unit": "nA"}, "current"),
])
def test_validate_current_inputs_bad_values(overrides, field):
    inputs, errors = _cvalidate(**overrides)
    assert inputs is None and errors.get(field)


def test_all_current_errors_reported_together():
    inputs, errors = _cvalidate(current_text="", voltage_text="0", mass_text="-1", distance_text="x", theta_text="95",
                                geometry_text="", collector_text="0")
    assert inputs is None
    assert set(errors) == {"current", "voltage", "mass", "distance", "theta", "geometry", "collector"}


def test_direct_validator_alias():
    assert validate_inputs is validate_direct_inputs
