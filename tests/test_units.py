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


def _validate(n1="1e20", d="0.1", du="m", t="10", mode="radius", g="1.0", gu="mm"):
    return validate_inputs(n1, d, du, t, mode, g, gu)


def test_validate_inputs_success():
    normalized, errors = _validate()
    assert errors == {}
    assert normalized.n1 == 1e20
    assert normalized.distance_m == 0.1
    assert normalized.r1_m == 0.001
    assert normalized.theta_rad == deg_to_rad(10.0)


def test_validate_area_mode_matches_radius_mode():
    a, _ = _validate()
    b, errors = _validate(mode="area", g=repr(math.pi), gu="mm²")
    assert errors == {}
    assert math.isclose(a.r1_m, b.r1_m, rel_tol=1e-15)
    ra = calculate_plume(a.n1, a.r1_m, a.distance_m, a.theta_rad)
    rb = calculate_plume(b.n1, b.r1_m, b.distance_m, b.theta_rad)
    assert math.isclose(ra.n2, rb.n2, rel_tol=1e-10)


def test_equivalent_units_validate_to_same_si():
    a, _ = _validate(d="0.1", du="m", g="1", gu="mm")
    b, _ = _validate(d="100000", du="µm", g="0.1", gu="cm")
    assert math.isclose(a.distance_m, b.distance_m, rel_tol=1e-15)
    assert math.isclose(a.r1_m, b.r1_m, rel_tol=1e-15)


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
        ({"n1": "0"}, "n1", "Initial density n₁ must be greater than zero."),
        ({"n1": "-1e20"}, "n1", "Initial density n₁ must be greater than zero."),
    ],
)
def test_validate_inputs_messages(kwargs, field, message):
    normalized, errors = _validate(**kwargs)
    assert normalized is None
    assert errors[field] == message


@pytest.mark.parametrize("field,kwargs", [("n1", {"n1": "lots"}), ("n1", {"n1": "nan"}), ("distance", {"d": "inf"}),
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
    assert set(errors) == {"n1", "distance", "theta", "geometry"}


def test_zero_distance_is_valid():
    normalized, errors = _validate(d="0")
    assert errors == {} and normalized.distance_m == 0.0
