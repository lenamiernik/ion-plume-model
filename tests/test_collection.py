"""Centered-collector tests: overlap cap, saturation, separation from F, and range handling."""

import inspect
import math

import pytest

from collection import calculate_collection, collection_fraction
from formatting import format_percent
from plume_model import PlumeModelError, PlumeRangeError


@pytest.mark.parametrize(
    "collector_m,fraction,current_a",
    [(1e-3, 0.25, 0.25e-6), (2e-3, 1.0, 1e-6), (3e-3, 1.0, 1e-6)],
)
def test_specification_analytic_cases(collector_m, fraction, current_a):
    # V2 section 9.3: beam radius 2 mm, I0 = 1 uA.
    c = calculate_collection(1e-6, 2e-3, collector_m)
    assert math.isclose(c.fraction, fraction, rel_tol=1e-15)
    assert math.isclose(c.collected_current_a, current_a, rel_tol=1e-15)
    assert math.isclose(c.collector_area_m2, math.pi * collector_m**2, rel_tol=1e-15)


@pytest.mark.parametrize("collector_m", [2e-3, 2.000001e-3, 3e-3, 1.0, 1e100])
def test_saturation_is_exact(collector_m):
    c = calculate_collection(1e-6, 2e-3, collector_m)
    assert c.fraction == 1.0
    assert c.collected_current_a == 1e-6
    assert c.uncollected_fraction == 0.0
    assert c.saturated
    assert c.percent == 100.0
    assert format_percent(c.fraction, c.uncollected_fraction) == "100 %"


def test_just_below_saturation_is_not_full_capture():
    beam = 2e-3
    c = calculate_collection(1e-6, beam, beam * (1 - 1e-12))
    assert c.fraction < 1.0 and not c.saturated
    assert math.isclose(c.uncollected_fraction, 2e-12, rel_tol=1e-3)  # 1-(1-d)^2 ~ 2d, cancellation-free
    assert format_percent(c.fraction, c.uncollected_fraction) != "100 %"


def test_fraction_matches_area_ratio_when_smaller():
    beam, coll = 0.0186326980708465, 1e-3
    expected = (math.pi * coll**2) / (math.pi * beam**2)  # A_coll / A2 written independently
    assert math.isclose(collection_fraction(beam, coll), expected, rel_tol=1e-14)


def test_enormous_collector_does_not_overflow_ratio():
    # A_coll/A2 would be ~1e800; comparing radii first returns exactly 1.
    assert collection_fraction(1e-200, 1e200) == 1.0
    c = calculate_collection(1e-6, 1e-200, 1e150)  # A_coll ~ 3e300 is still representable
    assert c.fraction == 1.0 and math.isfinite(c.collector_area_m2)


def test_unrepresentable_collector_area_is_named():
    with pytest.raises(PlumeRangeError) as exc:
        calculate_collection(1e-6, 1e-3, 1e200)
    assert exc.value.quantity == "area_coll"


def test_fraction_underflow_is_named():
    with pytest.raises(PlumeRangeError) as exc:
        calculate_collection(1e-6, 1.0, 1e-200)
    assert exc.value.quantity in {"f_coll", "area_coll"}
    with pytest.raises(PlumeRangeError) as exc:
        collection_fraction(1.0, 1e-160)
    assert exc.value.quantity == "f_coll"


def test_collected_current_underflow_is_named():
    with pytest.raises(PlumeRangeError) as exc:
        calculate_collection(1e-300, 1.0, 1e-6)  # 1e-300 A * 1e-12
    assert exc.value.quantity == "I_coll"


def test_collected_current_scales_with_current_only():
    a = calculate_collection(1e-6, 2e-3, 1e-3)
    b = calculate_collection(7e-6, 2e-3, 1e-3)
    assert a.fraction == b.fraction
    assert math.isclose(b.collected_current_a, 7 * a.collected_current_a, rel_tol=1e-15)


def test_fraction_does_not_need_mass_or_voltage():
    assert list(inspect.signature(collection_fraction).parameters) == ["beam_radius_m", "collector_radius_m"]
    assert list(inspect.signature(calculate_collection).parameters) == [
        "source_current_a", "beam_radius_m", "collector_radius_m"]


@pytest.mark.parametrize("radius", [0.0, -1e-3, True, math.nan, math.inf, "1e-3"])
def test_invalid_collector_rejected(radius):
    with pytest.raises(PlumeModelError) as exc:
        calculate_collection(1e-6, 2e-3, radius)
    assert exc.value.field == "collector"
