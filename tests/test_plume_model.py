"""Physics tests for plume_model: benchmark, identities, conservation, domain and range."""

import math

import numpy as np
import pytest

from plume_model import (
    PlumeModelError,
    PlumeRangeError,
    calculate_plume,
    density_at_distance,
    plume_area,
    plume_radius,
)
from units import area_to_m2, deg_to_rad, length_to_m

RTOL = 1e-10

# PRD section 11.1 reference values (full frustum model).
BENCHMARK = {
    "r2_m": 0.0186326980708465,
    "area1_m2": 3.141592653589793e-6,
    "area2_m2": 1.090690086825855e-3,
    "n2": 2.880371511152640e17,
    "expansion": 347.1774373993268,
    "remaining_fraction": 0.002880371511152640,
    "remaining_percent": 0.2880371511152640,
    "reduction_percent": 99.71196284888474,
}


@pytest.fixture
def benchmark():
    return calculate_plume(1.0e20, length_to_m(1.0, "mm"), length_to_m(0.10, "m"), deg_to_rad(10.0))


def test_benchmark_inputs_normalized(benchmark):
    assert benchmark.r1_m == pytest.approx(0.001, rel=1e-15)
    assert benchmark.theta_rad == pytest.approx(0.174532925199, rel=1e-11)


@pytest.mark.parametrize("name,expected", BENCHMARK.items())
def test_benchmark_matches_reference(benchmark, name, expected):
    assert math.isclose(getattr(benchmark, name), expected, rel_tol=RTOL)


def test_benchmark_fraction_consistency(benchmark):
    assert math.isclose(benchmark.n2 / benchmark.n1, benchmark.remaining_fraction, rel_tol=RTOL)
    assert math.isclose(benchmark.area1_m2 / benchmark.area2_m2, benchmark.remaining_fraction, rel_tol=RTOL)
    assert math.isclose(benchmark.expansion * benchmark.remaining_fraction, 1.0, rel_tol=RTOL)
    assert math.isclose(benchmark.remaining_percent + benchmark.reduction_percent, 100.0, rel_tol=1e-14)


def test_formula_matches_prd_closed_form(benchmark):
    # n2 = n1 * (r1 / (r1 + l tan theta))^2, written independently of the model code path.
    expected = 1.0e20 * (0.001 / (0.001 + 0.10 * math.tan(math.radians(10.0)))) ** 2
    assert math.isclose(benchmark.n2, expected, rel_tol=RTOL)


@pytest.mark.parametrize("distance,theta", [(0.0, deg_to_rad(30.0)), (0.5, 0.0), (0.0, 0.0), (1e6, 0.0)])
def test_identity_cases_exact(distance, theta):
    r = calculate_plume(3.3e18, 2.5e-3, distance, theta)
    assert r.n2 == r.n1
    assert r.r2_m == r.r1_m
    assert r.area2_m2 == r.area1_m2
    assert r.expansion == 1.0
    assert r.remaining_fraction == 1.0
    assert r.reduction_fraction == 0.0
    assert r.reduction_percent == 0.0


def test_density_decreases_with_distance():
    theta = deg_to_rad(15.0)
    n2 = [calculate_plume(1e20, 1e-3, d, theta).n2 for d in np.linspace(0.0, 2.0, 50)]
    assert all(a > b for a, b in zip(n2, n2[1:]))


def test_density_decreases_with_angle():
    n2 = [calculate_plume(1e20, 1e-3, 0.2, deg_to_rad(t)).n2 for t in np.linspace(0.0, 89.0, 60)]
    assert all(a > b for a, b in zip(n2, n2[1:]))


@pytest.mark.parametrize("n1", [1e6, 1e20, 1e30])
@pytest.mark.parametrize("r1", [1e-6, 1e-3, 0.5])
@pytest.mark.parametrize("distance", [1e-4, 0.1, 10.0])
@pytest.mark.parametrize("theta_deg", [0.5, 10.0, 45.0, 80.0])
def test_conservation_and_bounds(n1, r1, distance, theta_deg):
    r = calculate_plume(n1, r1, distance, deg_to_rad(theta_deg))
    # Conserved particle flow at equal axial velocity: n1*A1 == n2*A2.
    assert math.isclose(r.n1 * r.area1_m2, r.n2 * r.area2_m2, rel_tol=RTOL)
    assert 0.0 < r.remaining_fraction <= 1.0
    assert r.expansion >= 1.0
    assert 0.0 < r.n2 <= r.n1
    assert math.isclose(r.expansion * r.remaining_fraction, 1.0, rel_tol=RTOL)
    assert math.isclose(r.remaining_fraction + r.reduction_fraction, 1.0, rel_tol=1e-14)
    # Geometry: r2 = r1 + l tan(theta), A = pi r^2.
    assert math.isclose(r.r2_m, r1 + distance * math.tan(deg_to_rad(theta_deg)), rel_tol=1e-15)
    assert math.isclose(r.area2_m2, math.pi * r.r2_m**2, rel_tol=1e-15)


def test_unit_equivalence_gives_same_result():
    theta = deg_to_rad(10.0)
    base = calculate_plume(1e20, 0.001, 0.10, theta)
    for r_val, r_unit in [(1.0, "mm"), (0.1, "cm"), (1000.0, "µm"), (0.001, "m")]:
        for d_val, d_unit in [(0.10, "m"), (10.0, "cm"), (100.0, "mm"), (1e5, "µm")]:
            r = calculate_plume(1e20, length_to_m(r_val, r_unit), length_to_m(d_val, d_unit), theta)
            assert math.isclose(r.n2, base.n2, rel_tol=RTOL)
            assert math.isclose(r.r2_m, base.r2_m, rel_tol=RTOL)
            assert math.isclose(r.area2_m2, base.area2_m2, rel_tol=RTOL)


@pytest.mark.parametrize("unit", ["m²", "cm²", "mm²", "µm²"])
def test_radius_and_area_modes_agree(unit):
    r1 = 1e-3
    factor = {"m²": 1.0, "cm²": 1e-4, "mm²": 1e-6, "µm²": 1e-12}[unit]
    area_in_unit = math.pi * r1**2 / factor
    r1_from_area = math.sqrt(area_to_m2(area_in_unit, unit) / math.pi)
    theta = deg_to_rad(10.0)
    a = calculate_plume(1e20, r1, 0.10, theta)
    b = calculate_plume(1e20, r1_from_area, 0.10, theta)
    assert math.isclose(a.n2, b.n2, rel_tol=RTOL)
    assert math.isclose(a.area1_m2, b.area1_m2, rel_tol=RTOL)


def test_far_field_inverse_square_limit():
    n1, r1, theta = 1e20, 1e-6, deg_to_rad(20.0)
    for distance in (1.0, 10.0):
        r = calculate_plume(n1, r1, distance, theta)
        approx = n1 * r1**2 / (distance**2 * math.tan(theta) ** 2)
        assert math.isclose(r.n2, approx, rel_tol=1e-4)


def test_near_ninety_degrees_is_finite_and_flagged():
    r = calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(89.999))
    assert math.isfinite(r.n2) and 0.0 < r.n2 < r.n1
    assert r.high_angle_sensitivity
    assert not calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(84.999)).high_angle_sensitivity
    assert calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(85.0)).high_angle_sensitivity


def test_tiny_positive_angle_resolves_reduction():
    theta = deg_to_rad(1e-9)
    r = calculate_plume(1e20, 1e-3, 0.1, theta)
    expected_reduction = 1.0 - (1e-3 / (1e-3 + 0.1 * math.tan(theta))) ** 2  # ~3.5e-9, cancellation-limited
    exact_reduction = 2 * 0.1 * theta / 1e-3  # first-order series, accurate to ~1e-9 relative here
    assert r.reduction_fraction > 0.0
    assert r.remaining_fraction <= 1.0
    assert math.isclose(r.reduction_fraction, exact_reduction, rel_tol=1e-8)
    assert math.isclose(r.reduction_fraction, expected_reduction, rel_tol=1e-6)


def test_ninety_degrees_float_is_rejected():
    with pytest.raises(PlumeModelError) as exc:
        calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(90.0))
    assert exc.value.field == "theta"


@pytest.mark.parametrize(
    "args,field",
    [
        ((1e20, 1e-3, -0.1, 0.1), "distance"),
        ((0.0, 1e-3, 0.1, 0.1), "n1"),
        ((-1e20, 1e-3, 0.1, 0.1), "n1"),
        ((1e20, 0.0, 0.1, 0.1), "r1"),
        ((1e20, -1e-3, 0.1, 0.1), "r1"),
        ((1e20, 1e-3, 0.1, -1e-9), "theta"),
        ((1e20, 1e-3, 0.1, math.pi / 2), "theta"),
        ((1e20, 1e-3, 0.1, 2.0), "theta"),
        ((math.nan, 1e-3, 0.1, 0.1), "n1"),
        ((1e20, math.inf, 0.1, 0.1), "r1"),
        ((1e20, 1e-3, math.inf, 0.1), "distance"),
        ((1e20, 1e-3, 0.1, math.nan), "theta"),
        (("1e20", 1e-3, 0.1, 0.1), "n1"),
        ((True, 1e-3, 0.1, 0.1), "n1"),
    ],
)
def test_invalid_inputs_rejected(args, field):
    with pytest.raises(PlumeModelError) as exc:
        calculate_plume(*args)
    assert exc.value.field == field
    assert not isinstance(exc.value, PlumeRangeError)


@pytest.mark.parametrize(
    "args,quantity",
    [
        ((1e20, 1e-3, 1e306, deg_to_rad(89.99)), "delta_r"),  # l tan(theta) > float max
        ((1e20, 1e308, 1e308, deg_to_rad(45.0)), "r2"),  # r1 + delta_r > float max
        ((1e20, 1e200, 0.0, 0.0), "area1"),  # pi r1^2 overflows
        ((1e20, 1e153, 1e154, deg_to_rad(45.0)), "area2"),  # pi r2^2 overflows
        ((1e20, 1e-200, 0.0, 0.0), "area1"),  # pi r1^2 underflows
        ((1e20, 1e-150, 1e10, deg_to_rad(45.0)), "F"),  # (r1/r2)^2 ~ 1e-320 underflows
        ((1e-300, 1e-6, 1e6, deg_to_rad(45.0)), "n2"),  # n1 * 1e-24 underflows
        ((1e20, 1e10, 1e-3, 1e-320), "reduction"),  # 1 - F below float resolution
    ],
)
def test_unrepresentable_values_raise_named_range_error(args, quantity):
    with pytest.raises(PlumeRangeError) as exc:
        calculate_plume(*args)
    assert exc.value.quantity == quantity
    assert str(exc.value)


def test_plume_radius_and_area_helpers():
    assert plume_radius(1e-3, 0.0, 0.5) == 1e-3
    assert plume_radius(1e-3, 0.3, 0.0) == 1e-3
    assert math.isclose(plume_radius(1e-3, 0.1, math.pi / 4), 0.101, rel_tol=1e-14)
    assert math.isclose(plume_area(2.0), 4 * math.pi, rel_tol=1e-15)
    with pytest.raises(PlumeModelError):
        plume_area(0.0)
    with pytest.raises(PlumeRangeError):
        plume_area(1e200)


def test_density_at_distance_scalar_and_array():
    theta = deg_to_rad(10.0)
    assert density_at_distance(1e20, 1e-3, 0.0, theta) == 1e20
    assert isinstance(density_at_distance(1e20, 1e-3, 0.05, theta), float)
    x = np.array([0.0, 0.05, 0.1])
    n = density_at_distance(1e20, 1e-3, x, theta)
    assert isinstance(n, np.ndarray) and n.shape == (3,)
    assert n[-1] == calculate_plume(1e20, 1e-3, 0.1, theta).n2
    with pytest.raises(PlumeModelError):
        density_at_distance(1e20, 1e-3, [-0.1, 0.1], theta)
    with pytest.raises(PlumeModelError):
        density_at_distance(1e20, 1e-3, [np.nan], theta)
    with pytest.raises(PlumeModelError):
        density_at_distance(1e20, 1e-3, 0.1, deg_to_rad(90.0))
