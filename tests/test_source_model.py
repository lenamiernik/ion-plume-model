"""Source-normalization physics: amu conversion, v0, n0 and current reconstruction."""

import inspect
import math
from decimal import Decimal, getcontext

import pytest

from constants import AMU_KG, ELEMENTARY_CHARGE_C, MODELED_CHARGE_C, SPEED_OF_LIGHT_M_S
from plume_model import PlumeModelError, PlumeRangeError
from source_model import calculate_source_from_current, characteristic_velocity, source_current
from units import amu_to_kg

getcontext().prec = 50
RTOL = 1e-10
# Independent literals from the V2 specification, section 3.1.
Q = 1.602176634e-19
AMU = 1.66053906660e-27

# V2 specification section 9.2 (source part).
BENCHMARK = {
    "mass_kg": 1.66053906660e-25,
    "velocity_m_s": 43928.42636759329,
    "area0_m2": 3.141592653589793e-6,
    "n0": 4.522661536932149e13,
}


@pytest.fixture
def benchmark():
    return calculate_source_from_current(1.0e-6, 1000.0, amu_to_kg(100.0), 1.0e-3)


def test_constants_match_specification():
    assert ELEMENTARY_CHARGE_C == Q
    assert AMU_KG == AMU
    assert MODELED_CHARGE_C == ELEMENTARY_CHARGE_C
    assert SPEED_OF_LIGHT_M_S == 299_792_458.0


def test_default_charge_is_one_elementary_charge():
    default = inspect.signature(calculate_source_from_current).parameters["charge_c"].default
    assert default == Q


@pytest.mark.parametrize("name,expected", BENCHMARK.items())
def test_benchmark_matches_reference(benchmark, name, expected):
    assert math.isclose(getattr(benchmark, name), expected, rel_tol=RTOL)


def test_benchmark_normalized_inputs(benchmark):
    assert benchmark.current_a == 1.0e-6
    assert benchmark.voltage_v == 1000.0
    assert benchmark.charge_c == Q
    assert benchmark.r0_m == 1.0e-3
    assert math.isclose(benchmark.kinetic_energy_j, Q * 1000.0, rel_tol=1e-15)


def test_whiteboard_form_gives_same_n0(benchmark):
    # n0 = [I0/(q pi r0^2)] sqrt(m/(2 q V0)): a different algebraic route than I0/(q v0 A0).
    expected = (1e-6 / (Q * math.pi * 1e-3**2)) * math.sqrt(100 * AMU / (2 * Q * 1000.0))
    assert math.isclose(benchmark.n0, expected, rel_tol=RTOL)


def test_energy_relation_holds(benchmark):
    # (1/2) m v0^2 = q V0 in joules.
    kinetic = 0.5 * benchmark.mass_kg * benchmark.velocity_m_s**2
    assert math.isclose(kinetic, Q * 1000.0, rel_tol=1e-14)


def test_current_reconstruction(benchmark):
    # I0 = n0 q v0 A0 recovers the entered current.
    assert math.isclose(source_current(benchmark.n0, benchmark.velocity_m_s, benchmark.area0_m2), 1e-6, rel_tol=RTOL)
    plain = benchmark.n0 * Q * benchmark.velocity_m_s * benchmark.area0_m2
    assert math.isclose(plain, 1e-6, rel_tol=RTOL)


def test_n0_has_inverse_volume_dimension():
    # n0 = I/(q v A) must scale like 1/(velocity * area) at fixed current: v -> v/k (mass * k^2)
    # and A -> A/k^2 (radius / k) together give n0 -> k^3 n0, consistent with units of m^-3.
    base = calculate_source_from_current(1e-6, 1000.0, 100 * AMU, 1e-3)
    k = 10.0
    scaled = calculate_source_from_current(1e-6, 1000.0, 100 * AMU * k**2, 1e-3 / k)
    assert math.isclose(scaled.n0, base.n0 * k**3, rel_tol=1e-12)


@pytest.mark.parametrize("k", [0.25, 3.0, 1e3])
def test_current_scaling(k):
    base = calculate_source_from_current(1e-6, 1000.0, 100 * AMU, 1e-3)
    scaled = calculate_source_from_current(1e-6 * k, 1000.0, 100 * AMU, 1e-3)
    assert scaled.velocity_m_s == base.velocity_m_s  # current never changes velocity
    assert math.isclose(scaled.n0, base.n0 * k, rel_tol=1e-14)


@pytest.mark.parametrize("k", [0.01, 4.0, 9.0])
def test_voltage_scaling(k):
    base = calculate_source_from_current(1e-6, 1000.0, 100 * AMU, 1e-3)
    scaled = calculate_source_from_current(1e-6, 1000.0 * k, 100 * AMU, 1e-3)
    assert math.isclose(scaled.velocity_m_s, base.velocity_m_s * math.sqrt(k), rel_tol=1e-14)
    assert math.isclose(scaled.n0, base.n0 / math.sqrt(k), rel_tol=1e-14)


@pytest.mark.parametrize("k", [0.01, 4.0, 1e4])
def test_mass_scaling(k):
    base = calculate_source_from_current(1e-6, 1000.0, 100 * AMU, 1e-3)
    scaled = calculate_source_from_current(1e-6, 1000.0, 100 * AMU * k, 1e-3)
    assert math.isclose(scaled.velocity_m_s, base.velocity_m_s / math.sqrt(k), rel_tol=1e-14)
    assert math.isclose(scaled.n0, base.n0 * math.sqrt(k), rel_tol=1e-14)


@pytest.mark.parametrize("k", [0.1, 2.0, 30.0])
def test_source_size_scaling(k):
    base = calculate_source_from_current(1e-6, 1000.0, 100 * AMU, 1e-3)
    scaled = calculate_source_from_current(1e-6, 1000.0, 100 * AMU, 1e-3 * k)
    assert math.isclose(scaled.n0, base.n0 / k**2, rel_tol=1e-14)
    assert scaled.velocity_m_s == base.velocity_m_s


def test_configurable_charge_for_future_charge_states():
    single = calculate_source_from_current(1e-6, 1000.0, 100 * AMU, 1e-3)
    double = calculate_source_from_current(1e-6, 1000.0, 100 * AMU, 1e-3, charge_c=2 * Q)
    assert math.isclose(double.velocity_m_s, single.velocity_m_s * math.sqrt(2), rel_tol=1e-14)
    assert math.isclose(double.n0, single.n0 / (2 * math.sqrt(2)), rel_tol=1e-14)


def test_relativistic_notice_threshold():
    assert not calculate_source_from_current(1e-6, 1000.0, 100 * AMU, 1e-3).relativistic_notice
    electron_like = calculate_source_from_current(1e-6, 1000.0, 9.1093837e-31, 1e-3)  # v0 ~ 6 % of c
    assert electron_like.relativistic_notice
    assert 0.05 < electron_like.speed_fraction_of_c < 0.07


def test_speed_of_light_is_rejected():
    with pytest.raises(PlumeModelError) as exc:
        calculate_source_from_current(1e-6, 1e6, 1e-30, 1e-3)  # classical v0 ~ 5.7e8 m/s
    assert exc.value.field == "velocity"
    assert not isinstance(exc.value, PlumeRangeError)


def test_preventable_underflow_in_velocity_is_avoided():
    # 2 q V0 / m = 3.2e-329 underflows if formed directly, but v0 ~ 5.7e-165 m/s is representable.
    v0 = characteristic_velocity(1e-300, 1e10)
    expected = (Decimal(2) * Decimal(Q) * Decimal(1e-300) / Decimal(1e10)).sqrt()
    assert math.isclose(v0, float(expected), rel_tol=1e-14)


def test_preventable_underflow_in_density_denominator_is_avoided():
    # q v0 A0 is subnormal (~3e-313) when formed directly, yet n0 ~ 3.5e300 is representable.
    s = calculate_source_from_current(1e-12, 1e-10, 1e-20, 1e-145)
    expected = Decimal(1e-12) / (Decimal(Q) * Decimal(s.velocity_m_s) * Decimal(s.area0_m2))
    assert math.isclose(s.n0, float(expected), rel_tol=1e-14)


@pytest.mark.parametrize(
    "args,field",
    [
        ((0.0, 1000.0, 1e-25, 1e-3), "current"),
        ((-1e-6, 1000.0, 1e-25, 1e-3), "current"),  # the core takes magnitudes; signs are a UI convention
        ((1e-6, 0.0, 1e-25, 1e-3), "voltage"),
        ((1e-6, -1000.0, 1e-25, 1e-3), "voltage"),
        ((1e-6, 1000.0, 0.0, 1e-3), "mass"),
        ((1e-6, 1000.0, -1e-25, 1e-3), "mass"),
        ((1e-6, 1000.0, 1e-25, 0.0), "r0"),
        ((1e-6, 1000.0, 1e-25, -1e-3), "r0"),
        ((True, 1000.0, 1e-25, 1e-3), "current"),
        ((1e-6, math.nan, 1e-25, 1e-3), "voltage"),
        ((1e-6, 1000.0, math.inf, 1e-3), "mass"),
        (("1e-6", 1000.0, 1e-25, 1e-3), "current"),
    ],
)
def test_invalid_inputs_rejected(args, field):
    with pytest.raises(PlumeModelError) as exc:
        calculate_source_from_current(*args)
    assert exc.value.field == field


def test_invalid_charge_rejected():
    for charge in (0.0, -Q, False, math.nan):
        with pytest.raises(PlumeModelError) as exc:
            calculate_source_from_current(1e-6, 1000.0, 1e-25, 1e-3, charge_c=charge)
        assert exc.value.field == "charge"


@pytest.mark.parametrize(
    "args,quantity",
    [
        ((1e-6, 1000.0, 1e-320, 1e-3), "m_i"),  # subnormal mass
        ((1e-320, 1000.0, 1e-25, 1e-3), "I0"),  # subnormal current
        ((1.0, 1e3, 1e-25, 1e-150), "n0"),  # n0 overflows
        ((1e-300, 1000.0, 1e-25, 1e100), "n0"),  # n0 underflows
        ((1e-6, 1000.0, 1e-25, 1e200), "area0"),  # A0 overflows
    ],
)
def test_unrepresentable_values_raise_named_range_error(args, quantity):
    with pytest.raises(PlumeRangeError) as exc:
        calculate_source_from_current(*args)
    assert exc.value.quantity == quantity
