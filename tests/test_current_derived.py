"""End-to-end current-derived physics: benchmark, cross-mode equivalence, conservation, scaling."""

import math

import numpy as np
import pytest

from current_derived import calculate_current_derived
from plume_model import calculate_plume, plume_profile
from units import (
    amu_to_kg,
    area_to_m2,
    current_to_a,
    deg_to_rad,
    length_to_m,
    validate_current_inputs,
    voltage_to_v,
)

RTOL = 1e-10
Q = 1.602176634e-19

# V2 specification section 9.2.
BENCHMARK = {
    ("source", "mass_kg"): 1.66053906660e-25,
    ("source", "velocity_m_s"): 43928.42636759329,
    ("source", "area0_m2"): 3.141592653589793e-6,
    ("source", "n0"): 4.522661536932149e13,
    ("plume", "r2_m"): 0.0186326980708465,
    ("plume", "area2_m2"): 1.090690086825855e-3,
    ("plume", "remaining_fraction"): 0.002880371511152640,
    ("plume", "expansion"): 347.1774373993268,
    ("plume", "n2"): 1.302694544556517e11,
    ("collection", "collector_area_m2"): 3.141592653589793e-6,
    ("collection", "fraction"): 0.002880371511152640,
    ("collection", "percent"): 0.2880371511152640,
    ("collection", "collected_current_a"): 2.880371511152640e-9,
}


def _run(current_a=1e-6, voltage_v=1000.0, mass_amu=100.0, r0_m=1e-3, distance_m=0.10, theta_deg=10.0,
         collector_m=1e-3, **kw):
    return calculate_current_derived(current_a, voltage_v, amu_to_kg(mass_amu), r0_m, distance_m,
                                     deg_to_rad(theta_deg), collector_m, **kw)


@pytest.fixture
def benchmark():
    # Inputs converted from the UI units used in the specification.
    return calculate_current_derived(current_to_a(1.0, "µA"), voltage_to_v(1.0, "kV"), amu_to_kg(100.0),
                                     length_to_m(1.0, "mm"), length_to_m(0.10, "m"), deg_to_rad(10.0),
                                     length_to_m(1.0, "mm"))


@pytest.mark.parametrize("path,expected", BENCHMARK.items(), ids=lambda v: str(v))
def test_benchmark_matches_reference(benchmark, path, expected):
    part, name = path
    assert math.isclose(getattr(getattr(benchmark, part), name), expected, rel_tol=RTOL)


def test_benchmark_downstream_check(benchmark):
    # n2 = I0/(q v0 A2), written out independently of the model's scaled evaluation.
    s, p = benchmark.source, benchmark.plume
    assert math.isclose(p.n2, 1e-6 / (Q * s.velocity_m_s * p.area2_m2), rel_tol=RTOL)
    assert math.isclose(benchmark.n2_from_current, p.n2, rel_tol=RTOL)


def test_cross_mode_equivalence(benchmark):
    direct = calculate_plume(benchmark.source.n0, 1e-3, 0.10, deg_to_rad(10.0))
    assert direct == benchmark.plume  # same propagation law, identical outputs
    x_d, n_d = plume_profile(direct)
    x_c, n_c = plume_profile(benchmark.plume)
    assert np.array_equal(x_d, x_c) and np.array_equal(n_d, n_c)


def test_full_beam_current_is_conserved_not_reduced_by_f(benchmark):
    s, p = benchmark.source, benchmark.plume
    downstream = p.n2 * Q * s.velocity_m_s * p.area2_m2
    assert math.isclose(downstream, s.current_a, rel_tol=RTOL)
    assert math.isclose(benchmark.downstream_beam_current_a, s.current_a, rel_tol=RTOL)
    assert p.remaining_fraction < 0.01  # density falls ~350x while the full beam current does not


def test_collection_uses_full_current_not_f_scaled_current(benchmark):
    c, p = benchmark.collection, benchmark.plume
    assert math.isclose(c.collected_current_a, 1e-6 * c.fraction, rel_tol=1e-15)
    assert not math.isclose(c.collected_current_a, 1e-6 * p.remaining_fraction * c.fraction, rel_tol=1e-3)


@pytest.mark.parametrize("k", [0.1, 5.0, 1e3])
def test_current_scaling(k):
    a, b = _run(), _run(current_a=1e-6 * k)
    assert b.source.velocity_m_s == a.source.velocity_m_s
    assert b.collection.fraction == a.collection.fraction
    for get in (lambda r: r.source.n0, lambda r: r.plume.n2, lambda r: r.collection.collected_current_a):
        assert math.isclose(get(b), k * get(a), rel_tol=1e-14)


@pytest.mark.parametrize("kv,km", [(4.0, 1.0), (0.25, 1.0), (1.0, 9.0), (1.0, 0.01)])
def test_voltage_and_mass_scaling(kv, km):
    a, b = _run(), _run(voltage_v=1000.0 * kv, mass_amu=100.0 * km)
    assert math.isclose(b.source.velocity_m_s, a.source.velocity_m_s * math.sqrt(kv / km), rel_tol=1e-14)
    assert math.isclose(b.source.n0, a.source.n0 * math.sqrt(km / kv), rel_tol=1e-14)
    # Collection depends only on current and geometry.
    assert b.collection.fraction == a.collection.fraction
    assert b.collection.collected_current_a == a.collection.collected_current_a


@pytest.mark.parametrize("k", [0.5, 2.0, 10.0])
def test_source_size_scaling(k):
    a, b = _run(), _run(r0_m=1e-3 * k)
    assert math.isclose(b.source.n0, a.source.n0 / k**2, rel_tol=1e-14)
    # Downstream density from the conserved current with the new geometry, not a fixed-n0 argument.
    r2 = 1e-3 * k + 0.10 * math.tan(deg_to_rad(10.0))
    expected_n2 = 1e-6 / (Q * b.source.velocity_m_s * math.pi * r2**2)
    assert math.isclose(b.plume.n2, expected_n2, rel_tol=RTOL)


@pytest.mark.parametrize("distance_m,theta_deg", [(0.0, 10.0), (0.3, 0.0), (0.0, 0.0)])
def test_identity_cases(distance_m, theta_deg):
    r = _run(distance_m=distance_m, theta_deg=theta_deg)
    p = r.plume
    assert p.n2 == r.source.n0 == p.n0
    assert p.area2_m2 == p.area0_m2 == r.source.area0_m2
    assert p.remaining_fraction == 1.0 and p.expansion == 1.0
    assert p.reduction_fraction == 0.0
    x, n = plume_profile(p)
    assert np.all(n == p.n0)
    if distance_m == 0.0:
        assert x.tolist() == [0.0]


@pytest.mark.parametrize(
    "collector_m,fraction,current_a",
    [(1e-3, 0.25, 0.25e-6), (2e-3, 1.0, 1e-6), (3e-3, 1.0, 1e-6)],
)
def test_specification_collection_table(collector_m, fraction, current_a):
    # V2 section 9.3: theta = 0, r0 = 2 mm, any l, I0 = 1 uA, any valid mass/voltage.
    r = _run(r0_m=2e-3, distance_m=0.37, theta_deg=0.0, collector_m=collector_m, mass_amu=250.0, voltage_v=3e3)
    p, c = r.plume, r.collection
    assert p.r2_m == 2e-3
    assert p.remaining_fraction == 1.0 and p.expansion == 1.0 and p.n2 == p.n0
    assert p.reduction_percent == 0.0
    if fraction == 1.0:
        assert c.fraction == 1.0 and c.collected_current_a == 1e-6
    else:
        assert math.isclose(c.fraction, fraction, rel_tol=1e-15)
        assert math.isclose(c.collected_current_a, current_a, rel_tol=1e-15)


@pytest.mark.parametrize("collector_m", [1e-4, 5e-4, 2e-3, 0.0186, 0.05, 10.0])
def test_collector_geometry_separation(collector_m):
    base, other = _run(), _run(collector_m=collector_m)
    assert other.source == base.source
    assert other.plume == base.plume
    x_a, n_a = plume_profile(base.plume)
    x_b, n_b = plume_profile(other.plume)
    assert np.array_equal(x_a, x_b) and np.array_equal(n_a, n_b)
    if collector_m >= base.plume.r2_m:
        assert other.collection.fraction == 1.0
    else:
        assert math.isclose(other.collection.fraction, (collector_m / base.plume.r2_m) ** 2, rel_tol=1e-15)


def test_f_coll_and_f_are_distinct_quantities():
    same_area = _run(collector_m=1e-3)  # collector area equals source area
    assert same_area.collection.fraction == same_area.plume.remaining_fraction
    bigger = _run(collector_m=5e-3)
    assert bigger.collection.fraction != bigger.plume.remaining_fraction
    assert bigger.plume.remaining_fraction == same_area.plume.remaining_fraction


def test_unit_equivalence():
    base = _run()
    for current in (current_to_a(1000.0, "nA"), current_to_a(0.001, "mA"), current_to_a(1e-6, "A")):
        for voltage in (voltage_to_v(1000.0, "V"), voltage_to_v(1.0, "kV")):
            r = calculate_current_derived(current, voltage, amu_to_kg(100.0), length_to_m(0.1, "cm"),
                                          length_to_m(10.0, "cm"), deg_to_rad(10.0), length_to_m(1000.0, "µm"))
            for part, name in BENCHMARK:
                assert math.isclose(getattr(getattr(r, part), name), getattr(getattr(base, part), name), rel_tol=RTOL)


def test_area_entry_matches_radius_entry():
    r0_from_area = math.sqrt(area_to_m2(math.pi, "mm²") / math.pi)
    coll_from_area = math.sqrt(area_to_m2(math.pi * 1e-6, "m²") / math.pi)
    a, b = _run(), _run(r0_m=r0_from_area, collector_m=coll_from_area)
    assert math.isclose(a.plume.n2, b.plume.n2, rel_tol=RTOL)
    assert math.isclose(a.collection.collected_current_a, b.collection.collected_current_a, rel_tol=RTOL)


def _validated(**overrides):
    fields = dict(current_text="1.0", current_unit="µA", voltage_text="1.0", voltage_unit="kV", mass_text="100",
                  distance_text="0.10", distance_unit="m", theta_text="10", geometry_mode="radius",
                  geometry_text="1.0", geometry_unit="mm", collector_mode="radius", collector_text="1.0",
                  collector_unit="mm")
    fields.update(overrides)
    inputs, errors = validate_current_inputs(**fields)
    assert errors == {}
    return inputs, calculate_current_derived(inputs.current_a, inputs.voltage_v, inputs.mass_kg, inputs.r0_m,
                                             inputs.distance_m, inputs.theta_rad, inputs.collector_radius_m,
                                             inputs.charge_c)


@pytest.mark.parametrize("current,voltage", [("-1.0", "1.0"), ("1.0", "-1.0"), ("−1.0", "−1.0")])
def test_signed_conventions_give_same_nonnegative_results(current, voltage):
    _, positive = _validated()
    inputs, signed = _validated(current_text=current, voltage_text=voltage)
    assert signed == positive
    assert signed.collection.collected_current_a > 0
    assert inputs.current_sign_negative == current.startswith(("-", "−"))
    assert inputs.voltage_sign_negative == voltage.startswith(("-", "−"))


def test_charge_parameter_does_not_change_propagation_law():
    single, double = _run(), _run(charge_c=2 * Q)
    assert double.plume.remaining_fraction == single.plume.remaining_fraction
    assert math.isclose(double.plume.n2 / double.source.n0, single.plume.n2 / single.source.n0, rel_tol=1e-15)
