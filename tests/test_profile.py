"""Density-profile and sampling-grid tests."""

import math

import numpy as np
import pytest

from plume_model import (
    MIN_PROFILE_SAMPLES,
    PlumeModelError,
    PlumeRangeError,
    calculate_plume,
    density_at_distance,
    plume_profile,
    profile_grid,
)
from units import deg_to_rad


def _check_grid(x, distance):
    assert x[0] == 0.0
    assert x[-1] == distance
    assert np.all(np.diff(x) > 0)
    assert len(x) >= MIN_PROFILE_SAMPLES


def test_profile_endpoints_match_model():
    result = calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(10.0))
    x, n = plume_profile(result)
    _check_grid(x, 0.1)
    assert n[0] == result.n1
    assert n[-1] == result.n2
    assert np.all(np.diff(n) < 0)


def test_profile_matches_closed_form_everywhere():
    n1, r1, theta = 1e20, 1e-3, deg_to_rad(10.0)
    result = calculate_plume(n1, r1, 0.1, theta)
    x, n = plume_profile(result)
    expected = n1 * (r1 / (r1 + x * math.tan(theta))) ** 2
    np.testing.assert_allclose(n, expected, rtol=1e-12)


def test_zero_angle_profile_is_constant_and_linear_grid():
    result = calculate_plume(5e17, 2e-3, 0.3, 0.0)
    x, n = plume_profile(result)
    _check_grid(x, 0.3)
    assert np.all(n == 5e17)
    np.testing.assert_allclose(np.diff(x), np.diff(x)[0], rtol=1e-9)


def test_zero_distance_profile_is_single_point():
    result = calculate_plume(1e20, 1e-3, 0.0, deg_to_rad(30.0))
    x, n = plume_profile(result)
    assert x.tolist() == [0.0]
    assert n.tolist() == [1e20]


def test_steep_decay_profile_is_resolved_near_source():
    r1, theta, distance = 1e-6, deg_to_rad(45.0), 1.0
    result = calculate_plume(1e20, r1, distance, theta)
    assert result.remaining_fraction < 1e-11
    x, n = plume_profile(result)
    _check_grid(x, distance)
    length_scale = r1 / math.tan(theta)  # distance over which the radius doubles
    assert np.count_nonzero((x > 0) & (x < length_scale)) >= 20
    assert np.all(np.diff(n) < 0)
    assert n[-1] == result.n2
    # The far tail is also sampled for linear plotting.
    assert np.max(np.diff(x)) <= distance / 50


def test_tiny_angle_profile():
    result = calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(1e-9))
    x, n = plume_profile(result)
    _check_grid(x, 0.1)
    assert n[0] == 1e20 and n[-1] == result.n2
    assert np.all(np.diff(n) <= 0)


def test_near_ninety_profile():
    result = calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(89.9))
    x, n = plume_profile(result)
    _check_grid(x, 0.1)
    assert np.all(np.isfinite(n)) and np.all(n > 0)


def test_grid_rejects_invalid_domain():
    with pytest.raises(PlumeModelError):
        profile_grid(1e-3, -1.0, 0.1)
    with pytest.raises(PlumeModelError):
        profile_grid(0.0, 1.0, 0.1)


def test_profile_underflow_is_range_error():
    with pytest.raises(PlumeRangeError):
        density_at_distance(1e-300, 1e-6, np.array([0.0, 1e6]), deg_to_rad(45.0))
