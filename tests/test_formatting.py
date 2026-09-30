"""Display-formatting tests: significant figures and honest percentages."""

import pytest

from formatting import format_area, format_length, format_percent, format_sig
from plume_model import calculate_plume
from units import deg_to_rad


@pytest.mark.parametrize(
    "value,sig,text",
    [
        (2.880371511152640e17, 3, "2.88 × 10¹⁷"),
        (1e20, 3, "1.00 × 10²⁰"),
        (347.1774373993268, 3, "347"),
        (0.002880371511152640, 3, "0.00288"),
        (0.0186326980708465, 3, "0.0186"),
        (1.0, 3, "1.00"),
        (3.141592653589793e-6, 3, "3.14 × 10⁻⁶"),
        (2.880371511152640e17, 10, "2.880371511 × 10¹⁷"),
        (0.0, 3, "0"),
    ],
)
def test_format_sig(value, sig, text):
    assert format_sig(value, sig) == text


def test_engineering_units():
    assert format_length(0.0186326980708465) == "0.0186 m (1.86 cm)"
    assert format_length(0.001) == "0.00100 m (1.00 mm)"
    assert format_length(2.5) == "2.50 m"
    assert format_length(3e-6) == "3.00 × 10⁻⁶ m (3.00 µm)"
    assert format_area(3.141592653589793e-6) == "3.14 × 10⁻⁶ m² (3.14 mm²)"


def test_benchmark_percentages():
    r = calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(10.0))
    assert format_percent(r.remaining_fraction, r.reduction_fraction) == "0.288 %"
    assert format_percent(r.reduction_fraction, r.remaining_fraction) == "99.7 %"


def test_tiny_remaining_fraction_never_shows_zero_or_hundred():
    r = calculate_plume(1e20, 1e-9, 1e3, deg_to_rad(60.0))  # F ~ 3e-25
    remaining = format_percent(r.remaining_fraction, r.reduction_fraction)
    reduction = format_percent(r.reduction_fraction, r.remaining_fraction)
    assert "10⁻" in remaining and not remaining.startswith("0 ")
    assert reduction.startswith("100 % − ")


def test_tiny_reduction_never_shows_zero_or_hundred():
    r = calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(1e-7))  # 1 - F ~ 3.5e-7
    remaining = format_percent(r.remaining_fraction, r.reduction_fraction)
    reduction = format_percent(r.reduction_fraction, r.remaining_fraction)
    assert remaining == "99.99997 %"  # 100 % - 3.49e-5 %, shown to 5 decimals
    assert reduction == "3.49 × 10⁻⁵ %"


def test_percentages_near_hundred_distinguishable():
    for gap in (0.3, 0.04, 0.00999, 1e-6, 1e-11):
        text = format_percent(1 - gap / 100, gap / 100)
        assert text != "100 %" and not text.startswith("100.0")


def test_exact_identity_percentages():
    r = calculate_plume(1e20, 1e-3, 0.0, 0.3)
    assert format_percent(r.remaining_fraction, r.reduction_fraction) == "100 %"
    assert format_percent(r.reduction_fraction, r.remaining_fraction) == "0 %"
