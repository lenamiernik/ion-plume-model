"""Figure-construction tests for the geometry diagram and density profile."""

import matplotlib.pyplot as plt
import pytest

from plume_model import calculate_plume, plume_profile
from units import deg_to_rad
from visualization import SCHEMATIC_LABEL, TO_SCALE_LABEL, geometry_figure, profile_figure


def _texts(fig):
    return " ".join(t.get_text() for ax in fig.axes for t in ax.texts)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_benchmark_geometry_is_to_scale_with_labels():
    result = calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(10.0))
    fig, schematic = geometry_figure(result)
    text = _texts(fig)
    assert not schematic and TO_SCALE_LABEL in text
    for label in ("P₀", "P₂", "r₀", "r₂", "A₀", "A₂", "l =", "θ =", "centerline", "Instrument"):
        assert label in text
    assert fig.axes[0].get_aspect() == 1.0


def test_extreme_aspect_ratio_uses_schematic():
    result = calculate_plume(1e20, 1e-6, 10.0, deg_to_rad(1.0))
    fig, schematic = geometry_figure(result)
    assert schematic and SCHEMATIC_LABEL in _texts(fig)
    fig2, schematic2 = geometry_figure(calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(10.0)), view="schematic")
    assert schematic2 and SCHEMATIC_LABEL in _texts(fig2)


def test_zero_distance_geometry_single_plane():
    result = calculate_plume(1e20, 1e-3, 0.0, deg_to_rad(10.0))
    fig, schematic = geometry_figure(result)
    text = _texts(fig)
    assert "P₀ = P₂" in text and "n₂ = n₀" in text
    assert not schematic


def test_zero_angle_geometry_is_cylinder():
    result = calculate_plume(1e20, 1e-2, 0.1, 0.0)
    fig, _ = geometry_figure(result)
    assert "θ = 0°" in _texts(fig)


@pytest.mark.parametrize("log_y", [True, False])
@pytest.mark.parametrize("distance", [0.0, 0.1])
def test_profile_figure_axes(log_y, distance):
    result = calculate_plume(1e20, 1e-3, distance, deg_to_rad(10.0))
    x, n = plume_profile(result)
    fig = profile_figure(result, x, n, log_y=log_y, illustrative=True)
    ax = fig.axes[0]
    assert ax.get_yscale() == ("log" if log_y else "linear")
    assert ax.get_xscale() == "linear"
    assert "Illustrative example values" in _texts(fig)
    if distance == 0.0:
        assert "coincide" in _texts(fig)


def test_steep_decay_log_profile():
    result = calculate_plume(1e20, 1e-6, 1.0, deg_to_rad(45.0))
    x, n = plume_profile(result)
    fig = profile_figure(result, x, n, log_y=True)
    lo, hi = fig.axes[0].get_ylim()
    assert lo <= result.n2 and hi >= result.n0


# ---------------------------------------------------------------- V2: centered collector
from visualization import COLLECTOR_LABEL  # noqa: E402


def test_collector_annotation_replaces_instrument_marker():
    result = calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(10.0))
    fig, schematic = geometry_figure(result, collector_radius_m=1e-3)
    text = _texts(fig)
    assert not schematic
    assert COLLECTOR_LABEL in text and "r_coll" in text and "A_coll" in text and "at x = l" in text
    assert "P₀" in text and "A₂" in text and "r₂" in text  # source and full beam stay labelled
    assert "Instrument" not in text


def test_oversized_collector_uses_schematic():
    result = calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(10.0))
    fig, schematic = geometry_figure(result, collector_radius_m=1.0)  # ~54x the beam radius
    assert schematic and SCHEMATIC_LABEL in _texts(fig)
    assert "1.00 m" in _texts(fig)  # true collector size still labelled


def test_collector_at_zero_distance():
    result = calculate_plume(1e20, 2e-3, 0.0, 0.0)
    fig, schematic = geometry_figure(result, collector_radius_m=1e-3)
    text = _texts(fig)
    assert "P₀ = P₂" in text and COLLECTOR_LABEL in text and not schematic
    fig2, schematic2 = geometry_figure(result, collector_radius_m=1.0)
    assert schematic2 and SCHEMATIC_LABEL in _texts(fig2)


def test_profile_uses_source_notation():
    result = calculate_plume(1e20, 1e-3, 0.1, deg_to_rad(10.0))
    x, n = plume_profile(result)
    text = _texts(profile_figure(result, x, n, log_y=True))
    assert "P₀: n₀" in text and "P₂: n₂" in text
