"""UI workflow checks for both tabs using Streamlit's headless AppTest runner."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
DIRECT, CURRENT = 0, 1


@pytest.fixture
def at():
    return AppTest.from_file(APP, default_timeout=60).run()


def _text(elements):
    return " ".join(str(e.value) for e in elements)


def _metrics(at, tab):
    return [m.value for m in at.tabs[tab].metric]


def _stale(at, tab):
    return "OUT OF DATE" in _text(at.tabs[tab].warning)


def _markdown(at, tab):
    return _text(at.tabs[tab].markdown)


# ---------------------------------------------------------------- layout and initial state
def test_initial_state(at):
    assert not at.exception
    assert at.title[0].value == "Ion Plume Geometric Expansion Model"
    assert [t.label for t in at.tabs] == ["Direct Density", "Current-Derived Density"]
    for key in ("d_source_text", "c_source_text", "c_collector_text"):
        assert at.text_input(key=key).value == ""
    assert at.button(key="d_calculate").disabled
    assert at.button(key="c_calculate").disabled
    assert len(at.metric) == 0


def test_current_tab_shows_required_help_and_labels(at):
    info = _text(at.tabs[CURRENT].info)
    assert "one representative ion population with fixed elementary charge" in info
    assert at.text_input(key="c_current_text").label == "Beam current magnitude |I₀|"
    assert at.text_input(key="c_voltage_text").label == "Accelerating voltage magnitude |V₀|"
    assert "divide it by two" in _text(at.tabs[CURRENT].caption)


# ---------------------------------------------------------------- direct tab (V1 regression)
def test_direct_example_reproduces_v1_benchmark(at):
    at.button(key="d_load_example").click().run()
    assert not at.exception
    values = _metrics(at, DIRECT)
    assert values[0] == "2.88 × 10¹⁷ particles/m³"
    assert "0.288 %" in values and "99.7 %" in values
    assert "Illustrative example" in _text(at.tabs[DIRECT].info)
    assert not _stale(at, DIRECT)
    assert _metrics(at, CURRENT) == []  # the other tab is untouched
    assert len(at.tabs[DIRECT].get("image")) == 2  # geometry diagram and density profile


def test_direct_invalid_input_reports_errors_and_preserves_text(at):
    at.text_input(key="d_n0_text").input("1e20").run()
    at.text_input(key="d_distance_text").input("-3").run()
    at.text_input(key="d_theta_text").input("95").run()
    at.text_input(key="d_source_text").input("abc").run()
    at.button(key="d_calculate").click().run()
    assert not at.exception
    errors = _text(at.tabs[DIRECT].error)
    assert "Axial distance must be zero or greater." in errors
    assert "Half-angle must be at least 0° and less than 90°." in errors
    assert "source radius must be a number" in errors
    assert at.text_input(key="d_distance_text").value == "-3"
    assert at.text_input(key="d_source_text").value == "abc"
    assert _metrics(at, DIRECT) == []


def test_direct_range_error_is_reported(at):
    at.text_input(key="d_n0_text").input("1e-300").run()
    at.text_input(key="d_distance_text").input("1e6").run()
    at.text_input(key="d_theta_text").input("45").run()
    at.selectbox(key="d_source_radius_unit").set_value("µm").run()
    at.text_input(key="d_source_text").input("1").run()
    at.button(key="d_calculate").click().run()
    assert "Numerical range error (n2)" in _text(at.tabs[DIRECT].error)
    assert _metrics(at, DIRECT) == []


def test_direct_stale_and_plot_controls(at):
    at.button(key="d_load_example").click().run()
    at.radio(key="d_density_scale").set_value("Linear").run()
    at.radio(key="d_diagram_view").set_value("Schematic").run()
    assert not _stale(at, DIRECT)
    at.text_input(key="d_distance_text").input("10").run()
    at.selectbox(key="d_distance_unit").set_value("cm").run()  # same physical distance
    assert not _stale(at, DIRECT)
    at.text_input(key="d_theta_text").input("20").run()
    assert _stale(at, DIRECT)
    at.button(key="d_calculate").click().run()
    assert not _stale(at, DIRECT)
    assert _metrics(at, DIRECT)[0] != "2.88 × 10¹⁷ particles/m³"


def test_direct_zero_distance_both_scales(at):
    at.button(key="d_load_example").click().run()
    at.text_input(key="d_distance_text").input("0").run()
    at.button(key="d_calculate").click().run()
    values = _metrics(at, DIRECT)
    assert values[0] == "1.00 × 10²⁰ particles/m³"
    assert "100 %" in values and "0 %" in values
    for scale in ("Linear", "Logarithmic"):
        at.radio(key="d_density_scale").set_value(scale).run()
        assert not at.exception


# ---------------------------------------------------------------- current-derived tab
def test_current_example_reproduces_v2_benchmark(at):
    at.button(key="c_load_example").click().run()
    assert not at.exception
    values = _metrics(at, CURRENT)
    assert values[0] == "1.30 × 10¹¹ particles/m³"  # primary metric stays n2
    assert "4.52 × 10¹³ particles/m³" in values  # calculated n0
    assert "2.88 nA" in values  # I_coll
    assert values.count("0.288 %") == 2  # remaining density and collected percentage (equal radii)
    assert "Calculated source density n₀" in " ".join(m.label for m in at.tabs[CURRENT].metric)
    assert "Illustrative example" in _text(at.tabs[CURRENT].info)
    markdown = _markdown(at, CURRENT)
    assert "4.39 × 10⁴ m/s" in markdown  # v0
    assert "1.602176634 × 10⁻¹⁹ C" in markdown  # fixed charge disclosed in details
    assert "1.66053906660 × 10⁻²⁷ kg" in markdown  # amu conversion disclosed
    assert "1.660539067 × 10⁻²⁵ kg" in markdown  # converted mass
    assert _metrics(at, DIRECT) == []
    assert len(at.tabs[CURRENT].get("image")) == 2


def test_current_calculate_requires_source_and_collector(at):
    at.text_input(key="c_source_text").input("1").run()
    assert at.button(key="c_calculate").disabled
    at.text_input(key="c_collector_text").input("1").run()
    assert not at.button(key="c_calculate").disabled
    at.text_input(key="c_source_text").input("").run()
    assert at.button(key="c_calculate").disabled


def test_hidden_invalid_fields_in_other_tab_do_not_block(at):
    at.text_input(key="d_n0_text").input("garbage").run()
    at.button(key="c_load_example").click().run()
    assert _metrics(at, CURRENT)[0] == "1.30 × 10¹¹ particles/m³"
    assert _text(at.tabs[CURRENT].error) == ""
    assert _text(at.tabs[DIRECT].error) == ""  # direct tab not validated until its own Calculate
    at.text_input(key="c_mass_text").input("bad").run()
    at.button(key="d_load_example").click().run()
    assert _metrics(at, DIRECT)[0] == "2.88 × 10¹⁷ particles/m³"


@pytest.mark.parametrize("key,value", [
    ("c_mass_text", "200"), ("c_current_text", "2.0"), ("c_voltage_text", "2.0"), ("c_collector_text", "5.0"),
    ("c_source_text", "2.0"), ("c_distance_text", "0.2"), ("c_theta_text", "15"),
])
def test_current_physical_changes_mark_stale(at, key, value):
    at.button(key="d_load_example").click().run()
    at.button(key="c_load_example").click().run()
    at.text_input(key=key).input(value).run()
    assert _stale(at, CURRENT)
    assert not _stale(at, DIRECT)  # tabs keep separate snapshots


def test_current_plot_and_view_changes_do_not_mark_stale(at):
    at.button(key="c_load_example").click().run()
    at.radio(key="c_density_scale").set_value("Linear").run()
    at.radio(key="c_diagram_view").set_value("Schematic").run()
    assert not _stale(at, CURRENT)


def test_unit_change_with_same_text_is_stale_but_equivalent_units_are_not(at):
    at.button(key="c_load_example").click().run()
    at.selectbox(key="c_current_unit").set_value("nA").run()  # "1.0" now means 1 nA
    assert _stale(at, CURRENT)
    at.text_input(key="c_current_text").input("1000").run()  # 1000 nA = 1 µA
    assert not _stale(at, CURRENT)
    at.text_input(key="c_voltage_text").input("1000").run()
    at.selectbox(key="c_voltage_unit").set_value("V").run()  # 1000 V = 1 kV
    assert not _stale(at, CURRENT)


def test_signed_entries_use_magnitude(at):
    at.button(key="c_load_example").click().run()
    reference = _metrics(at, CURRENT)
    at.text_input(key="c_current_text").input("-1.0").run()
    at.text_input(key="c_voltage_text").input("−1.0").run()
    assert not _stale(at, CURRENT)  # same physical magnitudes
    assert "Signed entry interpreted by magnitude" in _text(at.tabs[CURRENT].caption)
    at.button(key="c_calculate").click().run()
    assert _metrics(at, CURRENT) == reference
    assert "entered with a negative sign; magnitude used" in _markdown(at, CURRENT)


def test_zero_current_rejected_and_old_result_marked_stale(at):
    at.button(key="c_load_example").click().run()
    at.text_input(key="c_current_text").input("0").run()
    at.button(key="c_calculate").click().run()
    assert "Beam current magnitude must be nonzero; zero current is not supported." in _text(at.tabs[CURRENT].error)
    assert at.text_input(key="c_current_text").value == "0"
    assert _stale(at, CURRENT)  # the previous result is never relabelled as current


def test_speed_of_light_rejected_in_ui(at):
    at.button(key="c_load_example").click().run()
    at.text_input(key="c_mass_text").input("1e-4").run()
    at.text_input(key="c_voltage_text").input("1000").run()
    at.button(key="c_calculate").click().run()
    assert "speed of light" in _text(at.tabs[CURRENT].error)
    assert _stale(at, CURRENT)


def test_collector_saturation_display(at):
    at.button(key="c_load_example").click().run()
    at.text_input(key="c_collector_text").input("100").run()
    at.button(key="c_calculate").click().run()
    values = _metrics(at, CURRENT)
    assert "1.00 µA" in values and "100 %" in values
    assert values[0] == "1.30 × 10¹¹ particles/m³"  # density unchanged by collector size
    assert "1 (exactly)" in _markdown(at, CURRENT)
    assert "covers the entire modeled beam" in _text(at.tabs[CURRENT].info)


def test_current_zero_distance_both_scales(at):
    at.button(key="c_load_example").click().run()
    at.text_input(key="c_distance_text").input("0").run()
    at.button(key="c_calculate").click().run()
    assert not at.exception
    values = _metrics(at, CURRENT)
    assert values[0] == "4.52 × 10¹³ particles/m³" == values[1]  # n2 == n0
    for scale in ("Linear", "Logarithmic"):
        at.radio(key="c_density_scale").set_value(scale).run()
        assert not at.exception


# ---------------------------------------------------------------- toggles
def test_source_and_collector_toggles_are_independent_and_round_trip(at):
    at.button(key="c_load_example").click().run()
    for _ in range(3):
        at.radio(key="c_source_mode").set_value("area").run()
        assert at.text_input(key="c_source_text").value == "3.14159265358979"
        assert at.text_input(key="c_collector_text").value in ("1.0", "1")  # untouched
        at.radio(key="c_collector_mode").set_value("area").run()
        assert at.text_input(key="c_collector_text").value == "3.14159265358979"
        assert at.selectbox(key="c_collector_area_unit").value == "mm²"
        assert not _stale(at, CURRENT)
        at.radio(key="c_source_mode").set_value("radius").run()
        assert at.text_input(key="c_source_text").value == "1"
        at.radio(key="c_collector_mode").set_value("radius").run()
        assert at.text_input(key="c_collector_text").value == "1"
    assert not _stale(at, CURRENT)


def test_direct_toggle_round_trip_and_changed_unit(at):
    at.button(key="d_load_example").click().run()
    at.radio(key="d_source_mode").set_value("area").run()
    assert at.text_input(key="d_source_text").value == "3.14159265358979"
    at.radio(key="d_source_mode").set_value("radius").run()
    assert at.text_input(key="d_source_text").value == "1"
    at.radio(key="d_source_mode").set_value("area").run()
    at.selectbox(key="d_source_area_unit").set_value("cm²").run()  # now means 3.14159... cm²
    assert _stale(at, DIRECT)
    at.radio(key="d_source_mode").set_value("radius").run()
    assert float(at.text_input(key="d_source_text").value) == pytest.approx(1.0, rel=1e-14)
    assert at.selectbox(key="d_source_radius_unit").value == "cm"


def test_invalid_value_cleared_on_mode_switch(at):
    at.text_input(key="c_collector_text").input("-4").run()
    at.radio(key="c_collector_mode").set_value("area").run()
    assert at.text_input(key="c_collector_text").value == ""
    assert "previous collector value" in _text(at.tabs[CURRENT].warning)


# ---------------------------------------------------------------- reset
def test_reset_affects_only_its_own_tab(at):
    at.button(key="d_load_example").click().run()
    at.button(key="c_load_example").click().run()
    at.button(key="c_reset").click().run()
    assert at.text_input(key="c_source_text").value == ""
    assert at.text_input(key="c_collector_text").value == ""
    assert at.text_input(key="c_mass_text").value == "100"  # other inputs kept (V1 behavior)
    assert _metrics(at, CURRENT) == []
    assert at.button(key="c_calculate").disabled
    assert _metrics(at, DIRECT)[0] == "2.88 × 10¹⁷ particles/m³"
    assert at.text_input(key="d_source_text").value == "1.0"
    at.button(key="d_reset").click().run()
    assert at.text_input(key="d_source_text").value == ""
    assert _metrics(at, DIRECT) == []
