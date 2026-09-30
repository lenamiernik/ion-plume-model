"""UI workflow checks using Streamlit's headless AppTest runner."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def at():
    return AppTest.from_file(APP, default_timeout=30).run()


def _text(elements):
    return " ".join(str(e.value) for e in elements)


def test_initial_state(at):
    assert not at.exception
    assert at.title[0].value == "Ion Plume Geometric Expansion Model"
    assert at.text_input(key="geometry_text").value == ""
    assert at.button(key="calculate").disabled
    assert len(at.metric) == 0


def test_load_example_shows_benchmark(at):
    at.button(key="load_example").click().run()
    assert not at.exception
    assert at.metric[0].value == "2.88 × 10¹⁷ particles/m³"
    values = [m.value for m in at.metric]
    assert "0.288 %" in values and "99.7 %" in values
    assert "Illustrative example" in _text(at.info)
    assert "OUT OF DATE" not in _text(at.warning)


def test_edit_marks_result_stale_and_scale_does_not(at):
    at.button(key="load_example").click().run()
    at.radio(key="density_scale").set_value("Linear").run()
    assert "OUT OF DATE" not in _text(at.warning)
    at.text_input(key="theta_text").input("20").run()
    assert "OUT OF DATE" in _text(at.warning)
    at.button(key="calculate").click().run()
    assert "OUT OF DATE" not in _text(at.warning)
    assert at.metric[0].value != "2.88 × 10¹⁷ particles/m³"


def test_equivalent_unit_change_is_not_stale(at):
    at.button(key="load_example").click().run()
    at.text_input(key="distance_text").input("10").run()
    at.selectbox(key="distance_unit").set_value("cm").run()
    assert "OUT OF DATE" not in _text(at.warning)


def test_invalid_input_reports_error_and_preserves_text(at):
    at.text_input(key="n1_text").input("1e20").run()
    at.text_input(key="distance_text").input("-3").run()
    at.text_input(key="theta_text").input("95").run()
    at.text_input(key="geometry_text").input("abc").run()
    at.button(key="calculate").click().run()
    assert not at.exception
    errors = _text(at.error)
    assert "Axial distance must be zero or greater." in errors
    assert "Half-angle must be at least 0° and less than 90°." in errors
    assert "source radius must be a number" in errors
    assert at.text_input(key="distance_text").value == "-3"
    assert at.text_input(key="geometry_text").value == "abc"
    assert len(at.metric) == 0


def test_range_error_is_reported(at):
    at.text_input(key="n1_text").input("1e-300").run()
    at.text_input(key="distance_text").input("1e6").run()
    at.text_input(key="theta_text").input("45").run()
    at.selectbox(key="radius_unit").set_value("µm").run()
    at.text_input(key="geometry_text").input("1").run()
    at.button(key="calculate").click().run()
    assert "Numerical range error (n2)" in _text(at.error)
    assert len(at.metric) == 0


def test_mode_switch_converts_geometry(at):
    at.button(key="load_example").click().run()
    at.radio(key="geometry_mode").set_value("area").run()
    assert at.text_input(key="geometry_text").value == "3.14159265358979"
    assert at.selectbox(key="area_unit").value == "mm²"
    assert "OUT OF DATE" not in _text(at.warning)
    at.radio(key="geometry_mode").set_value("radius").run()
    assert at.text_input(key="geometry_text").value == "1"
    assert at.selectbox(key="radius_unit").value == "mm"


def test_mode_switch_respects_changed_unit(at):
    at.button(key="load_example").click().run()
    at.radio(key="geometry_mode").set_value("area").run()
    at.selectbox(key="area_unit").set_value("cm²").run()  # now means 3.14159... cm²
    assert "OUT OF DATE" in _text(at.warning)
    at.radio(key="geometry_mode").set_value("radius").run()
    # sqrt(3.14159265358979 cm² / π) = 0.999999999999999… cm; reusing the old mm radius would give 0.1.
    assert float(at.text_input(key="geometry_text").value) == pytest.approx(1.0, rel=1e-14)
    assert at.selectbox(key="radius_unit").value == "cm"


def test_invalid_value_cleared_on_mode_switch(at):
    at.text_input(key="geometry_text").input("-4").run()
    at.radio(key="geometry_mode").set_value("area").run()
    assert at.text_input(key="geometry_text").value == ""
    assert "was cleared" in _text(at.warning)


def test_reset_clears_geometry_and_results(at):
    at.button(key="load_example").click().run()
    at.button(key="reset").click().run()
    assert at.text_input(key="geometry_text").value == ""
    assert len(at.metric) == 0
    assert at.button(key="calculate").disabled


def test_zero_distance_workflow(at):
    at.button(key="load_example").click().run()
    at.text_input(key="distance_text").input("0").run()
    at.button(key="calculate").click().run()
    assert not at.exception
    assert at.metric[0].value == "1.00 × 10²⁰ particles/m³"
    values = [m.value for m in at.metric]
    assert "100 %" in values and "0 %" in values
    for scale in ("Linear", "Logarithmic"):
        at.radio(key="density_scale").set_value(scale).run()
        assert not at.exception
