"""Verify each relation gets a visually distinct SVG signature."""
from sevim.pipeline import run_pipeline


def test_causes_renders_solid_arrow():
    svg = run_pipeline("A causes B.").svg
    assert 'marker-end="url(#arrow)"' in svg
    assert 'stroke-dasharray' not in svg


def test_used_for_renders_dashed_arrow():
    svg = run_pipeline("The algorithm uses a binary heap.").svg
    assert 'stroke-dasharray="6,4"' in svg
    assert 'marker-end="url(#arrow)"' in svg


def test_measures_renders_dotted_with_equals():
    # "X is divided by Y" still routes through `measures`; the math extension
    # captured the verbal "equals" form into the new `equals` relation.
    svg = run_pipeline("The probability is divided by the total.").svg
    assert 'stroke-dasharray="1,3"' in svg
    assert ">=<" in svg  # equals label


def test_equals_renders_solid_with_equals_label():
    # New math relation: "X equals Y" → `equals`, solid line + '=' midpoint.
    svg = run_pipeline("The perimeter equals four times the side.").svg
    assert 'e_equals_' in svg
    assert ">=<" in svg


def test_similar_to_renders_parallel_double():
    svg = run_pipeline("A dolphin is similar to a whale.").svg
    # Two <line> elements inside a <g data-eid="...similar_to...">
    assert 'e_similar_to_' in svg
    seg = svg[svg.index('e_similar_to_'):svg.index('e_similar_to_') + 500]
    assert seg.count('<line') >= 2


def test_opposes_renders_bar_markers():
    svg = run_pipeline("Heat opposes cold.").svg
    assert 'marker-start="url(#bar)"' in svg


def test_requires_renders_hollow_triangle():
    svg = run_pipeline("Merge sort requires additional memory.").svg
    assert 'marker-end="url(#hollow-tri)"' in svg


def test_reduces_to_renders_as_filled_path():
    svg = run_pipeline("The problem reduces to sorting.").svg
    assert 'e_reduces_to_' in svg
    idx = svg.index('e_reduces_to_')
    seg = svg[max(0, idx - 100): idx + 300]
    assert '<path' in seg
    # Funnel fill is a named gray; keep this loose to tolerate tuning.
    assert 'fill="#' in seg
