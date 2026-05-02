"""Deterministic geometry overlap detector."""
from sevim.overlap import OverlapError, assert_no_overlaps, detect_overlaps
from sevim.pipeline import run_pipeline


def test_simple_scene_has_no_overlap():
    r = run_pipeline("A causes B.")
    assert detect_overlaps(r.placed) == []


def test_containment_is_not_reported_as_overlap():
    r = run_pipeline("The arm is part of the body.")
    # arm is legitimately nested inside body; must not appear as a finding.
    findings = detect_overlaps(r.placed)
    assert not [f for f in findings if f["kind"] == "shape_overlap"], findings


def test_short_label_fits_in_box():
    r = run_pipeline("A causes B.")
    findings = [f for f in detect_overlaps(r.placed) if f["kind"] == "label_overflow"]
    assert findings == []


def test_long_label_overflow_is_caught():
    r = run_pipeline("Extremely_long_unbreakable_identifier causes another_equally_long_token.")
    findings = [f for f in detect_overlaps(r.placed) if f["kind"] == "label_overflow"]
    assert findings, "expected at least one label_overflow finding"
    # stable ordering
    ids = [f["a"] for f in findings]
    assert ids == sorted(ids)


def test_multi_clause_complex_scene_stays_clean():
    sample = (
        "A network contains filters. "
        "Filters require training. "
        "Training reduces to gradient descent. "
        "Gradient descent is part of backpropagation."
    )
    r = run_pipeline(sample)
    overlaps = [f for f in detect_overlaps(r.placed) if f["kind"] == "shape_overlap"]
    assert overlaps == [], overlaps


def test_strict_mode_raises_on_overflow():
    try:
        assert_no_overlaps(run_pipeline(
            "Extremely_long_unbreakable_identifier causes another_equally_long_token."
        ).placed)
    except OverlapError as e:
        assert e.findings
        return
    raise AssertionError("OverlapError not raised")
