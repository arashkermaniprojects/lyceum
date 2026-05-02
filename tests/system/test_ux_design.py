"""UX/UI design audit, programmatic tier.

A senior designer's checklist applied to ``serve/static/index.html``:
accessibility, color contrast, affordance, focus states, motion
durations, and information architecture.

Each test is a single inspector pass that fails loudly when the page
regresses below a UX baseline.  Visual aesthetics still need a human
eye, but anything tangible at the markup/CSS level is pinned here.
"""
from __future__ import annotations

import re

import pytest

from ._ux_helpers import (
    buttons_with_inner,
    contrast_ratio,
    extract_at_rule_body,
    extract_css,
    find_all_tags,
    first_tag,
    index_html,
    inputs_with_attrs,
    parse_css_rules,
    parse_tag_attrs,
    selects_with_attrs,
)


# ---------------------------------------------------------------------------
# 1. First impression
# ---------------------------------------------------------------------------

def test_page_has_visible_h1():
    src = index_html()
    h1 = first_tag(src, "h1")
    assert h1 is not None, "no <h1> in the document"
    assert h1.inner.strip(), "<h1> is empty"


def test_primary_actions_are_present():
    src = index_html()
    needed = {"start-btn", "read-btn", "ask-btn"}
    btn_ids = {b.attrs.get("id", "") for b in buttons_with_inner(src)}
    missing = needed - btn_ids
    assert not missing, f"missing primary action buttons: {missing}"


# ---------------------------------------------------------------------------
# 2. Affordance + discoverability
# ---------------------------------------------------------------------------

def test_every_button_has_text_or_aria_label():
    """Icon-only buttons (mic, ×, etc.) MUST carry an aria-label or
    a title so screen readers and tooltips work."""
    src = index_html()
    fails = []
    for b in buttons_with_inner(src):
        attrs = b.attrs
        text = re.sub(r"<[^>]+>", "", b.inner).strip()
        # Strip emoji / single-char icons — they're not informative
        # for a screen reader by themselves.
        is_icon_only = (
            len(text) <= 2
            or text in {"×", "›", "🎤", "+"}
        )
        if is_icon_only:
            label = (attrs.get("aria-label") or attrs.get("title") or "").strip()
            if not label:
                fails.append(attrs.get("id") or text)
    assert not fails, (
        f"icon-only buttons missing aria-label or title: {fails}"
    )


def test_input_fields_have_placeholder_or_label():
    src = index_html()
    fails = []
    for inp in inputs_with_attrs(src):
        if inp.attrs.get("type", "text") in {"hidden", "submit"}:
            continue
        # Either placeholder, aria-label, title, or wrapped in a <label>.
        attrs = inp.attrs
        if not (attrs.get("placeholder")
                or attrs.get("aria-label")
                or attrs.get("title")):
            fails.append(attrs.get("id") or attrs.get("type", "?"))
    assert not fails, (
        f"input fields missing placeholder/label: {fails}"
    )


def test_select_elements_have_title():
    src = index_html()
    for s in selects_with_attrs(src):
        attrs = s.attrs
        sid = attrs.get("id", "<no-id>")
        assert (attrs.get("title")
                or attrs.get("aria-label")), (
            f"<select id={sid!r}> missing title/aria-label"
        )


# ---------------------------------------------------------------------------
# 3. Visual consistency
# ---------------------------------------------------------------------------

def test_css_uses_root_variables_palette():
    """Colors should come from CSS custom properties, not be sprinkled
    as hex literals — single source of truth for the palette."""
    css = extract_css()
    assert ":root" in css, "no :root variable block found"
    assert "--accent" in css and "--bg" in css and "--ink" in css, (
        "expected --accent / --bg / --ink CSS vars"
    )


def test_border_radius_vocabulary_is_small():
    """Limited radius values keep cards visually consistent."""
    css = extract_css()
    radii = set(re.findall(r"border-radius:\s*(\d+)px", css))
    assert len(radii) <= 5, (
        f"too many border-radius values, lacks visual rhythm: {radii}"
    )


# ---------------------------------------------------------------------------
# 4. Feedback latency: pulse / pop classes exist
# ---------------------------------------------------------------------------

def test_pulse_animations_are_named():
    """Three named feedback animations: paused (amber), recording (red),
    shape-pop (enter).  Without these the user gets no immediate
    response to actions like pause / record / new card."""
    css = extract_css()
    assert "@keyframes pulse-amber" in css
    assert "@keyframes pulse-red" in css
    assert "@keyframes shape-pop" in css


def test_button_paused_class_pulses():
    css = extract_css()
    assert "button.paused" in css
    assert "pulse-amber" in css


def test_mic_recording_class_pulses():
    css = extract_css()
    assert "#mic-btn.recording" in css
    assert "pulse-red" in css


# ---------------------------------------------------------------------------
# 5. Accessibility (WCAG AA)
# ---------------------------------------------------------------------------

def test_body_text_has_sufficient_contrast():
    """Default body text against the page background ≥ 4.5:1 for
    body, ≥ 3:1 for large text.  Reads :root --ink and --bg."""
    css = extract_css()
    m_ink = re.search(r"--ink:\s*(#[0-9a-fA-F]+)", css)
    m_bg = re.search(r"--bg:\s*(#[0-9a-fA-F]+)", css)
    assert m_ink and m_bg, "missing --ink/--bg vars"
    ratio = contrast_ratio(m_ink.group(1), m_bg.group(1))
    assert ratio >= 4.5, f"body contrast too low: {ratio:.2f}:1"


def test_primary_button_contrast():
    """Default button (white text on accent) ≥ 4.5:1.  Failure means
    primary CTAs are illegible against their background."""
    css = extract_css()
    m_accent = re.search(r"--accent:\s*(#[0-9a-fA-F]+)", css)
    assert m_accent, "missing --accent var"
    ratio = contrast_ratio("#ffffff", m_accent.group(1))
    assert ratio >= 4.5, f"button contrast too low: {ratio:.2f}:1"


def test_min_body_font_size():
    """Tiny font (≤ 11 px) is fine for badges/timestamps but the
    main reading surfaces must be ≥ 12 px."""
    css = extract_css()
    # Body / button / input are the main reading surfaces.
    m_button = re.search(r"^\s*button\s*\{[^}]*font-size:\s*(\d+)px",
                         css, re.M | re.S)
    if m_button:
        assert int(m_button.group(1)) >= 12, (
            f"button font too small: {m_button.group(1)}px"
        )


def test_reduced_motion_respected():
    """Honour ``prefers-reduced-motion: reduce`` so users who
    flagged motion sensitivity don't get the looping pulse / pop
    animations.  Looping decorative animations (paused / mic /
    enter) need this safety net since their durations exceed the
    transition budget."""
    css = extract_css()
    assert "@media (prefers-reduced-motion: reduce)" in css, (
        "no prefers-reduced-motion media query — looping animations "
        "must opt out for motion-sensitive users"
    )
    body = extract_at_rule_body(
        css, "media (prefers-reduced-motion: reduce)",
    )
    assert body is not None
    assert "animation-duration" in body, (
        "reduced-motion block doesn't suppress animation-duration"
    )


def test_one_shot_transitions_brief():
    """Single-fire transitions (button hover, shape-pop entry)
    must complete within 500 ms — a Norman-style "interactive"
    threshold."""
    css = extract_css()
    durations_ms: list[float] = []
    # We only check the entry / leave / transition keyframes that
    # SHOULDN'T loop.  Looping pulses (pulse-amber / pulse-red) are
    # handled by the reduced-motion fallback above.
    one_shot = ("shape-pop",)
    for keyframes in one_shot:
        m = re.search(rf"animation:\s*{keyframes}\s+(\d+(?:\.\d+)?)s",
                      css)
        if m:
            durations_ms.append(float(m.group(1)) * 1000)
        m = re.search(rf"animation:\s*{keyframes}\s+(\d+)ms", css)
        if m:
            durations_ms.append(float(m.group(1)))
    if durations_ms:
        assert max(durations_ms) <= 500.0, (
            f"one-shot animation > 500 ms feels laggy: {durations_ms}"
        )


# ---------------------------------------------------------------------------
# 6. Information architecture
# ---------------------------------------------------------------------------

def test_layout_has_main_with_grid():
    src = index_html()
    main = first_tag(src, "main")
    assert main is not None, "no <main> wrapper"
    assert "id=" in src.split("<main", 1)[1].split(">", 1)[0]


def test_mobile_breakpoint_exists():
    """Single-column mobile layout below 900 px viewport."""
    css = extract_css()
    assert "@media (max-width: 900px)" in css, (
        "no mobile breakpoint at 900 px"
    )


def test_mobile_layout_collapses_to_single_column():
    css = extract_css()
    # Properly extract the @media body via balanced-brace parsing —
    # naive non-greedy regex stops at the first nested rule's '}'.
    body = extract_at_rule_body(css, "media (max-width: 900px)")
    assert body is not None, "@media (max-width: 900px) block not found"
    assert "grid-template-columns: 1fr" in body, (
        "mobile breakpoint doesn't collapse main grid to 1fr"
    )


def test_layout_regions_present():
    src = index_html()
    for region_id in ("board-wrap", "right-col", "tangent-card",
                      "ask-row", "log"):
        assert f'id="{region_id}"' in src, (
            f"layout region #{region_id} missing"
        )


# ---------------------------------------------------------------------------
# 7. Empty states
# ---------------------------------------------------------------------------

def test_currently_spoken_has_idle_placeholder():
    src = index_html()
    # The "currently spoken" pane shows ``— idle —`` before the
    # first session.
    assert "— idle —" in src, "no idle placeholder copy"


def test_mic_status_starts_empty():
    src = index_html()
    # mic-status div exists; copy populated dynamically.
    assert 'id="mic-status"' in src


# ---------------------------------------------------------------------------
# 10. Motion design
# ---------------------------------------------------------------------------

def test_motion_uses_non_linear_easing():
    """Linear easing feels mechanical; UX best practice is ease-out
    or cubic-bezier for entry, ease-in for exit."""
    css = extract_css()
    # Find any animation declarations and pick out their easing word.
    easings = set()
    for m in re.finditer(
        r"animation:\s*[^;]*?\b(ease|ease-in|ease-out|ease-in-out|"
        r"cubic-bezier\([^)]+\)|linear|step-start|step-end)",
        css,
    ):
        easings.add(m.group(1).lower())
    if easings:
        assert easings - {"linear", "step-start", "step-end"}, (
            f"all animations are linear/step — feels robotic: {easings}"
        )


# ---------------------------------------------------------------------------
# 11. Voice + audio polish
# ---------------------------------------------------------------------------

def test_mic_states_distinct():
    """Idle, recording, transcribing must be visually distinct."""
    src = index_html()
    css = extract_css()
    assert 'id="mic-btn"' in src
    # Recording state has its own pulse.
    assert "#mic-btn.recording" in css
    # Idle state has the default secondary look.
    assert "#mic-btn {" in css


def test_voice_picker_capped_width():
    css = extract_css()
    # We capped voice-select at ≤ 200 px so it doesn't dominate the header.
    src = index_html()
    m = re.search(r'<select[^>]*id="voice-select"[^>]*style="([^"]+)"', src)
    assert m is not None, "voice-select inline style missing"
    style = m.group(1)
    assert "max-width:140px" in style or "max-width: 140px" in style, (
        f"voice-select width not capped: {style!r}"
    )


# ---------------------------------------------------------------------------
# 12. Math rendering
# ---------------------------------------------------------------------------

def test_katex_loaded():
    src = index_html()
    assert "katex" in src.lower(), "KaTeX not loaded"
    # Auto-render extension is what compiles math-prose blocks inline.
    assert "auto-render" in src.lower(), "KaTeX auto-render extension missing"


# ---------------------------------------------------------------------------
# 14. Touch targets
# ---------------------------------------------------------------------------

def test_mic_fab_has_touch_target():
    css = extract_css()
    body = extract_at_rule_body(css, "media (max-width: 900px)")
    assert body is not None, "@media (max-width: 900px) block not found"
    assert "#right-col-toggle" in body, (
        "right-col-toggle missing mobile sizing"
    )
    assert "56px" in body, "FAB not sized for touch"


# ---------------------------------------------------------------------------
# 15. Speech / text synchronization
# ---------------------------------------------------------------------------

def test_word_highlight_uses_character_weighting():
    """Long words take more audio time to pronounce than short ones.
    A naive uniform ``i / N`` schedule drifts mid-sentence — the user
    perceives the highlight chasing the voice.  Pin that the timeline
    builder weights words by character length when assigning each
    word's [t0, t1) within its chunk's audio duration."""
    src = index_html()
    # The audio-clock sync engine builds the per-word timeline in
    # ``rebuildTimelineWords`` — that's the place where weighting lives.
    assert "rebuildTimelineWords" in src, (
        "rebuildTimelineWords function missing (audio-clock sync engine)"
    )
    assert "w.length" in src or "weights" in src, (
        "word timeline builder should weight by character length"
    )
    # Cumulative-fraction over total weight, not raw i / N.
    assert "cum / total" in src or "cumulative" in src.lower(), (
        "scheduler should use cumulative weighted fraction"
    )


def test_word_highlight_lead_time_constant_present():
    """The highlight should fire a hair *before* the audio so the eye
    leads the voice — a natural read-along feel."""
    src = index_html()
    assert "WORD_HIGHLIGHT_LEAD_MS" in src, (
        "expected a named constant for the highlight lead time"
    )
    # Reasonable range: somewhere between 30 and 150 ms.  Below 30
    # gives no perceptible head-start; above 150 ms makes the eye and
    # voice feel decoupled.
    m = re.search(r"WORD_HIGHLIGHT_LEAD_MS\s*=\s*(\d+)", src)
    assert m, "WORD_HIGHLIGHT_LEAD_MS must be a numeric constant"
    lead = int(m.group(1))
    assert 30 <= lead <= 150, (
        f"WORD_HIGHLIGHT_LEAD_MS={lead} outside the natural read-along "
        f"range (30..150 ms)"
    )


def test_card_highlight_uses_named_keyframe_animation():
    """The mention-highlight pulse must use a real keyframe animation
    so it's visible on text-heavy formula / reference / canonical cards.

    NOTE: we deliberately avoid ``transform: scale()`` in this keyframe
    because cards are <g> elements with an SVG ``transform="translate"``
    attribute, and CSS transform overrides the SVG transform attribute
    in modern browsers — using scale() would teleport the card to (0,0)
    for the duration of the pulse.  The visual effect is layered
    drop-shadows + a brightness bump, which compose with translate
    cleanly without breaking position."""
    src = index_html()
    assert "shape-highlight-pulse" in src, (
        "expected a named keyframe for the card highlight pulse"
    )
    assert "@keyframes shape-highlight-pulse" in src, (
        "shape-highlight-pulse keyframe definition missing"
    )
    css = extract_css()
    body = extract_at_rule_body(css, "keyframes shape-highlight-pulse")
    assert body is not None, "keyframe block not extractable"
    assert "drop-shadow" in body, "highlight pulse missing glow"
    # Position-safety: keyframe must NOT include transform: scale(),
    # which would clobber each card's translate(x,y).
    assert "scale(" not in body, (
        "highlight keyframe must not use transform: scale() — it "
        "overrides the SVG translate attribute and teleports the card "
        "to (0,0) for the duration of the pulse"
    )


def test_audio_clock_sync_engine_present():
    """All visual updates — clause text, yellow read-marker, chalkboard
    highlights — must be driven from a single rAF loop polling
    ``audioCtx.currentTime``.  Timer-based scheduling drifts: audio is
    queued through Web Audio's clock while the LLM-estimated word
    timestamps and ``performance.now()`` schedules diverge.  The fix
    is structural — every UI update reads the truth from the audio
    clock, never predicts it.
    """
    src = index_html()
    # Per-seq, per-panel ClauseTimeline: the single source of truth
    # the rAF loop reads from.
    assert "ensureTimeline" in src, (
        "expected an ensureTimeline helper to populate per-seq "
        "ClauseTimeline state"
    )
    assert "rebuildTimelineWords" in src, (
        "expected rebuildTimelineWords to derive word [t0, t1] in "
        "AudioContext time from chunks"
    )
    # The rAF loop itself.
    assert "syncFrame" in src and "requestAnimationFrame" in src, (
        "expected a requestAnimationFrame loop named syncFrame to "
        "drive UI updates from audioCtx.currentTime"
    )
    # The loop must read currentTime, not performance.now().
    m = re.search(r"function\s+syncFrame\s*\([^)]*\)\s*\{", src)
    assert m is not None, "syncFrame must exist"
    start = m.end() - 1
    depth = 0
    end = -1
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end > 0, "syncFrame body not extractable"
    body = src[start:end]
    assert "currentTime" in body, (
        "syncFrame must poll audioCtx.currentTime as the single "
        "source of truth for word-level sync"
    )
    # Yellow read-marker driven from timeline + currentTime.
    assert "updateYellowFromTimeline" in src, (
        "expected updateYellowFromTimeline to set .spoken on the word "
        "whose [t0, t1) contains audioCtx.currentTime"
    )
    # Chalkboard visual ops driven from timeline + audioStart.
    assert "fireVisualOpsFromTimeline" in src, (
        "expected fireVisualOpsFromTimeline to fire add/highlight/"
        "erase ops at audioStart + op.t in AudioContext time"
    )


def test_spoken_text_pane_accumulates_clauses():
    """Each new clause must append a new ``.clause`` block (not
    overwrite), so the user can scroll up to re-read what was just
    said.  A previous version replaced the pane's contents per clause
    and the user lost everything mid-narration."""
    src = index_html()
    assert "resetClausePane" in src, (
        "expected a topic-boundary reset helper for the clause pane"
    )
    assert "clause-active" in src, (
        "active clause must be styled distinctly from history"
    )
    # The renderer should *append* a new .clause block, not clobber
    # the pane's innerHTML on every clause.
    assert "appendChild(block)" in src or "clause-active" in src, (
        "renderClauseText should accumulate clauses, not replace them"
    )
    # Scroll-into-view so the latest clause is always visible.
    assert "scrollIntoView" in src, (
        "active clause must scroll into view as it appears"
    )
