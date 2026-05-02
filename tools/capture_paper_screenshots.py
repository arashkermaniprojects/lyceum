"""Capture the 5 screenshots referenced as placeholders in paper/lyceum.tex.

Outputs:
    paper/figures/fig_voice_input.png
    paper/figures/fig_chapter_zoom_shot.png
    paper/figures/fig_tangent_shot.png
    paper/figures/fig_reference_card.png
    paper/figures/fig_theorem_card.png

Assumes the HTTP server is up on http://127.0.0.1:8001/ and that the
text-LLM (port 8000) and embedding (port 8003) are reachable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT = Path('/home/ara/Documents/Programming/sevim_math')
FIG_DIR = PROJECT / 'paper' / 'figures'
BASE_URL = 'http://127.0.0.1:8001/'
VIEWPORT = {'width': 1920, 'height': 1080}


def _make_page(pw, *, mic: bool = False):
    browser = pw.chromium.launch(headless=True)
    perms = ['microphone'] if mic else []
    ctx = browser.new_context(viewport=VIEWPORT, permissions=perms)
    if mic:
        ctx.grant_permissions(['microphone'], origin=BASE_URL)
    page = ctx.new_page()
    page.goto(BASE_URL, wait_until='domcontentloaded', timeout=20000)
    try:
        page.wait_for_load_state('networkidle', timeout=4000)
    except Exception:
        pass
    page.wait_for_timeout(1500)
    return browser, ctx, page


def _click_first(page, *selectors: str) -> bool:
    for sel in selectors:
        loc = page.locator(sel)
        if loc.count():
            loc.first.click()
            return True
    return False


def cap_voice_input(out: Path) -> None:
    """Capture the page right after clicking the mic toggle."""
    print(f'[voice] capturing -> {out.name}')
    with sync_playwright() as pw:
        browser, ctx, page = _make_page(pw, mic=True)
        try:
            mic = page.locator('#mic-btn-header')
            if mic.count() == 0:
                mic = page.locator('#mic-btn')
            mic.first.click()
            # Short delay so the recording-state CSS / pulse activates.
            page.wait_for_timeout(800)
            page.screenshot(path=str(out), full_page=False)
        finally:
            browser.close()


def cap_chapter_zoom(out: Path, chapter_label: str = 'Ch') -> None:
    """Click a chapter in the TOC; wait for the chapter-zoom treemap to
    render and the first clause to start playing.
    """
    print(f'[chapter] capturing -> {out.name}')
    with sync_playwright() as pw:
        browser, ctx, page = _make_page(pw)
        try:
            # Expand the book in the TOC.
            page.locator('.toc-node').first.click()
            page.wait_for_timeout(500)
            chapters = page.locator('.toc-chapter')
            n = chapters.count()
            if n == 0:
                raise RuntimeError('no .toc-chapter rows found in TOC')
            # Pick chapter 5 if available (RKHS, regularization, ...) -
            # else the first chapter.
            target = chapters.nth(min(4, n - 1))
            target.scroll_into_view_if_needed()
            target.click()
            # Let the chapter-zoom render and the punch-line clause play.
            page.wait_for_timeout(20000)
            page.screenshot(path=str(out), full_page=False)
        finally:
            browser.close()


def _start_topic(page, topic: str) -> None:
    """Drive the hidden #topic + #start-btn pair.

    The visible Ask box only opens a tangent-style answer; to get a
    real main-board lecture (passage cards, formula cards, reference
    cards, ...) we have to fire the same code path the legacy
    \"Read book\"/\"Topic\" buttons used.  Both DOM nodes are still
    present, just hidden -- Playwright can poke them via evaluate.
    """
    page.evaluate(
        '''(t) => {
            const ti = document.getElementById('topic');
            const sb = document.getElementById('start-btn');
            if (!ti || !sb) throw new Error('topic/start-btn missing');
            ti.value = t;
            sb.click();
        }''',
        topic,
    )


def cap_tangent_panel(out: Path) -> None:
    """Start a topic narration, ask a follow-up while the lecture is
    playing, screenshot the moment the side panel has streamed in a
    couple of cards but before its audio tail closes it.

    To make the tangent panel visible in the screenshot we need to
    capture *mid-answer* — the panel auto-closes a beat after the
    answer's last clause finishes.  Empirically ~12s after the click
    is the sweet spot: long enough for the question + at least one
    answer card to render, short enough to be before close.
    """
    print(f'[tangent] capturing -> {out.name}')
    with sync_playwright() as pw:
        browser, ctx, page = _make_page(pw)
        try:
            _start_topic(page, 'regularization')
            page.wait_for_timeout(18000)
            page.evaluate(
                '''(q) => {
                    const ai = document.getElementById('ask-input');
                    const ab = document.getElementById('ask-btn');
                    if (!ai || !ab) throw new Error('ask DOM missing');
                    ai.value = q;
                    ab.click();
                }''',
                'what is the bias variance tradeoff',
            )
            # Wait for layout flip into tangent mode.
            try:
                page.wait_for_selector(
                    'main.with-tangent', state='attached', timeout=10000
                )
            except Exception:
                pass
            # Just enough for first answer cards to land.
            page.wait_for_timeout(12000)
            page.screenshot(path=str(out), full_page=False)
        finally:
            browser.close()


def cap_board_with_cards(out: Path, topic: str, wait_s: float) -> None:
    """Generic board capture during a topic narration. Used for both
    reference-card and theorem-card screenshots.
    """
    print(f'[board:{topic}] capturing -> {out.name}')
    with sync_playwright() as pw:
        browser, ctx, page = _make_page(pw)
        try:
            _start_topic(page, topic)
            page.wait_for_timeout(int(wait_s * 1000))
            page.screenshot(path=str(out), full_page=False)
        finally:
            browser.close()


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ('voice_input',   lambda: cap_voice_input(FIG_DIR / 'fig_voice_input.png')),
        ('chapter_zoom',  lambda: cap_chapter_zoom(FIG_DIR / 'fig_chapter_zoom_shot.png')),
        ('reference',     lambda: cap_board_with_cards(
            FIG_DIR / 'fig_reference_card.png',
            topic='regularization', wait_s=45)),
        ('theorem',       lambda: cap_board_with_cards(
            FIG_DIR / 'fig_theorem_card.png',
            topic='reproducing kernel hilbert spaces', wait_s=55)),
        ('tangent',       lambda: cap_tangent_panel(FIG_DIR / 'fig_tangent_shot.png')),
    ]
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        targets = [t for t in targets if t[0] in wanted]
    for name, fn in targets:
        t0 = time.time()
        try:
            fn()
            print(f'  [{name}] ok ({time.time() - t0:.1f}s)')
        except Exception as e:
            print(f'  [{name}] FAIL: {e!r}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
