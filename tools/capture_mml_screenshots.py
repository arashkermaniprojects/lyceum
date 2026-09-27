"""Capture illustrative UI screenshots of Lyceum running against the MML
corpus (Mathematics for Machine Learning, Deisenroth/Faisal/Ong, 2020).

The original ESLII-based screenshots reproduced copyrighted body-text
in their UI; MML serves the same illustrative purpose.  The MML PDF is
still copyrighted (its free author-hosted copy is for personal use), so
screenshots produced here should not be redistributed.

Outputs:
    screenshots/mml_chapter_zoom_shot.png   — Ch 9 Linear Regression
    screenshots/mml_tangent_shot.png        — main + tangent
    screenshots/mml_reference_card.png      — passage card + ref
    screenshots/mml_live_clustered.png      — t≈180s SVM narration
    screenshots/mml_live_late.png           — t≈380s SVM narration

Assumes the HTTP server is up on http://127.0.0.1:8001/ with MML.json,
that vLLM (8000 text + 8003 embed) is reachable, and Kokoro TTS is
configured.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / 'screenshots'   # gitignored output directory
BASE_URL = 'http://127.0.0.1:8001/'
VIEWPORT = {'width': 1920, 'height': 1080}


def _make_page(pw):
    browser = pw.chromium.launch(headless=True)
    ctx = browser.new_context(viewport=VIEWPORT)
    page = ctx.new_page()
    page.goto(BASE_URL, wait_until='domcontentloaded', timeout=20000)
    try:
        page.wait_for_load_state('networkidle', timeout=4000)
    except Exception:
        pass
    page.wait_for_timeout(1500)
    return browser, ctx, page


def _start_topic(page, topic: str) -> None:
    """Drive the hidden #topic + #start-btn pair to fire a main-board
    lecture (passage + formula + reference cards)."""
    page.evaluate(
        """(t) => {
            const ti = document.getElementById('topic');
            const sb = document.getElementById('start-btn');
            if (!ti || !sb) throw new Error('topic/start-btn missing');
            ti.value = t;
            sb.click();
        }""",
        topic,
    )


def cap_chapter_zoom(out: Path) -> None:
    """Click MML Chapter 9 (Linear Regression) in the TOC and capture
    the chapter-zoom layout once the punch-line clause has played."""
    print(f'[chapter] capturing -> {out.name}')
    with sync_playwright() as pw:
        browser, ctx, page = _make_page(pw)
        try:
            page.locator('.toc-node').first.click()
            page.wait_for_timeout(500)
            chapters = page.locator('.toc-chapter')
            n = chapters.count()
            if n == 0:
                raise RuntimeError('no .toc-chapter rows found')
            print(f'  TOC has {n} chapter rows')
            # MML structure: parts/sections at the top.  Chapter 9 sits in
            # part II.  Try a few indices and pick whichever scrolls into
            # view fastest.  Indices 8 (Ch 9) is the principal target.
            target = chapters.nth(min(8, n - 1))
            target.scroll_into_view_if_needed()
            target.click()
            page.wait_for_timeout(20000)
            page.screenshot(path=str(out), full_page=False)
        finally:
            browser.close()


def cap_board_with_cards(out: Path, topic: str, wait_s: float) -> None:
    """Generic board capture during a topic narration.  Used for
    reference-card and live-late captures."""
    print(f'[board:{topic}] capturing -> {out.name}')
    with sync_playwright() as pw:
        browser, ctx, page = _make_page(pw)
        try:
            _start_topic(page, topic)
            page.wait_for_timeout(int(wait_s * 1000))
            page.screenshot(path=str(out), full_page=False)
        finally:
            browser.close()


def cap_tangent_panel(out: Path) -> None:
    """Start a topic narration, ask a follow-up while the lecture is
    playing, screenshot mid-answer with the tangent panel still open."""
    print(f'[tangent] capturing -> {out.name}')
    with sync_playwright() as pw:
        browser, ctx, page = _make_page(pw)
        try:
            _start_topic(page, 'linear regression')
            page.wait_for_timeout(18000)
            page.evaluate(
                """(q) => {
                    const ai = document.getElementById('ask-input');
                    const ab = document.getElementById('ask-btn');
                    if (!ai || !ab) throw new Error('ask DOM missing');
                    ai.value = q;
                    ab.click();
                }""",
                'what is regularization',
            )
            try:
                page.wait_for_selector(
                    'main.with-tangent', state='attached', timeout=10000
                )
            except Exception:
                pass
            page.wait_for_timeout(12000)
            page.screenshot(path=str(out), full_page=False)
        finally:
            browser.close()


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ('chapter_zoom',  lambda: cap_chapter_zoom(FIG_DIR / 'mml_chapter_zoom_shot.png')),
        ('reference',     lambda: cap_board_with_cards(
            FIG_DIR / 'mml_reference_card.png',
            topic='linear regression', wait_s=45)),
        ('live_clustered',lambda: cap_board_with_cards(
            FIG_DIR / 'mml_live_clustered.png',
            topic='support vector machines', wait_s=180)),
        ('live_late',     lambda: cap_board_with_cards(
            FIG_DIR / 'mml_live_late.png',
            topic='support vector machines', wait_s=380)),
        ('tangent',       lambda: cap_tangent_panel(FIG_DIR / 'mml_tangent_shot.png')),
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
