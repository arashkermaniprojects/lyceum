"""Browser driver helpers for the autonomous inspector agent.

These wrappers hide the Playwright boilerplate so the agent can call
``snap_baseline()`` / ``snap_after_question(q)`` and reason about the
returned screenshot path.

Every helper restarts/keeps a fresh ``http://127.0.0.1:8001/`` session
so each iteration of the inspect-improve loop starts from a clean
slate.  Screenshots are saved under /tmp/sevim_inspect/.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from playwright.sync_api import Page, sync_playwright


SCREEN_DIR = Path('/tmp/sevim_inspect')
SCREEN_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = 'http://127.0.0.1:8001/'


def _wait_settled(page: Page, ms: int = 2000) -> None:
    """Wait for the page to settle: any pending fetches done + a small
    grace so cards finish their CSS transitions before the screenshot.
    """
    try:
        page.wait_for_load_state('networkidle', timeout=ms)
    except Exception:
        pass
    page.wait_for_timeout(ms)


def _new_browser():
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={'width': 1920, 'height': 1080})
    return pw, browser, ctx


def screenshot_baseline(name: str = 'baseline.png') -> str:
    """Open the page fresh and screenshot the idle state."""
    out = SCREEN_DIR / name
    pw, browser, ctx = _new_browser()
    try:
        page = ctx.new_page()
        page.goto(BASE_URL, wait_until='domcontentloaded', timeout=15000)
        _wait_settled(page, 1500)
        page.screenshot(path=str(out), full_page=True)
    finally:
        browser.close()
        pw.stop()
    return str(out)


def screenshot_after_read_book(
    name: str = 'after_read_book.png', wait_s: float = 25.0,
) -> str:
    """Click the 'Read book' button and wait for the lecture to fill the
    chalkboard with a few cards before screenshotting.
    """
    out = SCREEN_DIR / name
    pw, browser, ctx = _new_browser()
    try:
        page = ctx.new_page()
        page.goto(BASE_URL, wait_until='domcontentloaded', timeout=15000)
        _wait_settled(page, 1000)
        # The Read book button is a header <button>.  Locator by text.
        btn = page.get_by_role('button', name='Read book', exact=False)
        if btn.count() == 0:
            btn = page.locator('button:has-text("Read book")')
        btn.first.click()
        # Give the lecture time to start synthing + emit a few clauses.
        page.wait_for_timeout(int(wait_s * 1000))
        page.screenshot(path=str(out), full_page=True)
    finally:
        browser.close()
        pw.stop()
    return str(out)


def screenshot_after_question(
    question: str, *,
    name: str = 'after_question.png',
    pre_wait_s: float = 12.0,
    post_wait_s: float = 25.0,
) -> str:
    """Start a Read-book session, wait for it to lay down a few cards,
    then ask *question* via the Ask button.  Screenshot after the
    answer has had time to materialise in the tangent panel.

    This is the main scenario the user keeps re-testing: in-session
    question while a main lecture is running.
    """
    out = SCREEN_DIR / name
    pw, browser, ctx = _new_browser()
    try:
        page = ctx.new_page()
        page.goto(BASE_URL, wait_until='domcontentloaded', timeout=15000)
        _wait_settled(page, 1000)
        # Start a main session.
        read_btn = page.locator('button:has-text("Read book")')
        if read_btn.count() == 0:
            read_btn = page.get_by_role('button', name='Read book')
        read_btn.first.click()
        page.wait_for_timeout(int(pre_wait_s * 1000))
        # Type the question, click Ask.
        ask_input = page.locator('#ask-input')
        ask_input.fill(question)
        ask_btn = page.locator('#ask-btn')
        ask_btn.click()
        page.wait_for_timeout(int(post_wait_s * 1000))
        page.screenshot(path=str(out), full_page=True)
    finally:
        browser.close()
        pw.stop()
    return str(out)


def screenshot_no_session_question(
    question: str, *,
    name: str = 'no_session_question.png',
    wait_s: float = 30.0,
) -> str:
    """Ask without starting a main session first — exercises the
    /api/answer path that builds main_orch from the question's section.
    """
    out = SCREEN_DIR / name
    pw, browser, ctx = _new_browser()
    try:
        page = ctx.new_page()
        page.goto(BASE_URL, wait_until='domcontentloaded', timeout=15000)
        _wait_settled(page, 1000)
        ask_input = page.locator('#ask-input')
        ask_input.fill(question)
        ask_btn = page.locator('#ask-btn')
        ask_btn.click()
        page.wait_for_timeout(int(wait_s * 1000))
        page.screenshot(path=str(out), full_page=True)
    finally:
        browser.close()
        pw.stop()
    return str(out)


def collect_console_errors(
    url: str = BASE_URL, *, wait_s: float = 5.0,
) -> list[str]:
    """Open the page and collect any console.error / page errors.
    Useful for catching JS exceptions the screenshot wouldn't reveal.
    """
    errs: list[str] = []
    pw, browser, ctx = _new_browser()
    try:
        page = ctx.new_page()
        page.on('pageerror', lambda exc: errs.append(f'PAGE: {exc}'))
        page.on('console', lambda msg: (
            errs.append(f'CONSOLE {msg.type}: {msg.text}')
            if msg.type in ('error', 'warning') else None
        ))
        page.goto(url, wait_until='domcontentloaded', timeout=15000)
        page.wait_for_timeout(int(wait_s * 1000))
    finally:
        browser.close()
        pw.stop()
    return errs


def restart_server(book_path: str = 'books/ESLII.json',
                   port: int = 8001) -> None:
    """Kill any running server, start a fresh one, wait for HTTP 200."""
    import subprocess
    subprocess.run(['pkill', '-f', 'serve.server books'], check=False)
    time.sleep(1.0)
    log = open('/tmp/sevim_server.log', 'w')
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    cwd = '/home/ara/Documents/Programming/sevim_math'
    subprocess.Popen(
        [f'{cwd}/.venv/bin/python3', '-m', 'serve.server',
         book_path, '--port', str(port)],
        stdout=log, stderr=log, cwd=cwd, env=env,
        start_new_session=True,
    )
    # Poll for readiness.
    import urllib.request
    for _ in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/',
                                        timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            continue
    raise RuntimeError('server failed to come up on port {port}')


if __name__ == '__main__':
    # Quick CLI for sanity checks.
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
    if cmd == 'baseline':
        print(screenshot_baseline())
    elif cmd == 'read':
        print(screenshot_after_read_book())
    elif cmd == 'ask':
        q = sys.argv[2] if len(sys.argv) > 2 else 'what is a gradient'
        print(screenshot_after_question(q))
    elif cmd == 'no-session':
        q = sys.argv[2] if len(sys.argv) > 2 else 'what is a gradient'
        print(screenshot_no_session_question(q))
    elif cmd == 'errors':
        for e in collect_console_errors():
            print(e)
    elif cmd == 'restart':
        restart_server()
        print('restarted')
    else:
        print(f'unknown cmd {cmd!r}; '
              f'use baseline|read|ask|no-session|errors|restart')
