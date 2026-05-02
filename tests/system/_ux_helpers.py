"""Shared helpers for UX-oriented tests.

Parses ``serve/static/index.html`` once per test session and exposes
the structured artefacts the tests need: CSS rules, the body DOM,
button + input + style metadata.  Hand-rolled parsing so tests don't
depend on heavy dependencies like BeautifulSoup.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional


_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "serve", "static", "index.html",
)


def index_html() -> str:
    """Return the served index.html source."""
    with open(_INDEX_PATH, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# CSS extraction
# ---------------------------------------------------------------------------

_CSS_BLOCK_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)


def extract_css() -> str:
    src = index_html()
    parts = _CSS_BLOCK_RE.findall(src)
    return "\n".join(parts)


def parse_css_rules(css: str) -> list[tuple[str, dict[str, str]]]:
    """Strip comments + at-rules and return ``[(selector, decls), …]``.

    Naive parser — enough for sniffing major rules; doesn't fully
    handle nested at-rules.  We unwrap simple ``@media`` blocks so
    media-scoped rules show up too (with the selector prefixed by
    ``@media …`` for visibility).
    """
    # Strip comments.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out: list[tuple[str, dict[str, str]]] = []
    pos = 0
    n = len(css)

    def _parse_block(prefix: str, body: str) -> None:
        # ``body`` is the inside of a brace pair; may contain nested
        # rules.  For our purposes we just split on rule boundaries.
        i = 0
        m = len(body)
        while i < m:
            # Find selector up to '{'.
            brace = body.find("{", i)
            if brace < 0:
                break
            sel = body[i:brace].strip()
            # Find matching '}'.
            depth = 1
            j = brace + 1
            while j < m and depth > 0:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                j += 1
            inner = body[brace + 1:j - 1]
            full_selector = (prefix + " " + sel).strip() if prefix else sel
            if "{" in inner:
                # Nested at-rule — recurse.
                _parse_block(full_selector, inner)
            else:
                decls: dict[str, str] = {}
                for part in inner.split(";"):
                    part = part.strip()
                    if not part or ":" not in part:
                        continue
                    k, v = part.split(":", 1)
                    decls[k.strip()] = v.strip()
                out.append((full_selector, decls))
            i = j

    _parse_block("", css)
    return out


# ---------------------------------------------------------------------------
# Color contrast (WCAG)
# ---------------------------------------------------------------------------

def _hex_to_rgb(s: str) -> Optional[tuple[int, int, int]]:
    s = s.strip().lower()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return None
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None


def _rgb_to_lum(rgb: tuple[int, int, int]) -> float:
    """sRGB → relative luminance (WCAG)."""
    def _channel(c: int) -> float:
        cs = c / 255.0
        return cs / 12.92 if cs <= 0.03928 else ((cs + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    fg = _hex_to_rgb(fg_hex)
    bg = _hex_to_rgb(bg_hex)
    if fg is None or bg is None:
        return 0.0
    L1 = _rgb_to_lum(fg)
    L2 = _rgb_to_lum(bg)
    bright = max(L1, L2)
    dark = min(L1, L2)
    return (bright + 0.05) / (dark + 0.05)


# ---------------------------------------------------------------------------
# DOM helpers
# ---------------------------------------------------------------------------

@dataclass
class TagInfo:
    tag: str
    attrs: dict[str, str]
    inner: str


_TAG_OPEN_RE = re.compile(
    r"<([a-zA-Z][a-zA-Z0-9-]*)\b([^>]*?)(/?)>",
)


def parse_tag_attrs(attr_str: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(
        r'([a-zA-Z][a-zA-Z0-9-]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))',
        attr_str,
    ):
        key = m.group(1).lower()
        val = m.group(2) or m.group(3) or m.group(4) or ""
        out[key] = val
    # Boolean attrs (no equals sign).
    for m in re.finditer(r'(?:^|\s)([a-zA-Z][a-zA-Z0-9-]*)(?=\s|$|/)',
                         attr_str):
        key = m.group(1).lower()
        out.setdefault(key, key)
    return out


def find_all_tags(src: str, tag: str) -> list[TagInfo]:
    """Return every occurrence of ``<tag …>`` (open + self-closing) with
    attrs and inner text up to the matching close (best-effort)."""
    tag_lc = tag.lower()
    out: list[TagInfo] = []
    pos = 0
    while True:
        m = _TAG_OPEN_RE.search(src, pos)
        if not m:
            break
        if m.group(1).lower() != tag_lc:
            pos = m.end()
            continue
        attrs = parse_tag_attrs(m.group(2))
        self_close = m.group(3) == "/"
        if self_close or tag_lc in {"input", "img", "br", "meta", "link"}:
            out.append(TagInfo(tag=tag_lc, attrs=attrs, inner=""))
            pos = m.end()
            continue
        # Search for closing tag.
        close = re.search(rf"</\s*{re.escape(tag_lc)}\s*>",
                          src[m.end():], re.I)
        if close:
            inner = src[m.end():m.end() + close.start()]
            out.append(TagInfo(tag=tag_lc, attrs=attrs, inner=inner))
            pos = m.end() + close.end()
        else:
            out.append(TagInfo(tag=tag_lc, attrs=attrs, inner=""))
            pos = m.end()
    return out


def first_tag(src: str, tag: str) -> Optional[TagInfo]:
    tags = find_all_tags(src, tag)
    return tags[0] if tags else None


def buttons_with_inner(src: str) -> list[TagInfo]:
    """Every <button> in the document with inner text/markup."""
    return find_all_tags(src, "button")


def inputs_with_attrs(src: str) -> list[TagInfo]:
    return find_all_tags(src, "input")


def selects_with_attrs(src: str) -> list[TagInfo]:
    return find_all_tags(src, "select")


def extract_at_rule_body(css: str, prelude: str) -> Optional[str]:
    """Return the body of ``@<prelude> { … }`` with proper brace
    matching.  ``prelude`` must include the at-rule name and any
    parameters but NOT the opening brace (e.g. ``"media (max-width: 900px)"``).
    Returns ``None`` if not found.

    Necessary because a naive non-greedy regex stops at the first
    nested ``}``.
    """
    needle = "@" + prelude
    idx = css.find(needle)
    if idx < 0:
        return None
    open_brace = css.find("{", idx)
    if open_brace < 0:
        return None
    depth = 1
    j = open_brace + 1
    n = len(css)
    while j < n and depth > 0:
        if css[j] == "{":
            depth += 1
        elif css[j] == "}":
            depth -= 1
        j += 1
    if depth != 0:
        return None
    return css[open_brace + 1:j - 1]
