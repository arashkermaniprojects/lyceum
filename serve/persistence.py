"""Disk-backed persistence for sessions.

Each session is a JSON file under ``$SEVIM_SESSION_DIR`` (default
``~/.sevim/sessions``).  Atomic writes via temp + rename so a server
crash mid-write can never corrupt a session.

A persisted session captures dialogue history, the
:class:`~serve.session.SessionKnowledge`, the chalkboard shape list,
and a focus pointer — enough for a reload to drop the user back into
the conversation with the chalkboard intact.

Used by:
  * the server's ``_register`` hook to install an auto-save callback
    on every new ``Session``;
  * the server's startup scan to rehydrate sessions from prior runs;
  * the ``GET /api/session/<plan_id>`` and ``GET /api/sessions``
    endpoints that the frontend hits after a page reload.

Determinism: pure JSON, no pickling, no serialised lambdas.  Safe to
hand-edit and to ship between machines that share the same book.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from typing import Any, Optional


_SAFE_PLAN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def session_dir() -> str:
    """Directory where session JSON files live.  Created lazily."""
    cfg = os.environ.get("SEVIM_SESSION_DIR")
    if cfg:
        path = os.path.expanduser(cfg)
    else:
        path = os.path.expanduser("~/.sevim/sessions")
    os.makedirs(path, exist_ok=True)
    return path


def _path_for(plan_id: str) -> Optional[str]:
    """Resolve a plan_id to a JSON file path, refusing path-traversal."""
    if not plan_id or not _SAFE_PLAN_ID_RE.match(plan_id):
        return None
    return os.path.join(session_dir(), f"{plan_id}.json")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_session_dict(plan_id: str, snap: dict) -> bool:
    """Write *snap* to ``<session_dir>/<plan_id>.json`` atomically.

    Returns True on success, False on any error (caller logs).
    """
    target = _path_for(plan_id)
    if target is None:
        return False
    snap = dict(snap)
    snap.setdefault("plan_id", plan_id)
    snap["updated_at"] = time.time()
    try:
        # Write to a temp file in the same directory, then rename so
        # the swap is atomic on POSIX.
        d = os.path.dirname(target)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8",
            dir=d, suffix=".tmp", delete=False,
        ) as tmp:
            json.dump(snap, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, target)
        return True
    except Exception as e:
        print(f"[persistence] save failed for {plan_id}: {e}")
        return False


def load_session_dict(plan_id: str) -> Optional[dict]:
    """Load a session snapshot by plan_id; ``None`` if missing/corrupt."""
    target = _path_for(plan_id)
    if target is None or not os.path.isfile(target):
        return None
    try:
        with open(target, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[persistence] load failed for {plan_id}: {e}")
        return None


def list_session_summaries(*, limit: int = 50) -> list[dict]:
    """Return ``[{plan_id, updated_at, dialogue_len, last_focus_topic,
    book_title}, …]`` newest-first, capped at *limit*."""
    d = session_dir()
    out: list[dict] = []
    try:
        names = [n for n in os.listdir(d) if n.endswith(".json")]
    except FileNotFoundError:
        return []
    for name in names:
        path = os.path.join(d, name)
        plan_id = name[:-len(".json")]
        try:
            with open(path, encoding="utf-8") as f:
                snap = json.load(f)
        except Exception:
            continue
        if not isinstance(snap, dict):
            continue
        out.append({
            "plan_id": plan_id,
            "updated_at": snap.get("updated_at", 0.0),
            "dialogue_len": len(snap.get("dialogue_history", []) or []),
            "last_focus_topic": snap.get("last_focus_topic", ""),
            "book_title": snap.get("book_title", ""),
            "book_name": snap.get("book_name", ""),
        })
    out.sort(key=lambda s: -s.get("updated_at", 0))
    return out[:limit]


def delete_session(plan_id: str) -> bool:
    """Remove a session file by plan_id.  Returns True if removed."""
    target = _path_for(plan_id)
    if target is None or not os.path.isfile(target):
        return False
    try:
        os.remove(target)
        return True
    except Exception as e:
        print(f"[persistence] delete failed for {plan_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_old(*, max_age_days: float = 14.0) -> int:
    """Delete sessions whose ``updated_at`` is older than the given
    cutoff.  Returns the number deleted."""
    cutoff = time.time() - max_age_days * 86400
    deleted = 0
    for s in list_session_summaries(limit=10_000):
        if s.get("updated_at", 0.0) < cutoff:
            if delete_session(s["plan_id"]):
                deleted += 1
    return deleted
