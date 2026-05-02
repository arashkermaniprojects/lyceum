"""FastAPI shim exposing sevim.run_pipeline over HTTP.

Launch:
    uvicorn service.app:app --host 127.0.0.1 --port 8003

Endpoints:
    GET  /health                 liveness probe
    GET  /ontology               list the 12 relations + visual patterns
    POST /render                 text → SVG + IR + trace (stateless)
    POST /render/session/{sid}   text → SVG (extends the session's prior graph)
    DELETE /session/{sid}        clear a session

All bodies are JSON. Intended for same-machine use: bind to 127.0.0.1.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sevim.ir import SceneGraph  # noqa: E402
from sevim.pipeline import run_pipeline  # noqa: E402
from sevim.s3_map import _RELATION_PATTERN  # noqa: E402


app = FastAPI(
    title="SeVim",
    description="Deterministic semantic-to-visual mapping. "
                "Exposes sevim.run_pipeline over HTTP.",
    version="0.2.0",
)

_sessions: dict[str, SceneGraph] = {}
_sessions_lock = Lock()


class RenderReq(BaseModel):
    text: str
    utterance_id: str = "u0"


class RenderResp(BaseModel):
    svg: str
    ir: dict
    trace: list[dict]


def _serialize(result) -> RenderResp:
    return RenderResp(
        svg=result.svg,
        ir={
            "revision": result.graph.revision,
            "nodes": [asdict(n) for n in result.graph.nodes],
            "edges": [asdict(e) for e in result.graph.edges],
        },
        trace=[
            {"stage": t.stage, "message": t.message, "refs": t.refs}
            for t in result.trace
        ],
    )


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(_sessions)}


@app.get("/ontology")
def ontology():
    return {
        "version": "v2",
        "relations": [
            {"relation": rel, "visual_pattern": pat}
            for rel, pat in _RELATION_PATTERN.items()
        ],
    }


@app.post("/render", response_model=RenderResp)
def render(req: RenderReq):
    if not req.text.strip():
        raise HTTPException(400, "text must be non-empty")
    result = run_pipeline(req.text, utterance_id=req.utterance_id)
    return _serialize(result)


@app.post("/render/session/{sid}", response_model=RenderResp)
def render_session(sid: str, req: RenderReq):
    if not req.text.strip():
        raise HTTPException(400, "text must be non-empty")
    with _sessions_lock:
        prev = _sessions.get(sid)
    result = run_pipeline(req.text, utterance_id=sid, graph=prev)
    with _sessions_lock:
        _sessions[sid] = result.graph
    return _serialize(result)


@app.delete("/session/{sid}")
def clear_session(sid: str):
    with _sessions_lock:
        existed = _sessions.pop(sid, None) is not None
    return {"cleared": existed, "sid": sid}
