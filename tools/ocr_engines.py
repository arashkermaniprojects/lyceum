"""DEPRECATED — kept for import-compatibility.

The pipeline now uses the local Qwen-VL endpoint exclusively (see
``tools.vlm_engine``).  No OCR-specific models (nougat, pix2tex) are
loaded or invoked.  This module re-exports a no-op shim so older
imports do not crash.
"""
from __future__ import annotations


class NullEngine:
    def ocr(self, pdf_path: str, page_index: int, eq_hits) -> dict:
        return {h.label: "" for h in eq_hits}


# Back-compat aliases — both now route to the VLM engine.
class NougatEngine(NullEngine):
    def __init__(self) -> None:
        raise RuntimeError(
            "nougat path is deprecated; use tools.vlm_engine.VLMEngine "
            "(local Qwen-VL at $VLM_BASE_URL)."
        )


class Pix2TexEngine(NullEngine):
    def __init__(self) -> None:
        raise RuntimeError(
            "pix2tex path is deprecated; use tools.vlm_engine.VLMEngine "
            "(local Qwen-VL at $VLM_BASE_URL)."
        )
