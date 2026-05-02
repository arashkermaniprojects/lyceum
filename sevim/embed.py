"""Frozen encoder: mean-pooled last-hidden-state from a local HF model.

Default model is Qwen/Qwen2.5-7B (base, not Instruct), loaded via the
Transformers library. Override with env var SEVIM_EMBED_MODEL. If
transformers/torch are unavailable or the model fails to load, ``encode()``
returns an empty tuple and downstream stages (S2 cosine-merge, S3 numeric
phi) gracefully fall back to label-only / salience-only paths.

Determinism notes: the model is loaded in ``eval()`` mode with gradient
globally disabled; fp16 on CUDA or fp32 on CPU; no dropout at inference;
outputs are reproducible on a fixed machine + model revision.

Set ``SEVIM_STRICT_DET=1`` to pin single-threaded CPU ops.
"""
from __future__ import annotations

import os
import threading

MODEL_NAME = os.environ.get("SEVIM_EMBED_MODEL", "Qwen/Qwen2.5-7B")
MAX_TOKENS = int(os.environ.get("SEVIM_EMBED_MAX_TOKENS", "128"))

_lock = threading.Lock()
_model = None
_tokenizer = None
_device = None
_dim = 0
_tried = False


def _load():
    """Load the encoder model once; subsequent calls return the cached objects.

    Thread-safe: a lock guards the one-time initialisation.  Returns
    (None, None, None) when SEVIM_DISABLE_EMBED=1 or on any load error, so
    callers always get a consistent 3-tuple regardless of availability.
    """
    global _model, _tokenizer, _device, _dim, _tried
    with _lock:
        if _tried:
            return _model, _tokenizer, _device
        _tried = True
        if os.environ.get("SEVIM_DISABLE_EMBED") == "1":
            return None, None, None
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            torch.set_grad_enabled(False)
            if os.environ.get("SEVIM_STRICT_DET") == "1":
                torch.set_num_threads(1)

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token

            has_cuda = torch.cuda.is_available()
            dtype = torch.float16 if has_cuda else torch.float32
            _model = AutoModel.from_pretrained(
                MODEL_NAME,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            _device = torch.device("cuda" if has_cuda else "cpu")
            _model = _model.to(_device)
            _model.eval()
            _dim = int(getattr(_model.config, "hidden_size", 0))
        except Exception as exc:  # noqa: BLE001 — deliberately broad
            print(f"[sevim.embed] failed to load {MODEL_NAME}: {exc}")
            _model = None
            _tokenizer = None
            _device = None
        return _model, _tokenizer, _device


def is_available() -> bool:
    """Return True if the encoder loaded successfully and is ready to use."""
    m, _, _ = _load()
    return m is not None


def embedding_dim() -> int:
    """Return the hidden-state dimension of the loaded model (0 if unavailable)."""
    _load()
    return _dim


def encode(text: str) -> tuple[float, ...]:
    """Return the mean-pooled last-hidden-state embedding for *text*.

    Truncates to MAX_TOKENS tokens.  Returns an empty tuple when the encoder
    is unavailable; downstream stages (S2 cosine-merge gate, S3 φ) handle
    this gracefully by falling back to label-only / salience-only paths.
    """
    m, tok, dev = _load()
    if m is None or tok is None:
        return ()
    import torch
    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding=True,
    ).to(dev)
    with torch.no_grad():
        out = m(**inputs, output_hidden_states=False, use_cache=False)
    last = out.last_hidden_state  # (1, T, D)
    mask = inputs["attention_mask"].unsqueeze(-1).to(last.dtype)
    summed = (last * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    pooled = (summed / counts).squeeze(0).to(torch.float32).cpu().tolist()
    return tuple(float(x) for x in pooled)
