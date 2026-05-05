"""Service entry points for Lyceum.

Thin top-level wrappers that compose the lower-level ``serve`` package
into multi-book deployments.  Most users start a session with::

    python -m serve.server books/<book>.json [books/<other>.json ...]

and never touch this package directly.  ``service`` exists so a
multi-book operator can layer book-selection, active-book switching,
and shared-state warm-up on top of the per-session orchestrator
without modifying the core server.
"""
