"""Legacy product modules (pre-Wevra import path).

This subpackage is the original orchestration surface for the
profile-driven SWE-bench-Lite-shape evaluation pipeline. It is
kept importable for backwards compatibility, but the **stable
public contract is** ``vision_mvp.wevra`` — which re-exports the
modules here under their durable names.

New code should import from ``vision_mvp.wevra`` (e.g.
``from vision_mvp.wevra import profiles, report, ci_gate,
import_data, run, RunSpec``). The modules below are thin
orchestrators over the research primitives in
``vision_mvp.experiments.*``; they do not replace the per-phase
experiment scripts.
"""

from __future__ import annotations

__all__ = ("profiles", "runner", "report")
