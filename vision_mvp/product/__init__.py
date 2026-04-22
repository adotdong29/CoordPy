"""Phase-45 product-grade orchestration surface.

This subpackage is the one-command product entrypoint for the
context-zero SWE-style evaluation pipeline. It is a thin
orchestrator over the already-validated Phase-40..44 primitives:

  * ``phase44_public_readiness.run_readiness``
  * ``phase42_parser_sweep`` (parser-mode axis)
  * ``phase44_semantic_residue`` (raw capture + refined residue)

It does not replace the per-phase experiment scripts; it wires
them together behind a stable profile-driven CLI and emits a
single machine-readable report.
"""

from __future__ import annotations

__all__ = ("profiles", "runner", "report")
