"""Impossibility Theorems for Multi-Agent Context Management.

This package contains formal proofs and runtime demonstrations of three
fundamental impossibility theorems that establish why context must be
typed, immutable, and auditable:

  IS-1: Causality + Audit + Composability Cannot Coexist Without Capsules
  IS-2: Cross-Domain Type Unification Requires Closed Vocabulary
  IS-3: Formal Verification at Scale Requires Immutable Context

Each module provides:
  - Formal statement (as docstring)
  - Proof sketch
  - Runtime demonstration code
  - Test cases

Together they establish that the Capsule Contract is not just sufficient
but necessary for multi-agent systems to guarantee certain fundamental
properties.
"""

from vision_mvp.theorems import impossibility, domain_unification, verification_complexity

__all__ = [
    "impossibility",
    "domain_unification",
    "verification_complexity",
]
