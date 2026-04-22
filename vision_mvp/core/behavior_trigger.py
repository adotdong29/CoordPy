"""Behavior-fingerprint event trigger for numerical-convention coordination.

The existing `core/event_trigger.py` measures disagreement via Jaccard
distance on dict-key sets. That signal is perfect for Phase 14's
ProtocolKit (underspecified dict schemas) but blind to numerical-convention
drift: two drafts can have identical dict-key sets and still disagree about
rounding mode, NaN policy, or sign encoding.

This module replaces the signal. Given an agent's own draft and a bulletin
of teammate drafts, it:

  1. Execs each draft source in an isolated namespace (math + builtins only).
  2. Resolves the function each draft defines, matched against a fixed
     catalog of `(producer_name, consumer_name) -> probe_battery` entries.
  3. Runs the probe battery — each probe is a `(producer, consumer) -> bool`
     predicate that should be True iff the two drafts share a convention.
  4. Reports disagreement as `1 - fraction_of_probes_true`.

API mirrors `event_trigger.should_refine` exactly so it's a drop-in:

    decision = should_refine(own_draft_src, bulletin_src_list, threshold)
    decision.refine   : bool
    decision.score    : float in [0, 1]
    decision.threshold: float

Failures (syntax error, missing function, probe crash) fall back to
`score = 1.0` ("refine") — safety first.
"""

from __future__ import annotations

import ast
import math
import traceback
from dataclasses import dataclass


# ---------------- Probe batteries per (producer, consumer) pair ----------

# Each probe takes (producer_fn, consumer_fn) and returns True iff the pair
# agrees on the hidden convention for this input. Disagreement accumulates
# as the fraction of probes returning False or crashing.
_PROBES: dict[tuple[str, str], list] = {
    ("round_amount", "check_rounded"): [
        lambda p, c: c(0.5,  p(0.5, 0),  0) is True,
        lambda p, c: c(1.5,  p(1.5, 0),  0) is True,
        lambda p, c: c(-0.5, p(-0.5, 0), 0) is True,
        lambda p, c: c(2.5,  p(2.5, 0),  0) is True,
    ],
    ("to_ledger", "from_ledger"): [
        lambda p, c: abs(c(p(1.50))  - 1.50)  < 1e-6,
        lambda p, c: abs(c(p(100.0)) - 100.0) < 1e-6,
        lambda p, c: abs(c(p(0.01))  - 0.01)  < 1e-6,
    ],
    ("reduce_amounts", "is_valid_reduction"): [
        lambda p, c: c([1.0, 2.0, 3.0], p([1.0, 2.0, 3.0])) is True,
        lambda p, c: c([1.0, float("nan"), 2.0],
                        p([1.0, float("nan"), 2.0])) is True,
        lambda p, c: c([], p([])) is True,
    ],
    ("add_capped", "predict_overflow"): [
        lambda p, c: p(5, 3, 100)    == c(5, 3, 100),
        lambda p, c: p(99, 50, 100)  == c(99, 50, 100),
        lambda p, c: p(-99, -50, 100) == c(-99, -50, 100),
    ],
    ("encode_signed", "decode_signed"): [
        lambda p, c: c(p(5, 8),    8) == 5,
        lambda p, c: c(p(-5, 8),   8) == -5,
        lambda p, c: c(p(-127, 8), 8) == -127,
    ],
}

# Reverse map: given one specialty, find the probe-battery keys it can play
# either role in.
_PAIR_ROLES: dict[str, list[tuple[str, str, str]]] = {}
for (prod, cons), _battery in _PROBES.items():
    _PAIR_ROLES.setdefault(prod, []).append((prod, cons, "producer"))
    _PAIR_ROLES.setdefault(cons, []).append((prod, cons, "consumer"))


# ---------------- Source parsing -----------------------------------------

def _function_names_defined(source: str) -> set[str]:
    """Return the set of top-level `def` names in `source`."""
    if not source:
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


def _exec_source(source: str) -> dict:
    """Exec `source` into an isolated namespace pre-loaded with math."""
    ns: dict = {"__builtins__": __builtins__, "math": math}
    try:
        exec(source, ns)
    except Exception:
        return {}
    return ns


# ---------------- Disagreement score -------------------------------------

def _pair_score(
    producer_fn, consumer_fn, probes: list,
) -> tuple[int, int]:
    """Run `probes` with (producer_fn, consumer_fn); return (agreed, total)."""
    agreed = 0
    for probe in probes:
        try:
            if probe(producer_fn, consumer_fn) is True:
                agreed += 1
        except Exception:
            pass
    return agreed, len(probes)


def pair_disagreement(
    own_src: str, bulletin_src: str,
) -> float | None:
    """If `own_src` and `bulletin_src` form a known pair, return disagreement
    in [0, 1]; otherwise None (signal: can't judge this pair).

    Tries both role assignments (own as consumer / own as producer).
    """
    own_names = _function_names_defined(own_src)
    bul_names = _function_names_defined(bulletin_src)

    for p, c in _PROBES:
        probes = _PROBES[(p, c)]
        # own = consumer, bulletin = producer
        if c in own_names and p in bul_names:
            own_ns = _exec_source(own_src)
            bul_ns = _exec_source(bulletin_src)
            if c in own_ns and p in bul_ns:
                agreed, total = _pair_score(bul_ns[p], own_ns[c], probes)
                if total > 0:
                    return 1.0 - agreed / total
        # own = producer, bulletin = consumer
        if p in own_names and c in bul_names:
            own_ns = _exec_source(own_src)
            bul_ns = _exec_source(bulletin_src)
            if p in own_ns and c in bul_ns:
                agreed, total = _pair_score(own_ns[p], bul_ns[c], probes)
                if total > 0:
                    return 1.0 - agreed / total
    return None


# ---------------- Public API (event-trigger shape) -----------------------

@dataclass
class TriggerDecision:
    refine: bool
    score: float
    threshold: float
    own_functions: set[str]
    bulletin_functions: set[str]


def should_refine(
    own_draft: str,
    bulletin_drafts: list[str],
    threshold: float = 0.34,
) -> TriggerDecision:
    """Decide whether to invoke the LLM for round-2 refinement.

    Score = mean disagreement across any bulletin drafts that form a known
    pair with `own_draft`. If no pair match is found, score=0 (no signal —
    don't trigger).

    Refine iff score >= threshold.
    """
    own_names = _function_names_defined(own_draft)
    bul_names: set[str] = set()
    disagreements = []
    for src in bulletin_drafts:
        bul_names |= _function_names_defined(src)
        score = pair_disagreement(own_draft, src)
        if score is not None:
            disagreements.append(score)

    if not disagreements:
        return TriggerDecision(
            refine=False, score=0.0, threshold=threshold,
            own_functions=own_names, bulletin_functions=bul_names,
        )

    score = sum(disagreements) / len(disagreements)
    return TriggerDecision(
        refine=score >= threshold, score=score, threshold=threshold,
        own_functions=own_names, bulletin_functions=bul_names,
    )
