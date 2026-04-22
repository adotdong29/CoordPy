"""Phase 38 Part A — two-layer ensemble composition.

Phase 37 Theorem P37-4 formalised the asymmetry: a *reply-axis*
ensemble combined strictly below the point at which noise
enters cannot defend against that noise, no matter how many
redundant repliers it stacks. Conjecture C37-2 named the next
step: a **two-layer** composition that places an independent
path strictly above the noise point — the extractor-axis
analogue of Phase 34's ``UnionExtractor`` for the reply axis.

This module ships the layer and the compositional shape that
Phase 38 Part A measures.

Two axes, two layers
--------------------

The coordination pipeline has two ensemble-addressable
boundaries:

  * **Extractor axis (Phase 34 / 38 layer-1).** The
    ``extract_claims_for_role`` boundary. A role turns raw
    events into typed claims. Failure modes here: missed
    claims (narrative phrasing regex cannot parse), dropped
    claims (adversarial extractor), spurious claims. An
    ``UnionExtractor(primary, secondary)`` is the
    Phase-34 defense.
  * **Reply axis (Phase 37 / 38 layer-2).** The
    ``causality_extractor(scenario, role, kind, payload)``
    boundary. A producer classifies one of its own claims.
    Failure modes: malformed JSON, biased semantic
    mislabel, adversarial drop_root. ``EnsembleReplier``
    is the Phase-37 defense.

Phase 38's design surface adds a *third* combiner on top of
the two axis-local ensembles: the ``PathUnion`` combiner
takes two full (extractor + replier) paths, each with its
own possibly noise-wrapped output, and combines their
emitted causality class strictly above any noise wrapper.
This closes the last Phase-37 hole: the noise-above-ensemble
symmetry that Theorem P37-4 identified.

The three compositional shapes
------------------------------

  * **Axis-local ensembles, single path.**
    ``UnionExtractor`` at layer-1 *or* ``EnsembleReplier`` at
    layer-2, used in isolation. Phase 34 and Phase 37 respectively.
  * **Stacked axis-local ensembles, single path.** Both layer-1
    and layer-2 ensembles active within the same causality path.
    Defends against disjoint failure modes at the two axes:
    dropped-claim at the extractor axis AND biased-primary at
    the reply axis. Empirically measurable on the conjunction
    cell.
  * **Path-union (Phase-38 new).** Two full paths
    ``(extractor_i, replier_i, noise_i)`` combined by a
    causality-class combiner that operates strictly above
    any per-path noise wrapper. This recovers where
    single-layer reply-axis ensembles provably cannot
    (Theorem P37-4 → Theorem P38-1 in RESULTS_PHASE38.md).

Combiner modes
--------------

  * ``PATH_MODE_DUAL_AGREE``    — emit the shared class iff
    both paths agree; emit UNCERTAIN under disagreement.
    Conservative; analogue of Phase-37 ``MODE_DUAL_AGREE``
    but above the noise wrapper.
  * ``PATH_MODE_UNION_ROOT``    — emit INDEPENDENT_ROOT if
    at least one path emitted it and no path emitted a
    contradictory IR on a different candidate (the latter
    check is a per-call property of this module; the thread's
    resolution rule handles cross-candidate conflicts
    separately). Generous; recovers when one path is dropped
    under adversarial-drop_root.
  * ``PATH_MODE_VERIFIED``      — primary path's class is
    used iff the secondary path's class matches (or primary
    emitted UNCERTAIN, which is strictly safe). The secondary
    path functions as a cross-check. Used when the primary
    is preferred (trained, expensive) but a deterministic
    secondary catches the malicious/unusual primary outputs.

Scope discipline
----------------

  * This module does NOT modify ``EnsembleReplier`` or
    ``UnionExtractor``. It adds a causality-class combiner
    that lives strictly above them.
  * The combiner is a ``CausalityExtractor``
    (``(scenario, role, kind, payload) -> str``); it plugs
    into ``run_dynamic_coordination`` and
    ``run_adaptive_sub_coordination`` via their existing
    ``causality_extractor`` parameter. No substrate change.
  * The ``PathUnion`` combiner is NOT aware of thread /
    adaptive-sub structure. Per-candidate conflict resolution
    stays inside ``EscalationThread.close_thread`` — we only
    emit a single causality class per call.
  * Two-layer does NOT subsume single-layer. Each layer is
    load-bearing for a different failure family; running both
    is additive defense, not redundancy.

Theoretical anchor: RESULTS_PHASE38.md § B.1 (Theorem P38-1,
Conjecture C38-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from vision_mvp.core.reply_noise import (
    CAUSALITY_DOWNSTREAM_PREFIX, CAUSALITY_INDEPENDENT_ROOT,
    CAUSALITY_UNCERTAIN,
)


# =============================================================================
# Combiner modes
# =============================================================================


PATH_MODE_DUAL_AGREE = "path_dual_agree"
PATH_MODE_UNION_ROOT = "path_union_root"
PATH_MODE_VERIFIED = "path_verified"

ALL_PATH_MODES = (PATH_MODE_DUAL_AGREE, PATH_MODE_UNION_ROOT,
                  PATH_MODE_VERIFIED)


CausalityExtractor = Callable[[object, str, str, str], str]


# =============================================================================
# Stats
# =============================================================================


@dataclass
class PathUnionStats:
    """Per-run counters for a two-layer path-union combiner.

    * ``n_calls``          — outer calls.
    * ``n_agree``          — calls where both paths emit the
      same concrete class.
    * ``n_disagree``       — calls where paths disagree on a
      concrete class.
    * ``n_one_uncertain``  — calls where one path is UNCERTAIN
      and the other is concrete.
    * ``n_both_uncertain`` — calls where both paths are
      UNCERTAIN.
    * ``n_primary_used``   — calls where the combiner returns
      the primary path's class.
    * ``n_secondary_used`` — calls where the combiner returns
      the secondary path's class.
    * ``n_merged``         — calls where the combiner's output
      is a class that neither path emitted on this call (only
      meaningful in ``union_root`` mode when both paths are
      UNCERTAIN on different candidates — does not occur in
      single-call semantics but tracked for auditability).
    * ``final_class_hist`` — histogram of the returned class
      string.
    """

    n_calls: int = 0
    n_agree: int = 0
    n_disagree: int = 0
    n_one_uncertain: int = 0
    n_both_uncertain: int = 0
    n_primary_used: int = 0
    n_secondary_used: int = 0
    n_merged: int = 0
    final_class_hist: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_agree": self.n_agree,
            "n_disagree": self.n_disagree,
            "n_one_uncertain": self.n_one_uncertain,
            "n_both_uncertain": self.n_both_uncertain,
            "n_primary_used": self.n_primary_used,
            "n_secondary_used": self.n_secondary_used,
            "n_merged": self.n_merged,
            "final_class_hist": dict(self.final_class_hist),
        }


# =============================================================================
# PathUnionCausalityExtractor — the Phase-38 combiner
# =============================================================================


def _is_independent_root(c: str) -> bool:
    return c == CAUSALITY_INDEPENDENT_ROOT


def _is_downstream_symptom(c: str) -> bool:
    return c.startswith(CAUSALITY_DOWNSTREAM_PREFIX)


def _class_class(c: str) -> str:
    """Bucket a causality class string to {IR, DS, UNCERTAIN}."""
    if _is_independent_root(c):
        return "IR"
    if _is_downstream_symptom(c):
        return "DS"
    return "UNCERTAIN"


@dataclass
class PathUnionCausalityExtractor:
    """Compose two independent causality extractors with a class-
    level combiner that sits strictly above any per-path noise
    wrapper.

    ``primary`` and ``secondary`` are ``CausalityExtractor``
    callables. Each may be (and typically is) a replier wrapped
    with a noise / adversarial wrapper — the whole point of this
    module is that the combiner is ABOVE those wrappers.

    Modes:

      * ``PATH_MODE_DUAL_AGREE``: if both paths agree on a
        concrete class, emit it; else UNCERTAIN. Matches the
        Phase-37 ``MODE_DUAL_AGREE`` but composes at the path
        level, not the replier level. Recovers chatty-wrong
        primaries when secondary is clean.
      * ``PATH_MODE_UNION_ROOT``: if at least one path emits
        INDEPENDENT_ROOT and the other emits {IR, UNCERTAIN},
        emit IR. If the other emits DS (contradicting IR), emit
        UNCERTAIN (conservative; let the thread's multi-
        candidate resolution handle the conflict). If both
        emit DS, emit DS. If both UNCERTAIN, emit UNCERTAIN.
        Recovers adversarial drop_root when the adversary
        attacks only one path.
      * ``PATH_MODE_VERIFIED``: primary's concrete class is
        emitted only if secondary agrees on the class bucket
        (IR / DS / UNCERTAIN). If primary is IR and secondary
        disagrees → UNCERTAIN. If primary is DS and secondary
        is UNCERTAIN → DS is still emitted (UNCERTAIN is a
        null vote — this is the realistic semantics when the
        secondary is weaker than the primary).

    The output has the ``CausalityExtractor`` shape; it plugs
    directly into ``run_dynamic_coordination(causality_extractor=
    ...)`` and ``run_adaptive_sub_coordination(causality_
    extractor=...)``.
    """

    primary: CausalityExtractor
    secondary: CausalityExtractor
    mode: str = PATH_MODE_UNION_ROOT
    stats: PathUnionStats = field(default_factory=PathUnionStats)

    def __post_init__(self) -> None:
        if self.mode not in ALL_PATH_MODES:
            raise ValueError(f"unknown path mode {self.mode!r}")

    def __call__(self, scenario: object, role: str,
                 kind: str, payload: str) -> str:
        self.stats.n_calls += 1
        c_p = self.primary(scenario, role, kind, payload)
        c_s = self.secondary(scenario, role, kind, payload)
        bp = _class_class(c_p)
        bs = _class_class(c_s)

        if bp == bs and bp != "UNCERTAIN":
            self.stats.n_agree += 1
        elif bp == bs and bp == "UNCERTAIN":
            self.stats.n_both_uncertain += 1
        elif bp == "UNCERTAIN" or bs == "UNCERTAIN":
            self.stats.n_one_uncertain += 1
        else:
            self.stats.n_disagree += 1

        out = self._combine(c_p, c_s, kind)
        self.stats.final_class_hist[_class_class(out)] = (
            self.stats.final_class_hist.get(_class_class(out), 0)
            + 1)
        return out

    def _combine(self, c_p: str, c_s: str, kind: str) -> str:
        bp = _class_class(c_p)
        bs = _class_class(c_s)
        if self.mode == PATH_MODE_DUAL_AGREE:
            return self._combine_dual_agree(c_p, c_s, bp, bs, kind)
        if self.mode == PATH_MODE_UNION_ROOT:
            return self._combine_union_root(c_p, c_s, bp, bs, kind)
        return self._combine_verified(c_p, c_s, bp, bs, kind)

    def _combine_dual_agree(self, c_p, c_s, bp, bs, kind):
        if bp == bs and bp != "UNCERTAIN":
            # Both agreed on a concrete class.
            self.stats.n_primary_used += 1
            return c_p
        return CAUSALITY_UNCERTAIN

    def _combine_union_root(self, c_p, c_s, bp, bs, kind):
        if bp == "IR" and bs == "DS":
            # Contradictory concrete classes — safe UNCERTAIN.
            return CAUSALITY_UNCERTAIN
        if bp == "DS" and bs == "IR":
            return CAUSALITY_UNCERTAIN
        if bp == "IR" or bs == "IR":
            # At least one IR, no concrete contradiction.
            if bp == "IR":
                self.stats.n_primary_used += 1
                return c_p
            self.stats.n_secondary_used += 1
            return c_s
        if bp == "DS" or bs == "DS":
            # At least one DS, no concrete contradiction.
            if bp == "DS":
                self.stats.n_primary_used += 1
                return c_p
            self.stats.n_secondary_used += 1
            return c_s
        return CAUSALITY_UNCERTAIN

    def _combine_verified(self, c_p, c_s, bp, bs, kind):
        if bp == "IR":
            if bs == "IR":
                self.stats.n_primary_used += 1
                return c_p
            # Secondary disagrees or is UNCERTAIN. Treat
            # UNCERTAIN as a non-vote — conservative verifier
            # rejects the primary's IR unless the secondary
            # concretely confirms.
            return CAUSALITY_UNCERTAIN
        if bp == "DS":
            if bs == "IR":
                # Primary says DS, secondary says IR — on a
                # sibling candidate this is how adversarial-IR
                # may inject. Safe path: UNCERTAIN.
                return CAUSALITY_UNCERTAIN
            # Secondary DS or UNCERTAIN — DS is accepted.
            self.stats.n_primary_used += 1
            return c_p
        # Primary UNCERTAIN.
        if bs == "IR" or bs == "DS":
            # Primary is UNCERTAIN and secondary is concrete:
            # the verifier is generous here — fall through to
            # secondary. (This is the inverse of dual_agree and
            # makes verified a useful mode under a bypass
            # attack that nullifies primary without attacking
            # secondary.)
            self.stats.n_secondary_used += 1
            return c_s
        return CAUSALITY_UNCERTAIN


# =============================================================================
# TwoLayerDefense — the full stacked shape
# =============================================================================


@dataclass
class TwoLayerDefense:
    """Descriptor record for a Phase-38 two-layer defense.

    This is a record, not an orchestrator — the concrete wiring
    is done by the driver. The point is to make the two layers
    observable in the summary output (so the reader can tell
    which axes are active).

    Fields:

      * ``claim_extractor``        — layer-1 object, plugs into
        ``run_contested_handoff_protocol(claim_extractor=...)``.
        Optional (None = use Phase-31 default).
      * ``causality_extractor``    — layer-2 object, plugs
        into ``run_dynamic_coordination(causality_extractor=...)``.
        Optional (None = use Phase-35 deterministic oracle).
      * ``label``                  — short descriptor (e.g.
        ``"ext=union_narr;reply=dual"``).
      * ``n_layer1_active``        — 1 if layer-1 ensemble
        active, else 0.
      * ``n_layer2_active``        — 1 if layer-2 ensemble
        active, else 0.
      * ``path_union_mode``        — if a PathUnion combiner
        is installed at layer-2, its mode; else None.
    """

    label: str
    claim_extractor: Callable | None = None
    causality_extractor: CausalityExtractor | None = None
    n_layer1_active: int = 0
    n_layer2_active: int = 0
    path_union_mode: str | None = None

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "n_layer1_active": self.n_layer1_active,
            "n_layer2_active": self.n_layer2_active,
            "path_union_mode": self.path_union_mode,
        }


__all__ = [
    "PATH_MODE_DUAL_AGREE", "PATH_MODE_UNION_ROOT",
    "PATH_MODE_VERIFIED", "ALL_PATH_MODES",
    "PathUnionStats", "PathUnionCausalityExtractor",
    "TwoLayerDefense",
]
