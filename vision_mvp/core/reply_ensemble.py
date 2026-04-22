"""Phase 37 Part B — reply-axis ensemble defenses.

Phase 34 closed the extractor-axis ensemble story: a
``UnionExtractor`` over regex + LLM recovered from a Phase-34
adversarial-extractor collapse on narrative-phrased payloads.
Phase 36 then *measured* the analogous collapse on the reply
axis (Theorem P36-2): a single adversarial drop_root flip
collapses both dynamic primitives to the static baseline.
Conjecture C36-7 named the required defensive-depth layer on
the reply axis — *redundant replies* with robust aggregation —
but did not implement it.

This module is that layer. It provides three ensemble patterns
that differ in exactly one design axis (how replies are combined
into a single resolver input) and compose cleanly with the
Phase-36 ``LLMThreadReplier`` / ``CalibratingReplier`` pipeline:

  * ``MODE_DUAL_AGREE`` — two independent paths emit replies in
    parallel. The ensemble emits the shared ``reply_kind`` only
    if both paths agree on a well-formed in-vocab answer.
    Otherwise the ensemble returns UNCERTAIN. This is the
    reply-axis analogue of an AND-gated union — a *conservative*
    combiner: it refuses to act under disagreement, which the
    Phase-35 decoder correctly maps to NO_CONSENSUS.
  * ``MODE_PRIMARY_FALLBACK`` — the primary path's reply is
    used if it is well-formed; otherwise the ensemble falls
    back to the secondary path. This recovers *malformed* and
    *out-of-vocab* collapses but offers no defense against a
    semantically-wrong well-formed primary.
  * ``MODE_VERIFIED`` — the primary's reply is used only if a
    *deterministic verifier* accepts it. The verifier here is a
    classifier over the payload (e.g. the Phase-36
    ``ScenarioAwareMockReplier`` logic promoted to a verifier,
    or any hand-coded payload-pattern check). The ensemble
    defends against the dominant Phase-37 failure mode
    (semantic mislabel on well-formed JSON) at the cost of
    requiring a trusted verifier on the same axis.

Why three, not one
------------------

Phase 36 showed the reply-noise axis has at least three
orthogonal failure modes:

  * malformed / OOV      — syntactic.
  * drop (UNCERTAIN)     — recall loss.
  * semantic mislabel    — the LLM emits well-formed wrong JSON.

No single ensemble pattern dominates all three in the
abstract. ``MODE_DUAL_AGREE`` is strong against uncorrelated
random drift but weak against two paths that share the same
bias. ``MODE_PRIMARY_FALLBACK`` is strong against syntax noise
but blind to semantic error. ``MODE_VERIFIED`` is strong under
a trusted verifier but inherits the verifier's gaps.

Phase-37 measures all three on the Phase-35 contested bank
under: clean inputs, synthetic malformed_prob, synthetic
mislabel_prob, and adversarial drop_root.

Scope discipline
----------------

  * The ensemble replier has the same
    ``(scenario, role, kind, payload, other_candidates,
    role_events)`` -> ``(reply_kind, witness, well_formed)``
    shape as ``LLMThreadReplier``. Drop-in. No change to
    ``EscalationThread`` or ``AdaptiveSubRouter``.
  * ``MODE_DUAL_AGREE`` does NOT model a quorum > 2 — a 3-way
    quorum is straightforward but not needed to test the
    basic claim.
  * The verifier in ``MODE_VERIFIED`` is the caller's; it can
    be the deterministic oracle (trivial defense — upper
    bound) or a practical hand-coded classifier (realistic).
  * Token / wall overhead is additive: ``MODE_DUAL_AGREE``
    costs 2× replier calls; ``MODE_PRIMARY_FALLBACK`` costs
    1× normally, 2× on the fallback path; ``MODE_VERIFIED``
    costs 1× replier + 1× verifier (verifier is cheap).

Theoretical anchor: RESULTS_PHASE37.md § B.2 (Theorem P37-3,
Conjecture C37-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from vision_mvp.core.dynamic_comm import (
    REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN,
)
from vision_mvp.core.llm_thread_replier import LLMThreadReplier


# Ensemble modes.
MODE_DUAL_AGREE = "dual_agree"
MODE_PRIMARY_FALLBACK = "primary_fallback"
MODE_VERIFIED = "verified"

ALL_ENSEMBLE_MODES = (MODE_DUAL_AGREE, MODE_PRIMARY_FALLBACK,
                       MODE_VERIFIED)


# A ReplyCallable has the LLMThreadReplier shape.
ReplyCallable = Callable[
    [object, str, str, str,
     Sequence[tuple[str, str, str]],
     Sequence[object] | None],
    tuple[str, str, bool]]


# A Verifier gives a verdict on a (scenario, role, kind, payload,
# replied_kind) claim: True if it accepts, False if it rejects.
# Implementations include payload-pattern classifiers and
# deterministic oracles.
Verifier = Callable[[object, str, str, str, str], bool]


@dataclass
class EnsembleStats:
    """Per-run counters for the ensemble replier.

    * ``n_calls``         — outer calls.
    * ``n_primary_calls`` — primary path invocations.
    * ``n_secondary_calls`` — secondary / verifier path
      invocations (path-dependent on mode).
    * ``n_agree``         — calls where primary and secondary
      emitted the same in-vocab reply_kind.
    * ``n_disagree``      — well-formed disagreement.
    * ``n_verified``      — calls where verifier returned True.
    * ``n_rejected``      — calls where verifier returned False.
    * ``n_fallback_used`` — calls that consumed the fallback
      path.
    * ``final_kind_hist`` — histogram of the returned reply_kind.
    """

    n_calls: int = 0
    n_primary_calls: int = 0
    n_secondary_calls: int = 0
    n_agree: int = 0
    n_disagree: int = 0
    n_verified: int = 0
    n_rejected: int = 0
    n_fallback_used: int = 0
    final_kind_hist: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_primary_calls": self.n_primary_calls,
            "n_secondary_calls": self.n_secondary_calls,
            "n_agree": self.n_agree,
            "n_disagree": self.n_disagree,
            "n_verified": self.n_verified,
            "n_rejected": self.n_rejected,
            "n_fallback_used": self.n_fallback_used,
            "final_kind_hist": dict(self.final_kind_hist),
        }


@dataclass
class EnsembleReplier:
    """Configurable reply-axis ensemble.

    ``__call__`` matches ``LLMThreadReplier.__call__`` so the
    ensemble drops into ``causality_extractor_from_replier``.

    Fields:
      * ``mode``       — one of the ``MODE_*`` constants.
      * ``primary``    — primary replier (shape matches
        ``LLMThreadReplier.__call__``).
      * ``secondary``  — secondary replier (required in
        ``dual_agree`` and ``primary_fallback``; ignored in
        ``verified`` mode if ``verifier`` is set).
      * ``verifier``   — optional verifier (required in
        ``verified`` mode).
      * ``fallback_reply_kind`` — returned on disagreement /
        verifier rejection. Defaults to ``UNCERTAIN``.
      * ``stats``      — per-run counters.
    """

    mode: str
    primary: ReplyCallable
    secondary: ReplyCallable | None = None
    verifier: Verifier | None = None
    fallback_reply_kind: str = REPLY_UNCERTAIN
    stats: EnsembleStats = field(default_factory=EnsembleStats)

    def __post_init__(self) -> None:
        if self.mode not in ALL_ENSEMBLE_MODES:
            raise ValueError(f"unknown ensemble mode {self.mode!r}")
        if self.mode in (MODE_DUAL_AGREE, MODE_PRIMARY_FALLBACK):
            if self.secondary is None:
                raise ValueError(
                    f"mode {self.mode!r} requires a secondary replier")
        if self.mode == MODE_VERIFIED and self.verifier is None:
            raise ValueError(
                "mode 'verified' requires a verifier callable")

    def __call__(self,
                 scenario: object,
                 role: str,
                 kind: str,
                 payload: str,
                 other_candidates: Sequence[tuple[str, str, str]] = (),
                 role_events: Sequence[object] | None = None,
                 ) -> tuple[str, str, bool]:
        self.stats.n_calls += 1
        if self.mode == MODE_DUAL_AGREE:
            out = self._dual_agree(
                scenario, role, kind, payload,
                other_candidates, role_events)
        elif self.mode == MODE_PRIMARY_FALLBACK:
            out = self._primary_fallback(
                scenario, role, kind, payload,
                other_candidates, role_events)
        else:
            out = self._verified(
                scenario, role, kind, payload,
                other_candidates, role_events)
        self.stats.final_kind_hist[out[0]] = \
            self.stats.final_kind_hist.get(out[0], 0) + 1
        return out

    # ------- mode implementations -------

    def _dual_agree(self, scenario, role, kind, payload,
                    other_candidates, role_events):
        self.stats.n_primary_calls += 1
        rk_p, w_p, wf_p = self.primary(
            scenario, role, kind, payload,
            other_candidates, role_events)
        self.stats.n_secondary_calls += 1
        rk_s, w_s, wf_s = self.secondary(  # type: ignore[misc]
            scenario, role, kind, payload,
            other_candidates, role_events)
        if wf_p and wf_s and rk_p == rk_s:
            self.stats.n_agree += 1
            witness = w_p if len(w_p) >= len(w_s) else w_s
            return rk_p, witness, True
        self.stats.n_disagree += 1
        # Conservative fallback — refuse to emit a class.
        return self.fallback_reply_kind, "", False

    def _primary_fallback(self, scenario, role, kind, payload,
                          other_candidates, role_events):
        self.stats.n_primary_calls += 1
        rk_p, w_p, wf_p = self.primary(
            scenario, role, kind, payload,
            other_candidates, role_events)
        if wf_p:
            return rk_p, w_p, True
        self.stats.n_fallback_used += 1
        self.stats.n_secondary_calls += 1
        return self.secondary(  # type: ignore[misc]
            scenario, role, kind, payload,
            other_candidates, role_events)

    def _verified(self, scenario, role, kind, payload,
                  other_candidates, role_events):
        self.stats.n_primary_calls += 1
        rk_p, w_p, wf_p = self.primary(
            scenario, role, kind, payload,
            other_candidates, role_events)
        if not wf_p:
            self.stats.n_rejected += 1
            return self.fallback_reply_kind, "", False
        self.stats.n_secondary_calls += 1
        ok = self.verifier(  # type: ignore[misc]
            scenario, role, kind, payload, rk_p)
        if ok:
            self.stats.n_verified += 1
            return rk_p, w_p, True
        self.stats.n_rejected += 1
        return self.fallback_reply_kind, "", False


# =============================================================================
# Verifier helpers
# =============================================================================


def verifier_from_oracle(
        oracle: Callable[[object, str, str, str], str],
        ) -> Verifier:
    """Promote a causality oracle
    (``(scenario, role, kind, payload) -> oracle_class``) into a
    ``Verifier`` (``(scenario, role, kind, payload,
    replied_kind) -> bool``).

    The verifier is an *upper bound* — using the deterministic
    oracle as the verifier makes the ensemble strictly
    dominated by the oracle's answer. Useful as a theoretical
    ceiling for Phase-37 measurements; a realistic verifier
    uses a payload-pattern classifier that is strictly weaker
    than the oracle.
    """

    def _v(scenario: object, role: str, kind: str, payload: str,
           replied_kind: str) -> bool:
        oracle_out = oracle(scenario, role, kind, payload)
        if replied_kind == REPLY_INDEPENDENT_ROOT:
            return oracle_out == "INDEPENDENT_ROOT"
        if replied_kind == REPLY_DOWNSTREAM_SYMPTOM:
            return oracle_out.startswith("DOWNSTREAM_SYMPTOM_OF:")
        if replied_kind == REPLY_UNCERTAIN:
            return oracle_out == "UNCERTAIN"
        return False

    return _v


def verifier_accept_ir_only_on_payload_marker(
        marker_to_accept: dict[str, tuple[str, ...]],
        ) -> Verifier:
    """Return a realistic verifier: accepts an INDEPENDENT_ROOT
    reply only if the payload contains one of the markers for
    the declared ``claim_kind``. Rejects INDEPENDENT_ROOT for
    payloads without a declared root marker, and always accepts
    DOWNSTREAM_SYMPTOM / UNCERTAIN replies.

    This models a "cheap, partial" verifier — the real case a
    production pipeline ships: hand-coded rules that rule out
    obvious false INDEPENDENT_ROOT claims without claiming full
    oracle power.

    Example ``marker_to_accept``:

        {"TLS_EXPIRED": ("reason=expired",),
         "DEADLOCK_SUSPECTED": ("deadlock", "pg_deadlock"),
         "CRON_OVERRUN": ("exit=137",)}

    For any claim kind not in the map, the verifier accepts the
    INDEPENDENT_ROOT claim (no marker constraint).
    """

    def _v(scenario: object, role: str, kind: str, payload: str,
           replied_kind: str) -> bool:
        if replied_kind != REPLY_INDEPENDENT_ROOT:
            return True
        markers = marker_to_accept.get(kind)
        if not markers:
            return True
        return any(m in payload for m in markers)

    return _v


def verifier_from_payload_classifier(
        classifier: Callable[[str, str, str], str],
        ) -> Verifier:
    """Promote a scenario-free payload-pattern classifier
    (``(role, kind, payload) -> reply_kind``) into a
    ``Verifier``.

    Used in ``MODE_VERIFIED`` to cross-check a chatty LLM reply
    against a rule-based (no-LLM) payload classifier. If the
    classifier agrees, accept; if it disagrees, reject.
    """

    def _v(scenario: object, role: str, kind: str, payload: str,
           replied_kind: str) -> bool:
        c_kind = classifier(role, kind, payload)
        return c_kind == replied_kind

    return _v


# =============================================================================
# Adapter: EnsembleReplier -> CausalityExtractor
# =============================================================================


def causality_extractor_from_ensemble(
        ensemble: EnsembleReplier,
        ) -> Callable[[object, str, str, str], str]:
    """Wrap an ``EnsembleReplier`` as a ``CausalityExtractor``
    (shape ``(scenario, role, kind, payload) -> str``) so it is
    a drop-in for ``run_dynamic_coordination(causality_extractor=
    ...)`` and ``run_adaptive_sub_coordination(causality_extractor=
    ...)``.

    Mirrors ``causality_extractor_from_replier`` but goes through
    the ensemble's mode.
    """

    def _extract(scenario: object, role: str,
                 kind: str, payload: str) -> str:
        reply_kind, _witness, well_formed = ensemble(
            scenario=scenario, role=role, kind=kind,
            payload=payload, other_candidates=(),
            role_events=None)
        if not well_formed:
            return "UNCERTAIN"
        if reply_kind == REPLY_INDEPENDENT_ROOT:
            return "INDEPENDENT_ROOT"
        if reply_kind == REPLY_DOWNSTREAM_SYMPTOM:
            return "DOWNSTREAM_SYMPTOM_OF:" + kind
        return "UNCERTAIN"

    return _extract


__all__ = [
    "MODE_DUAL_AGREE", "MODE_PRIMARY_FALLBACK", "MODE_VERIFIED",
    "ALL_ENSEMBLE_MODES",
    "EnsembleReplier", "EnsembleStats", "Verifier",
    "verifier_from_oracle",
    "verifier_accept_ir_only_on_payload_marker",
    "verifier_from_payload_classifier",
    "causality_extractor_from_ensemble",
]
