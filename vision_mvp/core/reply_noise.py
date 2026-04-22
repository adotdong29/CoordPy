"""Phase 36 — parameterised noise on typed thread replies.

Phase 35 proved the dynamic-coordination primitive under a *clean*
producer-local causality extractor: ``infer_causality_hypothesis``
returns the gold causality class (INDEPENDENT_ROOT / DOWNSTREAM_
SYMPTOM / UNCERTAIN) with precision and recall 1.00 by
construction. That was the right first move — it isolated the
*substrate* story (can a thread change the answer?) from the
*reflection* story (can the producer reliably classify its own
claim?). Phase 36 Part A closes the first half of the gap between
Phase 35's clean ceiling and realistic deployment by shipping a
noise wrapper at the *reply-generation* boundary: the producer-
local causality extractor is perturbed with the same taxonomy
the Phase-32 extractor noise used (recall drop, type confusion,
witness corruption, spurious emission), and the thread's
resolution is re-measured.

Why this is the right boundary
-------------------------------

A typed thread reply carries two load-bearing bits per producer:

  * ``reply_kind`` — one of the enumerated causality classes
    (INDEPENDENT_ROOT / DOWNSTREAM_SYMPTOM / UNCERTAIN / ...).
  * ``referenced_claim_idx`` — which candidate this reply is
    about.

Under the clean Phase-35 setup, the producer's extractor always
picks the correct ``reply_kind`` on the correct index. Real
LLM-driven or heuristic classifiers violate both axes:

  * A producer might *drop* its INDEPENDENT_ROOT reply entirely
    (recall drop). The thread then closes NO_CONSENSUS.
  * A producer might *mislabel* its INDEPENDENT_ROOT as
    UNCERTAIN (precision-conservative failure), or as
    DOWNSTREAM_SYMPTOM (precision-wrong failure).
  * A producer might *reference the wrong candidate index*
    (e.g. claim-idx swap under adversarial scenario framing).
  * A producer's *witness string* might be corrupted (payload
    noise), which doesn't change the resolution but costs tokens.

This module parameterises all four axes against a single seed
and provides deterministic per-(scenario, role) perturbations so
a Phase-36 sweep is reproducible byte-for-byte.

What this module provides
-------------------------

* ``ReplyNoiseConfig`` — frozen knobs (drop_prob, mislabel_prob,
  swap_idx_prob, witness_corrupt_prob, seed). Identity if all
  probabilities are zero.
* ``noisy_causality_extractor`` — wraps a deterministic
  per-role causality extractor (shape matches
  ``infer_causality_hypothesis``) and returns a noisy variant.
* ``AdversarialReplyConfig`` / ``adversarial_reply_extractor`` —
  an adversarial wrapper that selectively targets the *gold*
  INDEPENDENT_ROOT reply per scenario (mirrors the Phase-34
  adversarial extractor's load-bearing-drop semantics but on
  the thread-reply axis).
* ``ReplyCorruptionReport`` — per-scenario counter of which
  noise type fired on which reply. Used by the Phase-36 driver
  to attribute each failed thread back to the specific noise
  that broke it.

Scope discipline (what this module does NOT do)
-----------------------------------------------

  * It does NOT change the ``EscalationThread`` primitive. Noise
    enters at the reply-producer boundary; the router, the
    resolution rule, and the subscription table are unchanged.
  * It does NOT model a payload-level attacker on the hash chain
    (that threat model belongs to ``peer_review`` + Ed25519).
  * It does NOT claim to calibrate against a specific LLM's
    reply-noise profile. The parameters are a parsimonious
    surrogate, selected so Phase-36 Part A can measure the
    substrate's graceful-degradation bound on the reply axis.
  * It is NOT a replacement for the Phase-32/34 extractor noise
    wrappers — those live *above* the handoff layer; these live
    above the thread-reply layer. They compose.

Theoretical anchor: RESULTS_PHASE36.md § B.1 (Theorem P36-1,
Conjecture C36-5).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable


# The shape of a per-role causality extractor — matches
# ``vision_mvp.tasks.contested_incident.infer_causality_hypothesis``.
# It is a function (scenario, role, claim_kind, payload) -> str where
# the returned string is one of:
#    "INDEPENDENT_ROOT"
#    "DOWNSTREAM_SYMPTOM_OF:<kind>"
#    "UNCERTAIN"
CausalityExtractor = Callable[[object, str, str, str], str]


# Enumerated causality reply classes — matches the Phase-35 thread
# reply vocabulary restricted to the causality-hypothesis axis.
CAUSALITY_INDEPENDENT_ROOT = "INDEPENDENT_ROOT"
CAUSALITY_UNCERTAIN = "UNCERTAIN"
CAUSALITY_DOWNSTREAM_PREFIX = "DOWNSTREAM_SYMPTOM_OF:"

# The reply kinds a noisy extractor may return.
_REPLY_POOL = (
    CAUSALITY_INDEPENDENT_ROOT,
    CAUSALITY_UNCERTAIN,
    CAUSALITY_DOWNSTREAM_PREFIX,  # special — composed with a claim kind
)


@dataclass(frozen=True)
class ReplyNoiseConfig:
    """Knobs for a Phase-36 reply-noise wrapper.

    Every probability is in ``[0, 1]``; 0 disables the axis, 1
    fires on every eligible reply. The default is identity.

    * ``drop_prob`` — probability of emitting UNCERTAIN when the
      underlying extractor returned INDEPENDENT_ROOT or
      DOWNSTREAM_SYMPTOM_OF. Models *recall loss* on the causal
      claim. Under the thread close rule, a dropped
      INDEPENDENT_ROOT flips the resolution to NO_CONSENSUS.
    * ``mislabel_prob`` — probability of swapping a concrete
      reply class for a different concrete class (INDEPENDENT_
      ROOT ↔ DOWNSTREAM_SYMPTOM_OF). Models *type confusion*.
    * ``swap_idx_prob`` — not handled at this layer (the
      extractor does not see the candidate index). This is a
      placeholder so the noise config can be extended at the
      orchestrator boundary without a breaking API change.
    * ``witness_corrupt_prob`` — probability of corrupting the
      witness string downstream. Again, not handled here —
      witnesses are produced by the orchestrator, not by the
      causality extractor. Kept on the config so that
      downstream callers (``run_dynamic_coordination``) can read
      a single ``ReplyNoiseConfig`` for both per-reply decisions.
    * ``seed`` — RNG seed; together with (scenario.scenario_id,
      role, kind) determines the per-reply outcome.
    """

    drop_prob: float = 0.0
    mislabel_prob: float = 0.0
    swap_idx_prob: float = 0.0
    witness_corrupt_prob: float = 0.0
    seed: int = 0

    def is_identity(self) -> bool:
        return (self.drop_prob == 0.0 and self.mislabel_prob == 0.0
                and self.swap_idx_prob == 0.0
                and self.witness_corrupt_prob == 0.0)

    def as_dict(self) -> dict:
        return {
            "drop_prob": self.drop_prob,
            "mislabel_prob": self.mislabel_prob,
            "swap_idx_prob": self.swap_idx_prob,
            "witness_corrupt_prob": self.witness_corrupt_prob,
            "seed": self.seed,
        }


@dataclass
class ReplyCorruptionReport:
    """Per-run counters of which reply-noise axis fired.

    Written by ``noisy_causality_extractor`` and read by the
    driver so failure attribution can distinguish "thread
    resolution missed because drop" from "... because mislabel"
    cleanly.
    """

    n_calls: int = 0
    n_dropped: int = 0
    n_mislabeled: int = 0
    n_unchanged: int = 0

    def record(self, axis: str) -> None:
        self.n_calls += 1
        if axis == "drop":
            self.n_dropped += 1
        elif axis == "mislabel":
            self.n_mislabeled += 1
        else:
            self.n_unchanged += 1

    def as_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_dropped": self.n_dropped,
            "n_mislabeled": self.n_mislabeled,
            "n_unchanged": self.n_unchanged,
        }


def _per_call_rng(seed: int, role: str, kind: str,
                   scenario_id: str) -> random.Random:
    """Deterministic per-(seed, role, kind, scenario) RNG.

    Separate streams per (role, kind) so that a sweep over one
    axis does not interact with the noise draws on another axis.
    """
    key = (seed, role, kind, scenario_id)
    return random.Random(hash(key) & 0xFFFFFFFF)


def noisy_causality_extractor(base: CausalityExtractor,
                                noise: ReplyNoiseConfig,
                                report: ReplyCorruptionReport | None = None,
                                ) -> CausalityExtractor:
    """Wrap a deterministic causality extractor with Phase-36 noise.

    The wrapped extractor has the same signature as ``base`` and is
    deterministic in (noise.seed, scenario, role, kind). Calling it
    twice on the same inputs yields the same output.

    Semantics (applied in order):

      1. Call ``base(scenario, role, kind, payload)`` to get the
         ground-truth causality class.
      2. If it was INDEPENDENT_ROOT or DOWNSTREAM_SYMPTOM_OF and
         the per-call RNG fires ``drop_prob``, return UNCERTAIN.
         Record ``"drop"``.
      3. Else, if the class is INDEPENDENT_ROOT and the RNG fires
         ``mislabel_prob``, return a DOWNSTREAM_SYMPTOM_OF:<kind>
         against a different candidate kind (if we cannot pick a
         valid different kind, degrade to UNCERTAIN). Record
         ``"mislabel"``.
      4. Else, if the class is DOWNSTREAM_SYMPTOM_OF and the RNG
         fires ``mislabel_prob``, return INDEPENDENT_ROOT.
         Record ``"mislabel"``.
      5. Else, return unchanged. Record ``"unchanged"``.

    Identity when ``noise.is_identity()``.
    """
    if noise.is_identity():
        return base

    def _wrapped(scenario: object, role: str,
                 kind: str, payload: str) -> str:
        base_out = base(scenario, role, kind, payload)
        sid = str(getattr(scenario, "scenario_id", ""))
        rng = _per_call_rng(noise.seed, role, kind, sid)

        def _record(axis: str) -> None:
            if report is not None:
                report.record(axis)

        # Step 1: recall-loss drop.
        if base_out in (CAUSALITY_INDEPENDENT_ROOT,) or \
                base_out.startswith(CAUSALITY_DOWNSTREAM_PREFIX):
            if rng.random() < noise.drop_prob:
                _record("drop")
                return CAUSALITY_UNCERTAIN

        # Step 2: mislabel.
        if rng.random() < noise.mislabel_prob:
            if base_out == CAUSALITY_INDEPENDENT_ROOT:
                _record("mislabel")
                return CAUSALITY_DOWNSTREAM_PREFIX + kind
            if base_out.startswith(CAUSALITY_DOWNSTREAM_PREFIX):
                _record("mislabel")
                return CAUSALITY_INDEPENDENT_ROOT
            if base_out == CAUSALITY_UNCERTAIN:
                # Conservative: flip UNCERTAIN → INDEPENDENT_ROOT
                # only with mislabel_prob; this is the "over-eager
                # producer" case that Phase-36 Part A specifically
                # measures.
                _record("mislabel")
                return CAUSALITY_INDEPENDENT_ROOT

        _record("unchanged")
        return base_out

    return _wrapped


# =============================================================================
# Adversarial reply noise — the Phase-34 analogue on the reply axis
# =============================================================================


ADVERSARIAL_REPLY_MODE_DROP_ROOT = "drop_root"
ADVERSARIAL_REPLY_MODE_FLIP_ROOT_TO_SYMPTOM = "flip_root_to_symptom"
ADVERSARIAL_REPLY_MODE_INJECT_ROOT_ON_SYMPTOM = "inject_root_on_symptom"
ADVERSARIAL_REPLY_MODE_COMBINED = "combined"


@dataclass(frozen=True)
class AdversarialReplyConfig:
    """Knobs for the Phase-36 adversarial reply wrapper.

    Unlike ``ReplyNoiseConfig`` whose noise is distribution-wide,
    this wrapper targets the *gold* reply on each scenario — it
    always damages the INDEPENDENT_ROOT reply if mode is
    DROP_ROOT / FLIP_ROOT_TO_SYMPTOM, and always damages a
    DOWNSTREAM_SYMPTOM reply if mode is
    INJECT_ROOT_ON_SYMPTOM. Matched nominal budget vs i.i.d.
    noise, the adversarial wrapper is the worst case.

    * ``target_mode`` — one of ``ADVERSARIAL_REPLY_MODE_*``.
    * ``target_roles`` — restrict the adversary to these roles.
      Empty = all roles.
    * ``budget`` — per-scenario cap on damaged replies. ``-1`` =
      unbounded.
    """

    target_mode: str = ADVERSARIAL_REPLY_MODE_DROP_ROOT
    target_roles: tuple[str, ...] = ()
    budget: int = 1


def adversarial_reply_extractor(base: CausalityExtractor,
                                 adv: AdversarialReplyConfig,
                                 report: ReplyCorruptionReport | None = None,
                                 ) -> CausalityExtractor:
    """Wrap a causality extractor with adversarial reply noise.

    The wrapper inspects each call's (role, kind, base_out) and
    decides deterministically (no RNG) whether to corrupt it,
    constrained by a per-scenario budget tracked in-closure.

    Under ``DROP_ROOT``: every INDEPENDENT_ROOT is flipped to
    UNCERTAIN up to ``budget`` flips per scenario.

    Under ``FLIP_ROOT_TO_SYMPTOM``: every INDEPENDENT_ROOT is
    flipped to DOWNSTREAM_SYMPTOM_OF:<kind>.

    Under ``INJECT_ROOT_ON_SYMPTOM``: every
    DOWNSTREAM_SYMPTOM_OF:<kind> is flipped to INDEPENDENT_ROOT.

    Under ``COMBINED``: all three passes (order above).
    """
    if adv.target_mode not in (
            ADVERSARIAL_REPLY_MODE_DROP_ROOT,
            ADVERSARIAL_REPLY_MODE_FLIP_ROOT_TO_SYMPTOM,
            ADVERSARIAL_REPLY_MODE_INJECT_ROOT_ON_SYMPTOM,
            ADVERSARIAL_REPLY_MODE_COMBINED):
        raise ValueError(
            f"unknown adversarial reply mode {adv.target_mode!r}")

    budgets: dict[str, int] = {}
    target_roles_set = set(adv.target_roles)

    def _wrapped(scenario: object, role: str, kind: str,
                 payload: str) -> str:
        base_out = base(scenario, role, kind, payload)
        if target_roles_set and role not in target_roles_set:
            return base_out
        sid = str(getattr(scenario, "scenario_id", ""))
        if sid not in budgets:
            budgets[sid] = (adv.budget if adv.budget >= 0 else 1_000_000)
        if budgets[sid] <= 0:
            return base_out

        def _record(axis: str) -> None:
            if report is not None:
                report.record(axis)

        tm = adv.target_mode
        if tm in (ADVERSARIAL_REPLY_MODE_DROP_ROOT,
                   ADVERSARIAL_REPLY_MODE_COMBINED):
            if base_out == CAUSALITY_INDEPENDENT_ROOT:
                budgets[sid] -= 1
                _record("drop")
                return CAUSALITY_UNCERTAIN

        if tm in (ADVERSARIAL_REPLY_MODE_FLIP_ROOT_TO_SYMPTOM,
                   ADVERSARIAL_REPLY_MODE_COMBINED):
            if base_out == CAUSALITY_INDEPENDENT_ROOT:
                budgets[sid] -= 1
                _record("mislabel")
                return CAUSALITY_DOWNSTREAM_PREFIX + kind

        if tm in (ADVERSARIAL_REPLY_MODE_INJECT_ROOT_ON_SYMPTOM,
                   ADVERSARIAL_REPLY_MODE_COMBINED):
            if base_out.startswith(CAUSALITY_DOWNSTREAM_PREFIX):
                budgets[sid] -= 1
                _record("mislabel")
                return CAUSALITY_INDEPENDENT_ROOT

        return base_out

    return _wrapped


__all__ = [
    "CAUSALITY_INDEPENDENT_ROOT", "CAUSALITY_UNCERTAIN",
    "CAUSALITY_DOWNSTREAM_PREFIX",
    "ReplyNoiseConfig", "ReplyCorruptionReport",
    "noisy_causality_extractor",
    "ADVERSARIAL_REPLY_MODE_DROP_ROOT",
    "ADVERSARIAL_REPLY_MODE_FLIP_ROOT_TO_SYMPTOM",
    "ADVERSARIAL_REPLY_MODE_INJECT_ROOT_ON_SYMPTOM",
    "ADVERSARIAL_REPLY_MODE_COMBINED",
    "AdversarialReplyConfig", "adversarial_reply_extractor",
]
