"""Phase 37 Part A — per-call calibration of a thread-reply path.

Phase 36 built the synthetic reply-noise channel
(``core/reply_noise.ReplyNoiseConfig``) with Bernoulli drop /
mislabel knobs. The gap Phase 37 closes: *is the synthetic
parameterisation a good surrogate for what a real LLM actually
does?*

Two observations from real runs on the Phase-35 contested bank
motivated this module:

  1. Real 0.5b and 7b Ollama models return **well-formed JSON**
     with a parseable ``reply_kind`` on ≫ 90 % of the calls.
     Malformed-output rate is small.
  2. The *semantic* fidelity — whether the emitted ``reply_kind``
     matches the oracle — is much lower. On TLS_EXPIRED (oracle
     ``INDEPENDENT_ROOT``), qwen2.5:0.5b and qwen2.5-coder:7b
     both emit ``DOWNSTREAM_SYMPTOM`` under the default prompt.

That split — "JSON is fine, but the class is wrong" — is
invisible to ``LLMReplierStats`` as it stands: it logs
well-formed vs malformed vs out-of-vocab but not the per-call
semantic bucket. This module adds that bucket by wrapping the
replier with an oracle comparator.

Calibration buckets
-------------------

Per call ``(scenario, role, kind, payload)``, the replier emits
``(reply_kind, witness, well_formed)``. Comparing against the
oracle's answer partitions the call into exactly one bucket:

  * ``CAL_CORRECT``              — well-formed, reply_kind matches
    the oracle.
  * ``CAL_MALFORMED``            — parser returned ``well_formed =
    False`` and no JSON was recovered.
  * ``CAL_OUT_OF_VOCAB``         — JSON recovered but reply_kind
    is not in the allowed enum.
  * ``CAL_SEM_ROOT_AS_SYMPTOM``  — oracle says INDEPENDENT_ROOT,
    replier says DOWNSTREAM_SYMPTOM. Dominant Phase-37 finding on
    weak models.
  * ``CAL_SEM_ROOT_AS_UNCERTAIN``— oracle says INDEPENDENT_ROOT,
    replier says UNCERTAIN. Under the Phase-36 reply-resolution
    rule this collapses the thread to NO_CONSENSUS.
  * ``CAL_SEM_SYMPTOM_AS_ROOT``  — oracle says DOWNSTREAM_SYMPTOM,
    replier says INDEPENDENT_ROOT. Creates a false CONFLICT on
    two-candidate scenarios.
  * ``CAL_SEM_SYMPTOM_AS_UNCERTAIN`` — mirror.
  * ``CAL_SEM_UNCERTAIN_AS_ROOT`` — oracle says UNCERTAIN, replier
    says INDEPENDENT_ROOT. Creates a false IR on the shadow
    candidate.
  * ``CAL_SEM_UNCERTAIN_AS_SYMPTOM`` — benign in Phase-35; the
    shadow reply was already UNCERTAIN, now it is a non-load-
    bearing DOWNSTREAM_SYMPTOM.
  * ``CAL_WITNESS_TRUNCATED``    — witness was clamped to the
    token cap. Not a correctness bucket; counted separately so we
    know how often the cap fires.

The rate of each bucket, computed on the Phase-35 bank, is the
*real reply-noise profile* of the LLM under test. The Phase-37
headline compares this to the synthetic
``ReplyNoiseConfig(drop_prob, mislabel_prob)`` that best
reproduces the same accuracy curve — the answer is: the
synthetic model under-counts the semantic-mislabel axis.

Scope discipline
----------------

  * This module does NOT change the ``LLMThreadReplier`` itself.
    It wraps the replier and intercepts its ``(reply_kind,
    witness, well_formed)`` tuple to record a bucket. The thread
    primitive sees the same tuple it would have seen without
    this wrapper.
  * The oracle passed in is the caller's responsibility; we do
    not re-implement scenario-specific reasoning here. The
    Phase-35 oracle is ``infer_causality_hypothesis``.
  * The ``witness_token_cap`` read here is the *same* cap that
    the replier's ``LLMReplyConfig`` uses — mismatched caps make
    the truncation counter meaningless.

Theoretical anchor: RESULTS_PHASE37.md § B.1 (Theorem P37-1,
Conjecture C37-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from vision_mvp.core.dynamic_comm import (
    REPLY_AGREE, REPLY_DEFER_TO, REPLY_DISAGREE,
    REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN,
)
from vision_mvp.core.llm_thread_replier import LLMThreadReplier
from vision_mvp.core.reply_noise import (
    CAUSALITY_DOWNSTREAM_PREFIX, CAUSALITY_INDEPENDENT_ROOT,
    CAUSALITY_UNCERTAIN,
)


CausalityOracle = Callable[[object, str, str, str], str]


# Calibration bucket names.
CAL_CORRECT = "correct"
CAL_MALFORMED = "malformed"
CAL_OUT_OF_VOCAB = "out_of_vocab"
CAL_SEM_ROOT_AS_SYMPTOM = "sem_root_as_symptom"
CAL_SEM_ROOT_AS_UNCERTAIN = "sem_root_as_uncertain"
CAL_SEM_SYMPTOM_AS_ROOT = "sem_symptom_as_root"
CAL_SEM_SYMPTOM_AS_UNCERTAIN = "sem_symptom_as_uncertain"
CAL_SEM_UNCERTAIN_AS_ROOT = "sem_uncertain_as_root"
CAL_SEM_UNCERTAIN_AS_SYMPTOM = "sem_uncertain_as_symptom"

# Meta-bucket, counted orthogonally to correctness. A call can be
# both ``correct`` and ``witness_truncated``; the first counter
# tracks the correctness axis, this one tracks the budget axis.
CAL_WITNESS_TRUNCATED = "witness_truncated"


ALL_CAL_BUCKETS = (
    CAL_CORRECT, CAL_MALFORMED, CAL_OUT_OF_VOCAB,
    CAL_SEM_ROOT_AS_SYMPTOM, CAL_SEM_ROOT_AS_UNCERTAIN,
    CAL_SEM_SYMPTOM_AS_ROOT, CAL_SEM_SYMPTOM_AS_UNCERTAIN,
    CAL_SEM_UNCERTAIN_AS_ROOT, CAL_SEM_UNCERTAIN_AS_SYMPTOM,
)


def _oracle_to_reply_kind(oracle_out: str) -> str:
    """Map the oracle's causality-class string to a thread-reply
    kind. Mirrors the adapter in
    ``contested_incident.run_dynamic_coordination``.
    """
    if oracle_out == CAUSALITY_INDEPENDENT_ROOT:
        return REPLY_INDEPENDENT_ROOT
    if oracle_out.startswith(CAUSALITY_DOWNSTREAM_PREFIX):
        return REPLY_DOWNSTREAM_SYMPTOM
    return REPLY_UNCERTAIN


@dataclass
class ReplyCalibrationReport:
    """Per-call bucket counters, plus aggregate rates.

    ``record`` is called once per (scenario, role, kind, payload)
    call. Aggregate rates are returned by ``rates()``.

    Witness truncation is tracked orthogonally because a
    correctness bucket and a witness-truncation flag are
    independent dimensions of the same call.
    """

    n_calls: int = 0
    buckets: dict[str, int] = field(default_factory=lambda: {
        b: 0 for b in ALL_CAL_BUCKETS})
    n_witness_truncated: int = 0
    # Raw (oracle_kind, replied_kind, correct_bool) rows — kept
    # for post-hoc analysis. Bounded in practice (≤ 6 scenarios ·
    # 2 candidates · 3 strategies per cell on the contested bank).
    rows: list[dict] = field(default_factory=list)

    def record(self, *, scenario_id: str, role: str, kind: str,
               payload: str, oracle_kind: str, replied_kind: str,
               well_formed: bool, had_json: bool,
               witness_truncated: bool) -> str:
        """Record one call. Returns the correctness-bucket name."""
        self.n_calls += 1
        if witness_truncated:
            self.n_witness_truncated += 1
        bucket = self._classify(
            oracle_kind=oracle_kind, replied_kind=replied_kind,
            well_formed=well_formed, had_json=had_json)
        self.buckets[bucket] = self.buckets.get(bucket, 0) + 1
        self.rows.append({
            "scenario_id": scenario_id, "role": role,
            "kind": kind, "payload_head": payload[:60],
            "oracle_kind": oracle_kind,
            "replied_kind": replied_kind,
            "well_formed": bool(well_formed),
            "bucket": bucket,
            "witness_truncated": bool(witness_truncated),
        })
        return bucket

    def _classify(self, *, oracle_kind: str, replied_kind: str,
                  well_formed: bool, had_json: bool) -> str:
        if not well_formed:
            if not had_json:
                return CAL_MALFORMED
            return CAL_OUT_OF_VOCAB
        if replied_kind == oracle_kind:
            return CAL_CORRECT
        # Well-formed disagreement — a semantic error. Six named
        # buckets cover the 3 × 3 − 3 = 6 distinct confusion pairs.
        o = oracle_kind
        r = replied_kind
        # Normalise anything not in the core trio (AGREE / DEFER_TO
        # / DISAGREE) by mapping to UNCERTAIN; the Phase-35
        # extractor rounds those to UNCERTAIN for the decoder.
        r_norm = self._normalize_reply(r)
        o_norm = self._normalize_reply(o)
        if o_norm == REPLY_INDEPENDENT_ROOT:
            if r_norm == REPLY_DOWNSTREAM_SYMPTOM:
                return CAL_SEM_ROOT_AS_SYMPTOM
            if r_norm == REPLY_UNCERTAIN:
                return CAL_SEM_ROOT_AS_UNCERTAIN
        if o_norm == REPLY_DOWNSTREAM_SYMPTOM:
            if r_norm == REPLY_INDEPENDENT_ROOT:
                return CAL_SEM_SYMPTOM_AS_ROOT
            if r_norm == REPLY_UNCERTAIN:
                return CAL_SEM_SYMPTOM_AS_UNCERTAIN
        if o_norm == REPLY_UNCERTAIN:
            if r_norm == REPLY_INDEPENDENT_ROOT:
                return CAL_SEM_UNCERTAIN_AS_ROOT
            if r_norm == REPLY_DOWNSTREAM_SYMPTOM:
                return CAL_SEM_UNCERTAIN_AS_SYMPTOM
        return CAL_OUT_OF_VOCAB

    @staticmethod
    def _normalize_reply(r: str) -> str:
        if r == REPLY_INDEPENDENT_ROOT:
            return REPLY_INDEPENDENT_ROOT
        if r == REPLY_DOWNSTREAM_SYMPTOM:
            return REPLY_DOWNSTREAM_SYMPTOM
        if r in (REPLY_UNCERTAIN, REPLY_AGREE, REPLY_DISAGREE,
                 REPLY_DEFER_TO):
            return REPLY_UNCERTAIN
        return REPLY_UNCERTAIN

    def rates(self) -> dict:
        """Return aggregate per-bucket rates in ``[0, 1]``."""
        n = max(1, self.n_calls)
        out = {
            "n_calls": self.n_calls,
            "n_witness_truncated": self.n_witness_truncated,
            "witness_truncation_rate": self.n_witness_truncated / n,
        }
        semantic_sum = 0
        for b in ALL_CAL_BUCKETS:
            c = self.buckets.get(b, 0)
            out[b] = c
            out[b + "_rate"] = c / n
            if b.startswith("sem_"):
                semantic_sum += c
        out["semantic_wrong_count"] = semantic_sum
        out["semantic_wrong_rate"] = semantic_sum / n
        # Effective rates that map to Phase-36 ReplyNoiseConfig.
        # drop_prob analogue: replier emitted UNCERTAIN on an IR
        # ground-truth.
        drop_count = self.buckets.get(CAL_SEM_ROOT_AS_UNCERTAIN, 0)
        ir_ground_truth = (
            self.buckets.get(CAL_SEM_ROOT_AS_UNCERTAIN, 0)
            + self.buckets.get(CAL_SEM_ROOT_AS_SYMPTOM, 0))
        out["effective_drop_prob_conditional_on_ir"] = \
            drop_count / max(1, ir_ground_truth
                             + self._ir_correct_count())
        # mislabel_prob analogue: replier emitted the wrong IR /
        # SYMPTOM class.
        mis_count = (
            self.buckets.get(CAL_SEM_ROOT_AS_SYMPTOM, 0)
            + self.buckets.get(CAL_SEM_SYMPTOM_AS_ROOT, 0))
        out["effective_mislabel_count"] = mis_count
        out["effective_mislabel_rate"] = mis_count / n
        return out

    def _ir_correct_count(self) -> int:
        # Oracle == IR AND replied == IR. Requires a row pass.
        n = 0
        for row in self.rows:
            if (row["oracle_kind"] == CAUSALITY_INDEPENDENT_ROOT
                    and row.get("replied_kind")
                    == REPLY_INDEPENDENT_ROOT
                    and row.get("bucket") == CAL_CORRECT):
                n += 1
        return n

    def as_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "buckets": dict(self.buckets),
            "n_witness_truncated": self.n_witness_truncated,
            "rows": list(self.rows),
            "rates": self.rates(),
        }


@dataclass
class CalibratingReplier:
    """Wrap an ``LLMThreadReplier`` with per-call oracle comparison.

    Calls are forwarded to ``inner``; the replier's own ``(reply_
    kind, witness, well_formed)`` tuple is returned unchanged so
    the thread primitive sees the same output it would without
    calibration. On each call, ``report`` records a bucket.

    The oracle is called to obtain the ground-truth causality
    class for the same ``(scenario, role, kind, payload)``.

    Optional ``witness_token_cap`` — if provided and the replier
    emitted a witness whose raw token count *before* the
    replier's own clamp exceeded the cap, the ``witness_
    truncated`` flag is set. We cannot observe the *raw* LLM
    reply directly after the replier; the post-clamp witness
    length is our proxy.
    """

    inner: LLMThreadReplier
    oracle: CausalityOracle
    report: ReplyCalibrationReport = field(
        default_factory=ReplyCalibrationReport)
    witness_token_cap: int | None = None

    def __call__(self,
                 scenario: object,
                 role: str,
                 kind: str,
                 payload: str,
                 other_candidates: Sequence[tuple[str, str, str]] = (),
                 role_events: Sequence[object] | None = None,
                 ) -> tuple[str, str, bool]:
        reply_kind, witness, well_formed = self.inner(
            scenario=scenario, role=role, kind=kind,
            payload=payload, other_candidates=other_candidates,
            role_events=role_events)
        oracle_out = self.oracle(scenario, role, kind, payload)
        oracle_reply_kind = _oracle_to_reply_kind(oracle_out)
        cap = self.witness_token_cap or self.inner.config.witness_token_cap
        # A clamped witness has exactly ``cap`` tokens; the signal
        # is not perfect (a call that *naturally* emitted exactly
        # ``cap`` tokens counts as "truncated"), but it upper-
        # bounds the truncation count.
        witness_tokens = witness.split() if witness else []
        witness_truncated = len(witness_tokens) >= cap
        # Inner stats tells us if a JSON was parsed; but the public
        # interface does not expose it per call. Re-derive using
        # the inner's cumulative stats deltas is fragile; we fall
        # back to the convention: replier returned ``well_formed ==
        # False`` and ``reply_kind`` equals the fallback
        # (UNCERTAIN) means either malformed or OOV. We cannot
        # distinguish without the raw text. We mark as MALFORMED
        # when replier ran but the inner-stats malformed counter
        # increased; else OOV. This is best-effort.
        had_json_flag = self._heuristic_had_json(well_formed)
        bucket = self.report.record(
            scenario_id=str(getattr(scenario, "scenario_id", "")),
            role=role, kind=kind, payload=payload,
            oracle_kind=oracle_reply_kind,
            replied_kind=reply_kind,
            well_formed=well_formed,
            had_json=had_json_flag,
            witness_truncated=witness_truncated,
        )
        # Keep bucket-rows aligned with inner stats.
        return reply_kind, witness, well_formed

    def _heuristic_had_json(self, well_formed: bool) -> bool:
        """Best-effort OOV vs malformed signal from inner stats.

        ``well_formed`` is False iff the reply was malformed
        (no JSON) OR out-of-vocab (JSON, unknown kind). The inner
        replier increments exactly one of ``n_malformed`` /
        ``n_out_of_vocab``. We read the inner counters and pick
        the one that increased on this call. First-call edge is
        handled by the inequality.
        """
        if well_formed:
            return True
        # Track last-known counters as attributes on the wrapper.
        last_m = getattr(self, "_last_malformed", 0)
        last_o = getattr(self, "_last_oov", 0)
        cur_m = self.inner.stats.n_malformed
        cur_o = self.inner.stats.n_out_of_vocab
        had_json = cur_o > last_o
        self._last_malformed = cur_m
        self._last_oov = cur_o
        return had_json


def causality_extractor_from_calibrating_replier(
        wrapper: CalibratingReplier,
        ) -> Callable[[object, str, str, str], str]:
    """Build a ``CausalityExtractor`` shape from a
    ``CalibratingReplier`` — drop-in for
    ``run_dynamic_coordination(causality_extractor=...)`` and
    ``run_adaptive_sub_coordination(causality_extractor=...)``.

    Mirrors ``causality_extractor_from_replier`` but keeps the
    calibration wrapper alive so per-call buckets are recorded.
    """

    def _extract(scenario: object, role: str,
                 kind: str, payload: str) -> str:
        reply_kind, _witness, well_formed = wrapper(
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
    "CAL_CORRECT", "CAL_MALFORMED", "CAL_OUT_OF_VOCAB",
    "CAL_SEM_ROOT_AS_SYMPTOM", "CAL_SEM_ROOT_AS_UNCERTAIN",
    "CAL_SEM_SYMPTOM_AS_ROOT", "CAL_SEM_SYMPTOM_AS_UNCERTAIN",
    "CAL_SEM_UNCERTAIN_AS_ROOT", "CAL_SEM_UNCERTAIN_AS_SYMPTOM",
    "CAL_WITNESS_TRUNCATED", "ALL_CAL_BUCKETS",
    "ReplyCalibrationReport", "CalibratingReplier",
    "causality_extractor_from_calibrating_replier",
]
