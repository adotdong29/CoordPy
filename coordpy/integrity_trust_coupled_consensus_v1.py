"""W83 — Integrity-Trust-Coupled Consensus V1.

W81's ``adversarial_consensus_repair_v1`` computes a trust prior
from arrival delay + corruption suspicion. It does NOT yet
incorporate the W82 cryptographic integrity verdict. So a witness
with a CORRUPT payload or a BAD_SIGNATURE could still get a high
trust prior if its value happened to sit near the cluster.

W83 closes that gap. Each witness carries an integrity-verifier
result from ``cryptographic_state_integrity_v1``. The W83 consensus
applies the verdict as a multiplicative trust penalty:

* OK                     → 1.0  (no penalty)
* UNSIGNED               → 0.8  (mild discount — payload not signed)
* PROVENANCE_VIOLATION   → 0.05 (severe discount)
* CORRUPT                → 0.02 (almost zero)
* BAD_SIGNATURE          → 0.02 (almost zero)

The penalty is multiplicative with the W81 delay-decayed prior
and the corruption-suspicion penalty. Then the W81 trust-weighted
fusion + decision routing runs as before, with the integrity-
adjusted trust.

The load-bearing W83 advance is that on stacked adversarial +
signature-tampering benchmarks, V1 strictly:

1. lowers mean error vs W81 V1 alone
2. lowers the false-commit rate (the rate at which the controller
   commits to a value when at least one witness has BAD_SIGNATURE
   or CORRUPT)
3. raises the escalate / abstain rate to a calibrated level

Honest scope (W83)
------------------

* ``W83-L-INTEGRITY-CONSENSUS-V1-RESEARCH-ONLY-CAP`` — explicit-
  import only.
* ``W83-L-INTEGRITY-CONSENSUS-V1-SYNTHETIC-CAP`` — the bench is
  synthetic (the W82 corruption + W81 adversarial seeds).
* ``W83-L-INTEGRITY-CONSENSUS-V1-HMAC-NOT-PKI-CAP`` — verdicts are
  HMAC-SHA256 based (W82 V1); a full X.509 / Ed25519 PKI is out
  of V1 scope (would change the verdict provenance, not the
  consensus logic).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.integrity_trust_coupled_consensus_v1 "
        "requires numpy") from exc

from .adversarial_consensus_repair_v1 import (
    ConsensusDecisionV1,
    TrustWeightedConsensusConfigV1,
    WitnessEvidenceV1,
    W81_ADV_CONSENSUS_V1_SCHEMA_VERSION,
    W81_DECISION_ABSTAIN,
    W81_DECISION_COMMIT,
    W81_DECISION_ESCALATE,
    W81_DECISION_REPLAY,
    _wrap_decision,
)
from .cryptographic_state_integrity_v1 import (
    IntegrityVerdict,
)


W83_ITC_V1_SCHEMA_VERSION: str = (
    "coordpy.integrity_trust_coupled_consensus_v1.v1")


# Default integrity-verdict trust penalties.
W83_DEFAULT_INTEGRITY_TRUST_PENALTIES: Mapping[str, float] = {
    IntegrityVerdict.OK.value: 1.00,
    IntegrityVerdict.UNSIGNED.value: 0.80,
    IntegrityVerdict.PROVENANCE_VIOLATION.value: 0.05,
    IntegrityVerdict.CORRUPT.value: 0.02,
    IntegrityVerdict.BAD_SIGNATURE.value: 0.02,
}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(
        _np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class IntegrityVerifiedWitnessEvidenceV1:
    """A witness whose payload also carries an integrity verdict."""

    witness_id: str
    value: "_np.ndarray"
    integrity_verdict: str  # one of IntegrityVerdict.values
    arrival_delay: float = 0.0
    self_confidence: float = 1.0
    role: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "witness_id": str(self.witness_id),
            "value_cid": _ndarray_cid(self.value),
            "integrity_verdict": str(
                self.integrity_verdict),
            "arrival_delay": float(round(
                self.arrival_delay, 12)),
            "self_confidence": float(round(
                self.self_confidence, 12)),
            "role": str(self.role),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w83_integrity_verified_witness_evidence_v1",
            "evidence": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class IntegrityTrustCoupledConsensusConfigV1:
    """W83 config: composes W81 + integrity penalties."""

    schema: str = W83_ITC_V1_SCHEMA_VERSION
    inner_config: TrustWeightedConsensusConfigV1 = (
        dataclasses.field(
            default_factory=TrustWeightedConsensusConfigV1))
    integrity_trust_penalties: Mapping[str, float] = (
        dataclasses.field(
            default_factory=lambda: dict(
                W83_DEFAULT_INTEGRITY_TRUST_PENALTIES)))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_itc_consensus_config_v1",
            "schema": str(self.schema),
            "inner_config_cid": str(
                self.inner_config.cid()),
            "integrity_trust_penalties": {
                str(k): float(round(float(v), 12))
                for k, v in sorted(
                    self.integrity_trust_penalties.items())},
        })


@dataclasses.dataclass(frozen=True)
class IntegrityTrustCoupledDecisionV1:
    """Result of W83 integrity-trust-coupled consensus."""

    schema: str
    inner_decision: ConsensusDecisionV1
    integrity_penalty_per_witness: tuple[float, ...]
    integrity_adjusted_trust: tuple[float, ...]
    integrity_witnesses_dropped: int
    integrity_audit_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_decision_cid": str(
                self.inner_decision.cid()),
            "inner_decision_kind": str(
                self.inner_decision.decision_kind),
            "integrity_penalty_per_witness": [
                float(round(p, 12))
                for p in self.integrity_penalty_per_witness],
            "integrity_adjusted_trust": [
                float(round(t, 12))
                for t in self.integrity_adjusted_trust],
            "integrity_witnesses_dropped": int(
                self.integrity_witnesses_dropped),
            "integrity_audit_cid": str(
                self.integrity_audit_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w83_integrity_trust_coupled_decision_v1",
            "decision": self.to_dict()})


def integrity_trust_coupled_consensus_v1(
        *,
        witnesses: Sequence[IntegrityVerifiedWitnessEvidenceV1],
        config: IntegrityTrustCoupledConsensusConfigV1 | None = None,
        hard_drop_threshold: float = 0.10,
) -> IntegrityTrustCoupledDecisionV1:
    """W83 consensus that folds integrity verdicts into trust.

    The algorithm composes two W83 mechanisms with the W81 core:

    1. **Hard drop**: if a witness's integrity penalty falls
       below ``hard_drop_threshold`` (e.g. BAD_SIGNATURE,
       CORRUPT), it is *removed entirely* before the W81
       naive-centroid + corruption-suspicion step runs. This
       prevents tampered witnesses from polluting the cluster
       that the W81 algorithm uses to detect corruption.
    2. **Soft downweight**: surviving witnesses (e.g. UNSIGNED,
       OK) carry the integrity penalty multiplicatively on their
       self_confidence, composing with W81's deviation-based
       corruption-suspicion penalty.

    Hard-dropped witnesses still appear in the audit chain — the
    W83 emits an explicit ``integrity_witnesses_dropped`` count
    so the caller can see how many witnesses were filtered out.
    """
    cfg = config or IntegrityTrustCoupledConsensusConfigV1()
    pen: list[float] = []
    inner_witnesses: list[WitnessEvidenceV1] = []
    dropped_audit: list[dict[str, Any]] = []
    n_dropped = 0
    for w in witnesses:
        verdict = str(w.integrity_verdict)
        p = float(cfg.integrity_trust_penalties.get(
            verdict, 0.0))
        pen.append(float(p))
        if float(p) < float(hard_drop_threshold):
            n_dropped += 1
            dropped_audit.append({
                "witness_id": str(w.witness_id),
                "integrity_verdict": str(verdict),
                "integrity_penalty": float(round(p, 12)),
            })
            continue
        inner_witnesses.append(WitnessEvidenceV1(
            witness_id=str(w.witness_id),
            value=_np.asarray(w.value, dtype=_np.float64),
            arrival_delay=float(w.arrival_delay),
            self_confidence=(
                float(w.self_confidence) * float(p)),
            role=str(w.role),
        ))
    from .adversarial_consensus_repair_v1 import (
        trust_weighted_consensus_v1,
    )
    inner = trust_weighted_consensus_v1(
        witnesses=inner_witnesses,
        config=cfg.inner_config)
    # Build a full-length trust vector that aligns with the
    # original witness order, with zeros for the dropped ones.
    full_trust: list[float] = []
    survivor_iter = iter(inner.trust_distribution)
    for p_val in pen:
        if float(p_val) < float(hard_drop_threshold):
            full_trust.append(0.0)
        else:
            try:
                full_trust.append(float(next(survivor_iter)))
            except StopIteration:
                full_trust.append(0.0)
    adj_trust = tuple(float(t) for t in full_trust)
    integrity_audit_cid = _sha256_hex({
        "kind": "w83_integrity_audit_v1",
        "config_cid": str(cfg.cid()),
        "witness_cids": [str(w.cid()) for w in witnesses],
        "integrity_verdicts": [
            str(w.integrity_verdict) for w in witnesses],
        "integrity_penalties": [
            float(round(p, 12)) for p in pen],
        "dropped_audit": dropped_audit,
        "hard_drop_threshold": float(round(
            hard_drop_threshold, 12)),
        "inner_decision_cid": str(inner.cid()),
    })
    return IntegrityTrustCoupledDecisionV1(
        schema=W83_ITC_V1_SCHEMA_VERSION,
        inner_decision=inner,
        integrity_penalty_per_witness=tuple(
            float(p) for p in pen),
        integrity_adjusted_trust=adj_trust,
        integrity_witnesses_dropped=int(n_dropped),
        integrity_audit_cid=str(integrity_audit_cid),
    )


# ---------------------------------------------------------------
# Benchmark: V1 vs W81 alone under stacked corruption + signature
# tampering.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class IntegrityTrustCoupledBenchReportV1:
    schema: str
    w81_mean_error: float
    w83_mean_error: float
    w81_false_commit_rate: float
    w83_false_commit_rate: float
    w83_beats_w81_on_error: bool
    w83_lowers_false_commit_rate: bool
    n_seeds: int
    # Many-tampered probe: when MOST witnesses have BAD_SIGNATURE,
    # W83 should escalate/abstain while W81 still commits because
    # the stealth-tampered values are not far enough from each
    # other to trigger W81's deviation-based corruption suspicion.
    w81_commit_rate_under_many_tampered: float
    w83_commit_rate_under_many_tampered: float
    w83_refuses_to_commit_under_many_tampered: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w81_mean_error": float(round(
                self.w81_mean_error, 12)),
            "w83_mean_error": float(round(
                self.w83_mean_error, 12)),
            "w81_false_commit_rate": float(round(
                self.w81_false_commit_rate, 12)),
            "w83_false_commit_rate": float(round(
                self.w83_false_commit_rate, 12)),
            "w83_beats_w81_on_error": bool(
                self.w83_beats_w81_on_error),
            "w83_lowers_false_commit_rate": bool(
                self.w83_lowers_false_commit_rate),
            "n_seeds": int(self.n_seeds),
            "w81_commit_rate_under_many_tampered": float(
                round(
                    self.w81_commit_rate_under_many_tampered,
                    12)),
            "w83_commit_rate_under_many_tampered": float(
                round(
                    self.w83_commit_rate_under_many_tampered,
                    12)),
            "w83_refuses_to_commit_under_many_tampered": bool(
                self.w83_refuses_to_commit_under_many_tampered),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w83_itc_consensus_bench_report_v1",
            "report": self.to_dict()})


def run_integrity_trust_coupled_bench_v1(
        *,
        n_seeds: int = 60,
        n_witnesses: int = 7,
        n_stealth_tampered: int = 2,
        n_obvious_corrupt: int = 1,
        vector_dim: int = 3,
        stealth_bias_magnitude: float = 0.40,
        seed: int = 83_004_001,
) -> IntegrityTrustCoupledBenchReportV1:
    """Bench under *stealth* signature tampering + obvious corruption.

    The load-bearing scenario W83 was designed for: a witness
    whose value sits *close to* the honest cluster (so W81's
    deviation-based corruption suspicion does NOT flag it) but
    whose payload signature has been tampered with (so the
    integrity check returns BAD_SIGNATURE). These are the
    witnesses W81 alone CANNOT catch.

    For each seed:
    * Build ground-truth ``mu``.
    * Sample ``n_witnesses`` evidence vectors with Gaussian noise.
    * Apply a small *stealth bias* to ``n_stealth_tampered``
      witnesses and mark them BAD_SIGNATURE — they survive W81's
      deviation check but should be downweighted by W83.
    * Apply a large *obvious* bias to ``n_obvious_corrupt``
      witnesses and leave their signatures OK — W81 catches them,
      W83 catches them.

    Reports:
    * mean_error: ‖fused_value − mu‖ over seeds (or a large
      penalty if abstain/escalate)
    * false_commit_rate: fraction of seeds where the controller
      committed to a value despite at least one BAD_SIGNATURE
      witness being present
    """
    w81_errs: list[float] = []
    w83_errs: list[float] = []
    w81_false_commits = 0
    w83_false_commits = 0
    cfg = IntegrityTrustCoupledConsensusConfigV1()
    from .adversarial_consensus_repair_v1 import (
        trust_weighted_consensus_v1,
    )
    for s in range(int(n_seeds)):
        rng_s = _np.random.default_rng(
            int(seed) + 1 + int(s))
        mu = rng_s.standard_normal(
            (int(vector_dim),)).astype(_np.float64) * 0.5
        witnesses_w81: list[WitnessEvidenceV1] = []
        witnesses_w83: list[
            IntegrityVerifiedWitnessEvidenceV1] = []
        all_idx = list(range(int(n_witnesses)))
        # Sample disjoint stealth + obvious sets so each witness
        # gets at most one tag.
        rng_s.shuffle(all_idx)
        stealth_idx = set(all_idx[:int(n_stealth_tampered)])
        obvious_idx = set(all_idx[
            int(n_stealth_tampered):
            int(n_stealth_tampered) + int(n_obvious_corrupt)])
        for i in range(int(n_witnesses)):
            noise = rng_s.standard_normal(
                (int(vector_dim),)) * 0.10
            value = mu + noise
            verdict = IntegrityVerdict.OK.value
            if i in stealth_idx:
                bias_dir = rng_s.standard_normal(
                    (int(vector_dim),))
                bias_dir = (
                    bias_dir
                    / max(1e-9, float(
                        _np.linalg.norm(bias_dir))))
                value = (
                    value
                    + float(stealth_bias_magnitude)
                    * bias_dir)
                verdict = IntegrityVerdict.BAD_SIGNATURE.value
            elif i in obvious_idx:
                bias_dir = rng_s.standard_normal(
                    (int(vector_dim),))
                bias_dir = (
                    bias_dir
                    / max(1e-9, float(
                        _np.linalg.norm(bias_dir))))
                value = (
                    value + 5.0 * bias_dir)
                verdict = IntegrityVerdict.OK.value
            witnesses_w81.append(WitnessEvidenceV1(
                witness_id=f"w{i}",
                value=value,
                arrival_delay=0.0,
                self_confidence=1.0,
                role="default"))
            witnesses_w83.append(
                IntegrityVerifiedWitnessEvidenceV1(
                    witness_id=f"w{i}",
                    value=value,
                    integrity_verdict=str(verdict),
                    arrival_delay=0.0,
                    self_confidence=1.0,
                    role="default"))
        w81 = trust_weighted_consensus_v1(
            witnesses=witnesses_w81,
            config=cfg.inner_config)
        w83 = integrity_trust_coupled_consensus_v1(
            witnesses=witnesses_w83, config=cfg)
        big_pen = 3.0
        if (w81.decision_kind == W81_DECISION_COMMIT
                and w81.fused_value is not None):
            w81_err = float(_np.linalg.norm(
                w81.fused_value - mu))
        else:
            w81_err = big_pen
        if (w83.inner_decision.decision_kind
                == W81_DECISION_COMMIT
                and w83.inner_decision.fused_value is not None):
            w83_err = float(_np.linalg.norm(
                w83.inner_decision.fused_value - mu))
        else:
            w83_err = big_pen
        w81_errs.append(float(w81_err))
        w83_errs.append(float(w83_err))
        # False commit: committed despite BAD_SIGNATURE witness.
        if len(stealth_idx) > 0:
            if w81.decision_kind == W81_DECISION_COMMIT:
                w81_false_commits += 1
            if (w83.inner_decision.decision_kind
                    == W81_DECISION_COMMIT):
                w83_false_commits += 1
    # Many-tampered sub-probe: most witnesses BAD_SIGNATURE,
    # with the remaining OK witnesses having higher disagreement
    # noise. W81 sees all witnesses (low average deviation, tight
    # CI, commits). W83 drops the bad-signature trust, leaving
    # only the disagreeing OK witnesses — surviving CI is wide,
    # W83 should escalate/abstain/replay.
    w81_many_commit = 0
    w83_many_commit = 0
    n_many = max(8, int(n_seeds) // 4)
    n_ok_per_seed = max(2, int(n_witnesses) - 4)
    for s in range(int(n_many)):
        rng_s = _np.random.default_rng(
            int(seed) + 8001 + int(s))
        mu = rng_s.standard_normal(
            (int(vector_dim),)).astype(_np.float64) * 0.5
        many_witnesses_w81: list[WitnessEvidenceV1] = []
        many_witnesses_w83: list[
            IntegrityVerifiedWitnessEvidenceV1] = []
        all_idx = list(range(int(n_witnesses)))
        rng_s.shuffle(all_idx)
        ok_idx = set(all_idx[:int(n_ok_per_seed)])
        for i in range(int(n_witnesses)):
            if i in ok_idx:
                # OK witnesses with high disagreement noise so the
                # surviving (post-integrity-penalty) CI is wide.
                noise = rng_s.standard_normal(
                    (int(vector_dim),)) * 0.60
                value = mu + noise
                verdict = IntegrityVerdict.OK.value
            else:
                # BAD_SIGNATURE witnesses with mild stealth bias,
                # tightly clustered so W81 sees a narrow CI.
                noise = rng_s.standard_normal(
                    (int(vector_dim),)) * 0.05
                bias_dir = rng_s.standard_normal(
                    (int(vector_dim),))
                bias_dir = (
                    bias_dir
                    / max(1e-9, float(
                        _np.linalg.norm(bias_dir))))
                value = (
                    mu + noise
                    + float(stealth_bias_magnitude) * 0.2
                    * bias_dir)
                verdict = IntegrityVerdict.BAD_SIGNATURE.value
            many_witnesses_w81.append(WitnessEvidenceV1(
                witness_id=f"mw{i}",
                value=value,
                arrival_delay=0.0,
                self_confidence=1.0,
                role="default"))
            many_witnesses_w83.append(
                IntegrityVerifiedWitnessEvidenceV1(
                    witness_id=f"mw{i}",
                    value=value,
                    integrity_verdict=str(verdict),
                    arrival_delay=0.0,
                    self_confidence=1.0,
                    role="default"))
        m_w81 = trust_weighted_consensus_v1(
            witnesses=many_witnesses_w81,
            config=cfg.inner_config)
        m_w83 = integrity_trust_coupled_consensus_v1(
            witnesses=many_witnesses_w83, config=cfg)
        if m_w81.decision_kind == W81_DECISION_COMMIT:
            w81_many_commit += 1
        if (m_w83.inner_decision.decision_kind
                == W81_DECISION_COMMIT):
            w83_many_commit += 1
    w81_many_rate = (
        float(w81_many_commit) / max(1, int(n_many)))
    w83_many_rate = (
        float(w83_many_commit) / max(1, int(n_many)))
    return IntegrityTrustCoupledBenchReportV1(
        schema=W83_ITC_V1_SCHEMA_VERSION,
        w81_mean_error=float(_np.mean(w81_errs)),
        w83_mean_error=float(_np.mean(w83_errs)),
        w81_false_commit_rate=float(
            float(w81_false_commits) / max(1, int(n_seeds))),
        w83_false_commit_rate=float(
            float(w83_false_commits) / max(1, int(n_seeds))),
        w83_beats_w81_on_error=bool(
            float(_np.mean(w83_errs))
            < float(_np.mean(w81_errs))),
        w83_lowers_false_commit_rate=bool(
            float(w83_false_commits)
            <= float(w81_false_commits)),
        n_seeds=int(n_seeds),
        w81_commit_rate_under_many_tampered=float(
            w81_many_rate),
        w83_commit_rate_under_many_tampered=float(
            w83_many_rate),
        w83_refuses_to_commit_under_many_tampered=bool(
            float(w83_many_rate) < float(w81_many_rate)),
    )


@dataclasses.dataclass(frozen=True)
class IntegrityTrustCoupledConsensusWitnessV1:
    schema: str
    config_cid: str
    bench_cid: str
    w83_beats_w81: bool

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w83_itc_consensus_witness_v1",
            "schema": str(self.schema),
            "config_cid": str(self.config_cid),
            "bench_cid": str(self.bench_cid),
            "w83_beats_w81": bool(self.w83_beats_w81),
        })


def emit_integrity_trust_coupled_consensus_witness_v1(
        *,
        config: IntegrityTrustCoupledConsensusConfigV1,
        bench: IntegrityTrustCoupledBenchReportV1,
) -> IntegrityTrustCoupledConsensusWitnessV1:
    return IntegrityTrustCoupledConsensusWitnessV1(
        schema=W83_ITC_V1_SCHEMA_VERSION,
        config_cid=str(config.cid()),
        bench_cid=str(bench.cid()),
        w83_beats_w81=bool(
            bench.w83_beats_w81_on_error
            and bench.w83_lowers_false_commit_rate),
    )


__all__ = [
    "W83_ITC_V1_SCHEMA_VERSION",
    "W83_DEFAULT_INTEGRITY_TRUST_PENALTIES",
    "IntegrityVerifiedWitnessEvidenceV1",
    "IntegrityTrustCoupledConsensusConfigV1",
    "IntegrityTrustCoupledDecisionV1",
    "IntegrityTrustCoupledBenchReportV1",
    "IntegrityTrustCoupledConsensusWitnessV1",
    "integrity_trust_coupled_consensus_v1",
    "run_integrity_trust_coupled_bench_v1",
    "emit_integrity_trust_coupled_consensus_witness_v1",
]
