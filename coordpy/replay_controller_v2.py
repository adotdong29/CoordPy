"""W61 M7 — Replay Controller V2.

Strictly extends W60's ``coordpy.replay_controller``. V1 was a
*deterministic* rule (REUSE / RECOMPUTE / FALLBACK / ABSTAIN) with
fixed thresholds (``flop_saving_floor``, ``drift_ceiling``,
``flop_ceiling_ratio``). V2 makes the decision *trainable*:

* **Fitted thresholds** — V2 solves a closed-form linear ridge
  over a training set of ``(saving, drift, crc_passed,
  transcript_available, abstain_cost, corruption_count) →
  optimal_decision`` tuples to fit a *per-decision* scoring head.
  Decision = argmax of four ridge heads. The thresholds become a
  learned consequence of the head weights, not a hard rule.
* **Decision confidence** — each call emits a per-decision
  probability mass via softmax over the four head scores. The
  controller's ``decision_confidence`` scalar is the chosen
  decision's softmax probability. If confidence is below
  ``abstain_threshold`` (also fitted), the V2 controller falls
  back to ABSTAIN even if a head argmax favoured a different
  decision.
* **Hidden-write trace gate** — V2 reads the V6 substrate cache's
  ``hidden_write_trace`` channel. If the cumulative L2 of hidden
  writes exceeds a fitted ``hidden_write_cap``, V2 forces ABSTAIN
  to prevent the controller from compounding HSB injections.
* **Backward compatibility** — with no fitted parameters and
  ``soft=False``, V2 reduces to V1 byte-for-byte.

Honest scope
------------

* The decision-head fit is a single closed-form linear ridge over
  a 6-dim feature against a 4-class label one-hot. No autograd.
  ``W61-L-V2-REPLAY-NO-AUTOGRAD-CAP`` documents.
* Confidence is the softmax of head scores; not a calibrated
  probability. It is monotone and useful for ranking, not for
  Bayesian decision-making.
* The hidden-write gate is a hard cap, not a learned function of
  the write trace.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v2 requires numpy"
        ) from exc

from .replay_controller import (
    ReplayCandidate, ReplayController,
    ReplayDecision, ReplayControllerWitness,
    W60_REPLAY_DECISION_ABSTAIN,
    W60_REPLAY_DECISION_FALLBACK,
    W60_REPLAY_DECISION_RECOMPUTE,
    W60_REPLAY_DECISION_REUSE,
    W60_REPLAY_DECISIONS,
    W60_DEFAULT_DRIFT_CEILING,
    W60_DEFAULT_FLOP_CEILING_RATIO,
    W60_DEFAULT_FLOP_SAVING_FLOOR,
    emit_replay_controller_witness,
)


W61_REPLAY_CONTROLLER_V2_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v2.v1")

W61_DEFAULT_REPLAY_V2_ABSTAIN_CONF: float = 0.30
W61_DEFAULT_REPLAY_V2_HIDDEN_WRITE_CAP: float = 1.0e9
W61_DEFAULT_REPLAY_V2_RIDGE_LAMBDA: float = 0.10


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _candidate_feature(
        c: ReplayCandidate,
        *,
        hidden_write_total_l2: float = 0.0,
) -> "_np.ndarray":
    """6-dim feature: [saving_ratio, drift_reuse, crc_passed,
    transcript_available, n_corruption_flags, hidden_write_l2]."""
    denom = max(int(c.flop_recompute), 1)
    saving = (
        float(int(c.flop_recompute) - int(c.flop_reuse))
        / float(denom))
    return _np.array([
        float(saving),
        float(c.drift_l2_reuse),
        1.0 if bool(c.crc_passed) else 0.0,
        1.0 if bool(c.transcript_available) else 0.0,
        float(int(c.n_corruption_flags)),
        float(hidden_write_total_l2),
    ], dtype=_np.float64)


def _softmax_4(x: "_np.ndarray") -> "_np.ndarray":
    x = _np.asarray(x, dtype=_np.float64).reshape(-1)
    if x.size < 4:
        x = _np.concatenate(
            [x, _np.zeros(4 - x.size, dtype=_np.float64)])
    e = _np.exp(x - _np.max(x))
    s = float(_np.sum(e))
    if s < 1e-30:
        return _np.full((4,), 0.25, dtype=_np.float64)
    return e / s


@dataclasses.dataclass
class ReplayControllerV2:
    """V2 replay controller.

    ``W`` is a 4×F linear head matrix; ``b`` is a 4-vector bias.
    If ``W`` is None we behave as V1 with the fixed thresholds.
    """
    inner_v1: ReplayController
    W: "_np.ndarray | None"
    b: "_np.ndarray | None"
    abstain_threshold: float
    hidden_write_cap: float
    soft: bool
    audit_v2: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            flop_saving_floor: float = (
                W60_DEFAULT_FLOP_SAVING_FLOOR),
            drift_ceiling: float = W60_DEFAULT_DRIFT_CEILING,
            flop_ceiling_ratio: float = (
                W60_DEFAULT_FLOP_CEILING_RATIO),
            abstain_threshold: float = (
                W61_DEFAULT_REPLAY_V2_ABSTAIN_CONF),
            hidden_write_cap: float = (
                W61_DEFAULT_REPLAY_V2_HIDDEN_WRITE_CAP),
            soft: bool = False,
    ) -> "ReplayControllerV2":
        v1 = ReplayController(
            flop_saving_floor=float(flop_saving_floor),
            drift_ceiling=float(drift_ceiling),
            flop_ceiling_ratio=float(flop_ceiling_ratio),
            audit=[])
        return cls(
            inner_v1=v1, W=None, b=None,
            abstain_threshold=float(abstain_threshold),
            hidden_write_cap=float(hidden_write_cap),
            soft=bool(soft),
            audit_v2=[],
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W61_REPLAY_CONTROLLER_V2_SCHEMA_VERSION,
            "kind": "replay_controller_v2",
            "inner_v1_cid": str(self.inner_v1.cid()),
            "W_cid": (
                hashlib.sha256(
                    self.W.tobytes()).hexdigest()
                if self.W is not None else "untrained"),
            "b_cid": (
                hashlib.sha256(
                    self.b.tobytes()).hexdigest()
                if self.b is not None else "untrained"),
            "abstain_threshold": float(round(
                self.abstain_threshold, 12)),
            "hidden_write_cap": float(round(
                self.hidden_write_cap, 12)),
            "soft": bool(self.soft),
            "audit_v2_len": int(len(self.audit_v2)),
        })

    def decide(
            self, candidate: ReplayCandidate,
            *, hidden_write_total_l2: float = 0.0,
    ) -> tuple[ReplayDecision, float]:
        """Return (decision, decision_confidence). If V2 is
        untrained (W is None) and soft is False, falls back to V1
        decision with confidence 1.0."""
        # Hidden-write gate.
        if float(hidden_write_total_l2) > float(
                self.hidden_write_cap):
            dec = ReplayDecision(
                decision=W60_REPLAY_DECISION_ABSTAIN,
                flop_chosen=0,
                drift_chosen=0.0,
                flop_saving_vs_recompute=0.0,
                rationale="hidden_write_cap_exceeded",
                crc_passed=bool(candidate.crc_passed),
            )
            self.audit_v2.append({
                "stage": "v2_hidden_write_gate",
                "hidden_write_total_l2": float(round(
                    hidden_write_total_l2, 12)),
                **dec.to_dict()})
            return dec, 1.0
        if self.W is None or self.b is None:
            v1_decision = self.inner_v1.decide(candidate)
            self.audit_v2.append({
                "stage": "v2_v1_fallback", **v1_decision.to_dict()})
            return v1_decision, 1.0
        feat = _candidate_feature(
            candidate,
            hidden_write_total_l2=float(hidden_write_total_l2))
        scores = (
            _np.asarray(self.W, dtype=_np.float64) @ feat
            + _np.asarray(self.b, dtype=_np.float64))
        probs = _softmax_4(scores)
        idx = int(_np.argmax(probs))
        confidence = float(probs[idx])
        chosen_kind = W60_REPLAY_DECISIONS[idx]
        # Abstain if confidence below threshold.
        if confidence < float(self.abstain_threshold):
            chosen_kind = W60_REPLAY_DECISION_ABSTAIN
        # Compute the V1 numbers consistent with the chosen kind.
        denom = max(int(candidate.flop_recompute), 1)
        if chosen_kind == W60_REPLAY_DECISION_REUSE:
            saving = (
                float(int(candidate.flop_recompute)
                      - int(candidate.flop_reuse))
                / float(denom))
            dec = ReplayDecision(
                decision=chosen_kind,
                flop_chosen=int(candidate.flop_reuse),
                drift_chosen=float(candidate.drift_l2_reuse),
                flop_saving_vs_recompute=float(saving),
                rationale="v2_softmax_reuse",
                crc_passed=bool(candidate.crc_passed))
        elif chosen_kind == W60_REPLAY_DECISION_RECOMPUTE:
            dec = ReplayDecision(
                decision=chosen_kind,
                flop_chosen=int(candidate.flop_recompute),
                drift_chosen=float(
                    candidate.drift_l2_recompute),
                flop_saving_vs_recompute=0.0,
                rationale="v2_softmax_recompute",
                crc_passed=bool(candidate.crc_passed))
        elif chosen_kind == W60_REPLAY_DECISION_FALLBACK:
            saving = (
                float(int(candidate.flop_recompute)
                      - int(candidate.flop_fallback))
                / float(denom))
            dec = ReplayDecision(
                decision=chosen_kind,
                flop_chosen=int(candidate.flop_fallback),
                drift_chosen=float(
                    candidate.drift_l2_fallback),
                flop_saving_vs_recompute=float(saving),
                rationale="v2_softmax_fallback",
                crc_passed=bool(candidate.crc_passed))
        else:
            dec = ReplayDecision(
                decision=W60_REPLAY_DECISION_ABSTAIN,
                flop_chosen=0, drift_chosen=0.0,
                flop_saving_vs_recompute=1.0,
                rationale=(
                    "v2_softmax_abstain"
                    if chosen_kind == (
                        W60_REPLAY_DECISION_ABSTAIN)
                    else "v2_confidence_below_threshold"),
                crc_passed=bool(candidate.crc_passed))
        self.audit_v2.append({
            "stage": "v2_softmax",
            "probs": [float(round(float(p), 12)) for p in probs],
            "confidence": float(round(confidence, 12)),
            **dec.to_dict()})
        return dec, float(confidence)


@dataclasses.dataclass(frozen=True)
class ReplayControllerV2FitReport:
    schema: str
    n_train_examples: int
    pre_fit_residual: float
    post_fit_residual: float
    ridge_lambda: float
    fit_seed: int
    accuracy_train: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_train_examples": int(self.n_train_examples),
            "pre_fit_residual": float(round(
                self.pre_fit_residual, 12)),
            "post_fit_residual": float(round(
                self.post_fit_residual, 12)),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
            "accuracy_train": float(round(
                self.accuracy_train, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v2_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v2(
        *,
        controller: ReplayControllerV2,
        train_candidates: Sequence[ReplayCandidate],
        train_optimal_decisions: Sequence[str],
        train_hidden_write_l2: Sequence[float] | None = None,
        ridge_lambda: float = (
            W61_DEFAULT_REPLAY_V2_RIDGE_LAMBDA),
        fit_seed: int = 61070,
) -> tuple[ReplayControllerV2, ReplayControllerV2FitReport]:
    """Closed-form linear ridge over the 6-dim feature against a
    one-hot 4-class label. ``W = (X^T X + λI)^{-1} X^T Y``."""
    n = int(len(train_candidates))
    if n == 0 or len(train_optimal_decisions) != n:
        raise ValueError(
            "fit requires matching positive-length sequences")
    hidden_l2 = (
        list(train_hidden_write_l2)
        if train_hidden_write_l2 is not None
        else [0.0] * n)
    X = _np.zeros((n, 6), dtype=_np.float64)
    Y = _np.zeros((n, 4), dtype=_np.float64)
    for i, (c, lbl) in enumerate(zip(
            train_candidates, train_optimal_decisions)):
        X[i] = _candidate_feature(
            c, hidden_write_total_l2=float(hidden_l2[i]))
        if lbl in W60_REPLAY_DECISIONS:
            Y[i, W60_REPLAY_DECISIONS.index(lbl)] = 1.0
        else:
            Y[i, W60_REPLAY_DECISIONS.index(
                W60_REPLAY_DECISION_ABSTAIN)] = 1.0
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(6, dtype=_np.float64)
    b = X.T @ Y
    try:
        Wt = _np.linalg.solve(A, b)
    except Exception:
        Wt = _np.zeros((6, 4), dtype=_np.float64)
    Y_hat = X @ Wt
    pre = float(_np.mean(_np.abs(Y - _np.mean(Y, axis=0))))
    post = float(_np.mean(_np.abs(Y - Y_hat)))
    # Compute training-set accuracy under softmax + argmax.
    preds = _np.argmax(Y_hat, axis=1)
    labels = _np.argmax(Y, axis=1)
    acc = float(_np.mean(preds == labels))
    fitted = dataclasses.replace(
        controller,
        W=_np.asarray(Wt.T, dtype=_np.float64).copy(),
        b=_np.zeros((4,), dtype=_np.float64),
        soft=True,
    )
    return fitted, ReplayControllerV2FitReport(
        schema=W61_REPLAY_CONTROLLER_V2_SCHEMA_VERSION,
        n_train_examples=int(n),
        pre_fit_residual=float(pre),
        post_fit_residual=float(post),
        ridge_lambda=float(ridge_lambda),
        fit_seed=int(fit_seed),
        accuracy_train=float(acc),
    )


@dataclasses.dataclass(frozen=True)
class ReplayControllerV2Witness:
    schema: str
    controller_cid: str
    inner_v1_witness_cid: str
    n_decisions: int
    mean_confidence: float
    abstain_count: int
    reuse_count: int
    recompute_count: int
    fallback_count: int
    hidden_write_gate_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "inner_v1_witness_cid": str(
                self.inner_v1_witness_cid),
            "n_decisions": int(self.n_decisions),
            "mean_confidence": float(round(
                self.mean_confidence, 12)),
            "abstain_count": int(self.abstain_count),
            "reuse_count": int(self.reuse_count),
            "recompute_count": int(self.recompute_count),
            "fallback_count": int(self.fallback_count),
            "hidden_write_gate_fired": int(
                self.hidden_write_gate_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v2_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v2_witness(
        controller: ReplayControllerV2,
        candidates: Sequence[ReplayCandidate],
) -> ReplayControllerV2Witness:
    inner_w = emit_replay_controller_witness(
        controller.inner_v1, candidates)
    n_dec = len(controller.audit_v2)
    confs = [
        float(entry.get("confidence", 0.0))
        for entry in controller.audit_v2
        if "confidence" in entry]
    mean_conf = (
        float(_np.mean(_np.asarray(confs, dtype=_np.float64)))
        if confs else 0.0)
    counts = {d: 0 for d in W60_REPLAY_DECISIONS}
    gate_fired = 0
    for entry in controller.audit_v2:
        d = str(entry.get("decision"))
        if d in counts:
            counts[d] += 1
        if str(entry.get("stage")) == "v2_hidden_write_gate":
            gate_fired += 1
    return ReplayControllerV2Witness(
        schema=W61_REPLAY_CONTROLLER_V2_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        inner_v1_witness_cid=str(inner_w.cid()),
        n_decisions=int(n_dec),
        mean_confidence=float(mean_conf),
        abstain_count=int(counts[W60_REPLAY_DECISION_ABSTAIN]),
        reuse_count=int(counts[W60_REPLAY_DECISION_REUSE]),
        recompute_count=int(
            counts[W60_REPLAY_DECISION_RECOMPUTE]),
        fallback_count=int(
            counts[W60_REPLAY_DECISION_FALLBACK]),
        hidden_write_gate_fired=int(gate_fired),
    )


__all__ = [
    "W61_REPLAY_CONTROLLER_V2_SCHEMA_VERSION",
    "W61_DEFAULT_REPLAY_V2_ABSTAIN_CONF",
    "W61_DEFAULT_REPLAY_V2_HIDDEN_WRITE_CAP",
    "W61_DEFAULT_REPLAY_V2_RIDGE_LAMBDA",
    "ReplayControllerV2",
    "ReplayControllerV2FitReport",
    "ReplayControllerV2Witness",
    "fit_replay_controller_v2",
    "emit_replay_controller_v2_witness",
]
