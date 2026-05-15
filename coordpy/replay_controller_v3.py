"""W62 M7 — Replay Controller V3.

Strictly extends W61's ``coordpy.replay_controller_v2``. V2 fit a
single 6×4 linear ridge head against a 4-class one-hot label. V3
makes the decision *regime-aware*:

* **Per-regime ridge head** — V3 fits **R** separate 6×4 ridge
  heads (one per regime) by closed-form ridge over the union of
  training candidates with regime labels. At decision time, V3
  first classifies the candidate into a regime via a 5-dim regime
  gate (closed-form 5×R ridge over regime features), then applies
  the per-regime head.
* **Replay-dominance scalar** — each decision returns a
  ``replay_dominance`` scalar = chosen head's softmax probability
  minus the second-best head's softmax probability. This is the
  W62 measurable replay-dominance signal that the LHR V14
  retention head reads.
* **Hidden-vs-KV regime classifier** —
  ``fit_hidden_vs_kv_regime_classifier`` is a separate closed-form
  ridge over 5 regime features against a 3-class label
  (hidden_beats_kv / kv_beats_hidden / tie). This is the H165
  bar.

Honest scope
------------

* All ridge fits are closed-form linear. ``W62-L-V3-REPLAY-NO-
  AUTOGRAD-CAP`` documents.
* Replay-dominance is a softmax margin, not a calibrated
  probability.
* The hidden-vs-KV regime classifier is a separate small model;
  it does NOT replace the deterministic comparison in
  ``compare_hidden_vs_kv_v7``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v3 requires numpy"
        ) from exc

from .replay_controller import (
    ReplayCandidate, ReplayDecision,
    W60_REPLAY_DECISIONS,
    W60_REPLAY_DECISION_ABSTAIN,
    W60_REPLAY_DECISION_RECOMPUTE,
    W60_REPLAY_DECISION_REUSE,
    W60_REPLAY_DECISION_FALLBACK,
)
from .replay_controller_v2 import (
    ReplayControllerV2,
    _candidate_feature,
    _softmax_4,
)
from .tiny_substrate_v3 import _sha256_hex


W62_REPLAY_CONTROLLER_V3_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v3.v1")

W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION: str = (
    "synthetic_corruption")
W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: str = (
    "crc_passed_low_drift")
W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: str = (
    "hidden_write_heavy")
W62_REPLAY_REGIME_TRANSCRIPT_ONLY: str = "transcript_only"

W62_REPLAY_REGIMES: tuple[str, ...] = (
    W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION,
    W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT,
    W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY,
    W62_REPLAY_REGIME_TRANSCRIPT_ONLY,
)

W62_HIDDEN_VS_KV_LABELS: tuple[str, ...] = (
    "hidden_beats_kv", "kv_beats_hidden", "tie")

W62_DEFAULT_REPLAY_V3_RIDGE_LAMBDA: float = 0.10


def _regime_feature(
        c: ReplayCandidate,
        hidden_write_total_l2: float = 0.0,
) -> "_np.ndarray":
    """5-dim regime feature: [drift_reuse, crc_flag, transcript_flag,
    n_corruption_flags, hidden_write_total_l2]."""
    return _np.array([
        float(c.drift_l2_reuse),
        1.0 if bool(c.crc_passed) else 0.0,
        1.0 if bool(c.transcript_available) else 0.0,
        float(int(c.n_corruption_flags)),
        float(hidden_write_total_l2),
    ], dtype=_np.float64)


@dataclasses.dataclass
class ReplayControllerV3:
    inner_v2: ReplayControllerV2
    # Per-regime decision heads: dict of regime → (4, 6) head matrix.
    per_regime_heads: dict[str, "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    # Regime gate: (R, 5) matrix mapping regime features → regime
    # logits.
    regime_gate: "_np.ndarray | None" = None
    # Hidden-vs-KV classifier: (3, 5) ridge head.
    hidden_vs_kv_classifier: "_np.ndarray | None" = None
    audit_v3: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, inner_v2: ReplayControllerV2 | None = None,
    ) -> "ReplayControllerV3":
        if inner_v2 is None:
            inner_v2 = ReplayControllerV2.init()
        return cls(
            inner_v2=inner_v2, per_regime_heads={},
            regime_gate=None,
            hidden_vs_kv_classifier=None,
            audit_v3=[])

    def cid(self) -> str:
        regime_cid = "untrained"
        if self.per_regime_heads:
            payload = sorted(
                (k, hashlib.sha256(
                    v.tobytes()).hexdigest())
                for k, v in self.per_regime_heads.items())
            regime_cid = _sha256_hex(payload)
        return _sha256_hex({
            "schema": W62_REPLAY_CONTROLLER_V3_SCHEMA_VERSION,
            "kind": "replay_controller_v3",
            "inner_v2_cid": str(self.inner_v2.cid()),
            "per_regime_heads_cid": regime_cid,
            "regime_gate_cid": (
                hashlib.sha256(
                    self.regime_gate.tobytes()).hexdigest()
                if self.regime_gate is not None
                else "untrained"),
            "hidden_vs_kv_classifier_cid": (
                hashlib.sha256(
                    self.hidden_vs_kv_classifier.tobytes()
                    ).hexdigest()
                if self.hidden_vs_kv_classifier is not None
                else "untrained"),
            "audit_v3_len": int(len(self.audit_v3)),
        })

    def classify_regime(
            self, c: ReplayCandidate,
            *, hidden_write_total_l2: float = 0.0,
    ) -> str:
        """Nearest-centroid classifier in 5-dim regime feature
        space. ``regime_gate`` is the (R, 5) matrix of centroids."""
        if self.regime_gate is None:
            return W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT
        feat = _regime_feature(
            c, hidden_write_total_l2=hidden_write_total_l2)
        centroids = _np.asarray(
            self.regime_gate, dtype=_np.float64)
        # Squared L2 distance to each centroid.
        diffs = centroids - feat.reshape(1, -1)
        dists = _np.sum(diffs * diffs, axis=-1)
        idx = int(_np.argmin(dists))
        if 0 <= idx < int(len(W62_REPLAY_REGIMES)):
            return W62_REPLAY_REGIMES[idx]
        return W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT

    def decide(
            self, c: ReplayCandidate,
            *, hidden_write_total_l2: float = 0.0,
    ) -> tuple[ReplayDecision, float, float, str]:
        """Return (decision, decision_confidence,
        replay_dominance, regime_used)."""
        regime = self.classify_regime(
            c, hidden_write_total_l2=hidden_write_total_l2)
        head = self.per_regime_heads.get(regime)
        if head is None or self.inner_v2.W is None:
            # Fall back to V2's softmax decision.
            v2_dec, v2_conf = self.inner_v2.decide(
                c, hidden_write_total_l2=hidden_write_total_l2)
            self.audit_v3.append({
                "stage": "v3_v2_fallback",
                "regime": str(regime),
                **v2_dec.to_dict()})
            return (
                v2_dec, float(v2_conf),
                float(v2_conf - 0.25), str(regime))
        feat = _candidate_feature(
            c, hidden_write_total_l2=hidden_write_total_l2)
        scores = _np.asarray(
            head, dtype=_np.float64) @ feat
        probs = _softmax_4(scores)
        idx = int(_np.argmax(probs))
        confidence = float(probs[idx])
        chosen_kind = W60_REPLAY_DECISIONS[idx]
        # Compute replay-dominance: top - second-best.
        sorted_probs = _np.sort(probs)[::-1]
        replay_dominance = float(
            sorted_probs[0] - sorted_probs[1])
        # Apply abstain threshold (inherited from V2 inner).
        if confidence < float(self.inner_v2.abstain_threshold):
            chosen_kind = W60_REPLAY_DECISION_ABSTAIN
        denom = max(int(c.flop_recompute), 1)
        if chosen_kind == W60_REPLAY_DECISION_REUSE:
            saving = (
                float(int(c.flop_recompute)
                      - int(c.flop_reuse))
                / float(denom))
            dec = ReplayDecision(
                decision=chosen_kind,
                flop_chosen=int(c.flop_reuse),
                drift_chosen=float(c.drift_l2_reuse),
                flop_saving_vs_recompute=float(saving),
                rationale=f"v3_regime_{regime}_reuse",
                crc_passed=bool(c.crc_passed))
        elif chosen_kind == W60_REPLAY_DECISION_RECOMPUTE:
            dec = ReplayDecision(
                decision=chosen_kind,
                flop_chosen=int(c.flop_recompute),
                drift_chosen=float(c.drift_l2_recompute),
                flop_saving_vs_recompute=0.0,
                rationale=f"v3_regime_{regime}_recompute",
                crc_passed=bool(c.crc_passed))
        elif chosen_kind == W60_REPLAY_DECISION_FALLBACK:
            saving = (
                float(int(c.flop_recompute)
                      - int(c.flop_fallback))
                / float(denom))
            dec = ReplayDecision(
                decision=chosen_kind,
                flop_chosen=int(c.flop_fallback),
                drift_chosen=float(c.drift_l2_fallback),
                flop_saving_vs_recompute=float(saving),
                rationale=f"v3_regime_{regime}_fallback",
                crc_passed=bool(c.crc_passed))
        else:
            dec = ReplayDecision(
                decision=W60_REPLAY_DECISION_ABSTAIN,
                flop_chosen=0, drift_chosen=0.0,
                flop_saving_vs_recompute=1.0,
                rationale=f"v3_regime_{regime}_abstain",
                crc_passed=bool(c.crc_passed))
        self.audit_v3.append({
            "stage": "v3_per_regime",
            "regime": str(regime),
            "probs": [float(round(float(p), 12)) for p in probs],
            "confidence": float(round(confidence, 12)),
            "replay_dominance": float(round(
                replay_dominance, 12)),
            **dec.to_dict()})
        return (
            dec, float(confidence),
            float(replay_dominance), str(regime))


@dataclasses.dataclass(frozen=True)
class ReplayControllerV3FitReport:
    schema: str
    n_regimes: int
    n_train_per_regime: tuple[int, ...]
    per_regime_pre_residual: tuple[float, ...]
    per_regime_post_residual: tuple[float, ...]
    converged: bool
    ridge_lambda: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_regimes": int(self.n_regimes),
            "n_train_per_regime": list(
                self.n_train_per_regime),
            "per_regime_pre_residual": [
                float(round(float(x), 12))
                for x in self.per_regime_pre_residual],
            "per_regime_post_residual": [
                float(round(float(x), 12))
                for x in self.per_regime_post_residual],
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v3_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v3_per_regime(
        *, controller: ReplayControllerV3,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[
            str, Sequence[str]],
        train_hidden_write_l2_per_regime: dict[
            str, Sequence[float]] | None = None,
        ridge_lambda: float = W62_DEFAULT_REPLAY_V3_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV3, ReplayControllerV3FitReport]:
    """Fit one closed-form ridge head per regime + a regime gate.

    The regime gate is fit by per-regime *centroid* of the
    5-dim regime feature, producing a (R, 5) matrix whose rows
    are the regime centroids.
    """
    regimes = list(train_candidates_per_regime.keys())
    R = int(len(regimes))
    if R == 0:
        raise ValueError("must provide ≥ 1 regime")
    n_per_regime: list[int] = []
    per_regime_heads: dict[str, "_np.ndarray"] = {}
    per_pre: list[float] = []
    per_post: list[float] = []
    centroids = _np.zeros((R, 5), dtype=_np.float64)
    for r_idx, regime in enumerate(regimes):
        cands = list(train_candidates_per_regime[regime])
        labels = list(train_decisions_per_regime[regime])
        hl2 = (
            list(train_hidden_write_l2_per_regime[regime])
            if train_hidden_write_l2_per_regime is not None
            and regime in train_hidden_write_l2_per_regime
            else [0.0] * len(cands))
        n = int(len(cands))
        n_per_regime.append(n)
        if n == 0:
            per_pre.append(0.0)
            per_post.append(0.0)
            continue
        X = _np.zeros((n, 6), dtype=_np.float64)
        Y = _np.zeros((n, 4), dtype=_np.float64)
        regime_feats = _np.zeros((n, 5), dtype=_np.float64)
        for i, (c, lbl) in enumerate(zip(cands, labels)):
            X[i] = _candidate_feature(
                c, hidden_write_total_l2=float(hl2[i]))
            regime_feats[i] = _regime_feature(
                c, hidden_write_total_l2=float(hl2[i]))
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
        per_pre.append(float(_np.mean(_np.abs(Y))))
        per_post.append(float(_np.mean(_np.abs(Y - Y_hat))))
        per_regime_heads[regime] = _np.asarray(
            Wt.T, dtype=_np.float64).copy()
        centroids[r_idx] = regime_feats.mean(axis=0)
    fitted = dataclasses.replace(
        controller,
        per_regime_heads=per_regime_heads,
        regime_gate=centroids.copy(),
    )
    converged = all(
        po <= pr + 1e-9 for pr, po in zip(per_pre, per_post))
    report = ReplayControllerV3FitReport(
        schema=W62_REPLAY_CONTROLLER_V3_SCHEMA_VERSION,
        n_regimes=int(R),
        n_train_per_regime=tuple(n_per_regime),
        per_regime_pre_residual=tuple(per_pre),
        per_regime_post_residual=tuple(per_post),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_hidden_vs_kv_regime_classifier(
        *, controller: ReplayControllerV3,
        train_features: Sequence[Sequence[float]],
        train_labels: Sequence[str],
        ridge_lambda: float = W62_DEFAULT_REPLAY_V3_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV3, dict[str, Any]]:
    """Fit a 3-class closed-form ridge classifier over 5 regime
    features. Labels must come from
    ``W62_HIDDEN_VS_KV_LABELS``."""
    n = int(len(train_features))
    if n == 0 or len(train_labels) != n:
        raise ValueError(
            "fit requires non-empty matching inputs")
    X = _np.asarray(train_features, dtype=_np.float64)
    if X.shape[1] != 5:
        raise ValueError("regime features must be 5-dim")
    Y = _np.zeros((n, 3), dtype=_np.float64)
    for i, lbl in enumerate(train_labels):
        if lbl in W62_HIDDEN_VS_KV_LABELS:
            Y[i, W62_HIDDEN_VS_KV_LABELS.index(lbl)] = 1.0
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(5, dtype=_np.float64)
    b = X.T @ Y
    try:
        Wt = _np.linalg.solve(A, b)
    except Exception:
        Wt = _np.zeros((5, 3), dtype=_np.float64)
    Y_hat = X @ Wt
    preds = _np.argmax(Y_hat, axis=1)
    labels = _np.argmax(Y, axis=1)
    acc = float(_np.mean(preds == labels))
    fitted = dataclasses.replace(
        controller,
        hidden_vs_kv_classifier=_np.asarray(
            Wt.T, dtype=_np.float64).copy())
    audit = {
        "schema": W62_REPLAY_CONTROLLER_V3_SCHEMA_VERSION,
        "kind": "hidden_vs_kv_classifier_fit",
        "n_train": int(n),
        "accuracy_train": float(acc),
    }
    return fitted, audit


def predict_hidden_vs_kv_regime(
        controller: ReplayControllerV3,
        feature: Sequence[float],
) -> str:
    if controller.hidden_vs_kv_classifier is None:
        return "tie"
    f = _np.asarray(feature, dtype=_np.float64).reshape(-1)
    if f.size < 5:
        f = _np.concatenate([
            f, _np.zeros(5 - f.size, dtype=_np.float64)])
    elif f.size > 5:
        f = f[:5]
    scores = (
        _np.asarray(controller.hidden_vs_kv_classifier,
                    dtype=_np.float64) @ f)
    idx = int(_np.argmax(scores))
    if 0 <= idx < int(len(W62_HIDDEN_VS_KV_LABELS)):
        return W62_HIDDEN_VS_KV_LABELS[idx]
    return "tie"


@dataclasses.dataclass(frozen=True)
class ReplayControllerV3Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    mean_confidence: float
    mean_replay_dominance: float
    regime_count: dict[str, int]
    hidden_vs_kv_classifier_trained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "mean_confidence": float(round(
                self.mean_confidence, 12)),
            "mean_replay_dominance": float(round(
                self.mean_replay_dominance, 12)),
            "regime_count": {
                str(k): int(v)
                for k, v in self.regime_count.items()},
            "hidden_vs_kv_classifier_trained": bool(
                self.hidden_vs_kv_classifier_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v3_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v3_witness(
        controller: ReplayControllerV3,
) -> ReplayControllerV3Witness:
    confs = [
        float(entry.get("confidence", 0.0))
        for entry in controller.audit_v3
        if "confidence" in entry]
    doms = [
        float(entry.get("replay_dominance", 0.0))
        for entry in controller.audit_v3
        if "replay_dominance" in entry]
    regime_count: dict[str, int] = {}
    for entry in controller.audit_v3:
        r = str(entry.get("regime", ""))
        if r:
            regime_count[r] = int(regime_count.get(r, 0) + 1)
    return ReplayControllerV3Witness(
        schema=W62_REPLAY_CONTROLLER_V3_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v3)),
        mean_confidence=(
            float(_np.mean(_np.asarray(
                confs, dtype=_np.float64)))
            if confs else 0.0),
        mean_replay_dominance=(
            float(_np.mean(_np.asarray(
                doms, dtype=_np.float64)))
            if doms else 0.0),
        regime_count=regime_count,
        hidden_vs_kv_classifier_trained=bool(
            controller.hidden_vs_kv_classifier is not None),
    )


__all__ = [
    "W62_REPLAY_CONTROLLER_V3_SCHEMA_VERSION",
    "W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION",
    "W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT",
    "W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY",
    "W62_REPLAY_REGIME_TRANSCRIPT_ONLY",
    "W62_REPLAY_REGIMES",
    "W62_HIDDEN_VS_KV_LABELS",
    "W62_DEFAULT_REPLAY_V3_RIDGE_LAMBDA",
    "ReplayControllerV3",
    "ReplayControllerV3FitReport",
    "fit_replay_controller_v3_per_regime",
    "fit_hidden_vs_kv_regime_classifier",
    "predict_hidden_vs_kv_regime",
    "ReplayControllerV3Witness",
    "emit_replay_controller_v3_witness",
]
