"""W63 M7 — Replay Controller V4.

Strictly extends W62's ``coordpy.replay_controller_v3``. V3 fit
**R = 4** per-regime 6×4 ridge heads with a 5-dim regime gate +
a 5×3 hidden-vs-KV classifier. V4 makes the decision *richer*:

* **Six regimes** — V4 introduces two new regimes on top of V3's
  four:
    - ``hidden_wins_regime`` (positive hidden_vs_kv_contention),
    - ``cache_corruption_recovered`` (CRC failed but
      cache_controller_v6.retrieval_repair > threshold).
* **Per-regime 8×4 ridge head** — V4 adds two new candidate
  features on top of V3's six: ``hidden_vs_kv_contention`` and
  ``prefix_reuse_trust``.
* **Three-way bridge classifier** —
  ``fit_three_way_bridge_classifier`` is a closed-form ridge over
  7 regime features against a *three-way* label
  (kv_wins / hidden_wins / prefix_wins). This is the H184 bar.
* **Replay-determinism bonus** — at decision time, V4 adds the
  V8 substrate's ``replay_determinism_channel`` mean over the
  candidate's slots to its REUSE score (encouraging reuse where
  the substrate cache is deterministic).

Honest scope
------------

* All ridge fits are closed-form linear. ``W63-L-V4-REPLAY-NO-
  AUTOGRAD-CAP`` documents.
* The two new regimes are still based on deterministic feature
  thresholds, not measured behaviour.
* The three-way bridge classifier is fit on *synthetic*
  supervision; it does NOT prove that prefix bridges beat
  hidden bridges in general.
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
        "coordpy.replay_controller_v4 requires numpy"
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
    _candidate_feature, _softmax_4,
)
from .replay_controller_v3 import (
    ReplayControllerV3,
    W62_HIDDEN_VS_KV_LABELS,
    W62_REPLAY_REGIMES,
    W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT,
    W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY,
    W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION,
    W62_REPLAY_REGIME_TRANSCRIPT_ONLY,
)
from .tiny_substrate_v3 import _sha256_hex


W63_REPLAY_CONTROLLER_V4_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v4.v1")

W63_REPLAY_REGIME_HIDDEN_WINS: str = "hidden_wins_regime"
W63_REPLAY_REGIME_CACHE_CORRUPTION_RECOVERED: str = (
    "cache_corruption_recovered")

W63_REPLAY_REGIMES_V4: tuple[str, ...] = (
    *W62_REPLAY_REGIMES,
    W63_REPLAY_REGIME_HIDDEN_WINS,
    W63_REPLAY_REGIME_CACHE_CORRUPTION_RECOVERED,
)

W63_BRIDGE_LABELS: tuple[str, ...] = (
    "kv_wins", "hidden_wins", "prefix_wins")

W63_DEFAULT_REPLAY_V4_RIDGE_LAMBDA: float = 0.10
W63_DEFAULT_REPLAY_V4_DETERMINISM_BONUS: float = 0.05


def _candidate_feature_v4(
        c: ReplayCandidate,
        *, hidden_write_total_l2: float = 0.0,
        hidden_vs_kv_contention: float = 0.0,
        prefix_reuse_trust: float = 0.0,
) -> "_np.ndarray":
    """8-dim feature: V2's six features + the two new V4 features."""
    base = _candidate_feature(
        c, hidden_write_total_l2=float(hidden_write_total_l2))
    return _np.concatenate([
        base,
        _np.array([
            float(hidden_vs_kv_contention),
            float(prefix_reuse_trust)],
            dtype=_np.float64)])


def _regime_feature_v4(
        c: ReplayCandidate,
        *, hidden_write_total_l2: float = 0.0,
        hidden_vs_kv_contention: float = 0.0,
        prefix_reuse_trust: float = 0.0,
) -> "_np.ndarray":
    """7-dim regime feature: V3's five + the two new V4 features."""
    return _np.array([
        float(c.drift_l2_reuse),
        1.0 if bool(c.crc_passed) else 0.0,
        1.0 if bool(c.transcript_available) else 0.0,
        float(int(c.n_corruption_flags)),
        float(hidden_write_total_l2),
        float(hidden_vs_kv_contention),
        float(prefix_reuse_trust),
    ], dtype=_np.float64)


@dataclasses.dataclass
class ReplayControllerV4:
    inner_v3: ReplayControllerV3
    # Per-regime decision heads: dict of regime → (4, 8) head.
    per_regime_heads_v4: dict[str, "_np.ndarray"] = (
        dataclasses.field(default_factory=dict))
    # Regime gate: (R, 7) matrix of regime centroids.
    regime_gate_v4: "_np.ndarray | None" = None
    # Three-way bridge classifier: (3, 7) ridge head.
    three_way_bridge_classifier: "_np.ndarray | None" = None
    determinism_bonus: float = (
        W63_DEFAULT_REPLAY_V4_DETERMINISM_BONUS)
    audit_v4: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            inner_v3: ReplayControllerV3 | None = None,
            determinism_bonus: float = (
                W63_DEFAULT_REPLAY_V4_DETERMINISM_BONUS),
    ) -> "ReplayControllerV4":
        if inner_v3 is None:
            inner_v3 = ReplayControllerV3.init()
        return cls(
            inner_v3=inner_v3,
            per_regime_heads_v4={},
            regime_gate_v4=None,
            three_way_bridge_classifier=None,
            determinism_bonus=float(determinism_bonus),
            audit_v4=[])

    def cid(self) -> str:
        regime_cid = "untrained"
        if self.per_regime_heads_v4:
            payload = sorted(
                (k, hashlib.sha256(
                    v.tobytes()).hexdigest())
                for k, v in self.per_regime_heads_v4.items())
            regime_cid = _sha256_hex(payload)
        return _sha256_hex({
            "schema": W63_REPLAY_CONTROLLER_V4_SCHEMA_VERSION,
            "kind": "replay_controller_v4",
            "inner_v3_cid": str(self.inner_v3.cid()),
            "per_regime_heads_v4_cid": regime_cid,
            "regime_gate_v4_cid": (
                hashlib.sha256(
                    self.regime_gate_v4.tobytes()
                    ).hexdigest()
                if self.regime_gate_v4 is not None
                else "untrained"),
            "three_way_bridge_classifier_cid": (
                hashlib.sha256(
                    self.three_way_bridge_classifier.tobytes()
                    ).hexdigest()
                if self.three_way_bridge_classifier is not None
                else "untrained"),
            "determinism_bonus": float(round(
                self.determinism_bonus, 12)),
            "audit_v4_len": int(len(self.audit_v4)),
        })

    def classify_regime_v4(
            self, c: ReplayCandidate,
            *, hidden_write_total_l2: float = 0.0,
            hidden_vs_kv_contention: float = 0.0,
            prefix_reuse_trust: float = 0.0,
    ) -> str:
        """Nearest-centroid classifier in 7-dim feature space."""
        if self.regime_gate_v4 is None:
            return W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT
        feat = _regime_feature_v4(
            c,
            hidden_write_total_l2=hidden_write_total_l2,
            hidden_vs_kv_contention=hidden_vs_kv_contention,
            prefix_reuse_trust=prefix_reuse_trust)
        centroids = _np.asarray(
            self.regime_gate_v4, dtype=_np.float64)
        diffs = centroids - feat.reshape(1, -1)
        dists = _np.sum(diffs * diffs, axis=-1)
        idx = int(_np.argmin(dists))
        if 0 <= idx < int(len(W63_REPLAY_REGIMES_V4)):
            return W63_REPLAY_REGIMES_V4[idx]
        return W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT

    def decide_v4(
            self, c: ReplayCandidate,
            *, hidden_write_total_l2: float = 0.0,
            hidden_vs_kv_contention: float = 0.0,
            prefix_reuse_trust: float = 0.0,
            replay_determinism_mean: float = 0.0,
    ) -> tuple[ReplayDecision, float, float, str]:
        """Return (decision, decision_confidence,
        replay_dominance, regime_used)."""
        regime = self.classify_regime_v4(
            c,
            hidden_write_total_l2=hidden_write_total_l2,
            hidden_vs_kv_contention=hidden_vs_kv_contention,
            prefix_reuse_trust=prefix_reuse_trust)
        head = self.per_regime_heads_v4.get(regime)
        if head is None:
            # Fall back to V3.
            v3_dec, v3_conf, v3_dom, _ = self.inner_v3.decide(
                c,
                hidden_write_total_l2=hidden_write_total_l2)
            self.audit_v4.append({
                "stage": "v4_v3_fallback",
                "regime": str(regime),
                **v3_dec.to_dict()})
            return (
                v3_dec, float(v3_conf), float(v3_dom),
                str(regime))
        feat = _candidate_feature_v4(
            c,
            hidden_write_total_l2=hidden_write_total_l2,
            hidden_vs_kv_contention=hidden_vs_kv_contention,
            prefix_reuse_trust=prefix_reuse_trust)
        scores = _np.asarray(
            head, dtype=_np.float64) @ feat
        # Determinism bonus: add to the REUSE score.
        reuse_idx = W60_REPLAY_DECISIONS.index(
            W60_REPLAY_DECISION_REUSE)
        scores[reuse_idx] = float(scores[reuse_idx]) + float(
            self.determinism_bonus
            * float(replay_determinism_mean))
        probs = _softmax_4(scores)
        idx = int(_np.argmax(probs))
        confidence = float(probs[idx])
        chosen_kind = W60_REPLAY_DECISIONS[idx]
        sorted_probs = _np.sort(probs)[::-1]
        replay_dominance = float(
            sorted_probs[0] - sorted_probs[1])
        if confidence < float(
                self.inner_v3.inner_v2.abstain_threshold):
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
                rationale=f"v4_regime_{regime}_reuse",
                crc_passed=bool(c.crc_passed))
        elif chosen_kind == W60_REPLAY_DECISION_RECOMPUTE:
            dec = ReplayDecision(
                decision=chosen_kind,
                flop_chosen=int(c.flop_recompute),
                drift_chosen=float(c.drift_l2_recompute),
                flop_saving_vs_recompute=0.0,
                rationale=f"v4_regime_{regime}_recompute",
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
                rationale=f"v4_regime_{regime}_fallback",
                crc_passed=bool(c.crc_passed))
        else:
            dec = ReplayDecision(
                decision=W60_REPLAY_DECISION_ABSTAIN,
                flop_chosen=0, drift_chosen=0.0,
                flop_saving_vs_recompute=1.0,
                rationale=f"v4_regime_{regime}_abstain",
                crc_passed=bool(c.crc_passed))
        self.audit_v4.append({
            "stage": "v4_per_regime",
            "regime": str(regime),
            "probs": [float(round(float(p), 12)) for p in probs],
            "confidence": float(round(confidence, 12)),
            "replay_dominance": float(round(
                replay_dominance, 12)),
            "replay_determinism_mean": float(round(
                replay_determinism_mean, 12)),
            **dec.to_dict()})
        return (
            dec, float(confidence),
            float(replay_dominance), str(regime))


@dataclasses.dataclass(frozen=True)
class ReplayControllerV4FitReport:
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
            "kind": "replay_controller_v4_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v4_per_regime(
        *, controller: ReplayControllerV4,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[
            str, Sequence[str]],
        train_hidden_write_l2_per_regime: dict[
            str, Sequence[float]] | None = None,
        train_hidden_vs_kv_per_regime: dict[
            str, Sequence[float]] | None = None,
        train_prefix_reuse_per_regime: dict[
            str, Sequence[float]] | None = None,
        ridge_lambda: float = W63_DEFAULT_REPLAY_V4_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV4, ReplayControllerV4FitReport]:
    """Fit one closed-form ridge head per regime + a regime gate.
    """
    regimes = list(train_candidates_per_regime.keys())
    R = int(len(regimes))
    if R == 0:
        raise ValueError("must provide ≥ 1 regime")
    n_per_regime: list[int] = []
    per_regime_heads: dict[str, "_np.ndarray"] = {}
    per_pre: list[float] = []
    per_post: list[float] = []
    centroids = _np.zeros((R, 7), dtype=_np.float64)
    for r_idx, regime in enumerate(regimes):
        cands = list(train_candidates_per_regime[regime])
        labels = list(train_decisions_per_regime[regime])
        hl2 = (
            list(train_hidden_write_l2_per_regime[regime])
            if train_hidden_write_l2_per_regime is not None
            and regime in train_hidden_write_l2_per_regime
            else [0.0] * len(cands))
        hvkv = (
            list(train_hidden_vs_kv_per_regime[regime])
            if train_hidden_vs_kv_per_regime is not None
            and regime in train_hidden_vs_kv_per_regime
            else [0.0] * len(cands))
        prt = (
            list(train_prefix_reuse_per_regime[regime])
            if train_prefix_reuse_per_regime is not None
            and regime in train_prefix_reuse_per_regime
            else [0.0] * len(cands))
        n = int(len(cands))
        n_per_regime.append(n)
        if n == 0:
            per_pre.append(0.0)
            per_post.append(0.0)
            continue
        X = _np.zeros((n, 8), dtype=_np.float64)
        Y = _np.zeros((n, 4), dtype=_np.float64)
        regime_feats = _np.zeros((n, 7), dtype=_np.float64)
        for i, (c, lbl) in enumerate(zip(cands, labels)):
            X[i] = _candidate_feature_v4(
                c, hidden_write_total_l2=float(hl2[i]),
                hidden_vs_kv_contention=float(hvkv[i]),
                prefix_reuse_trust=float(prt[i]))
            regime_feats[i] = _regime_feature_v4(
                c, hidden_write_total_l2=float(hl2[i]),
                hidden_vs_kv_contention=float(hvkv[i]),
                prefix_reuse_trust=float(prt[i]))
            if lbl in W60_REPLAY_DECISIONS:
                Y[i, W60_REPLAY_DECISIONS.index(lbl)] = 1.0
            else:
                Y[i, W60_REPLAY_DECISIONS.index(
                    W60_REPLAY_DECISION_ABSTAIN)] = 1.0
        lam = max(float(ridge_lambda), 1e-9)
        A = X.T @ X + lam * _np.eye(8, dtype=_np.float64)
        b = X.T @ Y
        try:
            Wt = _np.linalg.solve(A, b)
        except Exception:
            Wt = _np.zeros((8, 4), dtype=_np.float64)
        Y_hat = X @ Wt
        per_pre.append(float(_np.mean(_np.abs(Y))))
        per_post.append(float(_np.mean(_np.abs(Y - Y_hat))))
        per_regime_heads[regime] = _np.asarray(
            Wt.T, dtype=_np.float64).copy()
        centroids[r_idx] = regime_feats.mean(axis=0)
    fitted = dataclasses.replace(
        controller,
        per_regime_heads_v4=per_regime_heads,
        regime_gate_v4=centroids.copy(),
    )
    converged = all(
        po <= pr + 1e-9 for pr, po in zip(per_pre, per_post))
    report = ReplayControllerV4FitReport(
        schema=W63_REPLAY_CONTROLLER_V4_SCHEMA_VERSION,
        n_regimes=int(R),
        n_train_per_regime=tuple(n_per_regime),
        per_regime_pre_residual=tuple(per_pre),
        per_regime_post_residual=tuple(per_post),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_three_way_bridge_classifier(
        *, controller: ReplayControllerV4,
        train_features: Sequence[Sequence[float]],
        train_labels: Sequence[str],
        ridge_lambda: float = W63_DEFAULT_REPLAY_V4_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV4, dict[str, Any]]:
    """Fit a 3-class closed-form ridge classifier over 7 regime
    features. Labels must come from ``W63_BRIDGE_LABELS``."""
    n = int(len(train_features))
    if n == 0 or len(train_labels) != n:
        raise ValueError(
            "fit requires non-empty matching inputs")
    X = _np.asarray(train_features, dtype=_np.float64)
    if X.shape[1] != 7:
        raise ValueError("regime features must be 7-dim")
    Y = _np.zeros((n, 3), dtype=_np.float64)
    for i, lbl in enumerate(train_labels):
        if lbl in W63_BRIDGE_LABELS:
            Y[i, W63_BRIDGE_LABELS.index(lbl)] = 1.0
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(7, dtype=_np.float64)
    b = X.T @ Y
    try:
        Wt = _np.linalg.solve(A, b)
    except Exception:
        Wt = _np.zeros((7, 3), dtype=_np.float64)
    Y_hat = X @ Wt
    preds = _np.argmax(Y_hat, axis=1)
    labels = _np.argmax(Y, axis=1)
    acc = float(_np.mean(preds == labels))
    fitted = dataclasses.replace(
        controller,
        three_way_bridge_classifier=_np.asarray(
            Wt.T, dtype=_np.float64).copy())
    audit = {
        "schema": W63_REPLAY_CONTROLLER_V4_SCHEMA_VERSION,
        "kind": "three_way_bridge_classifier_fit",
        "n_train": int(n),
        "accuracy_train": float(acc),
    }
    return fitted, audit


def predict_three_way_bridge(
        controller: ReplayControllerV4,
        feature: Sequence[float],
) -> str:
    if controller.three_way_bridge_classifier is None:
        return "kv_wins"
    f = _np.asarray(feature, dtype=_np.float64).reshape(-1)
    if f.size < 7:
        f = _np.concatenate([
            f, _np.zeros(7 - f.size, dtype=_np.float64)])
    elif f.size > 7:
        f = f[:7]
    scores = (
        _np.asarray(
            controller.three_way_bridge_classifier,
            dtype=_np.float64) @ f)
    idx = int(_np.argmax(scores))
    if 0 <= idx < int(len(W63_BRIDGE_LABELS)):
        return W63_BRIDGE_LABELS[idx]
    return "kv_wins"


@dataclasses.dataclass(frozen=True)
class ReplayControllerV4Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    mean_confidence: float
    mean_replay_dominance: float
    regime_count: dict[str, int]
    three_way_bridge_classifier_trained: bool

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
            "three_way_bridge_classifier_trained": bool(
                self.three_way_bridge_classifier_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v4_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v4_witness(
        controller: ReplayControllerV4,
) -> ReplayControllerV4Witness:
    confs = [
        float(entry.get("confidence", 0.0))
        for entry in controller.audit_v4
        if "confidence" in entry]
    doms = [
        float(entry.get("replay_dominance", 0.0))
        for entry in controller.audit_v4
        if "replay_dominance" in entry]
    regime_count: dict[str, int] = {}
    for entry in controller.audit_v4:
        r = str(entry.get("regime", ""))
        if r:
            regime_count[r] = int(regime_count.get(r, 0) + 1)
    return ReplayControllerV4Witness(
        schema=W63_REPLAY_CONTROLLER_V4_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v4)),
        mean_confidence=(
            float(_np.mean(_np.asarray(
                confs, dtype=_np.float64)))
            if confs else 0.0),
        mean_replay_dominance=(
            float(_np.mean(_np.asarray(
                doms, dtype=_np.float64)))
            if doms else 0.0),
        regime_count=regime_count,
        three_way_bridge_classifier_trained=bool(
            controller.three_way_bridge_classifier is not None),
    )


__all__ = [
    "W63_REPLAY_CONTROLLER_V4_SCHEMA_VERSION",
    "W63_REPLAY_REGIME_HIDDEN_WINS",
    "W63_REPLAY_REGIME_CACHE_CORRUPTION_RECOVERED",
    "W63_REPLAY_REGIMES_V4",
    "W63_BRIDGE_LABELS",
    "W63_DEFAULT_REPLAY_V4_RIDGE_LAMBDA",
    "W63_DEFAULT_REPLAY_V4_DETERMINISM_BONUS",
    "ReplayControllerV4",
    "ReplayControllerV4FitReport",
    "fit_replay_controller_v4_per_regime",
    "fit_three_way_bridge_classifier",
    "predict_three_way_bridge",
    "ReplayControllerV4Witness",
    "emit_replay_controller_v4_witness",
]
