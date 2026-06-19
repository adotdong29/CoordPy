"""W64 M7 — Replay Controller V5.

Strictly extends W63's ``coordpy.replay_controller_v4``. V4 fit
**R = 6** per-regime 8×4 ridge heads with a 7-dim regime gate +
a 7×3 three-way bridge classifier. V5 makes the decision *richer*:

* **Seven regimes** — V5 introduces one new regime on top of V4's
  six:
    - ``replay_dominance_primary_regime`` (V9 substrate's
      replay_dominance_witness mean above threshold ⇒ replay
      dominance is the primary signal).
* **Per-regime 10×4 ridge head** — V5 adds two new candidate
  features on top of V4's eight: ``replay_dominance_witness_mean``
  and ``hidden_wins_primary_score``. 10-dim total.
* **Four-way bridge classifier** —
  ``fit_four_way_bridge_classifier`` is a closed-form ridge over
  9 regime features against a *four-way* label
  (kv_wins / hidden_wins / prefix_wins / replay_wins).
* **Replay-dominance-as-primary head** —
  ``fit_replay_dominance_primary_head_v5`` fits a 9-dim ridge
  head whose decisions favour REUSE when the V9 substrate's
  replay_dominance_witness is high.

Honest scope (W64)
------------------

* All ridge fits are closed-form linear. ``W64-L-V5-REPLAY-NO-
  AUTOGRAD-CAP`` documents.
* The new regime is based on a deterministic feature threshold,
  not measured behaviour.
* The four-way bridge classifier is fit on *synthetic*
  supervision; it does NOT prove that replay bridges beat hidden
  / prefix / KV bridges in general.
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
        "coordpy.replay_controller_v5 requires numpy"
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
from .replay_controller_v4 import (
    ReplayControllerV4,
    W63_REPLAY_REGIMES_V4,
    _candidate_feature_v4, _regime_feature_v4,
)
from .tiny_substrate_v3 import _sha256_hex


W64_REPLAY_CONTROLLER_V5_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v5.v1")

W64_REPLAY_REGIME_REPLAY_DOMINANCE_PRIMARY: str = (
    "replay_dominance_primary_regime")

W64_REPLAY_REGIMES_V5: tuple[str, ...] = (
    *W63_REPLAY_REGIMES_V4,
    W64_REPLAY_REGIME_REPLAY_DOMINANCE_PRIMARY,
)

W64_BRIDGE_LABELS_V5: tuple[str, ...] = (
    "kv_wins", "hidden_wins", "prefix_wins", "replay_wins")

W64_DEFAULT_REPLAY_V5_RIDGE_LAMBDA: float = 0.10
W64_DEFAULT_REPLAY_V5_DOMINANCE_PRIMARY_BONUS: float = 0.10


def _candidate_feature_v5(
        c: ReplayCandidate,
        *, hidden_write_total_l2: float = 0.0,
        hidden_vs_kv_contention: float = 0.0,
        prefix_reuse_trust: float = 0.0,
        replay_dominance_witness_mean: float = 0.0,
        hidden_wins_primary_score: float = 0.0,
) -> "_np.ndarray":
    """10-dim feature: V4's eight + the two new V5 features."""
    base = _candidate_feature_v4(
        c,
        hidden_write_total_l2=float(hidden_write_total_l2),
        hidden_vs_kv_contention=float(hidden_vs_kv_contention),
        prefix_reuse_trust=float(prefix_reuse_trust))
    return _np.concatenate([
        base,
        _np.array([
            float(replay_dominance_witness_mean),
            float(hidden_wins_primary_score)],
            dtype=_np.float64)])


def _regime_feature_v5(
        c: ReplayCandidate,
        *, hidden_write_total_l2: float = 0.0,
        hidden_vs_kv_contention: float = 0.0,
        prefix_reuse_trust: float = 0.0,
        replay_dominance_witness_mean: float = 0.0,
        hidden_wins_primary_score: float = 0.0,
) -> "_np.ndarray":
    """9-dim regime feature: V4's seven + the two new V5 features.
    """
    base = _regime_feature_v4(
        c,
        hidden_write_total_l2=float(hidden_write_total_l2),
        hidden_vs_kv_contention=float(hidden_vs_kv_contention),
        prefix_reuse_trust=float(prefix_reuse_trust))
    return _np.concatenate([
        base,
        _np.array([
            float(replay_dominance_witness_mean),
            float(hidden_wins_primary_score)],
            dtype=_np.float64)])


@dataclasses.dataclass
class ReplayControllerV5:
    inner_v4: ReplayControllerV4
    # Per-regime decision heads: dict of regime → (4, 10) head.
    per_regime_heads_v5: dict[str, "_np.ndarray"] = (
        dataclasses.field(default_factory=dict))
    # Regime gate: (R, 9) matrix of regime centroids.
    regime_gate_v5: "_np.ndarray | None" = None
    # Four-way bridge classifier: (4, 9) ridge head.
    four_way_bridge_classifier: "_np.ndarray | None" = None
    # Replay-dominance-primary head: (4, 9) ridge head.
    replay_dominance_primary_head: "_np.ndarray | None" = None
    dominance_primary_bonus: float = (
        W64_DEFAULT_REPLAY_V5_DOMINANCE_PRIMARY_BONUS)
    audit_v5: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            inner_v4: ReplayControllerV4 | None = None,
            dominance_primary_bonus: float = (
                W64_DEFAULT_REPLAY_V5_DOMINANCE_PRIMARY_BONUS),
    ) -> "ReplayControllerV5":
        if inner_v4 is None:
            inner_v4 = ReplayControllerV4.init()
        return cls(
            inner_v4=inner_v4,
            per_regime_heads_v5={},
            regime_gate_v5=None,
            four_way_bridge_classifier=None,
            replay_dominance_primary_head=None,
            dominance_primary_bonus=float(
                dominance_primary_bonus),
            audit_v5=[])

    def cid(self) -> str:
        regime_cid = "untrained"
        if self.per_regime_heads_v5:
            payload = sorted(
                (k, hashlib.sha256(
                    v.tobytes()).hexdigest())
                for k, v in self.per_regime_heads_v5.items())
            regime_cid = _sha256_hex(payload)
        return _sha256_hex({
            "schema": W64_REPLAY_CONTROLLER_V5_SCHEMA_VERSION,
            "kind": "replay_controller_v5",
            "inner_v4_cid": str(self.inner_v4.cid()),
            "per_regime_heads_v5_cid": regime_cid,
            "regime_gate_v5_cid": (
                hashlib.sha256(
                    self.regime_gate_v5.tobytes()
                    ).hexdigest()
                if self.regime_gate_v5 is not None
                else "untrained"),
            "four_way_bridge_classifier_cid": (
                hashlib.sha256(
                    self.four_way_bridge_classifier.tobytes()
                    ).hexdigest()
                if self.four_way_bridge_classifier is not None
                else "untrained"),
            "replay_dominance_primary_head_cid": (
                hashlib.sha256(
                    self.replay_dominance_primary_head.tobytes()
                    ).hexdigest()
                if self.replay_dominance_primary_head is not None
                else "untrained"),
            "dominance_primary_bonus": float(round(
                self.dominance_primary_bonus, 12)),
            "audit_v5_len": int(len(self.audit_v5)),
        })

    def classify_regime_v5(
            self, c: ReplayCandidate,
            *, hidden_write_total_l2: float = 0.0,
            hidden_vs_kv_contention: float = 0.0,
            prefix_reuse_trust: float = 0.0,
            replay_dominance_witness_mean: float = 0.0,
            hidden_wins_primary_score: float = 0.0,
    ) -> str:
        """Nearest-centroid classifier in 9-dim feature space."""
        if self.regime_gate_v5 is None:
            return W64_REPLAY_REGIMES_V5[0]
        feat = _regime_feature_v5(
            c,
            hidden_write_total_l2=hidden_write_total_l2,
            hidden_vs_kv_contention=hidden_vs_kv_contention,
            prefix_reuse_trust=prefix_reuse_trust,
            replay_dominance_witness_mean=(
                replay_dominance_witness_mean),
            hidden_wins_primary_score=(
                hidden_wins_primary_score))
        centroids = _np.asarray(
            self.regime_gate_v5, dtype=_np.float64)
        diffs = centroids - feat.reshape(1, -1)
        dists = _np.sum(diffs * diffs, axis=-1)
        idx = int(_np.argmin(dists))
        if 0 <= idx < int(len(W64_REPLAY_REGIMES_V5)):
            return W64_REPLAY_REGIMES_V5[idx]
        return W64_REPLAY_REGIMES_V5[0]

    def decide_v5(
            self, c: ReplayCandidate,
            *, hidden_write_total_l2: float = 0.0,
            hidden_vs_kv_contention: float = 0.0,
            prefix_reuse_trust: float = 0.0,
            replay_determinism_mean: float = 0.0,
            replay_dominance_witness_mean: float = 0.0,
            hidden_wins_primary_score: float = 0.0,
    ) -> tuple[ReplayDecision, float, float, str, bool]:
        """Return (decision, decision_confidence,
        replay_dominance, regime_used, dominance_primary_active).
        """
        regime = self.classify_regime_v5(
            c,
            hidden_write_total_l2=hidden_write_total_l2,
            hidden_vs_kv_contention=hidden_vs_kv_contention,
            prefix_reuse_trust=prefix_reuse_trust,
            replay_dominance_witness_mean=(
                replay_dominance_witness_mean),
            hidden_wins_primary_score=(
                hidden_wins_primary_score))
        head = self.per_regime_heads_v5.get(regime)
        if head is None:
            # Fall back to V4.
            v4_dec, v4_conf, v4_dom, v4_reg = (
                self.inner_v4.decide_v4(
                    c,
                    hidden_write_total_l2=hidden_write_total_l2,
                    hidden_vs_kv_contention=(
                        hidden_vs_kv_contention),
                    prefix_reuse_trust=prefix_reuse_trust,
                    replay_determinism_mean=(
                        replay_determinism_mean)))
            self.audit_v5.append({
                "stage": "v5_v4_fallback",
                "regime": str(regime),
                **v4_dec.to_dict()})
            return (
                v4_dec, float(v4_conf), float(v4_dom),
                str(regime), False)
        feat = _candidate_feature_v5(
            c,
            hidden_write_total_l2=hidden_write_total_l2,
            hidden_vs_kv_contention=hidden_vs_kv_contention,
            prefix_reuse_trust=prefix_reuse_trust,
            replay_dominance_witness_mean=(
                replay_dominance_witness_mean),
            hidden_wins_primary_score=(
                hidden_wins_primary_score))
        scores = _np.asarray(
            head, dtype=_np.float64) @ feat
        # Replay-dominance-primary bonus: add to REUSE score.
        reuse_idx = W60_REPLAY_DECISIONS.index(
            W60_REPLAY_DECISION_REUSE)
        dominance_primary_active = bool(
            (regime
             == W64_REPLAY_REGIME_REPLAY_DOMINANCE_PRIMARY)
            or (float(replay_dominance_witness_mean)
                >= float(self.dominance_primary_bonus)))
        if dominance_primary_active:
            scores[reuse_idx] = float(scores[reuse_idx]) + float(
                self.dominance_primary_bonus
                * float(replay_dominance_witness_mean))
        # V4's determinism bonus too.
        scores[reuse_idx] = float(scores[reuse_idx]) + float(
            self.inner_v4.determinism_bonus
            * float(replay_determinism_mean))
        probs = _softmax_4(scores)
        idx = int(_np.argmax(probs))
        confidence = float(probs[idx])
        chosen_kind = W60_REPLAY_DECISIONS[idx]
        sorted_probs = _np.sort(probs)[::-1]
        replay_dominance = float(
            sorted_probs[0] - sorted_probs[1])
        if confidence < float(
                self.inner_v4.inner_v3.inner_v2.abstain_threshold):
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
                rationale=f"v5_regime_{regime}_reuse",
                crc_passed=bool(c.crc_passed))
        elif chosen_kind == W60_REPLAY_DECISION_RECOMPUTE:
            dec = ReplayDecision(
                decision=chosen_kind,
                flop_chosen=int(c.flop_recompute),
                drift_chosen=float(c.drift_l2_recompute),
                flop_saving_vs_recompute=0.0,
                rationale=f"v5_regime_{regime}_recompute",
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
                rationale=f"v5_regime_{regime}_fallback",
                crc_passed=bool(c.crc_passed))
        else:
            dec = ReplayDecision(
                decision=W60_REPLAY_DECISION_ABSTAIN,
                flop_chosen=0, drift_chosen=0.0,
                flop_saving_vs_recompute=1.0,
                rationale=f"v5_regime_{regime}_abstain",
                crc_passed=bool(c.crc_passed))
        self.audit_v5.append({
            "stage": "v5_per_regime",
            "regime": str(regime),
            "probs": [float(round(float(p), 12)) for p in probs],
            "confidence": float(round(confidence, 12)),
            "replay_dominance": float(round(
                replay_dominance, 12)),
            "replay_dominance_witness_mean": float(round(
                replay_dominance_witness_mean, 12)),
            "dominance_primary_active": bool(
                dominance_primary_active),
            **dec.to_dict()})
        return (
            dec, float(confidence),
            float(replay_dominance), str(regime),
            bool(dominance_primary_active))


@dataclasses.dataclass(frozen=True)
class ReplayControllerV5FitReport:
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
            "kind": "replay_controller_v5_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v5_per_regime(
        *, controller: ReplayControllerV5,
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
        train_dominance_witness_per_regime: dict[
            str, Sequence[float]] | None = None,
        train_hidden_wins_primary_per_regime: dict[
            str, Sequence[float]] | None = None,
        ridge_lambda: float = (
            W64_DEFAULT_REPLAY_V5_RIDGE_LAMBDA),
) -> tuple[ReplayControllerV5, ReplayControllerV5FitReport]:
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
    centroids = _np.zeros((R, 9), dtype=_np.float64)
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
        dw = (
            list(train_dominance_witness_per_regime[regime])
            if train_dominance_witness_per_regime is not None
            and regime in train_dominance_witness_per_regime
            else [0.0] * len(cands))
        hwp = (
            list(train_hidden_wins_primary_per_regime[regime])
            if train_hidden_wins_primary_per_regime is not None
            and regime in train_hidden_wins_primary_per_regime
            else [0.0] * len(cands))
        n = int(len(cands))
        n_per_regime.append(n)
        if n == 0:
            per_pre.append(0.0)
            per_post.append(0.0)
            continue
        X = _np.zeros((n, 10), dtype=_np.float64)
        Y = _np.zeros((n, 4), dtype=_np.float64)
        regime_feats = _np.zeros((n, 9), dtype=_np.float64)
        for i, (c, lbl) in enumerate(zip(cands, labels)):
            X[i] = _candidate_feature_v5(
                c, hidden_write_total_l2=float(hl2[i]),
                hidden_vs_kv_contention=float(hvkv[i]),
                prefix_reuse_trust=float(prt[i]),
                replay_dominance_witness_mean=float(dw[i]),
                hidden_wins_primary_score=float(hwp[i]))
            regime_feats[i] = _regime_feature_v5(
                c, hidden_write_total_l2=float(hl2[i]),
                hidden_vs_kv_contention=float(hvkv[i]),
                prefix_reuse_trust=float(prt[i]),
                replay_dominance_witness_mean=float(dw[i]),
                hidden_wins_primary_score=float(hwp[i]))
            if lbl in W60_REPLAY_DECISIONS:
                Y[i, W60_REPLAY_DECISIONS.index(lbl)] = 1.0
            else:
                Y[i, W60_REPLAY_DECISIONS.index(
                    W60_REPLAY_DECISION_ABSTAIN)] = 1.0
        lam = max(float(ridge_lambda), 1e-9)
        A = X.T @ X + lam * _np.eye(10, dtype=_np.float64)
        b = X.T @ Y
        try:
            Wt = _np.linalg.solve(A, b)
        except Exception:
            Wt = _np.zeros((10, 4), dtype=_np.float64)
        Y_hat = X @ Wt
        per_pre.append(float(_np.mean(_np.abs(Y))))
        per_post.append(float(_np.mean(_np.abs(Y - Y_hat))))
        per_regime_heads[regime] = _np.asarray(
            Wt.T, dtype=_np.float64).copy()
        centroids[r_idx] = regime_feats.mean(axis=0)
    fitted = dataclasses.replace(
        controller,
        per_regime_heads_v5=per_regime_heads,
        regime_gate_v5=centroids.copy(),
    )
    converged = all(
        po <= pr + 1e-9 for pr, po in zip(per_pre, per_post))
    report = ReplayControllerV5FitReport(
        schema=W64_REPLAY_CONTROLLER_V5_SCHEMA_VERSION,
        n_regimes=int(R),
        n_train_per_regime=tuple(n_per_regime),
        per_regime_pre_residual=tuple(per_pre),
        per_regime_post_residual=tuple(per_post),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_four_way_bridge_classifier(
        *, controller: ReplayControllerV5,
        train_features: Sequence[Sequence[float]],
        train_labels: Sequence[str],
        ridge_lambda: float = (
            W64_DEFAULT_REPLAY_V5_RIDGE_LAMBDA),
) -> tuple[ReplayControllerV5, dict[str, Any]]:
    """Fit a 4-class closed-form ridge classifier over 9 regime
    features. Labels must come from ``W64_BRIDGE_LABELS_V5``."""
    n = int(len(train_features))
    if n == 0 or len(train_labels) != n:
        raise ValueError(
            "fit requires non-empty matching inputs")
    X = _np.asarray(train_features, dtype=_np.float64)
    if X.shape[1] != 9:
        raise ValueError("regime features must be 9-dim")
    Y = _np.zeros((n, 4), dtype=_np.float64)
    for i, lbl in enumerate(train_labels):
        if lbl in W64_BRIDGE_LABELS_V5:
            Y[i, W64_BRIDGE_LABELS_V5.index(lbl)] = 1.0
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(9, dtype=_np.float64)
    b = X.T @ Y
    try:
        Wt = _np.linalg.solve(A, b)
    except Exception:
        Wt = _np.zeros((9, 4), dtype=_np.float64)
    Y_hat = X @ Wt
    preds = _np.argmax(Y_hat, axis=1)
    labels = _np.argmax(Y, axis=1)
    acc = float(_np.mean(preds == labels))
    fitted = dataclasses.replace(
        controller,
        four_way_bridge_classifier=_np.asarray(
            Wt.T, dtype=_np.float64).copy())
    audit = {
        "schema": W64_REPLAY_CONTROLLER_V5_SCHEMA_VERSION,
        "kind": "four_way_bridge_classifier_fit",
        "n_train": int(n),
        "accuracy_train": float(acc),
    }
    return fitted, audit


def predict_four_way_bridge(
        controller: ReplayControllerV5,
        feature: Sequence[float],
) -> str:
    if controller.four_way_bridge_classifier is None:
        return "kv_wins"
    f = _np.asarray(feature, dtype=_np.float64).reshape(-1)
    if f.size < 9:
        f = _np.concatenate([
            f, _np.zeros(9 - f.size, dtype=_np.float64)])
    elif f.size > 9:
        f = f[:9]
    scores = (
        _np.asarray(
            controller.four_way_bridge_classifier,
            dtype=_np.float64) @ f)
    idx = int(_np.argmax(scores))
    if 0 <= idx < int(len(W64_BRIDGE_LABELS_V5)):
        return W64_BRIDGE_LABELS_V5[idx]
    return "kv_wins"


def fit_replay_dominance_primary_head_v5(
        *, controller: ReplayControllerV5,
        train_features: Sequence[Sequence[float]],
        train_decisions: Sequence[str],
        ridge_lambda: float = (
            W64_DEFAULT_REPLAY_V5_RIDGE_LAMBDA),
) -> tuple[ReplayControllerV5, dict[str, Any]]:
    """Fit a 4-decision closed-form ridge over 9 regime features
    that favours REUSE under high replay dominance."""
    n = int(len(train_features))
    if n == 0 or len(train_decisions) != n:
        raise ValueError(
            "fit requires non-empty matching inputs")
    X = _np.asarray(train_features, dtype=_np.float64)
    if X.shape[1] != 9:
        raise ValueError("regime features must be 9-dim")
    Y = _np.zeros((n, 4), dtype=_np.float64)
    for i, dec in enumerate(train_decisions):
        if dec in W60_REPLAY_DECISIONS:
            Y[i, W60_REPLAY_DECISIONS.index(dec)] = 1.0
        else:
            Y[i, W60_REPLAY_DECISIONS.index(
                W60_REPLAY_DECISION_ABSTAIN)] = 1.0
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(9, dtype=_np.float64)
    b = X.T @ Y
    try:
        Wt = _np.linalg.solve(A, b)
    except Exception:
        Wt = _np.zeros((9, 4), dtype=_np.float64)
    Y_hat = X @ Wt
    pre = float(_np.mean(_np.abs(Y)))
    post = float(_np.mean(_np.abs(Y - Y_hat)))
    fitted = dataclasses.replace(
        controller,
        replay_dominance_primary_head=_np.asarray(
            Wt.T, dtype=_np.float64).copy())
    audit = {
        "schema": W64_REPLAY_CONTROLLER_V5_SCHEMA_VERSION,
        "kind": "replay_dominance_primary_head_fit",
        "n_train": int(n),
        "pre": float(pre),
        "post": float(post),
        "converged": bool(post <= pre + 1e-9),
    }
    return fitted, audit


@dataclasses.dataclass(frozen=True)
class ReplayControllerV5Witness:
    schema: str
    controller_cid: str
    n_decisions: int
    mean_confidence: float
    mean_replay_dominance: float
    regime_count: dict[str, int]
    four_way_bridge_classifier_trained: bool
    replay_dominance_primary_head_trained: bool
    n_dominance_primary_active: int

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
            "four_way_bridge_classifier_trained": bool(
                self.four_way_bridge_classifier_trained),
            "replay_dominance_primary_head_trained": bool(
                self.replay_dominance_primary_head_trained),
            "n_dominance_primary_active": int(
                self.n_dominance_primary_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v5_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v5_witness(
        controller: ReplayControllerV5,
) -> ReplayControllerV5Witness:
    confs = [
        float(entry.get("confidence", 0.0))
        for entry in controller.audit_v5
        if "confidence" in entry]
    doms = [
        float(entry.get("replay_dominance", 0.0))
        for entry in controller.audit_v5
        if "replay_dominance" in entry]
    regime_count: dict[str, int] = {}
    for entry in controller.audit_v5:
        r = str(entry.get("regime", ""))
        if r:
            regime_count[r] = int(regime_count.get(r, 0) + 1)
    n_dpa = sum(
        1 for entry in controller.audit_v5
        if bool(entry.get("dominance_primary_active", False)))
    return ReplayControllerV5Witness(
        schema=W64_REPLAY_CONTROLLER_V5_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit_v5)),
        mean_confidence=(
            float(_np.mean(_np.asarray(
                confs, dtype=_np.float64)))
            if confs else 0.0),
        mean_replay_dominance=(
            float(_np.mean(_np.asarray(
                doms, dtype=_np.float64)))
            if doms else 0.0),
        regime_count=regime_count,
        four_way_bridge_classifier_trained=bool(
            controller.four_way_bridge_classifier is not None),
        replay_dominance_primary_head_trained=bool(
            controller.replay_dominance_primary_head
            is not None),
        n_dominance_primary_active=int(n_dpa),
    )


__all__ = [
    "W64_REPLAY_CONTROLLER_V5_SCHEMA_VERSION",
    "W64_REPLAY_REGIME_REPLAY_DOMINANCE_PRIMARY",
    "W64_REPLAY_REGIMES_V5",
    "W64_BRIDGE_LABELS_V5",
    "W64_DEFAULT_REPLAY_V5_RIDGE_LAMBDA",
    "W64_DEFAULT_REPLAY_V5_DOMINANCE_PRIMARY_BONUS",
    "ReplayControllerV5",
    "ReplayControllerV5FitReport",
    "fit_replay_controller_v5_per_regime",
    "fit_four_way_bridge_classifier",
    "predict_four_way_bridge",
    "fit_replay_dominance_primary_head_v5",
    "ReplayControllerV5Witness",
    "emit_replay_controller_v5_witness",
]
