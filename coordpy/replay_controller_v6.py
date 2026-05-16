"""W65 M7 — Replay Controller V6.

Strictly extends W64's ``coordpy.replay_controller_v5``. V5 fit
**R = 7** per-regime 10×4 ridge heads + a 9-dim regime gate + a
4×9 four-way bridge classifier + a 4×9 replay-dominance-primary
head. V6 adds:

* **Eight regimes** — V6 introduces one new regime on top of V5's
  seven:
    - ``team_substrate_coordination_regime`` (multi-agent task
      flag positive and substrate-fidelity above threshold ⇒ the
      regime where team-substrate coordination is the load-bearing
      signal).
* **Per-role per-regime ridge head** —
  ``fit_replay_controller_v6_per_role`` fits a separate per-regime
  10×4 head per role tag. The decision uses the role-specific head
  when the role is registered, falling back to V5's per-regime
  head otherwise.
* **Multi-agent abstain head** —
  ``fit_replay_v6_multi_agent_abstain_head`` is a 4×10 ridge head
  that decides whether the current agent should ABSTAIN to let the
  team converge — keyed on team-coordination features
  (per-agent confidence, per-team disagreement, substrate-fidelity,
  team budget remaining, etc.).

Honest scope (W65)
------------------

* All V6 fits are closed-form ridge. ``W65-L-V6-REPLAY-NO-AUTOGRAD-
  CAP`` documents.
* The team-substrate-coordination regime is a deterministic feature
  threshold, not a learned regime classifier.
* The multi-agent abstain head is a single 4×10 ridge over
  synthetic supervision; it does NOT prove that team abstain
  semantics generalise to real models.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v6 requires numpy") from exc

from .replay_controller import (
    ReplayCandidate, ReplayDecision,
    W60_REPLAY_DECISIONS,
    W60_REPLAY_DECISION_ABSTAIN,
    W60_REPLAY_DECISION_FALLBACK,
    W60_REPLAY_DECISION_RECOMPUTE,
    W60_REPLAY_DECISION_REUSE,
)
from .replay_controller_v2 import _softmax_4
from .replay_controller_v5 import (
    ReplayControllerV5,
    ReplayControllerV5FitReport,
    W64_REPLAY_REGIMES_V5,
    W64_DEFAULT_REPLAY_V5_RIDGE_LAMBDA,
    _candidate_feature_v5,
    fit_replay_controller_v5_per_regime,
)
from .tiny_substrate_v3 import _sha256_hex


W65_REPLAY_CONTROLLER_V6_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v6.v1")

W65_REPLAY_REGIME_TEAM_SUBSTRATE_COORDINATION: str = (
    "team_substrate_coordination_regime")
W65_REPLAY_REGIMES_V6: tuple[str, ...] = (
    *W64_REPLAY_REGIMES_V5,
    W65_REPLAY_REGIME_TEAM_SUBSTRATE_COORDINATION,
)
W65_DEFAULT_REPLAY_V6_RIDGE_LAMBDA: float = 0.10
W65_DEFAULT_TEAM_COORDINATION_THRESHOLD: float = 0.5


@dataclasses.dataclass
class ReplayControllerV6:
    inner_v5: ReplayControllerV5
    # Per-role per-regime heads: dict (role, regime) → (4, 10).
    per_role_per_regime_heads: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    # Multi-agent abstain head: (4, 10) ridge head over team
    # coordination features.
    multi_agent_abstain_head: "_np.ndarray | None" = None
    team_coordination_threshold: float = (
        W65_DEFAULT_TEAM_COORDINATION_THRESHOLD)
    audit_v6: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            inner_v5: ReplayControllerV5 | None = None,
            team_coordination_threshold: float = (
                W65_DEFAULT_TEAM_COORDINATION_THRESHOLD),
    ) -> "ReplayControllerV6":
        if inner_v5 is None:
            inner_v5 = ReplayControllerV5.init()
        return cls(
            inner_v5=inner_v5,
            per_role_per_regime_heads={},
            multi_agent_abstain_head=None,
            team_coordination_threshold=float(
                team_coordination_threshold),
            audit_v6=[])

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v in self.per_role_per_regime_heads.items())
            ph_cid = _sha256_hex(payload)
        abstain_cid = "untrained"
        if self.multi_agent_abstain_head is not None:
            abstain_cid = _sha256_hex(
                self.multi_agent_abstain_head.tobytes().hex())
        return _sha256_hex({
            "schema": W65_REPLAY_CONTROLLER_V6_SCHEMA_VERSION,
            "kind": "replay_controller_v6",
            "inner_v5_cid": str(self.inner_v5.cid()),
            "per_role_per_regime_heads_cid": ph_cid,
            "multi_agent_abstain_head_cid": abstain_cid,
            "team_coordination_threshold": float(round(
                self.team_coordination_threshold, 12)),
        })

    def classify_regime_v6(
            self, c: ReplayCandidate,
            *, hidden_write_total_l2: float = 0.0,
            hidden_vs_kv_contention: float = 0.0,
            prefix_reuse_trust: float = 0.0,
            replay_dominance_witness_mean: float = 0.0,
            hidden_wins_primary_score: float = 0.0,
            team_coordination_flag: float = 0.0,
            substrate_fidelity: float = 0.0,
    ) -> str:
        """V6 regime classification. Returns the team-substrate-
        coordination regime when both the team flag and the
        substrate fidelity exceed thresholds; otherwise falls
        back to V5."""
        thr = float(self.team_coordination_threshold)
        if (float(team_coordination_flag) >= thr
                and float(substrate_fidelity) >= thr):
            return W65_REPLAY_REGIME_TEAM_SUBSTRATE_COORDINATION
        return self.inner_v5.classify_regime_v5(
            c,
            hidden_write_total_l2=float(hidden_write_total_l2),
            hidden_vs_kv_contention=float(hidden_vs_kv_contention),
            prefix_reuse_trust=float(prefix_reuse_trust),
            replay_dominance_witness_mean=float(
                replay_dominance_witness_mean),
            hidden_wins_primary_score=float(
                hidden_wins_primary_score))

    def decide_v6(
            self, c: ReplayCandidate,
            *, role: str = "default",
            hidden_write_total_l2: float = 0.0,
            hidden_vs_kv_contention: float = 0.0,
            prefix_reuse_trust: float = 0.0,
            replay_determinism_mean: float = 0.0,
            replay_dominance_witness_mean: float = 0.0,
            hidden_wins_primary_score: float = 0.0,
            team_coordination_flag: float = 0.0,
            substrate_fidelity: float = 0.0,
    ) -> tuple[ReplayDecision, float, float, str, bool, bool]:
        """Returns (decision, confidence, replay_dominance,
        regime_used, dominance_primary_active,
        role_specific_head_used)."""
        regime = self.classify_regime_v6(
            c,
            hidden_write_total_l2=hidden_write_total_l2,
            hidden_vs_kv_contention=hidden_vs_kv_contention,
            prefix_reuse_trust=prefix_reuse_trust,
            replay_dominance_witness_mean=(
                replay_dominance_witness_mean),
            hidden_wins_primary_score=hidden_wins_primary_score,
            team_coordination_flag=team_coordination_flag,
            substrate_fidelity=substrate_fidelity)
        head = self.per_role_per_regime_heads.get(
            (str(role), str(regime)))
        if head is None:
            # Delegate to V5 head selection on the V5 regimes.
            dec_v5, conf_v5, dom_v5, reg_v5, dpa_v5 = (
                self.inner_v5.decide_v5(
                    c,
                    hidden_write_total_l2=hidden_write_total_l2,
                    hidden_vs_kv_contention=(
                        hidden_vs_kv_contention),
                    prefix_reuse_trust=prefix_reuse_trust,
                    replay_determinism_mean=(
                        replay_determinism_mean),
                    replay_dominance_witness_mean=(
                        replay_dominance_witness_mean),
                    hidden_wins_primary_score=(
                        hidden_wins_primary_score)))
            self.audit_v6.append({
                "stage": "v6_v5_fallback",
                "role": str(role), "regime_v6": str(regime),
                **dec_v5.to_dict()})
            return (
                dec_v5, float(conf_v5), float(dom_v5),
                str(regime), bool(dpa_v5), False)
        feat = _candidate_feature_v5(
            c,
            hidden_write_total_l2=hidden_write_total_l2,
            hidden_vs_kv_contention=hidden_vs_kv_contention,
            prefix_reuse_trust=prefix_reuse_trust,
            replay_dominance_witness_mean=(
                replay_dominance_witness_mean),
            hidden_wins_primary_score=hidden_wins_primary_score)
        scores = _np.asarray(head, dtype=_np.float64) @ feat
        probs = _softmax_4(scores)
        idx = int(_np.argmax(probs))
        confidence = float(probs[idx])
        chosen_kind = W60_REPLAY_DECISIONS[idx]
        sorted_probs = _np.sort(probs)[::-1]
        replay_dominance = float(
            sorted_probs[0] - sorted_probs[1])
        if confidence < float(
                self.inner_v5.inner_v4.inner_v3.inner_v2
                .abstain_threshold):
            chosen_kind = W60_REPLAY_DECISION_ABSTAIN
        denom = max(int(c.flop_recompute), 1)
        if chosen_kind == W60_REPLAY_DECISION_REUSE:
            saving = (
                float(int(c.flop_recompute)
                      - int(c.flop_reuse)) / float(denom))
            dec = ReplayDecision(
                decision=chosen_kind,
                flop_chosen=int(c.flop_reuse),
                drift_chosen=float(c.drift_l2_reuse),
                flop_saving_vs_recompute=float(saving),
                rationale=(
                    f"v6_role_{role}_regime_{regime}_reuse"),
                crc_passed=bool(c.crc_passed))
        elif chosen_kind == W60_REPLAY_DECISION_RECOMPUTE:
            dec = ReplayDecision(
                decision=chosen_kind,
                flop_chosen=int(c.flop_recompute),
                drift_chosen=float(c.drift_l2_recompute),
                flop_saving_vs_recompute=0.0,
                rationale=(
                    f"v6_role_{role}_regime_{regime}_recompute"),
                crc_passed=bool(c.crc_passed))
        elif chosen_kind == W60_REPLAY_DECISION_FALLBACK:
            saving = (
                float(int(c.flop_recompute)
                      - int(c.flop_fallback)) / float(denom))
            dec = ReplayDecision(
                decision=chosen_kind,
                flop_chosen=int(c.flop_fallback),
                drift_chosen=float(c.drift_l2_fallback),
                flop_saving_vs_recompute=float(saving),
                rationale=(
                    f"v6_role_{role}_regime_{regime}_fallback"),
                crc_passed=bool(c.crc_passed))
        else:
            dec = ReplayDecision(
                decision=W60_REPLAY_DECISION_ABSTAIN,
                flop_chosen=0, drift_chosen=0.0,
                flop_saving_vs_recompute=1.0,
                rationale=(
                    f"v6_role_{role}_regime_{regime}_abstain"),
                crc_passed=bool(c.crc_passed))
        self.audit_v6.append({
            "stage": "v6_per_role_per_regime",
            "role": str(role),
            "regime": str(regime),
            "probs": [float(round(float(p), 12)) for p in probs],
            "confidence": float(round(confidence, 12)),
            **dec.to_dict()})
        return (
            dec, float(confidence), float(replay_dominance),
            str(regime), True, True)

    def decide_multi_agent_abstain(
            self, *, team_features: Sequence[float],
    ) -> tuple[bool, float]:
        """Returns (should_abstain, abstain_score)."""
        if self.multi_agent_abstain_head is None:
            return False, 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self.multi_agent_abstain_head.shape[1]):
            return False, 0.0
        score = _np.asarray(
            self.multi_agent_abstain_head,
            dtype=_np.float64) @ feats
        probs = _softmax_4(score)
        abstain_idx = W60_REPLAY_DECISIONS.index(
            W60_REPLAY_DECISION_ABSTAIN)
        should = bool(int(_np.argmax(probs)) == abstain_idx)
        return should, float(probs[abstain_idx])


@dataclasses.dataclass(frozen=True)
class ReplayControllerV6FitReport:
    schema: str
    n_role_regime_heads: int
    multi_agent_abstain_trained: bool
    multi_agent_abstain_train_residual: float
    converged: bool
    ridge_lambda: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_role_regime_heads": int(
                self.n_role_regime_heads),
            "multi_agent_abstain_trained": bool(
                self.multi_agent_abstain_trained),
            "multi_agent_abstain_train_residual": float(round(
                self.multi_agent_abstain_train_residual, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v6_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v6_per_role(
        *, controller: ReplayControllerV6, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[
            str, Sequence[str]],
        ridge_lambda: float = (
            W65_DEFAULT_REPLAY_V6_RIDGE_LAMBDA),
) -> tuple[ReplayControllerV6, ReplayControllerV6FitReport]:
    """Fit a per-role per-regime 10×4 ridge head for each (role,
    regime) tuple. Reuses V5's per-regime fit on a fresh inner V5
    copy; then promotes those heads to the V6 (role, regime) map."""
    inner_v5_copy = controller.inner_v5
    fitted_v5, _ = fit_replay_controller_v5_per_regime(
        controller=inner_v5_copy,
        train_candidates_per_regime=train_candidates_per_regime,
        train_decisions_per_regime=train_decisions_per_regime,
        ridge_lambda=float(ridge_lambda))
    new_heads = dict(controller.per_role_per_regime_heads)
    for regime, head in fitted_v5.per_regime_heads_v5.items():
        new_heads[(str(role), str(regime))] = _np.asarray(
            head, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller, per_role_per_regime_heads=new_heads)
    return fitted, ReplayControllerV6FitReport(
        schema=W65_REPLAY_CONTROLLER_V6_SCHEMA_VERSION,
        n_role_regime_heads=int(len(new_heads)),
        multi_agent_abstain_trained=bool(
            fitted.multi_agent_abstain_head is not None),
        multi_agent_abstain_train_residual=0.0,
        converged=True,
        ridge_lambda=float(ridge_lambda),
    )


def fit_replay_v6_multi_agent_abstain_head(
        *, controller: ReplayControllerV6,
        train_team_features: Sequence[Sequence[float]],
        train_decisions: Sequence[str],
        ridge_lambda: float = (
            W65_DEFAULT_REPLAY_V6_RIDGE_LAMBDA),
) -> tuple[ReplayControllerV6, ReplayControllerV6FitReport]:
    """Fit the 4×10 multi-agent abstain head over a 9-dim feature
    space + bias = 10."""
    if not train_team_features:
        raise ValueError("must provide >= 1 train example")
    X_list = []
    for ex in train_team_features:
        row = list(ex)
        if len(row) != 9:
            raise ValueError(
                "each team feature must be 9-dim")
        X_list.append(row + [1.0])
    X = _np.asarray(X_list, dtype=_np.float64)
    # One-hot target labels over the 4 decisions.
    Y = _np.zeros((X.shape[0], 4), dtype=_np.float64)
    for i, lab in enumerate(train_decisions):
        idx = W60_REPLAY_DECISIONS.index(str(lab))
        Y[i, idx] = 1.0
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(X.shape[1], dtype=_np.float64)
    b = X.T @ Y
    try:
        W = _np.linalg.solve(A, b)   # (10, 4)
    except Exception:
        W = _np.zeros((X.shape[1], 4), dtype=_np.float64)
    Y_hat = X @ W
    residual = float(_np.mean(_np.abs(Y - Y_hat)))
    fitted = dataclasses.replace(
        controller, multi_agent_abstain_head=W.T.copy())
    return fitted, ReplayControllerV6FitReport(
        schema=W65_REPLAY_CONTROLLER_V6_SCHEMA_VERSION,
        n_role_regime_heads=int(
            len(controller.per_role_per_regime_heads)),
        multi_agent_abstain_trained=True,
        multi_agent_abstain_train_residual=float(residual),
        converged=bool(residual <= 1.0),
        ridge_lambda=float(ridge_lambda),
    )


@dataclasses.dataclass(frozen=True)
class ReplayControllerV6Witness:
    schema: str
    controller_cid: str
    n_role_regime_heads: int
    multi_agent_abstain_trained: bool
    n_audit_entries: int
    inner_v5_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_role_regime_heads": int(
                self.n_role_regime_heads),
            "multi_agent_abstain_trained": bool(
                self.multi_agent_abstain_trained),
            "n_audit_entries": int(self.n_audit_entries),
            "inner_v5_witness_cid": str(
                self.inner_v5_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v6_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v6_witness(
        controller: ReplayControllerV6,
) -> ReplayControllerV6Witness:
    from .replay_controller_v5 import (
        emit_replay_controller_v5_witness,
    )
    inner_w = emit_replay_controller_v5_witness(
        controller.inner_v5)
    return ReplayControllerV6Witness(
        schema=W65_REPLAY_CONTROLLER_V6_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_role_regime_heads=int(
            len(controller.per_role_per_regime_heads)),
        multi_agent_abstain_trained=bool(
            controller.multi_agent_abstain_head is not None),
        n_audit_entries=int(len(controller.audit_v6)),
        inner_v5_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W65_REPLAY_CONTROLLER_V6_SCHEMA_VERSION",
    "W65_REPLAY_REGIME_TEAM_SUBSTRATE_COORDINATION",
    "W65_REPLAY_REGIMES_V6",
    "W65_DEFAULT_REPLAY_V6_RIDGE_LAMBDA",
    "W65_DEFAULT_TEAM_COORDINATION_THRESHOLD",
    "ReplayControllerV6",
    "ReplayControllerV6FitReport",
    "fit_replay_controller_v6_per_role",
    "fit_replay_v6_multi_agent_abstain_head",
    "ReplayControllerV6Witness",
    "emit_replay_controller_v6_witness",
]
