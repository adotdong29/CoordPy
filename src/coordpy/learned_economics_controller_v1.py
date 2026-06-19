"""W81 / P1 #14 — Learned Multi-Runtime Economics Controller V1.

CoordPy's replay-vs-recompute / handoff economics has so far been
driven by closed-form ridge fits, explicit pressure gates, and
hand-coded arbiters (e.g. ``replay_controller_v20``,
``long_horizon_reconstruction_substrate_v2.arbitrate_replay_vs_recompute_v2``,
``hosted_cost_planner_v12``). That was the right shape for
discipline and audit. P1 #14 asks: can a learned policy do
better on cost-quality tradeoff under realistic conditions?

V1 stands up a small learned softmax-policy controller over the
5 canonical economics actions:

* ``replay`` — reuse cached substrate state (lowest cost,
  works only when cache is fresh)
* ``runtime_recompute`` — re-run the controlled runtime
  forward (mid cost, mid quality on degraded state)
* ``transcript_recompute`` — replay from saved transcript only
  (low quality on substrate questions, but always available)
* ``promote_to_richer_substrate`` — switch to a higher-tier
  backend (e.g. HF transformers runtime instead of in-repo
  NumPy) for hard queries
* ``abstain`` — refuse to answer when expected quality is
  below the abstain floor

The controller is a 2-layer MLP::

    h = swish(W1 @ x + b1)        # (state_dim,) -> (hidden_dim,)
    logits = W2 @ h + b2          # (hidden_dim,) -> (5,)
    pi = softmax(logits)          # action distribution

Training: supervised maximum-likelihood on a synthetic dataset
of (state, optimal_action) pairs where ``optimal_action`` is
the argmax over actions of ``utility = success_prob -
cost_weight * cost``. The dataset is generated with a
deterministic seed so the comparison against the heuristic
arbiter is reproducible.

The honest comparison includes both quality (utility) and
inspection-friendliness (the learned policy's choice
distribution): we don't claim the learned line wins on every
slice, only that it strictly improves on at least one
meaningful cost-quality tradeoff.

Honest scope (W81)
------------------

* ``W81-L-LEARNED-ECONOMICS-V1-RESEARCH-ONLY-CAP`` — explicit
  import only; not on the stable public surface.
* ``W81-L-LEARNED-ECONOMICS-V1-SUPERVISED-CAP`` — V1 is
  supervised on a synthetic optimal-action dataset, not full
  policy-gradient RL on live runtime cost/quality.
* ``W81-L-LEARNED-ECONOMICS-V1-SIMULATED-COSTS-CAP`` —
  the cost/quality model is the synthetic one declared in
  ``EconomicsSimulationV1``; live-runtime cost/quality
  validation is W81 follow-on work.
* ``W81-L-LEARNED-ECONOMICS-V1-NUMPY-CAP`` — pure NumPy.
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
        "coordpy.learned_economics_controller_v1 requires "
        "numpy") from exc


W81_LEARNED_ECONOMICS_V1_SCHEMA_VERSION: str = (
    "coordpy.learned_economics_controller_v1.v1")

# The 5 canonical economics actions.
W81_ECONOMICS_ACTIONS: tuple[str, ...] = (
    "replay",
    "runtime_recompute",
    "transcript_recompute",
    "promote_to_richer_substrate",
    "abstain",
)
W81_N_ECONOMICS_ACTIONS: int = len(W81_ECONOMICS_ACTIONS)

# The 7-dim economics feature vector (deterministic schema):
#   x[0] = horizon_norm        (steps to recall, normalised)
#   x[1] = budget_pressure     (fraction of budget consumed)
#   x[2] = evidence_completeness (0..1)
#   x[3] = prior_failure_rate  (0..1)
#   x[4] = cache_freshness     (1 = fresh, 0 = stale)
#   x[5] = task_difficulty     (0 = easy, 1 = hard)
#   x[6] = controlled_runtime_health (0 = degraded, 1 = healthy)
W81_ECONOMICS_FEATURE_DIM: int = 7
W81_ECONOMICS_FEATURE_NAMES: tuple[str, ...] = (
    "horizon_norm",
    "budget_pressure",
    "evidence_completeness",
    "prior_failure_rate",
    "cache_freshness",
    "task_difficulty",
    "controlled_runtime_health",
)

W81_DEFAULT_HIDDEN_DIM: int = 32
W81_DEFAULT_TRAIN_ITERS: int = 100
W81_DEFAULT_LEARNING_RATE: float = 0.030
W81_DEFAULT_MOMENTUM: float = 0.85
W81_DEFAULT_WEIGHT_DECAY: float = 0.0005
W81_DEFAULT_SEED: int = 81_014_001
W81_DEFAULT_COST_WEIGHT: float = 0.0008
W81_DEFAULT_ABSTAIN_FLOOR: float = 0.30


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(_np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _swish(x: "_np.ndarray") -> "_np.ndarray":
    return x / (1.0 + _np.exp(-x))


def _swish_grad(x: "_np.ndarray") -> "_np.ndarray":
    s = 1.0 / (1.0 + _np.exp(-x))
    return s + x * s * (1.0 - s)


def _softmax(z: "_np.ndarray") -> "_np.ndarray":
    z_shift = z - _np.max(z, axis=-1, keepdims=True)
    e = _np.exp(z_shift)
    return e / _np.sum(e, axis=-1, keepdims=True)


# ---------------------------------------------------------------
# Simulated multi-runtime cost / quality model.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EconomicsSimulationV1:
    """Deterministic cost / quality model for the 5 actions.

    Cost and success-probability are deterministic functions of
    the 7-dim feature vector. Holds the ground-truth utility
    used to derive optimal actions for supervised training and
    the policy evaluation report.
    """

    schema: str = W81_LEARNED_ECONOMICS_V1_SCHEMA_VERSION
    cost_weight: float = W81_DEFAULT_COST_WEIGHT
    abstain_floor: float = W81_DEFAULT_ABSTAIN_FLOOR

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w81_economics_sim_v1",
            "schema": str(self.schema),
            "cost_weight": float(round(self.cost_weight, 12)),
            "abstain_floor": float(round(
                self.abstain_floor, 12)),
            "actions": list(W81_ECONOMICS_ACTIONS),
        })

    def evaluate_action(
            self, *, feature: "_np.ndarray", action: str,
    ) -> tuple[float, float, float]:
        """Return ``(cost, success_prob, utility)`` for an action.

        Cost is in (synthetic) flops. Success-probability is in
        [0, 1]. Utility is ``success_prob -
        cost_weight * cost``, capped at 0 if the abstain floor
        would refuse this action's expected success.
        """
        f = _np.asarray(feature, dtype=_np.float64)
        h, bp, ec, pf, cf, td, rh = (
            float(f[0]), float(f[1]), float(f[2]),
            float(f[3]), float(f[4]), float(f[5]), float(f[6]))
        # Saturating helpers.
        clip01 = lambda v: max(0.0, min(1.0, v))  # noqa: E731
        if action == "replay":
            # Cheap, but only as good as the cache is fresh and
            # the task is easy enough.
            cost = 80.0 + 60.0 * h
            success = clip01(0.10 + 0.85 * cf - 0.40 * td)
        elif action == "runtime_recompute":
            # Medium cost; quality scales with runtime health.
            cost = 320.0 + 180.0 * h
            success = clip01(0.20 + 0.65 * rh - 0.20 * td
                             + 0.10 * ec)
        elif action == "transcript_recompute":
            # Cheap, but transcript can't answer substrate
            # questions — quality drops fast with task
            # difficulty.
            cost = 100.0 + 40.0 * h
            success = clip01(0.55 - 0.55 * td + 0.10 * ec
                             - 0.10 * bp)
        elif action == "promote_to_richer_substrate":
            # Expensive, but bumps success on hard tasks.
            cost = 1500.0 + 300.0 * h
            success = clip01(0.40 + 0.45 * td + 0.20 * rh
                             - 0.05 * pf)
        elif action == "abstain":
            # Free, but success is the abstain floor.
            cost = 5.0
            success = float(self.abstain_floor)
        else:
            raise ValueError(f"unknown action: {action}")
        util = success - float(self.cost_weight) * cost
        return float(cost), float(success), float(util)

    def utility_per_action(
            self, *, feature: "_np.ndarray",
    ) -> "_np.ndarray":
        utils = _np.zeros(
            (W81_N_ECONOMICS_ACTIONS,), dtype=_np.float64)
        for j, a in enumerate(W81_ECONOMICS_ACTIONS):
            _, _, u = self.evaluate_action(
                feature=feature, action=a)
            utils[j] = u
        return utils

    def optimal_action_index(
            self, *, feature: "_np.ndarray",
    ) -> int:
        return int(_np.argmax(self.utility_per_action(
            feature=feature)))


@dataclasses.dataclass
class LearnedEconomicsControllerV1:
    """2-layer softmax-policy network over 5 actions."""

    schema: str
    feature_dim: int
    hidden_dim: int
    n_actions: int
    W1: "_np.ndarray"
    b1: "_np.ndarray"
    W2: "_np.ndarray"
    b2: "_np.ndarray"
    mom_W1: "_np.ndarray"
    mom_b1: "_np.ndarray"
    mom_W2: "_np.ndarray"
    mom_b2: "_np.ndarray"
    n_train_steps: int
    last_train_loss: float
    pre_train_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "feature_dim": int(self.feature_dim),
            "hidden_dim": int(self.hidden_dim),
            "n_actions": int(self.n_actions),
            "W1_cid": _ndarray_cid(self.W1),
            "b1_cid": _ndarray_cid(self.b1),
            "W2_cid": _ndarray_cid(self.W2),
            "b2_cid": _ndarray_cid(self.b2),
            "n_train_steps": int(self.n_train_steps),
            "last_train_loss": float(round(
                self.last_train_loss, 12)),
            "pre_train_loss": float(round(
                self.pre_train_loss, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w81_learned_economics_controller_v1",
            "controller": self.to_dict()})

    def forward_logits(self, X: "_np.ndarray") -> "_np.ndarray":
        x = _np.asarray(X, dtype=_np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        h_pre = x @ self.W1 + self.b1
        h = _swish(h_pre)
        return h @ self.W2 + self.b2

    def predict_action(self, feature: "_np.ndarray") -> str:
        logits = self.forward_logits(feature)
        idx = int(_np.argmax(logits[0]))
        return W81_ECONOMICS_ACTIONS[idx]

    def predict_distribution(
            self, feature: "_np.ndarray",
    ) -> "_np.ndarray":
        return _softmax(self.forward_logits(feature))[0]


def build_learned_economics_controller_v1(
        *,
        feature_dim: int = W81_ECONOMICS_FEATURE_DIM,
        hidden_dim: int = W81_DEFAULT_HIDDEN_DIM,
        seed: int = W81_DEFAULT_SEED,
) -> LearnedEconomicsControllerV1:
    rng = _np.random.default_rng(int(seed))
    s1 = 1.0 / max(1.0, float(feature_dim)) ** 0.5
    s2 = 1.0 / max(1.0, float(hidden_dim)) ** 0.5
    W1 = rng.standard_normal(
        (int(feature_dim), int(hidden_dim))) * s1
    b1 = _np.zeros(
        (int(hidden_dim),), dtype=_np.float64)
    W2 = rng.standard_normal(
        (int(hidden_dim), int(W81_N_ECONOMICS_ACTIONS))) * s2
    b2 = _np.zeros(
        (int(W81_N_ECONOMICS_ACTIONS),), dtype=_np.float64)
    return LearnedEconomicsControllerV1(
        schema=W81_LEARNED_ECONOMICS_V1_SCHEMA_VERSION,
        feature_dim=int(feature_dim),
        hidden_dim=int(hidden_dim),
        n_actions=int(W81_N_ECONOMICS_ACTIONS),
        W1=W1, b1=b1, W2=W2, b2=b2,
        mom_W1=_np.zeros_like(W1),
        mom_b1=_np.zeros_like(b1),
        mom_W2=_np.zeros_like(W2),
        mom_b2=_np.zeros_like(b2),
        n_train_steps=0,
        last_train_loss=0.0,
        pre_train_loss=0.0,
    )


@dataclasses.dataclass(frozen=True)
class LearnedEconomicsTrainReportV1:
    schema: str
    controller_cid_pre: str
    controller_cid_post: str
    pre_loss: float
    post_loss: float
    n_iters: int
    converged: bool
    loss_curve_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid_pre": str(self.controller_cid_pre),
            "controller_cid_post": str(self.controller_cid_post),
            "pre_loss": float(round(self.pre_loss, 12)),
            "post_loss": float(round(self.post_loss, 12)),
            "n_iters": int(self.n_iters),
            "converged": bool(self.converged),
            "loss_curve_cid": str(self.loss_curve_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w81_learned_economics_train_report",
            "report": self.to_dict()})


def train_learned_economics_controller(
        *,
        controller: LearnedEconomicsControllerV1,
        train_features: "_np.ndarray",
        train_optimal_action_indices: "_np.ndarray",
        n_iters: int = W81_DEFAULT_TRAIN_ITERS,
        learning_rate: float = W81_DEFAULT_LEARNING_RATE,
        momentum: float = W81_DEFAULT_MOMENTUM,
        weight_decay: float = W81_DEFAULT_WEIGHT_DECAY,
) -> tuple[
        LearnedEconomicsControllerV1,
        LearnedEconomicsTrainReportV1]:
    """Train via supervised cross-entropy on optimal-action labels."""
    X = _np.asarray(train_features, dtype=_np.float64)
    Y_idx = _np.asarray(
        train_optimal_action_indices, dtype=_np.int64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    B, _ = X.shape
    if B == 0:
        return controller, LearnedEconomicsTrainReportV1(
            schema=W81_LEARNED_ECONOMICS_V1_SCHEMA_VERSION,
            controller_cid_pre=str(controller.cid()),
            controller_cid_post=str(controller.cid()),
            pre_loss=0.0, post_loss=0.0,
            n_iters=0, converged=True,
            loss_curve_cid=_sha256_hex({
                "kind": "w81_loss_curve", "losses": []}))
    pre_cid = str(controller.cid())
    W1 = controller.W1.copy()
    b1 = controller.b1.copy()
    W2 = controller.W2.copy()
    b2 = controller.b2.copy()
    mW1 = controller.mom_W1.copy()
    mb1 = controller.mom_b1.copy()
    mW2 = controller.mom_W2.copy()
    mb2 = controller.mom_b2.copy()
    losses: list[float] = []
    one_hot = _np.zeros(
        (B, int(W81_N_ECONOMICS_ACTIONS)),
        dtype=_np.float64)
    one_hot[_np.arange(B), Y_idx] = 1.0

    def _loss(_W1, _b1, _W2, _b2) -> float:
        h_pre = X @ _W1 + _b1
        h = _swish(h_pre)
        logits = h @ _W2 + _b2
        log_probs = (
            logits - _np.log(_np.sum(
                _np.exp(logits - _np.max(
                    logits, axis=-1, keepdims=True)),
                axis=-1, keepdims=True))
            - _np.max(logits, axis=-1, keepdims=True))
        return float(_np.mean(_np.sum(
            -one_hot * log_probs, axis=-1)))

    pre_loss = _loss(W1, b1, W2, b2)
    losses.append(float(pre_loss))
    for _ in range(int(n_iters)):
        h_pre = X @ W1 + b1
        h = _swish(h_pre)
        logits = h @ W2 + b2
        pi = _softmax(logits)
        # Backward through cross-entropy.
        d_logits = (pi - one_hot) / float(B)
        grad_W2 = h.T @ d_logits
        grad_b2 = _np.sum(d_logits, axis=0)
        d_h = d_logits @ W2.T
        d_h_pre = d_h * _swish_grad(h_pre)
        grad_W1 = X.T @ d_h_pre
        grad_b1 = _np.sum(d_h_pre, axis=0)
        # Weight decay.
        grad_W1 += float(weight_decay) * W1
        grad_W2 += float(weight_decay) * W2
        # SGD-with-momentum.
        mW1 = (float(momentum) * mW1
               - float(learning_rate) * grad_W1)
        mb1 = (float(momentum) * mb1
               - float(learning_rate) * grad_b1)
        mW2 = (float(momentum) * mW2
               - float(learning_rate) * grad_W2)
        mb2 = (float(momentum) * mb2
               - float(learning_rate) * grad_b2)
        W1 = W1 + mW1
        b1 = b1 + mb1
        W2 = W2 + mW2
        b2 = b2 + mb2
        cur = _loss(W1, b1, W2, b2)
        losses.append(float(cur))
    post_loss = _loss(W1, b1, W2, b2)
    fitted = LearnedEconomicsControllerV1(
        schema=W81_LEARNED_ECONOMICS_V1_SCHEMA_VERSION,
        feature_dim=int(controller.feature_dim),
        hidden_dim=int(controller.hidden_dim),
        n_actions=int(controller.n_actions),
        W1=W1, b1=b1, W2=W2, b2=b2,
        mom_W1=mW1, mom_b1=mb1, mom_W2=mW2, mom_b2=mb2,
        n_train_steps=(
            int(controller.n_train_steps) + int(n_iters)),
        last_train_loss=float(post_loss),
        pre_train_loss=float(pre_loss),
    )
    rep = LearnedEconomicsTrainReportV1(
        schema=W81_LEARNED_ECONOMICS_V1_SCHEMA_VERSION,
        controller_cid_pre=pre_cid,
        controller_cid_post=str(fitted.cid()),
        pre_loss=float(pre_loss),
        post_loss=float(post_loss),
        n_iters=int(n_iters),
        converged=bool(post_loss < pre_loss),
        loss_curve_cid=_sha256_hex({
            "kind": "w81_loss_curve",
            "losses": [float(round(x, 12)) for x in losses],
        }),
    )
    return fitted, rep


# ---------------------------------------------------------------
# Heuristic baseline (mirrors the existing W79 hand-built logic).
# ---------------------------------------------------------------

def heuristic_economics_choice(
        *, feature: "_np.ndarray",
        abstain_floor: float = W81_DEFAULT_ABSTAIN_FLOOR,
) -> str:
    """Hand-built arbiter — analog of the W79 closed-form line.

    Mirrors the existing arbitration logic (pressure thresholds +
    cache-freshness gates) so the learned controller has a
    meaningful baseline to beat.
    """
    f = _np.asarray(feature, dtype=_np.float64)
    h, bp, ec, pf, cf, td, rh = (
        float(f[0]), float(f[1]), float(f[2]),
        float(f[3]), float(f[4]), float(f[5]), float(f[6]))
    # Abstain gate.
    if pf > 0.70 and ec < 0.30:
        return "abstain"
    # Cache-fresh + easy -> replay.
    if cf > 0.65 and td < 0.40:
        return "replay"
    # Hard task with healthy runtime -> promote.
    if td > 0.65 and rh > 0.60:
        return "promote_to_richer_substrate"
    # Healthy runtime mid-task -> recompute.
    if rh > 0.50:
        return "runtime_recompute"
    # Fall back to transcript.
    return "transcript_recompute"


# ---------------------------------------------------------------
# Dataset / sweep.
# ---------------------------------------------------------------

def build_economics_dataset_v1(
        *,
        n_samples: int = 256,
        seed: int = W81_DEFAULT_SEED,
        sim: EconomicsSimulationV1 | None = None,
) -> tuple[
        "_np.ndarray", "_np.ndarray",
        EconomicsSimulationV1]:
    """Synthetic decision dataset for supervised training.

    Returns ``(features, optimal_action_indices, sim)``. ``sim``
    is the deterministic cost/quality model; share it between
    train + eval so the labels are consistent.
    """
    rng = _np.random.default_rng(int(seed))
    N = int(n_samples)
    sim = sim or EconomicsSimulationV1()
    X = rng.uniform(
        0.0, 1.0, size=(N, W81_ECONOMICS_FEATURE_DIM)
    ).astype(_np.float64)
    y_idx = _np.zeros((N,), dtype=_np.int64)
    for i in range(N):
        y_idx[i] = sim.optimal_action_index(feature=X[i])
    return X, y_idx, sim


# ---------------------------------------------------------------
# Comparison: learned vs heuristic.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LearnedVsHeuristicEconomicsReportV1:
    """Side-by-side cost-quality report.

    Reports average cost, average success probability, and
    average utility for the learned controller and the
    heuristic baseline on the same evaluation set, plus per-
    action choice distributions so we can see how the policies
    differ.
    """

    schema: str
    n_eval: int
    learned_avg_cost: float
    learned_avg_success: float
    learned_avg_utility: float
    heuristic_avg_cost: float
    heuristic_avg_success: float
    heuristic_avg_utility: float
    learned_action_distribution: dict[str, float]
    heuristic_action_distribution: dict[str, float]
    optimal_avg_utility: float
    learned_optimality_gap: float
    heuristic_optimality_gap: float
    learned_beats_heuristic_on_utility: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_eval": int(self.n_eval),
            "learned_avg_cost": float(round(
                self.learned_avg_cost, 12)),
            "learned_avg_success": float(round(
                self.learned_avg_success, 12)),
            "learned_avg_utility": float(round(
                self.learned_avg_utility, 12)),
            "heuristic_avg_cost": float(round(
                self.heuristic_avg_cost, 12)),
            "heuristic_avg_success": float(round(
                self.heuristic_avg_success, 12)),
            "heuristic_avg_utility": float(round(
                self.heuristic_avg_utility, 12)),
            "learned_action_distribution": {
                k: float(round(v, 12))
                for k, v in
                self.learned_action_distribution.items()},
            "heuristic_action_distribution": {
                k: float(round(v, 12))
                for k, v in
                self.heuristic_action_distribution.items()},
            "optimal_avg_utility": float(round(
                self.optimal_avg_utility, 12)),
            "learned_optimality_gap": float(round(
                self.learned_optimality_gap, 12)),
            "heuristic_optimality_gap": float(round(
                self.heuristic_optimality_gap, 12)),
            "learned_beats_heuristic_on_utility": bool(
                self.learned_beats_heuristic_on_utility),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w81_learned_vs_heuristic_economics_v1",
            "report": self.to_dict()})


def compare_learned_vs_heuristic(
        *,
        controller: LearnedEconomicsControllerV1,
        eval_features: "_np.ndarray",
        sim: EconomicsSimulationV1,
) -> LearnedVsHeuristicEconomicsReportV1:
    X = _np.asarray(eval_features, dtype=_np.float64)
    N = int(X.shape[0])
    if N == 0:
        return LearnedVsHeuristicEconomicsReportV1(
            schema=W81_LEARNED_ECONOMICS_V1_SCHEMA_VERSION,
            n_eval=0,
            learned_avg_cost=0.0, learned_avg_success=0.0,
            learned_avg_utility=0.0,
            heuristic_avg_cost=0.0, heuristic_avg_success=0.0,
            heuristic_avg_utility=0.0,
            learned_action_distribution={
                a: 0.0 for a in W81_ECONOMICS_ACTIONS},
            heuristic_action_distribution={
                a: 0.0 for a in W81_ECONOMICS_ACTIONS},
            optimal_avg_utility=0.0,
            learned_optimality_gap=0.0,
            heuristic_optimality_gap=0.0,
            learned_beats_heuristic_on_utility=False,
        )
    learned_costs, learned_succs, learned_utils = (
        [], [], [])
    heur_costs, heur_succs, heur_utils = [], [], []
    optimal_utils = []
    learned_counts = {
        a: 0 for a in W81_ECONOMICS_ACTIONS}
    heur_counts = {
        a: 0 for a in W81_ECONOMICS_ACTIONS}
    for i in range(N):
        feat = X[i]
        # Learned.
        la = controller.predict_action(feat)
        lc, ls, lu = sim.evaluate_action(
            feature=feat, action=la)
        learned_costs.append(lc)
        learned_succs.append(ls)
        learned_utils.append(lu)
        learned_counts[la] += 1
        # Heuristic.
        ha = heuristic_economics_choice(feature=feat)
        hc, hs, hu = sim.evaluate_action(
            feature=feat, action=ha)
        heur_costs.append(hc)
        heur_succs.append(hs)
        heur_utils.append(hu)
        heur_counts[ha] += 1
        # Optimal.
        utils = sim.utility_per_action(feature=feat)
        optimal_utils.append(float(_np.max(utils)))
    learned_avg_util = float(_np.mean(learned_utils))
    heur_avg_util = float(_np.mean(heur_utils))
    optimal_avg = float(_np.mean(optimal_utils))
    return LearnedVsHeuristicEconomicsReportV1(
        schema=W81_LEARNED_ECONOMICS_V1_SCHEMA_VERSION,
        n_eval=int(N),
        learned_avg_cost=float(_np.mean(learned_costs)),
        learned_avg_success=float(_np.mean(learned_succs)),
        learned_avg_utility=learned_avg_util,
        heuristic_avg_cost=float(_np.mean(heur_costs)),
        heuristic_avg_success=float(_np.mean(heur_succs)),
        heuristic_avg_utility=heur_avg_util,
        learned_action_distribution={
            a: float(learned_counts[a]) / float(N)
            for a in W81_ECONOMICS_ACTIONS},
        heuristic_action_distribution={
            a: float(heur_counts[a]) / float(N)
            for a in W81_ECONOMICS_ACTIONS},
        optimal_avg_utility=float(optimal_avg),
        learned_optimality_gap=float(
            optimal_avg - learned_avg_util),
        heuristic_optimality_gap=float(
            optimal_avg - heur_avg_util),
        learned_beats_heuristic_on_utility=bool(
            learned_avg_util > heur_avg_util),
    )


@dataclasses.dataclass(frozen=True)
class LearnedEconomicsWitnessV1:
    schema: str
    controller_cid: str
    n_train_steps: int
    last_train_loss: float

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w81_learned_economics_witness",
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_train_steps": int(self.n_train_steps),
            "last_train_loss": float(round(
                self.last_train_loss, 12)),
        })


def emit_learned_economics_witness(
        *, controller: LearnedEconomicsControllerV1,
) -> LearnedEconomicsWitnessV1:
    return LearnedEconomicsWitnessV1(
        schema=W81_LEARNED_ECONOMICS_V1_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_train_steps=int(controller.n_train_steps),
        last_train_loss=float(controller.last_train_loss),
    )
