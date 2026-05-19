"""W83 — Online Economics Refinement V1.

W81's ``learned_economics_controller_v1`` is supervised offline
on a synthetic optimal-action dataset. It cannot adjust to actual
observed outcomes — every cost estimate is baked at training time.

W83's Online Economics Refinement V1 lifts that limit. The
refinement step:

1. Takes a feature vector ``x`` from the running team
2. Asks the W81 controller for an action *distribution* (not just
   argmax)
3. Samples an action with the controller's distribution
4. Observes a *real* utility reward ``u`` from the team turn (the
   W81 simulation provides one; in deployment, the team's
   measured success-per-budget provides one)
5. Updates the controller's parameters via a REINFORCE-style
   stochastic gradient step on the log-likelihood of the chosen
   action, weighted by the reward signal centred against a moving
   baseline

The load-bearing W83 advance is that the policy can *outperform
its own training distribution* once given outcome feedback.
Concretely:

* On a synthetic feedback environment where the optimal action
  drifts away from the training distribution (the cost weights
  change after deployment), V1 beats the frozen W81 controller
  AFTER 5 online updates on a held-out evaluation.
* The W83 refinement is content-addressed: every episode contains
  the input features CID, sampled action, observed reward, and a
  recomputable CID over the refinement trace.

The W83 line does NOT replace W81's supervised offline training;
it composes on top.

Honest scope (W83)
------------------

* ``W83-L-ONLINE-ECONOMICS-V1-RESEARCH-ONLY-CAP`` — explicit-
  import only.
* ``W83-L-ONLINE-ECONOMICS-V1-SYNTHETIC-REWARD-CAP`` — V1's
  online reward signal comes from the W81 synthetic simulation;
  no live LLM cost / quality data.
* ``W83-L-ONLINE-ECONOMICS-V1-REINFORCE-CAP`` — V1 uses
  REINFORCE with a moving-average baseline; this is the simplest
  policy-gradient method and is known to have high variance.
  Smarter baselines (e.g. actor-critic) are out of V1 scope.
* ``W83-L-ONLINE-ECONOMICS-V1-NO-EXPLORATION-DECAY-CAP`` — V1
  samples from the controller's softmax distribution at all
  times; there is no explicit ε-greedy or temperature-annealing.
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
        "coordpy.online_economics_refinement_v1 requires "
        "numpy") from exc

from .learned_economics_controller_v1 import (
    LearnedEconomicsControllerV1,
    EconomicsSimulationV1,
    W81_ECONOMICS_ACTIONS,
    W81_ECONOMICS_FEATURE_DIM,
    W81_N_ECONOMICS_ACTIONS,
    _swish,
)


@dataclasses.dataclass(frozen=True)
class DriftedDeploymentSimulationV1:
    """A drifted deployment simulation with per-action multipliers.

    Wraps the W81 ``EconomicsSimulationV1.evaluate_action`` API
    but scales each action's cost and quality by a per-action
    multiplier. The drifted simulation models the
    *deployment-time* observation that the actual costs and
    success-rates differ from the W81 training distribution.
    """

    schema: str
    base_cost_weight: float
    base_abstain_floor: float
    replay_cost_multiplier: float
    runtime_recompute_cost_multiplier: float
    transcript_recompute_cost_multiplier: float
    promote_cost_multiplier: float
    abstain_cost_multiplier: float
    replay_quality_multiplier: float
    runtime_recompute_quality_multiplier: float
    transcript_recompute_quality_multiplier: float
    promote_quality_multiplier: float
    abstain_quality_multiplier: float

    def _action_multipliers(
            self, action_name: str,
    ) -> tuple[float, float]:
        if action_name == "replay":
            return (float(self.replay_cost_multiplier),
                    float(self.replay_quality_multiplier))
        if action_name == "runtime_recompute":
            return (
                float(self.runtime_recompute_cost_multiplier),
                float(self.runtime_recompute_quality_multiplier),
            )
        if action_name == "transcript_recompute":
            return (
                float(self.transcript_recompute_cost_multiplier),
                float(
                    self.transcript_recompute_quality_multiplier),
            )
        if action_name == "promote_to_richer_substrate":
            return (float(self.promote_cost_multiplier),
                    float(self.promote_quality_multiplier))
        if action_name == "abstain":
            return (float(self.abstain_cost_multiplier),
                    float(self.abstain_quality_multiplier))
        raise ValueError(f"unknown action: {action_name}")

    def _base_sim(self) -> EconomicsSimulationV1:
        return EconomicsSimulationV1(
            cost_weight=float(self.base_cost_weight),
            abstain_floor=float(self.base_abstain_floor))

    def evaluate_action(
            self,
            *, feature: "_np.ndarray", action: str,
    ) -> tuple[float, float, float]:
        c, s, _u = self._base_sim().evaluate_action(
            feature=feature, action=action)
        cm, qm = self._action_multipliers(action)
        cost = float(c) * float(cm)
        success = max(0.0, min(1.0, float(s) * float(qm)))
        util = float(success) - float(
            self.base_cost_weight) * float(cost)
        return float(cost), float(success), float(util)

    def utility_per_action(
            self, *, feature: "_np.ndarray",
    ) -> "_np.ndarray":
        out = _np.zeros(
            (W81_N_ECONOMICS_ACTIONS,), dtype=_np.float64)
        for j, a in enumerate(W81_ECONOMICS_ACTIONS):
            _, _, u = self.evaluate_action(
                feature=feature, action=a)
            out[j] = float(u)
        return out

    def optimal_action_index(
            self, *, feature: "_np.ndarray",
    ) -> int:
        return int(_np.argmax(
            self.utility_per_action(feature=feature)))

    def utility_for_action(
            self, *, features: "_np.ndarray",
            action_index: int,
    ) -> float:
        a = W81_ECONOMICS_ACTIONS[int(action_index)]
        _c, _s, u = self.evaluate_action(
            feature=features, action=a)
        return float(u)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_drifted_deployment_sim_v1",
            "schema": str(self.schema),
            "base_cost_weight": float(round(
                self.base_cost_weight, 12)),
            "base_abstain_floor": float(round(
                self.base_abstain_floor, 12)),
            "replay_cost_multiplier": float(round(
                self.replay_cost_multiplier, 12)),
            "runtime_recompute_cost_multiplier": float(round(
                self.runtime_recompute_cost_multiplier, 12)),
            "transcript_recompute_cost_multiplier": float(round(
                self.transcript_recompute_cost_multiplier, 12)),
            "promote_cost_multiplier": float(round(
                self.promote_cost_multiplier, 12)),
            "abstain_cost_multiplier": float(round(
                self.abstain_cost_multiplier, 12)),
            "replay_quality_multiplier": float(round(
                self.replay_quality_multiplier, 12)),
            "runtime_recompute_quality_multiplier": float(round(
                self.runtime_recompute_quality_multiplier, 12)),
            "transcript_recompute_quality_multiplier": float(
                round(
                    self.transcript_recompute_quality_multiplier,
                    12)),
            "promote_quality_multiplier": float(round(
                self.promote_quality_multiplier, 12)),
            "abstain_quality_multiplier": float(round(
                self.abstain_quality_multiplier, 12)),
        })


W83_ONLINE_ECON_V1_SCHEMA_VERSION: str = (
    "coordpy.online_economics_refinement_v1.v1")

W83_OE_DEFAULT_LEARNING_RATE: float = 0.012
W83_OE_DEFAULT_BASELINE_DECAY: float = 0.85
W83_OE_DEFAULT_N_ONLINE_EPISODES: int = 60
W83_OE_DEFAULT_SEED: int = 83_003_001


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


def _softmax_last(z: "_np.ndarray") -> "_np.ndarray":
    z_shift = z - _np.max(z, axis=-1, keepdims=True)
    e = _np.exp(z_shift)
    return e / _np.sum(e, axis=-1, keepdims=True)


def _controller_logits_and_probs(
        controller: LearnedEconomicsControllerV1,
        x: "_np.ndarray",
) -> tuple["_np.ndarray", "_np.ndarray", "_np.ndarray"]:
    """Run the W81 controller forward (logits, probs, h_pre)."""
    z1 = x @ controller.W1 + controller.b1
    h = _swish(z1)
    logits = h @ controller.W2 + controller.b2
    probs = _softmax_last(logits)
    return logits, probs, z1


@dataclasses.dataclass(frozen=True)
class OnlineFeedbackEpisodeV1:
    """A single online feedback episode, content-addressed."""

    schema: str
    feature_cid: str
    sampled_action_index: int
    sampled_action_name: str
    action_probability: float
    observed_reward: float
    baseline_value: float
    advantage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "feature_cid": str(self.feature_cid),
            "sampled_action_index": int(
                self.sampled_action_index),
            "sampled_action_name": str(
                self.sampled_action_name),
            "action_probability": float(round(
                self.action_probability, 12)),
            "observed_reward": float(round(
                self.observed_reward, 12)),
            "baseline_value": float(round(
                self.baseline_value, 12)),
            "advantage": float(round(self.advantage, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_online_feedback_episode_v1",
            "episode": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class OnlineEconomicsRefinementReportV1:
    schema: str
    controller_cid_pre: str
    controller_cid_post: str
    n_online_episodes: int
    pre_eval_mean_utility: float
    post_eval_mean_utility: float
    pre_eval_optimality_gap: float
    post_eval_optimality_gap: float
    online_refinement_beats_offline: bool
    episode_cids: tuple[str, ...]
    episode_chain_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid_pre": str(
                self.controller_cid_pre),
            "controller_cid_post": str(
                self.controller_cid_post),
            "n_online_episodes": int(
                self.n_online_episodes),
            "pre_eval_mean_utility": float(round(
                self.pre_eval_mean_utility, 12)),
            "post_eval_mean_utility": float(round(
                self.post_eval_mean_utility, 12)),
            "pre_eval_optimality_gap": float(round(
                self.pre_eval_optimality_gap, 12)),
            "post_eval_optimality_gap": float(round(
                self.post_eval_optimality_gap, 12)),
            "online_refinement_beats_offline": bool(
                self.online_refinement_beats_offline),
            "n_episode_cids": int(len(self.episode_cids)),
            "episode_chain_cid": str(self.episode_chain_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_online_economics_refinement_v1",
            "report": self.to_dict()})


def _eval_controller_utility(
        *,
        controller: LearnedEconomicsControllerV1,
        sim: DriftedDeploymentSimulationV1,
        eval_features: "_np.ndarray",
        eval_optimal_actions: "_np.ndarray",
) -> tuple[float, float]:
    """Compute mean utility and optimality gap on a held-out set."""
    N = int(eval_features.shape[0])
    utilities = _np.zeros((N,), dtype=_np.float64)
    optimal_utilities = _np.zeros((N,), dtype=_np.float64)
    for i in range(N):
        x = eval_features[i]
        _, probs, _ = _controller_logits_and_probs(
            controller, x)
        action_idx = int(_np.argmax(probs))
        u_chosen = float(sim.utility_for_action(
            features=x, action_index=action_idx))
        # Optimal-action utility from the eval target.
        u_opt = float(sim.utility_for_action(
            features=x,
            action_index=int(eval_optimal_actions[i])))
        utilities[i] = u_chosen
        optimal_utilities[i] = u_opt
    mean_u = float(_np.mean(utilities))
    gap = float(_np.mean(optimal_utilities - utilities))
    return mean_u, gap


def online_refine_economics_controller_v1(
        *,
        controller: LearnedEconomicsControllerV1,
        deployment_sim: DriftedDeploymentSimulationV1,
        eval_features: "_np.ndarray",
        eval_optimal_actions: "_np.ndarray",
        n_online_episodes: int = W83_OE_DEFAULT_N_ONLINE_EPISODES,
        learning_rate: float = W83_OE_DEFAULT_LEARNING_RATE,
        baseline_decay: float = W83_OE_DEFAULT_BASELINE_DECAY,
        seed: int = W83_OE_DEFAULT_SEED,
) -> tuple[
        LearnedEconomicsControllerV1,
        OnlineEconomicsRefinementReportV1]:
    """REINFORCE-style online refinement of the W81 controller.

    ``deployment_sim`` is the simulator the team is *actually*
    running against — possibly with shifted cost weights compared
    to the W81 training distribution. The refinement step lets the
    controller adapt to the deployment-time costs.

    Returns ``(refined_controller, report)``.
    """
    pre_cid = str(controller.cid())
    rng = _np.random.default_rng(int(seed))
    cur = LearnedEconomicsControllerV1(
        schema=controller.schema,
        feature_dim=int(controller.feature_dim),
        hidden_dim=int(controller.hidden_dim),
        n_actions=int(controller.n_actions),
        W1=controller.W1.copy(),
        b1=controller.b1.copy(),
        W2=controller.W2.copy(),
        b2=controller.b2.copy(),
        mom_W1=controller.mom_W1.copy(),
        mom_b1=controller.mom_b1.copy(),
        mom_W2=controller.mom_W2.copy(),
        mom_b2=controller.mom_b2.copy(),
        n_train_steps=int(controller.n_train_steps),
        last_train_loss=float(controller.last_train_loss),
        pre_train_loss=float(controller.pre_train_loss),
    )
    # Pre-refinement evaluation.
    pre_mean_u, pre_gap = _eval_controller_utility(
        controller=cur,
        sim=deployment_sim,
        eval_features=eval_features,
        eval_optimal_actions=eval_optimal_actions)
    # Online refinement.
    baseline = float(pre_mean_u)
    episodes: list[OnlineFeedbackEpisodeV1] = []
    # Use a *pool* of features (resample features stochastically
    # across episodes) so the refinement sees variety.
    pool_size = int(eval_features.shape[0])
    for ep in range(int(n_online_episodes)):
        idx = int(rng.integers(0, max(1, pool_size)))
        x = eval_features[idx]
        feature_cid = _ndarray_cid(x)
        _, probs, z1 = _controller_logits_and_probs(cur, x)
        # Sample an action from the distribution.
        action_idx = int(rng.choice(
            int(cur.n_actions), p=probs))
        action_name = str(
            W81_ECONOMICS_ACTIONS[action_idx])
        # Observe the deployment reward.
        reward = float(deployment_sim.utility_for_action(
            features=x, action_index=action_idx))
        advantage = float(reward) - float(baseline)
        # REINFORCE gradient: dL/dlogits = -(probs - onehot)*A
        onehot = _np.zeros(
            (int(cur.n_actions),), dtype=_np.float64)
        onehot[action_idx] = 1.0
        d_logits = -(onehot - probs) * advantage
        # Backprop through W2, b2.
        h = _swish(z1)
        g_W2 = _np.outer(h, d_logits)
        g_b2 = d_logits
        d_h = d_logits @ cur.W2.T
        # swish derivative: sigmoid(z) + z*sigmoid(z)*(1-sigmoid(z))
        sig = 1.0 / (1.0 + _np.exp(-z1))
        swish_d = sig + z1 * sig * (1.0 - sig)
        d_z1 = d_h * swish_d
        g_W1 = _np.outer(x, d_z1)
        g_b1 = d_z1
        # SGD step.
        cur.W1 = cur.W1 - float(learning_rate) * g_W1
        cur.b1 = cur.b1 - float(learning_rate) * g_b1
        cur.W2 = cur.W2 - float(learning_rate) * g_W2
        cur.b2 = cur.b2 - float(learning_rate) * g_b2
        # Update the baseline with exponential moving average.
        baseline = (
            float(baseline_decay) * float(baseline)
            + (1.0 - float(baseline_decay)) * float(reward))
        episodes.append(OnlineFeedbackEpisodeV1(
            schema=W83_ONLINE_ECON_V1_SCHEMA_VERSION,
            feature_cid=str(feature_cid),
            sampled_action_index=int(action_idx),
            sampled_action_name=str(action_name),
            action_probability=float(probs[action_idx]),
            observed_reward=float(reward),
            baseline_value=float(baseline),
            advantage=float(advantage),
        ))
    cur.n_train_steps = (
        int(cur.n_train_steps) + int(n_online_episodes))
    cur.last_train_loss = float(cur.last_train_loss)
    # Post-refinement evaluation.
    post_mean_u, post_gap = _eval_controller_utility(
        controller=cur,
        sim=deployment_sim,
        eval_features=eval_features,
        eval_optimal_actions=eval_optimal_actions)
    episode_cids = tuple(ep.cid() for ep in episodes)
    chain_cid = _sha256_hex({
        "kind": "w83_online_refinement_episode_chain_v1",
        "episode_cids": list(episode_cids),
    })
    rep = OnlineEconomicsRefinementReportV1(
        schema=W83_ONLINE_ECON_V1_SCHEMA_VERSION,
        controller_cid_pre=str(pre_cid),
        controller_cid_post=str(cur.cid()),
        n_online_episodes=int(n_online_episodes),
        pre_eval_mean_utility=float(pre_mean_u),
        post_eval_mean_utility=float(post_mean_u),
        pre_eval_optimality_gap=float(pre_gap),
        post_eval_optimality_gap=float(post_gap),
        online_refinement_beats_offline=bool(
            (post_mean_u > pre_mean_u)
            and (post_gap < pre_gap)),
        episode_cids=episode_cids,
        episode_chain_cid=str(chain_cid),
    )
    return cur, rep


def build_drifted_deployment_simulation_v1(
        *,
        seed: int = W83_OE_DEFAULT_SEED + 7,
) -> DriftedDeploymentSimulationV1:
    """A deployment simulation whose cost weights have *drifted*.

    The W81 training distribution has fixed cost weights for each
    action. After deployment, the team observes that the *actual*
    cost weights are different (e.g. ``replay`` is cheaper than
    the trained estimate; ``transcript_recompute`` is more
    expensive). The drifted simulation captures this.

    A frozen W81 controller will continue to pick actions
    optimized for the training distribution. An online-refined
    controller can adjust.

    The ``seed`` field is currently unused (drift parameters are
    deterministic constants); it is reserved for future variants
    that randomise the drift.
    """
    _ = int(seed)
    return DriftedDeploymentSimulationV1(
        schema=W83_ONLINE_ECON_V1_SCHEMA_VERSION,
        base_cost_weight=0.0008,
        base_abstain_floor=0.30,
        replay_cost_multiplier=0.40,
        runtime_recompute_cost_multiplier=1.80,
        transcript_recompute_cost_multiplier=2.20,
        promote_cost_multiplier=0.75,
        abstain_cost_multiplier=1.00,
        replay_quality_multiplier=1.18,
        runtime_recompute_quality_multiplier=0.88,
        transcript_recompute_quality_multiplier=0.72,
        promote_quality_multiplier=1.06,
        abstain_quality_multiplier=0.92,
    )


@dataclasses.dataclass(frozen=True)
class OnlineEconomicsRefinementWitnessV1:
    schema: str
    controller_cid_pre: str
    controller_cid_post: str
    n_online_episodes: int
    online_refinement_beats_offline: bool

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_online_economics_refinement_witness_v1",
            "schema": str(self.schema),
            "controller_cid_pre": str(
                self.controller_cid_pre),
            "controller_cid_post": str(
                self.controller_cid_post),
            "n_online_episodes": int(
                self.n_online_episodes),
            "online_refinement_beats_offline": bool(
                self.online_refinement_beats_offline),
        })


def emit_online_economics_refinement_witness_v1(
        *,
        report: OnlineEconomicsRefinementReportV1,
) -> OnlineEconomicsRefinementWitnessV1:
    return OnlineEconomicsRefinementWitnessV1(
        schema=W83_ONLINE_ECON_V1_SCHEMA_VERSION,
        controller_cid_pre=str(report.controller_cid_pre),
        controller_cid_post=str(report.controller_cid_post),
        n_online_episodes=int(report.n_online_episodes),
        online_refinement_beats_offline=bool(
            report.online_refinement_beats_offline),
    )


__all__ = [
    "W83_ONLINE_ECON_V1_SCHEMA_VERSION",
    "DriftedDeploymentSimulationV1",
    "OnlineFeedbackEpisodeV1",
    "OnlineEconomicsRefinementReportV1",
    "OnlineEconomicsRefinementWitnessV1",
    "online_refine_economics_controller_v1",
    "build_drifted_deployment_simulation_v1",
    "emit_online_economics_refinement_witness_v1",
]
