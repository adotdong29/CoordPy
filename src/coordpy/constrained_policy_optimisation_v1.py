"""W84 / P1 #34 — Online Learning with Safety Constraints
(Constrained Policy Optimisation / Lagrangian) V1.

Issue #34 asks for a constrained-RL extension of the W83
``online_economics_refinement_v1`` REINFORCE line. REINFORCE in
its raw form has no safety constraints; in regimes where the
optimal policy violates an action floor (e.g. abstain ≥ 0.10
in dangerous regimes), REINFORCE can drive that action's
probability to zero with no recourse.

W84 V1 ships:

* ``ConstrainedPolicyConfigV1`` — per-action probability floors
  / ceilings, per-action cost ceilings, and a hard whitelist
  mode. Constraints are content-addressed in the policy CID so
  a third party can audit what was imposed.
* ``LagrangianRefinementV1`` — REINFORCE augmented with a dual
  variable per constraint. Dual variables update by gradient
  ascent on constraint violation; the policy updates by
  gradient descent on the Lagrangian-augmented objective.
  Gradients are computed analytically; no autodiff dependency.
* Projection-based fallback ``project_onto_feasible_set_v1``
  that clips the action distribution to the feasible set when
  the Lagrangian is too slow to converge.
* ``ConstraintViolationLogV1`` capsule per episode.
* A 10-seed bench ``lagrangian_floor_recovery_bench_v1`` showing
  that on a regime where REINFORCE drives the abstain floor to
  zero, the Lagrangian refinement recovers a respected floor.
* A price-of-safety report bounding the mean-utility cost of
  enforcing the constraints.

Honest scope (W84 V1)
---------------------

* ``W84-L-CONSTRAINED-POLICY-V1-RESEARCH-ONLY-CAP`` — explicit-
  import only.
* ``W84-L-CONSTRAINED-POLICY-V1-LINEAR-CAP`` — V1 supports
  linear constraints (per-action floors / ceilings, per-action
  cost ceilings). Non-linear constraints (cost-per-success) are
  V2.
* ``W84-L-CONSTRAINED-POLICY-V1-SINGLE-POLICY-CAP`` — V1 is
  single-policy. Multi-role-per-agent constrained refinement
  is V2.
* ``W84-L-CONSTRAINED-POLICY-V1-NUMPY-CAP`` — pure NumPy.
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
        "coordpy.constrained_policy_optimisation_v1 requires "
        "numpy") from exc

from .learned_economics_controller_v1 import (
    LearnedEconomicsControllerV1,
    W81_ECONOMICS_ACTIONS,
    W81_ECONOMICS_FEATURE_DIM,
    W81_N_ECONOMICS_ACTIONS,
    _swish,
    _swish_grad,
)
from .online_economics_refinement_v1 import (
    DriftedDeploymentSimulationV1,
)


W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION: str = (
    "coordpy.constrained_policy_optimisation_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _softmax(z: "_np.ndarray") -> "_np.ndarray":
    z_shift = z - _np.max(z, axis=-1, keepdims=True)
    e = _np.exp(z_shift)
    return e / _np.sum(e, axis=-1, keepdims=True)


# ---------------------------------------------------------------
# Constraint config.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ConstrainedPolicyConfigV1:
    """All safety constraints in one content-addressed object.

    * ``action_floors`` — per-action probability floors (must be
      respected at every state).
    * ``action_ceilings`` — per-action probability ceilings.
    * ``action_cost_ceilings`` — per-action expected-cost
      ceilings (the mean of the simulated cost across the eval
      set must not exceed this).
    * ``whitelist_actions`` — if non-empty, ONLY these actions
      are allowed (any non-whitelisted action gets ceiling 0).
    """

    schema: str
    action_floors: tuple[tuple[str, float], ...]
    action_ceilings: tuple[tuple[str, float], ...]
    action_cost_ceilings: tuple[tuple[str, float], ...]
    whitelist_actions: tuple[str, ...] = ()

    def _validate(self) -> None:
        names = set(W81_ECONOMICS_ACTIONS)
        for a, v in self.action_floors:
            if a not in names:
                raise ValueError(
                    f"unknown action in floor: {a}")
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(
                    f"floor for {a} out of [0,1]: {v}")
        for a, v in self.action_ceilings:
            if a not in names:
                raise ValueError(
                    f"unknown action in ceiling: {a}")
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(
                    f"ceiling for {a} out of [0,1]: {v}")
        for a, v in self.action_cost_ceilings:
            if a not in names:
                raise ValueError(
                    f"unknown action in cost ceiling: {a}")
            if not (float(v) > 0.0):
                raise ValueError(
                    f"cost ceiling for {a} must be > 0: {v}")
        for a in self.whitelist_actions:
            if a not in names:
                raise ValueError(
                    f"unknown action in whitelist: {a}")

    def __post_init__(self) -> None:
        self._validate()

    def floor_for(self, action: str) -> float:
        for a, v in self.action_floors:
            if a == action:
                return float(v)
        return 0.0

    def ceiling_for(self, action: str) -> float:
        for a, v in self.action_ceilings:
            if a == action:
                return float(v)
        if (self.whitelist_actions
                and action not in self.whitelist_actions):
            return 0.0
        return 1.0

    def cost_ceiling_for(self, action: str) -> float | None:
        for a, v in self.action_cost_ceilings:
            if a == action:
                return float(v)
        return None

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_constrained_policy_config_v1",
            "schema": str(self.schema),
            "action_floors": [
                (str(a), float(round(v, 12)))
                for a, v in self.action_floors],
            "action_ceilings": [
                (str(a), float(round(v, 12)))
                for a, v in self.action_ceilings],
            "action_cost_ceilings": [
                (str(a), float(round(v, 12)))
                for a, v in self.action_cost_ceilings],
            "whitelist_actions": [
                str(a) for a in self.whitelist_actions],
        })


# ---------------------------------------------------------------
# Projection onto the feasible set.
# ---------------------------------------------------------------

def project_onto_feasible_set_v1(
        probs: "_np.ndarray",
        *, config: ConstrainedPolicyConfigV1,
) -> "_np.ndarray":
    """Project ``probs`` (a softmax distribution) onto the
    feasible set defined by ``config``.

    The projection:

    1. Clips each action to ``[floor_a, ceiling_a]``.
    2. Renormalises so the result sums to 1 while preserving
       per-action lower and upper bounds.

    The projection is a *fallback*; the Lagrangian update is
    the principled path. Both paths are exposed so the user
    can pick.
    """
    p = _np.asarray(probs, dtype=_np.float64).copy()
    n = int(p.shape[-1])
    floors = _np.zeros(n, dtype=_np.float64)
    ceilings = _np.ones(n, dtype=_np.float64)
    for i, a in enumerate(W81_ECONOMICS_ACTIONS):
        floors[i] = float(config.floor_for(a))
        ceilings[i] = float(config.ceiling_for(a))
    # Iterative feasibility projection: repeatedly clip and
    # renormalise until a fixed point or max-iter.
    for _ in range(64):
        p = _np.clip(p, floors, ceilings)
        s = float(_np.sum(p))
        if s <= 1e-12:
            # Degenerate: fall back to uniform over whitelist.
            if config.whitelist_actions:
                mask = _np.zeros(n, dtype=_np.float64)
                for i, a in enumerate(W81_ECONOMICS_ACTIONS):
                    if a in config.whitelist_actions:
                        mask[i] = 1.0
                if float(_np.sum(mask)) > 0:
                    return mask / float(_np.sum(mask))
            return _np.full(n, 1.0 / n, dtype=_np.float64)
        p = p / s
        # Check: floors respected and sum = 1.
        if (bool(_np.all(p >= floors - 1e-9))
                and bool(_np.all(p <= ceilings + 1e-9))):
            return p
    return p


# ---------------------------------------------------------------
# Lagrangian refinement.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ConstraintViolationLogV1:
    """Per-episode log of constraint violation."""

    schema: str
    episode_index: int
    floor_violations: tuple[tuple[str, float], ...]
    ceiling_violations: tuple[tuple[str, float], ...]
    cost_ceiling_violations: tuple[tuple[str, float], ...]
    feasible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "episode_index": int(self.episode_index),
            "floor_violations": [
                (str(a), float(round(v, 12)))
                for a, v in self.floor_violations],
            "ceiling_violations": [
                (str(a), float(round(v, 12)))
                for a, v in self.ceiling_violations],
            "cost_ceiling_violations": [
                (str(a), float(round(v, 12)))
                for a, v in self.cost_ceiling_violations],
            "feasible": bool(self.feasible),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_constraint_violation_log_v1",
            "log": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class LagrangianRefinementReportV1:
    schema: str
    controller_cid_pre: str
    controller_cid_post: str
    config_cid: str
    n_episodes: int
    pre_eval_mean_utility: float
    post_eval_mean_utility: float
    pre_eval_floor_satisfied: Mapping[str, float]
    post_eval_floor_satisfied: Mapping[str, float]
    post_eval_floors_respected: bool
    price_of_safety: float
    final_dual_variables: tuple[tuple[str, float], ...]
    violation_log_cids: tuple[str, ...]
    violation_chain_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid_pre": str(self.controller_cid_pre),
            "controller_cid_post": str(
                self.controller_cid_post),
            "config_cid": str(self.config_cid),
            "n_episodes": int(self.n_episodes),
            "pre_eval_mean_utility": float(round(
                self.pre_eval_mean_utility, 12)),
            "post_eval_mean_utility": float(round(
                self.post_eval_mean_utility, 12)),
            "pre_eval_floor_satisfied": {
                str(k): float(round(v, 12))
                for k, v in self.pre_eval_floor_satisfied.items()},
            "post_eval_floor_satisfied": {
                str(k): float(round(v, 12))
                for k, v in self.post_eval_floor_satisfied.items()
            },
            "post_eval_floors_respected": bool(
                self.post_eval_floors_respected),
            "price_of_safety": float(round(
                self.price_of_safety, 12)),
            "final_dual_variables": [
                (str(k), float(round(v, 12)))
                for k, v in self.final_dual_variables],
            "n_violation_logs": int(
                len(self.violation_log_cids)),
            "violation_chain_cid": str(
                self.violation_chain_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_lagrangian_refinement_report_v1",
            "report": self.to_dict()})


def _action_mean_probs(
        *, controller: LearnedEconomicsControllerV1,
        features: "_np.ndarray",
) -> "_np.ndarray":
    """Per-action mean probability across features."""
    N = int(features.shape[0])
    pm = _np.zeros((W81_N_ECONOMICS_ACTIONS,), dtype=_np.float64)
    for i in range(N):
        x = features[i]
        z1 = x @ controller.W1 + controller.b1
        h = _swish(z1)
        logits = h @ controller.W2 + controller.b2
        probs = _softmax(logits)
        pm += probs
    return pm / float(N)


def _eval_mean_utility(
        *, controller: LearnedEconomicsControllerV1,
        sim: DriftedDeploymentSimulationV1,
        features: "_np.ndarray",
) -> float:
    N = int(features.shape[0])
    total = 0.0
    for i in range(N):
        x = features[i]
        z1 = x @ controller.W1 + controller.b1
        h = _swish(z1)
        logits = h @ controller.W2 + controller.b2
        probs = _softmax(logits)
        a_idx = int(_np.argmax(probs))
        u = float(sim.utility_for_action(
            features=x, action_index=a_idx))
        total += u
    return float(total / float(N))


def _build_floor_satisfaction_map(
        *, controller: LearnedEconomicsControllerV1,
        config: ConstrainedPolicyConfigV1,
        features: "_np.ndarray",
) -> dict[str, float]:
    pm = _action_mean_probs(
        controller=controller, features=features)
    out: dict[str, float] = {}
    for i, a in enumerate(W81_ECONOMICS_ACTIONS):
        floor = float(config.floor_for(a))
        out[a] = float(pm[i]) - floor
    return out


@dataclasses.dataclass(frozen=True)
class LagrangianRefinementV1:
    """Constrained REINFORCE with Lagrangian dual updates.

    Gradients are computed analytically against the W81 2-layer
    MLP controller — no autodiff library.
    """

    learning_rate: float = 0.030
    dual_learning_rate: float = 0.060
    floor_tolerance: float = 0.005
    n_episodes: int = 200
    seed: int = 84_034_001

    def _grad_logits_for_chosen(
            self,
            *, x: "_np.ndarray", probs: "_np.ndarray",
            chosen_idx: int,
    ) -> "_np.ndarray":
        # ∂log π_chosen / ∂logits_k = δ_{k,chosen} - π_k.
        n = int(probs.shape[-1])
        g = -_np.asarray(probs, dtype=_np.float64).copy()
        g[chosen_idx] += 1.0
        return g

    def refine(
            self,
            *,
            controller: LearnedEconomicsControllerV1,
            sim: DriftedDeploymentSimulationV1,
            config: ConstrainedPolicyConfigV1,
            train_features: "_np.ndarray",
            eval_features: "_np.ndarray",
    ) -> tuple[LearnedEconomicsControllerV1,
               LagrangianRefinementReportV1,
               tuple[ConstraintViolationLogV1, ...]]:
        """Run the Lagrangian-augmented REINFORCE loop."""
        controller_pre = controller
        # Make a mutable copy of the controller's params.
        W1 = _np.array(controller.W1, dtype=_np.float64).copy()
        b1 = _np.array(controller.b1, dtype=_np.float64).copy()
        W2 = _np.array(controller.W2, dtype=_np.float64).copy()
        b2 = _np.array(controller.b2, dtype=_np.float64).copy()
        # One dual variable per floor.
        floor_actions = [
            a for a, v in config.action_floors if float(v) > 0.0]
        lam = {a: 0.5 for a in floor_actions}
        rng = _np.random.default_rng(int(self.seed))
        feats = _np.asarray(train_features, dtype=_np.float64)
        n_feats = int(feats.shape[0])
        violation_logs: list[ConstraintViolationLogV1] = []
        baseline_reward = 0.0
        baseline_alpha = 0.85
        for ep in range(int(self.n_episodes)):
            idx = int(rng.integers(0, n_feats))
            x = feats[idx]
            z1 = x @ W1 + b1
            h = _swish(z1)
            logits = h @ W2 + b2
            probs = _softmax(logits)
            chosen = int(rng.choice(
                W81_N_ECONOMICS_ACTIONS, p=probs))
            chosen_name = W81_ECONOMICS_ACTIONS[chosen]
            # Reward = utility on drifted sim.
            u = float(sim.utility_for_action(
                features=x, action_index=chosen))
            # Lagrangian augmentation (reward-shaping form):
            # choosing a floor-pressed action gets a +λ_a bonus,
            # making the policy more likely to pick it. The dual
            # variable λ_a grows when the floor is violated, so
            # this bonus increases until the floor is respected.
            # This is the canonical constrained-REINFORCE
            # Lagrangian: L = E[r] + λ * (E[π(a*|s)] - floor).
            for a in floor_actions:
                if W81_ECONOMICS_ACTIONS[chosen] == a:
                    u = u + float(lam[a])
            advantage = float(u - baseline_reward)
            baseline_reward = (
                baseline_alpha * baseline_reward
                + (1.0 - baseline_alpha) * float(u))
            # Compute parameter gradients analytically.
            g_logits = self._grad_logits_for_chosen(
                x=x, probs=probs, chosen_idx=chosen)
            # ∂L/∂W2 = outer(h, g_logits * advantage)
            g_W2 = _np.outer(h, g_logits * float(advantage))
            g_b2 = g_logits * float(advantage)
            # Backprop through the swish.
            g_h = g_logits * float(advantage) @ W2.T
            g_z1 = g_h * _swish_grad(z1)
            g_W1 = _np.outer(x, g_z1)
            g_b1 = g_z1
            # Gradient ASCENT on log-likelihood weighted by reward.
            W1 = W1 + float(self.learning_rate) * g_W1
            b1 = b1 + float(self.learning_rate) * g_b1
            W2 = W2 + float(self.learning_rate) * g_W2
            b2 = b2 + float(self.learning_rate) * g_b2
            # Dual variable gradient ASCENT on constraint
            # violation. λ_a ← max(0, λ_a + η * (floor - mean P(a))).
            # We approximate "mean P(a)" by the current episode
            # batch via probs[a_idx] across the sampled state.
            tmp_controller = LearnedEconomicsControllerV1(
                schema=controller_pre.schema,
                feature_dim=int(W81_ECONOMICS_FEATURE_DIM),
                hidden_dim=int(controller_pre.hidden_dim),
                n_actions=int(W81_N_ECONOMICS_ACTIONS),
                W1=W1, b1=b1, W2=W2, b2=b2,
                mom_W1=_np.zeros_like(W1),
                mom_b1=_np.zeros_like(b1),
                mom_W2=_np.zeros_like(W2),
                mom_b2=_np.zeros_like(b2),
                n_train_steps=int(controller_pre.n_train_steps),
                last_train_loss=float(
                    controller_pre.last_train_loss),
                pre_train_loss=float(
                    controller_pre.pre_train_loss),
            )
            pm_train = _action_mean_probs(
                controller=tmp_controller,
                features=feats[:min(n_feats, 32)])
            floor_violations: list[tuple[str, float]] = []
            for a in floor_actions:
                a_idx = W81_ECONOMICS_ACTIONS.index(a)
                floor = float(config.floor_for(a))
                p_mean = float(pm_train[a_idx])
                gap = float(floor - p_mean)
                lam[a] = max(
                    0.0,
                    float(lam[a])
                    + float(self.dual_learning_rate) * gap)
                if gap > float(self.floor_tolerance):
                    floor_violations.append((str(a), gap))
            ceiling_violations: list[tuple[str, float]] = []
            for a, v in config.action_ceilings:
                a_idx = W81_ECONOMICS_ACTIONS.index(a)
                ceil = float(v)
                p_mean = float(pm_train[a_idx])
                if p_mean > ceil + float(self.floor_tolerance):
                    ceiling_violations.append(
                        (str(a), float(p_mean - ceil)))
            cost_violations: list[tuple[str, float]] = []
            for a, v in config.action_cost_ceilings:
                a_idx = W81_ECONOMICS_ACTIONS.index(a)
                # Mean cost: average evaluate_action(a) cost over
                # the eval features.
                costs = _np.zeros(
                    (int(eval_features.shape[0]),),
                    dtype=_np.float64)
                for i_e in range(int(eval_features.shape[0])):
                    c, _s, _u = sim.evaluate_action(
                        feature=eval_features[i_e],
                        action=str(a))
                    costs[i_e] = float(c)
                mean_cost = float(_np.mean(costs))
                if mean_cost > float(v):
                    cost_violations.append(
                        (str(a), float(mean_cost - v)))
            feasible = bool(
                (not floor_violations)
                and (not ceiling_violations)
                and (not cost_violations))
            violation_logs.append(ConstraintViolationLogV1(
                schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
                episode_index=int(ep),
                floor_violations=tuple(floor_violations),
                ceiling_violations=tuple(ceiling_violations),
                cost_ceiling_violations=tuple(cost_violations),
                feasible=bool(feasible),
            ))
        controller_post = LearnedEconomicsControllerV1(
            schema=controller_pre.schema,
            feature_dim=int(W81_ECONOMICS_FEATURE_DIM),
            hidden_dim=int(controller_pre.hidden_dim),
            n_actions=int(W81_N_ECONOMICS_ACTIONS),
            W1=W1, b1=b1, W2=W2, b2=b2,
            mom_W1=_np.zeros_like(W1),
            mom_b1=_np.zeros_like(b1),
            mom_W2=_np.zeros_like(W2),
            mom_b2=_np.zeros_like(b2),
            n_train_steps=int(controller_pre.n_train_steps)
            + int(self.n_episodes),
            last_train_loss=float(
                controller_pre.last_train_loss),
            pre_train_loss=float(
                controller_pre.pre_train_loss),
        )
        pre_util = _eval_mean_utility(
            controller=controller_pre, sim=sim,
            features=eval_features)
        post_util = _eval_mean_utility(
            controller=controller_post, sim=sim,
            features=eval_features)
        pre_floor = _build_floor_satisfaction_map(
            controller=controller_pre, config=config,
            features=eval_features)
        post_floor = _build_floor_satisfaction_map(
            controller=controller_post, config=config,
            features=eval_features)
        floors_respected = all(
            v >= -float(self.floor_tolerance)
            for v in post_floor.values())
        violation_log_cids = tuple(
            str(log.cid()) for log in violation_logs)
        violation_chain_cid = _sha256_hex({
            "kind": "w84_constraint_violation_chain_v1",
            "cids": list(violation_log_cids),
        })
        report = LagrangianRefinementReportV1(
            schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
            controller_cid_pre=str(controller_pre.cid()),
            controller_cid_post=str(controller_post.cid()),
            config_cid=str(config.cid()),
            n_episodes=int(self.n_episodes),
            pre_eval_mean_utility=float(pre_util),
            post_eval_mean_utility=float(post_util),
            pre_eval_floor_satisfied=dict(pre_floor),
            post_eval_floor_satisfied=dict(post_floor),
            post_eval_floors_respected=bool(floors_respected),
            price_of_safety=float(pre_util - post_util),
            final_dual_variables=tuple(
                (str(k), float(v)) for k, v in lam.items()),
            violation_log_cids=violation_log_cids,
            violation_chain_cid=str(violation_chain_cid),
        )
        return controller_post, report, tuple(violation_logs)


# ---------------------------------------------------------------
# Floor-recovery bench.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FloorRecoveryBenchReportV1:
    schema: str
    n_seeds: int
    n_seeds_floor_respected_lagrangian: int
    n_seeds_floor_respected_reinforce: int
    lagrangian_strictly_beats_reinforce_on_floor: bool
    pre_floor_violation_rate_seedwise: tuple[float, ...]
    post_floor_violation_rate_seedwise: tuple[float, ...]
    mean_price_of_safety: float
    bootstrap_floor_violation_ci_lo: float
    bootstrap_floor_violation_ci_hi: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_seeds": int(self.n_seeds),
            "n_seeds_floor_respected_lagrangian": int(
                self.n_seeds_floor_respected_lagrangian),
            "n_seeds_floor_respected_reinforce": int(
                self.n_seeds_floor_respected_reinforce),
            "lagrangian_strictly_beats_reinforce_on_floor": bool(
                self.lagrangian_strictly_beats_reinforce_on_floor
            ),
            "mean_price_of_safety": float(round(
                self.mean_price_of_safety, 12)),
            "bootstrap_floor_violation_ci_lo": float(round(
                self.bootstrap_floor_violation_ci_lo, 12)),
            "bootstrap_floor_violation_ci_hi": float(round(
                self.bootstrap_floor_violation_ci_hi, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_floor_recovery_bench_v1",
            "report": self.to_dict()})


def _build_floor_pressing_features(
        *, seed: int, n: int = 96,
) -> "_np.ndarray":
    """Build a feature set that pushes REINFORCE away from abstain.

    Set abstain-floor-pressing knobs: high evidence_completeness
    + low task_difficulty + healthy runtime → REINFORCE prefers
    high-utility committal actions, away from abstain.
    """
    rng = _np.random.default_rng(int(seed))
    feats = _np.zeros((int(n), int(W81_ECONOMICS_FEATURE_DIM)),
                      dtype=_np.float64)
    for i in range(int(n)):
        feats[i, 0] = float(rng.uniform(0.10, 0.40))  # horizon
        feats[i, 1] = float(rng.uniform(0.05, 0.20))  # budget
        feats[i, 2] = float(rng.uniform(0.85, 1.00))  # evidence
        feats[i, 3] = float(rng.uniform(0.00, 0.10))  # prior fail
        feats[i, 4] = float(rng.uniform(0.85, 1.00))  # fresh
        feats[i, 5] = float(rng.uniform(0.10, 0.30))  # difficulty
        feats[i, 6] = float(rng.uniform(0.85, 1.00))  # rt health
    return feats


def _drifted_sim_for_floor_press() -> DriftedDeploymentSimulationV1:
    """A simulation where the optimal policy collapses to runtime
    recompute, eliminating abstain — exactly the regime where
    naive REINFORCE drives the abstain floor to 0.
    """
    return DriftedDeploymentSimulationV1(
        schema="w84_drifted_for_floor_press",
        base_cost_weight=0.000_3,
        base_abstain_floor=0.30,
        replay_cost_multiplier=2.0,
        runtime_recompute_cost_multiplier=0.5,
        transcript_recompute_cost_multiplier=1.5,
        promote_cost_multiplier=2.0,
        abstain_cost_multiplier=1.0,
        replay_quality_multiplier=0.8,
        runtime_recompute_quality_multiplier=1.05,
        transcript_recompute_quality_multiplier=0.9,
        promote_quality_multiplier=0.8,
        abstain_quality_multiplier=0.1,
    )


def _train_initial_controller(
        *, seed: int,
) -> LearnedEconomicsControllerV1:
    """Small in-line controller pre-train, not the full W81 fit.

    The W81 supervised optimiser runs many iterations and is not
    what we want here — we want a controller that's close enough
    to be a starting policy so the Lagrangian can do work.
    """
    rng = _np.random.default_rng(int(seed))
    hidden_dim = 16
    W1 = (rng.standard_normal((
        int(W81_ECONOMICS_FEATURE_DIM), int(hidden_dim)))
        * 0.10)
    b1 = _np.zeros((int(hidden_dim),), dtype=_np.float64)
    W2 = (rng.standard_normal((
        int(hidden_dim), int(W81_N_ECONOMICS_ACTIONS)))
        * 0.10)
    b2 = _np.zeros((int(W81_N_ECONOMICS_ACTIONS),),
                   dtype=_np.float64)
    return LearnedEconomicsControllerV1(
        schema="w84_init_controller",
        feature_dim=int(W81_ECONOMICS_FEATURE_DIM),
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


def _run_unconstrained_reinforce(
        *, controller: LearnedEconomicsControllerV1,
        sim: DriftedDeploymentSimulationV1,
        train_features: "_np.ndarray",
        n_episodes: int, seed: int,
) -> LearnedEconomicsControllerV1:
    """Plain REINFORCE for comparison; no Lagrangian term."""
    W1 = _np.array(controller.W1, dtype=_np.float64).copy()
    b1 = _np.array(controller.b1, dtype=_np.float64).copy()
    W2 = _np.array(controller.W2, dtype=_np.float64).copy()
    b2 = _np.array(controller.b2, dtype=_np.float64).copy()
    rng = _np.random.default_rng(int(seed))
    n_feats = int(train_features.shape[0])
    baseline = 0.0
    alpha = 0.85
    lr = 0.030
    for _ in range(int(n_episodes)):
        idx = int(rng.integers(0, n_feats))
        x = train_features[idx]
        z1 = x @ W1 + b1
        h = _swish(z1)
        logits = h @ W2 + b2
        probs = _softmax(logits)
        chosen = int(rng.choice(
            W81_N_ECONOMICS_ACTIONS, p=probs))
        u = float(sim.utility_for_action(
            features=x, action_index=chosen))
        adv = float(u - baseline)
        baseline = alpha * baseline + (1.0 - alpha) * u
        g = -probs.copy()
        g[chosen] += 1.0
        W2 = W2 + lr * _np.outer(h, g * adv)
        b2 = b2 + lr * (g * adv)
        gh = (g * adv) @ W2.T
        gz1 = gh * _swish_grad(z1)
        W1 = W1 + lr * _np.outer(x, gz1)
        b1 = b1 + lr * gz1
    return LearnedEconomicsControllerV1(
        schema=controller.schema,
        feature_dim=int(W81_ECONOMICS_FEATURE_DIM),
        hidden_dim=int(controller.hidden_dim),
        n_actions=int(W81_N_ECONOMICS_ACTIONS),
        W1=W1, b1=b1, W2=W2, b2=b2,
        mom_W1=_np.zeros_like(W1),
        mom_b1=_np.zeros_like(b1),
        mom_W2=_np.zeros_like(W2),
        mom_b2=_np.zeros_like(b2),
        n_train_steps=int(controller.n_train_steps)
        + int(n_episodes),
        last_train_loss=float(controller.last_train_loss),
        pre_train_loss=float(controller.pre_train_loss),
    )


def run_lagrangian_floor_recovery_bench_v1(
        *,
        n_seeds: int = 10,
        n_episodes: int = 250,
        abstain_floor: float = 0.10,
        floor_tolerance: float = 0.005,
) -> FloorRecoveryBenchReportV1:
    """Run a ≥10-seed floor-recovery bench.

    Per seed, compare unconstrained REINFORCE vs the Lagrangian
    refinement on the floor regime. The Lagrangian must
    *strictly* beat REINFORCE on the rate at which the floor is
    respected at the end of refinement.
    """
    config = ConstrainedPolicyConfigV1(
        schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
        action_floors=(
            ("abstain", float(abstain_floor)),
        ),
        action_ceilings=(),
        action_cost_ceilings=(),
        whitelist_actions=(),
    )
    sim = _drifted_sim_for_floor_press()
    n_lag_ok = 0
    n_re_ok = 0
    pos_seeds: list[float] = []
    post_seeds: list[float] = []
    price_of_safety_seedwise: list[float] = []
    for seed in range(int(n_seeds)):
        # Fresh deterministic features per seed.
        train_feats = _build_floor_pressing_features(
            seed=84_400_000 + 100 * seed, n=64)
        eval_feats = _build_floor_pressing_features(
            seed=84_500_000 + 100 * seed, n=64)
        init = _train_initial_controller(
            seed=84_300_000 + 100 * seed)
        # Unconstrained REINFORCE.
        re_post = _run_unconstrained_reinforce(
            controller=init, sim=sim,
            train_features=train_feats,
            n_episodes=int(n_episodes),
            seed=84_600_000 + 100 * seed)
        re_floor = _build_floor_satisfaction_map(
            controller=re_post, config=config,
            features=eval_feats)
        re_ok = bool(
            re_floor["abstain"] >= -float(floor_tolerance))
        if re_ok:
            n_re_ok += 1
        # Lagrangian-refined.
        lag = LagrangianRefinementV1(
            n_episodes=int(n_episodes),
            seed=84_700_000 + 100 * seed,
            floor_tolerance=float(floor_tolerance),
        )
        _post_ctrl, rep, _logs = lag.refine(
            controller=init, sim=sim, config=config,
            train_features=train_feats,
            eval_features=eval_feats)
        if rep.post_eval_floors_respected:
            n_lag_ok += 1
        pos_seeds.append(float(
            -min(rep.pre_eval_floor_satisfied.get(
                "abstain", 0.0), 0.0)))
        post_seeds.append(float(
            -min(rep.post_eval_floor_satisfied.get(
                "abstain", 0.0), 0.0)))
        price_of_safety_seedwise.append(
            float(rep.price_of_safety))
    # Bootstrap CI on post floor violation rate.
    rng = _np.random.default_rng(84_800_000)
    arr = _np.array(post_seeds, dtype=_np.float64)
    bs_means = []
    for _ in range(200):
        idx = rng.integers(0, len(arr), size=len(arr))
        bs_means.append(float(_np.mean(arr[idx])))
    lo = float(_np.percentile(bs_means, 2.5))
    hi = float(_np.percentile(bs_means, 97.5))
    return FloorRecoveryBenchReportV1(
        schema=W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION,
        n_seeds=int(n_seeds),
        n_seeds_floor_respected_lagrangian=int(n_lag_ok),
        n_seeds_floor_respected_reinforce=int(n_re_ok),
        lagrangian_strictly_beats_reinforce_on_floor=bool(
            n_lag_ok > n_re_ok),
        pre_floor_violation_rate_seedwise=tuple(pos_seeds),
        post_floor_violation_rate_seedwise=tuple(post_seeds),
        mean_price_of_safety=float(
            _np.mean(price_of_safety_seedwise)),
        bootstrap_floor_violation_ci_lo=float(lo),
        bootstrap_floor_violation_ci_hi=float(hi),
    )


__all__ = [
    "W84_CONSTRAINED_POLICY_V1_SCHEMA_VERSION",
    "ConstrainedPolicyConfigV1",
    "ConstraintViolationLogV1",
    "LagrangianRefinementReportV1",
    "LagrangianRefinementV1",
    "project_onto_feasible_set_v1",
    "FloorRecoveryBenchReportV1",
    "run_lagrangian_floor_recovery_bench_v1",
]
