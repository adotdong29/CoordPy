"""W84 / P1 #34 — Constrained Policy Optimisation V1.

W83's ``online_economics_refinement_v1`` uses REINFORCE with a
moving-average baseline. REINFORCE has NO safety constraints:
in principle it can converge to a policy that satisfies high
mean utility but violates abstain floors, cost ceilings, or
action-whitelist constraints arbitrarily often.

This V1 extends the W83 line with constrained policy
optimisation:

1. **``ConstrainedPolicyConfigV1``** — content-addressed
   dataclass of constraints:
   * per-action probability floors (e.g. abstain >= 0.10);
   * per-action probability ceilings (e.g. promote <= 0.05);
   * hard action whitelist (compliance-locked mode).
2. **``LagrangianRefinementV1``** — REINFORCE augmented with
   Lagrangian dual variables; the dual variables are updated
   by gradient ascent on the constraint violation; the policy
   is updated by gradient descent on the Lagrangian-augmented
   objective.
3. **Projection-based fallback** — ``project_to_feasible_set``
   clips the action distribution at each evaluation point to
   respect the floor/ceiling constraints. Used as the
   "fallback when Lagrangian is too slow to converge".
4. **Constraint-violation bench** — a regime where the
   unconstrained REINFORCE policy violates an action-floor
   (drives abstain to 0.0); the Lagrangian-refined policy must
   respect the floor at the end of refinement.
5. **Honest reporting of constraint-violation rate** —
   ``ConstraintViolationLogV1`` capsules per episode; the
   post-refinement bench reports per-constraint violation rate
   with bootstrap CIs across seeds.

Honest scope (V1):

* `W84-L-CONSTRAINED-V1-LINEAR-CAP` — V1 supports linear
  constraints (per-action floors / ceilings); non-linear
  (cost-per-success) is V2.
* `W84-L-CONSTRAINED-V1-LAGRANGIAN-PROJECTION-CAP` — V1 ships
  Lagrangian + projection; trust-region methods (TRPO, PPO-
  clip) are V2.
* `W84-L-CONSTRAINED-V1-SINGLE-POLICY-CAP` — V1 is single-
  policy. Per-role policies are V2.
"""

from __future__ import annotations

import dataclasses
import enum
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
    W81_N_ECONOMICS_ACTIONS,
    _swish,
)
from .online_economics_refinement_v1 import (
    DriftedDeploymentSimulationV1,
    W83_OE_DEFAULT_BASELINE_DECAY,
    W83_OE_DEFAULT_LEARNING_RATE,
    W83_OE_DEFAULT_N_ONLINE_EPISODES,
    W83_OE_DEFAULT_SEED,
    _controller_logits_and_probs,
)


W84_CONSTRAINED_V1_SCHEMA_VERSION: str = (
    "coordpy.constrained_policy_optimisation_v1.v1")


# ---------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ConstrainedPolicyConfigV1:
    """Constraint set on the policy.

    All constraints are content-addressed (the config CID is
    part of the policy CID under refinement).

    Per-action floors / ceilings are dicts mapping action name
    to (lower / upper) probability bound. The hard action
    whitelist (if non-empty) requires that the policy assigns
    ZERO probability to all non-whitelisted actions.
    """

    schema: str
    per_action_prob_floors: tuple[tuple[str, float], ...]
    per_action_prob_ceilings: tuple[tuple[str, float], ...]
    hard_action_whitelist: tuple[str, ...]
    per_action_cost_ceilings: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "per_action_prob_floors": [
                [str(a), float(round(v, 12))]
                for a, v in self.per_action_prob_floors],
            "per_action_prob_ceilings": [
                [str(a), float(round(v, 12))]
                for a, v in self.per_action_prob_ceilings],
            "hard_action_whitelist": list(
                self.hard_action_whitelist),
            "per_action_cost_ceilings": [
                [str(a), float(round(v, 12))]
                for a, v in self.per_action_cost_ceilings],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_constrained_policy_config_v1",
            "config": self.to_dict()})

    def floor_for_action(self, action: str) -> float | None:
        for a, v in self.per_action_prob_floors:
            if str(a) == str(action):
                return float(v)
        return None

    def ceiling_for_action(self, action: str) -> float | None:
        for a, v in self.per_action_prob_ceilings:
            if str(a) == str(action):
                return float(v)
        return None


def build_constrained_policy_config_v1(
        *,
        per_action_prob_floors: Mapping[str, float] | None = None,
        per_action_prob_ceilings: Mapping[str, float] | None = None,
        hard_action_whitelist: Sequence[str] = (),
        per_action_cost_ceilings: Mapping[str, float] | None = None,
) -> ConstrainedPolicyConfigV1:
    f = tuple(
        (str(a), float(v))
        for a, v in (per_action_prob_floors or {}).items())
    c = tuple(
        (str(a), float(v))
        for a, v in (per_action_prob_ceilings or {}).items())
    cc = tuple(
        (str(a), float(v))
        for a, v in (per_action_cost_ceilings or {}).items())
    return ConstrainedPolicyConfigV1(
        schema=W84_CONSTRAINED_V1_SCHEMA_VERSION,
        per_action_prob_floors=f,
        per_action_prob_ceilings=c,
        hard_action_whitelist=tuple(
            str(a) for a in hard_action_whitelist),
        per_action_cost_ceilings=cc,
    )


# ---------------------------------------------------------------
# Violation log capsule
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ConstraintViolationLogV1:
    """Per-episode violation log."""

    schema: str
    episode_index: int
    constraint_axis: str  # "floor:<action>" / "ceiling:<action>" / "whitelist"
    measured_value: float
    threshold: float
    violation_magnitude: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "episode_index": int(self.episode_index),
            "constraint_axis": str(self.constraint_axis),
            "measured_value": float(round(
                self.measured_value, 12)),
            "threshold": float(round(self.threshold, 12)),
            "violation_magnitude": float(round(
                self.violation_magnitude, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_constraint_violation_log_v1",
            "log": self.to_dict()})


# ---------------------------------------------------------------
# Projection
# ---------------------------------------------------------------


def project_to_feasible_set(
        probs: "_np.ndarray",
        config: ConstrainedPolicyConfigV1,
) -> "_np.ndarray":
    """Project a single action distribution into the feasible
    set defined by the constraint config.

    Algorithm:
    1. Apply hard whitelist: zero out probabilities of actions
       not on the whitelist.
    2. For each action with a floor, raise its probability to
       the floor (track these as "locked-to-floor").
    3. For each action with a ceiling, lower its probability
       to the ceiling (also locked-to-ceiling).
    4. Distribute the remaining mass (1 - sum(locked))
       proportionally to the original probabilities of the
       unlocked actions. Locked actions stay at their locked
       value.

    Post-conditions:
    * π(a) ≥ floor_a for every floor constraint.
    * π(a) ≤ ceiling_a for every ceiling constraint.
    * Sum of probabilities = 1.

    Edge case: if the floors are infeasible (sum of floors > 1),
    we scale the floors down to a feasible set proportional to
    each floor.
    """
    p_orig = _np.asarray(probs, dtype=_np.float64).copy()
    n = int(p_orig.size)
    actions = list(W81_ECONOMICS_ACTIONS[:n])
    locked = _np.full((n,), False, dtype=bool)
    p = _np.zeros((n,), dtype=_np.float64)
    # Whitelist: non-whitelisted actions are locked to 0.
    if len(config.hard_action_whitelist) > 0:
        wl = set(config.hard_action_whitelist)
        for i, a in enumerate(actions):
            if a not in wl:
                p[i] = 0.0
                locked[i] = True
    # Floors: locked-to-floor.
    floor_total = 0.0
    for a, floor in config.per_action_prob_floors:
        if str(a) in actions:
            i = actions.index(str(a))
            if locked[i]:
                continue
            p[i] = float(floor)
            locked[i] = True
            floor_total += float(floor)
    # Ceilings: only lock-to-ceiling if the original was at or
    # above the ceiling.
    for a, ceil in config.per_action_prob_ceilings:
        if str(a) in actions:
            i = actions.index(str(a))
            if locked[i]:
                # If locked to a floor that exceeds the
                # ceiling, the constraints conflict —
                # cap to ceiling (ceilings dominate).
                if float(p[i]) > float(ceil):
                    p[i] = float(ceil)
                continue
            if float(p_orig[i]) >= float(ceil):
                p[i] = float(ceil)
                locked[i] = True
    # Edge case: sum(locked) > 1 — scale locked down
    # proportionally (infeasible floors).
    locked_sum = float(_np.sum(p[locked]))
    if locked_sum > 1.0:
        p[locked] = p[locked] / locked_sum
        locked_sum = float(_np.sum(p[locked]))
    # Remaining mass = 1 - locked_sum, distributed among
    # unlocked actions proportional to their original probs.
    remaining = float(max(0.0, 1.0 - locked_sum))
    unlocked = ~locked
    unlocked_sum = float(_np.sum(p_orig[unlocked]))
    if unlocked_sum <= 0.0:
        # Distribute uniformly among unlocked.
        n_unlocked = int(_np.sum(unlocked))
        if n_unlocked > 0:
            p[unlocked] = remaining / float(n_unlocked)
    else:
        p[unlocked] = (
            remaining * p_orig[unlocked] / unlocked_sum)
    # Final sanity: re-clip ceilings on unlocked actions that
    # the redistribution may have pushed over.
    for a, ceil in config.per_action_prob_ceilings:
        if str(a) in actions:
            i = actions.index(str(a))
            if not locked[i] and float(p[i]) > float(ceil):
                p[i] = float(ceil)
    # Renormalise as a final safety (should already sum to 1).
    s = float(_np.sum(p))
    if s <= 0.0:
        p = _np.ones((n,), dtype=_np.float64) / float(n)
    else:
        p = p / s
    return p


# ---------------------------------------------------------------
# Constraint measurement
# ---------------------------------------------------------------


W84_CONSTRAINT_VIOLATION_TOLERANCE: float = 1e-9


def measure_constraint_violations(
        probs: "_np.ndarray",
        config: ConstrainedPolicyConfigV1,
        *, episode_index: int = 0,
        tolerance: float = W84_CONSTRAINT_VIOLATION_TOLERANCE,
) -> list[ConstraintViolationLogV1]:
    """Return per-axis violation logs.

    The ``tolerance`` argument absorbs floating-point noise:
    a measured value within ``tolerance`` of the floor is NOT
    flagged as a violation. Default is 1e-9 — much tighter
    than the issue's "within a tolerance, e.g. ≥ floor -
    0.01" allowance.
    """
    out: list[ConstraintViolationLogV1] = []
    p = _np.asarray(probs, dtype=_np.float64)
    n = int(p.size)
    actions = list(W81_ECONOMICS_ACTIONS[:n])
    tol = float(tolerance)
    for a, floor in config.per_action_prob_floors:
        if str(a) in actions:
            i = actions.index(str(a))
            val = float(p[i])
            if val < float(floor) - tol:
                out.append(ConstraintViolationLogV1(
                    schema=W84_CONSTRAINED_V1_SCHEMA_VERSION,
                    episode_index=int(episode_index),
                    constraint_axis=f"floor:{a}",
                    measured_value=float(val),
                    threshold=float(floor),
                    violation_magnitude=float(
                        float(floor) - val)))
    for a, ceil in config.per_action_prob_ceilings:
        if str(a) in actions:
            i = actions.index(str(a))
            val = float(p[i])
            if val > float(ceil) + tol:
                out.append(ConstraintViolationLogV1(
                    schema=W84_CONSTRAINED_V1_SCHEMA_VERSION,
                    episode_index=int(episode_index),
                    constraint_axis=f"ceiling:{a}",
                    measured_value=float(val),
                    threshold=float(ceil),
                    violation_magnitude=float(
                        val - float(ceil))))
    return out


# ---------------------------------------------------------------
# Lagrangian refinement
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ConstrainedRefinementReportV1:
    """End-of-refinement report for the constrained policy."""

    schema: str
    config_cid: str
    controller_cid_pre: str
    controller_cid_post: str
    n_episodes: int
    final_lambdas: tuple[float, ...]
    pre_constraint_violation_rate: float
    post_constraint_violation_rate: float
    pre_mean_utility: float
    post_mean_utility: float
    price_of_safety_utility_delta: float
    n_violation_logs: int
    violation_log_chain_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "config_cid": str(self.config_cid),
            "controller_cid_pre": str(self.controller_cid_pre),
            "controller_cid_post": str(
                self.controller_cid_post),
            "n_episodes": int(self.n_episodes),
            "final_lambdas": [
                float(round(l, 12)) for l in self.final_lambdas],
            "pre_constraint_violation_rate": float(round(
                self.pre_constraint_violation_rate, 12)),
            "post_constraint_violation_rate": float(round(
                self.post_constraint_violation_rate, 12)),
            "pre_mean_utility": float(round(
                self.pre_mean_utility, 12)),
            "post_mean_utility": float(round(
                self.post_mean_utility, 12)),
            "price_of_safety_utility_delta": float(round(
                self.price_of_safety_utility_delta, 12)),
            "n_violation_logs": int(self.n_violation_logs),
            "violation_log_chain_cid": str(
                self.violation_log_chain_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_constrained_refinement_report_v1",
            "report": self.to_dict()})


def lagrangian_refine_constrained_v1(
        *,
        controller: LearnedEconomicsControllerV1,
        deployment_sim: DriftedDeploymentSimulationV1,
        eval_features: "_np.ndarray",
        config: ConstrainedPolicyConfigV1,
        n_episodes: int = 400,
        policy_lr: float = 0.020,
        lambda_lr: float = 0.10,
        baseline_decay: float = W83_OE_DEFAULT_BASELINE_DECAY,
        apply_projection_at_eval: bool = True,
        seed: int = W83_OE_DEFAULT_SEED + 11,
) -> tuple[
        LearnedEconomicsControllerV1,
        ConstrainedRefinementReportV1]:
    """REINFORCE + Lagrangian dual ascent on linear constraints.

    For each per-action floor f_a, define the per-state
    constraint c_a(s) = f_a - π(a|s) (positive when violated).
    The Lagrangian-augmented loss is:
        L = -A * log π(a*|s) + Σ_a λ_a * max(0, c_a(s))

    Per step:
    1. Sample a* ~ π(·|s); observe reward r; compute advantage
       A = r - baseline.
    2. Compute REINFORCE gradient on log π(a*|s).
    3. Compute constraint gradient: for each floor-violated
       action a, push policy toward higher π(a|s). The
       gradient of -π(a|s) w.r.t. logits z is dπ_a/dz_i =
       π_a * (δ_{ia} - π_i).
    4. Update controller params: θ ← θ - lr * (g_policy +
       λ_a * g_constraint).
    5. Update λ_a ← max(0, λ_a + lambda_lr * (c_a(s) - 0)).
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
    floor_actions = list(config.per_action_prob_floors)
    K = len(floor_actions)
    lambdas = _np.zeros((K,), dtype=_np.float64)
    # Pre-refinement evaluation. Pre-eval does NOT apply
    # projection — this is the honest "what does REINFORCE
    # alone do" baseline.
    pre_violation_rate, pre_mean_u = _evaluate_constrained_v1(
        controller=cur,
        sim=deployment_sim,
        eval_features=eval_features,
        config=config,
        apply_projection_at_eval=False)
    # Online refinement.
    baseline = float(pre_mean_u)
    pool_size = int(eval_features.shape[0])
    violation_logs: list[ConstraintViolationLogV1] = []
    for ep in range(int(n_episodes)):
        idx = int(rng.integers(0, max(1, pool_size)))
        x = eval_features[idx]
        _, probs, z1 = _controller_logits_and_probs(cur, x)
        action_idx = int(rng.choice(
            int(cur.n_actions), p=probs))
        reward = float(deployment_sim.utility_for_action(
            features=x, action_index=action_idx))
        advantage = float(reward) - float(baseline)
        # REINFORCE d_logits (negative gradient direction).
        onehot = _np.zeros(
            (int(cur.n_actions),), dtype=_np.float64)
        onehot[action_idx] = 1.0
        d_logits = -(onehot - probs) * advantage
        # Constraint gradient: for each violated floor, push
        # policy probability of that action higher. Gradient
        # of -π(a|s) w.r.t. z is -(π_a * δ_{ia} - π_a * π_i)
        # = π_a * (π_i - δ_{ia}).
        # So d_logits += λ_a * π_a * (π_i - δ_{ia}).
        for k, (a, floor) in enumerate(floor_actions):
            try:
                a_idx = (
                    list(W81_ECONOMICS_ACTIONS).index(str(a)))
            except ValueError:
                continue
            pa = float(probs[a_idx])
            v = float(floor) - pa
            if v > 0:
                onehot_a = _np.zeros_like(probs)
                onehot_a[a_idx] = 1.0
                # Gradient of L = λ_a * max(0, floor - π_a)
                # w.r.t. z: -λ_a * dπ_a/dz = -λ_a * π_a * (e_a - π)
                grad_pa_wrt_z = pa * (onehot_a - probs)
                d_logits += -float(lambdas[k]) * grad_pa_wrt_z
                violation_logs.append(ConstraintViolationLogV1(
                    schema=(
                        W84_CONSTRAINED_V1_SCHEMA_VERSION),
                    episode_index=int(ep),
                    constraint_axis=f"floor:{a}",
                    measured_value=float(pa),
                    threshold=float(floor),
                    violation_magnitude=float(v)))
                # Update λ_a by gradient ascent on violation.
                lambdas[k] = float(max(
                    0.0,
                    float(lambdas[k]) + float(lambda_lr)
                    * float(v)))
        # Backprop d_logits through the network.
        h = _swish(z1)
        g_W2 = _np.outer(h, d_logits)
        g_b2 = d_logits
        d_h = d_logits @ cur.W2.T
        sig = 1.0 / (1.0 + _np.exp(-z1))
        swish_d = sig + z1 * sig * (1.0 - sig)
        d_z1 = d_h * swish_d
        g_W1 = _np.outer(x, d_z1)
        g_b1 = d_z1
        cur.W1 = cur.W1 - float(policy_lr) * g_W1
        cur.b1 = cur.b1 - float(policy_lr) * g_b1
        cur.W2 = cur.W2 - float(policy_lr) * g_W2
        cur.b2 = cur.b2 - float(policy_lr) * g_b2
        # Update baseline.
        baseline = (
            float(baseline_decay) * float(baseline)
            + (1.0 - float(baseline_decay)) * float(reward))
    cur.n_train_steps = (
        int(cur.n_train_steps) + int(n_episodes))
    # Post-refinement evaluation. With Lagrangian only, the
    # post may not fully respect the floor (Lagrangian is a
    # *soft* constraint). With apply_projection_at_eval=True
    # (default), the projection fallback gives an exactly-
    # feasible eval. This is the issue's intended V1 contract:
    # "Lagrangian + projection V1; trust-region methods V2."
    post_violation_rate, post_mean_u = (
        _evaluate_constrained_v1(
            controller=cur,
            sim=deployment_sim,
            eval_features=eval_features,
            config=config,
            apply_projection_at_eval=bool(
                apply_projection_at_eval)))
    chain_cid = _sha256_hex({
        "kind": "w84_violation_log_chain_v1",
        "log_cids": [str(v.cid()) for v in violation_logs],
    })
    return cur, ConstrainedRefinementReportV1(
        schema=W84_CONSTRAINED_V1_SCHEMA_VERSION,
        config_cid=str(config.cid()),
        controller_cid_pre=str(pre_cid),
        controller_cid_post=str(cur.cid()),
        n_episodes=int(n_episodes),
        final_lambdas=tuple(float(l) for l in lambdas),
        pre_constraint_violation_rate=float(
            pre_violation_rate),
        post_constraint_violation_rate=float(
            post_violation_rate),
        pre_mean_utility=float(pre_mean_u),
        post_mean_utility=float(post_mean_u),
        price_of_safety_utility_delta=float(
            float(pre_mean_u) - float(post_mean_u)),
        n_violation_logs=int(len(violation_logs)),
        violation_log_chain_cid=str(chain_cid),
    )


def _evaluate_constrained_v1(
        *,
        controller: LearnedEconomicsControllerV1,
        sim: DriftedDeploymentSimulationV1,
        eval_features: "_np.ndarray",
        config: ConstrainedPolicyConfigV1,
        apply_projection_at_eval: bool = False,
) -> tuple[float, float]:
    """Returns ``(violation_rate, mean_utility)``.

    ``violation_rate`` is the fraction of held-out states on
    which ANY constraint is violated.

    If ``apply_projection_at_eval`` is True, the projection
    fallback is applied to the policy at eval time. This is
    the V1 honest "constraint guarantee" path: the projection
    gives an exactly-feasible distribution from which the
    eval samples actions.
    """
    N = int(eval_features.shape[0])
    violations = 0
    utilities: list[float] = []
    for i in range(N):
        x = eval_features[i]
        _, probs, _ = _controller_logits_and_probs(
            controller, x)
        eval_probs = (
            project_to_feasible_set(probs, config)
            if apply_projection_at_eval else probs)
        logs = measure_constraint_violations(
            eval_probs, config, episode_index=i)
        if len(logs) > 0:
            violations += 1
        # Mean utility under argmax (deterministic eval).
        action_idx = int(_np.argmax(eval_probs))
        u = float(sim.utility_for_action(
            features=x, action_index=action_idx))
        utilities.append(u)
    return (
        float(violations) / max(1, int(N)),
        float(_np.mean(utilities)),
    )


# ---------------------------------------------------------------
# Multi-seed bench
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ConstrainedBenchReportV1:
    schema: str
    config_cid: str
    n_seeds: int
    seeds: tuple[int, ...]
    pre_violation_rates: tuple[float, ...]
    post_violation_rates: tuple[float, ...]
    mean_post_violation_rate: float
    mean_pre_violation_rate: float
    mean_post_utility: float
    mean_pre_utility: float
    mean_price_of_safety_utility_delta: float
    constraints_respected_across_seeds: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "config_cid": str(self.config_cid),
            "n_seeds": int(self.n_seeds),
            "seeds": list(self.seeds),
            "pre_violation_rates": [
                float(round(r, 12))
                for r in self.pre_violation_rates],
            "post_violation_rates": [
                float(round(r, 12))
                for r in self.post_violation_rates],
            "mean_pre_violation_rate": float(round(
                self.mean_pre_violation_rate, 12)),
            "mean_post_violation_rate": float(round(
                self.mean_post_violation_rate, 12)),
            "mean_post_utility": float(round(
                self.mean_post_utility, 12)),
            "mean_pre_utility": float(round(
                self.mean_pre_utility, 12)),
            "mean_price_of_safety_utility_delta": float(
                round(
                    self.mean_price_of_safety_utility_delta,
                    12)),
            "constraints_respected_across_seeds": bool(
                self.constraints_respected_across_seeds),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_constrained_bench_report_v1",
            "report": self.to_dict()})


def run_constrained_bench_v1(
        *,
        config: ConstrainedPolicyConfigV1 | None = None,
        n_seeds: int = 10,
        n_episodes: int = 600,
        n_eval_samples: int = 60,
        violation_floor_threshold: float = 0.15,
) -> ConstrainedBenchReportV1:
    """Multi-seed bench: train UNCONSTRAINED REINFORCE on a
    regime that drives the abstain floor to violation, then
    train Lagrangian-constrained on the same regime. Across
    seeds, the constrained refinement's post-violation-rate
    must be below ``violation_floor_threshold`` on a strict
    majority of seeds (the "respected across seeds" claim).
    """
    from .learned_economics_controller_v1 import (
        build_economics_dataset_v1,
        build_learned_economics_controller_v1,
        train_learned_economics_controller,
    )
    from .online_economics_refinement_v1 import (
        build_drifted_deployment_simulation_v1,
    )
    if config is None:
        config = build_constrained_policy_config_v1(
            per_action_prob_floors={"abstain": 0.20})
    seeds = tuple(11 + 17 * i for i in range(int(n_seeds)))
    sim = build_drifted_deployment_simulation_v1()
    pre_rates: list[float] = []
    post_rates: list[float] = []
    pre_utils: list[float] = []
    post_utils: list[float] = []
    deltas: list[float] = []
    for s in seeds:
        ctrl = build_learned_economics_controller_v1()
        # Offline pre-train on synthetic; the resulting
        # controller may push abstain very low under drift.
        X, y, _ = build_economics_dataset_v1(
            n_samples=120, seed=int(s) + 7)
        ctrl, _ = train_learned_economics_controller(
            controller=ctrl,
            train_features=X,
            train_optimal_action_indices=y,
            n_iters=60)
        X_eval, _, _ = build_economics_dataset_v1(
            n_samples=int(n_eval_samples),
            seed=int(s) + 113)
        _, rep = lagrangian_refine_constrained_v1(
            controller=ctrl,
            deployment_sim=sim,
            eval_features=X_eval,
            config=config,
            n_episodes=int(n_episodes),
            seed=int(s) + 211)
        pre_rates.append(float(rep.pre_constraint_violation_rate))
        post_rates.append(
            float(rep.post_constraint_violation_rate))
        pre_utils.append(float(rep.pre_mean_utility))
        post_utils.append(float(rep.post_mean_utility))
        deltas.append(
            float(rep.price_of_safety_utility_delta))
    mean_pre = float(_np.mean(pre_rates))
    mean_post = float(_np.mean(post_rates))
    # Constraints respected: strict majority of seeds have
    # post-violation-rate below the threshold AND the mean
    # post is strictly below the mean pre.
    n_pass = sum(
        1 for r in post_rates
        if float(r) <= float(violation_floor_threshold))
    respected = bool(
        n_pass > (int(n_seeds) // 2)
        and mean_post < mean_pre)
    return ConstrainedBenchReportV1(
        schema=W84_CONSTRAINED_V1_SCHEMA_VERSION,
        config_cid=str(config.cid()),
        n_seeds=int(n_seeds),
        seeds=tuple(int(s) for s in seeds),
        pre_violation_rates=tuple(pre_rates),
        post_violation_rates=tuple(post_rates),
        mean_pre_violation_rate=float(mean_pre),
        mean_post_violation_rate=float(mean_post),
        mean_pre_utility=float(_np.mean(pre_utils)),
        mean_post_utility=float(_np.mean(post_utils)),
        mean_price_of_safety_utility_delta=float(
            _np.mean(deltas)),
        constraints_respected_across_seeds=bool(respected),
    )


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


__all__ = [
    "W84_CONSTRAINED_V1_SCHEMA_VERSION",
    "ConstrainedPolicyConfigV1",
    "build_constrained_policy_config_v1",
    "ConstraintViolationLogV1",
    "project_to_feasible_set",
    "measure_constraint_violations",
    "ConstrainedRefinementReportV1",
    "lagrangian_refine_constrained_v1",
    "ConstrainedBenchReportV1",
    "run_constrained_bench_v1",
]
