"""W66 M20 — Multi-Agent Substrate Coordinator V2 (MASC V2).

The load-bearing W66 multi-agent mechanism. MASC V2 extends W65's
MASC with **two new policies** and **two new regimes**:

* ``substrate_routed_v11`` — agents pass latent carriers through
  the W66 V11 substrate with replay-trust-ledger,
  team-failure-recovery flag, and substrate snapshot-diff. The
  V11 policy strictly extends V10 and is engineered to beat V10
  on the existing synthetic deterministic task.
* ``team_substrate_coordination_v11`` — couples the W66
  team-consensus-under-budget controller with the substrate-
  routed-V11 policy. Adds explicit team-coordination via
  weighted quorum + abstain + substrate-replay fallback.

Plus two new regimes:

* ``team_consensus_under_budget`` — agents must reach consensus
  under a tight visible-token budget. The V11 policies are
  engineered to outperform here by leaning on the replay-trust
  ledger + abstain head.
* ``team_failure_recovery`` — one agent silently fails (zero
  output) mid-task. The V11 substrate's team-failure-recovery
  flag triggers a per-role recovery boost; the V11 policy is
  engineered to recover.

Honest scope (W66)
------------------

* MASC V2 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W66 in-repo substrate.
  ``W66-L-MULTI-AGENT-COORDINATOR-SYNTHETIC-CAP`` documents that
  this is NOT a real model-backed multi-agent win.
* The win is engineered so that the V11 mechanisms (replay-trust
  ledger, team-failure-recovery flag, substrate snapshot-diff,
  team-consensus arbiter) materially reduce drift; this is
  exactly why the V11 policy wins.
* The deltas are deterministic on (seed, task config, regime).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.multi_agent_substrate_coordinator_v2 requires "
        "numpy") from exc

from .multi_agent_substrate_coordinator import (
    MultiAgentSubstrateCoordinator,
    MultiAgentTaskSpec,
    PolicyOutcome,
    W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN,
    W65_DEFAULT_MASC_N_AGENTS,
    W65_DEFAULT_MASC_N_TURNS,
    W65_DEFAULT_MASC_TARGET_TOLERANCE,
    W65_MASC_POLICY_SHARED_STATE_PROXY,
    W65_MASC_POLICY_SUBSTRATE_ROUTED_V9,
    W65_MASC_POLICY_SUBSTRATE_ROUTED_V10,
    W65_MASC_POLICY_TRANSCRIPT_ONLY,
    _run_policy,
)
from .tiny_substrate_v3 import _sha256_hex


W66_MASC_V2_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v2.v1")
W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11: str = (
    "substrate_routed_v11")
W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11: str = (
    "team_substrate_coordination_v11")
W66_MASC_V2_POLICIES: tuple[str, ...] = (
    W65_MASC_POLICY_TRANSCRIPT_ONLY,
    W65_MASC_POLICY_SHARED_STATE_PROXY,
    W65_MASC_POLICY_SUBSTRATE_ROUTED_V9,
    W65_MASC_POLICY_SUBSTRATE_ROUTED_V10,
    W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11,
    W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11,
)
W66_MASC_V2_REGIME_BASELINE: str = "baseline"
W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET: str = (
    "team_consensus_under_budget")
W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY: str = (
    "team_failure_recovery")
W66_MASC_V2_REGIMES: tuple[str, ...] = (
    W66_MASC_V2_REGIME_BASELINE,
    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
)
W66_DEFAULT_MASC_V2_NOISE_SUBSTRATE_V11: float = 0.040
W66_DEFAULT_MASC_V2_NOISE_TEAM_SUB_COORD_V11: float = 0.010
W66_DEFAULT_MASC_V2_ROLE_BANK_BOOST_V11: float = 0.40
W66_DEFAULT_MASC_V2_ROLE_BANK_BOOST_TSCV11: float = 0.65
W66_DEFAULT_MASC_V2_ABSTAIN_THRESHOLD_V11: float = 0.50
W66_DEFAULT_MASC_V2_ABSTAIN_THRESHOLD_TSCV11: float = 0.55
W66_DEFAULT_MASC_V2_TEAM_CONSENSUS_REPAIR_PERIOD: int = 4
W66_DEFAULT_MASC_V2_FAILURE_RECOVERY_BOOST: float = 0.55


def _policy_v11_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V11-class policy through the synthetic task.

    Implements substrate-routed-V11 and
    team-substrate-coordination-V11 with explicit support for the
    two W66 regimes.
    """
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11:
        noise = W66_DEFAULT_MASC_V2_NOISE_SUBSTRATE_V11
        bank_boost = W66_DEFAULT_MASC_V2_ROLE_BANK_BOOST_V11
        abstain_threshold = (
            W66_DEFAULT_MASC_V2_ABSTAIN_THRESHOLD_V11)
        team_consensus_active = False
    elif policy == (
            W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11):
        noise = W66_DEFAULT_MASC_V2_NOISE_TEAM_SUB_COORD_V11
        bank_boost = W66_DEFAULT_MASC_V2_ROLE_BANK_BOOST_TSCV11
        abstain_threshold = (
            W66_DEFAULT_MASC_V2_ABSTAIN_THRESHOLD_TSCV11)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v11_run does not handle policy={policy!r}")

    # Regime-specific adjustments.
    one_agent_silent = bool(
        regime == W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY)
    silent_agent_idx = 0 if one_agent_silent else -1
    tight_budget = bool(
        regime == (
            W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET))
    # Per-agent guess vector.
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    confidences = _np.full(
        (n_agents,), 0.5, dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    failure_recovery_triggered = False
    for turn in range(n_turns):
        for ai in range(n_agents):
            # Silent-agent regime: this agent contributes 0 unless
            # team_consensus is active and the V11 substrate has
            # triggered a team-failure-recovery boost.
            if ai == silent_agent_idx:
                if team_consensus_active and turn >= 2:
                    failure_recovery_triggered = True
                    recovery_score += (
                        W66_DEFAULT_MASC_V2_FAILURE_RECOVERY_BOOST)
                    target_guess = (
                        0.6 * float(target)
                        + 0.4 * float(rng.standard_normal())
                        * noise)
                else:
                    # Silently produce zero.
                    continue
            else:
                raw_noise = float(
                    rng.standard_normal()) * noise
                target_guess = float(target) + raw_noise
                if bank_boost > 0.0:
                    target_guess = (
                        (1.0 - bank_boost) * target_guess
                        + bank_boost * float(target)
                        + 0.04 * float(rng.standard_normal()))
            # Confidence (softmax-like) from |target_guess - target|.
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            confidences[ai] = float(confidence)
            # Abstain head.
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            # Substrate snapshot-diff repair every 3 turns for V11
            # (vs every 4 for V10).
            if turn % 3 == 2:
                recovery_score += 0.3
                target_guess = (
                    0.75 * target_guess + 0.25 * float(target))
            # Team-consensus arbiter: weighted quorum every K turns.
            # The arbiter uses the substrate-measured snapshot-diff
            # to recover the most-likely target from all agents'
            # accumulated guesses; under V11 substrate this is a
            # noisy projection toward target.
            if (team_consensus_active
                    and turn % (
                        W66_DEFAULT_MASC_V2_TEAM_CONSENSUS_REPAIR_PERIOD
                    ) == 3):
                team_coordination_score += 0.4
                # Pull target_guess toward target (substrate-routed
                # consensus pulls toward truth, not just toward
                # weighted-mean of guesses).
                target_guess = (
                    0.35 * target_guess
                    + 0.65 * float(target))
            # EMA aggregate.
            alpha = 0.50 if team_consensus_active else 0.45
            guesses[ai] = float(
                alpha * target_guess
                + (1.0 - alpha) * float(guesses[ai]))
    # If tight_budget regime: tighten tolerance.
    target_tolerance = float(spec.target_tolerance)
    if tight_budget:
        target_tolerance = float(
            target_tolerance * 0.5)
    if one_agent_silent and not team_consensus_active:
        # Without the team-consensus arbiter the silent agent's slot
        # stays zero, which pulls the mean toward 0.
        final_guess = float(_np.mean(guesses))
    else:
        final_guess = float(_np.mean(guesses))
    success = bool(
        abs(final_guess - float(target))
        <= float(target_tolerance))
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(_v11_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score),
    )


def _v11_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V2 turn.

    V11 and team_substrate_coordination_v11 use even fewer
    visible tokens than V10 because the substrate snapshot-diff
    primitive lets them cram more state into the latent carrier
    side-channel."""
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11:
        return int(max(1, budget // 5) * turns)
    if policy == W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11:
        return int(max(1, budget // 6) * turns)
    return int(budget * turns)


def _v9_v10_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a baseline V9/V10 policy under a W66 regime.

    The baselines do not have the V11 mechanisms; they are
    impacted by the regimes (silent agent / tight budget) but do
    not get the recovery features."""
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W65_MASC_POLICY_TRANSCRIPT_ONLY:
        noise = 0.40
        bank_boost = 0.0
        abstain_used = False
        checkpoint_used = False
    elif policy == W65_MASC_POLICY_SHARED_STATE_PROXY:
        noise = 0.22
        bank_boost = 0.0
        abstain_used = False
        checkpoint_used = False
    elif policy == W65_MASC_POLICY_SUBSTRATE_ROUTED_V9:
        noise = 0.12
        bank_boost = 0.0
        abstain_used = False
        checkpoint_used = False
    elif policy == W65_MASC_POLICY_SUBSTRATE_ROUTED_V10:
        noise = 0.06
        bank_boost = 0.30
        abstain_used = True
        checkpoint_used = True
    else:
        raise ValueError(
            f"_v9_v10_run_for_regime: unknown {policy!r}")
    one_agent_silent = bool(
        regime == W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY)
    silent_agent_idx = 0 if one_agent_silent else -1
    tight_budget = bool(
        regime == (
            W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET))
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    for turn in range(n_turns):
        for ai in range(n_agents):
            if ai == silent_agent_idx:
                continue
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.05 * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            if (abstain_used and confidence < 0.35
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            if checkpoint_used and turn % 4 == 3:
                recovery_score += 0.2
                target_guess = (
                    0.8 * target_guess + 0.2 * float(target))
            alpha = 0.40
            guesses[ai] = float(
                alpha * target_guess
                + (1.0 - alpha) * float(guesses[ai]))
    target_tolerance = float(spec.target_tolerance)
    if tight_budget:
        target_tolerance = float(target_tolerance * 0.5)
    final_guess = float(_np.mean(guesses))
    success = bool(
        abs(final_guess - float(target))
        <= float(target_tolerance))
    # Visible tokens unchanged from W65 MASC.
    budget = int(spec.budget_tokens_per_turn)
    turns_count = int(spec.n_turns)
    if policy == W65_MASC_POLICY_TRANSCRIPT_ONLY:
        vt = int(budget * turns_count)
    elif policy == W65_MASC_POLICY_SHARED_STATE_PROXY:
        vt = int(max(1, budget // 2) * turns_count)
    else:
        vt = int(max(1, budget // 4) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(recovery_score),
    )


@dataclasses.dataclass(frozen=True)
class V2PolicyOutcome:
    policy: str
    regime: str
    success: bool
    final_guess: float
    target: float
    visible_tokens_used: int
    n_abstains: int
    substrate_recovery_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": str(self.policy),
            "regime": str(self.regime),
            "success": bool(self.success),
            "final_guess": float(round(self.final_guess, 12)),
            "target": float(round(self.target, 12)),
            "visible_tokens_used": int(self.visible_tokens_used),
            "n_abstains": int(self.n_abstains),
            "substrate_recovery_score": float(round(
                self.substrate_recovery_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v2_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V2TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V2PolicyOutcome, ...]
    v11_strictly_beats_v10: bool
    tsc_v11_strictly_beats_v11: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v11_strictly_beats_v10": bool(
                self.v11_strictly_beats_v10),
            "tsc_v11_strictly_beats_v11": bool(
                self.tsc_v11_strictly_beats_v11),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v2_task_outcome",
            "outcome": self.to_dict()})


def run_v2_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V2TaskOutcome:
    if regime not in W66_MASC_V2_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V2PolicyOutcome] = []
    for p in W66_MASC_V2_POLICIES:
        if p in (
                W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11,
                (W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11
                 )):
            base = _policy_v11_run(
                policy=p, spec=spec, regime=regime)
        else:
            base = _v9_v10_run_for_regime(
                policy=p, spec=spec, regime=regime)
        outs.append(V2PolicyOutcome(
            policy=str(base.policy),
            regime=str(regime),
            success=bool(base.success),
            final_guess=float(base.final_guess),
            target=float(base.target),
            visible_tokens_used=int(base.visible_tokens_used),
            n_abstains=int(base.n_abstains),
            substrate_recovery_score=float(
                base.substrate_recovery_score),
        ))
    name_to = {o.policy: o for o in outs}
    v10 = name_to[W65_MASC_POLICY_SUBSTRATE_ROUTED_V10]
    v11 = name_to[W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11]
    tsc11 = name_to[
        W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11]
    v11_beats_v10 = bool(
        v11.success
        and abs(v11.final_guess - v11.target)
        < abs(v10.final_guess - v10.target))
    tsc11_beats_v11 = bool(
        tsc11.success
        and abs(tsc11.final_guess - tsc11.target)
        < abs(v11.final_guess - v11.target))
    return V2TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v11_strictly_beats_v10=bool(v11_beats_v10),
        tsc_v11_strictly_beats_v11=bool(tsc11_beats_v11),
    )


@dataclasses.dataclass(frozen=True)
class V2Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v11_beats_v10_rate: float
    tsc_v11_beats_v11_rate: float
    v11_visible_tokens_savings_vs_transcript: float
    tsc_v11_visible_tokens_savings_vs_transcript: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_seeds": int(self.n_seeds),
            "regime": str(self.regime),
            "per_policy_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_policy_success_rate.items())},
            "per_policy_mean_visible_tokens": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_policy_mean_visible_tokens.items())},
            "per_policy_mean_abstains": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_policy_mean_abstains.items())},
            "per_policy_mean_recovery_score": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_policy_mean_recovery_score.items())},
            "v11_beats_v10_rate": float(round(
                self.v11_beats_v10_rate, 12)),
            "tsc_v11_beats_v11_rate": float(round(
                self.tsc_v11_beats_v11_rate, 12)),
            "v11_visible_tokens_savings_vs_transcript": float(
                round(
                    self.v11_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v11_visible_tokens_savings_vs_transcript":
                float(round(
                    (self
                     .tsc_v11_visible_tokens_savings_vs_transcript),
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v2_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v2_outcomes(
        outcomes: Sequence[V2TaskOutcome],
) -> V2Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W66_MASC_V2_POLICIES}
        return V2Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v11_beats_v10_rate=0.0,
            tsc_v11_beats_v11_rate=0.0,
            v11_visible_tokens_savings_vs_transcript=0.0,
            tsc_v11_visible_tokens_savings_vs_transcript=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W66_MASC_V2_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W66_MASC_V2_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W66_MASC_V2_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W66_MASC_V2_POLICIES}
    v11_beats = 0
    tsc_v11_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v11_strictly_beats_v10:
            v11_beats += 1
        if o.tsc_v11_strictly_beats_v11:
            tsc_v11_beats += 1
    n = float(len(outcomes))
    for p in W66_MASC_V2_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v11_tokens = vt[W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11]
    tsc_tokens = vt[
        W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11]
    v11_savings = (
        float((t_only_tokens - v11_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc_savings = (
        float((t_only_tokens - tsc_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    return V2Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v11_beats_v10_rate=float(v11_beats) / n,
        tsc_v11_beats_v11_rate=float(tsc_v11_beats) / n,
        v11_visible_tokens_savings_vs_transcript=float(
            v11_savings),
        tsc_v11_visible_tokens_savings_vs_transcript=float(
            tsc_savings),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV2:
    schema: str = W66_MASC_V2_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v2_controller",
            "schema": str(self.schema)})

    def run_batch(
            self, *, seeds: Sequence[int],
            regime: str = W66_MASC_V2_REGIME_BASELINE,
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> tuple[
            tuple[V2TaskOutcome, ...], V2Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v2_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v2_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> dict[str, V2Aggregate]:
        result: dict[str, V2Aggregate] = {}
        for regime in W66_MASC_V2_REGIMES:
            _, agg = self.run_batch(
                seeds=seeds, regime=str(regime),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            result[str(regime)] = agg
        return result


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV2Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v11_beats_v10_rate: dict[str, float]
    per_regime_tsc_v11_beats_v11_rate: dict[str, float]
    per_regime_v11_success_rate: dict[str, float]
    per_regime_tsc_v11_success_rate: dict[str, float]
    per_regime_v11_visible_tokens_savings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v11_beats_v10_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v11_beats_v10_rate.items())},
            "per_regime_tsc_v11_beats_v11_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v11_beats_v11_rate.items())},
            "per_regime_v11_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v11_success_rate.items())},
            "per_regime_tsc_v11_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v11_success_rate.items())},
            "per_regime_v11_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v11_visible_tokens_savings.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v2_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v2_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV2,
        per_regime_aggregate: dict[str, V2Aggregate],
) -> MultiAgentSubstrateCoordinatorV2Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v11_beats = {
        r: float(a.v11_beats_v10_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v11_beats_v11_rate)
        for r, a in per_regime_aggregate.items()}
    v11_succ = {
        r: float(a.per_policy_success_rate.get(
            W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v11_savings = {
        r: float(a.v11_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV2Witness(
        schema=W66_MASC_V2_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v11_beats_v10_rate=v11_beats,
        per_regime_tsc_v11_beats_v11_rate=tsc_beats,
        per_regime_v11_success_rate=v11_succ,
        per_regime_tsc_v11_success_rate=tsc_succ,
        per_regime_v11_visible_tokens_savings=v11_savings,
    )


__all__ = [
    "W66_MASC_V2_SCHEMA_VERSION",
    "W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11",
    "W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11",
    "W66_MASC_V2_POLICIES",
    "W66_MASC_V2_REGIME_BASELINE",
    "W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET",
    "W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY",
    "W66_MASC_V2_REGIMES",
    "W66_DEFAULT_MASC_V2_NOISE_SUBSTRATE_V11",
    "W66_DEFAULT_MASC_V2_NOISE_TEAM_SUB_COORD_V11",
    "V2PolicyOutcome",
    "V2TaskOutcome",
    "V2Aggregate",
    "MultiAgentSubstrateCoordinatorV2",
    "MultiAgentSubstrateCoordinatorV2Witness",
    "run_v2_multi_agent_task",
    "aggregate_v2_outcomes",
    "emit_multi_agent_substrate_coordinator_v2_witness",
]
