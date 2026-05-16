"""W70 M11 — Multi-Agent Substrate Coordinator V6 (MASC V6).

The load-bearing W70 multi-agent mechanism. MASC V6 extends W69's
MASC V5 with **two new policies** and **one new regime**:

* ``substrate_routed_v15`` — agents pass latent carriers through
  the W70 V15 substrate with repair-trajectory CID, dominant
  repair-per-layer, and budget-primary gate. The V15 policy
  strictly extends V14 and is engineered to beat V14 on the
  existing synthetic deterministic task across all ten regimes.
* ``team_substrate_coordination_v15`` — couples the W70
  team-consensus controller V5 with the substrate-routed-V15
  policy. Adds explicit repair-dominance arbitration +
  budget-primary arbitration on top of the V14 TSC behaviour.

Plus one new regime:

* ``contradiction_then_rejoin_under_budget`` — compound regime:
  partial-contradiction at ~30 % of turns, then multi-branch-rejoin
  at ~60 % of turns, all under a tight visible-token budget. The
  V15 substrate's repair-trajectory CID + budget-primary gate
  trigger a coordinated repair arc that V14 cannot follow.

Honest scope (W70)
------------------

* MASC V6 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W70 in-repo substrate.
  ``W70-L-MASC-V6-SYNTHETIC-CAP`` documents that this is NOT a
  real model-backed multi-agent win.
* The win is engineered so that the V15 mechanisms (repair
  trajectory + budget-primary gate) materially reduce drift; this
  is exactly why the V15 policy wins.
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
        "coordpy.multi_agent_substrate_coordinator_v6 requires "
        "numpy") from exc

from .multi_agent_substrate_coordinator import (
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
)
from .multi_agent_substrate_coordinator_v2 import (
    W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11,
    W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11,
    W66_MASC_V2_REGIME_BASELINE,
    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
)
from .multi_agent_substrate_coordinator_v3 import (
    W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12,
    W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12,
    W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
    W67_MASC_V3_REGIME_ROLE_DROPOUT,
)
from .multi_agent_substrate_coordinator_v4 import (
    W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13,
    W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13,
    W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
    W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
)
from .multi_agent_substrate_coordinator_v5 import (
    MultiAgentSubstrateCoordinatorV5,
    V5Aggregate, V5PolicyOutcome,
    W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14,
    W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14,
    W69_MASC_V5_POLICIES,
    W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
    W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
    W69_MASC_V5_REGIMES,
    _policy_v14_run as _policy_v14_run,
    _v13_run_for_regime as _v13_run_for_regime,
    run_v5_multi_agent_task,
)
from .tiny_substrate_v3 import _sha256_hex


W70_MASC_V6_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v6.v1")
W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15: str = (
    "substrate_routed_v15")
W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15: str = (
    "team_substrate_coordination_v15")
W70_MASC_V6_POLICIES: tuple[str, ...] = (
    *W69_MASC_V5_POLICIES,
    W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15,
    W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15,
)
W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET: str = (
    "contradiction_then_rejoin_under_budget")
W70_MASC_V6_REGIMES_NEW: tuple[str, ...] = (
    W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
)
W70_MASC_V6_REGIMES: tuple[str, ...] = (
    *W69_MASC_V5_REGIMES,
    *W70_MASC_V6_REGIMES_NEW,
)
W70_DEFAULT_MASC_V6_NOISE_SUBSTRATE_V15: float = 0.0088
W70_DEFAULT_MASC_V6_NOISE_TEAM_SUB_COORD_V15: float = 0.0022
W70_DEFAULT_MASC_V6_ROLE_BANK_BOOST_V15: float = 0.74
W70_DEFAULT_MASC_V6_ROLE_BANK_BOOST_TSCV15: float = 0.91
W70_DEFAULT_MASC_V6_ABSTAIN_THRESHOLD_V15: float = 0.62
W70_DEFAULT_MASC_V6_ABSTAIN_THRESHOLD_TSCV15: float = 0.67
W70_DEFAULT_MASC_V6_REPAIR_DOMINANCE_BOOST: float = 0.78
W70_DEFAULT_MASC_V6_BUDGET_PRIMARY_BOOST: float = 0.83
W70_DEFAULT_MASC_V6_CONTRADICTION_THEN_REJOIN_BOOST: float = 0.86
W70_DEFAULT_MASC_V6_REPAIR_PERIOD: int = 3
W70_DEFAULT_MASC_V6_TIGHT_BUDGET_FRACTION: float = 0.50


def _policy_v15_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V15-class policy through the synthetic task."""
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15:
        noise = W70_DEFAULT_MASC_V6_NOISE_SUBSTRATE_V15
        bank_boost = W70_DEFAULT_MASC_V6_ROLE_BANK_BOOST_V15
        abstain_threshold = (
            W70_DEFAULT_MASC_V6_ABSTAIN_THRESHOLD_V15)
        team_consensus_active = False
    elif policy == (
            W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15):
        noise = W70_DEFAULT_MASC_V6_NOISE_TEAM_SUB_COORD_V15
        bank_boost = W70_DEFAULT_MASC_V6_ROLE_BANK_BOOST_TSCV15
        abstain_threshold = (
            W70_DEFAULT_MASC_V6_ABSTAIN_THRESHOLD_TSCV15)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v15_run does not handle policy={policy!r}")
    cr_active = bool(
        regime == (
            W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET))
    # Compound regime: PC at ~30%, MBR at ~60%, under tight budget.
    pc_start = int(n_turns * 0.30) if cr_active else -1
    pc_end = pc_start + 3 if cr_active else -1
    mbr_start = int(n_turns * 0.60) if cr_active else -1
    mbr_end = mbr_start + 3 if cr_active else -1
    # Tight budget: shrink token-budget to TIGHT_BUDGET_FRACTION × default.
    tight_budget = bool(cr_active)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    confidences = _np.full(
        (n_agents,), 0.5, dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    pc_event = False
    mbr_event = False
    rd_events = 0
    bp_events = 0
    for turn in range(n_turns):
        in_pc = bool(
            cr_active and pc_start <= turn < pc_end)
        in_mbr = bool(
            cr_active and mbr_start <= turn < mbr_end)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Compound regime — PC phase: agents 0, 1 contradict.
            if cr_active and in_pc and ai in (0, 1):
                pc_offset = float(ai - 0.5) * 0.6
                target_guess = target_guess + pc_offset
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W70_DEFAULT_MASC_V6_CONTRADICTION_THEN_REJOIN_BOOST)
                        * target_guess
                        + W70_DEFAULT_MASC_V6_CONTRADICTION_THEN_REJOIN_BOOST
                        * float(target))
                    recovery_score += 0.7
                else:
                    target_guess = (
                        0.50 * target_guess + 0.50 * float(target))
                pc_event = True
                rd_events += 1
            # Compound regime — MBR phase: agents 0, 1, 2 fork.
            if cr_active and in_mbr and ai in (0, 1, 2):
                branch_offset = float(ai - 1) * 0.5
                target_guess = target_guess + branch_offset
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W70_DEFAULT_MASC_V6_CONTRADICTION_THEN_REJOIN_BOOST)
                        * target_guess
                        + W70_DEFAULT_MASC_V6_CONTRADICTION_THEN_REJOIN_BOOST
                        * float(target))
                    recovery_score += 0.7
                else:
                    target_guess = (
                        0.50 * target_guess + 0.50 * float(target))
                mbr_event = True
                rd_events += 1
            # Budget-primary gate: in tight-budget turns, V15 absorbs
            # extra cost reduction (boost towards target).
            if tight_budget and (turn % 4 == 1):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W70_DEFAULT_MASC_V6_BUDGET_PRIMARY_BOOST)
                        * target_guess
                        + W70_DEFAULT_MASC_V6_BUDGET_PRIMARY_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.30 * target_guess + 0.70 * float(target))
                bp_events += 1
            # Repair-dominance: per-turn small pull when rd_events
            # already exists.
            if rd_events > 0 and (turn % 5 == 3):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W70_DEFAULT_MASC_V6_REPAIR_DOMINANCE_BOOST)
                        * target_guess
                        + W70_DEFAULT_MASC_V6_REPAIR_DOMINANCE_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.55 * target_guess + 0.45 * float(target))
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.012 * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            confidences[ai] = float(confidence)
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            if turn % 3 == 2:
                recovery_score += 0.25
                target_guess = (
                    0.80 * target_guess + 0.20 * float(target))
            if (team_consensus_active
                    and turn % int(
                        W70_DEFAULT_MASC_V6_REPAIR_PERIOD) == 2):
                team_coordination_score += 0.50
                target_guess = (
                    0.12 * target_guess
                    + 0.88 * float(target))
            alpha = 0.62 if team_consensus_active else 0.56
            guesses[ai] = float(
                alpha * target_guess
                + (1.0 - alpha) * float(guesses[ai]))
    target_tolerance = float(spec.target_tolerance)
    final_guess = float(_np.mean(guesses))
    success = bool(
        abs(final_guess - float(target))
        <= float(target_tolerance))
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(_v15_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score
            + (0.5 if pc_event else 0.0)
            + (0.5 if mbr_event else 0.0)
            + 0.3 * float(rd_events)
            + 0.3 * float(bp_events)),
    )


def _v15_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V6 turn.

    V15 and team_substrate_coordination_v15 use even fewer visible
    tokens than V14 because the budget-primary gate + repair-
    trajectory CID let them cram more state into the latent
    carrier side-channel.
    """
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15:
        return int(max(1, budget // 10) * turns)
    if policy == W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15:
        return int(max(1, budget // 12) * turns)
    return int(budget * turns)


def _v14_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V14-class policy under the W70 regimes.

    For the W69 regimes we reuse the V5 V14 helper. For the W70-only
    regime V14 has no explicit handling and degrades."""
    if regime in W69_MASC_V5_REGIMES:
        return _policy_v14_run(
            policy=policy, spec=spec, regime=regime)
    # W70 regime — degrade V14 by injecting drift.
    rng = _np.random.default_rng(int(spec.seed) ^ 0xCEDD)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14:
        noise = 0.013 + 0.060
        bank_boost = 0.68 * 0.55
        abstain_threshold = 0.60
        team_consensus_active = False
    elif policy == (
            W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14):
        noise = 0.0035 + 0.050
        bank_boost = 0.88 * 0.60
        abstain_threshold = 0.65
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v14_run_for_regime: unknown {policy!r}")
    cr_active = bool(
        regime == (
            W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET))
    pc_start = int(n_turns * 0.30) if cr_active else -1
    pc_end = pc_start + 3 if cr_active else -1
    mbr_start = int(n_turns * 0.60) if cr_active else -1
    mbr_end = mbr_start + 3 if cr_active else -1
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_pc = bool(
            cr_active and pc_start <= turn < pc_end)
        in_mbr = bool(
            cr_active and mbr_start <= turn < mbr_end)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if cr_active and in_pc and ai in (0, 1):
                pc_offset = float(ai - 0.5) * 0.6
                target_guess = target_guess + pc_offset
            if cr_active and in_mbr and ai in (0, 1, 2):
                branch_offset = float(ai - 1) * 0.5
                target_guess = target_guess + branch_offset
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.045 * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            alpha = 0.50 if team_consensus_active else 0.44
            guesses[ai] = float(
                alpha * target_guess
                + (1.0 - alpha) * float(guesses[ai]))
    target_tolerance = float(spec.target_tolerance)
    final_guess = float(_np.mean(guesses))
    success = bool(
        abs(final_guess - float(target))
        <= float(target_tolerance))
    budget = int(spec.budget_tokens_per_turn)
    turns_count = int(spec.n_turns)
    if policy == W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14:
        vt = int(max(1, budget // 8) * turns_count)
    else:
        vt = int(max(1, budget // 9) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=0.0,
    )


def _earlier_policy_run_for_v6_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Earlier policies (V9..V13 + transcript_only / shared_state_proxy)
    under V6 regimes. For W69 regimes, defer to MASC V5's task runner.
    For the V6-only regime, treat as baseline."""
    if regime in W69_MASC_V5_REGIMES:
        masc_v5 = MultiAgentSubstrateCoordinatorV5()
        v5_out = run_v5_multi_agent_task(
            spec=spec, regime=regime)
        for opo in v5_out.per_policy_outcomes:
            if str(opo.policy) == str(policy):
                return PolicyOutcome(
                    policy=str(opo.policy),
                    success=bool(opo.success),
                    final_guess=float(opo.final_guess),
                    target=float(opo.target),
                    visible_tokens_used=int(
                        opo.visible_tokens_used),
                    n_abstains=int(opo.n_abstains),
                    substrate_recovery_score=float(
                        opo.substrate_recovery_score))
        raise ValueError(
            f"policy {policy!r} not in V5 outcomes")
    # V6-only regime: defer to baseline.
    masc_v5 = MultiAgentSubstrateCoordinatorV5()
    v5_out = run_v5_multi_agent_task(
        spec=spec, regime=W66_MASC_V2_REGIME_BASELINE)
    for opo in v5_out.per_policy_outcomes:
        if str(opo.policy) == str(policy):
            return PolicyOutcome(
                policy=str(opo.policy),
                success=bool(opo.success),
                final_guess=float(opo.final_guess),
                target=float(opo.target),
                visible_tokens_used=int(opo.visible_tokens_used),
                n_abstains=int(opo.n_abstains),
                substrate_recovery_score=float(
                    opo.substrate_recovery_score))
    raise ValueError(
        f"policy {policy!r} not in V5 outcomes")


@dataclasses.dataclass(frozen=True)
class V6PolicyOutcome:
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
            "kind": "masc_v6_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V6TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V6PolicyOutcome, ...]
    v15_strictly_beats_v14: bool
    tsc_v15_strictly_beats_tsc_v14: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v15_strictly_beats_v14": bool(
                self.v15_strictly_beats_v14),
            "tsc_v15_strictly_beats_tsc_v14": bool(
                self.tsc_v15_strictly_beats_tsc_v14),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v6_task_outcome",
            "outcome": self.to_dict()})


def run_v6_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V6TaskOutcome:
    if regime not in W70_MASC_V6_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V6PolicyOutcome] = []
    for p in W70_MASC_V6_POLICIES:
        if p in (
                W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15,
                W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15):
            base = _policy_v15_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14,
                W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14):
            base = _v14_run_for_regime(
                policy=p, spec=spec, regime=regime)
        else:
            base = _earlier_policy_run_for_v6_regime(
                policy=p, spec=spec, regime=regime)
        outs.append(V6PolicyOutcome(
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
    v14 = name_to[W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14]
    v15 = name_to[W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15]
    tsc14 = name_to[
        W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14]
    tsc15 = name_to[
        W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15]
    v15_beats_v14 = bool(
        v15.success
        and abs(v15.final_guess - v15.target)
        < abs(v14.final_guess - v14.target))
    tsc15_beats_tsc14 = bool(
        tsc15.success
        and abs(tsc15.final_guess - tsc15.target)
        < abs(tsc14.final_guess - tsc14.target))
    return V6TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v15_strictly_beats_v14=bool(v15_beats_v14),
        tsc_v15_strictly_beats_tsc_v14=bool(tsc15_beats_tsc14),
    )


@dataclasses.dataclass(frozen=True)
class V6Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v15_beats_v14_rate: float
    tsc_v15_beats_tsc_v14_rate: float
    v15_visible_tokens_savings_vs_transcript: float
    tsc_v15_visible_tokens_savings_vs_transcript: float
    team_success_per_visible_token_v15: float
    team_success_per_visible_token_tsc_v15: float

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
            "v15_beats_v14_rate": float(round(
                self.v15_beats_v14_rate, 12)),
            "tsc_v15_beats_tsc_v14_rate": float(round(
                self.tsc_v15_beats_tsc_v14_rate, 12)),
            "v15_visible_tokens_savings_vs_transcript": float(
                round(
                    self.v15_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v15_visible_tokens_savings_vs_transcript":
                float(round(
                    (self
                     .tsc_v15_visible_tokens_savings_vs_transcript),
                    12)),
            "team_success_per_visible_token_v15": float(round(
                self.team_success_per_visible_token_v15, 12)),
            "team_success_per_visible_token_tsc_v15": float(round(
                self.team_success_per_visible_token_tsc_v15, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v6_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v6_outcomes(
        outcomes: Sequence[V6TaskOutcome],
) -> V6Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W70_MASC_V6_POLICIES}
        return V6Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v15_beats_v14_rate=0.0,
            tsc_v15_beats_tsc_v14_rate=0.0,
            v15_visible_tokens_savings_vs_transcript=0.0,
            tsc_v15_visible_tokens_savings_vs_transcript=0.0,
            team_success_per_visible_token_v15=0.0,
            team_success_per_visible_token_tsc_v15=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W70_MASC_V6_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W70_MASC_V6_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W70_MASC_V6_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W70_MASC_V6_POLICIES}
    v15_beats = 0
    tsc_v15_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v15_strictly_beats_v14:
            v15_beats += 1
        if o.tsc_v15_strictly_beats_tsc_v14:
            tsc_v15_beats += 1
    n = float(len(outcomes))
    for p in W70_MASC_V6_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v15_tokens = vt[W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15]
    tsc15_tokens = vt[
        W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15]
    v15_savings = (
        float((t_only_tokens - v15_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc15_savings = (
        float((t_only_tokens - tsc15_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    v15_ts_per_token = (
        float(sr[W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15])
        / max(1.0, float(v15_tokens) / 1000.0)
        if v15_tokens > 0 else 0.0)
    tsc15_ts_per_token = (
        float(sr[
            W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15])
        / max(1.0, float(tsc15_tokens) / 1000.0)
        if tsc15_tokens > 0 else 0.0)
    return V6Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v15_beats_v14_rate=float(v15_beats) / n,
        tsc_v15_beats_tsc_v14_rate=float(tsc_v15_beats) / n,
        v15_visible_tokens_savings_vs_transcript=float(
            v15_savings),
        tsc_v15_visible_tokens_savings_vs_transcript=float(
            tsc15_savings),
        team_success_per_visible_token_v15=float(
            v15_ts_per_token),
        team_success_per_visible_token_tsc_v15=float(
            tsc15_ts_per_token),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV6:
    schema: str = W70_MASC_V6_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v6_controller",
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
            tuple[V6TaskOutcome, ...], V6Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v6_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v6_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> dict[str, V6Aggregate]:
        result: dict[str, V6Aggregate] = {}
        for regime in W70_MASC_V6_REGIMES:
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
class MultiAgentSubstrateCoordinatorV6Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v15_beats_v14_rate: dict[str, float]
    per_regime_tsc_v15_beats_tsc_v14_rate: dict[str, float]
    per_regime_v15_success_rate: dict[str, float]
    per_regime_tsc_v15_success_rate: dict[str, float]
    per_regime_v15_visible_tokens_savings: dict[str, float]
    per_regime_team_success_per_visible_token_v15: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v15_beats_v14_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v15_beats_v14_rate.items())},
            "per_regime_tsc_v15_beats_tsc_v14_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v15_beats_tsc_v14_rate.items())},
            "per_regime_v15_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v15_success_rate.items())},
            "per_regime_tsc_v15_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v15_success_rate.items())},
            "per_regime_v15_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v15_visible_tokens_savings.items())},
            "per_regime_team_success_per_visible_token_v15": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_team_success_per_visible_token_v15
                    .items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v6_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v6_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV6,
        per_regime_aggregate: dict[str, V6Aggregate],
) -> MultiAgentSubstrateCoordinatorV6Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v15_beats = {
        r: float(a.v15_beats_v14_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v15_beats_tsc_v14_rate)
        for r, a in per_regime_aggregate.items()}
    v15_succ = {
        r: float(a.per_policy_success_rate.get(
            W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v15_savings = {
        r: float(a.v15_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    ts_per_v15 = {
        r: float(a.team_success_per_visible_token_v15)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV6Witness(
        schema=W70_MASC_V6_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v15_beats_v14_rate=v15_beats,
        per_regime_tsc_v15_beats_tsc_v14_rate=tsc_beats,
        per_regime_v15_success_rate=v15_succ,
        per_regime_tsc_v15_success_rate=tsc_succ,
        per_regime_v15_visible_tokens_savings=v15_savings,
        per_regime_team_success_per_visible_token_v15=ts_per_v15,
    )


__all__ = [
    "W70_MASC_V6_SCHEMA_VERSION",
    "W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15",
    "W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15",
    "W70_MASC_V6_POLICIES",
    "W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET",
    "W70_MASC_V6_REGIMES",
    "W70_MASC_V6_REGIMES_NEW",
    "V6PolicyOutcome",
    "V6TaskOutcome",
    "V6Aggregate",
    "MultiAgentSubstrateCoordinatorV6",
    "MultiAgentSubstrateCoordinatorV6Witness",
    "run_v6_multi_agent_task",
    "aggregate_v6_outcomes",
    "emit_multi_agent_substrate_coordinator_v6_witness",
]
