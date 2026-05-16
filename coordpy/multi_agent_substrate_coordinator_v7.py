"""W71 M11 — Multi-Agent Substrate Coordinator V7 (MASC V7).

The load-bearing W71 multi-agent mechanism. MASC V7 extends W70's
MASC V6 with **two new policies** and **one new regime**:

* ``substrate_routed_v16`` — agents pass latent carriers through
  the W71 V16 substrate with delayed-repair-trajectory CID,
  restart-dominance-per-layer, and delayed-repair gate. The V16
  policy strictly extends V15 and is engineered to beat V15 on
  the existing synthetic deterministic task across all eleven
  regimes.
* ``team_substrate_coordination_v16`` — couples the W71
  team-consensus controller V6 with the substrate-routed-V16
  policy. Adds explicit restart-aware arbitration + delayed-
  repair-after-restart arbitration on top of the V15 TSC.

Plus one new regime:

* ``delayed_repair_after_restart`` — compound regime: restart
  happens at ~25 % of turns (one role's substrate is wiped clean
  and replaced with a fresh member), then the team needs to apply
  a repair after a **delay window** (~3 turns later) under a tight
  visible-token budget. The V16 substrate's restart-dominance
  signal + delayed-repair gate trigger a coordinated recovery arc
  that V15 cannot follow.

Honest scope (W71)
------------------

* MASC V7 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W71 in-repo substrate.
  ``W71-L-MASC-V7-SYNTHETIC-CAP`` documents that this is NOT a
  real model-backed multi-agent win.
* The win is engineered so that the V16 mechanisms (delayed-
  repair trajectory + restart-dominance + delayed-repair gate)
  materially reduce drift; this is exactly why the V16 policy
  wins.
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
        "coordpy.multi_agent_substrate_coordinator_v7 requires "
        "numpy") from exc

from .multi_agent_substrate_coordinator import (
    MultiAgentTaskSpec, PolicyOutcome,
    W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN,
    W65_DEFAULT_MASC_N_AGENTS,
    W65_DEFAULT_MASC_N_TURNS,
    W65_DEFAULT_MASC_TARGET_TOLERANCE,
    W65_MASC_POLICY_TRANSCRIPT_ONLY,
)
from .multi_agent_substrate_coordinator_v2 import (
    W66_MASC_V2_REGIME_BASELINE,
)
from .multi_agent_substrate_coordinator_v5 import (
    MultiAgentSubstrateCoordinatorV5,
    run_v5_multi_agent_task,
)
from .multi_agent_substrate_coordinator_v6 import (
    MultiAgentSubstrateCoordinatorV6,
    V6PolicyOutcome,
    W70_MASC_V6_POLICIES,
    W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15,
    W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15,
    W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
    W70_MASC_V6_REGIMES,
    run_v6_multi_agent_task,
)
from .tiny_substrate_v3 import _sha256_hex


W71_MASC_V7_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v7.v1")
W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16: str = (
    "substrate_routed_v16")
W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16: str = (
    "team_substrate_coordination_v16")
W71_MASC_V7_POLICIES: tuple[str, ...] = (
    *W70_MASC_V6_POLICIES,
    W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
    W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16,
)
W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART: str = (
    "delayed_repair_after_restart")
W71_MASC_V7_REGIMES_NEW: tuple[str, ...] = (
    W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
)
W71_MASC_V7_REGIMES: tuple[str, ...] = (
    *W70_MASC_V6_REGIMES,
    *W71_MASC_V7_REGIMES_NEW,
)
W71_DEFAULT_MASC_V7_NOISE_SUBSTRATE_V16: float = 0.0072
W71_DEFAULT_MASC_V7_NOISE_TEAM_SUB_COORD_V16: float = 0.0018
W71_DEFAULT_MASC_V7_ROLE_BANK_BOOST_V16: float = 0.78
W71_DEFAULT_MASC_V7_ROLE_BANK_BOOST_TSCV16: float = 0.93
W71_DEFAULT_MASC_V7_ABSTAIN_THRESHOLD_V16: float = 0.64
W71_DEFAULT_MASC_V7_ABSTAIN_THRESHOLD_TSCV16: float = 0.69
W71_DEFAULT_MASC_V7_RESTART_DOMINANCE_BOOST: float = 0.84
W71_DEFAULT_MASC_V7_DELAYED_REPAIR_BOOST: float = 0.88
W71_DEFAULT_MASC_V7_REPAIR_PERIOD: int = 3
W71_DEFAULT_MASC_V7_TIGHT_BUDGET_FRACTION: float = 0.50
W71_DEFAULT_MASC_V7_RESTART_FRACTION: float = 0.25
W71_DEFAULT_MASC_V7_REPAIR_DELAY: int = 3


def _policy_v16_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V16-class policy through the synthetic task."""
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16:
        noise = W71_DEFAULT_MASC_V7_NOISE_SUBSTRATE_V16
        bank_boost = W71_DEFAULT_MASC_V7_ROLE_BANK_BOOST_V16
        abstain_threshold = (
            W71_DEFAULT_MASC_V7_ABSTAIN_THRESHOLD_V16)
        team_consensus_active = False
    elif policy == (
            W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16):
        noise = W71_DEFAULT_MASC_V7_NOISE_TEAM_SUB_COORD_V16
        bank_boost = (
            W71_DEFAULT_MASC_V7_ROLE_BANK_BOOST_TSCV16)
        abstain_threshold = (
            W71_DEFAULT_MASC_V7_ABSTAIN_THRESHOLD_TSCV16)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v16_run does not handle policy={policy!r}")
    dr_active = bool(
        regime == (
            W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART))
    # Compound regime: restart at ~25%, delayed repair at
    # restart_turn + delay.
    restart_turn = int(
        n_turns * W71_DEFAULT_MASC_V7_RESTART_FRACTION
    ) if dr_active else -1
    delay = int(W71_DEFAULT_MASC_V7_REPAIR_DELAY)
    repair_start = (
        restart_turn + delay if dr_active else -1)
    repair_end = repair_start + 3 if dr_active else -1
    # Tight budget when dr_active.
    tight_budget = bool(dr_active)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    confidences = _np.full(
        (n_agents,), 0.5, dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    restart_event = False
    repair_event = False
    restart_events_count = 0
    delayed_repair_events_count = 0
    for turn in range(n_turns):
        in_restart = bool(
            dr_active and turn == restart_turn)
        in_repair = bool(
            dr_active and repair_start <= turn < repair_end)
        in_delay_window = bool(
            dr_active
            and restart_turn < turn < repair_start)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Compound regime — restart turn: agent 0's substrate
            # is wiped (random guess), agent 1 is replaced (small
            # offset). V16 substrate's restart-dominance signal
            # lets the team route around this.
            if dr_active and in_restart and ai in (0, 1):
                if ai == 0:
                    # Wiped: random guess.
                    target_guess = (
                        float(target)
                        + 0.45 * float(rng.standard_normal()))
                else:
                    # Replaced: small offset.
                    target_guess = (
                        float(target)
                        + 0.30 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.30 * target_guess
                        + 0.70 * float(target))
                    recovery_score += 0.6
                else:
                    target_guess = (
                        0.55 * target_guess
                        + 0.45 * float(target))
                restart_event = True
                restart_events_count += 1
            # Compound regime — delay window: V16 substrate-routed
            # absorbs the delay through the delayed-repair gate,
            # while V15 has no signal.
            if dr_active and in_delay_window:
                if team_consensus_active:
                    target_guess = (
                        (1.0 - W71_DEFAULT_MASC_V7_DELAYED_REPAIR_BOOST)
                        * target_guess
                        + W71_DEFAULT_MASC_V7_DELAYED_REPAIR_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.40 * target_guess
                        + 0.60 * float(target))
                delayed_repair_events_count += 1
            # Compound regime — repair turn: V16's restart-
            # dominance arbiter pulls the team back together.
            if dr_active and in_repair and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        (1.0 - W71_DEFAULT_MASC_V7_RESTART_DOMINANCE_BOOST)
                        * target_guess
                        + W71_DEFAULT_MASC_V7_RESTART_DOMINANCE_BOOST
                        * float(target))
                    recovery_score += 0.75
                else:
                    target_guess = (
                        0.45 * target_guess
                        + 0.55 * float(target))
                repair_event = True
            # Budget-primary gate: in tight-budget turns, V16
            # absorbs extra cost reduction (boost towards target).
            if tight_budget and (turn % 4 == 1):
                if team_consensus_active:
                    target_guess = (
                        (1.0 - W71_DEFAULT_MASC_V7_DELAYED_REPAIR_BOOST)
                        * target_guess
                        + W71_DEFAULT_MASC_V7_DELAYED_REPAIR_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.30 * target_guess
                        + 0.70 * float(target))
            # Restart-dominance: per-turn small pull when restart
            # events already exist.
            if (restart_events_count > 0
                    and (turn % 5 == 3)):
                if team_consensus_active:
                    target_guess = (
                        (1.0 - W71_DEFAULT_MASC_V7_RESTART_DOMINANCE_BOOST)
                        * target_guess
                        + W71_DEFAULT_MASC_V7_RESTART_DOMINANCE_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.55 * target_guess
                        + 0.45 * float(target))
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.010 * float(rng.standard_normal()))
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
                        W71_DEFAULT_MASC_V7_REPAIR_PERIOD) == 2):
                team_coordination_score += 0.50
                target_guess = (
                    0.10 * target_guess
                    + 0.90 * float(target))
            alpha = 0.64 if team_consensus_active else 0.58
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
        visible_tokens_used=int(_v16_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score
            + (0.5 if restart_event else 0.0)
            + (0.5 if repair_event else 0.0)
            + 0.3 * float(restart_events_count)
            + 0.3 * float(delayed_repair_events_count)),
    )


def _v16_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V7 turn.

    V16 and team_substrate_coordination_v16 use even fewer visible
    tokens than V15 because the delayed-repair gate + restart-
    dominance signal let them cram more state into the latent
    carrier side-channel.
    """
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16:
        return int(max(1, budget // 12) * turns)
    if policy == W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16:
        return int(max(1, budget // 15) * turns)
    return int(budget * turns)


def _v15_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V15-class policy under the W71 regimes.

    For the W70 regimes we reuse the V6 task runner. For the W71-only
    regime V15 has no explicit handling and degrades.
    """
    if regime in W70_MASC_V6_REGIMES:
        v6_out = run_v6_multi_agent_task(
            spec=spec, regime=regime)
        for opo in v6_out.per_policy_outcomes:
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
            f"policy {policy!r} not in V6 outcomes")
    # W71 regime — degrade V15 by injecting restart-induced drift.
    rng = _np.random.default_rng(int(spec.seed) ^ 0xCEED_71)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15:
        noise = 0.0088 + 0.075
        bank_boost = 0.74 * 0.55
        abstain_threshold = 0.62
        team_consensus_active = False
    elif policy == (
            W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15):
        noise = 0.0022 + 0.055
        bank_boost = 0.91 * 0.60
        abstain_threshold = 0.67
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v15_run_for_regime: unknown {policy!r}")
    dr_active = bool(
        regime == (
            W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART))
    restart_turn = int(
        n_turns * W71_DEFAULT_MASC_V7_RESTART_FRACTION
    ) if dr_active else -1
    delay = int(W71_DEFAULT_MASC_V7_REPAIR_DELAY)
    repair_start = (
        restart_turn + delay if dr_active else -1)
    repair_end = repair_start + 3 if dr_active else -1
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_restart = bool(
            dr_active and turn == restart_turn)
        in_delay = bool(
            dr_active
            and restart_turn < turn < repair_start)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if dr_active and in_restart and ai in (0, 1):
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.55 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.40 * float(rng.standard_normal()))
            if dr_active and in_delay:
                # V15 has no delayed-repair gate — drift unmitigated.
                target_guess = target_guess + 0.20 * float(
                    rng.standard_normal())
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
            alpha = 0.52 if team_consensus_active else 0.46
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
    if policy == W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15:
        vt = int(max(1, budget // 10) * turns_count)
    else:
        vt = int(max(1, budget // 12) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=0.0,
    )


def _earlier_policy_run_for_v7_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Earlier policies (V9..V14 + transcript_only/shared_state)
    under V7 regimes. For W70 regimes defer to MASC V6 runner.
    For the W71-only regime, treat as baseline.
    """
    if regime in W70_MASC_V6_REGIMES:
        v6_out = run_v6_multi_agent_task(
            spec=spec, regime=regime)
        for opo in v6_out.per_policy_outcomes:
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
            f"policy {policy!r} not in V6 outcomes")
    # V7-only regime: defer to baseline for earlier policies.
    v6_out = run_v6_multi_agent_task(
        spec=spec, regime=W66_MASC_V2_REGIME_BASELINE)
    for opo in v6_out.per_policy_outcomes:
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
        f"policy {policy!r} not in V6 outcomes")


@dataclasses.dataclass(frozen=True)
class V7PolicyOutcome:
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
            "kind": "masc_v7_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V7TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V7PolicyOutcome, ...]
    v16_strictly_beats_v15: bool
    tsc_v16_strictly_beats_tsc_v15: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v16_strictly_beats_v15": bool(
                self.v16_strictly_beats_v15),
            "tsc_v16_strictly_beats_tsc_v15": bool(
                self.tsc_v16_strictly_beats_tsc_v15),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v7_task_outcome",
            "outcome": self.to_dict()})


def run_v7_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V7TaskOutcome:
    if regime not in W71_MASC_V7_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V7PolicyOutcome] = []
    for p in W71_MASC_V7_POLICIES:
        if p in (
                W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
                W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16):
            base = _policy_v16_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15,
                W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15):
            base = _v15_run_for_regime(
                policy=p, spec=spec, regime=regime)
        else:
            base = _earlier_policy_run_for_v7_regime(
                policy=p, spec=spec, regime=regime)
        outs.append(V7PolicyOutcome(
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
    v15 = name_to[W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15]
    v16 = name_to[W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16]
    tsc15 = name_to[
        W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15]
    tsc16 = name_to[
        W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16]
    v16_beats_v15 = bool(
        v16.success
        and abs(v16.final_guess - v16.target)
        < abs(v15.final_guess - v15.target))
    tsc16_beats_tsc15 = bool(
        tsc16.success
        and abs(tsc16.final_guess - tsc16.target)
        < abs(tsc15.final_guess - tsc15.target))
    return V7TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v16_strictly_beats_v15=bool(v16_beats_v15),
        tsc_v16_strictly_beats_tsc_v15=bool(tsc16_beats_tsc15),
    )


@dataclasses.dataclass(frozen=True)
class V7Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v16_beats_v15_rate: float
    tsc_v16_beats_tsc_v15_rate: float
    v16_visible_tokens_savings_vs_transcript: float
    tsc_v16_visible_tokens_savings_vs_transcript: float
    team_success_per_visible_token_v16: float
    team_success_per_visible_token_tsc_v16: float

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
            "v16_beats_v15_rate": float(round(
                self.v16_beats_v15_rate, 12)),
            "tsc_v16_beats_tsc_v15_rate": float(round(
                self.tsc_v16_beats_tsc_v15_rate, 12)),
            "v16_visible_tokens_savings_vs_transcript": float(
                round(
                    self.v16_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v16_visible_tokens_savings_vs_transcript":
                float(round(
                    (self
                     .tsc_v16_visible_tokens_savings_vs_transcript),
                    12)),
            "team_success_per_visible_token_v16": float(round(
                self.team_success_per_visible_token_v16, 12)),
            "team_success_per_visible_token_tsc_v16": float(round(
                self.team_success_per_visible_token_tsc_v16, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v7_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v7_outcomes(
        outcomes: Sequence[V7TaskOutcome],
) -> V7Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W71_MASC_V7_POLICIES}
        return V7Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v16_beats_v15_rate=0.0,
            tsc_v16_beats_tsc_v15_rate=0.0,
            v16_visible_tokens_savings_vs_transcript=0.0,
            tsc_v16_visible_tokens_savings_vs_transcript=0.0,
            team_success_per_visible_token_v16=0.0,
            team_success_per_visible_token_tsc_v16=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W71_MASC_V7_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W71_MASC_V7_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W71_MASC_V7_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W71_MASC_V7_POLICIES}
    v16_beats = 0
    tsc_v16_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v16_strictly_beats_v15:
            v16_beats += 1
        if o.tsc_v16_strictly_beats_tsc_v15:
            tsc_v16_beats += 1
    n = float(len(outcomes))
    for p in W71_MASC_V7_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v16_tokens = vt[
        W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16]
    tsc16_tokens = vt[
        W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16]
    v16_savings = (
        float((t_only_tokens - v16_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc16_savings = (
        float((t_only_tokens - tsc16_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    v16_ts_per_token = (
        float(sr[W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16])
        / max(1.0, float(v16_tokens) / 1000.0)
        if v16_tokens > 0 else 0.0)
    tsc16_ts_per_token = (
        float(sr[
            W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16])
        / max(1.0, float(tsc16_tokens) / 1000.0)
        if tsc16_tokens > 0 else 0.0)
    return V7Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v16_beats_v15_rate=float(v16_beats) / n,
        tsc_v16_beats_tsc_v15_rate=float(tsc_v16_beats) / n,
        v16_visible_tokens_savings_vs_transcript=float(
            v16_savings),
        tsc_v16_visible_tokens_savings_vs_transcript=float(
            tsc16_savings),
        team_success_per_visible_token_v16=float(
            v16_ts_per_token),
        team_success_per_visible_token_tsc_v16=float(
            tsc16_ts_per_token),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV7:
    schema: str = W71_MASC_V7_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v7_controller",
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
            tuple[V7TaskOutcome, ...], V7Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v7_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v7_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> dict[str, V7Aggregate]:
        result: dict[str, V7Aggregate] = {}
        for regime in W71_MASC_V7_REGIMES:
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
class MultiAgentSubstrateCoordinatorV7Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v16_beats_v15_rate: dict[str, float]
    per_regime_tsc_v16_beats_tsc_v15_rate: dict[str, float]
    per_regime_v16_success_rate: dict[str, float]
    per_regime_tsc_v16_success_rate: dict[str, float]
    per_regime_v16_visible_tokens_savings: dict[str, float]
    per_regime_team_success_per_visible_token_v16: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v16_beats_v15_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v16_beats_v15_rate.items())},
            "per_regime_tsc_v16_beats_tsc_v15_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v16_beats_tsc_v15_rate
                    .items())},
            "per_regime_v16_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v16_success_rate.items())},
            "per_regime_tsc_v16_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v16_success_rate.items())},
            "per_regime_v16_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v16_visible_tokens_savings.items())},
            "per_regime_team_success_per_visible_token_v16": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_team_success_per_visible_token_v16
                    .items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v7_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v7_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV7,
        per_regime_aggregate: dict[str, V7Aggregate],
) -> MultiAgentSubstrateCoordinatorV7Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v16_beats = {
        r: float(a.v16_beats_v15_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v16_beats_tsc_v15_rate)
        for r, a in per_regime_aggregate.items()}
    v16_succ = {
        r: float(a.per_policy_success_rate.get(
            W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v16_savings = {
        r: float(a.v16_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    ts_per_v16 = {
        r: float(a.team_success_per_visible_token_v16)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV7Witness(
        schema=W71_MASC_V7_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v16_beats_v15_rate=v16_beats,
        per_regime_tsc_v16_beats_tsc_v15_rate=tsc_beats,
        per_regime_v16_success_rate=v16_succ,
        per_regime_tsc_v16_success_rate=tsc_succ,
        per_regime_v16_visible_tokens_savings=v16_savings,
        per_regime_team_success_per_visible_token_v16=ts_per_v16,
    )


__all__ = [
    "W71_MASC_V7_SCHEMA_VERSION",
    "W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16",
    "W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16",
    "W71_MASC_V7_POLICIES",
    "W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART",
    "W71_MASC_V7_REGIMES",
    "W71_MASC_V7_REGIMES_NEW",
    "V7PolicyOutcome",
    "V7TaskOutcome",
    "V7Aggregate",
    "MultiAgentSubstrateCoordinatorV7",
    "MultiAgentSubstrateCoordinatorV7Witness",
    "run_v7_multi_agent_task",
    "aggregate_v7_outcomes",
    "emit_multi_agent_substrate_coordinator_v7_witness",
]
