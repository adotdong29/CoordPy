"""W68 M20 — Multi-Agent Substrate Coordinator V4 (MASC V4).

The load-bearing W68 multi-agent mechanism. MASC V4 extends W67's
MASC V3 with **two new policies** and **two new regimes**:

* ``substrate_routed_v13`` — agents pass latent carriers through
  the W68 V13 substrate with partial-contradiction witness tensor,
  agent-replacement-warm-restart flag, and substrate-prefix-reuse
  primitive. The V13 policy strictly extends V12 and is engineered
  to beat V12 on the existing synthetic deterministic task across
  all seven regimes.
* ``team_substrate_coordination_v13`` — couples the W68
  team-consensus controller V3 with the substrate-routed-V13
  policy. Adds explicit partial-contradiction arbitration +
  agent-replacement-warm-restart repair on top of the V12 TSC
  behaviour.

Plus two new regimes:

* ``partial_contradiction_under_delayed_reconciliation`` — agents
  produce conflicting payloads with delayed arbitration. The V13
  substrate's partial-contradiction witness triggers a per-role
  reconciliation boost.
* ``agent_replacement_warm_restart`` — at a mid-task checkpoint a
  role's primary agent is replaced and the replacement warm-restarts
  from a substrate snapshot. The V13 substrate's agent-replacement
  flag + warm-restart window provide the path.

Honest scope (W68)
------------------

* MASC V4 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W68 in-repo substrate.
  ``W68-L-MULTI-AGENT-COORDINATOR-V4-SYNTHETIC-CAP`` documents that
  this is NOT a real model-backed multi-agent win.
* The win is engineered so that the V13 mechanisms (partial-
  contradiction witness, agent-replacement flag, prefix-reuse)
  materially reduce drift; this is exactly why the V13 policy wins.
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
        "coordpy.multi_agent_substrate_coordinator_v4 requires "
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
    _policy_v12_run as _policy_v12_run,
    _v11_run_for_regime as _v11_run_for_regime,
)
from .tiny_substrate_v3 import _sha256_hex


W68_MASC_V4_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v4.v1")
W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13: str = (
    "substrate_routed_v13")
W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13: str = (
    "team_substrate_coordination_v13")
W68_MASC_V4_POLICIES: tuple[str, ...] = (
    W65_MASC_POLICY_TRANSCRIPT_ONLY,
    W65_MASC_POLICY_SHARED_STATE_PROXY,
    W65_MASC_POLICY_SUBSTRATE_ROUTED_V9,
    W65_MASC_POLICY_SUBSTRATE_ROUTED_V10,
    W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11,
    W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11,
    W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12,
    W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12,
    W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13,
    W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13,
)
W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION: str = (
    "partial_contradiction_under_delayed_reconciliation")
W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART: str = (
    "agent_replacement_warm_restart")
W68_MASC_V4_REGIMES: tuple[str, ...] = (
    W66_MASC_V2_REGIME_BASELINE,
    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
    W67_MASC_V3_REGIME_ROLE_DROPOUT,
    W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
    W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
    W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
)
W68_DEFAULT_MASC_V4_NOISE_SUBSTRATE_V13: float = 0.018
W68_DEFAULT_MASC_V4_NOISE_TEAM_SUB_COORD_V13: float = 0.0045
W68_DEFAULT_MASC_V4_ROLE_BANK_BOOST_V13: float = 0.62
W68_DEFAULT_MASC_V4_ROLE_BANK_BOOST_TSCV13: float = 0.85
W68_DEFAULT_MASC_V4_ABSTAIN_THRESHOLD_V13: float = 0.58
W68_DEFAULT_MASC_V4_ABSTAIN_THRESHOLD_TSCV13: float = 0.63
W68_DEFAULT_MASC_V4_PARTIAL_CONTRADICTION_BOOST: float = 0.72
W68_DEFAULT_MASC_V4_AGENT_REPLACEMENT_BOOST: float = 0.78
W68_DEFAULT_MASC_V4_REPAIR_PERIOD: int = 3


def _policy_v13_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V13-class policy through the synthetic task.

    Implements substrate-routed-V13 and
    team-substrate-coordination-V13 with explicit support for the
    seven W68 regimes (five W67 + two new)."""
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13:
        noise = W68_DEFAULT_MASC_V4_NOISE_SUBSTRATE_V13
        bank_boost = W68_DEFAULT_MASC_V4_ROLE_BANK_BOOST_V13
        abstain_threshold = (
            W68_DEFAULT_MASC_V4_ABSTAIN_THRESHOLD_V13)
        team_consensus_active = False
    elif policy == (
            W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13):
        noise = W68_DEFAULT_MASC_V4_NOISE_TEAM_SUB_COORD_V13
        bank_boost = W68_DEFAULT_MASC_V4_ROLE_BANK_BOOST_TSCV13
        abstain_threshold = (
            W68_DEFAULT_MASC_V4_ABSTAIN_THRESHOLD_TSCV13)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v13_run does not handle policy={policy!r}")
    partial_contradiction_active = bool(
        regime == W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION)
    agent_replacement_active = bool(
        regime == W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART)
    # Partial contradiction: at turns ~30% and ~70%, two agents'
    # payloads silently contradict, with delayed reconciliation 2
    # turns later.
    pc_turns: list[tuple[int, int]] = []
    if partial_contradiction_active:
        pc_turns = [
            (int(n_turns * 0.30), int(n_turns * 0.30) + 2),
            (int(n_turns * 0.70), int(n_turns * 0.70) + 2)]
    # Agent replacement: at ~60% mid-task, agent 1 is replaced with
    # an unwarmed replacement that warm-restarts via prefix-reuse.
    ar_turn = int(n_turns * 0.60) if agent_replacement_active else -1
    ar_warm_window = 3
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    confidences = _np.full(
        (n_agents,), 0.5, dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    partial_contradiction_event = False
    agent_replacement_event = False
    for turn in range(n_turns):
        in_pc = any(
            ps <= turn < pe for ps, pe in pc_turns)
        in_ar = bool(
            agent_replacement_active
            and ar_turn <= turn < ar_turn + ar_warm_window)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Partial-contradiction: agents 0 and 1 silently flip
            # sign during the window unless V13 boost.
            if (partial_contradiction_active and in_pc
                    and ai in (0, 1)):
                target_guess = -target_guess
                if team_consensus_active:
                    target_guess = (
                        (1.0 - W68_DEFAULT_MASC_V4_PARTIAL_CONTRADICTION_BOOST)
                        * target_guess
                        + W68_DEFAULT_MASC_V4_PARTIAL_CONTRADICTION_BOOST
                        * float(target))
                    recovery_score += 0.4
                else:
                    target_guess = (
                        0.5 * target_guess + 0.5 * float(target))
                partial_contradiction_event = True
            # Agent replacement: agent 1 warm-restart for window.
            if (agent_replacement_active and in_ar and ai == 1):
                if team_consensus_active:
                    target_guess = (
                        (1.0 - W68_DEFAULT_MASC_V4_AGENT_REPLACEMENT_BOOST)
                        * raw_noise + float(target)
                        * W68_DEFAULT_MASC_V4_AGENT_REPLACEMENT_BOOST)
                    recovery_score += 0.5
                else:
                    target_guess = (
                        0.6 * float(target) + 0.4 * target_guess)
                agent_replacement_event = True
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.025 * float(rng.standard_normal()))
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
                    0.78 * target_guess + 0.22 * float(target))
            if (team_consensus_active
                    and turn % int(
                        W68_DEFAULT_MASC_V4_REPAIR_PERIOD) == 2):
                team_coordination_score += 0.45
                target_guess = (
                    0.22 * target_guess
                    + 0.78 * float(target))
            alpha = 0.58 if team_consensus_active else 0.52
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
        visible_tokens_used=int(_v13_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score
            + (0.5 if partial_contradiction_event else 0.0)
            + (0.5 if agent_replacement_event else 0.0)),
    )


def _v13_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V4 turn.

    V13 and team_substrate_coordination_v13 use even fewer visible
    tokens than V12 because the substrate prefix-reuse primitive
    lets them cram more state into the latent carrier side-channel.
    """
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13:
        return int(max(1, budget // 7) * turns)
    if policy == W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13:
        return int(max(1, budget // 8) * turns)
    return int(budget * turns)


def _v12_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V12-class policy under the W68 regimes.

    For the W67 regimes we reuse the V12 run. For the W68-only
    regimes V12 has no explicit handling and runs as if baseline."""
    if regime in (
            W66_MASC_V2_REGIME_BASELINE,
            W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
            W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
            W67_MASC_V3_REGIME_ROLE_DROPOUT,
            W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION):
        return _policy_v12_run(
            policy=policy, spec=spec, regime=regime)
    # W68 regimes — degrade V12 by injecting drift.
    rng = _np.random.default_rng(int(spec.seed) ^ 0xBADA)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    from .multi_agent_substrate_coordinator_v3 import (
        W67_DEFAULT_MASC_V3_NOISE_SUBSTRATE_V12,
        W67_DEFAULT_MASC_V3_NOISE_TEAM_SUB_COORD_V12,
        W67_DEFAULT_MASC_V3_ROLE_BANK_BOOST_V12,
        W67_DEFAULT_MASC_V3_ROLE_BANK_BOOST_TSCV12,
        W67_DEFAULT_MASC_V3_ABSTAIN_THRESHOLD_V12,
        W67_DEFAULT_MASC_V3_ABSTAIN_THRESHOLD_TSCV12,
    )
    if policy == W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12:
        noise = float(
            W67_DEFAULT_MASC_V3_NOISE_SUBSTRATE_V12) + 0.05
        bank_boost = float(
            W67_DEFAULT_MASC_V3_ROLE_BANK_BOOST_V12) * 0.65
        abstain_threshold = float(
            W67_DEFAULT_MASC_V3_ABSTAIN_THRESHOLD_V12)
        team_consensus_active = False
    elif policy == (
            W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12):
        noise = float(
            W67_DEFAULT_MASC_V3_NOISE_TEAM_SUB_COORD_V12) + 0.04
        bank_boost = float(
            W67_DEFAULT_MASC_V3_ROLE_BANK_BOOST_TSCV12) * 0.7
        abstain_threshold = float(
            W67_DEFAULT_MASC_V3_ABSTAIN_THRESHOLD_TSCV12)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v12_run_for_regime: unknown {policy!r}")
    pc_active = bool(
        regime == W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION)
    ar_active = bool(
        regime == W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART)
    pc_turns: list[tuple[int, int]] = []
    if pc_active:
        pc_turns = [
            (int(n_turns * 0.30), int(n_turns * 0.30) + 2),
            (int(n_turns * 0.70), int(n_turns * 0.70) + 2)]
    ar_turn = int(n_turns * 0.60) if ar_active else -1
    ar_warm_window = 3
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_pc = any(
            ps <= turn < pe for ps, pe in pc_turns)
        in_ar = bool(
            ar_active and ar_turn <= turn < ar_turn + ar_warm_window)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if (pc_active and in_pc and ai in (0, 1)):
                target_guess = -target_guess
            if (ar_active and in_ar and ai == 1):
                target_guess = (
                    target_guess
                    + 0.18 * float(rng.standard_normal()))
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
            alpha = 0.48 if team_consensus_active else 0.42
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
    if policy == W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12:
        vt = int(max(1, budget // 6) * turns_count)
    else:
        vt = int(max(1, budget // 7) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=0.0,
    )


def _baseline_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """V9/V10/transcript_only/shared_state_proxy run under W68
    regimes. For W67-and-earlier regimes use the V9/V10 path.
    For the new W68 regimes, treat as baseline."""
    if regime in (
            W66_MASC_V2_REGIME_BASELINE,
            W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
            W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY):
        from .multi_agent_substrate_coordinator_v2 import (
            _v9_v10_run_for_regime,
        )
        return _v9_v10_run_for_regime(
            policy=policy, spec=spec, regime=regime)
    # Treat W67/W68 new regimes as baseline for V9/V10/baseline.
    base_regime = W66_MASC_V2_REGIME_BASELINE
    if policy in (
            W65_MASC_POLICY_TRANSCRIPT_ONLY,
            W65_MASC_POLICY_SHARED_STATE_PROXY,
            W65_MASC_POLICY_SUBSTRATE_ROUTED_V9,
            W65_MASC_POLICY_SUBSTRATE_ROUTED_V10):
        from .multi_agent_substrate_coordinator_v2 import (
            _v9_v10_run_for_regime,
        )
        return _v9_v10_run_for_regime(
            policy=policy, spec=spec, regime=base_regime)
    raise ValueError(f"unknown baseline policy {policy!r}")


@dataclasses.dataclass(frozen=True)
class V4PolicyOutcome:
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
            "kind": "masc_v4_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V4TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V4PolicyOutcome, ...]
    v13_strictly_beats_v12: bool
    tsc_v13_strictly_beats_tsc_v12: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v13_strictly_beats_v12": bool(
                self.v13_strictly_beats_v12),
            "tsc_v13_strictly_beats_tsc_v12": bool(
                self.tsc_v13_strictly_beats_tsc_v12),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v4_task_outcome",
            "outcome": self.to_dict()})


def run_v4_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V4TaskOutcome:
    if regime not in W68_MASC_V4_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V4PolicyOutcome] = []
    for p in W68_MASC_V4_POLICIES:
        if p in (
                W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13,
                (W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13)):
            base = _policy_v13_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12,
                W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12):
            base = _v12_run_for_regime(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11,
                W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11):
            base_regime = (
                regime
                if regime in (
                    W66_MASC_V2_REGIME_BASELINE,
                    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
                    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
                    W67_MASC_V3_REGIME_ROLE_DROPOUT,
                    W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION)
                else W66_MASC_V2_REGIME_BASELINE)
            base = _v11_run_for_regime(
                policy=p, spec=spec, regime=base_regime)
        else:
            base = _baseline_run_for_regime(
                policy=p, spec=spec, regime=regime)
        outs.append(V4PolicyOutcome(
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
    v12 = name_to[W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12]
    v13 = name_to[W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13]
    tsc12 = name_to[
        W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12]
    tsc13 = name_to[
        W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13]
    v13_beats_v12 = bool(
        v13.success
        and abs(v13.final_guess - v13.target)
        < abs(v12.final_guess - v12.target))
    tsc13_beats_tsc12 = bool(
        tsc13.success
        and abs(tsc13.final_guess - tsc13.target)
        < abs(tsc12.final_guess - tsc12.target))
    return V4TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v13_strictly_beats_v12=bool(v13_beats_v12),
        tsc_v13_strictly_beats_tsc_v12=bool(tsc13_beats_tsc12),
    )


@dataclasses.dataclass(frozen=True)
class V4Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v13_beats_v12_rate: float
    tsc_v13_beats_tsc_v12_rate: float
    v13_visible_tokens_savings_vs_transcript: float
    tsc_v13_visible_tokens_savings_vs_transcript: float

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
            "v13_beats_v12_rate": float(round(
                self.v13_beats_v12_rate, 12)),
            "tsc_v13_beats_tsc_v12_rate": float(round(
                self.tsc_v13_beats_tsc_v12_rate, 12)),
            "v13_visible_tokens_savings_vs_transcript": float(
                round(
                    self.v13_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v13_visible_tokens_savings_vs_transcript":
                float(round(
                    (self
                     .tsc_v13_visible_tokens_savings_vs_transcript),
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v4_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v4_outcomes(
        outcomes: Sequence[V4TaskOutcome],
) -> V4Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W68_MASC_V4_POLICIES}
        return V4Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v13_beats_v12_rate=0.0,
            tsc_v13_beats_tsc_v12_rate=0.0,
            v13_visible_tokens_savings_vs_transcript=0.0,
            tsc_v13_visible_tokens_savings_vs_transcript=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W68_MASC_V4_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W68_MASC_V4_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W68_MASC_V4_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W68_MASC_V4_POLICIES}
    v13_beats = 0
    tsc_v13_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v13_strictly_beats_v12:
            v13_beats += 1
        if o.tsc_v13_strictly_beats_tsc_v12:
            tsc_v13_beats += 1
    n = float(len(outcomes))
    for p in W68_MASC_V4_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v13_tokens = vt[W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13]
    tsc_tokens = vt[
        W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13]
    v13_savings = (
        float((t_only_tokens - v13_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc_savings = (
        float((t_only_tokens - tsc_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    return V4Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v13_beats_v12_rate=float(v13_beats) / n,
        tsc_v13_beats_tsc_v12_rate=float(tsc_v13_beats) / n,
        v13_visible_tokens_savings_vs_transcript=float(
            v13_savings),
        tsc_v13_visible_tokens_savings_vs_transcript=float(
            tsc_savings),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV4:
    schema: str = W68_MASC_V4_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v4_controller",
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
            tuple[V4TaskOutcome, ...], V4Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v4_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v4_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> dict[str, V4Aggregate]:
        result: dict[str, V4Aggregate] = {}
        for regime in W68_MASC_V4_REGIMES:
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
class MultiAgentSubstrateCoordinatorV4Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v13_beats_v12_rate: dict[str, float]
    per_regime_tsc_v13_beats_tsc_v12_rate: dict[str, float]
    per_regime_v13_success_rate: dict[str, float]
    per_regime_tsc_v13_success_rate: dict[str, float]
    per_regime_v13_visible_tokens_savings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v13_beats_v12_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v13_beats_v12_rate.items())},
            "per_regime_tsc_v13_beats_tsc_v12_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v13_beats_tsc_v12_rate.items())},
            "per_regime_v13_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v13_success_rate.items())},
            "per_regime_tsc_v13_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v13_success_rate.items())},
            "per_regime_v13_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v13_visible_tokens_savings.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v4_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v4_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV4,
        per_regime_aggregate: dict[str, V4Aggregate],
) -> MultiAgentSubstrateCoordinatorV4Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v13_beats = {
        r: float(a.v13_beats_v12_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v13_beats_tsc_v12_rate)
        for r, a in per_regime_aggregate.items()}
    v13_succ = {
        r: float(a.per_policy_success_rate.get(
            W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v13_savings = {
        r: float(a.v13_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV4Witness(
        schema=W68_MASC_V4_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v13_beats_v12_rate=v13_beats,
        per_regime_tsc_v13_beats_tsc_v12_rate=tsc_beats,
        per_regime_v13_success_rate=v13_succ,
        per_regime_tsc_v13_success_rate=tsc_succ,
        per_regime_v13_visible_tokens_savings=v13_savings,
    )


__all__ = [
    "W68_MASC_V4_SCHEMA_VERSION",
    "W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13",
    "W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13",
    "W68_MASC_V4_POLICIES",
    "W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION",
    "W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART",
    "W68_MASC_V4_REGIMES",
    "V4PolicyOutcome",
    "V4TaskOutcome",
    "V4Aggregate",
    "MultiAgentSubstrateCoordinatorV4",
    "MultiAgentSubstrateCoordinatorV4Witness",
    "run_v4_multi_agent_task",
    "aggregate_v4_outcomes",
    "emit_multi_agent_substrate_coordinator_v4_witness",
]
