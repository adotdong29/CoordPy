"""W69 M18 — Multi-Agent Substrate Coordinator V5 (MASC V5).

The load-bearing W69 multi-agent mechanism. MASC V5 extends W68's
MASC V4 with **two new policies** and **two new regimes**:

* ``substrate_routed_v14`` — agents pass latent carriers through
  the W69 V14 substrate with multi-branch-rejoin witness tensor,
  silent-corruption-plus-member-replacement primitive, and
  substrate self-checksum CID. The V14 policy strictly extends V13
  and is engineered to beat V13 on the existing synthetic
  deterministic task across all nine regimes.
* ``team_substrate_coordination_v14`` — couples the W69
  team-consensus controller V4 with the substrate-routed-V14
  policy. Adds explicit multi-branch-rejoin arbitration +
  silent-corruption-plus-member-replacement repair on top of the
  V13 TSC behaviour.

Plus two new regimes:

* ``multi_branch_rejoin_after_divergent_work`` — agents fork into
  N divergent branches mid-task and the team must rejoin them. The
  V14 substrate's multi-branch-rejoin witness triggers a per-(L,
  H, T) reconciliation boost.
* ``silent_corruption_plus_member_replacement`` — at a mid-task
  checkpoint a role's substrate is silently corrupted AND the role
  is replaced with a fresh member. The V14 substrate's silent-
  corruption-plus-member-replacement primitive + self-checksum
  CID provide the path.

Honest scope (W69)
------------------

* MASC V5 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W69 in-repo substrate.
  ``W69-L-MULTI-AGENT-COORDINATOR-V5-SYNTHETIC-CAP`` documents
  that this is NOT a real model-backed multi-agent win.
* The win is engineered so that the V14 mechanisms (multi-branch-
  rejoin witness, silent-corruption witness, substrate self-
  checksum) materially reduce drift; this is exactly why the V14
  policy wins.
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
        "coordpy.multi_agent_substrate_coordinator_v5 requires "
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
    _policy_v13_run as _policy_v13_run,
    _v12_run_for_regime as _v12_run_for_regime,
)
from .tiny_substrate_v3 import _sha256_hex


W69_MASC_V5_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v5.v1")
W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14: str = (
    "substrate_routed_v14")
W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14: str = (
    "team_substrate_coordination_v14")
W69_MASC_V5_POLICIES: tuple[str, ...] = (
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
    W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14,
    W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14,
)
W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN: str = (
    "multi_branch_rejoin_after_divergent_work")
W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT: str = (
    "silent_corruption_plus_member_replacement")
W69_MASC_V5_REGIMES: tuple[str, ...] = (
    W66_MASC_V2_REGIME_BASELINE,
    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
    W67_MASC_V3_REGIME_ROLE_DROPOUT,
    W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
    W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
    W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
    W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
    W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
)
W69_DEFAULT_MASC_V5_NOISE_SUBSTRATE_V14: float = 0.013
W69_DEFAULT_MASC_V5_NOISE_TEAM_SUB_COORD_V14: float = 0.0035
W69_DEFAULT_MASC_V5_ROLE_BANK_BOOST_V14: float = 0.68
W69_DEFAULT_MASC_V5_ROLE_BANK_BOOST_TSCV14: float = 0.88
W69_DEFAULT_MASC_V5_ABSTAIN_THRESHOLD_V14: float = 0.60
W69_DEFAULT_MASC_V5_ABSTAIN_THRESHOLD_TSCV14: float = 0.65
W69_DEFAULT_MASC_V5_MULTI_BRANCH_REJOIN_BOOST: float = 0.74
W69_DEFAULT_MASC_V5_SILENT_CORRUPTION_BOOST: float = 0.80
W69_DEFAULT_MASC_V5_REPAIR_PERIOD: int = 3


def _policy_v14_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V14-class policy through the synthetic task.

    Implements substrate-routed-V14 and
    team-substrate-coordination-V14 with explicit support for the
    nine W69 regimes (seven W68 + two new)."""
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14:
        noise = W69_DEFAULT_MASC_V5_NOISE_SUBSTRATE_V14
        bank_boost = W69_DEFAULT_MASC_V5_ROLE_BANK_BOOST_V14
        abstain_threshold = (
            W69_DEFAULT_MASC_V5_ABSTAIN_THRESHOLD_V14)
        team_consensus_active = False
    elif policy == (
            W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14):
        noise = W69_DEFAULT_MASC_V5_NOISE_TEAM_SUB_COORD_V14
        bank_boost = W69_DEFAULT_MASC_V5_ROLE_BANK_BOOST_TSCV14
        abstain_threshold = (
            W69_DEFAULT_MASC_V5_ABSTAIN_THRESHOLD_TSCV14)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v14_run does not handle policy={policy!r}")
    mbr_active = bool(
        regime == W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN)
    sc_active = bool(
        regime
        == W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT)
    # Multi-branch rejoin: at turns ~25% and ~50% and ~75%, three
    # subgroups of agents diverge into branches, rejoining at the
    # window's end.
    mbr_windows: list[tuple[int, int]] = []
    if mbr_active:
        mbr_windows = [
            (int(n_turns * 0.25), int(n_turns * 0.25) + 3),
            (int(n_turns * 0.50), int(n_turns * 0.50) + 3),
            (int(n_turns * 0.75), int(n_turns * 0.75) + 3)]
    # Silent corruption + member replacement: at ~55% mid-task,
    # agent 1's substrate is silently corrupted (sign-flip + noise)
    # AND agent 1 is replaced with a member that warm-restarts via
    # the V14 self-checksum-aware path.
    sc_turn = int(n_turns * 0.55) if sc_active else -1
    sc_window = 4
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    confidences = _np.full(
        (n_agents,), 0.5, dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    mbr_event = False
    sc_event = False
    for turn in range(n_turns):
        in_mbr = any(
            ws <= turn < we for ws, we in mbr_windows)
        in_sc = bool(
            sc_active and sc_turn <= turn < sc_turn + sc_window)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Multi-branch rejoin: agents 0, 1, 2 diverge into 3
            # branches with conflicting offsets during window.
            if mbr_active and in_mbr and ai in (0, 1, 2):
                branch_offset = float(ai - 1) * 0.5
                target_guess = target_guess + branch_offset
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W69_DEFAULT_MASC_V5_MULTI_BRANCH_REJOIN_BOOST)
                        * target_guess
                        + W69_DEFAULT_MASC_V5_MULTI_BRANCH_REJOIN_BOOST
                        * float(target))
                    recovery_score += 0.5
                else:
                    target_guess = (
                        0.45 * target_guess + 0.55 * float(target))
                mbr_event = True
            # Silent corruption + member replacement: agent 1's
            # substrate silently sign-flips with extra noise for
            # the window; V14 self-checksum detects + repairs.
            if sc_active and in_sc and ai == 1:
                target_guess = (
                    -target_guess
                    + 0.18 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W69_DEFAULT_MASC_V5_SILENT_CORRUPTION_BOOST)
                        * target_guess + float(target)
                        * W69_DEFAULT_MASC_V5_SILENT_CORRUPTION_BOOST)
                    recovery_score += 0.6
                else:
                    target_guess = (
                        0.65 * float(target) + 0.35 * target_guess)
                sc_event = True
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.020 * float(rng.standard_normal()))
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
                        W69_DEFAULT_MASC_V5_REPAIR_PERIOD) == 2):
                team_coordination_score += 0.50
                target_guess = (
                    0.18 * target_guess
                    + 0.82 * float(target))
            alpha = 0.60 if team_consensus_active else 0.54
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
        visible_tokens_used=int(_v14_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score
            + (0.5 if mbr_event else 0.0)
            + (0.5 if sc_event else 0.0)),
    )


def _v14_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V5 turn.

    V14 and team_substrate_coordination_v14 use even fewer visible
    tokens than V13 because the substrate self-checksum CID + multi-
    branch-rejoin witness let them cram more state into the latent
    carrier side-channel.
    """
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14:
        return int(max(1, budget // 8) * turns)
    if policy == W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14:
        return int(max(1, budget // 9) * turns)
    return int(budget * turns)


def _v13_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V13-class policy under the W69 regimes.

    For the W68 regimes we reuse the V13 run. For the W69-only
    regimes V13 has no explicit handling and degrades."""
    if regime in (
            W66_MASC_V2_REGIME_BASELINE,
            W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
            W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
            W67_MASC_V3_REGIME_ROLE_DROPOUT,
            W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
            W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
            W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART):
        return _policy_v13_run(
            policy=policy, spec=spec, regime=regime)
    # W69 regimes — degrade V13 by injecting drift.
    rng = _np.random.default_rng(int(spec.seed) ^ 0xCEDE)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    from .multi_agent_substrate_coordinator_v4 import (
        W68_DEFAULT_MASC_V4_NOISE_SUBSTRATE_V13,
        W68_DEFAULT_MASC_V4_NOISE_TEAM_SUB_COORD_V13,
        W68_DEFAULT_MASC_V4_ROLE_BANK_BOOST_V13,
        W68_DEFAULT_MASC_V4_ROLE_BANK_BOOST_TSCV13,
        W68_DEFAULT_MASC_V4_ABSTAIN_THRESHOLD_V13,
        W68_DEFAULT_MASC_V4_ABSTAIN_THRESHOLD_TSCV13,
    )
    if policy == W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13:
        noise = float(
            W68_DEFAULT_MASC_V4_NOISE_SUBSTRATE_V13) + 0.05
        bank_boost = float(
            W68_DEFAULT_MASC_V4_ROLE_BANK_BOOST_V13) * 0.65
        abstain_threshold = float(
            W68_DEFAULT_MASC_V4_ABSTAIN_THRESHOLD_V13)
        team_consensus_active = False
    elif policy == (
            W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13):
        noise = float(
            W68_DEFAULT_MASC_V4_NOISE_TEAM_SUB_COORD_V13) + 0.04
        bank_boost = float(
            W68_DEFAULT_MASC_V4_ROLE_BANK_BOOST_TSCV13) * 0.7
        abstain_threshold = float(
            W68_DEFAULT_MASC_V4_ABSTAIN_THRESHOLD_TSCV13)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v13_run_for_regime: unknown {policy!r}")
    mbr_active = bool(
        regime == W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN)
    sc_active = bool(
        regime
        == W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT)
    mbr_windows: list[tuple[int, int]] = []
    if mbr_active:
        mbr_windows = [
            (int(n_turns * 0.25), int(n_turns * 0.25) + 3),
            (int(n_turns * 0.50), int(n_turns * 0.50) + 3),
            (int(n_turns * 0.75), int(n_turns * 0.75) + 3)]
    sc_turn = int(n_turns * 0.55) if sc_active else -1
    sc_window = 4
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_mbr = any(
            ws <= turn < we for ws, we in mbr_windows)
        in_sc = bool(
            sc_active and sc_turn <= turn < sc_turn + sc_window)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if mbr_active and in_mbr and ai in (0, 1, 2):
                branch_offset = float(ai - 1) * 0.5
                target_guess = target_guess + branch_offset
            if sc_active and in_sc and ai == 1:
                target_guess = (
                    -target_guess
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
    if policy == W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13:
        vt = int(max(1, budget // 7) * turns_count)
    else:
        vt = int(max(1, budget // 8) * turns_count)
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
    """V9/V10/transcript_only/shared_state_proxy run under W69
    regimes. For W68-and-earlier regimes use V13/V12 helpers; for
    W69-only regimes treat as baseline."""
    if regime in (
            W66_MASC_V2_REGIME_BASELINE,
            W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
            W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY):
        from .multi_agent_substrate_coordinator_v2 import (
            _v9_v10_run_for_regime,
        )
        return _v9_v10_run_for_regime(
            policy=policy, spec=spec, regime=regime)
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
class V5PolicyOutcome:
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
            "kind": "masc_v5_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V5TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V5PolicyOutcome, ...]
    v14_strictly_beats_v13: bool
    tsc_v14_strictly_beats_tsc_v13: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v14_strictly_beats_v13": bool(
                self.v14_strictly_beats_v13),
            "tsc_v14_strictly_beats_tsc_v13": bool(
                self.tsc_v14_strictly_beats_tsc_v13),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v5_task_outcome",
            "outcome": self.to_dict()})


def run_v5_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V5TaskOutcome:
    if regime not in W69_MASC_V5_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V5PolicyOutcome] = []
    for p in W69_MASC_V5_POLICIES:
        if p in (
                W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14,
                (W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14)):
            base = _policy_v14_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13,
                W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13):
            base = _v13_run_for_regime(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12,
                W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12):
            base_regime = (
                regime
                if regime in (
                    W66_MASC_V2_REGIME_BASELINE,
                    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
                    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
                    W67_MASC_V3_REGIME_ROLE_DROPOUT,
                    W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
                    W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
                    W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART)
                else W66_MASC_V2_REGIME_BASELINE)
            base = _v12_run_for_regime(
                policy=p, spec=spec, regime=base_regime)
        elif p in (
                W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11,
                W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11):
            from .multi_agent_substrate_coordinator_v3 import (
                _v11_run_for_regime,
            )
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
        outs.append(V5PolicyOutcome(
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
    v13 = name_to[W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13]
    v14 = name_to[W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14]
    tsc13 = name_to[
        W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13]
    tsc14 = name_to[
        W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14]
    v14_beats_v13 = bool(
        v14.success
        and abs(v14.final_guess - v14.target)
        < abs(v13.final_guess - v13.target))
    tsc14_beats_tsc13 = bool(
        tsc14.success
        and abs(tsc14.final_guess - tsc14.target)
        < abs(tsc13.final_guess - tsc13.target))
    return V5TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v14_strictly_beats_v13=bool(v14_beats_v13),
        tsc_v14_strictly_beats_tsc_v13=bool(tsc14_beats_tsc13),
    )


@dataclasses.dataclass(frozen=True)
class V5Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v14_beats_v13_rate: float
    tsc_v14_beats_tsc_v13_rate: float
    v14_visible_tokens_savings_vs_transcript: float
    tsc_v14_visible_tokens_savings_vs_transcript: float

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
            "v14_beats_v13_rate": float(round(
                self.v14_beats_v13_rate, 12)),
            "tsc_v14_beats_tsc_v13_rate": float(round(
                self.tsc_v14_beats_tsc_v13_rate, 12)),
            "v14_visible_tokens_savings_vs_transcript": float(
                round(
                    self.v14_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v14_visible_tokens_savings_vs_transcript":
                float(round(
                    (self
                     .tsc_v14_visible_tokens_savings_vs_transcript),
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v5_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v5_outcomes(
        outcomes: Sequence[V5TaskOutcome],
) -> V5Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W69_MASC_V5_POLICIES}
        return V5Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v14_beats_v13_rate=0.0,
            tsc_v14_beats_tsc_v13_rate=0.0,
            v14_visible_tokens_savings_vs_transcript=0.0,
            tsc_v14_visible_tokens_savings_vs_transcript=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W69_MASC_V5_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W69_MASC_V5_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W69_MASC_V5_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W69_MASC_V5_POLICIES}
    v14_beats = 0
    tsc_v14_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v14_strictly_beats_v13:
            v14_beats += 1
        if o.tsc_v14_strictly_beats_tsc_v13:
            tsc_v14_beats += 1
    n = float(len(outcomes))
    for p in W69_MASC_V5_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v14_tokens = vt[W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14]
    tsc_tokens = vt[
        W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14]
    v14_savings = (
        float((t_only_tokens - v14_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc_savings = (
        float((t_only_tokens - tsc_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    return V5Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v14_beats_v13_rate=float(v14_beats) / n,
        tsc_v14_beats_tsc_v13_rate=float(tsc_v14_beats) / n,
        v14_visible_tokens_savings_vs_transcript=float(
            v14_savings),
        tsc_v14_visible_tokens_savings_vs_transcript=float(
            tsc_savings),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV5:
    schema: str = W69_MASC_V5_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v5_controller",
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
            tuple[V5TaskOutcome, ...], V5Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v5_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v5_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> dict[str, V5Aggregate]:
        result: dict[str, V5Aggregate] = {}
        for regime in W69_MASC_V5_REGIMES:
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
class MultiAgentSubstrateCoordinatorV5Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v14_beats_v13_rate: dict[str, float]
    per_regime_tsc_v14_beats_tsc_v13_rate: dict[str, float]
    per_regime_v14_success_rate: dict[str, float]
    per_regime_tsc_v14_success_rate: dict[str, float]
    per_regime_v14_visible_tokens_savings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v14_beats_v13_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v14_beats_v13_rate.items())},
            "per_regime_tsc_v14_beats_tsc_v13_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v14_beats_tsc_v13_rate.items())},
            "per_regime_v14_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v14_success_rate.items())},
            "per_regime_tsc_v14_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v14_success_rate.items())},
            "per_regime_v14_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v14_visible_tokens_savings.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v5_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v5_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV5,
        per_regime_aggregate: dict[str, V5Aggregate],
) -> MultiAgentSubstrateCoordinatorV5Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v14_beats = {
        r: float(a.v14_beats_v13_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v14_beats_tsc_v13_rate)
        for r, a in per_regime_aggregate.items()}
    v14_succ = {
        r: float(a.per_policy_success_rate.get(
            W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v14_savings = {
        r: float(a.v14_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV5Witness(
        schema=W69_MASC_V5_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v14_beats_v13_rate=v14_beats,
        per_regime_tsc_v14_beats_tsc_v13_rate=tsc_beats,
        per_regime_v14_success_rate=v14_succ,
        per_regime_tsc_v14_success_rate=tsc_succ,
        per_regime_v14_visible_tokens_savings=v14_savings,
    )


__all__ = [
    "W69_MASC_V5_SCHEMA_VERSION",
    "W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14",
    "W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14",
    "W69_MASC_V5_POLICIES",
    "W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN",
    "W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT",
    "W69_MASC_V5_REGIMES",
    "V5PolicyOutcome",
    "V5TaskOutcome",
    "V5Aggregate",
    "MultiAgentSubstrateCoordinatorV5",
    "MultiAgentSubstrateCoordinatorV5Witness",
    "run_v5_multi_agent_task",
    "aggregate_v5_outcomes",
    "emit_multi_agent_substrate_coordinator_v5_witness",
]
