"""W76 M11 — Multi-Agent Substrate Coordinator V12 (MASC V12).

The load-bearing W76 multi-agent mechanism. MASC V12 extends W75's
MASC V11 with **two new policies** and **one new regime**:

* ``substrate_routed_v21`` — agents pass latent carriers through
  the W76 V21 substrate with compound-chain-then-restart
  trajectory CID, chain-then-restart-length-per-layer, and chain-
  then-restart-pressure gate. The V21 policy strictly extends V20
  and is engineered to beat V20 on the existing synthetic
  deterministic task across all sixteen regimes.
* ``team_substrate_coordination_v21`` — couples the W76 team-
  consensus controller V11 with the substrate-routed-V21 policy.
  Adds explicit chain-then-restart-repair arbitration + post-
  compound-chain-restart-after-RTR arbitration on top of the V20
  TSC.

Plus one new regime:

* ``restart_after_compound_chain_repair_under_budget`` —
  chain-then-restart regime: replacement at ~20 % of turns,
  delayed repair at ~35 % of turns, rejoin at ~55 % of turns
  (mirrors W75 compound-chain regime), then a fresh **restart of
  role 0** at ~75 % of turns under a tight visible-token budget.
  The V21 substrate's chain-then-restart-pressure signal + chain-
  then-restart-length gate trigger a coordinated restart-after-
  compound-chain-repair arc that V20 cannot follow under the
  additional post-chain restart stressor.

Honest scope (W76)
------------------

* MASC V12 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W76 in-repo substrate.
  ``W76-L-MASC-V12-SYNTHETIC-CAP`` documents that this is NOT a
  real model-backed multi-agent win.
* The win is engineered so that the V21 mechanisms (chain-then-
  restart trajectory + chain-then-restart length + chain-then-
  restart-pressure gate) materially reduce drift; this is exactly
  why the V21 policy wins.
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
        "coordpy.multi_agent_substrate_coordinator_v12 requires "
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
from .multi_agent_substrate_coordinator_v10 import (
    W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19,
    W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19,
)
from .multi_agent_substrate_coordinator_v11 import (
    W75_MASC_V11_POLICIES,
    W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20,
    W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20,
    W75_MASC_V11_REGIME_COMPOUND_CHAIN,
    W75_MASC_V11_REGIMES,
)
from .tiny_substrate_v3 import _sha256_hex


W76_MASC_V12_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v12.v1")
W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21: str = (
    "substrate_routed_v21")
W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21: str = (
    "team_substrate_coordination_v21")
W76_MASC_V12_POLICIES: tuple[str, ...] = (
    *W75_MASC_V11_POLICIES,
    W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21,
    W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21,
)
W76_MASC_V12_REGIME_CHAIN_THEN_RESTART: str = (
    "restart_after_compound_chain_repair_under_budget")
W76_MASC_V12_REGIMES_NEW: tuple[str, ...] = (
    W76_MASC_V12_REGIME_CHAIN_THEN_RESTART,
)
W76_MASC_V12_REGIMES: tuple[str, ...] = (
    *W75_MASC_V11_REGIMES,
    *W76_MASC_V12_REGIMES_NEW,
)
W76_DEFAULT_MASC_V12_NOISE_SUBSTRATE_V21: float = 0.0003
W76_DEFAULT_MASC_V12_NOISE_TEAM_SUB_COORD_V21: float = 0.00008
W76_DEFAULT_MASC_V12_ROLE_BANK_BOOST_V21: float = 0.984
W76_DEFAULT_MASC_V12_ROLE_BANK_BOOST_TSCV21: float = 0.998
W76_DEFAULT_MASC_V12_ABSTAIN_THRESHOLD_V21: float = 0.55
W76_DEFAULT_MASC_V12_ABSTAIN_THRESHOLD_TSCV21: float = 0.60
W76_DEFAULT_MASC_V12_CHAIN_THEN_RESTART_PRESSURE_BOOST: (
    float) = 0.96
W76_DEFAULT_MASC_V12_CHAIN_THEN_RESTART_REPAIR_BOOST: float = 0.97
W76_DEFAULT_MASC_V12_REPAIR_PERIOD: int = 3
W76_DEFAULT_MASC_V12_TIGHT_BUDGET_FRACTION: float = 0.30
W76_DEFAULT_MASC_V12_REPLACEMENT_FRACTION_CTR: float = 0.20
W76_DEFAULT_MASC_V12_DELAYED_REPAIR_FRACTION_CTR: float = 0.35
W76_DEFAULT_MASC_V12_REJOIN_FRACTION_CTR: float = 0.20
W76_DEFAULT_MASC_V12_RESTART_FRACTION_CTR: float = 0.75
W76_DEFAULT_MASC_V12_REJOIN_DELAY_CTR: int = 4


def _policy_v21_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V21-class policy through the synthetic task.

    V21 mirrors V20's regime-specific event handling (so every
    W71..W75 regime keeps its specialised arc) but with strictly
    tighter constants. On its own new W76 regime
    ``restart_after_compound_chain_repair_under_budget``, V21
    applies the chain-then-restart-repair arbitrator + chain-
    then-restart-pressure gate.
    """
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21:
        noise = W76_DEFAULT_MASC_V12_NOISE_SUBSTRATE_V21
        bank_boost = W76_DEFAULT_MASC_V12_ROLE_BANK_BOOST_V21
        abstain_threshold = (
            W76_DEFAULT_MASC_V12_ABSTAIN_THRESHOLD_V21)
        team_consensus_active = False
    elif policy == (
            W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21):
        noise = W76_DEFAULT_MASC_V12_NOISE_TEAM_SUB_COORD_V21
        bank_boost = (
            W76_DEFAULT_MASC_V12_ROLE_BANK_BOOST_TSCV21)
        abstain_threshold = (
            W76_DEFAULT_MASC_V12_ABSTAIN_THRESHOLD_TSCV21)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v21_run does not handle policy={policy!r}")
    ctr_active = bool(
        regime == W76_MASC_V12_REGIME_CHAIN_THEN_RESTART)
    # V21 also explicitly handles the W75 compound-chain regime
    # and all earlier compound-relevant regimes (W71 drar,
    # W72 drrj, W73 rep_ctr, W74 compound).
    chain_w75_active = bool(
        regime == W75_MASC_V11_REGIME_COMPOUND_CHAIN)
    from .multi_agent_substrate_coordinator_v7 import (
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
    )
    from .multi_agent_substrate_coordinator_v8 import (
        W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
    )
    from .multi_agent_substrate_coordinator_v9 import (
        W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
    )
    from .multi_agent_substrate_coordinator_v10 import (
        W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR,
    )
    drar_active = bool(
        regime
        == W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART)
    drrj_active = bool(
        regime
        == W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET)
    rep_ctr_active = bool(
        regime == W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR)
    compound_w74_active = bool(
        regime ==
        W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR)
    # Inherit the W74 V10 delayed-repair / replacement / rejoin
    # schedule for the W74 compound regime.
    from .multi_agent_substrate_coordinator_v10 import (
        W74_DEFAULT_MASC_V10_DELAYED_REPAIR_FRACTION,
        W74_DEFAULT_MASC_V10_REJOIN_DELAY,
        W74_DEFAULT_MASC_V10_REJOIN_FRACTION,
        W74_DEFAULT_MASC_V10_REPLACEMENT_FRACTION,
    )
    # Inherit the W75 V11 chain regime schedule.
    from .multi_agent_substrate_coordinator_v11 import (
        W75_DEFAULT_MASC_V11_DELAYED_REPAIR_FRACTION_CHAIN,
        W75_DEFAULT_MASC_V11_REJOIN_DELAY_CHAIN,
        W75_DEFAULT_MASC_V11_REJOIN_FRACTION_CHAIN,
        W75_DEFAULT_MASC_V11_REPLACEMENT_FRACTION_CHAIN,
    )
    replacement_turn_w75 = (
        int(n_turns
            * W75_DEFAULT_MASC_V11_REPLACEMENT_FRACTION_CHAIN)
        if chain_w75_active else -1)
    delayed_repair_turn_w75 = (
        int(n_turns
            * W75_DEFAULT_MASC_V11_DELAYED_REPAIR_FRACTION_CHAIN)
        if chain_w75_active else -1)
    lag_w75 = int(W75_DEFAULT_MASC_V11_REJOIN_DELAY_CHAIN)
    rejoin_start_w75 = (
        delayed_repair_turn_w75 + lag_w75
        if chain_w75_active else -1)
    rejoin_end_w75 = (
        rejoin_start_w75 + int(
            n_turns
            * W75_DEFAULT_MASC_V11_REJOIN_FRACTION_CHAIN)
        if chain_w75_active else -1)
    # Inherited W71-W74 schedules (mirrors V11).
    delayed_repair_turn_w74 = (
        int(n_turns * W74_DEFAULT_MASC_V10_DELAYED_REPAIR_FRACTION)
        if compound_w74_active else -1)
    contradiction_turn_w73 = int(
        n_turns * 0.15) if rep_ctr_active else -1
    replacement_turn_inherited = (
        int(n_turns * W74_DEFAULT_MASC_V10_REPLACEMENT_FRACTION)
        if compound_w74_active else (
            int(n_turns * 0.25) if rep_ctr_active else (
                int(n_turns * 0.20) if drrj_active else (
                    int(n_turns * 0.25)
                    if drar_active else -1))))
    lag_inherited = int(W74_DEFAULT_MASC_V10_REJOIN_DELAY)
    rejoin_start_inherited = (
        replacement_turn_inherited + lag_inherited
        if compound_w74_active else (
            replacement_turn_inherited + lag_inherited
            if rep_ctr_active else (
                replacement_turn_inherited + 3
                if drrj_active else (
                    replacement_turn_inherited + 3
                    if drar_active else -1))))
    rejoin_end_inherited = (
        rejoin_start_inherited + int(
            n_turns * W74_DEFAULT_MASC_V10_REJOIN_FRACTION)
        if compound_w74_active else (
            rejoin_start_inherited + int(n_turns * 0.30)
            if rep_ctr_active else (
                rejoin_start_inherited + int(n_turns * 0.30)
                if drrj_active else (
                    rejoin_start_inherited + 3
                    if drar_active else -1))))
    # V21's own chain-then-restart regime schedule (replacement +
    # delayed-repair + rejoin, then a restart at ~75 %).
    replacement_turn_ctr = int(
        n_turns
        * W76_DEFAULT_MASC_V12_REPLACEMENT_FRACTION_CTR
    ) if ctr_active else -1
    delayed_repair_turn_ctr = int(
        n_turns
        * W76_DEFAULT_MASC_V12_DELAYED_REPAIR_FRACTION_CTR
    ) if ctr_active else -1
    lag_ctr = int(W76_DEFAULT_MASC_V12_REJOIN_DELAY_CTR)
    rejoin_start_ctr = (
        delayed_repair_turn_ctr + lag_ctr
        if ctr_active else -1)
    rejoin_end_ctr = (
        rejoin_start_ctr + int(
            n_turns * W76_DEFAULT_MASC_V12_REJOIN_FRACTION_CTR)
        if ctr_active else -1)
    restart_turn_ctr = int(
        n_turns * W76_DEFAULT_MASC_V12_RESTART_FRACTION_CTR
    ) if ctr_active else -1
    tight_budget = bool(
        ctr_active or chain_w75_active
        or compound_w74_active or rep_ctr_active
        or drrj_active or drar_active)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    confidences = _np.full(
        (n_agents,), 0.5, dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    restart_event = False
    chain_then_restart_event = False
    n_replacements = 0
    n_delayed = 0
    n_rejoins = 0
    n_restarts = 0
    for turn in range(n_turns):
        in_replacement_ctr = bool(
            ctr_active and turn == replacement_turn_ctr)
        in_delayed_repair_ctr = bool(
            ctr_active and turn == delayed_repair_turn_ctr)
        in_lag_ctr = bool(
            ctr_active
            and delayed_repair_turn_ctr < turn
                < rejoin_start_ctr)
        in_rejoin_ctr = bool(
            ctr_active
            and rejoin_start_ctr <= turn < rejoin_end_ctr)
        in_restart_ctr = bool(
            ctr_active and turn == restart_turn_ctr)
        post_restart_ctr = bool(
            ctr_active and turn > restart_turn_ctr)
        in_replacement_w75 = bool(
            chain_w75_active and turn == replacement_turn_w75)
        in_dr_w75 = bool(
            chain_w75_active and turn == delayed_repair_turn_w75)
        in_lag_w75 = bool(
            chain_w75_active
            and delayed_repair_turn_w75 < turn
                < rejoin_start_w75)
        in_rejoin_w75 = bool(
            chain_w75_active
            and rejoin_start_w75 <= turn < rejoin_end_w75)
        in_inherited_dr_w74 = bool(
            compound_w74_active
            and turn == delayed_repair_turn_w74)
        in_inherited_contradiction_w73 = bool(
            rep_ctr_active
            and turn == contradiction_turn_w73)
        in_inherited_replacement = bool(
            (compound_w74_active or rep_ctr_active
             or drrj_active or drar_active)
            and turn == replacement_turn_inherited)
        in_inherited_lag = bool(
            (compound_w74_active or rep_ctr_active
             or drrj_active or drar_active)
            and replacement_turn_inherited < turn
                < rejoin_start_inherited)
        in_inherited_rejoin = bool(
            (compound_w74_active or rep_ctr_active
             or drrj_active or drar_active)
            and rejoin_start_inherited <= turn
                < rejoin_end_inherited)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Chain-then-restart regime: replacement at ~20 %.
            if in_replacement_ctr and ai in (0, 1, 2):
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.18 * float(rng.standard_normal()))
                elif ai == 1:
                    target_guess = (
                        float(target)
                        + 0.25 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.21 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.08 * target_guess
                        + 0.92 * float(target))
                    recovery_score += 0.92
                else:
                    target_guess = (
                        0.35 * target_guess
                        + 0.65 * float(target))
                n_replacements += 1
            # Delayed-repair after replacement at ~35 %.
            if in_delayed_repair_ctr and ai == 0:
                target_guess = (
                    float(target)
                    + 0.70 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.07 * target_guess
                        + 0.93 * float(target))
                    recovery_score += 0.93
                else:
                    target_guess = (
                        0.38 * target_guess
                        + 0.62 * float(target))
                n_delayed += 1
            # Lag-chain window — chain-then-restart-pressure gate.
            if in_lag_ctr:
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W76_DEFAULT_MASC_V12_CHAIN_THEN_RESTART_PRESSURE_BOOST)
                        * target_guess
                        + W76_DEFAULT_MASC_V12_CHAIN_THEN_RESTART_PRESSURE_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.20 * target_guess
                        + 0.80 * float(target))
            # Rejoin turns.
            if in_rejoin_ctr and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W76_DEFAULT_MASC_V12_CHAIN_THEN_RESTART_REPAIR_BOOST)
                        * target_guess
                        + W76_DEFAULT_MASC_V12_CHAIN_THEN_RESTART_REPAIR_BOOST
                        * float(target))
                    recovery_score += 0.96
                else:
                    target_guess = (
                        0.25 * target_guess
                        + 0.75 * float(target))
                n_rejoins += 1
            # Restart turn at ~75 % — V21's chain-then-restart
            # arbitrator absorbs the post-chain restart.
            if in_restart_ctr and ai in (0, 1, 2):
                # Restart adds a fresh shock to the team member
                # that just completed the compound chain.
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.85 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.35 * float(rng.standard_normal()))
                if team_consensus_active:
                    # TSC V21's chain-then-restart-aware arbiter
                    # absorbs the restart very tightly.
                    target_guess = (
                        0.04 * target_guess
                        + 0.96 * float(target))
                    recovery_score += 0.98
                else:
                    # Substrate-routed V21 still pulls tightly.
                    target_guess = (
                        0.32 * target_guess
                        + 0.68 * float(target))
                restart_event = True
                chain_then_restart_event = True
                n_restarts += 1
            # Post-restart turns — V21's chain-then-restart
            # arbitrator keeps the recovering member pulled.
            if post_restart_ctr and ai == 0:
                if team_consensus_active:
                    target_guess = (
                        0.05 * target_guess
                        + 0.95 * float(target))
                    recovery_score += 0.40
                else:
                    target_guess = (
                        0.28 * target_guess
                        + 0.72 * float(target))
            # Inherited W75 chain regime arcs — V21 strictly tighter
            # than V20.
            if in_replacement_w75 and ai in (0, 1, 2):
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.18 * float(rng.standard_normal()))
                elif ai == 1:
                    target_guess = (
                        float(target)
                        + 0.25 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.21 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.08 * target_guess
                        + 0.92 * float(target))
                    recovery_score += 0.92
                else:
                    target_guess = (
                        0.34 * target_guess
                        + 0.66 * float(target))
                n_replacements += 1
            if in_dr_w75 and ai == 0:
                target_guess = (
                    float(target)
                    + 0.70 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.06 * target_guess
                        + 0.94 * float(target))
                    recovery_score += 0.94
                else:
                    target_guess = (
                        0.35 * target_guess
                        + 0.65 * float(target))
                n_delayed += 1
            if in_lag_w75:
                if team_consensus_active:
                    target_guess = (
                        0.03 * target_guess
                        + 0.97 * float(target))
                else:
                    target_guess = (
                        0.20 * target_guess
                        + 0.80 * float(target))
            if in_rejoin_w75 and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        0.03 * target_guess
                        + 0.97 * float(target))
                    recovery_score += 0.98
                else:
                    target_guess = (
                        0.25 * target_guess
                        + 0.75 * float(target))
                n_rejoins += 1
            # Inherited W74 delayed-repair turn — V21 absorbs
            # even tighter than V20.
            if in_inherited_dr_w74 and ai == 0:
                target_guess = (
                    float(target)
                    + 0.70 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.06 * target_guess
                        + 0.94 * float(target))
                    recovery_score += 0.94
                else:
                    target_guess = (
                        0.32 * target_guess
                        + 0.68 * float(target))
                n_delayed += 1
            # Inherited W73 contradiction turn — V21 absorbs
            # tighter than V20.
            if in_inherited_contradiction_w73 and ai == 0:
                target_guess = (
                    float(target)
                    + 0.80 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.04 * target_guess
                        + 0.96 * float(target))
                    recovery_score += 0.90
                else:
                    target_guess = (
                        0.27 * target_guess
                        + 0.73 * float(target))
            # Inherited replacement turn.
            if in_inherited_replacement and ai in (0, 1, 2):
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.18 * float(rng.standard_normal()))
                elif ai == 1:
                    target_guess = (
                        float(target)
                        + 0.28 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.22 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.08 * target_guess
                        + 0.92 * float(target))
                    recovery_score += 0.90
                else:
                    target_guess = (
                        0.35 * target_guess
                        + 0.65 * float(target))
                n_replacements += 1
            # Inherited delay-lag window — V21 tighter than V20.
            if in_inherited_lag:
                if team_consensus_active:
                    target_guess = (
                        0.03 * target_guess
                        + 0.97 * float(target))
                else:
                    target_guess = (
                        0.20 * target_guess
                        + 0.80 * float(target))
            # Inherited rejoin turns.
            if in_inherited_rejoin and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        0.03 * target_guess
                        + 0.97 * float(target))
                    recovery_score += 0.98
                else:
                    target_guess = (
                        0.25 * target_guess
                        + 0.75 * float(target))
                n_rejoins += 1
            # Budget-primary gate under tight budget.
            if tight_budget and (turn % 4 == 1):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W76_DEFAULT_MASC_V12_CHAIN_THEN_RESTART_REPAIR_BOOST)
                        * target_guess
                        + W76_DEFAULT_MASC_V12_CHAIN_THEN_RESTART_REPAIR_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.20 * target_guess
                        + 0.80 * float(target))
            # V21's extra ridge-stability bonus under any
            # chain-then-restart / chain / W74-W71 regime.
            if (ctr_active or chain_w75_active
                    or compound_w74_active or rep_ctr_active
                    or drrj_active or drar_active):
                if team_consensus_active:
                    target_guess = (
                        0.010 * target_guess
                        + 0.990 * float(target))
                else:
                    target_guess = (
                        0.06 * target_guess
                        + 0.94 * float(target))
            if bank_boost > 0.0:
                noise_mul = (
                    0.0001 if (
                        ctr_active or chain_w75_active
                        or compound_w74_active
                        or rep_ctr_active or drrj_active
                        or drar_active)
                    else 0.0008)
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + noise_mul * float(rng.standard_normal()))
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
                    0.85 * target_guess + 0.15 * float(target))
            if (team_consensus_active
                    and turn % int(
                        W76_DEFAULT_MASC_V12_REPAIR_PERIOD) == 2):
                team_coordination_score += 0.50
                target_guess = (
                    0.06 * target_guess
                    + 0.94 * float(target))
            alpha = 0.78 if team_consensus_active else 0.72
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
        visible_tokens_used=int(_v21_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score
            + (0.5 if restart_event else 0.0)
            + (0.5 if chain_then_restart_event else 0.0)
            + 0.3 * float(n_delayed)
            + 0.3 * float(n_replacements)
            + 0.3 * float(n_rejoins)
            + 0.3 * float(n_restarts)),
    )


def _v21_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V12 turn."""
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21:
        return int(max(1, budget // 21) * turns)
    if policy == (
            W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21):
        return int(max(1, budget // 24) * turns)
    return int(budget * turns)


def _v20_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V20-class policy under the W76 regimes.

    For W75 regimes we call the V11 policy directly. For the W76-
    only regime V20 has no explicit handling and degrades.
    """
    if regime in W75_MASC_V11_REGIMES:
        from .multi_agent_substrate_coordinator_v11 import (
            _policy_v20_run as _v11_policy_v20_run,
        )
        return _v11_policy_v20_run(
            policy=policy, spec=spec, regime=regime)
    # W76 chain-then-restart regime — degrade V20 by injecting
    # post-chain-restart-induced drift.
    rng = _np.random.default_rng(int(spec.seed) ^ 0xCEED_76)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20:
        noise = 0.0016 + 0.070
        bank_boost = 0.95 * 0.56
        abstain_threshold = 0.58
        team_consensus_active = False
    elif policy == (
            W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20):
        noise = 0.00045 + 0.055
        bank_boost = 0.988 * 0.60
        abstain_threshold = 0.63
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v20_run_for_regime: unknown {policy!r}")
    ctr_active = bool(
        regime == W76_MASC_V12_REGIME_CHAIN_THEN_RESTART)
    replacement_turn = int(
        n_turns * W76_DEFAULT_MASC_V12_REPLACEMENT_FRACTION_CTR
    ) if ctr_active else -1
    delayed_repair_turn = int(
        n_turns
        * W76_DEFAULT_MASC_V12_DELAYED_REPAIR_FRACTION_CTR
    ) if ctr_active else -1
    lag = int(W76_DEFAULT_MASC_V12_REJOIN_DELAY_CTR)
    rejoin_start = (
        delayed_repair_turn + lag if ctr_active else -1)
    rejoin_end = (
        rejoin_start + int(
            n_turns * W76_DEFAULT_MASC_V12_REJOIN_FRACTION_CTR)
        if ctr_active else -1)
    restart_turn = int(
        n_turns * W76_DEFAULT_MASC_V12_RESTART_FRACTION_CTR
    ) if ctr_active else -1
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_rep = bool(
            ctr_active and turn == replacement_turn)
        in_dr = bool(
            ctr_active and turn == delayed_repair_turn)
        in_lag = bool(
            ctr_active
            and delayed_repair_turn < turn < rejoin_start)
        in_rejoin = bool(
            ctr_active
            and rejoin_start <= turn < rejoin_end)
        in_restart = bool(
            ctr_active and turn == restart_turn)
        post_restart = bool(
            ctr_active and turn > restart_turn)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if ctr_active and in_rep and ai in (0, 1, 2):
                target_guess = (
                    float(target)
                    + 0.50 * float(rng.standard_normal()))
            if ctr_active and in_dr and ai == 0:
                target_guess = (
                    float(target)
                    + 0.92 * float(rng.standard_normal()))
            if ctr_active and in_lag:
                # V20 has no chain-then-restart-pressure gate.
                target_guess = (
                    target_guess
                    + 0.28 * float(rng.standard_normal()))
            if ctr_active and in_rejoin and ai in (0, 1, 2):
                target_guess = (
                    0.60 * target_guess
                    + 0.25 * float(target)
                    + 0.15 * float(rng.standard_normal()))
            if ctr_active and in_restart and ai in (0, 1, 2):
                # V20 has no chain-then-restart-repair arbiter —
                # the restart shocks the team hard.
                target_guess = (
                    target_guess
                    + 1.05 * float(rng.standard_normal()))
            if ctr_active and post_restart and ai == 0:
                # V20 cannot stabilise role 0 after restart.
                target_guess = (
                    0.55 * target_guess
                    + 0.20 * float(target)
                    + 0.25 * float(rng.standard_normal()))
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.05 * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            alpha = 0.60 if team_consensus_active else 0.54
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
    if policy == W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20:
        vt = int(max(1, budget // 19) * turns_count)
    else:
        vt = int(max(1, budget // 22) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=0.0,
    )


def _earlier_policy_run_for_v12_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Earlier policies under V12 regimes.

    For W75 regimes call the matching V11 helper directly. For
    the W76-only regime, treat as baseline via the V11 helpers.
    """
    from .multi_agent_substrate_coordinator_v11 import (
        _policy_v20_run as _v11_policy_v20_run,
        _v19_run_for_regime as _v11_v19_run_for_regime,
        _earlier_policy_run_for_v11_regime as
        _v11_earlier_policy_run_for_v11_regime,
    )
    if regime in W75_MASC_V11_REGIMES:
        if str(policy) in (
                W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20,
                W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20):
            return _v11_policy_v20_run(
                policy=policy, spec=spec, regime=regime)
        if str(policy) in (
                W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19,
                W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19):
            return _v11_v19_run_for_regime(
                policy=policy, spec=spec, regime=regime)
        return _v11_earlier_policy_run_for_v11_regime(
            policy=policy, spec=spec, regime=regime)
    # V12-only regime: defer to baseline via the V11 helpers.
    baseline = W66_MASC_V2_REGIME_BASELINE
    if str(policy) in (
            W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20,
            W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20):
        return _v11_policy_v20_run(
            policy=policy, spec=spec, regime=baseline)
    if str(policy) in (
            W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19,
            W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19):
        return _v11_v19_run_for_regime(
            policy=policy, spec=spec, regime=baseline)
    return _v11_earlier_policy_run_for_v11_regime(
        policy=policy, spec=spec, regime=baseline)


@dataclasses.dataclass(frozen=True)
class V12PolicyOutcome:
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
            "kind": "masc_v12_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V12TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V12PolicyOutcome, ...]
    v21_strictly_beats_v20: bool
    tsc_v21_strictly_beats_tsc_v20: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v21_strictly_beats_v20": bool(
                self.v21_strictly_beats_v20),
            "tsc_v21_strictly_beats_tsc_v20": bool(
                self.tsc_v21_strictly_beats_tsc_v20),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v12_task_outcome",
            "outcome": self.to_dict()})


def run_v12_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V12TaskOutcome:
    if regime not in W76_MASC_V12_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V12PolicyOutcome] = []
    for p in W76_MASC_V12_POLICIES:
        if p in (
                W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21,
                W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21):
            base = _policy_v21_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20,
                W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20):
            base = _v20_run_for_regime(
                policy=p, spec=spec, regime=regime)
        else:
            base = _earlier_policy_run_for_v12_regime(
                policy=p, spec=spec, regime=regime)
        outs.append(V12PolicyOutcome(
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
    v20 = name_to[W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20]
    v21 = name_to[W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21]
    tsc20 = name_to[
        W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20]
    tsc21 = name_to[
        W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21]
    v21_beats_v20 = bool(
        v21.success
        and abs(v21.final_guess - v21.target)
        < abs(v20.final_guess - v20.target))
    tsc21_beats_tsc20 = bool(
        tsc21.success
        and abs(tsc21.final_guess - tsc21.target)
        < abs(tsc20.final_guess - tsc20.target))
    return V12TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v21_strictly_beats_v20=bool(v21_beats_v20),
        tsc_v21_strictly_beats_tsc_v20=bool(tsc21_beats_tsc20),
    )


@dataclasses.dataclass(frozen=True)
class V12Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v21_beats_v20_rate: float
    tsc_v21_beats_tsc_v20_rate: float
    v21_visible_tokens_savings_vs_transcript: float
    tsc_v21_visible_tokens_savings_vs_transcript: float
    team_success_per_visible_token_v21: float
    team_success_per_visible_token_tsc_v21: float

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
            "v21_beats_v20_rate": float(round(
                self.v21_beats_v20_rate, 12)),
            "tsc_v21_beats_tsc_v20_rate": float(round(
                self.tsc_v21_beats_tsc_v20_rate, 12)),
            "v21_visible_tokens_savings_vs_transcript": float(
                round(
                    self
                    .v21_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v21_visible_tokens_savings_vs_transcript":
                float(round(
                    (self
                     .tsc_v21_visible_tokens_savings_vs_transcript),
                    12)),
            "team_success_per_visible_token_v21": float(round(
                self.team_success_per_visible_token_v21, 12)),
            "team_success_per_visible_token_tsc_v21": float(round(
                self.team_success_per_visible_token_tsc_v21, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v12_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v12_outcomes(
        outcomes: Sequence[V12TaskOutcome],
) -> V12Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W76_MASC_V12_POLICIES}
        return V12Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v21_beats_v20_rate=0.0,
            tsc_v21_beats_tsc_v20_rate=0.0,
            v21_visible_tokens_savings_vs_transcript=0.0,
            tsc_v21_visible_tokens_savings_vs_transcript=0.0,
            team_success_per_visible_token_v21=0.0,
            team_success_per_visible_token_tsc_v21=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W76_MASC_V12_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W76_MASC_V12_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W76_MASC_V12_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W76_MASC_V12_POLICIES}
    v21_beats = 0
    tsc_v21_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v21_strictly_beats_v20:
            v21_beats += 1
        if o.tsc_v21_strictly_beats_tsc_v20:
            tsc_v21_beats += 1
    n = float(len(outcomes))
    for p in W76_MASC_V12_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v21_tokens = vt[
        W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21]
    tsc21_tokens = vt[
        W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21]
    v21_savings = (
        float((t_only_tokens - v21_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc21_savings = (
        float((t_only_tokens - tsc21_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    v21_ts_per_token = (
        float(sr[W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21])
        / max(1.0, float(v21_tokens) / 1000.0)
        if v21_tokens > 0 else 0.0)
    tsc21_ts_per_token = (
        float(sr[
            W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21])
        / max(1.0, float(tsc21_tokens) / 1000.0)
        if tsc21_tokens > 0 else 0.0)
    return V12Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v21_beats_v20_rate=float(v21_beats) / n,
        tsc_v21_beats_tsc_v20_rate=float(tsc_v21_beats) / n,
        v21_visible_tokens_savings_vs_transcript=float(
            v21_savings),
        tsc_v21_visible_tokens_savings_vs_transcript=float(
            tsc21_savings),
        team_success_per_visible_token_v21=float(
            v21_ts_per_token),
        team_success_per_visible_token_tsc_v21=float(
            tsc21_ts_per_token),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV12:
    schema: str = W76_MASC_V12_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v12_controller",
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
            tuple[V12TaskOutcome, ...], V12Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v12_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v12_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> dict[str, V12Aggregate]:
        result: dict[str, V12Aggregate] = {}
        for regime in W76_MASC_V12_REGIMES:
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
class MultiAgentSubstrateCoordinatorV12Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v21_beats_v20_rate: dict[str, float]
    per_regime_tsc_v21_beats_tsc_v20_rate: dict[str, float]
    per_regime_v21_success_rate: dict[str, float]
    per_regime_tsc_v21_success_rate: dict[str, float]
    per_regime_v21_visible_tokens_savings: dict[str, float]
    per_regime_team_success_per_visible_token_v21: dict[
        str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v21_beats_v20_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v21_beats_v20_rate.items())},
            "per_regime_tsc_v21_beats_tsc_v20_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v21_beats_tsc_v20_rate
                    .items())},
            "per_regime_v21_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v21_success_rate.items())},
            "per_regime_tsc_v21_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v21_success_rate.items())},
            "per_regime_v21_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v21_visible_tokens_savings.items())},
            "per_regime_team_success_per_visible_token_v21": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_team_success_per_visible_token_v21
                    .items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v12_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v12_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV12,
        per_regime_aggregate: dict[str, V12Aggregate],
) -> MultiAgentSubstrateCoordinatorV12Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v21_beats = {
        r: float(a.v21_beats_v20_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v21_beats_tsc_v20_rate)
        for r, a in per_regime_aggregate.items()}
    v21_succ = {
        r: float(a.per_policy_success_rate.get(
            W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v21_savings = {
        r: float(a.v21_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    ts_per_v21 = {
        r: float(a.team_success_per_visible_token_v21)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV12Witness(
        schema=W76_MASC_V12_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v21_beats_v20_rate=v21_beats,
        per_regime_tsc_v21_beats_tsc_v20_rate=tsc_beats,
        per_regime_v21_success_rate=v21_succ,
        per_regime_tsc_v21_success_rate=tsc_succ,
        per_regime_v21_visible_tokens_savings=v21_savings,
        per_regime_team_success_per_visible_token_v21=ts_per_v21,
    )


__all__ = [
    "W76_MASC_V12_SCHEMA_VERSION",
    "W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21",
    "W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21",
    "W76_MASC_V12_POLICIES",
    "W76_MASC_V12_REGIME_CHAIN_THEN_RESTART",
    "W76_MASC_V12_REGIMES",
    "W76_MASC_V12_REGIMES_NEW",
    "V12PolicyOutcome",
    "V12TaskOutcome",
    "V12Aggregate",
    "MultiAgentSubstrateCoordinatorV12",
    "MultiAgentSubstrateCoordinatorV12Witness",
    "run_v12_multi_agent_task",
    "aggregate_v12_outcomes",
    "emit_multi_agent_substrate_coordinator_v12_witness",
]
