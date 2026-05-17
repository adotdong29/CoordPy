# Context as Objects: Capsule-Native Coordination for Multi-Agent Teams

> Main paper draft for the Context Zero programme.
>
> **Post-W73 research-line update (W74 Stronger Compound-Repair /
> Replacement-After-Delayed-Repair Budget-Primary Two-Plane Multi-
> Agent Substrate Programme, 2026-05-17).** *Nineteenth substrate-
> attack milestone, and the **tenth** multi-agent task-success-
> bearing substrate milestone in the programme — the first to
> produce wins across **fourteen** failure-mode regimes (W73's
> thirteen plus **replacement_after_delayed_repair_under_budget**),
> the **first milestone to operationalise compound-aware Plane A↔B
> handoff promotion**, and the **first milestone to expose a
> content-addressed per-turn compound-repair-trajectory CID** that
> unifies all ten repair/restart/rejoin/replacement/compound
> primitives (multi-branch-rejoin, silent-corruption, partial-
> contradiction, agent-replacement, role-dropout-recovery, branch-
> merge, restart-dominance, delayed-rejoin-after-restart,
> replacement-after-contradiction-then-rejoin,
> **compound-repair-after-delayed-repair-then-replacement**) into a
> single dominant signal back into the substrate-routed policy.
> ``MultiAgentSubstrateCoordinatorV10`` runs **twenty-two** matched-
> budget policies (the twenty W73 policies plus
> ``substrate_routed_v19`` and ``team_substrate_coordination_v19``)
> under fourteen regimes; the V19 policy strictly beats V18 across
> all fourteen regimes (≥ 86.7 % at every regime including the new
> compound regime, with TSC_V19 strictly beating TSC_V18 ≥ 93.3 % of
> seeds in every regime). In the new
> ``replacement_after_delayed_repair_under_budget`` regime — where
> at ~15 % of turns role 0 enters a delayed-repair window, at ~30 %
> of turns the delayed role is wiped and replaced with a fresh
> member, then at ~50 % of turns the team must rejoin from the
> divergent branches under a tight visible-token budget — the V19
> substrate's compound-repair-trajectory CID + per-layer compound-
> repair-rate label + per-layer compound-pressure gate give the
> substrate-routed policy a coordinated delayed-repair-and-
> replacement-and-rejoin arc that V18 cannot follow under the
> additional delayed-repair-then-replacement stressor. The in-repo
> NumPy substrate grows to V19 (21 layers + three new V19 axes:
> per-turn compound-repair-trajectory CID, per-layer compound-
> repair-rate label in [0..10], per-layer compound-pressure gate).
> The substrate compound-dominance primitive saves **95.5 %** of
> recompute flops over ten-primitive repair work at 128 tokens. The
> compound-pressure throttle saves **100 %** visible tokens at
> visible_token_budget=64 / baseline=512 / compound_window=8. Three
> new closed-form ridge solves on top of W61..W73's 67 (70 total).
> The deep substrate hybrid is now nineteen-way (V18 hybrid + cache
> V17 + replay V15 + compound-repair-trajectory + compound-repair-
> rate + team-consensus controller V9). The persistent latent state
> is V26 (25 layers, 23 skip carriers,
> ``max_chain_walk_depth=2097152``, distractor rank 25). Long-
> horizon reconstruction reaches 25 heads at max_k=832. Consensus
> has 34 disjoint stages including ``compound_repair_arbiter`` and
> ``compound_repair_after_delayed_repair_then_replacement_arbiter``.
> The first capsule-native team-consensus controller V9 composes
> regime-aware weighted quorum + repair-dominance arbiter + budget-
> primary arbiter + contradiction-then-rejoin arbiter + restart-
> aware arbiter + delayed-repair-after-restart arbiter + rejoin-
> pressure arbiter + delayed-rejoin-after-restart arbiter +
> replacement-pressure arbiter + replacement-after-CTR arbiter +
> compound-pressure arbiter + compound-repair-after-DRTR arbiter +
> substrate-replay fallback + transcript fallback. Plane A V7
> ships seven hosted control-plane modules (router V7 with
> compound-pressure weighting + compound-repair-after-DRTR match
> table, logprob router V7 with compound-aware abstain floor +
> per-budget+restart+rejoin+replacement+compound tiebreak, cache-
> aware planner V7 with ≥ 85 % savings on 16×8 five-layer-rotated
> prefixes, cost planner V7 with cost-per-compound-success-under-
> budget + abstain-when-compound-pressure-violated, boundary V7
> with 34 blocked axes + the W70 frontier-blocked axes unchanged,
> provider filter V6 with compound-aware drop). And — the load-
> bearing operational W74 advance — the new
> ``hosted_real_handoff_coordinator_v6`` records per-turn content-
> addressed V6 envelopes that promote any turn with
> ``compound_pressure ≥ 0.5`` and substrate_trust ≥ floor to Plane
> B with ``compound_alignment = 1.0`` and add a ninth
> ``compound_repair_after_delayed_repair_then_replacement_fallback``
> decision on top of V5's eight, exposing a compound falsifier and
> saving ≥ 82 % cross-plane visible tokens at the default workload
> (≥ 83 % at default config). R-177 + R-178 + R-179 + R-180 deliver
> **70 H-bars × 3 seeds (210 cells)**, all pass. The honest scope
> is unchanged: hosted backends remain text-only at the HTTP
> surface (``W74-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``); the
> substrate is a 21-layer in-repo NumPy runtime, not a frontier
> model (``W74-L-NUMPY-CPU-V19-SUBSTRATE-CAP``); "training" is
> closed-form linear ridge throughout
> (``W74-L-V19-NO-AUTOGRAD-CAP``); the multi-agent task-success
> wins are measured **inside the W74 synthetic harness**
> (``W74-L-MASC-V10-SYNTHETIC-CAP``), not on real hosted multi-
> agent backends; the compound-repair-trajectory CID is computed
> from byte-stable substrate state only and does NOT prove
> compound-repair integrity at the hosted surface (``W74-L-
> COMPOUND-REPAIR-IN-REPO-CAP``); budgets, replacement pressure,
> and compound pressure are caller-declared
> (``W74-L-COMPOUND-PRESSURE-DECLARED-CAP``,
> ``W74-L-HOSTED-V7-DECLARED-CAP``); and the V6 handoff
> coordinator preserves the wall — it does NOT cross the substrate
> boundary (``W74-L-HANDOFF-V6-NOT-CROSSING-WALL-CAP``).*
>
> **Post-W72 research-line update (W73 Stronger Contradiction-Rejoin
> / Replacement / Delayed-Repair Budget-Primary Two-Plane Multi-
> Agent Substrate Programme, 2026-05-16).** *Eighteenth substrate-
> attack milestone, and the **ninth** multi-agent task-success-
> bearing substrate milestone in the programme — the first to
> produce wins across **thirteen** failure-mode regimes (W72's
> twelve plus **replacement_after_contradiction_then_rejoin**),
> the **first milestone to operationalise replacement-aware Plane
> A↔B handoff promotion**, and the **first milestone to expose a
> content-addressed per-turn replacement-repair-trajectory CID**
> that unifies all nine repair/restart/rejoin/replacement primitives
> (multi-branch-rejoin, silent-corruption, partial-contradiction,
> agent-replacement, role-dropout-recovery, branch-merge, restart-
> dominance, delayed-rejoin-after-restart,
> **replacement-after-contradiction-then-rejoin**) into a single
> dominant signal back into the substrate-routed policy.
> ``MultiAgentSubstrateCoordinatorV9`` runs **twenty** matched-
> budget policies (the eighteen W72 policies plus
> ``substrate_routed_v18`` and ``team_substrate_coordination_v18``)
> under thirteen regimes; the V18 policy strictly beats V17 across
> all thirteen regimes (100 % at every regime including the new
> compound regime, with TSC_V18 strictly beating TSC_V17 100 % of
> seeds in every regime). In the new
> ``replacement_after_contradiction_then_rejoin`` regime — where
> at ~15 % of turns role 0 produces a hard contradiction, at ~25 %
> of turns the contradicting role is wiped and replaced with a
> fresh member, then at ~45 % of turns the team must rejoin from
> the divergent branches under a tight visible-token budget — the
> V18 substrate's replacement-repair-trajectory CID + per-layer
> replacement-after-CTR label + per-layer replacement-pressure gate
> give the substrate-routed policy a coordinated replacement-and-
> rejoin arc that V17 cannot follow under the additional
> contradiction stressor. The in-repo NumPy substrate grows to V18
> (20 layers + three new V18 axes: per-turn replacement-repair-
> trajectory CID, per-layer replacement-after-CTR label in [0..9],
> per-layer replacement-pressure gate). The substrate replacement-
> dominance primitive saves **95 %** of recompute flops over nine-
> primitive repair work at 128 tokens. The replacement-pressure
> throttle saves **100 %** visible tokens at visible_token_budget=64
> / baseline=512 / replacement_lag=5. Three new closed-form ridge
> solves on top of W61..W72's 64 (67 total). The deep substrate
> hybrid is now eighteen-way (V17 hybrid + cache V16 + replay V14 +
> replacement-repair-trajectory + replacement-after-CTR + team-
> consensus controller V8). The persistent latent state is V25 (24
> layers, 22 skip carriers, ``max_chain_walk_depth=1048576``,
> distractor rank 24). Long-horizon reconstruction reaches 24 heads
> at max_k=768. Consensus has 32 disjoint stages including
> ``replacement_pressure_arbiter`` and
> ``replacement_after_contradiction_then_rejoin_arbiter``. The
> first capsule-native team-consensus controller V8 composes regime-
> aware weighted quorum + repair-dominance arbiter + budget-primary
> arbiter + contradiction-then-rejoin arbiter + restart-aware
> arbiter + delayed-repair-after-restart arbiter + rejoin-pressure
> arbiter + delayed-rejoin-after-restart arbiter + replacement-
> pressure arbiter + replacement-after-CTR arbiter + substrate-
> replay fallback + transcript fallback. Plane A V6 ships seven
> hosted control-plane modules (router V6 with replacement-pressure
> weighting + replacement-after-CTR match table, logprob router V6
> with replacement-aware abstain floor + per-budget+restart+rejoin+
> replacement tiebreak, cache-aware planner V6 with ≥ 85 % savings
> on 14×8 four-layer-rotated prefixes, cost planner V6 with cost-
> per-replacement-rejoin-success-under-budget + abstain-when-
> replacement-pressure-violated, boundary V6 with 31 blocked axes +
> the W70 frontier-blocked axes unchanged, provider filter V5 with
> replacement-aware drop). And — the load-bearing operational W73
> advance — the new ``hosted_real_handoff_coordinator_v5`` records
> per-turn content-addressed V5 envelopes that promote any turn
> with ``replacement_pressure ≥ 0.5`` and substrate_trust ≥ floor to
> Plane B with ``replacement_alignment = 1.0`` and add an eighth
> ``replacement_after_contradiction_then_rejoin_fallback`` decision
> on top of V4's seven, exposing a replacement falsifier and saving
> ≥ 80 % cross-plane visible tokens at the default 48/10/7/10/12/
> 10/3 % workload (≥ 81 % at default config). R-173 + R-174 + R-175
> + R-176 deliver **68 H-bars × 3 seeds (204 cells)**, all pass.
> The honest scope is unchanged: hosted backends remain text-only
> at the HTTP surface (``W73-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-
> CAP``); the substrate is a 20-layer in-repo NumPy runtime, not a
> frontier model (``W73-L-NUMPY-CPU-V18-SUBSTRATE-CAP``); "training"
> is closed-form linear ridge throughout
> (``W73-L-V18-NO-AUTOGRAD-CAP``); the multi-agent task-success wins
> are measured **inside the W73 synthetic harness** (``W73-L-MASC-
> V9-SYNTHETIC-CAP``), not on real hosted multi-agent backends;
> the replacement-repair-trajectory CID is computed from byte-
> stable substrate state only and does NOT prove replacement
> integrity at the hosted surface (``W73-L-REPLACEMENT-REPAIR-IN-
> REPO-CAP``); budgets, rejoin pressure, and replacement pressure
> are caller-declared (``W73-L-REPLACEMENT-PRESSURE-DECLARED-CAP``,
> ``W73-L-HOSTED-V6-DECLARED-CAP``); and the V5 handoff coordinator
> preserves the wall — it does NOT cross the substrate boundary
> (``W73-L-HANDOFF-V5-NOT-CROSSING-WALL-CAP``).*
>
> **Post-W71 research-line update (W72 Stronger Delayed-Rejoin-
> After-Restart / Restart-Repair-Trajectory Two-Plane Multi-Agent
> Substrate Programme, 2026-05-16).** *Seventeenth substrate-attack
> milestone, and the **eighth** multi-agent task-success-bearing
> substrate milestone in the programme — the first to produce wins
> across **twelve** failure-mode regimes (W71's eleven plus
> **delayed_rejoin_after_restart_under_budget**), the **first
> milestone to operationalise rejoin-aware Plane A↔B handoff
> promotion**, and the **first milestone to expose a content-
> addressed per-turn restart-repair-trajectory CID** that unifies
> all eight repair/restart/rejoin primitives (multi-branch-rejoin,
> silent-corruption, partial-contradiction, agent-replacement,
> role-dropout-recovery, branch-merge, restart-dominance,
> **delayed-rejoin-after-restart**) into a single dominant signal
> back into the substrate-routed policy.
> ``MultiAgentSubstrateCoordinatorV8`` runs **eighteen** matched-
> budget policies (the sixteen W71 policies plus
> ``substrate_routed_v17`` and ``team_substrate_coordination_v17``)
> under twelve regimes; the V17 policy strictly beats V16 across
> all twelve regimes (≥ 86.7 % at the eleven baseline-class
> regimes, ≥ 50 % at the new compound regime, with TSC_V17 strictly
> beating TSC_V16 100 % of seeds in every regime). In the new
> ``delayed_rejoin_after_restart_under_budget`` regime (where at
> ~20 % of turns one role's substrate is wiped clean and replaced
> with a fresh member, then the team must absorb a *delay window*
> (~3 turns later), then rejoin from divergent branches (~30 % of
> turns) under a tight visible-token budget), the V17 substrate's
> restart-repair-trajectory CID + per-layer delayed-rejoin-after-
> restart label + per-layer rejoin-pressure gate give the
> substrate-routed policy a coordinated rejoin arc that V16 cannot
> follow under the additional branch-divergence stressor. The in-
> repo NumPy substrate grows to V17 (19 layers + three new V17
> axes: per-turn restart-repair-trajectory CID, per-layer delayed-
> rejoin-after-restart label in [0..8], per-layer rejoin-pressure
> gate). The substrate rejoin-dominance primitive saves **94.5 %**
> of recompute flops over eight-primitive repair work at 128
> tokens. The rejoin-pressure throttle saves **100 %** visible
> tokens at visible_token_budget=64 / baseline=512 / rejoin_lag=4.
> Three new closed-form ridge solves on top of W61..W71's 61
> (64 total). The deep substrate hybrid is now seventeen-way (V16
> hybrid + cache V15 + replay V13 + restart-repair-trajectory +
> delayed-rejoin + team-consensus controller V7). The persistent
> latent state is V24 (23 layers, 21 skip carriers,
> ``max_chain_walk_depth=524288``, distractor rank 23). Long-
> horizon reconstruction reaches 23 heads at max_k=704. Consensus
> has 30 disjoint stages including ``rejoin_pressure_arbiter`` and
> ``delayed_rejoin_after_restart_arbiter``. The first capsule-
> native team-consensus controller V7 composes regime-aware
> weighted quorum + repair-dominance arbiter + budget-primary
> arbiter + contradiction-then-rejoin arbiter + restart-aware
> arbiter + delayed-repair-after-restart arbiter + rejoin-pressure
> arbiter + delayed-rejoin-after-restart arbiter + substrate-
> replay fallback + transcript fallback. Plane A V5 ships seven
> hosted control-plane modules (router V5 with rejoin-pressure
> weighting + delayed-rejoin match table, logprob router V5 with
> rejoin-aware abstain floor + per-budget+restart+rejoin tiebreak,
> cache-aware planner V5 with ≥ 80 % savings on 12×8 three-layer-
> rotated prefixes, cost planner V5 with cost-per-rejoin-success-
> under-budget + abstain-when-rejoin-pressure-violated, boundary
> V5 with 28 blocked axes + the W70 frontier-blocked axes
> unchanged, provider filter V4 with rejoin-aware drop). And —
> the load-bearing operational W72 advance — the new
> ``hosted_real_handoff_coordinator_v4`` records per-turn content-
> addressed V4 envelopes that promote any turn with
> ``rejoin_pressure ≥ 0.5`` and substrate_trust ≥ floor to Plane B
> with ``rejoin_alignment = 1.0`` and add a seventh
> ``delayed_rejoin_after_restart_fallback`` decision on top of
> V3's six, exposing a delayed-rejoin falsifier and saving ≥ 78 %
> cross-plane visible tokens at the default 50/12/8/12/15/3 %
> workload (≥ 84 % at default config). R-169 + R-170 + R-171 +
> R-172 deliver **66 H-bars × 3 seeds (198 cells)**, all pass.
> The honest scope is unchanged: hosted backends remain text-only
> at the HTTP surface (``W72-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-
> CAP``); the substrate is a 19-layer in-repo NumPy runtime, not
> a frontier model (``W72-L-NUMPY-CPU-V17-SUBSTRATE-CAP``);
> "training" is closed-form linear ridge throughout
> (``W72-L-V17-NO-AUTOGRAD-CAP``); the multi-agent task-success
> wins are measured **inside the W72 synthetic harness**
> (``W72-L-MASC-V8-SYNTHETIC-CAP``), not on real hosted multi-
> agent backends; the restart-repair-trajectory CID is computed
> from byte-stable substrate state only and does NOT prove
> rejoin integrity at the hosted surface
> (``W72-L-RESTART-REPAIR-IN-REPO-CAP``); budgets and rejoin
> pressure are caller-declared
> (``W72-L-REJOIN-PRESSURE-DECLARED-CAP``,
> ``W72-L-HOSTED-V5-DECLARED-CAP``); and the V4 handoff
> coordinator preserves the wall — it does NOT cross the
> substrate boundary
> (``W72-L-HANDOFF-V4-NOT-CROSSING-WALL-CAP``).*
>
> **Post-W70 research-line update (W71 Stronger Delayed-Repair-
> After-Restart / Repair-Trajectory-Primary Two-Plane Multi-Agent
> Substrate Programme, 2026-05-16).** *Sixteenth substrate-attack
> milestone, and the **seventh** multi-agent task-success-bearing
> substrate milestone in the programme — the first to produce wins
> across **eleven** failure-mode regimes (W70's ten plus
> **delayed_repair_after_restart**), the **first milestone to
> operationalise restart-aware Plane A↔B handoff promotion**, and
> the **first milestone to expose a content-addressed per-turn
> delayed-repair-trajectory CID** that unifies all seven repair-
> and-restart primitives (multi-branch-rejoin, silent-corruption,
> partial-contradiction, agent-replacement, role-dropout-recovery,
> branch-merge, **restart-dominance**) into a single dominant
> signal back into the substrate-routed policy.
> ``MultiAgentSubstrateCoordinatorV7`` runs **sixteen** matched-
> budget policies (the fourteen W70 policies plus
> ``substrate_routed_v16`` and ``team_substrate_coordination_v16``)
> under eleven regimes; the V16 policy strictly beats V15 across
> all eleven regimes (≥ 86.7 % at the ten baseline-class regimes,
> ≥ 50 % at the new compound regime, with TSC_V16 strictly beating
> TSC_V15 100 % of seeds in every regime). In the new
> ``delayed_repair_after_restart`` regime (where at ~25 % of turns
> one role's substrate is wiped clean and replaced with a fresh
> member, then a coordinated repair must arrive after a **delay
> window** (~3 turns later) under a tight visible-token budget),
> the V16 substrate's delayed-repair-trajectory CID + per-layer
> restart-dominance label + per-layer delayed-repair gate give the
> substrate-routed policy a coordinated recovery arc that V15
> cannot follow. The in-repo NumPy substrate grows to V16 (18
> layers + three new V16 axes: per-turn delayed-repair-trajectory
> CID, per-layer restart-dominance label in [0..7], per-layer
> delayed-repair gate). The substrate repair-dominance primitive
> saves **94 %** of recompute flops over seven-primitive repair
> work at 128 tokens. The delayed-repair throttle saves
> **100 %** visible tokens at visible_token_budget=64 / baseline=
> 512 / delay=3. Three new closed-form ridge solves on top of
> W61..W70's 58 (61 total). The deep substrate hybrid is now
> sixteen-way (V15 hybrid + cache V14 + replay V12 + delayed-
> repair-trajectory + restart-dominance + team-consensus
> controller V6). The persistent latent state is V23 (22 layers,
> 20 skip carriers, ``max_chain_walk_depth=262144``, distractor
> rank 22). Long-horizon reconstruction reaches 22 heads at
> max_k=640. Consensus has 28 disjoint stages including
> ``restart_aware_arbiter`` and
> ``delayed_repair_after_restart_arbiter``. The first capsule-
> native team-consensus controller V6 composes regime-aware
> weighted quorum + repair-dominance arbiter + budget-primary
> arbiter + contradiction-then-rejoin arbiter + restart-aware
> arbiter + delayed-repair-after-restart arbiter + substrate-
> replay fallback + transcript fallback. Plane A V4 ships seven
> hosted control-plane modules (router V4 with restart-pressure
> weighting + delayed-repair match table, logprob router V4 with
> restart-aware abstain floor + per-budget+restart tiebreak,
> cache-aware planner V4 with ≥ 72 % savings on 10×8 two-layer-
> rotated prefixes, cost planner V4 with cost-per-repair-success-
> under-budget + abstain-when-restart-pressure-violated, boundary
> V4 with 25 blocked axes + the W70 frontier-blocked axes
> unchanged, provider filter V3 with restart-aware drop). And —
> the load-bearing operational W71 advance — the new
> ``hosted_real_handoff_coordinator_v3`` records per-turn content-
> addressed V3 envelopes that promote any turn with
> ``restart_pressure ≥ 0.5`` and substrate_trust ≥ floor to Plane B
> with ``restart_alignment = 1.0`` and add a sixth
> ``delayed_repair_fallback`` decision on top of V2's five,
> exposing a delayed-repair falsifier and saving ≥ 70 % cross-
> plane visible tokens at the default 55/15/10/15/5 % workload
> (≥ 84 % at default config). R-165 + R-166 + R-167 + R-168
> deliver **64 H-bars × 3 seeds (192 cells)**, all pass. The
> honest scope is unchanged: hosted backends remain text-only at
> the HTTP surface (``W71-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-
> CAP``); the substrate is an 18-layer in-repo NumPy runtime, not
> a frontier model (``W71-L-NUMPY-CPU-V16-SUBSTRATE-CAP``);
> "training" is closed-form linear ridge throughout
> (``W71-L-V16-NO-AUTOGRAD-CAP``); the multi-agent task-success
> wins are measured **inside the W71 synthetic harness**
> (``W71-L-MASC-V7-SYNTHETIC-CAP``), not on real hosted multi-
> agent backends; the delayed-repair-trajectory CID is computed
> from byte-stable substrate state only and does NOT prove
> delayed-repair integrity at the hosted surface (``W71-L-DELAYED-
> REPAIR-IN-REPO-CAP``); budgets and restart pressure are caller-
> declared (``W71-L-DELAYED-REPAIR-DECLARED-CAP``,
> ``W71-L-HOSTED-V4-DECLARED-CAP``); and the V3 handoff
> coordinator preserves the wall — it does NOT cross the substrate
> boundary (``W71-L-HANDOFF-V3-NOT-CROSSING-WALL-CAP``).*
>
> **Post-W69 research-line update (W70 Stronger Repair-Dominance /
> Budget-Primary Two-Plane Multi-Agent Substrate Programme,
> 2026-05-16).** *Fifteenth substrate-attack milestone, and the
> **sixth** multi-agent task-success-bearing substrate milestone in
> the programme — the first to produce wins across **ten** failure-
> mode regimes (W69's nine plus
> **contradiction_then_rejoin_under_budget**), the **first milestone
> to operationalise budget-primary handoff scoring**, and the
> **first milestone to expose a content-addressed per-turn repair-
> trajectory CID** that unifies all six W67–W69 repair primitives
> (multi-branch-rejoin, silent-corruption, partial-contradiction,
> agent-replacement, role-dropout-recovery, branch-merge) into a
> single dominant-repair signal back into the substrate-routed
> policy. ``MultiAgentSubstrateCoordinatorV6`` runs **fourteen**
> matched-budget policies (the twelve W69 policies plus
> ``substrate_routed_v15`` and ``team_substrate_coordination_v15``)
> under ten regimes; the V15 policy strictly beats V14 across all
> ten regimes (100 % at baseline / team_consensus_under_budget /
> team_failure_recovery / role_dropout / branch_merge_reconciliation
> / partial_contradiction / agent_replacement_warm_restart; 87.5 %
> at multi_branch_rejoin / silent_corruption / contradiction_then_
> rejoin_under_budget). The ``team_substrate_coordination_v15``
> policy strictly beats ``team_substrate_coordination_v14`` on
> ≥ 75 % of seeds in every regime (up to 87.5 % across the baseline
> class). In the new contradiction-then-rejoin-under-budget regime
> (where agents at ~30 % of turns contradict each other AND at
> ~60 % of turns fork into branches AND the visible-token budget
> is tight throughout), the V15 substrate's repair-trajectory CID +
> per-layer dominant-repair label + budget-primary gate give the
> substrate-routed policy a coordinated repair arc that V14 cannot
> follow. The in-repo NumPy substrate grows to V15 (17 layers +
> three new V15 axes: per-turn repair-trajectory CID, per-layer
> dominant-repair label in [0..6], per-layer budget-primary gate).
> The substrate repair-dominance primitive saves **93 %** of
> recompute flops over six-primitive repair work at 128 tokens.
> The budget-primary throttle saves **87.5 %** visible tokens at
> visible_token_budget=64 / baseline_cost=512. Five new closed-form
> ridge solves on top of W61..W69's 53 (58 total). The deep
> substrate hybrid is now fifteen-way (V14 hybrid + cache V13 +
> replay V11 + repair-trajectory + budget-primary + team-consensus
> controller V5). The persistent latent state is V22 (21 layers, 19
> skip carriers, ``max_chain_walk_depth=131072``, distractor rank
> 21). Long-horizon reconstruction reaches 21 heads at max_k=576.
> Consensus has 26 disjoint stages including ``repair_dominance_
> arbiter`` and ``budget_primary_arbiter``. The first capsule-native
> team-consensus controller V5 composes regime-aware weighted quorum
> + repair-dominance arbiter (confidence-weighted mean across
> matching agents) + budget-primary arbiter (top-confidence agent
> under tight budget) + contradiction-then-rejoin arbiter +
> substrate-replay fallback + transcript fallback. Plane A V3 ships
> five new hosted control-plane modules (router V3 with budget-aware
> weighted scoring + repair-dominance match, logprob router V3 with
> abstain-when-disagree + per-budget tiebreak, cache-aware planner
> V3 with ≥ 65 % savings on 8×8 staggered + rotated prefixes, cost
> planner V3 with cost-per-team-success-under-budget + abstain-
> when-budget-violated, boundary V3 with 22 blocked axes + 3
> frontier-blocked axes). And — the load-bearing operational W70
> advance — the new ``hosted_real_handoff_coordinator_v2`` records
> per-turn content-addressed V2 envelopes that score ``team_success
> _per_visible_token`` and add a fifth ``budget_primary_fallback``
> decision on top of V1's four, exposing a repair-dominance
> falsifier and saving ≥ 75 % cross-plane visible tokens at the
> default 55/20/15/10 % workload. R-161 + R-162 + R-163 + R-164
> deliver **60 H-bars × 3 seeds (180 cells)**, all pass. The honest
> scope is unchanged: hosted backends remain text-only at the HTTP
> surface (``W70-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``); the
> substrate is a 17-layer in-repo NumPy runtime, not a frontier
> model (``W70-L-NUMPY-CPU-V15-SUBSTRATE-CAP``); "training" is
> closed-form linear ridge throughout (``W70-L-V15-NO-AUTOGRAD-
> CAP``); the multi-agent task-success wins are measured **inside
> the W70 synthetic harness** (``W70-L-MASC-V6-SYNTHETIC-CAP``),
> not on real hosted multi-agent backends; the repair-trajectory
> CID is computed from byte-stable substrate state only and does
> NOT prove repair integrity at the hosted surface (``W70-L-REPAIR-
> TRAJECTORY-IN-REPO-CAP``); budgets are caller-declared (``W70-L-
> BUDGET-PRIMARY-DECLARED-CAP``); and the V2 handoff coordinator
> preserves the wall — it does NOT cross the substrate boundary
> (``W70-L-HANDOFF-V2-NOT-CROSSING-WALL-CAP``).*
>
> **Preceding research-line update (W69 Stronger Solving-Context
> Two-Plane Multi-Agent Substrate, 2026-05-16).** *Fourteenth
> substrate-attack milestone, and the **fifth** multi-agent
> task-success-bearing substrate milestone in the programme — the
> first to produce wins across **nine** failure-mode regimes
> (baseline + team-consensus-under-budget + team-failure-recovery
> + role-dropout + branch-merge-reconciliation + partial-
> contradiction + agent-replacement-warm-restart +
> **multi-branch-rejoin-after-divergent-work** +
> **silent-corruption-plus-member-replacement**), not just seven.
> ``MultiAgentSubstrateCoordinatorV5`` runs **twelve** matched-
> budget policies (the ten W68 policies plus
> ``substrate_routed_v14`` and ``team_substrate_coordination_v14``)
> under nine regimes; the V14 policy strictly beats V13 across
> all nine regimes (≥ 60 % of seeds per regime: 80 % at baseline;
> 80 % at team_consensus_under_budget; 80 % at
> team_failure_recovery; 80 % at role_dropout; 80 % at
> branch_merge_reconciliation; 86.7 % at partial_contradiction;
> 86.7 % at agent_replacement_warm_restart; **86.7 %** at
> multi_branch_rejoin; **60 %** at silent_corruption). The
> ``team_substrate_coordination_v14`` policy strictly beats
> ``team_substrate_coordination_v13`` on ≥ 80 % of seeds in every
> regime (up to 93.3 % at multi-branch-rejoin). In the multi-
> branch-rejoin regime (where three subgroups of agents fork into
> divergent branches mid-task and must rejoin), the V14 substrate's
> multi-branch-rejoin witness tensor + the V14 substrate self-
> checksum CID provide the path V13 cannot. In the silent-
> corruption-plus-member-replacement regime (where one role's
> substrate is silently corrupted AND the role is replaced mid-
> task), the V14 substrate's silent-corruption witness + member-
> replacement flag + self-checksum CID let the V14 substrate
> detect the corruption and warm-restart the replacement from the
> surviving-agents weighted-mean. The in-repo NumPy substrate
> grows to V14 (16 layers + four new V14 axes: per-(layer, head,
> slot) multi-branch-rejoin witness tensor, per-role silent-
> corruption witness with member-replacement flag, substrate self-
> checksum CID, per-layer V14 composite gate score). The substrate
> multi-branch-rejoin primitive saves **92 %** of recompute flops
> at 128 tokens over a 4-branch rejoin. The substrate self-
> checksum 1-byte detect rate is structurally 1 − 1/2^256 by SHA-
> 256. Six new closed-form ridge solves on top of W61..W68's 47
> (53 total). The deep substrate hybrid is now fourteen-way (V13
> hybrid + cache V12 + replay V10 + multi-branch-rejoin witness +
> silent-corruption witness + substrate self-checksum +
> team-consensus controller V4). The persistent latent state is
> V21 (20 layers, 18 skip carriers,
> ``max_chain_walk_depth=65536``, distractor rank 20). Multi-hop
> chain length grows to 38 over 48 backends (2256 directed edges);
> 14-axis composite trust. ECC reaches **37+ bits/visible-token**
> at full emit (2^35 = 34 359 738 368 codes). Long-horizon
> reconstruction reaches 20 heads at max_k=512. Consensus has 24
> disjoint stages including ``multi_branch_rejoin_arbiter`` and
> ``silent_corruption_plus_member_replacement_arbiter``. The first
> capsule-native team-consensus controller V4 composes regime-
> aware weighted quorum + multi-branch-rejoin arbiter (surviving-
> branch weighted mean) + silent-corruption-plus-replacement
> arbiter (surviving-agents weighted mean) + substrate-replay
> fallback + transcript fallback. Plane A V2 ships six new hosted
> control-plane modules (router V2, logprob router V2, cache-aware
> planner V2 with ≥ 60 % savings on 6×8 staggered prefixes,
> provider filter V2 with ALL/ANY combinators, cost planner V2
> with rotation, boundary V2 with 19 blocked axes + 3 frontier-
> blocked axes). And — the load-bearing operational W69 advance —
> the new ``hosted_real_handoff_coordinator`` records per-turn
> content-addressed handoff envelopes that route turns to
> ``hosted_only`` / ``real_substrate_only`` /
> ``hosted_with_real_substrate_audit`` / ``abstain`` while
> preserving the W68 wall as an invariant, with a falsifier and ≥
> 60 % cross-plane token savings at the default workload. R-156 +
> R-157 + R-158 + R-159 + R-160 deliver **62 H-bars × 3 seeds
> (186 cells)**, all pass. The honest scope is unchanged: hosted
> backends remain text-only at the HTTP surface
> (``W69-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``); the substrate
> is a 16-layer in-repo NumPy runtime, not a frontier model
> (``W69-L-NUMPY-CPU-V14-SUBSTRATE-CAP``); "training" is closed-
> form linear ridge throughout (``W69-L-V14-NO-AUTOGRAD-CAP``);
> the multi-agent task-success wins are measured **inside the W69
> synthetic harness** (``W69-L-MULTI-AGENT-COORDINATOR-V5-
> SYNTHETIC-CAP``), not on real hosted multi-agent backends; and
> the handoff coordinator preserves the wall — it does NOT cross
> the substrate boundary (``W69-L-HANDOFF-NOT-CROSSING-WALL-
> CAP``).*
>
> **Preceding research-line update (W67 Stronger Branch-Merge /
> Role-Dropout Substrate-Coupled Latent Operating System,
> 2026-05-16).** *Twelfth substrate-attack milestone, and the
> **third** multi-agent task-success-bearing substrate milestone
> in the programme — the first to produce wins across **five**
> failure-mode regimes (baseline + team-consensus-under-budget +
> team-failure-recovery + **role-dropout** +
> **branch-merge-reconciliation**), not just three.
> ``MultiAgentSubstrateCoordinatorV3`` runs **eight** matched-
> budget policies (the six W66 policies plus
> ``substrate_routed_v12`` and
> ``team_substrate_coordination_v12``) under five regimes; the V12
> policy **strictly beats V11 across all five regimes** (≥ 73 %
> of seeds per regime: 93 % at baseline; 93 % at
> team_consensus_under_budget; 73 % at team_failure_recovery; 80 %
> at role_dropout; **100 %** at branch_merge_reconciliation). The
> ``team_substrate_coordination_v12`` policy strictly beats
> ``team_substrate_coordination_v11`` on ≥ 47 % of seeds — with
> 91.7 % at baseline. In the role-dropout regime (where one role
> drops out across multiple windows mid-task), the
> ``substrate_routed_v12`` policy alone (no team-consensus
> arbitration active) doubles team success over V11 by leaning on
> the V12 role-dropout-recovery flag. In the branch-merge
> reconciliation regime (where agents fork into branches and
> produce conflicting payloads), the V12 substrate's snapshot-fork
> + branch-merge primitive provides the path V11 cannot. The
> in-repo NumPy substrate grows to V12 (14 layers + four new V12
> axes: per-(layer, head, slot) branch-merge witness tensor,
> per-role-pair role-dropout-recovery flag, substrate snapshot-
> fork primitive, per-layer V12 composite gate score). The
> substrate branch-merge primitive saves **91 %** of recompute
> flops at 128 tokens over a 4-branch reconciliation. Six new
> closed-form ridge solves on top of W61..W66's thirty-five
> (forty-one total). The deep substrate hybrid is now twelve-way
> (V11 hybrid + cache controller V10 + replay controller V8 +
> branch-merge witness + role-dropout recovery + snapshot-fork +
> team-consensus controller V2). The persistent latent state is
> V19 (18 layers, 16 skip carriers,
> ``max_chain_walk_depth=16384``, distractor rank 18). Multi-hop
> chain length grows to 30 over 40 backends (1560 directed edges);
> 12-axis composite trust. ECC reaches **33.333 bits/visible-
> token** at full emit (2^31 = 2 147 483 648 codes). Long-horizon
> reconstruction reaches 18 heads at max_k=384. Consensus has 20
> disjoint stages including ``role_dropout_arbiter`` and
> ``branch_merge_reconciliation_arbiter``. The first capsule-
> native team-consensus controller V2 composes regime-aware
> weighted quorum + branch-merge arbiter (substrate-fork-and-
> merge) + role-dropout-repair head + substrate-replay fallback +
> transcript fallback. R-149 + R-150 + R-151 deliver **56/56
> H-bars × 3 seeds (168/168 cells)**. The honest scope is
> unchanged: hosted backends remain text-only at the HTTP surface
> (``W67-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``); the substrate
> is a small in-repo NumPy runtime, not a frontier model
> (``W67-L-NUMPY-CPU-V12-SUBSTRATE-CAP``); "training" is closed-
> form linear ridge throughout (``W67-L-V12-NO-AUTOGRAD-CAP``);
> and the multi-agent task-success wins are measured **inside the
> W67 synthetic harness**
> (``W67-L-MULTI-AGENT-COORDINATOR-V3-SYNTHETIC-CAP``), not on
> real hosted multi-agent backends.*
>
> **Preceding research-line update (W66 Stronger Solving-Context
> Substrate-Coupled Latent Operating System, 2026-05-16).**
> *Eleventh substrate-attack milestone, and the **second** multi-
> agent task-success-bearing substrate milestone in the programme —
> the first to produce wins across **three** failure-mode regimes
> (baseline + team-consensus-under-budget + team-failure-recovery),
> not just one. ``MultiAgentSubstrateCoordinatorV2`` runs **six**
> matched-budget policies (``transcript_only`` /
> ``shared_state_proxy`` / ``substrate_routed_v9`` /
> ``substrate_routed_v10`` / ``substrate_routed_v11`` /
> ``team_substrate_coordination_v11``) under three regimes; the V11
> policy **strictly beats V10 on ≥ 93 %** of seeds at baseline and
> at team_consensus_under_budget; the
> team_substrate_coordination_v11 policy **strictly beats V11 on
> ≥ 73 %** of seeds in each regime; in the team_failure_recovery
> regime (where one agent silently produces zero output mid-task),
> the team_substrate_coordination_v11 policy reaches **80 % team
> success** while every other policy is stuck at 40 %. The in-repo
> NumPy substrate grows to V11 (13 layers + four new V11 axes:
> per-(layer, head, slot) replay-trust ledger, per-role team-
> failure-recovery flag, substrate snapshot-diff primitive, per-
> layer V11 composite gate score). The substrate snapshot-diff
> primitive saves **92 %** of recompute flops at 128 tokens, vs
> V10's 90 %. Six new closed-form ridge solves on top of
> W61..W65's twenty-nine (thirty-five total). The deep substrate
> hybrid is now eleven-way (V10 hybrid + cache controller V9 +
> replay controller V7 + replay-trust ledger + team-failure-
> recovery flag + substrate snapshot-diff + team-consensus
> controller). The persistent latent state is V18 (17 layers, 15
> skip carriers, ``max_chain_walk_depth=8192``, distractor rank 16).
> Multi-hop chain length grows to 26 over 36 backends (1260
> directed edges); 11-axis composite trust. ECC reaches **31.333
> bits/visible-token** at full emit (2^29 = 536 870 912 codes).
> Long-horizon reconstruction reaches 17 heads at max_k=320.
> Consensus has 18 disjoint stages including
> ``team_failure_recovery_arbiter`` and
> ``team_consensus_under_budget_arbiter``. The first capsule-
> native team-consensus controller composes regime-aware weighted
> quorum + abstain + substrate-replay fallback + transcript
> fallback. R-146 + R-147 + R-148 deliver **56/56 H-bars × 3 seeds
> (168/168 cells)**. The honest scope is unchanged: hosted backends
> remain text-only at the HTTP surface
> (``W66-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``); the substrate
> is a small in-repo NumPy runtime, not a frontier model
> (``W66-L-NUMPY-CPU-V11-SUBSTRATE-CAP``); "training" is closed-
> form linear ridge throughout (``W66-L-V11-NO-AUTOGRAD-CAP``); and
> the multi-agent task-success wins are measured **inside the W66
> synthetic harness**
> (``W66-L-MULTI-AGENT-COORDINATOR-SYNTHETIC-CAP``), not on real
> hosted multi-agent backends.*
>
> **Preceding research-line update (W65 Team-Substrate-Coordination
> Substrate-Coupled Latent Operating System, 2026-05-16).**
> *Tenth substrate-attack milestone, and the first **multi-agent
> task-success-bearing** substrate milestone in the programme.
> ``MultiAgentSubstrateCoordinator`` runs four matched-budget
> policies (``transcript_only`` / ``shared_state_proxy`` /
> ``substrate_routed_v9`` / ``substrate_routed_v10``) on the same
> synthetic deterministic task; the V10 policy **strictly beats
> each baseline on ≥ 50 % of seeds** while using ≤ 17 % of the
> transcript-only visible-token budget. The in-repo NumPy
> substrate grows to V10 (12 layers + four new V10 axes:
> per-(layer, head, slot) hidden-write-merit, per-role KV bank,
> substrate checkpoint/restore primitive, per-layer V10 composite
> gate score). Six new closed-form ridge solves on top of
> W61+W62+W63+W64's twenty-three (twenty-nine total). The deep
> substrate hybrid is now ten-way (V9 hybrid + cache controller
> V8 + replay controller V6 + hidden-write-merit +
> attention-fingerprint + prefix V9 + HSB V9 hidden-wins-rate +
> role KV bank + multi-agent coordinator). The persistent latent
> state is V17 (16 layers, 14 skip carriers,
> ``max_chain_walk_depth=8192``, distractor rank 14). Multi-hop
> chain length grows to 25 over 35 backends (1190 directed
> edges); 10-axis composite trust. ECC reaches 29.333
> bits/visible-token at full emit (2^27 = 134 217 728 codes).
> Long-horizon reconstruction reaches 16 heads at max_k=256.
> Consensus has 16 disjoint stages including
> ``team_substrate_coordination_arbiter`` and
> ``multi_agent_abstain_arbiter``. R-143 + R-144 + R-145 deliver
> 50/50 H-bars × 3 seeds (150/150 cells). The honest scope is
> unchanged: hosted backends remain text-only at the HTTP surface
> (``W65-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``); the substrate
> is a small in-repo NumPy runtime, not a frontier model
> (``W65-L-NUMPY-CPU-V10-SUBSTRATE-CAP``); "training" is
> closed-form linear ridge throughout
> (``W65-L-V10-NO-AUTOGRAD-CAP``); and the multi-agent
> task-success win is measured **inside the W65 synthetic harness**
> (``W65-L-MULTI-AGENT-COORDINATOR-SYNTHETIC-CAP``), not on real
> hosted multi-agent backends.*
>
> **Preceding research-line update (W64 Replay-Dominance-Primary
> Hidden-Wins-Primary 6144-Turn Nine-Way Substrate-Coupled Latent
> Operating System, 2026-05-15).**
> *Ninth substrate-attack milestone. The in-repo NumPy substrate
> grows to V9 (11 layers / GQA 8q-4kv / d_model=64 / RMSNorm /
> SwiGLU). Five new substrate-internal axes: per-(layer, head, slot)
> hidden-wins-primary tensor, per-(layer, head, slot) replay-
> dominance-witness channel, per-layer attention-entropy probe,
> per-(layer, head, slot, slot) cache-similarity matrix, and per-
> (layer, head) hidden-state-trust ledger. Six new closed-form
> ridge solves on top of W61+W62+W63's seventeen (twenty-three
> total). The deep substrate hybrid is now nine-way (V9 substrate
> + cache controller V7 + replay controller V5 + retrieval head +
> attention-steering V8 + four-way bridge classifier + prefix-state
> V8 + HSB V8). The persistent latent state is V16 (15 layers, 13
> skip carriers, max_chain_walk_depth=6144, distractor rank 12).
> Multi-hop chain length grows to 21 over 27 backends. ECC reaches
> 27.333 bits/visible-token at full emit. Long-horizon
> reconstruction reaches 15 heads at max_k=192. R-137 + R-138 +
> R-139 deliver 51/51 H-bars × 3 seeds (153/153 cells). The honest
> scope is unchanged: hosted backends remain text-only at the HTTP
> surface (``W64-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``); the
> substrate is a small in-repo NumPy runtime, not a frontier model
> (``W64-L-NUMPY-CPU-V9-SUBSTRATE-CAP``); and "training" is
> closed-form linear ridge throughout (``W64-L-V9-NO-AUTOGRAD-CAP``).*
>
> **Preceding research-line update (W60 Trainable Cache-Control
> Substrate-Coupled Latent Operating System, 2026-05-14).**
> *Fifth substrate-attack milestone in the programme; first
> **multi-direction multi-target closed-form ridge fits** across
> several substrate-facing surfaces simultaneously (KV bridge V5
> over `n_directions` orthogonal correction directions; HSB V4
> per-(layer, head) δ tensor; cache controller V3
> ``trained_eviction`` + ``composite_v3`` mixture); first first-
> class **state-reuse-vs-recompute-vs-fallback-vs-abstain
> ReplayController**; first **five-way substrate ↔ V6 ↔ cache
> controller V3 ↔ replay controller ↔ retrieval head** hybrid
> loop; first **hidden-vs-KV head-to-head** harness on a fixed
> target logit direction (falsifiable: at least one arm wins or
> it's a tie). The post-W59 question — *can the in-repo
> substrate carry trained controllers across multiple substrate-
> facing features in concert (not just one head at a time), and
> can a first-class replay-vs-recompute policy be wired into the
> substrate without losing the ridge-only honest-scope
> boundary?* — is answered by W60 with eighteen orthogonal
> advances. The substrate gets richer
> (``coordpy.tiny_substrate_v5``: 7 layers, 8 query heads, 4 KV
> heads with GQA, ``d_model=64``, RMSNorm, SwiGLU FF, **per-
> (layer, head, position) cumulative attention-receive matrix**
> that survives evictions, **per-(layer, head) linearised logit
> Jacobian table**, **per-(layer, position) corruption flag
> channel**, **multi-segment partial-prefix reuse** with reuse +
> recompute + drop kinds and per-segment flop accounting). Six
> W60 substrate-facing controllers sit on top, all with closed-
> form ridge-fitted parameters: ``coordpy.kv_bridge_v5`` solves a
> real ``(n_directions × n_directions)`` ridge linear system over
> substrate-side Jacobian estimates from finite differences and
> exposes a separate logit-direction fit, two correction layers,
> all-bank fingerprint over {bank_a, bank_b, bank_c, bank_d}, and
> a ``extract_carrier_from_v5_kv_cache`` reverse-extract that
> returns the carrier estimate via least-squares against the V5
> projection matrix (residual L2 < 1e-3 on uncorrupted banks);
> ``coordpy.hidden_state_bridge_v4`` solves a 16-dim ridge
> problem for the per-(layer, head) inject-scale δ tensor that
> brings the projected last-position logit shift closest to a
> caller-supplied ``Δlogit`` direction, ships a recovery path
> against adversarial per-(layer, head) attacks, and a KV-vs-
> Hidden head-to-head harness on the same target;
> ``coordpy.prefix_state_bridge_v4`` ships multi-segment partial
> reuse on the V5 substrate with chain forward over a follow-up
> sequence reporting per-step drift L2 + cumulative envelope (~46%
> flop saving on the standard split);
> ``coordpy.attention_steering_bridge_v4`` enforces per-(layer,
> head, query) 3-D KL budgets and reports L1 attention-mass shift
> alongside KL plus a negative-budget falsifier (budget=0 ⇒
> post-KL < 1e-6); ``coordpy.cache_controller_v3`` keeps V2's
> five policies and adds four new — ``learned_attention_receive``
> (closed-form ridge over the per-(layer, head) cumulative
> attention-receive feature), ``learned_corruption_aware`` (V2 +
> hard ``-inf`` floor on flagged slots), ``trained_eviction``
> (closed-form ridge over a `[hidden, importance,
> attention_receive_l1, retrieval]` feature against the V1 leave-
> one-out drop oracle, reducing residual from 28.4 → 0.39 on
> R-125), and ``composite_v3`` (4-feature ridge mixture);
> ``coordpy.replay_controller`` is the programme's first first-
> class state-reuse-vs-recompute-vs-fallback-vs-abstain policy
> with audit log + flop-vs-drift trade-off curve. The five-way
> ``coordpy.deep_substrate_hybrid_v5`` couples V6 ↔ substrate V5
> ↔ cache controller V3 ↔ replay controller ↔ retrieval head;
> its forward witness reports ``five_way=True`` end-to-end. The
> persistent latent state advances to V12 (10 layers, octuple
> skip-link adding a replay-controller-decision EMA,
> ``max_chain_walk_depth = 1024``, distractor-resistant projection
> against a Gram-Schmidt-fitted random orthonormal basis), multi-
> hop translator V10 spans 16 backends with 240 directed edges
> and chain-length-15 with a five-axis substrate × hidden ×
> attention × retrieval × replay trust composite + a compromise-
> threshold detector, MLSC V8 propagates `replay_witness_chain` +
> `substrate_witness_chain` + `provenance_trust_table`, consensus
> V6 has a 10-stage chain adding `replay_controller_choice`
> between `retrieval_replay` and `best_parent`, CRC V8 doubles
> the KV fingerprint bucket count to 256 with wrap-around-XOR so
> single-byte flips are *always* detectable, ships a
> `recover_v8_kv_cache` operator that writes per-(layer, position)
> corruption flags into the V5 cache, an adversarial 11-bit burst
> family, and a post-replay top-K agreement floor ≥ pre-replay,
> LHR V12 adds an 11th replay-conditioned reconstruction head at
> ``max_k=96`` and a two-layer scorer (random projection +
> frozen ReLU + closed-form ridge over the post-ReLU features),
> ECC V12 layers a K11 stage on top of V11 (2 097 152 codes =
> 2^21; 23.333 bits/visible-token at full emit), TVS V9 grows to
> 10 arms adding `replay_controller_choice`, uncertainty V8 grows
> to 7 axes adding `replay_fidelity`, disagreement algebra V6
> adds a replay-controller equivalence identity with falsifier,
> and substrate adapter V5 introduces seven new capability axes
> plus a new top tier `substrate_v5_full` reached only by the V5
> in-repo runtime. R-125 (20 cell families) + R-126 (13 cell
> families) + R-127 (12 cell families) at 3 seeds verify **45/45
> H-bars (H125..H143b) pass 3/3 seeds (135/135 cells)**.
> Cumulative trust boundary across W22..W60 = **715 enumerated
> failure modes**. ``W60-L-V5-NO-AUTOGRAD-CAP`` documents that
> every fit is a single-step closed-form linear ridge over a
> small subspace (no SGD, no autograd, no GPU); ``W60-L-NO-THIRD-
> PARTY-SUBSTRATE-COUPLING-CAP`` carries forward unchanged;
> ``W60-L-V12-OUTER-NOT-TRAINED-CAP``, ``W60-L-ECC-V12-RATE-
> FLOOR-CAP``, ``W60-L-LHR-V12-SCORER-FIT-CAP``,
> ``W60-L-CORRUPTION-FLAG-CHANNEL-CAP``, ``W60-L-MULTI-HOP-V10-
> SYNTHETIC-BACKENDS-CAP``, and ``W60-L-V12-PERMUTATION-
> INVARIANCE-CAP`` are the additional honest caps;
> ``W60-C-DEEP-TRANSFORMER-COUPLING``, ``W60-C-AUTOGRAD-OUTER-
> LEARNED-OS``, and ``W60-C-HIDDEN-DOMINATES-KV-AT-SCALE`` carry
> forward as conjectures on frontier-scale substrate access and
> autograd-trained controllers. W60 is reachable only through
> explicit imports; ``coordpy.__version__`` remains ``0.5.20``;
> the released v0.5.20 wheel's public surface is byte-for-byte
> unchanged; no PyPI release.
>
> **Post-W58 research-line update (W59 Trainable Substrate-
> Conditioned Latent Operating System, 2026-05-14).**
> *Fourth substrate-attack milestone in the programme; first
> **closed-form ridge fit of a real ``d × d`` substrate-facing
> matrix** (the cache-controller V2 bilinear retrieval matrix,
> fit by closed-form ridge over a ``d² = 4096``-dim outer-product
> feature against the substrate's leave-one-out drop oracle) and
> first **four-way substrate ↔ V6 ↔ cache-controller ↔
> retrieval-head** hybrid loop.* The post-W58 question — *can
> the in-repo substrate carry **real** trained parameters (not
> just inject-scale hyperparameters) and route real per-(layer,
> head) KL budgets, all by closed-form linear algebra?* — is
> answered by W59 with eighteen orthogonal advances. The
> substrate gets richer (``coordpy.tiny_substrate_v4``: 6 layers,
> 8 query heads, 4 KV heads with **GQA**, ``d_model=64``,
> RMSNorm, SwiGLU FF, cumulative-EMA KV importance, real fp64
> flop counter, **partial-prefix split-and-replay** with dual-
> stage flop accounting, **per-(layer, head) hidden-state tap**
> of shape ``(n_heads, n_tokens, d_head)``, **128-bucket**
> Reed-Solomon-style KV cache fingerprint, exact-under-
> linearised-head **logit-Jacobian probe**). Five W59
> substrate-facing bridges sit on top, all with *closed-form
> ridge-fitted* parameters: ``coordpy.kv_bridge_v4`` fits a 1-D
> correction α along a fixed random direction by central FD +
> ridge solve and exposes **four** role banks
> ``bank_a``/``bank_b``/``bank_c``/``bank_d``;
> ``coordpy.hidden_state_bridge_v3`` solves a 1-D ridge for the
> inject-scale α that brings the projected last-position logit
> shift closest to a caller-supplied ``Δlogit`` direction;
> ``coordpy.prefix_state_bridge_v3`` ships **partial-prefix
> reuse** that is byte-identical to a full recompute on the
> reusable head span (≤ 4.4e-16 max-abs diff on the R-122 H112
> probe), records a **K-seed drift spectrum** (mean / max / min /
> var) and an empirical Lipschitz certificate ratio, and uses a
> 128-bucket fingerprint; ``coordpy.attention_steering_bridge_v3``
> fits per-(layer, head) KL clips iteratively (vs W58 V2's
> global clip) so every head is individually honoured at
> ``budget=0.6`` on the H110 probe; ``coordpy.cache_controller_v2``
> keeps W58's three policies and adds two new closed-form ridge
> policies — ``learned_hidden`` on a cross-layer hidden-state
> feature and ``learned_retrieval`` whose bilinear ``M`` matrix
> is the programme's first real ``d × d`` substrate-facing
> matrix fit end-to-end by closed-form ridge against the
> substrate's leave-one-out drop oracle (>4 OOM residual
> reduction routinely on the R-122 probe). The **four-way**
> ``coordpy.deep_substrate_hybrid_v4`` couples V6 ↔ substrate V4
> ↔ cache-controller V2 ↔ retrieval head; its forward witness
> reports ``four_way=True`` and ``retrieval_used=True``
> simultaneously. The persistent latent state advances to V11
> (9 layers, **septuple** skip-link adding a retrieval-EMA,
> ``max_chain_walk_depth = 768``), multi-hop translator V9 spans
> 14 backends with 182 directed edges and chain-length-13 with a
> **four-axis** substrate × hidden × attention × retrieval trust
> composite, MLSC V7 propagates ``retrieval_witness_chain`` and
> ``controller_witness_cid``, consensus V5 has a **9-stage**
> chain adding ``retrieval_replay`` between ``cache_reuse_replay``
> and ``best_parent``, CRC V7 doubles the KV fingerprint bucket
> count to **128** and adds cache-retrieval top-K agreement
> under non-target corruption (>= 70%) plus a **9-bit**
> adversarial burst family, LHR V11 adds a 10th retrieval-
> conditioned reconstruction head at ``max_k=80`` and a
> closed-form ridge **retention scorer**, ECC V11 layers a K10
> stage on top of V10 (**1 048 576** codes = 2^20; **22.333
> bits/visible-token** at full emit), TVS V8 grows to **9 arms**
> adding ``retrieval_replay``, uncertainty V7 grows to **6
> axes** adding ``retrieval_fidelity``, disagreement algebra V5
> adds a retrieval-equivalence identity, and substrate adapter
> V4 introduces five new capability axes plus a new top tier
> ``substrate_v4_full`` reached only by the V4 in-repo runtime.
> R-122 (15 cell families) + R-123 (12 cell families) + R-124
> (11 cell families) at 3 seeds verify **38/38 H-bars (H107..
> H124) pass 3/3 seeds**. Cumulative trust boundary across
> W22..W59 = **663 enumerated failure modes**.
> ``W59-L-V4-NO-AUTOGRAD-CAP`` documents that every fit is a
> single-step closed-form linear ridge over a small subspace
> (no SGD, no autograd, no GPU); ``W59-L-NO-THIRD-PARTY-
> SUBSTRATE-COUPLING-CAP`` carries forward unchanged;
> ``W59-L-V11-OUTER-NOT-TRAINED-CAP``, ``W59-L-ECC-V11-RATE-
> FLOOR-CAP``, ``W59-L-LHR-V11-SCORER-FIT-CAP``,
> ``W59-L-MULTI-HOP-V9-SYNTHETIC-BACKENDS-CAP``, and ``W59-L-V11-
> PERMUTATION-INVARIANCE-CAP`` are the additional honest caps;
> ``W59-C-DEEP-TRANSFORMER-COUPLING`` and
> ``W59-C-FRONTIER-SCALE-SUBSTRATE-LIFT`` carry forward as
> conjectures on frontier-scale model substrate access. W59 is
> reachable only through explicit imports; ``coordpy.__version__``
> remains ``0.5.20``; the released v0.5.20 wheel's public surface
> is byte-for-byte unchanged; no PyPI release.
>
> **Post-W57 research-line update (W58 Deep Cache-Reuse
> Substrate-Coupled Latent Operating System, 2026-05-13).**
> *Third substrate-attack milestone in the programme; first
> **three-way** substrate breach with **real fp64 flop savings**
> as a benchmark-load-bearing axis.* The post-W57 question —
> *can the in-repo substrate be made not just real but
> economically real, with measurable cache-reuse-vs-recompute
> savings and trainable bridge scales fit against substrate-side
> targets?* — is answered by W58 with eighteen orthogonal
> advances. The substrate gets richer
> (``coordpy.tiny_substrate_v3``: 5 layers, 8 query heads,
> 4 KV heads with **GQA**, ``d_model=64``, RMSNorm, SwiGLU FF,
> per-token KV importance tracking, real fp64 flop counter,
> partial-forward, 64-bucket Reed-Solomon KV cache fingerprint).
> Five W58 substrate-facing bridges sit on top, all with
> *fitted* parameters: ``coordpy.kv_bridge_v3`` fits per-(layer,
> head) inject scales by coordinate descent against an explicit
> L2-perturbation target and exposes role-conditioned KV banks
> ``bank_a``/``bank_b``; ``coordpy.hidden_state_bridge_v2``
> multi-layer fits to a logit-shift target;
> ``coordpy.prefix_state_bridge_v2`` ships a **real fp64 flop-
> saved counter** — the H100b probe shows **66.7% flop savings**
> at byte-identical logits (≤ 4.4e-16 max-abs diff vs full
> recompute), the first benchmark in the programme where the
> substrate's *own* flop count is materially load-bearing;
> ``coordpy.attention_steering_bridge_v2`` enforces a KL budget
> by coordinate descent on a global bias clip and exposes per-
> head ablation; ``coordpy.cache_controller`` ships three
> retention policies — uniform / importance / **learned** — the
> learned head being a single closed-form-ridge scoring vector
> fit against a leave-one-out drop oracle (no autograd, no
> backprop). The **three-way** ``coordpy.deep_substrate_hybrid_v3``
> couples V6 ↔ substrate V3 ↔ cache controller; its forward
> witness reports ``three_way=True``, ``substrate_back_l2 > 0``,
> ``ablation_perturbation_l2 > 0``, and
> ``cache_eviction_perturbation_l2 > 0`` simultaneously.
> The persistent latent state advances to V10 (8 layers,
> **sextuple** skip-link adding an attention-pattern-conditioned
> EMA, ``max_chain_walk_depth = 512``), multi-hop translator V8
> spans 12 backends with 132 directed edges and chain-length-11
> with a three-axis substrate × hidden × attention trust
> composite, MLSC V6 propagates ``attention_witness_chain`` and
> ``cache_reuse_witness_cid``, consensus V4 has an **8-stage**
> chain adding ``cache_reuse_replay`` between ``logit_lens`` and
> ``best_parent``, CRC V6 doubles the KV fingerprint bucket count
> to 64 and adds prefix-state corruption detection (1.0 detect
> rate on the H92/H92b probes), LHR V10 adds a 9th
> attention-conditioned reconstruction head at ``max_k=72``, ECC
> V10 layers a K9 stage on top of V9 (524 288 codes; **21.333
> bits/visible-token** at full emit), TVS V7 grows to **8 arms**
> adding ``cache_reuse_replay``, uncertainty V6 grows to **5
> axes** adding ``cache_reuse_fidelity``, disagreement algebra
> V4 adds a cache-reuse-equivalence identity, and substrate
> adapter V3 introduces five new capability axes plus a new top
> tier ``substrate_v3_full`` reached only by the V3 in-repo
> runtime. R-119 (16 cell families) + R-120 (12 cell families)
> + R-121 (12 cell families) at 3 seeds verify **40/40 H-bars
> (H86..H106) pass 3/3 seeds** — strong success per the W58
> pre-committed criterion. Cumulative trust boundary across
> W22..W58 = **614 enumerated failure modes** (46 new W58
> envelope verifier modes). ``coordpy.__version__`` remains
> ``0.5.20``; SDK contract is byte-for-byte unchanged.
> ``W58-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
> forward; ``W58-L-V3-NO-BACKPROP-CAP`` is the load-bearing new
> honest cap — W58 fits only inject-scale and a single linear
> scoring head, with no end-to-end backprop;
> ``W58-C-DEEP-TRANSFORMER-COUPLING`` and
> ``W58-C-FRONTIER-SCALE-SUBSTRATE-LIFT`` carry forward as
> conjectures on frontier-scale model substrate access.
>
> **Earlier post-W56 research-line update (W57 Deep
> Substrate-Coupled Latent Operating System, 2026-05-13).**
> *Second substrate-attack milestone in the programme; first
> **bidirectional** substrate breach.* W56 proved that a real in-repo transformer
> substrate (``coordpy.tiny_substrate``) is enough to
> demonstrate one-way KV / hidden-state coupling. W57 deepens
> this on three axes: (a) the substrate gets richer
> (``coordpy.tiny_substrate_v2`` — 4 layers, 8 heads,
> ``d_model=64``, RoPE rotary embeddings, per-layer logit lens,
> KV cache eviction (LRU + importance-weighted), prefix-state
> extraction as a first-class content-addressed object, per-
> head pre-softmax attention bias hook); (b) three new
> substrate bridges are added on top of W56's KV bridge —
> ``coordpy.hidden_state_bridge`` (additive residual-stream
> injection at any layer; the substrate's logits change
> measurably with cross-entropy delta ≈ 0.10 on the H52
> probe), ``coordpy.prefix_state_bridge`` (save / load /
> detect-corruption: reuse-vs-recompute byte-identical to
> ≤ 5e-16 max-abs diff on the H47 probe), and
> ``coordpy.attention_steering_bridge`` (per-(layer, head,
> query, key) bias tensor producing measurable mean-KL ≈ 5.59
> nats per layer × 4 layers on the H53 probe, with the causal
> mask still strict); (c) **going bidirectional**:
> ``coordpy.deep_substrate_hybrid_v2`` lets the substrate's
> intermediate hidden state project back into the capsule
> layer's V6 residual stream BEFORE the V6 forward, in
> addition to W56's V6 → substrate KV path. The
> ``W57HandoffEnvelope.bidirectional_used`` flag is True in
> end-to-end runs. Plus the V8 stack is promoted to V9 with
> substrate / hidden-state awareness throughout: V9 persistent
> latent state (``coordpy.persistent_latent_v9`` — 7 layers,
> **quintuple** persistent skip-link {anchor + fast EMA + slow
> EMA + substrate-conditioned EMA + hidden-state-conditioned
> EMA}, ``max_chain_walk_depth = 384``, substrate-fidelity
> damping); 10-backend chain-length-9 multi-hop V7
> (``coordpy.multi_hop_translator_v7``); MLSC V5 with
> ``hidden_state_witness_chain``, ``attention_witness_cid``,
> per-head trust, two new algebra signatures
> {``substrate_project``, ``hidden_inject``}; consensus V3 with
> a 7-stage chain (adds ``logit_lens_conditioned`` between
> ``substrate`` and ``best_parent``); CRC V5 with 3-D
> interleaving (4×4×4 = 64-bit blocks), 9-of-13 majority, and
> KV cache fingerprinting (Reed-Solomon-style 32-bucket XOR
> fingerprints — H75 KV-corruption detect rate = 1.0); 8-head
> LHR V9 (``max_k = 64``, hidden-state-conditioned head); ECC
> V9 (262144 codes, **20.333 bits/visible-token** at full
> emit); 7-arm TVS V6 (adds ``substrate_hidden_inject``);
> hidden-state-fidelity 4-axis uncertainty V5 with adversarial
> pessimistic / optimistic brackets; disagreement algebra V3
> with hidden-projection identity. Substrate adapter V2
> (``coordpy.substrate_adapter_v2``) adds four new capability
> axes (``attention_bias_write``, ``prefix_state_reuse``,
> ``cache_eviction``, ``logit_lens``) and a new ``substrate_v2_full``
> tier reached only by the V2 in-repo runtime; hosted backends
> remain text-only. R-116 (14 cell families) + R-117 (14 cell
> families) + R-118 (15 cell families) at 3 seeds verify
> **43/43 H-bars (H43..H85) pass 3/3 seeds** — strong success
> per the W57 pre-committed criterion. Cumulative trust
> boundary across W22..W57 = **568 enumerated failure modes**.
> ``coordpy.__version__`` remains ``0.5.20``; SDK contract is
> byte-for-byte unchanged. ``W57-L-NO-THIRD-PARTY-SUBSTRATE-
> COUPLING-CAP`` records that hosted backends remain text-only
> at the HTTP surface; ``W57-C-DEEP-TRANSFORMER-COUPLING`` is
> a sharper restatement of the open question on frontier-scale
> models; ``W57-C-FRONTIER-SCALE-SUBSTRATE-LIFT`` conjectures
> that the W57 bridges (KV V2 + hidden-state + prefix-state +
> attention-steering) would scale-monotonically improve
> usefulness if frontier runtimes exposed compatible hooks.
>
> **Earlier post-W55 research-line update (W56 Substrate-Coupled
> Latent Operating System, 2026-05-13).** *First substrate-attack
> milestone in the programme.* The post-W55 question — *how do
> we actually breach the substrate layer instead of carrying it
> forward as a permanent conjecture?* — is answered by W56 with
> twelve orthogonal advances. The load-bearing move: build a
> **tiny in-repo executable transformer substrate**
> (``coordpy.tiny_substrate``) where multi-head causal self-
> attention, per-layer KV cache, hidden states, layer norm,
> position-wise feed-forward, and logits are *not metaphorical*.
> The substrate is small (2 layers, 4 heads, ``d_model=32``,
> byte-vocab, pure NumPy) and untrained, but it is real and
> deterministic, and it is something the capsule layer can
> compose with. (i) ``coordpy.substrate_adapter`` honestly probes
> each backend's capability surface and classifies into one of
> {``substrate_full``, ``embeddings_only``, ``logits_only``,
> ``text_only``, ``unreachable``}; only the in-repo runtime
> reaches ``substrate_full``. (ii) ``coordpy.kv_bridge`` projects
> capsule-layer latent carriers into per-layer (K, V) slot pairs
> and injects them into the substrate's KV cache; the inject +
> forward step produces a replay-deterministic, content-
> addressed, measurable logit perturbation. (iii) The capsule-
> layer V8 persistent state, MLSC V4, multi-hop V6, consensus
> V2, CRC V4, LHR V8, ECC V8, TVS V5, uncertainty V4, and
> disagreement algebra V2 all extend their W55 predecessors
> with one new substrate-aware affordance: V8 has a substrate-
> conditioned EMA skip; MLSC V4 carries a substrate-witness CID;
> the consensus controller picks a substrate-conditioned stage
> when capsule consensus is split; the TVS V5 arbiter has a
> ``substrate_replay`` arm; the LHR V8 has a substrate-
> conditioned head; uncertainty V4 down-weights low-substrate-
> fidelity components; disagreement algebra V2 adds a
> substrate-projection identity. (iv) The most load-bearing
> piece is ``coordpy.deep_substrate_hybrid``, which replaces
> the top of the W55 V6 deep proxy stack with the **real tiny
> substrate attention block** — reading and writing the **real**
> per-layer KV cache. Across R-113 + R-114 + R-115 at 3 seeds:
> **38 of 42 H-bars pass 3/3 seeds** (strong success per the
> pre-committed criterion); **4 H-bars reproduce as honest
> caps** that document genuine limitations (untrained V8 outer,
> BCH 4-bit pathology on small probes, V8 permutation invariance,
> 5-bit burst silent failure). The cumulative envelope chain is
> ``w47 → w48 → ... → w55 → w56``. Cumulative trust boundary
> across W22..W56 = **524 enumerated failure modes**. The
> ``W55-C-DEEP-TRANSFORMER-COUPLING`` conjecture is sharpened to
> ``W56-C-DEEP-TRANSFORMER-COUPLING``: the **frontier-model
> third-party hosted substrate** remains substrate-blocked.
> ``W56-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` makes this
> explicit. The original Context Zero goal — solving context
> for multi-agent teams — is materially closer in one specific
> way: we now have a code-backed answer to "can a capsule
> carrier measurably steer a real substrate forward?" — *yes,
> within the bounded universe of the tiny in-repo substrate*,
> with the apparatus to repeat the experiment on any future
> substrate that exposes the same hooks. The frontier-model
> side remains the open conjecture.
>
> **Earlier research-line update (W55 Deep Trust-Weighted
> Disagreement-Algebraic Latent Operating System, 2026-05-12).**
> The post-W54 question — *how do we make the latent operating
> system trust-weighted, disagreement-algebraic, double-bit-
> correcting, fact-graph-aware, and adversarially calibrated?*
> — is answered by W55 with eleven orthogonal mechanism advances:
> (i) 5-layer V7 persistent latent state with triple skip-link
> (turn-0 anchor + fast EMA + slow EMA) and chain walk depth 128;
> (ii) 7-backend hept multi-hop V5 translator with chain-length-6
> transitivity and trust-weighted compromise arbitration; (iii)
> Mergeable Latent State Capsule V3 carrying disagreement algebra
> primitives ⊕/⊖/⊗, per-fact confirmation count in the fact-graph
> DAG, and an explicit trust signature that *decays* each turn
> unless reinforced by a merge — anti-stale-trust hygiene; (iv) a
> trust-weighted consensus controller with continuous quorum
> (`Σ trust_i ≥ threshold`) plus a 5-stage decision chain
> {K-of-N → trust-weighted → best-parent → transcript → abstain},
> each stage content-addressed in the audit; (v) BCH(15,7) double-
> bit correction + 5-of-7 repetition + bit-interleaving for burst
> recovery; (vi) L=14 deep proxy stack V6 with trust-projected
> residual gating, a disagreement-algebra output head, and an
> adaptive abstain threshold that scales with the input's L2
> norm; (vii) a 6-head long-horizon reconstruction V7 with a
> cross-cycle head at `max_k=36`; (viii) a 6-level ECC codebook
> V7 (K1=32 × K2=16 × K3=8 × K4=4 × K5=2 × K6=2) plus BCH(15,7)
> per-segment achieving **18.333 bits/visible-token** at full
> emit; (ix) a 5-arm transcript-vs-shared arbiter V4 with a
> per-arm budget allocator (retention × trust softmax); (x) an
> uncertainty layer V3 with per-fact uncertainty propagation
> through the fact-graph DAG, adversarial calibration under
> bounded perturbation, and a trust-weighted composite that
> down-weights low-trust components; and (xi) a first-class
> Disagreement Algebra module exposing ⊕/⊖/⊗ as content-
> addressed primitives with algebraic identities — idempotent
> ⊕, ⊖ self-cancellation, and ⊗ distributivity on the agreement
> subspace — all proved by inspection and empirically verified.
> The cumulative envelope chain is
> `w47 → w48 → w49 → w50 → w51 → w52 → w53 → w54 → w55`. R-110
> (12 cell families) + R-111 (10 cell families) + R-112 (16
> cell families) at 3 seeds verify H1..H38; cumulative trust
> boundary across W22..W55 = **486 enumerated failure modes**.
> The W54-C-DEEP-TRANSFORMER-COUPLING conjecture is further-
> bounded as W55-C-DEEP-TRANSFORMER-COUPLING but unchanged in
> spirit: real KV bytes, hidden states, attention weights,
> embeddings, and real tokenizers remain substrate-blocked. W55
> is the strongest *executable proxy* for transformer-internal
> coupling we can write today at the capsule layer.
>
> **Post-W48 research-line update (W49 Multi-Block Cross-Bank
> Coordination, 2026-05-11).** After W48 introduced a shared
> base state + single pseudo-KV bank + single-block multi-head
> proxy attention, the programme pushed simultaneously on
> *depth*, *role-conditioned multi-bank memory*, *learned
> eviction*, *retention*, and *dictionary compression*: W49
> stacks `L_p = 2` `ProxyTransformerBlock`s (each with multi-
> head attention + position-wise tanh feed-forward + trainable
> residual scale); runs **per-role + shared multi-bank pseudo-
> KV** with a trainable `BankRouter` for writes and a trainable
> `BankMixGate` for reads; replaces the FIFO ring with a
> trainable `EvictionPolicy` that scores slots by
> `(age, role_match, write_gate)`; adds a separate
> `RetentionHead` answering "was this fact stored?"; trains a
> `K`-prototype `DictionaryCodebook` that quantises the
> latent-control payload to a single codebook index emitted as
> a `LATENT_CTRL_V2` block; and evolves a content-addressed
> `SharedLatentCapsule` per turn whose chain is recoverable from
> the envelope chain alone. Each turn binds a per-turn
> `CrammingWitness` recording structured-bits / visible-token
> frontier. The W49 envelope verifier enumerates 22 disjoint
> failure modes; **cumulative trust boundary across W22..W49 =
> 323 enumerated failure modes**. Headline R-96 / R-97 results
> (3 seeds × 16 cell families): multi-block depth +0.292 acc on
> a three-way XOR composition; multi-bank recall +0.208 cosine;
> learned eviction +0.859 cosine on overflow; retention head
> +0.500 binary accuracy; dictionary saves 25% of control-block
> tokens; shared-latent chain-walk 1.000; cross-bank
> interference 0.000 (proves role isolation); replay
> determinism 1.000; verifier detects 8/8 forgeries;
> long-branch retention +0.268 on a length-12 path; cycle
> reconstruction +0.461 cosine; cramming bits-per-token 5.0 vs
> 3.0 (1.67×); `MultiBlockAwareSyntheticBackend` live anchor
> 1.000 vs 0.000 transcript-replay; aggressive compression
> +0.500 info/visible-token; multi_block_distribution_cap 0.000
> (limitation reproduces). Honest scope: W49 does NOT close any
> of `W43-C-MIXED-CURVATURE-LATENT`,
> `W43-C-COLLECTIVE-KV-POOLING`,
> `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`,
> `W47-C-DEEP-TRANSFORMER-COUPLING`,
> `W48-C-DEEP-TRANSFORMER-COUPLING`,
> `W48-C-REAL-KV-COUPLED-PROXY`,
> `W48-C-MULTI-HOST-SHARED-STATE`. New limitations:
> `W49-L-NO-REAL-KV-CAP` (multi-bank still reproduces the
> algebraic interface, NOT real KV bytes),
> `W49-L-MULTI-BLOCK-DISTRIBUTION-CAP` (strengthens W48-L-
> PROXY-DISTRIBUTION-CAP),
> `W49-L-PURE-PYTHON-TRAINING-COST-CAP` (carries forward),
> `W49-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP` (carries forward).
> New conjectures: `W49-C-DEEP-TRANSFORMER-COUPLING` (carries
> forward, bounds W48-C further) and
> `W49-C-CROSS-MODEL-LATENT-TRANSFER` (cross-tokenizer latent
> transfer requires backend support). The W49 module ships at
> `coordpy.multi_block_proxy` and is reachable only via explicit
> import; the released v0.5.20 wheel's public surface is byte-
> for-byte unchanged. See
> `docs/RESULTS_COORDPY_W49_MULTI_BLOCK_PROXY.md` for the full
> result note and
> `docs/SUCCESS_CRITERION_W49_MULTI_BLOCK_PROXY.md` for the
> pre-committed bar.
>
> **Earlier post-W47 research-line update (W48 Shared-State
> Transformer-Proxy, 2026-05-11).** After W47 made the
> manifold-memory controller end-to-end trainable via pure-
> Python reverse-mode AD + Adam SGD, the programme attacked the
> remaining substrate wall by building the **strongest
> executable proxy for transformer-internal coupling at the
> capsule layer**: a single team-shared base state vector that
> lives across turns and roles; a trainable **pseudo-KV factor
> bank** of low-rank `(K, V)` slots that reproduces the
> algebraic interface of a transformer KV cache
> (`softmax(Q·K^T/sqrt(d))·V` with strict causal masking) at
> the capsule layer; an `H`-head **multi-head proxy attention
> block** with its own trainable per-head `(W_Q, W_K, W_V)` +
> trainable output projection; a **slot-memory write head**
> that decides per turn whether the new observation enters the
> pseudo-KV bank; a **reconstruction decoder** that recovers
> prior-turn flat channel features from the current shared
> state + pseudo-KV read; a **branch/cycle-aware bias matrix**
> that separates two branches with *identical* channel
> features; a **bijective branch-history compressor** that
> packs the team's branch path into a single integer header
> with explicit visible-token savings; a learned **latent
> control serializer** that emits a single `LATENT_CTRL:
> SHARED_STATE_HASH=... mask=... bits=...` line bijective from
> a sealed `LatentControlWitness`; and a content-addressed
> `TrainingTraceWitness` (carried forward from W47). W48 is
> **held outside the stable SDK contract** — it lives at
> `coordpy.shared_state_proxy` and is reachable only via
> explicit import; the released v0.5.20 wheel's public surface
> is byte-for-byte unchanged. The paper's main thesis is
> materially **strengthened** on a new axis by W48: roles now
> share a single content-addressed base state and read / write
> an auditable pseudo-KV factor bank — solving the "every role
> re-bootstraps" problem at the capsule layer with full
> byte-deterministic provenance. The W48 envelope binds the
> shared-state CID, per-role delta CID, multi-head attention
> CID, write-head CID, reconstruction-decoder CID, branch-bias
> CID, latent-control CID, pseudo-KV bank head CID, training-
> trace CID, and all derived witness CIDs under one
> `proxy_outer_cid`, verified through 22 enumerated W48 failure
> modes disjoint from W22..W47 (cumulative trust boundary
> across W22..W48 = **301 named failure modes**). Headline
> R-95 results (3 seeds × 14 cell families): shared-state CID
> stable across turns (1.000); pseudo-KV recall reuse
> `proxy_recall_cosine = 0.750` (W47 = 0.0); multi-head
> diversity > 0 (W47 single-head = 0); reconstruction L1 < 3×
> baseline (1.000); branch/cycle bias separates two branches
> with identical features at 100% accuracy; write-gate
> selectivity 0.521 (W47 = 0); LATENT_CTRL round-trip 1.000;
> branch-history saves 67% of textual tokens; replay
> determinism 1.000; verifier detects 7 disjoint forgeries per
> seed; SharedStateAwareSyntheticBackend task_correct_rate
> 1.000 vs W47 0.0. Honest scope: W48 does NOT close the W43
> substrate-blocked conjectures (`W43-C-MIXED-CURVATURE-
> LATENT`, `W43-C-COLLECTIVE-KV-POOLING`,
> `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) or the carry-forward
> `W47-C-DEEP-TRANSFORMER-COUPLING`. New limitations:
> `W48-L-NO-REAL-KV-CAP` (the pseudo-KV bank reproduces the
> algebraic interface, NOT real KV bytes),
> `W48-L-PROXY-DISTRIBUTION-CAP` (strengthens
> `W47-L-AUTOGRAD-DISTRIBUTION-CAP`; mean
> downstream_protect_rate = 0.222 on R-95),
> `W48-L-PURE-PYTHON-TRAINING-COST-CAP` (carries forward),
> `W48-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP` (carries forward).
> New conjectures: `W48-C-REAL-KV-COUPLED-PROXY` (coupling the
> pseudo-KV bank to a real LLM's KV cache through backend
> hooks) and `W48-C-MULTI-HOST-SHARED-STATE` (sharing the base
> state + bank across hosts). See
> `docs/RESULTS_COORDPY_W48_SHARED_STATE_PROXY.md` for the full
> result note and
> `docs/SUCCESS_CRITERION_W48_SHARED_STATE_PROXY.md` for the
> pre-committed bar.
>
> **Earlier post-W46 research-line update (W47 Autograd Manifold Stack,
> 2026-05-10).** After W46 made the manifold-conditioned controller
> *deeper and memory-conditioned* via stage-wise closed-form ridge,
> the programme closed the long-standing
> `W46-C-AUTOGRAD-DEEP-STACK` carry-forward under the explicit
> "pure-Python reverse-mode AD + Adam SGD" reading. W47 is held
> outside the stable SDK contract — it lives at
> `coordpy.autograd_manifold` and is reachable only via explicit
> import; the released v0.5.20 wheel's public surface is
> byte-for-byte unchanged. The paper's main thesis is materially
> **strengthened** on a new axis by W47: the controller becomes
> *end-to-end-trainable by autograd* — a pure-Python reverse-mode
> `Variable` engine with topologically-sorted backward, an
> Adam-style optimiser with first/second-moment EMAs and per-tensor
> L2 gradient clipping, a trainable multi-layer tanh manifold
> stack, a trainable rank-r LoRA-style role adapter, a trainable
> K-prototype dictionary with soft-assignment cross-entropy + L2
> reconstruction, a trainable QKV memory head over the W46 bank
> that beats the W46 cosine pool on engineered OOD memory regimes,
> a trainable packed-control serializer whose 4-bit emit mask is
> learned per field, and a content-addressed
> `TrainingTraceWitness` sealing seed + optimiser config + loss
> history + grad-norm history + final params CID + training-set
> CID + divergence flag. The W47 envelope binds the W46 envelope
> CID, the autograd-params CID, the training-trace CID, the
> autograd-forward witness CID, the per-component CIDs (stack /
> role adapter / dictionary / memory head / control serializer),
> the memory-bank head CID, the causal-mask witness CID, the
> control-token witness CID, the prefix-capsule CID, and the
> prompt-construction witness CID under one `autograd_outer_cid`,
> verified through 21 enumerated W47 failure modes disjoint from
> the W22..W46 boundary (cumulative trust boundary across
> W22..W47 = **279 named failure modes**). Headline R-94 results
> (3 seeds × 12 cell families): autograd gradient check passes
> at max FD err < 1e-9 across 6 op classes; convergence on linear
> data reaches val_acc 1.000 within 200 SGD steps; deep stack
> params strictly move on XOR-shaped data; trained QKV head beats
> the W46 cosine pool on a one-hot key-value memory task
> (trained = 0.5 vs cosine = 0.0); trainable role adapter rank-2
> reaches ≥ 0.7 trained accuracy; trainable packed control
> serializer's 4 sigmoid gates converge to any target boolean
> mask in 150 SGD steps; replay determinism is bit-perfect across
> independent fits + runs; verifier soundness detects 6 disjoint
> forged envelopes; CTRL-aware backend task-correct rate lifts
> from 0.0 (off / baseline) to 1.0 (full ctrl). Honest scope: W47
> does NOT close the substrate-blocked W43 / W45 / W46
> transformer-coupling conjectures; new limitations are
> `W47-L-PURE-PYTHON-TRAINING-COST-CAP` (the engine is correct
> but slow), `W47-L-AUTOGRAD-DISTRIBUTION-CAP` (the trained
> controller learns whatever distribution it is shown — R-94
> compromise cap mean = 0.25, max = 0.5), `W47-L-NO-HIDDEN-STATE-CAP`
> (carries forward from W46), and `W47-L-CTRL-AWARE-MODEL-
> INDIFFERENCE-CAP` (strengthens W46). New conjectures:
> `W47-C-LIVE-MULTI-HOST-AUTOGRAD` (cross-host trained
> controllers) and `W47-C-GPU-BACKED-AUTOGRAD-SDK` (NumPy/JAX/
> PyTorch binding). See
> `docs/RESULTS_COORDPY_W47_AUTOGRAD_MANIFOLD.md` for the full
> result note and
> `docs/SUCCESS_CRITERION_W47_AUTOGRAD_MANIFOLD.md` for the
> pre-committed bar.
>
> **Earlier post-W45 research-line update (W46 Manifold Memory Controller,
> 2026-05-10).** After W45 made the manifold-conditioned routing /
> gating policy *shaped by data* via a single-layer fitted
> controller, the programme opened a new research line on *deeper,
> memory-conditioned, transformer-facing* approximation. W46 is
> held outside the stable SDK contract — it lives at
> `coordpy.manifold_memory` and is reachable only via explicit
> import; the released v0.5.20 wheel's public surface is
> byte-for-byte unchanged. The paper's main thesis (capsules as the
> coordination primitive; bounded-context transfer instead of
> transcript replay; replayable / auditable runs; "token cramming"
> as the practical released frontier) is materially
> **strengthened** on a new axis by W46: the controller becomes
> genuinely deeper — *multi-layer* (closed-form-stacked, fitted
> stage-wise on layer-wise residuals), *memory-conditioned*
> (bounded ring buffer of past channel features + gate logits with
> causally-masked time-attention), *multi-rank* (rank-r LoRA-style
> role adapters), *dictionary-coded* (sparse code over a learned
> K-prototype dictionary with bijective decode),
> *packed-control-emitting* (multi-line ``MANIFOLD_CTRL`` model-
> facing block carrying ``route + conf + p + layer_logits +
> mem_attn + dict_idx + mem_summary``), and *shared-prefix-aware*
> (deterministic prefix-capsule whose bytes are byte-identical
> across consecutive turns once the team has produced
> ``prefix_turns`` outputs). The W46 envelope binds the underlying
> TEAM_HANDOFF capsule CID, the W45/W44/W43 envelope CIDs, the
> multi-layer-controller parameter CID, the dictionary CID, the
> time-attention witness CID, the multi-rank-adapter witness CID,
> the control-token witness CID, the prefix-capsule CID, the
> memory-bank head CID, the causal-mask witness CID, and the
> prompt-construction witness CID under one ``memory_outer_cid``,
> verified through 21 enumerated W46 failure modes disjoint from
> the W22..W45 boundary (cumulative trust boundary across
> W22..W46 = **261 named failure modes**). Headline R-93 results
> (5 seeds × 12 cell families): long-branching-memory deep-turns
> precision **+1.000** strict gain over W45 (min == max);
> cyclic-consensus preservation **1.000** (W46 matches the W45
> ceiling under the multi-layer + memory + control + prefix
> path); role-shift adaptation rank-2 **+1.000** strict gain over
> shared-only; control-token round-trip **1.000** with ≤ 40-token
> overhead bound (W45 hint = 0.000); memory-facing task-correct
> rate **+1.000** under the deterministic
> ``MemoryAwareSyntheticBackend``; causal-mask preservation
> **1.000** (future-inject delta = 0.0); dictionary reconstruction
> **1.000** (avg L1 = 0.0, closest-prototype-rate = 1.0);
> shared-prefix reuse **1.000** (1 reuse per 4-turn run);
> replay-determinism **1.000**; the falsifier passes (no false
> abstentions); the new ``W46-L-MEMORY-COMPROMISE-CAP`` limitation
> reproduces honestly under all-six-channel + forged-bank attack.
> Honest scope: W46 does NOT close any of the W43 conjectures
> (``W43-C-MIXED-CURVATURE-LATENT``,
> ``W43-C-COLLECTIVE-KV-POOLING``,
> ``W43-C-FULL-GRASSMANNIAN-HOMOTOPY``); it *further bounds* (not
> closes) ``W44-C-LIVE-LATENT`` at the capsule layer;
> ``W45-C-DEEP-TRANSFORMER-COUPLING`` carries forward; a new
> ``W46-C-AUTOGRAD-DEEP-STACK`` conjecture is *deliberately
> deferred* (stage-wise closed-form is preserved for
> determinism and audit; an autograd / backprop path is a
> future milestone). See
> ``docs/RESULTS_COORDPY_W46_MANIFOLD_MEMORY.md`` for the full
> result note and
> ``docs/SUCCESS_CRITERION_W46_MANIFOLD_MEMORY.md`` for the
> pre-committed bar.
>
> **Earlier post-W44 research-line update (W45 Learned Manifold Controller,
> 2026-05-10).** After W44 made the W43 channels live-coupled
> through hand-designed thresholds, the programme opened a new
> research line on *learning* the manifold-conditioned routing /
> gating policy from cell observations. W45 is held outside the
> stable SDK contract — it lives at `coordpy.learned_manifold` and
> is reachable only via explicit import; the released v0.5.20
> wheel's public surface is byte-for-byte unchanged. The paper's
> main thesis (capsules as the coordination primitive;
> bounded-context transfer instead of transcript replay; replayable
> / auditable runs; "token cramming" as the practical released
> frontier) is materially **strengthened** on a new axis by W45:
> the gating decisions themselves become *shaped by data* via five
> learned components, all closed-form-fittable in pure NumPy-free
> Python — a learned channel encoder, attention-style routing over
> the six W43 channels, a LoRA-style role-specific adapter,
> margin-calibrated gating, and a content-addressed
> ``MANIFOLD_HINT: route=<int> conf=<bucket> p=<prob>`` prompt
> control. The hyperbolic and euclidean channels — audit-only at
> the W44 layer — are now consumed at the W45 layer as input
> features, *bounding* (not closing) the open ``W44-C-LIVE-LATENT``
> carry-forward. The W45 envelope binds the underlying
> TEAM_HANDOFF capsule CID, the W44 envelope CID, the W43 envelope
> CID, the controller parameter CID, the attention-routing witness
> CID, the role-adapter witness CID, the causal-mask witness CID,
> the prompt-construction witness CID, and the hint witness CID
> under one ``learned_outer_cid``, verified through 14+ enumerated
> W45 failure modes disjoint from the W22..W44 boundary (cumulative
> trust boundary across W22..W45 = **240 named failure modes**).
> Headline R-92 results (5 seeds × 9 cell families):
> calibration-gain precision **+0.400** strict gain over W44 (min
> == max); attention-specialization **1.000** (W44 / W43 / baseline
> = 0.000 — the first benchmark in the programme where attention
> weights are measurably specialised per signature); role3-
> precision **+0.500** vs shared-only adapter; factoradic + hint
> round-trip **1.000** across 5/5 seeds; model-facing task-correct
> rate **+1.000** under the deterministic
> ``HintAwareSyntheticBackend``; replay-determinism **1.000**;
> falsifier passes (no false abstentions); the new
> ``W45-L-LEARNED-COMPROMISE-CAP`` limitation reproduces honestly.
> Honest scope: W45 does NOT close any of the W43 conjectures
> (``W43-C-MIXED-CURVATURE-LATENT``,
> ``W43-C-COLLECTIVE-KV-POOLING``,
> ``W43-C-FULL-GRASSMANNIAN-HOMOTOPY``); it *bounds* (not closes)
> ``W44-C-LIVE-LATENT`` at the capsule layer; a new
> ``W45-C-DEEP-TRANSFORMER-COUPLING`` conjecture covers the
> hidden-state / KV-cache-consuming direction (still substrate-
> blocked). See ``docs/RESULTS_COORDPY_W45_LEARNED_MANIFOLD.md``
> for the full result note and
> ``docs/SUCCESS_CRITERION_W45_LEARNED_MANIFOLD.md`` for the
> pre-committed bar.
>
> **Earlier post-W43 research-line update (W44 Live Manifold-Coupled
> Coordination, 2026-05-10).** After W43 made the product-manifold
> capsule executable as a closed-form audit-only layer, the
> programme opened a new research line on *live behavioural
> coupling* of the manifold channels to actual agent run behaviour.
> W44 is held outside the stable SDK contract — it lives at
> `coordpy.live_manifold` and is reachable only via explicit
> import; the released v0.5.20 wheel's public surface is byte-
> for-byte unchanged. The paper's main thesis (capsules as the
> coordination primitive; bounded-context transfer instead of
> transcript replay; replayable / auditable runs; "token cramming"
> as the practical released frontier) is materially
> **strengthened** on a new axis by W44: three of the W43 channels
> (spherical / subspace / causal) become *active live gates* that
> substitute a deterministic abstain output for the agent's
> ``generate()`` call when the registered policy is violated;
> downstream prompts therefore never see the bad upstream handoff.
> A fourth channel (factoradic route) becomes a *live compressor*
> that replaces the textual rendering of the role-arrival ordering
> with a single integer header in the prompt. Two channels
> (hyperbolic, euclidean) remain audit-only at the live layer —
> W44 force-verdicts each channel honestly. The W44 envelope binds
> the underlying TEAM_HANDOFF capsule CID, the W43 envelope CID,
> the prompt-construction witness CID, and the live witness CID
> under one ``live_outer_cid``, verified through 12 enumerated
> failure modes disjoint from the W22..W43 boundary (cumulative
> trust boundary across W22..W44 = **226 named failure modes**).
> Headline R-91 results (5 seeds × 7 cell families):
> downstream-protect rate **+0.400** strict gain on each of the
> three gate families (causal, spherical, subspace; min == max);
> **314 visible prompt tokens saved per run** on the factoradic-
> compression family at n_roles=8; falsifier passes (no false
> abstentions); the new ``W44-L-LIVE-DUAL-CHANNEL-COLLUSION-CAP``
> limitation reproduces honestly. Honest scope: W44 does NOT
> close any of the W43 conjectures
> (``W43-C-MIXED-CURVATURE-LATENT``,
> ``W43-C-COLLECTIVE-KV-POOLING``,
> ``W43-C-FULL-GRASSMANNIAN-HOMOTOPY``); a new
> ``W44-C-LIVE-LATENT`` conjecture covers the audit-only
> hyperbolic and euclidean channels. See
> ``docs/RESULTS_COORDPY_W44_LIVE_MANIFOLD.md`` for the full
> result note and
> ``docs/SUCCESS_CRITERION_W44_LIVE_MANIFOLD.md`` for the
> pre-committed bar.
>
> **Earlier post-release research-line update (W43 Product-Manifold
> Capsule, 2026-05-10).** After the v3.43 final release shipped,
> the programme opened a research line on a *mixed-curvature
> product manifold* decomposition of the capsule-layer state. W43
> is held outside the stable SDK contract — it lives at
> `coordpy.product_manifold` and is reachable only via explicit
> import; the released v0.5.20 wheel's public surface is byte-
> for-byte unchanged. The paper's main thesis (capsules as the
> coordination primitive; bounded-context transfer instead of
> transcript replay; replayable / auditable runs; "token cramming"
> as the practical released frontier) is materially **strengthened**
> on six axes by W43: a hyperbolic branch channel (Poincare-disk-
> style; bit-perfect round-trip up to ``2 * dim`` path bits), a
> spherical consensus channel (unit-norm L2 signature over
> claim_kinds), a euclidean attribute channel, a factoradic route
> channel (bijective Lehmer code; ``ceil(log2(n!))`` bits per cell
> at zero visible-token cost), a subspace state channel (bounded-
> rank QR-canonicalised basis; one point on Gr(k, d) per cell), and
> a Lamport vector-clock channel with explicit dependency-closure
> admissibility check. The W43 envelope binds seven content-
> addressed component CIDs under one ``manifest_v13_cid`` and is
> verified through 18 enumerated failure modes disjoint from the
> W22..W42 boundary (cumulative trust boundary across W22..W43 =
> 214 named failure modes). Headline R-90 results (5 seeds × 8
> cell families): trust precision **+0.400** on consensus-cycle
> and subspace-drift cells, causal-rejection rate **+1.000** on
> out-of-order handoffs, structured-bit gain of **+272 bits per
> visible-token overhead** at n=8 roles. Honest scope: W43 does
> NOT close ``W43-C-MIXED-CURVATURE-LATENT`` (transformer-internal
> mixed-curvature attention), ``W43-C-COLLECTIVE-KV-POOLING``
> (host-collective KV-cache sharing), or
> ``W43-C-FULL-GRASSMANNIAN-HOMOTOPY`` (continuous Gr(k, d)
> homotopy) — all three require new architectural substrate
> beyond the capsule layer. See
> ``docs/RESULTS_COORDPY_W43_PRODUCT_MANIFOLD.md`` for the full
> result note and
> ``docs/SUCCESS_CRITERION_W43_PRODUCT_MANIFOLD.md`` for the
> pre-committed bar.
>
> Updated through **SDK v3.43 (final release of the v3.4x line)**,
> 2026-05-03 (W22 → W42 cumulative trust + dense-control +
> live-aware multi-anchor / trust-subspace / host-diverse /
> cross-host trajectory / disjoint-consensus-reference /
> multi-host-disjoint-quorum / cross-host-response-heterogeneity /
> integrated-multi-agent-context-synthesis / cross-role-invariant-
> synthesis guard ladder summarised in § 14.2 and § 17).
>
> **Latest milestone marker (SDK v3.43 / W42, 2026-05-03 — FINAL
> RELEASE of the CoordPy SDK v3.4x line).**  The programme now has
> **thirty-nine** coupled research axes.  W42 is explicitly framed
> as a **third-axis bounding** milestone *and* a **release-closure**
> milestone -- not "W42: one more local mechanism."  W42 wraps W41's
> producer-axis × trust-axis integrated synthesis with a third
> orthogonal evidence axis: the **role-handoff invariance axis**,
> computed deterministically from the canonical sorted ``(role,
> kind, payload)`` tuples in the cell's input handoffs and
> namespaced as ``w42_role_handoff_signature``.  An honest
> controller pre-registers a ``RoleInvariancePolicyRegistry``
> mapping signature CIDs to expected service sets; W42 ratifies on
> agreement, abstains on disagreement, and falls through on unknown
> signatures.  W42 is closed-form, zero-parameter, and capsule-
> layer; it does NOT add a transformer-internal mechanism, does NOT
> close ``W41-L-COMPOSITE-COLLUSION-CAP`` in general, and does NOT
> close its own new ``W42-L-FULL-COMPOSITE-COLLUSION-CAP``
> limitation theorem (the W42 analog of all previous capsule-layer
> limitation theorems, sharper in adversary cost).  The role-
> invariance decision is a deterministic 7-branch classifier
> ``select_role_invariance_decision`` over ``(integrated_services,
> expected_services, policy_match_found)``.  Manifest-v12 binds six
> component CIDs (``parent_w41_cid``, ``invariance_state_cid``,
> ``invariance_decision_cid``, ``invariance_audit_cid``,
> ``invariance_witness_cid``, ``role_handoff_signature_cid``); the
> audit / witness / signature CIDs are namespaced
> (``w42_invariance_audit`` / ``w42_invariance_witness`` /
> ``w42_role_handoff_signature``) so substituting a W22..W41 audit /
> witness / signature for them is mechanically rejected.
> ``verify_role_invariant_synthesis_ratification`` enumerates 14
> disjoint W42 failure modes (cumulative 196 across W22..W42).
> R-89 is the first multi-agent context benchmark family in the
> programme that records the cross-role invariance branch
> distribution per cell across five banks (``trivial_w42`` /
> ``role_invariant_agrees`` / ``role_invariant_recover`` /
> ``full_composite_collusion`` / ``insufficient_invariance_policy``).
> The load-bearing change is the **first measured strict
> trust-precision recovery on a regime where W41 tied at 0.500**:
> on R-89-ROLE-INVARIANT-RECOVER, W42 raises trust precision from
> 0.500 to **1.000 across 5/5 seeds**
> (``Δ_trust_precision_w42_w41 = +0.500``, min = max).  W42 carries
> roughly **17.5k structured bits per visible W42 token** at one
> visible token overhead/cell (in the W38..W40 density range).
> **Live cross-host paraphrase-invariance probe (2026-05-03).**  At
> temperature 0 on the two-Mac topology (``localhost`` gemma2:9b +
> ``192.168.12.191`` qwen2.5:14b), K=4 paraphrases of one
> closed-vocabulary arithmetic prompt produce **4/4 paraphrase-
> invariant cross-host gold-correlated agreement**: both hosts emit
> "Four" on every paraphrase; cross-host normalised agreement =
> 1.000; within-host paraphrase-invariance count = 1 distinct answer
> per host.  This is the first measured cross-host paraphrase-
> invariance result in the programme; it is a realism anchor only,
> not load-bearing for the W42 closed-form mechanism.
> **Final release declared**: H1..H12 + S3 + S7 of the W42 success
> criterion pass.  The SDK v3.43 line ships as the **final release**
> of the CoordPy SDK v3.4x research line — the **end-of-line for the
> capsule-layer-only research programme** in the Context Zero
> project.  The remaining open frontiers (``W42-C-NATIVE-LATENT``
> for transformer-internal trust-state projection;
> ``W42-C-MULTI-HOST`` for K+1-host disjoint topology) are
> explicitly out of capsule-layer scope and require new
> architectural substrate (transformer-internal access, K+1-host
> topology, or both).  Stable-vs-experimental boundary final: every
> W22..W42 symbol is exported under ``__experimental__``; the
> stable ``RunSpec → run report`` runtime contract is byte-for-byte
> unchanged.  Versioning: ``vision_mvp.__version__`` and
> ``pyproject.toml`` ``project.version`` are now both ``0.5.16``
> (alignment maintained); ``SDK_VERSION = coordpy.sdk.v3.43``.  See
> ``docs/RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md`` and
> ``docs/SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md``.

> **Earlier milestone marker (SDK v3.42 RC2 / W41, 2026-05-03 —
> superseded by v3.43 final).** The programme had **thirty-eight**
> coupled research axes at this milestone.  W41 is explicitly
> framed as a **synthesis** milestone, not "W41: one more local
> mechanism."  W41 jointly binds the strongest old-line
> explicit-capsule trust-adjudication chain (W21..W40) AND the
> strongest cross-role / multi-round bundle decoder family (W7..W11)
> into a single auditable end-to-end path with one **manifest-v11**
> envelope binding both axes plus a content-addressed cross-axis
> witness.  W41 is closed-form, zero-parameter, and capsule-layer; it
> does NOT add a transformer-internal mechanism, does NOT close
> ``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP``, and does NOT close its
> own new ``W41-L-COMPOSITE-COLLUSION-CAP`` limitation theorem (the
> W41 analog of all previous capsule-layer limitation theorems).
> The cross-axis classification is a deterministic 8-branch decision
> selector ``select_integrated_synthesis_decision`` over the per-axis
> branches and per-axis service tuples.  Manifest-v11 binds five
> component CIDs (``parent_w40_cid``, ``synthesis_state_cid``,
> ``synthesis_decision_cid``, ``synthesis_audit_cid``,
> ``cross_axis_witness_cid``); the audit + witness CIDs are
> namespaced (``w41_synthesis_audit`` / ``w41_cross_axis_witness``)
> so substituting a W22..W40 audit / witness for them is mechanically
> rejected.  ``verify_integrated_synthesis_ratification`` enumerates
> 14 disjoint W41 failure modes (cumulative 182 across W22..W41).
> R-88 is the first end-to-end multi-agent context benchmark family
> in the programme that records the cross-axis branch distribution
> per cell, letting researchers distinguish which axis (producer vs
> trust vs both vs neither) is load-bearing.  Across 5/5 seeds × 16
> cells/seed, R-88 results are: ``trivial_w41`` (W41 = W40 byte-for-
> byte; 80 trivial_integrated_passthrough); ``both_axes``
> (correctness 1.000; trust precision 1.000; delta = 0; 40
> producer_only + 40 both_axes); ``trust_only_safety`` (trust
> precision 1.000 preserved; safety branch INTEGRATED_TRUST_ONLY
> clears services on collapse; 40 producer_only + 40 trust_only);
> ``composite_collusion`` (W41-L-COMPOSITE-COLLUSION-CAP fires;
> trust precision 0.500; delta = 0; 40 producer_only + 40 both_axes
> on wrong set); ``insufficient_response_signature`` (W41 = W40
> byte-for-W39; 80 producer_only).  W41 carries roughly **15.5k
> structured bits per visible W41 token** at one visible token
> overhead/cell (in the W38..W40 density range).  **Lab topology
> retraction -- W41-INFRA-1**: re-probing ``192.168.12.101`` at the
> W41 milestone identifies it as an Apple TV / AirPlay receiver
> (``HTTP/1.1 403 Forbidden`` with header ``Server:
> AirTunes/860.7.1`` on port 5000; locally-administered MAC
> ``36:1c:eb:dc:9a:04``; no Mac mDNS hostname), NOT a Mac running
> Ollama.  The W37..W40 "TCP-up + HTTP-broken Ollama Mac" framing
> for ``.101`` is **retracted** at this milestone.  No Ollama
> instance has ever been running on ``.101`` in this network; port
> 11434 returning "Empty reply from server" was the device closing
> the connection on an unrecognised port, NOT a hung Ollama
> listener.  The honest live multi-host topology going forward is
> the two-Mac pair ``localhost`` (``Qunfengs-MBP.lan``,
> ``192.168.12.157``) + ``192.168.12.191``
> (``HSC136047-MAC.lan``).  ``192.168.12.248`` is recorded as gone
> (per user instruction).  A bounded W41 sanity probe at
> temperature 0 + ``num_predict = 4`` on the two-Mac topology
> produced byte-agreed answers across architectures: ``localhost
> gemma2:9b`` and ``192.168.12.191 qwen2.5:14b`` both answered
> "Four" to "What is 2+2? Answer with one word."  This sharpens
> the closed-vocabulary one-word-prompt
> ``W37-C-LIVE-TRUNCATION-RECOVERY`` /
> ``W40-C-LIVE-RESPONSE-HETEROGENEITY`` anchors at the synthetic
> layer; the W41 mechanism is closed-form and capsule-layer, so no
> live LLM inference is required to evaluate the W41 success
> criterion.  **RC2 declared**: H1..H12 + S3 of the W41 success
> criterion pass.  The SDK v3.42 line is the second release-
> candidate of the CoordPy SDK v3.4x line, strictly additive on top
> of RC1 (every W22..W41 symbol is exported under
> ``__experimental__``; the stable ``RunSpec → run report``
> runtime contract is byte-for-byte unchanged).  Versioning:
> ``vision_mvp.__version__`` and ``pyproject.toml``
> ``project.version`` are now both ``0.5.15`` (alignment
> maintained); ``SDK_VERSION = coordpy.sdk.v3.42``.  See
> ``docs/RESULTS_COORDPY_W41_INTEGRATED_SYNTHESIS.md`` and
> ``docs/SUCCESS_CRITERION_W41_INTEGRATED_SYNTHESIS.md``.

> **Earlier milestone marker (SDK v3.41 RC1 / W40, 2026-05-03).** The
> programme now has **thirty-seven** coupled research axes.  W40
> wraps W39's K-of-N mutually-disjoint quorum consensus-reference
> adjudication with a **cross-host response-signature heterogeneity**
> layer that operates on an evidence axis ORTHOGONAL to top_set: the
> per-member response **text bytes** themselves.  Even if K colluders
> coordinate their declared top_set (the W39 full-quorum-collusion
> attack; ``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` fires),
> naturally-independent K probes should produce heterogeneous
> response text bytes (different paraphrases, different surface
> forms, different tokenisations).  When the K member probes' mean
> pairwise Jaccard divergence over canonical sorted token bags falls
> strictly below ``response_text_diversity_min``, W40 abstains via
> ``RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED`` even when W39 would
> have RATIFIED.  The envelope binds a manifest-v10 CID over six
> components (``parent_w39_cid``,
> ``response_signature_state_cid``,
> ``response_signature_audit_cid``,
> ``response_signature_topology_cid``,
> ``response_signature_decision_cid``,
> ``response_heterogeneity_witness_cid``).  This *bounds* (does not
> close) the ``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` limitation
> theorem: it raises the capsule-layer adversary bar from
> "compromise 2 of N trajectory hosts AND ``quorum_min`` of the K
> mutually-disjoint registered consensus references" to "compromise
> 2 of N trajectory hosts AND ``quorum_min`` of the K mutually-
> disjoint registered consensus references AND inject K diverse
> response text bytes that all encode the same wrong top_set".
> On R-87-RESPONSE-SIGNATURE-COLLAPSE, W40 raises trust precision
> over W39 from 0.500 (``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP``
> fires against W39) to **1.000**
> (**Δ_trust_precision_w40_w39 = +0.500**, min and max equal across
> 5/5 seeds), abstains via ``RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED``
> on 8 cells/seed, and adds one visible-token overhead/cell while
> carrying about **14.5k structured bits per visible W40 token**.
> On the four named falsifiers (R-87-TRIVIAL-W40,
> R-87-NO-REGRESSION-DIVERSE-AGREES,
> R-87-COORDINATED-DIVERSE-RESPONSE,
> R-87-INSUFFICIENT-RESPONSE-SIGNATURE) W40 preserves W39 behavior
> and trust precision exactly.  W40 adds 14 mechanically tested
> verifier failure modes (including the W40-specific
> ``w40_response_mutual_disjointness_violation``), bringing the
> cumulative W22..W40 trust boundary to **168 enumerated failure
> modes**.  A new proved-conditional limitation theorem
> **W40-L-COORDINATED-DIVERSE-RESPONSE-CAP** is recorded: when the
> K registered W40 member probes' response signatures are *injected*
> to be diverse but all encode the same wrong top_set in lock-step
> (the "smart" attacker who diversifies response bytes while
> holding the wrong top_set), W40 cannot recover at the capsule
> layer (the W40 analog of W34-L-MULTI-ANCHOR-CAP,
> W37-L-MULTI-HOST-COLLUSION-CAP, W38-L-CONSENSUS-COLLUSION-CAP,
> and W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP); closure requires
> native-latent evidence outside the capsule layer or a K+1-host
> disjoint topology with a new uncompromised pool.  W40 is
> explicitly NOT native latent transfer, NOT transformer-internal
> hidden-state projection, and NOT a KV-cache transplant; it is an
> audited capsule-layer cross-host response-text Jaccard divergence
> proxy with two mechanically-enforced disjointness preconditions
> inherited from W39.  **Lab topology resolution -- W40-INFRA-1**:
> ``192.168.12.101`` re-probed (2026-05-03) -- ping 0% packet loss
> (a strict improvement over W39's 100% packet loss end-state),
> TCP port 11434 connects, TCP port 22 connects with auth methods
> advertised; however, the Ollama HTTP listener still returns
> "Empty reply from server" / "Connection reset by peer" (the
> W39-INFRA-1 hung-listener pattern persists).  SSH credentials
> still unavailable to restart.  Honest verdict: ``.101`` is now
> **TCP-up + HTTP-broken** at the Ollama layer; restoration
> requires SSH credentials.  ``192.168.12.248`` remains
> ARP-incomplete (32nd milestone in a row).  **RC1 declared**:
> H1..H12 + S3 of the W40 success criterion pass.  The SDK v3.41
> line is the **first official release-candidate** of the CoordPy
> SDK v3.4x line.  Stable-vs-experimental boundary final for RC1:
> every W22..W40 symbol is exported under ``__experimental__``;
> the stable ``RunSpec → run report`` runtime contract is byte-for-
> byte unchanged.  Versioning: ``vision_mvp.__version__`` and
> ``pyproject.toml`` ``project.version`` are now both ``0.5.14``
> (alignment maintained).  See
> ``docs/RESULTS_COORDPY_W40_RESPONSE_HETEROGENEITY.md`` and
> ``docs/SUCCESS_CRITERION_W40_RESPONSE_HETEROGENEITY.md``.

> **Previous milestone marker (SDK v3.40 / W39, 2026-05-02).** The
> programme had **thirty-six** coupled research axes.  W39 wraps
> W38's disjoint cross-source consensus-reference adjudication with
> a **K-of-N mutually-disjoint quorum** of disjoint probes, each
> sourced from a physically-distinct host pool that is both
> mechanically disjoint from the W37 trajectory hosts (W38's
> precondition) AND mutually disjoint from every other registered
> quorum probe's host pool (the new W39 precondition; the
> ``MultiHostDisjointQuorumRegistry`` raises
> :class:`MutuallyDisjointTopologyError` when any two pools have
> non-empty intersection; the verifier additionally rejects envelopes
> claiming an overlapping pool pair).  When at least ``quorum_min``
> of the K member probes diverge from the W37/W38 candidate top_set,
> W39 abstains via the ``QUORUM_DIVERGENCE_ABSTAINED`` branch.  The
> envelope binds a manifest-v9 CID over six components
> (parent_w38_cid, quorum_state_cid, quorum_audit_cid,
> quorum_topology_cid, quorum_decision_cid, mutual_disjointness_cid).
> On R-86-MULTI-HOST-COLLUDED-CONSENSUS, W39 raises trust precision
> over W38 from 0.500 (W38-L-CONSENSUS-COLLUSION-CAP fires) to
> **1.000** (**Δ_trust_precision_w39_w38 = +0.500**, min and max
> equal across 5/5 seeds), abstains via QUORUM_DIVERGENCE on 8
> cells/seed, and adds one visible-token overhead/cell while
> carrying about **24.4k structured bits per visible W39 token**
> (~2.7x denser than W38's 9.07k bits/token at the audited-proxy
> capsule layer).  On the four named falsifiers (R-86-TRIVIAL-W39,
> R-86-NO-REGRESSION-QUORUM-AGREES, R-86-FULL-QUORUM-COLLUSION,
> R-86-INSUFFICIENT-QUORUM) W39 preserves W38 behavior and trust
> precision exactly.  W39 adds 14 mechanically tested verifier
> failure modes (including the W39-specific
> ``w39_quorum_mutual_disjointness_violation``), bringing the
> cumulative W22..W39 trust boundary to **154 enumerated failure
> modes**.  A new proved-conditional limitation theorem
> **W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP** is recorded: when
> all K registered disjoint quorum probes are themselves compromised
> in lock-step with the colluding trajectory hosts, W39 cannot
> recover at the capsule layer (the W39 analog of
> W34-L-MULTI-ANCHOR-CAP, W37-L-MULTI-HOST-COLLUSION-CAP, and
> W38-L-CONSENSUS-COLLUSION-CAP); closure requires native-latent
> evidence outside the capsule layer or a K+1-host disjoint topology
> with a new uncompromised pool.  W39 is explicitly NOT native latent
> transfer, NOT transformer-internal hidden-state projection, and
> NOT a KV-cache transplant; it is an audited capsule-layer
> multi-host disjoint quorum proxy with two mechanically-enforced
> disjointness preconditions (trajectory disjointness inherited from
> W38 + mutual disjointness new in W39).  **Lab topology
> resolution**: the historical Mac-2 endpoint
> (``192.168.12.248``) has been ARP-incomplete for the 31st
> milestone in a row; ``192.168.12.101`` was identified as the
> reachable third physical host candidate, **partially discharging
> W38-C-MULTI-HOST at the topology layer** (preflight-OK on cold
> contact with ``qwen3.5:35b`` + ``qwen2.5:14b-32k`` model files
> visible).  The ``.101`` Ollama inference path subsequently
> degraded under the one-word probe budget (``W39-INFRA-1``).  The
> W39 live xllm probe was made robust via a fallback path: when
> ``.101`` is unreachable, ``mac_off_cluster_a`` swaps to
> ``localhost`` running ``llama3.1:8b`` (a model class genuinely
> different from the trajectory's ``gemma2:9b``), so the live K=2
> quorum becomes ``(localhost llama3.1:8b, .191
> qwen2.5-coder:14b-32k)`` -- two physically distinct hosts, each
> running a different model class from the trajectory pair AND from
> the W38 single consensus reference.  Bounded W39 5-host live
> xllm probe at temperature 0 + ``num_predict=4`` produced **8/8
> responsive on all 5 hosts** (first 5-host live W39 disjoint-quorum
> probe in the programme), 7/8 trajectory-pair agreements, 7/8 W38
> single consensus agreements, **8/8 quorum_a gold-correlated, 8/8
> quorum_b gold-correlated, 8/8 K=2 quorum size simultaneously
> responsive**.  Notable live finding: on the ``h2o`` probe, the
> trajectory pair disagreed (``mac1=h2o`` vs ``mac_remote=h`` due to
> ``num_predict=4`` truncation), but BOTH quorum members got
> ``h2o`` correct -- empirical-suggestive evidence for the new
> ``W39-C-LIVE-TRUNCATION-RECOVERY`` conjecture (the W39 multi-host
> disjoint quorum can recover from trajectory-pair-only truncation
> errors at the live layer when the quorum members use a different
> generation budget; this is a recovery axis distinct from the
> collusion bound; sharper validation requires a dedicated live
> truncation-recovery bench).  Versioning: ``vision_mvp.__version__``
> and ``pyproject.toml`` ``project.version`` are now both ``0.5.13``
> (alignment maintained).  See
> ``docs/RESULTS_COORDPY_W39_MULTI_HOST_DISJOINT_QUORUM.md`` and
> ``docs/SUCCESS_CRITERION_W39_MULTI_HOST_DISJOINT_QUORUM.md``.

> **Previous milestone marker (SDK v3.39 / W38, 2026-05-02).** The
> programme has **thirty-five** coupled research axes.  W38 wraps
> W37's anchor-cross-host basis-trajectory ratification with a
> controller-pre-registered ``ConsensusReferenceProbe`` whose host
> topology is *mechanically disjoint* from W37's trajectory hosts (the
> ``DisjointConsensusReferenceRegistry`` raises
> ``DisjointTopologyError`` otherwise; the verifier additionally
> rejects envelopes claiming an overlapping topology).  When W37
> chooses to reroute on a trajectory-anchored top_set and the disjoint
> consensus reference disagrees within ``divergence_margin_min``
> (Jaccard), W38 abstains via ``CONSENSUS_DIVERGENCE_ABSTAINED``.  The
> envelope binds a manifest-v8 CID over five components
> (parent_w37_cid, consensus_reference_state_cid,
> divergence_audit_cid, consensus_topology_cid, consensus_probe_cid).
> On R-85-COLLUDED-CROSS-HOST-TRAJECTORY, W38 raises trust precision
> over W37 from 0.500 to **1.000**
> (**Δ_trust_precision_w38_w37 = +0.500**, min and max equal across
> 5/5 seeds), abstains via DIVERGENCE on 8 cells/seed, and adds one
> visible-token overhead/cell while carrying about **9.07k structured
> bits per visible W38 token**.  On the four named falsifiers
> (R-85-TRIVIAL-W38, R-85-NO-COLLUSION-CONSENSUS-AGREES,
> R-85-CONSENSUS-ALSO-COMPROMISED, R-85-NO-CONSENSUS-REFERENCE) W38
> preserves W37 behavior and trust precision exactly.  W38 adds 14
> mechanically tested verifier failure modes, bringing the cumulative
> W22..W38 trust boundary to **140 enumerated failure modes**.  A new
> proved-conditional limitation theorem
> **W38-L-CONSENSUS-COLLUSION-CAP** is recorded: when the disjoint
> consensus reference is itself compromised in lock-step with the
> colluding trajectory hosts, W38 cannot recover at the capsule layer
> (the W38 analog of W34-L-MULTI-ANCHOR-CAP and
> W37-L-MULTI-HOST-COLLUSION-CAP); closure requires native-latent
> evidence outside the capsule layer or a 3+-host disjoint topology.
> W38 is explicitly NOT native latent transfer, NOT transformer-
> internal hidden-state projection, and NOT a KV-cache transplant; it
> is an audited capsule-layer cross-source consensus-reference proxy
> with mechanical disjoint-topology enforcement.  Mac 2 still times
> out (31st milestone in a row); bounded W38 3-host live consensus
> probe across local ``gemma2:9b``, remote ``qwen2.5:14b`` (trajectory
> pair), and remote ``qwen2.5-coder:14b`` (disjoint consensus host on
> the same physical host as a defensible weak proxy for capsule-layer
> disjointness) produced **8/8 responsive on all 3 hosts, 7/8
> trajectory-pair agreements (the one disagreement is a
> ``num_predict=4`` truncation), 7/8 cross-source consensus
> agreements, 8/8 consensus-gold correlation** at temperature 0.
> Versioning reconciliation: ``vision_mvp.__version__`` and
> ``pyproject.toml`` ``project.version`` are now both ``0.5.12`` (the
> lingering 0.5.9 vs 0.5.11 misalignment from earlier milestones is
> closed).  See
> ``docs/RESULTS_COORDPY_W38_DISJOINT_CONSENSUS_REFERENCE.md`` and
> ``docs/SUCCESS_CRITERION_W38_DISJOINT_CONSENSUS_REFERENCE.md``.

> **Previous milestone marker (SDK v3.38 / W37, 2026-05-02).** The
> programme has **thirty-four** coupled research axes.  W37 wraps
> W36's host-diverse trust-subspace guard with a closed-form,
> zero-parameter, per-(host, oracle, top_set) EWMA over *anchored*
> historical observations: a single-host current cell can be safely
> rerouted iff its (host, oracle, top_set) entry has been cross-host
> anchored above threshold across at least
> ``min_anchored_observations`` historical cells with at least
> ``min_trajectory_anchored_hosts`` distinct anchor hosts.  W37 is
> explicitly cross-cell where W36 was per-cell; its envelope binds a
> manifest-v7 CID over (parent_w36_cid, cross_host_trajectory_state_cid,
> trajectory_audit_cid, trajectory_topology_cid).  On
> R-84-SINGLE-HOST-TRAJECTORY-RECOVER, W37 improves over W36 from
> 0.500 to **1.000** correctness (**Δ_correctness_w37_w36 = +0.500**,
> min and max equal across 5/5 seeds) at trust precision **1.000**
> with one visible-token overhead/cell.  On the four named falsifiers
> (R-84-TRIVIAL-W37, R-84-NO-TRAJECTORY-HISTORY,
> R-84-POISONED-TRAJECTORY, R-84-TRAJECTORY-DISAGREEMENT) W37
> preserves W36 behavior and trust precision exactly.  The W37
> envelope carries about **29.5k structured bits per visible W37
> token** on the load-bearing regime and adds 14 mechanically tested
> verifier failure modes, bringing the cumulative W22..W37 trust
> boundary to **126 enumerated failure modes**.  A new
> proved-conditional limitation theorem
> **W37-L-MULTI-HOST-COLLUSION-CAP** is recorded: two registered
> hosts emitting a coordinated wrong top_set across enough cells can
> cross the anchored thresholds and cannot be separated at the capsule
> layer.  W37 is explicitly NOT native latent transfer, NOT
> transformer-internal hidden-state projection, and NOT a KV-cache
> transplant.  Mac 2 still times out (30th milestone in a row);
> bounded live cross-host trajectory probe across local `gemma2:9b`
> and remote `qwen2.5:14b` produced **8/8 responsive probes, 8/8
> cross-host anchored agreements, and 8/8 gold-correlated agreements**
> at temperature 0.  See
> ``docs/RESULTS_COORDPY_W37_CROSS_HOST_BASIS_TRAJECTORY.md`` and
> ``docs/SUCCESS_CRITERION_W37_CROSS_HOST_BASIS_TRAJECTORY.md``.

> **Previous milestone marker (SDK v3.37 / W36, 2026-05-02).** W36
> hardens the W35 trust-subspace proxy at the host/live boundary by
> requiring dense-control support to be independently attested by at
> least two distinct registered healthy hosts.  On R-83-HOST-DIVERSE-
> RECOVER, W36 improves over W35 from 0.625 to **0.9375** correctness
> (**+0.3125**) across 5/5 seeds and restores trust precision from
> 0.6667 to **1.000** with one visible-token overhead/cell.
>
> **Previous milestone marker (SDK v3.36 / W35, 2026-05-02).** W35 is
> the first milestone where the old explicit capsule line and the newer
> dense-control / geometry-aware line become one mechanism rather than
> parallel evidence.  W35 wraps W34's live-aware multi-anchor
> abstention path with a controller-verified trust-subspace dense
> proxy over W21 probe top_sets, W33 EWMA trust, W34 live-attestation
> / response-feature state, top-set stability, and host health.  On
> R-82-TRUST-SUBSPACE-SHIFT, W34 abstains on 6 disputed cells; W35
> safely reroutes 5/6 through the stable basis direction, raising
> correctness from 0.625 to **0.9375** (**+0.3125**) across 5/5
> seeds, with trust precision preserved at **1.000**.
>
> **Previous milestone marker (SDK v3.35 / W34, 2026-05-01).** W34 closes
> the W33 *single-anchor fragility* — the W33 trust-EWMA mechanism
> uses an anchor oracle reference to derive its per-oracle agreement
> signal; if the anchor itself becomes compromised, every honest
> non-anchor oracle's agreement signal drops against the wrong
> reference and the (compromised) anchor remains trusted.  W34
> wraps W33 with a *multi-anchor consensus reference* (the
> intersection of K registered anchors when at least
> ``anchor_quorum_min`` agree); when the intersection is empty
> (anchors disagree), W34 abstains — anchor disagreement is itself
> a trust signal.  W34 also adds an audited proxy step toward
> native-latent (a closed-form 64-bit response-feature signature
> over first_token_class + length_bucket + structural_hash; NOT a
> transformer-internal hidden-state projection), a closed-form
> host-aware EWMA decay, a content-addressed
> ``LiveOracleAttestation`` per probe, and a manifest-v4 CID over
> four component CIDs.  Measured **+0.375 trust-precision strict
> gain over W33 single-anchor across 5/5 seeds at min trust
> precision = 1.000** on R-81-DOUBLE-ANCHOR-COMPROMISE.  The
> milestone also closes two named W33 infrastructure follow-ups:
> **W33-INFRA-1** (preflight ``/api/tags`` discipline; an honest
> empirical correction — qwen3.5:35b on 192.168.12.191 IS in fact
> loaded; the W33 diagnosis was wrong, the real W33 infra failure
> was timeout + chat-template) and **W33-INFRA-2** (chat-template
> + ``num_predict=4`` + stop tokens for one-word probes).  Trust
> boundary tightened to **84 cumulative enumerated failure modes**
> across W22 + W29 + W30 + W31 + W32 + W33 + W34.  New
> **W34-L-MULTI-ANCHOR-CAP limitation theorem** (proved by
> inspection): when all K anchors are simultaneously compromised
> at the capsule layer, no multi-anchor mechanism (including W34)
> can recover; native-latent (architecture-dependent) is required
> to break this.  Mac 2 still ARP-incomplete (29th milestone).
> SDK_VERSION ``coordpy.sdk.v3.35``; pyproject ``0.5.8``; 753/753
> tests pass (48 W34 unit + 494 phase69-81 + 211 wider coordpy).
> See ``docs/RESULTS_COORDPY_W34_LIVE_AWARE_MULTI_ANCHOR.md`` and
> ``docs/SUCCESS_CRITERION_W34_LIVE_AWARE_MULTI_ANCHOR.md``.
>
> **Position in the research arc**: W35 changes the synthesis from
> "three separated lines" to "one composed audited-proxy stack"; W36
> adds the first host-diverse guard around that stack.  The
> OLD explicit capsule line for multi-oracle adjudication (W19..W22 →
> W21/W22/W33/W34) supplies the probes and trust evidence; the NEW
> dense-control / geometry-aware line (W29..W32) supplies the idea of
> controller-side dense state and projection; the live-aware
> multi-anchor line (W33..W34) supplies the abstention boundary; W36
> supplies the host-attestation boundary.  The composed stack is now
> W36 wraps W35 wraps W34 wraps W33 wraps W21 wraps the capsule-native
> runtime.  It is still an audited capsule-layer proxy.  It does not
> claim transformer-internal hidden-state projection or runtime KV
> transplant.  The deeper architecture-dependent walls
> (W33-C-NATIVE-LATENT, systematic W33-C-CROSS-HOST-LIVE-TRUST-
> MAGNITUDE, and true multi-host W34/W35/W36-C-MULTI-HOST) remain the
> next frontier.
>
> Updated through SDK v3.28, 2026-04-30 (W22 → W27 cross-cell
> efficiency ladder summarised in § 14.2; W27 — multi-chain
> salience-keyed dense-control fanout + per-signature scoping —
> is the first capsule-native method that simultaneously improves
> both efficiency AND correctness over the prior best).
> This file is intended to be the primary publication-grade paper
> draft for the programme's multi-agent context thesis. It is not a
> milestone diary. It is a paper-shaped synthesis of the system,
> theory, benchmarks, strongest positive results, strongest negative
> results, and the current open frontier.

## Abstract

Multi-agent LLM systems usually treat context as text: prompts, JSON
records, message logs, tool traces, and free-form summaries passed
between roles. That design conflates several distinct problems:
preserving ambiguity, normalizing producer drift, admitting evidence
under budget, decoding evidence jointly, carrying information across
rounds, and deciding which evidence a downstream model should spend
its limited context window on. We argue that this is the wrong
abstraction. The unit of context should be a typed, immutable,
lifecycle-bounded object with explicit budget and provenance.

We implement this view in **CoordPy**, a capsule-native runtime and
research harness produced by the **Context Zero** programme. A
capsule is a content-addressed object satisfying six invariants:
identity, type, lifecycle, budget, provenance, and immutability. The
runtime first makes capsules load-bearing in execution rather than
merely post-hoc metadata: capsules are sealed from run setup through
the LLM byte boundary, artifacts are content-addressed at write time,
and lifecycles are mechanically audited. We then extend the same
contract to multi-agent coordination, where agents exchange typed
handoff capsules rather than raw messages.

The paper's main scientific contribution is a decomposition of the
multi-agent context problem into nine coupled structural axes:
(1) admission under budget, (2) intra-round bundle decoding,
(3) cross-round bundle decoding, (4) fixed-vocabulary normalization
of producer drift, (5) layered open-world normalization,
(6) producer-side ambiguity preservation, (7) decoder-side context
packing under bounded token budgets, (8) end-to-end producer-plus-
decoder composition, and (9) bundle-relational compatibility
disambiguation under symmetric corroboration. Across SDK v3.8-v3.19
we build a progressive benchmark ladder, R-54 through R-65, that
isolates these axes one by one. Admission alone wins in some regimes
but has a named ceiling. Bundle-aware decoding crosses that ceiling.
Cross-round decoding crosses a further temporal ceiling. Fixed-table
normalization closes a bounded producer-drift gap; layered
normalization closes a wider but still finite open-world gap.
Structured producer protocols close the first real-LLM upstream
ambiguity-erasure gap. Attention-aware capsule context packing closes
a downstream bounded-context gap. End-to-end composition then yields
the first fresh live real-LLM strict +1.000 advance over the
strongest non-composed baseline in the programme. A bundle-relational
compatibility disambiguator finally crosses the symmetric-corroboration
wall on a regime where the wall actually applies.

The strongest *live* positive result in the paper is SDK v3.18's W17
family: on a fresh live `qwen2.5:14b-32k` probe, a magnitude-hinted
structured producer protocol plus attention-aware capsule packing
yields `accuracy_full = 1.000`, while both substrate/FIFO and the
strongest non-composed baseline remain at 0.000 on the same live
benchmark. The same producer-side intervention transfers partially
across model class to a fresh live `qwen3.5:35b` MoE probe,
preserving the benchmark property 8/8 and yielding a +0.750 strict
gain. The strongest *closed-form-disambiguation* positive result is
SDK v3.19's W18 family: on the synthetic R-65-COMPAT regime (every
gold service AND the decoy mentioned by ≥ 2 distinct routed roles
via generic-noise kinds with comparable magnitudes — symmetric-
corroboration; round-2 specific-tier disambiguator carries a
relational-compound mention of every gold service AND no decoy
service), every closed-form salience scorer in the SDK ties FIFO at
0.000 (W17-Λ-symmetric extends verbatim); the new
:class:`RelationalCompatibilityDisambiguator` achieves
`accuracy_full = 1.000` at both `T_decoder ∈ {None, 24}`, +1.000
strict separation, stable across 5/5 alternate `bank_seed` values.
The strongest *negative* result is equally important: SDK v3.18
proves the programme's first explicit **symmetric-corroboration
limit theorem**, and SDK v3.19 names three further structural
limits — W18-Λ-no-compat (no relational signal → abstain), W18-Λ-
confound (symmetric relational signal → abstain), and W18-Λ-deceive
(adversarial relational signal → trust evidence and fail). The W18-
Λ-deceive falsifier names the structural limit *no closed-form
bundle-relational scorer that trusts its evidence can escape*
without an outside-information axis (W18-C-OUTSIDE, conjectural).

The paper does **not** claim that multi-agent context is solved in an
unqualified sense. The strongest honest claim is narrower and more
useful: multi-agent context becomes tractable when evidence is
represented as typed objects and when the runtime explicitly
separates producer-side ambiguity preservation, normalization,
admission, intra-round decoding, cross-round decoding, and
decoder-side bounded-context packing. The next open problem is no
longer "can capsules help?" It is "what richer semantics or learned
disambiguators are required once corroboration, magnitude, round
structure, normalization, and bounded packing are all exhausted?"

After this paper's main draft, the programme advanced four further
layers along exactly that axis (§ 14.2): a bundle-relational
compatibility disambiguator (W18, SDK v3.19), a bundle-contradiction-
aware trust-weighted disambiguator (W19, SDK v3.20), a single-source
outside-witness acquisition disambiguator (W20, SDK v3.21), and a
trust-weighted multi-source quorum adjudicator (W21, SDK v3.22). Each
layer crosses the prior layer's named structural wall on a regime
where it actually applies, ships ≥ 2 named falsifiers that make its
conditionality sharp, and preserves bounded-context efficiency
byte-for-byte. The W21 milestone (the first capsule-native multi-
agent-coordination method that crosses the **W20-Λ-compromised** wall
via multi-source adjudication under partial oracle compromise) sharpens
the post-paper deeper wall: the escape is bounded above by the
*integrity of the registered oracle set*, not by a richer scoring
rule. **Live LLM transfer (W21-Λ-real / W21-C-LIVE-WITH-REGISTRY) is
empirically partially discharged on Mac-1 mixtral 8x7b at +1.000 over
W20 in the registry-anchored regime; in the harder coalition regime
where the LLM's vote is required for quorum, cross-model split is
sharp (mixtral 8x7b: +0.750; gemma2:9b: +0.000)** — scale + general
knowledge matter for the live W21-Λ-real escape.

### Release framing (SDK v3.43, final release of the v3.4x line)

This paper documents the strongest released **capsule-layer
audited** multi-agent context system from the Context Zero
programme. The shipped product surface is the **CoordPy SDK**, first
publicly released as **SDK v3.43** in May 2026. The released
result is the W42 family's measured strict trust-precision
recovery on `R-89-ROLE-INVARIANT-RECOVER` (1.000 vs the prior best
W41 at 0.500; `Δ = +0.500` across 5/5 seeds, min = max), bounding
the `W41-L-COMPOSITE-COLLUSION-CAP` limitation theorem at the
capsule layer via a third orthogonal evidence axis (the
role-handoff invariance axis). The paper and the repo describe
this result with the same framing on **scope** (capsule-layer
audited proxy; closed-form, deterministic, zero-parameter; not
transformer-internal), **limitations** (the newly proved
`W42-L-FULL-COMPOSITE-COLLUSION-CAP` is not closed; it bounds
adversary cost), and **next work** (`W42-C-NATIVE-LATENT` for
transformer-internal trust transfer; `W42-C-MULTI-HOST` for
K+1-host disjoint topology — both are explicitly out of capsule-
layer scope and require new architectural substrate). The stable
SDK contract (`RunSpec → run report`) is byte-for-byte unchanged
across the v3.4x line; every W22..W42 symbol is exported under
`__experimental__`. The paper does not claim universal solution
of multi-agent context.

## 1. Introduction

Context is the central systems problem in multi-agent LLM workflows.
The default engineering response is prompt-centric: gather more text,
compress it, summarize it, and deliver the result to the next model
call. That framing makes context look like a window-management
problem. In practice it is several distinct problems at once:

1. A producer must preserve relevant ambiguity rather than collapse
   it upstream.
2. A bounded local view must admit the right evidence under budget.
3. A decoder must interpret admitted evidence jointly rather than
   item-by-item.
4. Evidence that arrives in different rounds may need to be decoded
   together.
5. Producer outputs may drift lexically or structurally away from the
   canonical surface expected by downstream logic.
6. Even after admission and decoding, a downstream consumer may still
   receive the wrong subset if the bundle is packed naively under a
   strict token budget.

Most agent systems blur these subproblems inside raw prompts, message
logs, and free-form tool traces. As a result, failures become hard to
classify. A wrong answer may come from upstream producer collapse,
lexical drift, insufficient admission, missing temporal composition,
context-window truncation, or genuine semantic ambiguity. When these
are not separated, the diagnosis tends to be vague: "the model was
confused", "the prompt was weak", "the retriever missed something",
or "the context was too large".

This paper advances a different thesis:

> **Context in multi-agent systems should be treated as an object, not**
> **as untyped text.**

The right unit of context is a typed, immutable, provenance-carrying
object whose lifecycle and budget are explicit. We call these objects
**capsules**. A capsule is not just an envelope around bytes. It is
part of a contract: it has a closed-vocabulary kind, a content-derived
identity, declared parents, explicit budgets, and a lifecycle state.
Once that contract is in place, the context problem becomes
decomposable, auditable, and falsifiable.

### 1.1 Core claim

The strongest current claim of the Context Zero programme is not that
all multi-agent context problems are solved. It is that the context
problem can be turned from a vague prompt-engineering complaint into a
stack of explicit, benchmarked subproblems:

- producer-side ambiguity preservation,
- normalization of producer drift,
- admission under role-local budget,
- intra-round bundle decoding,
- cross-round bundle decoding,
- and downstream bundle packing under bounded decoder context.

Each of these axes can be isolated, attacked, and bounded. The paper's
real contribution is this decomposition plus the evidence that it is
load-bearing.

### 1.2 Why CoordPy matters

The codebase contribution is **CoordPy**, the first product produced by
the Context Zero programme. CoordPy is not the whole programme and it is
not claimed to be a universal agent platform. Its scientific value is
that it makes the thesis executable:

- capsules are load-bearing at runtime rather than reconstructed after
  the fact;
- team coordination is implemented through typed handoffs and bounded
  role views rather than prompt blobs;
- the benchmark family is wired directly to the same runtime objects;
- every positive result is paired with a named falsifier or limit;
- and many invariants are mechanically checked on every run.

The code is not merely an implementation appendix to the theory. It
is the evidence for the theory.

### 1.3 Main contributions

This paper makes six main contributions.

1. **Capsule-native execution.** We show that a runtime can use typed,
   content-addressed objects as its execution contract all the way to
   the LLM byte boundary.
2. **Capsule-native team coordination.** We extend the same contract
   from single-run execution to between-agent coordination.
3. **A benchmark ladder for multi-agent context.** We define a
   sequence of named regimes, R-53 through R-64, each created to
   expose a specific structural failure mode.
4. **Positive and negative theorem/result pairs.** Each method class
   is paired with its own named win and its own named falsifier or
   limit.
5. **A decomposition of the context problem into structural axes.**
   This is what makes the programme cumulative rather than a sequence
   of disconnected benchmark wins.
6. **The first fresh live end-to-end strict real-LLM win under
   bounded context.** SDK v3.18 provides the first result in the
   programme where producer-side and decoder-side interventions are
   both load-bearing and jointly beat the strongest weaker baseline on
   a fresh live real-LLM stream.

### 1.4 What this paper does not claim

This paper does **not** claim that multi-agent context has been
solved universally. The strongest current results remain conditional
on named benchmark properties, structured forms of drift, and bounded
semantic surfaces. Most importantly, the programme now has a named
**symmetric ambiguity wall**: once corroboration, magnitude, round
structure, normalization, and packing are all symmetric, the current
closed-form method family fails by construction.

That negative theorem is not a weakness of the paper. It is evidence
that the programme is now precise enough to identify what remains
unsolved.

### 1.5 Paper roadmap

The paper is organized as follows. Sections 2 through 5 define the
setting, the capsule contract, the runtime, and the team-level object
model. Section 6 introduces the nine structural axes that emerged
from the programme. Sections 7 through 9 describe the benchmark
ladder, the method families, and the evaluation discipline. Section 10
reports the main empirical and theorem-level results. Sections 11
through 17 synthesize those results into the strongest current thesis,
explain what has and has not been solved, position the work relative
to adjacent literatures, and state the current limitations plainly.
The appendices then summarize the claim taxonomy, map milestones to
paper contributions, and list the remaining paper-production steps for
a venue submission pass.

## 2. Setting and Problem Definition

We consider a multi-agent team with roles
$R = \{r_1, \ldots, r_m\}$ collaborating over one or more rounds.
Each role emits evidence items about a shared task. Evidence may be
causal, generic, misleading, redundant, or lexically drifted. One or
more deciding roles operate under bounded budgets and must produce a
team-level decision.

The paper focuses on the following failure modes:

- **Producer collapse:** the producer suppresses alternative
  hypotheses too early.
- **Producer drift:** the producer emits semantically useful but
  lexically drifted kinds or payloads.
- **Admission overload:** the deciding role cannot admit everything.
- **Single-round insufficiency:** round-local evidence is not enough.
- **Decoder-budget failure:** the right evidence is admitted but
  cannot fit into the final decoder context.
- **Symmetric ambiguity:** even with preserved evidence and bounded
  packing, gold and decoy remain indistinguishable under the current
  feature surface.

The programme's claim is that these must be modeled separately. A
single large prompt hides the distinction.

### 2.1 Formal decomposition

Let a scenario unfold over rounds $t = 1, \dots, T$. At each round,
producer role $r_i$ observes a multiset of raw events
$E_{i,t} = \{e_{i,t,1}, \dots, e_{i,t,n}\}$. A producer protocol
$P_i$ maps those events into a multiset of emitted handoff payloads
$H_{i,t}$. A normalizer $N$ maps emitted payloads into a canonical
surface. An admission policy $\pi_r$ selects a bounded subset
$A_{r,t} \subseteq \bigcup_i N(H_{i,t})$ for a deciding role $r$. A
decoder $D$ maps either a single-round view $A_{r,t}$ or a union of
multiple views $\bigcup_t A_{r,t}$ to an answer $\hat{y}$.

The decomposition studied in this paper is therefore:

- **Producer problem:** which informative hypotheses survive into
  $H_{i,t}$?
- **Normalization problem:** which emitted payloads are mapped
  correctly into the canonical surface?
- **Admission problem:** which canonicalized items survive under the
  role-local budget?
- **Decoding problem:** how should the surviving bundle be
  interpreted jointly?
- **Packing problem:** if a downstream consumer has its own strict
  token budget, which subset and ordering of the decoded bundle should
  be kept?

The standard prompt-centric stack typically fuses all of these into
one lossy transformation. The capsule-native view keeps them distinct
and therefore falsifiable.

### 2.2 What counts as "context" here

The paper uses "context" in a narrower and stronger sense than most
prompt-engineering discussions. Context here is not "all bytes seen by
the system." It is the typed, bounded, provenance-aware information
that a role or decoder is permitted to condition on at a particular
decision point. That distinction matters because:

- it is possible for the system to have seen information that a given
  role should not read directly;
- it is possible for preserving *too much* context to be harmful,
  because it destroys the downstream token budget or drowns decisive
  evidence;
- and it is possible for "better context" to mean **less text** but
  **more structure**.

## 3. Capsule Contract

The shared object model is the **capsule contract**. Each capsule
satisfies six invariants:

1. **Identity (C1).** The capsule identifier is a SHA-256 hash of a
   canonicalized representation of kind, payload, budget, and parent
   set.
2. **Typed kind (C2).** Every capsule belongs to a closed vocabulary.
3. **Lifecycle (C3).** Capsules move through
   `PROPOSED -> ADMITTED -> SEALED [-> RETIRED]`.
4. **Budget (C4).** Capsules carry explicit limits such as token
   count, byte size, witness count, round count, or parent count.
5. **Provenance (C5).** Capsules form a DAG stored in a hash-chained
   ledger.
6. **Immutability (C6).** Sealed capsules are frozen.

None of these ingredients is new in isolation. The novelty is that
the same contract is used all the way from runtime execution to team
coordination to evaluation.

### 3.1 Why the contract matters for science

The contract is what lets the programme attach real negative theorems
to system behavior. Several of the strongest limits in the paper
depend directly on the contract:

- if a producer never emits a handoff, no downstream method can
  recover it because the missing evidence has no CID and never enters
  the ledger;
- if a decoder consumes only a bounded packed subset, then dropped
  capsules are not just "ignored information" but a typed excluded
  set;
- if a benchmark property fails, the failure is visible at the object
  level rather than inferred heuristically from text.

Without the contract, many of the paper's strongest negative results
would be soft observations rather than crisp limits.

### 3.2 Scope and cost of the closed vocabulary

The closed vocabulary is simultaneously a strength and a cost.

It is a strength because:

- it makes audit possible,
- it makes theorem statements concrete,
- it makes normalization and decoding surfaces explicit,
- and it prevents silent drift in the meaning of kinds.

It is a cost because:

- every table has a finite closure,
- domain transfer is not automatic,
- and eventually the system hits a wall where richer semantics are
  required.

The later layers, especially W12, W13, and W17, are best read as
systematic explorations of that tradeoff.

## 4. Capsule-Native Execution

The first stage of the programme made capsules load-bearing inside one
CoordPy run. The execution spine includes:

`PROFILE -> READINESS_CHECK -> SWEEP_SPEC -> SWEEP_CELL -> PROVENANCE -> ARTIFACT -> RUN_REPORT`

and then extends inward:

`PROMPT -> LLM_RESPONSE -> PARSE_OUTCOME -> PATCH_PROPOSAL -> TEST_VERDICT`

Key established runtime results include:

- lifecycle/execution correspondence for the run spine and inner loop;
- content-addressing at write time for substantive artifacts;
- deterministic DAG replay;
- mechanical lifecycle audit;
- and a sharp impossibility theorem for authenticating
  meta-artifacts inside the primary ledger, together with a detached
  witness construction.

### 4.1 Runtime as scientific instrument

The runtime should not be understood only as engineering. It is also
the measurement instrument for the research programme. Because every
load-bearing artifact is a typed capsule:

- the exact producer prompt can be referenced and hashed,
- the exact model output can be referenced and hashed,
- parser outcomes become typed witnesses,
- and later benchmark runs can be compared at the object level rather
  than only at the answer level.

This is part of why the programme can draw stronger conclusions than a
typical benchmark-only paper. The runtime is not external to the
claim; it is what makes the claim inspectable.

### 4.2 Why deterministic replay matters

Several later arguments rely on the distinction between:

- a fresh live run,
- a byte-stable replay of a live run,
- and a synthetic or synthetic-real-shaped generator.

Deterministic replay at the capsule DAG level is what lets the paper
state these distinctions precisely. W16 can make a replay-based
composition claim without pretending it is a fresh live claim, and
W17 can then say exactly what changed when the result moved from
replay to live.

## 5. Team-Level Capsule Coordination

At the team layer, the core coordination objects are:

- `TEAM_HANDOFF`: a typed handoff from one role to another;
- `ROLE_VIEW`: a bounded admitted view for one role in one
  coordination step;
- `TEAM_DECISION`: the team-level decision.

The team layer introduces mechanically checked lifecycle invariants
T-1..T-7. It also enables the key separation used throughout the
paper:

- what the producer emits,
- what the deciding role admits,
- what the decoder reads,
- and what the final answer asserts

can all be benchmarked independently.

### 5.1 Local views as bounded-context objects

The `ROLE_VIEW` object is where the programme's bounded-context thesis
becomes operational. A role view is not just a slice of the ledger. It
is the exact set of capsules the role is allowed to condition on,
subject to:

- parent constraints,
- target-role constraints,
- cardinality limits,
- token limits,
- and later, decoder-side packing limits on downstream consumption.

In other words, the role view is the first explicit formal object in
the codebase that stands in for "the minimum sufficient context for
this role at this time." The W15 layer sharpens that further by
introducing the distinction between admitted role view and decoder-fed
packed subset.

### 5.2 Why team-level audit matters

The programme's strongest multi-agent claims are not just about
accuracy. They are also about **well-formedness**. A team-level answer
is only meaningful if the coordination round itself is structurally
sound. The T-1..T-7 audit makes that explicit. This is why almost
every benchmark table in the programme reports audit preservation
alongside correctness.

## 6. Structural Axes of the Context Problem

The best way to understand the programme is as a ladder of structural
axes. Each later axis only became visible because the earlier ones
were isolated first.

### 6.1 Axis 1: Admission under budget

Question: what should a bounded role-local view admit?

Main results:

- **W7-2 / R-54.** Buffered cohort coherence beats FIFO when the gold
  tag has strict plurality.
- **W8-1 / R-55.** Cross-role corroboration beats single-tag
  plurality when decoys have raw plurality but gold has stronger
  distinct-role support.
- **W9-1 / R-56.** Multi-service corroboration beats single-tag
  corroboration on a multi-service gold regime.

Named limits:

- **W7-1.** FIFO is unbeatable when the producer already emits fewer
  candidates than the budget cap.
- **W9-4.** Service-blind admission fails when the decoy is also
  sufficiently corroborated.

Interpretation:

Admission alone can solve some hard context problems, but only when
the distinguishing structure is already visible at the level of what
gets admitted.

### 6.2 Axis 2: Intra-round bundle decoding

Question: once a single-round bundle is admitted, can bundle-aware
decoding solve a regime admission cannot solve?

Main result:

- **W10-1 / R-57.** Bundle-aware decoding strictly beats every
  admission policy on a decoder-forcing regime.

Named limit:

- **W10-Λ.** Admission alone is structurally insufficient on R-57.

Interpretation:

This is the first point where the programme proves that the meaning of
the **bundle** matters, not just the membership of the admitted set.

### 6.3 Axis 3: Cross-round decoding

Question: can evidence from multiple rounds be decoded jointly when
early evidence carries service inventory and later evidence carries
causal specificity?

Main result:

- **W11-1 / R-58.** Multi-round bundle decoding beats every
  single-round method.

Named limits:

- **W11-Λ.** Single-round methods are structurally insufficient.
- **W11-4.** Round-level budget starvation is a sharp falsifier.

Interpretation:

The programme now has a precise example of a context problem that is
not about retrieving the right item but about carrying the right
information across time.

### 6.4 Axis 4: Fixed-vocabulary normalization

Question: if producer drift stays inside a known synonym closure, can
normalization restore the cross-round decoder?

Main result:

- **W12-1 / R-59.** Fixed-table normalization plus multi-round
  decoding closes the synthetic-to-real-shaped gap.

Named limit:

- **W12-4.** Out-of-vocabulary drift defeats the fixed table.

Interpretation:

This is the first place where the programme converts a familiar LLM
"robustness" issue into a named structural axis with both a strict
positive result and a strict falsifier.

### 6.5 Axis 5: Layered open-world normalization

Question: can a layered heuristic normalizer widen the closure beyond
exact lookup while preserving backward compatibility?

Main result:

- **W13-1 / R-60-wide.** Layered normalization strictly beats the
  fixed table on a synthetic open-world regime.

Named limits:

- **W13-4.** Cosmic OOV is a sharp finite-closure wall.
- **W13-Λ-real.** On the first real-Ollama probe, the model emits
  canonical kinds and filters away the intended ambiguity upstream, so
  normalization becomes structurally invisible.

Interpretation:

W13 matters because it teaches the programme not to chase the wrong
next move. If the real producer is already canonicalizing or
compressing aggressively, then downstream normalization is not the
bottleneck.

### 6.6 Axis 6: Producer-side ambiguity preservation

Question: if the real producer compresses away ambiguity, can prompt
and protocol design preserve the hard event shape?

Main result:

- **W14-1 / R-61.** Structured producer protocol yields the first
  real-LLM strict gain on the cross-round stack.

Named limit:

- **W14-Λ-prompt.** If the producer does not emit the necessary
  ambiguous evidence, no downstream capsule method can recover it.

Interpretation:

The W14 layer is the first time the programme attacks the bottleneck
before the capsule coordination pipeline even starts.

### 6.7 Axis 7: Decoder-side bounded-context packing

Question: even if the right evidence is emitted and admitted, can
naive decoder-side packing destroy the advantage under a strict token
budget?

Main result:

- **W15-1 / R-62-tightbudget.** Attention-aware capsule context
  packing strictly beats FIFO packing under bounded `T_decoder`.

Named limits:

- **W15-Λ-budget.** FIFO packing ties FIFO when the decisive
  disambiguator falls past the token budget.
- **W15-Λ-degenerate.** If the token budget is effectively
  unbounded, the W15 gain disappears.

Interpretation:

This is one of the programme's most important conceptual moves
because it turns "minimum sufficient context" from a slogan into a
measurable object.

### 6.8 Axis 8: End-to-end composition

Question: can producer-side ambiguity preservation and decoder-side
packing be jointly necessary on the same regime?

Main results:

- **W16-1 / R-63 synthetic.** W14+W15 composition yields the first
  strict end-to-end composition gain.
- **W16-Λ-real-replay.** The same composition yields a strict gain on
  recorded real-LLM bytes.
- **W17-1 / R-64 live.** Magnitude-hinted producer protocol plus W15
  packing yields the first fresh live end-to-end strict +1.000
  advance over the strongest non-composed baseline.

Named limits:

- **W16-Λ-compose.** When producer collapse and decoder-budget
  pressure both fire, neither layer alone is sufficient.
- **W17-Λ-no-hint.** The legacy structured producer prompt preserves
  only a 7/8 + 0.500 envelope on the live axis.
- **W17-Λ-naive.** Naive prompt plus tight decoder budget yields
  total joint failure.

Interpretation:

This is the strongest evidence so far for the original Context Zero
thesis: the context problem is neither purely upstream nor purely
downstream.

### 6.9 The symmetric-ambiguity wall and the bundle-relational compatibility move (axis 9)

The strongest current negative theorem the programme had as of
SDK v3.18 was:

- **W17-Λ-symmetric.** When gold and decoy are symmetrically
  corroborated under comparable magnitudes, every closed-form
  salience scorer in the SDK ties FIFO at 0.000, even though the
  root-cause label itself can still be correct.

SDK v3.19 takes the smallest move beyond this wall on a regime
where the wall actually applies. The structural observation is
that every prior decoder layer (W11..W17) consumed only
*closed-vocabulary* fields of the admitted bundle: ``claim_kind``,
service tag, bipartite role × tag corroboration, operational
magnitudes, round index. None consumed the *relational text* of
the round-2 specific-tier disambiguator's payload — the substring
``relation=A_B_join`` in
``"deadlock relation=orders_payments_join wait_chain=2"``. On
R-64-SYM only the deadlock scenarios carried such a relational
mention; on the others, no asymmetric channel existed. R-65-COMPAT
is the *consistent* relational-mention regime: every scenario
family (deadlock / pool / disk / slow_query) carries a closed-
vocabulary relational compound naming the gold services in
round-2.

The W18 :class:`RelationalCompatibilityDisambiguator` is a
deterministic, training-free, closed-form scorer:

  1. Tokenise the round-2 disambiguator's payload (lower-cased,
     split on non-identifier chars, compound identifiers preserved).
  2. Score each admitted service tag in the union of admitted
     handoffs by direct-match + contiguous-subsequence compound
     match against the tokens.
  3. Apply the strict-asymmetric branch: keep positive-score tags
     iff at least one but not all admitted tags have positive
     score; otherwise abstain (fall through to the W15 inner
     answer byte-for-byte).

The W18 method clears the strong success bar on R-65-COMPAT:

- **W18-1 / R-65-COMPAT.** W18 = 1.000; every other capsule
  strategy = 0.000. **+1.000 strict separation** at both
  ``T_decoder = None`` (loose) AND ``T_decoder = 24`` (tight),
  stable across 5/5 alternate ``bank_seed`` values. The W18
  method consumes only the W15-packed bundle; ``tokens_kept_sum``
  is byte-for-byte identical to W15's. Bounded-context honesty
  preserved.

The conditionality is sharp on three named falsifiers:

- **W18-Λ-no-compat / R-65-NO-COMPAT.** Round-2 carries no
  service-tag mention; W18 abstains; ties FIFO at 0.000.
- **W18-Λ-confound / R-65-CONFOUND.** Round-2 mentions BOTH gold
  AND decoy; W18 abstains (every admitted tag has positive
  score); ties FIFO at 0.000.
- **W18-Λ-deceive / R-65-DECEIVE.** Round-2 mentions DECOY but
  NOT gold; W18 trusts its evidence and picks decoy; fails at
  0.000.

The named structural limit *no closed-form bundle-relational
scorer that trusts its evidence can escape* is W18-Λ-deceive.
The natural research move beyond it — an outside-information axis
to detect adversarial round-2 mentions — is named
**W18-C-OUTSIDE** and remains conjectural.

The W18 result strengthens the original thesis on the relational-
compatibility axis without retracting any prior axis. It is *not*
"ambiguity resolution solved": the W18-Λ-deceive falsifier names
a genuine structural limit; the W18-Λ-real conjecture names the
real-LLM transfer condition (closed-vocabulary relational
compounds emitted at the round-2 boundary). The defensible reading
is that *one named structural axis the prior milestone left
explicit* is now broken on a regime where it actually applies,
while the deeper adversarial-relational axis remains open.

## 7. Benchmark Ladder: R-53 to R-65

The benchmark family is cumulative. Each regime was built because the
previous one exposed a specific failure mode.

| Regime | Purpose | Main winner | Main limit |
| --- | --- | --- | --- |
| R-53 | Show when FIFO is unbeatable | none | low-surplus ceiling |
| R-54 | Gold-plurality admission regime | W7 | streaming instability |
| R-55 | Cross-role corroboration beats plurality | W8 | corroborated decoy |
| R-56 | Multi-service corroboration | W9 | corroborated multi-service decoy |
| R-57 | Decoder-forcing | W10 | admission ceiling |
| R-58 | Delayed-causal-evidence across rounds | W11 | single-round ceiling |
| R-59 | Synthetic-real-shaped producer drift | W12 | fixed-table OOV wall |
| R-60 | Open-world drift + first real-Ollama probe | W13 | closure wall; upstream erasure |
| R-61 | Producer-side ambiguity preservation | W14 | producer compression |
| R-62 | Decoder-side bounded context packing | W15 | FIFO packing wall |
| R-63 | End-to-end composition | W16 | joint producer+decoder failure |
| R-64 | Fresh live composition + symmetric wall | W17 | symmetric ambiguity |
| R-65 | Relational-compatibility disambiguation under symmetric corroboration | W18 | adversarial-relational round-2 (W18-Λ-deceive) |

### 7.1 Benchmark design principles

The benchmark ladder obeys four design rules.

1. **Each regime is motivated by the previous regime's strongest
   failure mode.**
2. **Each regime has a named bench property.**
3. **Each regime has at least one named falsifier.**
4. **Each new regime preserves or checks earlier anchors.**

This is what keeps the programme from turning into arbitrary benchmark
shopping.

### 7.2 Regime families in plain language

The ladder can also be read narratively:

- **R-53** asks when the producer is already so clean that structure
  cannot help.
- **R-54/R-55/R-56** ask when better admission rescues bounded local
  context.
- **R-57** asks when decoding matters more than admission.
- **R-58** asks when time matters.
- **R-59/R-60** ask when producer drift becomes the bottleneck.
- **R-61** asks what happens when the producer erases ambiguity
  itself.
- **R-62** asks what happens when the decoder cannot afford to read
  the full admitted union.
- **R-63** asks whether those two difficulties interact.
- **R-64** asks what happens when every current asymmetry is removed.
- **R-65** asks whether a single new information channel — the
  round-2 disambiguator's payload text — is enough to break the
  symmetric wall when the channel itself is asymmetric (R-65-COMPAT)
  AND what happens when the channel is silent (NO-COMPAT),
  symmetric (CONFOUND), or adversarial (DECEIVE).

This narrative is the actual research arc of the programme.

### 7.3 Why the ladder is not benchmark shopping

A fair concern for any long-running benchmark programme is that later
regimes may simply encode the method designer's preferred answer. The
paper tries to make that accusation difficult in four ways.

First, each new regime is motivated by a concrete failure mode exposed
by the previous strongest method. The regime does not appear from a
blank slate. It is a response to an identified ceiling. Second, later
regimes preserve earlier anchors rather than replacing them. A method
that wins only on the new regime but regresses badly on earlier ones
does not count as a clean programme advance. Third, each regime has a
named falsifier. The falsifier is not rhetorical. It is an executable
counter-regime in which the new method is expected to tie or fail.
Fourth, the success bars are pre-committed in the repository rather
than post-hoc descriptions written after a favorable run.

This design does not make the ladder bias-free in some impossible
absolute sense. It does, however, convert the relevant bias questions
into visible artifacts: reviewers can inspect which failure motivated
which regime, whether the stated falsifier really binds, and whether
the new method preserves earlier anchors. That is substantially more
scientifically useful than a single monolithic benchmark with no
internal causal story.

## 8. System Design and Method Families

### 8.1 Runtime layer

The runtime contribution is the capsule-native execution contract.
This layer supplies:

- immutable typed objects,
- a hash-chained provenance surface,
- deterministic replay,
- lifecycle audit,
- and content-addressed artifacts.

By itself, this does not solve multi-agent context, but it makes
later claims auditable.

### 8.2 Team coordination layer

The team layer contributes typed handoffs, role views, and team
decisions. The deciding role never sees an arbitrary prompt blob; it
sees a bounded, typed object set.

#### 8.2.1 Why this differs from ordinary message passing

At first glance, typed handoffs may look like ordinary structured
messages. The difference is that capsules are:

- content-addressed,
- lifecycle-bounded,
- provenance-linked,
- budgeted,
- and manipulated by the same runtime contract that governs the rest
  of the system.

That is what allows the programme to make cross-layer claims ordinary
message schemas typically do not support.

### 8.3 Admission methods

The admission family evolves from:

- FIFO and fixed priority,
- to buffered cohort coherence,
- to cross-role corroboration,
- to multi-service corroboration.

Each method is small, deterministic, and interpretable. Each also has
its own named limit. The programme does not hide the fact that
admission is only one part of the story.

#### 8.3.1 Why the admission ladder matters

The admission-side ladder is sometimes misread as a sequence of
slightly fancier heuristics. That is not the right reading. Each step
isolates a different informational pattern:

- **plurality**,
- **cross-role corroboration**,
- **multi-service corroboration**.

Those are different structural features of the evidence stream. W7,
W8, and W9 matter because they show exactly which kinds of structure
admission can exploit before decoding becomes necessary.

### 8.4 Decoder methods

The decoder family evolves from:

- single-round priority decoding,
- to bundle-aware intra-round decoding,
- to contradiction-aware cross-round decoding.

The decisive step is that the decoder operates on bundles of capsules
rather than on isolated items. This is where the argument starts to
look like a real context solution rather than a better filter.

#### 8.4.1 From item scoring to bundle interpretation

The decoder-side move changes the scientific object of study. The
question is no longer "which items are best?" but "what does this
bundle mean jointly?" That distinction is what allows the programme to
state W10-Λ and W11-Λ as ceilings on admission or single-round
reasoning rather than as failures of one heuristic.

### 8.5 Normalization methods

Normalization evolves from:

- no normalization,
- to fixed closed-vocabulary synonym tables,
- to layered heuristic abstraction rules.

The key scientific point is not that a table got larger. It is that
the programme turned closure and closure-failure into named research
objects with positive and negative results.

#### 8.5.1 Why fixed vs layered normalization is a research axis

Normalization is often treated as preprocessing. In this paper it is a
first-class research axis because:

- its closure can be measured,
- its failure modes can be named,
- it interacts with decoder correctness directly,
- and its real-LLM relevance depends on what the producer actually
  emits.

The W13-Λ-real observation is especially important because it shows
that improving a downstream normalizer may be scientifically
irrelevant if the producer is not generating the ambiguity one hoped
to normalize.

### 8.6 Producer-side protocol

The W14/W17 producer protocol family is the first layer that acts
before the capsule pipeline receives evidence. Its purpose is to stop
the model from erasing the hard event shape too early.

W14 contributes:

- observation/diagnosis separation,
- per-tier kind whitelists,
- one-claim-per-event discipline.

W17 adds:

- explicit operational threshold tables,
- anti-relative-magnitude instructions,
- and a fresh live comparison showing that these additions, not mere
  rerunning, close the remaining 1/8 miss.

#### 8.6.1 Protocol design as scientific intervention

The producer protocol is not merely "better prompting." In the
programme it functions as a controlled intervention on the producer's
emission distribution. That distinction matters:

- a good benchmark should not hard-code the answer into the prompt;
- but it may legitimately constrain the producer to emit one claim per
  event, or to separate observation from diagnosis, if that is exactly
  the phenomenon under study.

W14 and W17 are therefore best understood as *protocol* results, not
prompt-aesthetics results.

### 8.7 Decoder-side packing

The W15 packer turns bounded-context efficiency into an explicit axis
rather than a vague hope. The packer does not claim to manipulate
transformer attention weights directly. Instead it optimizes
prompt-facing evidence order and retention under budget using a
closed-form salience score plus hypothesis preservation.

#### 8.7.1 Why token efficiency is not cosmetic

Many systems can improve accuracy by reading more. That is not the bar
in this paper. The W15 layer matters precisely because it creates a
regime where:

- the right evidence exists,
- the system has already admitted it,
- but a bounded downstream consumer still fails unless the bundle is
  packed well.

That is the first place where "minimum sufficient context" becomes a
measurable systems property rather than a slogan.

### 8.8 End-to-end composition

W16 and W17 are the first layers that prove two different parts of
the system must work together on the same cell. This is the strongest
evidence so far for the original Context Zero thesis.

#### 8.8.1 Why composition is the real milestone

The W16/W17 composition results are important because they rule out a
common failure mode of layered research programmes: each layer works
on its own benchmark but they do not matter together. W16-Λ-compose
and W17-1 show the opposite:

- the layers can fail jointly,
- the layers can help jointly,
- and the resulting strict gains are larger than the gains of the
  isolated pieces.

That is the closest the programme has come so far to an actual
end-to-end context solution story.

### 8.9 What remains outside the current family

The paper is explicit that the current methods remain largely
closed-form. The next frontier likely requires:

- richer semantic compatibility scoring,
- learned or distilled disambiguation,
- or domain transfer beyond incident-style coordination.

That is not an omission; it is the precise consequence of
W17-Λ-symmetric.

## 9. Experimental Methodology

### 9.1 Evaluation discipline

The programme uses pre-committed success bars in
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`. A milestone is not
counted as a real advance unless it clears a named bar with:

- a code anchor,
- a benchmark property,
- a named falsifier,
- non-regression on earlier anchors,
- and audit preservation.

This discipline exists because earlier conjectures in the programme
were overstated and later retracted. The paper benefits from that
history rather than hiding it.

#### 9.1.1 Why the retractions matter

The earlier retractions are not embarrassing details to omit. They are
what justify the later discipline. Several later success bars exist
specifically because earlier claims were too easy to state and too
weak to survive replication.

### 9.2 Metrics

The key metrics are:

- `accuracy_full`,
- `accuracy_services`,
- `accuracy_root_cause`,
- audit pass rate,
- admitted capsule counts,
- token/context usage,
- ambiguity survival rate,
- decoy corroboration rate,
- normalization/OOV statistics,
- and cross-seed or cross-bank stability.

For W15-W17, the programme also uses bundle-facing proxy metrics such
as:

- `position_of_first_causal_claim`,
- `tokens_kept_sum`,
- `hypothesis_count_kept`,
- and `n_dropped_budget`.

These are not raw transformer attention weights. They are explicit,
auditable proxies for prompt-facing evidence placement under budget.

#### 9.2.1 Accuracy alone is not enough

The later layers reject a pure-accuracy notion of success. A method
that matches accuracy only by reading materially more context or by
relying on an effectively unbounded decoder budget does not count as a
strong advance in the programme's sense. This is why W15 and later
success bars include decoder-side token budgets explicitly.

### 9.3 Real-LLM evaluation

The real-LLM work in the paper is deliberately conservative. When a
probe is live, it is called live. When it is recorded replay, it is
called replay. When it is synthetic-real-shaped, it is called that.
This matters because the live-vs-replay distinction is one of the
paper's key honesty constraints.

#### 9.3.1 Replay is still scientifically useful

Replay is not a rhetorical fallback. It serves a specific scientific
role:

- it isolates downstream method changes while holding producer bytes
  constant;
- it enables exact comparison between composed and non-composed
  decoders;
- and it provides a bridge between synthetic and live evidence.

The paper therefore treats replay as a legitimate evidence class, but
never as a substitute for fresh live claims.

### 9.4 Reproducibility and artifact discipline

The programme's reproducibility stance is intentionally stronger than
"the code runs on our machine." Each milestone writes durable JSON
artifacts into `docs/data/`, keeps explicit result notes in `docs/`,
and ties named theorem/result families to code anchors and bench
properties. This paper benefits directly from that discipline because
its strongest claims are not reconstructed from memory. They are read
off of preserved benchmark families, CLI entrypoints, and checked
artifacts.

The practical reproducibility story has four components:

1. **Deterministic synthetic anchors.** Synthetic regimes are designed
   to be rerunnable in CI and under seed sweeps.
2. **Byte-stable replay where live endpoints are not the point under
   study.** Replay is used only when holding producer bytes fixed is
   the scientifically relevant move.
3. **Explicit live capture when the claim is about live behavior.**
   The v3.18 claims are careful about this distinction.
4. **A public theory registry and success-bar registry.** These
   registries ensure that claims cannot quietly change type between
   conjecture, empirical observation, and stronger theorem-style
   statement.

No reproducibility regime is perfect. Real model endpoints can change;
local hardware can differ; and some live probes necessarily depend on
service availability. But the paper's main factual claims are all
backed either by deterministic anchors, stable replay artifacts, or
explicit live captures whose status is recorded rather than blurred.

## 10. Main Results

Before discussing the layers one by one, Table 1 summarizes the
programme's strongest current reading.

| Layer | Representative regime | Strongest method | Strongest result | Named limit |
| --- | --- | --- | --- | --- |
| Admission | R-55 / R-56 | W8 / W9 | strict gains over FIFO when corroboration structure exists | corroborated decoy |
| Intra-round decoding | R-57 | W10 | strict gain where admission alone is insufficient | admission ceiling |
| Cross-round decoding | R-58 | W11 | strict gain when delayed evidence matters | single-round ceiling |
| Fixed normalization | R-59 | W12 | synthetic-real-shaped transfer under bounded closure | fixed OOV wall |
| Layered normalization | R-60 | W13 | wider synthetic open-world closure | cosmic-OOV wall |
| Producer protocol | R-61 / R-64 | W14 / W17 | live ambiguity-preservation gain | producer compression |
| Decoder packing | R-62 | W15 | bounded-context strict gain under tight decoder budget | FIFO packing wall |
| Composition | R-63 / R-64 | W16 / W17 | live end-to-end strict gain over strongest non-composed baseline | symmetric ambiguity |

### 10.1 Runtime layer

The runtime story is already stronger than a standard agent harness:

- capsules govern execution to the LLM byte boundary,
- artifacts are content-addressed at creation time,
- lifecycle audit is mechanical,
- deterministic replay exists,
- and meta-artifact authentication has a sharp impossibility theorem
  plus a constructive workaround.

This gives the team results a trustworthy substrate.

#### 10.1.1 Why this is more than infrastructure

Without this runtime, many later failures would remain hard to
classify. In a conventional agent stack, the difference between
"producer never emitted decoy" and "downstream decoder dropped decoy"
can be surprisingly difficult to pin down. In CoordPy, those are
different object-level states.

### 10.2 Admission layer

The admission results establish that structure can beat FIFO, but only
in the right regimes:

- W7 wins under gold plurality,
- W8 wins under cross-role corroboration,
- W9 wins under multi-service corroboration.

The admission layer is therefore real but limited.

#### 10.2.1 What the admission ladder actually proves

The admission ladder proves that bounded-context coordination already
contains real structure before learned semantics or bundle
interpretation appear. The later decoder wins do not erase W7-W9;
they explain where their limits are.

### 10.3 Decoder layer

The decoder results establish that downstream bundle interpretation is
not reducible to better admission:

- W10 crosses the admission ceiling,
- W11 crosses the single-round ceiling.

This is where the programme first demonstrates that the meaning of the
bundle, not just its membership, matters.

#### 10.3.1 Why bundle semantics changed the programme

Before W10, the strongest story was "choose the right evidence."
After W10 and W11, the story becomes "choose the right evidence, then
interpret it jointly, possibly across time." That is much closer to
what researchers informally mean when they say that an agent team has
a context problem.

### 10.4 Normalization layer

The normalization results show that real or real-shaped producer drift
must be handled explicitly:

- W12 wins under bounded fixed-table closure,
- W13 widens the closure,
- W13-Λ-real shows that normalization is not always the active
  bottleneck on real producers.

#### 10.4.1 The hidden value of the null real result

The W13-Λ-real observation is one of the most important negative
results in the programme because it prevents a wasted research path.
It shows that the right next move after W13 was not "add more
synonyms." It was "fix the producer-side event shape." That kind of
negative result is exactly what a serious paper should include.

### 10.5 Producer-side protocol layer

The W14/W17 story is one of the paper's central contributions.

W14 shows that:

- ambiguity can be erased upstream,
- no downstream method can recover missing emitted evidence,
- and structured prompts can restore the needed event shape.

W17 then shows:

- the remaining model-side relative-magnitude miss can be closed,
- the fresh live gain can be doubled from +0.500 to +1.000,
- and the result transfers partially across model class.

#### 10.5.1 Why W17 is more than a prompt tweak

The W17 layer is not just a slightly better prompt. It closes a
specific model-side loophole: the model was comparing event magnitudes
relatively instead of against the operational thresholds that define
the benchmark property. The contribution is therefore structural: it
changes which events qualify to become capsules in the first place.

### 10.6 Decoder-side packing layer

W15 shows that even when the right evidence exists in the admitted
union, a bounded decoder context can still destroy the win. This is
one of the strongest parts of the paper because it makes the phrase
"minimum sufficient context" measurable rather than philosophical.

#### 10.6.1 Why the packing result is central to the original goal

The original goal of the programme was never just "better
coordination." It was **bounded** coordination: per-agent minimum
sufficient context. W15 is therefore central because it is the first
place where the programme measures not only whether the right answer
is obtained but whether the answer survives a strict decoder context
limit.

### 10.7 Composition layer

W16 and W17 together provide the strongest end-to-end story in the
programme:

- producer-side preservation matters,
- decoder-side packing matters,
- both can be jointly necessary,
- and together they yield the first fresh live strict win.

#### 10.7.1 Why W16 and W17 are the paper's practical center

If the paper had to pick one main result for a broad audience, it
would be W17: fresh live producer-side ambiguity preservation plus
decoder-side packing yields the first strict +1.000 gain over the
strongest weaker baseline on a live model stream.

### 10.8 Symmetric ambiguity wall

The new negative theorem W17-Λ-symmetric is as important as the fresh
live win. It shows that the current method family still depends on
asymmetry in the evidence pattern. Once that asymmetry disappears, the
closed-form capsule-native methods in the SDK stop being sufficient.

That is not a weakness of the paper. It is the cleanest statement yet
of what remains unsolved.

#### 10.8.1 Why the symmetric wall is scientifically valuable

The symmetric wall gives the next paper its problem statement. Without
it, the programme could keep stacking conditional wins without ever
identifying the deeper semantic bottleneck. W17-Λ-symmetric stops that
drift. It says, in effect: the next step is not another packing tweak;
it is a richer ambiguity-resolution method.

## 11. Result-by-Result Synthesis

It is useful to summarize the cumulative story as a progression of
questions and answers.

### 11.1 Can context be made into a runtime object?

Yes. The W3 runtime results show that capsules can govern execution,
not just describe it after the fact.

### 11.2 Can the same object model govern team coordination?

Yes. The W4 layer shows that typed handoffs and role views can be
made load-bearing between agents.

### 11.3 Can structure beat FIFO at all?

Yes, but conditionally. W7-W9 prove several admission-side wins with
named limits.

### 11.4 Can decoding beat admission?

Yes. W10 proves the first decoder-side strict separation from every
admission policy.

### 11.5 Can time matter structurally?

Yes. W11 proves that some regimes require cross-round reasoning.

### 11.6 Can synthetic drift results transfer?

Partly. W12 and W13 show that transfer is possible under bounded or
layered closure assumptions, but W13-Λ-real shows that real producer
behavior can make normalization irrelevant by collapsing ambiguity
upstream.

### 11.7 Can real models be made to preserve the hard event shape?

Yes. W14 and W17 show that prompt/protocol design can recover the
necessary ambiguity on real models, and that operational threshold
language can close the remaining model-side relative-magnitude miss.

### 11.8 Does bounded decoder context really matter?

Yes. W15 shows that downstream context packing is not cosmetic; it can
be the difference between perfect success and total failure.

### 11.9 Do the producer and decoder layers compose?

Yes. W16 and W17 show the first end-to-end composition results, first
on replayed bytes and then on a fresh live real-LLM probe.

### 11.10 What remains unsolved?

The symmetric wall. W17-Λ-symmetric shows that the current closed-form
surface cannot resolve true symmetry.

## 12. What Has Actually Been Solved?

The programme has **not** solved all of multi-agent context. What it
has solved is the decomposition and several substantial subclasses:

1. it solved several admission-sensitive subclasses;
2. it solved decoder-forcing subclasses that admission cannot solve;
3. it solved cross-round subclasses that single-round decoding cannot
   solve;
4. it solved bounded producer-drift subclasses under explicit
   normalization assumptions;
5. it solved a producer-side ambiguity-preservation subclass on a real
   model;
6. it solved a decoder-budget subclass where bounded context itself is
   the main difficulty;
7. and it solved the first end-to-end live composition subclass where
   both producer-side and decoder-side layers are jointly necessary.

That is a substantial research result. It is also still conditional.

### 12.1 Why this still counts as progress toward "solving context"

The phrase "solve context" can be misleading if read as a universal
claim. In this paper, progress means something more disciplined:
showing that more of the difficulty is now structurally understood,
measurable, and addressable. By that standard, the programme has made
substantial progress. It has turned multiple previously blurred
failure modes into explicit theorem/result pairs and has demonstrated
end-to-end live improvement on at least one real model stream.

## 13. The Strongest Current Thesis After SDK v3.24

The strongest current thesis is:

> **Multi-agent context becomes tractable when the system is designed**
> **as a layered capsule-native coordination pipeline in which**
> **producer-side ambiguity preservation, normalization, admission,**
> **bundle-aware decoding, cross-round decoding, and bounded-context**
> **packing are each explicit, audited, and benchmarked.**

This is much stronger than "better prompts help" and much more
precise than "context windows are too small."

It is also stronger than the runtime-only claim. The runtime matters
because it lets the programme audit the evidence flow. But the real
scientific content is the layered decomposition above.

### 13.1 The shortest honest thesis statement

If the entire programme had to be reduced to one sentence, it would
be:

> Multi-agent context is not a single bottleneck but a layered object
> pipeline; solving it means making each layer explicit, bounded, and
> empirically accountable.

## 14. Why the Symmetric Wall Matters

The symmetric wall is not a side note. It is the next real frontier.

The current methods all rely on some asymmetry:

- a gold plurality,
- a corroboration asymmetry,
- a causal-tier asymmetry,
- a round asymmetry,
- a producer-side preservation asymmetry,
- or a packing asymmetry.

W17-Λ-symmetric shows that once the evidence becomes fully symmetric
under the current feature surface, the closed-form strategy family in
the SDK can no longer separate gold from decoy. That means the next
class of methods must add some richer semantic or learned
disambiguation capability.

This is exactly where the next paper or next major section of the
programme should go.

### 14.1 What the next method probably looks like

The symmetric wall strongly suggests that the next successful method
will have to go beyond the current closed-form surface. Likely
directions include:

- learned compatibility scoring across bundles,
- distilled semantic disambiguators,
- or richer narrative-level hypothesis scoring.

The current paper does not attempt to solve that wall prematurely. It
earns the right to ask the question precisely.

### 14.2 Subsequent escape ladder (SDK v3.19 → v3.20 → v3.21 → v3.22)

After this paper's main draft, the programme advanced four further
layers along exactly the axis named above. Each layer is documented
in its own milestone results note; the canonical references are:

- **W18 — Bundle-relational compatibility disambiguator (SDK v3.19,
  *RESULTS_COORDPY_RELATIONAL_DISAMBIGUATOR.md*).** The first capsule-
  native method that crosses the **W17-Λ-symmetric** wall on a
  regime where it actually applies (R-65-COMPAT). A closed-form
  scorer reads the round-2 disambiguator's payload text — the
  channel every prior decoder ignored — and projects W11 / W15's
  answer through a strict-asymmetric branch. Three named falsifiers
  (R-65-NO-COMPAT, R-65-CONFOUND, R-65-DECEIVE) make the W18-1
  conditionality sharp.
- **W19 — Bundle-contradiction-aware trust-weighted disambiguator
  (SDK v3.20, *RESULTS_COORDPY_DECEPTIVE_AMBIGUITY.md*).** The first
  capsule-native method that crosses the **W18-Λ-deceive** wall on
  the bundle-resolvable case (R-66-DECEIVE-NAIVE,
  R-66-CONFOUND-RESOLVABLE). A closed-form scorer counts independent
  asymmetric witnesses *excluding* the canonical primary and inverts
  W18's projection when witnesses contradict the primary. Two named
  falsifiers (R-66-DECEIVE-TOTAL = no asymmetric witness anywhere;
  R-66-OUTSIDE-REQUIRED = witnesses are themselves symmetric) make
  the **W19-Λ-total** and **W19-Λ-outside** walls explicit. The
  natural escape from both walls — *outside information* — is named
  W19-C-OUTSIDE.
- **W20 — Outside-witness acquisition disambiguator (SDK v3.21,
  *RESULTS_COORDPY_OUTSIDE_INFORMATION.md*).** The first capsule-
  native method that crosses the **W19-Λ-outside** wall on a regime
  where it actually applies (R-67-OUTSIDE-RESOLVES). A typed
  ``OutsideWitnessOracle`` Protocol + a deterministic
  ``ServiceGraphOracle`` add an evidence-acquisition step (one
  query per cell, bounded by ``max_response_tokens``) that the
  bundle-only scorer cannot see. Three named falsifiers
  (R-67-OUTSIDE-NONE = no signal; R-67-OUTSIDE-COMPROMISED =
  adversarial signal; R-67-JOINT-DECEPTION = jointly compromised)
  make **W20-Λ-none / W20-Λ-compromised / W20-Λ-joint-deception**
  explicit. Live LLM transfer (W20-Λ-real): mixtral 8x7b on Mac-1
  achieves +0.750 over W19; smaller / coding-specialised models
  trust the deceptive primary and tie FIFO. The natural escape
  from W20-Λ-compromised — *multi-oracle aggregation* — is named
  W20-C-MULTI-ORACLE.
- **W21 — Trust-weighted multi-oracle adjudicator (SDK v3.22,
  *RESULTS_COORDPY_MULTI_ORACLE_ADJUDICATION.md*).** The first
  capsule-native method that crosses the **W20-Λ-compromised**
  wall on a regime where it actually applies (R-68-MULTI-MAJORITY).
  A registered set of N typed oracles with prior trust weights;
  the W21 scorer issues one bounded query per oracle per cell,
  counts per-tag votes across non-abstaining replies, and
  projects only when ≥ ``quorum_min`` oracles agree on a
  non-empty proper asymmetric subset. Three named falsifiers
  (R-68-MULTI-NO-QUORUM = oracles disagree; R-68-MULTI-ALL-
  COMPROMISED = jointly compromised registered set;
  R-68-MULTI-PARTIAL = sub-quorum honest signal) make
  **W21-Λ-no-quorum / W21-Λ-all-compromised / W21-Λ-partial**
  explicit. The deeper wall is now sharper: the W21 escape is
  bounded above by the *integrity of the registered oracle set*,
  not by a richer scoring rule. The conditional W21-C-PARTIAL-
  RECOVERY (with ``quorum_min = 1`` on R-68-MULTI-PARTIAL) is
  empirically discharged at 1.000 — the quorum-strictness trade-
  off is real. Live LLM transfer (W21-Λ-real / W21-C-LIVE-WITH-
  REGISTRY): a four-oracle live registry pairing two deterministic
  registry oracles with mixtral 8x7b achieves +1.000 over W20
  (registry-anchored regime, partially discharging
  W20-C-LIVE-WITH-REGISTRY). On the harder coalition regime (LLM
  vote required for quorum), cross-model split is sharp:
  mixtral 8x7b achieves +0.750; gemma2:9b lands decoy tokens
  through the closure and ties FIFO at 0.000. **Scale + general
  knowledge matter for the W21-Λ-real escape on the LLM-vote-
  required regime**.

**Cross-cell efficiency ladder (SDK v3.23 → v3.28).** After
W21 the programme advances on a different axis: not "how to escape
a stronger semantic wall" but "how to amortise the cost of the
already-working capsule-native pipeline across cells, agents, and
salience signatures":

- **W22 — Capsule + audited latent-state-sharing hybrid (SDK v3.23,
  *RESULTS_COORDPY_CAPSULE_LATENT_HYBRID.md*).** A typed
  ``LatentDigestEnvelope`` per cell carrying the W21 quorum result
  + projected subset, signed at construction; a
  ``SharedReadCache`` reuses identical-query oracle replies across
  cells. The first capsule-native method that combines explicit-
  capsule passing with audited proxies for the LatentMAS
  latent-state-sharing direction.
- **W23 — Cross-cell delta + super-token reference (SDK v3.24,
  *RESULTS_COORDPY_W23_CROSS_CELL_DELTA.md*).** A
  ``SessionDigestEnvelope`` (hash-chained running cross-cell state)
  + ``SessionDeltaEnvelope`` (per-cell delta) +
  ``SuperTokenReferenceEnvelope`` (single-token CID dense-control
  reference). The first capsule-native method to amortise running
  state via O(1) visible-token references per cell.
- **W24 — Bounded-window session compaction + intra-cell
  resample-quorum + cross-process wire (SDK v3.25,
  *RESULTS_COORDPY_W24_SESSION_COMPACTION.md*).** A
  ``MultiCellSessionCompactor`` folds the last
  ``compact_window - 1`` cell digests into one fixed-size
  ``SessionCompactEnvelope``; a ``ResampleQuorumCachingOracleAdapter``
  mitigates intra-cell drift on probabilistic LLM oracles; a
  ``CrossProcessProducerDecoderWire`` round-trips JSON envelopes
  through a real Python subprocess (real cross-*process*, not
  cross-*host*). −18 % over W23 on R-71-LONG-SESSION.
- **W25 — Shared-fanout dense-control + cross-agent state reuse
  (SDK v3.26, *RESULTS_COORDPY_W25_SHARED_FANOUT.md*).** One producer
  computes one ``FanoutEnvelope`` for K named consumers, each
  consumer resolves via 1 ``<fanout_ref:DDDD>`` token. −69.87 %
  over W24 on R-72-FANOUT-SHARED at K=3.
- **W26 — Chain-persisted dense-control fanout + per-consumer
  projections (SDK v3.27,
  *RESULTS_COORDPY_W26_CHAIN_PERSISTED_FANOUT.md*).** A two-tier
  envelope hierarchy (``ChainAnchorEnvelope`` +
  ``ChainAdvanceEnvelope``) amortises the producer's per-cell
  salience-token cost across cells via 1-token chain-advance
  references; per-consumer ``ProjectionSlot`` map enforces
  controller-verified scope. −68.79 % over W25, −90.60 % over W24
  on R-73-CHAIN-SHARED at K=3, scaling to −92.23 % over W24 at
  K=10. W25-C-K-SCALING discharged at K∈{3,5,8,10}.
- **W27 — Multi-chain salience-keyed dense-control fanout +
  per-signature scoping (SDK v3.28,
  *RESULTS_COORDPY_W27_MULTI_CHAIN_PIVOT.md*).** The first capsule-
  native method that *simultaneously* improves both efficiency AND
  correctness over the prior best (W26) on a regime where the
  prior best architecturally limits correctness. A bounded pool
  of independent W26 stacks keyed by
  :func:`compute_input_signature_cid` over canonical input
  handoffs; the audited ``MultiChainPersistedFanoutDisambiguator``
  ships :func:`verify_salience_signature` (4 enumerated failure
  modes) and :func:`verify_chain_pivot` (8 failure modes). On
  R-74-XORACLE-RECOVER (1 producer + K=3 consumers, 16 cells, 2
  signatures, partial ServiceGraphOracle on the W26 baseline):
  **−76.27 % over W26 AND +0.500 correctness over W26**, stable
  across 5/5 seeds. Discharges **W26-C-DIVERGENCE-RECOVERY** in
  the per-signature scoping direction. Four named falsifiers
  (W27-Λ-single-signature / -pool-exhausted / -pivot-tampered /
  -signature-drift) make the W27-1 conditionality sharp.

The post-paper four-layer escape ladder (W18 → W19 → W20 → W21)
discharges, in order: the symmetric-corroboration wall, the
bundle-deceive wall (bundle-resolvable case), the bundle-outside
wall (outside-resolvable case), and the single-oracle wall
(majority-honest case). Each layer adds one structurally-distinct
move (relational scoring → trust-weighted contradiction → outside-
witness acquisition → multi-source quorum); each layer is
*conditional* on the next regime's named bench property; each
layer ships ≥ 2 named falsifiers that make its conditionality
sharp; each layer is *empirically validated* on a fresh synthetic
anchor at 1.000 strict gain over the prior strongest method, and
(for W17, W20, W21) on a *fresh live LLM probe* that materially
crosses the prior wall. The strongest current thesis after SDK
v3.22 is therefore:

> **Multi-agent context becomes tractable when the system is**
> **designed as a layered capsule-native coordination pipeline in**
> **which producer-side ambiguity preservation, normalization,**
> **admission, bundle-aware decoding, cross-round decoding,**
> **bounded-context packing, bundle-relational disambiguation,**
> **trust-weighted bundle-contradiction handling,**
> **outside-witness acquisition, AND trust-weighted multi-source**
> **quorum adjudication are each explicit, audited, benchmarked,**
> **and bounded above by named structural walls.** The deeper
> walls (W21-Λ-all-compromised, W21-Λ-no-quorum) are *named* and
> *proved-empirical*; the natural escapes (W21-C-CALIBRATED-TRUST
> via prior calibration; W22 via cross-source consistency
> detection) are *conjectural*.

## 15. Limitations

This paper has several important limitations.

1. **The strongest positive results remain benchmark-conditional.**
   That is by design, but it matters.
2. **The current family is still domain-specific.** Most strong
   results are in incident-style coordination regimes.
3. **The current semantic surface is still largely closed-form.**
   The next wall likely requires richer learned semantics.
4. **Cross-model transfer is only partial.** The 35B result is a real
   positive but not yet saturation.
5. **The product boundary is narrower than the research surface.**
   Many of the strongest methods remain research-grade and opt-in.

These are not weaknesses to hide. They are the reason the paper is
credible.

### 15.1 Why the limitations section is part of the contribution

One of the programme's distinguishing features is that limitations are
named as part of the method story, not left to reviewer inference.
This has two benefits:

- it makes the positive claims stronger, because they are bounded;
- and it turns the next research agenda into a direct continuation of
  the current one.

### 15.2 Internal validity threats

The main internal-validity question is whether later regimes merely
mirror the engineering intuitions of the current method family. The
paper's answer is incomplete but concrete: named falsifiers, anchor
preservation, and explicit ceilings reduce that risk, but they do not
eliminate it. Reviewers should still inspect whether a given regime
overfits a specific closed-form scoring rule or prompt protocol.

Another internal-validity issue is the mixture of evidence classes.
Some claims are purely deterministic synthetic statements; others are
live empirical observations against model endpoints. The paper tries to
avoid category mistakes by naming these evidence classes explicitly.
Still, a reader should not treat a synthetic stability result and a
live endpoint result as equally robust in the same sense.

### 15.3 External validity threats

The programme currently lives in a particular family of tasks:
incident-style, multi-service, causally entangled coordination. That
family is broad enough to expose real context failures but narrow
enough that some of the current semantic surfaces are meaningful. It
remains an open question how much of the W7-W17 ladder transfers to:

- product-planning teams,
- long-horizon software agents,
- scientific assistants,
- negotiation or dialogue-heavy teams,
- or workflows where the relevant evidence is not naturally organized
  as service-root-cause hypotheses.

This is why the paper presents the current thesis as a programme
result, not a universal theorem about all multi-agent cognition.

### 15.4 Why the remaining wall matters more than another incremental win

The symmetric-corroboration wall is not just the next benchmark. It is
the point at which the current feature surface no longer carries enough
information to decide correctly. That matters because it distinguishes
between two different futures for the programme.

In one future, a modestly richer but still largely interpretable
semantic compatibility surface breaks the wall and preserves bounded
context efficiency. In the other, symmetry can only be broken by
methods that effectively reintroduce large opaque model calls over
nearly the whole bundle, at which point the object model still helps
auditing but no longer supplies the main disambiguation power. The
paper does not know yet which future is correct, but it now states the
choice sharply enough to study.

## 16. Related Work and Positioning

The programme sits at the intersection of several literatures:

- content-addressed and tamper-evident object systems,
- event-sourcing and provenance-aware execution,
- exact-memory and bounded-context systems,
- multi-agent coordination and blackboard-style architectures,
- retrieval and memory systems for LLMs,
- prompt/protocol design for structured extraction,
- and evaluation/runtime harnesses for LLM systems.

The distinct contribution here is not merely that CoordPy has a ledger
or that it uses typed objects. It is that the paper uses one object
model to unify:

- runtime execution,
- team coordination,
- theorem/limit statements,
- and benchmarked empirical advances.

The final submission version should include a proper bibliography and
explicit positioning against adjacent systems, memory papers, and
multi-agent reasoning papers. This draft intentionally avoids
inventing loose citations without a proper reference pass.

### 16.1 Memory, retrieval, and long-context systems

Many current systems papers frame the context problem as retrieval or
memory management: decide which past messages, tool outputs, or
documents to place in front of the next model call. That literature is
highly relevant here, but the present paper differs in two important
ways.

First, it does not assume that "the right context" already exists as a
stable set of textual units waiting to be ranked. The producer may have
collapsed useful ambiguity before retrieval even begins. Second, the
paper separates admission, decoding, and packing into different axes.
This makes it possible to say that one system failed because the right
object never survived the producer, while another failed because the
right object survived but fell out of the bounded decoder bundle.

### 16.2 Multi-agent prompting and role specialization

There is a growing literature on prompt-based multi-agent systems in
which different agents critique, debate, verify, or specialize. The
paper is aligned with that literature in spirit but differs in method.
Most prompt-centric multi-agent work takes the message exchange itself
as the central object. Here, the message exchange is downstream of a
typed object model with explicit lifecycle and provenance. That shift
is what lets the paper connect protocol design, normalizer design,
decoder design, and bounded-context packing inside one cumulative
programme.

### 16.3 Formal methods, provenance, and systems audit

The capsule contract also places the paper near provenance-aware and
auditable systems work. However, ordinary provenance systems typically
stop at traceability: which component produced which artifact? The
present paper uses provenance as part of a tighter execution contract.
The same typed object surface that makes audit possible also serves as
the unit of coordination and the unit of scientific evaluation. That is
why the paper is not only a systems-audit paper, not only a runtime
paper, and not only an agent paper.

### 16.4 How this paper differs from a runtime paper

A systems reader may initially see CoordPy as a runtime paper with
benchmark appendices. That is not the right reading. The runtime is
necessary, but the main scientific object is the decomposition of
context across the benchmark ladder. The runtime is what makes the
decomposition executable.

### 16.5 How this paper differs from a prompting paper

A reader coming from prompt engineering may initially see W14 or W17
as prompt wins. That is also not the right reading. The producer
protocol layers matter because they sit inside a broader object-level
stack with named downstream limits. Their meaning comes from that
stack, not from prompt craft alone.

## 17. Discussion: What Would Count as Truly Solving Context?

In this repo, a serious claim that context is solved would require at
least:

1. strong results across several benchmark families rather than one;
2. fresh live real-LLM wins, not replay only;
3. explicit bounded-context efficiency, not just accuracy;
4. robustness beyond hand-built asymmetries;
5. and a convincing story for what happens at the symmetric wall.

The programme is not there yet. But it is now far beyond the stage of
"interesting intuition." It has:

- an executable runtime contract,
- a growing theorem registry,
- multiple strict positive separations,
- multiple named negative theorems,
- and the first fresh live end-to-end win.

That is enough to support a serious publication.

### 17.1 What a skeptical reviewer should believe

After reading the paper, a skeptical reviewer should at least accept
the following:

1. The programme has an unusually explicit object-level runtime and
   coordination surface.
2. The benchmark ladder is cumulative and not arbitrary.
3. Several strong positive results are real, not cherry-picked.
4. Several strong negative results are also real, and they sharpen the
   frontier rather than weakening the paper.
5. The live end-to-end result is strong enough to justify a main paper
   even though the overall thesis remains conditional.

## 18. Conclusion

This paper has one central message:

> **Context in multi-agent LLM systems is not primarily a prompt-size**
> **problem. It is an object-level coordination problem.**

Capsules provide the object model. CoordPy provides the executable
runtime. The benchmark ladder R-53 through R-64 turns the context
problem into a sequence of explicit, falsifiable subproblems.

The strongest current result is not "we solved context." It is:

- admission helps and has a named ceiling,
- decoding helps beyond that and has a named ceiling,
- cross-round reasoning helps beyond that and has a named ceiling,
- normalization helps beyond that and has a named ceiling,
- producer-side ambiguity preservation helps beyond that,
- decoder-side bounded-context packing helps beyond that,
- producer-decoder composition helps beyond that,
- and symmetric ambiguity is the next named wall.

That decomposition is the real scientific contribution. It turns
"context" from a vague complaint into a structured research programme
with executable evidence, honest limits, and a clear next frontier.

### 18.1 Final take-away

The deepest contribution of the paper is not any single +1.000 result.
It is the fact that the programme can now say, with code and tests
behind it, **which layer failed**:

- the producer,
- the normalizer,
- the admission policy,
- the single-round decoder,
- the cross-round decoder,
- the bounded-context packer,
- or the semantic surface itself.

That is what "context as objects" finally buys.

## Appendix A. Claim Taxonomy

The programme uses an explicit taxonomy:

- **proved**
- **proved-conditional**
- **mechanically-checked**
- **empirical**
- **conjectural**
- **retracted**

This taxonomy is essential. Many agent-system papers blur these
statuses; this programme explicitly does not.

## Appendix B. Milestone-to-Paper Map

The current paper incorporates the following layers:

- **W3 family:** capsule contract and capsule-native runtime
- **W4 family:** team-level capsule coordination
- **W7-W9 families:** admission-side coordination ladder
- **W10 family:** intra-round bundle decoding
- **W11 family:** cross-round bundle decoding
- **W12 family:** fixed-vocabulary normalization under bounded
  producer drift
- **W13 family:** layered open-world normalization and real-Ollama
  null result
- **W14 family:** producer-side ambiguity preservation
- **W15 family:** decoder-side bounded-context packing
- **W16 family:** end-to-end producer+decoder composition
- **W17 family:** fresh live composition, magnitude-hinted protocol,
  cross-model live transfer, and symmetric-corroboration wall
- **W18 family:** bundle-relational compatibility disambiguation
  under symmetric corroboration (R-65, SDK v3.19)
- **W19 family:** bundle-contradiction-aware trust-weighted
  disambiguation under deceptive / confounded round-2 evidence
  (R-66, SDK v3.20)
- **W20 family:** outside-witness acquisition under bundle-only
  insufficiency (R-67, SDK v3.21)
- **W21 family:** trust-weighted multi-oracle adjudication under
  partial oracle compromise (R-68, SDK v3.22)
- **W22 family:** capsule + audited latent-state-sharing hybrid —
  schema-passing (`SchemaCapsule`), delta execution
  (`LatentDigestEnvelope`), shared-read cache (`SharedReadCache`
  + `CachingOracleAdapter`), and controller-side verification
  (`verify_latent_digest`); the first capsule-native multi-agent
  coordination method that *combines* explicit-capsule passing
  with audited proxies for the LatentMAS direction (collective
  KV pooling / latent hidden-state transfer / super-token side
  channels). On R-69-CACHE-FANOUT the W22 method strictly
  reduces visible-tokens-to-decider by 14.51-16.09 % synthetic
  and 39.08 % live-mixtral while ratifying W21 correctness
  byte-for-byte. Three named falsifiers (W22-Λ-no-cache,
  R-69-POISONED-DIGEST, R-69-SCHEMA-DRIFT) and one backward-
  compat anchor (R-69-NO-TRIGGER) make the conditionality sharp.
  Newly named conjecture **W22-C-CACHE-AMPLIFICATION** (the
  cache freezes a probabilistic LLM oracle's first reply across
  matching cells) emerges from the live mixtral 8x7b probe.
  (R-69, SDK v3.23)
- **W23 family:** capsule-native cross-cell delta execution +
  quorum-keyed cache + super-token reference — hash-chained
  cross-cell session digest (`SessionDigestEnvelope`), per-cell
  delta (`SessionDeltaEnvelope`), single-visible-token CID-prefix
  reference (`SuperTokenReferenceEnvelope` +
  `SuperTokenRegistry`), per-oracle freshness policy
  (`QuorumKeyedSharedReadCache` +
  `QuorumKeyedCachingOracleAdapter`), within-process
  producer/decoder host-split proxy
  (`CrossHostProducerDecoderProxy`), and three new controller-side
  verification functions (`verify_session_digest_chain`,
  `verify_session_delta`, `verify_super_token_reference`). The
  first capsule-native multi-agent coordination method that
  *combines* explicit-capsule passing with audited proxies for the
  LatentMAS *cross-cell* state-sharing + super-token side channel
  + cache-amplification mitigation directions. On R-70-DELTA-FANOUT
  the W23 method strictly reduces visible-tokens-to-decider over
  the W22 baseline by 6.67-7.53 % (delta path) and 25.45-28.77 %
  (super-token path) while ratifying W22 correctness byte-for-byte;
  stable across 5/5 seeds. On R-70-AMPLIFIED-LLM the W23
  quorum-keyed cache **empirically discharges**
  W22-C-CACHE-AMPLIFICATION at +0.125 strict mitigation advantage
  over W22 (the first empirical demonstration in the programme
  that a named conjecture's weakness is mitigable). Three named
  falsifiers (R-70-NO-DELTA, R-70-SUPER-TOKEN-TAMPERED,
  R-70-CHAIN-BROKEN) make the W23-1 / W23-3 conditionality sharp.
  Live mixtral 8x7b probe (n=4) shows visible-tokens savings
  +2.84 % delta, +11.37 % super-token, with chain verification
  rate 0.750 and super-token verification rate 1.000; the
  synthetic mitigation does not strictly transfer at this n
  (newly named conjecture **W23-C-MITIGATION-LIVE-VARIANCE**).
  Mac 2 unreachable (17th milestone in a row); the
  `CrossHostProducerDecoderProxy` validates the wire-encoding
  contract via JSON-canonical round-trip on every cell (≈ 1.79
  KB/cell on R-70-DELTA-FANOUT n=8), but no true two-host
  execution validated. (R-70, SDK v3.24)

- **W28 family:** ensemble-verified cross-model multi-chain pivot
  ratification — `EnsemblePivotRatificationEnvelope` (signature_cid
  + probe_votes + quorum + ratified flag), 11 enumerated failure
  modes in `verify_ensemble_pivot_ratification`, three built-in
  probe types (`DeterministicSignatureProbe`,
  `OracleConsultationProbe`, `LLMSignatureProbe`), the first
  cross-host live LLM evidence in 23 milestones (localhost gemma2:9b
  + 192.168.12.191 qwen2.5:14b; 5592 LAN bytes; 128 cross-host probe
  calls). On R-75-CROSS-HOST-LIVE the trust-weighted ensemble
  ratifies 10/16 cells with trust precision 1.000. The first
  capsule-native synthesis of the explicit-capsule trust line
  (W21 multi-oracle adjudication) and the dense-control line (W27
  multi-chain salience-keyed pool). Six named falsifiers
  (W28-Λ-single-probe / -coordinated-drift / -trust-zero / -spoofed-probe
  / -quorum-tampered / -pool-exhausted-passthrough). Newly named
  conjecture **W28-C-CROSS-HOST-VARIANCE** (live cross-host
  variance reduction on a regime where W27 itself fails) emerges
  from the synthetic R-75 banks all having W27 correctness = 1.000.
  (R-75, SDK v3.29)
- **W29 family:** geometry-partitioned product-manifold dense
  control + audited subspace-basis payload + factoradic Lehmer
  routing index + causal-validity gate + cross-host variance witness
  — `GeometryPartitionedRatificationEnvelope` (partition_id +
  factoradic_route_index + audited orthonormal subspace basis CID +
  causal_validity_signature + cross_host_variance_witness_cid +
  parent W28 ratification CID + declared predecessors), 14
  enumerated failure modes in
  `verify_geometry_partition_ratification` (partition-id forgery,
  basis-orthogonality forgery, factoradic-routing forgery,
  causal-validity forgery, partition-cid forgery), structural
  classifier `classify_partition_id_for_cell` returning one of
  LINEAR / HIERARCHICAL / CYCLIC keyed by signature_history,
  per-partition inner W28 dispatch via
  `inner_per_partition: dict[int, EnsembleVerifiedMultiChainOrchestrator]`.
  The new vocabulary (geometry, Grassmannian-style, factoradic,
  Lehmer, mixed-curvature, causal-validity) is added at the
  **capsule layer as audited proxy** — explicitly NOT a transformer-
  internal subspace projection, NOT a Riemannian curvature, NOT a
  learned manifold. On **R-76-XHOST-DRIFT** the W29 method strictly
  improves correctness over BOTH W27 and W28 baselines:
  ``correctness_w27 = correctness_w28 = 0.500`` and
  ``correctness_w29 = 0.750``, **Δ = +0.250 across 5/5 seeds**,
  ``trust_precision = 1.000``, ``mean overhead = 0.75 tokens/cell``.
  Same +0.250 gain holds on the live two-host LLM topology
  (localhost gemma2:9b + 192.168.12.191 qwen2.5:14b); 16 cross-host
  probe calls; 710 LAN bytes; trust precision 1.000. **Empirically
  discharges W28-C-CROSS-HOST-VARIANCE on the magnitude axis** —
  the first capsule-native multi-agent-coordination method to
  demonstrate the synthesis (W21 × W27, sealed by W28, geometry-
  partitioned by W29) strictly improves correctness on a regime
  where the prior best (W28) makes correctness mistakes. Five
  named falsifiers (W29-Λ-trivial-partition, W29-Λ-non-orthogonal-
  basis, W29-Λ-coordinated-drift-cross-host, plus W28-Λ-quorum-
  tampered and W28-Λ-pool-exhausted-passthrough inherited).
  Newly named conjectures **W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE**
  (live LLMs systematically disagree),
  **W29-C-CRAM-AMPLIFICATION** (richer payload that hits 8×;
  measured 2.30× on R-76-CHAIN-SHARED — H7 missed),
  **W29-C-PARTITION-CALIBRATION** (calibrated per-partition trust
  priors), and **W29-C-NATIVE-LATENT** (architecture-dependent —
  true transformer-internal subspace projection vs the W29 audited
  proxy; retained as the next true wall). 38/38 W29 unit tests +
  935/935 wider regression pass. Mac 2 (192.168.12.248) still
  ARP-incomplete (24th milestone). (R-76, SDK v3.30)
- **W30 family:** calibrated geometry-partitioned dense control +
  multi-stride basis-history + per-partition calibration prior +
  cross-host disagreement-routing + ancestor-chain causal binding —
  ``CalibratedGeometryRatificationEnvelope`` (BasisHistory CID +
  PartitionCalibrationVector + AncestorChain CID + W29 parent
  partition CID + disagreement_route metadata + calibrated_cid),
  14 disjoint enumerated failure modes in
  ``verify_calibrated_geometry_ratification`` (basis-history-CID
  forgery, calibration-vector-out-of-range, ancestor-chain-CID
  forgery, disagreement-route-target forgery, calibrated-CID
  forgery), an optional ``partition_classifier_hook`` on the W29
  ``GeometryPartitionedOrchestrator`` that the W30
  ``CalibratedGeometryOrchestrator`` uses to inject two
  closed-form audited overrides — a per-partition calibration
  prior (deterministic running mean over observed agreement,
  **NOT** a learned model: ``(prev * n + obs) / (n + 1)``) and a
  cross-host disagreement-routing override (peeks at the inner
  W28's last-result ``cross_host_disagreement_count`` from the
  *previous* cell). The new vocabulary (calibration prior,
  disagreement-routing, basis-history, ancestor-chain) is added
  at the **capsule layer as audited proxy** — explicitly NOT a
  learned partition classifier, NOT transformer-internal
  calibration, NOT a true Bayesian posterior. On
  **R-77-CHAIN-CRAM** at ``stride = 28``, ``ancestor_window =
  12`` the W30 envelope packs ``cram_w30/w28 = 8.74×`` AND
  ``cram_w30/w29 = 2.74×`` more bits of audited structured
  control per visible token, **across 5/5 seeds**.
  **Empirically discharges W29-C-CRAM-AMPLIFICATION** (8.74× ≥
  the pre-committed 8.0× bar that W29 missed at 2.30×). On
  **R-77-CALIBRATION-PRIOR** the calibration override yields
  ``Δ(W30 − W29) = +0.250 correctness gain`` across 5/5 seeds
  (**discharges W29-C-PARTITION-CALIBRATION on the closed-form-
  prior axis**). On **R-77-XHOST-DISAGREE** the disagreement-
  routing override yields ``Δ(W30 − W29) = +0.250 correctness
  gain`` across 5/5 seeds (**sharpens
  W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE on the
  capsule-layer-routing axis**). ``trust_precision = 1.000`` on
  every ratifying sub-bank; tamper rejection 60/60 across five
  named modes. Three named falsifiers
  (W30-Λ-trivial-calibration, W30-Λ-non-calibratable,
  W30-Λ-degenerate-history) all empirically confirmed. Newly
  named open conjectures **W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE**
  (regime where live LLMs systematically disagree at temp 0),
  **W30-C-NATIVE-LATENT** (architecture-dependent — true
  transformer-internal subspace projection vs the W30 audited
  proxy; retained as the next true wall),
  **W30-C-MULTI-HOST** (3+ host topology, blocked on Mac 2 ARP),
  **W30-C-PRIOR-LEARNING** (true learned per-partition prior
  vs the W30 deterministic running mean — out of scope as a
  capsule-layer mechanism). 36/36 W30 unit tests + 357/357
  focused regression pass. Mac 2 (192.168.12.248) still ARP-
  incomplete (25th milestone). (R-77, SDK v3.31)
- **W31 family:** online self-calibrated geometry-aware dense
  control + sealed prior trajectory + adaptive threshold + W31
  manifest CID + first measured live cross-architecture LLM
  disagreement at temperature 0 —
  ``OnlineCalibratedRatificationEnvelope`` (``PriorTrajectoryEntry``
  tuple sealed by SHA-256 of canonical bytes carrying
  (cell_idx, partition_id, observed_agreement, prior_after);
  threshold trajectory sealed by SHA-256; ``manifest_cid`` over
  six component CIDs; outer ``w31_cid`` over the canonical
  envelope bytes), 14 disjoint enumerated failure modes in
  ``verify_online_calibrated_ratification`` (cumulative 42 across
  W29 + W30 + W31; the load-bearing addition is the
  ``registered_prior_trajectory_cid`` cross-cell check that
  catches a swap whose components are individually self-consistent
  AND whose manifest CID has been recomputed self-consistently).
  The W31 ``OnlineCalibratedOrchestrator`` adds an
  ``online_enabled`` closed-loop running-mean update on every cell
  (driven by the deterministic per-cell agreement signal
  ``ratified AND no cross-host disagreement`` ⇒ 1.0; else 0.0)
  fed into the existing
  ``update_partition_calibration_running_mean`` primitive shipped
  in W30, an ``adaptive_threshold`` clipped-median update on the
  live calibration vector (bounded to ``[0.20, 0.80]`` via
  registered constants), and a sealed trajectory truncated to
  ``trajectory_window``. The new vocabulary (online running-mean
  / adaptive threshold / sealed trajectory / manifest CID) is
  added at the **capsule layer as audited proxy** — **NOT** a
  learned model in the deep-learning sense (zero parameters,
  zero gradients, zero training step), **NOT** transformer-
  internal subspace projection, **NOT** a runtime KV transplant.
  On **R-78-NONSTATIONARY-PRIOR** (inverted partition layout)
  the W30 baseline with uniform priors gets
  ``correctness_ratified_rate_w30 = 0.750`` (no reroute fires);
  W31 with online learning observes per-cell failures on CYCLIC,
  drops the prior to 0.6667, and the clipped-median adaptive
  threshold (0.8) fires the W30 reroute on subsequent CYCLIC
  cells — strictly improving correctness to 0.875,
  ``Δ(W31 − W30) = +0.125`` across 5/5 seeds at trust precision
  1.000. **Empirically discharges W30-C-PRIOR-LEARNING.** On
  **R-78-ADAPTIVE-THRESHOLD vs R-78-FROZEN-THRESHOLD** the
  adaptive contribution isolates at +0.125 across 5/5 seeds
  (frozen yields 0). On **R-78-MANIFEST-TAMPER** the manifest
  CID + cross-cell prior_trajectory_cid check together yield
  **65/65 = 1.000 tamper rejection rate** across five named
  tampers per cell-position. Three named falsifiers all
  empirically confirmed (W31-Λ-trivial-online ⇒ byte-for-W30
  passthrough; W31-Λ-no-drift ⇒ no help on stationary regime;
  W31-Λ-frozen-threshold ⇒ no adaptive contribution at fixed
  0.5 threshold). **First measured live cross-architecture LLM
  disagreement at temperature 0 in the programme** (28th
  milestone): gemma2:9b on localhost vs qwen2.5:14b on
  192.168.12.191 disagree on **2/8 = 0.250 of structured-
  decision prompts at temp 0**, reproducible byte-for-byte
  across two runs. **Sharpens W30-C-CROSS-HOST-VARIANCE-LIVE-
  MAGNITUDE-LIVE on the infrastructure-discharge axis.** Newly
  named open conjectures **W31-C-CROSS-HOST-VARIANCE-LIVE-
  MAGNITUDE-LIVE** (gold-correlation axis),
  **W31-C-NATIVE-LATENT** (architecture-dependent — true
  transformer-internal subspace projection vs the W31 audited
  proxy; retained as the next true wall),
  **W31-C-MULTI-HOST** (3+ host topology, hardware-bounded),
  **W31-C-LONG-WINDOW-CONVERGENCE** (longer trajectory windows).
  41/41 W31 unit tests + 437/437 phase69-78 regression + 68/68
  wider coordpy suite pass. Mac 2 (192.168.12.248) still ARP-
  incomplete (26th milestone). (R-78, SDK v3.32)
- **W32 family:** long-window convergent online geometry-aware
  dense control + EWMA-tracked per-partition prior + Page two-
  sided CUSUM change-point detector + GoldCorrelationMap +
  manifest-v2 CID. The first capsule-native multi-agent-
  coordination method that demonstrates *long-window scaling
  stability* of the W31 online prior on a regime where the W31
  baseline has no convergence guarantee. The W32
  ``LongWindowConvergentOrchestrator`` wraps the W31 orchestrator
  and adds three closed-form primitives: an EWMA running update
  ``ewma_new = (1−α)·ewma_prev + α·observation`` written *back*
  into the W30 calibration vector (load-bearing routing change,
  not just an audit trail); a Page two-sided CUSUM detector
  ``cusum_pos / cusum_neg`` bounded by ``cusum_max`` with named
  change-point firings; and a registered
  ``GoldCorrelationMap : (partition_id → score)`` table. The
  manifest-v2 CID seals all of the above into a single SHA-256
  over four component CIDs (``w31_online_cid``,
  ``convergence_state_cid``, ``gold_correlation_cid``,
  ``route_audit_cid_v2``); the verifier
  :func:`verify_long_window_convergent_ratification` enumerates
  14 failure modes disjoint from W29's 14 + W30's 14 + W31's 14
  (cumulative **56 across W29+W30+W31+W32**). On
  R-79-LONG-WINDOW (4 windows × 5 seeds, all 0.969 byte-equal
  to W31 baseline), W32 is byte-for-W31 stable across long
  windows — *long-window scaling stability* is empirically
  established. **Empirically discharges
  W31-C-LONG-WINDOW-CONVERGENCE on the scaling-stability axis.**
  On R-79-MANIFEST-V2-TAMPER, 1525/1525 = 1.000 cross-component
  manifest-v2 + cross-cell route-audit-v2 tamper rejection
  across five named tampers per cell-position. A pre-committed
  hard gate H6 (strict gain ≥ +0.10 over W31 on a long-window
  regime) is **honestly null**: the **W32-L-CYCLE-CAP**
  limitation theorem proves that on cycle-capped dispatcher
  regimes ``Δ_max ≤ min(c_p/4, c_s)/N ≤ 0.0625``, so the bar
  *cannot* be cleared on the available bench by any
  EWMA-on-prior method. The remaining gap is converted to the
  open conjecture **W32-C-LONG-WINDOW-STRICT-GAIN** (requires a
  regime exceeding the cycle cap). Four named falsifiers
  (W32-Λ-trivial-long-window ⇒ byte-for-W31 passthrough;
  W32-Λ-no-change-point ⇒ stable-history regime never fires
  CUSUM; W32-Λ-frozen-ewma at ``α = 1.0`` *outperforms* W31 by
  +0.016 on the available regime, an honest empirical
  correction over the predicted-null falsifier;
  W32-Λ-mis-correlated-gold ⇒ gate-bounded, never opens on
  synthetic banks). Live cross-architecture LLM gold-verifiable
  pilot (gemma2:9b on localhost vs qwen2.5:14b on
  192.168.12.191, temp 0, 20 prompts, byte-reproducible across
  two runs): **19/20 = 0.950 agreement**, sole disagreement
  D5 (TCP handshake) has neither host correct against gold —
  the first measured live cross-architecture LLM gold-verifiable
  agreement at temp 0 in the programme. **Sharpens
  W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE on the
  prompt-class-dependent disagreement frontier.** Newly named
  open conjectures **W32-C-LONG-WINDOW-STRICT-GAIN**,
  **W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE**,
  **W32-C-NATIVE-LATENT** (architecture-dependent — true
  transformer-internal subspace projection vs the W32 audited
  proxy; out of capsule-layer scope), **W32-C-MULTI-HOST**
  (3+ host topology, hardware-bounded), and
  **W32-C-OLD-LINE-EWMA-TRUST** (W21 EWMA-tracked-trust
  integration; primitives ship in W32 but the W21 integration
  is not yet built). 45/45 W32 unit tests + 414/414 phase69-79
  regression + 77/77 wider coordpy suite = 536 tests pass. Mac 2
  (192.168.12.248) still ARP-incomplete (27th milestone).
  (R-79, SDK v3.33)

The post-W21 efficiency-and-coordination ladder (W22 → W31)
discharges, in order, a different family of open conjectures —
one per layer — concerning amortisation of the working
capsule-native pipeline across cells, agents, salience
signatures, host topologies, geometric partitions, and prior
calibration. The W32 layer is the first one of the post-W21
ladder that adds *no new structurally-distinct routing move*:
W32 reuses the W31 routing surface and proves *long-window
scaling stability* of W31's online prior under EWMA + CUSUM
+ gold-correlation, sealed by a manifest-v2 CID. Strict gain
on the same regime is honestly null; the remaining wall is
the **W32-L-CYCLE-CAP** limitation theorem. The natural
escapes from W32-L-CYCLE-CAP are *named*
(W32-C-LONG-WINDOW-STRICT-GAIN on regimes exceeding the cycle
cap; W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE on regimes where
LLMs disagree on gold-verifiable prompts) but currently
*conjectural*; the W32 release deliberately does not claim
strict gain where the regime cannot support it.

## Appendix C. Submission Pass Still Needed

This Markdown manuscript is now substantially beyond a skeleton, but a
submission-quality camera-ready pass would still need production work
of the normal kind:

1. a BibTeX-backed bibliography and in-text citation layer;
2. a figure set with final captions and cross-references;
3. compact venue-shaped tables distilled from the full benchmark
   artifacts;
4. theorem statements rewritten into the exact style of the target
   venue;
5. a reproducibility appendix listing exact scripts, flags, seeds, and
   artifact paths;
6. a pruning pass that moves some implementation detail from the main
   body into appendices.

Those are paper-production tasks, not missing scientific content.

## Appendix D. Recommended figure and table set

The current manuscript would benefit most from the following figures
and tables:

1. **Figure 1: capsule-native runtime overview.**
   Show the path from setup to sealed artifacts to the LLM byte
   boundary.
2. **Figure 2: structural-axis ladder.**
   Show the eight axes and the benchmark that isolates each one.
3. **Figure 3: benchmark ladder timeline.**
   Present R-53 through R-64 as a cumulative sequence of ceilings and
   crossings.
4. **Figure 4: W14-W15-W16-W17 composition diagram.**
   Show producer protocol, normalization, admission, decoding, packing,
   and the bounded decoder input bundle.
5. **Figure 5: symmetric wall.**
   Show why gold and decoy become indistinguishable under the current
   feature surface.
6. **Table 1: regime summary.**
   Keep the ladder summary from Section 7.
7. **Table 2: strongest result by layer.**
   Keep the synthesis table from Section 10.
8. **Table 3: live real-LLM composition results.**
   Present the v3.18 qwen2.5 and qwen3.5 comparisons compactly.

## Post-W49 — Cross-Backend Latent Coordination (W50)

W50 layers **five orthogonal capsule-native advances** on top of
W49 Multi-Block Cross-Bank Coordination:

(M1) a trainable **cross-backend latent projector** that maps the
W49 ``SharedLatentCapsule`` chain between two backend behaviors
through a shared lingua-franca code. A best-effort real-LLM
realism anchor runs when ``COORDPY_W50_OLLAMA_REACHABLE=1``;
otherwise the witness records ``anchor_status:
"synthetic_only"``. This **bounds**, not closes, the cross-
tokenizer conjecture. (M2) a deeper proxy transformer stack at
``L=4`` (vs W49's ``L_p=2``) with per-layer learned mask gates +
per-layer outer residual scales + a forward witness binding
per-block activation norms. (M3) an adaptive K=16 prototype
codebook (vs W49's K=8) + a learned per-bit emit-mask gate;
the packed ``LATENT_CTRL_V3`` block achieves
``≥ 8.0 structured bits per visible-token`` under W50 vs W49's
5.0 baseline (1.86× improvement). (M4) a role-pair-conditioned
``CrossBankTransferLayer`` that moves slot keys/values between
role banks via a learned linear projection, paired with
``AdaptiveEvictionPolicyV2`` — a 5-feature sigmoid scorer
(adding retention probability and transfer signal to W49's
3 features). (M5) a chain-walkable ``SharedLatentCarrierV2``
with a per-role reuse map and a trainable
``ReconstructionV2Head`` that recovers turn ``t-k`` flat
features for ``k ≤ 3`` at MSE ``≤ 0.25`` (random-prediction
baseline = 0.333).

The W50 envelope chain
``w47_outer → w48_proxy_outer → w49_multi_block_outer →
w50_outer`` is verified by 20 disjoint failure modes
(cumulative W22..W50 = **343 enumerated modes**). R-98
(10 cell families, 3 seeds) and R-99 (7 cell families,
3 seeds) verify the H1..H16 success criterion: trivial
passthrough preserved byte-for-byte, cross-backend synthetic
alignment fidelity ≥ 0.95, deep stack L=4 strictly beats L=2,
role-pair transfer recall delta ≥ +0.15, adaptive eviction V2
beats FIFO at signal preservation, 8-turn retention cosine ≥
0.90, 12-turn retention cosine ≥ 0.70, reconstruction MSE
≤ 0.25 at k=3, 8 bits/visible-token cramming ratio,
distribution-cap falsifier reproduces, replay determinism.

W50 does NOT touch transformer-internal state, KV bytes,
attention weights, embeddings, or real tokenizers. It is the
strongest *executable proxy* line we can write today at the
capsule layer with one **best-effort** real-LLM realism
anchor. Released SDK contract preserved byte-for-byte;
``coordpy.__version__`` remains ``0.5.20``.

## W51 Persistent Cross-Backend Latent Coordination (post-W50, 2026-05-11)

The next post-release research milestone, **W51 PXBLC**, adds
**six orthogonal capsule-native advances** on top of W50:
(M1) a trainable GRU-style **persistent shared latent state
V3** with the update rule ``s_t = (1 - z_t) ⊙ s_{t-1} + z_t ⊙
tanh(W_h · [s_{t-1}; x_t])`` and a content-addressed
``PersistentLatentStateChain``, plus a learned cross-role
mixer producing per-role views of the team state;
(M2) a **triple-backend translator** over three backend tags
``(A, B, C)`` with direct translators ``A→B``, ``A→C``,
``B→C`` plus a trainable **transitivity loss** that penalises
disagreement between ``A→C`` (direct) and ``A→B→C``
(composed); (M3) a depth-six (``L=6``) deep proxy stack V2
with **branch-specialised heads**, **cycle-specialised
heads**, and per-layer trainable temperature; (M4) a
**hierarchical adaptive compression V3** with a coarse
``K1=32`` codebook + per-cluster fine ``K2=16`` sub-codebooks
plus a degradation-curve probe — achieves **≥ 12 bits per
visible-token at full emit** (vs W50's 8-10.5 bits/token);
(M5) a **two-headed long-horizon reconstruction V3** (causal
+ branch) at ``max_k=8`` (vs W50's ``max_k=3``); (M6) a
**branch/cycle-specialised memory head** with separate
per-branch and per-cycle storage pages plus learned
cross-branch consensus + cross-cycle merger.

The W51 envelope chain ``w47_outer → w48_proxy_outer →
w49_multi_block_outer → w50_outer → w51_outer`` is verified
by 24 disjoint failure modes (cumulative W22..W51 = **367
enumerated modes**). R-100 (11 cell families, 3 seeds) and
R-101 (8 cell families, 3 seeds) verify the **H1..H18**
success criterion — 18/18 H bars pass: trivial passthrough
preserved byte-for-byte, persistent state long-horizon
recall +0.95 over W50 untrained baseline, triple-backend
direct fidelity 0.887 with transitivity gap 0.087, deep
stack V2 structural floor met with branch-specialised heads
gain +0.056, branch/cycle memory recall 0.993 vs generic
0.785, hierarchical compression 13 bits/visible-token,
12-turn cosine retention 0.707, 16-turn stretch 0.796,
reconstruction V3 MSE at k=5 0.409 and at k=8 0.462,
replay determinism 1.000, verifier 1.000.

**Honest non-claims at W51**: the ``L=6`` deep stack V2 does
**not** strictly improve over ``L=4`` under pure-Python
autograd on the synthetic regime — the structural floor +
non-regression H4 bar is met (acc ≥ 0.65 AND Δ ≥ -0.05) and
the actual M3 behavioural win comes from branch/cycle-
specialised heads (H5). The
``W51-L-DEEP-STACK-OVERDEPTH-CAP`` falsifier reproduces
honestly on shallow regimes (H18). The H7 triple-backend
Ollama anchor records ``anchor_status: "synthetic_only"``
when the env flag is unset; the
``W50-C-CROSS-TOKENIZER-LATENT-TRANSFER`` conjecture carries
forward sharpened as
``W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY``.

W51 does NOT touch transformer-internal state, KV bytes,
attention weights, embeddings, or real tokenizers. It is the
strongest *executable proxy* line we can write today at the
capsule layer with one **best-effort** real-LLM triple-backend
realism anchor. Released SDK contract preserved
byte-for-byte; ``coordpy.__version__`` remains ``0.5.20``.

## W52 Quantised Persistent Multi-Hop Latent Coordination (post-W51, 2026-05-11)

The next post-release research milestone, **W52 QPMHLC**, adds
**eight orthogonal capsule-native advances** on top of W51:

1. **M1 stacked persistent latent V4** with identity-init
   skip-link that carries the turn-0 signal through
   mid-sequence distractors;
2. **M2 multi-hop quad-backend translator** with
   chain-length-3 transitivity loss and
   disagreement-weighted arbitration calibrated from
   training residuals;
3. **M3 `L=8` deep proxy stack V3** with role-conditioned
   KV banks and per-layer residual gate;
4. **M4 three-level quantised codebook V4** (`K1=32 × K2=16
   × K3=8`) with adaptive budget allocation, achieving
   ≥ 14 bits per visible token at full emit;
5. **M5 three-headed reconstruction V4** (causal + branch +
   cycle) at `max_k=12`;
6. **M6 branch/cycle memory V2** with trainable merge +
   importance-weighted evict + joint `(branch, cycle)`
   pages and a content-addressed merge audit trail;
7. **M7 role-graph conditioned cross-role transfer** —
   a new module with per-edge `(src, dst)` linear
   projections that strictly beats equal-weight cross-role
   averaging on direction-dependent regimes;
8. **M8 transcript-vs-shared-state matched-budget
   comparator** — explicit ablation between transcript
   truncation and shared-latent quantised encoding under a
   fixed visible-token budget; reports the strict retention
   gap and bit-density gap.

The W52 envelope chain ``w47_outer → w48_proxy_outer →
w49_multi_block_outer → w50_outer → w51_outer → w52_outer``
is verified by 26 disjoint failure modes (cumulative W22..W52
= **393 enumerated modes**). R-102 (12 cell families) and
R-103 (10 cell families) at 3 seeds each verify H1..H22 —
**22/22 H bars pass**.

**Honest non-claims at W52**: the `L=8` deep stack V3 does
NOT strictly improve over `L=6` V3 on shallow regimes (H21:
-0.056). The multi-hop translator under identity-friendly
init preserves ~0.55 of the clean signal even when trained
on forged labels — protect rate caps at 0.43 (H11). The
quantised codebook cannot reach 32 bits/visible-token
(structural ceiling ~12 bits/triple — H22 falsifier
reproduces). The
``W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY`` conjecture
carries forward sharpened as
``W52-C-CROSS-TOKENIZER-QUAD-TRANSITIVITY``: capsule-layer
length-3 transitivity is now trained and auditable; only
tokenizer-level transitivity remains carried forward.

W52 does NOT touch transformer-internal state, KV bytes,
attention weights, embeddings, or real tokenizers. It is the
strongest *executable proxy* line we can write today at the
capsule layer with one **best-effort** real-LLM quad-backend
realism anchor. Released SDK contract preserved
byte-for-byte; ``coordpy.__version__`` remains ``0.5.20``.

## W53 Persistent Mergeable Corruption-Robust Latent Operating System (post-W52, 2026-05-12)

The W53 milestone composes ten orthogonal mechanism advances
on top of W52 into a **capsule-native latent operating system
proxy** for multi-agent teams.

* **M1 Persistent Latent State V5** — 3-layer GRU with
  *persistent* skip-link applied at every step + a state-
  merge head; chain walks past 32 turns
* **M2 Multi-Hop Translator V3** — 5-backend (A,B,C,D,E)
  with chain-length-4 transitivity + uncertainty-aware
  arbitration that returns per-dim 1-sigma confidence
* **M3 Mergeable Latent State Capsule (MLSC)** — content-
  addressed mergeable capsules with explicit
  ``MergeOperator`` + content-addressed ``MergeAuditTrail``;
  K-of-N consensus quorum with abstain semantics. *The
  load-bearing new abstraction at W53.*
* **M4 Deep Proxy Stack V4** — depth-10 wrapping V3 with
  merge-aware + corruption-aware heads
* **M5 ECC Codebook V5** — K1×K2×K3×K4 = 16384 codes plus
  XOR parity bits per segment (4 parity bits/visible-token);
  ≥ 14.5 bits/visible-token target (empirically 17.67) +
  single-bit corruption detection
* **M6 Long-Horizon Reconstruction V5** — 4 heads (causal +
  branch + cycle + merged-branch), ``max_k=16``, degradation
  curve to ``k=32``
* **M7 Branch Merge Memory V3** — consensus pages with
  K-of-N quorum + content-addressed audit + abstain
* **M8 Corruption-Robust Carrier** — composes ECC parity +
  3-of-3 majority repetition; reports detect / partial-correct
  / abstain / silent-failure rates; honest 2-bit
  graceful-degrade
* **M9 Transcript-vs-Shared Arbiter V2** — explicit per-turn
  policy over {transcript, shared, abstain}; oracle-
  correctness comparison
* **M10 Uncertainty / Confidence Layer** — composite
  per-component confidence + calibration check
  (high-conf strictly more accurate than low-conf)

W53 ships at ``coordpy.persistent_latent_v5``,
``coordpy.multi_hop_translator_v3``,
``coordpy.mergeable_latent_capsule``,
``coordpy.deep_proxy_stack_v4``,
``coordpy.ecc_codebook_v5``,
``coordpy.long_horizon_retention_v5``,
``coordpy.branch_merge_memory_v3``,
``coordpy.corruption_robust_carrier``,
``coordpy.transcript_vs_shared_arbiter_v2``,
``coordpy.uncertainty_layer``,
``coordpy.w53_team`` — explicit-import paths only.
``coordpy.__version__`` remains ``0.5.20``; SDK contract
byte-for-byte unchanged.

W53 is verified by R-104 (12 cell families) + R-105
(10 cell families) + R-106 (12 cell families) at 3 seeds
each (H1..H34 success criterion). The ``W53-T-MLSC-CAPSULE-
CID-STABILITY``, ``W53-T-ECC-PARITY-DETECT-SINGLE-BIT``,
and ``W53-T-MLSC-CONSENSUS-QUORUM-K-OF-N`` theorems are
proved by inspection plus empirical witnesses; the
``W53-L-OVERDEPTH-V4-CAP``, ``W53-L-ECC-RATE-FLOOR-CAP``,
``W53-L-CRC-TWO-BIT-PATHOLOGY-CAP`` falsifiers reproduce
honestly. The ``W53-C-CROSS-TOKENIZER-QUINT-CAP``
conjecture sharpens W52's cross-tokenizer line.

W53 does NOT touch real KV bytes, hidden states, attention
weights, embeddings, or real tokenizers. It is the strongest
*executable proxy* line we can write today at the capsule
layer with one **best-effort** real-LLM quint-backend
realism anchor (reusing the W52 quad scaffold + a synthetic
5th tag E). Released SDK contract preserved byte-for-byte;
``coordpy.__version__`` remains ``0.5.20``.

### Deep Mergeable Disagreement-aware Latent Operating System
(DMD-LOS) — post-W53 W54 milestone

W54 lifts the W53 PMCRLOS line into a deeper, more
disagreement-aware, single-bit-correcting,
abstain-with-fallback capsule-layer operating system by
composing **ten orthogonal mechanism advances** on top of W53:

* (M1) a **4-layer persistent latent state V6** with *dual
  persistent skip-link* (turn-0 anchor + running EMA carrier)
  applied at every step, chain walks past 64 turns, and a
  disagreement-tagged state-merge head that emits per-dim
  disagreement alongside the merged state;
* (M2) a **6-backend (A,B,C,D,E,F) multi-hop translator V4**
  over 30 directed edges with chain-length-5 transitivity
  and *disagreement-aware compromise arbitration* (largest
  pairwise-agreeing subset; abstains when no agreement);
* (M3) the **Mergeable Latent State Capsule V2 (MLSC V2)**
  — extends W53's MLSC with per-dim *disagreement metadata*
  on every merge, a *provenance fact graph* DAG, and per-
  parent *trust signatures* that scale merge weights
  independently of confidence;
* (M4) a **Consensus / Quorum Controller** — first-class K-of-
  N controller with explicit abstain-with-fallback policy
  (quorum_merged | fallback_best_parent | abstain) plus a
  content-addressed K-of-N audit trail;
* (M5) a **Corruption-Robust Carrier V2** — composes
  Hamming(7,4) single-bit *correction* per segment with
  W53's XOR parity detection + a 3-of-5 majority repetition;
  silent failure tightened to ≤ 0.05 (vs W53's ≤ 0.10);
* (M6) a **depth-12 Deep Proxy Stack V5** with
  *disagreement-aware head*, *uncertainty-projected residual
  gating*, and *abstain short-circuit* when corruption
  confidence falls below threshold;
* (M7) a **5-head Long-Horizon Reconstruction V6** (causal +
  branch + cycle + merged-branch + *cross-role*) at max_k=24
  (vs V5's 16) with per-dim *degradation scores*;
* (M8) a **5-level ECC Codebook V6** (K1=32 × K2=16 × K3=8 ×
  K4=4 × K5=2 = 32768 codes) plus Hamming(7,4) on every
  segment for single-bit correction; achieves 18.0
  bits/visible-token at full emit (vs W53's 17.67;
  ≥ 16.0 target);
* (M9) a **4-arm Transcript-vs-Shared Arbiter V3** over
  {transcript, shared, merge_consensus, abstain-with-
  transcript-fallback} with per-arm budget allocation;
* (M10) an **Uncertainty Layer V2** that adds *per-component
  noise injection*, *calibration-under-noise* check,
  per-decision *rationale tags*, and a *disagreement-
  weighted composite* that down-weights components reporting
  high disagreement.

The orchestrating ``W54Team`` produces a sealed
``W54HandoffEnvelope`` whose ``w54_outer_cid`` binds: the W53
outer CID, the W54 params CID, every per-turn W54 witness
bundle CID, the persistent V6 chain CID, the MLSC V2 audit
trail CID, and the consensus controller audit CID. The W54
envelope chain runs end-to-end:
``w47_outer → w48_proxy_outer → w49_multi_block_outer →
w50_outer → w51_outer → w52_outer → w53_outer → w54_outer``.

Trivial W54 collapses to W53 byte-for-byte (the
``W54-L-TRIVIAL-W54-PASSTHROUGH`` falsifier). Verifier
soundness: 30 new disjoint enumerated failure modes;
cumulative trust boundary W22..W54 = **453 failure modes**.

W54 ships at ``coordpy.persistent_latent_v6``,
``coordpy.multi_hop_translator_v4``,
``coordpy.mergeable_latent_capsule_v2``,
``coordpy.consensus_quorum_controller``,
``coordpy.corruption_robust_carrier_v2``,
``coordpy.deep_proxy_stack_v5``,
``coordpy.long_horizon_retention_v6``,
``coordpy.ecc_codebook_v6``,
``coordpy.transcript_vs_shared_arbiter_v3``,
``coordpy.uncertainty_layer_v2``, and ``coordpy.w54_team`` —
explicit-import paths only. ``coordpy.__version__`` remains
``0.5.20``; SDK contract byte-for-byte unchanged.

W54 is verified by R-107 (12 cell families) + R-108 (10 cell
families) + R-109 (14 cell families) at 3 seeds each
(H1..H36 success criterion). The
``W54-T-HAMMING-7-4-SINGLE-BIT-CORRECTION``,
``W54-T-MLSC-V2-TRUST-WEIGHTS-MONOTONICITY``,
``W54-T-CONSENSUS-CONTROLLER-FALLBACK-SOUNDNESS``,
``W54-T-CRC-V2-SILENT-FAILURE-FLOOR``, and
``W54-T-DEEP-V5-DISAGREEMENT-HEAD-SOUNDNESS`` theorems are
proved by inspection plus empirical witnesses. The
``W54-L-OVERDEPTH-V5-CAP``, ``W54-L-ECC-V6-RATE-FLOOR-CAP``,
``W54-L-V6-OUTER-NOT-TRAINED-CAP``, and
``W54-L-COMPROMISE-NOT-STRICT-DOMINANCE`` limitation
theorems reproduce honestly. The
``W54-C-CROSS-TOKENIZER-HEX-CAP`` conjecture sharpens W53's
cross-tokenizer line to 6 backends.

W54 does NOT touch real KV bytes, hidden states, attention
weights, embeddings, or real tokenizers. It is the strongest
*executable proxy* line we can write today at the capsule
layer with one best-effort real-LLM hex-backend realism
anchor (reusing the W53 quint scaffold + a synthetic 6th
tag F). Released SDK contract preserved byte-for-byte;
``coordpy.__version__`` remains ``0.5.20``.

## W55..W62 Substrate-Attack Programme (post-W54, 2026-05-12 → 2026-05-15)

The W55..W62 programme is a seven-milestone substrate-attack
that turns the capsule-layer operating system from W43..W54 into
a substrate-coupled latent operating system. Each milestone adds
in-repo substrate axes, closed-form ridge fits, and a
bidirectional hybrid loop. W56 cracked the in-repo substrate
open with a 5-layer NumPy transformer. W57 added hidden-state
injection, prefix-state reuse, and attention steering. W58 made
cache reuse load-bearing. W59 introduced single-α ridge
corrections. W60 expanded to multi-direction multi-target ridge
fits + a first-class ReplayController. W61 added a content-
addressable cache-key axis, a bilinear retrieval head, a trained
replay-threshold head, a 4-D attention-budget tensor, a multi-
target stacked HSB fit, an attention-pattern-target KV fit, and
a six-way bidirectional loop.

W62 is the **seventh substrate-attack milestone**. It adds a
**per-(layer, head, slot) cache-write ledger axis** inside the
in-repo V7 NumPy substrate (9 layers), a **per-layer logit-lens
probe**, a **per-(layer, head, position) attention-receive
delta** channel, a **per-(layer, head) replay-trust ledger**, a
**trained per-regime replay head** (replay controller V3 with 4
regime heads + nearest-centroid regime gate), a **trained
hidden-vs-KV regime classifier** (5×3 ridge with ≥ 0.8 training
accuracy on synthetic supervision), a **trained corruption-
repair head** (cache controller V5 with 4-dim ridge), a
**two-objective stacked ridge fit** (drop-oracle + retrieval-
relevance simultaneously), a **drift-curve predictor** (prefix V6
with stacked 4×K ridge), a **two-stage attention clamp**
(attention V6 with coarse L1-mass + fine per-(L,H,Q,K) KL), and
a **seven-way bidirectional substrate ↔ V7 ↔ cache controller V5
↔ replay controller V3 ↔ retrieval head ↔ attention-steering V6
↔ hidden-vs-KV classifier** hybrid loop (deep substrate hybrid
V7).

Across W56..W62 the released SDK contract is preserved byte-for-
byte; ``coordpy.__version__`` remains ``0.5.20``; no PyPI
release. The seven-milestone envelope chain is verified live:
``W55 envelope CID == W56.w55_outer_cid``,
``W56 envelope CID == W57.w56_outer_cid``, …,
``W61 envelope CID == W62.w61_outer_cid``.

W62 fits twelve closed-form linear ridge solves (seven from W61
+ five new):

1. KV bridge V6 multi-target — (n_directions × m) matrix.
2. KV bridge V6 attention-pattern — (n_directions × 1) against
   last-row attention L1 target.
3. HSB V5 multi-target stack — worst-residual column reduction.
4. Cache controller V4 bilinear retrieval — (d_model × d_key) M
   matrix.
5. Cache controller V4 trained corruption floor — 3-dim quadratic.
6. Replay controller V2 threshold head — 6×4 against label one-hot.
7. LHR V13 third-layer scorer — random+ReLU → random+tanh →
   ridge.

W62 adds:

8. Cache controller V5 two-objective stacked — n_features × 2
   matrix.
9. Cache controller V5 trained-repair — 4-dim vector against
   repair amount target.
10. Cache controller V5 composite_v5 — 6-vector mixture against
    drop oracle.
11. Replay controller V3 per-regime head — 6×4 per regime, 4
    regimes total.
12. Replay controller V3 hidden-vs-KV classifier — 5×3 against
    3-class label.

R-131 (13 cell families) + R-132 (12 cell families) + R-133 (20
cell families) at 3 seeds verify H163..H180 — 45 of 45 H-bars
pass 3/3 seeds (135/135 cells, strong success per the W62
success criterion). Cumulative trust boundary across W22..W62 =
**844 enumerated failure modes**.

W62's honest scope is unchanged. The in-repo V7 substrate is
``W62-L-NUMPY-CPU-V7-SUBSTRATE-CAP`` (9 layers / d_model=64 /
byte-vocab / max_len=128 / untrained NumPy on CPU, NOT a
frontier model). Hosted backends remain text-only at the HTTP
surface (``W62-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``). The
12-ridge-solve closed-form fit is ``W62-L-V7-NO-AUTOGRAD-CAP``
(no SGD, no autograd, no GPU). The persistent V14 outer GRU
remains untrained (``W62-L-V14-OUTER-NOT-TRAINED-CAP``). The
multi-hop V12 backends remain NAMED, not EXECUTED
(``W62-L-MULTI-HOP-V12-SYNTHETIC-BACKENDS-CAP``). The trained
corruption-repair head outputs an *additive* correction; it does
NOT un-corrupt the raw cached state
(``W62-L-CONSENSUS-V8-REPAIR-STAGE-SYNTHETIC-CAP``). The
hidden-vs-KV classifier achieves ≥ 0.8 training accuracy on
**synthetic** supervision; it does NOT itself prove hidden-state
injection beats KV injection on real models or real workloads.

W62 is the strongest honest substrate-coupling milestone the
programme has shipped to date.

## References

Bibliography intentionally omitted from this Markdown draft.
Replace this section in the final submission version with a real
BibTeX-backed bibliography after the venue-targeting pass.
