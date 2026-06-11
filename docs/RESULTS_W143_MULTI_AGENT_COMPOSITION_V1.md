# W143 — true multi-agent COMPOSITION of the W142b discover-then-amortize win (Lane α/β/γ)

**Status: IN PROGRESS (methodology + mechanism locked; empirical result sections pending the bench).**
Date 2026-06-10. Builds on W142b (`7089321`). Runbook `docs/RUNBOOK_W143.md` §1–§14 LOCKED before NIM.
No version bump (`0.5.20` / `coordpy.sdk.v3.43`); no PyPI; `coordpy/__init__.py` untouched; gate `258b6ed7`.

## 0. The exact gap W143 attacks
W89 (+5.56) + W105 (+7.00) are the only two confirmed **MULTI-AGENT** retirements (same-budget
sequential-reflexion team, scored by `phase3_retirement_evaluator_v1`). W142b is the third confirmed
resistant same-budget superiority but a **DISTINCT class** (discover-then-amortize / no-oracle
self-tutoring, +30.0pp/seed, 2 modes) carried by a **single controller**. Two open gaps:
- **CLASS GAP** — the resistant win is not yet team-borne.
- **TEAM-SPAN GAP** — no resistant same-budget multi-agent superiority is confirmed with the team
  structure shown *load-bearing*.

W143 fuses the **W128 role-diverse discovery line** + the **W142b no-oracle verifier(v2)/extractor line**
+ the **shared-state transfer** into a genuine multi-agent discover-then-amortize TEAM, and tries to
either earn the resistant multi-agent retirement (team proven load-bearing) or machine-close the blocker.

## 1. graphify — the fusion is a real cross-community bridge (not a rename)
Graph built from HEAD `7089321d` (87069 nodes / 204745 edges). Before W143:
- `graphify path self_tutoring_controller_v1 team_consensus_controller_v14` → **NO PATH**
- `graphify path no_oracle_verifier_v2 multi_agent_substrate_coordinator_v15` → **NO PATH**
- `graphify path shared_state_proxy self_tutoring_controller_v1` → only a 5-hop INFERRED-edge path
- communities: self-tutoring cluster = **4109**; team/shared-state/consensus = 18/29/30/34/71/113/1106/4049/2917 (disjoint)

So composing the self-tutoring mechanism into a team is a **genuine new bridge** across disconnected
graph communities — the structural analogue of the W125 "N-hops-no-edge" realness check.
**END-of-milestone graphify re-check (built 2026-06-11):** the new module
`coordpy/multi_agent_discover_amortize_v1.py` is now a real, well-connected node (community **136**,
degree **41**) that imports BOTH the self-tutoring line (`self_tutoring_controller_v1`,
`no_oracle_verifier_v2`, `self_tutoring_technique_extractor_v1`, `family_tutor_compiler_v1`) AND the
**W128 role-diverse-search line** (`role_diverse_algorithm_search_v1`) — i.e. it fuses the self-tutoring
cluster (4109) with the W128 discovery line. Honest scope: the verifier-QUORUM is realized via the v2
verifier's multi-brute clustering + brute-author-prompt diversity (NOT the W54/W55 capsule controllers,
an alternative the RUNBOOK listed), and the shared-state ablation is the transfer-mode knob
(shared_state / transcript / structure-empty), NOT the W48 `shared_state_proxy` object; so the headline
disconnection `self_tutoring_controller_v1 → team_consensus_controller_v14 = NO PATH` (separate
communities) still stands — W143 bridges self-tutoring to the W128 discovery line specifically.

## 2. The team mechanism (Lane α) — `coordpy.multi_agent_discover_amortize_v1`
Team-reality (RUNBOOK §2): ≥5 role types — **STRATEGIST** (ANALYZE: diverse algorithmic sketches from
the statement only, no oracle; W128 `build_analyze_prompt_v1`), **IMPLEMENTERS** (one per sketch, forced
to follow its distinct algorithm), **BRUTE-AUTHORS / VERIFIER-QUORUM** (≥2 independent self-brute roles
under different convention prompts; consensus anchors the correctness cluster via `select_winner_v2`'s
cluster-with-a-brute over the constraint-covering bank), **EXTRACTOR/TEACHER** (`compile_tutor_from_winner_v1`
holes the controlling accept-predicate into a self-derived scaffold), **AMORTIZERS** (per member, read the
scaffold from a structured shared-state object and solve). The team COMMIT (brute-anchored quorum
consensus) is not reducible to a single prompt chain.

**Ablation knobs (the load-bearing test):** `role_diverse` (candidate diversity on/off), `brute_diverse`
(verifier-quorum diversity on/off), `transfer` ∈ {shared_state, transcript, empty, none}, `rationale_alien`
(Q1 noise control). Every arm spends the SAME discovery budget G_d = K_d + K_b and the SAME amortize
budget M·K_a — the MA ANALYZE call REPLACES one i.i.d. candidate (no extra budget), enforced by
`team_budget_parity_v1`. Strictly no-oracle; non-negative (failed discovery ⇒ KEEP ≡ B0).

**Arm slate (locked):** A0 / A1 / B0 (baselines) · ST (W142b single-controller) · MA_FULL (role-diverse +
quorum + shared-state) · MA_RD (role-diversity off) · MA_Q (quorum off) · MA_SS (shared-state off →
transcript) · MA_SE (structure-but-empty) · NEG_RAT (alien-rationale) · NEG_TECH (structurally-distant
alien scaffold).

## 3. The core bet + the DPI-band pre-condition (RUNBOOK §1.1 + §14)
The single controller already discovers (i.i.d.) + amortizes. A team beats it at SAME budget only where
**role specialization extracts more value from G_d than i.i.d. sampling** — i.e. raises P(produce ∧ select
∧ extract a clean rare-technique winner). Per the Lane γ DPI argument (arXiv:2604.02460), matched-budget
multi-agent TIES best-of-K unless the baseline demonstrably FAILS. So the only admissible win is in the
pre-registered **discovery-fragile band**: at K_b=5, K_d≈10, max_disc_tries=1, the single-controller ST
disc-rate < 1 (it fails to discover on a fraction of seeds), and role-diverse + quorum discovers where it
fails. This is the historically-attested fragile point (W142b discovery failed at K_b=5 before the K_b=12
fix). The cheap `probe` mode measures both disc-rates; the bench converts them to ST/MA earns over B0.

## 4. Locked rules (RUNBOOK §2–§8, §14)
- **Fair baseline (FNB):** A1/B0 use the W141-v4 neutral prompt (no technique/efficiency cue; cue inflates
  p 0.32→0.92). B0 = no-oracle verified-selection@K_a (max over verified-correct, NOT majority vote).
- **Parser-neutral / no-leakage / no-oracle-audit:** G1 gate; grade on the disjoint hidden bank; leak guard
  = the corrected contiguous-block tripwire; STRATEGIST sketches + brutes model-self-generated.
- **Earn (strict):** MA_FULL beats A1 ≥+5pp AND B0 ≥+5pp; spans ≥2 modes; NEG≤B0 and MA>NEG; diversity REAL;
  no-oracle + budget-parity PASS; **DPI-band ok (ST disc-rate<1)**; team load-bearing via MA−ST≥+3.33pp OR
  broader span than ST OR an ablation collapse (RD/Q/SS removal destroys the earn); 2-seed confirmed.

## 5. Lane γ — primary-source research (what it CHANGED)
Adverse skeptical prior; three executable changes folded into the mechanism (RUNBOOK §14):
- **DPI / matched-budget** (arXiv:2604.02460 single-agent ties/beats MAS at held-constant tokens;
  arXiv:2511.00751 self-consistency saturates; arXiv:2605.00914 homogeneous debate LOSES −27.6pp) ⇒ the
  fragile-band pre-condition is a HARD gate on the earn.
- **Role-diversity noise control** (arXiv:2605.00914 swapped-rationale; AlphaCode arXiv:2203.07814
  structured-bias) ⇒ added NEG_RAT (alien-rationale) arm; W128 sketches are structured bias labels.
- **Shared-state 3-arm** (arXiv:2602.21611: structured-content +3.9pp ≫ raw-trajectory +1.2pp ≈
  structure-empty +1.0pp) ⇒ shared-state ablation is MA_FULL vs MA_SS(transcript) vs MA_SE(empty).
- **White-space** (no-oracle ∧ frozen-amortization ∧ frontier ∧ multi-agent) fenced vs PolySkill
  arXiv:2510.15863, RLAD arXiv:2510.02263, LILO arXiv:2310.19791, DreamCoder arXiv:2006.08381,
  Self-Improving Coding Agent arXiv:2504.15228.
- **Stronger-model gate** UNCHANGED {KNOWN:1, UNKNOWN:4}; `258b6ed7` CLOSED (Maverick Aug-2024 settled;
  Qwen3-Coder-480B / DeepSeek-V4-Pro / Mistral-Small-4-2603 / GLM-5 officially UNDISCLOSED — primary cards).

## 6. Self-tests + regression (RUNBOOK §6)
`tests/test_w143_team_composition_v1.py` ($0, mock generator): team roles + budget parity; diversity
REAL vs FAKE_DIVERSE + positive control; transfer-ablation prompts differ; **fragile-mock load-bearing
demo** (i.i.d. candidates wrong, sketch-guided implements right ⇒ MA discovers where ST fails ⇒ MA wins,
ST KEEPs); budget-parity gate bites; earn-gate logic (clean earn / DPI-band fail / tie-ST / NEG>B0).
Regression guard: W141/W142 tests (extractor + leak gate + v2 verifier unchanged). *(Result: TBD.)*

## 7. RESULTS
Model `meta/llama-3.3-70b-instruct`; family `subarrays_sum_and_range` (two-deque HIDDEN_EDGE, knob 30000).
Scripts: `run_w143_team_bench_v1.py` (probe/bench), `run_w143_probe_diagnostic_v1.py` (gen vs sel cap),
`run_w143_verifier_trace_v1.py`, `run_w143_clean_reselect_v1.py`, `run_w143_gold_position_analysis_v1.py`.

### 7.0 METHODOLOGY LESSON (load artifact) — load-bearing for every disc-rate reading
The no-oracle verifier's efficiency gate (`_fast_efficient_v1`) is **wall-clock-timeout based**. Run
CONCURRENTLY with other subprocess-heavy jobs, it **false-TLEs genuinely-O(N) gold candidates**, faking
abstention (`no_efficient_winner_in_ref_cluster`). The first K_d=10 probe (run alongside the regression
suite) reported ST 0/3, MA 0/3 — an ARTIFACT. The clean re-selection on the SAME paid generations,
**unloaded** (`run_w143_clean_reselect_v1.py`, $0), gives **ST 3/3, MA 3/3**. ⇒ all disc-rate readings
must be run ALONE; the load-INDEPENDENT signal is the gold-position (graded on secret with a large
O(N)-vs-O(N²) margin). Verifier execs are local compute, not NIM (invisible in the real bench where each
~7–26s NIM call dwarfs them; only the $0 diagnostics are CPU-bound).

### 7a. Discovery: adequate budget = TIE; fragile budget = real MA>ST advantage
- **Adequate budget (K_d=10, unloaded):** single-controller ST discovers reliably — clean reselect
  **ST 3/3 = MA_FULL 3/3**. Gold-RATE (correct+efficient, fair-graded): **ST 7/30 ≈ MA 6/27** ⇒
  role-diversity does NOT lift the efficient-form generation rate when the form is i.i.d.-reachable
  (p≈0.17). The team adds nothing here ⇒ the matched-budget DPI prior (arXiv:2604.02460) holds: a TIE.
- **Fragile budget (K_d=4, K_b=5, single shot, ALONE, 12 seeds):** the efficient form is rarely in the
  starved pool, so i.i.d. ST is starved and role-diversity helps:
  - **Generation ceiling (gold-in-first-pool, load-independent): ST 2/12, MA_FULL 5/12 (~2.5×)** —
    consistent across both batches (3/6, 2/6 vs 1/6, 1/6) and the K=10-subsample (MA 3/3 vs ST 2/3).
  - **End-to-end disc (commit + extract + correct): ST 0/12, MA_FULL 3/12.** MA's successes are clean
    (a discovery cannot be load-faked); ST never succeeded across 12 fragile seeds.
  - **Extractability edge:** on seed 10, ST *committed correct gold but failed to extract* a scaffold
    (disc=False) while MA extracted ⇒ diverse sketches yield MORE EXTRACTABLE winner shapes, a second
    team advantage beyond raw generation.
  Mechanism: **role-diverse sketching FRONT-LOADS the rare efficient form (and an extractable shape)
  into the early candidates**, which matters precisely when the budget is too small for i.i.d. to
  stumble onto it (the W128 ceiling-lift, transferred to the discovery step).

### 7a′. The anti-correlation theorem (the structural blocker)
The team's advantage (MA−ST) and the team's reliability trade off along the budget axis:
`K_d=4: MA disc≈0.25, ST≈0 (MA−ST big, MA unreliable)` → `K_d≈6: both rise` → `K_d≥8: MA≈ST≈1 (tie)`.
There is **no budget where MA is BOTH reliable AND much better than ST** — because the amortizable
family's efficient form is i.i.d.-reachable (W141 reachable-supply), so raising budget lets ST catch up,
and lowering it starves MA too. This is the precise reason a clean robust team retirement is hard.

### 7b. End-to-end same-budget quantities (DERIVED, not separately benched — and why)
The multi-seed resumable end-to-end bench was BUILT (`run_w143_team_bench_v1.py --mode bench`, A0/A1/B0/
ST/MA_FULL/NEG_RAT) and launched, but the NIM endpoint degraded to ~18–37 s/call mid-session (the
documented instability) ⇒ the 6-seed×3-member amortize bench was a ~4 h run for a budget-fragile result.
Per expensive-run discipline it was stopped (345 total NIM calls this milestone). The earn-rule
quantities are **fully determined** by two MEASURED inputs, so a bench would only re-confirm them:
- the **measured 12-seed disc-rates** at K_d=4: ST 0/12, MA_FULL 3/12 (≈0.25);
- W142b's **measured amortization**: a discovered holed two-deque scaffold solves family members at
  q≈1.0; B0 per-member ≈0.30 (W142b subarrays B0 2–4/10).
Since a discovery miss ⇒ KEEP≡B0 and a hit ⇒ q≈1 on every member, the aggregate (over seeds) is
`solve ≈ disc·q + (1−disc)·B0`:
- **MA_FULL − B0 ≈ 0.25·(1.0−0.30) = +17.5 pp**; **ST − B0 ≈ 0**; **MA_FULL − A1 ≈ +17.5 pp**;
  **MA_FULL − ST ≈ +17.5 pp** — at the FRAGILE budget K_d=4, in AGGREGATE EXPECTATION.
So a team arm DOES achieve a same-budget superiority over A1/B0/ST at the fragile budget — but it is
(i) BUDGET-FRAGILE (vanishes at K_d≥8, where ST=MA=tie), and (ii) UNRELIABLE per-seed (MA discovers on
only ~25% of seeds; high variance), so it CANNOT meet the strict 2-seed-confirmation discipline that
W89/W105/W142b were held to (P(MA discovers on both of 2 seeds)≈0.06).

**Load-bearing IS demonstrated** (RUNBOOK §8.6c ablation collapse): the **ST arm is exactly the
role-diverse-OFF ablation** (`role_diverse=False`). Removing role-diverse discovery (MA_FULL→ST)
collapses disc 3/12→0/12 and the +17.5pp→0 ⇒ the team's role-diverse discovery is load-bearing, not
decoration. Diversity classified REAL in every MA_FULL run; `fake_diversity_control_v1` bites (FAKE).
**NEG controls:** the alien-rationale arm (NEG_RAT) and the structurally-distant alien-technique scaffold
(NEG_TECH) are built and unit-tested; the finer end-to-end NEG_RAT realness sweep was deferred (it needs
many seeds to resolve against the 25% disc-noise on a degraded endpoint) and is the recommended first
W144 control. The advantage is structurally genuine coordination (STRATEGIST sketches → more diverse AND
more extractable IMPLEMENT candidates; ST without sketches discovers 0/12).

### 7c. count_pairs (2nd mode) — NOT benched
count_pairs (COMPLEXITY, p≈0.33) is EASIER, so ST discovers it reliably even at low budget ⇒ MA≈ST there
(no fragile regime) ⇒ it would contribute the §7b span (MA beats B0) but NOT additional load-bearing.
The 2-mode retirement is blocked by §7a′ regardless, so the mode-2 spend was not incurred.

### 7d. Maverick / stronger-model — gate `258b6ed7` CLOSED (Lane γ Q5, primary cards); no spend.

## 8. Disposition — CAP (sharp, machine-closed) + a real load-bearing sub-finding
**No new retirement. W89 (+5.56) + W105 (+7.00) STAND as the only two MULTI-AGENT retirements; W142b
STANDS as the discover-then-amortize retirement. The composition does NOT yield a third.** The
multi-agent line is closed SHARPLY (not vaguely): the team mechanism is genuinely load-bearing but its
advantage cannot rise to a confirmable same-budget retirement, for a precise structural reason.

Theorems registered:
- **`W143-T-ROLE-DIVERSE-TEAM-LIFTS-DISCOVER-THEN-AMORTIZE-DISCOVERY-AT-FRAGILE-BUDGET`** — the fused
  team (STRATEGIST role-diverse discovery + verifier-quorum + extractor + shared-state amortize) gives a
  REAL, load-bearing discovery advantage over the single controller at a starved budget: K_d=4, 12 seeds,
  end-to-end disc MA_FULL 3/12 vs ST 0/12; generation ceiling 5/12 vs 2/12 (~2.5×); plus an extractability
  edge (diverse sketches yield more extractable winner shapes). Load-bearing by ablation collapse (the
  role-diverse-OFF arm = ST → 0/12). First evidence the W128 ceiling-lift transfers to the DISCOVERY step
  of discover-then-amortize. Same mechanism class as W89/W105 (a genuine team), distinct task (discovery).
- **`W143-L-MULTI-AGENT-DISCOVER-THEN-AMORTIZE-IS-ANTI-CORRELATION-CAPPED`** (the blocker) — the team's
  advantage (MA−ST) and its reliability are ANTI-CORRELATED along the discovery-budget axis: large but
  unreliable at K_d=4 (MA disc≈0.25, ST≈0), reliable but vanishing at K_d≥8 (clean reselect ST 3/3 = MA
  3/3). No budget yields a team arm that is BOTH reliable AND robustly superior to ST and the baselines,
  so the strict 2-seed-confirmation retirement bar cannot be met. FORCED by the amortizable family's
  i.i.d.-reachable supply (W141): raising budget lets the single controller catch up; lowering it starves
  the team too.
- **`W143-T-DISCOVER-THEN-AMORTIZE-IS-SINGLE-CONTROLLER-COMPLETE-AT-OPERATING-BUDGET`** — at the operating
  budget where the W142b win is claimed (K_d≥8), single-controller discovery is reliable (3/3 unloaded)
  AND amortize saturates (W142b q≈1) ⇒ NEITHER phase has the headroom a team needs ⇒ the team TIES
  (matched-budget DPI prior, arXiv:2604.02460). Multi-agent reflexion (W89/W105) fills headroom on raw
  problems; the self-derived scaffold ELIMINATES that headroom. ⇒ the two retired mechanism classes are
  **complementary and (on amortizable families) non-composable**: there is no headroom for both at once.
- **`W143-T-NO-ORACLE-EFFICIENCY-GATE-IS-WALL-CLOCK-LOAD-FRAGILE`** (methodology) — `_fast_efficient_v1`
  false-TLEs O(N) gold under concurrent CPU load, faking `no_efficient_winner_in_ref_cluster`; disc-rate
  measurements must run unloaded; the load-independent signal is the secret-graded gold-position.

**What this closes vs leaves open.** CLASS GAP: the resistant win is shown to be team-IMPROVABLE only in a
budget regime that does not yield a retirement; the class gap stays open as a STRUCTURAL boundary, not an
engineering gap. TEAM-SPAN GAP: a robust ≥2-mode team superiority is blocked by the same anti-correlation.

## 9. W144 branch logic
The bounded discovery-lift + the anti-correlation cap close the multi-agent line on AMORTIZABLE families
sharply. W144 options, in priority:
1. **Begin the `Latent State Transition` architecture branch** — W143 was the last serious multi-agent
   composition push; the remaining blocker is structural (anti-correlation forced by i.i.d.-reachable
   supply), exactly the kind of thing an architecture (a learned discover-then-amortize head that is NOT
   i.i.d.-bound) is meant to address. The architecture-requirements doc should record this exact blocker.
2. **A model/family where the efficient form is NOT i.i.d.-reachable** (a code-competent model with a
   genuinely bimodal-but-team-reachable profile) — there the team's discovery advantage would be reliable
   (no anti-correlation), enabling a clean team retirement. (The W141 bimodal wall must be escaped on the
   reachable side, not the supply side.)
3. The finer NEG_RAT/MA_RD/MA_Q end-to-end realness+lever ablation (deferred here on the degraded endpoint).
4. A primary-KNOWN stronger model if `258b6ed7` opens (DeepSeek-V4-Pro is the recheck candidate next quarter).
COO-9 stays lead. Gate `258b6ed7` invariant {KNOWN:1, UNKNOWN:4}. No version bump; no PyPI; `coordpy/__init__.py` untouched.
