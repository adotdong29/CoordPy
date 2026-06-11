# RUNBOOK W143 — true multi-agent COMPOSITION of the W142b discover-then-amortize win + resistant same-budget TEAM-retirement attempt (only if the team honestly earns it)

**Status: LOCK §1–§13 before any NIM spend. `ultracode` OFF. Work on `main`. COO-9 lead. No version bump (`0.5.20` / `coordpy.sdk.v3.43`); no PyPI; `coordpy/__init__.py` untouched; advanced work explicit-import only. Gate `258b6ed7` invariant.**

Built from HEAD `7089321d` (graphify confirms graph built-from == HEAD; 87069 nodes / 204745 edges).

---

## 0. One-line thesis

W142b earned the FIRST confirmed §7b **discover-then-amortize / no-oracle self-tutoring** resistant same-budget superiority (count_pairs COMPLEXITY + subarrays HIDDEN_EDGE, +30.0pp/seed, 2 modes, NEG no-lift) — but with a **single controller**, NOT a team. W89 (+5.56) + W105 (+7.00) remain the only two confirmed **MULTI-AGENT** retirements. The open programme gap is therefore exact:

> **CLASS GAP** — the resistant win exists but not yet as a genuinely team-borne mechanism.
> **TEAM-SPAN GAP** — no resistant same-budget multi-agent superiority has been confirmed on a locked resistant family/mode set with the team structure shown to be *load-bearing*.

W143 fuses the **W128 role-diverse discovery line** + the **W142b no-oracle verifier(v2)/extractor line** + the **W54/W55 quorum-consensus controllers** + the **W48 shared-state proxy** into a genuine multi-agent **discover-then-amortize TEAM**, and tries to either (a) earn the resistant multi-agent retirement with the team structure proven load-bearing, or (b) machine-close the exact blocker. graphify confirms the two lines are **disconnected** (self_tutoring in community 4109; team/shared-state/consensus in communities 18/29/30/34/71/113/1106/4049/2917; `graphify path self_tutoring_controller_v1 team_consensus_controller_v14` = NO PATH; `… no_oracle_verifier_v2 multi_agent_substrate_coordinator_v15` = NO PATH) ⇒ the fusion is a **real cross-community bridge**, not a rename.

W143 is NOT: another architecture note; another plain-reflexion rerun on a dead resistant field; another single-controller self-tutoring lap; another prompt-only "team" that is fake-different.

---

## 1. The mechanism question, stated precisely (the spine)

W142b's single-controller ST already does discovery (K_d i.i.d. draws → no-oracle-verify → extract → scaffold) and amortization (K_a scaffolded draws/member). **What can a TEAM add at SAME total budget that the single controller cannot?**

A team of distinct roles splitting a fixed budget G can only beat N i.i.d. draws if **role specialization extracts more value from G than i.i.d. sampling** — i.e. if the team raises the probability of producing *and selecting* a clean, extractable, verified winner of the rare technique, at the same call budget. This is exactly the W128 result restated for the discovery step: **role-diverse search LIFTS the generation ceiling** (diverse roles reach programs i.i.d. misses) but is **SELECTION-capped** (the no-oracle selector commits fewer than the ceiling).

**W143's load-bearing claim (the bet):** at a **DISCOVERY-FRAGILE budget** — a budget low enough that i.i.d. single-controller discovery (ST) frequently *abstains / false-negatives / fails to extract* (disc=False ⇒ ST KEEPs ≡ B0 ⇒ no win) — a **role-diverse + verifier-quorum** team discovers a clean extractable verified winner reliably, converts it to the W142b amortization win over B0, and the team structure is *load-bearing*: removing role-diversity (→ i.i.d. = ST), the quorum (→ single brute), or the shared-state transfer (→ transcript-only) each collapses the earn.

This is honest and historically motivated: W142b's own discovery was fragile at K_b=5 (seed1 subarrays drew 0 correct two-constraint brutes → FP → KEEP → no win) and only became reliable at **K_b=12 + multi-winner extraction** — i.e. the single controller bought reliability with *more i.i.d. budget*. W143 asks whether **diversity + quorum buys the same reliability at the same (lean) budget** — and is load-bearing exactly where i.i.d. is fragile.

**Same-budget accounting is the heart of the claim** (see §8). The team and ST get the SAME total discovery budget `G_d` (= candidate calls + brute calls) and the SAME amortize budget `M·K_a`. The team even pays a planning tax (1 ANALYZE call out of its discovery budget) — so a team win is a win *despite fewer raw candidate draws*. The only differences are ALLOCATION (diverse sketches vs i.i.d.), SELECTION (quorum vs single cluster), and TRANSFER (shared-state vs none).

### 1.1 Why this is non-vacuous and falsifiable
- If at the fragile budget MA ties ST (both discover, or both fail) ⇒ the team adds nothing ⇒ register a cap (§13). Iterate-on-fail (§12) first: lower the budget further or test a harder family, root-causing whether the bottleneck is generation (diversity doesn't reach the winner) or selection (quorum can't pick it) — the W128 distinction.
- If MA beats ST only because it spent MORE budget ⇒ budget-accounting gate (§8) FAILS the run. Disallowed.
- If MA's lift survives a fake-different relabel (NEG) ⇒ it was decoration ⇒ killed.

---

## 2. The team-reality rule (LOCKED before results)

An arm counts as **genuinely multi-agent** only if ALL of:
1. **≥3 explicit roles with distinct responsibilities.** W143's team has ≥5 role TYPES: **STRATEGIST** (ANALYZE: propose diverse efficient algorithmic sketches from the statement only), **IMPLEMENTERS** (one per sketch, each forced to follow its assigned distinct algorithm), **BRUTE-AUTHORS / VERIFIER QUORUM** (≥2 independent roles, each writing a self-brute under a *different* convention prompt), **EXTRACTOR/TEACHER** (compile the holed-skeleton scaffold from the verified winner), **AMORTIZERS** (per member, read the scaffold from shared-state and solve).
2. **Explicit artifact or shared-state transfer across roles.** The transfer chain is: sketches (STRATEGIST) → candidate programs (IMPLEMENTERS) → verified winner (VERIFIER QUORUM consensus) → holed-skeleton `FamilyTutorV1` (EXTRACTOR) → **structured shared-state object** → scaffolded solutions (AMORTIZERS). The scaffold crosses the discover→amortize boundary as a structured artifact, not raw text.
3. **A team decision/commit step not reducible to a single prompt chain.** The **verifier-quorum consensus** over ≥2 independent brute-authors (cluster-with-a-brute via `select_winner_v2`, or `ConsensusQuorumController.decide` / `TrustWeightedConsensusController.decide`) is a genuine cross-role aggregation: the committed winner is the one anchored by a quorum of independent verifiers, not one chain's pick.
4. **An ablation path that can show the team structure is load-bearing** (§7 ablation arms).

**It is NOT multi-agent** if: it is a single prompt chain with relabeled "roles" (NEG); it is the W142b single controller wrapped in role names with no real bias/selection/transfer difference; more model calls alone with no role decomposition; a self-tutoring win with no team ablation showing load-bearing value. (HOW_NOT_TO_OVERSTATE: a teacher-compiled tutor and a weak-model lift are NOT multi-agent frontier superiority.)

### 2.1 Brainstorm (done BEFORE locking the slate)

**≥10 candidate team compositions (enumerated; kept ones marked ✓, killed ones marked ✗ + reason):**
1. ✓ **Role-diverse discovery + verifier-quorum + shared-state amortize** (the LEAD; MA-FULL). Real bias diversity, real quorum, real structured transfer.
2. ✓ **Verifier-quorum-only team** (i.i.d. candidates, but ≥2 independent brute-author roles + consensus selection) — MA2 / the quorum-isolating arm.
3. ✓ **Role-diverse-discovery-only team** (diverse sketches, single brute) — MA1 / the diversity-isolating arm.
4. ✓ **Shared-state amortize team** (scaffold transferred as structured holed-skeleton object) vs ✗ transcript-only — the shared-state ablation pair (MA-SS vs MA-TR).
5. ✗ **Sequential reflexion team relabeled as discover/amortize** — this is the W89/W105 mechanism, not discover-then-amortize; reusing it here would be the OLD retirement, not a composition. Kept ONLY as conceptual reference; not an arm.
6. ✗ **Debate team (two solvers argue, judge picks)** — the judge has no oracle; reduces to a verifier with extra calls (no added discovery supply). Killed as budget-wasteful.
7. ✗ **Critic-refiner pair** — collapses to a single chain (critic → refiner is reflexion); fails team-reality #3.
8. ✗ **N-identical-persona "team"** — fake-different by construction; becomes the NEG control, not a real arm.
9. ✗ **Trust-weighted consensus over agents with hand-assigned trust** — trust must be *earned* (matches-predicted-expected, W128 RDA4 axis), not assigned, else decorative. Folded into the quorum arm's trust axis, not a separate arm.
10. ✓ **Quorum-of-quorums / multi-winner extraction across the quorum's verified set** — the W142b multi-winner loop made a *team* asset: every role's verified winner is an extraction candidate. Folded into the LEAD.
11. ✗ **Manager/worker hierarchical team** — the manager is an extra prompt layer with no oracle; adds calls without supply. Killed.
12. ✗ **Tool-augmented single agent relabeled team** — fails team-reality #1 (one role). Killed (W125 lesson: a controller is not a team).

**≥8 ways a "team" can secretly collapse to a single-controller mechanism (each gets a guard):**
1. Roles share one prompt with cosmetic name tags → guard: **fake-diversity detector** (`compute_diversity_v1` REAL iff sketches jaccard-distinct, ≥2 distinct impls, counterexamples-new) + the NEG relabel control MUST classify FAKE_DIVERSE/FAKE_DIFFERENT.
2. The "quorum" is one brute copied N times → guard: brute-authors use distinct convention prompts; the quorum's *added* value is measured by the quorum-OFF ablation (single brute).
3. Role-diversity that doesn't change the candidate distribution → guard: measure pool-distinct-algorithm-signatures; require the diversity report REAL.
4. The scaffold is identical regardless of which role discovered it → guard: that's fine for the scaffold, but the *discovery rate* (disc=True frequency) must differ between MA and ST at the fragile budget, else no team value.
5. The team only wins by spending more calls → guard: **budget-parity gate** (`discover_amortize_accounting_v1` + a new team-budget identity) FAILS the run if MA's total > ST's total.
6. The extractor/teacher is the only thing doing work (so it's W142b single-controller extraction) → guard: the load-bearing test is at the DISCOVERY step (does diversity+quorum discover where i.i.d. fails), not the extraction step which is shared by ST and MA.
7. Shared-state is just the transcript renamed → guard: §2.2 shared-state-realness rules + the transcript-only ablation.
8. The "team commit" is actually argmax of one role's confidence → guard: the commit is the brute-anchored cluster consensus; the quorum-OFF ablation isolates it.
9. Diversity is real but selection always picks the same i.i.d.-reachable candidate → guard: this is the W128 SELECTION cap; if it bites, register it as the exact blocker (not a win).

**≥6 ways a shared-state claim can be fake or decorative (each gets a guard):**
1. Shared-state carries the same bytes as the transcript → guard: shared-state = the *compiled AST-derived holed skeleton* (`FamilyTutorV1.to_prompt_block()`), structurally distinct from raw generation text; the transcript-only arm gets the raw winning generation text instead.
2. Amortizers ignore the shared-state and re-derive from the statement → guard: measure scaffolded rate q vs plain rate p per member; if q≈p the scaffold isn't used (W140 "additive iff A1≈0" gate — only families where plain rate is low qualify).
3. Shared-state passed but never read (write-only) → guard: the amortizer prompt embeds `scaffold.to_prompt_block()`; the transcript-only ablation removes exactly that.
4. The "object" is decorative metadata, not the load-bearing content → guard: the load-bearing content is the holed-skeleton hole structure; the leak gate (`tutor_leak_gate_v1`) proves it teaches the technique without leaking the answer.
5. Shared-state value is really just "more context tokens" → guard: the transcript-only arm gets MORE or equal tokens (raw transcript ≥ compiled skeleton); if shared-state still wins, it's the STRUCTURE not the token count.
6. Shared-state only helps because it leaks a hidden-test-passing snippet → guard: no-oracle audit + leak gate + NEG (scaffold from a verifier-REJECTED sample must not lift).

---

## 3. Fair-baseline and family-screen rule (LOCKED before spend)

**Fair neutral baseline (FNB), byte-stable, inherited from W141/W142 §2.** The baseline arms A1/B0 use the exact W141-v4 neutral prompt (`self_tutoring_controller_v1._efficient_prompt`): names **NO technique, NO efficiency/time-limit/large-input/data-structure cue, NO scaffold**. (W141: naming the technique inflates raw `p` 0.32→0.92; even "largest input/time limit" inflates to 0.92 — a cue hands B0 the discovery free and MASKS the amortization win.) `p̂` = passes-on-hidden / K_screen (a hidden pass ⇒ correct AND efficient by construction). Where a fresh `p` is measured, report the median over ≥2 truly-neutral phrasings (FormatSpread arXiv:2310.11324) + spread.

**Family-screen rule.** A family is admitted to the main bench iff it is EITHER:
- (a) **already W142b-proven** as a resistant discover-then-amortize family — i.e. `count_pairs_sum_le_t` (sort+two-pointer, COMPLEXITY, p≈0.33) and `subarrays_sum_and_range` (two-deque, HIDDEN_EDGE, p≈0.17), the locked 2-mode set; OR
- (b) passes the W142 $0 admission gates **G1 parser-neutrality** ∧ **G2 exact-oracle-discriminating** ∧ **G3 gated-accumulator extractability** ∧ **G4 novelty**, AND the FNB fair-`p` band **`p̂ ∈ [0.10, 0.50]`** (Wilson-95% excluding 0 and 1), making a team attempt non-vacuous.

The main bench uses the (a) set (the proven 2-mode span) so the team claim is tested on the SAME resistant ground W142b earned on — the cleanest possible class-gap test. A 3rd family via (b) is attempted ONLY if a cheap screen finds one inside p∈(0.1,0.4) (carry-forward from COO-67), and only to widen span, never to rescue a failed 2-mode result.

---

## 4. Parser-neutrality / no-leakage / no-technique-cue rule (LOCKED, code-enforced)

- **G1 parser-neutral I/O:** every family uses `IoShapeV1` + `render_normal_form_v1` via `make_pn_template`; `parser_neutrality_gate_v1` must PASS (strict per-line reader AND read-all-tokens reader recover byte-identical structured data). Regression fixture must BITE (a flattened input FAILS; normal-form PASSES).
- **No-leakage:** model sees ONLY statement + PUBLIC samples. Never model-facing: `ref_source`/`naive_source`/`brute_source`, hidden `secret_cases`. Grading on the disjoint hidden bank (`_passes_secret` is SCORING-ONLY). The leak guard is the **corrected contiguous-block tripwire** `reproduces_accepted_block_v1` (NOT the W127 per-line check that false-positived on boilerplate); the scaffold must clear the falsifiable `tutor_leak_gate_v1`.
- **No-oracle audit:** the mechanism's code path provably never reads the hidden bank / ref / naive / brute answer-key. The STRATEGIST sketches and the brute-authors' brutes are model-self-generated in the audited subprocess. Machine-checkable (reuse W141 `no_oracle_audit_v1` discipline).
- **No-technique-cue for the team's roles:** the STRATEGIST proposes diverse strategies from the STATEMENT ONLY (W128 `build_analyze_prompt_v1`); it is NOT told the target technique. Role-diversity is self-derived. (This preserves no-oracle: the team self-derives BOTH the candidate diversity AND the scaffold; only the FNB baseline must be technique-neutral — the mechanism is allowed to be cleverer than the baseline, that is the whole point.)

---

## 5. The team-mechanism slate (LOCKED before any NIM) + ablation arms

Same hidden grader, pass/fail-only (the W105/W141/W142b discipline). Per family member:
- **A0** — plain single-shot (1 draw).
- **A1** — FNB self-consistency pool@K_a (floor bar).
- **B0** — no-oracle verified-selection@K_a per member (the STRONG bar; must re-discover per problem). `select_winner_v2` over K_a plain candidates + brutes.
- **ST** — **W142b single-controller** discover-then-amortize at the fragile discovery budget: i.i.d. K_d_frag candidates from the FNB prompt + K_b_frag i.i.d. brutes → `select_winner_v2` → `compile_tutor_from_winner_v1` (multi-winner) → amortize K_a scaffolded draws/member; KEEP≡B0 if disc fails.
- **MA-FULL** — the LEAD team: STRATEGIST (1 ANALYZE) + (K_d_frag−1) IMPLEMENTERS across diverse sketches + K_b_frag brutes from ≥2 distinct brute-author roles → **verifier-quorum** consensus winner → EXTRACTOR scaffold → **shared-state** transfer → AMORTIZERS (K_a scaffolded/member). Same total discovery + amortize budget as ST.
- **NEG** — fake-different relabeled team: identical role *labels*, but sketches are duplicated (no real bias diversity) and the scaffold is compiled from a **verifier-REJECTED / structurally-distant alien** sample. MUST classify FAKE_DIVERSE (`compute_diversity_v1`) and MUST NOT lift (NEG ≤ B0). The strict-alien NEG (W142b `sum_nearest_smaller_left` for the counting families) is retained.

**Pre-registered ABLATION arms (the load-bearing test):**
- **MA−RD** (role-diversity OFF) — same quorum + shared-state, but discoverers are i.i.d. from the FNB prompt (no STRATEGIST sketches). Isolates role-diverse discovery. Equivalent to "quorum-only".
- **MA−Q** (quorum OFF) — same role-diverse discovery + shared-state, but a single brute-author (no consensus). Isolates the verifier-quorum.
- **MA−SS** (shared-state OFF / transcript-only) — same role-diverse discovery + quorum, but amortizers receive the winner's RAW generation transcript text instead of the compiled holed-skeleton object. Isolates structured shared-state (the W52 transcript-vs-shared distinction for code).

Dominated/optional arms are dropped (W140 lesson). MA3 (one additional honest team variant) is included ONLY if a specific blocker motivates it during iterate-on-fail.

---

## 6. Self-test + regression-fixture rule (LOCKED, all $0, before any NIM)

A **MOCK generator** (deterministic, emits canned programs — efficient/naive/brute/sketch variants keyed by prompt content) drives every plumbing test at $0:
- **Team plumbing:** MA-FULL runs end-to-end on the mock; ≥3 distinct roles fire; the scaffold crosses discover→amortize via the shared-state object; the verifier-quorum commits a brute-anchored winner.
- **Each ablation BITES on the mock:** MA−RD collapses diversity (diversity report FAKE/low); MA−Q reverts to single-brute selection; MA−SS feeds raw transcript (no holed skeleton in the amortizer prompt).
- **NEG control BITES:** the relabeled/alien team classifies FAKE_DIVERSE and does not lift.
- **Budget-parity gate BITES:** an arm declared at G that spends >G FAILS `team_budget_parity_v1`.
- **Fake-diversity positive control:** `fake_diversity_control_v1()` classifies FAKE_DIVERSE (detector bites).

**Regression fixtures (must pass before the pilot is trusted):**
- (a) the **W142b count_pairs + subarrays earn** re-extracts cleanly (extractor + leak gate unchanged) — run the existing `tests/test_w142_*` / W142b validations as a guard;
- (b) the **W140 tutor cross-tier lift** families still compile;
- (c) the **W125 looks-right-fails-hidden** failure mode still fails the no-oracle verifier (the v2 verifier rejects the sum-only naive on subarrays);
- (d) the **W142b strict-alien NEG** check still shows NEG_strict ≈ B0 ≪ ST.

---

## 7. Budget-accounting rule (LOCKED, pre-registered)

The retirement claim is the **EQUAL-TOTAL-FAMILY-BUDGET** claim. Over `M` same-technique members:
- **B0:** total `M·K_a`; per-member `P_solve = 1−(1−p)^{K_a}`.
- **ST:** total `G_d^ST + M·K_a` where `G_d^ST = K_d_frag` (candidates) `+ K_b_frag` (brutes), one-time discovery; per-member superiority over B0 = `(1−p)^{K_a}` when disc=True, else 0.
- **MA-FULL:** total `G_d^MA + M·K_a` where `G_d^MA = 1` (ANALYZE) `+ (K_d_frag−1)` (implements) `+ K_b_frag` (brutes) = **exactly `G_d^ST`** (the planning call replaces one candidate call). Same `M·K_a` amortize budget.

**`G_d^MA == G_d^ST` is enforced** by a new `team_budget_parity_v1` (extends `discover_amortize_accounting_v1`): it counts model calls per arm and FAILS the run if any arm exceeds its declared `G`. `K_d` is reported SEPARATELY as a one-time family cost with per-member share `K_d/M → 0`; NEVER folded into a per-problem same-budget claim. Verifier executions (brute/complexity runs) are LOCAL compute, not model calls — reported, not counted. No hidden-test calls anywhere.

**The discovery-fragile budget (pre-registered + justified, not cherry-picked):** `K_b_frag = 5`, `K_d_frag ∈ {8,10,12}` (the LEAD probe uses `K_d_frag = 10`, `K_b_frag = 5`). Justification: W142b's discovery was empirically fragile at K_b=5 (seed1 subarrays drew 0 correct two-constraint brutes → no win) and reliable only at K_b=12 + multi-winner. K_b=5 is the historically-attested fragile point. A **$0 pre-check** (replay over the W142b call logs / mock) must confirm ST i.i.d. discovery at the fragile budget has P(disc)<1 on at least the subarrays mode — else the budget is not actually fragile and is lowered. K_a held at **4** (the W142b amortize budget) for all arms.

---

## 8. Multi-agent EARN rule (retirement-grade, STRICT — LOCKED before results)

On the LOCKED resistant family set (count_pairs COMPLEXITY + subarrays HIDDEN_EDGE), single seed then a confirming second seed, the team arm **MA-FULL** earns a resistant multi-agent retirement iff ALL:
1. **MA-FULL beats A1 by ≥ +5 pp** (aggregated, equal total family budget);
2. **MA-FULL beats B0 by ≥ +5 pp** (aggregated, equal total family budget);
3. **spans ≥2 modes or ≥3 families** (the §7b span; the locked set is 2 modes);
4. **NEG ≤ B0** (fake-different/alien team does not lift) AND **MA-FULL > NEG**; `compute_diversity_v1` REAL on MA-FULL, FAKE on NEG;
5. **no_oracle_audit PASS**, leak gate clean, budget-parity gate PASS;
6. **TEAM LOAD-BEARING** demonstrated by ≥1 of:
   - **(6a)** `MA-FULL − ST ≥ +3.33 pp` aggregate, OR
   - **(6b)** MA-FULL spans a broader family/mode set than ST at the same budget (e.g. ST disc=False on a mode where MA-FULL disc=True), OR
   - **(6c)** a pre-registered ablation collapse: removing role-diverse discovery (MA−RD), the quorum (MA−Q), or shared-state (MA−SS) **destroys the earn** (the ablated arm drops below the +5pp bar that MA-FULL clears).
7. **Confirmed on a 2nd seed** (single-seed = demonstration, never a retirement).

A team arm that merely **ties ST everywhere** does NOT close the class gap unless an ablation (6c) proves load-bearing. A one-family win is NOT enough. A biased baseline is NOT enough. A fake-different team is NOT enough. More calls are NOT coordination. If MA beats A1 but not B0 ⇒ selection not coordination ⇒ bounded claim, not a retirement.

**If not earned:** register the EXACT machine-checkable blocker (§13) — e.g. `W143-L-ROLE-DIVERSE-DISCOVERY-LIFTS-CEILING-BUT-QUORUM-SELECTION-CAPPED` (W128 class at the discovery step), or `W143-L-MA-TIES-ST-NO-TEAM-VALUE-AT-FRAGILE-BUDGET`, or `W143-L-SHARED-STATE-DECORATIVE-TRANSCRIPT-TIES`.

---

## 9. Frontier / retirement-target rule (LOCKED)

Default frontier anchor = `meta/llama-3.3-70b-instruct` (the exact W105/W142b retirement model; resistant by construction; primary-KNOWN cutoff ~Dec-2023). Maverick (`meta/llama-4-maverick-17b-128e-instruct`) = OPTIONAL push-button cross-scale check on the same admitted slice, separate from the main claim, only if its deployment is healthy AND the §10 primary-cutoff gate stays as-is (Maverick Aug-2024 KNOWN but SETTLED ⇒ redundant by default). No stronger-model frontier spend unless the `258b6ed7` gate genuinely opens.

---

## 10. Primary-source research rule (Lane γ, mandatory)

Primary sources only (arXiv / OpenReview / official venue). Must answer: (Q1) do role-diverse generators add real supply vs duplicate resampling at matched budget; (Q2) how to prove shared state is load-bearing not decorative; (Q3) the skeptical prior — does matched-budget multi-agent reliably tie best-of-K on code; (Q4) no-oracle self-tutoring white-space vs DreamCoder/Voyager/RLAD; (Q5) stronger-model cutoff recheck (Maverick / Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4-2603 / GLM-5 / newly reachable). Use literature ONLY if it changes the mechanism or evaluation (no literature-summary-as-output). Disconfirming evidence is wanted. Re-run the stronger-model gate recheck (decision CID `258b6ed7`); do NOT block W143 on it if it stays closed. Do NOT start the architecture branch; architecture notes updated only if needed to record the final multi-agent blocker.

---

## 11. graphify deliverables (LOCKED)

- **START (done):** `graphify update .` from HEAD `7089321d`; confirmed `GRAPH_REPORT.md` built-from == HEAD; `explain`/`path`/`affected` on the W142b + team clusters; the two lines are DISCONNECTED (no path) ⇒ fusion is a real bridge.
- **END:** after the new modules + docs land, `graphify update .` again so `graphify-out/` matches repo truth; confirm the new team-composition module is a real 1-hop bridge connecting the self-tutoring cluster (community 4109) to the consensus/shared-state clusters — i.e. `graphify path` now finds a SHORT EXTRACTED path where there was none.

---

## 12. Iterate-on-fail protocol (LOCKED — operator directive)

Do NOT accept the first apparent null. On a serious failure:
1. **Root-cause** with the W128 generation-vs-selection split: is `pool_pass` (any role's candidate is a clean extractable winner) high while `committed` is low (SELECTION cap), or is `pool_pass` itself low (GENERATION cap)? Use the $0 diversity report + the per-arm disc-rate.
2. **If SELECTION-capped:** strengthen the quorum (more independent brutes within the same budget by reallocating from candidate calls; or the trust-weighted consensus W55) — within the bounded branch.
3. **If GENERATION-capped:** strengthen STRATEGIST diversity (more distinct sketches) or test whether the family's `p≈0` half is an unbreakable bimodal wall (then it's a field-supply blocker, not a mechanism blocker — register it).
4. **If MA ties ST (team adds nothing):** lower the discovery budget further (more fragile) where i.i.d. must fail; if MA still ties, the team is not load-bearing on this family ⇒ test the 2nd mode / a 3rd family ⇒ if still ties, machine-close `W143-L-MA-TIES-ST`.
5. Only stop when EITHER the multi-agent result is earned (§8) OR the exact blocker is machine-closed (a $0-checkable theorem). Harden and rerun within the bounded branch; expensive reruns only after a $0 diagnosis predicts the fix.

---

## 13. Anti-overstatement rule + carry-forward + W144 branch logic (LOCKED)

**Anti-overstatement (HOW_NOT_TO_OVERSTATE + HONEST_FRAMING):**
- W89 (+5.56) + W105 (+7.00) STAND as the only two MULTI-AGENT retirements unless a clean multi-seed `PASS` with the team proven load-bearing is registered. W142b stands as the third resistant result (distinct class). Headline unchanged: "we did not solve context in multi-agent systems."
- A wrapped single-controller win is NOT a team retirement. A sequential prompt chain is NOT multi-agent unless role decomposition is real. A self-tutoring win is NOT multi-agent unless the team ablation shows load-bearing value. More calls are NOT coordination. A close/contaminated edge is NOT a win.
- Single-seed = demonstration; rescue-concentrated margins reported as inflated UPPER BOUNDS; small-n (n=2–4/cell) noisy.

**Disposition / carry-forward:**
- **Earned (MA-FULL ≥+5pp over A1 AND B0, ≥2 modes, NEG no-lift, load-bearing via §8.6, 2-seed):** register `W143-T-MULTI-AGENT-DISCOVER-THEN-AMORTIZE-IS-A-RESISTANT-TEAM-RETIREMENT` + the THIRD multi-agent retirement (the class gap CLOSED); W144 = cross-scale (8B democratization + Maverick) + productionize the team discover-amortize controller + the architecture branch is then entitled to begin.
- **Capped:** register the EXACT machine-checkable blocker (`W143-L-…` per §8/§12). W144 target = the named blocker. The class gap stays open; W89+W105 stand.

**Stable boundary:** gate `258b6ed7` invariant {KNOWN:1, UNKNOWN:4}; no version bump (`0.5.20` / `coordpy.sdk.v3.43`); no PyPI; `coordpy/__init__.py` untouched; advanced work explicit-import only; `ultracode` OFF; COO-9 lead unless evidence forces a code-line move.

**W143 is NOT complete if it only updates docs** — it must land executable code + script assets in the team-composition / resistant-team-bench path.

---

## 14. Lane γ integration — primary-source findings that CHANGE the mechanism (LOCKED, folded before NIM)

The Lane γ research (primary sources only; full report in `docs/RESULTS_W143_*.md`) returned an **adverse skeptical prior** and three executable changes. All are folded in here.

- **[Q3 — the DPI prior, decisive] Matched-budget multi-agent reliably TIES best-of-K** (arXiv:2604.02460 Data-Processing-Inequality argument: no post-hoc multi-agent recombination adds information beyond an ideal single context; self-consistency saturates fast, arXiv:2511.00751; homogeneous debate even LOSES −27.6pp at matched budget, arXiv:2605.00914). **Change:** the only defensible win is the pre-registered **discovery-fragile band where single-context best-of-K demonstrably fails to discover** (so there is no ideal single context to dominate the team). This is now a HARD pre-condition on the earn (§8): a §8 earn is admissible ONLY on a (family, seed) where the single-controller **ST disc-rate < 1 at the fragile budget** (verified by the §7 fragility pre-check). Outside the fragile band, MA is EXPECTED to tie — a tie there is the predicted null, not evidence against the mechanism.
- **[Q1 — role-diversity realness] Add the alien-rationale NOISE CONTROL** (arXiv:2605.00914 swapped-rationale control; AlphaCode arXiv:2203.07814: diversity must be structured bias labels, not temperature noise). **Change:** add arm **NEG-RAT** = MA-FULL but the STRATEGIST's sketches are generated from a **different-family problem** (alien rationale). If NEG-RAT discovers/lifts as well as MA-FULL, the role-diversity is decoration ⇒ team killed. (W128 sketches are already structured-bias-labelled, satisfying the AlphaCode requirement.) This is in ADDITION to the W142b alien-technique NEG (structurally-distant scaffold).
- **[Q2 — shared-state load-bearing] 3-arm shared-state ablation, not 2** (arXiv:2602.21611: vanilla→structure-empty +1.0pp ≈ raw-trajectory +1.2pp ≪ abstracted shared memory +3.9pp; the load-bearing gap is the structured-CONTENT vs transcript gap). **Change:** the shared-state ablation is **MA-FULL (holed-skeleton object) vs MA−SS (raw transcript) vs MA−SE (structure-empty, = `make_negative_control_tutor_v1`)**; the load-bearing-shared-state claim is the MA-FULL − MA−SS gap, with MA−SE ≈ MA−SS as the predicted control. Note the transcript arm carries MORE literal tokens than the holed skeleton, so a shared-state win is the STRUCTURE, not the token count.
- **[Q3/Q1 — budget honesty] Log per-arm TOKENS and CALLS** (arXiv:2604.02460 flags budget-control artifacts inflating multi-agent gains). **Change:** the sidecar already records `prompt_len`/`response_len` per call; the bench aggregates per-arm total tokens + calls and the budget-parity gate checks calls; tokens are reported so a reviewer can recompute solve@token.
- **[Q4 — white-space] The defensible novelty** = "discover a reusable scaffold on problems verifiable with NO ground-truth tests/specs (model-authored brute-force consensus) and amortize the frozen scaffold to unseen same-family problems, frontier model, multi-agent." Fence with: PolySkill arXiv:2510.15863 (semi-oracle, web), RLAD arXiv:2510.02263 (per-problem, oracle-warm), LILO arXiv:2310.19791 / DreamCoder arXiv:2006.08381 (spec-warm), Self-Improving Coding Agent arXiv:2504.15228 (benchmark-scored). Recorded in the results doc; no novelty-of-skill-learning-broadly claim.
- **[Q5 — stronger-model gate] UNCHANGED {KNOWN:1, UNKNOWN:4}; `258b6ed7` stays CLOSED.** Maverick Aug-2024 (settled); Qwen3-Coder-480B (arXiv:2505.09388, no cutoff), DeepSeek-V4-Pro (HF card, no cutoff — recheck next quarter), Mistral-Small-4-2603 (docs.mistral.ai, no cutoff), GLM-5 (HF, no cutoff) all officially UNDISCLOSED. No stronger-model spend.

**Net effect on the bet (honest):** the literature predicts a TIE unless the fragile-band pre-condition genuinely bites. W143 therefore lives or dies on whether role-diversity + brute-author quorum discovers reliably *where i.i.d. fails at the same budget*. If it does and the ablations collapse, that is a genuine, literature-defensible team result. If it ties even in the fragile band, the cap is the W128 selection cap pushed to the discovery step — a clean machine-closeable blocker.

---
*§1–§14 LOCKED (byte-stable) as of the W143 build, before any NIM spend. The team-reality rule (§2), the FNB + family-screen rule (§3), the parser-neutral/no-leakage/no-cue rule (§4), the mechanism+ablation slate (§5 + §14 controls), the budget-parity + fragile-budget rule (§7), the multi-agent earn rule (§8 + the §14 DPI-band pre-condition), and the Lane γ changes (§14) are fixed before results. Iterate-on-fail (§12) is the only sanctioned response to a null; it never relaxes §8.*
