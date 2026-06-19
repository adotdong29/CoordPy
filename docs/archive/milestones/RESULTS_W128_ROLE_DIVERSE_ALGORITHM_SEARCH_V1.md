# RESULTS — W128: role-diverse algorithm SEARCH on the non-scaffoldable resistant ICPC clusters + same-family hard-cluster dev bench + targeted resistant probe (3 lanes)

**Date:** 2026-06-01 · executes the pre-committed `docs/RUNBOOK_W128.md` α/β/γ branch logic,
locked BEFORE any NIM and BEFORE any result was interpreted. `ultracode` OFF. Decision CID
`258b6ed7` invariant. No version bump, no PyPI, `coordpy/__init__.py` untouched.

> **Verdict.** W128 stopped treating the W127 scaffold miss as the end of the mechanism search.
> It (α) built a genuinely different mechanism — a **role-diverse algorithm SEARCH**
> (`coordpy.role_diverse_algorithm_search_v1`) that turns single-shot generation into
> generate → verify → select → abstain at the SAME K=5 budget, bridging the real W41/W42
> synthesis decisions + the executor digest (the W79 substrate-controller literal bridge was
> examined and KILLED as fake-inapplicable); (β) validated it on a disjoint same-family EXPOSED
> hard-cluster dev bench (11 targets: simulation_grid 4 / adhoc_math 6 / greedy_scheduling 1;
> graph_flow EXPOSED supply = **0** ⇒ resistant-probe-only). The mechanism is **REAL** — all 11
> runs classify genuine REAL-diversity and the role-diverse SEARCH **lifts the generation
> ceiling** (pool **3/11** > plain baseline **2/11**, reaching a simulation_grid program plain
> i.i.d. sampling did not) — **but it is NOT EARNED**: the full RDA4 mechanism commits **2/11**
> (net **+0** over baseline: +1 unique `blueberrywaffle` − 1 regression `pawnshop`), tying both
> the plain baseline and the W127 scaffold line. **The bottleneck is the verification-based
> SELECTION layer, not generation** — without a hidden-test oracle the public-sample +
> derived-counterexample signal cannot convert the lifted ceiling into committed wins (abstains
> 7/11; the W125 looks-right-fails-hidden problem, localized at the selection layer). (γ) T1
> (dev EARNED) is FALSE ⇒ **no targeted resistant probe** ⇒ **$0 resistant NIM**; stronger-model
> gate CLOSED. New carry-forward `W128-L-ROLE-DIVERSE-HARD-CLUSTER-DEV-BENCH-CAP` (the
> SELECTION-lever sibling of the W123→W127 cap taxonomy) + the sharper diagnosis
> `W128-T-ROLE-DIVERSE-SEARCH-LIFTS-GENERATION-CEILING-BUT-SELECTION-CAPPED`. **W89 (+5.56) +
> W105 (+7.00) STAND as the only two retirements.** `COO-9` stays lead.

## Why W128 is NOT another lap

W120–W127 closed every prior lever (battlefield → local encoder → $0 re-routing → $0 synthesis
→ family-scaffold fresh generation). W127's `RESISTANT_SCAFFOLD_FRESH_GEN_CAP` proved a better
SCAFFOLD does not transfer; its atlas showed the resistant failures are 95% wrong-algorithm and
algorithmically DIVERSE. W128 asks the genuinely different question the operator named: **can a
role-diverse algorithm SEARCH attack the non-scaffoldable clusters (`graph_flow`,
`simulation_grid`) that no skeleton covers?** This is not more scaffolding, not re-routing, not
deterministic synthesis — it is fresh generate-verify-select-abstain with enforced
algorithmic diversity. Bounded-context / compaction / summarization remain anti-patterns,
explicitly not pursued.

## Lane α — non-scaffoldable cluster hardening + role-diverse mechanism (NIM-free)

**Hard-cluster target rule + protocol (`results/w128/cluster_protocol/cluster_protocol_v1.json`).**
From the W127 atlas, the resistant hard-cluster target = the **8** problems with public family
∈ {graph_flow, simulation_grid}, ALL `scaffoldable_flag = False`: `graph_flow` = {andor,
balancingart, bigand, buyingjerseys}; `simulation_grid` = {brownianbears, chesssolitaire,
enchantedmaze, spiesvsspies} (2 of the simulation_grid are hidden-only failures —
looks-right-fails-hidden). Per-cluster protocol records dominant evidence + why a scaffold is
the wrong mechanism + what search signal might help (graph-model elimination via derived edge
cases; rule-hypothesis agreement on derived grid edge cases). EXPOSED supply census:
**graph_flow = 0** (registered `W128-L-GRAPH-FLOW-EXPOSED-SUPPLY-CAP` ⇒ graph_flow is
resistant-probe-only); non-scaffoldable total = 11 (simulation_grid 4 / adhoc_math 6 /
greedy_scheduling 1).

**Mechanism (`coordpy.role_diverse_algorithm_search_v1`, explicit-import only).** One target =
**K=5 model calls = 1 ANALYZE + 4 IMPLEMENT** (matched to baseline A1's K=5; the analyze call
costs one generation, so the mechanism implements 4 enforced-distinct sketches vs baseline's 5
i.i.d. — a generation DISADVANTAGE that makes any win more convincing). All RDA1–RDA4 selection
variants are computed NIM-free over the same 5 generations:

* **RDA1** role-diverse sketch search (ANALYZE → SPEC + invariants + complexity + 4 distinct
  algorithm sketches + derived counterexamples; one IMPLEMENT per sketch; select = first
  public-sample passer).
* **RDA2** counterexample-guided elimination — run public-survivors on the DERIVED
  counterexamples (executor-grounded via `parse_failure_digest_v1`), commit the majority
  agreement class.
* **RDA3** role-invariant ABSTAIN — **REAL bridge** to `role_invariant_synthesis.select_role_invariance_decision`;
  irreconcilable divergence with no strict-majority quorum ⇒ abstain.
* **RDA4** two-axis consensus + fallback — **REAL bridge** to
  `integrated_synthesis.select_integrated_synthesis_decision`; commit on producer∧trust
  agreement (trust = candidate matching the model's predicted-expected on the derived cases),
  abstain on divergence, producer-only fallback otherwise. **RDA4 = the full mechanism / earn
  arm.**

**Fake-diversity kill (NIM-free, the W125 `MechanismFingerprintV1` analogue).** A run is
`diversity_real = REAL` iff ≥2 sketches AND max pairwise sketch Jaccard < 0.80 AND ≥2 distinct
AST-normalized implementations AND ≥1 NEW derived counterexample AND non-empty invariants.
`fake_diversity_control_v1` (identical sketches) classifies `FAKE_DIVERSE` (positive control,
tested). A win on a `FAKE_DIVERSE` run does NOT count (R1c).

**Honest mining (RDA4 — which candidate died and why).**
`examine_substrate_controller_applicability_v1` records that the operator-named W79 substrate
controllers (`team_consensus_controller_v14` / `consensus_fallback_controller_v25` /
`hosted_cost_planner_v12` / `hosted_real_handoff_coordinator_v11`) are ALL substrate-trust-
specific (parameterised over `replacement_then_restart_after_long_delay` pressure / trust
floors, not code-candidate consensus) ⇒ a literal bridge would be fake-different ⇒ the
literal-controller bridge is KILLED; the consensus/abstain role is filled by the W41/W42
synthesis decisions (genuinely aligned). `W128-T-SUBSTRATE-CONTROLLERS-NOT-CODE-CONSENSUS-APPLICABLE`.

## Measurement-bug recalibration (outcome-relevant; documented transparently)

The FIRST dev-bench run (165 NIM) was **INVALIDATED** by a sketch-parser false-negative: the
hosted model formats sketches as markdown `#### SKETCH A:` (4 hashes), which the locked
`_SECTION_RE` (`#{0,3}`) and the fallback splitter (`^\s*SKETCH`) both missed ⇒ **0 sketches
parsed** ⇒ the mechanism degenerated to a padded single default sketch ⇒ 9/11 spurious
`FAKE_DIVERSE` (not the model's fault). Direct inspection of the stored analyze responses
confirmed the model HAD produced genuinely different algorithms (e.g. familyvisits: "Greedy
with Priority Queue" / "Dynamic Programming" / "Binary Search"). The parser was corrected
(`#{0,6}` + markdown-prefix-tolerant fallback); re-parsing the SAME stored responses NIM-free
flipped 0→4 sketches on all 11 (positive control: `fake_diversity_control_v1` still bites). This
is directly analogous to the W127 boilerplate-FP / W126 emoticons-FP recalibrations — applied
transparently, and it required a **clean re-run** (the implementations had been conditioned on
the wrong default sketch). A separate leakage-closure bug (mis-reading `LeakageVerdictV1.clean`)
was also caught at `--limit 2` validation and fixed before the valid run. Total EXPOSED dev
spend: 30 (validation) + 165 (invalidated) + 165 (valid) = **360 NIM calls, all EXPOSED dev,
$0 resistant** — the iterative-validation cost the operator authorized for mechanism validation.

## Lane β — EXPOSED hard-cluster dev bench (`results/w128/dev_bench/hard_cluster_dev_bench_verdict.json`)

11 hard-cluster dev targets (simulation_grid 4 / adhoc_math 6 / greedy_scheduling 1;
`hard_dev_target_cid 546c1466…`, teacher disjoint `ffa027db…`). Three arms, matched budget,
pass graded on the official secret cases:

| arm | result |
|---|---|
| `plain` baseline (K=5 i.i.d. @ T=0.7, == W120/W121/W127 A1) | **2/11** |
| `scaffold` (W127 G2→G3, K=5) | **2/11** |
| `rda` RDA4 committed (the full mechanism, 5 calls) | **2/11** |
| `rda` pool ceiling (any of the 4 diverse implementations passes secret) | **3/11** |

**All 11 runs classify REAL-diversity** (after the parser fix). Per-target: RDA4 **uniquely
solves `blueberrywaffle`** (simulation_grid — a problem plain i.i.d. sampling at the same budget
did NOT reach: a genuine role-diversity generation win) but **regresses `pawnshop`** (baseline
solved it; the RDA4 pool ALSO reached a correct `pawnshop` program but the SELECTION committed a
wrong candidate). `sunandmoon` is solved by all arms. Net RDA4 gain = **+0** (1 unique − 1
regression); `net_vs_scaffold = +0`.

**Earn gate R1′ = NOT EARNED** (`ROLE_DIVERSE_HARD_CLUSTER_DEV_BENCH_NOT_EARNED`): failed **R1a′**
(`net_rda_gain +0 < +2`). The mechanism is REAL (R1c′ diversity-real + leakage-clean would pass)
and it does not under-perform the scaffold (R1e′ +0 ≥ +0), but it does not clear the +2 net bar
— a close, clean negative, not an earn.

**The load-bearing finding (`W128-T-ROLE-DIVERSE-SEARCH-LIFTS-GENERATION-CEILING-BUT-SELECTION-CAPPED`):**
the diverse SEARCH's pool ceiling (3/11) EXCEEDS plain baseline (2/11) — enforced algorithmic
diversity reaches a correct program i.i.d. sampling misses — but the **verification-based
SELECTION** (public samples + the model's derived counterexamples, with NO hidden-test oracle)
cannot convert that ceiling into committed wins: it commits the right program where the public
signal discriminates (blueberrywaffle), abstains where it diverges (7/11), and **mis-commits
where the signal is non-discriminating** (pawnshop — pool had a correct program, RDA4 committed a
wrong one). So the full mechanism nets +0 at matched budget. **The bottleneck is SELECTION, not
generation** — a sharper localization than W125/W126/W127, and consistent with W125's
`looks_right_fails_hidden` (the in-loop signal does not discriminate correct from
plausible-wrong on this family).

## Lane γ — stronger-model gate + targeted resistant probe

**Stronger-model gate (NIM-free; `results/w128/stronger_model_gate/gate_recheck_v1.json`):**
`NO_CERTIFIABLE_STRONGER_MODEL`, decision CID `258b6ed7` **invariant**, {KNOWN:1, UNKNOWN:4}
(Maverick KNOWN Aug-2024 certifiable-but-settled; Qwen3-Coder-480B / DeepSeek-V4-pro /
Mistral-Small-4-119B-2603 / GLM-5 UNKNOWN-from-primary). Maverick stays the hosted target; the
W123/W124/W125/W126/W127 caps stay closed.

**Targeted resistant probe (gated by T1 ∧ T2; `results/w128/targeted_probe/targeted_resistant_probe_verdict.json`):**
`scripts/run_w128_targeted_resistant_probe_v1.py` ENCODES the RUNBOOK § 6 earn gate. T2 (a named
hard family in the dev-earned families) is technically **True** — the lone unique solve
`blueberrywaffle` is simulation_grid, so the would-be cluster-matched subset is the 4
simulation_grid resistant problems {brownianbears, chesssolitaire, enchantedmaze, spiesvsspies}.
But **T1 (dev EARNED) is FALSE** (net +0). T1 ∧ T2 therefore fails ⇒ **probe NOT launched, $0
resistant NIM**, carry-forward `W128-L-ROLE-DIVERSE-HARD-CLUSTER-DEV-BENCH-CAP`. Per RUNBOOK
§6/§7 the exposed control is also NOT bought (resistant-first; no probe to disambiguate).

## graphify deliverables (RUNBOOK § 9)

START refreshed from HEAD `0b323e6` (force-stamped; the W127 module content was already in the
graph). END refreshed after the W128 module/scripts land (recorded in the END refresh). The new
`role_diverse_algorithm_search_v1` creates the FIRST semantic bridge between the role-diverse
SYNTHESIS stack (`integrated_synthesis` / `role_invariant_synthesis`, community 35) and the ICPC
resistant-code path (`icpc_reflexion_bench_v1` / `family_scaffold_generation_v1` /
`resistant_capability_atlas_v1` / `executor_grounded_patcher_v1`, communities 174/329) — START
they were 5–8 hops apart with no semantic edge (the `role_invariant_synthesis` path ran through
`Exception`/INFERRED).

## Carry-forward + named claims

Exactly **TWO** confirmed retirements stand — **W89** (+5.56) + **W105** (+7.00), both
contamination-EXPOSED HumanEval-family at 70B. W128 retires none and adds none.

* `W128-L-ROLE-DIVERSE-HARD-CLUSTER-DEV-BENCH-CAP` (empirical): a genuinely role-diverse
  algorithm-search line (REAL — all 11 dev runs classify genuine diversity; the W41/W42 synthesis
  bridges + the executor digest + the abstain layer all work) does NOT beat plain hosted
  generation on held-out EXPOSED hard-cluster problems by a real margin (net committed +0 < +2;
  ties baseline AND the W127 scaffold line) ⇒ the mechanism is not validated ⇒ no resistant spend
  earned. The SELECTION-lever sibling of W123 battlefield / W124 encoder / W125 re-routing /
  W126 deterministic-synthesis / W127 scaffold-fresh-generation caps.
* `W128-T-ROLE-DIVERSE-SEARCH-LIFTS-GENERATION-CEILING-BUT-SELECTION-CAPPED` (empirical): the
  diverse search's generation ceiling (pool 3/11) exceeds plain baseline (2/11) — it reaches a
  simulation_grid program i.i.d. sampling missed — but verification-based selection without a
  hidden-test oracle commits only 2/11 (mis-commits the one pool-only win), so the full mechanism
  nets +0 at matched budget; the bottleneck is SELECTION, not generation.
* `W128-L-GRAPH-FLOW-EXPOSED-SUPPLY-CAP` (empirical): the EXPOSED hard-cluster corpus
  (`/tmp/w121_icpc`, 38 problems) has ZERO `graph_flow` problems ⇒ graph_flow is
  resistant-probe-only and cannot be exposed-dev-validated at this corpus.
* `W128-T-SUBSTRATE-CONTROLLERS-NOT-CODE-CONSENSUS-APPLICABLE` (mechanically-checked): the W79
  substrate-trust controllers' decision logic is substrate-specific; a literal code-candidate
  consensus bridge would be fake-different; the consensus/abstain role is genuinely provided by
  `role_invariant_synthesis` + `integrated_synthesis`.

## Artifacts

- `coordpy/role_diverse_algorithm_search_v1.py` (Lane α; RDA1–RDA4 + fake-diversity detector +
  W41/W42 synthesis bridge + executor digest + earn gate + substrate-controller examination).
- `scripts/run_w128_cluster_protocol_v1.py` (Lane α; hard-cluster protocol + EXPOSED supply
  census + controller-kill record).
- `scripts/run_w128_hard_cluster_dev_bench_v1.py` (Lane β; 3-arm matched-budget dev bench).
- `scripts/run_w128_stronger_model_gate_recheck_v1.py`,
  `scripts/run_w128_targeted_resistant_probe_v1.py` (Lane γ).
- `tests/test_w128_role_diverse_algorithm_search_v1.py` (15 tests; falsifiability-first incl. the
  fake-diversity + leakage positive controls + the substrate-controller kill + the earn-gate
  guards; direct-execution + pytest).
- `results/w128/cluster_protocol/…`, `results/w128/dev_bench/…`,
  `results/w128/stronger_model_gate/…`, `results/w128/targeted_probe/…`.
- `docs/RUNBOOK_W128.md` (pre-registration, locked before any NIM).
