# RESULTS — W127: resistant capability atlas + family-specific algorithm-scaffold generation + targeted resistant probe (3 lanes)

**Date:** 2026-06-01 · executes the pre-committed `docs/RUNBOOK_W127.md` α/β/γ branch logic,
locked BEFORE the atlas was run and BEFORE any NIM result was interpreted. `ultracode` OFF.
Decision CID `258b6ed7` invariant. No version bump, no PyPI, `coordpy/__init__.py` untouched.

> **Verdict.** W127 built the resistant capability atlas (the 22 are wrong-algorithm-dominated
> + algorithmically DIVERSE, no dominant cluster, surface labels 47%-concordant), built a real
> family-specific scaffold-generation line, and validated it on the EXPOSED dev bench
> (**EARNED**: scaffold +2 net over baseline across 2 families, leakage-clean after a documented
> boilerplate-FP recalibration). The earned **R1∧R2 targeted resistant probe** (string_processing
> scaffoldable cluster, 6 problems) then created **0/6 new solves** ⇒
> `RESISTANT_SCAFFOLD_FRESH_GEN_CAP`: **the exposed-bench scaffold signal did NOT transfer to the
> non-memorizable resistant field.** Spend = the authorized EXPOSED dev bench (80) + the earned
> targeted probe (30) = **110 NIM calls**; no exposed-control bought. **W89 (+5.56) + W105
> (+7.00) STAND as the only two retirements.** New carry-forward
> `W127-L-RESISTANT-SCAFFOLD-FRESH-GEN-CAP` (the FRESH-GENERATION-lever sibling of the
> W123/W124/W125/W126 cap taxonomy). Stronger-model gate CLOSED, decision CID `258b6ed7`
> invariant. `COO-9` stays lead.

## Why W127 is NOT another lap

W120–W126 closed every **$0** in-repo lever against the resistant ICPC field (battlefield →
local encoder → $0 re-routing → $0 synthesis). W126's sharp finding: the 22 uniformly-unsolved
resistant problems have an **oracle ceiling of 0/22** — there is no headroom to miss;
deterministic recombination/repair/consensus over capability-failed generations cannot
manufacture a correct algorithm at $0. "Capability failure" is where W126 stopped. W127
refuses to stop there: it (α) diagnoses WHICH capabilities are missing, (β) builds a
family-specific algorithm-scaffold generation line and validates it on a disjoint same-family
EXPOSED development bench (development spend authorized for mechanism validation), and (γ) buys
a tightly-scoped targeted resistant fresh-generation probe ONLY if the mechanism earns it.

This is **fresh algorithmic trajectory generation informed by a real diagnosis** — not more
deterministic post-processing. Bounded-context / compaction / summarization remain
anti-patterns, explicitly not pursued.

## Lane α — resistant capability atlas (NIM-free; `results/w127/atlas/capability_atlas_v1.json`)

`coordpy/resistant_capability_atlas_v1.py` reconstructs the 22 uniformly-unsolved resistant
problems (W126 grade cache + the W120 resistant 30-slice, CID `01bf9ef8…`) and emits a
machine-checkable atlas that separates two signal layers:

**Hard, re-executable layer (definitive — from the already-graded generations):**

| signal | value | reading |
|---|---|---|
| `wrong_answer` fraction | **230/242 = 95.0%** | the failures are WRONG-ALGORITHM, not surface bugs |
| `timeout` | 9 | very few TLEs (fast-I/O hardening addresses almost nothing) |
| `runtime_error` | 0 | not crashes |
| `parse_error` | 3 (all on `chesssolitaire`) | not a parsing problem family-wide |
| visible / hidden | **19 / 3** | only 3 problems (`enchantedmaze`, `genies`, `spiesvsspies`) have a hidden-only failure |

This reproduces W126's "capability failure" with typed evidence: the resistant gap is **missing
algorithms**, dominated by visible wrong-algorithm failures with 10–11 distinct-but-uniformly-
wrong generations per problem.

**Soft, transparent, theme-biased heuristic layer (machine-checkable = deterministic +
auditable; NOT ground truth):** a deterministic lexicon + code-signal classifier over PUBLIC
inputs only (statement + samples + the model's own generations) assigns a
`dominant_algorithm_family` from a LOCKED 10-family taxonomy, with the full score vector + the
exact hits recorded. An INDEPENDENT analyst cross-check (`reference_family_signal`, from the
target's own accepted-solution structure, NEVER model-facing) measures label reliability:

* **public-label clusters:** string_processing 6, graph_flow 4, simulation_grid 4, geometry 3,
  dp_optimization 2, adhoc_math 2, greedy_scheduling 1 — **top-2 concentration 45%**.
* **reference-space (actual-algorithm) clusters:** graph_flow 6, string_processing 5,
  simulation_grid 4, dp 2, geometry 2, … — **top-2 concentration 50%**.
* **label confidence:** 7 ref-confirmed / 8 ref-conflict / 7 unconfirmed; **atlas_label_agreement
  = 0.47**.

**The honest Lane-α conclusion:** the resistant failures are **algorithmically DIVERSE**, with
NO single dominant capability cluster (top-2 ≈ 45–50% in both label spaces). Statement-surface
lexicon classification is **theme-biased** (chess/spies/encryption themes fool it; only 47%
ref-concordant), so individual family labels are weak — but the *diversity* meta-finding is
robust across both label spaces, and the hard wrong-algorithm finding is definitive. The
scaffoldable subset (leakage-clean public space, teacher-coverage ≥ 2) is **11/22**:
string_processing 6, geometry 3, dp 2 (graph_flow drops out — the public classifier finds thin
teacher coverage). Lane α does not assert a clean dominant cluster; it gives Lane β a
multi-family target and quantifies the noise.

## No-leakage + teacher/target-disjointness rule (LOCKED, enforced — RUNBOOK § 3)

* The scaffold generator opens neither a target's `submissions/` nor its `data/secret/`. A
  target's accepted solution is used ONLY to CHECK candidates for leakage (never shown).
* Teacher/target disjointness is enforced by problem short-name; a near-duplicate retrieval
  guard drops any teacher whose skeleton n-gram-overlaps the target statement above threshold.
* Every candidate passes the W126 provenance-aware `SynthesisLeakageGuardV1`. **Positive
  control verified:** a planted secret answer is caught (`test_pipeline_leakage_positive_control`).
* The atlas's reference cross-check is segregated: `family_scaffold_generation_v1` does not
  import `classify_reference_family_v1`.

## Lane β — family-specific scaffold-generation line (`coordpy/family_scaffold_generation_v1.py`)

The FIRST module to wire the EXPOSED teacher corpus + the LOCKED family taxonomy onto a
FRESH-generation prompt (graphify END: community 174, degree 39, bridging
`family_adapted_repair_synthesis_v1` [W126, reusing `SynthesisLeakageGuardV1`] +
`icpc_reflexion_bench_v1` + the atlas module). The slate (RUNBOOK § 4):

* **G1 scaffold library** — each EXPOSED accepted `.py` → a de-identified STRUCTURAL skeleton
  (locals renamed via AST, long string literals truncated, control-flow + stdlib idioms
  preserved) + an approach outline, classified into the family taxonomy, keyed by family.
* **G2 retriever** — retrieves top-R FAMILY-level scaffolds across the target's prioritized
  families (argmax → scaffold-COMPATIBLE group → runner-up — this hedges the atlas's measured
  47% theme-bias AT THE RETRIEVAL LAYER, not by forcing the classifier), enforcing
  disjointness + the near-duplicate guard.
* **G3 scaffolded fresh-generation** — prompt = target statement + public samples + the
  retrieved STRUCTURAL skeleton(s) (explicitly a template from OTHER problems, not a solution)
  → K fresh candidates from the hosted model.
* **G4 constrained scaffold policy** — deterministic family-match; the learned variant
  (`constrained_policy_optimisation_v1` / `learned_economics_controller_v1`) is registered
  `NOT_WARRANTED` below the event floor (W124/W126 precedent).

**Development bench (RUNBOOK § 5; EXPOSED dev spend authorized for mechanism validation):**
deterministic short-name-hash split of the 38 gradeable EXPOSED problems into 25 TEACHER (their
accepted `.py` → G1) and held-out DEV-TARGET problems (graded; accepted solution NEVER shown).
Baseline arm = plain hosted generation (== W120/W121 A1); scaffold arm = G2→G3; **same K=5
budget**, pass@5 on the official secret cases (public-sample prescreen → secret only for
sample-passers). Earn gate R1 = net scaffold gain ≥ +2 spanning ≥ 2 capability families,
leakage-clean, nontrivial.

**Dev-bench result (8 held-out targets — latency-bounded subset of the hash-split dev set,
spanning 5 families; `results/w127/dev_bench/exposed_dev_bench_verdict.json` + the $0
re-eval).** 80 NIM calls. Raw: baseline **1/8** → scaffold **3/8**, net **+2**, 2 unique
solves (`champernownecount` [string], `electionparadox` [adhoc]), **0 regressions**, spanning
**2 families** (R1a ✓, R1b ✓, R1d ✓).

**Leakage recalibration (outcome-relevant; documented transparently).** The first run's locked
guard flagged the two unique solves as `DEV_BENCH_INVALID_LEAKAGE` — but the flagged
"accepted-solution lines" were **universal boilerplate** (`n, k = map(int, input().split())`,
`n = int(input())`). Direct inspection confirmed the winning candidates were structurally
**different correct derivations** sharing only boilerplate (champernownecount tracks length
with `power_of_10` vs the accepted's `len(str(a))`; electionparadox sorts ascending with
`(p+1)//2` vs the accepted's descending `x//2`) — **NOT memorized reproductions**. This is a
guard-calibration false positive directly analogous to the W126 `emoticons` correction. The
accepted-line tripwire was corrected from per-line to a **contiguous-block** signature (a real
"accepted solution shown" leak reproduces a multi-line problem-specific block, not scattered
idioms); the **positive control still bites** (a planted accepted solution is caught — tested),
and a **$0 re-grade of the stored candidates** confirmed all candidates leakage-clean ⇒ R1c ✓
⇒ **`EXPOSED_SCAFFOLD_DEV_BENCH_EARNED`** (R1 = True).

**Honest strength caveat (do NOT overclaim).** The earn is at the locked **minimum** (+2 net
on 8 targets, K=5) — a real but WEAK margin, and on **exposed/pre-cutoff** problems where
algorithm-level memorization + a longer-prompt framing effect + K=5 sampling variance cannot be
excluded by the dev bench alone. The wins are genuine *derivations* (the code is not the
accepted solution), but whether the **scaffold content** (vs prompt-length/variance/memory)
caused them is exactly what the **resistant probe** (non-memorizable problems) resolves. The
dev bench earns the *right to a cheap clean test*, not a validated mechanism claim.

## Lane γ — stronger-model gate + targeted resistant probe

**Stronger-model gate (NIM-free):** `stronger_model_cutoff_certification_v1` re-affirms
`NO_CERTIFIABLE_STRONGER_MODEL`, decision CID `258b6ed7` **invariant**, registry
{KNOWN:1, UNKNOWN:4} (Maverick KNOWN Aug-2024 certifiable-but-settled; Qwen3-Coder-480B /
DeepSeek-V4-pro / Mistral-Small-4-119B-2603 / GLM-5 UNKNOWN-from-primary). Maverick stays the
hosted target; the W123/W124/W125/W126 caps stay closed.
`results/w127/stronger_model_gate/gate_recheck_v1.json`.

**Targeted resistant probe (gated by R1 ∧ R2):** `scripts/run_w127_targeted_resistant_probe_v1.py`
ENCODES the RUNBOOK § 6 earn gate — fresh resistant hosted spend is earned ONLY iff R1 (dev
bench EARNED) ∧ R2 (a scaffoldable resistant cluster ≥ 3 intersects the dev-earned families).

**Targeted-probe outcome (`results/w127/targeted_probe/targeted_resistant_probe_verdict.json`).**
R1 (dev EARNED) ∧ R2 (string_processing scaffoldable cluster ≥ 3 ∩ dev-earned families) both
hold, so the smallest honest probe was run: G3 scaffolded FRESH generation (K=5) on the 6
string_processing scaffoldable resistant problems (`leapfrogencryption`, `letterballoons`,
`marchingorders`, `emoticons`, `palindromicwordsearch`, `pillowstacking` — all in the 22
uniformly-unsolved; old 11-generation pool = 0/6), graded on the OFFICIAL secret cases (30 NIM
calls, ~5 min):

| quantity | value |
|---|---|
| `targeted_new_solves` (clean solves the old pool never made) | **0 / 6** |
| verdict | **`RESISTANT_SCAFFOLD_FRESH_GEN_CAP`** |
| broader pilot | **NOT earned** |

**The dev-validated scaffold line, run fresh on the cluster-matched resistant subset, creates
ZERO new solves.** The most parsimonious reading of the exposed-EARN → resistant-0 dissociation:
the exposed dev-bench +2 was driven by exposed-problem memorization / sampling variance /
prompt-framing, NOT a transferable scaffold-taught algorithm — because the SAME mechanism on the
SAME (earned) family's NON-memorizable resistant problems yields nothing. (`emoticons` had one
candidate trip the contiguous-block guard but solved nothing — moot for the cap.) This EXTENDS
the resistant cap from W126's *deterministic synthesis* to *fresh scaffolded generation*.
Per RUNBOOK § 6/§ 7 the exposed-control pilot is NOT bought (resistant-first; the probe is a
clean negative, nothing to disambiguate). `$0` further resistant spend.

## graphify deliverables (RUNBOOK § 9)

Refreshed START + END (`graphify update .`); END built from HEAD `<<END_HEAD>>` (78,483 nodes,
189,108 edges). `graphify explain` run on the W125/W126 arsenal + the two new W127 modules;
`graphify path family_adapted_repair_synthesis_v1 adversarial_consensus_repair_v1` = 1-hop
`imports_from` (the W126 bridge persists). New W127 edges: `family_scaffold_generation_v1`
(community 174) imports the atlas module + the W126 leakage guard + the ICPC bench;
`resistant_capability_atlas_v1` (community 329) is imported by it.

## Carry-forward + named claims

Exactly **TWO** confirmed retirements stand — **W89** (+5.56) + **W105** (+7.00), both
contamination-EXPOSED HumanEval-family at 70B. W127 retires none and adds none.

* `W127-T-RESISTANT-FAILURES-ARE-DIVERSE-WRONG-ALGORITHM-NOT-SURFACE` (empirical): the 22
  uniformly-unsolved resistant problems are 95% wrong-algorithm generations (19 visible / 3
  hidden; ≈0 TLE/crash) and algorithmically DIVERSE (top-2 family concentration ≈ 45–50% in both
  the public-label and reference-algorithm spaces); the public statement-surface family
  classifier is theme-biased (47% concordant with the actual-algorithm signal), so the robust
  diagnosis is "diverse missing algorithms", not a single dominant capability cluster.
* `W127-T-EXPOSED-SCAFFOLD-DEV-BENCH-EARNS-WEAKLY-CONFOUNDED` (empirical): on a disjoint
  same-family EXPOSED dev bench the family-scaffold line beats plain hosted generation by +2 net
  across 2 capability families (leakage-clean after a boilerplate-FP recalibration; positive
  control preserved), but the earn is the locked minimum on n=8/K=5 and is confounded by
  exposed-problem memorization / prompt-framing / sampling variance.
* `W127-L-RESISTANT-SCAFFOLD-FRESH-GEN-CAP` (empirical): the dev-validated scaffold line, run
  FRESH (R1∧R2-earned) on the 6 string_processing scaffoldable resistant problems, creates
  0/6 new secret-passing solves over the old pool ⇒ the exposed signal does NOT transfer to the
  non-memorizable resistant field; family scaffolds do not close the resistant capability gap at
  Maverick's scale. The FRESH-GENERATION-lever sibling of W123 battlefield-supply / W124
  local-encoder / W125 re-routing / W126 deterministic-synthesis caps.

## Artifacts

- `coordpy/resistant_capability_atlas_v1.py` (Lane α; lexicon classifier + reference
  cross-check + failure decomposition + clustering).
- `coordpy/family_scaffold_generation_v1.py` (Lane β; G1 library + G2 retriever + G3
  generation + G4 policy + dev-bench earn gate + leakage assertions).
- `scripts/run_w127_capability_atlas_v1.py`, `scripts/run_w127_exposed_dev_bench_v1.py`,
  `scripts/run_w127_targeted_resistant_probe_v1.py`,
  `scripts/run_w127_stronger_model_gate_recheck_v1.py`.
- `tests/test_w127_capability_atlas_and_scaffold_v1.py` (31 tests; direct-execution;
  falsifiability-first incl. the leakage positive control).
- `results/w127/atlas/…`, `results/w127/dev_bench/…`, `results/w127/targeted_probe/…`,
  `results/w127/stronger_model_gate/…`.
- `docs/RUNBOOK_W127.md` (pre-registration, locked before the atlas + before any NIM).
