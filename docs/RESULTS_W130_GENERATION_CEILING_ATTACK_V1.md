# RESULTS — W130: generation-ceiling attack on the hard ICPC clusters + same-family EXPOSED dev bench + targeted resistant probe (3 lanes)

**Date:** 2026-06-01 · executes the pre-committed `docs/RUNBOOK_W130.md` α/β/γ branch logic,
locked BEFORE any NIM and BEFORE any dev-bench result was interpreted. `ultracode` OFF. Decision
CID `258b6ed7` invariant. No version bump, no PyPI, `coordpy/__init__.py` untouched. Filled ONLY
from emitted verdict JSON (`results/w130/**`).

> **Verdict.** W129 proved the binding cap is GENERATION, not selection. W130 attacked the
> generator directly. (α) The generator-failure atlas (`coordpy.generator_failure_atlas_v1`)
> decomposed the 11-target hard-cluster pool: **8/11 are pool-DEAD (generation-bound)** — 3
> `WRONG_ALGORITHM_ADMISSIBLE` + 2 `HIDDEN_EDGE_STATE_MISS` + 2 `WRONG_ALGORITHM_NO_SKETCH`
> (capability) + 1 `PARSE_IO_FAILURE`; the 2 pool-bearing misses are `SELECTION_FIXABLE` (W129's
> domain). (β) A stronger same-budget generator slate (`coordpy.stronger_generator_slate_v1`:
> GG1 complexity-gated handoff / GG2 counterexample-to-rewrite / GG3 family anti-pattern coach /
> GG4 budget router / GGLEAD = GG1→GG2), run on the SAME 11 targets at MATCHED K=5 with the W129
> selector held FIXED downstream (275 NIM), produced **exactly 1 NEW pool solve**: GG2's in-loop
> rewrite cracked `doubleup` (`HIDDEN_EDGE_STATE_MISS`, adhoc_math) — a problem the ENTIRE old
> W128/W129 pool missed — and the fixed selector committed it. This is the **FIRST W120–W130
> mechanism to crack a pool-DEAD problem by GENERATION**. But it is **NOT EARNED**: 1 < the +2
> earn bar and it does not span ≥2 families/modes. The 3 `WRONG_ALGORITHM_ADMISSIBLE` problems —
> the dominant generator-fixable mode — were cracked by NO arm, confirming the atlas
> idiom-overlap "admissible" is an UPPER BOUND (a named technique ≠ a correct algorithm =
> capability-bound). GG1/GG3/GG4/GGLEAD added 0. (γ) T1 (dev EARNED) FALSE ⇒ **$0 resistant
> NIM**; stronger-model gate CLOSED. New carry-forward `W130-L-GENERATION-CEILING-DEV-BENCH-CAP`
> + the diagnoses `W130-T-COUNTEREXAMPLE-REWRITE-LIFTS-ONE-HIDDEN-EDGE` +
> `W130-T-ADMISSIBLE-SKETCH-IS-CAPABILITY-NOT-GENERATION-FIXABLE`. **W89 (+5.56) + W105 (+7.00)
> STAND as the only two retirements.** `COO-9` stays lead.

## Why W130 is NOT another lap

W129 attacked the SELECTOR and proved committed ≤ pool ceiling (baseline+1) < the +2 bar
regardless of selector quality ⇒ the cap is GENERATION. W130 is NOT another selector/oracle/
scaffold retry. It (1) refined the generator-FAILURE atlas, (2) built a stronger SAME-BUDGET
generator, (3) held the W129 selector FIXED downstream, and (4) gated a resistant probe on the
generator actually creating new headroom. Bounded-context / compaction / summarization remain
anti-patterns, explicitly not pursued.

## Lane α — generator-failure atlas (`results/w130/atlas/generator_failure_atlas_v1.json`, $0 NIM)

`coordpy.generator_failure_atlas_v1` reconstructs the FULL old W128/W129 pool (plain ∪ scaffold ∪
rda) per hard-cluster dev target from the stored W128 sidecar (replay_misses = **0** — perfect
reconstruction), grades every candidate with a mechanical failure signature (the official
execution path), cross-checks an OFFLINE accepted-algorithm reference (NEVER model-facing), and
classifies each problem's dominant generator-failure mode (taxonomy LOCKED in code).

| problem | family | dominant mode | pool | gen-fixable | sel-fixable | adm |
|---|---|---|---|---|---|---|
| `amazingpuzzle` | simulation_grid | `PARSE_IO_FAILURE` | dead | ✓ | | ✓ |
| `blueberrywaffle` | simulation_grid | `SELECTION_FIXABLE` | bearing | | ✓ | |
| `colortubes` | simulation_grid | `WRONG_ALGORITHM_NO_SKETCH` | dead | | | |
| `doubleup` | adhoc_math | `HIDDEN_EDGE_STATE_MISS` | dead | ✓ | | |
| `electionparadox` | adhoc_math | `WRONG_ALGORITHM_NO_SKETCH` | dead | | | |
| `familyvisits` | greedy_scheduling | `WRONG_ALGORITHM_ADMISSIBLE` | dead | ✓ | | ✓ |
| `foodprocessor` | adhoc_math | `WRONG_ALGORITHM_ADMISSIBLE` | dead | ✓ | | ✓ |
| `hilbertshedgemaze` | simulation_grid | `WRONG_ALGORITHM_ADMISSIBLE` | dead | ✓ | | ✓ |
| `pawnshop` | adhoc_math | `SELECTION_FIXABLE` | bearing | | ✓ | |
| `slidecount` | adhoc_math | `HIDDEN_EDGE_STATE_MISS` | dead | ✓ | | |
| `sunandmoon` | adhoc_math | `SOLVED` | bearing | | | |

**Headline diagnosis.** Of the 8 pool-DEAD (generation-bound) problems: **6 atlas-labeled
generator-fixable** (3 admissible-wrong-algorithm + 2 hidden-edge + 1 parse/IO), **2 capability
failures** (`WRONG_ALGORITHM_NO_SKETCH` — no sketch named the accepted approach). The 2
pool-bearing misses are `SELECTION_FIXABLE` (W129's domain, NOT generation). **Honest caveat
(load-bearing for Lane β):** `WRONG_ALGORITHM_ADMISSIBLE` is an idiom-overlap HEURISTIC — a sketch
NAMED the accepted technique family (e.g. dp/binary_search/greedy/graph) — so it is an **UPPER
BOUND** on generator-fixability, NOT proof a correct algorithm was reachable (the W127
47%-theme-classifier lesson). Implementation/parsing is NOT the bottleneck (every candidate
parses); 7/8 pool-dead problems have ZERO public survivors (the algorithm is wrong on the PUBLIC
samples) — these are wrong-algorithm failures, not IO bugs.

## Lane β — stronger same-budget generator dev bench (`results/w130/dev_bench/gg_dev_bench_verdict.json`)

`coordpy.stronger_generator_slate_v1`, 5 arms on the SAME 11 hard-cluster EXPOSED dev targets
(`hard_dev_target_cid 546c1466…`, teacher disjoint `ffa027db…` — both byte-identical to W128),
MATCHED K=5 budget (every arm spends exactly 5 model calls), W129 selector held FIXED downstream
(`select_so_v1` SOLEAD, NIM-free). **275 NIM calls, ~13 min.** The old W128/W129 pool solved
{`pawnshop`, `sunandmoon`, `blueberrywaffle`} on secret; the other 8 are pool-DEAD.

| arm | NEW pool solves (pool-dead, absent from old pool) | new committed |
|---|---|---|
| GG1 — complexity-gated handoff | **0** | 0 |
| GG2 — counterexample-to-rewrite | **1** {`doubleup`} | 1 {`doubleup`} |
| GG3 — family anti-pattern coach | **0** | 0 |
| GG4 — budget router | **0** | 0 |
| GGLEAD = GG1→GG2 | **0** | 0 |

**The one real generation lift (`doubleup`, GG2).** GG2 produced 3 implements (A0/B1/C2) that all
failed; the best (A0) failed a PUBLIC sample (`AssertionError` = wrong answer). GG2's in-loop
rewrite — driven by the typed PUBLIC failure digest (NEVER the secret) — produced a
structurally-NEW candidate `rw` (`rewrite_structurally_new = True`) that was the unique public
survivor; the fixed W129 selector committed it (`SINGLE_SURVIVOR`) and it PASSED secret,
leakage-clean. The entire old W128/W129 pool (plain ∪ scaffold ∪ rda) never produced a
secret-passing `doubleup` candidate ⇒ this is a genuine GENERATION-ceiling lift on a `HIDDEN_EDGE`
problem — **the first time in the W120–W130 chain a pool-DEAD problem was cracked by generation**
(W128 lifted `blueberrywaffle` but that pool was selection-capped; W129 was pure selection).

**Earn gate R2W = NOT EARNED** (`GG_EXPOSED_DEV_BENCH_NOT_EARNED`): best arm GG2 created **1** new
pool solve (need ≥ 2 spanning ≥ 2 families/modes); `spans_two = False`. The dominant
generator-fixable mode — the 3 `WRONG_ALGORITHM_ADMISSIBLE` problems (`familyvisits` /
`foodprocessor` / `hilbertshedgemaze`) — was cracked by **NO** arm, empirically confirming the
atlas "admissible" was an UPPER BOUND: the model NAMED the technique family but never DERIVED a
correct algorithm ⇒ these are capability-failures-in-disguise, not generation-engineering-bound.
The `PARSE_IO` (`amazingpuzzle`) and the 2 capability `NO_SKETCH` problems were also 0. GG1's
complexity gate, GG3's coach, and GG4's router added nothing.

**Stored-regression trio (`blueberrywaffle` / `pawnshop` / `sunandmoon`) — PRESERVED.** The fixed
W129 selector had **0 new mis-commits**: every committed candidate on the trio across all arms
PASSED secret. `blueberrywaffle` committed-correct by GG3 + GGLEAD (others safely abstained the
under-determined tie); `pawnshop` committed-correct by GG2 + GG3 (one arm via `LEAD_FALSIFIER_UNIQUE`
— the GG-generated pool happened to be falsifier-separable, so the selector even cashed it out;
no arm re-committed the W128 wrong A0); `sunandmoon` committed-correct by GG2 + GG3 + GG4. The
W129 abstain discipline holds; the stronger generator did NOT regress selector behavior.

**Realness controls + honest kills.** `gg1_gate_control` PASS (O(N²)@1e6 inadmissible /
O(N log N) admissible / unstated unjudgeable); `gg2_rewrite_control` PASS (failing-case finder
bites a wrong candidate, not a correct one); the hosted-controller examination KILLS the literal
planner/handoff/substrate bridge as fake-different (the cache planner is efficiency-only KV-prefix
savings, NOT a capability lever — graphify-confirmed no ICPC-code path). **GG3 killed:** its
family coach tripped the contiguous-block leakage guard on `foodprocessor` + `pawnshop`
(`coach_is_scaffold = False` but a CANDIDATE reproduced a family-boilerplate accepted block — the
W127 boilerplate-FP pattern; the guard correctly DROPPED those candidates), and GG3 added 0 new
solves ⇒ the anti-pattern coach is a leakage-risk with no upside. The one earning candidate
(`doubleup` GG2 `rw`) is leakage-clean.

## Lane γ — stronger-model gate + targeted resistant probe (`results/w130/stronger_model_gate/gate_recheck_v1.json`)

**Stronger-model gate ($0):** `NO_CERTIFIABLE_STRONGER_MODEL`, decision CID `258b6ed7`
**invariant** (W114→W130), {KNOWN:1 (Maverick Aug-2024), UNKNOWN:4 (Qwen3-Coder-480B /
DeepSeek-V4-pro / Mistral-Small-4-119B-2603 / GLM-5)}. Gate CLOSED; Maverick stays the hosted
target.

**Targeted resistant probe (gated by T1 ∧ T2):** **T1 (dev EARNED) is FALSE** (R2W not earned —
1 < 2 spanning) ⇒ T1 ∧ T2 fails ⇒ **probe NOT launched, $0 resistant NIM**. Per RUNBOOK §8 the
exposed control is also NOT bought (resistant-first; no probe to disambiguate). The W123–W129
caps stay closed.

## Carry-forward + named claims

Exactly **TWO** confirmed retirements stand — **W89** (+5.56) + **W105** (+7.00), both
contamination-EXPOSED HumanEval-family at 70B. W130 retires none and adds none.

* `W130-L-GENERATION-CEILING-DEV-BENCH-CAP` (empirical): a stronger SAME-BUDGET generator slate
  (GG1 complexity-gate / GG2 counterexample-rewrite / GG3 coach / GG4 router / GGLEAD), with the
  W129 selector held fixed, creates only **1** NEW EXPOSED hard-cluster solve (< the +2 earn bar,
  does not span ≥2 families/modes) ⇒ the hard-cluster generation ceiling is PARTIALLY liftable but
  NOT to the earn bar at the Maverick scale + K=5 budget. The GENERATION-line sibling of the cap
  taxonomy: W123 battlefield → W124 encoder → W125 re-routing → W126 synthesis → W127 scaffold-gen
  → W128 role-diverse-search → W129 selection-oracle → **W130 generator-line**.
* `W130-T-COUNTEREXAMPLE-REWRITE-LIFTS-ONE-HIDDEN-EDGE` (empirical): GG2's in-loop PUBLIC/derived
  digest-driven REWRITE cracked a `HIDDEN_EDGE_STATE_MISS` problem (`doubleup`) the ENTIRE old
  W128/W129 pool missed, committed by the fixed selector — the first W120–W130 GENERATION crack of
  a pool-DEAD problem. The load-bearing lever is the rewrite (not selection, not diversity, not a
  scaffold).
* `W130-T-ADMISSIBLE-SKETCH-IS-CAPABILITY-NOT-GENERATION-FIXABLE` (empirical): the 3
  atlas-`WRONG_ALGORITHM_ADMISSIBLE` problems (a sketch named the accepted technique family) were
  cracked by NONE of the 5 generator arms ⇒ the atlas idiom-overlap "admissible" over-counts
  generator-fixability; naming a technique ≠ a correct algorithm; the dominant pool-dead mode is
  capability-bound, not generation-engineering-bound.
* `W130-T-ANTI-PATTERN-COACH-RISKS-BOILERPLATE-LEAKAGE` (mechanically-checked): GG3's family coach
  nudges the model toward family boilerplate that can reproduce a contiguous accepted block (the
  W127 FP pattern); the provenance-aware guard caught + dropped it; GG3 earned nothing ⇒ killed.
* Carried forward: `W128-L-GRAPH-FLOW-EXPOSED-SUPPLY-CAP` +
  `W129-L-HARD-CLUSTER-GENERATION-CEILING-CAPS-SELECTION-EARN`.

## Artifacts

- `coordpy/generator_failure_atlas_v1.py` (Lane α; locked taxonomy + mechanical classifier +
  offline admissibility heuristic + atlas aggregation).
- `coordpy/stronger_generator_slate_v1.py` (Lane β; GG1–GG4 + GGLEAD + complexity gate +
  counterexample-rewrite + family coach + budget router + fixed-selector finalizer + realness
  controls + honest hosted-controller examination + R2W earn gate).
- `scripts/run_w130_generator_failure_atlas_v1.py` (Lane α builder).
- `scripts/run_w130_gg_dev_bench_v1.py` (Lane β; 5-arm matched-budget dev bench, W129 selector fixed).
- `scripts/run_w130_stronger_model_gate_recheck_v1.py` (Lane γ).
- `tests/test_w130_generator_atlas_and_slate_v1.py` (15 tests; falsifiability-first incl. the
  complexity gate, the GG1/GG2 controls, the controller-mining kill, the leakage-blocks-earn +
  span-requires-two earn-gate guards; direct-execution + pytest).
- `results/w130/atlas/…`, `results/w130/dev_bench/…`, `results/w130/stronger_model_gate/…`.
- `docs/RUNBOOK_W130.md` (pre-registration, locked before any NIM).
