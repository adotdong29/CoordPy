# RUNBOOK W137 — parser-neutral HARD battlefield v2 + model-ladder hardness calibration + conditional frontier rerun

**Status: LOCKED before any NIM.** Operator-greenlit bounded battlefield-repair + model-ladder milestone (`COO-9` sibling; `ultracode` OFF). This runbook pre-commits the repair rule, the target-family rule, the no-leakage rule, the five hardness-calibration gates (HC1–HC5), the model-ladder calibration protocol, the same-budget mechanism arms, the §7a/§7b/§7c earn rule, and the frontier target — **all CIDs that can be computed at $0 are recorded here BEFORE calibration / dev / frontier spend.**

## 0. One line

W136 root-caused the W132–W135 "wrong-algorithm ceiling" as an **I/O-FORMAT CONFOUND** (the model wrote correct algorithms but misparsed the whitespace-flattened input) AND found that, once I/O is normalised, the textbook-DP traps are **one-shot / low-algorithm-headroom** for a 70B model. W137 stops treating that field as a valid algorithm benchmark and **rebuilds it parser-neutral and hardness-calibrated across a model ladder**, then re-tests the surviving mechanisms only if the repaired field earns it.

## 1. Stable boundary (invariant)

- `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`.
- No PyPI publish; `coordpy/__init__.py` untouched; advanced work explicit-import only.
- `W89 (+5.56)` + `W105 (+7.00)` remain the ONLY two retirements unless a later clean MULTI-SEED `PASS_MECHANISM_DRIVEN`. `COO-9` stays lead.
- Stronger-model cutoff gate decision CID `258b6ed7` (`{KNOWN:1, UNKNOWN:4}`) re-derived in Lane γ; default frontier target `meta/llama-3.3-70b-instruct` (the W105 retirement model; KNOWN cutoff ~Dec-2023).

## 2. Repair rule (the I/O-confound fix) — LOCKED

Every generated task uses **parser-neutral, standard competitive-programming I/O**: a canonical **one-logical-item-per-line normal form** declared by an `IoShapeV1` and rendered by `coordpy.parser_neutral_io_v1.render_normal_form_v1`:
- line 0 = the header (counts + global scalars);
- each subsequent line is **exactly one logical item** — a k-tuple is k whitespace-separated integers on its own line; a grid row is one contiguous string of length C on its own line; a 1-D array is one line of values;
- **no cross-item interleaving** (the W132 defect: `_case` flattened the whole body onto one line).

A task is REJECTED unless a **strict per-line reader** (the parser a model writes from the statement) and a **read-all-tokens reader** recover **byte-identical structured data** for every minted case — the **HC1 parser-neutrality gate** (`parser_neutrality_gate_v1`). This makes the W136 confound machine-impossible to recur.

## 3. Target-family rule — LOCKED

Required core modes (the W130/W131 atlas failure families): `WRONG_ALGORITHM_ADMISSIBLE`, `SEARCH_ENUM`, `COMPLEXITY_BLIND`. Optional 4th: `HIDDEN_EDGE_STATE_MISS` (included — it survives HC1+HC2 cleanly and is the mode the counterexample witness should lift on a clean field).

**Headroom thesis (Lane-γ primary sources, locked):** real 70B headroom lives where the NAIVE is implementation-trivial but wrong by COMPLEXITY (passes small samples, TLEs the large hidden case) or by an unhandled CASE — NOT "an obviously different algorithm" (LiveCodeBench-Pro arXiv:2506.11928; AlgBench arXiv:2601.04996; competitive-error-analysis arXiv:2506.22954). This is also where reflexion has room to help (USACO arXiv:2404.10952).

**Candidate slate (LOCKED before calibration):** 17 templates, `slate_fingerprint_cid = 2ce207c567324e4322f308e58a1fc2c88a8d4bdd0e340d2ec8a1b867d82b3f70`.
- COMPLEXITY_BLIND (7): `cb_count_inversions_v2`, `cb_longest_subarray_sum_le_s_v2`, `cb_count_pairs_sum_le_t_v2`, `cb_count_pairs_absdiff_le_d_v2`, `cb_sum_nearest_smaller_left_v2`, `cb_count_subarrays_sum_le_s_v2`, `cb_max_j_minus_i_le_v2`.
- WRONG_ALGORITHM_ADMISSIBLE (5): `wa_house_robber_circular_v2`, `wa_min_coins_arbitrary_v2`, `wa_weighted_interval_scheduling_v2`, `wa_knapsack_01_v2`, `wa_min_subset_diff_v2`.
- HIDDEN_EDGE_STATE_MISS (2): `he_max_subarray_kadane_v2`, `he_longest_subarray_sum_k_v2`.
- SEARCH_ENUM (3): `se_climb_stairs_123_v2`, `se_lattice_paths_blocked_nf_v2`, `se_compositions_1234_v2`.

## 4. No-leakage rule — LOCKED (unchanged from W132/W133)

The model under test sees ONLY the `statement` + PUBLIC `samples`. NEVER model-facing: `ref_source` / `naive_source` / `brute_source`, the graded `secret_cases`, any hidden-case input/output. Grading is `grade_on_secret_v1` on the DISJOINT hidden bank ⇒ memorising shown values cannot pass ⇒ tests GENERALISATION. Mechanism feedback objects (Lane β) are reconstructible only from (a) the candidate's own program, (b) the public statement+samples, (c) owned-oracle executions on FRESH witness-seed probe inputs byte-disjoint from `secret_cases` — emitting only oracle OUTPUTS + derived structural summaries (the W133/W135 witness discipline).

## 5. The five hardness-calibration gates — LOCKED

- **HC1 parser-neutrality** ($0): every minted case passes `parser_neutrality_gate_v1` (dual-parser agreement + canonical normal form). Self-tested on every corpus problem.
- **HC2 exact-oracle discrimination** ($0): every problem passes the W132 framework gates — ref-solvable; INDEPENDENT brute == ref small-case agreement (`n_brute_checked >= 1`); naive looks-right-on-all-public / fails ≥1 hidden with the declared kind (TIMEOUT for COMPLEXITY_BLIND, WRONG_ANSWER else); public/hidden split integrity.
- **HC3 strong-anchor A0 headroom** (NIM): reject a template the strong anchor (`meta/llama-3.3-70b-instruct`) one-shots — A0 pass-rate `>= hc3_ceiling = 0.80` on the calibration instances (a saturated item has no mechanism headroom; the W136 failure).
- **HC4 floor / not universally dead** (NIM): reject a template on which the strong anchor passes NOTHING even with K samples (A0 == 0 AND A1(K=3) == 0) — capability-dead at this scale (the W128–W131 generation ceiling).
- **HC5 template diversity** ($0): distinct templates have statement char-5-gram Jaccard `< 0.55` (the W132 novelty ceiling). **Recorded pre-calibration: max pairwise Jaccard = 0.4737, all_distinct = True, n_modes = 4** ⇒ the field is not one family repeated.

Quality gates required before any frontier rerun: HC1 + HC2 + deterministic regeneration + novelty/near-dup guard + the model-ladder headroom report + the family-balance report.

## 6. Model-ladder calibration protocol — LOCKED before calibration NIM

- **Ladder (LOCKED):** small = `meta/llama-3.1-8b-instruct`; strong anchor = `meta/llama-3.3-70b-instruct`. (≥2 tiers; the strong tier is the frontier anchor the earn rule references.)
- **Per template:** mint `n_a0 = 3` calibration instances from the DISJOINT calibration seed range (`CALIBRATION_SEED_BASE = 137_900_000`, byte-disjoint from the train/dev/eval/frontier seed ranges `137_1xx..137_4xx`). Run A0 (single-shot, temp 0.0) at each ladder model on all 3 instances; run A1 (K=3, temp 0.7, any-of-K) at the strong anchor on 1 instance.
- **Same budget params:** `max_tokens = 1536`, `executor_timeout_s = 8.0`.
- **Admission rule (HC3 ∧ HC4):** ADMIT iff `strong_A0_rate < 0.80` AND `strong_best_rate (max A0,A1) > 0`. Discrimination (`strong_best > small_best`) is recorded but not hard-required.
- Calibration `calibration_cid` is computed and recorded BEFORE the corpus admission / Lane-β spend.

## 7. Corpus assembly + thresholds — LOCKED

- A corpus problem = (surviving template, split, disjoint seed). Splits `train / dev / eval / frontier` use disjoint global-seed bases (`137_1/2/3/4 * 10^5`) ⇒ no instance in two splits.
- Replica count per split is set AFTER calibration so `surviving_templates × replicas` targets the thresholds; the **admitted** corpus must hit **≥40 train, ≥40 dev, ≥40 eval, ≥30 frontier**.
- **If the thresholds cannot be hit honestly, land the instrument anyway and register the exact blocker machine-checkably** (`W137-L-...`); do NOT pad with one-shot or dead templates.
- The admitted corpus CID + per-split CIDs are LOCKED before any Lane-β dev NIM; eval/frontier slice CIDs are frozen before eval/frontier NIM and the bench asserts the locked prefixes (refuses on drift).

## 7a. §7a DEV gate (go/no-go for EVAL spend) — LOCKED

On the held-out repaired **dev** slice, the LEAD mechanism arm must satisfy ALL of:
1. `(lead − B0) >= +3.33 pp`, AND
2. `(lead − A1) >= +3.33 pp`, AND
3. rescues span ≥ **2 distinct modes** OR ≥ **3 distinct template families**, AND
4. every counted rescue is STRUCTURAL/algorithmic — **a parsing/formatting-only gain is NOT counted** (the W136 §7b exclusion), AND
5. no regression nets the gain below bar.
FAIL ⇒ register the cap, **$0 eval, $0 frontier.**

## 7b. §7b EVAL / frontier-earn rule — LOCKED (operator-locked)

A mechanism arm earns the frontier rerun ONLY if, on the held-out repaired **eval** slice with fresh baselines:
1. `(lead − A1) >= +5.00 pp`, AND
2. `(lead − B0) >= +5.00 pp`, AND
3. rescues span ≥ **2 distinct modes** OR ≥ **3 distinct template families**, AND
4. **per-rescue audit: every counted rescue is STRUCTURAL/algorithmic** — a formatting/parsing-only gain is NOT an earn; a complexity-only single-family gain does NOT count, AND
5. **same-sign gain on ≥ 2 model tiers**, one of which is the frontier anchor (`meta/llama-3.3-70b-instruct`) or a faithful dev proxy.
A one-model fluke / a parsing-only gain / a single-template blip is NOT an earn.

## 7c. §7c FRONTIER outcome — LOCKED (single seed; NEVER a retirement by itself)

On the LOCKED frontier 30-slice, target `meta/llama-3.3-70b-instruct`, same exact-oracle grader, pass-fail only: `PASS_MECHANISM_DRIVEN` iff `(lead − A1) >= +5 pp` AND `MLB-1 >= 33%` AND `MLB-2 >= 33%`. A single-seed pass ⇒ next-milestone MULTI-SEED confirmation toward retirement — NOT a retirement. Maverick (KNOWN cutoff Aug-2024) is an OPTIONAL push-button cross-scale check; W137 does NOT block on it.

## 8. Lane β — same-budget mechanism arms — LOCKED before dev NIM

All arms K=5, same model, temp 0.7, `max_tokens=1536`, `executor_timeout_s=8.0`. Attempt-0 = byte-identical standard prompt; K attempts; one model call per attempt; no early stop; graded on `secret_cases`; arm scored in the "B" slot so `arm − A1 ≡ B − A1` via the verbatim W108 `_evaluate_phase2_gates` / `_mlb_rates`. Witness/feedback generation (oracle execution on disjoint probes) is **$0 NIM, paid OUTSIDE the K budget**.

- **A0** — single shot, pass@1.
- **A1** — K=5 self-consistency, pass@K (the headroom baseline).
- **B0** — the W132/W120 blind reflexion (judge-reject bit + stderr + public-sample results).
- **C0** — the W133/W134 exact-oracle COMPLEXITY witness arm, **scored on COMPLEXITY tasks only** (the oracle-grounded reference; the W133 single-mode gain).
- **M1** — best surviving FAMILY-ROUTED mechanism: per-problem, route to the counterexample witness (WRONG_ALGORITHM / HIDDEN_EDGE / SEARCH_ENUM) or the complexity witness (COMPLEXITY_BLIND), parser-neutral assumptions. Consumes: the candidate's own code + owned-oracle on FRESH probes.
- **M2** — best surviving DEPLOYABLE mechanism, oracle-free, **stripped of any parsing-only rescue** (the W134 deployable complexity witness on COMPLEXITY; blind reflexion elsewhere).
- **M3** — a deliberately simple NEGATIVE-CONTROL mechanism (relabeled reflexion) that the structural fake-different test must classify `FAKE_DIFFERENT` (so the discipline still bites).

Each arm: hypothesis first; exactly which repaired-field signal it consumes; why it should survive the W136 confound correction; KILL it if it collapses to fake-different prompt decoration. Regression fixtures before any fresh run: the W136 parser-confound diagnostics (a confounded input must FAIL HC1), the W133/W134 genuine complexity gains, the W132 frontier failures.

## 9. Spend discipline — LOCKED

Battlefield repair + calibration first. Held-out generated-field dev spend allowed. **Frontier rerun spend is NOT automatic — it is earned by §7b.** Maverick cross-scale optional/separate. No exposed frontier-control spend by default. No new seed-chasing on old official benchmarks. No stronger-model frontier spend unless the `258b6ed7` primary-cutoff gate genuinely opens (Lane γ re-derives it). A close edge / a contaminated field / a parsing effect is NOT a success.

## 10. Carry-forward caps that STAND unless new evidence changes them

W123 official-supply / W124 local-encoder / W125 rerouting / W126 deterministic-synthesis / W127 scaffold / W128 selection / W129 generation-cap / W130 sub-bar generator / W131 disclosure-bound stronger-model / W132 resistant-by-construction pilot / W133 witness single-mode / W134 deployable-complexity dev / W135 structure-witness dev / W136 generated-field I/O-confound correction (REVISED W133-L/W135-L). W89+W105 stand as the only two retirements.

## 11. W138 branch logic

- §7a dev FAIL ⇒ register `W137-L-...` cap; W138 = a genuinely different axis / a code-competent local model / a primary-KNOWN stronger model when the gate opens.
- §7a PASS, §7b eval FAIL ⇒ register the eval cap; W138 = strengthen the surviving arm or accept the bounded claim.
- §7b PASS ⇒ §7c single-seed frontier; if `PASS_MECHANISM_DRIVEN`, W138 = multi-seed confirmation toward a third retirement.
- Lane α threshold MISS ⇒ land the instrument + machine-checkable blocker; W138 = widen the hard-family supply or accept the repaired-field cap.

## 12. Deliverables

New explicit-import-only modules: `parser_neutral_io_v1`, `hard_battlefield_slate_v2`, `hard_battlefield_corpus_v2`, `model_ladder_calibration_v1` (+ Lane-β mechanism bench module). Scripts: build+selftest, model-ladder calibration, mechanism bench, conditional frontier. Unit tests incl. the HC1 parser-neutrality + the W136 confound-regression fixture. graphify START+END. Truth-surface docs + Linear sync. No version bump; no PyPI; `coordpy/__init__.py` untouched.
