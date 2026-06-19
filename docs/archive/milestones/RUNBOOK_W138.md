# RUNBOOK W138 — headroom-band battlefield search + cross-scale mechanism validation + architecture-requirements extraction + conditional frontier rerun

**Status: LOCKED before any NIM.** Operator-greenlit bounded battlefield-search / cross-scale validation / conditional-frontier-rerun milestone (`COO-9` sibling; `ultracode` OFF). This runbook pre-commits the band rule, the target-family rule, the no-leakage rule, the band-admission slate, the model-ladder protocol, the same-budget mechanism arms, the §7a/§7b/§7c earn rule, the frontier target, the primary-source rule, the architecture-requirements deliverable, the graphify deliverables, and the W139 branch logic. **Every CID computable at $0 is recorded BEFORE calibration / dev / eval / frontier spend.**

## 0. One line

W137 proved the parser-neutral repaired field is **BIMODAL at 70B** (every template's A1∈{0,1}; 1/17 survivors ⇒ no headroom band). BUT the W137 dev mechanism bench shows the M1 family-routed witness controller **already earned +33.33 pp (B−A1) and passed BOTH MLB sub-gates** on the surviving complexity cells — W137's $0-frontier verdict was a **SPAN failure (modes=1<2, families=1<3), NOT a mechanism failure.** Two confounds narrowed the field: (a) the band is **measured** with `n_a1=1` (so a1_rate∈{0,1} by construction — `count_pairs_absdiff_le_d` was culled `HC4_DEAD` on a single coin-flip yet the bench rescued it at +33pp); (b) the family slate spans too few **fixable modes**. W138 stops treating a no-headroom field as evidence about mechanisms and instead **engineers a parser-neutral, resistant-by-construction field with a real `0<A1<1` band that SPANS ≥2 fixable modes × ≥3 families across a model ladder**, then re-tests the surviving mechanisms there and reruns the frontier ONLY if the field and a mechanism both earn it.

## 1. Stable boundary (invariant)

- `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`.
- No PyPI publish; `coordpy/__init__.py` untouched; advanced work explicit-import only.
- `W89 (+5.56)` + `W105 (+7.00)` remain the ONLY two retirements unless a later clean MULTI-SEED `PASS_MECHANISM_DRIVEN`. `COO-9` stays lead.
- Stronger-model cutoff gate decision CID `258b6ed7` (`{KNOWN:1, UNKNOWN:4}`) re-derived in Lane γ; default frontier target `meta/llama-3.3-70b-instruct` (the W105 retirement model; KNOWN cutoff ~Dec-2023).
- New work is explicit-import-only NEW LEAF modules (`coordpy.headroom_band_slate_v3`, `coordpy.headroom_band_calibration_v2`, `coordpy.headroom_band_corpus_v3`, `coordpy.band_mechanism_bench_v1`) + new `scripts/run_w138_*` — reusing the W137/W120 machinery by name. Zero edits to existing modules.

## 2. The three lanes (α / β / γ) — branch logic LOCKED

- **Lane α (construction):** mint candidate continuous-knob families → calibrate across the ladder with A1-as-RATE → admit the band cells (HC1∧HC2∧HB3∧HB4) → assemble train/dev/eval/frontier corpus. Succeeds iff it lands ≥40 train / ≥40 dev / ≥40 eval / ≥30 frontier admitted instances across **≥3 distinct surviving template families spanning ≥2 mechanism-fixable modes.** On miss: land the search instrument + machine-checkable blocker.
- **Lane β (validation):** if Lane α lands a band, run the same-budget arms (A0/A1/B0/C0/N0/X1/X2) on the held-out **dev** slice at the strong anchor (+ the small tier for the cross-tier check) → §7a dev gate. If §7a PASS → held-out **eval** slice → §7b earn rule.
- **Lane γ (research + gate + truth):** primary-source research (DONE, §9); architecture-requirements artifact `docs/ARCHITECTURE_REQUIREMENTS_W138_V1.md` (DONE, §10); re-derive the `258b6ed7` stronger-model gate; gate the frontier rerun; graphify START+END; truth-surface + Linear sync.
- **Frontier:** earned ONLY if Lane α gates pass AND Lane β §7b passes AND the locked frontier slice + budget are frozen before spend. Then §7c single-seed on `meta/llama-3.3-70b-instruct`. Maverick optional/separate; W138 does NOT block on it.

## 3. Parser-neutrality + no-leakage rule — LOCKED (unchanged from W137/W132)

- **Parser-neutral I/O (HC1):** every minted task uses the canonical one-logical-item-per-line normal form declared by an `IoShapeV1` and rendered by `coordpy.parser_neutral_io_v1.render_normal_form_v1`. A task is REJECTED unless a strict per-line reader and a read-all-tokens reader recover **byte-identical structured data** for every minted+secret case (`parser_neutrality_gate_v1`). **No family is admitted if its difficulty comes only from awkward serialization** (the W136 confound is machine-impossible to recur).
- **No-leakage:** the model under test sees ONLY the `statement` + PUBLIC `samples`. NEVER model-facing: `ref_source`/`naive_source`/`brute_source`, the graded `secret_cases`, any hidden I/O. Grading is `grade_on_secret_v1` on the DISJOINT hidden bank ⇒ memorising shown values cannot pass ⇒ tests GENERALISATION. Mechanism feedback is reconstructible only from (a) the candidate's own program, (b) the public statement+samples, (c) owned-oracle executions on FRESH witness-seed probes byte-disjoint from `secret_cases` — emitting only oracle OUTPUTS + derived structural summaries.
- **No-confound / anti-cheat (LOCKED before results):** no official-task paraphrases; no accepted-solution reuse; no secret-case reuse; no formatting artifact that explains failures (HC1 enforces); no family whose hardness comes only from serialization; per-instance novelty (statement char-5-gram Jaccard `< NOVELTY_JACCARD_MAX` within a template, not just across templates).

## 4. Target-family rule — LOCKED before generation

Required: **multiple families, ≥1 complexity-sensitive (continuous knob = input size N, discriminator TIMEOUT) AND ≥1 non-complexity (discriminator WRONG_ANSWER), all with a continuous or quasi-continuous hardness knob.** Headroom thesis (Lane-γ primary sources): an item carries mechanism-relevant information only in the partial-correctness band `0<p<1` (IRT θ=b; saturated/dead = zero-information — tinyBenchmarks arXiv:2402.14992, metabench arXiv:2407.12844, Lost-in-Benchmarks arXiv:2505.15055); a SMOOTH difficulty curve comes from a **continuous structural knob** (input size N — CLRS-Text arXiv:2406.04229; recursion/composition depth, distractor/constraint count — FuncBenchGen arXiv:2509.26553), NOT surface re-skinning; a mechanism can only beat pass@K with **oracle-grounded counterexample / complexity feedback that changes generation** (Self-Debugging arXiv:2304.05128, AlphaCodium arXiv:2401.08500), because intrinsic self-correction does not help verifiable code (arXiv:2310.01798) and self-consistency is inert off-band (arXiv:2203.11171).

**Candidate slate (LOCKED before calibration; knob-parameterized factories, over-provisioned ~50% expected survival):**
- **CX (COMPLEXITY_BLIND, TIMEOUT, complexity-witness-fixable):**
  - `bcx_pairs_sum_le_t` — count pairs i<j with A[i]+A[j] ≤ T; naive O(N²), ref sort+two-pointer O(N log N). Knob = hidden N. (W137 survivor family; control + CX.)
  - `bcx_pairs_absdiff_le_d` — count pairs |A[i]−A[j]| ≤ D; naive O(N²), ref sort+two-pointer. Knob = N. (W137 near-miss mis-culled by n=1.)
  - `bcx_subarrays_sum_le_s` — count contiguous subarrays with sum ≤ S (non-neg); naive O(N²), ref two-pointer O(N). Knob = N. (W137 HC4-dead at high N; sweep N down to band.)
- **MS (compositional, WRONG_ANSWER, counterexample-fixable — NEW mode):**
  - `bms_mod_then_maxsub` — B[i]=A[i] mod M, output max-subarray-sum of B (and a d=3 variant adding a difference stage). naive DROPS the mod stage; public cases benign (A[i]<M ⇒ mod is a no-op), hidden cases A[i]≥M (mod matters). Knob = composition depth d∈{2,3}. Counterexample reveals an A[i]≥M input.
- **CE (multi-constraint, WRONG_ANSWER, counterexample-fixable):**
  - `bce_subarrays_sum_and_range` — count subarrays with sum ≤ S AND (max−min) ≤ D; naive checks ONLY the sum constraint (drops the range constraint); public cases benign (range never binds), hidden cases the range binds. Knob = #constraints / range-tightness. Counterexample reveals a sum-OK / range-violating subarray.

Modes covered: COMPLEXITY_BLIND (3 families), WRONG_ALGORITHM/compositional (MS), HIDDEN_EDGE/multi-constraint (CE) ⇒ design target ≥2 fixable modes × ≥3 families with calibration margin. The exact admitted set is decided EMPIRICALLY by calibration (§6), never asserted here. `slate_fingerprint_cid` recorded at build before any NIM.

## 5. Band-field split rule — LOCKED

- A corpus instance = (admitted family, calibrated knob, split, disjoint seed). Splits `train/dev/eval/frontier` use disjoint global-seed bases **`138_1/2/3/4 × 10^5`** ⇒ no instance in two splits. Calibration seeds use a byte-disjoint base **`138_900_000`**.
- Replica count per split is set AFTER calibration so `surviving_cells × replicas` targets the thresholds; the **admitted** corpus must hit **≥40 train, ≥40 dev, ≥40 eval, ≥30 frontier** across **≥3 surviving families spanning ≥2 modes.**
- **If the thresholds cannot be hit honestly, land the search instrument anyway and register the exact blocker machine-checkably (`W138-L-...`); do NOT pad with one-shot or dead cells.**
- The admitted `corpus_cid_v3` + per-split CIDs are LOCKED before any Lane-β dev NIM; the eval/frontier slice CIDs are frozen before eval/frontier NIM and the bench asserts the locked prefixes (refuses on drift).

## 6. Hardness-calibration + band-admission slate — LOCKED before calibration NIM

- **Ladder (LOCKED):** small = `meta/llama-3.1-8b-instruct`; strong anchor = `meta/llama-3.3-70b-instruct` (the frontier anchor the earn rule references). ≥2 tiers; an optional reachable stronger dev proxy may be added but W138 does NOT block on it.
- **A1-as-RATE (the W137 fix):** per (family, knob) cell, mint **`n_cal ≥ 8`** calibration instances (disjoint seeds from `138_900_000`). Run **A0** (single-shot, temp 0.0) at EACH ladder model on all `n_cal`; run **A1** (K=5, temp 0.7, any-of-K) at the strong anchor on all `n_cal` so the strong **`a1_rate` is a population rate over `n_cal` instances** (NOT n=1). Same-budget params: `max_tokens = 1536`, `executor_timeout_s = 8.0`. Coarse knob sweep first (locate the band), then confirm.
- **Band-admission rule (HC1 ∧ HC2 ∧ HB3 ∧ HB4):**
  - **HC1 parser-neutrality** ($0): every minted case passes `parser_neutrality_gate_v1`.
  - **HC2 exact-oracle discrimination** ($0): every instance passes the W132 gates — ref-solvable; INDEPENDENT brute == ref small-case agreement (`n_brute_checked ≥ 1`); naive looks-right-on-all-public / fails ≥1 hidden with the declared kind; public/hidden split integrity.
  - **HB3 intermediate band** (NIM): the strong anchor's `a1_rate ∈ [BAND_LO, BAND_HI] = [0.15, 0.85]` (point estimate) AND the Wilson-95% interval of the pass count **excludes 0 and 1** (genuinely intermediate, not a small-sample 0/1) AND strong `a0_rate < HC3_ceiling = 0.80` (not one-shot saturated). Cells are RANKED by closeness to `a1_rate ≈ 0.5` (peak Fisher information). This is the metabench filter (drop mean-acc>95% / zero-variance) operationalised.
  - **HB4 cross-scale discrimination** (NIM): `strong_best_rate > small_best_rate` (same-direction ladder signal). Recorded for every cell; required for admission (the IRT discrimination criterion; W137 recorded but did not require it).
- **HB5 feedback-reachability** is checked at DEV (§7a), not calibration, to bound calibration spend: a cell is mechanism-sensitive iff a routed witness lifts ≥1 A1-failing instance. A cell that is in-band (HB3) but reachability-dead at dev contributes to the §7a verdict honestly.
- `calibration_cid` (ladder, knob grid, n_cal, per-cell verdicts) is computed and recorded BEFORE corpus admission / Lane-β spend.

## 7. Self-test + regression-fixture rule — LOCKED (all $0, run before any NIM)

Build self-test asserts: (1) **HC1** parser-neutrality on every minted sample+secret case of every candidate family; (2) **HC2** exact-oracle gates admit every minted instance (ref-solvable, brute==ref, naive discriminating with the declared kind, split integrity); (3) **deterministic regeneration** — same seed ⇒ byte-identical corpus (CID stable); (4) **novelty / near-duplicate** guard (per-instance + per-template Jaccard < ceiling). Regression fixtures that MUST bite:
- **W136 I/O-confound fixture:** a W132-flattened input FAILS `parser_neutrality_gate_v1`; the normal-form input PASSES.
- **W137 bimodality detector:** the calibration correctly labels a saturated cell (a0=1.0) HC3-reject and a dead cell (best=0) HB-reject — i.e. the instrument reproduces the W137 verdict on a known-bimodal control.
- **W133/W134 genuine complexity gain:** the complexity-witness arm (C0) is structurally distinct from blind reflexion (`witness_is_genuinely_new_v1`) and the fake-different report `.bites` (M3/B0 → FAKE_DIFFERENT, ≥1 witness → REAL).

## 8. Same-budget mechanism arms + earn rule — LOCKED before dev NIM

All arms **K=5, same model, temp 0.7, `max_tokens=1536`, `executor_timeout_s=8.0`.** Attempt-0 = byte-identical standard prompt; K attempts; one model call per attempt; no early stop; graded on `secret_cases`; arm scored in the "B" slot so `arm − A1 ≡ B − A1` via the verbatim W108 `_evaluate_phase2_gates`/`_mlb_rates` (the exact evaluator that scored W89/W105/W120/W132–W137). Witness/feedback generation (owned-oracle on disjoint probes) is **$0 NIM, paid OUTSIDE the K budget**. Each arm: hypothesis first; exactly which band-field signal it consumes; why it should survive the W137 SPAN/bimodality narrowing; KILLED if it collapses to fake-different decoration.

- **A0** — single shot, pass@1.
- **A1** — K=5 self-consistency, pass@K (the headroom baseline the lead must beat). *Hypothesis: on the band 0<A1<1, A1 is beatable in principle (off-band it is not — arXiv:2203.11171).*
- **B0** — the W120/W132 blind reflexion (judge-reject bit + stderr + public-sample results). *Consumes: blind in-loop signal only. The second baseline the lead must beat (excludes "any feedback helps").*
- **C0** — exact-oracle COMPLEXITY witness (`exact_oracle_witness_v1` ARM_C2), scored on COMPLEXITY cells. *Consumes: owned-oracle timing curve on FRESH probes. Should survive: W133 proved REAL+load-bearing; clean parser-neutral field ⇒ timing is the only signal.*
- **N0** — exact-oracle COUNTEREXAMPLE witness (`exact_oracle_witness_v1` ARM_C1), scored on the NON-complexity (MS/CE) cells — **the arm W138 must vindicate.** *Consumes: minimal counterexample on FRESH probes. Should survive: W133's EW1 +0 was on a bimodal field with no almost-right-fixable CE instances; W138 CONSTRUCTS intermediate counterexample-fixable MS/CE cells. If N0 still adds +0, that is a sharp architecture-requirement finding (R3/R7), not a benchmark artifact.*
- **X1** — best FAMILY-ROUTED composite (`exact_oracle_witness_v1` ARM_C3 controller): per problem, route to the complexity witness (COMPLEXITY) or the counterexample witness (MS/CE). **The LEAD arm.** *Consumes: owned-oracle minimal-counterexample OR timing curve routed per candidate failure. This is the M1 controller that already scored +33pp on W137's complexity cells — W138 tests whether it spans modes.*
- **X2** — deliberately simple NEGATIVE CONTROL (relabeled reflexion) the structural fake-different test must classify `FAKE_DIFFERENT` (so the discipline still bites). Scored at $0 (≡ B0 by construction).

**§7a DEV gate (go/no-go for EVAL spend) — LOCKED.** On the held-out **dev** slice, the LEAD arm (X1) must satisfy ALL: (1) `(lead−B0) ≥ +3.33pp`; (2) `(lead−A1) ≥ +3.33pp`; (3) rescues span **≥2 distinct modes OR ≥3 distinct families**; (4) every counted rescue is STRUCTURAL/algorithmic — a parsing/formatting-only gain is NOT counted, a complexity-only single-family gain does NOT count toward span; (5) no regression nets the gain below bar. FAIL ⇒ register the cap, **$0 eval, $0 frontier.** (`evaluate_gate_v1`, `margin_pp=3.33`.)

**§7b EVAL / frontier-earn rule — LOCKED (operator-locked).** On the held-out **eval** slice with fresh baselines: (1) `(lead−A1) ≥ +5.00pp`; (2) `(lead−B0) ≥ +5.00pp`; (3) rescues span **≥2 modes OR ≥3 families**; (4) per-rescue audit ALL STRUCTURAL (formatting/parsing-only NOT an earn; complexity-only single-family does NOT count); (5) **same-sign gain on ≥2 model tiers**, one being the frontier anchor or a faithful dev proxy. A one-model fluke / parsing gain / single-template blip is NOT an earn. (`evaluate_gate_v1`, `margin_pp=5.00`, `two_tier_same_sign`.)

**§7c FRONTIER outcome — LOCKED (single seed; NEVER a retirement by itself).** On the LOCKED frontier 30-slice, target `meta/llama-3.3-70b-instruct`, same exact-oracle grader, pass-fail only: `PASS_MECHANISM_DRIVEN` iff `(lead−A1) ≥ +5pp` AND `MLB-1 ≥ 33%` AND `MLB-2 ≥ 33%`. A single-seed pass ⇒ NEXT-milestone MULTI-SEED confirmation toward a third retirement — NOT a retirement.

## 9. Primary-source research rule — LOCKED (DONE)

Restricted to primary sources (arXiv/OpenReview/official venue pages). Used (and how each changed W138): metabench arXiv:2407.12844 / tinyBenchmarks arXiv:2402.14992 / Lost-in-Benchmarks arXiv:2505.15055 (IRT zero-information framing ⇒ the band-admission HB3 metabench filter + rank-by-p≈0.5); CLRS-Text arXiv:2406.04229 / FuncBenchGen arXiv:2509.26553 (continuous structural knob ⇒ §4 knob-parameterized families, not re-skins); arXiv:2310.01798 / arXiv:2404.17140 (intrinsic self-correction fails on verifiable code; verifier must exceed generator ⇒ oracle-grounded witnesses, §8); Self-Debugging arXiv:2304.05128 / AlphaCodium arXiv:2401.08500 (counterexample/test feedback drives repair, largest on the hard tier ⇒ C0/N0/X1); self-consistency arXiv:2203.11171 (off-band inert ⇒ HB3 `0<A1<1`). No literature-summary-as-output; every cited finding is wired into an executable gate or family.

## 10. Architecture-requirements deliverable — LOCKED (DONE)

`docs/ARCHITECTURE_REQUIREMENTS_W138_V1.md` — 12 requirements (R1–R12) the eventual coordination-native architecture MUST satisfy to survive the W120–W137 chain and reproduce W89/W105, each traced to specific milestones + primary sources (new-trajectory generation, internal verifier stronger than generator, feedback-form routing, in-band difficulty estimator, matched-compute allocation, I/O-confound robustness, capability-bound wrong-algorithm limit, exposed→resistant transfer, multi-mode span, band-conditional superiority, auditable/falsifiable coordination, continuous structural knobs). W138 specifically tests the triad R1+R2+R3/R9 under the band condition R4/R10/R12.

## 11. Graphify deliverables — LOCKED

START: `git rev-parse HEAD`; `graphify update .`; inspect `graphify-out/GRAPH_REPORT.md`; the required `explain`/`path`/`affected` commands on the W137 + witness modules (DONE — confirmed all new work is leaf modules reusing the W137/W120 hubs with `coordpy/__init__.py` untouched; AST-only refresh, report stamp `d821f920` = last full-rebuild commit, no `GEMINI_API_KEY`). END: `graphify update .` after material code/doc changes so `graphify-out/` matches repo truth.

## 12. Spend discipline + W139 branch logic — LOCKED

**Spend:** field search + calibration first; held-out dev spend allowed; frontier rerun NOT automatic (earned by §7b); Maverick optional/separate; no exposed frontier-control spend; no seed-chasing on old official benchmarks; no stronger-model frontier spend unless `258b6ed7` genuinely opens. W138-specific: no reopening MBPP+ V2 / frozen cross-modal lines / the closed Llama-3.1 rescue branch; no APPS main-lane NIM; no dirty synthetic sold as a frontier win; no official-task paraphrase sold as resistant-by-construction; no 405B run. `ultracode` stays OFF unless the work unexpectedly expands into a real dynamic-workflow problem (multiple band fields earning live reruns at once / repo-wide field-search integration / broad external verification at once) — stated explicitly before any mode change.

**Carry-forward caps that STAND** unless new evidence genuinely changes them: W123 official-supply / W124 local-encoder / W125 rerouting / W126 deterministic-synthesis / W127 scaffold / W128 selection / W129 generation-cap / W130 sub-bar generator / W131 disclosure-bound stronger-model / W132 resistant-by-construction pilot / W133 witness single-mode / W134 deployable-complexity dev / W135 structure-witness dev / W136 I/O-confound correction / W137 repaired-field no-headroom-band (bimodality). W89+W105 stand as the only two retirements.

**W139 branch logic:**
- Lane α band MISS (no ≥2-mode × ≥3-family band, or thresholds unmet) ⇒ register `W138-L-...`; land the search instrument + machine-checkable blocker; W139 = a genuinely different difficulty axis / a code-competent model with a native populated band / a primary-KNOWN stronger model when the gate opens.
- Lane α PASS, §7a dev FAIL ⇒ register the dev cap (e.g. `W138-L-COUNTEREXAMPLE-WITNESS-DOES-NOT-SPAN-AT-70B` if N0 adds +0 on designed-fixable CE/MS); W139 = strengthen the non-complexity witness / a different fixable mode / a stronger model.
- §7a PASS, §7b eval FAIL ⇒ register the eval cap; W139 = strengthen the surviving arm or accept the bounded claim.
- §7b PASS ⇒ §7c single-seed frontier; if `PASS_MECHANISM_DRIVEN`, W139 = MULTI-SEED confirmation toward a third retirement (the first since W105).

## 13. Deliverables

New explicit-import-only modules: `headroom_band_slate_v3`, `headroom_band_calibration_v2`, `headroom_band_corpus_v3`, `band_mechanism_bench_v1`. Scripts: build+selftest, model-ladder band calibration, mechanism dev/eval bench, conditional frontier. Unit tests incl. HC1 parser-neutrality + the W136 confound-regression + the W137 bimodality-detector fixtures. `docs/ARCHITECTURE_REQUIREMENTS_W138_V1.md`. graphify START+END. Truth-surface docs (RESULTS, THEOREM_REGISTRY, HOW_NOT_TO_OVERSTATE, RESEARCH_STATUS, CHANGELOG) + Linear sync. No version bump; no PyPI; `coordpy/__init__.py` untouched.
