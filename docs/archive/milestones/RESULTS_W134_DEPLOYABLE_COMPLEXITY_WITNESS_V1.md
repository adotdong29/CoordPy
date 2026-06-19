# RESULTS — W134: deployable complexity witness + held-out complexity-only eval + conditional frontier rerun

Executes the pre-committed `docs/RUNBOOK_W134.md` (locked before any NIM). COO-9 sibling (COO-59).
`coordpy.__version__ == "0.5.20"` · `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"` · no PyPI ·
`coordpy/__init__.py` untouched · `ultracode` OFF.

## One line

W133 proved the exact-oracle EW2 complexity witness is REAL and load-bearing but SINGLE-MODE, and
its gain consumed the oracle's timing. **W134 distils it into a DEPLOYABLE, oracle-free
public-signal complexity witness** — parse `N_max` from the public statement (DW1 constraint-derived
budget) + measure the candidate's OWN runtime growth across a public-format size ladder (DW2
stress-growth, multi-shape, log-log fit + extrapolation) → a structured "your code is O(N^p),
~Xs at N_max, rewrite to O(N log N)" diagnostic, using NO reference solution and emitting NO oracle
output. It is validated on a dedicated COMPLEXITY-ONLY held-out corpus, with a targeted frontier
complexity rerun gated on the held-out earn rule.

## Lane α — deployable witness instrument + complexity-only corpus ($0) — SUCCESS

**Instrument** `coordpy.deployable_complexity_witness_v1` (explicit-import only). DW1
`derive_budget_fact_v1` (bridges `public_signal_selection_oracle_v1.parse_max_constraint_v1`:
parses `N≤100000`; ops budget `5e8`; admissible exponent ceiling ≈1.74 ⇒ O(N²) inadmissible).
DW2 `build_ladder_v1` + `measure_growth_v1` (geometric ladder `[1000,2000,4000,8000]`, shapes
`{descending,constant,random}` adversarial-first with early-exit, baseline-subtracted, log-log
exponent fit + extrapolation to `N_max`). DW3 rewrite prompt; DW4 KEEP/REWRITE/ABSTAIN controller
(bridges `stronger_generator_slate_v1.complexity_admissible_v1` / `COMPLEXITY_OPS_BUDGET`). The
same-budget arm `run_deployable_witness_arm_v1` consumes ONLY `(code, statement, public samples)` —
no template/ref/naive/secret on any path (the deployability proof). Per the SwiftSolve
(arXiv:2510.22626) / GuessCompx (arXiv:1911.01420) small-n-instability caveat, a non-TLE
super-linear verdict requires BOTH an R²≥0.7 confident fit AND a significant runtime
(`MIN_SIGNIFICANT_S=0.1`) at the ladder top; otherwise it ABSTAINS to blind reflexion — a
calibrated diagnostic, NOT a ground-truth oracle. graphify START (`cd1e3d40`): the module is a REAL
1-hop `imports_from` bridge to BOTH `public_signal_selection_oracle_v1` AND
`stronger_generator_slate_v1` AND the witness/reflexion path (degree 44, community 260).

**Corpus** `coordpy.complexity_only_corpus_v1` — the 9 `cb_*` COMPLEXITY_BLIND templates × 5
seed-disjoint seeds per split. Admitted **train 45 / dev 45 / eval 45 / frontier 45** (slice 30) —
all ≥ the 36/36/36/30 floors. Held-out integrity TRUE: pairwise-disjoint per-instance `content_cid`
+ pairwise-disjoint seeds across all four splits. **LOCKED CIDs** (from the $0 build at
`timeout_s=8.0`, predating all β NIM): `corpus_cid 191d9954…`, `eval_split_cid 748dd6fa…`,
`frontier_slice_cid 31a81304…`.

**Self-tests + faithfulness gate (all $0):** witness reproducibility ✓, deterministic ladder ✓,
public-spec-consistent stress ✓ (every ladder input parses under the public format), deterministic
split regeneration ✓. **Naive/ref separation** on every admitted train problem: the deployable
witness fires inadmissible on the O(N²) `naive_source` and stays silent on the O(N log N)
`ref_source` on **45/45** problems (after the SwiftSolve significance-gate fix that suppressed one
contention-induced false-positive on a sub-millisecond correct `monotonic_stack` ref), vs the
exact-oracle EW2's **45/45** — deployable-vs-oracle agreement **45/45**, and the deployable witness
is `genuinely_new` (carries a ≥2-size measured curve + a growth verdict + NO oracle output) on
**43/45**; the 2 exceptions are the fastest-TLE naives that fire via an early N=2000 TLE + the DW1
constraint-derived budget fact but yield <2 clean pre-TLE curve points (they still carry strictly
more than B0's blind reject bit — a TLE-at-a-size + the over-budget verdict — just not a multi-point
curve; the ≥2-point bar is held, NOT loosened post-hoc).
Regression fixtures: the deployable witness fires on the `naive_source` of the W132 B-unique rescue
family (`cb_pairs_absdiff_le_d`) and the W133 four complexity-rescue families
(`cb_distinct_in_windows`, `cb_pairs_sum_eq_t`, `cb_subarrays_sum_eq_k`, `cb_pairs_absdiff_le_d`);
negative control: a FAST value-wrong program does NOT fire (complexity-specific, not a generic
rewrite nudge). `W134-T-DEPLOYABLE-COMPLEXITY-WITNESS-INSTRUMENT-MINTABLE` +
`W134-T-DEPLOYABLE-WITNESS-FAITHFUL-TO-EXACT-ORACLE-ON-NAIVE-REF-SEPARATION`.

Artifacts: `results/w134/corpus/{corpus_build_v1,separation_characterization_v1,selftest_v1}.json`
+ `corpus_cache.pkl` (deterministic cache so the bench never re-mints).

## Lane β — held-out complexity-only mechanism bench — DEV GATE FAILS ⇒ $0 eval, $0 frontier

Executed DEV bench: 18 held-out complexity problems (9 families × 2 seeds), `meta/llama-3.3-70b-instruct`,
1 seed × K=5, 522 NIM calls, wall 9936 s. Arms same-budget K=5, each scored in the "B" slot so
`arm − A1 ≡ B − A1` (verbatim W108 evaluator).

| arm | pass@1 | − A1 | − B0 | C0 − arm | rescues vs B0 (families) | MLB-2 |
| -- | -- | -- | -- | -- | -- | -- |
| A0 (single-shot) | 44.44 % | — | — | — | — | — |
| A1 (self-consistency) | 55.56 % | — | — | — | — | — |
| **B0** (blind reflexion) | **88.89 %** | +33.33 | — | +11.11 | (B0−A1 +33.33) | 80 % |
| **C0** (exact-oracle EW2 = UPPER BOUND) | **100.00 %** | +44.44 | **+11.11** | +0.00 | 2 (**1 family**) | 100 % |
| **D1** (deployable rewrite) | **94.44 %** | +38.89 | **+5.56** | +5.56 | 2 (**1 family**) | 91 % |
| **D2** (deployable + gate) | 94.44 % | +38.89 | +5.56 | +5.56 | 2 (1 family) | 90 % |
| **D3** (deployable controller, pre-committed LEAD) | 88.89 % | +33.33 | **+0.00** | +11.11 | 1 (1 family) | 80 % |

**Three findings.** (1) **The blind reflexion baseline B0 is already very strong on a dedicated
complexity-only field (88.89 %; `B0−A1 = +33.33 pp`):** on a field that is entirely "too-slow"
traps, the blind judge-reject bit already conveys "your code is too slow — use a faster algorithm,"
and Llama-3.3-70B knows the textbook fast algorithms (two-pointer / prefix-sum / Fenwick / Kadane /
monotonic-stack), so the reflexion loop itself does the heavy lifting and the witness has little
headroom to add. (2) **Even the EXACT-ORACLE C0's entire +11.11 pp gain over B0 is concentrated in
a SINGLE family** (`window_distinct_sum`, both `cb_distinct_in_windows` instances) ⇒ the dedicated
field's residual headroom over blind reflexion at 70B is single-family even for the oracle, so the
≥2-family earn bar would block C0 itself — a FIELD property, not a deployable-witness weakness.
(3) **The deployable witness is REAL but sub-oracle:** D1/D2 beat B0 by +5.56 pp (MLB-2 90 %+, all
algorithmic), capturing the same single-family rescue as the oracle but with one witness-induced
regression, leaving +5.56 pp on the table vs C0. The pre-committed LEAD D3 (controller) UNDER-
performs at +0.00 over B0: its KEEP/ABSTAIN conservatism (20 ABSTAIN / 34 KEEP / 18 REWRITE across
attempts) deferred to blind reflexion on the rescuable problems — the W128/W129 abstain-discipline
(good for SELECTION) here merely neutralised the witness; D1 (always-rewrite-on-fire) > D2 (gated) >
D3 (controller) on this field.

**§7a DEV gate (pre-committed: lead beats B0 ≥+3.33 pp ∧ rescues span ≥2 families ∧ within 3.33 pp
of C0): FAILS for every arm.** The ≥2-family condition fails for ALL arms (including C0); D1/D2 also
exceed the C0-tracking tolerance (C0−D = +5.56 > 3.33); D3 misses the +3.33 margin (+0.00). ⇒ the
locked rule fires: **$0 eval, $0 frontier — the frontier rerun is NOT earned** ⇒
`W134-L-DEPLOYABLE-COMPLEXITY-WITNESS-DEV-CAP`. The D1 +5.56-over-B0 is a real but single-family,
sub-oracle edge and is NOT hand-waved into an earn. Artifacts:
`results/w134/dev/w134_dev_meta_llama-3.3-70b-instruct_*/w134_dev_report.json`.

## Lane γ — research + stronger-model gate + frontier

Primary-source research (arXiv/OpenReview/official venues) is decisive and CHANGED the mechanism:
**SwiftSolve (arXiv:2510.22626, 2025)** is a near-identical published instantiation — multi-size
profiling → log-log slope → complexity class vs `n_max` budget → structured rewrite, **oracle-free
(public statement + self-generated inputs, no reference solution)** — direct primary-source evidence
the mechanism is deployable and operational; its documented "slope-fit unstable at small n" caveat
(echoed by **GuessCompx arXiv:1911.01420** and grounded in **trend-prof, FSE'07** measure-and-fit)
motivated the R²-confidence + significance guards W134 adopts. Execution-feedback self-debug
(Self-Debugging 2304.05128 / Reflexion 2303.11366 / CodeT 2207.10397 / LDB 2402.16906) and PBT/CEGIS
LLM repair (2506.18315) are uniformly CORRECTNESS-driven — none uses a growth-curve signal — so the
deployable complexity witness is genuinely new relative to that family. Verdict: the
measure-growth + constraint-budget signal is **SUPPORTED as a sound deployable oracle-free signal**,
with the literature's own caveats bounding it to a calibrated diagnostic (not a ground-truth oracle).

Stronger-model gate re-derived `NO_CERTIFIABLE_STRONGER_MODEL`, decision CID `258b6ed7` invariant
(registry `{KNOWN:1, UNKNOWN:3}` this recheck) ⇒ frontier target stays `meta/llama-3.3-70b-instruct`
(the W105 model; Maverick infra-down). No 405B.

**Frontier: NOT launched.** The §7a dev gate failed, so per the locked spend rule the frontier
complexity rerun was NOT earned and $0 frontier NIM was spent. No Maverick cross-scale check was run
(its deployment remains infra-down and the frontier was not earned). No stronger-than-Maverick model
became primary-KNOWN/certifiable (gate CLOSED). graphify END: refreshed (`graphify update .`), graph
rebuilt from current HEAD; the new `deployable_complexity_witness_v1` is a REAL 1-hop `imports_from`
bridge to BOTH `public_signal_selection_oracle_v1` AND `stronger_generator_slate_v1` AND the
witness/reflexion path (degree 44, community 260) — the graphify deliverable the runbook §12 required.

## Net

W134 LANDS a real, executable, oracle-free deployable complexity-witness instrument + a dedicated
held-out complexity-only corpus (Lane α SUCCESS), and the held-out dev bench gives a sharp, honest
verdict: on a dedicated complexity-only field at 70B the deployable witness is REAL (D1/D2 +5.56 pp
over B0, MLB-2 90 %+) but **sub-oracle and single-family**, while the strong blind-reflexion baseline
(B0 88.89 %) leaves so little headroom that even the EXACT-oracle upper bound's gain is single-family
— so the §7a dev gate fails and the frontier rerun is genuinely NOT earned (`$0` eval, `$0` frontier).
This LOCALISES the W133 single-mode cap one notch further: the complexity-witness lever, even in its
exact-oracle form, does NOT produce a broad multi-family superiority over blind reflexion on the
dedicated field at 70B, and a deployable approximation captures only ~half of the oracle's already-
small edge. **W89 (+5.56) + W105 (+7.00) remain the only two retirements; W134 retires none.** COO-9
stays lead. The deployable witness instrument + complexity corpus STAND as reusable, push-button
assets (eval slice `748dd6fa…` + frontier slice `31a81304…` locked + cached). No version bump
(0.5.20 / coordpy.sdk.v3.43); no PyPI; `coordpy/__init__.py` untouched.

W135 (per RUNBOOK §13, dev-fail branch) = accept `W134-L-DEPLOYABLE-COMPLEXITY-WITNESS-DEV-CAP`; the
remaining levers are a genuinely different axis, the Maverick cross-scale push-button (if its
deployment recovers), or a primary-KNOWN stronger-than-Maverick model when the gate opens.

## Carry-forwards added

* `W134-T-DEPLOYABLE-COMPLEXITY-WITNESS-INSTRUMENT-MINTABLE` — the oracle-free constraint-derived +
  stress-growth witness is buildable + reusable; bridges `parse_max_constraint_v1` +
  `complexity_admissible_v1` onto the witness path.
* `W134-T-DEPLOYABLE-WITNESS-FAITHFUL-TO-EXACT-ORACLE-ON-NAIVE-REF-SEPARATION` — 45/45 naive/ref
  separation, agreement 45/45 with EW2, genuinely-new 43/45 (2 fastest-TLE naives carry a single TLE
  point + budget fact, still > B0's bit); $0.
* `W134-T-DEPLOYABLE-COMPLEXITY-CORPUS-CID-TIMEOUT-INVARIANT` — the corpus CID `191d9954…` is
  byte-identical at mint timeout 8.0 s and 3.0 s (the O(N²) naives TLE at both; ref/brute fast at
  both), confirming content-addressed determinism.
* `W134-T-DEDICATED-COMPLEXITY-FIELD-B0-CEILING-HIGH-AND-SINGLE-FAMILY-HEADROOM` — blind reflexion
  alone solves 16/18 on a complexity-only field (`B0−A1 = +33.33 pp`); even the exact-oracle's
  +11.11 pp gain over B0 is single-family ⇒ the ≥2-family earn bar blocks even C0.
* `W134-T-DEPLOYABLE-COMPLEXITY-WITNESS-IS-REAL-BUT-SUB-ORACLE-ON-HELD-OUT-DEV` — D1/D2 +5.56 over
  B0 (half of C0's +11.11), single-family, with 1 witness-induced regression.
* `W134-T-CONTROLLER-ABSTAIN-DISCIPLINE-NEUTRALISES-WITNESS-ON-COMPLEXITY-FIELD` — D3 KEEP/ABSTAIN
  ties B0 (+0.00); D1 (always-rewrite) > D2 (gated) > D3 (controller).
* `W134-L-DEPLOYABLE-COMPLEXITY-WITNESS-DEV-CAP` — §7a dev gate fails (single-family + sub-oracle) ⇒
  $0 eval, $0 frontier; the deployable-complexity lever does not earn a frontier rerun at 70B.
