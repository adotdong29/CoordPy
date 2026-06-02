# RESULTS W132 — resistant-by-construction hard-family battlefield + exact-oracle gates + executed frontier pilot (Llama-3.3-70B; Maverick infra-down) — clean FAIL

**One line.** W123 bounded the OFFICIAL resistant-package supply and W131 found the model
axis blocked on cutoff DISCLOSURE; both share one upstream dependency — a contamination-
RESISTANT battlefield we either inherit or certify against a disclosed cutoff. **W132
removes that dependency by MINTING the battlefield ourselves** (a CoordPy-owned, exact-
oracle, novelty-guarded, deterministic battlefield of **33 admitted** resistant-by-
construction problems targeted at the failure families that actually beat the mechanism
stack) **and RAN the frontier pilot on it**. The locked target Maverick was infra-down
this session (machine-checked: same key, `llama-3.1-8b`/`llama-3.3-70b` respond instantly,
maverick chat hangs 0-bytes), so the frontier target was substituted **pre-spend** to
**`meta/llama-3.3-70b-instruct` — the exact model that earned the W105 retirement** (KNOWN
cutoff ~Dec-2023; the minted field is resistant for it by date AND construction). The
executed pilot is a **clean FAIL**: **A0 = A1 = 76.67%, B = 80.00%, B − A1 = +3.33 pp**
(< the +5 pp bar; 7/9 Phase-2 gates; MLB-1 = 26.7% FAIL, MLB-2 = 25.0% FAIL) ⇒
`VERDICT = FAIL` ⇒ **`W132-L-RESISTANT-BY-CONSTRUCTION-PILOT-CAP`**. **This closes the
"maybe the official benchmarks were the wrong test" escape**: on a battlefield minted
specifically in the failure families that beat the stack, at the exact retirement model +
scale, the same-budget mechanism still does NOT clear the retirement bar (+3.33 pp — the
SAME sub-floor one-problem nudge as the W121 matched-exposed-ICPC control). **W89 (+5.56) +
W105 (+7.00) remain the only two retirements; W132 retires none.** Maverick stays the
push-button cross-scale check when its deployment recovers.

All numbers below are filled ONLY from emitted verdict JSON
(`results/w132/**`).

---

## Lane α — resistant-by-construction battlefield (MAIN empirical, $0 NIM) — **SUCCESS**

New modules (explicit-import only):
`coordpy.resistant_by_construction_battlefield_v1` (framework: minted-problem record,
the three-program oracle harness, the six per-problem quality gates, the novelty guard,
the manifest/CID, Maverick resistance certification) +
`coordpy.resistant_by_construction_slate_v1` (the 33-template slate).

Each minted problem ships THREE independent complete stdin/stdout programs so correctness
is machine-checked, never asserted: `ref_source` (the scalable correct program whose
stdout IS the answer key), `brute_source` (an INDEPENDENT obviously-correct exhaustive
oracle, run on the small cases to cross-check `ref_source`), and `naive_source` (the
admissible-wrong trap). The model under test sees ONLY the statement + the PUBLIC samples
and is graded by the audited `grade_icpc_candidate_case_v1` token-diff / float oracle
(exit-0-iff-every-hidden-case-passes, NO LLM judge). Minted problems are emitted as
`IcpcPilotProblemV1` so the *already-validated* W120 reflexion bench consumes them verbatim.

**Build verdict (`results/w132/battlefield/battlefield_verdict_v1.json`; re-derivable
byte-identically):**

| field | value |
|---|---|
| templates minted | 33 |
| gate-passing | 33 |
| **admitted (novelty-clean)** | **33 ≥ 30** ✓ |
| mode histogram | COMPLEXITY_BLIND 9 / HIDDEN_EDGE_STATE_MISS 8 / WRONG_ALGORITHM_ADMISSIBLE 8 / SEARCH_ENUM 8 |
| distinct families | 33 (one per problem) |
| `manifest_cid` | `562aafbd62f550d1…` |
| `raw_cid` | `1e9a2a42f20f05ec…` |
| core 30-slice `core_slice_cid` | `f6a2ebed3da2f13b…` (mode-stratified 8/8/7/7) |
| deterministic regeneration | TRUE (re-mint → byte-identical manifest CID) |
| Maverick resistance | RESISTANT (minted 2026-06-02 > cutoff 2024-08-31 by date AND fresh instances by construction; reused W114 C1..C4 gate corroborates `certifiable`) |
| minted date / seed / exec timeout | 2026-06-02 / 132 / 8.0 s |

**Quality gates (every admitted problem passes ALL):** exact-oracle self-test +
small-vs-large `brute`↔`ref` agreement (non-vacuous); reference solvable in budget;
discriminating-hidden-case (`naive` PASSES every public sample AND FAILS ≥1 hidden case in
its declared mode — TIMEOUT for COMPLEXITY_BLIND, WRONG_ANSWER otherwise); public/hidden
split integrity; deterministic regeneration; pass-fail-only.

**Hard-family target rule (§3 of the runbook), what it selected:** the slate prioritises
the W130/W131 atlas modes — `WRONG_ALGORITHM_ADMISSIBLE` (named-but-wrong greedy where DP
is required: coins / weighted-interval / knapsack / partition / LIS / house-robber /
max-product / LCS), `HIDDEN_EDGE_STATE_MISS` (wrap / overlap / bracket-type / inclusivity
/ tie / sign / sort corners), `COMPLEXITY_BLIND` (O(N²) naive TLEs on the large stress
case while the O(N log N)/O(N) reference finishes), and `SEARCH_ENUM` (small-n exhaustive
oracle exact; naive = ordered-vs-unordered / wrong-recurrence / blocks-ignored miscount).

**Novelty / near-duplicate rule, what it rejected:** `novelty_filter_v1` rejects a problem
iff its statement char-5-gram Jaccard ≥ 0.55 with an already-accepted minted problem, or
its statement embeds an official ICPC identity token (the W120 short-names, paraphrase
guard). On the first build it correctly fired on a near-duplicate (`wa_max_product_subarray`
J=0.745 vs `cb_max_subarray_sum`); the two are genuinely different algorithms (Kadane sum
vs min/max product DP), so the product statement was differentiated and BOTH retained. The
guard is validated by a planted-duplicate positive control + an official-identity control
in the tests.

**No-leakage rule:** `MintedProblemV1.to_pilot_problem` ships ONLY the statement + public
samples + the hidden grader; `ref_source`/`naive_source`/`brute_source` are never in any
model-facing prompt; the bench's reflexion feedback uses ONLY the public samples + judge
verdict bit + executor stderr (verbatim the W120 anti-cheat). A unit test asserts the
model-facing payload contains no solver token tells.

---

## Lane β — generated-family mechanism validation + executed frontier pilot — **clean FAIL**

**Earn check (RUNBOOK §8b): the battlefield EARNED the pilot** — ≥30 admitted ✓, all
quality gates pass ✓, resistance-certified ✓, A0/A1/B same-budget eval rule locked before
spend ✓. Arms = the *already-validated* W88/W89 three-arm mechanism (A0 single-shot / A1
self-consistency-pass@K / B sequential reflexion) at K=5, 1 seed (132001), graded by the
audited oracle, scored by the verbatim W108 `_mlb_rates` + `_evaluate_phase2_gates` — the
SAME code that scored W89/W105/W120.

**Infra-forced model substitution (RUNBOOK §8d, locked PRE-SPEND).** The locked target
Maverick was infra-down this session — machine-checked: `GET /v1/models` returns 200 in
0.15 s and lists maverick, and `meta/llama-3.1-8b-instruct` returns in 0.42 s with the SAME
key, but every maverick chat/completions call (streaming + non-streaming, 8/16 tokens) returns
**0 bytes** and times out (45–75 s, `time_starttransfer=0`) ⇒ a model-specific server-side
outage. So the frontier target was substituted pre-spend to **`meta/llama-3.3-70b-instruct`
— the exact model that earned the W105 retirement** (primary-KNOWN cutoff ~Dec-2023;
reachable at ~7 s/call), making this the MOST on-target transfer test. Same core 30-slice
(`f6a2ebed…`), same gates, same eval rule.

**Calibration (RUNBOOK §8a) cleared non-degenerate** (6 mode-spanning problems, 66 calls):
A0 = A1 = B = 66.7%, A1 in the useful band, 2/6 attempt-0 failures (MLB invocable).

**Executed pilot (Llama-3.3-70B × minted resistant-by-construction core 30-slice, 1 seed ×
K=5 = 330 calls, 80.3 min, 429-throttled but completed via backoff)
(`results/w132/pilot/.../w132_pilot_report.json`):**

| metric | value |
|---|---|
| A0 / A1 / B (pass@1) | 76.67% / 76.67% / 80.00% |
| **B − A1** | **+3.33 pp** (< the +5 pp bar; **G4 FAIL**) |
| B − A0 | +3.33 pp (G5 FAIL) |
| Phase-2 gates | **7/9** (G4, G5 fail; G1–G3, G6–G9 pass; per-problem majority 30/30) |
| MLB-1 invocation | 8/30 = **26.7% FAIL** (reflexion genuinely invoked on the hard quarter) |
| MLB-2 rescue | 2/8 = **25.0% FAIL** (sub-floor; reflexion not load-bearing) |
| **verdict** | **`FAIL` ⇒ `RESISTANT_BY_CONSTRUCTION_PILOT_CAP`** |

**Per-mode pass counts (A0 / A1 / B):** COMPLEXITY_BLIND 4/4/**5**, HIDDEN_EDGE_STATE_MISS
7/7/7, SEARCH_ENUM 6/6/6, WRONG_ALGORITHM_ADMISSIBLE 6/6/6. **B − A1 = +3.33 pp is exactly 1
B-unique rescue, 0 regressions:** reflexion rescued exactly ONE problem A1 missed —
`cb_pairs_absdiff_le_d` (COMPLEXITY: after the judge returned "rejected on hidden tests",
reflexion switched the O(N²) attempt to the efficient two-pointer) — a REAL but isolated
mechanism win, the only complexity rescue. The **6 problems unsolved by ALL arms** are the
capability-bound traps the 70B cannot crack even with reflexion (3 COMPLEXITY +
`he_interval_union_length` [overlap double-count] + `se_lattice_paths_blocked` [binomial-
ignores-blocks] + `wa_knapsack_01` [ratio-greedy]); reflexion-on-public-feedback cannot fix
a wrong algorithm whose public samples pass.

**This is the strongest possible statement of the bounded ceiling.** On a battlefield minted
SPECIFICALLY in the failure families that beat the stack, with exact oracles + novelty
guards, at the EXACT W105 retirement model and 70B scale, the same-budget mechanism gets
**+3.33 pp — below the +5 pp bar, MLB sub-floor** — the SAME margin as the W121 matched-
exposed-ICPC control (+3.33 pp). The "maybe the official benchmarks were the wrong test"
escape is **closed**: we built the right test and the mechanism still does not transfer.

**DEV_ONLY local-Ollama characterization** (`qwen2.5-coder:7b`, run while NIM was fully
down) is retained only as pipeline-execution evidence: `PARTIAL_THROUGHPUT_LIMITED`, 8 clean
end-to-end cycles, ~100 s/call throughput-impractical; `frontier_claim=false`,
`can_retire=false` — NOT a frontier or retirement result.

---

## Lane γ — stronger-model gate / graphify / truth ($0) — gate CLOSED

`scripts/run_w132_stronger_model_gate_recheck_v1.py` re-derived the cutoff gate
(`results/w132/stronger_model_gate/gate_recheck_v1.json`): verdict
`NO_CERTIFIABLE_STRONGER_MODEL`, **decision CID `258b6ed7…` invariant** (byte-identical
W114→W132), registry split {KNOWN:1, UNKNOWN:3 (+GLM-5 = 4)}, gate **CLOSED**. No new
primary cutoff disclosure since the W131 re-check earlier on 2026-06-02.

**The W132 γ contribution:** resistance-by-construction REMOVES the W131 cutoff-disclosure
dependency. The minted instances did not exist before the mint date, so the field is
resistant for ANY model regardless of disclosure — UNKNOWN-cutoff stronger models may be
used as DEV_ONLY characterization, while the FRONTIER claim stays on Maverick precisely
because its cutoff is KNOWN (Aug-2024) and the mint date strictly post-dates it. The
W123→W131 caps stay closed; no 405B run.

**graphify:** refreshed at START (HEAD `3708ea5d`; `results/w132/graphify/graphify_start_w132.txt`)
and END; the new `resistant_by_construction_battlefield_v1` creates the first semantic
bridge from a minted-task GENERATOR to BOTH the validated bench (`icpc_reflexion_bench_v1`,
`IcpcPilotProblemV1`) AND the audited grader (`coordpy_icpc_battlefield_v1`,
`grade_icpc_candidate_case_v1`).

---

## Net + carry-forward

* **W89 (+5.56) + W105 (+7.00) remain the only two confirmed retirements.** W132 retires
  none (executed FAIL, +3.33 pp < +5 pp, MLB sub-floor).
* **NEW `W132-T-RESISTANT-BY-CONSTRUCTION-BATTLEFIELD-MINTABLE`** — CoordPy CAN mint a
  genuinely-new, exact-oracle, post-cutoff, novelty-clean, deterministic resistant
  battlefield (33 ≥ 30) targeted at the failure families that beat the stack; this removes
  the W123 official-supply cap and the W131 cutoff-disclosure dependency as blockers on the
  EXISTENCE of a resistant instrument. The instrument STANDS as a reusable resistant
  battlefield.
* **NEW `W132-L-RESISTANT-BY-CONSTRUCTION-PILOT-CAP`** (the EXECUTED FAIL) — on the minted
  resistant-by-construction field, at the exact W105 retirement model (Llama-3.3-70B) and
  70B scale, the same-budget mechanism gets B − A1 = +3.33 pp (< +5 pp; MLB-1 26.7%, MLB-2
  25%, both FAIL): a single isolated complexity rescue, 0 regressions, 6 capability-bound
  traps unsolved by all arms. The bounded contamination-EXPOSED-HumanEval-family-at-70B
  ceiling HOLDS — and is now STRENGTHENED, because the "wrong test" escape is closed (the
  field was built specifically in the failure families and the mechanism still did not
  transfer). +3.33 pp matches the W121 matched-exposed-ICPC control exactly.
* **`W132-T-RESISTANT-BY-CONSTRUCTION-MINTABLE-FOR-ANY-CUTOFF`** (γ) — resistance-by-
  construction removes the W131 cutoff-disclosure dependency (resistant for ANY model);
  Maverick stays the eventual push-button CROSS-SCALE check (deployment infra-down this
  session; `--model meta/llama-4-maverick-17b-128e-instruct` re-runs on the same slice when
  it recovers).
* W123→W131 caps carried forward unchanged; decision CID `258b6ed7…` invariant.

**W133** (per RUNBOOK §12): the executed pilot FAILed, so W133 = accept the registered
resistant-by-construction pilot cap (the minted instrument STANDS as a reusable resistant
battlefield); the remaining levers are the Maverick CROSS-SCALE check on the SAME slice
(push-button when its deployment recovers), an operator-greenlit multi-seed confirmation of
the +3.33 pp on the minted field, or a genuinely different axis. A retirement is registered
ONLY on a later clean multi-seed `PASS_MECHANISM_DRIVEN`. Bounded-context / compaction
remain anti-patterns. COO-9 stays lead.

**No version bump (0.5.20 / coordpy.sdk.v3.43); no PyPI; `coordpy/__init__.py` untouched.**
