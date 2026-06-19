# RUNBOOK W132 — resistant-by-construction hard-family battlefield + exact-oracle quality gates + Maverick pilot if honestly earned

**Status: LOCKED PRE-REGISTRATION.** Written BEFORE any W132 NIM (calibration or pilot)
call and BEFORE any pilot result is interpreted. The Lane-α battlefield CONSTRUCTION and
its quality gates (§ 2, § 6) are **$0-NIM** (deterministic generation + the answer-key /
oracle subprocess; no model inference) and are allowed before this lock — the W129/W130/
W131 $0-recon discipline. They are already emitted (`results/w132/battlefield/`;
`manifest_cid 562aafbd…`, `core_slice_cid f6a2ebed…`). The Lane-γ cutoff gate (§ 9) is
also $0 and already emitted (decision CID `258b6ed7…` invariant). Fill
`docs/RESULTS_W132_*` ONLY from emitted verdict JSON
(`feedback_never_prewrite_results_before_data`). The pre-committed code rules below are
the branch authority, not any prior or hope.

`ultracode` OFF. W132 is a bounded instrument-construction / conditional-pilot milestone,
not a repo-wide dynamic-workflow job. It turns ON only if the work unexpectedly expands
into a genuine dynamic-workflow problem (multiple minted families all earn live pilots at
once / repo-wide task-generator integration / broad multi-surface external verification at
once) — and only after an explicit mode-change note.

Stable boundary (unchanged, asserted in tests): `coordpy.__version__ == "0.5.20"`,
`coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI publish, `coordpy/__init__.py`
untouched, advanced work explicit-import only.

---

## § 0 — Why W132 is NOT another supply memo / selector retry / prompt-only generator lap

W123 bounded the OFFICIAL resistant-package supply; W131 found the model axis blocked on
cutoff DISCLOSURE (every reachable stronger code model is UNKNOWN-cutoff ⇒ DEV_ONLY). Both
caps share one upstream dependency: a contamination-RESISTANT battlefield we either inherit
(supply-capped) or certify against a disclosed cutoff (disclosure-capped). **W132 removes
that dependency by MINTING the battlefield ourselves** — freshly-generated algorithmic
problem instances that did not exist before the mint date (resistant by *construction*, for
ANY model cutoff), targeted at the failure families that actually beat the mechanism stack,
with exact executable oracles and a mechanical novelty guard. Bounded-context / compaction
/ summarization / "cram less / truncate better" remain explicit anti-patterns, NOT the path.

---

## § 1 — α / β / γ branch logic (pre-committed)

* **Lane α — resistant-by-construction battlefield (MAIN empirical, $0).** Build
  `coordpy.resistant_by_construction_battlefield_v1` (framework + gates + manifest) +
  `coordpy.resistant_by_construction_slate_v1` (the 33-template slate), mint the
  battlefield, run ALL § 6 quality gates, apply the § 3 novelty guard, certify Maverick
  RESISTANCE (§ 9), select the deterministic mode-stratified core 30-slice, emit
  `results/w132/battlefield/battlefield_verdict_v1.json` (+ manifest). Lane α SUCCEEDS iff
  it admits **≥ 30** clean tasks; if it cannot, land the instrument anyway and make the
  blocker machine-checkable.
* **Lane β — generated-family mechanism validation + Maverick pilot (MAIN spend).** Lock
  the arms (§ 7) BEFORE any NIM. Run a small **calibration slice** first (§ 8a); only if
  the battlefield earned (§ 8) AND calibration is non-degenerate, run the cheapest honest
  Maverick pilot on the deterministic core 30-slice via the *already-validated* W120
  reflexion bench (`run_icpc_reflexion_bench_v1`) + the verbatim W108 evaluator
  (`_mlb_rates` / `_evaluate_phase2_gates`). Same exact-oracle grader, NO LLM judge. Emit
  `results/w132/pilot/`.
* **Lane γ — stronger-model gate / graphify / truth (MANDATORY, $0).** Re-derive the
  cutoff gate (§ 9); refresh graphify START + END (§ 10); register carry-forward (§ 11);
  land executable code, not docs only.

Branch order: α (mint + gates, $0) → γ-gate ($0) → β-calibration (cheap NIM) → β-pilot
(NIM iff earned) → γ-graphify/truth.

---

## § 2 — Battlefield construction (LOCKED; already emitted, $0)

Mint params LOCKED: `global_seed = 132`, `minted_date = 2026-06-02`,
`exec_timeout_s = 8.0`. Each of the 33 templates ships THREE complete stdin/stdout
programs — `ref_source` (the scalable correct oracle whose stdout IS the answer key),
`brute_source` (an INDEPENDENT obviously-correct exhaustive oracle for the small-case
cross-check), and `naive_source` (the admissible-wrong trap) — plus seeded public/hidden
case generators. The minted problems are emitted as `IcpcPilotProblemV1` so the W120
reflexion bench consumes them verbatim.

LOCKED build result (re-derivable byte-identically): **33 admitted** (COMPLEXITY_BLIND 9 /
HIDDEN_EDGE_STATE_MISS 8 / WRONG_ALGORITHM_ADMISSIBLE 8 / SEARCH_ENUM 8), 33 distinct
families, `manifest_cid 562aafbd62f550d1…`, `raw_cid 1e9a2a42f20f05ec…`, deterministic
regeneration TRUE. Core 30-slice `core_slice_cid f6a2ebed3da2f13b…`, mode-stratified
(8 / 8 / 7 / 7).

---

## § 3 — Hard-family target rule (LOCKED, enforced in code)

Every minted template carries a `mode` in the W130/W131 atlas taxonomy, and the slate
prioritises the families that materially beat the mechanism stack:

* **`WRONG_ALGORITHM_ADMISSIBLE`** — a NAMED-but-wrong technique (a greedy where DP is
  required: coins / weighted-interval / knapsack / partition / LIS / house-robber /
  max-product / LCS). Hidden case = a constructed greedy-defeating instance.
* **`HIDDEN_EDGE_STATE_MISS`** — public samples underspecify a corner (wrap / overlap /
  bracket-type / inclusivity / tie / sign / sort). Hidden case = the constructed corner.
* **`COMPLEXITY_BLIND`** — the naive IS the correct algorithm but O(N^2); it TLEs on the
  large hidden stress case while the O(N log N)/O(N) reference finishes.
* **`SEARCH_ENUM`** — a small-n exhaustive oracle is exact; the naive is a plausible
  miscount (ordered-vs-unordered / wrong recurrence / blocks-ignored).

A template is ADMITTED only if its discriminating gate (§ 6) fires in its declared mode
(TIMEOUT for COMPLEXITY_BLIND; WRONG_ANSWER otherwise).

---

## § 4 — Novelty / near-duplicate rule (LOCKED, enforced in code)

`novelty_filter_v1` REJECTS a minted problem iff **(a)** its statement char-5-gram Jaccard
with an already-accepted minted problem `>= 0.55`, or **(b)** its statement embeds an
official ICPC identity token (the W120 listing short-names, the paraphrase guard). Family-
level inspiration (textbook algorithm families) is allowed; same-problem reuse is not. No
verbatim problem-statement reuse, no accepted-solution reuse, no official secret-case reuse
(all content is procedurally authored from scratch). The guard is validated by a planted
near-duplicate positive control in the tests.

---

## § 5 — No-leakage rule (LOCKED, enforced in code)

1. The model under test sees ONLY `statement` + the PUBLIC `samples`. `ref_source`,
   `naive_source`, `brute_source`, and the hidden `secret_cases` are NEVER placed in any
   model-facing prompt (`MintedProblemV1.to_pilot_problem` ships only statement + samples
   + the hidden grader; the bench's reflexion feedback uses ONLY public samples + the
   judge verdict bit + the executor stderr tail — verbatim the W120 anti-cheat).
2. The answer key is an executable program's stdout, not a hand-written constant; grading
   is the audited `grade_icpc_candidate_case_v1` (token-diff / float oracle), exit-0-iff-
   EVERY-secret-case-passes, NO LLM judge.
3. Public samples are a SUBSET of the secret cases (split integrity gate, § 6); the hidden
   discriminating cases are disjoint from the public samples.

---

## § 6 — Oracle / quality-gate rule (LOCKED; a problem is ADMITTED iff ALL pass)

* **exact-oracle self-test / small-vs-large agreement** — `brute_source` (independent,
  obviously-correct) equals `ref_source` (the answer key) on every case below
  `brute_cap_tokens`; the cross-check must be non-vacuous (≥ 1 checked).
* **reference solvable** — `ref_source` finishes within `exec_timeout_s` with rc 0 on
  every case (incl. the large stress case): the field is solvable.
* **discriminating-hidden-case** — `naive_source` PASSES every public sample (looks-right)
  AND FAILS ≥ 1 hidden case in its declared mode (TIMEOUT for COMPLEXITY_BLIND, else
  WRONG_ANSWER): the field genuinely RESISTS the admissible-wrong approach.
* **public/hidden split integrity** — samples ⊆ secret AND ∃ a secret case not in samples.
* **deterministic regeneration** — re-minting at the same seed yields a byte-identical
  manifest CID.
* **pass-fail-only** — `kind ∈ {passfail, passfail_float}`; no scoring/interactive.

---

## § 7 — Same-budget evaluation rule (LOCKED before NIM)

The arms are the *already-validated* W88/W89 three-arm same-budget mechanism (the one that
earned W89 + W105), run by `run_icpc_reflexion_bench_v1` at `K = 5`, 1 seed
(`132001`), `sampling_temperature = 0.7`, `max_tokens_per_call = 1536`,
`executor_timeout_s = 8.0`:

* **A0** — single-shot (1 model call, temperature 0).
* **A1** — same-budget self-consistency: K i.i.d. samples, pass iff ANY passes
  (oracle pass@K).
* **B** — same-budget sequential reflexion: K attempts, each conditioned ONLY on public
  feedback (judge verdict bit + stderr tail + public-sample results).

Same model on every arm; A1 and B spend EXACTLY K calls (no early stop, no selective
retry). **B is the canonical validated stack** (the only stack that earned a retirement);
the W128/W129/W130 refinements are NOT validated and are deliberately NOT used as B (using
them would be an unvalidated confound). Per problem 1 + K + K = 11 calls ⇒ 30 × 11 = **330
NIM calls** for the full pilot (+ the § 8a calibration). Gates scored by the verbatim W108
`_mlb_rates` + `_evaluate_phase2_gates`.

---

## § 8 — Maverick pilot earn rule (LOCKED)

**8a — Calibration (cheap, required before the full pilot).** Run A0/A1/B on a small
deterministic calibration slice (the first **6** core-slice problems, mode-spanning;
6 × 11 = 66 NIM calls). The pilot is **non-degenerate-cleared** iff: the oracles behave
(no executor crashes on the arms), A0 is not already saturated, and A1 is in a useful band
`0 < A1 < 0.90` with at least one attempt-0 failure (so reflexion can be invoked). If the
calibration is degenerate (A1 ≈ 0 floor, or A1 ≥ 0.90 saturated), do NOT force the full
pilot; register the calibration outcome and STOP.

**8b — Full pilot earned iff ALL hold:** (1) battlefield admits ≥ 30 (§ 2, TRUE); (2) all
§ 6 quality gates pass (TRUE); (3) Maverick is RESISTANCE-certified on the minted boundary
(§ 9, TRUE); (4) the § 7 evaluation rule is locked before spend (this document); (5) § 8a
calibration is non-degenerate-cleared. If earned, run the full 30-slice pilot once
(330 calls). The core-slice CID `f6a2ebed3da2f13b…` is asserted before spend (slice-drift
guard).

**8c — Outcome mapping (pre-committed; filled ONLY from JSON):**
* `PASS_MECHANISM_DRIVEN` (9/9 Phase-2 gates AND MLB-1 ≥ 33% AND MLB-2 ≥ 33%, i.e.
  B − A1 ≥ +5pp ∧ B − A0 ≥ +5pp ∧ A1 < 90% ∧ per-problem majority ≥ 16/30 ∧ reflexion
  load-bearing) ⇒ **clean resistant-by-construction superiority, single seed** ⇒ W133 =
  multi-seed same-budget confirmation to reach W89/W105 retirement-grade. A retirement is
  registered ONLY on a clean multi-seed `PASS_MECHANISM_DRIVEN` (NOT on a single seed).
* `PASS_NON_MECHANISM_DRIVEN` (9/9 but MLB sub-gate fails) ⇒ margin present but reflexion
  not load-bearing ⇒ register bounded; NOT a clean mechanism win.
* `FAIL` ⇒ the minted resistant-by-construction field does NOT yield same-budget
  superiority ⇒ register `W132-L-RESISTANT-BY-CONSTRUCTION-PILOT-CAP`; the bounded
  contamination-EXPOSED-HumanEval-family-at-70B ceiling STANDS; resistant superiority
  still 0 clean.

**No retirement is registered by W132 on a single seed** regardless of branch. W89 + W105
remain the only two unless a clean multi-seed mechanism-driven pass is later earned.

**§ 8d — Infra-forced model substitution (LOCKED PRE-SPEND, amended 2026-06-02).** The
locked target Maverick (`meta/llama-4-maverick-17b-128e-instruct`) is **infra-down this
session** — machine-checked, not assumed: `GET /v1/models` returns 200 in 0.15 s and lists
maverick, and `meta/llama-3.1-8b-instruct` returns in 0.42 s with the SAME key/endpoint/
request shape, but every maverick chat/completions call (streaming AND non-streaming, 8 and
16 tokens) returns **0 bytes** and times out (45–75 s, `time_starttransfer=0`) ⇒ a
model-specific server-side outage (NOT auth, NOT the endpoint, NOT our code; a new key
cannot fix it). Therefore the frontier target is substituted **pre-spend** to
**`meta/llama-3.3-70b-instruct`** — which is (a) reachable (HTTP 200, ~7 s for a 512-token
code generation), (b) **primary-KNOWN cutoff (~Dec-2023)** so the 2026-06-02 mint date
strictly post-dates it (resistant by date AND construction), and (c) **the exact model that
earned the W105 retirement (+7.00 pp on EXPOSED HumanEval+)** — making this the MOST
on-target transfer test (does the W105-retiring same-budget mechanism beat A1 on a
resistant-by-construction field at its own 70B scale?). The substitution is forced by infra
and made BEFORE any pilot result; the locked core 30-slice (`f6a2ebed…`), the § 6 quality
gates, the § 7 same-budget eval rule, and the § 8c outcome mapping are UNCHANGED. Maverick
remains the eventual push-button cross-scale check when its deployment recovers.

---

## § 9 — Frontier-eligible vs dev-only rule (Lane γ, LOCKED)

* **Maverick** (`meta/llama-4-maverick-17b-128e-instruct`, KNOWN Aug-2024) is the default
  frontier target: the mint date `2026-06-02` strictly post-dates its cutoff, so the field
  is resistant for it by DATE, and additionally by CONSTRUCTION (the instances are fresh).
  `certify_resistance_v1` records both, corroborated by the reused W114 C1..C4 gate.
* **Resistance-by-construction removes the W131 disclosure dependency**: the field is
  resistant for ANY model regardless of cutoff disclosure. An UNKNOWN-cutoff stronger model
  (Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4-119B-2603 / GLM-5) may be used ONLY
  as an explicitly-labelled `DEV_ONLY` characterization on a tiny auxiliary slice — it must
  NOT contaminate the frontier claim, which stays on Maverick (KNOWN cutoff).
* The stronger-model gate is re-derived (`decide_certification_v1`): decision CID
  `258b6ed7…` invariant, verdict `NO_CERTIFIABLE_STRONGER_MODEL`, gate CLOSED. No 405B run
  unless reachability changes and a pre-committed gate clears. The W123→W131 caps stay
  closed unless new evidence genuinely changes them.

---

## § 10 — graphify deliverables (LOCKED)

* Refresh `graphify update .` at START (built from HEAD `3708ea5d`; captured
  `results/w132/graphify/graphify_start_w132.txt`) and END (record END HEAD).
* `graphify explain` on the mined arsenal: `generator_failure_atlas_v1`,
  `public_signal_selection_oracle_v1`, `stronger_generator_slate_v1`,
  `role_diverse_algorithm_search_v1`, `resistant_capability_atlas_v1`,
  `family_scaffold_generation_v1`, `icpc_reflexion_bench_v1`,
  `coordpy_icpc_battlefield_v1`, and the NEW
  `resistant_by_construction_battlefield_v1` + `resistant_by_construction_slate_v1`.
* `graphify path stronger_generator_slate_v1 public_signal_selection_oracle_v1` +
  `graphify path generator_failure_atlas_v1 resistant_capability_atlas_v1` +
  `graphify affected stronger_generator_slate_v1`. `graphify query` secondary only.
* The new `resistant_by_construction_battlefield_v1` must create the FIRST semantic bridge
  from a minted-task GENERATOR to BOTH the validated bench (`icpc_reflexion_bench_v1`,
  `IcpcPilotProblemV1`) AND the audited grader (`coordpy_icpc_battlefield_v1`,
  `grade_icpc_candidate_case_v1`). The END graph must show these edges.

---

## § 11 — Carry-forward registration (LOCKED shape; filled ONLY from JSON)

* **W89 (+5.56) + W105 (+7.00)** remain the only two confirmed retirements unless a later
  multi-seed clean `PASS_MECHANISM_DRIVEN` is earned. W132 retires none on a single seed.
* On pilot `FAIL`: register `W132-L-RESISTANT-BY-CONSTRUCTION-PILOT-CAP` — even on a
  battlefield minted specifically in the failure families that beat the stack, with exact
  oracles + novelty guards, the same-budget mechanism does not beat A1 at 70B ⇒ a STRONGER
  statement of the bounded ceiling (the official-benchmark "maybe it was the wrong test"
  escape is closed). Record the per-mode B−A1 distribution.
* On `PASS_NON_MECHANISM_DRIVEN`: register the bounded margin (not mechanism-driven).
* On `PASS_MECHANISM_DRIVEN`: register the single-seed clean signal + the W133 multi-seed
  decision (NOT a retirement by itself).
* Always carry forward the W123→W131 caps + the new
  `W132-T-RESISTANT-BY-CONSTRUCTION-BATTLEFIELD-MINTABLE` (the instrument exists, ≥30,
  exact-oracle, novelty-clean, deterministic — the construction advance regardless of the
  pilot outcome).

---

## § 12 — W133 branch logic (pre-committed)

* If the pilot is `FAIL` ⇒ W133 = accept the registered resistant-by-construction pilot
  cap; the minted instrument STANDS as a reusable resistant battlefield (a genuinely
  different axis is the remaining lever, or an operator-greenlit DEV_ONLY stronger-model
  characterization on the minted field to localise whether the cap is generation- or
  mechanism-bound). Bounded-context / compaction remain anti-patterns.
* If `PASS_NON_MECHANISM_DRIVEN` ⇒ W133 = strengthen the mechanism or accept the bounded
  resistant ceiling.
* If `PASS_MECHANISM_DRIVEN` (single seed) ⇒ W133 = operator-greenlit multi-seed same-
  budget confirmation on the minted core slice (≥ 3 seeds); retire iff a clean multi-seed
  mechanism-driven +5.00pp margin.
* `COO-9` stays the lead path unless the evidence genuinely forces a different code-line
  move.
