# W112 — Milestone summary (stronger-model resistant-code gate + NIM-free M3 strengthening + bounded-claim discipline)

**One line:** the cross-scale-UP gate went LIVE for the first time (Llama-4-Maverick
reachable where 405B is 404×6); the earned BigCodeBench pilot reopened a +10.00 pp
same-budget reflexion margin — but on a benchmark that is contamination-EXPOSED
for Maverick (Aug-2024 cutoff > 2024-06 release), so it is a CONFOUNDED
exposed-column result, NOT a clean resistant reopening; the NIM-free Lane β shows
no fair M3 strengthening can clear the 33 % floor (structural). **W89 + W105
remain the only two confirmed retirements; W112 adds none; the contamination-confound
is STRENGTHENED a third time (first within-benchmark flip).**

---

## The three lanes

### Lane α — stronger-model resistant-code MAIN lane (LIVE; one earned expensive run)

* **Reachability sweep** (`run_w112_stronger_model_reachability_sweep_v1.py`,
  decision CID `a654956b…`): GET the live 118-model NIM catalogue + probed 13
  candidates. **405B → 6th consecutive 404.** 4 eligible reachable stronger
  targets: **Llama-4-Maverick** (tier-1, SELECTED per the locked § 1α
  Llama-family rule), Qwen3-Coder-480B, DeepSeek-V4-pro, Mistral-Small-4-119B (all
  tier-2). The cross-scale-UP gate is LIVE for the first time since the axis opened.
* **Earned pilot** (Maverick, EXACT W110 30-slice CID `b69bf3a0…`, K=5, reflexion B;
  §1α-earn canary confirmed the plain code path): **B − A1 = +10.00 pp; 9/9 core
  Phase-2 gates; MLB-2 = 37.5 % PASS; MLB-1 = 26.67 % FAIL ⇒
  `PASS_NON_MECHANISM_DRIVEN`**; 3 clean rescues (`/15`,`/26`,`/51`), 0 regressions.
* **The confound** (grounded): BigCodeBench 2024-06 is resistant for
  Llama-3.3-70B (~2024-01 cutoff) but EXPOSED for Llama-4 (**Aug-2024 cutoff** >
  release). The result is a structural twin of the W109 APPS exposed control
  (A0 = 73.33 % identical; same MLB-1-fail/MLB-2-pass/PASS_NON_MECHANISM_DRIVEN
  shape). The same slice flips +0.00 pp → +10.00 pp as the model cutoff crosses
  the release date. **Most parsimonious reading: exposed-column, not a clean
  resistant reopening.** Adds NO retirement.

### Lane β — NIM-free M3 strengthening lane (killed at $0; structural)

* **Harder mining** (`mine_w112_fair_reachability_v1.py`): on the W110 MLB-2
  denominator (12 invoked), the reliably fair-reachable ceiling is **8.3 %** (1/12);
  the best-conceivable bound (33.3 %, 4/12) only TOUCHES the floor; **58 %
  mock/fixture-coupled** (oracle-entangled, unreachable to any fair mechanism).
* **All four fair strengthenings** (richer typed digest / multi-candidate
  aggregation / patch-rejection / doctest invariants) killed at $0 — none expands
  the reliably-reachable set. **Verdict: `NO_FAIR_STRENGTHENING_CAN_CLEAR_FLOOR`.**
  Structurally strengthens the W111 empirical sub-floor finding; W111 already
  closed the 70B fair-mechanism branch, W112 shows the close is structural.

### Lane γ — graphify / claim-discipline lane

* graphify refreshed at start (HEAD `2985b55`, no topology change) → after adding
  the 2 W112 scripts (**75,728 nodes / 241,040 edges**) → and at end of milestone.
  `explain`/`path`/`affected` used on `run_executor_grounded_patcher_bench_v1`;
  `explain` on the new Lane β mining script confirmed fair-regime reuse
  (`run_bigcodebench_executor_v1` + `parse_failure_digest_v1`).
* Claim surfaces tightened (registry / status / honesty / consolidated narrative /
  CHANGELOG / contamination-control framing) so the bounded ceiling + the
  model-cutoff-relativity lesson are defensible.

---

## Truth surface after W112

* **Confirmed retirements: still exactly TWO** — W89 (base HumanEval, +5.56 pp) +
  W105 (HumanEval+, +7.00 pp), both `meta/llama-3.3-70b-instruct` @ 70B,
  contamination-EXPOSED HumanEval-family. W112 adds NONE and retires NONE.
* **Contamination-confound: STRENGTHENED a third time, still NOT proven.**
  Exposed/large-margin: W89 +5.56 / W105 +7.00 / W109 APPS +16.67 / W112
  Maverick-BigCodeBench +10.00 (exposed-FOR-Llama-4). Resistant/flat-negative (at
  the model the benchmark is resistant for): W108 LCB-2025/3.3 −3.33 / W110
  BigCodeBench-2024/3.3 +0.00. New: a WITHIN-benchmark resistant→exposed flip via
  model cutoff. Not proof (single-seed each; capability not excluded).
* **Resistant superiority: still 0 clean demonstrations.** Reflexion 0/2 at the
  model it is resistant for; M3 fair-strengthening structurally sub-floor; the
  stronger-model +10 pp is on a for-it-exposed benchmark.
* **New methodological finding:** contamination-resistance is MODEL-CUTOFF-RELATIVE
  (`W112-T-CONTAMINATION-RESISTANCE-IS-MODEL-CUTOFF-RELATIVE`).
* Still NOT cross-class, NOT a clean cross-scale-UP resistant win, NOT MBPP-family,
  NOT cross-modal, NOT "context solved".

## Entitlement delta

The programme is **NOT** entitled to a stronger SUPERIORITY/retirement claim (no
resistant win; no new retirement). It IS entitled to a slightly stronger
CONTAMINATION-CONFOUND claim (a third dissociation point + a cleaner mechanism:
model-cutoff-relativity), still short of proof.

## W113 (made obvious by W112)

**W113 = a benchmark VERIFIABLY contamination-resistant FOR Llama-4-Maverick**
(problem/contest dates > Aug 2024; a date-filtered LiveCodeBench slice is the
natural instrument — the § 1α-bench cross-check sharpened with a post-cutoff date
filter). If the +10 pp HOLDS there ⇒ scale genuinely reopens RESISTANT
superiority (new frontier → Phase-3 bench). If it COLLAPSES ⇒ the +10 pp was
exposure; bounded claim reinforced. Tier-2 reachable models (Qwen3-Coder-480B,
DeepSeek-V4-pro) likely share the post-2024-06 cutoff confound on BigCodeBench, so
the DATE-FILTERED resistant benchmark is the load-bearing instrument. `COO-9`
stays lead.

## Discipline / boundary

* 22nd consecutive preflight/earn-discipline validation (W93–W112): runbook locked
  before any NIM (incl. the sub-second sweep); target rule locked before probing;
  M3-strengthening earn-rule locked before any β NIM; the one expensive run earned
  under §1α-earn.
* Stable boundary preserved: `0.5.20` / `coordpy.sdk.v3.43`; no PyPI;
  `coordpy/__init__.py` untouched; ZERO new `coordpy.*` modules (2 new scripts only:
  the fair-reachability miner + the reachability sweep; the pilot reused the W110
  driver verbatim). 26 reused-module tests pass.
* `COO-9` stays the lead path.

## Anchors

`docs/RUNBOOK_W112.md`, `docs/RESULTS_W112_STRONGER_MODEL_BIGCODEBENCH_PILOT_V1.md`,
`docs/RESULTS_W112_STRONGER_MODEL_GATE_AND_SELECTION_V1.md`,
`docs/RESULTS_W112_FAIR_REACHABILITY_M3_STRENGTHENING_V1.md`,
`results/w112/stronger_model_reachability/`, `results/w112/stronger_model_pilot/`,
`results/w112/fair_reachability/w110_bcb_fair_reachability.json`.
