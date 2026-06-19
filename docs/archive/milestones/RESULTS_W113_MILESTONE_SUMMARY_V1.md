# W113 — Milestone summary (clean resistant-for-Llama-4 benchmark + earned Maverick pilot → exposure confirmed)

**One line:** W113 built a benchmark VERIFIABLY contamination-resistant for
Llama-4-Maverick (date-filtered LiveCodeBench, all problems strictly after the
Aug-2024 cutoff), earned the cheapest honest Maverick pilot on the EXACT W108
slice, and the W112 +10.00 pp **collapsed to +0.00 pp (FAIL)** — confirming the
+10 pp was contamination EXPOSURE, not a capability reopening of resistant
superiority. **W89 + W105 remain the only two retirements; W113 adds none; the
contamination-confound is STRENGTHENED by the sharpest dissociation yet (a
within-model exposed→resistant flip), still not proven.**

---

## The three lanes

### Lane α — clean resistant-for-Llama-4 main lane (LIVE; one earned expensive run)

* **Built the machine-checkable resistance rule first** (`coordpy.livecodebench_resistant_slice_v1`):
  RESISTANT-for-Maverick ⟺ `contest_date > 2024-08-31` (strictly after the stated
  August-2024 month; the entire ambiguous August window is EXCLUDED). A model-cutoff
  registry tags each cutoff KNOWN / ESTIMATED / UNKNOWN and certifies resistance only
  against a KNOWN cutoff.
* **Proved date integrity** (NIM-free preflight `run_w113_resistant_slice_preflight.py`,
  verdict CID `6f30990c…`): the `release_v6` functional subset is **63/63 resistant**
  for Maverick — every problem dated 2025-01-11..2025-04-05, **0 excluded** (0 missing,
  0 unparseable, 0 in-August). The deterministic resistant 30-slice CID is `2afc318c…`
  == the **W108 slice CID**, so the date filter did not perturb the problem set — the
  Maverick pilot runs on the EXACT problems 70B ran (model scale = the only variable).
* **Earned + ran the cheapest honest pilot** (Maverick, the W108 slice, 1×30×K=5 =
  330 calls, ~55 min): **A0 30.00 / A1 50.00 / B 50.00 %; B − A1 = +0.00 pp; 7/9
  gates; MLB-1 = 63.33 % PASS; MLB-2 = 21.05 % FAIL ⇒ `FAIL` → `EXPOSURE_CONFIRMED`.**
  Reflexion was genuinely invoked (more than at 70B) but rescued 4 / regressed 2 / net
  0; Maverick's A1 (50 %) was *below* 70B's (63.33 %) on the resistant slice.
* **Verdict:** the W112 +10 pp was EXPOSURE. The clean 2×2 is complete; resistant
  superiority is 0/2 across BOTH scales.

### Lane β — tier-2 stronger-model readiness (NIM-free; no spend earned)

* **Locked the ranking** (`coordpy.tier2_readiness_v1`): tier-2 = Qwen3-Coder-480B
  (1) → DeepSeek-V4-pro (2) → Mistral-Small-4-119B (3).
* **Locked the same-filtered-slice applicability rule** + the spend rule (delegated to
  the resistant-slice certification): a tier-2 pilot is worth NIM iff the main lane
  earns escalation AND ≥1 tier-2 model has a CERTIFIABLY-resistant slice.
* **Verdict (`results/w113/tier2_readiness/tier2_readiness.json`):** all three tier-2
  cutoffs are **UNKNOWN** (Qwen3-Coder released 2025-07, cutoff undisclosed;
  DeepSeek-V4-pro 2025+; Mistral-Small-4 "2603" = 2026-03) and plausibly overlap /
  post-date the 2025-01..04 slice ⇒ **NONE is certifiably resistant on the pinned
  corpus** ⇒ **tier-2 spend BLOCKED on a missing instrument under EVERY outcome.**
  `$0` tier-2 NIM. The next instrument for any tier-2 follow-up is a LATER date-filtered
  LiveCodeBench slice (release_v7+) with problems strictly after that model's
  first-KNOWN cutoff. This is the model-cutoff-relativity lesson made into a spend gate.

### Lane γ — graphify / claim-discipline

* graphify refreshed at start (HEAD `2985b55` → rebuilt to HEAD `00210b7`, **0 token
  cost**) and at end (this HEAD). `explain`/`path`/`affected` used on the reflexion
  bench runners for file selection + dependency checks; `query` located the
  contamination/claim surface; `explain` run on the new W113 modules after the
  end-refresh.
* Claim surfaces tightened (registry / status / honesty / consolidated narrative /
  CHANGELOG / new W113 contamination-control framing) so the model-cutoff-relativity
  lesson + the resistant-superiority-fails-across-scales boundary are defensible.

---

## Truth surface after W113

* **Confirmed retirements: still exactly TWO** — W89 (base HumanEval, +5.56 pp) + W105
  (HumanEval+, +7.00 pp), both `meta/llama-3.3-70b-instruct` @ 70B, contamination-EXPOSED
  HumanEval-family. W113 adds NONE and retires NONE.
* **Resistant superiority: still 0 clean demonstrations, now confirmed across BOTH
  scales.** Reflexion 0/2 at 70B (W108 −3.33 / W110 +0.00) + Maverick resistant FAIL
  (W113 +0.00); M3 fair-strengthening sub-floor at 70B.
* **Contamination-confound: STRENGTHENED a fourth time (sharpest yet), still NOT
  proven.** The within-model exposed→resistant flip (+10.00 → +0.00, same Maverick +
  same mechanism) is the cleanest dissociation; single-seed; difficulty not fully
  excluded.
* **The W112 +10 pp is now firmly an EXPOSED-column artifact.**
* Still NOT cross-class, NOT a resistant win, NOT MBPP-family, NOT cross-modal, NOT
  "context solved".

## Entitlement delta

NOT entitled to a stronger SUPERIORITY/retirement claim (no resistant win at either
scale; no new retirement). Entitled to a slightly stronger **contamination-confound**
claim (a fourth, cleanest within-model dissociation point), still short of proof. The
bounded two-retirement contamination-EXPOSED-HumanEval-family-at-70B claim is HARDER
than before W113.

## W114 (made obvious by W113)

**W114 = accept the bounded contamination-EXPOSED claim as the honest code ceiling and
pursue a GENUINELY DIFFERENT axis** (not another exposed rerun; not another same-scale
resistant reflexion pilot; tier-2 ONLY if a per-model-resistant slice is fetched +
certified). `COO-9` stays lead.

## Discipline / boundary

* 23rd consecutive preflight/earn-discipline validation (W93–W113): runbook locked
  before any NIM (incl. the canary); resistance rule + preflight built NIM-free; the one
  expensive run earned under § 4.
* Stable boundary preserved: `0.5.20` / `coordpy.sdk.v3.43`; no PyPI;
  `coordpy/__init__.py` untouched; 3 new explicit-import-only modules + 3 scripts; the
  pilot reused the canonical gate evaluator + the W108 NIM generator (namespace import).
  31 new W113 tests + 42 reused-module regression tests pass.
* `COO-9` stays the lead path.

## Anchors

`docs/RUNBOOK_W113.md`, `docs/RESULTS_W113_RESISTANT_PILOT_V1.md`,
`docs/CONTAMINATION_CONTROL_FRAMING_W113_V1.md`,
`results/w113/resistant_slice_preflight/`, `results/w113/resistant_pilot/`,
`results/w113/tier2_readiness/`.
