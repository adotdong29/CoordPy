# RUNBOOK W129 — public-signal SELECTION ORACLE on the W128 hard-cluster miss pattern + same-family EXPOSED dev bench + targeted resistant probe only if earned

**Status: LOCKED PRE-REGISTRATION.** Written BEFORE any W129 NIM call and BEFORE any
dev-bench / probe result is interpreted. The $0-NIM recon (§ 0) is allowed before this lock
(it spends no NIM and only re-grades already-paid generations). Fill `docs/RESULTS_W129_*`
ONLY from emitted verdict JSON (the "never pre-write results" discipline —
`feedback_never_prewrite_results_before_data`). The pre-committed code rules below are the
branch authority, not any prior or hope.

`ultracode` OFF. W129 is a bounded selector/oracle + conditional-probe milestone, not a
repo-wide dynamic-workflow job. It turns ON only if the work unexpectedly expands into a genuine
dynamic-workflow problem (multiple selector families all earn live runs at once / repo-wide
selector-oracle integration / broad multi-surface external verification at once) — and only
after an explicit mode-change note.

Stable boundary (unchanged, asserted in tests): `coordpy.__version__ == "0.5.20"`,
`coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI publish, `coordpy/__init__.py` untouched,
advanced work explicit-import only.

---

## § 0 — Why W129 is NOT another generation lap (the $0 recon)

W128 proved a role-diverse algorithm SEARCH is REAL and **lifts the generation ceiling**
(pool 3/11 > plain baseline 2/11) but is **NOT EARNED**: RDA4 committed only 2/11 (net +0 =
+1 `blueberrywaffle` − 1 `pawnshop`). The load-bearing finding: **the bottleneck is the
verification-based SELECTION layer, not generation**. W129 attacks the SELECTOR directly.

The $0 recon (`scripts/run_w129_stored_pool_recon_v1.py`, verdict
`results/w129/recon/stored_pool_recon_v1.json`) reconstructs the EXACT W128 candidate pools by
REPLAYING the stored generations (keyed by `prompt_sha256`) and grades every candidate
(A0/B1/C2/D3) on the PUBLIC samples AND the official SECRET cases. It localizes the W128 miss:

| problem | public survivors | hidden-correct | hidden-wrong | separable on derived? | brute-force sketch? | W128 commit |
|---|---|---|---|---|---|---|
| `blueberrywaffle` | A0,B1,C2 | A0,C2 | B1 | **YES** | no | A0 (✓ correct) |
| `sunandmoon` | B1,C2 | B1,C2 | — | n/a (both correct) | no | B1 (✓ correct) |
| `pawnshop` | A0,B1 | **B1** | **A0** | **NO** | no | **A0 (✗ MIS-COMMIT)** |

The decisive fact: on `pawnshop` the hidden-correct `B1` and hidden-wrong `A0` are two
near-identical greedy programs that emit **byte-identical output on every public sample AND
every model-derived case**, with **no brute-force/reference sketch**. The choice is
**public-signal UNDER-DETERMINED**: a 2-way behavioural tie with no reference cannot be broken
by ANY NIM-free agreement/consensus/falsifier oracle (this is the W125 `looks_right_fails_hidden`
phenomenon at the candidate level). `blueberrywaffle` IS separable (the wrong B1 diverges on
derived cases) ⇒ a falsifier/differential selector keeps it.

Consequence for the bench math: even a PERFECT selector on the stored 11-pool commits at most
the pool ceiling 3/11 = net **+1** over baseline (2/11) — so the stored pool is a
REGRESSION-PAIR / precursor test, NOT a +2 earn bench. The +2 earn requires a FRESH EXPOSED
dev bench. W129 is NOT another battlefield pivot / scaffold retry / raw-generation search lap;
bounded-context / compaction / summarization remain anti-patterns, explicitly NOT the path.

---

## § 1 — α / β / γ branch logic (pre-committed)

* **Lane α — public-signal selection-oracle construction (MAIN mechanism, mostly NIM-free).**
  Build `coordpy.public_signal_selection_oracle_v1` (explicit-import only): the SO1–SO4 slate
  (§ 4) + a fake-selection positive control + the honest trust-machinery examination
  (`examine_trust_machinery_applicability_v1`). Run the NIM-free SO1/SO2/SO4/SOLEAD over the
  STORED 11 pools (§ 5 regression pair). Mine `mathvista_bench_v2` (verifier-final),
  `integrated_synthesis` + `role_invariant_synthesis` (disagreement/abstain),
  `integrity_trust_coupled_consensus_v1` + `trust_weighted_consensus_controller` (trust-weighted
  abstain — honest mapping or KILL). Emits `results/w129/selector_stored/stored_selector_eval_v1.json`.
  **$0 NIM.**
* **Lane β — same-family EXPOSED hard-cluster dev bench (MAIN validation, EXPOSED dev spend
  ALLOWED).** Two tiers: (β0) a CHEAP SO3 verifier-final probe re-judging the STORED pool-bearing
  problems (≤ ~12 NIM) — the only lever that can break a public-signal-under-determined tie;
  (β1) — fired ONLY if β0 shows the verifier cashes out the stored pool ceiling without
  mis-commit — a FRESH EXPOSED hard-cluster dev bench (W128 § 5 discipline) comparing
  plain / scaffold / W128-RDA4 / W129 selector arms at MATCHED budget, with the R2′ earn gate
  (§ 6). Kill the selector sharply if fake or weak. Emits
  `results/w129/dev_bench/selector_dev_bench_verdict.json` (+ the β0 probe verdict).
* **Lane γ — targeted resistant probe / stronger-model gate / truth.** Re-check primary cutoffs
  (§ 9). A targeted resistant probe is earned ONLY iff T1 ∧ T2 (§ 7). If earned, run the
  smallest honest cluster-matched probe first. Else **$0 resistant NIM**, register the cap.
  Keep the W123–W128 caps closed unless new evidence genuinely changes them. Refresh graphify
  START + END (§ 10); land executable code, not docs only.

Branch order: α ($0, stored regression pair) → β0 (cheap SO3 stored probe) → β1 (fresh dev
bench, conditional) → γ (T1∧T2 → targeted resistant probe; else $0).

---

## § 2 — Hard-cluster SELECTOR TARGET rule (LOCKED)

* The W129 selector target = the **W128 hard-cluster miss pattern**: the pool-bearing problems
  where the GENERATION ceiling is positive (≥1 candidate passes secret) but RDA4 either
  mis-committed (`pawnshop`) or the pool exceeds the commit. The selector's job is to convert a
  POOL-ONLY hidden winner into a committed win WITHOUT a hidden-test oracle or leakage, and to
  STOP mis-commits.
* Stored regression fixtures (pinned by the W128 dev-bench verdict + recon):
  `blueberrywaffle` (separable pool win — must KEEP), `pawnshop` (under-determined mis-commit —
  must STOP: cash out B1 if a real public signal separates it, else ABSTAIN — never re-commit
  the wrong A0), `sunandmoon` (both-correct — must KEEP).
* The fresh EXPOSED dev-bench target families = `NON_SCAFFOLDABLE_FAMILIES` with
  `simulation_grid` PRIORITY (graph_flow EXPOSED supply = 0 — `W128-L-GRAPH-FLOW-EXPOSED-SUPPLY-CAP`).
* The atlas reference family signal is OFFLINE-ONLY and NEVER model-facing.

---

## § 3 — No-leakage + accepted-solution tripwire rule (LOCKED, enforced in code)

1. **NEVER** expose a target's accepted solution, secret input, secret answer, or validator
   internals to ANY model-facing selector/verifier path. Every selector prompt (SO3
   verifier-final) and every oracle CASE uses ONLY: the target's PUBLIC statement, PUBLIC
   samples, public-signal-derived cases (model DERIVED counterexamples + deterministic
   FORMAT-PRESERVING sample mutations), candidate SOURCE code, candidate OUTPUTS, and typed
   failure digests (`parse_failure_digest_v1`).
2. Auto-derived cases are FORMAT-PRESERVING mutations of the PUBLIC samples only (token
   rotations of integer lines; deterministic, no `Math.random`); they are used DIFFERENTIALLY
   (a case that breaks every survivor is a bad mutation, ignored) so a malformed mutation can
   never falsify a correct candidate.
3. The accepted solution remains a TRIPWIRE only: every committed/pool candidate passes the
   W126/W127 provenance-aware leakage guard (`SynthesisLeakageGuardV1` + the contiguous-block
   `reproduces_accepted_block_v1`). A run failing any guard is dropped, never counted as a win.
   Positive control preserved (a planted accepted solution is caught).
4. If ANY leakage check fails on the EARNING set ⇒ the earn is INVALID and the lane is killed
   honestly; resistant spend is NOT earned.

---

## § 4 — Selector / oracle slate (LOCKED before results)

New module `coordpy.public_signal_selection_oracle_v1`. Inputs = the SAME `(problem, role
artifacts, candidate impls)` W128 produced (selection is over generations, not new generation).

* **SO1 — public-derived falsifier stack** (NIM-free). Run public survivors on the model
  DERIVED cases + auto FORMAT-PRESERVING mutations; **differentially** falsify a candidate that
  crashes / TLEs / format-violates / contradicts the model's predicted-expected on a case where
  ≥1 other survivor stays clean. Commit the unique survivor; else first-survivor (the W128 RDA1
  contrast), `evidence_used` iff the falsifier eliminated ≥1 survivor.
* **SO2 — differential disagreement selector** (NIM-free; REAL bridge to `integrated_synthesis`).
  Group survivors by behaviour signature over the case union; producer axis = signature-majority,
  trust axis = falsifier-survivor set; combine via `select_integrated_synthesis_decision`.
  Commit the agreed rep; **ABSTAIN** on `INTEGRATED_AXES_DIVERGED_ABSTAINED` or on a no-majority
  tie with no discriminator (do NOT mis-commit — the pawnshop discipline).
* **SO3 — verifier-final chooser** (needs `gen`; mines the `mathvista_bench_v2`
  verifier-final pattern). A final model call SEES every candidate + each one's public/derived
  verdict + the invariants and makes a REAL final CHOICE (`CHOOSE <label>`) or ABSTAINs, on
  public-signal evidence only (NEVER the secret cases / accepted solution). A choice naming a
  non-survivor or unparseable ⇒ safety ABSTAIN. SO3 carries a SKEPTICAL prior
  (`W96-L-MATHVISTA-BENCH-V2-VLM-VERIFIER-FINAL-K5-CAP`: verifier-final did NOT earn cross-modally).
* **SO4 — trust-weighted abstain ensemble** (NIM-free). Each survivor → trust = falsifier-survival
  fraction; integrity = format/crash-clean. Realizes the `integrity_trust_coupled_consensus_v1`
  integrity-penalty + trust-weighted-quorum + ABSTAIN CONCEPT natively over HONEST
  code-correctness trust signals; commit the max-trust integrity-clean survivor iff its trust
  SHARE clears quorum (0.5) AND a strictly-lower-trust survivor exists, else ABSTAIN. The
  substrate `TrustWeightedConsensusController` (latent `MergeableLatentCapsuleV3` / cosine /
  merge) literal bridge is KILLED as latent-specific fake-different
  (`examine_trust_machinery_applicability_v1`, machine-checkable — the W128 W79 lesson).
* **SOLEAD — composition** = SO1 falsifier → SO2 differential → (SO3 verifier on residual tie if
  `gen` present). The LEAD arm.

**Fake-selection kill (NIM-free positive control).** `fake_selection_control_v1`: on a
behaviourally-identical 2-survivor tie, SO2 and SO4 MUST ABSTAIN with `evidence_used = False`
(no discriminating signal). A win driven by alphabetical-first / coin-flip (the W128 pawnshop
mis-commit pattern) does NOT count — a real SO commit must cite a discriminating signal (a
falsifier eliminated a survivor, a strict majority, or a verifier choice).

**Kill rules (honest):** a selector arm is killed if (i) the control fails (it commits a
no-evidence tie); (ii) on the fresh dev bench it is no better than W128-RDA4 net (selection =
prompt decoration); (iii) its only gains are trivial parse/format fixes; (iv) a win depends on
same-problem leakage; (v) SO3's choices are not better than chance at separating
hidden-correct from hidden-wrong survivors on the stored pool (it restates preferences, not a
real verification).

---

## § 5 — Stored regression-pair rule for `blueberrywaffle` / `pawnshop` (LOCKED)

Over the STORED W128 11 pools, NIM-free, BEFORE any fresh spend:

* **R-KEEP-BLUE**: the lead arm must COMMIT a hidden-correct `blueberrywaffle` candidate
  (A0 or C2) — keep the W128 unique win.
* **R-KEEP-SUN**: the lead arm must COMMIT a hidden-correct `sunandmoon` candidate (B1 or C2).
* **R-NO-MISCOMMIT-PAWN**: the lead arm must NOT commit the hidden-wrong `pawnshop` A0. The
  acceptable outcomes are (a) COMMIT B1 (cash out — only if a real public signal separates it),
  or (b) ABSTAIN. Re-committing A0 = the W128 mis-commit = regression-pair FAIL.
* **MIS-COMMITS metric**: across all 11, the lead arm's mis-commit count (committed a
  hidden-wrong candidate) is the key SAFETY metric; a principled selector drives it BELOW
  W128-RDA4's (which mis-committed pawnshop).

The NIM-free selectors are EXPECTED (per § 0) to keep blue+sun and to ABSTAIN on the
under-determined pawnshop tie (B1≡A0 on all public signal) — i.e. resolve the regression pair
in the **mis-commit→abstain** sense, NOT the cash-out sense, unless a public signal is found.
Cashing out pawnshop's B1 requires SO3 (a model verifier) — the β0 probe (§ 6) is the test.

---

## § 6 — EXPOSED hard-cluster dev-bench earn rule (LOCKED; R2′)

**β0 (cheap SO3 stored probe).** Re-judge the STORED pool-bearing problems (`pawnshop`,
`blueberrywaffle`, `sunandmoon`; optionally all 11 with ≥2 survivors) with ONE SO3
verifier-final call each (≤ ~12 NIM, EXPOSED dev — operator-greenlit). β0 PASSES iff the
verifier COMMITS pawnshop's `B1` (cashes out the under-determined tie) AND keeps
blueberrywaffle + sunandmoon AND has 0 mis-commits. If β0 fails (verifier abstains / picks A0 /
picks a non-survivor) ⇒ the in-loop signal is non-discriminating EVEN for a model judge ⇒
register the under-determination cap; **do NOT fire β1**; $0 beyond β0.

**β1 (fresh EXPOSED hard-cluster dev bench).** Fired ONLY if β0 passes. Reuse the W128 hard-cluster
EXPOSED bench discipline (`simulation_grid` + `adhoc_math` + `greedy_scheduling`; graph_flow
supply 0). Arms at MATCHED budget on `meta/llama-4-maverick-17b-128e-instruct`:
plain (K=5 i.i.d.) / scaffold (W127) / W128-RDA4 / **W129 SOLEAD** (with SO3 reshape: 1 ANALYZE
+ 3 IMPLEMENT + 1 VERIFIER_FINAL, matched K=5). Grade on official secret (public-sample
prescreen). Budget ceiling ≤ ~250 NIM. Canary first.

**R2′ EARNED iff ALL hold:**
* **R2a′** `net_solead_gain ≥ +2` over the plain baseline (the W128 DEV_MIN_NET_GAIN bar), AND
* **R2b′** the unique solves span ≥ 2 hard families OR include ≥ 1 `simulation_grid` solve, AND
* **R2c′** every unique-solve run is diversity-REAL + leakage-clean, AND
* **R2d′** the gain is nontrivial (not exclusively parse/format fixes), AND
* **R2e′** `net_solead_gain > net_rda4_gain` AND the SOLEAD wins are driven by the SELECTOR
  layer (a committed win the W128-RDA4 selector abstained-on or mis-committed — i.e. the
  selector cashed out a pool-only winner), not merely by generation variance, AND
* **R2f′** SOLEAD mis-commits ≤ W128-RDA4 mis-commits (no safety regression).

A weak/confounded tie is NOT an earn. A close edge is NOT sufficient. If R2′ FAILS ⇒ register
`W129-L-PUBLIC-SIGNAL-SELECTION-*-CAP`; **$0 resistant NIM**. Record which signal was
load-bearing (falsifier / differential / verifier-final / abstain discipline).

---

## § 7 — Targeted resistant-probe earn rule (LOCKED; T1 ∧ T2)

Fresh resistant hosted spend is earned ONLY iff BOTH:
* **T1** — Lane β1: the W129 SOLEAD selector shows REAL dev-bench value (R2′ EARNED), AND
* **T2** — the resistant atlas identifies a cluster-matched subset where the selector logic is
  specifically relevant (the EXPOSED-earned family ∩ the resistant hard target families ≠ ∅;
  prioritize resistant `simulation_grid` — where W128 showed the pool-only hidden winner).

If T1 ∧ T2: run the **smallest honest cluster-matched targeted resistant probe** first
(resistant hard problems in the earned family, ≤ 1 seed, SOLEAD mechanism), grading committed
+ pool on the official secret cases vs the old W120/W126 pool (0 on the 22 uniformly-unsolved).
Probe budget ceiling ≤ ~45 NIM. `targeted_new_solves` = #(cluster-subset problems SOLEAD
commits on secret that the old pool never solved, leakage-clean). If ≥ 1 ⇒ define whether a
broader resistant pilot is earned (separate, explicitly-flagged; NOT auto-run). If = 0 ⇒
register the resistant selection cap; no broader pilot.

If T1 ∧ T2 do NOT both hold ⇒ **$0 additional resistant NIM**; register the exact blocker. No
new n=30 seed-chasing. No stronger-model spend unless § 9 opens. No 405B. No reopening MBPP+ V2
/ frozen cross-modal / the closed Llama-3.1 rescue / APPS main-lane NIM. No dirty exposed
benchmark sold as a frontier win. A close blip / same-problem leak / one-trick fix is NOT a win.

---

## § 8 — Exposed-control earn / no-earn rule (LOCKED)

The matched exposed-frontier *control* pilot (W121-style; distinct from the EXPOSED *dev bench*
authorized by § 6) is downstream and NOT automatic. Buy it ONLY if a targeted resistant probe
is RUN AND produces a real interpretation-changing result that an exposed control would resolve
(mechanism-vs-exposure). If the probe is not earned/not run, or is a clean negative ⇒ exposed
control NOT earned and NOT bought (resistant-first).

---

## § 9 — Per-model disclosure status + certification rule (Lane γ, LOCKED)

Reuse `coordpy.stronger_model_cutoff_certification_v1` (C1∧C2∧C3∧C4; decision CID `258b6ed7`,
invariant W114→W128). Re-check PRIMARY sources for: Maverick, Qwen3-Coder-480B, DeepSeek-V4-pro,
Mistral-Small-4-119b-2603, GLM-5, and any newly reachable same-budget-comparable model. A model
SUPERSEDES Maverick as the hosted target ONLY if it becomes primary-KNOWN (disclosed cutoff) AND
certifiable on the matched ICPC family. Standing prior: **{KNOWN:1 (Maverick, Aug-2024),
UNKNOWN:4}** ⇒ Maverick is the only certifiable hosted target. No 405B run unless reachability
changes and a pre-committed gate clears. Emit `results/w129/stronger_model_gate/gate_recheck_v1.json`.

---

## § 10 — graphify deliverables (LOCKED)

* Refresh `graphify update .` at START (built from HEAD `0f47803`) and END (record END HEAD).
* `graphify explain` on the mined arsenal: `role_diverse_algorithm_search_v1`,
  `integrated_synthesis`, `role_invariant_synthesis`, `integrity_trust_coupled_consensus_v1`,
  `trust_weighted_consensus_controller`, `mathvista_bench_v2`, `executor_grounded_patcher_v1`,
  `resistant_capability_atlas_v1`, and the NEW `public_signal_selection_oracle_v1`.
* `graphify path public_signal_selection_oracle_v1 integrated_synthesis` +
  `graphify path public_signal_selection_oracle_v1 integrity_trust_coupled_consensus_v1` +
  `graphify affected public_signal_selection_oracle_v1`. `graphify query` only as a secondary
  claim-surface finder.
* The new module must create the FIRST semantic bridge from the verifier-final
  (`mathvista_bench_v2`) + trust-coupled-consensus (`integrity_trust_coupled_consensus_v1`)
  machinery onto the ICPC resistant-code SELECTION path; the END graph must show its edges.

---

## § 11 — Carry-forward registration (LOCKED shape; filled ONLY from JSON)

* **W89 (+5.56) + W105 (+7.00)** remain the only two confirmed retirements unless the targeted
  resistant probe earns AND a (separately-defined) broader pilot clears the +5.00pp clean-
  superiority bar. W129 retires none unless the JSON says so.
* On β0 fail (verifier cannot cash out the under-determined pawnshop tie): register
  `W129-L-PUBLIC-SIGNAL-SELECTION-UNDER-DETERMINED-CAP` — on the regression fixture the
  hidden-correct and hidden-wrong public-survivors are behaviourally identical on ALL
  public-derivable signal AND a model verifier cannot break the tie ⇒ NO public-signal
  selector (NIM-free or model-based) cashes out the pool ceiling; the W128 selection cap is a
  public-signal INFORMATION limit, not a weak-selector artifact (the selection-information
  sibling of the W123→W128 cap taxonomy). Also register the SAFETY positive: the W129
  abstain-disciplined selector converts the W128 mis-commit into an ABSTAIN (0 mis-commits).
* On β1 run + R2′ fail: register `W129-L-SELECTION-ORACLE-EXPOSED-DEV-BENCH-NOT-EARNED` (REAL
  selector, does not beat W128-RDA4 net on held-out EXPOSED hard clusters).
* On T1 ∧ T2 with `targeted_new_solves = 0`: register
  `W129-L-RESISTANT-SELECTION-ORACLE-CAP`.
* On T1 ∧ T2 with `targeted_new_solves ≥ 1`: register the new-solve evidence + broader-pilot
  decision (NOT a retirement by itself).
* Always carry forward `W128-L-GRAPH-FLOW-EXPOSED-SUPPLY-CAP`. Named claims filled ONLY from
  the emitted verdict JSON.

---

## § 12 — W130 branch logic (pre-committed)

* If β0 fails (under-determination) ⇒ W130 = accept the public-signal selection-information cap;
  the honest remaining lever is NOT a better in-loop selector but a STRONGER GENERATOR (a
  code-competent local model / a primary-KNOWN reachable stronger-than-Maverick model that
  produces fewer near-correct-but-wrong candidates AND selects better) or a genuinely different
  axis. Bounded-context / compaction remain anti-patterns.
* If β0 passes but R2′ fails on β1 ⇒ W130 = the selector is real but does not earn on held-out
  EXPOSED hard clusters; accept the bounded ceiling + the registered dev-bench cap; stronger /
  code-competent model.
* If R2′ holds but T2 fails ⇒ W130 = the selector is a real same-family mechanism but the
  resistant field's missing capability is not selection-addressable in the matched cluster.
* If T1 ∧ T2 and `targeted_new_solves = 0` ⇒ W130 = accept the resistant selection cap;
  stronger / code-competent model.
* If T1 ∧ T2 and `targeted_new_solves ≥ 1` ⇒ W130 = define + (operator-greenlit) run the broader
  cluster-matched resistant pilot; retire iff a clean +5.00pp multi-seed same-budget margin.
* `COO-9` stays the lead path unless the evidence genuinely forces a different code-line move.
