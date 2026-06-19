# W129 — Public-Signal Selection Oracle on the W128 hard-cluster miss pattern

**Status: result note. Filled ONLY from emitted verdict JSON** (`results/w129/**`), per the
"never pre-write results" discipline. Runbook: `docs/RUNBOOK_W129.md` (LOCKED before NIM).
NIM spend: **3 calls** (the β0 SO3 verifier-final probe). Everything else $0.

## TL;DR

W128 localized its cap to the verification-based SELECTION layer (pool 3/11 > committed 2/11).
W129 attacked the selector directly. The honest result has three sharp parts:

1. **The W128 `pawnshop` "selection miss" is a COMPLEXITY bug, not a hard information limit.**
   The $0 recon (`stored_pool_recon_v1.json`) shows pawnshop's hidden-correct `B1` and
   hidden-wrong `A0` are byte-identical on every public sample AND every model-derived case
   (`separable_on_derived=False`). But `A0` is `O(N²)` (recomputes `Counter(a[i:j+1])` each step)
   and `B1` is `O(N)` incremental; the statement bounds `N ≤ 300000`. Empirically A0 TLEs and B1
   finishes in 0.10s on a large instance. So the separating signal exists — it is COMPLEXITY.
2. **A model verifier (SO3) breaks the tie; generic NIM-free stress-testing does NOT.** The β0
   SO3 verifier-final (1 model call, reads the candidate CODE) chose pawnshop `B1` citing
   exactly the O(N²)-vs-O(N) complexity-vs-constraint argument — **refuting** the strong
   "the in-loop signal is non-discriminating even for a model judge" prior. But a GENERIC NIM-free
   stress generator (format-preserving sample scale-up) does NOT crack it: exposing A0's worst
   case needs an adversarially-structured large input (one big block), which a generic scale-up
   doesn't construct (constructing it ≈ solving the problem). We deliberately did NOT hand-tune a
   bespoke worst-case (that would be overfitting the single fixture).
3. **The binding cap MOVED from SELECTION (W128) to GENERATION (W129).** A selector can only
   commit candidates the pool CONTAINS, so committed ≤ pool ceiling = **3/11 = baseline+1**, which
   is BELOW the locked **+2** earn bar — regardless of selector quality. So R2′ is **structurally
   unreachable by a selection oracle** on the hard-cluster bench; the honest remaining lever is a
   stronger GENERATOR, not a better selector. T1 is FALSE ⇒ **$0 resistant NIM**.

**W89 (+5.56) + W105 (+7.00) remain the only two retirements; W129 adds none.**

## Lane α — selection-oracle construction ($0 NIM)

`coordpy.public_signal_selection_oracle_v1` (explicit-import only) — the SO1–SO4 slate + SOLEAD
composition + a fake-selection positive control + the honest trust-machinery examination.

* **SO1 — public-derived falsifier stack.** model DERIVED cases + auto FORMAT-PRESERVING sample
  mutations + a STRESS scale-up (to the parsed constraint); DIFFERENTIAL crash/TLE/format
  falsification (a case that breaks every survivor is ignored). Auto + stress are
  FALSIFICATION-ONLY (excluded from the agreement signature — they would otherwise split correct
  candidates on malformed/huge inputs).
* **SO2 — differential disagreement** (REAL bridge to `integrated_synthesis`): producer axis =
  behaviour-signature majority over the model DERIVED cases; trust axis = falsifier-survivor set;
  `select_integrated_synthesis_decision` commits the agreed rep, ABSTAINS on divergence / on a
  structurally-distinct unanimous tie (the pawnshop discipline — never mis-commit a coin-flip).
* **SO3 — verifier-final chooser** (mines `mathvista_bench_v2`): one model call sees every
  candidate + its public/derived verdict + the invariants and makes a REAL CHOICE or ABSTAINs.
* **SO4 — trust-weighted abstain ensemble**: trust = falsifier-survival fraction; integrity =
  format/crash-clean; realizes the `integrity_trust_coupled_consensus_v1` ABSTAIN concept
  natively. The substrate `TrustWeightedConsensusController` (latent capsule / cosine / merge)
  literal bridge is KILLED as fake-different (`examine_trust_machinery_applicability_v1`,
  machine-checkable) — the W128 W79 lesson.

NIM-free realness surface: `fake_selection_control_v1` (SO2/SO4 must ABSTAIN a no-evidence tie
with `evidence_used=False`) — **PASSES**. Trust-machinery kill — **substrate literal bridge
KILLED**. 29/29 module tests pass.

### Stored-pool regression pair + aggregate (NIM-free, over the W128 11-target pools)

From `stored_selector_eval_v1.json` (W128 baseline 2/11, RDA4 2/11, **pool ceiling 3/11**;
method: the selection logic replayed analytically on the recon-graded candidates — subprocess-free;
the generic stress component is not load-bearing here):

| arm | committed | SELECTION mis-commits | abstains | net vs baseline |
|---|---|---|---|---|
| W128-RDA4 (ref) | 2/11 {blue, sun} | **1** (pawnshop A0) | 7 | +0 |
| W129 SO1 (naive first-survivor) | 2/11 | 2 (pawnshop A0 + the generation-level `doubleup`) | 7 | +0 |
| W129 SO2 / **SOLEAD** (abstain-disciplined) | 1/11 {blue} | **0** (only the generation-level single-survivor `doubleup`) | 9 | −1 |
| W129 SO4 (trust-weighted-abstain) | 0/11 | 0 (only `doubleup`) | 10 | −2 |
| W129 β0 SO3 verifier-final (3 NIM) | 2/11 {**pawnshop B1**, blue} | **0** | (sun abstained) | +0 |

`control_passes = True`; `substrate_controller_literal_bridge_killed = True`. The
abstain-disciplined SOLEAD has **0 SELECTION mis-commits** (W128-RDA4 had 1) but lower net
(it abstains the under-determined ties W128 gambled on). No arm exceeds the pool ceiling (+1) —
the +2 earn bar is structurally unreachable by selection.

The decisive per-problem behaviour (confirmed):
* **`blueberrywaffle`** (separable): SO2/SOLEAD COMMIT a hidden-correct candidate via the
  signature majority (`INTEGRATED_producer_only`) — the W128 unique win is KEPT.
* **`pawnshop`** (under-determined on small/generic signal): SO2/SOLEAD **ABSTAIN**
  (`UNDER_DETERMINED_TIE_ABSTAIN`) — they do NOT re-commit the hidden-wrong `A0`. This converts
  the W128 **mis-commit into a safe abstain** (0 selection mis-commits). The generic stress
  scale-up does NOT crack it (cycle-structured scale-up keeps A0's window small).
* **`sunandmoon`** (both-correct tie): SO2/SOLEAD ABSTAIN — safe but loses a win the W128
  coin-flip happened to take.

## Lane β0 — SO3 verifier-final stored probe (3 NIM; `so3_stored_probe_verdict.json`)

* **`pawnshop` → CHOSE `B1` (CORRECT)** — the verifier read the code and reasoned A0's `O(N²)`
  recompute would TLE on `N ≤ 300000` while B1's `O(N)` would not. The under-determined tie IS
  breakable by a code-reading model judge.
* **`blueberrywaffle` → CHOSE `C2` (correct, kept)**.
* **`sunandmoon` → ABSTAIN** (over-cautious on a both-correct tie).
* `b0_pass = False` per the locked § 6 gate — but the failure is **over-abstention on a
  both-correct tie, NOT non-discrimination**: the verifier committed 2/11 with **0 selection
  mis-commits** and cashed the HARD fixture (pawnshop) the NIM-free layer could not. Net committed
  count is unchanged vs W128 (it trades the sunandmoon win for the pawnshop cash-out).

Because (a) the locked β0 gate FAILED and (b) committed ≤ pool ceiling = +1 < the +2 earn bar
(generation-bound — see TL;DR #3), the fresh β1 dev bench is **NOT** fired: it cannot clear the
+2 R2′ bar by SELECTION on a bench whose generation ceiling is +1, so a ~220-NIM fresh bench
would be hope-funded spend against a structural ceiling. **T1 = FALSE.**

## Lane γ — stronger-model gate + truth (`stronger_model_gate/gate_recheck_v1.json`)

`NO_CERTIFIABLE_STRONGER_MODEL`; decision CID `258b6ed7` **invariant** (W114→W129);
{KNOWN:1 (Maverick Aug-2024), UNKNOWN:4 (Qwen3-Coder-480B / DeepSeek-V4-pro /
Mistral-Small-4-119b-2603 / GLM-5)}. Gate **CLOSED**. T1 ∧ T2 not both true ⇒ **$0 resistant
NIM**; no targeted resistant probe; exposed control NOT bought. W123–W128 caps stay closed.

## Carry-forwards (filled from JSON)

* `W129-T-PAWNSHOP-SELECTION-MISS-IS-COMPLEXITY-VERIFIER-BREAKABLE` — the W128 pawnshop
  "selection cap" is a COMPLEXITY (O(N²) TLE) signal a code-reading verifier breaks (chose B1),
  not a hard public-signal information limit.
* `W129-L-NIMFREE-GENERIC-STRESS-CANNOT-CONSTRUCT-ADVERSARIAL-WORSTCASE` — a generic
  format-preserving scale-up does not construct the adversarial worst-case needed to TLE the slow
  candidate; constructing it generically ≈ solving the problem.
* `W129-L-HARD-CLUSTER-GENERATION-CEILING-CAPS-SELECTION-EARN` — committed ≤ pool ceiling
  (3/11 = baseline+1) < the +2 earn bar ⇒ R2′ unreachable by ANY selector on this bench ⇒ the
  binding cap is GENERATION, not selection (the cap taxonomy moves W123 battlefield → W124 encoder
  → W125 re-routing → W126 synthesis → W127 scaffold-gen → W128 role-diverse-selection → **W129
  selection-is-not-the-cap-generation-is**).
* `W129-T-ABSTAIN-DISCIPLINE-ELIMINATES-SELECTION-MISCOMMIT` — the abstain-disciplined SOLEAD has
  0 SELECTION mis-commits (vs W128-RDA4's 1 on pawnshop); the only residual mis-commit is the
  GENERATION-level single-survivor `doubleup` (the pool has no correct candidate).
* `W128-L-GRAPH-FLOW-EXPOSED-SUPPLY-CAP` carried forward.

Anchors: `docs/RUNBOOK_W129.md`; `results/w129/recon/stored_pool_recon_v1.json`;
`results/w129/selector_stored/stored_selector_eval_v1.json`;
`results/w129/dev_bench/so3_stored_probe_verdict.json`;
`results/w129/stronger_model_gate/gate_recheck_v1.json`;
`coordpy/public_signal_selection_oracle_v1.py`.
