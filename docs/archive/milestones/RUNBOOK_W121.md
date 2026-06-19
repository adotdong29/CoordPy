# RUNBOOK — W121: matched EXPOSED official-ICPC control + Maverick same-family contrast + optional paired-seed

**Status: LOCKED before any NIM call (2026-05-31).** Sibling of `COO-9`; executes the
three-lane W121 branch logic and the pre-committed `docs/RUNBOOK_W120.md` § 10 FAIL
branch ("genuinely different axis"). `ultracode` OFF. Stable boundary:
`coordpy.__version__ == "0.5.20"`, `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI,
`coordpy/__init__.py` untouched, advanced work explicit-import only.

W120 built a ≥30 official-ICPC **RESISTANT** battlefield, certified Maverick, ran the
earned pilot ⇒ **B−A1 = +0.00 pp, FAIL**. The remaining live objection: the EXPOSED
retirements (W89/W105 HumanEval-family) and the RESISTANT ICPC null had never been
matched **inside the SAME official ICPC package family**, so the resistant null could be
an ICPC-DIFFICULTY artifact rather than a CONTAMINATION one. W121 is **NOT** another
resistant rerun and **NOT** another blocker memo. It is the matched EXPOSED control on
the SAME official package family + the same-model same-mechanism contrast against W120.

---

## 1. Exact α / β / γ branch logic

* **Lane α — matched EXPOSED ICPC control battlefield (main empirical lane).** Build the
  exposed control from the SAME two official ICPC org surface families W120 used (the
  `icpc/na-ecna-archive` archive + the `icpc/na-rocky-mountain-*-public` repos), on the
  PRE-cutoff (exposed-for-Maverick) year-drops, under byte-identical R1/R2/R4/R5/R6/R7/R8
  discipline with the date rule FLIPPED (E3, § 2). **Success ⟺** ≥ 30 tier-1 pure
  pass-fail EXPOSED tasks + a deterministic manifest/SHA/CID/date-histogram + a
  self-test-passing grader on each exposed surface. *(W121 live: 42 ≥ 30.)*
* **Lane β — Maverick exposed control pilot + paired contrast (mandatory).** If Lane α
  yields ≥30 and Maverick EXPOSED-certifies (§ 5/§ 6), run the cheapest honest exposed
  pilot (same mechanism + gate discipline as W120), then contrast its margin directly
  against the LOCKED W120 resistant +0.00 (§ 3). Buy a paired seed ONLY in the § 4
  ambiguity band.
* **Lane γ — executable dual-battlefield control pipeline / graphify / truth (mandatory).**
  Ship the exposed-field constructor, exposed manifest builder, exposed preflight, the
  exposed certification gate, the dual-field comparison logic, the W122 fire condition,
  and tests. Keep graphify in the loop (§ 8). W121 is NOT complete if it only updates
  docs — it must land executable code/scripts in the control-comparison path.
* **Pilot ⟺ Lane α (≥30) ∧ Lane β (Maverick exposed-certifiable).** If earned, run; else
  `$0` NIM and the exact sub-threshold reason is made machine-checkably load-bearing (§ 7).

---

## 2. Exact EXPOSED ICPC battlefield rule (E1..E8)

A surface is admissible iff it is one of the SAME two official ICPC org surface families
W120 used (`github.com/icpc`: the `na-ecna-archive` archive year-folders + the
`na-rocky-mountain-*-public` repos), following the ICPC/Kattis Problem Package Format —
**NOT** mirrors/aggregators/scrapers. The exposed control mirrors W120's **2-ECNA +
2-RMRC** structure on the most-recent pre-cutoff artifact-complete year-drops:

| surface | source | contest date | matched W120 resistant surface |
|---|---|---|---|
| RMRC | `icpc/na-rocky-mountain-2021-public` | 2022-03-14 | RMRC 2024-2025 |
| ECNA | `icpc/na-ecna-archive` `2022-2023` | 2022-11-12 | ECNA 2024-2025 |
| RMRC | `icpc/na-rocky-mountain-2022-2023-public` | 2023-02-25 | RMRC 2025-2026 |
| ECNA | `icpc/na-ecna-archive` `2023-2024` | 2023-11-11 | ECNA 2025-2026 |

**Typed comparability exclusion:** `icpc/na-rocky-mountain-2023-2024-public` is MORE
recent (2024-01-16, still pre-cutoff) but ships a MINIMAL package — secret data only, NO
`problem_statement/*.tex` — so it cannot present a statement to the model (every W120
resistant problem shipped a statement). The rule advances to RMRC 2021 to keep all four
exposed surfaces artifact-complete (statement + grader).

The kind/tier rule is byte-identical to W120 R7 (tier-1 pure pass-fail / tier-2 float /
tier-3 custom-with-validator admitted; interactive / custom-no-validator / scoring
excluded). The **≥30 count gate AND the pilot use the STRICT tier-1 core only**. The
ONLY change vs W120 is the date rule:

* **E3 (the flipped W120 R3):** a problem is admitted (EXPOSED) iff its official contest
  date is **AT OR BEFORE** the target model's KNOWN cutoff (Maverick Aug-2024 ⇒ date ≤
  `2024-08-31`), the exact complement of W120's "strictly after". Post-cutoff problems
  are EXCLUDED (`post_cutoff_or_undated`). ICPC regionals never fall in the ambiguous
  August window, so the complement is clean.
* **E8 grader self-test (per surface):** an accepted reference runs in a fresh isolated
  subprocess against the official secret cases and passes, on EACH exposed surface.

**W121 live pass (SHA-pinned, NIM-free):** 50 seen → 48 admitted → **42 tier-1 pure
pass-fail** (+5 float +1 custom-with-validator; 2 custom-no-validator excluded);
`raw_classification_sha256 = 653e3682…`; `manifest_cid = 8acbc7cc…`; core 30-slice CID
`32d15db5…`; dates 2022-03-14..2023-11-11 (all ≤ 2024-08-31); month histogram
`{2022-03:13, 2022-11:12, 2023-02:11, 2023-11:12}`; grader self-test **30 all-pass
problems / 637 official secret cases, ≥4 problems per surface ⇒ each-surface PASS**.

---

## 3. Exact EXPOSED-vs-RESISTANT comparison rule

The contrast is **same model (Maverick) × same official ICPC package family × same grader
× same evaluator (verbatim W108 gates) × same K/budget × same difficulty class (ICPC
regional) — opposite cutoff sides.** The matched-family proof
(`MatchedFamilyComparisonV1`, NIM-free) asserts `differs_only_in_cutoff_side` and that
org/format/grader/tiers/difficulty/model are all shared; the empirical contrast is the
**exposed pilot B−A1 vs the LOCKED W120 resistant B−A1 = +0.00 pp**. The pre-committed
three-branch interpreter (`interpret_exposed_vs_resistant_v1`) maps the exposed margin:

* **exposed ≥ +5.00 pp** (W89/W105 grade) while resistant ~0 ⇒
  `EXPOSED_MARGIN_VS_RESISTANT_NULL_DIFFICULTY_LOOPHOLE_CLOSED` — within-family
  within-model exposure dissociation; difficulty/family held fixed, only exposure flips
  ⇒ the "resistant null is an ICPC-difficulty artifact" loophole is CLOSED.
* **|exposed| ≤ +3.34 pp** (within ~one K=5 rescue of the resistant null) ⇒
  `EXPOSED_NULL_TOO_CONTAMINATION_CONFOUND_WEAKENS_BOUNDED_CEILING_HARDENS` — on the same
  family at matched difficulty, flipping only exposure does NOT reopen the margin ⇒ the
  contamination reading of W89/W105 weakens; the bounded ceiling hardens (ICPC difficulty,
  not HumanEval-family exposure, is the more likely driver of the resistant null).
* **otherwise (between +3.34 and +5.00, or |·|>3.34 without ≥+5)** ⇒
  `AMBIGUOUS_PAIRED_SEED_EARNED` — earn exactly ONE paired seed (§ 4).

---

## 4. Exact ambiguity band + paired-seed earn rule

* `EXPOSED_MARGIN_PASS_PP = 5.0`; `AMBIGUITY_BAND_PP = 3.34`.
* A paired seed is **EARNED ⟺** the first exposed-control result is AMBIGUOUS (outcome
  `AMBIGUOUS_PAIRED_SEED_EARNED`): the exposed margin is neither a clean ≥+5 pp margin
  nor within ±3.34 pp of the resistant null. ONLY then is a second seed run on BOTH the
  exposed AND the resistant ICPC battlefields (the only move that can change the
  interpretation).
* Do **NOT** default to extra seeds. A clean margin (loophole closed) or a clean null
  (confound weakens) is decisive on one seed — buying a second seed there has no
  interpretation-changing power and is refused (W106 margin-cap discipline).

---

## 5. Exact per-model disclosure-status + certification rule (Lane β)

EXPOSED certification (`certify_model_exposed_v1`) is the mirror of the W114
`certify_model_v1` gate, with **C2 flipped to C2e**: C1 (cutoff KNOWN from a PRIMARY
source) ∧ **C2e (≥ `MIN_RESISTANT_SLICE`=30 problems in months AT OR BEFORE the cutoff
= EXPOSED)** ∧ C3 (reachable + stronger/comparable) ∧ C4 (un-run on THIS exposed
instrument; the exposed control is genuinely new). Disclosure matrix (primary sources,
re-checked 2026-05-31; reused W118 matrix, byte-identical to W119/W120):

| model | primary cutoff | status | reachable | EXPOSED-certifiable |
|---|---|---|---|---|
| `meta/llama-4-maverick-17b-128e-instruct` | **August 2024** (Meta llama4 MODEL_CARD) | KNOWN | ✅ | **YES** (C1✓ C2e✓[48≥30] C3✓ C4✓) |
| `qwen/qwen3-coder-480b-a35b-instruct` | none stated | UNKNOWN | ✅ | no (C1) |
| `deepseek-ai/deepseek-v4-pro` | none stated | UNKNOWN | ✅ | no (C1) |
| `mistralai/mistral-small-4-119b-2603` | none stated | UNKNOWN | ✅ | no (C1) |

`{KNOWN:1, UNKNOWN:3}`; nothing newly primary-disclosed since W120. **Default target =
Maverick** (the only model with a KNOWN cutoff, the only one whose exposure side is
anchorable). If a stronger-than-Maverick model becomes primary-KNOWN and certifies on
BOTH ICPC battlefields, prefer the strongest honest target.

---

## 6. Exact EXPOSED-pilot earn rule

A pilot is **EARNED ⟺ BOTH**: (a) Lane α tier-1 core ≥ 30, **and** (b) Lane β yields a
model that is EXPOSED-certifiable (C1∧C2e∧C3∧C4) ∧ grader-admissible (E7∧E8) ∧
slice-admissible. **On the W121 live pass both hold** (core = 42 ≥ 30; Maverick
exposed-certifies; grader self-test passes all four surfaces; `NVIDIA_API_KEY` present;
Maverick reachable) ⇒ **run the cheapest honest pilot.**

* Target `meta/llama-4-maverick-17b-128e-instruct`; instrument = the deterministic
  exposed tier-1 core 30-slice (surface×year stratified; CID `32d15db5…`); 1 seed × 30 ×
  K=5 ⇒ 330 NIM calls; A0 (T=0) + A1 (first-pass-among-K, T=0.7) + B (sequential
  reflexion K=5, T=0.7); seed 120001 (byte-identical to W120).
* **Grader = official secret cases** (token-diff oracle; exit-0-iff-every-secret-case;
  NO LLM judge). **Reflexion feedback = public samples + judge verdict bit + stderr ONLY
  (never secret data).**
* **Canary first:** `--n-problems 2 --label canary` (~22 calls) to validate the harness
  end-to-end; only then the full 30.
* Gates = the **verbatim W108** `_evaluate_phase2_gates` + `_mlb_rates` (byte-identical to
  W103/W105/W108/W113/W120): G1..G9 + MLB-1 (invocation ≥33%) + MLB-2 (rescue ≥33%).
* The OUTCOME is then mapped by § 3 (the exposed margin vs the resistant +0.00). A clean
  ≥+5 pp exposed margin while resistant FAILed CLOSES the difficulty loophole; an exposed
  null too WEAKENS the confound and HARDENS the bounded ceiling.

---

## 7. Exact no-go rule (exposed battlefield < 30 OR no exposed-certifiable model)

If exposed tier-1 core stays < 30, OR no model is exposed-certifiable: **NO pilot; `$0`
NIM is correct** (discipline, not omission). Make the exact sub-threshold reason
machine-checkably load-bearing via the exclusion audit + `assess_exposed_admissibility_v1.reason`
and name the precise missing artifact (the next pre-cutoff artifact-complete official
ICPC drop). *(W121 actual: core = 42 ⇒ this branch does NOT fire.)*

---

## 8. Exact graphify deliverables

* Refresh `graphify update .` at START (built from HEAD `8b15e97`) and at END (after all
  code/doc changes) so `graphify-out/` matches repo truth; record the END commit.
* Use graphify for file selection + dependency checks: `graphify explain
  run_exposed_control_construction_v1 / run_icpc_public_construction_v1 /
  run_icpc_stdin_executor_v1 / certify_model_v1 / build_icpc_manifest_v1`; `graphify path
  run_icpc_public_construction_v1 run_upstream_construction_v1`; `graphify affected
  run_icpc_public_construction_v1`; `graphify query` as a secondary claim-surface finder.
  Confirm the reuse edges (exposed control → battlefield classifier/oracle/slice-selector,
  → certify cutoff registry, → run_upstream_construction_v1 for the 258b6ed7 invariant).

---

## 9. Exact W122 branch logic

* **If the exposed pilot CLOSES the loophole (`…DIFFICULTY_LOOPHOLE_CLOSED`):** W121
  upgrades the programme — the within-family within-model exposure dissociation is the
  cleanest contamination evidence to date (still observational, single-seed each side).
  W122 = a stronger primary-KNOWN model on BOTH ICPC battlefields, or a second paired
  seed to harden the dissociation, or accept the strengthened claim and pursue a
  genuinely different axis.
* **If the exposed pilot is a NULL too (`…CONFOUND_WEAKENS…`):** the contamination
  reading of W89/W105 weakens and the bounded ceiling hardens (ICPC difficulty, not
  HumanEval-family exposure, drives the resistant null). W122 = accept the hardened
  bounded ceiling / pursue a genuinely different axis (NOT another ICPC rerun).
* **If AMBIGUOUS (`AMBIGUOUS_PAIRED_SEED_EARNED`):** earn ONE paired seed (§ 4) on BOTH
  battlefields; W122 carries the paired-seed verdict.
* **Cross-cutting (any outcome):** a reachable stronger-than-Maverick model disclosing a
  primary-KNOWN cutoff re-opens BOTH matched battlefields with the strongest honest target;
  a further official ICPC surface widens either side.

`COO-9` stays the lead path unless the evidence forces a different code-line move.
No version bump; no PyPI; `coordpy/__init__.py` untouched.
