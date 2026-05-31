# RUNBOOK — W120: close the ICPC count gap + certify Maverick on a ≥30 official resistant battlefield + run the earned pilot

**Status: LOCKED before any NIM call (2026-05-31).** Sibling of `COO-9`; executes the
three-lane W120 branch logic. `ultracode` OFF. Stable boundary: `coordpy.__version__ ==
"0.5.20"`, `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI, `coordpy/__init__.py`
untouched, advanced work explicit-import only.

W119 dissolved the W118 grader blocker (official ICPC packages ship a self-test-passing
executable grader) but the post-cutoff resistant pass-fail slice from the single RMRC
surface was **24 < MIN_RESISTANT_SLICE = 30** — the count was the sole remaining blocker.
W120 is **NOT** another "24 < 30" memo and **NOT** another exposed rerun. It is a
count-gap-closure attack + a stronger-model certification lane + an earned pilot the
moment the battlefield clears 30.

---

## 1. Exact α / β / γ branch logic

* **Lane α — count-gap closure (main empirical lane).** Two routes, same milestone:
  * **α1 exclusion audit:** re-derive every W119 RMRC exclusion problem-by-problem from
    each official `problem.yaml` + tree (`scripts`-side `rmrc_audit`). Recover any task
    honestly admissible under the extended evaluator; correct mislabels.
  * **α2 multi-surface aggregation:** sweep additional official ICPC org surfaces; admit
    those that pass the same resistant-date + grader-clean rule.
  * **Success ⟺** a combined official-ICPC battlefield with **≥ 30 tier-1 pure pass-fail**
    resistant tasks, a deterministic manifest/SHA/CID/date-histogram, exact inclusion +
    exclusion counts, and a self-test-passing grader on each admitted surface.
* **Lane β — stronger-model cutoff / certification (mandatory, parallel).** Re-check
  primary-source cutoffs for Maverick, Qwen3-Coder-480B, DeepSeek-V4-pro,
  Mistral-Small-4-119B, GLM-5, and any newly reachable stronger model; build the
  disclosure-status matrix; pick the strongest honestly-certifiable target (Maverick by
  default — KNOWN Aug-2024).
* **Lane γ — executable battlefield-to-pilot (mandatory).** Ship the exclusion-audit
  tool, validator-aware executor extension, official multi-surface aggregator, manifest
  builder, package validator, preflight/construction driver, certification matrix, the
  stdin/stdout reflexion bench, the pilot driver, the W121 fire condition, and tests.
* **Pilot ⟺ Lane α (≥30) ∧ Lane β (a certifiable model).** If earned, run the cheapest
  honest pilot; else `$0` NIM and the exact remaining exclusion reasons are made
  machine-checkably load-bearing.

---

## 2. Exact official ICPC public-package battlefield rule (R1)

A surface is admissible iff it is a repository of the **official ICPC GitHub org
(`github.com/icpc`)** or an official ICPC Foundation archive surface, following the
ICPC/Kattis Problem Package Format (`problem.yaml` + `data/secret/*.in`+`*.ans` +
`submissions/` + optional `output_validators/`). **NOT** mirrors, aggregators, or
third-party scrapers. W120 admits two surfaces, both in the icpc org:

* **RMRC** — `icpc/na-rocky-mountain-2024-2025-public` (2024-12-03),
  `icpc/na-rocky-mountain-2025-2026-public` (2025-11-13). (W119 surface.)
* **ECNA** — `icpc/na-ecna-archive`, year folders `2024-2025` (NA East Division 2024,
  packages dated 2024-11-11) and `2025-2026` (2025, dated 2025-11-10). (**W120 NEW**;
  per-problem `.zip` Kattis packages.)

The `icpc/archive` repo's `ContestList.tsv` is the official contest index; ECNA (`ecna`)
is an official ICPC NA regional listed there.

---

## 3. Exact validator-aware pass/fail admission rule (R7) — tiers

Each problem is classified by a **total deterministic** read of its official
`problem.yaml` (active, non-comment lines) + tree, into:

| kind | rule | tier | admitted |
|---|---|---|---|
| `passfail` | `validation` absent/`default` (incl. `case_sensitive`/space flags) | **tier-1** | ✅ core |
| `passfail_float` | `validation: default` + `validator_flags: float_*` | tier-2 | ✅ extended |
| `custom_with_validator` | `validation: custom` **and** ships `output_validators/` | tier-3 | ✅ extended |
| `custom_no_validator` | `validation: custom` **without** a shipped validator | — | ❌ no checker |
| `interactive` | `validation: custom interactive` or an interactor | — | ❌ P4 |
| `scoring` | `type: scoring` / scoring validation | — | ❌ optimization |

* **tier-1 (core):** graded by the default token-normalized diff oracle
  (`run_icpc_stdin_executor_v1`). **The ≥30 count gate AND the pilot use tier-1 ONLY** —
  the strictest, cleanest possible slice.
* **tier-2 (float):** graded by a **deterministic** float oracle (`judge_icpc_output_v1`:
  numeric tokens within absolute OR relative `validator_flags` tolerance, non-numeric
  exact, token counts equal). Allowed because it is deterministic, secret-data-backed,
  and executable without human judgment. Counted as battlefield breadth; **not required**
  for the ≥30 gate.
* **tier-3 (custom-with-validator):** a shipped deterministic checker. Counted; not in
  the pilot.
* **Excluded** kinds are never admitted regardless of count. **We do not loosen the
  battlefield to cross 30.**

**R8 grader self-test (per surface):** a shipped accepted reference solution runs in a
fresh isolated subprocess against the official secret cases and passes. RMRC: reuse W119
(videogames 8/8 + whattimeisitmrfox 8/8 = 16/16). ECNA: 6/6 Python-self-testable
problems, 149/149 cases PASS (incl. `valleygulls` 40/40 under the float oracle).

---

## 4. Exact multi-surface aggregation rule (R5/R6)

Aggregate across admitted official surfaces under the **same** resistant-date + tier
rules. Inclusion AND ordering are a total machine function of the official payload
(sorted by `(contest_date, source_repo, short_name)`); every exclusion is typed. The
combined manifest pins `raw_classification_sha256` (over the full classified listing) +
a content-addressed `manifest_cid` + a re-derivable month histogram. **Stop broadening
once tier-1 honestly exceeds 30** — one new official surface adding ≥6 clean tier-1 tasks
is sufficient (ECNA adds 23).

---

## 5. Exact resistant-date rule (R3)

A problem is **post-cutoff resistant** for the target model iff its official contest date
is **STRICTLY AFTER** the model's KNOWN cutoff. Target Maverick ⇒ boundary
`2024-08-31`. All four admitted contests (ECNA 2024-11-11 + RMRC 2024-12-03 + ECNA
2025-11-10 + RMRC 2025-11-13) are strictly after ⇒ resistant. Undated / pre-cutoff
problems are excluded (`pre_cutoff_or_undated`). Boundary is parsed by the reused
`normalize_contest_date_v1`.

---

## 6. Exact per-model disclosure-status + certification rule (Lane β)

Reused W114 `certify_model_v1` gate **C1∧C2∧C3∧C4** on the tier-1 core instrument:
* **C1** cutoff KNOWN from a PRIMARY source; **C2** ≥ `MIN_RESISTANT_SLICE`=30 resistant
  problems after the cutoff; **C3** reachable + stronger/comparable same-budget;
  **C4** not already settled on this instrument.

Disclosure matrix (primary sources, re-checked 2026-05-31):

| model | primary cutoff | status | NIM-reachable | certifiable |
|---|---|---|---|---|
| `meta/llama-4-maverick-17b-128e-instruct` | **August 2024** (Meta llama4 MODEL_CARD.md, verbatim ×3) | **KNOWN** | ✅ | **YES** (C1✓ C2✓[45≥30] C3✓ C4✓ — new instrument) |
| `qwen/qwen3-coder-480b-a35b-instruct` | none stated (HF card + blog + tech report) | UNKNOWN | ✅ | no (C1) |
| `deepseek-ai/deepseek-v4-pro` | none stated (V4 card; "Jul-2024" is V3 sys-prompt, non-primary) | UNKNOWN | ✅ | no (C1) |
| `mistralai/mistral-small-4-119b-2603` | none stated (Mistral docs) | UNKNOWN | ✅ | no (C1) |
| `zai-org/glm-5` | none primary (listicle only) | UNKNOWN | ❌ | no (C1 + reachability) |

Matrix `{KNOWN: 1, UNKNOWN: 4}`; nothing newly primary-disclosed since W119. **Default
target = Maverick.** If a stronger-than-Maverick model becomes primary-KNOWN and
certifies, prefer the strongest honest target.

---

## 7. Exact pilot-earning rule (battlefield ≥30)

A pilot is **EARNED ⟺ BOTH**: (a) Lane α tier-1 core ≥ 30, **and** (b) Lane β yields a
model that is identity-certifiable (C1..C4) ∧ grader-admissible (R7∧R8) ∧ slice-admissible.
**On the W120 live pass both hold** (core = 45 ≥ 30; Maverick certifies; grader self-test
passes both surfaces; `NVIDIA_API_KEY` present; Maverick reachable in the 118-model NIM
catalogue) ⇒ **run the cheapest honest pilot.**

* Target `meta/llama-4-maverick-17b-128e-instruct`; instrument = the deterministic
  tier-1 core 30-slice (surface×year stratified; CID `01bf9ef8…`); 1 seed × 30 × K=5 ⇒
  330 NIM calls; A0 (T=0) + A1 (first-pass-among-K, T=0.7) + B (sequential reflexion K=5,
  T=0.7).
* **Grader = official secret cases** (token-diff oracle; exit-0-iff-every-secret-case;
  NO LLM judge). **Reflexion feedback = public samples + judge verdict bit + stderr ONLY
  (never secret data).**
* **Canary first:** `--n-problems 2 --label canary` (~22 calls) to validate the harness
  end-to-end; only then the full 30.
* Gates = the **verbatim W108** `_evaluate_phase2_gates` + `_mlb_rates` (byte-identical to
  W103/W105/W108/W113): G1..G9 + MLB-1 (invocation ≥33%) + MLB-2 (rescue ≥33%).
* **Clean-superiority bar = `verdict_label == PASS_MECHANISM_DRIVEN`** (9/9 ∧ MLB-1 ∧
  MLB-2). Outcome mapping in `_interpret_w120` (§8 of the pilot driver). A single-seed
  PASS is **RESISTANT_SUPERIORITY_DEMONSTRATED_SINGLE_SEED** — strong, but multi-seed
  confirmation (W121) is required to reach W89/W105 retirement-grade. We do **not** count
  a `PASS_NON_MECHANISM_DRIVEN` or a `FAIL` as a resistant win.

---

## 8. Exact no-go rule (battlefield < 30)

If tier-1 core stays < 30 after both α routes: **NO pilot; `$0` NIM is correct** (discipline,
not omission). Do not run a 24-/26-/29-task "almost there" pilot. Make the exact remaining
exclusion reasons machine-checkably load-bearing via `exclusion_audit_v1` (typed counts +
excluded problem ids) and name the precise missing-artifact (`assess_battlefield_admissibility_v1.reason`).
*(W120 actual: core = 45 ⇒ this branch does NOT fire.)*

---

## 9. Exact graphify deliverables

* Refresh `graphify update .` at START (built from HEAD `106428c`) and at END (after all
  code/doc changes) so `graphify-out/` matches repo truth; record the END commit.
* Use graphify for file selection + dependency checks: `graphify explain
  run_battlefield_construction_v1 / run_icpc_reflexion_bench_v1 /
  run_icpc_public_construction_v1 / certify_model_v1 / build_icpc_manifest_v1`;
  `graphify path run_battlefield_construction_v1 run_upstream_construction_v1`;
  `graphify affected coordpy_icpc_battlefield_v1`; `graphify query` as a secondary
  claim-surface finder. Confirm the reuse edges (battlefield → certify_model_v1,
  → run_upstream_construction_v1, → run_icpc_stdin_executor_v1, bench → battlefield).

---

## 10. Exact W121 branch logic

* **If the pilot is `PASS_MECHANISM_DRIVEN`:** W121 = **multi-seed same-budget
  confirmation** on the same ≥30 official ICPC battlefield to reach W89/W105
  retirement-grade on RESISTANT code (the first such). Optionally widen with the next
  official ICPC surface for a second seed.
* **If `PASS_NON_MECHANISM_DRIVEN`:** register the bounded margin; W121 strengthens the
  mechanism (NIM-free-earned) or accepts the bounded resistant ceiling.
* **If `FAIL`:** the bounded contamination-EXPOSED-HumanEval-family-at-70B ceiling
  STANDS; resistant superiority still 0 clean; W121 = accept the bounded claim / pursue a
  genuinely different axis (NOT another exposed rerun).
* **Cross-cutting triggers (any outcome):** a reachable stronger-than-Maverick model
  disclosing a primary-KNOWN cutoff (prefer the strongest honest target on this same
  battlefield); a further official ICPC surface widening the resistant set.

`COO-9` stays the lead path unless the evidence forces a different code-line move.
No version bump; no PyPI; `coordpy/__init__.py` untouched.
