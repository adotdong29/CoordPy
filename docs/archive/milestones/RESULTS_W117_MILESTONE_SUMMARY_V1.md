# W117 — Milestone summary (upstream-DERIVED instrument CONSTRUCTION attack + deeper primary-source cutoff attack + construction/admission pipeline; $0 NIM)

**One line:** W117 escalated W116 from *packaged-release detection* to *construction
from provenance* — it attacked the upstream CONSTRUCTION provenance LIVE at EIGHT
authoritative surfaces (the revision/commit history + collection mechanism, not just
the release label), proved NO post-v6 instrument can be CONSTRUCTED from authoritative
provenance (the only post-v6 path — raw-contest hand-assembly — is refused by the
pre-committed B1 + B2 criteria), re-confirmed DEEPER from primary sources that the
DeepSeek V4 official model-card PDF discloses NO cutoff, and shipped a durable
upstream-DERIVED construction/admission pipeline that makes the next clean shot
push-button. The certification verdict re-derives `NO_CERTIFIABLE_STRONGER_MODEL`
(decision CID `258b6ed7…`, byte-identical to W114/W115/W116). **No pilot earned; $0
NIM. W89 + W105 remain the only two retirements.**

---

## The three lanes

### Lane α — upstream-derived CONSTRUCTION attack (LIVE; the main empirical lane)

W116 verified the latest *packaged release* at four surfaces. W117 attacks the
*construction provenance* at EIGHT surfaces (RUNBOOK_W117 § 2) — the revision history
+ collection mechanism:

1. **HF dataset commit/revision log**: 20 commits; latest data-bearing = "add v6"
   2025-04-21; HEAD = "fix typos" 2025-06-05. No post-v6 data commit; no `test7`.
2. **HF refs**: 1 branch (`main`), 0 tags. No staging / v7 branch.
3. **HF discussions/PRs**: newest = "LCB pull request" #14 + a v6-size clarification
   #13. No v7 / newer-data thread.
4. **GitHub commits**: newest 2025-07-16 (runner typo); recent = runner maintenance.
   No data-collection commit; no v7.
5. **GitHub tags**: empty (0 tags).
6. **GitHub repo pipeline structure**: `lcb_runner/` + `assets/` only; NO scraper /
   data / collect / contest directory — the public repo is the runner/eval harness,
   not a published collection pipeline.
7. **Dataset README provenance**: LeetCode / AtCoder / Codeforces; documents ONLY
   loading published releases — NO generation tool / manifest.
8. **Runner loader**: `code_generation.py` loads exclusively via
   `load_dataset("livecodebench/code_generation_lite", version_tag=…)` — no local
   scraping.

**Net:** the authoritative construction provenance IS the packaged HF release; LCB
publishes no collection pipeline or forward problem-id manifest. No post-v6
LCB-published artifact exists at any surface. The only post-v6 path (raw-contest
hand-assembly) is **CONSTRUCTION-INADMISSIBLE** (refused by A1 ∧ B1 ∧ B2). ⇒ **0
construction-admissible NEW instruments.** This is sharper than W116's "no packaged
v7": it proves a post-v6 instrument cannot be *constructed* from authoritative
provenance, only hand-curated (refused).

### Lane β — deeper primary-source model-cutoff attack (LIVE)

Probed DEEPER than a single card. Maverick = **KNOWN "August 2024"** (Meta
MODEL_CARD.md, verbatim) but C4-settled; Qwen3-Coder-480B = UNKNOWN (official HF card
raw README: no cutoff); **DeepSeek-V4-pro = UNKNOWN from primary** (official V4
model-card PDF re-checked: no cutoff; the only figure is a non-primary aggregator "Apr
2026" that is itself C2-exposed); Mistral-Small-4-119B-2603 = UNKNOWN from primary
(docs: no cutoff). No reachable stronger-than-Maverick model has a primary-KNOWN cutoff
≤ 2025-01; **nothing newly-disclosed since W116.** DeepSeek V4 now matches Mistral
Small 4's exact "UNKNOWN-primary + C2-exposed-aggregator" pattern.

### Lane γ — construction/admission pipeline (NIM-free; `coordpy.upstream_derived_instrument_construction_v1`)

* **Construction rule** (`assess_construction_admissibility_v1`): A1..A5 (reused) ∧ B1
  (authoritative LCB-published provenance) ∧ B2 (no operator curation). REFUSES the
  raw-contest / aggregator / hand-curated path.
* **Eight-surface provenance snapshot** (`UpstreamProvenanceSnapshotV1`), wrapping the
  W116 `UpstreamSupplySnapshotV1`.
* **Candidate-instrument constructor** (`construct_upstream_derived_candidate_v1`):
  raw-contest assembly triply refused (A1 ∧ B1 ∧ B2); LCB-pipeline template
  construction-admissible-in-principle but artifact-absent ⇒ `constructed = False` +
  the EXACT missing artifact named.
* **Sharpened disclosure matrix** (`W117_DISCLOSURE_MATRIX`) + the **W118 fire
  condition** (`W118FireConditionV1`: packaged / construction-provenance / cutoff).
* Verdict re-derives `NO_CERTIFIABLE_STRONGER_MODEL` (decision CID `258b6ed7…`; result
  CID `c3c60483…`); corpus `sha_ok` ✓ + `histogram_match` ✓. graphify-confirmed reuse:
  `run_upstream_construction_v1 --calls--> run_upstream_admission_v1`.

## Pilot decision

A pilot needs BOTH a construction-admissible NEW instrument AND a non-redundant
primary-KNOWN-cutoff stronger model — **neither exists** ⇒ **NO pilot earned; $0 NIM**
(discipline, not omission). No Maverick rerun (no genuinely new instrument; the
`release_v6` cell is settled — W106 redundant-run discipline).

## Stable boundary

`coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`; no
PyPI; `coordpy/__init__.py` untouched. 1 new explicit-import-only module
(`upstream_derived_instrument_construction_v1`) + 1 script
(`run_w117_upstream_construction_v1.py`) + 13 new tests (81 total across W113–W117).
27th consecutive preflight/earn-discipline validation (W93–W117).

## Carry-forwards

* **Added**: `W117-T-UPSTREAM-DERIVED-CONSTRUCTION-PIPELINE-SHIPS`,
  `W117-L-NO-CONSTRUCTION-ADMISSIBLE-POST-V6-INSTRUMENT-EIGHT-SURFACE-CAP`,
  `W117-T-LCB-CONSTRUCTION-PROVENANCE-IS-PACKAGED-RELEASE`,
  `W117-T-DEEPSEEK-V4-PRIMARY-PDF-RECONFIRMED-NO-CUTOFF`.
* **Retired**: none. W89 + W105 STAND; the W114/W115/W116 caps STAND.

## W118 (loaded)

Fires on (a) a packaged `release_v7`+ admitted to the loader, OR (b) an LCB-PUBLISHED
post-v6 CONSTRUCTION provenance (a dataset revision/commit/PR adding post-2025-04
functional problems, OR a published collection pipeline + problem-id manifest)
enabling a B1 ∧ B2 reproducible ≥30 functional post-cutoff slice, OR (c) a reachable
stronger-than-Maverick model disclosing a primary-KNOWN cutoff ≤ 2025-01. `COO-9` stays
lead.
