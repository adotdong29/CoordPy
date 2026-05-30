# W116 — Milestone summary (upstream instrument-supply ATTACK + primary-source cutoff ATTACK + upstream-admission pipeline; $0 NIM)

**One line:** W116 ATTACKED the supply side instead of waiting — it re-verified the
LCB instrument frontier LIVE at FOUR authoritative UPSTREAM surfaces (lite file tree
/ loader `ALLOWED_FILES` + `release_latest` resolution / full `code_generation`
dataset / GitHub), confirmed NO admissible new instrument beyond `release_v6`,
confirmed from PRIMARY sources that the last hypothesized tier-2 candidate
(`mistralai/mistral-small-4-119b-2603`) is now a REAL model whose official card
discloses NO cutoff, and shipped a durable upstream-ADMISSION pipeline that makes the
next clean shot push-button. The certification verdict re-derives
`NO_CERTIFIABLE_STRONGER_MODEL`. **No pilot earned; $0 NIM. W89 + W105 remain the
only two retirements.**

---

## The three lanes

### Lane α — upstream instrument-supply ATTACK (LIVE; the main empirical lane)

Re-verified from PRIMARY sources at FOUR upstream surfaces (RUNBOOK_W116 § 2),
vs W115's single lite-file-tree check:
* **Lite file tree**: highest = `test6.jsonl`; no `test7`+; lastModified 2025-06-05.
* **Loader `ALLOWED_FILES`**: `v_list = [v1..v6]`; **`release_latest` → the SAME
  files as `release_v6`**; no `release_v7`. (The authoritative definition of which
  releases exist — confirms there is no hidden newer-than-v6 supply.)
* **Full `code_generation` dataset**: README `release_v6` = May 2023–Apr 2025
  (1055 problems, frontier 2025-04-05) — the full set's frontier is also Apr-2025.
* **GitHub repo**: README tops out at `release_v6`; no v7 tag.
* The "planned v7" search hint is non-primary, contradicted by all four surfaces,
  and INADMISSIBLE (no artifact, no SHA) — recorded as the W117 watch signal only.
* **Net**: NO admissible new instrument; the functional frontier is conclusively
  2025-04-05. The instrument supply did NOT move. $0 NIM.

### Lane β — primary-source model-cutoff ATTACK (LIVE; mandatory)

Built the four-way disclosure-status matrix from PRIMARY sources:
* Maverick = **KNOWN** Aug-2024 (settled, C4); Qwen3-Coder-480B / DeepSeek-V4-pro =
  **UNKNOWN** (official cards: NO CUTOFF STATED); Mistral-Small-4-119B-2603 =
  **UNKNOWN from primary** (official Mistral docs card + official announcement state
  no cutoff) — **SHARPENED: now CONFIRMED REAL (119B MoE, 2026-03-16)**; the only
  cutoff figure is a non-primary aggregator (OpenRouter "2025-06") that is itself
  C2-exposed. Mistral-Small-3.2-24B = **KNOWN** ~Oct-2023 but sub-70B (C3).
* **Net**: no reachable stronger-than-Maverick model has a primary-KNOWN cutoff
  ≤ 2025-01; the model-disclosure blocker is sharper than W115.

### Lane γ — upstream-admission pipeline + graphify + truth (NIM-free; mandatory)

Shipped `coordpy.upstream_instrument_admission_v1` (explicit-import-only; reuses the
W113 registry + the W114 `certify_model_v1` / instrument + the W115
`run_frontier_certification_v1` / `frontier_date_summary_v1` / `FrontierSnapshotV1`
+ the loader's `LIVECODEBENCH_KNOWN_RELEASES`, no duplication) +
`scripts/run_w116_upstream_admission_v1.py`. It generalises the W115 snapshot-checker
into a real upstream-ADMISSION pipeline: a pre-committed admissibility rule
(A1..A5), a multi-surface upstream supply snapshot, an upstream-change detector
(richer than W115's single boolean), a certifiable-slice builder, the four-way
disclosure matrix, and a structured `W117FireConditionV1` — all driven by a
`UpstreamSupplySnapshotV1` (external state as DATA) so W117 is push-button. The
script re-verifies the histogram against the SHA-pinned corpus (`sha_ok` +
`histogram_match` ✓) and emits `results/w116/upstream_admission/
upstream_admission_verdict.json` (result CID `193164c4…`; **decision CID `258b6ed7…`
= the W114/W115 decision, byte-for-byte**). graphify refreshed from HEAD at start
(`5b3f75d`; 0 token cost) + re-ingested the new module/script at close;
`explain`/`path`/`affected`/`query` used for file selection + dependency checks.
Truth surfaces tightened across RESEARCH_STATUS / THEOREM_REGISTRY /
HOW_NOT_TO_OVERSTATE / CONSOLIDATED narrative / new CONTAMINATION_CONTROL_FRAMING_W116
/ new FRONTIER_RELEVANCE_AUDIT_W116 / CHANGELOG. 16 new W116 tests (incl. a
falsifiability test that an admissible NEW `release_v7` + a KNOWN-cutoff stronger
model DOES certify and fires W117) + 52 reused W113/W114/W115 tests pass (68 total).

---

## Truth surface after W116

* **Confirmed retirements: still exactly TWO** — W89 + W105, both Llama-3.3-70B @
  70B, contamination-EXPOSED-HumanEval-family. W116 adds NONE, retires NONE.
* **Resistant superiority: 0 clean across BOTH scales** (unchanged; registered
  ceiling).
* **Certification supply: BLOCKED, now attacked at the SOURCE.** No admissible new
  instrument exists beyond `release_v6` (four upstream surfaces); no reachable
  stronger-than-Maverick model has a primary-KNOWN cutoff ≤ 2025-01 (incl. the
  now-real Mistral Small 4). The blocker is precise, dated, per-surface +
  per-model.
* **Contamination-confound: UNCHANGED** (W116 tests certification supply, not the
  confound; STRENGTHENED-not-proven, per W113).
* Still NOT cross-class, NOT a resistant win, NOT MBPP-family, NOT cross-modal,
  NOT "context solved".

## Entitlement delta

NOT entitled to a stronger superiority/retirement claim. The bounded two-retirement
contamination-EXPOSED-HumanEval-family-at-70B claim is the registered ceiling
(W114), now with the supply blocker (a) attacked at FOUR upstream surfaces, (b)
sharpened on the model side by the confirmed-real Mistral Small 4 finding, and (c)
OPERATIONALISED into a push-button upstream-admission pipeline — a sharper, more
defensible position than W115's single-surface snapshot, not a vague "needs more
work".

## Carry-forward delta

* **Added**: `W116-L-UPSTREAM-SUPPLY-NO-ADMISSIBLE-NEW-INSTRUMENT-FOUR-SURFACE-CAP`
  + `W116-T-MISTRAL-SMALL-4-CONFIRMED-REAL-PRIMARY-NO-CUTOFF` +
  `W116-T-UPSTREAM-ADMISSION-PIPELINE-SHIPS`.
* **Retired**: none.
* **Re-affirmed**: the W114/W115 caps STAND.

## W117 (made push-button by W116)

W117 fires the moment `detect_upstream_change_v1` flags an admissible change: a newer
admitted `release_v7`+ (or `release_latest` re-pointing past v6, or a distinct
upstream functional dataset) with ≥30 post-frontier functional problems for a
KNOWN-cutoff stronger model, OR a reachable stronger-than-Maverick model disclosing a
primary-KNOWN cutoff month ≤ 2025-01. Re-run `run_upstream_admission_v1` against the
updated snapshot → if any model certifies on an admissible instrument, run the
pre-committed cheapest-honest pilot on the strongest target. Until then the bounded
ceiling STANDS and resistant-code NIM is BLOCKED. `COO-9` stays lead.

## Discipline / boundary

* 26th consecutive preflight/earn-discipline validation (W93–W116): runbook
  (`docs/RUNBOOK_W116.md`) locked before any NIM; the no-go branch is pre-committed
  by the rule, so the $0 spend is discipline, not omission.
* Stable boundary preserved: `0.5.20` / `coordpy.sdk.v3.43`; no PyPI;
  `coordpy/__init__.py` untouched; 1 new explicit-import-only module + 1 script +
  16 new tests.
* `ultracode` stayed OFF (a bounded upstream-supply + certification milestone, not
  a repo-wide dynamic-workflow job; the four-surface upstream sweep + four model
  cards were sequential and bounded). `COO-9` stays the lead path.

## Anchors

`docs/RUNBOOK_W116.md`, `docs/RESULTS_W116_UPSTREAM_ADMISSION_V1.md`,
`docs/CONTAMINATION_CONTROL_FRAMING_W116_V1.md`,
`docs/FRONTIER_RELEVANCE_AUDIT_W116_V1.md`,
`results/w116/upstream_admission/upstream_admission_verdict.json`.
