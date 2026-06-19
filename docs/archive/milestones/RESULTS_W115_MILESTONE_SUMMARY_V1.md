# W115 — Milestone summary (live external-frontier refresh + future-fire pipeline; $0 NIM)

**One line:** W115 RE-VERIFIED the external frontier LIVE from primary sources
(2026-05-29) — confirming no newer LiveCodeBench release and no newly-disclosed
KNOWN cutoff for any reachable model stronger than Maverick — AND shipped a durable
future-fire certification/instrument-supply pipeline that makes the next clean shot
push-button. The certification verdict re-derives `NO_CERTIFIABLE_STRONGER_MODEL`.
**No pilot earned; $0 NIM. W89 + W105 remain the only two retirements.**

---

## The three lanes

### Lane α — external-frontier refresh (LIVE; the main empirical lane)

Re-verified from PRIMARY sources (RUNBOOK_W115 § 2):
* **Latest LCB release**: HF file tree's highest test file is still `test6.jsonl`
  (no `test7`+; "add v6" ~1 yr ago) ⇒ `release_v6` STILL latest; functional
  frontier 2025-04-05 UNCHANGED.
* **Model cutoffs**: Llama-4-Maverick Aug-2024 KNOWN (reconfirmed, settled);
  Qwen3-Coder-480B NO cutoff (UNKNOWN, fetched live); **DeepSeek-V4-pro** — the
  official V4 model card now EXISTS (published 2026-04-27, 1.6T/49B) but discloses
  NO cutoff (UNKNOWN); Mistral-Small-4 UNKNOWN. No newly-reachable stronger model
  with a KNOWN cutoff ≤ ~Jan-2025.
* **Net**: both binding conditions still fail — no newer instrument, no reachable
  stronger model with a KNOWN cutoff ≤ ~Jan-2025. The external frontier did NOT move
  in any verdict-relevant way since W114. Reachability NOT re-probed (not the binding
  gate; W112 facts carried). $0 NIM.

### Lane β — future-fire certification/instrument-supply pipeline (NIM-free; mandatory)

Built `coordpy.frontier_certification_pipeline_v1` (explicit-import-only; reuses the
W113 registry + `partition_resistant_v1` + the W114 `certify_model_v1` /
`decide_certification_v1` / instrument + the loader's `LIVECODEBENCH_KNOWN_RELEASES`,
no duplication) + `scripts/run_w115_frontier_certification_v1.py`. The pipeline
generalises the W114 one-shot certification into a push-button supply-chain tool:
* a **latest-official-release detector** (`newer_release_available`) — turns "is
  there a release_v7+?" into an operational boolean (False on the W115 snapshot);
* a **frontier-date summary + threshold table** (max KNOWN cutoff month for a ≥30
  slice = 2025-01), generalised over ANY instrument;
* a **per-model go/no-go matrix** with exact blocker reasons, driven by a
  `FrontierSnapshotV1` (external state as DATA), plus a disclosure-consistency guard
  (live vs encoded registry — a divergence is the W116 update signal) and a
  structured `W116FireConditionV1`.
The script re-verifies the pinned histogram against the SHA-pinned corpus (`sha_ok`
+ `histogram_match` ✓) and emits `results/w115/frontier_certification/
frontier_certification_verdict.json` (result CID `6890419c…`; decision CID
`258b6ed7…` = the W114 decision, byte-for-byte).

### Lane γ — claim / graphify / readiness (NIM-free; mandatory)

graphify refreshed from HEAD at start (`f8b085d`; 0 token cost) + re-ingested the
new W115 module/script + re-refreshed at close; `explain`/`path`/`affected`/`query`
used for file selection + dependency checks (the pipeline is confirmed a wrapper
that reuses the W114 gate + W113 registry + loader, no duplication). Truth surfaces
tightened across RESEARCH_STATUS / THEOREM_REGISTRY / HOW_NOT_TO_OVERSTATE /
CONSOLIDATED narrative / new CONTAMINATION_CONTROL_FRAMING_W115 / new
FRONTIER_RELEVANCE_AUDIT_W115 / CHANGELOG. 10 new W115 tests (incl. a
falsifiability test that a newer release + a KNOWN-cutoff stronger model DOES
certify and fires W116) + 42 reused W113/W114 tests pass.

---

## Truth surface after W115

* **Confirmed retirements: still exactly TWO** — W89 + W105, both Llama-3.3-70B @
  70B, contamination-EXPOSED-HumanEval-family. W115 adds NONE, retires NONE.
* **Resistant superiority: 0 clean across BOTH scales** (unchanged; registered
  ceiling).
* **Certification supply: BLOCKED, LIVE-re-verified.** No reachable stronger-than-
  Maverick model is certifiable on the latest real instrument; the blocker is
  precise, dated, and now confirmed current (incl. against the new DeepSeek V4 card).
* **Contamination-confound: UNCHANGED** (W115 tests certification supply, not the
  confound; STRENGTHENED-not-proven, per W113).
* Still NOT cross-class, NOT a resistant win, NOT MBPP-family, NOT cross-modal,
  NOT "context solved".

## Entitlement delta

NOT entitled to a stronger superiority/retirement claim. The bounded two-retirement
contamination-EXPOSED-HumanEval-family-at-70B claim is the registered ceiling
(W114), now with the supply blocker LIVE-re-verified (W115) and OPERATIONALISED
into a push-button pipeline — a sharper, more defensible position than W114's
one-day-old snapshot, not a vague "needs more work".

## Carry-forward delta

* **Added**: `W115-L-EXTERNAL-FRONTIER-UNCHANGED-NO-CERTIFIABLE-SLICE-REVERIFIED-CAP`
  (the live re-check confirms the blocker) + `W115-T-FUTURE-FIRE-CERTIFICATION-
  PIPELINE-SHIPS` (the durable supply-chain asset).
* **Retired**: none.
* **Re-affirmed**: the W114 caps (`W114-L-RESISTANT-INSTRUMENT-FRONTIER-LAGS-MODEL-
  FRONTIER-CAP`, `W114-T-STRONGER-MODEL-CUTOFFS-OFFICIALLY-UNDISCLOSED`,
  `W114-T-BOUNDED-EXPOSED-CODE-CEILING-REGISTERED`) STAND.

## W116 (made push-button by W115)

W116 fires the moment the pipeline trigger flips: a newer admitted `release_v7`+
with ≥30 post-frontier functional problems for a KNOWN-cutoff stronger model, OR a
reachable stronger-than-Maverick model disclosing a KNOWN cutoff month ≤ 2025-01.
Re-run `run_frontier_certification_v1` against the updated snapshot → if any model
certifies, run the pre-committed cheapest-honest pilot on the strongest target.
Until then the bounded ceiling STANDS and resistant-code NIM is BLOCKED. `COO-9`
stays lead.

## Discipline / boundary

* 25th consecutive preflight/earn-discipline validation (W93–W115): runbook
  (`docs/RUNBOOK_W115.md`) locked before any NIM; the no-go branch is pre-committed
  by the rule, so the $0 spend is discipline, not omission.
* Stable boundary preserved: `0.5.20` / `coordpy.sdk.v3.43`; no PyPI;
  `coordpy/__init__.py` untouched; 1 new explicit-import-only module + 1 script +
  10 new tests.
* `ultracode` stayed OFF (a bounded frontier-refresh + supply-chain milestone, not
  a repo-wide dynamic-workflow job). `COO-9` stays the lead path.

## Anchors

`docs/RUNBOOK_W115.md`, `docs/RESULTS_W115_FRONTIER_CERTIFICATION_V1.md`,
`docs/CONTAMINATION_CONTROL_FRAMING_W115_V1.md`,
`docs/FRONTIER_RELEVANCE_AUDIT_W115_V1.md`,
`results/w115/frontier_certification/frontier_certification_verdict.json`.
