# W115 — external-frontier refresh (LIVE) + future-fire certification pipeline (no certifiable stronger model; $0 NIM)

**One line:** a LIVE primary-source re-verification (2026-05-29) confirms the
external frontier has NOT moved in any verdict-relevant way since W114 — LCB
`release_v6` is still the latest official release (no `test7`+; frontier
2025-04-05), and no reachable model stronger than Llama-4-Maverick has a KNOWN
cutoff ≤ ~Jan-2025 (Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4 cutoffs
remain officially undisclosed, incl. against the **now-existing** DeepSeek V4 card
published 2026-04-27). W115 ALSO ships a durable future-fire **supply-chain
pipeline** (`coordpy.frontier_certification_pipeline_v1`) that makes the next clean
shot push-button. Verdict re-derives `NO_CERTIFIABLE_STRONGER_MODEL`; **$0 NIM**;
no pilot earned. W89 + W105 STAND; W115 adds none.

---

## The question (RUNBOOK_W115 § 1 Lane α)

Not "find a dirty benchmark and squeeze a margin" but the time-sensitive supply
question: **has the external world changed enough — a newer official LiveCodeBench
release, or a newly-disclosed KNOWN cutoff for a reachable stronger model — that a
clean, certifiably-resistant pilot for a model STRONGER than Maverick is now
possible?**

## LIVE primary-source verification pass (2026-05-29)

Sources are PRIMARY only (official HF dataset/model cards + vendor docs/PDFs;
aggregators corroborate at most). Tooling: WebSearch/WebFetch (the chrome MCP is
reserved for `/browse` and is not used) — the documented W114 convention.

| Surface | Primary source | Finding (vs W114) |
|---|---|---|
| Latest LCB release | HF file tree of `livecodebench/code_generation_lite`: `test.jsonl`..`test6.jsonl`; **highest = `test6.jsonl` (134 MB)**; latest commit "add v6" ~1 yr ago; **no `test7.jsonl`+** | **`release_v6` STILL latest — UNCHANGED.** Functional subset 63 problems, 2025-01-11..2025-04-05 (frontier 2025-04-05). |
| Llama-4-Maverick cutoff | Official Llama 4 model card (model-cards docs; multi-source corroborated) | **August 2024 — KNOWN.** UNCHANGED. Settled on `release_v6` (W113 resistant FAIL ⇒ C4). |
| Qwen3-Coder-480B cutoff | Official HF model card (`Qwen/Qwen3-Coder-480B-A35B-Instruct`) — fetched live | **NO CUTOFF STATED — UNKNOWN.** UNCHANGED. Released 2025-07-22; estimable cutoff ~2025 ≥ Apr-2025 frontier ⇒ C2-exposed even if disclosed. |
| DeepSeek-V4-pro cutoff | **Official DeepSeek V4 model card PDF** (`fe-static.deepseek.com/.../deepseek-V4-model-card-EN.pdf`; **published 2026-04-27**; Pro = 1.6T params / 49B activated) — fetched + text-extracted live | **NO "cutoff" string anywhere; no training-data date — UNKNOWN.** **SHARPENED**: W114 noted "no card published"; the official card now EXISTS (2026-04-27) and STILL discloses no cutoff. A 2026-04 release ⇒ real cutoff ≥2025 ⇒ C2-exposed. |
| Mistral-Small-4-2603 cutoff | Official Mistral docs/HF (real line = Mistral-Small-3.2-2506) | **NO CUTOFF STATED for the candidate — UNKNOWN.** The real reachable Mistral line (Small 3.2, 2025-06) is weaker than Maverick and not the candidate; the 2026-03 tag post-dates the whole window ⇒ C2-exposed regardless. |
| Any NEW reachable stronger code model w/ KNOWN cutoff ≤ ~Jan-2025 | Broad official-source sweep | **NONE.** Stronger-than-Maverick models that surfaced (DeepSeek V4, Qwen 3.5/3.6) are NEWER ⇒ later cutoffs ⇒ C2-exposed; none discloses a KNOWN cutoff ≤ ~Jan-2025. |

> **The one genuine external change since W114** — the DeepSeek V4 official model
> card was *published* (2026-04-27) — does **NOT** move the verdict: the card
> discloses no training cutoff, so DeepSeek-V4-pro stays UNKNOWN (C1-blocked), and
> a 2026-04 frontier model's real cutoff is ≥2025 (C2-exposed) anyway. The blocker
> is confirmed real and current, not a stale carry-forward.

## Instrument frontier (corpus-grounded, NIM-free)

From the SHA-pinned `release_v6` (`test6.jsonl`; SHA `bb4c364f…`; functional subset
= 63), re-verified live by the script (`sha_ok` ✓ + `histogram_match` ✓):

* functional `contest_date` span **2025-01-11 .. 2025-04-05**; month histogram
  2025-01 = 14, 2025-02 = 20, 2025-03 = 27, 2025-04 = 2 (total 63).
* threshold table (cutoff month → resistant count): `<before-all>` = 63,
  2025-01 = 49, 2025-02 = 29, 2025-03 = 2, 2025-04 = 0.
* **max KNOWN cutoff month admitting a ≥30 resistant slice = 2025-01.**

## Future-fire pipeline (Lane β; `coordpy.frontier_certification_pipeline_v1`)

result CID `6890419c…`; decision CID `258b6ed7…` (**byte-identical to the W114
decision** — the pipeline wraps, never forks, the W114 gate).
Artifact `results/w115/frontier_certification/frontier_certification_verdict.json`.

The W114 one-shot certification is generalised into a durable, push-button
supply-chain pipeline driven by a `FrontierSnapshotV1` (the external state as DATA):

1. **latest-official-release detector** (`detect_latest_release_v1`) — compares
   releases OBSERVED on the live source vs the loader-ADMITTED set; flags
   `newer_release_available`. On the W115 snapshot: admitted latest = observed
   latest = `release_v6` ⇒ **`newer_release_available = False`**.
2. **frontier-date summary** (`frontier_date_summary_v1`) — the month histogram +
   frontier date + threshold table + `max_cutoff_month_for_min_slice`, generalised
   over ANY instrument (not a hard-coded constant).
3. **certifiable-slice candidate builder + per-model go/no-go matrix**
   (`run_frontier_certification_v1`) — reuses the W114 `certify_model_v1` /
   `decide_certification_v1` gate (no duplication), adds the disclosure-consistency
   guard (live disclosures vs the encoded W113/W114 registry — a divergence is the
   W116 update signal) and the structured `W116FireConditionV1`.

### Per-model go/no-go matrix (W115 snapshot)

| model | rank | cutoff | C1 | C2 | C3 | C4 | certifiable |
|---|---|---|---|---|---|---|---|
| `meta/llama-4-maverick-17b-128e-instruct` | 1.1 | 2024-08-31 [KNOWN] | ✓ | ✓ (63) | ✓ | ✗ (W113 settled) | **No** |
| `qwen/qwen3-coder-480b-a35b-instruct` | 2.1 | 2025-07-01 [UNKNOWN] | ✗ | ✗ (0) | ✓ | ✓ | **No** |
| `deepseek-ai/deepseek-v4-pro` | 2.2 | 2025-01-01 [UNKNOWN] | ✗ | ✓ (49)¹ | ✓ | ✓ | **No** |
| `mistralai/mistral-small-4-119b-2603` | 2.3 | 2026-03-01 [UNKNOWN] | ✗ | ✗ (0) | ✓ | ✓ | **No** |

¹ DeepSeek's C2 passes only under its UNKNOWN *estimate* (2025-01-01 → 49); C1
(KNOWN-cutoff-only) blocks it regardless — the W113/W114 discipline.

**Verdict: `NO_CERTIFIABLE_STRONGER_MODEL`.** `disclosure_consistency_ok = True`
(all live disclosures match the encoded registry — nothing moved).
`W116 fires_now = False` (instrument_trigger_met = cutoff_trigger_met = False).

## What this establishes / does NOT

* **DOES** turn the W114 supply blocker from a one-day-old snapshot into a
  LIVE-re-verified, dated finding (incl. against the brand-new DeepSeek V4 card)
  (`W115-L-EXTERNAL-FRONTIER-UNCHANGED-NO-CERTIFIABLE-SLICE-REVERIFIED-CAP`).
* **DOES** ship a durable future-fire pipeline that makes W116 push-button
  (`W115-T-FUTURE-FIRE-CERTIFICATION-PIPELINE-SHIPS`): the operator updates ONE
  snapshot (newer observed release / newly-disclosed cutoff) and re-runs.
* **Does NOT** spend NIM (no model certifiable; Maverick redundant; no newer
  instrument), add a retirement, weaken W89/W105, move the contamination confound
  (W115 tests certification supply, not the confound), or re-probe reachability
  (not the binding gate; W112 facts carried).

## W116 (the loaded next move)

W116 fires the moment the pipeline's trigger flips:
* **instrument trigger** — a newer official LCB `release_v7`+ is observed +
  operator-fetched + SHA-pinned + admitted, with ≥30 functional problems dated
  strictly after a reachable stronger-than-Maverick model's KNOWN cutoff; **or**
* **cutoff trigger** — a reachable stronger-than-Maverick model discloses a KNOWN
  cutoff month ≤ 2025-01 (so `release_v6` admits a ≥30 resistant slice for it);
  update the registry + provenance, then re-run `run_frontier_certification_v1`.

Until one holds, the bounded ceiling STANDS and resistant-code NIM is BLOCKED.
`COO-9` stays lead.

Anchors: `docs/RUNBOOK_W115.md`, `docs/RESULTS_W115_MILESTONE_SUMMARY_V1.md`,
`docs/CONTAMINATION_CONTROL_FRAMING_W115_V1.md`,
`results/w115/frontier_certification/frontier_certification_verdict.json`.
