# W114 — per-model post-cutoff certification + instrument-frontier gate (no stronger model certifiable; $0 NIM)

**One line:** from the LATEST real LiveCodeBench release and the OFFICIAL model
cutoffs (verified 2026-05-29 from primary sources), **no reachable model stronger
than Llama-4-Maverick is certifiably contamination-resistant on the available
instrument** — the resistant FUNCTIONAL instrument frontier (Apr-2025) has aged
out relative to the reachable frontier-model class, whose cutoffs are officially
undisclosed. Certification verdict `NO_CERTIFIABLE_STRONGER_MODEL`; **$0 NIM**;
no pilot earned. W89 + W105 STAND; W114 adds none.

---

## The question (RUNBOOK_W114 § 1 Lane β)

Not "find another borderline positive" but: **can a NEW instrument be built that
is certifiably contamination-resistant for a model STRONGER than Maverick, from
the latest REAL release and OFFICIAL cutoffs, and earn one clean shot on it?**

## Primary-source verification pass (2026-05-29)

| Surface | Primary source | Finding |
|---|---|---|
| Latest LCB release | HF dataset file tree of `livecodebench/code_generation_lite` (`test.jsonl`..`test6.jsonl`; highest = `test6.jsonl`) | **`release_v6` is the latest**; no `test7`+. Full release May 2023–Apr 2025; the **FUNCTIONAL/lite subset = 63 problems, all 2025-01-11..2025-04-05** (frontier **2025-04-05**). The HF README prose lists only through v5 (stale); the file tree + the local SHA-pinned `test6.jsonl` + the "add v6" commits confirm v6. |
| Llama-4-Maverick cutoff | Official Llama 4 model card (llama.com / Meta GitHub `llama-models/.../llama4/MODEL_CARD.md`; NVIDIA build modelcard) | **August 2024 — KNOWN** ("pretraining data has a cutoff of August 2024"). |
| Qwen3-Coder-480B cutoff | Official HF model card (`Qwen/Qwen3-Coder-480B-A35B-Instruct`) + Qwen blog (`qwenlm.github.io/blog/qwen3-coder`) | **NO CUTOFF STATED — UNKNOWN.** Released 2025-07. |
| DeepSeek-V4-pro cutoff | DeepSeek official sources | **Not disclosed — UNKNOWN** (V3 ≈ Jul-2024 per a non-official system-prompt extraction; V4 post-dates it). |
| Mistral-Small-4-2603 cutoff | Official Mistral docs / HF model card (`mistral-small-4-0-26-03`) | **NO CUTOFF STATED — UNKNOWN.** Released 2026-03-16 (post-dates the entire release_v6 window). |

> Tool note: the primary-source verification used WebSearch/WebFetch against the
> official HF dataset/model cards + vendor blogs — the appropriate research
> primitives for fetching official documentation (NOT the chrome browser-
> automation MCP, which the global guidance reserves for `/browse`).

## Instrument frontier (corpus-grounded, NIM-free)

From the SHA-pinned `release_v6` (`test6.jsonl`; SHA `bb4c364f…`; functional
subset = 63), re-verified live by the script (`sha_ok` + `histogram_match`):

* functional `contest_date` span **2025-01-11 .. 2025-04-05**; month histogram
  (resistant-for-Maverick, > 2024-08-31): 2025-01 = 14, 2025-02 = 20,
  2025-03 = 27, 2025-04 = 2 (total 63).
* **A ≥ 30 functional resistant slice requires a KNOWN cutoff ≤ ~2025-01-31**
  (cutoff > 2025-01-31 → 49; > 2025-02-28 → 29 < 30; > 2025-03-31 → 2;
  > 2025-04-30 → 0).

## Certification (C1∧C2∧C3∧C4; `coordpy.stronger_model_cutoff_certification_v1`)

decision CID `258b6ed7…`; `results/w114/certification/certification_verdict.json`.

| model | rank | cutoff | C1 KNOWN | C2 ≥30 | C3 reach/stronger | C4 not-settled | certifiable |
|---|---|---|---|---|---|---|---|
| `meta/llama-4-maverick-17b-128e-instruct` | 1.1 | 2024-08-31 [KNOWN] | ✓ | ✓ (63) | ✓ | ✗ (W113 settled) | **No** |
| `qwen/qwen3-coder-480b-a35b-instruct` | 2.1 | 2025-07-01 [UNKNOWN] | ✗ | ✗ (0) | ✓ | ✓ | **No** |
| `deepseek-ai/deepseek-v4-pro` | 2.2 | 2025-01-01 [UNKNOWN] | ✗ | ✓ (49)¹ | ✓ | ✓ | **No** |
| `mistralai/mistral-small-4-119b-2603` | 2.3 | 2026-03-01 [UNKNOWN] | ✗ | ✗ (0) | ✓ | ✓ | **No** |

¹ DeepSeek's C2 passes only under its UNKNOWN *estimate* boundary (2025-01-01 →
49); C1 (KNOWN-cutoff-only) blocks it regardless — exactly the W113 discipline:
you cannot certify resistance against a cutoff you cannot verify.

**Verdict: `NO_CERTIFIABLE_STRONGER_MODEL`.** Maverick is C1∧C2∧C3 but
C4-blocked (settled at W113 → a second pilot is redundant, no verdict-changing
power). Every stronger-than-Maverick frontier model is C1-blocked (officially
undisclosed cutoff) and, where estimable, C2-blocked (cutoff at/after the
Apr-2025 frontier). The gaps COMPOUND.

## What this establishes / does NOT

* **DOES** establish the **certification-supply blocker**: the latest resistant
  FUNCTIONAL instrument has aged out relative to the reachable frontier-model
  class (`W114-L-RESISTANT-INSTRUMENT-FRONTIER-LAGS-MODEL-FRONTIER-CAP`), and the
  reachable frontier models' cutoffs are officially undisclosed
  (`W114-T-STRONGER-MODEL-CUTOFFS-OFFICIALLY-UNDISCLOSED`). This is a genuinely
  different axis (a certification-supply analysis), not another exposed/resistant
  reflexion pilot.
* **DOES** register the bounded ceiling as the honest code-superiority FLOOR
  (`W114-T-BOUNDED-EXPOSED-CODE-CEILING-REGISTERED`).
* **Does NOT** spend NIM (no model is certifiable; Maverick is redundant).
* **Does NOT** add a retirement, weaken W89/W105, move the contamination
  confound, or re-probe reachability (not the binding gate; W112 facts carried).

## W115 (the loaded next move)

W115 fires only when a resistant FUNCTIONAL instrument with ≥30 problems dated
strictly after a reachable frontier model's KNOWN cutoff exists: **(a)** a future
LCB `release_v7`+ (or equivalent freshly-dated functional benchmark) with
post-2025-04 functional problems, operator-fetched + SHA-pinned + loader-admitted,
AND **(b)** a reachable stronger-than-Maverick model whose cutoff is officially
DISCLOSED (KNOWN) and earlier than those problems. Until one holds, the bounded
ceiling STANDS and resistant-code NIM is BLOCKED. `COO-9` stays lead.

Anchors: `docs/RUNBOOK_W114.md`, `docs/RESULTS_W114_MILESTONE_SUMMARY_V1.md`,
`docs/CONTAMINATION_CONTROL_FRAMING_W114_V1.md`,
`results/w114/certification/certification_verdict.json`.
