# W116 — upstream instrument-supply ATTACK (LIVE) + primary-source cutoff ATTACK + upstream-admission pipeline (no admissible new instrument; no certifiable stronger model; $0 NIM)

**One line:** W116 did the honest aggressive supply-side move W115 set up — instead
of passively waiting for a `release_v7`, it ATTACKED the upstream instrument supply
at FOUR authoritative surfaces and the model-disclosure side from PRIMARY sources,
confirming (a) NO admissible new instrument exists beyond `release_v6` and (b) the
last hypothesized tier-2 candidate (`mistralai/mistral-small-4-119b-2603`) is now a
CONFIRMED REAL model whose official card discloses NO cutoff — and shipped a durable
upstream-ADMISSION pipeline that makes the next clean shot push-button. The
certification verdict re-derives `NO_CERTIFIABLE_STRONGER_MODEL` (decision CID
`258b6ed7…`, byte-identical to W114/W115). **No pilot earned; $0 NIM. W89 + W105
remain the only two retirements.**

---

## The question (RUNBOOK_W116 § 1 Lane α)

Not "has someone published `release_v7` for us yet?" but the active supply-side
question: **can we CONSTRUCT the next certifiable post-cutoff functional instrument
from official UPSTREAM sources — going one level upstream of the pinned
`release_v6` — and has any reachable stronger-than-Maverick model disclosed a
primary-source KNOWN cutoff since W115?**

## Lane α — upstream instrument-supply ATTACK (LIVE, primary sources, 2026-05-30)

Verified at FOUR authoritative surfaces (vs W115's single lite-file-tree check),
WebSearch/WebFetch against official sources only (the chrome MCP is reserved for
`/browse` and is not used):

| Surface | Primary source | Finding |
|---|---|---|
| 1. Lite dataset file tree | HF API tree `livecodebench/code_generation_lite` | `test.jsonl`..`test6.jsonl`; **highest = `test6.jsonl` (134 MB); NO `test7`+**; lastModified **2025-06-05**; commit `0fe84c39`. |
| 2. Loader version definition | HF raw `code_generation_lite.py` `ALLOWED_FILES` | `v_list = [v1..v6]`; **`release_latest` → the SAME six files as `release_v6`**; `DEFAULT_CONFIG_NAME = "release_latest"`. **No `release_v7`. The upstream "latest" alias resolves to v6 — no hidden newer supply.** |
| 3. Full `code_generation` dataset | HF tree + README | Single `test.jsonl` (9.4 GB); README **`release_v6` = May 2023–Apr 2025, 1055 problems** (frontier **2025-04-05**). The full set's frontier is also Apr-2025 — no newer-dated problems. |
| 4. GitHub repo | `LiveCodeBench/LiveCodeBench` README + `/tags` | README tops out at `release_v6`; `/tags` shows no releases. **No `release_v7`.** |
| (corroboration) LCB website intro | livecodebench.github.io | Intro text STALE ("May 2023–Feb 2024"); **NOT authoritative** for the dataset frontier ⇒ ignored as a frontier source (the HF dataset + loader are the authority). |
| (rumor) "planned v7" | non-primary WebSearch summary | A summary mentioned a "planned v7" through late-2025/early-2026; **contradicted by all four authoritative surfaces; INADMISSIBLE** (no artifact, no SHA) — recorded as the W117 watch signal, not as supply. |

> **Net Lane α:** NO admissible new instrument exists beyond `release_v6`. The
> functional frontier is conclusively **2025-04-05**, now confirmed at four upstream
> surfaces including the `release_latest` alias resolution and the full dataset —
> not just the lite file tree W115 checked. The instrument supply did NOT move. The
> honest aggressive move was to attack the supply at the loader/dataset level and
> confirm it is genuinely absent, AND to land the admission pipeline so the next
> change is immediately usable (Lane γ).

## Lane β — primary-source model-cutoff ATTACK (LIVE, 2026-05-30)

Four-way disclosure-status matrix (RUNBOOK_W116 § 4b):

| Model | Primary source | Disclosure | >70B | Certification blocker |
|---|---|---|---|---|
| `meta/llama-4-maverick-17b-128e-instruct` | Official Llama 4 card (reconfirmed) | **KNOWN** (Aug-2024) | ✓ | **C4** (settled on `release_v6`; W113 FAIL) |
| `qwen/qwen3-coder-480b-a35b-instruct` | Official HF card | **UNKNOWN** (NO CUTOFF STATED) | ✓ | C1; C2-exposed if estimated ~2025 |
| `deepseek-ai/deepseek-v4-pro` | Official HF card + V4 card PDF | **UNKNOWN** (NO CUTOFF STATED) | ✓ | C1; a 2026 release ⇒ C2-exposed |
| `mistralai/mistral-small-4-119b-2603` | **Official Mistral docs card + official announcement** | **UNKNOWN from primary** (NO CUTOFF STATED); aggregator OpenRouter "2025-06" + stale "2023-10" ⇒ CONTRADICTORY | ✓ | C1; even the 2025-06 aggregator estimate post-dates the Apr-2025 frontier ⇒ C2-exposed |
| `mistralai/mistral-small-3.2-24b-instruct-2506` | HF discussion / aggregator | **KNOWN** (~Oct-2023) | ✗ | **C3** (24B; NOT stronger than 70B) |

> **Net Lane β:** NO reachable stronger-than-Maverick model has a primary-KNOWN
> cutoff ≤ 2025-01. **SHARPENED vs W115:** `mistralai/mistral-small-4-119b-2603` —
> a hypothesized placeholder in W113/W114 — is now CONFIRMED REAL (119B MoE,
> released 2026-03-16) and its OFFICIAL card discloses NO cutoff; the only cutoff
> figure is a non-primary aggregator (OpenRouter "2025-06") that is itself
> C2-exposed (post-dates the Apr-2025 frontier). A KNOWN-cutoff Mistral exists
> (Small 3.2, ~Oct-2023) but it is sub-70B (C3). Broad sweep: Jan-2025-cutoff models
> that surfaced (e.g. a Gemini Flash-Lite) are closed / not reachable / not in the
> candidate set.

## Lane γ — upstream-admission pipeline (NIM-free; `coordpy.upstream_instrument_admission_v1`)

result CID `193164c4…`; **decision CID `258b6ed7…` (byte-identical to the
W114/W115 decision** — the pipeline reuses, never forks, the W115
`run_frontier_certification_v1` → W114 `decide_certification_v1` → `certify_model_v1`
chain; asserted by `test_decision_cid_is_byte_identical_to_w114_w115`). Artifact
`results/w116/upstream_admission/upstream_admission_verdict.json`; corpus
re-verification `sha_ok` ✓ + `histogram_match` ✓ against the SHA-pinned `bb4c364f…`
`release_v6`.

The W115 snapshot-checker is generalised into a real upstream-ADMISSION pipeline:

1. **Admissibility rule** (`AdmissibilityRuleV1` / `assess_instrument_admissibility_v1`)
   — the pre-committed A1..A5 gate (authoritative source / dated / functional /
   SHA-pinnable+admittable / reproducible histogram). On the W116 snapshot:
   `release_v6` + the full dataset + the `release_latest` alias are
   **admissible-but-NOT-newer**; the "planned v7" rumor is **REFUSED** (A1 + A4
   fail) ⇒ **0 admissible NEW instruments**.
2. **Multi-surface upstream supply snapshot** (`UpstreamSupplySnapshotV1`) — the
   four-surface upstream state as DATA, wrapping the W115 `FrontierSnapshotV1`.
3. **Upstream-change detector** (`detect_upstream_change_v1`) — flags WHAT changed
   across surfaces (new numbered release / `release_latest` re-point / frontier
   advance / lastModified bump / loader-version extension / new admissible
   instrument). On the W116 snapshot: **`any_change = False`** (the W117 signal is
   quiet).
4. **Certifiable-slice builder** (`build_certifiable_slice_candidate_v1`) — reuses
   `certify_model_v1`; surfaces the resistant-slice size + the binding blocker per
   `(model, instrument)`.
5. **Four-way disclosure-status matrix** (`W116_DISCLOSURE_MATRIX`) — the Lane β
   record (KNOWN ×2 / UNKNOWN ×3; **no usable NEW KNOWN-cutoff target**).
6. **Structured W117 fire condition** (`W117FireConditionV1`) — instrument trigger +
   cutoff trigger; `fires_now = False`.

### Per-model go/no-go matrix (W116 snapshot)

| model | rank | cutoff | C1 | C2 | C3 | C4 | certifiable |
|---|---|---|---|---|---|---|---|
| `meta/llama-4-maverick-17b-128e-instruct` | 1.1 | 2024-08-31 [KNOWN] | ✓ | ✓ (63) | ✓ | ✗ (W113 settled) | **No** |
| `qwen/qwen3-coder-480b-a35b-instruct` | 2.1 | 2025-07-01 [UNKNOWN] | ✗ | ✗ (0) | ✓ | ✓ | **No** |
| `deepseek-ai/deepseek-v4-pro` | 2.2 | 2025-01-01 [UNKNOWN] | ✗ | ✓ (49)¹ | ✓ | ✓ | **No** |
| `mistralai/mistral-small-4-119b-2603` | 2.3 | 2026-03-01 [UNKNOWN] | ✗ | ✗ (0) | ✓ | ✓ | **No** |

¹ DeepSeek's C2 passes only under its UNKNOWN *estimate*; C1 (KNOWN-only) blocks it
regardless — the W113/W114 discipline.

**Verdict: `NO_CERTIFIABLE_STRONGER_MODEL`.** `disclosure_consistency_ok = True`.
`n_admissible_new_instruments = 0`. `W117 fires_now = False`
(instrument_trigger_met = cutoff_trigger_met = False).

## Pilot decision (RUNBOOK_W116 § 6 / § 7)

A pilot is earned ONLY if BOTH (1) Lane α yields an admissible NEW instrument AND
(2) Lane β yields a non-redundant stronger model with a primary-KNOWN cutoff that
certifies. Lane α = 0 admissible NEW instruments; Lane β = no usable NEW KNOWN-cutoff
target ⇒ **NO pilot earned; $0 NIM** (the pre-committed no-go branch; the no-go is
discipline, not omission). No Maverick rerun (no GENUINELY NEW instrument exists;
the same-`release_v6` cell is settled — W106 redundant-run discipline).

## What this establishes / does NOT

* **DOES** turn the W115 supply blocker from a single-surface snapshot into a
  FOUR-surface upstream-verified, dated finding incl. the `release_latest` alias and
  the full dataset (`W116-L-UPSTREAM-SUPPLY-NO-ADMISSIBLE-NEW-INSTRUMENT-FOUR-SURFACE-CAP`).
* **DOES** confirm `mistralai/mistral-small-4-119b-2603` is now a REAL model whose
  primary card discloses NO cutoff — sharpening the model-disclosure blocker
  (`W116-T-MISTRAL-SMALL-4-CONFIRMED-REAL-PRIMARY-NO-CUTOFF`).
* **DOES** ship a durable upstream-ADMISSION pipeline (admissibility rule + multi-
  surface change detector + slice builder + disclosure matrix + W117 condition) that
  makes W117 push-button (`W116-T-UPSTREAM-ADMISSION-PIPELINE-SHIPS`).
* **Does NOT** spend NIM (no admissible new instrument; no certifiable model;
  Maverick redundant), add a retirement, weaken W89/W105, move the contamination
  confound (W116 tests certification supply, not the confound), or re-probe
  reachability (not the binding gate; W112 facts carried).

## W117 (the loaded next move)

W117 fires the moment `detect_upstream_change_v1` flags an admissible change — a
newer admitted `release_v7`+ (or `release_latest` re-pointing past v6, or a distinct
upstream functional dataset) with ≥30 functional problems dated strictly after a
reachable stronger-than-Maverick model's primary-KNOWN cutoff — OR a reachable
stronger-than-Maverick model disclosing a primary-KNOWN cutoff month ≤ 2025-01.
Re-run `run_upstream_admission_v1` against the updated snapshot → if any model
certifies on an admissible instrument, run the pre-committed cheapest-honest pilot.
Until one holds, the bounded ceiling STANDS and resistant-code NIM is BLOCKED.
`COO-9` stays lead.

Anchors: `docs/RUNBOOK_W116.md`, `docs/RESULTS_W116_MILESTONE_SUMMARY_V1.md`,
`docs/CONTAMINATION_CONTROL_FRAMING_W116_V1.md`,
`docs/FRONTIER_RELEVANCE_AUDIT_W116_V1.md`,
`results/w116/upstream_admission/upstream_admission_verdict.json`.
