# W117 — upstream-DERIVED instrument CONSTRUCTION attack (LIVE) + deeper primary-source cutoff attack + construction/admission pipeline (no construction-admissible instrument; no certifiable stronger model; $0 NIM)

**One line:** W117 did the harder aggressive move W116 set up — instead of only
checking for a *packaged* `release_v7`, it attacked the upstream CONSTRUCTION
provenance at EIGHT authoritative surfaces (the revision history + collection
mechanism, not just the release label) and the model-disclosure side DEEPER from
primary sources, proving (a) NO post-v6 instrument can be CONSTRUCTED from
authoritative provenance (the only post-v6 path — raw-contest hand-assembly — is
refused by the pre-committed B1 + B2 construction criteria) and (b) the DeepSeek V4
official model-card PDF, re-checked at primary, still discloses NO cutoff — and shipped
a durable upstream-DERIVED construction/admission pipeline that makes the next clean
shot push-button. The certification verdict re-derives `NO_CERTIFIABLE_STRONGER_MODEL`
(decision CID `258b6ed7…`, byte-identical to W114/W115/W116). **No pilot earned; $0
NIM. W89 + W105 remain the only two retirements.**

---

## The question (RUNBOOK_W117 § 1 Lane α)

Not "is there an admissible *packaged* `release_v7`?" (W116 answered No) but the harder
construction question: **even before a numbered `release_v7` is packaged, can we
CONSTRUCT an upstream-authoritative, machine-checkable, certifiable post-cutoff
functional instrument from official upstream PROVENANCE — the HF dataset revision/
commit/discussion history, the official LCB GitHub data-generation pipeline, and the
actual upstream contest provenance? And has any reachable stronger-than-Maverick model
disclosed a primary-source KNOWN cutoff since W116 when probed DEEPER than a single
card?**

## Lane α — upstream-derived CONSTRUCTION attack (LIVE, primary sources, 2026-05-30)

W116 verified the latest *packaged release* at four surfaces. W117 attacks the
*construction provenance* at EIGHT surfaces — the revision history and collection
mechanism (WebSearch/WebFetch against official HF/GitHub APIs + raw files; the chrome
MCP is reserved for `/browse` and is not used):

| # | Provenance surface | Primary source | Finding |
|---|---|---|---|
| 1 | HF dataset commit/revision log | `api/.../code_generation_lite/commits/main` | 20 commits; latest data-bearing = **"add v6" 2025-04-21**; HEAD = "fix typos (#4)" 2025-06-05. **No post-v6 data commit; no `test7`.** |
| 2 | HF refs (branches/tags) | `…/refs` | **1 branch (`main`), 0 tags.** No staging / preview / v7 branch. |
| 3 | HF discussions / PRs | dataset `/discussions` | Newest = "LCB pull request" #14 + "Clarification on v6 size (454 vs 175)" #13. **No v7 / newer-data thread.** |
| 4 | LCB GitHub commit history | `api/.../commits` | Newest **2025-07-16** ("fix: typo in Explorer URL"); recent = runner maintenance (model infos, gemini, ERRATA). **No data-collection commit; no v7.** |
| 5 | LCB GitHub tags/releases | `api/.../tags` | **Empty (0 tags).** |
| 6 | GitHub repo data-pipeline structure | repo root contents | Top-level = `lcb_runner/` + `assets/` + configs. **NO scraper / data / collect / contest directory** ⇒ the public repo is the runner/eval harness, NOT a published collection pipeline. |
| 7 | Dataset README provenance | dataset `raw/main/README.md` | Provenance = LeetCode / AtCoder / Codeforces; each `release_vN` a temporal snapshot (prose lags at v5). **Documents ONLY loading published releases — NO generation tool / script / manifest.** |
| 8 | Runner data-loading path | `lcb_runner/benchmarks/code_generation.py` | Loads **exclusively** via `load_dataset("livecodebench/code_generation_lite", version_tag=…)`; **no local scraping / construction.** |

> **Net Lane α:** the authoritative construction provenance IS the packaged HF release.
> No post-v6 LCB-published artifact exists at ANY of the eight surfaces; LCB does not
> publish its collection pipeline or a forward problem-id manifest. The provenance
> flows contest-sites → LCB's (private, unpublished) collection → HF release → runner,
> so construction can only enter at the HF-release level — which requires LCB to
> publish it. The ONLY post-v6 path is a raw-contest hand-assembly, which is
> **CONSTRUCTION-INADMISSIBLE** (refused by A1 ∧ B1 ∧ B2). ⇒ **0 construction-admissible
> NEW instruments.** This is sharper than "no v7": it proves a post-v6 instrument
> cannot be *constructed* from authoritative provenance, only hand-curated (refused)
> (`W117-L-NO-CONSTRUCTION-ADMISSIBLE-POST-V6-INSTRUMENT-EIGHT-SURFACE-CAP` +
> `W117-T-LCB-CONSTRUCTION-PROVENANCE-IS-PACKAGED-RELEASE`).

## Lane β — deeper primary-source model-cutoff attack (LIVE, 2026-05-30)

Sharpened disclosure-status matrix (RUNBOOK_W117 § 4b), probed DEEPER than a single
card (official PDFs / release notes / vendor docs):

| Model | Primary source (verbatim probe) | Disclosure | >70B | Blocker |
|---|---|---|---|---|
| `meta/llama-4-maverick-17b-128e-instruct` | Meta `llama-models/.../llama4/MODEL_CARD.md` — **"August 2024"** verbatim | **KNOWN** | ✓ | **C4** (settled; W113) |
| `qwen/qwen3-coder-480b-a35b-instruct` | Official HF card raw README — **NO CUTOFF STATED** | **UNKNOWN** | ✓ | C1 |
| `deepseek-ai/deepseek-v4-pro` | **Official DeepSeek V4 model-card PDF** — **NO CUTOFF STATED** (aggregator "Apr 2026" is non-primary) | **UNKNOWN** (primary) | ✓ | C1; aggregator figure (Apr-2026) post-dates frontier ⇒ C2-exposed |
| `mistralai/mistral-small-4-119b-2603` | Official Mistral docs models overview — **NO CUTOFF STATED** | **UNKNOWN** (primary) | ✓ | C1; aggregator "2025-06" post-dates frontier ⇒ C2-exposed |
| `mistralai/mistral-small-3.2-24b-instruct-2506` | HF discussion / aggregator | **KNOWN** (~Oct-2023) | ✗ | **C3** (24B) |

> **Net Lane β:** NO reachable stronger-than-Maverick model has a primary-KNOWN cutoff
> ≤ 2025-01; **nothing newly-disclosed since W116.** SHARPENED on DeepSeek: the
> official V4 model-card PDF re-checked at primary still states no cutoff — the only
> figure is a non-primary aggregator ("Apr 2026") that is itself C2-exposed (a year
> past the Apr-2025 frontier) — so DeepSeek V4 now matches Mistral Small 4's exact
> "UNKNOWN-from-primary + C2-exposed-aggregator" pattern. Maverick's "August 2024" was
> re-confirmed VERBATIM from the Meta MODEL_CARD.md. The broad sweep (Qwen3.5 / GLM-5 /
> Kimi K2.6 / Gemma 4 / DeepSeek R1) surfaced only listicles — none reachable+stronger
> with a primary-KNOWN cutoff ≤ 2025-01 (`W117-T-DEEPSEEK-V4-PRIMARY-PDF-RECONFIRMED-NO-CUTOFF`).

## Lane γ — construction/admission pipeline (NIM-free; `coordpy.upstream_derived_instrument_construction_v1`)

result CID `c3c60483…`; **decision CID `258b6ed7…` (byte-identical to the
W114/W115/W116 decision** — W117 reuses, never forks, the W116
`run_upstream_admission_v1` → W115 `run_frontier_certification_v1` → W114
`decide_certification_v1` → `certify_model_v1` chain; asserted by
`test_decision_cid_is_byte_identical_to_w114_w115_w116` + graphify-confirmed
`run_upstream_construction_v1 --calls--> run_upstream_admission_v1`). Artifact
`results/w117/upstream_construction/upstream_construction_verdict.json`; corpus
re-verification `sha_ok` ✓ + `histogram_match` ✓ against the SHA-pinned `bb4c364f…`
`release_v6`.

The W116 packaged-admission pipeline is generalised into a real upstream-DERIVED
CONSTRUCTION pipeline:

1. **Construction rule** (`ConstructionRuleV1` / `assess_construction_admissibility_v1`)
   — the pre-committed gate: A1..A5 (reused W116 assessor) PLUS **B1 authoritative
   construction provenance** (the problem set is fully defined by an LCB-PUBLISHED
   artifact — a dataset revision/commit/PR OR a published collection pipeline +
   problem-id manifest — NOT a raw-contest hand-assembly) AND **B2 no operator
   curation** (selection + ordering reproducible from the published provenance; no
   operator discretion — the anti-cherry-pick criterion).
2. **EIGHT-surface provenance snapshot** (`UpstreamProvenanceSnapshotV1`) — the
   construction-provenance state as DATA, wrapping the W116 `UpstreamSupplySnapshotV1`.
3. **Candidate-instrument constructor** (`construct_upstream_derived_candidate_v1`) —
   scans the surfaces + the construction proposals; if a proposal is
   construction-admissible AND its LCB-published artifact actually exists, it DERIVES
   the post-v6 observation; otherwise it returns the EXACT missing artifact.
4. **Sharpened disclosure matrix** (`W117_DISCLOSURE_MATRIX`) — the Lane β record
   (KNOWN ×2 / UNKNOWN ×3; **no usable NEW KNOWN-cutoff target; nothing newly-disclosed
   since W116**).
5. **Structured W118 fire condition** (`W118FireConditionV1`) — packaged-release
   trigger + **construction-provenance trigger** (NEW) + cutoff trigger;
   `fires_now = False`.

### Construction attempt (W117 snapshot)

| proposal | A1..A5 | B1 | B2 | construction-admissible | realizable |
|---|---|---|---|---|---|
| raw-contest hand-assembly (post-2025-04) | ✗ (A1) | ✗ | ✗ | **No** | No |
| LCB-published collection pipeline + manifest (template) | ✓ | ✓ | ✓ | **Yes (in principle)** | **No (artifact absent)** |

The raw-contest assembly is **triply refused** (not the official `livecodebench/*`
source → A1; not LCB-published provenance → B1; operator-curated → B2). The
LCB-published-pipeline template **would** be construction-admissible (A1..A5 ∧ B1 ∧ B2,
post-frontier) but its artifact **does not exist** on the live surface ⇒ **not
realizable**. `constructed = False`; `n_construction_admissible_new = 0`;
`n_construction_admissible_in_principle = 1`. The exact missing artifact is named (the
load-bearing W118 trigger).

### Per-model go/no-go matrix (W117 snapshot — unchanged, CID 258b6ed7)

| model | rank | cutoff | C1 | C2 | C3 | C4 | certifiable |
|---|---|---|---|---|---|---|---|
| `meta/llama-4-maverick-17b-128e-instruct` | 1.1 | 2024-08-31 [KNOWN] | ✓ | ✓ (63) | ✓ | ✗ (W113 settled) | **No** |
| `qwen/qwen3-coder-480b-a35b-instruct` | 2.1 | 2025-07-01 [UNKNOWN] | ✗ | ✗ (0) | ✓ | ✓ | **No** |
| `deepseek-ai/deepseek-v4-pro` | 2.2 | 2025-01-01 [UNKNOWN] | ✗ | ✓ (49)¹ | ✓ | ✓ | **No** |
| `mistralai/mistral-small-4-119b-2603` | 2.3 | 2026-03-01 [UNKNOWN] | ✗ | ✗ (0) | ✓ | ✓ | **No** |

¹ DeepSeek's C2 passes only under its UNKNOWN *registry estimate*; C1 (KNOWN-only)
blocks it regardless — the W113/W114 discipline.

**Verdict: `NO_CERTIFIABLE_STRONGER_MODEL`.** `disclosure_consistency_ok = True`.
`n_surfaces_with_post_frontier_artifact = 0`. `W118 fires_now = False`
(packaged / construction / cutoff triggers all met = False).

## Pilot decision (RUNBOOK_W117 § 6 / § 7)

A pilot is earned ONLY if BOTH (1) Lane α yields a construction-admissible NEW
instrument AND (2) Lane β yields a non-redundant stronger model with a primary-KNOWN
cutoff that certifies. Lane α = 0 construction-admissible NEW instruments (the only
post-v6 path is refused by B1 + B2); Lane β = no usable NEW KNOWN-cutoff target ⇒ **NO
pilot earned; $0 NIM** (the pre-committed no-go branch; the no-go is discipline, not
omission). No Maverick rerun (no GENUINELY NEW instrument exists; the same-`release_v6`
cell is settled — W106 redundant-run discipline).

## What this establishes / does NOT

* **DOES** prove the upstream supply blocker is a CONSTRUCTION-provenance blocker, not
  merely a packaging blocker: across EIGHT provenance surfaces no post-v6 LCB-published
  artifact exists, and the only post-v6 path (raw-contest hand-assembly) is refused by
  the pre-committed B1 + B2 criteria
  (`W117-L-NO-CONSTRUCTION-ADMISSIBLE-POST-V6-INSTRUMENT-EIGHT-SURFACE-CAP` +
  `W117-T-LCB-CONSTRUCTION-PROVENANCE-IS-PACKAGED-RELEASE`).
* **DOES** re-confirm, DEEPER from primary sources, that DeepSeek V4's official
  model-card PDF discloses no cutoff (the only figure is a non-primary aggregator that
  is itself C2-exposed) — sharpening the model-disclosure blocker
  (`W117-T-DEEPSEEK-V4-PRIMARY-PDF-RECONFIRMED-NO-CUTOFF`).
* **DOES** ship a durable upstream-DERIVED construction/admission pipeline (construction
  rule with B1 + B2 + a candidate constructor + a provenance validator + an
  eight-surface snapshot + the disclosure matrix + the W118 condition) that makes W118
  push-button (`W117-T-UPSTREAM-DERIVED-CONSTRUCTION-PIPELINE-SHIPS`).
* **Does NOT** spend NIM (no construction-admissible instrument; no certifiable model;
  Maverick redundant), add a retirement, weaken W89/W105, move the contamination
  confound (W117 tests construction supply, not the confound), or re-probe reachability
  (not the binding gate; W112 facts carried).

## W118 (the loaded next move)

W118 fires the moment the construction pipeline's detector flags an admissible change —
(a) a packaged `release_v7`+ admitted to the loader, OR (b) an LCB-PUBLISHED post-v6
CONSTRUCTION provenance (a dataset revision/commit/PR adding post-2025-04 functional
problems, OR a published collection pipeline + problem-id manifest) enabling a B1 ∧ B2
reproducible ≥30 functional post-cutoff slice, OR (c) a reachable stronger-than-Maverick
model disclosing a primary-KNOWN cutoff month ≤ 2025-01. Re-run
`run_upstream_construction_v1` against the updated snapshot → if any model certifies on
a construction-admissible instrument, run the pre-committed cheapest-honest pilot. Until
one holds, the bounded ceiling STANDS and resistant-code NIM is BLOCKED. `COO-9` stays
lead.

Anchors: `docs/RUNBOOK_W117.md`, `docs/RESULTS_W117_MILESTONE_SUMMARY_V1.md`,
`docs/CONTAMINATION_CONTROL_FRAMING_W117_V1.md`,
`docs/FRONTIER_RELEVANCE_AUDIT_W117_V1.md`,
`results/w117/upstream_construction/upstream_construction_verdict.json`.
