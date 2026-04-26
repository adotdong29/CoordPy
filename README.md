# Context Zero — Wevra

**Wevra is a context-capsule runtime.** Every piece of context that
crosses a role boundary, a layer boundary, or a run boundary is a
typed, content-addressed, lifecycle-bounded, budget-bounded,
provenance-stamped **capsule** — never a raw prompt string. One
`RunSpec` in, one reproducible report out, and that report is the
root of a sealed capsule graph you can audit, replay, and trust.

**As of SDK v3.7 (April 2026), capsules are load-bearing
*inside one Wevra run*, *between agents in a team*, *across the
model-class gradient*, and now — most sharply — across the
*model regime × admission strategy* grid on a real-LLM-driven
multi-agent coordination benchmark.** SDK v3.7's contribution is
the **Phase-53 stronger-model multi-agent benchmark**
([`vision_mvp/experiments/phase53_scale_vs_structure.py`](vision_mvp/experiments/phase53_scale_vs_structure.py)),
which replaces Phase-52's deterministic producer-role extractor
with a real-LLM extractor (qwen2.5:14b-32k, qwen3.5:35b on Mac 1
Ollama) and decomposes accuracy across (model regime × admission
strategy). The headline result (n=5 saturated, K_auditor=4):
every fixed admission strategy — substrate, capsule_fifo,
capsule_priority, capsule_coverage — achieves
``accuracy_full = 0.800`` in every model regime; only
``capsule_learned`` varies, scoring 0.400 on synthetic and 14B
and recovering to 0.800 on 35B. **``structure_gain`` is
*non-positive* at every regime tested** (-0.4 / -0.4 / 0.0);
**``scale_gain[capsule_learned] = +0.4`` while
``scale_gain[fixed] = 0.0``**; cross-(14B, 35B) candidate-kind
TVD = 0.167. The capsule-team lifecycle audit (T-1..T-7) holds
60/60 across (regime × capsule strategy × scenario) — Theorem
**W6-1**. The honest reading: the SDK v3.5 conjecture **W4-C1**
(learned admission policy beats fixed) is **conditionally
falsified** out-of-distribution on the real-LLM regime — it
holds on its anchor distribution (Phase-52 synthetic+noise
default config) but loses to FIFO on the cleaner real-LLM
candidate stream. Scale closes a *structure deficit* (created by
OOD over-rejection of the synthetic-trained scorer), **not** a
*structure surplus*. **The capsule layer's load-bearing
contribution at this benchmark is the lifecycle audit, not
admission policy gains.** Substrate FIFO is a stronger baseline
than the W4 family suggested when the LLM is the producer (the
LLM does its own implicit filtering). **Mac 2 is still offline**
(192.168.12.248 ARP "incomplete"); **no two-Mac sharded inference
ran in this milestone** — the
[`MLXDistributedBackend`](vision_mvp/wevra/llm_backend.py)
integration boundary is byte-for-byte unchanged from SDK v3.6
and waits for the runbook
([`docs/MLX_DISTRIBUTED_RUNBOOK.md`](docs/MLX_DISTRIBUTED_RUNBOOK.md))
when Mac 2 returns. The strongest model class actually exercised
is single-Mac qwen3.5:35b (36 B-MoE Q4) on Mac 1 Ollama. See
[`docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md`](docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md)
for the full SDK v3.7 milestone note,
[`docs/RESULTS_WEVRA_DISTRIBUTED.md`](docs/RESULTS_WEVRA_DISTRIBUTED.md)
for SDK v3.6 (cross-LLM parser-boundary + integration boundary),
and
[`docs/RESULTS_WEVRA_TEAM_COORD.md`](docs/RESULTS_WEVRA_TEAM_COORD.md)
for SDK v3.5 (multi-agent capsule-coordination research slice).
The Wevra single-run product runtime contract is byte-for-byte
unchanged.

Up through **SDK v3.5**, the multi-agent capsule coordination
**research slice**
([`vision_mvp/wevra/team_coord.py`](vision_mvp/wevra/team_coord.py)
+ [`vision_mvp/wevra/team_policy.py`](vision_mvp/wevra/team_policy.py))
added three closed-vocabulary capsule kinds (`TEAM_HANDOFF`,
`ROLE_VIEW`, `TEAM_DECISION`) and a `TeamCoordinator` that drove
one coordination round end-to-end, with a mechanically-checked
`audit_team_lifecycle` over invariants T-1..T-7 (Theorem **W4-1**).
Theorems W4-2 (proved-conditional: coverage-implies-correctness)
and W4-3 (proved-negative: per-role budget below the role's
causal-share floor cannot be rescued by *any* admission policy)
anchor the team-level mechanism formally. The W4-C1 conjecture
(learned admission policy beats the strongest fixed baseline) is
empirical-positive on its synthetic+noise anchor distribution
but is now annotated as **conditional** in
[`docs/THEOREM_REGISTRY.md`](docs/THEOREM_REGISTRY.md): SDK v3.7
empirically falsifies it out-of-distribution on the real-LLM
regime. Honest-reading rules in
[`docs/HOW_NOT_TO_OVERSTATE.md`](docs/HOW_NOT_TO_OVERSTATE.md).

Up through **SDK v3.4 (April 2026)**, capsules drive execution all the
way through the LLM byte boundary. Run-boundary stages
(profile / readiness / sweep_spec / sweep_cell / provenance /
artifact / run_report) seal capsules in flight as before
(W3-32..W3-35). Inside every LLM-backed cell, the end-to-end
inner-loop chain is **five typed sealed capsules**:

  `PROMPT → LLM_RESPONSE → PARSE_OUTCOME → PATCH_PROPOSAL → TEST_VERDICT`

The PROMPT capsule (parent: SWEEP_SPEC, **W3-42**) records the
prompt's SHA-256 + byte length + bounded text snippet; the
LLM_RESPONSE capsule (parent: PROMPT, **W3-43**) records the
response's SHA-256 + length + snippet + elapsed milliseconds;
the PARSE_OUTCOME capsule on the LLM-backed path parents on
both SWEEP_SPEC and the upstream LLM_RESPONSE (**W3-44**), with
coordinate consistency mechanically verified by audit rule
**L-11** (**W3-45** — soundness of the eleven-rule audit). A new
**in-process synthetic-LLM mode**
(`SweepSpec(mode="synthetic", synthetic_model_tag=<tag>)`) lets
the full chain run end-to-end in CI without an Ollama endpoint.
The cross-model parser-boundary research (**W3-C6**, empirical)
reports PARSE_OUTCOME failure-kind Total Variation Distance up
to **1.000** across the calibrated synthetic distribution
library and parser-mode (strict→robust) shift up to **1.000**
on `synthetic.unclosed`.

The SDK v3.3 layer is unchanged: PARSE_OUTCOME capsule
(W3-39), eight-rule lifecycle audit (W3-40, now extended to
eleven rules in W3-45), and `RunSpec(deterministic=True)` opt-in
that collapses the full capsule DAG byte-for-byte across runs
(W3-41). The meta-artefact boundary
(`product_report.json` / `capsule_view.json` / `product_summary.txt`)
remains a sharp circularity theorem (W3-36) with a constructive
detached `META_MANIFEST` witness in a secondary ledger.
`wevra-capsule verify` recomputes the chain from on-disk header
bytes (W3-37) and re-hashes every artefact at audit time (W3-38).
See
[`docs/RESULTS_WEVRA_INNER_LOOP.md`](docs/RESULTS_WEVRA_INNER_LOOP.md)
for the SDK v3.4 milestone note,
[`docs/RESULTS_WEVRA_DEEP_INTRA_CELL.md`](docs/RESULTS_WEVRA_DEEP_INTRA_CELL.md)
for SDK v3.3,
[`docs/RESULTS_WEVRA_INTRA_CELL.md`](docs/RESULTS_WEVRA_INTRA_CELL.md)
for SDK v3.2, and
[`docs/RESULTS_WEVRA_CAPSULE_NATIVE.md`](docs/RESULTS_WEVRA_CAPSULE_NATIVE.md)
for the SDK v3.1 run-boundary slice. For the unified paper-grade
write-up see
[`papers/wevra_capsule_native_runtime.md`](papers/wevra_capsule_native_runtime.md).

Wevra is the first shipped product from the **Context Zero** research
programme on per-agent minimum-sufficient context.

> ### The one-line mental model
>
> Traditional eval harnesses and agent frameworks pass *strings*
> between roles — prompts, JSON dicts, log lines. Wevra doesn't.
> Every unit of coordination that crosses a boundary in Wevra is a
> **`ContextCapsule`**: it has a SHA-256 content-address (`cid`), a
> typed `claim_kind` from a closed vocabulary, an explicit
> `lifecycle` (PROPOSED → ADMITTED → SEALED), an explicit
> `CapsuleBudget` (tokens / bytes / rounds / witnesses / parents),
> a parent-CID DAG, and a hash-chained audit history. "Context" in
> Wevra is not text — it is an object with identity, type,
> lifecycle, budget, and proof. SDK v3.1 lifted that contract from
> *audit description* to *runtime gate* on the run-boundary spine;
> SDK v3.2 extended the gate past the cell boundary into the inner
> parse→apply→test loop and formalised the meta-artefact circularity
> as a sharp limitation theorem with a constructive detached-witness
> boundary. SDK v3.3 extended the gate one further structural
> layer with a sub-intra-cell PARSE_OUTCOME capsule (parent:
> SWEEP_SPEC, W3-39), a runtime-checkable lifecycle audit (W3-40),
> and deterministic-mode replay collapsing the full DAG
> byte-for-byte across runs (W3-41). **SDK v3.4 extends the gate
> one further structural layer to the LLM byte boundary itself**:
> PROMPT and LLM_RESPONSE capsules (W3-42 / W3-43), the
> PARSE_OUTCOME → LLM_RESPONSE chain coordinate consistency
> theorem (W3-44, mechanically checked by audit rule L-11), the
> extended audit soundness over eleven invariants (W3-45), and
> a synthetic-LLM mode that lets the full chain run in CI
> end-to-end. See
> [`docs/RESULTS_WEVRA_CAPSULE.md`](docs/RESULTS_WEVRA_CAPSULE.md)
> for the contract (C1..C6),
> [`docs/RESULTS_WEVRA_CAPSULE_NATIVE.md`](docs/RESULTS_WEVRA_CAPSULE_NATIVE.md)
> for the v3.1 run-boundary slice (W3-32..W3-35),
> [`docs/RESULTS_WEVRA_INTRA_CELL.md`](docs/RESULTS_WEVRA_INTRA_CELL.md)
> for the v3.2 intra-cell + detached witness milestone
> (W3-32-extended / W3-36 / W3-37 / W3-38),
> [`docs/RESULTS_WEVRA_DEEP_INTRA_CELL.md`](docs/RESULTS_WEVRA_DEEP_INTRA_CELL.md)
> for the v3.3 deeper-slice / audit / determinism milestone
> (W3-39 / W3-40 / W3-41), and
> [`docs/RESULTS_WEVRA_INNER_LOOP.md`](docs/RESULTS_WEVRA_INNER_LOOP.md)
> for the v3.4 PROMPT / LLM_RESPONSE / synthetic mode /
> parser-boundary research milestone
> (W3-42 / W3-43 / W3-44 / W3-45 / W3-C6). Canonical theorem registry:
> [`docs/THEOREM_REGISTRY.md`](docs/THEOREM_REGISTRY.md). Canonical
> research-status:
> [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md). How not to
> overstate any claim:
> [`docs/HOW_NOT_TO_OVERSTATE.md`](docs/HOW_NOT_TO_OVERSTATE.md).

[![status](https://img.shields.io/badge/status-beta-blue.svg)](#)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)

> **New to this repo?** Read [`docs/START_HERE.md`](docs/START_HERE.md) first —
> it is the canonical one-pass orientation: what Context Zero is, what Wevra
> is, what is core substrate, what is product surface, what is boundary, and
> what is research-grade.
>
> **Naming.** `Context Zero` is the research programme — theorems, phase
> shards, and the 72-framework theoretical survey (see `PROOFS.md`,
> `EXTENDED_MATH_[1-7].md`, `vision_mvp/RESULTS_PHASE*.md`). `Wevra` is the
> shipped SDK/runtime under `vision_mvp/wevra/`, imported as
> `from vision_mvp import wevra`. The older `vision_mvp/product/` modules
> remain importable for backwards compatibility but are **not** the public
> contract; see the [Stability matrix](#stability-matrix).
>
> Wevra is **not** the whole research programme and **not** a universal
> agent platform. It is a context-capsule runtime with a narrow, stable
> product surface: profile-driven SWE-bench-Lite-shape evaluation runs
> with a stable report schema, a CI gate, a provenance manifest, and a
> capsule graph on every run.
>
> **CASR** (Causal-Abstraction Scale-Renormalized Routing) is the original
> substrate proposal from the programme; it lives in `vision_mvp.core.*` as
> research-grade code and informs Wevra's bounded-context guarantees, but
> it is not itself the product identity.

> Most multi-agent AI frameworks (AutoGen, CrewAI, LangGraph, …) cap out at
> around 10-100 agents because every agent has to read every other agent's
> output each round — context grows like O(N²). **Context Zero** ships a
> coordination layer whose per-agent context grows like **O(log N)**, so the
> same team design scales to 10 000, 100 000, or more agents without
> collapsing under its own context.

> **What this project is — and what it is not.** Context Zero is a *context
> substrate for teams of agents collaborating on a task*. It is NOT a repo
> knowledge-graph tool (Graphify-style or otherwise); those tools represent
> a corpus for a single assistant to traverse, and their object of study is
> the graph. Our object of study is the per-role, per-round information flow
> across a whole team — *who should know what, when, why, and with what loss
> profile*. Graph/index techniques show up *as one layer* in our stack
> (retrieval, call graph, interprocedural analysis); they are not the
> project's identity. See
> [``docs/context_zero_master_plan.md`` § 1.5](docs/context_zero_master_plan.md)
> for the durable version of this distinction, and
> [``vision_mvp/RESULTS_PHASE31.md``](vision_mvp/RESULTS_PHASE31.md)
> Theorem P31-5 for the formal statement (a single-agent corpus compressor
> cannot match a team's bounded-context guarantee by any universal
> compression; the team's guarantee is a property of *role-conditioned
> information flow*).

Empirically validated from **N=10** up to **N=100 000** agents with peak
per-agent context equal to ⌈log₂ N⌉ *exactly* — and **1 000 real local-LLM
agents coordinating on one laptop in 54 seconds** with 100 % accuracy on a
factual question, using **3 750 × fewer tokens** than naive broadcast would
require.

---

## Install

```bash
git clone <this-repo>
cd context-zero
pip install -e .
```

Only dependency is NumPy. The optional LLM-agent demo talks HTTP to a local
Ollama instance (no Python binding required).

Installing the package also registers three console scripts:

```bash
wevra --profile local_smoke --out-dir /tmp/cz-smoke
wevra-import --jsonl /path/to/swe_bench_lite.jsonl --out /tmp/audit.json
wevra-ci --report /tmp/cz-smoke/product_report.json --min-pass-at-1 1.0
```

Optional extras: `wevra[scientific]`, `wevra[dl]`, `wevra[heavy]`,
`wevra[docker]` (Docker-first sandbox — Slice 2), `wevra[dev]`.

---

## Wevra SDK — quick start

Wevra is a context-capsule runtime. One `RunSpec` in, one
reproducible, provenance-stamped **capsule graph** out.

```python
from vision_mvp.wevra import RunSpec, run

report = run(RunSpec(profile="local_smoke", out_dir="/tmp/cz-smoke"))
assert report["readiness"]["ready"]
assert report["provenance"]["schema"] == "wevra.provenance.v1"
# SDK v3 — every run ships a sealed capsule graph.
cv = report["capsules"]
assert cv["schema"] == "wevra.capsule_view.v1"
assert cv["chain_ok"]
print(f"RUN_REPORT CID = {cv['root_cid']}")
print(report["summary_text"])
```

### Every run is a sealed capsule DAG, built in flight

```python
from vision_mvp.wevra import (
    RunSpec, run, CONSTRUCTION_IN_FLIGHT, CapsuleKind,
)

report = run(RunSpec(profile="local_smoke", out_dir="/tmp/cz-smoke"))
# Capsule view tagged with its construction mode:
assert report["capsules"]["construction"] == CONSTRUCTION_IN_FLIGHT
# Every proposed capsule sealed; no failures.
assert report["capsules"]["in_flight_stats"]["n_failed"] == 0

# Types you'll see in a smoke run:
#   PROFILE           (root of the DAG)
#   READINESS_CHECK   (parent: PROFILE)
#   SWEEP_SPEC        (parent: PROFILE)
#   SWEEP_CELL × N    (parent: SWEEP_SPEC)
#   PROVENANCE        (parent: PROFILE)
#   ARTIFACT × K      (content-addressed at write time, parent: source capsule)
#   RUN_REPORT        (parents: ALL of the above)
```

For third parties who have a `product_report` dict from outside the
runtime, the post-hoc fold is still available:

```python
from vision_mvp.wevra import build_report_ledger
ledger, run_cid = build_report_ledger(report)
assert ledger.verify_chain()
```

The two paths produce CID-equivalent ledgers on non-ARTIFACT kinds
(Theorem W3-34); ARTIFACT capsules from a capsule-native run carry
real SHA-256 hashes, while the post-hoc fold's ARTIFACT capsules
carry `None`.

The RUN_REPORT capsule's CID is the durable identifier for the run —
send someone that CID plus `product_report.json` and they can
reproduce every upstream capsule, verify the hash chain end-to-end,
and know the bytes haven't drifted.

### Every run emits seven artifacts

  * `product_report.json` — schema `phase45.product_report.v2`,
    includes the `capsules` block (`wevra.capsule_view.v1`)
  * `capsule_view.json` — the sealed capsule graph, on disk
  * `product_summary.txt` — human-readable summary with a
    `capsules` line (kind histogram + chain_ok + root CID)
  * `readiness_verdict.json` — per-row readiness verdicts
  * `provenance.json` — git SHA, package version, Python, platform,
    profile, model, endpoint, sandbox, input JSONL + SHA-256, argv,
    timestamp, artifact list (schema `wevra.provenance.v1`)
  * `sweep_result.json` — the executed-sweep block when the
    profile's sweep ran in-process (mock or real-LLM acknowledged)
  * `meta_manifest.json` — **new in SDK v3.2** — the detached
    META_MANIFEST witness for the meta-artefacts above, in a
    secondary ledger; carries on-disk SHA-256 of
    `product_report.json` / `capsule_view.json` /
    `product_summary.txt` plus the primary `root_cid` and
    `chain_head` (Theorem W3-36 — meta-artefact circularity is
    sharp; the manifest is the strongest authentication achievable
    one trust hop beyond the primary view)

### CLIs

```bash
wevra --profile local_smoke --out-dir /tmp/cz-smoke
wevra-ci --report /tmp/cz-smoke/product_report.json --min-pass-at-1 1.0
# Capsule-graph inspection (new in SDK v3, strengthened in v3.2):
wevra-capsule view   --report /tmp/cz-smoke/product_report.json
wevra-capsule verify --report /tmp/cz-smoke/product_report.json
wevra-capsule cid    --report /tmp/cz-smoke/product_report.json
```

`wevra-capsule verify` (v3.2) runs four independent on-disk
checks and prints each verdict line plus a final `verdict = OK / BAD`:

```
chain_ok_embedded         = True    # writer's self-report
chain_recompute_embedded  = True    # we recompute the chain
chain_recompute_on_disk   = True    # ...from on-disk view bytes
on_disk_view_agrees       = True    # cross-check
artifacts_on_disk         = OK  (3/3 matched, 0 drifts, 0 missing)
meta_manifest_on_disk     = OK  (3/3 matched, 0 drifts, 0 missing)
verdict                   = OK
```

A drift in any check fails the verify (exit code 3) and prints the
specific path / sealed SHA / on-disk SHA tuple — the audit knows
exactly which file lied.

The SDK public surface is contract-tested in
`vision_mvp/tests/test_wevra_public_api.py` and the Capsule Contract
(invariants C1..C6) is tested in
`vision_mvp/tests/test_wevra_capsules.py` — any rename or removal is a
breaking change and requires bumping `wevra.SDK_VERSION` (currently
`wevra.sdk.v3.4`).

### Who Wevra is for

  * Research engineers running **profile-driven evaluations** on
    SWE-bench-Lite-shape banks who need a reproducible, provenance-
    stamped, capsule-graph artifact trail instead of ad-hoc scripts.
  * Teams wiring **CI gates** over evaluation quality (`wevra-ci`
    consumes the report and emits a pass/fail verdict with explicit
    blocker strings; `wevra-capsule verify` re-hashes the capsule
    chain so a third party can confirm the report bytes haven't
    drifted).
  * Operators who want to swap **profiles** (`local_smoke`,
    `bundled_57`, `aspen_mac1_coder`, `aspen_mac2_frontier`,
    `public_jsonl`, …) without editing core code.
  * Downstream framework authors who want to **lift their own
    substrate into capsules** — the `capsule_from_handle`,
    `capsule_from_handoff`, `capsule_from_sweep_cell` adapters
    let you expose your own typed objects under the same
    Contract, and `CapsuleLedger.admit_and_seal()` makes your
    artefacts composable with Wevra's.

Wevra is **not** (yet) for: arbitrary multi-agent orchestration,
agent-platform building, or non-SWE evaluation shapes. Those are
Context Zero research-programme territory, not Wevra product
territory.

### How to extend Wevra (current state, honest)

  * **New profiles**: add a declarative entry to
    `vision_mvp/product/profiles.py`. The `profiles` module is part
    of the SDK contract; a profile is a frozen dict with stable keys.
  * **New sandbox backends, new task banks, new reporting sinks**:
    the extension surface for these is *not yet stable*. The planned
    entry-point / registry-based plugin system is Slice 2 (see
    `docs/context_zero_master_plan.md` § Wevra SDK follow-ups).
    Today, extending these requires editing core modules; this is
    a known limitation marked **boundary** in the stability matrix.

---

## Stability matrix

| Layer | Scope | Stability | Import path |
|---|---|---|---|
| **Wevra SDK** — `RunSpec`, `run`, `SweepSpec`, `run_sweep`, `HeavyRunNotAcknowledged`, `WevraConfig`, `build_manifest`, `profiles`, `report`, `ci_gate`, `import_data`, `extensions`, schema constants. SDK v3.3 adds `RunSpec.deterministic` opt-in. **SDK v3.4 adds `SweepSpec(mode="synthetic", synthetic_model_tag=...)`** — in-process synthetic-LLM mode for CI-runnable end-to-end chain exercise. | The public product contract | **Stable v3.4** (contract-tested) | `vision_mvp.wevra` |
| **Context Capsule primitives** — `ContextCapsule`, `CapsuleKind`, `CapsuleLifecycle`, `CapsuleBudget`, `CapsuleLedger`, `CapsuleView`, `render_view`, `build_report_ledger`, `capsule_from_*` adapters | The **load-bearing SDK abstraction**: every cross-boundary artefact is a capsule | **Stable v1** (contract-tested: invariants C1..C6) | `vision_mvp.wevra.capsule` (re-exported from `vision_mvp.wevra`) |
| **Capsule-native runtime** — `CapsuleNativeRunContext`, `seal_and_write_artifact`, `ContentAddressMismatch`, `CONSTRUCTION_IN_FLIGHT`/`CONSTRUCTION_POST_HOC`, `RunSpec.capsule_native` + intra-cell `seal_patch_proposal` / `seal_test_verdict` + sub-intra-cell `seal_parse_outcome` (SDK v3.3) + **sub-sub-intra-cell `seal_prompt` / `seal_llm_response` (SDK v3.4)** + detached `seal_meta_manifest` + on-disk `verify_chain_from_view_dict` / `verify_artifacts_on_disk` / `verify_meta_manifest_on_disk` | Capsules drive runtime stage transitions at the run boundary AND inside the inner sweep loop AND on the parser axis AND at the LLM byte boundary; substantive artifacts are content-addressed at write time and re-verifiable at audit time; meta-artefacts are authenticated by a detached META_MANIFEST | **Stable v3.4** (contract-tested: theorems W3-32 / W3-33 / W3-34 / W3-35 / W3-32-extended / W3-36 / W3-37 / W3-38 / W3-39 / W3-40 / W3-41 / W3-42 / W3-43 / W3-44 / W3-45) | `vision_mvp.wevra.capsule_runtime` (re-exported from `vision_mvp.wevra`) |
| **Lifecycle audit** — `CapsuleLifecycleAudit`, `LifecycleAuditReport`, `audit_capsule_lifecycle`, `audit_capsule_lifecycle_from_view` (SDK v3.3, **extended in SDK v3.4 to L-9..L-11**) | Mechanically verifies **eleven** lifecycle invariants L-1..L-11 over a finished run (eight from v3.3 + L-9 / L-10 / L-11 covering the PROMPT / LLM_RESPONSE / coordinate-consistency chain). Returns OK / BAD / EMPTY plus typed counterexamples. | **Stable v1.1** (contract-tested: theorems W3-40 / W3-45) | `vision_mvp.wevra.lifecycle_audit` (re-exported) |
| **Wevra console scripts** — `wevra`, `wevra-import`, `wevra-ci`, `wevra-capsule` | CLI surface | **Stable v3** (Slice 3: `wevra-capsule view / verify / cid`) | `[project.scripts]` |
| **Provenance manifest** — `wevra.provenance.v1` | Reproducibility artifact | **Stable v1** | `vision_mvp.wevra.provenance` |
| **Capsule view artifact** — `wevra.capsule_view.v1` | Sealed capsule graph on disk | **Stable v1** | `capsule_view.json` next to every report |
| **Extension Protocols** — `SandboxBackend`, `TaskBankLoader`, `ReportSink` | Plugin surface | **Stable v1** (runtime-checkable Protocols, `entry_points` discovery) | `vision_mvp.wevra.extensions` |
| **Unified runtime** — `SweepSpec`, `run_sweep`, `wevra.sweep.v2` | One execution path for mock + real-executed + real-staged | **Stable v1** | `vision_mvp.wevra.runtime` |
| **Report / CI-gate schemas** — `phase45.product_report.v2` (v1 accepted), `phase46.ci_verdict.v1`, `phase46.import_audit.v1` | On-disk contract | **Stable** | — |
| **Core substrate** — CASR router, hierarchical router, ledger, exact_ops, role_handoff | Research substrate used *by* Wevra | **Settled** (proofs + tests) but **research API** | `vision_mvp.core.*` |
| **Legacy product path** — `vision_mvp.product.*` | Pre-Slice-1 import path | **Deprecated-compat** (still works; re-exported by `wevra`) | `vision_mvp.product` |
| **Docker sandbox** | Untrusted-input isolation | **Available** (backend registered as `wevra.extensions.get_sandbox("docker")`); **not yet the default** | `vision_mvp.wevra.extensions` |
| **Docker-first-by-default** for public JSONLs | Slice 3 target | **Boundary / next-slice** (default-flip) | n/a yet |
| **First real out-of-tree plugin** | Slice 3 community target | **Boundary / next-slice** | n/a yet |
| **Research shards** — Phases 1–44 RESULTS_*.md, EXTENDED_MATH_*.md, per-phase experiment scripts, 72-framework survey | The Context Zero research programme | **Research-grade** (empirical or proved per shard; no product-API guarantee) | `vision_mvp.experiments.*`, `vision_mvp.tasks.*`, docs |

See `docs/context_zero_master_plan.md` for the living version of this
matrix and the concrete next-slice follow-ups.

---

## Quick start

```python
from vision_mvp import CASRRouter
import numpy as np

# 1 000 agents, each carrying a 64-dim state vector.
router = CASRRouter(n_agents=1000, state_dim=64, task_rank=10)

# Feed observations each round — get back consensus estimates.
for _ in range(100):
    obs = np.random.randn(1000, 64)     # (N, d) noisy observations
    estimates = router.step(obs)         # (N, d) consensus estimates

print(router.stats)
# {'peak_context_per_agent': 10, 'total_tokens': 1_010_000,
#  'total_messages': 100_100, 'mean_context_per_agent': 10.0,
#  'manifold_dim': 10, 'workspace_size': 10, 'rounds_executed': 100}
```

Peak context per agent = 10 = ⌈log₂ 1000⌉. Workspace size also = 10. Both
independent of the state dimension d=64.

---

## CLI

```bash
python -m vision_mvp demo --n 500 --d 32 --rounds 20
python -m vision_mvp scale --n-values 10 50 200 1000 5000
python -m vision_mvp phase 3
python -m vision_mvp test
```

---

## What's in the repo

```
context-zero/
├── README.md                   # you are here
├── LICENSE                     # MIT
├── pyproject.toml              # installable as context-zero
├── PROOFS.md                   # 12 formal theorems underpinning the scaling claims
├── FRAMEWORK.md                # original problem formulation
├── ARCHITECTURE.md             # reference implementation design
├── EVALUATION.md               # metrics + benchmarks + 'solved' definition
├── MVP.md                      # Phase-1 spec
├── ROADMAP.md                  # 4-phase research plan
├── OPEN_QUESTIONS.md           # 7 open research questions
├── VISION_MILLIONS.md          # the 10-idea vision for million-agent teams
├── EXTENDED_MATH.md            # +6 volumes of mathematical grounding
├── EXTENDED_MATH_2..7.md       # 72 independent frameworks converging on O(log N)
└── vision_mvp/                 # the working implementation
    ├── api.py                  # public CASRRouter class
    ├── __main__.py             # CLI
    ├── core/                   # manifold, stigmergy, workspace, predictor, market, scales…
    ├── tasks/                  # consensus, drifting consensus, LLM QA
    ├── protocols/              # naive, gossip, manifold-only, full, adaptive,
    │                           # hierarchical, holographic, swarm, LLM
    ├── experiments/            # phase 1–5 experiment harnesses
    ├── tests/                  # 94 tests covering every module + the theorems
    ├── RESULTS.md … RESULTS_PHASE5.md
    ├── FINAL_RESULTS.md        # consolidated scaling table
    └── README.md               # implementation-level README
```

---

## The empirical story in one table

Measured on the drifting-consensus task (d=64, intrinsic_rank=⌈log₂ N⌉,
noise σ=1.0, drift σ=0.05, 3 seeds averaged).

| N | log₂ N | Peak ctx/agent | Writes/round | Workspace | Steady err |
|---:|---:|---:|---:|---:|---:|
| 50 | 5.6 | **6** | 6 | 6 | 0.22 |
| 200 | 7.6 | **8** | 8 | 8 | 0.25 |
| 1 000 | 10.0 | **10** | 10 | 10 | 0.08 |
| 5 000 | 12.3 | **13** | 13 | 13 | 0.11 |
| 10 000 | 13.3 | **14** | 14 | 14 | 0.09 |
| 50 000 | 15.6 | **16** | 16 | 16 | 0.12 |
| **100 000** | **16.6** | **17** | **17** | **17** | **0.15** |

Three independently-measured metrics — peak context, writes per round,
workspace capacity — **all equal ⌈log₂ N⌉** at every scale tested.

Compare with naive broadcast, which at N=100 000 would use **6 500 000 tokens
of peak per-agent context**. That's a **≈382 000× reduction**.

And for **real LLM agents** (qwen2.5:0.5b via local Ollama), wall time stays
essentially flat across two orders of magnitude of team size:

| N | Wall time | Accuracy | LLM tokens | Naive/vision ratio |
|---:|---:|---:|---:|---:|
| 100 | 46 s | 100 % | 5 669 | 43 × |
| 1 000 | 54 s | 100 % | 8 242 | **3 750 ×** |
| 2 000 | 24 s | 100 % | 5 934 | 15 745 × |
| **5 000** | **46 s** | **100 %** | **7 513** | **76 840 ×** |

At N = 5 000, naive broadcast would cost ~577 million tokens ≈ **333 days** of
continuous decoding on this laptop. The vision stack does the same job in
under a minute.

Reproduce with `python -m vision_mvp.experiments.phase6_llm_1000 --n 5000`.

### And on a real reasoning task

Phase 7: **100 AI agents review actual code and find the critical bug.**
Same CASR machinery, qwen2.5-coder:7b model, 25 specialist reviewer
personas. The team reviews a Python helper with an SQL-injection
vulnerability and produces a structured report:

> **CRITICAL ISSUE:** SQL injection vulnerability due to direct string
> concatenation of user input into the SQL query.

10/10 sampled agents flag SQL injection, all 100 do via NN heuristic,
the synthesis step outputs the report above. See
`vision_mvp/RESULTS_PHASE7.md` for the full trace and
`python -m vision_mvp.experiments.phase7_code_review --n 100 --task sql`
to reproduce.

A second run on a **different bug type** (race condition in a counter)
also succeeded: 50/50 agents correctly identified the thread-safety
issue, synthesis produced the correct structured report. Two tasks, two
distinct bug categories, both 100 % accuracy — the team preserves
*reasoning quality* across task types, not just the original demo.

### Phase 8 — truly distributed task: no single agent can solve it

The phases above are useful scaling demos but don't require
*collaboration* — each agent could independently name the SQL-injection
bug. Phase 8 is the first task where **no single agent has enough
information** to answer. Each of 16 agents sees ONE chunk (~47 words)
of a 757-word fictional incident review. The task: identify the top 3
systemic risks that emerge only from **cross-chunk patterns**.

| Mode | Risks found | What it proves |
|---|---:|---|
| Isolated (1 chunk) | **0 / 3** | Correctly refuses — task really is distributed |
| Oracle (full 757 words, 1 agent) | 2 / 3 | Upper bound with unbounded context |
| **Map-reduce team (16 × 1-chunk agents + synth)** | **2 / 3** | **Matches oracle; each member saw 1/16th of the doc** |

The map-reduce team, where **no single agent saw more than 47 words**,
produced the same quality report as a single agent with the full
document. Both identified vendor concentration (NordAxis across multiple
incidents) and documentation/runbook gaps. This is the qualitative shift
— **collaborative output exceeds any individual member's capability**.

See `vision_mvp/RESULTS_PHASE8.md` for the full trace. Reproduce with:

```bash
python -m vision_mvp.experiments.phase8_mapreduce --n 16
python -m vision_mvp.experiments.phase8_distributed --n 16  # full 3-way compare
```

### Phase 9 — multi-role pipeline + longer-than-context document

Phase 8 showed pooling across chunks. Phase 9 pushes two directions:

**9a — Multi-role pipeline (quant strategy).** 19 agents in 4 distinct
roles: researchers distill notes, market-analysts read individual asset
time-series, strategists combine signals, a PM produces the final
portfolio. Each role has its own prompt and reads different inputs.
Result: team produced a real book with written rationales (+1.05 %
gross on its 5 committed bets, 3/5 hit rate). Too conservative —
FLATted 7 of 12 assets because the PM prompt didn't enforce coverage.
Real end-to-end work across genuinely different roles, honestly modest
score.

**9b — Longer-than-context document.** Fictional 11 000-word / 14 500-
token incident review (~3.5× Ollama's default 4 k context). Single-
agent oracle **TIMED OUT after 300 s**: it simply cannot complete the
task at default settings. Map-reduce team (40 agents, each sees 1/40
of the document) continues running — the distributed approach works
because each chunk fits any single agent.

See `vision_mvp/RESULTS_PHASE9.md` for the full phase-9 writeup and
`MATH_AUDIT.md` for an honest accounting of which of the 72 frameworks
in the extended math docs are actually in the running code (6 USED,
13 STRUCTURAL, 3 BUILT-not-tested, 50 THEORY-only).

### Phase 10 — The Agent Network: 500 agents collaborating on one task

Phase 9 used a classical DAG pipeline — what CrewAI / LangGraph already do.
Phase 10 is the thing they can't: **hundreds of agents send targeted
messages to each other**, each picking up a piece of one interconnected
task, with per-agent context **bounded regardless of team size**.

The architecture wires in three mechanisms from the math that were
previously theory-only:

1. **Sparse MoE routing** (from Routing Transformer / Mixtral) — each
   agent has a learned key; messages route to top-k recipients by
   clustered cosine-similarity lookup, cost O(√N · d).
2. **Hyperbolic (Lorentz-model) address space** — tree-structured task
   decompositions embed without sibling-subtree crosstalk.
3. **Sheaf H¹ diagnostic** — per-edge discord localizes exactly which
   agents disagree, not whether the team as a whole does.

Plus a shared task board (claim/complete/deps) and per-agent inbox with
capacity limits.

### The scaling result (mock LLM, same 40-subtask interconnected build)

| | **N = 30** | **N = 200** | **N = 500** |
|---|---:|---:|---:|
| Subtasks completed | **40 / 40** | **40 / 40** | **40 / 40** |
| Rounds | 8 | 4 | **4** |
| **Max inbox per agent** | **73** | **42** | **48** |
| Integration score | 1.00 | 1.00 | **1.00** |
| Total inter-agent messages | 232 | 706 | 1 728 |
| Wall (mock) | 0.1 s | 0.2 s | 0.9 s |

**Max per-agent inbox stays bounded at 40–75 across a 17× range of team
size.** That's the thing AutoGen / CrewAI cannot do — their context grows
linearly with team size and breaks past ~100 agents.

Reproduce:
```bash
python -m vision_mvp.experiments.phase10_network --mock --n 500
```

Full phase-10 writeup: `vision_mvp/RESULTS_PHASE10.md`. Design:
`AGENT_NETWORK_DESIGN.md`.

### Phase 43 — public-style-scale audit, frontier semantic headroom, and the post-parser-recovery semantic taxonomy

Phase 42 closed the parser-compliance layer and shipped the
three-axis attribution surface. The residue on the
``qwen2.5-coder:14b`` cluster run (4/57 failures on every
strategy) was the first purely semantic residue — format-
compliant, byte-matching, structurally-valid patches that
still fail the hidden test. Phase 43 *characterises* that
residue and audits that the programme is ready for public
SWE-bench-Lite drop-in.

* **Part A — Semantic failure taxonomy**
  (``vision_mvp/tasks/swe_semantic_taxonomy.py``). A pure
  deterministic ``classify_semantic_outcome`` that takes
  ``(buggy_source, gold_patch, proposed_patch, error_kind,
  test_passed)`` and returns exactly one label from a closed
  nine-element set (``SEM_OK`` / ``SEM_PARSE_FAIL`` /
  ``SEM_WRONG_EDIT_SITE`` / ``SEM_RIGHT_SITE_WRONG_LOGIC`` /
  ``SEM_INCOMPLETE_MULTI_HUNK`` / ``SEM_TEST_OVERFIT`` /
  ``SEM_STRUCTURAL_SEMANTIC_INERT`` / ``SEM_SYNTAX_INVALID``
  / ``SEM_NO_MATCH_RESIDUAL``). ``SemanticCounter`` aggregates
  per-strategy + pooled histograms with a ``failure_mix``
  helper that normalises by non-``SEM_OK`` total so
  cross-model comparisons are composition-level.
* **Part B — Public-style loader self-test**
  (``phase43_frontier_headroom.verify_public_style_loader``).
  Round-trips every instance in a target JSONL through
  ``load_jsonl_bank → SWEBenchAdapter.from_swe_bench_dict
  → parse_unified_diff → apply_patch → run_patched_test``
  under the strict matcher. On the bundled 57-instance bank:
  **57 / 57 oracle saturation** (Theorem P41-2 reproduced at
  scale). The externalisation gap to public SWE-bench-Lite
  is now a ``--jsonl <path>`` drop-in.
* **Part C — Frontier semantic-headroom run.** The ASPEN
  cluster runs ``qwen3.5:35b`` (36B MoE) on mac1 at
  ``n_distractors = 6`` over the full 57-instance bank and
  on mac2 at ``n_distractors ∈ {0, 24}`` over a 20-instance
  subset for the bounded-context stress. The 35B needs
  ``--think off`` to free its 600-token output budget from
  internal thinking; the ``LLMClient.think`` Phase-43
  extension threads this through Ollama's ``/api/generate``
  body.
* **Part D — Analysis driver**
  (``phase43_frontier_headroom``). Ingests Phase-42-shape
  artifacts, re-derives per-cell semantic labels, and emits
  the cross-model summary JSON. Analysis-only, no LLM
  dependency.

| metric (Phase-43 canonical cell: parser=robust/nd=6/apply=strict) | pass@1 (N, R, S) | S−N gap | dominant residue label | residue mix |
|---|:-:|:-:|---|---|
| ``qwen3.5:35b`` (36B MoE, cluster mac1)      | **0.965 / 0.965 / 0.965 (55/57)** | **0.0 pp** | SEM_WRONG_EDIT_SITE* | 100 % wrong_site* (§ D.7 caveat: test_exception + test_assert merged) |
| ``qwen2.5-coder:14b`` (cluster mac1)        | 0.930 / 0.930 / 0.930 | **0.0 pp** | SEM_WRONG_EDIT_SITE | 50 % wrong_site / 25 % multi_hunk / 25 % no_match |
| ``qwen2.5-coder:7b`` (localhost)            | 0.842 / 0.842 / 0.842 | **0.0 pp** | SEM_WRONG_EDIT_SITE | 56 % wrong_site / 33 % no_match / 11 % syntax |
| ``gemma2:9b`` (28 subset)                    | 0.857 / 0.857 / 0.857 | **0.0 pp** | SEM_WRONG_EDIT_SITE | 50 % wrong_site / 25 % no_match / 25 % syntax |
| ``qwen2.5:14b-32k`` (general, cluster mac2)  | 0.544 / 0.544 / 0.526 | 1.8 pp      | SEM_SYNTAX_INVALID  | 52 % syntax / 46 % no_match / 3 % multi_hunk  |

**Three new theorems** (P43-1 bounded-context preservation
on the external-validity bank — substrate 205.9 tokens flat
across the full cross product; P43-2 post-parser-recovery
semantic residue is structurally classifiable — total,
exhaustive, deterministic; P43-3 semantic-ceiling separation
on coder-finetuned models at N ≥ 50 — gap-zero + strategy-
invariant composition + training-mix-indexed dominance) and
**four new conjectures** (C43-1 frontier coder closes
wrong-edit-site without re-opening the substrate gap; C43-2
residue composition is training-mix-indexed, not parameter-
count-indexed; C43-3 substrate bounded-context invariant is
model-independent; C43-4 semantic residue does not decompose
under existing substrate primitives).

The ``qwen3.5:35b`` cluster run surfaced a Phase-43
regression shape: the 35B closes the NEW block with ``<<``
instead of ``<<<`` at end-of-generation. Without the
Phase-43 fix, the ``RECOVERY_CLOSED_AT_EOS`` heuristic kept
the trailing ``<<`` in the NEW payload, producing syntax
errors on every instance (171/171). The one-pattern fix in
``_strip_trailing_prose`` (``\n\s*<{2,4}\s*\Z``) strips
partial or full trailing delimiters; byte-safe under
Theorem P42-2.

**Headline**: on every coder-class model tested at N ≥ 50 the
substrate-vs-naive pass@1 gap is **0 pp**, and the frontier
``qwen3.5:35b`` MoE **beats the 14B-coder** at 0.965 vs 0.930
— direct evidence for Conjecture C43-1 (frontier model closes
the semantic residue without re-opening the substrate gap).
The substrate's durable claim is *bounded active context per
role* — substrate prompt is flat at **205.9 tokens** across
the full parser × matcher × distractor cross product on the
full 57-instance bank, while naive grows 197 → 527 tokens.
The remaining residue is *model-shaped* (wrong-edit-site on
coder models, syntax-invalid on general-purpose) and neither
the parser nor the matcher nor the substrate can close it
without re-generating the patch.

Reproduce:
```bash
# Phase-43 cluster run — qwen3.5:35b on mac1, full 57-instance bank
python -m vision_mvp.experiments.phase42_parser_sweep \
    --mode real --model qwen3.5:35b \
    --ollama-url http://192.168.12.191:11434 \
    --sandbox subprocess \
    --apply-modes strict --parser-modes strict robust \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 6 \
    --think off --max-tokens 600 --llm-timeout 600 \
    --out vision_mvp/results_phase43_parser_35b_moe_mac1.json

# Phase-43 analysis — cross-model semantic summary
python -m vision_mvp.experiments.phase43_frontier_headroom \
    --artifacts \
        vision_mvp/results_phase42_parser_14b_coder.json \
        vision_mvp/results_phase42_parser_7b_coder.json \
        vision_mvp/results_phase42_parser_9b_gemma.json \
        vision_mvp/results_phase42_parser_14b_general.json \
        vision_mvp/results_phase43_parser_35b_moe_mac1.json \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --out vision_mvp/results_phase43_frontier_summary.json
```

### Phase 44 — raw-text semantic residue capture, refined taxonomy, and public-SWE-bench-Lite drop-in readiness

Phase 43 characterised the post-parser-recovery residue with a
nine-label closed vocabulary, but every Phase-43 result was
derived from a *sentinel* proposed-patch (§ D.7) because the
Phase-42 artifact schema does not preserve raw LLM output.
Phase 44 removes that limitation, runs the strongest practical
coder-class cell on the ASPEN cluster with raw capture on, and
promotes the public-SWE-bench-Lite drop-in claim from
documented to validated code.

* **Part A — Raw-text capture** (``vision_mvp/tasks/swe_raw_capture.py``).
  ``RawCaptureRecord`` / ``RawCaptureStore`` with schema version
  ``phase44.v1``. Each record carries raw LLM bytes + SHA-256,
  the ``ParseOutcome`` dict, proposed + applied ``(old, new)``
  pairs, and the patched-source SHA-256.
  ``make_capturing_generator`` wraps a bridge generator or a
  fresh ``llm_call`` and plumbs the raw text into the store
  while preserving the Phase-42 LLM-output cache discipline.
  Opt-in — every pre-Phase-44 path runs unchanged.

* **Part B — Refined semantic taxonomy (v2 classifier).** Five
  new sub-labels partition the Phase-43 coarse buckets when raw
  bytes are available: ``SEM_RIGHT_FILE_WRONG_SPAN``
  (anchored in right file, wrong span), ``SEM_RIGHT_SPAN_WRONG_LOGIC``,
  ``SEM_PARTIAL_MULTI_HUNK_SUCCESS``,
  ``SEM_NARROW_FIX_TEST_OVERFIT``,
  ``SEM_STRUCTURAL_VALID_INERT``.
  ``classify_semantic_outcome_v2`` subsumes v1 on sentinel
  inputs (Theorem P44-2). ``REFINEMENT_MAP`` is reflexive so
  v2 can stay at the coarse level when bytes don't allow
  narrowing.

* **Part C — Phase-44 driver**
  (``vision_mvp/experiments/phase44_semantic_residue.py``).
  Sweep mode runs the Phase-42-shape experiment with raw
  capture on; analyse-only mode consumes (parent, capture)
  pairs and emits a ``phase44.summary.v1`` JSON with per-cell
  coarse + refined counters + a ``coarse_to_refined_partition``
  audit.

* **Part D — Public-SWE-bench-Lite readiness validator**
  (``vision_mvp/experiments/phase44_public_readiness.py``).
  Five checks (``schema`` → ``adapter`` → ``parser`` →
  ``matcher`` → ``test_runner``) on any local JSONL. CI-gate
  verdict: ``{"ready": true, "n": 57, "n_passed_all": 57,
  "blockers": []}`` on the bundled bank in **5.2 s wall**
  through SubprocessSandbox. The externalisation gap to public
  SWE-bench-Lite is now purely a data-availability gap.

| metric (Phase-44 canonical cell: parser=robust/nd=6/apply=strict) | pass@1 (N, R, S) | S−N gap | refined dominant residue (v2) |
|---|:-:|:-:|---|
| ``qwen3.5:35b`` (36B MoE, cluster mac2)   | 0.965 / 0.965 / 0.965 (55/57) | **0.0 pp** | right_site / test_overfit (refined from v1 wrong_site) |
| ``qwen2.5-coder:14b`` (cluster mac1)     | 0.930 / 0.930 / 0.930 | **0.0 pp** | right_file_wrong_span + partial_multi_hunk (v1 wrong_site/multi_hunk partitioned) |

**Three new theorems** (P44-1 raw capture is a lossless
projection of pipeline state; P44-2 refined classifier monotone
on sentinel inputs; P44-3 public-readiness saturates on bundled
bank) and **four new conjectures** (C44-1 coder-class
wrong_edit_site refines to right_file_wrong_span; C44-2
frontier residue refines to overfit or wrong-logic; C44-3
substrate gap is refinement-invariant on coder-class; C44-4
readiness closed under row-level filtering).

**Headline**: the substrate-vs-naive gap is **0 pp** on every
coder-class model under the v2 refined classifier (Conjecture
C44-3), and the Phase-43 § D.7 sentinel-path limitation is
closed — the 14B-coder's 4/57 residue and the 35B's 2/57
residue are now attributable to refined sub-labels by
inspection of stored bytes, no LLM replay required.
The substrate's durable claim (bounded active context per role,
Theorem P43-1) is preserved: the v2 classifier does not move a
measurement between SEM_OK and non-SEM_OK.

Reproduce:
```bash
# Phase-44 public-SWE-bench-Lite readiness validator (bundled bank)
python -m vision_mvp.experiments.phase44_public_readiness \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --out vision_mvp/results_phase44_readiness_bundled.json

# Phase-44 sweep with raw capture — qwen2.5-coder:14b on mac1
python -m vision_mvp.experiments.phase44_semantic_residue \
    --mode real --model qwen2.5-coder:14b \
    --ollama-url http://192.168.12.191:11434 \
    --sandbox subprocess \
    --parser-modes strict robust --apply-modes strict \
    --n-distractors 6 --think default --max-tokens 400 \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --out-parent vision_mvp/results_phase44_parser_14b_coder.json \
    --out-capture vision_mvp/results_phase44_capture_14b_coder.json

# Phase-44 sweep with raw capture — qwen3.5:35b on mac2
python -m vision_mvp.experiments.phase44_semantic_residue \
    --mode real --model qwen3.5:35b \
    --ollama-url http://192.168.12.248:11434 \
    --sandbox subprocess \
    --parser-modes strict robust --apply-modes strict \
    --n-distractors 6 --think off --max-tokens 600 \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --out-parent vision_mvp/results_phase44_parser_35b_moe.json \
    --out-capture vision_mvp/results_phase44_capture_35b_moe.json

# Phase-44 analysis — refined cross-model summary
python -m vision_mvp.experiments.phase44_semantic_residue \
    --analyse-only \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --artifacts \
        vision_mvp/results_phase44_parser_14b_coder.json \
        vision_mvp/results_phase44_parser_35b_moe.json \
    --captures \
        vision_mvp/results_phase44_capture_14b_coder.json \
        vision_mvp/results_phase44_capture_35b_moe.json \
    --out vision_mvp/results_phase44_refined_summary.json
```

Full phase-43 writeup: ``vision_mvp/RESULTS_PHASE43.md``.

### Phase 45 — one-command product runner, release-candidate validation, finished-product checklist

Phase 44 closed the science side (raw capture + refined
taxonomy + five-check readiness validator). Phase 45 closes the
*operability* side: a declared profile set, a one-command
runner, a report renderer, and a durable Finished-Product
Checklist in the master plan (§9 of
``docs/context_zero_master_plan.md``).

* **One command** —
  ``python3 -m vision_mvp.product --profile <name> --out-dir <d>``
  composes the Phase-44 readiness validator, the Phase-42
  parser sweep, and the Phase-44 raw-capture / refined
  taxonomy behind a stable schema
  (``phase45.product_report.v1`` + ``phase45.profile.v1``).
* **Six stable profiles** — ``local_smoke``, ``bundled_57``,
  ``bundled_57_mock_sweep``, ``aspen_mac1_coder``,
  ``aspen_mac2_frontier``, ``public_jsonl``.
* **Release-candidate artifacts** — under
  ``vision_mvp/artifacts/phase45_rc_bundled/`` (readiness
  57/57, 5.16 s), ``vision_mvp/artifacts/phase45_rc_mock_sweep/``
  (oracle pass@1 = 1.000 every strategy every cell), and
  ``vision_mvp/artifacts/phase45_mac1_recorded/`` (real-LLM
  launch command recorded for the ASPEN cluster operator).
* **Three new theorems** — P45-1 (runner composition is
  faithful), P45-2 (readiness is a hard gate for the sweep),
  P45-3 (finished-product state is the logical product of the
  per-layer theorems). **Three new conjectures** (C45-1
  profile-set completeness; C45-2 operator overhead <5 %;
  C45-3 remaining blockers are model/data shaped, not
  architecture shaped).
* **What still materially blocks finished-product status** —
  named in §9.8 of the master plan: (1) a public SWE-bench-
  Lite JSONL on local disk (🧱 external data availability),
  and (2) a ≥70B local coder-finetuned model on the cluster
  (◐ engineering). Neither is an architecture blocker.

Reproduce the Phase-45 release-candidate validation pass:
```bash
# Local smoke (mock, 8 instances, <1 s wall)
python3 -m vision_mvp.product --profile local_smoke \
    --out-dir /tmp/cz-smoke

# Bundled 57 readiness release-candidate (subprocess, ~5 s)
python3 -m vision_mvp.product --profile bundled_57 \
    --out-dir vision_mvp/artifacts/phase45_rc_bundled

# Bundled 57 + mock oracle sweep (~30 s)
python3 -m vision_mvp.product --profile bundled_57_mock_sweep \
    --out-dir vision_mvp/artifacts/phase45_rc_mock_sweep

# Real-LLM launch command record (ASPEN mac1, no LLM budget)
python3 -m vision_mvp.product --profile aspen_mac1_coder \
    --out-dir vision_mvp/artifacts/phase45_mac1_recorded

# Public SWE-bench-Lite drop-in (requires JSONL)
python3 -m vision_mvp.product --profile public_jsonl \
    --jsonl /path/to/swe_bench_lite.jsonl \
    --out-dir vision_mvp/artifacts/phase45_public_lite
```

Full Phase-45 writeup: ``vision_mvp/RESULTS_PHASE45.md``.

### Phase 46 — External-exercise readiness: public-data import, CI gate, frontier-model slot

Phase 45 closed "finished product within programme control".
Phase 46 closes the *boundary* between the finished product and
the outside world — the remaining blockers named in master plan
§9.8 (public JSONL, ≥70B model, CI pipeline) now meet the code at
a single command.

* **Public-data import CLI** —
  ``python3 -m vision_mvp.product.import_data --jsonl X --out Y``.
  Schema audit (native / hermetic / ambiguous / unusable),
  duplicate-``instance_id`` detection, decode-error / non-object
  / empty-bank enumeration, delegated Theorem-P44-3 readiness.
  Exit codes: ``0`` clean / ``1`` blocker / ``2`` file-not-found.
* **CI gate** —
  ``python3 -m vision_mvp.product.ci_gate --report <product_report.json> ...``.
  Five checks (schema / profile / readiness threshold / sweep
  outcome / artifact presence), multi-report aggregation,
  ``--min-ready-fraction`` + ``--min-pass-at-1`` thresholds,
  ``--require-profile`` whitelist. Machine-readable
  ``phase46.ci_verdict.v1``.
* **Frontier-model slot** — profile
  ``aspen_mac1_coder_70b`` + ``profiles.model_availability()``.
  Adding a 70B coder model is a one-string config change; the
  runner attaches ``model_metadata`` to the recorded launch
  payload so downstream tooling knows whether the model is
  resident or pending.
* **Three new theorems** — P46-1 (import-audit saturates on the
  bundled bank), P46-2 (CI-gate composition is faithful), P46-3
  (capability declaration ≠ residency). **Three new conjectures**
  (C46-1 remaining blockers are boundary-shaped; C46-2 70B
  ceiling lift + substrate invariance; C46-3 CI consumption
  closed under profile extension).

Reproduce the Phase-46 boundary-surface exercises:
```bash
# Audit the bundled bank (57/57 rows, readiness READY)
python3 -m vision_mvp.product.import_data \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --out   vision_mvp/artifacts/phase46_public_audit/bundled_bank_audit.json

# Audit a missing path (exit 2)
python3 -m vision_mvp.product.import_data \
    --jsonl /nonexistent/public_swe_bench_lite.jsonl

# Gate the three Phase-45 RC artifacts (aggregate ok)
python3 -m vision_mvp.product.ci_gate \
    --report \
        vision_mvp/artifacts/phase45_rc_bundled/product_report.json \
        vision_mvp/artifacts/phase45_rc_mock_sweep/product_report.json \
        vision_mvp/artifacts/phase45_mac1_recorded/product_report.json \
    --out vision_mvp/artifacts/phase46_ci_gate/aggregate_verdict.json

# Exercise the frontier-model slot (no LLM invoked)
python3 -m vision_mvp.product --profile aspen_mac1_coder_70b \
    --out-dir vision_mvp/artifacts/phase46_frontier_slot
```

Full Phase-46 writeup: ``vision_mvp/RESULTS_PHASE46.md``.

### Phase 42 — parser-compliance attribution layer, 57-instance SWE-bench-Lite bank, cluster rerun

Phase 41 surfaced a new attribution boundary above the matcher
axis: the LLM-output parser. On the Phase-41 bank ``gemma2:9b``
emitted the semantically correct fix on every instance but
failed to close the bridge's ``<<<`` output delimiter, so every
patch landed as ``patch_no_match`` before the matcher axis
became measurable. Phase 42 promotes that boundary to a
first-class attribution surface and closes the ≥ 50-instance
external-validity threshold.

* **Part A — Parser-compliance layer**
  (``vision_mvp/tasks/swe_patch_parser.py``). A
  ``parse_patch_block(text, mode, unified_diff_parser)`` entry
  point with three modes (``PARSER_STRICT`` = Phase-41
  baseline; ``PARSER_ROBUST`` = Phase-42 default with five
  heuristics — tolerant block closing at end-of-generation,
  trailing-prose stripping, unified-diff fallback, two-fence
  pairing, ``OLD:``/``NEW:`` label-prefix; ``PARSER_UNIFIED``
  = diff-only) and a closed vocabulary: ten failure kinds
  (``PARSE_OK`` / ``PARSE_EMPTY_OUTPUT`` / ``PARSE_NO_BLOCK``
  / ``PARSE_UNCLOSED_NEW`` / ``PARSE_UNCLOSED_OLD`` /
  ``PARSE_MALFORMED_DIFF`` / ``PARSE_EMPTY_PATCH`` /
  ``PARSE_MULTI_BLOCK`` / ``PARSE_PROSE_ONLY`` /
  ``PARSE_FENCED_ONLY``), six recovery labels
  (``RECOVERY_NONE`` / ``RECOVERY_CLOSED_AT_EOS`` /
  ``RECOVERY_FENCED_CODE`` / ``RECOVERY_LABEL_PREFIX`` /
  ``RECOVERY_UNIFIED_DIFF`` / ``RECOVERY_LOOSE_DELIM``).
  ``ParserComplianceCounter`` aggregates per-cell and
  exposes ``compliance_rate`` / ``raw_compliance_rate`` /
  ``recovery_lift``; the bridge's ``llm_patch_generator``
  gains opt-in ``parser_mode`` / ``parser_counter`` /
  ``prompt_style`` kwargs. ``None`` preserves Phase-41
  behaviour.
* **Part B — 57-instance SWE-bench-Lite-style bank.** The
  Phase-41 28-instance ``swe_lite_style_bank.jsonl`` grown
  with 29 new instances covering broader edit classes
  (string manipulation, numeric guards, sequence
  construction, dict helpers, recursion/iteration,
  exception handling, set algebra, class state
  transitions, binary search, graph walk, multi-hunk
  class edits). Every instance round-trips through the
  oracle before being written — the bank-builder refuses
  any instance whose diff doesn't parse, whose OLD blocks
  aren't unique, or whose oracle-patched source doesn't
  pass the hidden test.
* **Part C — Parser sweep + cluster driver**
  (``experiments/phase42_parser_sweep``). Sweeps
  ``(parser_mode × apply_mode × n_distractors)`` with an
  LLM-output cache so the second parser cell costs only
  sandbox wall. ``LLMClient(base_url=…)`` + ``--ollama-url``
  forward to the ASPEN cluster: coding/generation runs on
  macbook-1 (``http://192.168.12.191:11434``), secondary
  runs on macbook-2 (``http://192.168.12.248:11434``) or
  localhost — fan out in parallel.

| metric (Phase-42 mock, 57 instances) | naive | routing | substrate |
|---|---:|---:|---:|
| pass@1 (oracle)                         | 1.000 | 1.000 | 1.000 |
| substrate prompt tokens (n_d ∈ {0..24}) | 197 → 527 | 93 → 423 | **205.9 (constant)** |
| events to patch_gen                     | 4 → 28 | 0 → 24 | **0** |
| wall (1 368 sandboxed measurements)     | — | — | **122 s** |

**Three new theorems** (P42-1 parser-compliance attribution
decomposition ``Δ pass@1 = |R_recovered_parser| −
|R_regressed_parser|``; P42-2 parser recovery cannot produce
a false pass — byte-provenance argument on recovery
heuristics; P42-3 robust parser dominates on format-
noncompliant generators whose dominant failure mode is one
of ``{unclosed_new, prose_only, fenced_only_2,
label_prefix, fence_wrapped_payload}``) and **three new
conjectures** (C42-1 substrate-vs-naive gap ≤ 1 pp at N ≥
50; C42-2 parser-compliance dominates matcher-
permissiveness at 7B–30B; C42-3 three-axis decomposition
completeness ``pass@1 = P_parse · P_match · P_semantic ·
P_sandbox``).

**Headline empirical result**: ``qwen2.5-coder:14b`` on
the ASPEN cluster macbook-1, 57 instances, strict matcher
— pass@1 jumps from **0.018 / 0.018 / 0.018**
(naive / routing / substrate, strict parser, 56 × 3
`patch_no_match` from fence-wrapped OLD payloads) to
**0.930 / 0.930 / 0.930** (robust parser, with
`RECOVERY_FENCE_WRAPPED`): **+91.2 percentage-point pass@1
lift, 52 instances recovered on every strategy, 0
regressed, substrate-vs-naive gap = 0 pp**. On cluster
macbook-2 ``qwen2.5:14b-32k`` (general) the same parser
axis lifts +1.8 pp (12 % of outputs fence-wrapped — model-
specific η). On localhost ``qwen2.5-coder:7b`` — the Phase-41
headline model at N = 57 — the parser axis is empirically
null (no fence-wrapping); pass@1 = **0.842** on every
strategy. And on localhost ``gemma2:9b`` (the Phase-41
§ D.4 failure-mode replication), pass@1 flips from
**0/28 → 24/28 = 0.857** on every strategy under the
robust parser — **+85.7 pp lift** from the exact same
LLM output, changing only the parser mode. Substrate-vs-
naive gap is **0 pp** on all four real-LLM cells
(14B-coder cluster, 14B-general cluster, 7B-coder
localhost, 9B-gemma localhost) — the strongest empirical
support for Conjecture C42-1 in the programme to date.
Full per-cell tables in
``vision_mvp/RESULTS_PHASE42.md`` § D.3, § D.4, § D.4b,
§ D.5.

Reproduce:
```bash
# Phase-42 mock — 1 368 measurements in ~122 s, no LLM, no docker
python -m vision_mvp.experiments.phase41_swe_lite_sweep \
    --mode mock --sandbox subprocess \
    --apply-modes strict lstrip \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 0 6 12 24 \
    --out vision_mvp/results_phase42_swe_lite_mock.json

# Phase-42 real LLM — cluster mac1 qwen2.5-coder:14b
python -m vision_mvp.experiments.phase42_parser_sweep \
    --mode real --model qwen2.5-coder:14b \
    --ollama-url http://192.168.12.191:11434 \
    --sandbox subprocess \
    --apply-modes strict --parser-modes strict robust \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 6 \
    --out vision_mvp/results_phase42_parser_14b_coder.json

# Phase-42 secondary — localhost qwen2.5-coder:7b
python -m vision_mvp.experiments.phase42_parser_sweep \
    --mode real --model qwen2.5-coder:7b \
    --ollama-url http://localhost:11434 \
    --sandbox subprocess \
    --apply-modes strict --parser-modes strict robust \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 6 \
    --out vision_mvp/results_phase42_parser_7b_coder.json

# Phase-42 test slice (parser + bank regression; 31 tests)
python -m pytest vision_mvp/tests/test_phase42_parser.py -q
```

### Phase 41 — larger SWE-bench-Lite-style empirical sweep, patch-matcher permissiveness attribution, stronger-model datapoint

Phase 40 proved the real SWE-bench-style loop exists. Phase
41 moves the next credibility step: **scale plus
attribution**. Three tightly coupled artifacts ship, keeping
the agent-team substrate central:

* **Part A — 28-instance real-shape bank**
  (``vision_mvp/tasks/data/swe_lite_style_bank.jsonl``).
  ~4.7× the Phase-40 mini bank, authored to cover a
  disciplined spectrum of edit shapes (single-hunk,
  multi-hunk, multi-function, operator-typo, off-by-one,
  wrong-branch, seed-wrong, aggregate-missing, mutation-
  vs-copy, parity-partition, slice-direction, index-
  return, polarity-flipped, empty-guard, type-conversion,
  unicode edge, ambiguous comparator). The bank-builder
  (``_build_swe_lite_bank.py``) refuses to register any
  instance whose diff doesn't parse, whose OLD blocks
  aren't unique, or whose oracle-patched source doesn't
  pass the hidden test.
* **Part B — Permissive patch-matcher axis.**
  ``apply_patch(..., mode=…)`` accepts one of
  ``strict`` (Phase-40 default, byte-exact),
  ``lstrip`` (leading-whitespace drift tolerance),
  ``ws_collapse`` (internal-whitespace drift),
  ``line_anchored`` (trailing-whitespace drift). All
  three permissive modes preserve the **unique-match
  discipline** — a normalised OLD that matches more than
  one source region is rejected. ``apply_mode`` is
  threaded through ``run_swe_loop``, ``Sandbox.run``, and
  ``run_swe_loop_sandboxed``; ``SWEReport.config`` records
  it for audit.
* **Part C — Attribution-aware driver**
  (``experiments/phase41_swe_lite_sweep``). Caches every
  LLM call per ``(instance, strategy, n_distractors)``
  so permissive cells reuse strict cells' proposals; emits
  a per-strategy ``{recovered, regressed, unchanged_pass,
  unchanged_fail}`` set delta between each permissive
  mode and strict.

| metric (Phase-41 mock, 28 instances) | naive | routing | substrate |
|---|---:|---:|---:|
| pass@1 (oracle)                       | 1.000 | 1.000 | 1.000 |
| substrate prompt chars (n_d ∈ {0..24}) | 807 → 2 126 | 373 → 1 692 | **746 (constant)** |
| events to patch_gen                   | 4 → 28 | 0 → 24 | **0** |
| wall (672 sandboxed measurements)     | — | — | **53.0 s** |

**Three new theorems** (P41-1 bounded-context
preservation at 4.7× scale, P41-2 oracle-ceiling is
matcher-mode-invariant, P41-3 matcher-permissiveness
attribution decomposition ``Δ pass@1 = |R_recovered| −
|R_regressed|``) and **five new conjectures** (C41-1
communication-bounded at ≥ 50 instances, C41-2
matcher-permissiveness saturation, C41-3 stronger-model
strict-floor saturation, C41-4 regime decomposition
``pass@1 = P_comm · P_gen``, C41-5 parser-compliance
attribution boundary). Real-LLM sweeps on
``qwen2.5-coder:7b`` (28 instances, **pass@1 = 0.929 /
0.929 / 0.893** naive / routing / substrate under strict;
byte-identical under permissive matchers;
``R_recovered = R_regressed = ∅``) and ``gemma2:9b`` (28
instances, **pass@1 = 0 / 0 / 0** — the general-purpose
9B emits the semantically correct fix but fails the
bridge's ``<<<`` output-delimiter contract, surfacing
a new attribution boundary named by Conjecture C41-5);
see ``vision_mvp/RESULTS_PHASE41.md`` § D.3 and § D.4.

Reproduce:
```bash
# Phase-41 mock — 672 measurements in ~ 53 s, no LLM, no docker
python -m vision_mvp.experiments.phase41_swe_lite_sweep \
    --mode mock --sandbox subprocess \
    --apply-modes strict lstrip \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 0 6 12 24 \
    --out vision_mvp/results_phase41_swe_lite_mock.json

# Phase-41 real LLM — qwen2.5-coder:7b on all 28 instances
python -m vision_mvp.experiments.phase41_swe_lite_sweep \
    --mode real --model qwen2.5-coder:7b --sandbox subprocess \
    --apply-modes strict lstrip \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 6 \
    --out vision_mvp/results_phase41_swe_lite_7b.json

# Phase-41 test slice (18 tests; ~ 28 s)
python -m pytest vision_mvp/tests/test_phase41_swe_lite.py -q
```

### Phase 40 — real SWE-bench-style loader, sandboxed execution boundary, first end-to-end real-shape evaluation

Phase 39 left two gaps to end-to-end SWE-bench: the
*mechanical* (unidiff parser + sandboxed runner +
JSONL loader, Theorem P39-4) and the *empirical*
(does substrate dominance hold at SWE-bench Lite scale,
Conjectures C39-3 / C39-4). Phase 40 closes the
mechanical gap end-to-end:

* **Part A — Real-shape loader / adapter
  (``tasks/swe_bench_bridge`` extension).**
  ``parse_unified_diff`` (a tolerant ``git diff``
  parser), ``SWEBenchAdapter.from_swe_bench_dict``
  (the real-shape adapter — derives ``buggy_function``
  from the diff hunk, promotes a ``test_patch`` to a
  runnable ``test_source``), and ``load_jsonl_bank``
  (hermetic JSONL loader with per-instance file
  namespacing). A bundled six-instance JSONL artifact
  (``vision_mvp/tasks/data/swe_real_shape_mini.jsonl``)
  exercises the full path offline in sub-second.
* **Part B — Sandboxed execution boundary
  (``tasks/swe_sandbox``).** Three backends behind
  one ``Sandbox`` protocol: ``InProcessSandbox``
  (Phase-39 wrapped), ``SubprocessSandbox`` (new —
  wall-clock timeout, tempdir cwd, sanitised env, JSON
  outcome protocol), ``DockerSandbox`` (new — optional;
  ``--network=none --read-only`` rootfs, ``tmpfs``
  ``/work``). ``select_sandbox("auto")`` picks
  Docker → subprocess → in-process by availability;
  ``run_swe_loop_sandboxed`` wires it into the bridge.
* **Part C — End-to-end driver
  (``experiments/phase40_real_swe_bridge``).** Loader
  + substrate + sandbox + (optional) real LLM patch
  generator. Mock run on the bundled JSONL across
  n_distractors ∈ {0, 6, 12, 24} = 72 measurements
  in **5.6 s**, pass@1 = 1.000 / 1.000 / 1.000 (oracle
  ceiling). Real-LLM ``qwen2.5:0.5b``: every cell
  hits ``patch_no_match`` (transcription-bounded
  regime). Real-LLM ``qwen2.5-coder:7b``: pass@1 =
  0.833 / 0.833 / 0.667 (naive / routing /
  substrate) on 6 instances — substrate ranks one
  instance below naive on byte-strict matcher
  variance, sitting cleanly in P39-2's
  transcription-bounded regime.

| metric (Phase-40 mock JSONL bank) | naive | routing | substrate |
|---|---:|---:|---:|
| pass@1 (oracle generator)         | 1.000 | 1.000 | 1.000 |
| substrate prompt chars (n_d ∈ {0..24}) | 826 → 2 145 | 373 → 1 692 | **813 (constant)** |
| events to patch_gen               | 4 → 28 | 0 → 24 | **0** |
| substrate handoffs                | 2     | 2     | **5**   |

| metric (Phase-40 real LLM, n_d=6) | naive | routing | substrate |
|---|---:|---:|---:|
| pass@1 (qwen2.5:0.5b)             | 0.000 | 0.000 | 0.000 |
| pass@1 (qwen2.5-coder:7b)         | 0.833 | 0.833 | 0.667 |
| dominant failure                  | patch_no_match (1/6 → 2/6 substrate) ||

**Three new theorems** (P40-1 unidiff round-trip,
P40-2 real-shape bounded-context, P40-3 sandbox-
boundary preservation) and **three new conjectures**
(C40-1 sandbox cost amortisable, C40-2 loader
sufficiency for SWE-bench Lite, C40-3 sandbox-axis
equivalence). See ``vision_mvp/RESULTS_PHASE40.md``.

Reproduce:
```bash
# Phase-40 mock — sub-second, no LLM, no docker required
python -m vision_mvp.experiments.phase40_real_swe_bridge \
    --mode mock --sandbox subprocess \
    --jsonl vision_mvp/tasks/data/swe_real_shape_mini.jsonl \
    --n-distractors 0 6 12 24 \
    --out vision_mvp/results_phase40_real_swe_bridge_mock.json

# Phase-40 real LLM — qwen2.5:0.5b (~ 100 s)
python -m vision_mvp.experiments.phase40_real_swe_bridge \
    --mode real --model qwen2.5:0.5b --sandbox subprocess \
    --jsonl vision_mvp/tasks/data/swe_real_shape_mini.jsonl \
    --n-distractors 6 \
    --out vision_mvp/results_phase40_real_swe_bridge_0p5b.json

# Phase-40 test slice (26 tests; ~ 11 s)
python -m pytest vision_mvp/tests/test_phase40_real_swe_bridge.py -q
```

### Phase 39 — real-LLM prompt-variant sweep, frontier-model substrate breadth, SWE-bench-style bridge

Phase 39 attacks three coupled gaps Phase 38 left open:
(A) the real-LLM prompt-variant measurement; (B)
cross-family frontier-model breadth on the substrate
slice; (C) a runnable SWE-bench-style bridge that wires
the existing typed-handoff substrate through a multi-role
patch / test team.

* **Part A — Real-LLM prompt-variant measurement
  (``--mode real``).** The Phase-38 mock predicted that
  ``rubric`` and ``contrastive`` variants would cut
  ``sem_wrong`` from 0.69 → 0.23. On real
  ``qwen2.5:0.5b`` the prediction is *wrong* — four of
  five variants reproduce the Phase-37 default
  distribution to within ±0 calls; the fifth
  (``forced_order``) merely converts semantic errors
  into malformed parses. **Theorem P39-1**: on the 0.5B
  / 7B size class, the Phase-37
  ``sem_root_as_symptom`` bias is *model-shaped, not
  prompt-shaped*. This empirically refutes the
  optimistic read of Conjecture C38-3 on these models.
* **Part B — Frontier-model bounded substrate sweep
  (``experiments/phase39_frontier_substrate``).** A
  cross-family bench on Phase-31 incident triage at
  k = 6, seed = 31 across mock + 2–3 local LLMs
  (``llama3.1:8b``, ``gemma2:9b``,
  ``qwen2.5-coder:7b``). Substrate-side correctness
  preservation reproduces across families.
* **Part C — SWE-bench-style bridge
  (``tasks/swe_bench_bridge`` +
  ``experiments/phase39_swe_bridge``).** A
  ``SWEBenchStyleTask`` schema mirroring SWE-bench's
  public instance shape; a four-instance ``MiniSWEBank``
  with real Python files, real bugs, real gold patches,
  real in-process tests; a four-role team
  (``issue_reader`` / ``code_searcher`` /
  ``patch_generator`` / ``test_runner``) wired through
  the unchanged Phase-31 ``HandoffRouter``.
  ``SWEBenchAdapter.from_dict`` documents the schema
  mapping for a future real-SWE-bench loader.
  **Theorem P39-3**: the substrate's bounded-context
  invariant extends to the SWE-style team — patch_
  generator prompt size is constant in n_distractors
  while naive grows linearly. **Theorem P39-4**: the
  schema gap to public SWE-bench is adapter-shaped, not
  architectural.

| metric (mini-SWE bank) | naive | routing | substrate |
|---|---:|---:|---:|
| pass@1 (oracle generator)  | 1.000 | 1.000 | 1.000 |
| pass@1 (qwen2.5:0.5b)      | 0.000 | 0.000 | 0.000 |
| pass@1 (qwen2.5-coder:7b)  | 0.250 | 0.250 | 0.250 |
| mean prompt chars (n_d=24) | 1936  | 1360  | **842** |
| events to patch_gen        | 14    | 10    | **0**   |
| substrate handoffs         | 2     | 2     | **5**   |

| prompt variant | 0.5b correct | 7b correct | 7b dyn ctst |
|---|---:|---:|---:|
| default        | 0.100 | 0.100 | 0.000 |
| contrastive    | 0.100 | **0.500** | 0.000 |
| few_shot       | 0.100 | 0.200 | **0.250** |
| rubric         | 0.100 | **0.400** | 0.000 |
| forced_order   | 0.100 | 0.200 | 0.000 |

(0.5B: every variant pinned at correct = 0.10 — model-shaped bias.
7B: contrastive lifts correct 5×; few_shot is the only variant
that lifts downstream contested accuracy — partial prompt-shape.)

| frontier substrate slice (incident triage, k=6) | naive | substrate | substrate_wrap |
|---|---:|---:|---:|
| qwen2.5-coder:7b acc_full | 0.000 | 0.400 | **0.800** |
| llama3.1:8b acc_full      | 0.000 | 0.200 | **0.600** |
| gemma2:9b acc_full        | 0.000 | 0.000 | **1.000** |
| gemma2:9b acc_root_cause  | 0.000 | 0.400 | **1.000** |
| substrate prompt tokens   | 573  | **196** | 229 |

(Cross-family reproduction of the Phase-31 substrate dominance:
substrate_wrap dominates naive by +60 to +100 pp on 7B / 8B / 9B;
**gemma2:9b saturates at 1.000/1.000** — first real LLM in the
programme to fully match the substrate ceiling on a non-code team.
Substrate prompt is constant at 196 chars regardless of model.)

See ``vision_mvp/RESULTS_PHASE39.md`` for theorems
(P39-1..P39-4) and conjectures (C39-1..C39-4).

Reproduce:
```bash
# Real LLM prompt-variant sweep (~ 100s on 0.5b)
python3 -m vision_mvp.experiments.phase38_prompt_calibration \
    --mode real --models qwen2.5:0.5b --seeds 35 \
    --distractor-counts 4 \
    --out vision_mvp/results_phase39_prompt_calibration_0p5b.json

# SWE-bench-style mini bridge (sub-second mock)
python3 -m vision_mvp.experiments.phase39_swe_bridge \
    --mode mock --n-distractors 0 6 12 24 \
    --out vision_mvp/results_phase39_swe_bridge_mock.json

# Frontier-model substrate sweep on incident triage
python3 -m vision_mvp.experiments.phase39_frontier_substrate \
    --models llama3.1:8b gemma2:9b qwen2.5-coder:7b \
    --domains incident --distractor-counts 6 --seeds 31 \
    --out vision_mvp/results_phase39_frontier_substrate.json
```

### Phase 38 — two-layer ensemble composition, minimum dynamic primitive ablation, prompt-shaped reply calibration

Phase 38 closes three Phase-37 frontier items at once: (A)
two-layer ensemble composition across the extractor and
reply axes; (B) a per-feature falsifier table for the
five-feature candidate minimum dynamic primitive; (C) a
pipeline for measuring whether the Phase-37
``sem_root_as_symptom`` bias is prompt-shaped.

* **Part A — Two-layer ensemble composition
  (``core/two_layer_ensemble`` +
  ``core/extractor_adversary``).**
  ``PathUnionCausalityExtractor`` + ``UnionClaimExtractor``
  stack layer-1 (extractor-axis) and layer-2 (reply-axis)
  ensembles. **Theorem P38-1**: on a conjunction attack
  (layer-1 adversary drops the gold claim AND layer-2
  biased primary emits IR on every candidate), the two-
  layer stack ``UnionClaimExtractor ∘
  EnsembleReplier(MODE_DUAL_AGREE)`` is the unique
  configuration that recovers (0.833 vs 0.333 for every
  single-layer alternative; 1.000 contested). **Theorem
  P38-2**: on the Phase-37 ``adv_drop_root`` cell where
  Theorem P37-4 proved every reply-axis ensemble powerless,
  ``PathUnionCausalityExtractor(PATH_MODE_UNION_ROOT)``
  above-noise combiner recovers to 1.000 accuracy.
* **Part B — Minimum primitive ablation
  (``core/primitive_ablation``).** Feature-flagged thread
  runner with five toggles; ablation table across Phase-35
  contested + Phase-37 nested banks. **Theorem P38-3**:
  ``typed_vocab``, ``terminating_resolution``, and
  ``round_aware_state`` are individually load-bearing;
  ``bounded_witness`` is null-control on accuracy but
  load-bearing for Theorem P35-2's context bound;
  ``frozen_membership`` is null-control on tested families
  (Conjecture C38-2 asserts a family exists where it is
  load-bearing).
* **Part C — Prompt-variant calibration
  (``core/prompt_variants``).** Five surgical variants of
  the Phase-36 default prompt (default, contrastive,
  few_shot, rubric, forced_order). Driver supports
  ``--mode mock`` (sub-second deterministic simulation)
  and ``--mode real`` (Ollama sweep). Mock headline:
  ``rubric`` and ``contrastive`` cut the semantic-wrong
  rate from 0.688 → 0.225. **Theorem P38-4**: every
  variant preserves the Phase-36 typed-reply contract;
  bias shifts are measurable without enlarging the
  substrate surface. Real-LLM measurement is one CLI
  parameter away (Conjecture C38-3).

| cell / config      | baseline | ext_only | reply_only | **two_layer** | two_layer_path_union |
|---|---:|---:|---:|---:|---:|
| clean              | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ext_drop_gold      | 0.333 | **0.833** | 0.333 | 0.833 | 0.833 |
| rep_biased_primary | 0.333 | 0.333 | **1.000** | **1.000** | 0.333 |
| **conjunction**    | 0.333 | 0.333 | 0.333 | **0.833** | 0.333 |
| adv_drop_root      | 0.333 | 0.333 | 0.333 | 0.333 | **1.000** |

| feature removed          | contested | nested |
|---|---:|---:|
| none (full)              | 1.000 | 1.000 |
| typed_vocab              | 0.500 | 0.333 |
| terminating_resolution   | 0.333 | 0.000 |
| round_aware_state        | 1.000 | 0.000 |
| bounded_witness          | 1.000 | 1.000 |
| frozen_membership        | 1.000 | 1.000 |

| prompt variant | correct_rate (mock) | sem_wrong (mock) |
|---|---:|---:|
| default        | 0.312 | 0.688 |
| **contrastive** | **0.775** | **0.225** |
| few_shot       | 0.463 | 0.537 |
| **rubric**     | **0.775** | **0.225** |
| forced_order   | 0.500 | 0.500 |

See ``vision_mvp/RESULTS_PHASE38.md`` for theorems
(P38-1..P38-4) and conjectures (C38-1..C38-3).

Reproduce:
```bash
# Two-layer ensemble sweep (sub-second)
python3 -m vision_mvp.experiments.phase38_two_layer_ensemble \
    --seeds 35 36 37 --distractor-counts 4 6 \
    --out vision_mvp/results_phase38_two_layer_ensemble.json

# Minimum primitive ablation (sub-second)
python3 -m vision_mvp.experiments.phase38_primitive_ablation \
    --seeds 35 36 --distractor-counts 4 6 \
    --out vision_mvp/results_phase38_primitive_ablation.json

# Prompt calibration, mock mode (sub-second)
python3 -m vision_mvp.experiments.phase38_prompt_calibration \
    --mode mock --seeds 35 36 --distractor-counts 4 6 \
    --out vision_mvp/results_phase38_prompt_calibration_mock.json

# Prompt calibration, real Ollama (one variant at ~20s on 0.5b)
python3 -m vision_mvp.experiments.phase38_prompt_calibration \
    --mode real --models qwen2.5:0.5b --seeds 35 \
    --distractor-counts 4 \
    --out vision_mvp/results_phase38_prompt_calibration_real.json
```

### Phase 37 — real-LLM reply calibration, reply-axis ensembles, nested-contest equivalence

Phase 37 sharpens three axes Phase 36 left open: (A) real-LLM
calibration of the Phase-36 synthetic reply-noise channel; (B)
reply-axis ensemble defenses; (C) thread-vs-adaptive equivalence
on a task family with multi-round state.

* **Part A — Real-LLM reply calibration
  (``core/reply_calibration``).** ``CalibratingReplier`` wraps
  an ``LLMThreadReplier`` with a per-call oracle comparator and
  records every call into a 9-bucket taxonomy. On the Phase-35
  contested bank: both ``qwen2.5:0.5b`` and ``qwen2.5-coder:7b``
  emit 100 % well-formed JSON, but 90 % semantically wrong
  (50 % ``sem_root_as_symptom`` + 40 %
  ``sem_uncertain_as_symptom``). **Theorem P37-1:** real-LLM
  reply noise is dominated by semantic mislabel, not syntactic
  failure — the Phase-36 synthetic malformed_prob knob is a
  near-useless surrogate on this task.
* **Part B — Reply-axis ensemble (``core/reply_ensemble``).**
  Three modes: ``dual_agree`` (AND-gated parallel),
  ``primary_fallback`` (chatty primary + deterministic
  fallback), ``verified`` (primary + deterministic verifier).
  **Theorem P37-2:** under a biased primary (always emits IR),
  dual_agree and verified recover from 33 % to 100 %.
  **Theorem P37-3:** under synthetic malformed_prob=0.5,
  primary_fallback recovers from 83 % to 100 %. **Theorem P37-4**
  (structural limit): under extractor-output-level noise
  (adversarial drop_root, synthetic mislabel), no ensemble
  mode helps — the ensemble is above the noise wrapper.
* **Part C — Nested-contest thread vs adaptive sub
  (``tasks/nested_contested_incident``).** Three scenarios
  where round-1 replies are insufficient. Four strategies ×
  18 measurements: static=0.000, adaptive_sub_1r=0.000,
  **adaptive_sub_2r=1.000** (18 briefings), **dynamic_nested_2r=
  1.000** (0 briefings). **Theorem P37-5:** accuracy
  equivalence EXTENDS to nested contests while exposing a
  structural-complexity separation — the thread reads round-1
  replies natively, adaptive-sub needs explicit inter-round
  briefing edges.

| noise cell | mode | dyn acc | dyn contested |
|---|---|---:|---:|
| clean                 | single            | 1.000 | 1.000 |
| synth_malformed_0.5   | single            | 0.833 | 0.750 |
| synth_malformed_0.5   | **primary_fallback** | **1.000** | **1.000** |
| biased_primary_ir     | single            | 0.333 | 0.000 |
| biased_primary_ir     | **dual_agree**    | **1.000** | **1.000** |
| biased_primary_ir     | **verified**      | **1.000** | **1.000** |
| adv_drop_root         | any mode          | 0.333 | 0.000 |
| synth_mislabel_0.5    | any mode          | 0.333 | 0.000 |

| strategy (nested bank) | acc | briefings |
|---|---:|---:|
| static_handoff        | 0.000  | 0  |
| adaptive_sub_1r       | 0.000  | 0  |
| **adaptive_sub_2r**   | **1.000** | **18** |
| **dynamic_nested_2r** | **1.000** | **0** |

See ``vision_mvp/RESULTS_PHASE37.md`` for theorems (P37-1..
P37-5) and conjectures (C37-1..C37-4).

Reproduce:
```bash
# Real-LLM calibration (Ollama required; 20s for 0.5b)
python3 -m vision_mvp.experiments.phase37_real_reply_calibration \
    --models qwen2.5:0.5b qwen2.5-coder:7b --seeds 35 \
    --distractor-counts 4 \
    --out vision_mvp/results_phase37_real_reply_calibration.json

# Reply-axis ensemble sweep (sub-second mock)
python3 -m vision_mvp.experiments.phase37_reply_ensemble \
    --seeds 35 36 --distractor-counts 6 \
    --out vision_mvp/results_phase37_reply_ensemble.json

# Nested-contest comparison (sub-second mock)
python3 -m vision_mvp.experiments.phase37_nested_contest \
    --seeds 37 38 39 --distractor-counts 4 6 \
    --out vision_mvp/results_phase37_nested_contest.json
```

### Phase 36 — dynamic coordination under reply noise, LLM-driven typed replies, and adaptive subscriptions

Phase 36 stresses the Phase-35 dynamic-coordination primitive
along three coupled axes:

* **Part A — Reply-axis noise (``core/reply_noise``).**
  ``ReplyNoiseConfig`` perturbs producer-local causality replies
  with Bernoulli drop / mislabel; ``AdversarialReplyConfig``
  targets the gold ``INDEPENDENT_ROOT`` reply under a per-scenario
  budget. **Theorem P36-1:** under i.i.d. noise the dynamic
  accuracy satisfies ``Pr[D_dyn = gold] = (1-p)·(1-q)``; static
  is capped at ``≤ 1/2``. Dominance persists for ``p + q < 1/2``.
  Empirically, dynamic is at 91.7 % at drop_prob=0.25 and
  degrades to the static baseline at drop_prob≥0.75.
  **Theorem P36-2:** a single targeted adversarial
  ``drop_root`` (budget ``b = 1``) collapses both dynamic and
  adaptive-sub to the static baseline (33.3 %).
* **Part B — LLM-driven typed replies
  (``core/llm_thread_replier``).** ``LLMThreadReplier`` drives
  a narrow LLM call (one JSON line — reply_kind ∈ Phase-35 enum,
  bounded witness), filters out-of-vocab / malformed at parse,
  falls back to UNCERTAIN. **Theorem P36-3:** under well-formed
  in-vocab replies, the LLM replier is behaviourally identical
  to the deterministic oracle (100 % contested accuracy).
  Graceful decay at ``malformed_prob = 0.5`` to 66.7 %.
* **Part C — Bounded adaptive subscriptions
  (``core/adaptive_sub``).** ``AdaptiveSubRouter`` extends the
  Phase-31 router with a bounded, TTL-expiring edge-install /
  tick primitive (hard cap ``max_active_edges``). A new strategy
  ``STRATEGY_ADAPTIVE_SUB`` installs one temporary edge per
  producer per contested scenario. **Theorem P36-4:** across
  96 paired measurements (drop × mislabel × k × seed) the
  dynamic-thread vs adaptive-sub accuracy gap is **0.000 pp** at
  every cell; token overhead is +12 %. Conjecture C35-5
  empirically confirmed on this task family.

| noise | dynamic | adaptive_sub | static |
|---|---:|---:|---:|
| drop=0.0, mis=0.0  | 1.000 | 1.000 | 0.333 |
| drop=0.25, mis=0.0 | 0.917 | 0.917 | 0.333 |
| drop=0.5, mis=0.0  | 0.667 | 0.667 | 0.333 |
| drop=1.0, mis=0.0  | 0.333 | 0.333 | 0.333 |
| adv/drop_root, b=1 | 0.333 | 0.333 | 0.333 |

See ``vision_mvp/RESULTS_PHASE36.md`` for theorems + the full
noise × primitive × seed grid.

Reproduce:
```bash
# Mock sweeps (all sub-second)
python3 -m vision_mvp.experiments.phase36_noisy_dynamic \
    --mock --seeds 35 36 \
    --drop-probs 0.0 0.1 0.25 0.5 0.75 1.0 --mislabel-probs 0.0 0.25 \
    --out vision_mvp/results_phase36_noisy_dynamic.json

python3 -m vision_mvp.experiments.phase36_llm_replies \
    --mock --seeds 35 36 --malformed-probs 0.0 0.1 0.25 0.5 \
    --out vision_mvp/results_phase36_llm_replies_mock.json

python3 -m vision_mvp.experiments.phase36_adaptive_sub \
    --mock --seeds 35 36 --distractor-counts 6 20 \
    --drop-probs 0.0 0.25 0.5 1.0 \
    --out vision_mvp/results_phase36_adaptive_sub.json
```

### Phase 35 — dynamic, bounded communication primitives and a contested-incident benchmark

Phases 31–34 established that **typed** handoffs + a **static**
role-subscription table suffice whenever the auditor's decoder can
pick the right answer from a fixed-priority rule over the
delivered bundle. Phase 35 identifies the smallest task family
where that precondition fails — *contested* incidents where two
plausible root-cause claims arrive with inverted static priority
— and ships a minimal primitive that recovers correctness while
preserving the Phase-31 bounded-context guarantee.

* **Escalation threads (``core/dynamic_comm``).** A typed,
  frozen-membership, explicitly-terminated coordination object
  with bounded per-member reply budget and bounded witness-token
  cap. The thread's only public output is a single
  ``CLAIM_THREAD_RESOLUTION`` handoff routed through the
  unchanged Phase-31 ``HandoffRouter``; thread-internal events are
  hash-chained in the existing log but never enter non-member
  inboxes (Theorem P35-4, no-leak invariant).
* **Contested-incident benchmark (``tasks/contested_incident``).**
  6-scenario bank: 4 contested root-cause pairs (deadlock vs
  shadow cron, TLS vs disk shadow, cron vs OOM shadow, DNS vs
  TLS shadow) + 2 controls. Under the mock auditor across
  k ∈ {6, 20, 60, 120} × 2 seeds × 4 strategies = 192
  measurements:

| k | strategy | full_acc | contested_acc | mean_tok |
|---:|---|---:|---:|---:|
| 6   | naive          | 0.333 | 0.000 |   591 |
| 6   | static_handoff | 0.333 | 0.000 |   215 |
| 6   | **dynamic**    | **1.000** | **1.000** |   **246** |
| 120 | naive          | 0.167 | 0.250 | 2 950 |
| 120 | static_handoff | 0.333 | 0.000 |   215 |
| 120 | **dynamic**    | **1.000** | **1.000** |   **246** |

  Dynamic coordination is **flat at 246 tokens / 100 % accuracy
  on every k**; static is **flat at 215 tokens but capped at
  33 % full / 50 % root-cause** (Theorem P35-1 separation).
  Messaging budget: exactly one 3-member thread per contested
  scenario, ≤ 2 replies of ≤ 12 witness tokens. Real-LLM spot
  check under ``qwen2.5:0.5b`` at k=6 seed=35 reproduces the
  separation: dynamic root-cause 1.00 vs static 0.50 (+50 pp).

* **Theorems.** P35-1 (expressivity separation between static
  handoffs and dynamic coordination — a pigeonhole argument on
  priority orderings), P35-2 (bounded-context preservation with
  additive ``T·R_max·W`` per role per round — independent of
  |X|), P35-3 (correctness under sound producer-local causality
  extraction), P35-4 (no-leak invariant). Conjectures C35-5
  (bounded threads ≡ bounded adaptive subscriptions in decoder
  correctness), C35-6 (dynamic coordination is necessary, not
  only sufficient — predicts an information-theoretic lower-
  bound dual of P35-1).

See ``vision_mvp/RESULTS_PHASE35.md`` for theorems + benchmark
tables.

Reproduce:
```bash
# Mock sweep (~0.2 s)
python3 -m vision_mvp.experiments.phase35_contested_incident \
    --mock --distractor-counts 6 20 60 120 --seeds 35 36 \
    --out vision_mvp/results_phase35_mock.json

# Real-LLM spot check (~70 s for 24 calls on qwen2.5:0.5b)
python3 -m vision_mvp.experiments.phase35_contested_incident \
    --model qwen2.5:0.5b --distractor-counts 6 --seeds 35 \
    --out vision_mvp/results_phase35_llm_0p5b.json
```

### Phase 34 — structured noise, adversarial noise, and honest ensemble extractors

Phase 34 closes the three medium-term frontier items Phase 33
surfaced: per-role-adaptive calibration (§ 4.11 bullet h),
adversarial extractor noise (§ 4.11 bullet i), ensemble extractor
composition (§ 4.11 bullet j).

* **Per-role calibration (Part A).**
  ``core/extractor_calibration.per_role_audit_summary`` + an audit-
  fit ``PerRoleNoiseConfig`` + ``per_role_noisy_extractor``. The
  Phase-34 mock benchmark across three domains shows max per-role
  drop-rate spread ≥ 0.33 (incident), 0.50 (compliance), 0.67
  (security) — the Phase-33 C33-3 pattern reproduces on every
  domain. Theorem P34-1: ``A_real ≤ Π_k (1 − δ_k) ≤
  (1 − δ̄)^{R*}`` by AM-GM; the pooled replay over-estimates
  accuracy whenever per-role noise is heterogeneous.
* **Adversarial noise (Part B).**
  ``core/extractor_noise.adversarial_extractor`` with three target
  modes (load-bearing drop with priority ordering, role silencing,
  severity escalation). At matched nominal budget δ·R*, the
  targeted-drop adversary collapses substrate accuracy to **0 %** at
  budget = 1 on all three domains while matched i.i.d. preserves
  20 %–80 % — pooled gap **+0.47 pp** (Theorem P34-2). Severity
  escalation confirms the Theorem-P33-3 precision-to-severity
  failure mode on the max-ordinal security decoder.
* **Ensemble extractor (Part C).**
  ``core/ensemble_extractor.UnionExtractor`` on a compliance *mixed*
  bank (5 canonical + 5 narrative; regex cannot parse narrative
  phrasings and narrative-LLM cannot match canonical). Regex alone
  **50 %**, LLM alone **0 %**, ensemble **100 %** at pooled
  δ_u = 0.00 ≤ δ_r · δ_l = 0.188 — Conjecture C33-4 promoted to
  empirical bound (Theorem P34-3).

See ``vision_mvp/RESULTS_PHASE34.md`` for theorems + benchmark tables.

Reproduce:
```bash
python -m vision_mvp.experiments.phase34_per_role_calibration \
    --mode mock --domains incident compliance security \
    --sweep-path vision_mvp/results_phase32_noise_sweep.json \
    --out vision_mvp/results_phase34_per_role_calibration_mock.json

python -m vision_mvp.experiments.phase34_adversarial_noise \
    --domains incident compliance security --seeds 34 35 \
    --drop-budgets 1 2 3 \
    --out vision_mvp/results_phase34_adversarial_noise.json

python -m vision_mvp.experiments.phase34_ensemble_extractor \
    --seeds 34 35 --distractor-counts 6 \
    --out vision_mvp/results_phase34_ensemble_extractor.json
```

### Phase 33 — LLM-driven extractors, real-noise calibration, and a third non-code domain

The Phase-31/32 substrate evaluated with *regex-perfect* extractors
— clean but unrealistic. Phase 33 closes that gap: an LLM-driven
extractor (``core/llm_extractor``) is a drop-in replacement for any
Phase-31/32 regex extractor, and a calibration layer
(``core/extractor_calibration``) measures its empirical noise
profile (δ̂ drop, ε̂ spurious, μ̂ mislabel, π̂ payload-corrupt)
against gold causal chains and maps it to the closest Phase-32
synthetic sweep grid point.

Headline on ``qwen2.5:0.5b`` against the compliance-review bank
(k = 6, seed = 33, 40 LLM calls, 91 s wall):

| metric | real 0.5b LLM extractor | Phase-32 closest (drop=0.5 sp=0.1 mis=0.25) | gap |
|---|---:|---:|---:|
| drop rate (δ̂)      | **0.70** | 0.50 | +0.20 |
| spurious per event (ε̂) | **0.12** | 0.10 | +0.02 |
| mislabel rate (μ̂) | **0.40** | 0.25 | +0.15 |
| substrate accuracy  | 0.00 | 0.00 | 0.00 |
| handoff recall      | **0.50** | 0.60 | −0.10 |
| handoff precision   | **0.27** | 0.21 | +0.06 |
| **verdict**         | | | **approximates** (γ = 0.10) |

The Phase-32 synthetic i.i.d. Bernoulli sweep **approximates** the
real LLM extractor's pooled noise profile on compliance — max-abs
gap of 0.10 on the recall axis. Per-role noise is highly
heterogeneous (legal 50 % drop, finance 100 % drop) — the pooled
match hides structure that Conjecture C33-3 names explicitly.

A **third non-code domain** — security-audit escalation
(``tasks/security_escalation``) — with a *max-ordinal severity +
claim-set classification* decoder (structurally distinct from both
prior domains) confirms the substrate signature at K = 3:

| k | strategy | tokens | acc (mock) |
|---:|---|---:|---:|
| 6   | naive          |   767 | 100 % |
| 6   | routing        |   151 |   0 % |
| 6   | **substrate**  | **242** | **100 %** |
| 120 | naive          | 4 216 |  20 % |
| 120 | **substrate**  | **242** | **100 %** |

Three domains, three decoder shapes (Phase 31 priority-order /
Phase 32 monotone-verdict + strict-flags / Phase 33 max-ordinal
severity + claim-set), one substrate module unchanged. Theorems
P33-1 (LLM-extractor subsumption), P33-2 (cross-domain
correctness at K = 3), P33-3 (two-regime bound on max-ordinal
decoder); Conjectures C33-3 (per-role heterogeneity), C33-4
(ensemble composition). See
``vision_mvp/RESULTS_PHASE33.md`` for theorems + benchmark tables.

Reproduce:
```bash
# LLM-extractor benchmark across all three domains (mock — instant)
python3 -m vision_mvp.experiments.phase33_llm_extractor --mode mock \
    --distractor-counts 6 20 60 120 --seeds 33 34 \
    --sweep-path vision_mvp/results_phase32_noise_sweep.json \
    --out vision_mvp/results_phase33_llm_extractor_mock.json

# Real 0.5b LLM calibration on compliance (~90 s)
python3 -m vision_mvp.experiments.phase33_llm_extractor --mode real \
    --model qwen2.5:0.5b --domains compliance --seeds 33 \
    --sweep-path vision_mvp/results_phase32_noise_sweep.json \
    --out vision_mvp/results_phase33_llm_extractor_0p5b.json

# Third-domain substrate benchmark (mock — instant)
python3 -m vision_mvp.experiments.phase33_security_escalation --mock \
    --distractor-counts 6 20 60 120 --seeds 33 34 \
    --out vision_mvp/results_phase33_security_mock.json
```

Writeup: `vision_mvp/RESULTS_PHASE33.md`.

### Phase 32 — cross-domain substrate, noisy-extractor robustness, frontier-model spot check

A second non-code domain confirms the Phase-31 substrate isn't
specific to operational telemetry. Five role-typed agents (legal,
security, privacy, finance, compliance officer) review a vendor
onboarding request — compound compliance issues (missing DPA,
uncapped liability, cross-border transfer without SCCs, weak
encryption, budget breach) require cross-role handoffs to resolve
correctly:

| k | strategy | mean tokens | accuracy (mock) | accuracy (7B) |
|---:|---|---:|---:|---:|
| 6   | naive          |   658 | 100 % |   0 % |
| 6   | routing        |   132 |   0 % | — |
| 6   | **substrate**  | **171** | **100 %** | **100 %** |
| 6   | **substrate_wrap** | 204 |   100 % | **100 %** |
| 120 | naive          | 4 047 |  40 % | — |
| 120 | **substrate**  | **171** | **100 %** | — |

Same substrate module (``core/role_handoff``) unchanged — the
*only* change is a new scenario catalogue, extractor set, and
decoder. Substrate flat at **171 tokens** across four orders of
magnitude of document-stream size; matches Phase-31's 196-token
flatline on a structurally distinct domain. On ``qwen2.5-coder:7b``
the substrate path **saturates the mock ceiling at 100 %** on
compliance review (vs naive's 0 %); on incident triage the 7B
reaches 80 % via ``substrate_wrap`` vs naive's 0 %. **Theorem
P32-1** formalises cross-domain correctness preservation;
**Theorem P32-2** formalises graceful degradation under extractor
noise (two regimes, closing Conjecture C31-7 in the monotone
case); **Theorem P32-3** shows the token bound survives bounded
noise as long as inbox capacity absorbs the spurious blow-up. The
Phase-32 noise sweep (``core/extractor_noise``, 96 noise points ×
2 domains, 0.5 s wall) confirms all three empirically, with a new
first-class failure attribution (``spurious_claim``) on the flag
side of the histogram.

Reproduce:
```bash
# Part A — cross-domain benchmark
python3 -m vision_mvp.experiments.phase32_compliance_review --mock \
    --distractor-counts 6 20 60 120 --seeds 32 33 \
    --out vision_mvp/results_phase32_compliance_mock.json

# Part B — noisy-extractor sweep (both domains)
python3 -m vision_mvp.experiments.phase32_noise_sweep --domain both \
    --drop-probs 0.0 0.1 0.25 0.5 --spurious-probs 0.0 0.05 0.1 \
    --mislabel-probs 0.0 0.25 --seeds 31 32 \
    --out vision_mvp/results_phase32_noise_sweep.json

# Part C — qwen2.5-coder:7b spot check on both non-code benchmarks
python3 -m vision_mvp.experiments.phase32_stronger_model \
    --model qwen2.5-coder:7b --seeds 32 --distractor-counts 6 \
    --out vision_mvp/results_phase32_llm_7b_spot.json
```

Writeup: `vision_mvp/RESULTS_PHASE32.md`.

### Phase 31 — typed handoffs and the first non-code multi-role benchmark

The same substrate, without code in sight. Five role-typed agents
(monitor, DBA, sysadmin, network, auditor) investigate a cascading
outage on a simulated fleet — each role owns a different slice of
telemetry. Under a deterministic five-scenario catalogue with a
distractor-density sweep (k ∈ {6, 20, 60, 120} per role), the
substrate path delivers:

| k | strategy | mean tokens | accuracy (mock) | accuracy (0.5b) |
|---:|---|---:|---:|---:|
| 6   | naive          |   574 | 100 % | 0 % |
| 6   | routing        |   147 |   0 % | 0 % |
| 6   | **substrate**  | **196** | **100 %** | **40 %** |
| 120 | naive          | 2 925 |  20 % | — |
| 120 | **substrate**  | **196** | **100 %** | — |

Substrate prompt size is **flat at 196 tokens** across four orders
of magnitude of event-stream size. Role-keyed routing alone cannot
rescue the auditor role — its concerns are content-level, so
header-filtering delivers nothing. The new substrate primitive is
``vision_mvp/core/role_handoff.py``: typed, content-addressed,
hash-chained handoffs between roles, with a role-subscription
table at claim-kind granularity. Theorems P31-1..P31-5 formalise
the separation; P31-5 is the formal statement behind the
master-plan distinction from graph/index tools.

Reproduce:
```bash
python3 -m vision_mvp.experiments.phase31_incident_triage --mock \
    --distractor-counts 6 20 60 120 --seeds 31 32 \
    --out vision_mvp/results_phase31_mock.json
```

Writeup: `vision_mvp/RESULTS_PHASE31.md`.

### Phase 30 — substrate vs naive full-context, with a real LLM on the answer path

First LLM-in-loop external-validity result. `qwen2.5:0.5b` via
local Ollama answers 20 SWE-style queries on the Python stdlib
`json` module under three delivery strategies:

| strategy | mean prompt tokens | accuracy | wall s/call |
|---|---:|---:|---:|
| naive full-context | 2 615 | 20 % | 19.97 |
| role routing       | 2 554 | 10 % |  0.86 |
| **substrate_wrap** | **163** | **80 %** |  **1.22** |

That's **16.0×** fewer tokens **and** **+60 percentage points** of
accuracy. Same harness, same model, same corpus — the only variable
is delivery strategy. See `vision_mvp/RESULTS_PHASE30.md` for the
theorem set, external causal-relevance numbers on `click` and
`json-stdlib`, and the reproduction commands.

Reproduce:
```bash
python -m vision_mvp.experiments.phase30_llm_swe_benchmark \
    --model qwen2.5:0.5b --corpora json-stdlib \
    --out vision_mvp/results_phase30_json_stdlib.json
```

The harness `vision_mvp/tasks/swe_loop_harness.py` takes any
`Callable[[str], str]`, so swapping in a frontier-model API or a
SWE-bench task generator is a drop-in replacement.

---

## Why this works

The repo separates the context substrate into five distinct layers,
each with its own loss profile:

| Layer | What it bounds | Loss profile | Where in code |
|---|---|---|---|
| **Routing** | who-talks-to-whom | lossy — projections / sparse selection are intentional | `core/api.CASRRouter`, `core/agent_keys`, `core/sparse_router`, `core/hierarchical_router` |
| **Trigger** | when to refine | lossy — drops drafts that look "agreed" | `core/trigger`, `core/general_trigger`, `core/event_trigger`, `core/behavior_trigger` |
| **Exact external memory** | what content is preserved | **lossless** — content-addressed Merkle DAG | `core/merkle_dag`, `core/context_ledger` |
| **Retrieval** | what content reaches the prompt | lossy in *ranking*; never lossy in *content* | `core/retrieval_store` (dense), `core/lexical_index` (BM25), hybrid RRF in `core/context_ledger.search(mode=)` |
| **Computation / planning** | how aggregations and joins answer without LLM-in-the-loop | **lossless and deterministic** — typed operators over handles | `core/exact_ops`, `core/query_planner`, `core/code_planner` |
| **Render** | whether the planner's exact answer is wrapped by the LLM or returned verbatim | **lossless on the direct-exact path** (zero LLM, zero prompt) | `experiments/phase22_codebase.run_direct_exact` |

The full project's research-first framing, arc structure, and
long-running open problems live in
[docs/context_zero_master_plan.md](docs/context_zero_master_plan.md).
That document is the durable reference; results notes below are
the per-phase empirical record.

Phases 1–18 built the routing and trigger layers (CASR + hybrid-structural
trigger). Phase 19 introduced the exact-memory + retrieval layers as a
lossless context substrate. Phase 20 strengthened retrieval with hybrid
BM25+dense and structural multi-hop expansion. Phase 21 added the
computation layer — a small natural-language → operator planner that
answers aggregation queries deterministically without an LLM in the
inner loop. Phase 22 generalised the substrate to real Python codebases
(AST-derived typed metadata via `core/code_index`) and added the
direct-exact render path that bypasses the LLM entirely when the planner
has the answer. Phase 23 extends Phase 22 to multi-codebase external
validity: direct-exact holds at 65 / 65 (100 %, σ = 0) across six real
Python corpora (vision-mvp modules / tests / experiments, the `click`
third-party CLI framework, and the stdlib `json` module), with a
reusable `CorpusRegistry` and a coverage-accounting ingestion pass
(`IngestionStats`). Phase 24 extends the exact slice from syntactic
structure to conservative *intraprocedural* static-semantic properties —
per-function predicates for `may_raise`, `is_recursive`,
`may_write_global`, `calls_subprocess`/`filesystem`/`network`, computed
from the AST via `core/code_semantics`. Direct-exact holds at **44 / 44
(100 %, σ = 0)** across the same six corpora on a 44-question semantic
battery with zero LLM calls. **Phase 25 extends the exact slice further
to conservative *interprocedural* semantic properties** — transitive
closures of the Phase-24 predicates over a local call graph plus exact
SCC-based recursion-cycle detection, computed by a new
`core/code_interproc` module that runs as a corpus-wide post-pass.
Direct-exact scores **50 / 50 (100 %, σ = 0)** on the Phase-25
interprocedural battery across the same six corpora, zero LLM calls,
zero prompt chars; retrieval-multihop averages 38.0 % (σ = 23.1) with
every failure attributed to `retrieval_miss`. The widening is dramatic
per corpus — on `click`, intra `may_raise = 46` becomes trans
`may_raise = 96` (+50 functions recovered); on `vision-core`, mutual
recursion is detected over a 19-function SCC. **Phase 26 introduces a
separate truth axis — *runtime calibration of the conservative
analyzer* — via instrumented probes over an executable snippet corpus.
On 21 snippets × 6 runtime-decidable predicates (126 applicable
measurements), the analyzer agrees with runtime-observed truth on
123 / 126 (97.6 %). The three divergences are one false-positive
(`may_raise` on `if False: raise` — the analyzer is control-flow-
insensitive by design) and two false-negatives (`calls_subprocess` via
`eval`, `calls_filesystem` via `getattr` — reflection holes explicitly
documented as Phase-24 boundaries). Every divergence lands on a pre-
documented boundary condition; the direct-exact planner round-trip
still matches the analyzer at 126 / 126 (100 %), confirming the
substrate's `render_error = 0` guarantee is independent of analyzer
runtime calibration.** **Phase 27 extends the runtime-calibration
axis from the curated 21-snippet corpus to *real corpus functions*.
On `vision-core` (~791 functions), the Phase-24/25 analyzer emits
flags for every function; only ~35.7 % are *runtime-calibratable*
under the Phase-27 default invocation-recipe strategy (ready_no_args
+ ready_typed + ready_curated). The gap — $|F_R| / |F_A| \approx
0.36$ — is the formal research finding: runtime truth at corpus
scale is **witness-availability-bounded, not planner-exactness-
bounded** (Theorem P27-1). On the entered subset, analyzer and
runtime agree on the overwhelming majority of predicates;
divergences concentrate exactly on the Phase-24 pre-documented
boundary classes (Conjecture P27-4). Planner-vs-analyzer round-trip
remains at 100 % on every predicate across every corpus — the
Phase-22 substrate guarantee is independent of Phase-27 coverage
(Theorem P27-2).** **Phase 28 extends runtime calibration from
one primary corpus (`vision-core`) to four local corpora
(`vision-core` / `vision-tasks` / `vision-tests` /
`vision-experiments`) in a single benchmark, and makes the
explicit-vs-implicit raise distinction first-class in both the
analyzer and the runtime observer. On the pooled entered slice
(306 observations across 2 140 functions), `may_raise_explicit`
is sound (FN = 0) with 98.7 % agreement; the new
`may_raise_implicit` predicate is essentially sound (FN = 1 /
116 on runtime-positives, ≈ 0.9 %) and over-approximating
by design. The Phase-24 `may_raise` contract and every Phase-22..27
substrate guarantee are preserved byte-for-byte. Coverage is
reported as a first-class cross-corpus variable — `ready_fraction`
ranges from **2.9 %** (`vision-tests`, 97 % methods) to **80.2 %**
(`vision-experiments`, 80 % typed top-level) — while
analyzer/runtime agreement on the other five predicates stays at
100 % on every corpus's entered slice (Theorem P28-2, P28-3,
P28-4).**

**Phase 29 runs the first task-scale falsifiability check of the
core routing/substrate thesis on a realistic SWE-style multi-role
task distribution drawn from the same four corpora, and closes the
Phase-27/28 method-coverage gap with a conservative
instance-auto-constructor. On 80 queries / 5 718 events across four
corpora, the pooled aggregator-role *causal*-relevance fraction
under naive broadcast is **4.54 %** (σ ≈ 0.002 across 5 seeds) —
strictly below the ROADMAP-specified 50 % confirmation gate. The
substrate matches 95 % of tasks and collapses aggregator context
from 13 849 → 13.75 tokens (**1 007×**) at **100 %** answer
correctness. Role-level routing alone reduces non-aggregator
context 1.3×–1 154× but leaves the aggregator untouched, confirming
that routing-by-type cannot resolve content-level aggregation
(Theorem P29-2). In parallel, the Phase-29 method-instance recipe
promotes methods on safely-zero-arg-constructable classes
(inherited-`object.__init__` / all-defaulted init / dataclass-all-
defaults) to a new `ready_method` status, lifting runtime
`ready_fraction` on `vision-tests` from **2.9 %** (Phase 28) to
**98.8 %** (Phase 29) and pooled entered slice 4.83× (306 → 1 477)
with `may_raise_explicit` FN preserved at 0 pooled and
construct-failed rate < 1 % (Theorem P29-5). Pooled falsifiability
decision: **CONFIRMED** (Theorem P29-1; full eight-theorem set in
`RESULTS_PHASE29.md`).**

**Phase 30 closes the theoretical–empirical bridge and runs the
programme's first LLM-in-loop external-validity benchmark. Four
theorems (P30-1..P30-4) formalise minimum-sufficient context
`T_i*`, connect it to the Phase-29 causal-relevance fraction,
and close one special case of OQ-1 (fixed-point convergence) in
the matched-substrate regime with a unique one-step fixed point.
Two conjectures (P30-5, P30-6) give OQ-1 its first concrete
mathematical shape under a stochastic answer path. The benchmark
runs a real local Ollama model on real external Python corpora:
on the Python stdlib `json` module under `qwen2.5:0.5b`, the
substrate path delivers a **16.0×** prompt-token reduction and
a **+60 percentage-point** answer-accuracy lift (80 % vs 20 %)
over naive full-context delivery. Substrate-matched slice
accuracy is **78.9 %** on 0.5b — bounded below by the model's
transcription fidelity, not by any substrate guarantee. Routing
alone *does not* rescue this model (10 % — confirming Phase-29
Theorem P29-2 on a live LLM). External corpora (`click`,
`json-stdlib`) land in the same causal-relevance band as the
internal four-corpus set (0.047–0.122 vs 0.032–0.056),
supporting Conjecture P30-5. The harness
(`vision_mvp/tasks/swe_loop_harness.py`,
`vision_mvp/experiments/phase30_llm_swe_benchmark.py`) accepts
any `Callable[[str], str]`; a future SWE-bench driver is a
drop-in replacement. See `vision_mvp/RESULTS_PHASE30.md`.**

See
`vision_mvp/RESULTS_PHASE19.md`, `RESULTS_PHASE20.md`,
`RESULTS_PHASE21.md`, `RESULTS_PHASE22.md`, `RESULTS_PHASE23.md`,
`RESULTS_PHASE24.md`, `RESULTS_PHASE25.md`, `RESULTS_PHASE26.md`,
`RESULTS_PHASE27.md`, `RESULTS_PHASE28.md`, `RESULTS_PHASE29.md`,
and `RESULTS_PHASE30.md`.

Five ideas stacked at the routing/trigger layers (see `VISION_MILLIONS.md`
for the full vision of 10):

1. **Shared Latent Manifold** — every agent projects to, and reads from, a
   small shared subspace instead of talking to every other agent directly.
2. **Streaming PCA** — the subspace is learned from observations, no oracle.
3. **Global Workspace** — only the ⌈log₂ N⌉ most-surprised agents get to
   write each round.
4. **Neural-net predictor per agent** — agents predict their own next state;
   surprise = prediction error (the "world model" of each agent).
5. **Decaying shared register** — the manifold forgets old evidence at an
   exponential rate, so drift is tracked without a sliding window.

The math behind why O(log N) is the right bound, not O(N), is derived from
72 independent mathematical frameworks in `EXTENDED_MATH_[1-7].md`:
Information Bottleneck, Kolmogorov cascade, gauge theory, spin glasses,
expander graphs, holographic entropy, TQFT, … — they all converge on the
same scaling law.

---

## Honest caveats

- **Low intrinsic rank assumption.** These results hold when the task's
  relevant structure has effective rank ≤ O(log N). Fully-general tasks
  with dim-d complexity need Ω(d) bandwidth by Theorem 11 in `PROOFS.md`.
- **LLM experiments were run at N=10.** The numpy experiments go to
  N=100 000 but the bridge (N=100 real LLMs) was out of scope for the
  initial pass — that's the obvious next thing to check.
- **Protocol is synchronous.** Async variants are straightforward in
  principle (CRDT semantics are already commutative) but not yet built.
- **Not yet peer-reviewed.** If you're a referee or reviewer, please dig
  in and break things.

---

## Project status

This is one continuous research push (Apr 2026) producing:
- a 72-framework theoretical survey (EXTENDED_MATH_[1-7].md)
- 12 formal theorems (PROOFS.md)
- 5 experiment phases from pure NumPy to local LLMs
- 94 passing unit + integration tests
- a clean public API (CASRRouter)

If it holds up under wider scrutiny, O(log N) coordination is the kind of
foundational result that would sit next to Shannon's channel capacity — a
statement about the minimum communication required for a class of
distributed problems. The way to find out is to throw it at more problems,
and to invite many pairs of eyes to check the math. That is what this
repository exists for.

---

## License

MIT.
