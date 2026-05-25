# W97 — Frontier relevance audit V1

> 2026-05-25.  Mandatory frontier-vs-stale classification of the
> CoordPy repo's mechanism inventory, performed at the start of
> W97 before any new expensive run.  Output: every load-bearing
> mechanism is honestly classified as (a) **active frontier
> arsenal**, (b) **useful baseline-only**, (c) **historical
> artifact**, (d) **dead direction (refuted)**, or
> (e) **anti-pattern (commodity context tricks; not core
> strategy)**.  Goal: stop the cross-modal lead path from
> drifting toward bounded-window / compaction / summary tricks
> as core strategy, and make explicit which mechanisms are still
> live for the RealWorldQA / cross-modal frontier.
>
> This audit does NOT rewrite history.  Modules that are
> baseline-only or historical artifacts are kept in-repo because
> they pass their own offline verifiers and continue to serve as
> falsifier targets or substrate-level evidence; they are simply
> *not the frontier path* for the cross-modal retirement
> programme.
>
> No code is removed by this audit.  No version bump.  No PyPI
> publish.  `coordpy.__version__` stays `0.5.20`;
> `coordpy.SDK_VERSION` stays `coordpy.sdk.v3.43`.

## Operating principle

CoordPy's frontier path is not "context management."  Bounded
windows, compaction, generic prose summarization, and shallow
prompt-massaging are commodity-LLM tricks — they exist in this
repo only as **baselines that the substrate-coupled methods are
supposed to BEAT**, not as the substrate-coupled methods
themselves.

The legitimate frontier arsenal is structurally different from
"cram less, truncate better."  It includes:

* explicit structural extraction (scene → graph; chart → table)
* geometry-aware representations
* tool-augmented reasoning with explicit budget accounting
* direct visual grounding across turns
* benchmark-specific decomposition with cheap-probe-tested
  hypothesis
* substrate-level hidden-state / cross-modal injection (when
  hardware permits)
* verifier turns when the mechanism is *empirically* load-bearing
  (the W96-C C1 case is a falsifier of "verifier is enough" at
  K=5 byte-exact)
* failure-cluster mining + sidecar-driven slice selection

The audit is mandatory for W97 specifically because the
RealWorldQA preflight passed at both 11B and 90B and the
*temptation* is to immediately commit to a W95-B0-shaped port.
The honest read is that scene-perception/reasoning entanglement
on RealWorldQA may punish the W95-B0 extract-then-reason shape;
the audit forces the architecture choice to come from arsenal
mining + structural argument, not from W95-B0 inertia.

## Active frontier arsenal (still credible attack surface)

These mechanisms are load-bearing for the cross-modal lead
path on RealWorldQA (or related W97+ work).  Inclusion here
means: *worth running on the next expensive bench if the cheap
evidence earns it*.

| Mechanism | Module(s) | Why still frontier |
|---|---|---|
| **W95-B0 scene-reader + text solver + executor-guided reflexion** | `coordpy/mathvista_bench_v1.py` | The preflight-earned baseline shape for W97 D2-B0.  W95 +3.67pp Phase 3 is empirically the strongest cross-modal team signal in the programme on a clean-executor benchmark.  Sub-retirement bar, but positive. |
| **Structured scene-graph extraction (D2-B1 candidate)** | NEW; lives in `coordpy/realworldqa_bench_v1.py` (W97) | The next structurally-motivated refinement: replace bullet-list free-text extraction with an explicit JSON schema for objects + positions + spatial relations + counts.  Forces lossless extraction.  Documented but not implemented in this milestone. |
| **Tool-augmented solver (C2 / arithmetic substrate)** | `coordpy/tool_call_substrate_v1.py`, `coordpy/code_substrate_v1.py` | Documented as W96-C C2 backup; addresses arithmetic-failure-mode orthogonal to extraction.  Less relevant for spatial-reasoning RealWorldQA, more relevant for arithmetic ChartQA-like benches if revived. |
| **Failure-cluster mining + sidecar-driven slice selection** | `coordpy/failure_cluster_miner_v1.py`, sidecars from W95 / W96-A / W96-C runs | Active load-bearing infrastructure for Phase 2 evidence reading + Phase 3 design.  Used to mine W95-B0 vs A1 disagreement structure. |
| **Cheap-preflight harness (W93 + W95 / W96-D composite)** | `coordpy/cross_modal_preflight_harness_v1.py`, `coordpy/mathvista_preflight_v1.py`, `coordpy/chartqa_preflight_v1.py`, `coordpy/realworldqa_preflight_v1.py` | Validated SIX consecutive times (W93/W94/W95/W96-A/W96-C/W96-D).  Active gate for every cross-modal candidate.  Pulls A1@K=5 saturation estimate, executor self-test, decomposition argument, and 5 W93 gates with $0 NIM spend. |
| **MathVista / ChartQA / RealWorldQA loaders + executors** | `coordpy/mathvista_loader_v1.py`, `coordpy/mathvista_executor_v1.py`, `coordpy/chartqa_loader_v1.py`, `coordpy/chartqa_executor_v1.py`, `coordpy/realworldqa_loader_v1.py`, `coordpy/realworldqa_executor_v1.py` | Canonical anti-cheat boundary for cross-modal benches.  Deterministic slice selectors + SHA-anchored parquet + no-judge executor (numeric / multi-choice / canonical text) per benchmark. |
| **NIM HTTPS chat-completions runtime** | `coordpy/nim_frontier_text_runtime_v1.py`, NIM client patterns in `scripts/run_w95_mathvista_pilot.py` (forked into W96-A/W96-C drivers) | The only path from the repo to live frontier-class models (Llama-3.2-Vision 11B / 90B; Llama-3.3-70B-Instruct).  Text-only at NIM; hidden-state work is substrate-side. |
| **Per-call sidecars + per-seed Merkle + bench Merkle audit chain** | `coordpy/mathvista_bench_v1.py` (capsule shapes), `scripts/verify_w95_mathvista_audit_chain.py` | Re-derives every bench verdict offline from sidecars.  Already 14/14 re-derivations confirmed on W95 Phase 3.  Anti-cheat infrastructure that survives across cross-modal benches. |
| **Linear ↔ GitHub durable bridge** | `linear_github_mapping.json`, `scripts/sync_linear_github_v1.py`, `docs/LINEAR_GITHUB_SYNC.md` | Validated bridge across 7 milestones (W89 / W93 / W94 / W95 / W96-A / W96-C / W96-D).  Re-runnable, idempotent. |
| **Vision substrate adapter + multi-modal payload (when GPU available)** | `coordpy/vision_substrate_v1.py`, `coordpy/multi_modal_payload_v1.py`, `coordpy/composed_multimodal_pipeline_v1.py` | Substrate-level cross-modal injection — the *hard* alternative when NIM HTTPS can no longer carry the cross-modal signal.  Currently blocked on NIM not exposing patch embeddings; load-bearing as documentation of the substrate-side path. |
| **Cross-scale Phase 2 rule (W96-C carry-over)** | `docs/RUNBOOK_W96C.md`, replicated in `docs/RUNBOOK_W96D.md` and W97 runbook | Active discipline mechanism: Phase 3 entitlement requires Phase 2 PASS at BOTH 11B AND 90B (or 90B with written justification).  Caught W96-A retroactively and W96-C prospectively. |

## Useful baselines (not the frontier path; kept honest)

These modules are correctly framed as **baselines that the
frontier path must beat**.  They are NOT the lead architecture
for any future cross-modal retirement attempt; they are the
falsifier-targets the substrate-coupled / structured-extraction
methods must demonstrate superiority over on regimes where their
bounded shape forecloses signal.

| Mechanism | Module(s) | Baseline status |
|---|---|---|
| **W79 V2 bounded-window k=64 + φ=0.20 summary** | `coordpy/bounded_window_baseline_v2.py` | Documented falsifier target for substrate-coupled long-horizon recovery (W79+).  Header explicitly labels itself "still bounded; not the frontier." |
| **W83 V3 bounded-window k=256 + dynamic rolling summary + cosine-retrieval** | `coordpy/bounded_window_baseline_v3.py` | The *strongest known bounded baseline* in the programme — explicitly built as the hardest falsifier target.  Module docstring says verbatim: "V3 is *still* bounded.  Substrate-coupled pipeline strictly beats V3 on regimes that require cross-window-boundary recall, multi-hop reconstruction, or horizons ≥ ~10k turns." |
| **W83 V1 bounded-window k=64 (legacy)** | `coordpy/bounded_window_baseline_v1.py` | Earliest bounded baseline; superseded by V2 and V3 but kept for regression coverage. |
| **A0 text-only / A1 unified-VLM K=5 baselines (per-bench)** | Bench modules (`mathvista_bench_v1`, `chartqa_bench` future, `realworldqa_bench` future) | A0 / A1 are NOT the candidate; they are the same-budget comparison surface that B must beat.  Useful baselines; not the frontier mechanism. |
| **Generic chat-completions text driver** | `coordpy/nim_frontier_text_runtime_v1.py` | The NIM HTTPS chat-completions path IS load-bearing infrastructure, but the *driver itself* is a commodity wrapper, not a research contribution.  Documented honestly in the module docstring as a frontier-class *text-only oracle*, not a substrate.  Useful tool; not the frontier mechanism. |

## Historical artifacts (kept for regression / audit; not active path)

These modules captured real evidence from prior milestones and
remain re-verifiable, but are *not* part of the active W97
plan.  Removing them would erase recoverable evidence;
demoting them keeps history honest.

| Mechanism | Module(s) | Reason it is no longer the active path |
|---|---|---|
| **W90 VLM-in-loop cross-modal bench** | `coordpy/cross_modal_vlm_loop_bench_v1.py` | LOST by −1..−7 pp on HumanEval-Visual K=5.  W93 failure-cluster diagnosis showed the pattern is closer to A1 i.i.d. K=5 than a structurally distinct team.  Kept as a refuted baseline. |
| **W92 role-specialized cross-modal bench (VLM-Planner + Code-Implementer + VLM-Verifier)** | `coordpy/cross_modal_role_specialized_bench_v1.py` | LOST by −10.71 pp on HumanEval-Visual K=5.  Three independent cross-modal architectures (split / VLM-in-loop / role-specialized) all decisively LOSE at K=5 on HumanEval-Visual.  Kept as evidence of the "HumanEval-Visual K=5 is the wrong battlefield" conclusion. |
| **W88 cross-modal code split (VLM-extract + code-LM)** | `coordpy/cross_modal_code_bench_v1.py` | LOST by −5.56 pp; first cross-modal team-architecture decisive negative.  Kept as historical reference. |
| **Earlier attention_steering_bridge V1–V13** | `coordpy/attention_steering_bridge_v{1..13}.py` | Superseded by V14+ family or by later substrate-coupled work.  Kept for regression / audit. |
| **W81 adversarial consensus repair** | `coordpy/adversarial_consensus_repair_v1.py` | Useful only with parallel candidate solvers; W95-B0 / W96-C / W97 D2 use sequential solver chains where adversarial consensus has no parallel branches.  Kept as substrate-side infrastructure for parallel-branch revivals. |
| **W83 long-horizon reconstruction substrate** | `coordpy/long_horizon_reconstruction_substrate_v{1,2}.py`, `coordpy/composed_long_horizon_multi_agent_recovery_v1.py` | Active in W78–W83 substrate work; outside the cross-modal K=5 surface.  Kept for substrate-side claims when local-runtime work resumes. |
| **W84 tool-call substrate** | `coordpy/tool_call_substrate_v1.py` | Documented as W96-C C2 backup; not active for W97 D2-B0 lead but kept for the arithmetic-failure-mode revival path. |

## Dead directions (refuted by the evidence; do not entertain as core strategy)

These mechanisms have empirical evidence against them on the
cross-modal retirement programme.  Do NOT pick them as the lead
candidate for any future expensive run.  Re-introducing one as
a lead requires NEW positive cheap-probe evidence that the
prior negative does not transfer.

| Mechanism | Evidence against |
|---|---|
| **Cross-modal team superiority on HumanEval-Visual K=5** | W88 / W90 / W91 / W92: three independent architecture families lose by −5.56 to −10.71 pp.  W94 explicitly retired HumanEval-Visual K=5 as a cross-modal battlefield. |
| **Scaling the VLM weight class on MathVista to retire** | W96-A 90B Phase 3: B−A1 = −5.00 pp.  Cross-scale shift from 11B to 90B = −8.67 pp.  Scaling the VLM HURTS the team's relative advantage on MathVista. |
| **VLM-Verifier-Final-Turn as a load-bearing rescue mechanism (W96-C C1)** | 11B Phase 2: verifier rescue rate 0/11 = 0.0 %; 90B Phase 2: verifier rescue rate 1/7 = 14.3 % (the 90B PASS is variance-driven, not mechanism-driven).  Mechanism is empirically NOT load-bearing at K=5 byte-exact. |
| **ChartQA as a clean-executor cross-modal battlefield for Llama-3.2-Vision K=5** | W96-D D1: A1@K=5 saturation 91.69 % at 11B and 92.75 % at 90B; residual 8.31 / 7.25 pp — far below the W95 +20 pp floor.  Battlefield killed at $0 NIM spend.  Reviving ChartQA requires either a non-saturated model family or a different K-budget shape. |
| **K=10 reflexion as the rescue lever (W93-C / W94 hypothesis)** | W94: A1 first-pass-among-K=10 saturates to 100 % ceiling on the 15-problem slice, leaving zero failure-residual for reflexion.  Killed in 90-min cheap pilot. |
| **i.i.d. wider 11B sampling as a substitute for cross-scale evidence** | W96-A's cross-scale shift on B−A1 already rules out the variance hypothesis at any wider 11B sample.  COO-18 de-prioritised. |

## Anti-patterns (NEVER promote as core strategy; baseline-only allowed)

These are commodity-LLM tricks that the CoordPy programme has
*explicitly rejected* as the lead path.  They may appear in-repo
as **baselines** (bounded_window_baseline_v{1,2,3}) — that is
healthy, because the substrate-coupled methods must beat them.
They MUST NOT become the cross-modal team-architecture's core
mechanism.

| Anti-pattern | Why it is not the frontier path |
|---|---|
| **Bounded context window as a product thesis** | The substrate-coupled work in W78–W83 was specifically built to *beat* the strongest known bounded baseline (W83 V3 = k=256 + dynamic rolling summary + cosine retrieval).  Reviving "bounded context" as the CoordPy thesis would erase that work and concede the field to commodity LLMs. |
| **Compaction / generic prose summarization as a "memory" mechanism** | Refuted as core strategy by W78–W83.  The W83 V3 falsifier explicitly composes "dynamic rolling summary" — and the substrate-coupled pipeline STRICTLY BEATS it.  Reviving compaction as the lead would be a regression. |
| **Shallow token compression without a real structural reason** | Anti-pattern.  Compression is acceptable when the *structure* is exploited (e.g., chart → table, scene → graph).  Compression as a generic prompt-massage is not a research contribution. |
| **Context-pruning theater** | Pruning that does not preserve structural signal is the same anti-pattern as shallow compression. |
| **"Cram less / truncate better" presented as a frontier memory system** | Explicitly rejected by the user's W97 instruction.  Bounded baselines are kept as falsifier targets only. |
| **Generic LLM-as-judge anywhere in the executor chain** | W88 anti-cheat clause.  Every cross-modal bench's executor is rule-based (numeric / multi-choice / canonical text).  No LLM judge is allowed in the executor truth surface. |
| **Selective retries to inflate pass rates** | W88 anti-cheat clause.  Same K=5 budget every arm; no retries on FAIL. |
| **Single-seed pilots used as retirement-grade evidence** | W95 / W96-A both demonstrated single-seed +10 pp narrowing to retirement-grade ≤ +5 pp.  Pilots are *cheap probes*, not retirement evidence. |
| **Architecture refinement chosen by vibe rather than arsenal mining** | W93 / W96-C / W96-D / W97 all require explicit arsenal-mining + structural argument before NIM spend. |

## Specific Linear / repo mirror

* `COO-6` (parent backlog) — already accurately reflects
  active frontier vs done vs deferred per the 2026-05-25 update.
  W97 frontier-audit appends here.
* `COO-12` (substrate-level cross-modal injection) — correctly
  classified as the *hard alternative* (Low priority).  Stays as
  documentation of the substrate-side path while NIM remains the
  primary cross-modal runtime.
* `COO-15` (GSM8K resolution at stronger scale) — Low; still
  unresolved.  Not the W97 lead path.
* `COO-14` (slice-selection helpers) — Medium; generalisation of
  W96-C Q4/Q5 probes.  Useful tooling, not frontier.
* `COO-9` (second code benchmark) — High but blocked behind the
  current cross-modal lead.  Promote when the RealWorldQA line
  is decided (PASS / FAIL Phase 2).
* `COO-18` (wider 11B sample on MathVista) — de-prioritised by
  W96-A's cross-scale negative.

## What this audit changes operationally

Nothing in-repo is deleted.  The classification is the
deliverable.  Operationally, it means:

1. **W97 D2 lead must be picked from the active frontier
   arsenal** — D2-B0 (W95-B0 scene-port) or D2-B1 (structured
   scene-graph extraction).  No bounded / compaction / summary
   shortcut.
2. **The W97 runbook MUST explicitly state which arsenal
   mechanism is load-bearing for the chosen candidate** —
   structural argument, not vibes.
3. **Refuted directions (W90 / W92 / verifier-rescue / VLM
   scaling / ChartQA) MUST NOT be revived as the W97 lead
   without new positive cheap-probe evidence.**
4. **Bounded baselines stay in-repo as falsifier targets
   only** — never promote them to the lead path or to the
   cross-modal team architecture.
5. **Linear COO-20 + COO-6 + the new W97 issue MUST quote this
   audit** so the Linear-side state is in sync with the audit.

## Honest scope of this audit

* This is a *classification*, not a benchmark.  No empirical
  claim is made here; every claim is sourced to a prior
  milestone's `RESULTS_*.md` doc.
* The audit does NOT close any carry-forward.  All W89 / W95 /
  W96-A / W96-C / W96-D carry-forwards remain active.
* The audit does NOT bump `coordpy.__version__` or
  `SDK_VERSION`; no PyPI publish.
* The audit lives in `docs/` because it is part of the
  programme's honesty surface; the Linear comment mirror is the
  team-facing surface.

## What this audit DOES NOT do

* It does NOT claim multi-agent context is solved.
* It does NOT claim cross-modal team superiority on RealWorldQA
  (no Phase 2 evidence yet).
* It does NOT retire any prior carry-forward.
* It does NOT erase historical / refuted modules; they remain
  in-repo as evidence.
* It does NOT propose new code in this milestone beyond
  documenting the classification; the W97 bench module + pilot
  script + runbook follow in the same milestone but are
  separate deliverables.
