# RUNBOOK — W124: transformer-native code-intervention line on matched ICPC + hosted translation only if earned

> Pre-commit contract. Locked BEFORE any expensive (NIM) call and BEFORE the
> decisive local probe is interpreted. Fill `docs/RESULTS_W124_*` ONLY from the
> emitted verdict JSON (see the "never pre-write results" discipline). `COO-9`
> remains the lead path unless the evidence forces a different code-line move.
> Stable boundary: `coordpy.__version__ == "0.5.20"`, `SDK_VERSION ==
> "coordpy.sdk.v3.43"`, no PyPI, `coordpy/__init__.py` untouched, advanced work
> explicit-import-only. `ultracode` stays OFF (single-line bounded mechanism
> search, not a repo-wide dynamic-workflow job).

## 0. Why W124 is NOT more battlefield

W120 (resistant ICPC, +0.00 FAIL), W121 (matched exposed ICPC, +3.33 FAIL),
W122 (3-seed paired closure → terminal B4, unresolvable at n=30), W123 (large-n
supply census → ≥100/field matched battlefield is SUPPLY-UNREACHABLE, resistant
hard-capped ~45 tier-1). **The matched ICPC battlefield line is saturated.** The
honest aggressive move is a *genuinely different mechanism line* using the repo's
unused transformer/substrate arsenal (AST-aware code reads, hidden-state
intercepts, learned projectors, tool-call substrate, learned memory), tested
LOCALLY on the already-built matched ICPC family first, and translated to hosted
Maverick **only if it earns the right**. W124 is NOT a new benchmark family, NOT
an n=30/n=100 supply memo, NOT a bounded-context / compaction / summarization
trick (those remain anti-patterns).

## 1. α/β/γ branch logic (pre-committed)

- **Lane α (local transformer-native code intervention — MAIN empirical lane):**
  build a LOCAL mechanism bench FIRST (not a hosted run first) using the real
  repo transformer/substrate stack (`transformers_runtime_v1`,
  `code_substrate_v1`, `hidden_state_intercept_bench_v1`,
  `cross_runtime_hidden_state_projector_v1`). Lock the mechanism slate (§2)
  BEFORE results. A candidate SURVIVES α only if it shows a real mechanism gain
  over local A1 / local reflexion on the matched ICPC mini-bench (§3, §4).
- **Lane β (learned memory / controller line — mandatory):** evaluate whether
  `differentiable_memory_substrate_v1` / `composed_learned_memory_v1` can serve
  as a *controller* over code-trace / failure-trace state (patch-vs-replan-vs-
  abstain; rank candidate repairs; carry forward structured failure state) rather
  than raw token generation. NIM-free. Kill honestly if fake-relevant or too
  synthetic; define the smallest decisive probe if it looks real (§5).
- **Lane γ (hosted translation gate / graphify / truth — mandatory regardless of
  α/β):** re-check primary-source cutoffs (§6); decide hosted Maverick earn (§7);
  refresh graphify at START+END (§8); tighten the truth surface so the outcome is
  defensible.

## 2. Local code-model / transformer-native mechanism slate (LOCKED before results)

Hardware ground truth (this host, established by
`scripts/w124_env_feasibility_probe_v1.py`): the ONLY env with both torch and
transformers is `/Users/qdong/opt/anaconda3/bin/python` (py 3.9.13, torch
1.13.1, transformers 4.28.1, CPU/MPS, **no GPU**). The HF cache holds **no
code-fine-tuned model**; the repo's `TransformersRuntimeV1` default and only
locally-loadable real transformer is **`distilbert/distilgpt2`** (82M, general
LM, 6 blocks, hidden_dim 768). A real *code-competent* local solver is ABSENT.

Consequence (honesty rule, pre-committed): distilgpt2 CANNOT generate competent
ICPC solutions (expected ~0% solve), so the local lane CANNOT run a
"mechanism-beats-A1-in-solve-rate" comparison with a real generator. The
decisive LOCAL question is the **necessary-precursor** for the whole
transformer-native line:

> **M4 (AST-boundary hidden-state readout + probe).** Reading a real transformer's
> hidden state at AST function boundaries (`code_substrate_v1` over distilgpt2 via
> `transformers_runtime_v1`), is there a learnable hidden-state direction that
> distinguishes ACCEPTED from FAILED ICPC solutions on the matched family,
> **beyond surface features**? If NO usable signal exists on the available
> encoder, no hidden-state intervention can be built on this host (the line is
> blocked at the precursor), and M5 is moot.

- **M4 — AST-boundary hidden-state readout + probe.** Encoder = distilgpt2 real
  hidden states at `extract_function_boundaries_v1` boundaries (mean-pooled over
  boundary token positions, last block). Probe = L2-logistic, evaluated under
  **problem-disjoint grouped K-fold CV** (a problem's accepted+failed samples may
  NOT straddle train/test — else the probe learns problem identity, not
  correctness). Control = a surface-feature baseline probe (code char-len,
  line-count, #AST functions, #tokens, indentation depth) under the SAME CV.
- **M5 — learned hidden-state projector / repair steering.** GATED on M4: only if
  M4 yields a usable accepted-vs-failed direction do we fit a
  `cross_runtime_hidden_state_projector_v1` learned projector on the
  accepted−failed hidden-state contrast and measure whether the projected
  representation separates better than raw. Steering (hidden-state WRITE) is
  **local-only** (see §7 — NIM exposes no hidden state).
- **M6 — tool-call-substrate mediated code controller.** A DETERMINISTIC
  controller (`tool_call_substrate_v1` plane + `executor_grounded_patcher_v1`
  failure-digest reads) that routes patch-vs-replan-vs-abstain on parsed executor
  failure state, materially different from prose reflexion, NEVER reading the
  hidden test source. Built + contract-tested NIM-free. M6 is the ONLY slate
  member that is text-level and therefore hosted-translatable (§7).

A candidate is KILLED if it is fake-different (only reuses prompt text through a
new wrapper) or signal-poor (fails the §4 gate).

## 3. Matched ICPC mini-bench rule (LOCKED)

- Source = the ALREADY-BUILT official ICPC family only (no new benchmark family):
  the W120 resistant 30-slice (`coordpy_icpc_battlefield_v1`, 30-slice CID
  `01bf9ef8`) and the W121 matched exposed 30-slice (`coordpy_icpc_exposed_control_v1`,
  30-slice CID `32d15db5`).
- Labels = OFFICIAL grader only (`icpc_reflexion_bench_v1.grade_on_secret_v1`),
  re-derived deterministically on the official secret cases — model-free.
- Dataset = the REAL Maverick generations already on disk
  (`results/w120/icpc_pilot/...icpc_reflexion_calls.jsonl` resistant +
  `results/w121/exposed_pilot/...exposed_reflexion_calls.jsonl` exposed; 330 each
  = 660 matched-core generations), code recovered via `extract_candidate_code_v1`,
  each labelled accepted/failed by re-grading on its problem's secret cases.
  Problem attribution = statement-substring match (each prompt embeds exactly one
  cleaned statement). Deterministic manifest = sha256 over (slice CIDs, model_id,
  n_samples, label vector). W122 (1320 more) = optional held-out confirmation only
  if the core is decisive.

## 4. Local mechanism-gain gate (pre-committed, decides α survival)

Let `AUC_h` = mean held-out ROC-AUC of the M4 hidden-state probe under
problem-disjoint grouped CV; `AUC_s` = the surface-baseline probe AUC under the
SAME folds; chance = 0.50.

- **M4 SHOWS REAL SIGNAL** iff `AUC_h ≥ 0.60` AND `AUC_h − AUC_s ≥ +0.05`
  (hidden state beats both chance AND surface confound by a margin), on the
  matched core, replicated on resistant AND exposed sub-slices (sign-consistent).
- **M4 SIGNAL-POOR (expected, given an 82M general-LM encoder on hard ICPC)** iff
  the above fails ⇒ the transformer-native hidden-state line is **blocked at the
  precursor on this host**; the blocker is hardware/model-supply (no
  code-competent local encoder + transformers 4.28.1 too old for modern code
  models), NOT a refutation of the mechanism idea. M5 not run; M6 stays
  contract-only.
- A close blip (`AUC_h − AUC_s` in `[0.00, +0.05)` or `AUC_h < 0.60`) is NOT a
  gain and does NOT earn anything (W106 margin-cap discipline).

## 5. Learned-memory / controller earn-no-earn rule (Lane β, LOCKED)

- β earns a "real controller line" verdict ONLY if a learned-memory module
  (`differentiable_memory_substrate_v1` / `composed_learned_memory_v1`) trained on
  the ICPC failure-trace decision problem (features from the on-disk traces:
  did-A0-pass, did-A1-pass, reflexion-first-pass-index, failure-kind) predicts a
  USEFUL controller decision (e.g., will-reflexion-rescue) on held-out problems
  **better than a trivial majority/base-rate baseline by ≥ +0.05 balanced
  accuracy**, under problem-disjoint CV.
- Else β = **too-synthetic / not-relevant** — the learned-memory arsenal is
  trained on synthetic content-addressed-recall / long-horizon datasets and does
  not transfer to the n≈30-per-slice, class-imbalanced code-repair decision; kill
  honestly. Either way Lane β lands an executable probe + verdict JSON, not just
  prose.

## 6. Per-model disclosure-status & certification rule (Lane γ, unchanged gate)

Reuse `coordpy.stronger_model_cutoff_certification_v1` (C1∧C2∧C3∧C4; decision CID
`258b6ed7`, invariant). Re-check PRIMARY sources for: Maverick, Qwen3-Coder-480B,
DeepSeek-V4-pro, Mistral-Small-4-119B-2603, GLM-5, and any newly reachable
same-budget-comparable model.

- A model SUPERSEDES Maverick as the hosted target ONLY if it becomes
  primary-KNOWN (disclosed cutoff) AND certifiable on the matched ICPC family
  (resistant side needs a KNOWN cutoff ≤ ~Aug-2024; exposed side needs ≤
  2024-08-31). Standing prior: `{KNOWN:1 (Maverick Aug-2024), UNKNOWN:4}` ⇒
  Maverick is the only certifiable hosted target. γ is "gate closed + Maverick
  sole target" unless a disclosure genuinely flips.

## 7. Hosted-translation earn rule + spend rules (LOCKED)

**Categorical fact (pre-committed):** the hosted Maverick surface is the NIM
OpenAI-compatible **text** API (`nim_frontier_text_runtime_v1`); it exposes NO
hidden states. Therefore M4/M5 (hidden-state READ/WRITE interventions) are
**fundamentally local-only and CANNOT be translated to a hosted probe as
hidden-state interventions.** Only a text-level controller (M6) is
hosted-translatable.

Hosted Maverick spend is EARNED iff ALL hold:
1. a Lane-α candidate shows a real LOCAL gain per §4 (M4 real signal), AND
2. it yields a **text-translatable** controller (M6-shaped) whose decision rule
   is honestly reproducible over the NIM text API (no hidden-state dependence),
   AND
3. the hosted probe has real verdict-changing power on the matched ICPC family
   (could plausibly move B−A1 past the W120/W121 null band).

If any fails ⇒ **$0 NIM, no pilot.** Spend rules: local/runtime work first; no
new ICPC n=30 seed-chasing by default; no stronger-model spend unless §6 opens;
no 405B; a close local blip is NOT sufficient to translate. No reopening MBPP+
V2 / frozen cross-modal / closed Llama-3.1 rescue / APPS main-lane NIM. No dirty
exposed benchmark sold as a frontier win.

## 8. graphify deliverables (LOCKED)

- Refresh `graphify update .` at START (built from current HEAD; done — built
  from `1f10943d`) and END (after code/doc changes; must match the final commit).
- `graphify explain` on `code_substrate_v1`, `transformers_runtime_v1`,
  `hidden_state_intercept_bench_v1`, `cross_runtime_hidden_state_projector_v1`,
  `tool_call_substrate_v1`, `differentiable_memory_substrate_v1`,
  `composed_learned_memory_v1`, and the NEW W124 mechanism module entry point.
- `graphify path code_substrate_v1 transformers_runtime_v1` (confirm the 1-hop
  import edge the mechanism reuses) and `graphify affected` on the new module to
  confirm it is a leaf that does not couple back into the hosted pilot path.

## 9. W125 branch logic (pre-committed)

- If W124 Lane α is **M4 SIGNAL-POOR** (expected): W125 = either (a) accept the
  standing bounded HumanEval-family ceiling (W89+W105) AND the now-registered
  *local-encoder-supply* limitation, OR (b) fire only when a **code-competent
  local model becomes loadable** (a newer transformers + a small code model, or a
  reachable GPU) so the M4 precursor can be re-tested on a real code encoder, OR
  (c) a primary-KNOWN reachable stronger-than-Maverick model opens §6.
- If W124 Lane α is **M4 REAL SIGNAL** (surprise): W125 = build M5 steering / the
  M6 text-translatable controller and, if §7 clears, run the earned hosted
  Maverick probe; carry the verdict.
- If Lane β earns a **real controller line**: W125 may promote the controller
  probe to a larger ICPC trace corpus.
- `COO-9` stays the lead unless the evidence forces a different code-line move.
