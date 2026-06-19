# RESULTS — W124: transformer-native code-intervention line on matched ICPC (3 lanes)

**Date:** 2026-05-31 · **Lane α verdict:** `M4_CLOSE_BLIP_NOT_A_GAIN` (no transformer-native signal beyond surface) · **m4_cid** `2fc8fa9b2be6`
**Spend:** **$0 NIM** (no hosted probe earned). Decision CID `258b6ed7` invariant. No version bump, no PyPI, `coordpy/__init__.py` untouched. `ultracode` OFF throughout.

## Why W124 left the battlefield

W120–W123 saturated the matched-ICPC battlefield (resistant +0.00 FAIL / exposed +3.33 FAIL / 3-seed paired B4 unresolvable at n=30 / ≥100/field supply-UNREACHABLE). W124 stops treating the battlefield as the only lever and, for the **first time, runs the repo's unused transformer/substrate arsenal against the code-closure problem**: AST-aware code reads (`code_substrate_v1`), real hidden-state intercepts (`transformers_runtime_v1` / `hidden_state_intercept_bench_v1`), a learned projector (`cross_runtime_hidden_state_projector_v1`), the tool-call substrate (`tool_call_substrate_v1`) + executor-grounded failure digest (`executor_grounded_patcher_v1`), and the learned-memory line (`differentiable_memory_substrate_v1` / `composed_learned_memory_v1`).

## Hardware ground truth (the honest envelope)

The only env with torch **and** transformers is `/Users/qdong/opt/anaconda3/bin/python` (py 3.9.13, **torch 1.13.1, transformers 4.28.1**, CPU, torch default 1 thread, no GPU). The HF cache holds **no code-fine-tuned model**; the repo's `TransformersRuntimeV1` default + only locally-loadable real transformer is **`distilbert/distilgpt2`** (82M general LM). The repo's `transformers_runtime_v1` itself **does not load** under transformers 4.28.1 (it passes `attn_implementation=` to `from_pretrained`, added ~4.36), so the M4 encoder reads real distilgpt2 hidden state via a **direct HF load** while reusing `code_substrate_v1.extract_function_boundaries_v1` for the AST plane. A code-**competent** local solver is ABSENT — so the local lane runs the **necessary precursor**, not a solve-rate contest.

## Lane α — local transformer-native code intervention (MAIN)

**Question (M4):** reading a REAL transformer's hidden state at AST function boundaries, is there a learnable direction that separates ACCEPTED from FAILED ICPC code **beyond surface features**?

**Dataset (matched, official-grader-labelled, NIM-free):** 1,570 real Maverick generations recovered from the on-disk W120 resistant + W121 exposed + W122 paired-seed `*_reflexion_calls.jsonl`, labelled by the bench's own official `grade_on_secret_v1` verdicts (read from the per-problem reports; arm-order deterministic). 52 positives / 1,518 negatives over 60 problems. Negatives deterministically subsampled to 130/field (seed 124000) → **n=312 (52 pos, 60 problems)** for the CPU encode; AUC is balance-robust.

**Probe:** distilgpt2 AST-boundary hidden state (768-d) vs an 11-feature **surface baseline** (length, lines, indent, #AST funcs, parseability, …), L2-logistic, **problem-disjoint grouped CV**, pooled out-of-fold ROC-AUC.

| | AUC (hidden) | AUC (surface) | hidden − surface |
|---|---|---|---|
| **pooled** | **0.6345** | **0.6343** | **+0.0001** |
| resistant | 0.6991 | 0.7639 | **−0.0648** (surface beats hidden) |
| exposed | 0.5751 | 0.3204 | +0.2547 (both weak/noisy) |

**Gate (pre-committed §4):** real signal requires `AUC_h ≥ 0.60` **and** `AUC_h − AUC_s ≥ +0.05` **and** sign-consistency across slices. Margin is **+0.0001** (≪ +0.05) and the resistant slice has surface **beating** hidden ⇒ **`M4_CLOSE_BLIP_NOT_A_GAIN`** (`real_signal=False`). The modest ~0.63 separability is **fully explained by surface confounds** — the transformer-native read adds **nothing**.

**Consequence:** the transformer-native hidden-state line is **blocked at the precursor on this host**, and the blocker is **hardware/model-supply** (no code-competent local encoder; transformers 4.28.1 too old for modern code models and for the repo's own `transformers_runtime_v1`), **not** a refutation of the mechanism idea. **M5** (learned repair-steering projector) is **GATED OFF** (`NOT_RUN_M4_SIGNAL_POOR`) — and is in any case local-only (NIM exposes no hidden state). **M6** (deterministic tool-call-substrate code controller: typed patch/replan/abstain router over the executor failure digest, materially different from prose reflexion, never reads the hidden test) ships as an **executable contract** (routes verified `PATCH/REPLAN/PATCH/ABSTAIN`) but stays **contract-only** — no competent local generator to demonstrate a gain.

## Lane β — learned-memory / controller relevance

**Question:** is there a learned-controller line in `differentiable_memory_substrate_v1` / `composed_learned_memory_v1` actually relevant to ICPC code repair, or are those memory lines still too synthetic to matter here?

Decision problem (the only place a learned controller adds value over A1): predict reflexion-rescue of an A1-failed problem from pre-decision trace features. Across the same 6 runs: **139 A1-failed decisions, 14 rescue events (base rate 10.07 %)**; learned CV **balanced accuracy = 0.502** vs majority 0.500 (Δ +0.002 ≪ +0.05 earn margin). The learned-memory modules are trained on **synthetic** content-addressed-recall / long-horizon datasets — architecturally mismatched and data-starved for this n≈30-per-slice tabular decision. **Verdict: `TOO_SYNTHETIC_NOT_WARRANTED`.** $0.

## Lane γ — hosted translation gate / cutoff disclosure

`coordpy.stronger_model_cutoff_certification_v1` re-affirmed: **`NO_CERTIFIABLE_STRONGER_MODEL`**, decision CID **`258b6ed794b45a18…` invariant**, registry `{KNOWN:1, UNKNOWN:4}` (Maverick KNOWN Aug-2024 **certifiable-but-settled**; Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4-119B-2603 / GLM-5 UNKNOWN-from-primary). Maverick is the sole hosted target. **Hosted spend is moot for W124** regardless: Lane α earned no translatable gain ⇒ **$0 NIM, no pilot**.

## Hosted-translation earn rule outcome

NOT earned. The earn rule (RUNBOOK_W124 §7) requires (1) a real local M4 gain — **absent** (+0.0001) — **and** (2) a text-translatable controller — note that M4/M5 are hidden-state mechanisms the NIM **text** API cannot expose, so even a local gain would not translate as a hidden-state intervention; only M6 is text-level, and it has no demonstrated gain. **$0 NIM.**

## Carry-forward (unchanged retirements)

Exactly **TWO** confirmed retirements stand — **W89** (base HumanEval ×llama-3.3-70b, +5.56pp) and **W105** (HumanEval+ ×llama-3.3-70b, +7.00pp), both contamination-EXPOSED HumanEval-family at 70B. W124 **retires none and adds none**; it adds **limitation** carry-forwards (`W124-L-*`) localizing the transformer-native blocker to **local code-model-encoder supply** (the model-axis sibling of W123's post-cutoff battlefield-supply cap) and recording the learned-memory line as not-warranted. The arsenal was genuinely mined and tested — the honest result is a sharp, executable **negative**, not an unexamined "battlefield capped".

## Artifacts

- `coordpy/transformer_native_code_intervention_v1.py` (explicit-import-only; M4 encoder + probe + gate, M5 gated projector, M6 controller).
- `scripts/run_w124_lane_alpha_mechanism_bench_v1.py`, `scripts/run_w124_lane_beta_controller_probe_v1.py`, `scripts/run_w124_stronger_model_gate_recheck_v1.py`, `scripts/w124_env_feasibility_probe_v1.py`.
- `tests/test_w124_transformer_native_code_intervention_v1.py` (11 tests; falsifiability-first; validated by direct execution — local pytest/attrs env is broken).
- `results/w124/lane_alpha/w124_lane_alpha_verdict.json`, `results/w124/lane_beta/w124_lane_beta_verdict.json`, `results/w124/stronger_model_gate/gate_recheck_v1.json`.
