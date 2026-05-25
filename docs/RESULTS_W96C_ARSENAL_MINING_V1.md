# W96-C arsenal-mining inventory (pre-pilot, NIM-free)

> 2026-05-24.  Documentation-only inventory of reusable
> mechanisms in the CoordPy repo that could attack the
> structural failure mode W96-A Phase 3 empirically identified:
> *the W95-B0 math_solver / reflexion turns are blind to the
> image, and 90B-Vision's unified A1 K=5 climbs into the
> residual on problems where the vlm_reader's text extraction
> is lossy.*  No NIM calls.  Goal: ensure W96-C is hypothesis-
> and arsenal-driven, not vibes-driven.

## Candidates surfaced by the mining pass

| Mechanism | Module | Vintage | Relevance | Selected for W96-C? |
|---|---|---|---|---|
| **VLM-Verifier role** | `coordpy.cross_modal_role_specialized_bench_v1` (W92) | 2026-05-23 | Direct template; verifier reads (image, prompt, candidate, executor stderr) → structured critique.  Same K=5 model-call budget shape. | **YES — C1 template** |
| **VLM-in-loop (image every turn)** | `coordpy.cross_modal_vlm_loop_bench_v1` (W90) | 2026-05-22 | Sequential VLM turns with image in context every turn.  K=5 budget.  Most direct attack: every solver turn sees the image. | Considered; rejected for V2 because the W95-B0 vlm_reader / math_solver decomposition already pays its 1-VLM-call cost and the V2 selection rule preserves W95-B0 wins; a full VLM-in-loop would not preserve those |
| **Tool-augmented solving** | `coordpy.tool_call_substrate_v1` (W84) | 2026-05-19 | `PythonExecSandboxToolV1` with wall-time cap.  Tool calls would need explicit budget accounting to stay K=5; attacks arithmetic correctness, not image access. | Considered; ranked **lower** as **C2** because W96-A failure mode is image-access loss, not arithmetic loss |
| **Multi-modal payload carry-state** | `coordpy.multi_modal_payload_v1`, `coordpy.vision_substrate_v1` (W87) | 2026-05-18 | Patch-level VLM embeddings; would let solver re-index into image without re-encoding. | **Infeasible** — requires VLM hidden-state access; the NIM HTTPS API does not expose it |
| **Adversarial repair / consensus** | `coordpy.adversarial_consensus_repair_v1` (W81) | 2026-05-18 | Trust-weighted consensus + abstain/escalate.  Useful when multiple parallel attempts exist. | Considered; rejected because V2's solver chain is sequential (executor-guided reflexion); no parallel branches |
| **Multi-round / bundle decoder** | `coordpy.capsule_decoder_v2`, `coordpy.capsule_bundle_decoder` | foundational | Compact information transfer between agents. | Considered; rejected because the V2 verifier prompt already carries the extraction + candidates + verdicts in plaintext within the token budget |
| **Memory / state graph** | `coordpy.manifold_memory`, `coordpy.event_sourced_memory_graph_v1` | various | Bounded ring buffers + content-addressed event graph; could carry per-turn state across solver turns. | Considered; rejected — adds machinery not load-bearing for the W96-A-identified mechanism (image access, not memory) |
| **Repair-integrity composition** | `coordpy.compose_repair_integrity_pipeline_v1` (W83) | 2026-05-18 | Substrate-restore → consensus repair → integrity fusion → Merkle anchor. | Considered; useful for audit-chain extensions but not load-bearing for the C1 mechanism |

## Selected lead candidate — C1 (VLM-Verifier-Final-Turn)

C1 reuses:

* `_b_reader_prompt` from `mathvista_bench_v1` (W95-B0 vision-
  reader prompt; verbatim).
* `_b_solver_initial_prompt` from `mathvista_bench_v1` (W95-B0
  solver prompt; verbatim).
* `_b_solver_reflexion_prompt` from `mathvista_bench_v1` (W95-B0
  reflexion prompt; verbatim).
* `_run_a0_text` and `_run_a1_vlm` from `mathvista_bench_v1`
  (W95-B0 baselines; verbatim — A0 and A1 are byte-identical
  to V1).
* `MathVistaArmCallCapsuleV1` and `MathVistaArmOutcomeCapsuleV1`
  from `mathvista_bench_v1` (W95-B0 capsule shapes; verbatim —
  same audit-chain code applies).
* `MathVistaSeedReportV1` from `mathvista_bench_v1`.
* `extract_candidate_answer_v1` from `mathvista_bench_v1`
  (anti-cheat extractor; verbatim — same extractor for every
  arm).
* `evaluate_answer_v1` from `mathvista_executor_v1` (W95 truth
  function; verbatim).
* `select_mathvista_subset_v1` from `mathvista_loader_v1`
  (deterministic slice; verbatim).

C1 adds:

* `_b_vlm_verifier_final_prompt_v2` — a new prompt template
  modelled on W92's `_VLM_VERIFIER_SYSTEM` but tailored to
  MathVista (single final answer, no critique-then-implementer
  pattern; the verifier IS the final answerer because MathVista
  answer space is small).
* `_run_b_vlm_team_v2` — the V2 B arm runner with:
  - 1 vlm_reader call (T=0.0, sees image)
  - 3 math_solver/reflexion calls (text-only, T=temperature)
  - 1 vlm_verifier_final call (T=0.0, sees image + extraction
    + 3 prior candidates + executor verdicts)
  - selection: first text-only PASS short-circuits → ship;
    else if verifier PASS → ship verifier; else ship verifier
    answer as best-effort image-grounded last guess.

## Rejected candidates and why

### Why not C-loop (full VLM-in-loop on MathVista)

The W90 VLM-in-loop pattern keeps the image in every solver
turn (K=5 sequential VLM calls with reflexion).  On
HumanEval-Visual at code, W90 lost to A1_vlm K=5 by ~−1 to
−7 pp (W93 failure-cluster diagnosis).  At MathVista, the
benchmark has a much larger A1 failure-residual (~25-30 pp),
so the W90 pattern might in principle win.  However, the
selection logic of W90 (ship first PASS) and the absence of an
explicit extraction step mean the W90 architecture is closer
to A1 i.i.d. K=5 than the W95-B0 team mechanism.  It would not
preserve the W95-B0 wins that the vlm_reader + math_solver
decomposition specifically produces (which the W96-A Phase 3
evidence shows are 30 problems at 90B and 44 at 11B per 300
pairs).  C1's selection rule explicitly preserves those wins
and only uses the verifier to rescue A1-only territory; this
is the dominant strategy when both wins and rescues need to
be preserved.

### Why not C2 (tool-augmented solver) as the lead

C2 would inject Python exec tools into the solver chain so the
solver can compute numeric / symbolic results explicitly
(`sympy.solve`, `eval`, etc.).  This addresses the failure
mode where the extraction is correct but the math is wrong.
The W96-A evidence does not identify this as the dominant
failure mode:

* B-only rescues at 90B: 30 problems — these are cases where
  W95-B0's text-only solver chain correctly computed an answer
  the unified VLM K=5 sampling could not.  Tool augmentation
  would not change this count materially (the solver is
  already getting these right).
* A1-only rescues at 90B: 45 problems — these are cases where
  the unified VLM K=5 succeeded but W95-B0's chain failed.
  The dominant failure mode for these is the math_solver's
  inability to see the image directly (the extraction missed
  something the unified VLM caught).  Tool augmentation cannot
  recover lost extraction information.

C2 stays as a follow-up candidate if C1 succeeds and a residual
remains in the arithmetic-error class.  It is documented in
`COO-19` but not implemented in this milestone.

### Why not C3 (vision substrate + multi-modal payload)

These W87 modules (`vision_substrate_v1`, `multi_modal_payload_v1`)
require VLM hidden-state access — they read the vision tower
output, projector output, and LLM hidden state at per-patch
granularity, and pass them as compact embeddings.  This is a
genuinely stronger mechanism for cross-modal carry-state, but
it requires Hugging Face transformers loading of the VLM
weights locally.  The W96-C path is NIM HTTPS API only (we do
not have local 11B / 90B Llama-3.2-Vision weights at this
session's surface), so C3 is infeasible at the W96-C scale.
It is a documented future direction if the project moves to a
local-runtime substrate (cf. W79 `controlled_runtime_substrate_v1`
+ `local_openai_compatible_facade_v1` infrastructure that
already supports local execution).

## Cheap-probe predictions (mined NIM-free from W96-A Phase 3)

The W96-C preflight (`scripts/run_w96c_mathvista_preflight.py`)
will mine the W96-A Phase 3 and W95 Phase 3 sidecars to compute
two probes:

* **Q4** — the share of W95-B0 passes whose 4th text-only
  solver turn produced a NEW candidate (different from the
  first 3).  This is the conservative upper bound on V2's
  worst-case loss.  Pre-pilot expectation: < 30 % (most W95-B0
  passes complete within the first 1-2 solver turns; the 4th
  turn is mostly retry-of-equivalent-text).
* **Q5** — the A1-only rescue pool share (problems where A1
  PASS and W95-B0 FAIL).  Pre-pilot expectation at 90B:
  ~15 % (45 / 300); at 11B: ~11 % (33 / 300).

If Q4 < 30 % and Q5 > 10 % at 90B → C1 has structural room to
win the cross-scale Phase 2 even with a modest verifier
rescue rate.  If Q4 > 50 % or Q5 < 5 % → C1's prior is weak;
the cheap pilot evidence will be the discriminator.

## Honest framing

This is a *mining inventory*, not a benchmark result.  The
selected C1 candidate has not been validated empirically;
the W96-C preflight + Phase 2 pilot is what will decide.  The
arsenal-mining pass is a discipline mechanism (the user's
instruction "use more of the arsenal, not less"), recorded so
the W96-C C1 selection can be audited against the alternatives
that were considered and rejected.  The Linear `COO-19` issue
already identified C1 and C2 as the two refinements; the
arsenal mining confirms C1 as the lead and rejects the
substrate-level mechanisms (C3) as infeasible at the W96-C
scale.
