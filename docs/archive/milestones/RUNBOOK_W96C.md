# W96-C — MathVista C1 (VLM-Verifier-Final-Turn) runbook

> **Pre-commit contract for W96-C C1, locked 2026-05-24 BEFORE
> any NIM call against the W96-C V2 candidate.**  W96-A Phase 3
> at Llama-3.2-90B-Vision decisively failed retirement with
> B − A1 = −5.00 pp on the same 3 × 100 slice set as W95 Phase 3
> at 11B (+3.67 pp).  The cross-scale shift on B − A1 is
> −8.67 pp; the W95-B0 math_solver / reflexion turns are blind
> to the image, and 90B's unified-VLM K=5 climbs into the
> residual on problems where the vlm_reader's text extraction
> is lossy (A1-only rescues 33 → 45 at 90B; B-only rescues
> 44 → 30).
>
> W96-C C1 replaces the **last** W95-B0 text-only reflexion turn
> with a VLM-Verifier-Final call (same K=5 budget byte-exact)
> that SEES the image, the structured extraction, and the
> 3 prior text-only candidates + executor verdicts.  Selection
> short-circuits to any text-only PASS (preserves W95-B0 wins),
> else ships the verifier's answer (rescues A1-only territory).
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`; `coordpy.SDK_VERSION` stays
> `coordpy.sdk.v3.43`.

## Linear

* `COO-19` (W96-C): Architecture refinement (verifier turn /
  tool-aug solver) Phase 2 pilot.
* Parent: `COO-6` (post-W96-A empirical frontier backlog).
* This runbook implements **C1** (VLM-Verifier-Final-Turn).
  **C2** (tool-augmented solver) is documented as a separate
  candidate that the arsenal-mining pass considered but ranked
  lower because the W96-A failure mode is image-access loss,
  not arithmetic loss; C2 attacks the latter but stays orthogonal
  to the former.  **C3** is not defined; arsenal mining did not
  surface a candidate that strictly dominates C1.

## Arsenal mining (locked 2026-05-24)

Before writing C1, the repo was searched for any reusable
mechanism that could attack the W96-A-identified failure mode
(vision → solver lossiness, K=5 budget).  The full mining
inventory is in `docs/RESULTS_W96C_ARSENAL_MINING_V1.md`; the
top candidates surfaced were:

| Mechanism | Module | Relevance to W96-A failure mode |
|---|---|---|
| VLM-Verifier role | `coordpy.cross_modal_role_specialized_bench_v1` (W92) | Direct template; verifier reads (image, signature, candidate, stderr) → critique |
| VLM-in-loop (image every turn) | `coordpy.cross_modal_vlm_loop_bench_v1` (W90) | Simplest pattern: keep image in context every solver turn |
| Tool-augmented solving | `coordpy.tool_call_substrate_v1` (W84) | Python exec sandbox; attacks arithmetic, not image-access |
| Multi-modal payload carry-state | `coordpy.multi_modal_payload_v1`, `coordpy.vision_substrate_v1` (W87) | Patch-level embeddings; requires hidden-state access (infeasible via NIM HTTPS) |
| Adversarial repair / consensus | `coordpy.adversarial_consensus_repair_v1` (W81) | Validation layer; useful only if there are multiple parallel solver attempts |

C1 was selected as the lead candidate because:

1. It is the simplest mechanism that *directly* attacks the
   W96-A-identified weakness (image access for the final
   reasoning step).
2. It stays in K=5 byte-exactly by re-allocating one text
   reflexion turn into a VLM call (no budget growth).
3. It reuses W92's VLM-Verifier prompt template + the W95-B0
   vlm_reader / math_solver primitives verbatim.
4. Its selection rule (text-only PASS short-circuits) means C1
   cannot regress on W95-B0 wins that complete within the
   first 3 solver turns — the worst-case downside is the share
   of W95-B0 passes whose load-bearing turn was the 4th solver
   turn (cheap-mined as Q4 in the W96-C preflight).
5. Substrate-level mechanisms (vision_substrate_v1,
   multi_modal_payload_v1) require model hidden-state access
   that the NIM HTTPS API does not expose; they are infeasible
   at the W96-C scale.

C2 (tool-augmented solver) was ranked lower because the W96-A
failure mode is image-access lossiness (A1-only rescues), not
arithmetic correctness (B-only failures with correct
extraction).  C2 would address a different failure mode and
could be a follow-up if C1 succeeds and a residual remains in
the arithmetic class.

## Hypothesis (locked 2026-05-24)

W96-A Phase 3 empirically identifies the W95-B0 architecture's
structural cap: the math_solver / reflexion turns are
**text-only**, so problems where the vlm_reader's bullet-list
extraction is lossy (small numbers, axis labels, color codes,
tick marks) cannot be recovered downstream.  At 90B-Vision, A1
climbs into exactly that residual: A1-only rescues rise from
33 (11B) to 45 (90B); B-only rescues fall from 44 to 30; B − A1
swings −8.67 pp uniformly across all 3 seeds.

W96-C C1 hypothesis: **giving the team's final turn direct
image access regains the rescue capacity that the W95-B0
text-only chain lost.**  The cheapest implementation is to
replace the 4th text-only solver turn with a VLM-Verifier-Final
call (1 VLM call, T=0.0) that re-reads the image with the
extraction + prior candidates + executor verdicts in context.

Two distinct outcomes are plausible:

* **H1 (C1 wins):** the VLM-Verifier rescues a meaningful share
  of the A1-only-rescue pool (≥ 30 % rescue rate × ≥ 10 % pool
  fraction → ≥ 3 pp net margin gain); the loss of the 4th
  text-only solver turn is small enough that net B_v2 − A1 ≥
  +5 pp on the W95 Phase 2 deterministic slice.
* **H2 (C1 ties or loses):** either the verifier fails to rescue
  enough A1-only problems (the residual is harder than expected
  even with image grounding), or the loss of the 4th text-only
  solver turn cancels the gain.  Net B_v2 − A1 < +5 pp on Phase
  2; C1 is killed cheaply.

**Pre-pilot prediction (locked BEFORE NIM):** "Genuinely
unknown but slightly H1-leaning at 11B (verifier's image access
is a structurally new capability the W95-B0 chain lacked) and
H2-leaning at 90B (the W96-A negative was uniformly negative
across seeds; the unified VLM at 90B may already absorb most of
the rescue room a verifier could exploit, making the verifier
agree with A1 most of the time)."  The cheap Phase 2 cross-scale
pilot decides.

## Baselines (locked 2026-05-24)

Identical to W95 / W96-A Phase 2 except the B arm is the W96-C
C1 V2 architecture (`coordpy.mathvista_bench_v2.B_vlm_team_v2`).

* **A0** — text-only mode (image=None) of the VLM at T=0.0,
  K=1.  Byte-identical to W95/W96-A A0.
* **A1** — unified VLM mode at T=0.7, K=5.  Byte-identical to
  W95/W96-A A1.
* **B0 (V1) — historical reference**, not run in this pilot.
* **B_vlm_team_v2 (W96-C C1)** — vlm_reader (1, T=0.0) +
  math_solver_initial (1, T=temp) + math_solver_reflexion×2
  (2, T=temp) + vlm_verifier_final (1, T=0.0) = 5 calls total.

## Pre-committed W96-C preflight gates

A W96-C NIM pilot launches IFF ALL of these hold.

### W95 P1..P4 + W93 G1..G5 composite (NIM-free)

`scripts/run_w96c_mathvista_preflight.py --candidate-model
meta/llama-3.2-{11b,90b}-vision-instruct` re-uses the W95
composite preflight with the V2-aware decomposition argument
(see `W96C_DECOMPOSITION_ARGUMENT_V2` in the preflight script).
Composite verdict must be `PASS`.

### W96-C-specific cheap probes (NIM-free; mine W95-B0 sidecars)

**Q4 — turn-4 upper-bound on V2's downside risk.**  The
preflight mines the W96-A Phase 3 + W95 Phase 3 sidecars to
estimate the share of W95-B0 passes whose 4th text-solver turn
produced a NEW candidate (different from the first 3).  This is
the conservative upper bound on the share of W95-B0 wins V2
could lose by removing the 4th turn.  Threshold: ≤ 50 % (loose
prior; the cheap pilot decides empirically).

**Q5 — A1-only-rescue pool size.**  The preflight mines the
same sidecars to count the share of (seed, problem) pairs where
A1 PASS and W95-B0 FAIL — the V2 verifier's potential upside
territory.  Threshold: ≥ 5 % (any meaningful pool size).

If composite + Q4 + Q5 all PASS → V2 is preflight-earned for a
NIM smoke test + Phase 2 cheap pilot.  If any fail → kill the
W96-C line and document the negative.

## NIM smoke test (only if preflight composite PASS)

1. One 1-token POST against the candidate model returns HTTP 200
   with a non-empty completion.  Recorded as the V2 sidecar's
   `kind=smoke_test` entry; does NOT count toward per-problem
   budget accounting.
2. Optional dry-run of the V2 bench on 1 problem with a real
   NIM call shape (1 A0 + 5 A1 + 5 B_v2 = 11 NIM calls; ~30-60s
   wall at 11B; ~60-120s at 90B).  Confirms the V2 wiring works
   against the live endpoint.

## Phase 2 — cheap NIM pilot (only if NIM smoke PASS)

### Pilot shape (locked 2026-05-24)

* **Bench**: `coordpy.mathvista_bench_v2` (V2; new B arm).
* **Model**: `meta/llama-3.2-{11b,90b}-vision-instruct` via NIM.
* **Slice**: **1 seed (95_005_001) × 30 problems** — *byte-
  identical* to W95 Phase 2 + W96-A Phase 2 so the
  V1-vs-V2 cross-architecture comparison stays problem-level
  fair (the W95-B0 V1 archive provides the head-to-head
  reference at the same 30 pids).
* **Budget**: K=5 calls per problem on A1 and B_v2; A0 = 1 call.
* **Expected cost**: 30 × 11 = 330 NIM calls / pilot; ~21 min
  wall at 11B (W95 Phase 2 wall), ~45-60 min at 90B (W96-A
  Phase 2 wall).

### Pre-committed Phase 2 pilot gates (W95 9-gate shape)

Phase 2 escalates to a Phase 3 pre-commit IFF ALL of these hold.
Gate texts are byte-identical to W95/W96-A Phase 2; the arm
under evaluation is `B_vlm_team_v2`.

1. **Slice pre-committed**: 30 problems by
   `select_mathvista_subset_v1(95_005_001, 30)` BEFORE any
   MathVista NIM call.  Slice SHA recorded.
2. **A1 < 90 %**: A1@K=5 pass rate on the 30-problem slice must
   stay below 90 %.
3. **B_v2 > A1**: `b_vlm_team_v2_pass_rate > a1_pass_rate`.
4. **Margin ≥ +5 pp**: `b_vlm_team_v2 − a1 ≥ 5.0 pp`.
5. **B_v2 > A0 by ≥ +5 pp**: image is load-bearing.
6. **Per-problem majority**: B_v2 ≥ A1 on ≥ 16 of 30 problems.
7. **Budget accounting exact**: per-problem 1 + 5 + 5 = 11.
8. **Audit chain re-derives**: bench + seed Merkle present.
9. **Executor stays clean**: executor invariants intact.

If any gate fails, **W96-C Phase 2 at the candidate scale is
KILLED** at that scale.  Cross-scale rule (see next section)
decides whether the candidate is killed overall or escalates to
Phase 3.

## Cross-scale rule (locked 2026-05-24 — POST-W96-A discipline)

W96-A taught that a +10 pp single-seed Phase 2 pilot can flip
to −5 pp at multi-seed retirement scale.  W96-A also showed
that the cross-scale shift on B − A1 was UNIFORMLY NEGATIVE
across the 11B → 90B step.  W96-C C1's Phase 2 entitles
Phase 3 IFF:

* **Either** Phase 2 PASS at *both* 11B and 90B (the strongest
  case: the architecture survives cross-scale, ready for Phase
  3 at the scale that produced the W96-A negative);
* **Or** Phase 2 PASS at 90B only, with a written justification
  for why 11B Phase 2 PASS is not required (the W96-A negative
  was at 90B; if C1 fixes 90B specifically, that is sufficient
  for the next Phase 3 at 90B).

A Phase 2 PASS at 11B only is **NOT** sufficient for Phase 3 at
90B: that would repeat the W96-A mistake of escalating from
single-scale single-seed pilot evidence.  In that case the C1
candidate gets a cross-scale Phase 2 at 90B to confirm or
falsify before any Phase 3 is launched.

If Phase 2 FAILS at the *first* tested scale (11B), the 90B
Phase 2 is **NOT** automatically run.  The author writes the
W96-C negative result doc and decides whether the 90B Phase 2
is informative enough to justify the additional ~330 NIM call
spend; this is recorded as a pre-commit decision before any
90B NIM call.

### Default cross-scale order

The pilot defaults to **11B first** because the 11B pilot is
cheaper per-call (~21 min wall vs ~60 min at 90B) and because
the W95-B0 V1 baseline at 11B is the longest-established
reference (+10 pp Phase 2; +3.67 pp Phase 3).  If 11B Phase 2
PASSES at +5 pp, the 90B Phase 2 is then run to confirm
cross-scale and either preflight-earn Phase 3 or kill the
candidate.

## Phase 3 — full bench (NOT in this milestone)

Phase 3 will require its own runbook section locked BEFORE any
Phase 3 NIM call, mirroring the W96-A Phase 3 section in
`docs/RUNBOOK_W96A.md`:

* 3 seeds × 100 problems × K=5 (same as W96-A).
* Same W88 6-bar retirement shape.
* Same audit-chain re-derivation requirement.
* Cross-scale: if both 11B and 90B Phase 2 PASS, Phase 3 runs
  at 90B (the scale where W96-A negative was observed); the
  11B Phase 3 is *not* required because W95 Phase 3 at 11B
  already provides the 3 × 100 reference for the same model
  family.

**No Phase 3 will be launched in this milestone.**  W96-C C1's
job in this milestone is to earn or kill the next expensive
run; the Phase 3 launch decision and runbook are explicitly
out of scope until Phase 2 evidence exists at both scales (or
at 90B with written justification).

## Anti-cheat (carry-forward from W88–W96-A)

All W88–W96-A anti-cheat clauses carry forward verbatim.
W96-C-specific additions:

* **Slice is byte-identical** to the W95 Phase 2 deterministic
  30-problem slice (`select_mathvista_subset_v1(95_005_001,
  30)`).  This makes the V1 vs V2 cross-architecture
  comparison problem-level fair: W95-B0 V1 already has 11B and
  90B passes at +10 pp on this slice; V2's per-problem outcomes
  can be compared directly to V1's archived outcomes for the
  same 30 pids.
* **Same VLM model on every arm** — A1, B-reader, B-verifier
  all use the candidate VLM.  Solver / reflexion turns use the
  same model in text-only mode (image=None).  No cross-family
  mixing.
* **Same K=5 budget byte-exact** — verifier always runs even
  when text-only PASSes early.  Budget gate enforced byte-
  exactly on every problem.
* **Same executor truth** — `evaluate_answer_v1` for every arm.
  No LLM judge.
* **Verifier prompt is locked** in the V2 bench module's
  `_b_vlm_verifier_final_prompt_v2` function.  Any change to
  the prompt requires a new V3 bench module (V2 stays frozen
  for audit).

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New module `coordpy.mathvista_bench_v2` is **explicit-import
  only**.  It does NOT enter the `coordpy/__init__.py` public
  surface (advanced work, not promoted to default API).
* New scripts `scripts/run_w96c_mathvista_{preflight,pilot}.py`
  are independent of W95 scripts.  W95 scripts are NOT modified.

## Operational plan

1. **(W96-C NIM-free preflight)**
   a. Run
      `scripts/run_w96c_mathvista_preflight.py
      --candidate-model meta/llama-3.2-11b-vision-instruct`
      and record the verdict to
      `results/w96/mathvista_preflight_c1/<RUN_ID>/`.
   b. Re-run with `--candidate-model
      meta/llama-3.2-90b-vision-instruct` for the 90B
      composite (no extra cost; same probes).
   c. Verify both composites + Q4 + Q5 PASS.
2. **(W96-C NIM smoke test — only if preflight passed)**
   a. One 1-token POST against each candidate model.
   b. Record smoke entry to the preflight run dir.
3. **(W96-C Phase 2 pilot at 11B — only if smoke PASS)**
   a. Launch `scripts/run_w96c_mathvista_pilot.py
      --vlm-model meta/llama-3.2-11b-vision-instruct
      --n-problems 30 --n-seeds 1 --out-dir
      results/w96/mathvista_c1_pilot`.
   b. Apply the 9 Phase 2 gates.
4. **(W96-C Phase 2 pilot at 90B — only if 11B PASS)**
   a. Launch `scripts/run_w96c_mathvista_pilot.py
      --vlm-model meta/llama-3.2-90b-vision-instruct
      --n-problems 30 --n-seeds 1 --out-dir
      results/w96/mathvista_c1_pilot`.
   b. Apply the 9 Phase 2 gates.
5. **(Cross-scale verdict)**
   a. If both PASS → write Phase 3 runbook section + commit;
      do NOT launch Phase 3 in this milestone.
   b. If only one PASS → record cross-scale verdict + decide
      Phase 3 scope per the cross-scale rule above.
   c. If neither PASS → document the C1 negative with
      `W96-L-MATHVISTA-V2-VERIFIER-FINAL-K5-PHASE2-CAP` carry-
      forward.  W96 advances to either W96-D (battlefield
      pivot) or W96-C C2 (tool-augmented solver) per the
      Linear-recommended ordering.
6. **(Linear ↔ GitHub sync)**
   a. Update `COO-19` with V2 verdict + run-dir CIDs.
   b. Update `COO-6` summary with the W96-C verdict.
   c. Append a `W96-C` entry to `linear_github_mapping.json`
      and run `scripts/sync_linear_github_v1.py`.

## Honest framing

W96-C C1's Phase 2 pilot is a 30-problem × 1-seed cheap probe
at each scale, locked to the same deterministic slice as
W95/W96-A Phase 2.  It cannot retire any carry-forward by
itself; it can only earn or kill the next expensive run, with
the post-W96-A cross-scale discipline.  A Phase 2 PASS at one
scale does NOT license a Phase 3 launch at that scale unless
the cross-scale check confirms the architecture survives the
scale step.  A Phase 2 FAIL kills the candidate at that scale
cheaply; the negative is preserved for the next architecture
candidate to learn from.
