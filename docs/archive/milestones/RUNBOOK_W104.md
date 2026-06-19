# W104 — HumanEval+ cross-scale confirmation at 405B + hardening + W105 planning (runbook)

> **Pre-commit contract for W104, locked 2026-05-26 BEFORE any
> W104 NIM call and BEFORE the W104 pilot driver is written.**
>
> W103 closed with the HumanEval+ cheap pilot at 70B
> `PASS_MECHANISM_DRIVEN` (B − A1 = +20.00 pp; 9/9 Phase 2 gates
> + MLB-1 56.67 % + MLB-2 47.06 % all PASS; matched the W89
> base-HumanEval retirement template's 47 % rescue rate byte-
> for-byte; helper-anchored slice CID
> `c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2`;
> bench Merkle `68f4a9669f1bd03e6b3cb393a436e4f04aca034a0bad9c4b5ea8a002faabfd6d`;
> verdict cid `4f57a2cf...` preflight reused).  Per **Branch A**
> of the pre-committed W103 RUNBOOK § Planning lane, W104 is
> entitled to launch a HumanEval+ cross-scale Phase 2 cheap
> pilot at a SECOND model class on the EXACT same helper-
> anchored slice.  `COO-9` REMAINS the lead path.
>
> W104 is NOT a one-step milestone.  It advances THREE lanes in
> the same milestone:
>
> 1. **Lead lane (cross-scale at 405B)** — execute the W103
>    helper-anchored 30-problem slice BYTE-FOR-BYTE unchanged on
>    `meta/llama-3.1-405b-instruct`; evaluate the SAME 9 Phase 2
>    gates + MLB-1 + MLB-2 sub-gates; emit a per-problem cross-
>    scale comparator against W103 70B.
> 2. **Hardening lane (cross-scale-discipline)** — codify the
>    durable guardrails that mattered empirically in W102/W103
>    (resume-after-socket-hang / 429-storm handling; mid-run
>    sidecar visibility; cross-scale comparator emitted
>    automatically; provenance schema/checks on the cross-scale
>    bench report).
> 3. **Planning lane (W105 entitlement OR fallback)** — pre-
>    commit BOTH the W105 Phase 3 retirement-bench slice pack
>    (3 seeds × 100 problems × K=5) AND the fallback code-line
>    ranking refresh, BEFORE the W104 verdict is in, so that
>    EITHER outcome leaves W105 trivially launchable.
>
> The pilot is CONDITIONAL on a sub-second reachability smoke
> test on the 405B endpoint.  The hardening lane and planning
> lane ship UNCONDITIONALLY.
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`; `coordpy.SDK_VERSION` stays
> `coordpy.sdk.v3.43`.  Advanced work remains explicit-import
> only.

## Linear

* New issue **`COO-28`** (W104): HumanEval+ cross-scale Phase 2
  cheap pilot at 405B + cross-scale-discipline hardening +
  W105 entitlement / fallback pre-commit.  Parent: `COO-6`.
  High priority.  Status: In Progress.
* Related: `COO-9` (lead path) — remains at High; W104 confirms
  the W103 EvalPlus-family attack at a second model class.
* Related: `COO-14` (helper) — same `coordpy.code_slice_selector_v1`
  consumption as W103; the W103 slice is reused BYTE-FOR-BYTE
  so no fresh helper invocation is required.
* Related: `COO-27` (W103; Done) — lead-lane infrastructure
  W104 consumes verbatim.
* Related: `COO-26` (W102; Done) — backup-lane bench module
  W104 consumes verbatim.

## What is NOT in scope (anti-drift contract)

W104 explicitly does NOT:

1. Re-open the cross-modal RealWorldQA arc.  RealWorldQA stays
   frozen at 11B per the W100 frontier audit.  W101 / W102 /
   W103 / W104 carry this verbatim.
2. Re-open the W95-B0 family, the typed-extract sub-family, or
   any RealWorldQA candidate.
3. Re-attempt MBPP+ V2 at 70B or 405B.  W102 FAIL is a cap;
   re-running on a fresh seed or scale would be hope-driven,
   not evidence-earned.
4. Promote `COO-12` (substrate-level cross-modal injection)
   absent fresh evidence; `COO-12` stays Low.
5. Build APPS / LiveCodeBench / SWE-bench-lite infrastructure.
   The W101 battlefield-selection matrix locked these out of
   scope; W104 inherits that decision verbatim.  Only Branch C
   (W104 FAIL) re-opens the LiveCodeBench preflight option,
   and even then only as the next preflight step, never as a
   running expensive bench.
6. Bump `coordpy.__version__` or `SDK_VERSION`.
7. Publish to PyPI.
8. Edit `coordpy/__init__.py`.  Any new W104 modules are
   explicit-import only.
9. Re-introduce any anti-pattern under a prettier name
   (bounded windowing; compaction; generic prose summarization;
   shallow token compression; context-pruning theater; "cram
   less / truncate better").  The W97 – W103 frontier-relevance
   audits stay in force verbatim.
10. Launch Phase 3 (3 seeds × 100 problems × K=5) at W104.
    Phase 3 is W105 only IF W104 PASSes Branch A, AND only
    once the W104 verdict doc is committed.
11. Re-instate `coordpy.mbpp_plus_loader_v1` / `..._executor_v1`
    / `..._reflexion_bench_v1` as live infrastructure.  These
    remain demoted to historical artifact + anti-pattern per
    the W102 audit.
12. Run a new candidate tournament.  The W101 battlefield-
    selection matrix and the W103 helper-anchored slice are
    BOTH carried forward unchanged.  W104 changes only the
    target model class.
13. Re-grade historical W88 / W91 / W103 responses against a
    new test surface and treat the result as cheap-pilot
    earning evidence.  Per the W102 anti-pattern carry-
    forward, that is an UPPER BOUND only.  Fresh-K=5 sampling
    on 405B is the W104 ground truth.

## Target-model selection rule (PRE-LOCKED BEFORE any NIM call)

The W104 lead-lane target model is selected by the following
deterministic rule, locked here verbatim BEFORE any NIM call:

1. **Reachability**: target must be a NIM-hosted chat-completion
   endpoint reachable from the W103 pilot driver shape unchanged
   (same `POST https://integrate.api.nvidia.com/v1/chat/completions`
   surface; same `Authorization: Bearer ${NVIDIA_API_KEY}`).
2. **Second-model-class legitimacy**: target must be a
   STRUCTURALLY DIFFERENT model class from `meta/llama-3.3-70b-
   instruct` along at least one of two axes:
   (a) parameter scale (≥ 3x larger or ≤ 1/3 smaller); OR
   (b) different model generation / fine-tuning lineage (e.g.,
   Llama 3.1 vs 3.3, or Nemotron variant).  Same-scale-same-
   generation does NOT count as cross-scale.
3. **Same-budget comparability**: target must accept the IDENTICAL
   K=5 sequential-reflexion mechanism via the unchanged
   `coordpy.humaneval_plus_reflexion_bench_v1.run_humaneval_
   plus_reflexion_bench_v1` entrypoint.  No prompt-format
   adaptation, no per-model special-casing, no budget tuning.
4. **Anti-saturation legitimacy** (cheap-pre-NIM probe): the
   predicted A1@K=5 on the W103 helper-anchored slice must be
   ≤ 80 % (clear of the 90 % saturation gate by ≥ 10 pp).
   Predicted from the published EvalPlus drop applied to a
   reasonable upper bound on the target's base-HumanEval
   pass@1, with the helper-anchored slice's historically-hard
   bias (63 % of slice is `b_only_wins` + `shared_fails` +
   `a1_only_wins`) shaving an additional ≥ 10 pp.
5. **Anti-pattern token absence**: target model name MUST NOT
   contain any forbidden token (`bounded_window`, `compaction`,
   `context_compaction`, `prose_summary`, `context_pruning`,
   `summarizer`).
6. **Practical runtime + cost**: target must fit within the
   cheap-pilot wall budget (≤ 4 h ceiling per W103 envelope)
   at the 330-call budget; expected per-call latency ≤ 30 s
   under nominal NIM throttling.

### Primary target (locked): `meta/llama-3.1-405b-instruct`

Applies the selection rule:

| Criterion | Verdict | Evidence |
|---|---|---|
| Reachability | PASS (pending W104 smoke probe) | NIM hosts `meta/llama-3.1-405b-instruct` on the standard `/v1/chat/completions` surface; same auth flow as the W103 70B target. |
| Cross-scale legitimacy | PASS | 405B is ~5.8x parameter scale vs 70B; both Llama-3.x family, both instruction-tuned; cross-scale-UP precedent matches the W96-A 11B→90B + W100 11B→90B patterns. |
| Same-budget comparability | PASS | Identical chat-completion API surface; identical `coordpy.humaneval_plus_reflexion_bench_v1` bench module; identical K=5 budget; identical prompt format (no model-specific adaptation). |
| Anti-saturation legitimacy | PASS (cheap-pre-NIM) | Llama 3.1 paper Table 7 lists 405B base-HumanEval pass@1 = 89.0; EvalPlus published min drop = 12.7 pp (Hoeffding lower bound) ⇒ 405B HumanEval+ A1@1 ≈ 76.3 % full corpus; pass@K=5 lift is bounded ≤ ~4-5 pp on this regime ⇒ 405B HumanEval+ A1@K=5 ≈ 80-81 % full corpus.  Helper-anchored slice (63 % historically-hard from arsenal mining) shaves an additional ≥ 10 pp ⇒ predicted slice A1@K=5 ≤ 71 %, clear of the 90 % gate by ≥ 19 pp.  Predicted MUCH cleaner than the 70B helper-anchored slice (W103 empirical A1 = 50 %; predicted full-corpus A1 = 72.86 %). |
| Anti-pattern token absence | PASS | `meta/llama-3.1-405b-instruct` contains no forbidden tokens. |
| Practical runtime + cost | EXPECTED PASS | 330 calls × 5-30 s/call = 30 min – 3 h wall; well within the W103 124 min envelope.  Cost ~5-15 USD on NIM rates (cheap pilot envelope). |

### Backup target (locked): `meta/llama-3.1-70b-instruct`

If the W104 reachability smoke probe FAILs on the primary 405B
target (HTTP 404 / 403 / model-not-found), the pre-locked
backup is `meta/llama-3.1-70b-instruct` (Llama 3.1 vs Llama
3.3 — cross-GENERATION at the same parameter scale).  This is
a WEAKER form of cross-scale (same parameter count) but
satisfies criterion 2(b) (different generation / fine-tuning
lineage).  The backup is used ONLY if 405B is unreachable; the
W104 verdict doc records WHICH target was used.

The backup is NOT a fallback to skip cross-scale evaluation.
If BOTH primary and backup are unreachable, W104 is paused at
the pre-NIM milestone and the runbook decision becomes "defer
W104 to next NIM-budget allocation"; this is recorded in
`docs/RESULTS_W104_HUMANEVAL_PLUS_PHASE2_405B_V1.md` with the
"DEFERRED on reachability" verdict.

## Operational state (cheap evidence in hand BEFORE W104 starts)

| Field | Value |
|---|---|
| `coordpy.__version__` | `0.5.20` |
| `coordpy.SDK_VERSION` | `coordpy.sdk.v3.43` |
| W103 70B Phase 2 verdict | **`PASS_MECHANISM_DRIVEN`**; B − A1 = +20.00 pp; MLB-2 = 47.06 % |
| W103 helper-anchored slice CID (helper-priority) | `c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2` |
| W103 helper-anchored slice CID (bench iteration order) | `d5364a2f5a6ab3d6febe69b99d8424f75a54ad6f1dbde9e5e8e2d7e62c9e3052` |
| W103 bench Merkle root | `68f4a9669f1bd03e6b3cb393a436e4f04aca034a0bad9c4b5ea8a002faabfd6d` |
| HumanEval+ corpus SHA-256 (LFS oid) | `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492` |
| W102 HumanEval+ preflight verdict cid | `4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4` (re-confirmed W103) |
| Llama 3.1 paper base-HumanEval 405B pass@1 (Hoeffding upper bound on full-corpus A1@1) | 89.0 % |
| EvalPlus published min base→plus drop (Hoeffding lower bound) | 12.7 pp |
| Predicted 405B HumanEval+ A1@K=5 (full corpus) | ≈ 80-81 % (saturation margin 9-10 pp on full corpus) |
| Predicted 405B HumanEval+ A1@K=5 (W103 helper-anchored slice; 63 % historically-hard) | ≤ 71 % (saturation margin ≥ 19 pp) |

## Critical W104 anti-pattern carry-forwards from W101 / W102 / W103

1. **`coordpy.mbpp_plus_loader_v1` is an anti-pattern**, not a
   loader.  W104 never touches it.
2. **Cross-bench arsenal-mining priors are an UPPER BOUND, not
   a Phase 2 earning signal.**  W104 records the W88+W91+W102
   HumanEval+ re-grade prior (+5.56 pp B − A1) and the W103
   empirical 70B result (+20.00 pp) as PROVENANCE inputs only.
   Neither earns the +5 pp margin bar at 405B.  Fresh-K=5
   sampling at 405B is the ground truth.
3. **MLB-2 rescue rate varies by benchmark family AND by
   scale.**  W102 MBPP+ V2 produced MLB-2 = 22.22 % (well below
   the W89 HumanEval-family 47 % retirement template).  W103
   HumanEval+ at 70B produced MLB-2 = 47.06 % (byte-for-byte
   match to W89).  W104 records BOTH as priors; the 405B cross-
   scale verdict reads against the W89 template; if MLB-2
   collapses similarly to W102 / W96-C / W100 the W104 verdict
   is downgraded to `PASS_NON_MECHANISM_DRIVEN`.
4. **Cross-scale collapse pattern from W96-A / W96-C / W100.**
   Three out of three cross-modal cross-scale 11B→90B
   confirmations showed margin shifts of −5 to −10 pp from
   the smaller-scale Phase 2 result.  W104 is the FIRST code-
   benchmark cross-scale confirmation; the cross-modal pattern
   is a structural prior to take seriously (cross-scale
   collapse is a real risk, not a remote one).
5. **Sidecar mid-run visibility.**  W102 buffered all 330
   sidecar entries until pilot exit; W103 added the per-write
   flush.  W104 inherits the W103 flush behaviour AND adds a
   "resume-from-sidecar" capability so a socket-hang or 429
   storm that requires a kill-and-restart does NOT lose
   evidence.

## Lead lane — HumanEval+ cross-scale cheap pilot at 405B (CONDITIONAL on reachability)

### Decision logic (pre-locked BEFORE driver is written)

1. **Pilot driver builds + unit-tested + Linear-synced**
   regardless of reachability smoke outcome.  The driver
   mirrors `scripts/run_w103_humaneval_plus_pilot.py` shape
   VERBATIM, with three additions:
   (a) accepts a `--target-model` argument with the W104
   pre-committed value defaulting to
   `meta/llama-3.1-405b-instruct`;
   (b) reuses the EXACT W103 slice via a `--reuse-slice` flag
   that loads the W103 slice CID list directly (zero re-
   invocation of the helper, byte-for-byte slice equality);
   (c) emits the cross-scale comparator block in the bench
   report (W104 cross-scale-discipline hardening lane
   deliverable).
2. **Reachability smoke probe** (sub-second NIM cost): the
   driver issues ONE chat completion call with a 4-character
   prompt to the primary 405B target.  HTTP 200 OK ⇒ primary
   target is used; HTTP 404 / 403 / 5xx persistent ⇒ backup
   target is used; both unreachable ⇒ W104 deferred (see
   above).
3. **Slice byte-equal to W103** (NEW W104 contract): the
   driver loads the W103 bench-iteration task_ids list
   directly from `results/w103/humaneval_plus_pilot/<RUN>/
   provenance.json` and recomputes the slice CID at run
   start.  Slice CID mismatch ⇒ refuse to run.  This makes
   per-problem cross-scale comparison APPLES-TO-APPLES.
4. **Cheap pilot launches** (1 seed × 30 problems × K=5 = 330
   NIM calls at the chosen target; ~30 min – 3 h wall subject
   to NIM throttling).  Single-seed slice with `--seed 104001`
   (preserves audit-chain isolation from W88 / W89 / W102 /
   W103 namespaces).
5. **Phase 2 9-gate + MLB-1 + MLB-2 sub-gates** evaluated per
   the W95-9-gate locked shape (carry-forward verbatim from
   W101 / W102 / W103).
6. **Cross-scale comparator block** computed against the W103
   bench report (per-problem diff: shared_wins / b_only_wins /
   a1_only_wins / shared_fails on the 30-problem slice at BOTH
   scales; cross-scale shift on B − A1 in pp; cluster mix
   delta).
7. **Decision branch** (pre-locked here; applied at verdict):
   * **Branch A — full PASS** (9 / 9 gates + MLB-1 + MLB-2
     both clear at 405B): register
     `W104-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-405B-PASS` (1-
     seed cheap pilot at 405B).  W105 = HumanEval+ Phase 3
     retirement bench (3 seeds × 100 problems × K=5).  The
     W105 slice pack is pre-built in this milestone (see
     Planning lane § below) so W105 is execution, not
     paperwork.
   * **Branch B — PASS with MLB-2 FAIL** (9 / 9 gates clear
     but reflexion rescue rate < 33 %): downgrade to
     `PASS_NON_MECHANISM_DRIVEN` (mirrors W96-C / W100
     precedent); W105 explores mechanism variations on
     HumanEval+ at 405B rather than Phase 3.  Add carry-
     forward `W104-L-HUMANEVAL-PLUS-MECHANISM-LOAD-
     BEARINGNESS-WEAK-AT-405B-CAP`.
   * **Branch C — pilot FAIL** (margin / G2 / per-problem-
     majority FAIL): add carry-forward
     `W104-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-405B-CAP`.  This
     would be a STRUCTURALLY INFORMATIVE outcome — it would
     refute the cross-scale generalisation of the W103 70B
     result.  W105 then refreshes the code-line ranking per
     the Branch C triage table below.  The fallback ranking
     refresh is pre-built in this milestone so W105 starts
     execution-ready.

### Phase 2 cheap-pilot gates (W95 9-gate shape; verbatim from W101 / W102 / W103)

1. **Slice pre-committed**: 30 problems by BYTE-EQUAL reuse of
   the W103 helper-anchored slice.  Slice CID verified at run
   start (must equal `c35155956ece605c...`); refuses to run on
   mismatch.
2. **A1 < 90 %**: A1 @ K=5 pass rate on the 30-problem slice
   must stay below 90 %.  Predicted 405B helper-anchored slice
   A1@K=5 ≤ 71 % (saturation margin ≥ 19 pp).
3. **B > A1**: `b_pass_rate > a1_pass_rate`.
4. **Margin ≥ +5 pp**: `b_pass_rate − a1_pass_rate ≥ 5 pp`.
5. **B > A0 by ≥ +5 pp**: reflexion mechanism is load-bearing.
6. **Per-problem majority**: B ≥ A1 on ≥ 16 of 30 problems.
7. **Budget accounting exact**: 1 + 5 + 5 = 11 calls per
   problem; 330 calls total.
8. **Audit chain re-derives**: per-call sidecars + per-seed
   Merkle + bench Merkle re-derive offline.
9. **Executor stays clean**: HumanEval+ canonical-solution
   self-test re-run at end-of-run → 100 % pass on the 30 slice
   problems.

### Mechanism-load-bearingness sub-gates (B only; W100 / W101 / W102 / W103 carry-forward verbatim)

* **MLB-1 — Reflexion-cycle invocation rate ≥ 33 %** of
  problems on the slice (≥ 10 / 30 problems where attempt 0
  FAILs and reflexion is exercised).
* **MLB-2 — Reflexion rescue rate ≥ 33 %** of MLB-1 numerator
  (≥ 1 in 3 reflexion-exercised problems end up PASSing).

A B PASS with MLB-2 < 33 % is downgraded to
`PASS_NON_MECHANISM_DRIVEN` (Branch B).

### Anti-cheat (verbatim from W88 – W103)

* Slice = byte-equal to W103; slice CID verified at run start.
* Same model on every arm (the W104 cross-scale target).
* Same K=5 byte-exact budget on A1 / B (sequential reflexion
  runs the FULL K=5 budget; no early-stop).
* Executor = `coordpy.humaneval_plus_executor_v1.run_humaneval_
  plus_executor_v1`.  No LLM judge; subprocess CPython.
* Corpus SHA-256-anchored at pilot start; mismatches refuse to
  run.
* No selective retries; each (seed, problem, arm) is exactly
  one set of calls.
* Per-call sidecars + per-seed Merkle + bench Merkle re-derive
  offline.

## Hardening lane — cross-scale discipline (UNCONDITIONAL)

W102 + W103 surfaced four guardrails that the W104 milestone
codifies into durable infrastructure:

1. **Sidecar mid-run visibility** — W102 buffered all 330
   sidecar entries until pilot exit; W103 added per-write
   flush.  W104 EXTENDS this with a **resume-from-sidecar**
   capability: if the pilot is killed mid-run (socket hang /
   429 storm), the driver can restart against the same
   sidecar file and skip already-completed (problem_idx, arm)
   tuples.  Tests assert: (a) per-write flush is called after
   every NIM call; (b) resume detects already-written entries
   by `(seed, p_idx, arm, attempt_idx)` tuple; (c) corrupted
   trailing line is detected and treated as not-yet-completed
   (not silently consumed).

2. **Cross-scale comparator schema/provenance guard** —
   `coordpy.cross_scale_comparator_v1` is the single new
   module (explicit-import only).  It takes two HumanEval+
   bench reports + their provenance JSONs and emits a
   structured cross-scale diff with the following fields per
   problem: `task_id`, `cluster_at_scale_a`,
   `cluster_at_scale_b`, `a0_at_scale_a/b`,
   `a1_at_scale_a/b`, `b_at_scale_a/b`, `first_pass_idx_at_
   scale_a/b`, `cross_scale_cluster_shift` (one of
   {`stayed`, `improved`, `regressed`, `flipped`}).  The
   comparator REFUSES to run if (a) the two reports do not
   share a slice CID, (b) the two reports do not share a
   corpus SHA-256, (c) either report's MLB block is missing,
   or (d) either report's schema version is unrecognised.
   This catches a 70B-vs-405B mix-up at write time, not at
   post-hoc autopsy time.

3. **Provenance-on-cross-scale guard** — the W104 cross-scale
   verdict doc records the EXACT same provenance fields as
   W103 (corpus_sha, helper_proposal_cid, mining_report_cid,
   preflight_verdict_cid, slice_cid_helper_priority,
   slice_cid_bench_order, arsenal_mining_prior with explicit
   `earning_status = "recorded; NOT a Phase 2 gate input
   (W102 anti-pattern carry-forward)"`) PLUS three new fields:
   `cross_scale_pair_a_run_id` (W103 70B run id),
   `cross_scale_pair_a_bench_merkle` (W103 bench Merkle),
   `target_selection_rule_version` (`coordpy.w104_target_
   selection_rule.v1`).

4. **Cross-scale comparator emitted automatically** — the
   pilot driver, on successful pilot completion, automatically
   invokes the comparator against the W103 bench report and
   emits `cross_scale_comparator_report.json` alongside the
   pilot bench report.  The verdict doc references this file
   verbatim.

### Hardening-lane deliverable (locked)

* `coordpy/cross_scale_comparator_v1.py` — single new
  explicit-import-only module (ZERO additions to
  `coordpy/__init__.py`); ≤ 200 lines; pure-Python; no NIM.
* `tests/test_w104_cross_scale_discipline_v1.py` — new test
  file; ≥ 8 tests covering (a) resume-from-sidecar regression
  guard; (b) per-write sidecar flush regression guard; (c)
  cross-scale comparator schema/provenance refuse-to-run
  paths; (d) cross-scale comparator cluster-shift correctness
  on a synthetic pair; (e) target-selection-rule deterministic
  evaluation; (f) anti-pattern token guard on the W104 target
  list; (g) slice CID equality with W103; (h) provenance
  schema-version pin.
* `scripts/run_w104_humaneval_plus_cross_scale_pilot.py` —
  pilot driver consuming `coordpy/cross_scale_comparator_v1`
  + the W103 slice + the W102 HumanEval+ bench module.
* `docs/RESULTS_W104_HUMANEVAL_PLUS_PHASE2_405B_V1.md` —
  verdict doc with the provenance fields + cross-scale
  comparator block verbatim.
* `docs/RESULTS_W104_CROSS_SCALE_COMPARATOR_V1.md` —
  standalone cross-scale comparator narrative for the
  verdict.

## Planning lane — W105 pre-commit (UNCONDITIONAL)

The W105 next step is pre-committed in this milestone so the
milestone boundary doesn't drift on outcome.  W105 is
*execution* (NOT paperwork) under BOTH the success branch
and the failure branch.

### Branch A — W104 PASS_MECHANISM_DRIVEN (full PASS at 405B)

W105 = HumanEval+ Phase 3 retirement bench (3 seeds × 100
problems × K=5 = 3 300 NIM calls / scale × 2 scales = 6 600
NIM calls).  The Phase 3 slice pack is pre-built in this
milestone (`docs/RESULTS_W104_HELPER_W105_PLANNING_V1.md` §
"W105 Phase 3 slice pack"):

* **Slice expansion rule**: take the W103 helper-anchored
  30-problem slice as the inner kernel; expand to 100
  problems via the W102 code-slice-selector helper consuming
  the full W101 + W102 arsenal-mining clusters; preserve the
  helper-priority order; SHA-anchored slice CID recorded.
* **Multi-seed rule**: 3 seeds (105 001 / 105 002 / 105 003)
  chosen at uniform spacing from the seed namespace to keep
  audit isolation; same K=5 mechanism.
* **Phase 3 gates**: the W88 / W89 / W95 6-bar retirement
  shape (margin ≥ +5 pp + per-seed majority + per-problem
  majority + audit chain).
* **Multi-scale rule**: Phase 3 is run at BOTH 70B (W103
  scale) AND 405B (W104 scale) on the SAME 100-problem slice,
  giving 6 seed-scale combinations.  Cross-scale retirement
  margin = (Phase 3 mean B − A1 across both scales).

### Branch B — W104 PASS_NON_MECHANISM_DRIVEN

W105 = HumanEval+ mechanism-variation slate at 405B (parallel
B-style variants: B1 = enforced-reflexion-on-attempt-0 (no
early-pass); B2 = test-aware decomposition reader+solver on
the EvalPlus extra-test surface; B3 = sidecar-driven failure-
cluster targeting at 405B).  Pre-commit cheap-pilot earning
rule: at least one B-variant must lift MLB-2 ≥ 33 % AT 405B
AND keep margin ≥ +5 pp.  This branch's slice pack is the
SAME W103 helper-anchored 30-problem slice (cheap-pilot
budget).

### Branch C — W104 FAIL

W105 = code-line ranking refresh.  The pre-committed
candidate triage table (carried forward from W103 RUNBOOK and
EXTENDED with W104 evidence shape):

| FAIL mode | W105 lead step | Rationale |
|---|---|---|
| Margin < 0 pp + MLB-2 < 33 % at 405B (mechanism-distribution shift; close to W102 MBPP+ V2 pattern) | LiveCodeBench preflight (NIM-free) | The W101 matrix ranked LiveCodeBench second-best; its lower per-problem ceiling at 70B + 405B makes it the natural next attack on the cross-bench mechanism. |
| G2 saturation (A1 ≥ 90 % on the slice; ceiling-pressure FAIL) | APPS preflight (NIM-free) | The W101 matrix ranked APPS C-grade on infra cost but its larger problem ceiling (1 + 10 + 5 = 16 K problems) defuses any A1 saturation; APPS preflight is the right move if 405B saturates HumanEval+. |
| Margin < +5 pp but ≥ 0 pp, MLB-2 ≥ 33 % (per-seed sampling variance) | HumanEval+ multi-seed cheap confirmation at 405B (3 seeds × 30 problems × K=5; same slice) | If reflexion is load-bearing but the per-seed margin missed by < 5 pp, the next cheap step is multi-seed sampling at 405B on the same slice rather than a new bench. |
| Branch-C-quad-bottom: margin < 0 pp AND MLB-2 ≥ 33 % AND G2 < 90 % (mechanism IS load-bearing but B regressed at 405B; close to W96-A 11B→90B Phase 3 pattern) | Cross-scale-collapse audit + W103 70B Phase 3 (3 seeds × 100 problems × K=5 at 70B ONLY) | If the mechanism is load-bearing at 405B but the margin reversed, the next move is to test whether the W103 70B PASS replicates at multi-seed and Phase 3 size at the SAME scale, isolating cross-scale collapse from per-seed sampling. |

SWE-bench-lite STAYS out of scope unconditionally — the W89
reflexion mechanism does not have the structural shape to
attack SWE-bench-lite's repo-level failure surface.

The dispatch decision is recorded in
`docs/RESULTS_W104_MILESTONE_SUMMARY_V1.md` post-pilot per the
applied branch.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* Exactly ONE new `coordpy.*` module in W104:
  `coordpy.cross_scale_comparator_v1` (explicit-import only;
  not added to `coordpy/__init__.py`).
* New W104 artefacts:
  * `coordpy/cross_scale_comparator_v1.py` (single new
    module; the cross-scale-discipline-hardening lane).
  * `scripts/run_w104_humaneval_plus_cross_scale_pilot.py`
    (cross-scale-pilot driver; consumes W102 HumanEval+
    infrastructure + reused W103 slice + cross-scale
    comparator).
  * `scripts/run_w104_w105_phase3_slice_pack.py` (W105 slice
    pack pre-builder; consumes `coordpy.code_slice_selector_
    v1` as a real downstream consumer for the second time).
  * `tests/test_w104_cross_scale_discipline_v1.py` (hardening-
    lane tests).
  * `docs/RUNBOOK_W104.md` (this file).
  * `docs/RESULTS_W104_HUMANEVAL_PLUS_PHASE2_405B_V1.md`
    (verdict doc; populated post-pilot).
  * `docs/RESULTS_W104_CROSS_SCALE_COMPARATOR_V1.md`
    (comparator doc; populated post-pilot).
  * `docs/RESULTS_W104_HELPER_W105_PLANNING_V1.md` (W105 slice
    pack + fallback ranking).
  * `docs/RESULTS_W104_MILESTONE_SUMMARY_V1.md` (milestone
    summary; populated at close).
  * `docs/FRONTIER_RELEVANCE_AUDIT_W104_V1.md` (frontier audit
    supplement; 14th preflight-discipline validation).

## Cross-scale rule (W96-C carry-over; verbatim from W102 / W103)

* **W104 405B Phase 2 entitled** because W103 70B Phase 2
  PASSed with MLB-1 + MLB-2 both clear.
* **W105 Phase 3 retirement bench** entitled IFF W104 PASSes
  Branch A.  Phase 3 spans BOTH scales on the SAME 100-problem
  slice (3 seeds × 100 problems × K=5 × 2 scales).
* A Phase 2 PASS at one scale alone is NOT sufficient for the
  Phase 3 retirement bench.
* A Phase 2 PASS at two scales with matching MLB-2 load-
  bearing is the strongest cheap-pilot evidence the programme
  has ever assembled for code-benchmark cross-bench
  generalisation — but it is STILL NOT retirement-grade.
  Retirement-grade requires Phase 3 multi-seed at the bar
  shape W88/W89 carry verbatim.

## Operational plan

### Phase 1 — done in W104 (NO NIM)

1. **(W104 hardening lane)** —
   `coordpy/cross_scale_comparator_v1.py` (≤ 200 lines;
   pure-Python; explicit-import only).
2. **(W104 hardening tests)** —
   `tests/test_w104_cross_scale_discipline_v1.py`; ≥ 8 tests;
   all PASS.
3. **(W104 lead-lane driver)** —
   `scripts/run_w104_humaneval_plus_cross_scale_pilot.py`;
   byte-equal slice reuse from W103; reachability smoke
   probe; cross-scale comparator emitted automatically;
   resume-from-sidecar.
4. **(W104 W105 slice-pack pre-builder)** —
   `scripts/run_w104_w105_phase3_slice_pack.py`; consumes
   `coordpy.code_slice_selector_v1`; emits the 100-problem
   slice + 3 seeds.
5. **(W104 planning artifact)** —
   `docs/RESULTS_W104_HELPER_W105_PLANNING_V1.md`; documents
   both the Phase 3 slice pack (Branch A) and the fallback
   ranking refresh (Branch C).
6. **(W104 frontier-relevance audit supplement)** —
   `docs/FRONTIER_RELEVANCE_AUDIT_W104_V1.md`; 14th
   consecutive preflight-discipline validation.
7. **(Linear ↔ GitHub sync)** — create `COO-28`; append a
   `W104` entry to `linear_github_mapping.json`; post W104
   verdict comments to `COO-6`, `COO-9`, `COO-14`, `COO-27`,
   `COO-28`.

### Phase 2 — conditional on reachability smoke probe (sub-second NIM cost)

1. **Launch cheap HumanEval+ cross-scale pilot** at the
   resolved target (primary `meta/llama-3.1-405b-instruct`;
   backup `meta/llama-3.1-70b-instruct` if 405B unreachable):
   ```bash
   NVIDIA_API_KEY=... python scripts/run_w104_humaneval_plus_cross_scale_pilot.py \
       --target-model meta/llama-3.1-405b-instruct \
       --reuse-slice results/w103/humaneval_plus_pilot/latest_run/provenance.json \
       --n-problems 30 --seed 104001
   ```
2. **Evaluate Phase 2 gates** (9 W95 gates + MLB-1 + MLB-2).
   Verdict goes in
   `docs/RESULTS_W104_HUMANEVAL_PLUS_PHASE2_405B_V1.md`.
3. **Emit cross-scale comparator** (automatic on successful
   pilot completion).  Doc:
   `docs/RESULTS_W104_CROSS_SCALE_COMPARATOR_V1.md`.
4. **Apply pre-locked decision branch** (A / B / C above).

### Phase 3 — DEFERRED to W105+ (Phase 3 retirement OR mechanism variation OR ranking refresh)

W104 explicitly does NOT pre-commit Phase 3 launch or Phase 2
mid-flight escalations.  W105 is pre-committed in the Planning
lane § above by the applicable branch.

## Pre-pilot prediction (recorded 2026-05-26 BEFORE W104 pilot)

> "Subjective priors over the W104 HumanEval+ cross-scale
> Phase 2 cheap pilot at 405B on the W103 helper-anchored
> 30-problem slice with K=5, conditional on reachability:
>
> * Probability A1@K=5 clears the saturation gate (< 90 %):
>   **~ 96 %** (predicted 405B helper-anchored slice A1@K=5
>   ≤ 71 % from Hoeffding bounds + helper-anchored hard-
>   cluster bias).
> * Probability B beats A1 on the mean: **~ 75 %** (W103
>   delivered +20 pp at 70B on the same slice; cross-scale
>   collapse pattern from W96-A / W96-C / W100 says cross-
>   scale can shift the margin meaningfully; the W103 +20 pp
>   gives substantial buffer above the +5 pp bar).
> * Probability B − A1 ≥ +5 pp: **~ 60-70 %** (the W103
>   +20 pp gives 15 pp of headroom for cross-scale shift; the
>   W96-A / W100 patterns showed -5 to -10 pp cross-scale
>   shifts).
> * Probability MLB-2 sub-gate clears (rescue rate ≥ 33 %):
>   **~ 55-65 %** (the W96-C MLB-2 11B → 90B collapse was
>   100 % → 14.3 %; the W100 MLB-2 11B → 90B collapse was
>   100 % → 11.1 %; HumanEval-family is structurally different
>   from RealWorldQA, and the W89 + W103 47 % rescue rate
>   precedent on the SAME bench at 70B is the strongest prior
>   we have — but mechanism-load-bearingness cross-scale
>   collapse on code benchmarks is empirically untested).
> * Probability the W104 verdict is `PASS_MECHANISM_DRIVEN`
>   (Branch A; full PASS + MLB clear): **~ 40-50 %**.  This
>   would be the FIRST code-benchmark cross-scale Phase 2
>   PASS in the programme and would entitle Phase 3 in W105.
> * Probability the W104 verdict is Branch B (PASS +
>   MLB-2 FAIL): **~ 20 %**.  Would mirror the W96-C 90B
>   pattern.
> * Probability the W104 verdict is Branch C (FAIL):
>   **~ 30-40 %**.  Would mirror the W96-A / W100 cross-
>   scale collapse pattern.
>
> If W104 PASSes Branch A, the programme would be entitled to
> claim that the W89 sequential-reflexion mechanism extends
> to a SECOND model class on HumanEval+ at cheap-pilot Phase 2
> scale.  Retirement-grade generalisation still requires W105
> Phase 3 multi-seed.  W104 alone is NOT a multi-benchmark
> multi-scale same-budget retirement.
>
> If W104 FAILs, the W104 verdict caps the W103 mechanism at
> the 70B scale on HumanEval+ at the cheap-pilot size, and
> the Branch C dispatch table determines the next code-line
> move."

## Honest framing

W104's job is to:

1. **Honestly test cross-scale** by reusing the EXACT W103
   slice byte-for-byte, isolating model-class shift from
   slice-distribution noise.  A cross-scale Phase 2 PASS on
   the SAME slice is the cleanest cross-scale evidence
   shape the programme can produce.
2. **Honestly ship hardening** that addresses W102 / W103
   failure modes empirically — resume-after-socket-hang;
   mid-run sidecar visibility; cross-scale comparator
   schema/provenance refuse-to-run on mix-up; provenance
   schema-version pin.
3. **Honestly pre-commit W105** by outcome so milestone
   boundaries don't drift on result.  The W105 Phase 3 slice
   pack (Branch A) AND the fallback ranking refresh (Branch
   C) are BOTH built in W104.
4. **Launch the cheap pilot ONLY IF** the reachability smoke
   probe PASSes.  No buying long runs from hope.

If W104 PASSes Branch A, the programme is entitled to the
*stronger* claim that the W89 reflexion mechanism extends to
HumanEval+ at TWO model scales at cheap-pilot Phase 2 quality.
Retirement-grade generalisation requires W105 Phase 3 multi-
seed at BOTH scales.  W104 alone is NOT a multi-scale same-
budget retirement.

If W104 FAILs, the W104 verdict caps the W103 70B PASS at the
single-scale-cheap-pilot level, and the Branch C dispatch
table determines the next code-line move.  Either outcome
preserves the W93 – W103 preflight-first + cross-scale +
multi-candidate-tournament-then-confirm + mechanism-load-
bearingness + silent-degradation-anti-pattern-guard +
arsenal-mining-prior-anti-pattern-guard discipline as the
14th consecutive validation.
