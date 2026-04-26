# SDK v3.3 — deeper intra-cell + lifecycle audit + deterministic mode

> Theory-forward milestone note. SDK v3.3, 2026-04-26. Anchors:
> `vision_mvp/wevra/capsule.py` (PARSE_OUTCOME),
> `vision_mvp/wevra/runtime.py` (`_make_intra_cell_hooks`),
> `vision_mvp/wevra/lifecycle_audit.py`,
> `vision_mvp/product/runner.py` (deterministic canonicalisation),
> `vision_mvp/tests/test_wevra_capsule_native_deeper.py` (18 tests).
> Status taxonomy follows
> [`docs/HOW_NOT_TO_OVERSTATE.md`](HOW_NOT_TO_OVERSTATE.md).

## What is materially newly true

1. **The parser axis is lifecycle-governed.** Up to SDK v3.2 the
   parser's `ParseOutcome` was a plain dataclass passed inside
   `llm_patch_generator`'s body. Under v3.3, every (task, strategy)
   inside a sweep cell seals a `PARSE_OUTCOME` capsule (parent:
   SWEEP_SPEC) BEFORE the corresponding `PATCH_PROPOSAL`. The
   parse → patch → verdict chain has a typed DAG witness —
   downstream consumers can navigate from a `TEST_VERDICT` back
   through its `PATCH_PROPOSAL` to its parser-axis decision
   without parsing rationale strings. Theorem W3-39.

2. **The lifecycle correspondence is mechanically checkable.** A
   `CapsuleLifecycleAudit` runs over a finished ledger and
   verifies eight invariants L-1..L-8. Counterexamples surface as
   typed `{rule, capsule_cid, capsule_kind, detail}` violations.
   The audit is also runnable from a forensic
   `capsule_view.json` alone (no in-process ctx required) via
   `audit_capsule_lifecycle_from_view`. Theorem W3-40.

3. **The full capsule DAG is reproducible across machines.**
   `RunSpec(deterministic=True)` strips per-run timestamps,
   wall-clock fields, host-local paths, and absolute output
   directories from PROVENANCE / READINESS_CHECK / RUN_REPORT
   capsule payloads and ARTIFACT capsule paths. Two independent
   runs of the same deterministic profile (mock mode,
   in_process / subprocess sandbox, frozen JSONL) produce
   byte-identical CIDs on every kind, identical chain head,
   identical root CID. Theorem W3-41.

## The new theorems / conjectures

### Theorem W3-39 (PARSE_OUTCOME lifecycle gate + DAG chain) — proved

For any capsule-native run with a sealed SWEEP_SPEC `s`, every
sealed PARSE_OUTCOME capsule `p` satisfies:

  - `parents(p) = (cid(s),)`.
  - `payload(p)` carries closed-vocabulary `failure_kind` from
    `swe_patch_parser.ALL_PARSE_KINDS ∪ {"oracle", "gen_error"}`,
    a `recovery` label from
    `swe_patch_parser.ALL_RECOVERY_LABELS`, an `ok` boolean, a
    `substitutions_count` integer, and a bounded `detail` string.
  - The corresponding PATCH_PROPOSAL capsule `q` (parented on
    `p`'s CID and `s`'s CID) has matching coordinates (instance,
    strategy, parser_mode, apply_mode, n_distractors) on its
    payload, AND its admission depends on `p`'s CID being already
    in the ledger (Capsule Contract C5).

**Proof.** By inspection of `seal_parse_outcome` (gates on
`spec_cap is not None`), `seal_patch_proposal` (admits with
parents `(spec_cap.cid, parse_outcome_cid)` and refuses if
`parse_outcome_cid not in self.ledger`), and
`_make_intra_cell_hooks` (closes over the cell coordinates and
calls `seal_parse_outcome` then `seal_patch_proposal`). $\square$

**Code anchor.** `seal_parse_outcome` + `seal_patch_proposal` in
`capsule_runtime.py`; hook plumbing in `runtime.py`;
`vision_mvp/tests/test_wevra_capsule_native_deeper.py
::ParseOutcomeLifecycleTests`.

### Theorem W3-40 (Lifecycle-audit soundness) — proved + mechanically-checked

For any `CapsuleNativeRunContext` `ctx` whose `audit_capsule_lifecycle(ctx)`
returns `verdict == "OK"`, the underlying ledger satisfies the
eight lifecycle invariants:

  - **L-1** Every PROPOSED-but-not-sealed in-flight entry has a
    non-empty `failure` string AND is not in the ledger.
  - **L-2** Every PATCH_PROPOSAL's parent set includes the
    SWEEP_SPEC's CID.
  - **L-3** Every TEST_VERDICT has exactly one parent, and that
    parent is a sealed PATCH_PROPOSAL.
  - **L-4** Every PARSE_OUTCOME's parent is exactly the SWEEP_SPEC.
  - **L-5** Every SWEEP_CELL's parent is exactly the SWEEP_SPEC.
  - **L-6** PARSE_OUTCOME ↔ PATCH_PROPOSAL ↔ TEST_VERDICT
    coordinate multisets are equal.
  - **L-7** Every PATCH_PROPOSAL whose parent set contains a
    PARSE_OUTCOME has matching coordinates with that PARSE_OUTCOME.
  - **L-8** Every TEST_VERDICT is sealed strictly after its
    PATCH_PROPOSAL in the ledger's append order.

**Proof.** By inspection of the eight `_check_l*` methods in
`CapsuleLifecycleAudit`. Each method enumerates every capsule of
the relevant kind and emits a typed violation on any failure;
`run()` returns OK iff every method emits zero violations. The
soundness is a tautology: OK ⇔ no violation emitted ⇔ every
invariant held on every checked capsule. $\square$

**Code anchor.** `vision_mvp/wevra/lifecycle_audit.py`;
`vision_mvp/tests/test_wevra_capsule_native_deeper.py
::LifecycleAuditTests`.

**Corollary (audit-from-view).**
`audit_capsule_lifecycle_from_view` reconstructs a synthetic
ledger from a rendered view dict and runs the same audit;
because the view's `payload_kinds_always` invariant
(`render_view`) guarantees that PARSE_OUTCOME / PATCH_PROPOSAL
/ TEST_VERDICT / ARTIFACT / META_MANIFEST payloads are always
included on disk, every L-2..L-8 check is recoverable from
disk bytes alone. L-1 is vacuous on a forensic view (failed
in-flight entries are not in the view).

### Theorem W3-41 (Deterministic-mode CID determinism) — proved + empirical

Let `Π` be a deterministic profile (mock mode,
in_process / subprocess sandbox, frozen JSONL whose SHA-256 is
stable). Let `R₁`, `R₂` be two runs under
`RunSpec(profile=Π, out_dir=O₁ ≠ O₂, deterministic=True)`.

**Claim.** For every kind `k ∈ CapsuleKind.ALL`, the multiset
of CIDs of kind `k` in `R₁`'s rendered view equals that in `R₂`'s
view. The chain head and root CID are byte-identical across
`R₁` and `R₂`.

**Proof sketch.** The CID of a capsule is
`SHA-256(canonical(kind, payload, budget, sort(parents)))`. The
canonicalisation in `_canonicalise_for_determinism`,
`_canonicalise_run_report_headers`,
`_canonicalise_readiness_verdict`, and
`_canonicalise_sweep_result` strips every payload field that
varies between runs of the same logical input (timestamps,
absolute paths, host names, wall-clock measurements, output
directories). The deterministic-mode `recorded_path` argument
to `seal_and_write_artifact` ensures ARTIFACT capsule
payloads carry basenames rather than absolute paths. After
canonicalisation, every payload is a deterministic function of
`Π`, the profile dict, the JSONL bytes, and the closed code
path; therefore every CID is deterministic; therefore every
parent set is deterministic; therefore every transitive chain
of CIDs (including chain head and root CID) is deterministic.
$\square$

**Empirical anchor.**
`DeterministicModeTests::test_w3_41_two_runs_collapse_to_identical_cids`
runs two independent local_smoke runs under
`deterministic=True` and asserts CID-set-equality on every
kind plus chain-head equality plus root-CID equality.

**Sharp scope.** Without `deterministic=True`, two runs produce
different PROVENANCE / RUN_REPORT CIDs by design (timestamp /
wall-clock variance). Real-LLM runs are non-deterministic by
construction (the LLM is sampling); deterministic mode does
not affect the LLM call graph. The on-disk product report
still records wall-clock fields for forensic context — the
determinism is on the capsule graph, not on wall clock.

### Theorem W3-32-extended² (intra-cell extension, second order) — proved

The lifecycle correspondence W3-32 (and its first-order extension
W3-32-extended for PATCH_PROPOSAL / TEST_VERDICT) lifts to the
sub-intra-cell PARSE_OUTCOME kind. Specifically: for the parser
axis, the in-progress / sealed / failed correspondence holds as
in W3-32, with the parent gate being SWEEP_SPEC.

**Proof.** Direct application of W3-32's argument with
`PARSE_OUTCOME` substituted for SWEEP_CELL and SWEEP_SPEC as
the parent gate. The argument is preserved because the
in-flight register and admit-and-seal pipeline are uniform
across kinds. $\square$

**Code anchor.** `seal_parse_outcome` mirrors `seal_sweep_cell`
in shape; the in-flight failure path is identical.

### Conjecture W3-C4 (new) — PARSE_OUTCOME failure_kind distribution

On the bundled `bundled_57` profile under a real LLM
(gemma2:9b or aspen_mac1_coder), the PARSE_OUTCOME capsules'
`failure_kind` distribution is *stable across runs* up to a
parser-recovery sub-axis. Specifically: the marginal over
`(failure_kind, recovery)` pairs is reproducible up to
$\pm$ 5 capsules out of 48 per cell across re-runs of the same
LLM tag with `temperature=0.0`.

**Falsifiers.** A run-to-run `failure_kind` distribution that
varies by more than 10 capsules out of 48 (tail variance in
the LLM despite temperature=0.0); a parser recovery heuristic
that fires non-deterministically (would be a bug in the parser).

**Status.** Conjectural, not yet measured. The empirical
measurement requires a real-LLM run on the cluster; this is
follow-up work.

### Conjecture W3-C5 (new) — sub-intra-cell PROMPT/LLM_RESPONSE slice

A future SDK extension that names the *LLM prompt bytes* and
*raw LLM response bytes* as PROMPT and LLM_RESPONSE capsules
(parents: SWEEP_SPEC, payload: bounded byte arrays under a
streaming budget) closes the inner-loop boundary while
preserving Theorem W3-34 spine equivalence.

**Falsifiers.** PROMPT / LLM_RESPONSE bytes too large for any
admission budget under realistic profiles (would force a
chunking design); spine CIDs drift under the new kind (would
require modifying SWEEP_SPEC's payload, breaking W3-34).

**Status.** Conjectural. Implementing this slice is the most
likely next milestone (SDK v3.4).

## What this changes about Wevra's originality and publishability

The pre-v3.3 framing was *"capsules drive execution at the run
boundary and the inner sweep loop pair (patch, verdict)"*. The
v3.3 framing is sharper:

> *Capsules drive execution at the run boundary, the inner sweep
> loop pair, AND the parser axis. The lifecycle correspondence is
> mechanically checkable on every finished run. The full capsule
> DAG is reproducible byte-for-byte across machines under a
> stated determinism flag.*

This is a meaningful upgrade for publishability:

- **A reviewer can run the lifecycle audit on any submitted
  artefact.** A paper that claims "our runtime maintains
  invariants L-1..L-8" can be verified by the reviewer pointing
  the audit at the submitted `capsule_view.json`. No "trust the
  authors" step.
- **A reviewer can replicate the capsule DAG byte-for-byte.**
  A paper that claims "this run produced root CID `X`" can be
  verified by re-running with `deterministic=True` and
  asserting `assert root_cid == "X"`. No reproducibility
  ambiguity.
- **A reviewer can locate the parser-axis behaviour without
  guessing.** Where pre-v3.3 the parser-axis taxonomy was buried
  in `ParseOutcome.failure_kind` rationale strings, post-v3.3 it
  is a typed capsule on the DAG. Aggregate parser-axis
  statistics become recoverable from `capsule_view.json` alone.

What is **not** changed by v3.3:

- The decoder-frontier results (W3-19 / W3-22 / W3-23 / W3-26 /
  W3-27 / W3-28 / W3-30 / W3-31) are unaffected. They live above
  the runtime; the runtime supports them but does not constrain
  them.
- The retraction of W3-C7 (strict reading) is unaffected. The
  defensible reading remains W3-C9.
- LLM prompts and raw LLM response bytes remain plain Python.
  Capsule discipline does not yet cover them.

## Honest scope claims

Following [`HOW_NOT_TO_OVERSTATE.md`](HOW_NOT_TO_OVERSTATE.md):

- Wevra is **not** "fully capsule-native". The parser axis is
  capsule-native; LLM prompts, raw LLM responses, sandbox
  stdout/stderr are not.
- The lifecycle audit is **mechanically-checked**, not
  **proved-correct-over-arbitrary-runtimes**. It is a runtime
  audit; its soundness is by inspection of the audit code.
- Deterministic mode is **not** "Wevra runs are reproducible".
  It is "the capsule DAG is reproducible under a frozen JSONL +
  deterministic profile + the explicit flag".
- W3-39 / W3-40 / W3-41 are **proved** (proofs by inspection of
  short code paths) and **empirically anchored** (18 contract
  tests passing in `test_wevra_capsule_native_deeper.py`).

## Empirical anchor (this milestone)

| Metric                                                | Value                              |
| ----------------------------------------------------- | ---------------------------------- |
| New SDK v3.3 contract tests                            | 18 (in `test_wevra_capsule_native_deeper.py`) |
| Total `vision_mvp.tests.test_wevra_*` + `test_capsule_*` passing | 165                                |
| `local_smoke` capsule count (default mode)             | 154 (was 106 pre-v3.3)             |
| `local_smoke` capsule count (deterministic mode)       | 154 (identical across two runs)    |
| `local_smoke` PARSE_OUTCOME count                      | 48 (one per (task, strategy) pair) |
| Lifecycle audit on local_smoke (default + deterministic) | OK on all 8 rules                |
| Capsule view chain_head deterministic across two runs   | YES (under `deterministic=True`)   |

## Cross-references

- Formal model: `docs/CAPSULE_FORMALISM.md` (theorems will be
  appended in § 4.I in a follow-up edit).
- Theorem registry: `docs/THEOREM_REGISTRY.md`.
- Research status: `docs/RESEARCH_STATUS.md`.
- Paper draft: `papers/wevra_capsule_native_runtime.md`.
- Master plan: `docs/context_zero_master_plan.md` § 4.20 (added
  in this milestone).
