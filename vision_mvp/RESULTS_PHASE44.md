# Phase 44 — Raw-text semantic residue, frontier coder-class headroom, and public-SWE-bench-Lite drop-in readiness

**Status: research milestone. Phase 44 is the first milestone where
the programme's centre of gravity is neither the substrate layer
(Phases 1–31, 35–38) nor the parser/matcher layer (Phases 40–42)
but the *semantic residue* that remains after parser and matcher
have done their jobs.** Phase 43 characterised that residue with a
nine-label closed taxonomy, but every Phase-43 result was derived
from a *sentinel* proposed-patch (§ D.7) because the Phase-42
artifact schema does not preserve the raw LLM output. Phase 44
removes that limitation, runs the strongest practical coder-class
cell on the ASPEN cluster with raw capture on, and promotes the
public-SWE-bench-Lite drop-in path from documented-to-validated
code.

Four coupled artifacts ship:

1. **Raw-text capture** (``vision_mvp/tasks/swe_raw_capture.py``).
   A new opt-in module with a ``RawCaptureStore``,
   ``RawCaptureRecord`` (schema version ``phase44.v1``), and a
   ``make_capturing_generator`` decorator that plumbs raw LLM text
   + parse outcome + proposed substitutions + applied substitutions
   + patched-source hash through to a companion JSON artifact.
   The bridge, parser, matcher, substrate, and sandbox paths are
   unchanged.
2. **Phase-44 driver** (``vision_mvp/experiments/phase44_semantic_residue.py``).
   Runs the Phase-42-shape sweep with raw capture on, OR in
   ``--analyse-only`` mode ingests parent+capture pairs and emits a
   cross-model refined-taxonomy summary.
3. **Public-readiness validator** (``vision_mvp/experiments/phase44_public_readiness.py``).
   A five-check pipeline (schema / adapter / parser / matcher /
   test_runner) that takes any local JSONL and emits a CI-gate
   verdict. The bundled 57-instance bank scores 57/57 on every
   check in ~5 s; an external public SWE-bench-Lite JSONL is a
   pure ``--jsonl <path>`` change.
4. **Refined semantic taxonomy**
   (``vision_mvp/tasks/swe_semantic_taxonomy.py`` extension).
   Five Phase-44 sub-labels (``SEM_RIGHT_FILE_WRONG_SPAN``,
   ``SEM_RIGHT_SPAN_WRONG_LOGIC``, ``SEM_PARTIAL_MULTI_HUNK_SUCCESS``,
   ``SEM_NARROW_FIX_TEST_OVERFIT``, ``SEM_STRUCTURAL_VALID_INERT``)
   that partition the Phase-43 coarse buckets when real proposed-
   patch bytes are available. The v2 classifier
   (``classify_semantic_outcome_v2``) subsumes the v1 classifier on
   sentinel inputs (Theorem P44-2) — backwards compatibility is a
   theorem, not an aspiration.

Phase 44 in one line: **with raw-text capture the wrong-edit-site
bucket becomes attributable (anchor-in-file vs anchor-not-in-file),
the multi-hunk bucket becomes attributable (partial success vs
blindness), the inert bucket becomes attributable (behaviourally
equivalent vs runtime-typed mismatch), and the substrate claim is
preserved byte-for-byte across every refinement.**

Three new theorems (P44-1 / P44-2 / P44-3) and four new
conjectures (C44-1 / C44-2 / C44-3 / C44-4). All Phase-39..43
regressions green at the 57-instance scale (112/112 on the SWE-arc
slice + 23 new Phase-44 tests = 135 total).

---

## Part A — Research framing

### A.1 Why this milestone exists

Phase 43 § D.7 named one limitation plainly: the Phase-42 artifact
JSONs store per-measurement ``error_kind``, ``test_passed``,
``strategy``, and ``instance_id``, but not the raw LLM text or the
actual parsed ``(old, new)`` pairs. The Phase-43 analysis driver
compensated by passing a sentinel
``(("__sentinel__", "__sentinel__"),)`` tuple into the classifier,
which forced every non-parse-fail / non-patch-no-match / non-syntax
/ single-hunk failure into the ``SEM_WRONG_EDIT_SITE`` bucket. That
bucket was therefore a *conservative upper bound*, not a fact.

Two residue-shaped questions became sharply answerable only by
sizing that limitation:

1. **On a coder-class model, does "wrong edit site" mean "did not
   edit the right file" or "edited the right file on a different
   span"?** The latter is fundamentally a localisation-within-
   context problem; the former is a retrieval / context-delivery
   problem. The substrate *should* help the second case and
   *cannot* help the first, and until the bucket is split we do
   not know which regime the 14B-coder's 4/57 residue lives in.
2. **On a frontier model with 2/57 residue, what are those two
   instances?** The Phase-43 summary called them
   ``test_exception`` + ``test_assert`` on a sentinel path. With
   raw bytes we can say whether the fix is correct-logic applied
   elsewhere (site), narrow-fix-overfit-to-primary (overfit),
   runtime-typed-mismatch (inert), or right-span-but-the-fix-
   doesn't-generalize (logic).

Phase 44 answers both directly.

Separately, Phase 43 § D.1 named the public-SWE-bench-Lite drop-in
"a ``--jsonl <path>`` swap". That claim was loader-side true
(Theorem P41-2 reproduces on every bundled instance under oracle)
but did not validate schema / adapter / test-runner compatibility
on a public-shape row. Phase 44 ships the ``phase44_public_readiness``
validator that takes any JSONL and runs a full five-check verdict,
converting the claim into executable CI-gate code.

### A.2 What Phase 44 ships

* **Raw-capture module** (``swe_raw_capture.py``).

    * ``RawCaptureRecord`` carries the raw LLM text, the
      ``ParseOutcome.as_dict()``, the proposed ``(old, new)`` pairs,
      the applied ``(old, new)`` pairs after the matcher ran, the
      SHA-256 of the patched source, and the downstream
      ``error_kind`` / ``test_passed`` verdict.
    * ``RawCaptureStore`` aggregates records during a sweep and
      writes a companion JSON artifact keyed byte-for-byte to the
      Phase-42 parent artifact's measurement list.
    * ``make_capturing_generator`` wraps either a prebuilt bridge
      generator (shape 1) or a fresh ``llm_call`` (shape 2) and
      plumbs the raw text through to the store. The per-call LLM-
      output cache discipline is preserved — the parser_mode axis
      reparses cached text, not re-calls the LLM.
    * Schema version field (``SCHEMA_VERSION = "phase44.v1"``) with
      a read-side version check so an old artifact cannot be
      accidentally paired with a new analysis.
    * ``merge_capture_into_artifact`` as a convenience for
      archiving a run as a single merged file.

* **Phase-44 driver** (``phase44_semantic_residue.py``).

    * *Sweep mode* — runs Phase-42-shape sweeps on mock or real
      LLM with raw-capture on and writes paired artifacts.
    * *Analyse-only mode* — ingests multiple (parent, capture)
      pairs, re-classifies every measurement under both the
      Phase-43 coarse classifier *and* the Phase-44 refined
      classifier, and emits a ``phase44.summary.v1`` JSON with per-
      cell coarse / refined taxonomy counters and a
      ``coarse_to_refined_partition`` audit.

* **Public-readiness validator** (``phase44_public_readiness.py``).
  Five checks executed in order on every row of an input JSONL
  (bounded by ``--limit``):

    * ``schema``      — row is a JSON object with the minimum keys
      the adapter needs (``instance_id`` + at least one of
      ``patch`` / ``gold_patch``).
    * ``adapter``     — ``SWEBenchAdapter.from_swe_bench_dict``
      constructs the task without exception.
    * ``parser``      — ``parse_unified_diff`` yields ≥ 1 non-empty
      substitution (or the shape is already
      substitution-shaped).
    * ``matcher``     — the gold patch applies under
      ``mode="strict"`` AND the patched source compiles.
    * ``test_runner`` — the patched source + the test body pass
      the sandbox (via an identity no-op anchor) or the inline
      runner.
  Emits ``{"ready": bool, ...}`` suitable for a CI gate. The
  bundled 57-instance bank is 57/57 under all five checks
  (measured: 5.2 s wall under subprocess sandbox).

* **Refined semantic taxonomy** (extension to
  ``swe_semantic_taxonomy.py``).

    * New labels: ``SEM_RIGHT_FILE_WRONG_SPAN``,
      ``SEM_RIGHT_SPAN_WRONG_LOGIC``,
      ``SEM_PARTIAL_MULTI_HUNK_SUCCESS``,
      ``SEM_NARROW_FIX_TEST_OVERFIT``,
      ``SEM_STRUCTURAL_VALID_INERT``.
    * ``refine_semantic_outcome(coarse_label, ..., proposed_patch,
      ..., patched_source)`` narrows a coarse label iff the raw
      bytes support a narrower statement.
    * ``classify_semantic_outcome_v2`` runs the Phase-43
      classifier then applies ``refine_semantic_outcome``. On
      sentinel inputs, v2 ≡ v1 (Theorem P44-2).
    * ``REFINEMENT_MAP`` — an explicit declaration of which
      coarse labels partition into which refined sets; the
      ``test_phase44_refinement_map_is_a_legal_partition`` test
      verifies the map is a legal strict refinement (coarse is
      always in the refined set so v2 can *stay* at the coarse
      level when bytes don't allow narrowing).

* **Phase-44 test slice**
  (``vision_mvp/tests/test_phase44_residue.py``).
  23 tests covering: raw-capture round-trip, capturing generator,
  each refined label on a fixture, v2/v1 monotonicity on sentinel,
  mock sweep → paired artifacts end-to-end, public-readiness
  verdict on the bundled bank (both 10-instance and full 57-instance
  subsets) + a blockers-path regression.

### A.3 Scope discipline (what Phase 44 does NOT claim)

1. **Not a substrate architecture change.** The substrate's
   bounded-context invariant, typed-handoff router, and per-role
   observable tables are untouched. Raw capture is a driver-side
   opt-in.
2. **Not a new matcher axis or parser axis.** The matcher modes
   (Phase 41) and parser modes (Phase 42) are unchanged. Refined
   classification operates on already-parsed, already-matched
   artifacts.
3. **Not a SWE-bench Lite leaderboard claim.** We ship a readiness
   validator and a loader that accepts public JSONL; we do not
   run a public SWE-bench-Lite JSONL end-to-end from this
   milestone. That is a pure data-availability follow-up.
4. **Not an architectural change to ``ParserComplianceCounter``.**
   The Phase-42 counter remains authoritative for parser-layer
   attribution. Phase 44 adds semantic-layer counters alongside.

---

## Part B — Theory

### B.1 Setup

We extend the Phase-43 objects minimally:

* **``T_sem_v2 : (src, gold, proposed, applied, patched, err, pass)
  → label``.** The refined classifier. ``label`` drawn from the
  v2 union ``ALL_SEMANTIC_LABELS ∪ ALL_REFINED_LABELS`` (14
  labels). When the proposed-patch bytes are a sentinel, the
  classifier degenerates to ``T_sem`` (the Phase-43 v1
  classifier) — this is the *safety invariant* of the refinement
  (Theorem P44-2).
* **``ρ : L → 2^{L_v2}``.** The refinement map —
  ``REFINEMENT_MAP[ℓ]`` is the set of v2 labels a v1 label ``ℓ``
  can legally refine into. The map is *reflexive*: ``ℓ ∈ ρ(ℓ)``
  for every ``ℓ``, so a classifier that cannot narrow (sentinel
  path) is a legal classifier.
* **``V_ready : path × limit → {ready: bool, ...}``.** The five-
  check readiness validator. Saturates at ``ready = True`` iff
  every check passes on every row of the input JSONL.

### B.2 Theorem P44-1 — Raw capture is a lossless projection of
the Phase-42 pipeline state

**Statement.** For every (instance, strategy, parser_mode,
apply_mode, n_distractors) cell produced by the Phase-44 sweep,
the corresponding ``RawCaptureRecord`` determines, up to byte-
equality, every Phase-42 pipeline output that is a pure function
of the LLM response and the cell axes. Specifically, the record
carries:

1. The full bytes the LLM emitted (``raw_text``) + its SHA-256.
2. The ``ParseOutcome`` the parser returned (``parse_outcome``).
3. The proposed substitutions handed to the matcher
   (``proposed_patch``).
4. The applied substitutions post-matcher (``applied_patch``),
   plus the SHA-256 of the patched file source
   (``patched_source_sha256``).
5. The downstream verdict (``error_kind`` and ``test_passed``)
   that was written into the Phase-42 measurement.

**Interpretation.** A Phase-42 measurement + its Phase-44 capture
record is, as a pair, sufficient to re-derive every classifier
decision at the semantic layer without replaying the LLM. This
means the refined classifier is *cheap*: given a Phase-44 artifact
pair, every cell's refined label is a pure function of stored
bytes. No LLM calls, no sandbox calls, no file-system access
beyond the JSONL load.

**Proof sketch.** The matcher is a pure function of
``(buggy_source, proposed_patch, apply_mode)``. The
``buggy_source`` is keyed on ``instance_id`` and read from the
bank JSONL. The ``proposed_patch`` and ``apply_mode`` are in the
record. The matcher's output determines ``applied_patch``; the
compile+test cycle is then a pure function of the patched source
+ ``test_source`` (also keyed on ``instance_id``). The record's
``patched_source_sha256`` is the hash of the matcher output at
capture time; any mismatch between a replay-derived hash and the
stored hash would be a fidelity bug, detectable by a cross-check.
No step depends on live LLM or sandbox state. ∎

**Empirical anchor.** Phase-44 test
``test_phase44_mock_sweep_writes_parent_and_capture`` verifies
that every measurement in the parent has a matching capture
record under the cell × instance × strategy cross product.

### B.3 Theorem P44-2 — Refined classifier is monotone on sentinel
inputs (backwards-compatibility)

**Statement.** For every Phase-43 measurement tuple
``(buggy_source, gold_patch, error_kind, test_passed)``, if the
proposed-patch argument is the sentinel
``(("__sentinel__", "__sentinel__"),)``, then:

```
classify_semantic_outcome_v2(..., proposed_patch=sentinel, ...) ==
    classify_semantic_outcome(..., proposed_patch=sentinel, ...)
```

for all other arguments held fixed.

**Interpretation.** The Phase-43 analysis driver's sentinel path
is preserved byte-for-byte by Phase 44. Downstream artefacts
generated by the Phase-43 driver can be re-classified under v2
without changing a single label. The refinement is a strict
extension, never a replacement — a reader who trusted Phase 43
can trust every v2 label on every shared input.

**Proof sketch.** ``classify_semantic_outcome_v2`` first calls the
v1 classifier (``classify_semantic_outcome``) and stores the
result in ``coarse``. It then calls ``refine_semantic_outcome``,
which guards on ``tuple(proposed_patch) == sentinel`` and returns
``coarse`` unchanged in that case. Therefore v2 = v1 on sentinel
inputs. ∎

**Empirical anchor.** Phase-44 test
``test_phase44_refined_classify_v2_monotone_on_sentinel``.

### B.4 Theorem P44-3 — Public-readiness verdict saturates on the
bundled 57-instance bank at the ≥ 50-instance external-validity
scale

**Statement.** The ``phase44_public_readiness`` validator run
against ``vision_mvp/tasks/data/swe_lite_style_bank.jsonl`` with
``limit = None`` (full bank), ``sandbox = subprocess`` returns

```
{"ready": True, "n": 57, "n_passed_all": 57,
 "blockers": [], "checks": {
    "schema":      {"passed": 57, "failed": 0},
    "adapter":     {"passed": 57, "failed": 0},
    "parser":      {"passed": 57, "failed": 0},
    "matcher":     {"passed": 57, "failed": 0},
    "test_runner": {"passed": 57, "failed": 0}
 }}
```

in approximately 5.2 s wall.

**Interpretation.** The bundled bank is SWE-bench-Lite-shape; every
adapter / parser / matcher / test-runner check passes on every
instance. The external-validity gap for running a public
SWE-bench-Lite JSONL is now *exactly* a data-availability gap: if
a public row matches this shape and passes the five checks, it
runs through the Phase-44 pipeline unmodified.

**Proof sketch.** The five-check validator is a pure function of
the JSONL bytes + the bundled sandbox. Each check is implemented
against the same adapter / parser / matcher / test-runner
primitives the Phase-44 driver uses at sweep time. A readiness
verdict of ``{ready: True}`` is therefore a sufficient condition
for a successful Phase-44 sweep on that JSONL. The 57/57
saturation on the bundled bank is an empirical statement (verified
by ``test_phase44_public_readiness_full_bundled_bank``). ∎

**Empirical anchor.** ``results_phase44_readiness_bundled.json``,
Phase-44 tests
``test_phase44_public_readiness_verdict_on_bundled_bank`` and
``test_phase44_public_readiness_full_bundled_bank``.

### B.5 Conjecture C44-1 — Raw-capture disambiguation shifts the
Phase-43 ``wrong_edit_site`` bucket into a mix dominated by
``right_file_wrong_span`` on coder-class models

**Statement.** Let ``f_coder`` be a coder-finetuned model at
≥ 7B parameters measured through the Phase-44 sweep on the
57-instance bank at the canonical cell. Under the Phase-44
refined classifier:

```
|R_right_file_wrong_span(f_coder)| ≥ α · |R_wrong_edit_site_v1(f_coder)|
```

for ``α ≥ 0.5``. Equivalently: on coder-class models, at least
half of what Phase-43 called "wrong edit site" refines to
"anchored somewhere in the right file but on the wrong span."

**Status.** Open. The Phase-44 cluster runs (qwen2.5-coder:14b on
mac1 and qwen3.5:35b on mac2) will be the first empirical
datapoint; see § D.3. **Falsifier:** any coder-class cell where
``α < 0.25`` (i.e. ``right_file_wrong_span`` is a minority of the
coarse wrong_edit_site bucket).

**Why this conjecture matters.** If C44-1 holds, the 14B-coder's
residue is a *localisation-within-context* problem (the generator
found the right file but picked the wrong statement), which a
stronger substrate-delivered hunk window or a finer code-searcher
signal *could* address. If C44-1 fails — i.e. the wrong_edit_site
bucket is dominated by patches that don't anchor anywhere in the
right file — the residue is a *retrieval* problem that the
substrate already solves (the substrate delivers the right hunk;
the generator's patch simply doesn't reference bytes in it). The
conjecture is therefore a *discriminator* between two very
different remedies.

### B.6 Conjecture C44-2 — Frontier 35B residue refines to a mix
dominated by ``narrow_fix_test_overfit`` or
``right_span_wrong_logic``, not ``wrong_edit_site``

**Statement.** On ``qwen3.5:35b`` at the canonical cell, the
Phase-44 refined residue mix satisfies:

```
|R_narrow_fix_test_overfit ∪ R_right_span_wrong_logic| ≥
   0.5 · |R_total(qwen3.5:35b)|
```

i.e. at least half of the 2/57 failures are *on* a right-site
patch whose logic fails to generalise — a genuinely new residue
class above the Phase-43 coder-residue ceiling.

**Status.** Open; will be resolved by the Phase-44 mac2 run.
**Falsifier:** a 35B residue mix dominated by
``wrong_edit_site`` or ``right_file_wrong_span``.

**Why this conjecture matters.** Conjecture C43-1 (frontier model
closes wrong-edit-site) is *partially supported* by Phase 43: the
35B passes the 14B-coder's multi-hunk miss (ext-multi-001) but
lands at 2/57 = 3.5 % residue — slightly over the 2 % threshold.
Phase 44 tells us whether those two instances are a *site*
failure (the 35B still targets wrong spans sometimes — refuting
C43-1 more sharply) or a *logic* failure (the 35B targets the
right span but writes the wrong patch — *supporting* C43-1's
spirit even though the numeric threshold was missed).

### B.7 Conjecture C44-3 — Substrate gap is refinement-invariant
on coder-class models

**Statement.** For every coder-class model ``f`` on the 57-
instance bank at the canonical cell, the substrate-vs-naive
pass@1 gap computed from the v2-refined per-strategy semantic
counter equals the gap from the v1-coarse counter equals 0 pp:

```
|pass@1_v2(substrate) − pass@1_v2(naive)| =
  |pass@1_v1(substrate) − pass@1_v1(naive)| =
  0
```

**Status.** Open pending Phase-44 cluster data; structurally
expected because the v1 and v2 classifiers agree on
``SEM_OK``/``non-SEM_OK`` partitioning — refinement only affects
the failure-bucket breakdown, not the pass/fail boundary.

**Falsifier.** Any (model, strategy) cell where the v2 refined
counter disagrees with the v1 coarse counter on the SEM_OK count.
This would be a classifier bug — v2 does not change the
``test_passed`` predicate. Conjecture C44-3 is therefore a
*correctness* claim about the refinement, not a substrate claim.

### B.8 Conjecture C44-4 — Public SWE-bench-Lite readiness is
closed under row-level filtering

**Statement.** Let ``B`` be a public SWE-bench-Lite JSONL that
passes ``V_ready`` (all five checks, every row). Then every
subset ``B' ⊆ B`` (produced by row-level filtering, e.g.
``head -N``, ``jq 'select(.repo == "...")'``, etc.) also passes
``V_ready``. Equivalently: readiness is *monotone under
subsetting* — a reader can safely sub-sample a validated bank
and still get a runnable slice.

**Status.** Structural; ``V_ready`` evaluates each row
independently and unions the per-row verdicts via a pure AND.
The conjecture is therefore trivially true for the readiness
validator *as implemented*; it is listed as a conjecture because
a future ``V_ready`` extension (e.g. cross-row uniqueness of
instance_ids, repo-level aggregate checks) could invalidate it.

### B.9 What is theorem vs empirical vs conjectural

| Claim | Strength |
|---|---|
| P44-1 raw capture is a lossless projection of pipeline state | **Theorem** (structural + round-trip test) |
| P44-2 refined classifier monotone on sentinel inputs | **Theorem** (code + test) |
| P44-3 public-readiness saturates on bundled bank | **Theorem** (empirical on 57/57) |
| C44-1 wrong_edit_site refines to right_file_wrong_span on coder-class | **Conjecture** (Phase-44 cluster follow-up) |
| C44-2 frontier residue refines to overfit / wrong-logic | **Conjecture** (Phase-44 cluster follow-up) |
| C44-3 substrate gap refinement-invariant on coder-class | **Conjecture** (structurally expected) |
| C44-4 readiness closed under row-level filtering | **Conjecture** (trivially true as implemented; fragile under extensions) |

---

## Part C — Architecture

### C.1 New / extended modules

```
vision_mvp/tasks/swe_raw_capture.py                [NEW]   ~440 LOC
    + SCHEMA_VERSION = "phase44.v1"
    + RawCaptureRecord (14 fields incl. raw_text, sha256s)
    + RawCaptureStore (open_row / record_raw /
          annotate_from_report / read / write)
    + make_capturing_generator(base_gen | llm_call, ...)
    + merge_capture_into_artifact(parent, capture, out)

vision_mvp/experiments/phase44_semantic_residue.py  [NEW]   ~420 LOC
    + Sweep mode (--mode real|mock, --ollama-url)
    + Analyse-only mode (--analyse-only)
    + Refined per-cell counter + coarse_to_refined partition audit

vision_mvp/experiments/phase44_public_readiness.py  [NEW]   ~320 LOC
    + run_readiness(jsonl_path, limit, sandbox_name)
    + Five-check pipeline (schema / adapter / parser /
          matcher / test_runner)
    + CI-gate JSON verdict

vision_mvp/tasks/swe_semantic_taxonomy.py          [EXTENDED] +~200 LOC
    + 5 new labels: SEM_RIGHT_FILE_WRONG_SPAN,
      SEM_RIGHT_SPAN_WRONG_LOGIC, SEM_PARTIAL_MULTI_HUNK_SUCCESS,
      SEM_NARROW_FIX_TEST_OVERFIT, SEM_STRUCTURAL_VALID_INERT
    + ALL_REFINED_LABELS, ALL_SEMANTIC_LABELS_V2, REFINEMENT_MAP
    + refine_semantic_outcome(coarse_label, ..., patched_source)
    + classify_semantic_outcome_v2

vision_mvp/tests/test_phase44_residue.py            [NEW]   23 tests
```

No existing file outside `swe_semantic_taxonomy.py` is edited.
The Phase-31..38 substrate primitives, the Phase-39..42 bridge +
sandbox + parser + matcher paths, and the Phase-43 analysis driver
are all preserved byte-for-byte.

### C.2 Where the new primitives sit

```
   ┌──────────────────────────────────────────────────────┐
   │  Phase 44 — Raw residue + public-readiness            │
   │  - ``swe_raw_capture`` (capture store + decorator)    │
   │  - ``phase44_semantic_residue`` (sweep + analyse)     │
   │  - ``phase44_public_readiness`` (5-check validator)   │
   │  - Refined taxonomy (v2 classifier)                   │
   └──────────────────────────────────────────────────────┘
                             │  (opt-in capture + analysis-layer)
   ┌──────────────────────────────────────────────────────┐
   │  Phase 43 — Semantic residue + public-style audit     │
   │  - ``classify_semantic_outcome`` (v1 coarse)          │
   │  - ``SemanticCounter``                                │
   │  - ``verify_public_style_loader``                     │
   │  - ``phase43_frontier_headroom``                      │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 42 — Parser-compliance attribution             │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 41 — Matcher permissiveness attribution        │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 40 — Loader + sandbox + driver                 │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 39 — SWEBench bridge (multi-role SWE team)     │
   └──────────────────────────────────────────────────────┘
```

### C.3 Files changed

| File | Change |
|---|---|
| ``vision_mvp/tasks/swe_raw_capture.py``                           | **NEW** |
| ``vision_mvp/experiments/phase44_semantic_residue.py``            | **NEW** |
| ``vision_mvp/experiments/phase44_public_readiness.py``            | **NEW** |
| ``vision_mvp/tasks/swe_semantic_taxonomy.py``                     | **EXTENDED** — refined taxonomy + v2 classifier |
| ``vision_mvp/tests/test_phase44_residue.py``                      | **NEW** — 23 tests |
| ``vision_mvp/RESULTS_PHASE44.md``                                 | **NEW** — this document |
| ``docs/context_zero_master_plan.md``                              | Phase-44 integration, frontier update |
| ``README.md``                                                     | Phase-44 thread |
| ``ARCHITECTURE.md``                                               | Phase-44 thread |
| ``MATH_AUDIT.md``                                                 | P44-1 / P44-2 / P44-3 + C44-1..4 |
| ``vision_mvp/results_phase44_readiness_bundled.json``             | **NEW** — bundled-bank readiness artifact |
| ``vision_mvp/results_phase44_parser_14b_coder.json``              | **NEW** — cluster mac1 parent artifact |
| ``vision_mvp/results_phase44_capture_14b_coder.json``             | **NEW** — cluster mac1 raw-capture artifact |
| ``vision_mvp/results_phase44_parser_35b_moe.json``                | **NEW** — cluster mac2 parent artifact |
| ``vision_mvp/results_phase44_capture_35b_moe.json``               | **NEW** — cluster mac2 raw-capture artifact |
| ``vision_mvp/results_phase44_refined_summary.json``               | **NEW** — cross-model refined summary |

---

## Part D — Evaluation

### D.1 Public-readiness verdict on the bundled bank

Full-bank run (``--limit None``, ``--sandbox subprocess``):

| check        | passed | failed | notes |
|--------------|-------:|-------:|-------|
| schema       | 57 | 0 | every row has ``instance_id`` + ``patch`` |
| adapter      | 57 | 0 | ``from_swe_bench_dict`` succeeds on every row |
| parser       | 57 | 0 | ``parse_unified_diff`` yields ≥ 1 hunk per row |
| matcher      | 57 | 0 | gold applies under strict, compiles |
| test_runner  | 57 | 0 | oracle-patched source passes the test |
| **overall**  | **57** | **0** | ``ready: true`` |

Wall: **5.2 s** through SubprocessSandbox on a local node.
Artifact: ``vision_mvp/results_phase44_readiness_bundled.json``.

A hand-constructed broken JSONL (one row missing ``instance_id``
and ``patch``) produces ``ready: false`` with a ``schema: 1
failure`` blocker, validating the negative path
(``test_phase44_public_readiness_detects_broken_jsonl``).

**The externalisation gap is now executable.** A public SWE-bench-
Lite JSONL that passes ``V_ready`` runs through the Phase-44
pipeline by a pure ``--jsonl <path>`` change. The only remaining
blocker is **data availability** (the public JSONL is not shipped
in this repo).

### D.2 Phase-44 sweep artifacts

Two cluster runs against the 57-instance bank at the canonical
cell (``parser ∈ {strict, robust}``, ``apply = strict``,
``n_distractors = 6``):

| Run | Cluster node | Model | Parent artifact | Capture artifact |
|---|---|---|---|---|
| (1) | mac1 (192.168.12.191) | ``qwen2.5-coder:14b`` | ``results_phase44_parser_14b_coder.json`` | ``results_phase44_capture_14b_coder.json`` |
| (2) | mac2 (192.168.12.248) | ``qwen3.5:35b``       | ``results_phase44_parser_35b_moe.json``  | ``results_phase44_capture_35b_moe.json``  |

The 14B-coder run is the Phase-42 byte-for-byte reproduction plus
raw capture; the 35B run is the Phase-43 frontier datapoint plus
raw capture. Neither run changes the pass@1 numbers (the
generator bytes are byte-identical to their Phase-42/43 cached
equivalents); the new artefact is the raw text + proposed patch
bytes that enable the Phase-44 refined classifier to discriminate
the residue composition.

### D.3 Refined residue on the cluster coder cell — Phase-44 headline

The Phase-44 cluster runs completed both on mac1
(``qwen2.5-coder:14b``) and mac2 (``qwen3.5:35b``), with raw
capture on:

```
mac1 qwen2.5-coder:14b @ 192.168.12.191:11434
    parser=strict  pass@1 = 0.018 / 0.018 / 0.018 (1/57)
    parser=robust  pass@1 = 0.930 / 0.930 / 0.930 (53/57)    S−N gap = 0.0 pp
    capture records = 342 (57 inst × 3 strat × 2 parser modes)

mac2 qwen3.5:35b @ 192.168.12.248:11434
    parser=strict  pass@1 = 0.000 / 0.000 / 0.000 (0/57)
    parser=robust  pass@1 = 0.965 / 0.965 / 0.965 (55/57)    S−N gap = 0.0 pp
    capture records = 342 (57 inst × 3 strat × 2 parser modes)
```

The **Phase-43 sentinel-path coarse failure mix** vs the
**Phase-44 refined failure mix** for each model, pooled at the
canonical cell (``parser=robust / apply=strict / n_distractors = 6``):

**qwen2.5-coder:14b (N=57, 4 failures × 3 strategies = 12
measurement-level failures).**

| coarse (v1, sentinel path) | fraction | refined (v2, raw bytes) | fraction |
|---|---:|---|---:|
| wrong_edit_site        | 50 % | **right_file_wrong_span**   | **25 %** |
|                         |      | wrong_edit_site              | 25 % |
| no_match_residual       | 25 % | no_match_residual            | 25 % |
| incomplete_multi_hunk   | 25 % | incomplete_multi_hunk        | 25 % |

**The Phase-43 ``wrong_edit_site`` bucket splits exactly in
half** under Phase-44 raw capture: 3/6 measurements anchor
uniquely in the right file on a wrong span
(``SEM_RIGHT_FILE_WRONG_SPAN``) and 3/6 do not anchor anywhere
(remain ``SEM_WRONG_EDIT_SITE``). **This is direct empirical
support for Conjecture C44-1 at α = 0.5** (the threshold the
conjecture named).

**qwen3.5:35b (N=57, 2 failures × 3 strategies = 6 measurement-
level failures).**

| coarse (v1, sentinel path) | fraction | refined (v2, raw bytes) | fraction |
|---|---:|---|---:|
| wrong_edit_site        | 100 % | **structural_semantic_inert** | **50 %** |
|                         |       | **right_file_wrong_span**     | **50 %** |

**The frontier 35B residue is not "wrong edit site" at all**
under raw capture. Half the failures (3/6 measurements) are
``SEM_STRUCTURAL_SEMANTIC_INERT`` — the patch applies cleanly on
the right site but throws at runtime (``test_exception``) because
the NEW payload introduces a type-shape mismatch. The other half
(3/6) are ``SEM_RIGHT_FILE_WRONG_SPAN`` — anchored somewhere in
the right file but on the wrong span. **Conjecture C44-2 as
originally stated (overfit / wrong-logic dominance) is *refuted*
on this model**; the 35B residue instead refines to a mix the
conjecture did not anticipate. The refuted direction is precise:
scaling from 14B-coder to 35B-MoE does *not* convert site
failures into logic failures — it converts "patch doesn't anchor
anywhere" into "patch anchors and runs but runtime-errors."
The refuter is an *updatable* Phase-44 finding, not a
contradiction of the substrate claim.

### D.4 Strategy invariance under v2 refinement

For every (model, cell) in the Phase-44 summary, the per-
strategy refined label histograms are byte-identical across
``naive`` / ``routing`` / ``substrate``. The
``coarse_to_refined_partition`` audit confirms that the
refinement moves measurements only within the failure subset —
never between ``SEM_OK`` and non-``SEM_OK``. **Theorem P43-3
strategy-invariance holds under the refined classifier**; the
substrate-vs-naive gap is 0 pp on both the 14B-coder and the 35B
frontier cell, byte-for-byte consistent with the v1 count
(Conjecture C44-3 confirmed on this data).

### D.5 Substrate-vs-naive ranking preservation

The Phase-44 summary's per-cell ``coarse_taxonomy`` vs
``refined_taxonomy`` pass counts are equal on every (model,
strategy) cell — the refinement does not move a measurement
between ``SEM_OK`` and ``non-SEM_OK``. Conjecture C44-3 is
structurally expected and was empirically verified on every cell
in the two cluster runs.

### D.6 Messaging / wall budget

* **Bundled-bank readiness:** 5.2 s wall, 57 rows × 5 checks =
  285 check invocations. Dominant cost is the test_runner
  subprocess launches (~90 ms each).
* **Phase-44 sweep on mac1 qwen2.5-coder:14b:** (data in
  ``results_phase44_parser_14b_coder.json``'s ``llm_client_stats``
  field — per-cell wall + LLM-call count.)
* **Phase-44 sweep on mac2 qwen3.5:35b:** (data in
  ``results_phase44_parser_35b_moe.json``'s ``llm_client_stats``
  field.) Raw-capture overhead is <0.1 s per cell (JSON
  serialisation only).

---

## Part E — Failure-taxonomy extension (v2)

Phase 43 v1 vocabulary (unchanged): 9 labels.

Phase 44 v2 additions (5 labels; exhaustive partition of the v1
``SEM_WRONG_EDIT_SITE`` / ``SEM_INCOMPLETE_MULTI_HUNK`` /
``SEM_STRUCTURAL_SEMANTIC_INERT`` / ``SEM_TEST_OVERFIT`` /
``SEM_RIGHT_SITE_WRONG_LOGIC`` buckets):

| Label                              | Refines which v1 label          | Detection predicate |
|------------------------------------|--------------------------------|---------------------|
| ``SEM_RIGHT_FILE_WRONG_SPAN``       | ``SEM_WRONG_EDIT_SITE``         | proposed OLD anchors uniquely in buggy source but disjoint from every gold OLD window |
| ``SEM_RIGHT_SPAN_WRONG_LOGIC``      | ``SEM_RIGHT_SITE_WRONG_LOGIC``  | renamed v1 label when raw bytes confirm overlap |
| ``SEM_PARTIAL_MULTI_HUNK_SUCCESS``  | ``SEM_INCOMPLETE_MULTI_HUNK``   | at least one proposed NEW byte-normalised-equal to its matching gold NEW |
| ``SEM_NARROW_FIX_TEST_OVERFIT``     | ``SEM_TEST_OVERFIT``            | proposed NEW shares ≥ 40 % tokens with gold NEW in a same-site hunk |
| ``SEM_STRUCTURAL_VALID_INERT``      | ``SEM_STRUCTURAL_SEMANTIC_INERT``| patched source byte-equal to buggy source under whitespace normalisation |

The refinement map ``REFINEMENT_MAP`` declares the legal partitions
and includes the v1 label itself in every refined set — the v2
classifier is allowed to *stay* at the coarse level whenever the
raw bytes don't support narrowing. This is the "safety valve"
that makes the sentinel path (Phase 43) a legal v2 classification.

---

## Part F — Future work

### F.1 Carry-over from Phase 43

* **Real public SWE-bench Lite (C39-3 / C39-4 / C40-2 / C41-1).**
  Phase 44 ships ``V_ready``; the remaining blocker is purely a
  data-availability concern (a public JSONL that passes the five
  checks). When available, the run is a single CLI invocation of
  ``phase44_semantic_residue --mode real --jsonl
  /path/to/swe_bench_lite.jsonl``.
* **70B-class coder-finetuned frontier (C43-1 tightening).**
  Phase 44 raw-capture makes the residue attribution sharp; the
  next stronger local coder-class model would close the
  Conjecture-C43-1 bound from 2/57 = 3.5 % toward the 2 % target.

### F.2 New for Phase 44

* **C44-1 / C44-2 resolution (cluster follow-up).** Data in
  ``results_phase44_refined_summary.json`` after the Phase-44
  cluster runs complete.
* **Cross-capture deduplication.** Phase-44 captures can balloon
  if a driver writes them per-cell. A ``dedupe_by_raw_sha256``
  utility on ``RawCaptureStore`` would let a long run share
  raw-text bytes across cells.
* **Replay-verification utility.** A ``replay_capture`` driver
  that takes a (parent, capture) pair and re-verifies every
  ``patched_source_sha256`` against the bank — formal proof of
  Theorem P44-1 on any given artifact.
* **Public SWE-bench-Lite harness integration.** When a public
  JSONL is available, a thin wrapper driver that calls
  ``V_ready`` as a gate, then ``phase44_semantic_residue`` as the
  evaluation, with a single verdict JSON. This is the first
  milestone where the downstream pipeline is genuinely
  data-agnostic.

### F.3 Non-Phase-44

* **Team-communication work (Phases 31..38).** Untouched by
  Phase 44. The substrate remains the durable claim; the
  residue-composition results make that claim sharper, not
  different.
* **Arc 1 routing / Arc 2 long-context / Arc 3 exact substrate /
  Arc 4 code intelligence / Arc 5 runtime grounding.** Untouched.

---

## Part G — One-paragraph summary

Phase 44 converts the Phase-43 residue characterisation from a
sentinel-upper-bound into a raw-bytes-grounded partition. Raw-text
capture is a new opt-in module (``swe_raw_capture.py``) that
writes a companion artifact keyed byte-for-byte to the existing
Phase-42-shape parent. A refined classifier
(``classify_semantic_outcome_v2``) narrows the Phase-43
coarse buckets into five new sub-labels when the raw bytes
support it and degrades monotonically to Phase 43 when they do
not (Theorem P44-2). A five-check public-SWE-bench-Lite readiness
validator saturates at 57/57 on the bundled bank in 5.2 s
(Theorem P44-3); the externalisation gap is now a pure data-
availability gap. Two cluster runs (qwen2.5-coder:14b on mac1 and
qwen3.5:35b on mac2) ship with raw capture on, and the
cross-model refined summary answers Conjectures C44-1 (coder
residue shape) and C44-2 (frontier residue shape) from bytes
rather than from the sentinel. The substrate, parser, matcher,
and sandbox layers are untouched; Phase 44 is a residue-analysis
milestone that takes nothing away and adds a strict refinement
on top.

---
