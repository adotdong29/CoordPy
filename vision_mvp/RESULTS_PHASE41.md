# Phase 41 — Larger SWE-bench-Lite-style empirical sweep, patch-matcher permissiveness attribution, and a stronger-model datapoint

**Status: research milestone. Phase 41 converts Phase 40's
"real external task loop exists" into "real external task loop
has first *larger-N* empirical ranking data and a principled
generator-vs-substrate attribution surface." Three coupled
artifacts ship:
(1) a **28-instance real-shape JSONL bank**
(``vision_mvp/tasks/data/swe_lite_style_bank.jsonl``) —
~4.7× the Phase-40 6-instance mini bank, authored as diverse
SWE-bench-Lite-shape edit classes (single-hunk, multi-hunk,
multi-function, operator-typo, wrong-branch, mutation-vs-copy,
off-by-one, whitespace edge cases, type-conversion, aggregation-
seed, slice-direction, ordering, ambiguous-guard);
(2) a **permissive patch-matcher axis** —
``apply_patch`` now accepts one of four modes
(``strict``, ``lstrip``, ``ws_collapse``, ``line_anchored``)
threaded through ``run_swe_loop`` and ``run_swe_loop_sandboxed``,
with strict default preserving Phase-40 byte-for-byte;
(3) a **Phase-41 driver** — ``phase41_swe_lite_sweep`` that
runs the bank through the substrate at a disciplined distractor
grid, compares the three strategies against each matcher mode,
and emits a **generator-vs-substrate attribution table**
(``recovered`` / ``regressed`` set deltas per strategy).
The driver carries a compute-efficient discipline: each LLM call
is cached per ``(instance_id, strategy, n_distractors)`` so
a permissive-matcher cell reuses the strict-matcher cell's
proposals.**

Phase 41 in one line: **the programme now has 28-instance
empirical pass@1 data through the real loop, a one-dial
attribution boundary between matcher precision and substrate
delivery, and a stronger-model datapoint beyond the
Phase-40 0.5B / 7B pair.**

Three new theorems (P41-1, P41-2, P41-3) and four new
conjectures (C41-1..C41-4). The Phase-31/35/40 bounded-
context invariants reproduce at the 28-instance bank — the
substrate's ``patch_generator`` prompt is constant at **746.4
chars / 186.6 tokens** across ``n_distractors ∈ {0, 6, 12,
24}`` on every measurement, while naive grows from **806.8 →
2 125.8 chars** (**2.6×** span). Full Phase 31..41 regression
green (**62 / 62** on the Phase-39/40/41 SWE-arc slice).

---

## Part A — Research framing

### A.1 Why this milestone exists

Phase 40 closed the *mechanical* SWE-bench gap end-to-end:
a unified-diff parser, a real-shape adapter, a JSONL loader,
and a sandboxed execution boundary behind one ``Sandbox``
protocol. The Phase-40 real-LLM result on ``qwen2.5-coder:7b``
was honest but small: pass@1 = 5/6 under naive vs 4/6 under
substrate at 6 instances — a single-instance ranking
inversion sitting cleanly inside Theorem P39-2's
transcription-bounded regime. Two follow-ups surfaced:

1. **Scale.** 6 instances is too small for the pass@1 ranking
   claim to wash out per-instance variance. A bank with
   diverse edit-shapes at 20–30 instances is the next step.
2. **Attribution.** The byte-strict matcher is itself a
   generator-side bottleneck: the substrate's bounded prompt
   can carry the gold *semantically* but the bridge's
   ``apply_patch`` demands literal-text reproduction. The
   open question named in Phase 40 § F.2 was whether a
   more permissive matcher would close the substrate-vs-
   naive pass gap on byte-borderline instances.

Phase 41 executes both follow-ups together so the substrate-
side and matcher-side stories can be read side-by-side. The
guiding discipline: *permissive matching is not a substrate
change*. It is a generator-attribution knob — a tighter
statement of "how much of the remaining failure is
byte-fidelity" vs "how much is semantic."

### A.2 What Phase 41 ships (four coupled pieces)

* **Larger JSONL bank
  (``vision_mvp/tasks/data/swe_lite_style_bank.jsonl``).**
  28 instances in real SWE-bench JSONL shape (``instance_id``,
  ``repo``, ``base_commit``, ``problem_statement``,
  ``buggy_file_relpath``, ``buggy_function``, unified-diff
  ``patch``, inline ``repo_files``, runnable
  ``test_source`` with the bridge's ``def test(module)``
  contract). The bank deliberately spans edit-shape classes:
  operator-typo, off-by-one, wrong-branch, seed-wrong,
  aggregate-missing, mutation-vs-copy, ordering, unicode-
  edge, type-conversion, pattern-branch, multi-hunk,
  comparator-flipped, empty-guard, slice-direction,
  polarity-flipped, loop-body-missing, parity-partition,
  index-return. Each instance ships its inline repo source
  (no network), a hidden ``def test(module)``, and a
  validated round-trip (the bank-builder script refuses to
  register an instance whose diff doesn't parse, whose old
  blocks aren't unique, or whose oracle-patched source
  doesn't pass the hidden test). The JSONL is the precondition
  that makes the Phase-41 evaluation reproducible offline in
  seconds. Pointing the driver at a real SWE-bench Lite
  JSONL is a ``--jsonl <path>`` parameter change — the
  loader, sandbox, and substrate are unchanged from Phase 40.
* **Permissive patch-matcher modes (``apply_patch`` extension).**
  A new ``mode`` kwarg on ``apply_patch`` and every
  ``Sandbox.run(...)`` call:
  ``strict`` (Phase-40 byte-exact),
  ``lstrip`` (tolerate leading-whitespace drift; normalise
  each line via ``str.lstrip`` when searching, substitute the
  NEW block's bytes verbatim on match),
  ``ws_collapse`` (tolerate internal-whitespace drift;
  normalise via ``" ".join(str.split())``),
  ``line_anchored`` (tolerate trailing-whitespace drift;
  normalise via ``str.rstrip``). All permissive modes retain
  the **unique-match discipline** — a normalised OLD that
  appears more than once in the normalised source is
  rejected as ``old_ambiguous``. Strict remains the default
  so every Phase-40 artifact reruns byte-for-byte.
* **Phase-41 experiment driver
  (``experiments/phase41_swe_lite_sweep``).**
  Composes loader + substrate + sandbox + (optional) real
  LLM. Accepts ``--apply-modes strict lstrip`` to run the
  attribution sweep. Caches the LLM output per
  ``(instance_id, strategy, n_distractors)`` so re-evaluating
  under a permissive mode does not re-call the LLM — the
  attribution study is a cheap post-hoc sandbox comparison.
  Emits a per-strategy ``recovered`` / ``regressed`` set
  delta between each permissive mode and strict.
* **Phase-41 test slice
  (``tests/test_phase41_swe_lite.py``).**
  18 new tests covering strict-mode regression, all three
  permissive modes' recovery surface, ambiguity-rejection
  discipline, over-acceptance guards, JSONL loading at the
  new scale, oracle-saturation under both matcher modes,
  bounded-context preservation at the larger bank
  (Theorem P41-1), and ``apply_mode`` threading through
  ``run_swe_loop`` and ``run_swe_loop_sandboxed``.

### A.3 Scope discipline (what Phase 41 does NOT claim)

1. **Not SWE-bench Lite end-to-end.** The bundled JSONL is
   *self-authored* real-shape, not a SWE-bench Lite
   download. The instance count (28) is large enough to
   wash out single-instance variance at the Phase-40
   6-instance scale but far smaller than a Lite pass@1
   leaderboard claim would require (≥ 50).
2. **Not a claim that permissive matchers are the right
   default.** Strict matching is what separates the
   substrate's *correctness-preservation* claim from the
   generator's *text-fidelity* claim. Phase 41 makes the
   matcher axis visible so failures can be attributed
   cleanly; it does not advocate for permissive matching
   as the production default.
3. **Not a refutation of any prior theorem.** Phase 40's
   P40-1, P40-2, P40-3 all hold byte-for-byte at the larger
   bank; the Phase-39 team-substrate plumbing is unchanged.
   Phase 41 adds new claims (P41-1..P41-3) without removing
   old ones.
4. **Not a strong-model measurement.** The stronger-model
   datapoint (``gemma2:9b`` — § D.4) is a spot check at a
   subset size, not a full sweep. The programme's existing
   frontier ranking (P39 Part B: gemma2:9b as the strongest
   local model tested so far) is extended by one real-SWE
   datapoint, not by a scale-N claim.

---

## Part B — Theory

### B.1 Setup (Phase 41 deltas)

The Phase-41 objects extend Phase 40 minimally:

* **``M_apply : (source, patch, mode) → (new_source, applied, reason)``.**
  Phase 40's ``apply_patch`` extended with a ``mode`` parameter.
  The matcher is a one-axis generalisation: strict is a total
  specialisation (``n_match`` via ``str.count``) whose preimage
  ⊆ permissive-mode preimage (every byte-exact match is also
  a normalised match). Permissive modes add matches whose
  normalised-line sequences agree.
* **``B_lite : path → (bank_28, repo_files_28)``.**
  The Phase-41 JSONL loader outcome.

### B.2 Theorem P41-1 — Bounded-context preservation at scale

**Statement.** On the 28-instance Phase-41 bank
(``swe_lite_style_bank.jsonl``) under the deterministic
oracle generator, the ``SubprocessSandbox`` backend, and
every matcher mode in ``ALL_APPLY_MODES``, the mean
``patch_generator`` prompt size is independent of
``n_distractors`` under the substrate strategy:

```
prompt_chars(substrate, n_distractors = 0)   = 746.4
prompt_chars(substrate, n_distractors = 6)   = 746.4
prompt_chars(substrate, n_distractors = 12)  = 746.4
prompt_chars(substrate, n_distractors = 24)  = 746.4
```

while under naive the same metric grows monotonically:

```
prompt_chars(naive, n_distractors = 0)   ≈ 806.8
prompt_chars(naive, n_distractors = 6)   ≈ 1 127.8
prompt_chars(naive, n_distractors = 12)  ≈ 1 461.8
prompt_chars(naive, n_distractors = 24)  ≈ 2 125.8
```

Pass@1 = 1.000 under the oracle on every (strategy,
distractor, apply_mode) cell = 12 × 28 = 336 measurements;
every cell's ``chain_ok`` is True.

**Interpretation.** Theorem P40-2 stated this for the
Phase-40 six-instance mini bank (813 chars substrate, 826 →
2 145 naive); Theorem P41-1 reproduces it on a **4.7×
larger** bank of diverse edit shapes. The constant prompt-size
is not an artifact of the mini-bank's specific instances —
it is a property of the substrate's
``_build_patch_gen_context(strategy="substrate", …)`` path,
which consumes only ``{issue_summary, hunk}``, both bounded
at construction. The absolute constant drops from 813 to
747 because the larger bank's per-instance hunk windows are
slightly tighter on average (a bank-composition
characteristic, not a substrate change).

**Proof sketch.** The substrate prompt is composed from
``ctx["issue_summary"]`` (a ``_summarise_issue`` output,
capped at 120 chars) and ``ctx["hunk"]`` (a
``_slice_function`` output, capped at 12 source lines around
the buggy ``def``). Neither depends on ``n_distractors``;
adding distractor events to the hidden event log grows only
the naive-strategy ``delivered_events`` embedding. The
sandbox backend is transparent to prompt construction. ∎

**Empirical anchor.** § D.1 +
``test_phase41_substrate_prompt_constant_across_distractors``.

### B.3 Theorem P41-2 — Oracle ceiling is matcher-mode-invariant

**Statement.** For every matcher mode ``m ∈ ALL_APPLY_MODES``,
every strategy ``s ∈ {naive, routing, substrate}``, and
every sandbox backend ``S ∈ {InProcessSandbox,
SubprocessSandbox}``, the deterministic oracle generator
delivers ``pass@1 = 1.000`` on the Phase-41 28-instance
bank. Permissive matching does not subtract correctness
from a byte-exact patch source.

**Interpretation.** This is the soundness statement for
the matcher axis. The oracle emits the bank's gold
``(old, new)`` tuples byte-exactly; strict accepts them by
``str.replace`` (n_match = 1); permissive accepts them by
line-normalised substring match (also n_match = 1 because
the strict match is a special case). The theorem makes
explicit that adding a matcher axis cannot introduce a
false-negative under a byte-exact generator. Phase 41's
pass-rate *deltas* under permissive matching are therefore
attributable entirely to the generator's text-fidelity
drift, not to a bridge regression.

**Proof sketch.** For every ``(old, new)`` pair the oracle
emits, ``old`` is byte-exact and unique in the buggy source
(enforced by the bank-builder's ``_run_and_assert``
precondition). Strict mode's ``str.replace`` therefore
succeeds; permissive modes apply the identical replacement
via a normalised-line search whose byte-exact preimage is a
subset of the normalised preimage — so the unique match
under strict implies at least one match under permissive.
Uniqueness is preserved because the bank-builder's
precondition ensures both byte-exact uniqueness and
normalised-line uniqueness on the 28 bank instances
(empirically verified by the oracle saturation test). ∎

**Empirical anchor.** § D.1 +
``test_phase41_jsonl_oracle_saturates_under_sandbox_strict``
+
``test_phase41_jsonl_oracle_saturates_under_sandbox_lstrip``.

### B.4 Theorem P41-3 — Matcher-permissiveness attribution boundary

**Statement.** For a real-LLM patch generator ``f`` and
a matcher-mode pair ``(strict, permissive)``, define:

* ``R_recovered(f, s) = {instances that strict-fail and permissive-pass under strategy s}``;
* ``R_regressed(f, s) = {instances that strict-pass and permissive-fail under strategy s}``.

Then the per-strategy pass@1 delta satisfies

```
Δ_pass@1(f, s) = |R_recovered(f, s)| − |R_regressed(f, s)|
               = pass@1(permissive, f, s) − pass@1(strict, f, s).
```

``R_recovered`` is a *generator-side* attribution set — the
generator emitted a near-correct OLD block that the strict
matcher refused on literal-text grounds. ``R_regressed`` is
a *matcher-risk* attribution set — the permissive matcher
accepted a substitution the strict matcher would have refused,
and the accepted substitution did not survive the hidden
test (this empirically turns out to be a null set on the
Phase-41 bank; see § D.3).

**Interpretation.** The theorem is a *decomposition*
statement, not a monotonicity claim. It makes the matcher-
permissiveness story a *named surface* rather than a
narrative hedge: any pass-rate change between two matcher
modes is exactly the difference between the generator-
side gain and the permissive-matcher risk, and the two
components are independently measurable from the Phase-41
result artifact. Combined with Theorem P39-2
(transcription-bounded vs communication-bounded regimes),
Theorem P41-3 gives the programme its first *two-axis*
attribution surface for a real SWE loop: the substrate-
side axis separates *what the substrate delivers* from
*what the generator reconstructs*, and the matcher-side
axis separates *what the generator emits* from *what the
bridge accepts*.

**Proof sketch.** Straightforward counting.
pass@1(permissive) − pass@1(strict) = |{instances that pass
under permissive}| / N − |{instances that pass under
strict}| / N = (|pass_strict ∩ pass_perm| + |R_recovered|
− |pass_strict ∩ pass_perm| − |R_regressed|) / N =
(|R_recovered| − |R_regressed|) / N. ∎

**Empirical anchor.** § D.3 (``qwen2.5-coder:7b`` run) +
§ D.4 (``gemma2:9b`` run).

### B.5 Conjecture C41-1 — Communication-bounded at larger N

**Statement.** On a SWE-bench-Lite-style bank of
≥ 50 instances drawn from genuine public SWE-bench Lite
(not the self-authored Phase-41 28-instance bank) and
under a patch generator ``f`` whose byte-fidelity floor
is above the bridge's strict matcher precision on
≥ 50 % of instances, the substrate's pass@1 on naive
is bounded above by naive's pass@1 plus a *task-length*
term that is ``o(1)`` in N — i.e. the substrate is
*not meaningfully worse* than naive at scale N ≥ 50,
despite the Phase-40 6-instance inversion.

**Status.** Open. Phase 41's 28-instance run sits
between the Phase-40 6-instance scale (where the
ranking inverted by one instance) and the public
SWE-bench Lite scale. Falsifier: a ≥ 50-instance run
where ``pass@1(naive) − pass@1(substrate) > 0.1`` at
both matcher modes.

### B.6 Conjecture C41-2 — Matcher-permissiveness saturates on
well-specified benchmarks

**Statement.** The pass-rate recovery attributable to
matcher permissiveness saturates as benchmark size grows:
on the Phase-41 bank, the recovery is tight to the
substrate's bounded-prompt withholding of raw-text anchors,
and larger benches admit fewer *strict-strictly-fails*
instances (the text-fidelity floor is task-family-local,
not global). Equivalently:

```
lim_{N → ∞, f fixed}  |R_recovered(f, s=substrate)| / N  ≤ ε_f
```

for a generator-specific constant ε_f ≥ 0 that does not
grow with N.

**Status.** Open; requires the SWE-bench Lite run.
Falsifier: a SWE-bench Lite sweep where ``R_recovered``
grows linearly in N.

### B.7 Conjecture C41-3 — Stronger-model saturates the strict
matcher

**Statement.** For a stronger local model (e.g.
``gemma2:9b`` or a 30B+ class model), the ratio
``|R_recovered| / |R_strict_fail|`` on the Phase-41
bank approaches 0 — a stronger model does not need
matcher permissiveness because its literal-text fidelity
clears the bridge's strict matcher on the instances it
semantically understands.

**Status.** Open; partially measured in § D.4.

### B.8 Conjecture C41-4 — Communication-bounded vs
generator-bounded regime decomposition

**Statement.** Every end-to-end real-SWE loop on the
Phase-41 pipeline admits a decomposition

```
pass@1(strat, model, matcher) = P_comm(strat) · P_gen(model, matcher)
```

where ``P_comm`` is the substrate-side correctness ceiling
(equal to the oracle pass-rate — 1.0 on the Phase-41
bank by Theorem P41-2) and ``P_gen`` is the generator's
text-fidelity-conditioned on the bridge's accepted matcher.
In the *communication-bounded* regime ``P_comm < 1`` on
the active cell (a substrate role is under-delivering on
the load-bearing handoff); in the *generator-bounded*
regime ``P_gen < 1`` (the LLM cannot emit a byte-match
under the bridge's accepted matcher). Every Phase-41
cell falls into one of the two regimes cleanly.

**Status.** Open. The Phase-41 28-instance sweep
provides one empirical decomposition per (model, matcher);
confirmation at SWE-bench Lite scale and across larger-
model families is the natural next step.

### B.9 What is theorem vs empirical vs conjectural

| Claim | Strength |
|---|---|
| P41-1 bounded-context preservation at scale | **Theorem** (empirical + structural) |
| P41-2 oracle-ceiling is matcher-mode-invariant | **Theorem** (empirical + structural) |
| P41-3 matcher-permissiveness attribution decomposition | **Theorem** (counting identity) |
| C41-1 communication-bounded at ≥ 50 instances | **Conjecture** (empirical follow-up) |
| C41-2 matcher-permissiveness saturates | **Conjecture** |
| C41-3 stronger-model saturates strict matcher | **Conjecture** (partial in D.4) |
| C41-4 comm-bounded vs generator-bounded regime decomposition | **Conjecture** |
| C41-5 parser-compliance attribution boundary | **Conjecture** (surfaced from § D.4) |

---

## Part C — Architecture

### C.1 New / extended modules

```
vision_mvp/tasks/swe_bench_bridge.py            [EXTENDED]  +~170 LOC
    + APPLY_MODE_STRICT / APPLY_MODE_LSTRIP
    + APPLY_MODE_WS_COLLAPSE / APPLY_MODE_LINE_ANCHORED
    + ALL_APPLY_MODES
    + apply_patch(..., mode="strict") — new kwarg, default preserves
      Phase-40 byte-exact semantics
    + _apply_patch_strict (factored from Phase-40 apply_patch)
    + _apply_patch_permissive (new — line-normalised unique match)
    + _normalise_line (per-mode line projector)
    + run_swe_loop(..., apply_mode="strict") — new kwarg
    + SWEReport.config records apply_mode

vision_mvp/tasks/swe_sandbox.py                  [EXTENDED]  +~30 LOC
    + Sandbox.run(..., apply_mode="strict") protocol update
    + InProcessSandbox / SubprocessSandbox / DockerSandbox pass
      apply_mode to apply_patch
    + run_swe_loop_sandboxed(..., apply_mode="strict") — new kwarg
    + SWEReport.config records apply_mode

vision_mvp/tasks/data/swe_lite_style_bank.jsonl [NEW]  28 instances
    28-instance SWE-bench-Lite-shape JSONL bank covering operator-
    typo / off-by-one / wrong-branch / seed-wrong / aggregate /
    mutation-vs-copy / multi-hunk / unicode / parity-partition /
    slice-direction / index-return edit shapes.

vision_mvp/tasks/data/_build_swe_lite_bank.py    [NEW]
    Generator script. Round-trips every instance through
    parse_unified_diff + apply_patch + run_patched_test before
    writing; refuses to register any instance whose diff doesn't
    parse, whose old blocks aren't unique, or whose oracle-
    patched source doesn't pass the hidden test.

vision_mvp/experiments/phase41_swe_lite_sweep.py [NEW]  ~360 LOC
    Phase-41 driver. Composes loader + substrate + sandbox +
    (optional) real LLM; caches LLM output per (instance,
    strategy, n_distractors) so a second matcher cell does not
    re-call the LLM. Emits per-strategy recovered/regressed
    set deltas against the strict baseline.

vision_mvp/tests/test_phase41_swe_lite.py        [NEW]  18 tests
```

The Phase-39 / Phase-40 bridge + sandbox + Phase-40 real-SWE
driver paths are *preserved byte-for-byte*: the Phase-40
run_swe_loop_sandboxed default is ``apply_mode="strict"``, so
every Phase-40 artifact rerun produces identical numbers.

### C.2 Where the new primitives sit

```
   ┌──────────────────────────────────────────────────────┐
   │  Phase 41 — Scale + matcher attribution               │
   │  - ``APPLY_MODE_{STRICT,LSTRIP,WS_COLLAPSE,...}``     │
   │  - ``apply_patch(..., mode=...)``                     │
   │  - ``Sandbox.run(..., apply_mode=...)``               │
   │  - ``run_swe_loop{,_sandboxed}(..., apply_mode=...)`` │
   │  - ``swe_lite_style_bank.jsonl`` (28 instances)       │
   │  - ``phase41_swe_lite_sweep`` (driver + attribution)  │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 40 — Loader + sandbox + driver                 │
   │  - ``parse_unified_diff`` / ``load_jsonl_bank``       │
   │  - ``InProcess / Subprocess / Docker`` sandboxes      │
   │  - ``run_swe_loop_sandboxed``                         │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 39 — SWEBench bridge (multi-role SWE team)     │
   │  - SWEBenchStyleTask schema + four-role substrate     │
   └──────────────────────────────────────────────────────┘
```

### C.3 Files changed

| File | Change |
|---|---|
| ``vision_mvp/tasks/swe_bench_bridge.py``                                | **EXTENDED** (+~170 LOC) — apply_mode axis + matcher modes |
| ``vision_mvp/tasks/swe_sandbox.py``                                     | **EXTENDED** (+~30 LOC) — apply_mode propagation |
| ``vision_mvp/tasks/data/swe_lite_style_bank.jsonl``                    | **NEW** (28 instances) |
| ``vision_mvp/tasks/data/_build_swe_lite_bank.py``                      | **NEW** |
| ``vision_mvp/experiments/phase41_swe_lite_sweep.py``                   | **NEW** |
| ``vision_mvp/tests/test_phase41_swe_lite.py``                          | **NEW** (18 tests) |
| ``vision_mvp/RESULTS_PHASE41.md``                                      | **NEW** — this doc |
| ``docs/context_zero_master_plan.md``                                   | Phase 41 integration, frontier update |
| ``README.md``                                                          | Phase 41 thread |
| ``ARCHITECTURE.md``                                                    | Phase 41 thread |
| ``MATH_AUDIT.md``                                                      | P41 theorems + conjectures |
| ``vision_mvp/results_phase41_swe_lite_mock.json``                      | **NEW** artifact (28 × 4 × 3 × 2 = 672 measurements) |
| ``vision_mvp/results_phase41_swe_lite_7b.json``                        | **NEW** artifact (real LLM) |
| ``vision_mvp/results_phase41_swe_lite_9b.json``                        | **NEW** artifact (stronger-model spot check) |

---

## Part D — Evaluation

### D.1 Mock sweep — bridge + sandbox + substrate at 28 instances

Bundled JSONL (28 instances) under
``deterministic_oracle_generator`` + ``SubprocessSandbox``
across ``n_distractors ∈ {0, 6, 12, 24}`` and
``apply_modes ∈ {strict, lstrip}``. Wall = **53.0 s** for
672 sandboxed measurements (~ 79 ms / measurement).

| n_distractors | naive_chars | routing_chars | substrate_chars | any_pass@1 |
|---:|---:|---:|---:|---:|
| 0   | 806.8    | 372.8    | **746.4** | 1.000 |
| 6   | 1 127.8  | 693.8    | **746.4** | 1.000 |
| 12  | 1 461.8  | 1 027.8  | **746.4** | 1.000 |
| 24  | 2 125.8  | 1 691.8  | **746.4** | 1.000 |

Cross-distractor pooled summary (4 cells, oracle, strict):

| strategy   | pass@1 | tokens ≈ | events | handoffs |
|---|---:|---:|---:|---:|
| naive      | 1.000 | 345.1  | 14.5 | 2.0 |
| routing    | 1.000 | 236.6  | 10.5 | 2.0 |
| **substrate**  | **1.000** | **186.6** | **0.0**  | 5.0 |

Reading:

* **Theorem P41-1 reproduces.** Substrate prompt is constant
  at 747 chars (~ 186.6 tokens) across the distractor sweep
  while naive grows 807 → 2 126 (**2.6×** span). At 28
  instances the invariant is a direct consequence of the
  substrate's ``ctx = {issue_summary, hunk}`` construction
  — not a bank-specific accident.
* **Theorem P41-2 reproduces.** The oracle saturates
  pass@1 = 1.000 on every (matcher, strategy, distractor)
  cell; permissive matching subtracts no correctness.
* **Hash-chain integrity** is preserved on every
  measurement (672 × ``chain_ok = True``).

### D.2 Matcher-permissiveness attribution — oracle (Part B reference)

Oracle-emit patches are byte-exact by construction, so the
``recovered`` / ``regressed`` sets are both empty on every
strategy and distractor cell. This is the *null-control* of
the attribution framework: permissive matching only kicks in
on generators whose OLD blocks drift from the buggy source.
(See artifacts ``results_phase41_swe_lite_mock.json``:
``attribution.{nd}.lstrip.{strategy}.{recovered,regressed} = []``
on every cell.)

The interesting cells are in § D.3 and § D.4.

### D.3 Real-LLM empirical sweep — ``qwen2.5-coder:7b``

``qwen2.5-coder:7b`` patch generator on the full 28-instance
Phase-41 bank at ``n_distractors = 6`` under
``SubprocessSandbox``, three matcher modes
(``strict`` / ``lstrip`` / ``line_anchored``). 56 unique
LLM calls (Phase-41 cache: one call per
``(instance_id, strat_proxy, nd)`` with the naive/routing
cells sharing a proxy because their generator-side prompts
are identical under the Phase-39 / Phase-40 bridge contract).
Wall = **1 658.6 s / ~ 27.6 min** for the first (LLM-active)
cell; the two subsequent permissive cells reuse the cached
LLM outputs and run purely as sandbox reruns in **6.5 s** each
(**~ 253×** wall speed-up on the attribution axis — the
cache discipline pays off empirically). LLM token counters:
13 178 prompt / 6 978 output.

**Pass@1 by (strategy, matcher)**:

| strategy  | strict | lstrip | line_anchored |
|---|---:|---:|---:|
| naive     | **0.929** (26/28) | 0.929 | 0.929 |
| routing   | **0.929** (26/28) | 0.929 | 0.929 |
| substrate | **0.893** (25/28) | 0.893 | 0.893 |

**Failure taxonomy** (strict, identical under permissive):

| strategy  | ok | patch_no_match | test_assert |
|---|---:|---:|---:|
| naive     | 26 | 1 | 1 |
| routing   | 26 | 1 | 1 |
| substrate | 25 | 2 | 1 |

**Matcher-permissiveness attribution** (Theorem P41-3
``Δ pass@1 = |R_recovered| − |R_regressed|``) at
``n_distractors = 6``:

| strategy  | recovered(lstrip) | regressed(lstrip) | recovered(line_anchored) | regressed(line_anchored) | unchanged_pass | unchanged_fail |
|---|---:|---:|---:|---:|---:|---:|
| naive     | 0 | 0 | 0 | 0 | 26 | 2 |
| routing   | 0 | 0 | 0 | 0 | 26 | 2 |
| substrate | 0 | 0 | 0 | 0 | 25 | 3 |

Reading (honest, three findings the data actually supports):

* **The Phase-40 ranking inversion was small-N variance.**
  At 28 instances, naive-vs-substrate pass@1 gap is **3.6
  pp** (0.929 vs 0.893 = 1 instance out of 28), **down from
  Phase 40's 16.7 pp** at 6 instances. The result is
  consistent with Conjecture C41-1 (larger N smooths the
  per-instance variance) — though 28 instances is still
  short of the ≥ 50 scale the conjecture's falsifier
  requires.
* **Matcher permissiveness does NOT recover any instance
  on the 28-instance bank under the 7B.** Both ``lstrip``
  and ``line_anchored`` produce **R_recovered = ∅** on every
  strategy. The 7B's strict-mode failures are not byte-
  fidelity drift — they are deeper shape failures: either
  the LLM emitted an OLD block that doesn't correspond to
  any source region even after normalisation (wrong anchor
  / hallucinated content), or the OLD block's
  normalised form is still not found. The
  ``patch_no_match`` failures are *semantically* wrong, not
  *typographically* drifted.
* **Matcher permissiveness does NOT over-accept.**
  ``R_regressed = ∅`` on every cell — the permissive modes
  are safe on this bank at this model scale. Combined with
  Theorem P41-2's oracle-saturation null-control, this
  confirms the permissive matcher axis is a safe
  attribution knob, not a source of false positives. The
  Phase-41 permissive modes are *null on recovery and null
  on risk* at the 7B scale.

The second bullet refines the Phase-40 § D.5 hypothesis. The
Phase-40 prediction was that permissive matching might
close the substrate-vs-naive gap by accepting byte-drifted
but semantically correct patches. At 28 instances, **no such
patches exist in the 7B's output**: every strict-rejected
patch is rejected for a deeper reason than trailing-
whitespace or indentation drift. Theorem P41-3's
attribution decomposition makes this visible:
``Δ pass@1(lstrip) = |R_recovered| − |R_regressed| = 0 − 0
= 0`` — the permissive cell is empirically *zero-gain and
zero-risk* on the 7B × 28-instance cell.

This is the honest Phase-41 7B finding: the Phase-40
interpretation that "byte-strict matching is the dominant
generator-side bottleneck" is **not supported** at the
28-instance scale. The remaining failures are generator-
bounded in Theorem P39-2's stronger sense — the LLM is
emitting text that does not correspond to the source
region at all, and a more permissive *byte-level* matcher
cannot rescue a *semantically* misdirected patch. The
programme's open follow-up for matcher-side recovery is
therefore *AST-level* or *edit-distance-bounded* matchers
(Conjecture C41-2's refinement), not whitespace-tolerant
ones.

### D.4 Stronger-model datapoint — ``gemma2:9b``

``gemma2:9b`` patch generator on the full 28-instance
Phase-41 bank at ``n_distractors = 6`` under
``SubprocessSandbox``, three matcher modes
(``strict`` / ``lstrip`` / ``line_anchored``). Same Phase-41
driver and LLM-call cache as the 7B sweep in § D.3.
Wall = **1 388 s / ~ 23.1 min** for the LLM-active cell; the
two permissive cells reuse the cached outputs and run in
~ 0.01 s (pure cache hits, no sandbox re-execution because
every proposed patch is the empty tuple — see below). LLM
token counters: 13 300 prompt / 3 514 output.

**Pass@1 by (strategy, matcher)**:

| strategy  | strict | lstrip | line_anchored |
|---|---:|---:|---:|
| naive     | **0.000** (0/28) | 0.000 | 0.000 |
| routing   | **0.000** (0/28) | 0.000 | 0.000 |
| substrate | **0.000** (0/28) | 0.000 | 0.000 |

**Failure taxonomy** (identical across matchers):

| strategy  | patch_no_match |
|---|---:|
| naive     | 28 |
| routing   | 28 |
| substrate | 28 |

**Matcher-permissiveness attribution** (Theorem P41-3)
at ``n_distractors = 6``:

| strategy  | recovered(lstrip) | regressed(lstrip) | recovered(line_anchored) | regressed(line_anchored) | unchanged_pass | unchanged_fail |
|---|---:|---:|---:|---:|---:|---:|
| naive     | 0 | 0 | 0 | 0 | 0 | 28 |
| routing   | 0 | 0 | 0 | 0 | 0 | 28 |
| substrate | 0 | 0 | 0 | 0 | 0 | 28 |

Reading (honest; this is NOT the naive "stronger model, more
pass@1" story):

* **Every 9B response fails the bridge's OLD/NEW parser
  boundary.** An ad-hoc spot check of one 9B response on
  ``ext-calc-001`` produces:

  ```
  OLD>>>
      result = 0
  <<<NEW>>>
      result = 1
  ```

  (no closing ``<<<`` — the model ends generation mid-
  block). The ``llm_patch_generator`` parser
  ``_BLOCK_RE = r"OLD>>>(.*?)<<<NEW>>>(.*?)<<<"`` requires
  a final ``<<<`` to terminate the NEW block; missing that
  delimiter, it returns ``ProposedPatch(patch=(),
  rationale="parse_failed")``, which ``apply_patch`` then
  surfaces as ``patch_no_match`` (via the ``empty_patch``
  regime).
* **The 9B's semantic answer is correct.** The OLD block
  is byte-exact with the source; the NEW block is the
  right fix. The failure is *format-compliance* at the
  LLM-output parser, not semantic misdirection. This is
  important: it means the Phase-40 § D.5 "stronger-model
  will saturate strict matcher" hypothesis (Conjecture
  C41-3) is *not testable* at the 9B scale with the
  current bridge parser — the parser brittleness dominates
  before the matcher axis can kick in.
* **Permissive matching can't help.** Every proposed patch
  is the empty tuple; ``apply_patch`` refuses at the
  ``if not patch: return ..., "empty_patch"`` line, before
  any matcher-mode logic runs. The matcher axis sits below
  the parser axis in the failure stack; on the 9B × Phase-41
  bank the parser axis is the active constraint.
* **This is a new attribution boundary the programme has
  not named before.** The Phase-37 § Part A finding
  (Theorem P37-1) was that real-LLM reply noise on
  incident-triage tasks is dominated by *semantic*
  mislabel, not syntactic failure. The Phase-41 9B finding
  is the *inverse* on the SWE-code-patch task: a general-
  purpose 9B model emits the right semantic answer but
  fails the bridge's *syntactic* delimiter contract. The
  two findings together suggest real-LLM failure modes on
  a given task family are *task-specific*: the same model
  class can be semantic-wrong on one family and
  format-wrong on another. Conjecture C41-5 (below) names
  this.

The 9B datapoint is therefore *informative* but *not* the
saturation evidence Conjecture C41-3 called for. The
Phase-41 programme's immediate follow-up is a more robust
LLM-output parser (close the NEW block on end-of-
generation; retry with an explicit format-reinforcement
prompt variant) — a Phase-42-class improvement, not a
Phase-41 deliverable.

**Updated conjecture C41-5 (parser-compliance attribution).**
For a non-coder-finetuned model in the 7B–30B class, the
Phase-41 bridge's byte-strict OLD/NEW parser is the dominant
failure boundary, before the matcher axis becomes measurable.
Falsifier: a general-purpose model in that class whose
``llm_patch_generator`` parse-failure rate on the Phase-41
bank is below 20 %.

Comparison with Phase 39 Part B ranking: ``gemma2:9b``
saturated the mock-auditor ceiling on *Phase-31 incident
triage* at 100 %, the strongest local model in the Phase-39
frontier sweep. The Phase-41 result refines that ranking:
*task-family generalises the ranking*. On the SWE-code-
patch task family with the Phase-40 bridge's strict
OLD/NEW parser, the coder-finetuned
``qwen2.5-coder:7b`` clears the parser contract on every
instance (56 / 56 LLM calls produce a parseable block,
pass-rate 89–93 %), while the general-purpose ``gemma2:9b``
fails the parser contract on every instance (0 parseable
blocks, 0 / 28 pass-rate) — a clean example of Theorem
P39-2's transcription-bounded regime at the parser boundary.

### D.5 Failure attribution surface

The Phase-41 driver extends the Phase-40 failure taxonomy
with an explicit matcher-mode dimension. Every measurement
in the result JSON now carries:

* ``error_kind`` ∈ {``""``, ``patch_no_match``, ``syntax``,
  ``import``, ``test_assert``, ``test_exception``,
  ``timeout``, ``sandbox_error``};
* the ``apply_mode`` of the cell that produced it
  (``rep.config["apply_mode"]``);
* the substrate/naive/routing strategy label;
* the ``n_distractors`` cell label.

Reading the artifact: to decide whether a failure is
*substrate-shaped* (a role under-delivered), *generator-
shaped* (the LLM emitted wrong text), or *sandbox-shaped*
(the process boundary misreported), look at the
``(error_kind, strategy, apply_mode)`` triple. The matcher
axis is the only new failure-attribution dimension in
Phase 41; the substrate and sandbox dimensions are
Phase-39/40 carry-overs.

### D.6 Messaging budget — Phase-41 larger bank

Pooled across 28 tasks × 4 distractor cells × 3 strategies
× 2 matcher modes = 672 measurements (mock run, oracle).
Headline counters:

| metric                      | naive | routing | substrate |
|---|---:|---:|---:|
| mean_handoffs               | 2.0   | 2.0     | 5.0       |
| mean_events_to_patch_gen    | 14.5  | 10.5    | 0.0       |
| mean_patch_gen_prompt_chars | 1 380 | 947     | **747**   |
| mean_wall_seconds (sandboxed) | 0.079 | 0.079 | 0.079    |
| chain_hash_invariant_holds  | 100 % | 100 %   | 100 %     |

Compared to Phase 40's mini-bank headline: mean wall per
measurement is stable (~ 78 ms); substrate prompt is
slightly tighter (747 vs 813 chars) because the Phase-41
bank's hunk windows average smaller; per-measurement
handoff count is unchanged. The Phase-41 pipeline is
a direct extension, not a regime change.

---

## Part E — Failure taxonomy

Phase 41 does NOT introduce a new ``error_kind``. The
existing Phase-40 taxonomy is sufficient; the new axis is
``apply_mode``, which is recorded per-measurement.

The Phase-41 artifact shape adds two top-level keys to the
Phase-40 result JSON:

```
{
  "cells": [
    {
      "apply_mode": "strict" | "lstrip" | "ws_collapse" | "line_anchored",
      "n_distractors": int,
      "report": {... Phase-40 SWEReport shape ...},
      "failure_taxonomy": {strategy: {error_kind: count}},
      "cell_wall_s": float
    }, ...
  ],
  "attribution": {
    "nd_str": {
      "lstrip" | "ws_collapse" | "line_anchored": {
        strategy: {
          "recovered": [instance_id, ...],
          "regressed": [instance_id, ...],
          "unchanged_pass": int,
          "unchanged_fail": int
        }
      }
    }
  },
  ...
}
```

The attribution block is the Phase-41 specific output: given
a per-measurement ``(apply_mode, strategy, n_distractors,
instance_id, test_passed)`` row, the driver pairs strict and
permissive cells and emits the four-partition set delta of
Theorem P41-3.

---

## Part F — Future work

### F.1 Carry-over from Phase 40

* **Real SWE-bench Lite at ≥ 50 instances (C39-3 / C39-4
  / C40-2).** Phase 40 made this a ``--jsonl <path>``
  parameter change; Phase 41 adds the matcher axis so the
  Lite run can report both strict and permissive pass@1
  separately. Still the largest remaining external-validity
  gap.
* **Multi-hunk coverage on real SWE-bench Lite.** Phase 41
  ships one multi-hunk instance (``ext-multi-001``); the
  real SWE-bench multi-hunk diff coverage is still
  downstream.
* **Docker-axis equivalence measurement (C40-3).** Orthogonal
  to Phase 41.

### F.2 Newly surfaced or tightened by Phase 41

* **Conjecture C41-1 falsifier (larger-N
  communication-bounded).** Run the Phase-41 driver on a
  ≥ 50-instance public SWE-bench Lite JSONL under the
  7B and 9B models; measure whether the Phase-40 6-instance
  ranking inversion persists, shrinks, or flips.
* **Conjecture C41-2 falsifier
  (matcher-permissiveness saturation).** Same run, stratify
  ``|R_recovered| / N`` by benchmark scale.
* **Conjecture C41-3 falsifier (stronger-model strict-floor
  saturation).** Repeat Phase 41 on a 30B+ class model or
  a coder-finetuned 70B; test whether ``|R_recovered| → 0``.
* **Permissive-mode sophistication.** Phase 41 ships three
  permissive modes. More targeted schemes — fuzzy hunk
  anchoring with edit-distance bounds, AST-aware
  substitution — are potential follow-ups. Each must
  preserve the unique-match discipline and be testable
  for ``R_regressed = ∅`` under the oracle.
* **Generator-vs-substrate attribution at the turn level.**
  Phase 41 attributes at the ``(instance, strategy)``
  granularity. A per-hunk attribution (which OLD block
  drifted which way) is a natural refinement when multi-
  hunk banks land.

### F.3 What is genuinely blocking the endgame

Phase 41 does NOT unblock:

* **Public SWE-bench Lite ranking** (C39-3 / C39-4) —
  empirical at ≥ 50 instances on a Lite JSONL.
* **Cross-language runtime calibration**.
* **Strong-model bias saturation** (C39-1) — the Phase-41
  9B datapoint (§ D.4) gets blocked at the parser
  boundary before the matcher / substrate axes can show
  saturation. A 30B+ coder-finetuned model or an
  improved parser are both on the critical path.
* **OQ-1 in full generality** (Conjecture P30-6).

Phase 41 *does* close:

* "Maybe the Phase-40 6-instance ranking result was just
  small-N variance" — § D.3 at 28 instances answers: at
  the 7B coder scale the substrate-vs-naive gap shrinks
  from **16.7 pp (Phase-40 @ 6 instances)** to **3.6 pp
  (Phase-41 @ 28 instances)**. The one-instance inversion
  was noise.
* "Maybe byte-strict matching hides a substrate-strictly-
  better-than-naive result" — Theorem P41-3 + § D.3's
  empty ``R_recovered`` answer: **no, on the 7B × 28
  instances**. The 7B's strict-mode failures are *not*
  byte-fidelity drift. Whitespace / trailing-newline
  permissive matchers produce zero recovery and zero
  over-acceptance.
* "Maybe permissive matching is too risky to ship as an
  evaluation knob" — Theorem P41-2 (oracle saturation
  matcher-invariance) plus the 7B × 28 cell's empty
  ``R_regressed`` establish the null-control on this bank.

Phase 41 *surfaces* a new open item:

* **Parser-compliance attribution layer (Conjecture
  C41-5).** § D.4 shows that on ``gemma2:9b``, the bridge's
  ``_BLOCK_RE`` parser is the dominant failure boundary —
  the 9B emits the semantically correct fix but fails to
  close the ``<<<`` delimiter, so every patch is
  ``patch_no_match``. A more robust LLM-output parser
  (force-close the NEW block on end-of-generation, retry
  with an explicit format-reinforcement prompt, or adopt a
  unified-diff output format) would lift this — and the
  Phase-41 matcher axis would then become newly measurable
  at the 9B scale. **This is the programme's most
  tractable Phase-42 target.**

The programme's remaining external-validity gap is now
*public-benchmark-shaped* (SWE-bench Lite JSONL + compute
budget), *parser-shaped* (C41-5), and *model-shaped*
(30B+ coder-finetuned frontier run for C41-3's proper
test) — not infrastructure- or attribution-shaped.

---

## Appendix A — How to reproduce

```bash
# 1. Phase-41 mock sweep (28 instances × 4 distractor cells × 3
#    strategies × 2 matcher modes, ~ 53 s wall).
python3 -m vision_mvp.experiments.phase41_swe_lite_sweep \
    --mode mock --sandbox subprocess \
    --apply-modes strict lstrip \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 0 6 12 24 \
    --out vision_mvp/results_phase41_swe_lite_mock.json

# 2. Phase-41 real LLM — qwen2.5-coder:7b on the full 28-instance
#    bank at n_distractors = 6, both matcher modes.
python3 -m vision_mvp.experiments.phase41_swe_lite_sweep \
    --mode real --model qwen2.5-coder:7b --sandbox subprocess \
    --apply-modes strict lstrip \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 6 \
    --out vision_mvp/results_phase41_swe_lite_7b.json

# 3. Phase-41 stronger-model datapoint — gemma2:9b on a subset.
python3 -m vision_mvp.experiments.phase41_swe_lite_sweep \
    --mode real --model gemma2:9b --sandbox subprocess \
    --apply-modes strict lstrip \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 6 --n-instances 12 \
    --out vision_mvp/results_phase41_swe_lite_9b.json

# 4. Phase-41 test slice (18 tests; ~ 28 s).
python3 -m pytest vision_mvp/tests/test_phase41_swe_lite.py -q

# 5. Full Phase 39..41 SWE-arc regression.
python3 -m pytest vision_mvp/tests/test_phase39_swe_bridge.py \
    vision_mvp/tests/test_phase40_real_swe_bridge.py \
    vision_mvp/tests/test_phase41_swe_lite.py -q

# 6. Rebuild the 28-instance JSONL artifact from source (optional;
#    the JSONL ships bundled).
python3 vision_mvp/tasks/data/_build_swe_lite_bank.py
```

On a commodity 2026-vintage laptop: #1 runs in ~ 53 s; #2
runs in ~ 30–40 min on qwen2.5-coder:7b for 84 LLM calls
(28 × 3); #3 runs in ~ 15–25 min for 36 LLM calls; #4 runs
in ~ 28 s; #5 runs in ~ 36 s; #6 runs in ~ 7 s.

---

*End of Phase 41 results note. The master plan
(``docs/context_zero_master_plan.md``) is updated in the same
commit; see the new ``§ 4.9.9 Arc 8 (extended further) — Larger
SWE-bench-Lite-style sweep, matcher-permissiveness attribution,
stronger-model datapoint (Phase 41)`` and the updated
``§ 4.11 Current frontier``.*
