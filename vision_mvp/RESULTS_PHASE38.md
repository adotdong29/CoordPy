# Phase 38 — Two-layer ensemble composition, minimum dynamic primitive ablation, prompt-shaped reply calibration

**Status: combined research milestone. Phase 38 closes three
coupled Phase-37 gaps: (A) does composing the Phase-34
extractor-axis ensemble with the Phase-37 reply-axis ensemble
defend where neither alone can; (B) which features of the
dynamic communication primitive are load-bearing on the
Phase-35 contested and Phase-37 nested task families; (C)
does the real-LLM ``sem_root_as_symptom`` bias move under a
disciplined set of prompt variants. Two new core modules
(``core/two_layer_ensemble``, ``core/primitive_ablation``),
one additional extractor-axis test fixture
(``core/extractor_adversary``), one prompt-variation module
(``core/prompt_variants``), three experiment drivers, four
new theorems (P38-1..P38-4), three new conjectures
(C38-1..C38-3). 35 new unit tests; full regression
(1,373 tests) clean.**

Phase 38 in one line: **on the Phase-35 contested bank, a
two-layer ensemble that composes the Phase-34 extractor-axis
``UnionClaimExtractor`` with the Phase-37 reply-axis
``EnsembleReplier`` is the unique configuration that closes
the conjunction of a layer-1 adversarial drop AND a layer-2
biased primary (83.3 % vs 33.3 % for every single-layer
alternative); the Phase-38 ``PathUnionCausalityExtractor``
closes the adv_drop_root cell that Theorem P37-4 proved no
reply-axis ensemble alone can close (100 % vs 33.3 %);
removing any of {``typed_vocab``, ``terminating_resolution``,
``round_aware_state``} from the thread primitive collapses it
on at least one task family, while ``bounded_witness`` and
``frozen_membership`` are null-control ablations on accuracy
but load-bearing for the Phase-35 P35-2 bounded-context
invariant; on a deterministic bias-shift mock the ``rubric``
and ``contrastive`` prompt variants cut the Phase-37 semantic-
wrong rate from 0.688 to 0.225.**

---

## Part A — Research framing

### A.1 Why this milestone exists

Phase 37 left four coupled open questions on the frontier:

1. **"Two-layer ensemble composition across the full
   communication stack is untested."** Conjecture C37-2
   named it as the extension of reply-axis ensembles (Phase
   37) to cover extractor-output noise that Theorem P37-4
   proved reply-axis ensembles powerless against. No
   artefact yet existed.
2. **"The minimum dynamic primitive is conjecturally
   five-featured."** Conjecture C37-4 listed the candidate
   feature set ({typed reply enum, bounded witness,
   terminating resolution, round-aware state, bounded-
   context invariant}) but did not construct a per-feature
   falsifier table.
3. **"Semantic reply bias might be prompt-shaped, not
   model-shaped."** Conjecture C37-1 named the hypothesis
   without a testable artefact.
4. **"The programme still lacks a single headline result
   on full-stack communication robustness."** Phase 37
   closed reply-axis depth; Phase 34 closed extractor-axis
   depth; no phase had measured the joint.

Phase 38 attacks all three with empirical instruments and an
explicit theory note. The stance is falsifiable: if
two-layer composition does not beat single-layer on the
conjunction cell, we report that; if no prompt variant
moves the semantic-wrong rate, we report that; if every
feature in C37-4 is individually load-bearing, the
conjecture is strengthened.

### A.2 What Phase 38 ships (five coupled pieces)

* **Part A — Two-layer ensemble composition
  (``core/two_layer_ensemble`` +
  ``core/extractor_adversary``).** A
  ``PathUnionCausalityExtractor`` combiner that places a
  class-level combiner strictly *above* any per-path
  noise wrapper, and a ``UnionClaimExtractor`` / adversarial
  / narrative extractor triple that exercises the
  extractor-axis. A driver
  (``experiments/phase38_two_layer_ensemble``) compares
  five configurations (baseline, extractor_only,
  reply_only, two_layer, two_layer_path_union) across five
  noise cells (clean, ext_drop_gold, rep_biased_primary,
  conjunction, adv_drop_root).
* **Part B — Minimum primitive ablation
  (``core/primitive_ablation``).** A feature-flagged
  thread runner with five toggles (``typed_vocab``,
  ``bounded_witness``, ``terminating_resolution``,
  ``round_aware_state``, ``frozen_membership``). A driver
  (``experiments/phase38_primitive_ablation``) measures
  per-feature collapse on the Phase-35 contested and
  Phase-37 nested banks.
* **Part C — Prompt calibration
  (``core/prompt_variants``).** Five Phase-35-compatible
  prompt variants (default, contrastive, few_shot, rubric,
  forced_order) plus a ``BiasShiftMockReplier`` and a
  real-LLM driver (``experiments/phase38_prompt_calibration``
  under ``--mode mock|real``).
* **Part D — Theory.** Four new theorems (P38-1..P38-4),
  three new conjectures (C38-1..C38-3). Master plan
  updated with Phase 38 integration.
* **Part E — Regression.** 35 new tests; full Phase
  30-38 regression clean (1,373 tests).

### A.3 Scope discipline (what Phase 38 does NOT claim)

1. **Not a replacement for Phase 34, 35, 36, or 37.** Every
   prior primitive is unchanged. Phase 38 is additive.
2. **Not a claim that the prompt-calibration study moves
   real-LLM bias.** Part C ships a deterministic mock that
   *validates the pipeline* and a real-LLM driver
   (``--mode real --models ...``) that any reader can run
   against their own Ollama install. The mock's per-variant
   bias table is a scaffold, not a predictive model.
3. **Not a claim that the minimum primitive is
   exhaustively characterised.** We ship a falsifier table
   for the five C37-4 features on two task families.
   ``frozen_membership`` is a null-control on both — we do
   NOT claim no task exists where it is load-bearing.
4. **Not a claim that two-layer ensembles subsume every
   failure mode.** The ``adv_drop_root`` cell is recovered
   by ``PathUnionCausalityExtractor`` specifically; a more
   aggressive adversary (e.g. multi-path correlated) would
   require further depth.
5. **Not a full SWE-bench validation.** The external-
   validity gap named in Phase 36/37 remains unchanged.

---

## Part B — Theory

### B.1 Setup

We inherit the Phase-37 setup. ``C(z)`` is the causal chain,
``D_dyn`` / ``D_adp`` / ``D_static`` the decoders. The two
ensemble boundaries (Phase 38 formalises the naming):

* **Layer 1 (extractor axis):** the
  ``extract_claims_for_role`` boundary.
  ``(role, events, scenario) → list[(claim_kind, payload,
  evids)]``. Ensemble point: Phase-34
  ``UnionExtractor``, Phase-38
  ``UnionClaimExtractor``.
* **Layer 2 (reply axis):** the
  ``causality_extractor`` boundary.
  ``(scenario, role, kind, payload) → causality_class``.
  Ensemble point: Phase-37 ``EnsembleReplier``, Phase-38
  ``PathUnionCausalityExtractor``.

A **noise wrapper** is a function ``ν`` that post-processes
the output of either boundary. A **path** is a concrete
``(extractor, replier, wrappers)`` tuple. A **combiner**
aggregates the outputs of two or more paths.

### B.2 Theorem P38-1 — Two-layer composition closes the conjunction cell that no single-layer closes

**Statement.** On the Phase-35 contested bank under the
joint noise cell ``C_∧ = {ext_drop_gold ∧
rep_biased_primary}``:

```
acc(D_dyn with baseline       on C_∧)  = 1/3   (contested 1/4)
acc(D_dyn with extractor_only on C_∧)  = 1/3   (contested 1/4)
acc(D_dyn with reply_only     on C_∧)  = 1/3   (contested 1/4)
acc(D_dyn with two_layer      on C_∧)  = 5/6   (contested 4/4)
```

i.e. the two-layer composition ``UnionClaimExtractor ∘
EnsembleReplier(MODE_DUAL_AGREE)`` is the unique
configuration tested that closes the joint cell.

**Interpretation.** The two attacks target disjoint
boundaries. ``ext_drop_gold`` fires on layer 1 (drops the
gold claim); ``rep_biased_primary`` fires on layer 2 (the
primary replier always emits INDEPENDENT_ROOT). A layer-1
defense alone (extractor_only) is powerless against layer-2
attack and vice versa (Theorem P37-4 for the other
direction). Only the stacked defense has one combiner per
layer.

**Proof sketch.** ``ext_drop_gold`` removes the gold
claim kind from ``extract_claims_for_role``'s emission on
the target role. Under ``baseline`` (Phase-31 extractor)
and ``reply_only``, the auditor's inbox no longer contains
the gold claim, so ``detect_contested_top`` cannot include
it — either no thread opens or a thread opens on the
remaining pair, neither of which resolves to the gold kind.
Under ``extractor_only`` and ``two_layer``, the
``UnionClaimExtractor``'s narrative secondary emits the
gold claim (matched via tag+signal), and the inbox contains
it again. Under ``reply_only`` and ``baseline``, the
``rep_biased_primary`` biased primary emits IR on both
contested candidates, producing CONFLICT under the
Phase-35 resolution rule. Under ``extractor_only`` and
``baseline``, this still fires. Only ``two_layer`` has both
(i) an inbox with gold present and (ii) a reply-axis
ensemble that AND-gates the biased primary against a clean
secondary — on the true IR the secondary agrees, on the
false IR the secondary disagrees, so the CONFLICT is
averted. ∎

**Empirical anchor.** § D.1 —
``results_phase38_two_layer_ensemble.json``, cell
``conjunction`` × config ``two_layer``.

### B.3 Theorem P38-2 — PathUnionCausalityExtractor closes the adv_drop_root cell that reply-axis ensembles cannot

**Statement.** On the Phase-35 contested bank under
``adv_drop_root`` (budget=1, noise at the extractor-output
boundary — i.e. strictly above the reply-axis ensemble
combiner per Theorem P37-4):

```
acc(D_dyn with baseline           ) = 1/3   (contested 0/4)
acc(D_dyn with extractor_only     ) = 1/3   (contested 0/4)
acc(D_dyn with reply_only         ) = 1/3   (contested 0/4)
acc(D_dyn with two_layer          ) = 1/3   (contested 0/4)
acc(D_dyn with two_layer_path_union) = 1     (contested 1)
```

i.e. the ``PathUnionCausalityExtractor`` with
``PATH_MODE_UNION_ROOT``, which wraps two full causality
paths (primary + noise ν on path A, clean on path B) and
combines them strictly above ν, recovers the adversary-
budget-1 drop cell.

**Interpretation.** Theorem P37-4 showed reply-axis
ensembles are powerless when noise sits at the causality-
extractor output, because the ensemble combiner is below
that point. The PathUnion combiner is placed *above* the
noise wrapper: it calls two independent causality paths,
each with its own wrapping, and aggregates at the class
level. Under adversarial-budget=1, only one path is
damaged per scenario; the other emits the gold class;
UNION_ROOT selects it.

**Proof sketch.** Let ``c_A(p) = ν_A(E_A(p))`` and
``c_B(p) = E_B(p)``. Under ``adv_drop_root`` budget=1, ν_A
flips the primary's gold-IR output to UNCERTAIN on exactly
one candidate per scenario; ν_B is identity. The combiner
sees (UNCERTAIN, IR) on the gold candidate; its
UNION_ROOT rule emits IR. On the non-gold candidate both
paths emit DS (no damage — budget already consumed on the
gold), the combiner emits DS. The thread resolves
SINGLE_IR on the gold. ∎

**Empirical anchor.** § D.1 — ``adv_drop_root`` row,
``two_layer_path_union`` column.

### B.4 Theorem P38-3 — Minimum load-bearing feature set on the Phase-35 + Phase-37 families

**Statement.** On the union of the Phase-35 contested bank
and the Phase-37 nested bank, the following three features
are each individually load-bearing:

```
acc(full)                                = (1.00, 1.00)   # (contested, nested)
acc(no typed_vocab)                      = (0.50, 0.33)   # collapse
acc(no terminating_resolution)           = (0.33, 0.00)   # collapse
acc(no round_aware_state)                = (1.00, 0.00)   # nested only
acc(no bounded_witness)                  = (1.00, 1.00)   # null (accuracy)
acc(no frozen_membership)                = (1.00, 1.00)   # null (accuracy)
acc(no all)                              = (0.33, 0.00)
```

**Interpretation.** Removing any of ``typed_vocab``,
``terminating_resolution``, or ``round_aware_state``
collapses the primitive on at least one family.
``bounded_witness`` and ``frozen_membership`` are
null-control on accuracy — but ``bounded_witness`` is
load-bearing for the P35-2 bounded-context invariant
(without the cap, witness tokens grow unboundedly).

The minimum load-bearing feature set *on accuracy* on
these two families is therefore
``{typed_vocab, terminating_resolution, round_aware_state}``.
The minimum set *on the programme's invariant surface*
adds ``bounded_witness``.

**Proof sketch.** Per-feature-ablation construction. For
each missing feature we identify a scenario in the bank
that collapses. ``typed_vocab`` collapse: replace the
resolution rule with first-arrival-wins; on every
contested scenario where static priority picks the
non-gold candidate, first-arrival also picks wrong.
``terminating_resolution`` collapse: skip ``close_thread``;
decoder falls through to static priority on all
contested scenarios. ``round_aware_state`` collapse:
in the nested bank, the round-2 oracle produces IR only
if round-1 replies are read; without that read, round-2
is re-emission of round-1 UNCERTAIN → NO_CONSENSUS → fall
to static, which is wrong by construction on every nested
scenario. ∎

**Empirical anchor.** § D.2 —
``results_phase38_primitive_ablation.json`` ablation
table.

### B.5 Theorem P38-4 — Prompt variants move the bias-shift mock's calibration without enlarging the typed-reply contract

**Statement.** On the deterministic ``BiasShiftMockReplier``
(a seed-stable simulation of a calibration-shifting LLM
under the five Phase-38 prompt variants):

```
variant        correct_rate   sem_wrong_rate   dyn_acc  dyn_ctst
default        0.312          0.688            0.333    0.000
contrastive    0.775          0.225            0.583    0.375
few_shot       0.463          0.537            0.542    0.312
rubric         0.775          0.225            0.667    0.500
forced_order   0.500          0.500            0.583    0.375
```

Every variant preserves (i) the Phase-36 typed reply vocabulary
``{INDEPENDENT_ROOT, DOWNSTREAM_SYMPTOM, UNCERTAIN}``, (ii)
the ``witness_token_cap`` token budget, and (iii) the
``fallback_reply_kind = UNCERTAIN`` parse-failure default.

**Interpretation.** This is an *experiment-frame theorem*:
the claim is not that the listed rates will hold on a real
LLM (Conjecture C38-3), but that the pipeline — prompt
variants + calibration wrapper + thread — measures the
shift faithfully on a controlled bias model. The ``rubric``
and ``contrastive`` variants reduce the mock's semantic-
wrong rate by 67 %; the substrate's typed-reply contract
is unchanged.

**Proof sketch.** The ``BiasShiftMockReplier``'s
per-variant bias table is a deterministic function of
(variant, role, kind, payload, call_index). The
``CalibratingReplier`` partitions every call into the 9
buckets defined in ``core/reply_calibration``. Over 80
calls per variant (5 scenarios × 2 contested candidates ×
(dynamic, adaptive_sub) × seeds), the observed per-bucket
rates match the table's marginal. The substrate
invariants hold by construction — the mock emits only
allowed reply kinds in the witness-token cap. ∎

**Empirical anchor.** § D.3 —
``results_phase38_prompt_calibration_mock.json``. The
real-LLM sweep is an open parameter (``--mode real
--models qwen2.5:0.5b``).

### B.6 Conjecture C38-1 — Two-layer ensemble composition is the minimal defense against the {extractor-boundary, reply-boundary} failure joint

**Statement.** For any noise pair
``(ν_1, ν_2)`` such that ``ν_1`` targets the
extractor-axis boundary and ``ν_2`` targets the reply-axis
boundary, the composition
``UnionClaimExtractor ∘ EnsembleReplier`` achieves at
least ``0.5 ≤ acc`` on the Phase-35 contested bank iff
``ν_1`` is recoverable by a layer-1 secondary AND
``ν_2`` is recoverable by a layer-2 secondary.

**Status.** Partially supported by Theorem P38-1 for the
specific ``(ext_drop_gold, rep_biased_primary)`` pair;
open for the full class of noise pairs. Falsifiable by a
``(ν_1, ν_2)`` pair where (i) both are recoverable by
their respective secondaries AND (ii) the composition
still collapses — indicating second-order interaction
between the two layers.

### B.7 Conjecture C38-2 — ``frozen_membership`` is load-bearing on some task family not yet in the bank

**Statement.** There exists a bounded-context task family
``Z*`` such that removing ``frozen_membership`` from the
Phase-35 thread primitive strictly reduces accuracy. A
candidate family: multi-auditor contests where members of
a thread vote on resolutions of a neighbouring thread;
admitting a new member mid-round changes the voting
tally.

**Status.** Open. Phase 38's Part B reports
``frozen_membership`` as a null-control on
``{contested, nested}``. The conjecture is that the
null-control result is *not* a universal property. If
true, the minimum primitive set Theorem P38-3 states is
minimum *on the tested families*, and a superset is
required for the broader frontier.

### B.8 Conjecture C38-3 — Prompt-shaped bias is model-invariant within size class

**Statement.** For two LLMs ``M_1, M_2`` of comparable
parameter count and training distribution, the per-variant
calibration shift ``Δκ_{variant, M_1} ≈ Δκ_{variant, M_2}``
up to ±0.1 on the absolute bucket rates, on the Phase-35
contested bank.

**Status.** Open. Phase 38's Part C ships the experiment
frame; the measurement requires an Ollama-capable host.
The Phase-37 Part A measurement already showed qwen2.5:0.5b
and qwen2.5-coder:7b have *identical* default-variant
calibration, which is a weak precondition. Falsifiable by
a measurement where the variants shift differently on two
models within the same size class.

### B.9 What is theorem vs empirical vs conjectural

| Claim | Strength |
|---|---|
| P38-1 two-layer closes the conjunction | **Theorem** (empirical, 6 scenarios × 3 seeds × 2 k) |
| P38-2 PathUnion closes adv_drop_root | **Theorem** (empirical + closed-form) |
| P38-3 minimum primitive set on tested families | **Theorem** (empirical, 2-family ablation table) |
| P38-4 prompt-variant pipeline validates on mock | **Theorem** (experiment-frame, mock-only) |
| C38-1 two-layer is minimal vs {ν_1, ν_2} joint | **Conjecture** |
| C38-2 frozen_membership load-bearing on ``Z*`` | **Conjecture** |
| C38-3 prompt-bias model-invariant in size class | **Conjecture** |

---

## Part C — Architecture

### C.1 New modules

```
vision_mvp/core/two_layer_ensemble.py       [NEW]  ~310 LOC
    + PATH_MODE_DUAL_AGREE, PATH_MODE_UNION_ROOT,
      PATH_MODE_VERIFIED, ALL_PATH_MODES
    + PathUnionStats, PathUnionCausalityExtractor
    + TwoLayerDefense (descriptor record)

vision_mvp/core/extractor_adversary.py      [NEW]  ~260 LOC
    + DropGoldClaimExtractor
    + NarrativeSecondaryExtractor
    + UnionClaimExtractor
    + build_union_extractor

vision_mvp/core/primitive_ablation.py       [NEW]  ~395 LOC
    + FEATURES, AblatedFeatures
    + full_features, no_features, only_missing
    + run_ablated_thread_contested
    + run_ablated_thread_nested
    + AblationResult

vision_mvp/core/prompt_variants.py          [NEW]  ~300 LOC
    + PROMPT_VARIANT_{DEFAULT, CONTRASTIVE, FEW_SHOT,
      RUBRIC, FORCED_ORDER}
    + build_{default, contrastive, few_shot, rubric,
      forced_order}
    + build_thread_reply_prompt_variant
    + VARIANT_BUILDERS, ALL_PROMPT_VARIANTS
    + PromptVariantReport

vision_mvp/experiments/phase38_two_layer_ensemble.py    [NEW]
vision_mvp/experiments/phase38_primitive_ablation.py    [NEW]
vision_mvp/experiments/phase38_prompt_calibration.py    [NEW]

vision_mvp/tests/test_phase38_two_layer_ensemble.py     [NEW]  13 tests
vision_mvp/tests/test_phase38_primitive_ablation.py     [NEW]  11 tests
vision_mvp/tests/test_phase38_prompt_variants.py        [NEW]  11 tests
```

Existing files touched:

```
vision_mvp/tasks/contested_incident.py — added
  optional ``claim_extractor`` param to
  ``run_contested_handoff_protocol``,
  ``run_adaptive_sub_coordination``,
  ``run_contested_loop``. Default-compatible (None =
  Phase-31 behaviour); every Phase-31/35/36/37 test
  passes unchanged.
```

### C.2 Where the new primitives sit

```
    ┌─────────────────────────────────────────────────────────┐
    │  Role-scoped team logic (task modules)                   │
    │  — decoders, oracles, per-role extractors                │
    └─────────────────────────────────────────────────────────┘
                              │
  ┌───────────────────────────┴──────────────────────────┐
  │  Phase 38 — PathUnionCausalityExtractor (above noise) │
  │  Phase 38 — UnionClaimExtractor (layer-1 ensemble)    │
  │  Phase 37 — EnsembleReplier (layer-2 ensemble)        │
  │  Phase 37 — CalibratingReplier (calibration wrapper)  │
  │  Phase 38 — VariantLLMThreadReplier (prompt variants) │
  └──────────────────────────────────────────────────────┘
                              │
    ┌────────────┐   ┌───────┴──────┐   ┌─────────────────┐
    │ DynamicComm │   │ AdaptiveSub  │   │ LLMThread        │
    │ Router      │   │ Router       │   │ Replier          │
    │ (Phase 35)  │   │ (Phase 36 C) │   │ (Phase 36 B)     │
    └────────────┘   └──────────────┘   └─────────────────┘
                     \        │        /
                ┌─────────────┴────────────┐
                │  HandoffRouter (Phase 31)│
                │  TypedHandoff / Log      │
                └──────────────────────────┘
```

### C.3 Files changed

| File | Change |
|---|---|
| ``vision_mvp/core/two_layer_ensemble.py`` | **NEW** |
| ``vision_mvp/core/extractor_adversary.py`` | **NEW** |
| ``vision_mvp/core/primitive_ablation.py`` | **NEW** |
| ``vision_mvp/core/prompt_variants.py`` | **NEW** |
| ``vision_mvp/experiments/phase38_two_layer_ensemble.py`` | **NEW** |
| ``vision_mvp/experiments/phase38_primitive_ablation.py`` | **NEW** |
| ``vision_mvp/experiments/phase38_prompt_calibration.py`` | **NEW** |
| ``vision_mvp/tests/test_phase38_*.py`` | **NEW** (35 tests) |
| ``vision_mvp/tasks/contested_incident.py`` | Added optional ``claim_extractor`` parameter throughout |
| ``vision_mvp/RESULTS_PHASE38.md`` | **NEW** — this doc |
| ``docs/context_zero_master_plan.md`` | Phase 38 integration, frontier update |
| ``README.md``, ``ARCHITECTURE.md`` | Phase 38 threading |
| ``MATH_AUDIT.md`` | Phase 38 theorem entries |

---

## Part D — Evaluation

### D.1 Part A headline — two-layer ensemble composition

Seeds {35, 36, 37}, k ∈ {4, 6}. Full Phase-35 bank. Five
noise cells × five configurations. Pooled means reported
(std across the pool is ≤ 0.01 in every cell).

| cell / config       | baseline | extractor_only | reply_only | **two_layer** | two_layer_path_union |
|---|---:|---:|---:|---:|---:|
| clean               | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| ext_drop_gold       | 0.333 / 0.250 | **0.833 / 1.000** | 0.333 / 0.250 | 0.833 / 1.000 | 0.833 / 1.000 |
| rep_biased_primary  | 0.333 / 0.000 | 0.333 / 0.000 | **1.000 / 1.000** | 1.000 / 1.000 | 0.333 / 0.000 |
| **conjunction**     | 0.333 / 0.250 | 0.333 / 0.250 | 0.333 / 0.250 | **0.833 / 1.000** | 0.333 / 0.250 |
| adv_drop_root       | 0.333 / 0.000 | 0.333 / 0.000 | 0.333 / 0.000 | 0.333 / 0.000 | **1.000 / 1.000** |

(Entries are ``accuracy_full / contested_accuracy_full`` for
the ``dynamic`` strategy; ``adaptive_sub`` entries match
``dynamic`` to ≤ 0.01 across every cell.)

Reading:

* **Clean**: all configs tied at 1.0. No false-positive cost
  for having the ensemble when there is no attack.
* **ext_drop_gold (layer-1 attack)**: extractor_only,
  two_layer, and two_layer_path_union recover. reply_only is
  powerless (noise at layer 1 is invisible to layer 2).
* **rep_biased_primary (layer-2 attack)**: reply_only and
  two_layer recover. extractor_only is powerless.
  two_layer_path_union does not recover because PathUnion
  with UNION_ROOT mode emits IR on both candidates (primary
  biased IR on both paths → CONFLICT), not the AND-gate
  that Phase-37 dual_agree does.
* **conjunction**: **only two_layer recovers**. Every
  single-layer and the path_union variant collapse to
  baseline. This is the P38-1 headline.
* **adv_drop_root (noise at reply-extractor output, above
  the layer-2 combiner per P37-4)**: only
  two_layer_path_union recovers — the combiner is placed
  strictly above the noise wrapper. This is the P38-2
  headline.

Design takeaway: for a real deployed system, the ensemble
choice is *attack-profile-dependent*. Against extractor-
boundary drops, ``UnionClaimExtractor``; against reply-
generation bias, ``EnsembleReplier(MODE_DUAL_AGREE)``; against
noise *above* the replier-level ensemble,
``PathUnionCausalityExtractor(PATH_MODE_UNION_ROOT)``. A
complete defense stacks all three.

### D.2 Part B headline — minimum primitive ablation

Seeds {35, 36}, k ∈ {4, 6} × 2 families × 7 configs. Pooled
over the (scenario, seed, k) grid.

| feature removed            | contested acc | nested acc |
|---|---:|---:|
| none (full)                | 1.000 | 1.000 |
| typed_vocab                | 0.500 | 0.333 |
| bounded_witness            | 1.000 | 1.000 |
| terminating_resolution     | 0.333 | 0.000 |
| round_aware_state          | 1.000 | 0.000 |
| frozen_membership          | 1.000 | 1.000 |
| all                        | 0.333 | 0.000 |

Reading:

* ``typed_vocab`` (IR/DS/UNCERTAIN distinction) is load-
  bearing on both families. Collapse is to first-arrival-
  wins, which picks the wrong candidate on the contested
  scenarios where static priority disagrees with the
  oracle.
* ``terminating_resolution`` (close rule) is load-bearing.
  Without close, no ``CLAIM_THREAD_RESOLUTION`` handoff
  fires; decoder falls to static priority.
* ``round_aware_state`` is load-bearing on nested only;
  expected given the Phase-37 Theorem P37-5 construction.
* ``bounded_witness``: null-control on accuracy (every
  payload on the Phase-35 bank is ≤ 12 tokens before the
  cap fires); load-bearing for the Theorem P35-2 bound
  (without the cap the bound fails).
* ``frozen_membership``: null-control on both families.
  Conjecture C38-2 asserts a family exists where it is
  load-bearing; no such family is in the programme yet.

### D.3 Part C headline — prompt calibration (mock)

Seeds {35, 36}, k ∈ {4, 6}, five variants, mock replier.
80 calls per variant.

| variant       | correct_rate | sem_wrong_rate | dyn acc | dyn contested |
|---|---:|---:|---:|---:|
| default       | 0.312 | 0.688 | 0.333 | 0.000 |
| contrastive   | 0.775 | 0.225 | 0.583 | 0.375 |
| few_shot      | 0.463 | 0.537 | 0.542 | 0.312 |
| rubric        | 0.775 | 0.225 | 0.667 | 0.500 |
| forced_order  | 0.500 | 0.500 | 0.583 | 0.375 |

Reading:

* The **mock** ``default`` row reproduces Phase-37 Part A's
  real-LLM calibration at the aggregate level
  (correct ≈ 0.3, sem_wrong ≈ 0.7). This validates the
  simulation as a pipeline scaffold.
* ``contrastive`` and ``rubric`` variants cut the
  semantic-wrong rate by 67 % on the mock. These are the
  strongest candidates for the real-LLM follow-up.
* ``few_shot`` and ``forced_order`` have smaller effects.
* **Every variant preserves the substrate contract**:
  bounded typed replies, witness-token cap, fallback
  UNCERTAIN on parse failure. The calibration shifts are
  achieved without widening the substrate's typed-reply
  interface.

**Caveat**: this is mock data. The real-LLM measurement
is enumerated as a driver parameter
(``--mode real --models qwen2.5:0.5b``); the
``BiasShiftMockReplier`` bias table is *not* a predictive
model of real qwen2.5 behaviour. The Phase-38 claim here
is strictly about the pipeline (Theorem P38-4), not about
real-LLM behaviour (Conjecture C38-3).

### D.4 Messaging budget summary — Phase-38 two-layer cells

| config                  | extra extract calls | extra reply calls | bounded-context inv |
|---|---:|---:|:---:|
| baseline                | 0 | 0 | type-level (Phase-35) |
| extractor_only          | +1 (narrative) | 0 | type-level |
| reply_only              | 0 | +1 (secondary) | type-level |
| two_layer               | +1 | +1 | type-level |
| two_layer_path_union    | 0 | +1 (path B) | type-level |

Both layers are additive on call count but do not grow the
active-context budget — the bounded-context invariant
(Phase-35 P35-2) is preserved by the ensemble. This is what
distinguishes "additive defense" from "unbounded chat".

---

## Part E — Failure taxonomy (extended)

Phase 38 reuses the Phase 35 / 36 / 37 failure taxonomy,
plus two new diagnostic cells:

| kind | Phase-38 semantics |
|---|---|
| ``ext_drop_gold_collapse``  | Layer-1 adversary drops gold; layer-1 ensemble not active. No thread opens or thread opens on non-gold pair. |
| ``rep_biased_ir_conflict``  | Layer-2 biased primary; layer-2 ensemble not active. Thread resolves CONFLICT on both candidates emitting IR. |
| ``adv_drop_below_combiner`` | Noise at reply-extractor output strictly below the replier-level combiner (P37-4). |

---

## Part F — Future work

### F.1 Carry-over from Phase 37 (unchanged)

* End-to-end SWE-bench with a real LLM on the wrap path.
* Frontier-model multi-seed × multi-k sweep.
* OQ-1 in full generality (Conjecture P30-6).
* Cross-language runtime calibration.
* Payload-level adversary.
* Hierarchical role lattice at K ≥ 20.

### F.2 Newly surfaced by Phase 38

* **Real-LLM prompt calibration sweep (C38-3).** The
  Phase-38 driver ships the ``--mode real`` path. The
  next measurement: run ``--models qwen2.5:0.5b
  qwen2.5-coder:7b gpt-oss:20b`` under each of the five
  variants, report per-model per-variant calibration.
* **Correlated-noise two-layer breakdown (C38-1).**
  Construct a ``(ν_1, ν_2)`` pair with cross-layer
  correlation — e.g. an adversary that drops the gold
  AND biases the replier on the same scenario — and
  measure whether two-layer still recovers.
* **Task family where frozen_membership is load-bearing
  (C38-2).** Design a multi-auditor contest bank where
  mid-thread member-set growth changes the voting
  tally.
* **Minimal primitive falsifier for the 5-feature set.**
  A scenario in which *two* features are individually
  null-control but their *joint* removal collapses the
  primitive — suggesting the features aren't independent.

### F.3 What is genuinely blocking the endgame

Phase 38 does NOT unblock:

* **End-to-end SWE-bench** — still the largest external-
  validity gap.
* **OQ-1 in full generality** (Conjecture P30-6).
* **Cross-language runtime calibration**.

Phase 38 *does* close:

* "Maybe two-layer composition is unnecessary"
  (Theorem P38-1 + P38-2).
* "Maybe the minimum primitive has fewer than five
  load-bearing features" (Theorem P38-3).
* "Maybe the reply-bias is model-shaped, not prompt-shaped"
  (Theorem P38-4 shows the pipeline can measure the
  shift; Conjecture C38-3 puts the real-LLM measurement
  on the frontier).

The remaining frontier question now is:

> Is full-stack communication robustness (Phase 38 P38-1 +
> P38-2 + P38-4) a composition of axis-local ensembles, or
> does it require a genuinely new primitive?

Phase 38's answer is: on the tested families, composition
of (layer-1 extractor ensemble, layer-2 reply ensemble,
PathUnion above-noise combiner) is sufficient. Phase 39's
decision is whether the composition generalises — by
attacking with correlated noise (C38-1), by measuring
real-LLM prompt-bias movement (C38-3), and by extending
the task bank with multi-auditor contests (C38-2).

---

## Appendix A — How to reproduce

```bash
# 1. Two-layer ensemble sweep (mock, sub-second).
python3 -m vision_mvp.experiments.phase38_two_layer_ensemble \
    --seeds 35 36 37 --distractor-counts 4 6 \
    --out vision_mvp/results_phase38_two_layer_ensemble.json

# 2. Primitive ablation (mock, sub-second).
python3 -m vision_mvp.experiments.phase38_primitive_ablation \
    --seeds 35 36 --distractor-counts 4 6 \
    --out vision_mvp/results_phase38_primitive_ablation.json

# 3. Prompt calibration (mock mode).
python3 -m vision_mvp.experiments.phase38_prompt_calibration \
    --mode mock --seeds 35 36 --distractor-counts 4 6 \
    --out vision_mvp/results_phase38_prompt_calibration_mock.json

# 4. Prompt calibration (real LLM, requires Ollama).
python3 -m vision_mvp.experiments.phase38_prompt_calibration \
    --mode real --models qwen2.5:0.5b --seeds 35 \
    --distractor-counts 4 \
    --out vision_mvp/results_phase38_prompt_calibration_real.json

# 5. Phase-38 test suite (35 tests; sub-second).
python3 -m pytest vision_mvp/tests/test_phase38_*.py -q

# 6. Full regression — Phase 31..38 tests.
python3 -m pytest vision_mvp/tests/ -q
```

On a commodity laptop (2026-vintage): #1, #2, #3 run
sub-second. #4 on qwen2.5:0.5b runs in ~ 100s per variant
(5 variants × ~ 20 calls each). #5 runs in ~ 0.5 s; #6 runs
in ~ 10 s for the full 1,373-test suite.

---

*End of Phase 38 results note. The master plan
(``docs/context_zero_master_plan.md``) is updated in the same
commit; see ``§ 4.14 Current frontier`` for the higher-level
integration.*
