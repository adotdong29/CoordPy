# Capsule-Native Runtimes: Lifecycle-Bounded Execution as a Type Discipline for Multi-Agent LLM Systems

> Working draft. Authors: the Context Zero programme.
> Status as of SDK v3.4 (2026-04-26): submission-quality structure
> with strict claim taxonomy. Extension of the SDK v3.1 / v3.2 /
> v3.3 milestone notes into a single paper-shaped argument.
> SDK v3.4 added the sub-sub-intra-cell **PROMPT / LLM_RESPONSE
> slice** — capsule-native execution now extends to the LLM byte
> boundary, the load-bearing inner-loop frontier identified in
> the prior version's "Future work" section.

## Abstract

Multi-agent LLM systems use raw bytes — prompt strings, tool-call
dicts, JSON blobs — as the unit of context flowing across role,
layer, and run boundaries. We propose **capsule-native execution**:
every cross-boundary artifact is a typed, content-addressed,
lifecycle-bounded, budget-bounded, provenance-stamped *capsule*,
and the runtime drives execution by sealing capsules in a
hash-chained ledger rather than by passing untyped state. We
implement this discipline in **CoordPy**, an open-source
SWE-bench-Lite-shape evaluation runtime, and prove five classes of
guarantees the discipline buys:

1. **Lifecycle correspondence** between runtime stages and capsule
   states (Theorem W3-32; intra-cell extension W3-32-extended;
   sub-intra-cell extension W3-39; **sub-sub-intra-cell extension
   W3-42 / W3-43 / W3-44 to the LLM byte boundary**, SDK v3.4).
2. **Content addressing at write time** for substantive artifacts
   (Theorem W3-33), with audit-time on-disk re-verification
   (Theorem W3-38). PROMPT / LLM_RESPONSE capsules extend
   content-addressing inside the LLM call boundary (SDK v3.4).
3. **A sharp impossibility theorem on meta-artifact authentication**
   inside the primary ledger (Theorem W3-36) and a constructive
   detached-witness corollary.
4. **Deterministic-mode replay** that collapses the full capsule DAG
   across runs of the same logical input (Theorem W3-41), enabling
   cross-machine CI gates with byte-identical capsule CIDs.
5. **Mechanically-checkable inner-loop chain.** A
   ``CapsuleLifecycleAudit`` verifies eleven invariants L-1..L-11
   on every finished run (Theorems W3-40 / W3-45). The end-to-end
   inner-loop chain
   ``PROMPT → LLM_RESPONSE → PARSE_OUTCOME → PATCH_PROPOSAL →
   TEST_VERDICT`` is observable as a typed DAG with strong
   parent-CID gating; coordinate consistency between
   PARSE_OUTCOME and LLM_RESPONSE is mechanically verified
   (rule L-11).

We separate strictly from the substrate-ML and decoder-frontier
results elsewhere in the programme: the contributions of this
paper are about the *runtime discipline*, not about model
accuracy. We do, however, anchor a small **parser-boundary
empirical study** (W3-C6, SDK v3.4): conditional on a calibrated
synthetic LLM-output distribution library, the parser's
PARSE_OUTCOME failure-kind closed vocabulary distinguishes
distributions with cross-distribution Total Variation Distance
up to 1.000, and the strict→robust parser-mode choice shifts
the failure footprint by up to 1.000 on
``synthetic.unclosed``. The honest scope: this is **not** a
real cross-LLM measurement; it is a calibrated synthetic
distribution library that exercises the parser's failure-kind
closed vocabulary's resolving power.

## 1. Problem statement

Modern agent harnesses cross many trust boundaries — between roles
in a team, between layers (parser → matcher → sandbox), between
runs of the same evaluation. The dominant unit of context across
those boundaries is a raw string or a raw dict: a prompt, a
tool-call envelope, a JSONL row, a JSON product report. These
units have:

- **No identity.** A prompt is its bytes; two prompts that differ
  by one trailing newline are nominally distinct but
  operationally identical, and there is no canonical content
  address.
- **No type.** A "context blob" passed from analyzer to model
  carries no closed-vocabulary kind; downstream code re-parses.
- **No lifecycle.** A blob goes from "doesn't exist" to "is in
  memory" to "is on disk" with no enforced ordering between
  consumers.
- **No budget.** Arbitrarily large blobs can flow through
  arbitrary layers; bounded-context discipline (if any) is
  enforced by ad hoc length checks.
- **No provenance.** The relationship between this blob and its
  upstream causes is implicit.

In multi-agent LLM systems, this absence costs reproducibility,
auditability, and (we argue) correctness. The standard fix —
"add metadata" — is necessary but not sufficient: metadata that
can be detached from the blob is metadata that can drift.

## 2. The Capsule Contract

A **Context Capsule** is an immutable object satisfying six
invariants C1–C6:

1. **Identity (C1).** `cid = SHA-256(canonical(kind, payload, budget,
   sort(parents)))`.
2. **Typed claim (C2).** `kind ∈ ALL` for a closed vocabulary.
3. **Lifecycle (C3).** `PROPOSED → ADMITTED → SEALED [→ RETIRED]`.
4. **Budget (C4).** Explicit `(max_tokens, max_bytes, max_rounds,
   max_witnesses, max_parents)`. Admission rejects.
5. **Provenance (C5).** Capsules form an acyclic DAG; the ledger
   is hash-chained.
6. **Frozen (C6).** Sealed capsules are immutable.

These invariants are inherited from Merkle DAGs / Git / IPFS
(C1, C5), actor and event-sourcing systems (C2, C3),
capability-style references (C2, C6), and tamper-evident logs (C5).
What is novel is the **end-to-end unification**: every
cross-boundary object in a CoordPy run satisfies the *same* contract,
and a runtime context drives the run through their lifecycles.

## 3. Capsule-native execution

Up to SDK v3.0, capsules were a *post-hoc fold* — a finished run's
product report was lifted into a capsule DAG after the fact. SDK
v3.1 introduced a runtime context (`CapsuleNativeRunContext`) that
seals one capsule per stage:

```
PROFILE → READINESS_CHECK → SWEEP_SPEC → SWEEP_CELL_n
                                       → PROVENANCE
                                       → ARTIFACT_k → RUN_REPORT
                                                    → META_MANIFEST (detached)
```

Each stage's capsule is sealed before the next stage can read its
result; the parent-CID gate (C5) refuses to admit a capsule whose
declared parents are not yet in the ledger. A failed stage leaves
a typed PROPOSED-but-never-sealed entry in an in-flight register
(`ctx.in_flight_failures()`).

SDK v3.2 extended the discipline past the cell boundary into the
inner sweep loop. Each `(task, strategy)` pair seals two capsules:
a **PATCH_PROPOSAL** (parent: SWEEP_SPEC) when the generator
produces a patch, and a **TEST_VERDICT** (parent: PATCH_PROPOSAL)
when the sandbox reports a verdict. The chain `patch → verdict` is
enforced at the type level.

SDK v3.3 extended the discipline one further structural layer with
the **PARSE_OUTCOME** capsule sealed *before* every PATCH_PROPOSAL.
The parser's structured outcome — `ok`, closed-vocabulary
`failure_kind` from `ALL_PARSE_KINDS`, `recovery` label,
substitutions count, bounded detail — becomes a typed witness on
the capsule DAG. The PATCH_PROPOSAL's parent set now contains both
the SWEEP_SPEC and the PARSE_OUTCOME; the parse → patch → verdict
chain has a typed DAG witness.

SDK v3.4 extended the discipline **one further structural layer**
into the LLM byte boundary itself. Two new kinds — **PROMPT** and
**LLM_RESPONSE** — wrap the LLM call. PROMPT carries
*coordinates + prompt SHA-256 + byte length + bounded text
snippet (≤ 4 KiB)*; its parent is SWEEP_SPEC. LLM_RESPONSE carries
*coordinates + response SHA-256 + byte length + bounded snippet +
elapsed milliseconds*; its parent is exactly the upstream PROMPT.
The PARSE_OUTCOME's parent set is then either `(SWEEP_SPEC,)` (on
the deterministic-oracle path, no LLM call) or
`(SWEEP_SPEC, LLM_RESPONSE)` (on the LLM-backed path). The
end-to-end inner-loop chain is therefore
`PROMPT → LLM_RESPONSE → PARSE_OUTCOME → PATCH_PROPOSAL →
TEST_VERDICT`, a **five-link typed DAG** rooted at the cell's
SWEEP_SPEC. Theorems W3-42 / W3-43 / W3-44 / W3-45 are proved by
inspection; lifecycle audit invariants L-9 / L-10 / L-11
mechanically check the chain on every run.

### 3.1 What is in flight vs out of flight (SDK v3.4)

| Layer                          | Capsule kind            | Status              | Parent      |
| ------------------------------ | ----------------------- | ------------------- | ----------- |
| Profile resolution             | PROFILE                 | capsule-native      | (root)      |
| Readiness verdict              | READINESS_CHECK         | capsule-native      | PROFILE     |
| Sweep configuration            | SWEEP_SPEC              | capsule-native      | PROFILE     |
| Per-cell metrics               | SWEEP_CELL              | capsule-native      | SWEEP_SPEC  |
| **Per-call LLM prompt bytes**  | **PROMPT**              | **capsule-native (SDK v3.4)** | SWEEP_SPEC  |
| **Per-call LLM response bytes** | **LLM_RESPONSE**       | **capsule-native (SDK v3.4)** | PROMPT      |
| **Per-(task,strat) parse**     | **PARSE_OUTCOME**       | **capsule-native**  | SWEEP_SPEC  (+ LLM_RESPONSE on LLM-backed path) |
| **Per-(task,strat) patch**     | **PATCH_PROPOSAL**      | **capsule-native**  | SWEEP_SPEC + PARSE_OUTCOME |
| **Per-(task,strat) test**      | **TEST_VERDICT**        | **capsule-native**  | PATCH_PROPOSAL |
| Provenance manifest            | PROVENANCE              | capsule-native      | PROFILE     |
| Substantive artifacts          | ARTIFACT                | capsule-native + content-addressed at write time | various |
| Run report                     | RUN_REPORT              | capsule-native      | spine kinds |
| Meta-artifact witness          | META_MANIFEST           | detached witness    | (secondary) |
| Sandbox stdout/stderr          | (none)                  | not capsule-native  | —           |
| Parser-internal regex/recovery state | (none)            | not capsule-native  | —           |

**The honest scope claim** of this paper: capsule-native execution
covers the *load-bearing* artifacts of a CoordPy run — every artifact
whose presence or absence is observable in the report or whose
bytes are content-checkable. SDK v3.4 brings the LLM byte boundary
inside this scope. Sandbox stdout/stderr and the parser's internal
recovery-heuristic state remain plain Python — these are
**volatile sub-step bytes** whose capsule wrapping is the natural
SDK v3.5 candidate.

## 4. Theorems

We list the load-bearing theorems and their epistemic status.

### 4.1 Identity / lifecycle / chain

- **W3-7** — Identity is a homomorphism on parent canonicalisation
  (proof: by inspection of `_capsule_cid`). [proved]
- **W3-9** — The ledger DAG is acyclic and topologically equal to
  append order (proof: `admit` requires every parent CID already
  in the ledger). [proved]
- **W3-10** — Chain tamper-evidence under SHA-256 second-preimage
  hardness. [proved, conditional on cryptographic assumption]

### 4.2 Capsule-native execution

- **W3-32** — Lifecycle ↔ execution-state correspondence on the
  spine kinds (PROFILE / READINESS_CHECK / SWEEP_SPEC /
  SWEEP_CELL / PROVENANCE / ARTIFACT / RUN_REPORT). [proved by
  inspection]
- **W3-32-extended** — W3-32 lifts to PATCH_PROPOSAL and
  TEST_VERDICT with patch → verdict ordering enforced at the
  parent-CID gate. [proved by inspection]
- **W3-33** — Content addressing at artifact creation time: the
  on-disk SHA-256 of any substantive artifact equals the SHA-256
  in its ARTIFACT capsule's payload at
  `seal_and_write_artifact` return. [proved by inspection +
  cross-validation test]
- **W3-34** — In-flight ↔ post-hoc CID equivalence on
  non-ARTIFACT spine kinds. [proved by per-kind set-equality test]
- **W3-35** — Parent-CID gating is the execution contract: a
  stage that misorders the lifecycle raises a typed exception,
  not an obscure downstream error. [proved by inspection]
- **W3-39 (new in SDK v3.3)** — PARSE_OUTCOME lifecycle gate +
  parse → patch → verdict DAG chain. A PARSE_OUTCOME outside a
  sealed SWEEP_SPEC is rejected; a PATCH_PROPOSAL declaring a
  non-sealed PARSE_OUTCOME parent is rejected (C5 carry-over).
  [proved by inspection + contract test]
- **W3-42 (new in SDK v3.4)** — PROMPT lifecycle gate. Every
  sealed PROMPT capsule has parent set exactly `(SWEEP_SPEC,)`;
  a PROMPT outside a sealed SWEEP_SPEC is rejected. The capsule
  payload carries the prompt's SHA-256, byte length, model
  tag, prompt style, and a bounded text snippet (≤ 4 KiB).
  Two byte-identical prompts collapse to one capsule (C1
  idempotence). [proved by inspection + contract test]
- **W3-43 (new in SDK v3.4)** — Prompt → response parent gate.
  Every LLM_RESPONSE capsule has exactly one parent, and that
  parent is a sealed PROMPT. An LLM_RESPONSE declaring a
  non-sealed prompt CID is rejected (C5). [proved by inspection
  + contract test]
- **W3-44 (new in SDK v3.4)** — PARSE_OUTCOME → LLM_RESPONSE
  chain coordinate consistency. When PARSE_OUTCOME's parent set
  contains an LLM_RESPONSE, their `(instance_id, parser_mode,
  apply_mode, n_distractors)` coordinate fields are equal. The
  `strategy` field is permitted to differ (multiple strategies
  share an LLM call when the prompt is identical). [proved by
  inspection + lifecycle audit invariant L-11]

### 4.3 Verification

- **W3-37** — Chain-from-headers verification: the chain step is
  a pure function of on-disk header fields (cid, kind, lifecycle),
  so an auditor recomputes the chain head from disk bytes alone.
  [proved by inspection]
- **W3-38** — ARTIFACT audit-time on-disk re-hash: every ARTIFACT
  capsule's payload SHA-256 is checkable against the on-disk
  file's SHA-256 at audit time. [proved by inspection]
- **W3-40 (new in SDK v3.3)** — Lifecycle-audit soundness. A
  `CapsuleLifecycleAudit` returning `verdict == "OK"` witnesses
  the eight invariants L-1..L-8 over the finished ledger.
  Counterexamples surface as typed violations. [proved by
  inspection + contract test]
- **W3-45 (new in SDK v3.4)** — Lifecycle-audit soundness
  extends to L-1..L-11 (the SDK v3.3 set plus
  L-9 / L-10 / L-11 added in v3.4). Same structural argument
  as W3-40. [proved by inspection + mechanically-checked]
- **W3-41 (new in SDK v3.3)** — Deterministic-mode CID
  determinism. Two runs of the same deterministic profile under
  `RunSpec(deterministic=True)` produce byte-identical capsule
  CIDs on every kind, identical chain head, identical root CID.
  [proved by inspection of the canonicalisation set + contract
  test asserting set-equality across two runs]

### 4.4 Impossibility / limitation theorems

- **W3-14** — Per-capsule budget locality: no per-capsule budget
  can enforce a cardinality invariant `Card_Φ(N)` globally.
  [proved by counterexample construction]
- **W3-16** — Relational limitation post-cohort-lifting: cohort
  admission is silent on relational invariants over distinct
  pairs of capsules. [proved by counterexample construction]
- **W3-17** — Admission-locality bound: every admission rule
  whose decision depends only on capsule headers is bounded above
  by the priority-decoder ceiling under ceiling-forcing spurious
  injection. [proved, conditional]
- **W3-21** — Linear-class sign-flip asymmetry: no class-agnostic
  linear decoder over a feature-set with cross-domain sign-flip
  closes a strict zero-shot transfer Gate 2. [proved]
- **W3-29** — Bayes-divergence zero-shot risk lower bound on the
  magnitude-monoid linear family. [proved]
- **W3-36 (SDK v3.2)** — Meta-artifact circularity: no extension
  of the primary ledger admits an ARTIFACT for any meta-artifact
  whose bytes encode the rendered RUN_REPORT view, without
  changing the rendered view. [proved by structural argument +
  ledger-extension contradiction]

The detached-witness corollary of W3-36: a META_MANIFEST capsule
in a *secondary* ledger is the strongest authentication achievable.
The trust unit is one explicit hop beyond the primary view.

## 5. Empirical anchor

We do not report a leaderboard. The capsule-native runtime is a
*systems contribution*, not an accuracy contribution. We anchor
the discipline empirically with:

- **101 contract tests** (`vision_mvp/tests/test_coordpy_*.py`,
  `test_capsule_*.py`) on SDK v3.2; **+18 tests** for SDK v3.3
  covering PARSE_OUTCOME, lifecycle audit, and deterministic
  mode; **+16 tests** for SDK v3.4 covering PROMPT /
  LLM_RESPONSE / synthetic mode / W3-C6 cross-model parser
  boundary (181 capsule tests passing locally as of 2026-04-26).
- **A bundled local_smoke profile** that produces a 154-capsule
  graph on a default mock-mode run: 1 PROFILE, 1
  READINESS_CHECK, 1 SWEEP_SPEC, 2 SWEEP_CELL, 48 PARSE_OUTCOME,
  48 PATCH_PROPOSAL, 48 TEST_VERDICT, 1 PROVENANCE, 3 ARTIFACT,
  1 RUN_REPORT, 1 META_MANIFEST (in secondary ledger). Chain-head
  verification passes; on-disk re-hash passes; lifecycle audit
  returns OK on all 11 rules (SDK v3.4).
- **Synthetic-mode end-to-end demonstration.** A 4-instance
  synthetic run (``mode="synthetic", synthetic_model_tag=
  "synthetic.unclosed"``) produces a chain
  ``PROMPT → LLM_RESPONSE → PARSE_OUTCOME → PATCH_PROPOSAL →
  TEST_VERDICT``. Under strict parser mode every PARSE_OUTCOME
  has ``failure_kind=unclosed_new``; under robust every
  PARSE_OUTCOME has ``failure_kind=ok, recovery=closed_at_eos``
  — the strict→robust shift is observable as a coordinate-aligned
  shift in the LLM_RESPONSE-keyed PARSE_OUTCOME multiset.
- **Cross-run determinism** under `deterministic=True`: two
  independent runs collapse to identical chain head and root
  CID across every capsule kind.

## 6. Honest claim taxonomy

We classify every theorem-style claim by epistemic status. The
goal is to make hostile-reviewer-grade audit cheap.

| Status                | Definition                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------- |
| **proved**            | Mathematical proof or proof-by-inspection of code; reviewer reads the proof.                |
| **proved-conditional**| Proof depends on a stated assumption (cryptographic hardness, distributional premise).      |
| **mechanically-checked** | Property is checked on every run by a runtime audit; soundness is by inspection.        |
| **empirical**         | Measured on the bundled bench; reproducible from a published seed.                          |
| **conjectural**       | Stated, falsifiable, but not yet proved or systematically tested.                           |
| **retracted**         | Earlier statement now believed false or scope-limited; prior reading is withdrawn.           |

| Claim   | Description                                                              | Status                |
| ------- | ------------------------------------------------------------------------ | --------------------- |
| W3-7    | CID identity is parent-permutation-invariant                             | proved                |
| W3-9    | Ledger DAG is acyclic and append-order is topo                            | proved                |
| W3-10   | Chain tamper-evidence under SHA-256                                       | proved-conditional    |
| W3-12   | Capsule view is a faithful header projection                              | proved                |
| W3-13   | DAG height ≤ 4 on canonical run                                           | proved                |
| W3-14   | Per-capsule budget locality                                               | proved (negative)     |
| W3-15   | Cohort lift                                                                | proved                |
| W3-16   | Relational limitation post-cohort-lifting                                 | proved (negative)     |
| W3-17   | Admission-locality bound                                                   | proved-conditional    |
| W3-18   | Plurality > priority on coherent-majority bundles                          | proved-conditional    |
| W3-19   | Learned bundle decoder breaks 0.200 ceiling at +15pp on $n=80$            | empirical             |
| W3-20   | Deep Sets sufficiency over linear class                                    | proved                |
| W3-21   | Linear-class sign-flip asymmetry                                            | proved                |
| W3-22   | Pooled-multitask symmetric transfer at $n=80$                              | empirical             |
| W3-23   | DeepSet best-cell 0.425 at $n=80$                                          | empirical             |
| W3-24   | Post-search winner's-curse bias                                             | proved                |
| W3-25   | (deferred — see master plan §4.16)                                         | proved-conditional    |
| W3-26   | DeepSet best-cell at $n=320$ drops to 0.362                                | empirical             |
| W3-27   | 6-family zero-shot max-penalty +0.112 at $n=320$                           | empirical             |
| W3-28   | Sign-stable DeepSet zero-shot gap = 0.000 at level 0.237                   | empirical             |
| W3-29   | Bayes-divergence zero-shot risk lower bound on linear family               | proved                |
| W3-30   | Strict separation: relational decoder vs. magnitude-monoid                  | proved                |
| W3-31   | Empirical relational-decoder level-lift                                     | empirical             |
| W3-32   | Lifecycle ↔ execution-state correspondence (spine)                          | proved                |
| W3-32-extended | W3-32 carry-over to intra-cell (PATCH_PROPOSAL, TEST_VERDICT)         | proved                |
| W3-33   | Content addressing at artifact creation time                                 | proved                |
| W3-34   | In-flight ↔ post-hoc CID equivalence on non-ARTIFACT spine                  | proved                |
| W3-35   | Parent-CID gating is the execution contract                                  | proved                |
| W3-36   | Meta-artifact circularity (sharp impossibility)                              | proved (negative)     |
| W3-37   | Chain-from-headers verification                                                | proved                |
| W3-38   | ARTIFACT audit-time on-disk re-hash                                            | proved                |
| **W3-39** | **PARSE_OUTCOME lifecycle gate + parse → patch → verdict DAG chain**       | **proved**            |
| **W3-40** | **Lifecycle-audit soundness on L-1..L-8**                                  | **proved + mechanically-checked** |
| **W3-41** | **Deterministic-mode CID determinism on full DAG**                         | **proved + empirical (set-equality test across two runs)** |
| **W3-42 (SDK v3.4)** | **PROMPT lifecycle gate (parent = SWEEP_SPEC; idempotent on content)**       | **proved**            |
| **W3-43 (SDK v3.4)** | **Prompt → response parent gate**                                            | **proved**            |
| **W3-44 (SDK v3.4)** | **PARSE_OUTCOME → LLM_RESPONSE chain coordinate consistency**                | **proved + mechanically-checked** |
| **W3-45 (SDK v3.4)** | **Lifecycle-audit soundness extends to L-1..L-11**                           | **proved + mechanically-checked** |
| W3-C1   | General subsumption — every Phase-N bounded-context theorem subsumes        | conjectural           |
| W3-C5 (legacy) | Relational-axis extension closes W3-16                                       | conjectural           |
| W3-C7 (strict) | Strict point-estimate Gate-1 + zero-shot Gate-2 paradigm-shift threshold | retracted (W3-26, W3-27) |
| W3-C9   | Refined paradigm-shift reading: $n=80$ point-estimate + zero-shot gap       | conjectural (candidate) |
| W3-C10  | Relational decoder level-ceiling                                                | conjectural           |
| **W3-C4 (legacy SDK v3.3)** | **PARSE_OUTCOME failure_kind distribution stable across LLM tags** | **superseded by W3-C6** |
| **W3-C5 (legacy SDK v3.3)** | **Sub-intra-cell PROMPT/LLM_RESPONSE capsule slice closes the inner-loop boundary without breaking spine equivalence W3-34** | **DISCHARGED in SDK v3.4 by W3-42 / W3-43 / W3-44 / W3-45** |
| **W3-C6 (new SDK v3.4)** | **Synthetic-LLM cross-distribution PARSE_OUTCOME failure-kind TVD ≥ 0.5; strict→robust shift up to 1.000** | **empirical (synthetic distribution library; not a real cross-LLM measurement)** |

## 7. How not to overstate this

A reader who finishes this paper and takes away any of the
following statements has misread the contribution.

- *"CoordPy makes LLM agent runs deterministic."* **Mis-reading.**
  The runtime is deterministic *on the capsule DAG* under
  `deterministic=True` for mock-mode profiles whose JSONL is
  frozen. Real-LLM mode is non-deterministic by construction
  (the LLM is sampling); the deterministic-mode flag does not
  change that.
- *"Capsules eliminate prompt injection."* **Mis-reading.** The
  capsule contract is a coordination-context discipline, not an
  adversarial-input filter. A capsule whose payload is itself a
  prompt-injection string seals normally; the capsule layer
  records but does not sanitise.
- *"Capsule-native means the entire runtime is capsule-tracked."*
  **Mis-reading.** LLM prompts, raw LLM responses, and sandbox
  stdout/stderr remain plain Python. The discipline covers the
  load-bearing run-boundary and intra-cell artifacts; it
  deliberately does not cover volatile sub-step bytes.
- *"Theorem W3-41 means every CoordPy run produces the same CID."*
  **Mis-reading.** Two runs collapse to the same CIDs only under
  `deterministic=True` and only on a deterministic profile.
  Without the flag, PROVENANCE / RUN_REPORT CIDs vary by design
  (timestamp / wall-clock variance).
- *"The detached META_MANIFEST signs the run."* **Mis-reading.**
  The META_MANIFEST is a *content-addressed witness*, not a
  cryptographic signature. Authenticity-against-an-adversary
  requires orthogonal signing; the manifest's claim is that
  *anyone with the bytes can verify the bytes are the bytes the
  run produced*, not that an adversary did not produce a different
  run.

## 8. What this changes about CoordPy's originality claim

The honest contribution of capsule-native execution is **not** a
new cryptographic primitive (we use SHA-256), **not** a new
DAG layout (Merkle DAGs are 50 years old), and **not** a new
coordination protocol (actor / event-sourcing systems have
content-addressed events). What is new is the
**unification at the runtime contract level**: every cross-boundary
artifact in a complete LLM agent runtime — profile, readiness,
sweep config, sweep cells, parser outcomes, patch proposals, test
verdicts, provenance, on-disk artifacts, meta-artifacts — satisfies
**one** contract, and the runtime drives execution by sealing
capsules in a hash-chained ledger rather than by passing raw state.

The contribution is sharper than the v3.1 framing because:

1. The intra-cell extension (v3.2) demonstrates the discipline
   *scales past the run boundary into the inner sweep loop*
   without breaking spine equivalence.
2. The sub-intra-cell extension (v3.3) demonstrates the
   discipline *scales one further structural layer* (parser axis)
   without breaking spine equivalence.
3. The deterministic-mode result (v3.3) demonstrates the
   discipline *enables byte-identical capsule DAG comparison
   across machines*, a property that is impossible for raw-state
   runtimes without after-the-fact normalisation.
4. The meta-artifact circularity theorem (v3.2) is a sharp
   limitation, not a hand-wave: the runtime cannot authenticate
   its own rendering inside the primary ledger; the detached
   witness is the constructive boundary.

## 9. Future work

- **Sub-intra-cell PROMPT / LLM_RESPONSE capsule slice (SDK
  v3.4).** ✅ **DISCHARGED** by Theorems W3-42 / W3-43 / W3-44 /
  W3-45. The bytes-size challenge was disposed of by recording
  the prompt / response SHA-256 + byte length + a bounded
  snippet (≤ 4 KiB), rather than the raw bytes. Two
  byte-identical prompts collapse to one capsule (idempotent
  on content) so cached LLM calls produce one capsule pair on
  the DAG. The on-disk view always carries PROMPT /
  LLM_RESPONSE payloads (additive to the prior
  ``payload_kinds_always`` invariant).
- **Sandbox stdout/stderr / parser-internal state slice (SDK
  v3.5 candidate).** The remaining inner-loop bytes that are
  still plain Python. Capsule-tracking the apply outcome could
  produce a six-link chain
  ``PROMPT → LLM_RESPONSE → PARSE_OUTCOME → PATCH_PROPOSAL →
  APPLY_OUTCOME → TEST_VERDICT`` and would relax the strict
  invariant that TEST_VERDICT has exactly one parent.
- **Real cross-LLM W3-C6 study.** ✅ **PARTIALLY DISCHARGED**
  in SDK v3.6 by W5-1 on the available model class. Real
  cross-LLM measurement on Mac 1 Ollama (``qwen2.5:14b-32k``
  vs ``qwen3.5:35b``) yields cross-model PARSE_OUTCOME
  failure-kind TVD = 1.000 on strict parsing (n=10), collapsing
  to 0.000 on robust — saturated and reproducible. The
  full discharge remaining is the 70 B-class measurement
  across both Macs via MLX distributed (Mac 2 currently offline
  at measurement time). See
  ``docs/archive/coordpy-milestones/RESULTS_COORDPY_DISTRIBUTED.md``.
- **Cryptographic signing on top of the META_MANIFEST.** Authenticity
  against an adversary is orthogonal but composable.
- **Cross-run determinism on full DAG without flag.** Could be
  achievable by stripping wall-clock and host-local fields
  unconditionally; the trade-off is loss of forensic
  context (when did the run happen?).
- **Synthetic→real-LLM cross-round transfer (SDK v3.13).**
  ✅ **PARTIALLY DISCHARGED** by the W12 family. The synthetic
  W11 cross-round bundle decoder (SDK v3.12) does **not** transfer
  to a calibrated real-LLM-shaped producer-noise channel
  (W12-Λ at the real-LLM axis: under ``synonym_prob=0.50,
  svc_token_alt_prob=0.30`` synthetic-noisy-LLM extractor, the
  un-normalised W11 ties FIFO at 0.000). A *normalising* multi-
  round bundle decoder
  (``RobustMultiRoundBundleDecoder = closed-vocabulary
  CLAIM_KIND_SYNONYMS + payload-rewrite layer ahead of W11``)
  closes the gap (W12-1: +1.000 vs un-normalised W11, stable 5/5
  seeds), conditional on the producer-noise channel being bounded
  by the closed-vocabulary closure (W12-2). The honest revised
  reading is three-layered: un-normalised admission does not
  transfer (W6-C2); un-normalised cross-round decoding does not
  transfer (W12-Λ); normalised cross-round decoding transfers,
  conditional on closure (W12-1). The next move is the W12-C2
  Phase-59 ``ollama`` opt-in mode against a real Ollama-served
  model. See ``docs/RESULTS_COORDPY_REAL_LLM_MULTI_ROUND.md`` and
  ``vision_mvp/experiments/phase59_real_llm_multi_round.py``.

## 10. Code anchor

- Capsule contract: `vision_mvp/coordpy/capsule.py`
- Capsule-native runtime: `vision_mvp/coordpy/capsule_runtime.py`,
  `vision_mvp/coordpy/runtime.py`,
  `vision_mvp/product/runner.py`
- Lifecycle audit: `vision_mvp/coordpy/lifecycle_audit.py`
- Synthetic LLM (SDK v3.4): `vision_mvp/coordpy/synthetic_llm.py`
- Cross-model parser-boundary experiment (SDK v3.4):
  `vision_mvp/experiments/parser_boundary_cross_model.py`
- Formal model: `docs/CAPSULE_FORMALISM.md` (§ 4.J for SDK v3.4)
- Milestone notes: `docs/archive/coordpy-milestones/RESULTS_COORDPY_INNER_LOOP.md` (SDK v3.4)
- Tests: `vision_mvp/tests/test_coordpy_capsule_native*.py`,
  `test_coordpy_capsule_native_deeper.py`,
  `test_coordpy_capsule_native_inner_loop.py` (SDK v3.4),
  `test_capsule_properties.py`,
  `test_capsule_subsumption.py`
- TLA+ spec: `vision_mvp/formal/CapsuleContract.tla`
- Theorems: `docs/THEOREM_REGISTRY.md` (canonical registry)

A reader who wants to falsify any claim in this paper can clone
the repository, run `pip install -e .[docker]`, and execute
`pytest vision_mvp/tests/test_coordpy_*.py -v`. The 18 SDK v3.3
contract tests are in `test_coordpy_capsule_native_deeper.py`; the
16 new SDK v3.4 tests are in `test_coordpy_capsule_native_inner_loop.py`;
their failure modes are the falsifiers for W3-39 / W3-40 / W3-41
(v3.3) and W3-42 / W3-43 / W3-44 / W3-45 / W3-C6 (v3.4).
