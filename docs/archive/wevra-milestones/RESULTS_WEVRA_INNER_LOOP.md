# SDK v3.4 — sub-sub-intra-cell PROMPT / LLM_RESPONSE slice + inner-loop research

> Theory-forward milestone note. SDK v3.4, 2026-04-26. Anchors:
> `vision_mvp/wevra/capsule.py` (PROMPT, LLM_RESPONSE),
> `vision_mvp/wevra/capsule_runtime.py` (`seal_prompt`,
> `seal_llm_response`),
> `vision_mvp/wevra/runtime.py` (`_make_intra_cell_hooks`,
> `_seal_prompt_response_pair`, `_real_cells`, `_synthetic_cells`),
> `vision_mvp/wevra/synthetic_llm.py` (`SyntheticLLMClient`,
> `SYNTHETIC_MODEL_PROFILES`),
> `vision_mvp/wevra/lifecycle_audit.py` (L-9 / L-10 / L-11),
> `vision_mvp/experiments/parser_boundary_cross_model.py`,
> `vision_mvp/tests/test_wevra_capsule_native_inner_loop.py`
> (16 new tests).
> Status taxonomy follows
> [`docs/HOW_NOT_TO_OVERSTATE.md`](HOW_NOT_TO_OVERSTATE.md).

## What is materially newly true

1. **The LLM byte boundary is lifecycle-governed.** Up to SDK
   v3.3 the prompt the patch generator sent and the raw bytes
   the LLM returned were plain Python strings inside
   ``_real_cells._gen``. Under v3.4, every LLM call seals two
   capsules in flight: a ``PROMPT`` capsule (parent: SWEEP_SPEC)
   recording the prompt's SHA-256, byte length, model tag,
   prompt style, and a bounded text snippet (≤ 4 KiB), and an
   ``LLM_RESPONSE`` capsule (parent: PROMPT) recording the
   response's SHA-256, byte length, snippet, and elapsed
   milliseconds. Theorems **W3-42** (PROMPT lifecycle gate) and
   **W3-43** (prompt → response parent gate) are proved by
   inspection.

2. **The full inner-loop chain is a typed DAG.** The
   PARSE_OUTCOME capsule may now parent on
   ``(SWEEP_SPEC, LLM_RESPONSE)`` so the inner-loop chain is
   ``PROMPT → LLM_RESPONSE → PARSE_OUTCOME → PATCH_PROPOSAL →
   TEST_VERDICT`` end-to-end on the capsule DAG. Coordinate
   consistency between PARSE_OUTCOME and LLM_RESPONSE is
   mechanically checked by the new lifecycle-audit invariant
   **L-11** (``CapsuleLifecycleAudit._check_l11``). Theorem
   **W3-44** is the structural claim; Theorem **W3-45** is the
   audit-soundness extension to L-1..L-11.

3. **Synthetic mode lets the inner-loop slice run in CI.**
   ``SweepSpec(mode="synthetic", synthetic_model_tag=<tag>)``
   uses a deterministic in-process synthetic LLM client (see
   ``vision_mvp.wevra.synthetic_llm``) instead of an Ollama
   endpoint. The full prompt/response/parse/patch/verdict chain
   exercises in-flight; no network is required. This lets
   contract tests + the cross-model parser-boundary experiment
   run on every commit.

4. **The parser-boundary attribution layer is empirically
   sharp.** Conjecture W3-C6 (the SDK v3.4 sharper reading of
   the legacy W3-C4) reports cross-distribution PARSE_OUTCOME
   failure-kind Total Variation Distance up to **1.000** on the
   bundled bank, and strict→robust parser-mode shift TVD up to
   **1.000** on ``synthetic.unclosed`` (the parser flips
   entirely from ``unclosed_new`` failure to ``ok +
   recovery=closed_at_eos``). The closed-vocabulary failure
   taxonomy is sharp enough to detect distribution shifts in
   LLM output.

## The new theorems / conjectures

### Theorem W3-42 (PROMPT lifecycle gate) — proved

For any capsule-native run with sealed SWEEP_SPEC `s`:

  1. Every sealed PROMPT capsule `ρ` has
     `parents(ρ) = (cid(s),)`.
  2. ``seal_prompt`` is **idempotent on content** (Capsule
     Contract C1).
  3. The PROMPT payload is *coordinates + content hash +
     bounded snippet* — not the full prompt bytes.

**Proof.** Direct inspection of ``seal_prompt`` in
``capsule_runtime.py``. $\square$

### Theorem W3-43 (Prompt → response parent gate) — proved

For every sealed LLM_RESPONSE capsule `ν`:

  1. `|parents(ν)| = 1`.
  2. `parents(ν)[0]` is a sealed PROMPT capsule's CID.
  3. ``seal_llm_response`` is idempotent on content.
  4. Admission of `ν` depends on the prompt being already
     sealed (Capsule Contract C5).

**Proof.** Direct inspection of ``seal_llm_response``. $\square$

### Theorem W3-44 (PARSE_OUTCOME → LLM_RESPONSE chain consistency) — proved + mechanically-checked

For every PARSE_OUTCOME `p` whose parent set contains an
LLM_RESPONSE `ν`:

  1. `cid(s) ∈ parents(p)` AND `cid(ν) ∈ parents(p)`.
  2. The coordinate fields
     `(instance_id, parser_mode, apply_mode, n_distractors)`
     in `p.payload` and `ν.payload` are equal.
  3. The ``strategy`` field is permitted to differ — multiple
     strategies sharing one LLM call collapse to one
     LLM_RESPONSE while seeding distinct PARSE_OUTCOMEs (the
     ``raw_cache`` deduplicates by strategy_proxy).

**Proof.** Inspection of ``seal_parse_outcome``
(``llm_response_cid`` parameter; raises if the CID is not
sealed) and ``_make_intra_cell_hooks`` /
``_seal_prompt_response_pair`` (the four non-strategy
coordinates are constants over the cell). The lifecycle-audit
rule L-11 mechanically checks coordinate consistency on every
finished run. $\square$

### Theorem W3-45 (Lifecycle-audit soundness extends to L-1..L-11) — proved + mechanically-checked

`audit_capsule_lifecycle(ctx).verdict == "OK"` iff the
underlying ledger satisfies the eleven invariants L-1..L-11
defined in ``lifecycle_audit.py`` (the SDK v3.3 set plus L-9 /
L-10 / L-11 added in v3.4).

**Proof.** Same structural argument as W3-40 — the audit's
``run()`` accumulates every violation across the eleven
``_check_l*`` methods; OK iff zero violations. $\square$

### Conjecture W3-C6 (synthetic-LLM cross-distribution PARSE_OUTCOME variance) — empirical

Conditional on the synthetic distribution library
``vision_mvp.wevra.synthetic_llm.SYNTHETIC_MODEL_PROFILES``,
the cross-distribution PARSE_OUTCOME failure-kind TVD ≥ 0.5
on at least one (distribution-pair, parser-mode) triple, and
the strict→robust parser-mode shift on
``synthetic.unclosed`` is exactly 1.000.

**Reproduction.**
```
python3 -m vision_mvp.experiments.parser_boundary_cross_model
```

On the bundled bank (57 instances) this reports
``max_cross_tvd=1.000`` and ``max_parser_mode_shift=1.000``.

**Honest scope.** The distributions are calibrated synthetic,
not real cross-LLM. The empirical claim is about the parser
failure-kind closed vocabulary's resolving power, not about
real LLM output distributions in the wild.

## Strict claim taxonomy

| Claim                  | Status                | Anchor                                                                                  |
| ---------------------- | --------------------- | --------------------------------------------------------------------------------------- |
| **W3-42**              | **proved**            | `capsule_runtime.py::seal_prompt`; `PromptCapsuleLifecycleTests`                         |
| **W3-43**              | **proved**            | `capsule_runtime.py::seal_llm_response`; `LLMResponseCapsuleLifecycleTests`             |
| **W3-44**              | **proved + mechanically-checked** | `lifecycle_audit.py::_check_l11`; `LifecycleAuditExtendedTests`             |
| **W3-45**              | **proved + mechanically-checked** | `lifecycle_audit.py::CapsuleLifecycleAudit.run`                              |
| **W3-C6**              | **empirical**         | `vision_mvp/experiments/parser_boundary_cross_model.py`                                  |
| **W3-C5 (legacy)**     | **DISCHARGED**        | Replaced by W3-42 / W3-43 / W3-44 / W3-45                                                |

## What this milestone does NOT claim

- It does **not** claim "Wevra is now fully capsule-native." The
  sandbox stdout/stderr layer and the parser's internal
  regex/recovery state remain plain Python.
- It does **not** claim a real cross-LLM study. The W3-C6
  empirical result uses calibrated synthetic distributions.
- It does **not** redefine W3-34 spine equivalence — the SWEEP_SPEC
  payload is unchanged; PROMPT / LLM_RESPONSE / PARSE_OUTCOME /
  PATCH_PROPOSAL / TEST_VERDICT are all non-spine kinds.
- It does **not** change the META_MANIFEST detached-witness
  trust unit — meta-artefact authentication is still one hop
  beyond the primary view.

## What's the next slice?

The smallest real, load-bearing inner-loop boundary that
remains plain Python is the **sandbox layer**: the patched
test execution, the syntax-check, and the apply step's
intermediate state. Capsule-tracking these would extend the
chain to ``PROMPT → LLM_RESPONSE → PARSE_OUTCOME →
PATCH_PROPOSAL → APPLY_OUTCOME → TEST_VERDICT`` and would
require rejecting the current strong invariant that
TEST_VERDICT has exactly one parent. That is the natural SDK
v3.5 candidate.

A second next-slice candidate is real cross-LLM W3-C6
(replacing the synthetic library with a sweep over real Ollama
LLMs); this is straightforward to layer on by passing
``llm_client=LLMClient(...)`` to ``_real_cells``.
