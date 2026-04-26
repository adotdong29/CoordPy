# How not to overstate this

> Canonical do-not-overstate rules for the Context Zero / Wevra
> programme. Every milestone note, paper draft, README claim, or
> README-of-README must satisfy these rules. Last touched: SDK
> v3.3, 2026-04-26.

The programme has a long history of moves where a candidate result
was written up too strongly and later had to be sharpened or
retracted. This document is the canonical rule-book that prevents
that; reviewers should reject any text that violates it.

## Status vocabulary (definitions)

These are the only labels permitted on theorem-style claims in
this repo. Unlabelled claims are forbidden.

- **proved** — Mathematical proof, or proof-by-inspection of code
  that a reviewer can read in under 30 lines. The proof is in
  `docs/CAPSULE_FORMALISM.md` or `PROOFS.md` or in the
  docstring of the relevant code. No simulation, no
  "experiments-show", no implicit cryptographic hardness.
- **proved-conditional** — Proof depends on a stated assumption.
  The assumption is named in the theorem statement (e.g. SHA-256
  second-preimage hardness; coherent-majority distribution
  premise; multinomial-logistic strict convexity).
- **mechanically-checked** — A runtime audit verifies the
  property on every run. The audit's soundness is by inspection
  (the audit code is short and the failure mode is enumerated).
  Compare to "test-passed-once" — mechanical-check is per-run.
- **empirical** — Measured on a published bench (`local_smoke`,
  `bundled_57`, `noisy_phase31`, etc.) from a published seed.
  Reproducible from `python -m vision_mvp.experiments.<phase>`
  with default args.
- **conjectural** — Stated, falsifiable, but not yet proved or
  systematically tested. The conjecture statement names the
  falsifier ("If we observe X, the conjecture is refuted").
- **retracted** — Earlier reading withdrawn; replaced by a more
  honest reading. The retraction names *why* the earlier reading
  was wrong (a counterexample, a tighter measurement, a sharper
  separation) and points to the replacement claim.

## Forbidden moves

### "Paradigm shift" without a stated reading

> *"This is a paradigm shift."*

If you write the phrase "paradigm shift" anywhere in this repo,
you must immediately specify the reading: under which
quantitative bar, on which bench, at which $n$. The phrase
"paradigm-shift candidate" is permitted only when followed by
"under W3-Cn" where W3-Cn is a named conjecture in
`THEOREM_REGISTRY.md`. The strict reading W3-C7 is **retracted**.

### "Solves" without a defining gate

> *"Wevra solves context."*

If you write "solves" or "closes" about any open problem, you
must name the *defining gate* and state which side of the gate
you are on. "Solves the bounded-context problem" is forbidden;
"closes the per-agent O(log N) bound on the routing-only setting
of Phases 1–10" is permitted because the bound is named.

### "Beats" without a baseline

> *"Our decoder beats the priority decoder."*

If you write "beats" or "outperforms", you must name the baseline
and the bench. "Beats" by 5pp on `local_smoke` at $n=80$ is fine;
"beats" without bench is not.

### "Honest" without a referent

> *"The honest reading is..."*

The phrase "honest" is overused. Reserve it for *explicit
contrast* with a named less-honest reading: "The strict reading is
retracted; the honest defensible reading is W3-C9." Do not use
"honest" as a generic intensifier.

### Retroactive scope reduction

> *"The result is at $n=80$."*

If a result was originally claimed at $n=320$ and is now reported
at $n=80$, you must state the retraction explicitly: "Originally
claimed at $n=320$; subsequent measurement at $n=320$ falsified
the strict reading (W3-26)." Silently dropping $n$ is forbidden.

### Asymmetric framing of negative results

> *"The decoder works."*

If a result holds in one direction and not the other, the
statement must reflect both. "Direction-invariant under sign-stable
DeepSet (gap = 0.000) at level 0.237" is permitted.
"DeepSet achieves symmetric zero-shot transfer" is forbidden
(it suggests level-matching, which is false).

### Authority citations without inspection

> *"Theorem W3-37 says…"*

Before you cite a theorem in a new milestone note, verify the
theorem still says what you remember. Theorems can be sharpened
(W3-32 → W3-32-extended), conditional premises can be added or
removed, and conjectures can be promoted or retracted. Always
check `THEOREM_REGISTRY.md` first.

### Labelling the runtime "fully capsule-native"

> *"The Wevra runtime is fully capsule-native."*

Forbidden until sandbox stdout/stderr, parser-internal regex /
recovery state, and on-the-wire LLM streaming chunks are
capsule-tracked. **SDK v3.4 narrows the gap**: PROMPT and
LLM_RESPONSE bytes ARE now capsule-tracked (Theorems W3-42 /
W3-43). The honest current reading is *"capsule-native at the
run boundary, intra-cell pair, parse outcome, and the
prompt/response boundary; not capsule-native at the sandbox
stdio layer or the parser's internal recovery state."*

### Labelling the META_MANIFEST a "signature"

> *"The META_MANIFEST signs the run."*

Forbidden. The META_MANIFEST is a *content-addressed witness*,
not a cryptographic signature. Its claim is that the bytes hash
to the recorded SHA, not that an adversary did not produce them.
Use "witness" or "manifest"; "signature" implies authentication
against an adversary, which we do not provide.

### "Fully reproducible" without scope

> *"Wevra runs are fully reproducible."*

Forbidden without scope. The accurate statement is:
"Under `RunSpec(deterministic=True)` with a frozen JSONL and a
deterministic-profile sandbox, the **capsule DAG** is reproducible
byte-for-byte (W3-41). Wall-clock fields and host-local paths
are stripped from CIDs; the on-disk product report still records
them for forensic context."

## Required moves

### State the falsifier

Every conjectural claim must name what would falsify it.
"W3-C5: a sub-intra-cell PROMPT capsule slice closes the
inner-loop boundary" is incomplete; "Falsifiers: PROMPT bytes too
large for any reasonable budget; spine CIDs drift under the new
kind" is correct.

### State the scope

Every theorem must name its scope. "W3-32: lifecycle ↔
execution-state correspondence" is incomplete; "on the spine
kinds (PROFILE / READINESS_CHECK / SWEEP_SPEC / SWEEP_CELL /
PROVENANCE / ARTIFACT / RUN_REPORT)" is the full statement.
W3-32-extended explicitly extends to PATCH_PROPOSAL / TEST_VERDICT.

### Cross-link the code

Every theorem must point to its code anchor. The code anchor is
the falsifier of the proof: "the proof says X holds; code Y
implements it; if Y does not implement X, the proof is wrong."

### Mark retractions clearly

When a claim is retracted, do not delete it. Keep it in the
registry with status `retracted` and a one-line explanation of
*why*. Future readers must be able to see what we previously
believed and why we stopped.

### Use the same name for the same claim

If a theorem is referenced in `CAPSULE_FORMALISM.md`,
`THEOREM_REGISTRY.md`, `RESEARCH_STATUS.md`,
`papers/wevra_capsule_native_runtime.md`, and a milestone note,
the *name and number* must be byte-identical across files.
"Theorem W3-32" never becomes "the lifecycle correspondence
result" without its number.

## How this document is enforced

1. New milestone notes start by reading
   `RESEARCH_STATUS.md` and `THEOREM_REGISTRY.md`. If a
   milestone-note claim contradicts these files, the milestone
   note must update the canonical files (or be sharpened).
2. New paper drafts must reproduce the claim taxonomy table in
   the paper itself, using the same status labels.
3. README / START_HERE must use status labels for any
   theorem-grade claim (or omit the claim).
4. PR review explicitly checks for forbidden phrases ("paradigm
   shift" without reading; "solves" without gate; "fully
   capsule-native"; "signs the run"). PRs that introduce
   forbidden phrases are rejected.

## Change log

- **2026-04-26 (SDK v3.3).** Initial canonical version. Adds
  PARSE_OUTCOME / lifecycle-audit / deterministic-mode rules
  ("not fully capsule-native", "not a signature", "not fully
  reproducible without scope").
- **2026-04-26 (SDK v3.4).** Sharpens "fully capsule-native" rule
  — PROMPT / LLM_RESPONSE bytes ARE now capsule-tracked
  (W3-42 / W3-43). Adds rule that synthetic-LLM cross-model
  research (W3-C6) must be cited as *synthetic*, not as a
  cross-LLM measurement. New forbidden phrase: "the parser
  failure-kind distribution is stable across LLMs" — the
  empirical claim only covers the calibrated synthetic
  distribution library, not real cross-LLM behaviour.
