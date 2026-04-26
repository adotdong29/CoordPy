# How not to overstate this

> Canonical do-not-overstate rules for the Context Zero / Wevra
> programme. Every milestone note, paper draft, README claim, or
> README-of-README must satisfy these rules. Last touched: SDK
> v3.9, 2026-04-26.

The programme has a long history of moves where a candidate result
was written up too strongly and later had to be sharpened or
retracted. This document is the canonical rule-book that prevents
that; reviewers should reject any text that violates it.

## Status vocabulary (definitions)

These are the only labels permitted on theorem-style claims in
this repo. Unlabelled claims are forbidden.

- **proved** — Mathematical proof, or proof-by-inspection of code
  that a reviewer can read in under 30 lines. The proof is in
  `docs/CAPSULE_FORMALISM.md` or `docs/archive/pre-wevra-theory/PROOFS.md` or in the
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

### Labelling the team-coordination layer "solves multi-agent context"

> *"Wevra solves context for multi-agent teams."*

Forbidden. SDK v3.5 ships *one* capsule-native team-coordination
slice over *one* synthetic benchmark family (Phase-52 incident-
triage) under a deterministic team decoder. The defensible
readings are:

* "On the Phase-52 incident-triage benchmark, the learned
  per-role admission policy (W4-C1) improves pooled team-decision
  accuracy by $+0.097$ full / $+0.156$ root_cause over the
  strongest fixed admission baseline at matched per-role budgets
  (default config, $n_\text{eval}=31$)."
* "Theorem W4-1 mechanically verifies team-lifecycle invariants
  T-1..T-7 on every coordination round. Theorem W4-2 proves
  coverage-implies-correctness conditional on a faithful decoder
  + sound admission. Theorem W4-3 proves a sharp local-view
  limitation: per-role budget below the role's causal-share
  floor cannot be rescued by any admission policy."

The phrase "solves" / "closes" applied to the team-coordination
slice must specify the **gate** (the named theorem and bench);
unqualified is forbidden.

### Labelling W4-C1 as a proven theorem or as a *strict* per-seed advantage

> *"The learned admission policy strictly improves accuracy over
> coverage-guided on every seed."*

Forbidden. The honest reading is the cross-seed table in
`docs/archive/wevra-milestones/RESULTS_WEVRA_TEAM_COORD.md` § "Cross-seed result":

* The learned policy admits *strictly fewer* handoffs than
  coverage-guided on every train seed (12/12) — this is the
  load-bearing positive empirical signal of W4-C1.
* The learned policy improves pooled `accuracy_full` over
  coverage-guided in 11/12 seeds (mean $+0.054$) and pooled
  `accuracy_root_cause` in 8/12 seeds (mean $+0.032$). The
  advantage is *mean-positive*, not strict per-seed. There is
  one outlier seed where root_cause underperforms by $-0.097$.
* At higher noise (spurious_prob = 0.50), coverage-guided beats
  the learned policy on root_cause (mean $-0.089$).

The accurate phrasings are: "*budget-efficiency dominance is
robust per-seed*"; "*the accuracy advantage is mean-positive on
the default noise config but not strict per-seed*"; "*the
advantage does not survive heavier noise*."

The phrasing "strictly improves" is permitted only on
``mean_n_admitted_auditor`` (handoff-count savings), not on
accuracy. **W4-C1 is a conjecture**, not a theorem.

Single-seed numbers (the historical $+0.097$ full / $+0.156$
root_cause result on one specific seed) may be cited only as
"upper-end single seed" with cross-seed numbers immediately
following. Reporting a single-seed number without the cross-seed
distribution is forbidden.

### Labelling team-level capsules as "production-grade"

> *"Wevra ships capsule-native multi-agent coordination in production."*

Forbidden. The TEAM_HANDOFF / ROLE_VIEW / TEAM_DECISION capsule
kinds ship in the SDK's closed vocabulary, but the **Wevra product
runtime** (the ``RunSpec`` → ``RUN_REPORT`` path, ``wevra-ci``,
``wevra-capsule verify``) does not seal any of them. They are
emitted only by ``TeamCoordinator`` — the multi-agent coordination
*research slice* (``vision_mvp/wevra/team_coord.py``). The honest
phrasing is: "SDK v3.5 ships a multi-agent capsule coordination
research slice that runs side-by-side with the Wevra single-run
runtime; the run-boundary product contract is unchanged."

### Conflating substrate typed handoffs with TEAM_HANDOFF capsules

> *"TypedHandoff and TEAM_HANDOFF are the same thing."*

Forbidden. They are *adjacent* but distinct:

* ``TypedHandoff`` (``vision_mvp.core.role_handoff``) is the
  Phase-31 substrate primitive — a frozen dataclass routed
  through ``HandoffRouter`` / ``RoleInbox``. The capsule layer
  does not see it natively; the ``capsule_from_handoff`` adapter
  produces a HANDOFF capsule (``CapsuleKind.HANDOFF``) from one.
* ``TEAM_HANDOFF`` (``CapsuleKind.TEAM_HANDOFF``,
  ``vision_mvp.wevra.team_coord``) is born as a capsule and has
  no substrate twin. Identity is content-addressed by the
  capsule's hash, not by a substrate ``handoff_id``.

The two paths can run side by side; they are not interchangeable
at the audit / lifecycle layer.

### Labelling the W7-2 cohort-coherence win as unconditional

> *"Cohort-coherence admission beats FIFO."*

Forbidden without the conditions. The defensible W7-2 reading
names the bench properties that the win depends on:

* **Surplus.** ``|candidates(scenario)| > K_auditor`` — without
  budget pressure, FIFO ≡ admit-all and structure_gain = 0
  identically (W7-1).
* **Foreign-service decoys.** Some auditor-routed candidates carry
  ``service=<tag>`` tokens whose tag differs from the gold
  service. Without this, cohort coherence has no signal to
  exploit.
* **Gold plurality.** The gold service tag has strictly more
  auditor-routed candidates than any decoy service tag. Without
  this, buffered cohort coherence picks the *decoy* plurality and
  ties FIFO at 0.000.
* **Buffered mode.** ``CohortCoherenceAdmissionPolicy(fixed_plurality_tag=...)``
  constructed via ``from_candidate_payloads``. The streaming
  variant (``fixed_plurality_tag=None``) is unstable under arrival
  permutation (W7-1-aux) and ties FIFO on the same bench.

Permitted: "On the Phase-54 default config (K_auditor=4, n_eval=10,
gold-plurality property, foreign-service decoys, surplus on every
scenario), the *buffered* ``CohortCoherenceAdmissionPolicy``
achieves ``accuracy_full = 1.000`` against ``capsule_fifo`` 0.000
— a +1.000 conditional structural win (W7-2)."

Forbidden: "Cohort coherence solves multi-agent context."
Forbidden: "Capsule structure beats FIFO." (Without the bench
conditions named.)

### Conflating Phase-53 and Phase-54 results

> *"SDK v3.8 reverses the SDK v3.7 result."*

Forbidden. The two milestones measure *different bench properties*
and are both true:

* **Phase-53** (SDK v3.7): real-LLM producer extractor;
  ``mean_n_admitted_auditor < K_auditor`` in every regime; FIFO
  ties every fixed strategy at ``accuracy_full = 0.800``;
  capsule layer's contribution is the **lifecycle audit**.
* **Phase-54** (SDK v3.8): deterministic candidate stream with
  cross-role service-tag coherence; gold-plurality property;
  surplus on every scenario; buffered cohort coherence beats
  FIFO by ``+1.000`` on accuracy_full.

Phase-53 is preserved exactly. Phase-54 measures a different slice
that the original bench did not surface. The honest combined
reading is the W7-1/W7-2 dichotomy: **substrate FIFO is unbeatable
under low surplus (W7-1, Phase-53); cohort coherence beats
substrate cleanly under surplus + foreign-service decoys +
gold-plurality (W7-2, Phase-54)**. Both are conditional on stated
bench properties. Neither is universal.

### Labelling the streaming cohort variant as the load-bearing policy

> *"Cohort-coherence admission is the SDK v3.8 win."*

Forbidden without specifying *which* cohort variant. The streaming
variant (``CohortCoherenceAdmissionPolicy()`` with
``fixed_plurality_tag=None``) is unstable under arrival permutation
(W7-1-aux); on the Phase-54 default it ties FIFO at 0.000. The
buffered variant is the load-bearing policy. The honest phrasing:
"The *buffered* ``CohortCoherenceAdmissionPolicy`` (constructed
via ``from_candidate_payloads``) is the SDK v3.8 win."

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
- **2026-04-26 (SDK v3.5).** Adds team-coordination rules:
  forbidden phrases "solves multi-agent context" without a
  named theorem-bench gate; W4-C1 cited as a theorem; team-level
  capsule kinds described as production-grade; conflation of
  ``TypedHandoff`` with ``TEAM_HANDOFF``. Adds the canonical
  defensible reading template for the Phase-52 result.
- **2026-04-26 (SDK v3.8).** Adds W7 rules: forbidden phrases
  "cohort-coherence admission beats FIFO" without the
  bench-property conditions; "SDK v3.8 reverses SDK v3.7" without
  the W7-1/W7-2 dichotomy framing; "cohort-coherence is the
  SDK v3.8 win" without specifying *buffered* (vs streaming)
  variant.
- **2026-04-26 (SDK v3.9).** Adds W8 rules: forbidden phrases
  "cross-role corroboration beats W7-2" without the
  bench-property conditions; "we solved multi-agent context"
  without naming the **strong success bar** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` § 1.1; "the
  three-regime win is universal" without naming the W8-4
  falsifier regime.
- **2026-04-26 (SDK v3.10).** Adds W9 rules: forbidden phrases
  "multi-service corroboration beats W8" without the
  bench-property conditions (multi-service-gold + both gold
  services 2-role corroborated + single-role decoy storm);
  "we solved multi-agent context" still forbidden after the
  Phase-56 result — it now spans **four** named regimes, not
  all of multi-agent reality; "the four-regime win is universal"
  without naming the W9-4 falsifier regime (decoy-corroborated
  decoy); "W8 was wrong" — W8 is *unchanged* and still wins on
  Phase 55, W9 is a strict generalisation that adds Phase 56,
  not a refutation.

### Labelling the W8-1 win "the W8 multi-service-gold falsifier" (named for SDK v3.10)

> *"On multi-service-gold benches, single-tag corroboration is
> sufficient."*

Forbidden. The **W8 multi-service-gold falsifier** is named in the
v3.10 milestone: when ``gold_services`` has size > 1, the W8 buffered
policy admits only the top-1 corroborated tag and the decoder's
``services`` set is a singleton, so ``services_correct`` fails by
set equality regardless of the corroboration signal's quality. The
SDK v3.10 ``MultiServiceCorroborationAdmissionPolicy`` (W9) was
built specifically to address this falsifier; the SDK v3.10
contract tests
(``Phase56DefaultConfigTests::test_w8_falsifies_on_phase56``) gate
the falsifier mechanically.

The defensible reading: "W8 is sufficient on single-service-gold
benches; W9 is required for multi-service gold."

### Labelling the W9-1 multi-service win as unconditional

> *"Multi-service corroboration beats W8 on multi-agent benchmarks."*

Forbidden without the conditions. The defensible W9-1 reading
names the bench properties:

* **Surplus.** ``|candidates(scenario)| > K_auditor`` — without
  budget pressure, FIFO ≡ admit-all and structure_gain = 0
  identically (W7-1).
* **Multi-service gold.** ``|gold_services| ≥ 2``. On
  single-service-gold benches, W9 collapses to W8 (W9-3) and
  beats nothing W8 doesn't.
* **Both gold services cross-role corroborated.** Each gold
  service has ≥ ``min_corroborated_roles`` distinct producer
  roles. Without this, the gold tag is below the role threshold
  and W9 admits nothing tagged.
* **Single-role decoy storm only.** Every decoy service has
  ≤ 1 distinct producer role. If a decoy is also corroborated by
  ≥ ``min_corroborated_roles`` distinct roles, W9 admits the
  decoy (the W9-4 falsifier).

If any of these fails, the W9-1 win does not necessarily hold.

### Labelling the SDK v3.10 result "we solved multi-agent context"

> *"SDK v3.10 solves multi-agent context."*

Still **forbidden** after the Phase-56 result. The defensible
reading is that SDK v3.10 ships the **second consecutive
strong-bar conditional structural win** (this time on R-56
multi-service gold), the structural win now spans **four** named
regimes (R-53 / R-54 / R-55 / R-56), and is the **first programme
result whose strict-gain regime is not solvable by the previous
SDK's strongest method** — but real multi-agent reality has more
axes than four pre-committed regimes (heterogeneous producers,
time-varying budgets, multi-round handoffs, multi-service
incidents with `|gold| ≥ 3`, decoder-side coordination).

Defensible phrasings:

* "SDK v3.10 clears the strong success bar of
  `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1 (R-56
  anchor) on the Phase-56 multi-service-gold + cross-role-
  corroborated bench."
* "SDK v3.10 is the first programme result to strictly separate
  multi-service top-K corroboration from single-tag corroboration
  (W9-1 vs W8); the win is conditional on the multi-service-gold
  + single-role-decoy property; the W9-4 falsifier regime is the
  named structural limit."
* "Four regimes anchored, the W9-1 conditional win is sharp; the
  W9-4 falsifier (decoy corroboration) is the next axis to attack
  — by the W9-C1 bundle-aware decoder companion."

### Labelling the W8-1 corroboration win as unconditional

> *"Cross-role corroboration beats W7-2 on multi-agent benchmarks."*

Forbidden without the conditions. The defensible W8-1 reading
names the bench properties:

* **Surplus.** ``|candidates(scenario)| > K_auditor`` — without
  budget pressure, FIFO ≡ admit-all and structure_gain = 0
  identically (W7-1).
* **Decoy raw plurality.** Some decoy service has strictly more
  raw mentions in the auditor stream than gold. Without this,
  W7-2 single-tag plurality also wins, so corroboration only
  matches W7-2 (W8-3 backward-compat — not a strict separation).
* **Cross-role-corroborated gold.** The gold service is mentioned
  by strictly more distinct producer roles than any decoy
  service. Without this, the corroboration policy picks the
  decoy plurality and ties FIFO at 0.000 (W8-4 falsifier).
* **Buffered mode.** ``CrossRoleCorroborationAdmissionPolicy(fixed_dominant_tag=...)``
  constructed via ``from_candidate_stream``. The streaming variant
  is unstable under arrival permutation in the same sense as
  W7-1-aux; do not cite it as the load-bearing variant.

Permitted: "On the Phase-55 default config (K_auditor=4,
n_eval=10, decoy-plurality + cross-role-corroborated-gold property),
the *buffered* ``CrossRoleCorroborationAdmissionPolicy`` achieves
``accuracy_full = 1.000`` against ``capsule_fifo`` 0.000 AND
``capsule_cohort_buffered`` (W7-2) 0.000 — a +1.000 strict
separation from both baselines (W8-1)."

Forbidden: "Cross-role corroboration solves multi-agent context."
Forbidden: "Capsule corroboration always beats W7-2."
Forbidden: "Capsule structure beats FIFO." (Without the bench
conditions named — same as the W7-2 rule.)

### Labelling the streaming corroboration variant as the load-bearing policy

> *"Cross-role corroboration is the SDK v3.9 win."*

Forbidden without specifying *buffered* mode. The streaming
variant (``CrossRoleCorroborationAdmissionPolicy()`` with
``fixed_dominant_tag=None``) is arrival-order-sensitive in the
same way W7-1-aux describes for streaming cohort coherence. The
buffered variant (constructed via ``from_candidate_stream``) is
the load-bearing one and the W8-1 anchor. The honest phrasing:
"The *buffered* ``CrossRoleCorroborationAdmissionPolicy`` is the
SDK v3.9 win."

### Labelling the SDK v3.9 result "we solved multi-agent context"

> *"SDK v3.9 solves multi-agent context."*

Forbidden. SDK v3.9 ships the strongest cross-regime conditional
structural-win the programme has ever produced (Phase 55 strict
gain + Phase 54 backward-compat + Phase 53 no-regression, stable
across 5/5 bank seeds, named falsifier regime correctly ties FIFO).
This clears the **strong success bar** of
``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` § 1.1 — a real
advance, not a null milestone. But "solved" remains forbidden:

* The W8-1 win is *conditional* on the named bench property
  (decoy-plurality + cross-role-corroborated gold). The W8-4
  falsifier regime is the named conditional limit.
* Real production multi-agent teams have additional axes
  (heterogeneous producers, time-varying budgets, multi-round
  handoffs, conflicting goals, multi-service gold answers) that
  Phase 55 does not test. W8-C1 / W8-C2 / W8-C3 are the
  conjectural extensions; none are yet shipped.
* Three named regimes is a stronger cross-regime result than
  two, but it is not "all regimes." Real-LLM under multi-service
  decoy chatter (W8-C2) is the natural next falsifier.

Permitted phrasings:

* "SDK v3.9 clears the strong success bar of
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` § 1.1 on
  three named regimes (no-regression on R-53, backward-compat on
  R-54, strict win on R-55) with cross-bank stability and a named
  falsifier."
* "On the Phase-55 default config (named bench property),
  buffered cross-role corroboration achieves +1.000 accuracy_full
  vs both substrate FIFO and SDK v3.8 W7-2 buffered cohort, the
  first strict separation between cross-role corroboration and
  single-tag plurality cohort in the programme."
* "Three regimes anchored, the W8-1 conditional win is sharp
  and falsifiable. We have not solved multi-agent context;
  we have made the strongest cross-regime conditional advance
  to date."

### Conflating Phase-54 and Phase-55 results

> *"SDK v3.9 reverses the SDK v3.8 result."*

Forbidden. The two milestones measure *different bench properties*
and both are true:

* **Phase-54** (SDK v3.8): deterministic candidate stream with
  cross-role service-tag coherence; **gold raw plurality**
  property; surplus on every scenario; buffered cohort coherence
  beats FIFO by +1.000 on accuracy_full.
* **Phase-55** (SDK v3.9): deterministic candidate stream with
  cross-role corroboration; **decoy raw plurality** + **gold
  cross-role corroboration** property (strict separation from
  Phase 54); surplus on every scenario; buffered cross-role
  corroboration beats both FIFO AND W7-2 by +1.000 on
  accuracy_full.

W7-2 on Phase 54 is preserved exactly. Corroboration on Phase 54
matches W7-2 (W8-3 backward-compat). The honest combined reading
is the W7-2 / W8-1 strict-generalisation hierarchy: **W7-2 wins
on gold-plurality benches; W8-1 wins on the strict superset
that includes decoy-plurality with cross-role-corroborated gold;
W8-3 backward-compat preserves W7-2's wins under W8.**

### Conflating Phase-53 and Phase-55 results

> *"Phase 55 makes Phase 53 obsolete."*

Forbidden. Phase 53 is the **real-LLM low-surplus** anchor for
W7-1 (FIFO unbeatability under low surplus). Phase 55 is the
**deterministic budget-pressured + decoy-plurality + gold-
corroborated** anchor for W8-1. They measure orthogonal axes:

* Phase 53 tests *real-LLM extractor variability* with a small
  candidate stream (low surplus → no admission policy can win).
* Phase 55 tests *cross-role admission decision quality* with a
  designed candidate stream where admission *can* win.

Both are true; both are conditional. The W8-1 win on Phase 55
**does not contradict** the W7-1 result on Phase 53 — they
operate in disjoint regimes (high surplus vs low surplus).
