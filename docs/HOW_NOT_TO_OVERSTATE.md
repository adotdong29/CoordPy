# How not to overstate this

> Canonical do-not-overstate rules for the Context Zero / Wevra
> programme. Every milestone note, paper draft, README claim, or
> README-of-README must satisfy these rules. Last touched: SDK
> v3.19, 2026-04-28.

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

### "W15 shapes transformer attention" or "the salience pack proves attention manipulation"

> *"W15 shapes transformer attention weights so the auditor's
> decoder pays attention to the right evidence."*

Forbidden. The W15 layer uses an *honest proxy* attention metric:
the ``position_of_first_causal_claim`` in the salience-ordered
pack. We do NOT manipulate transformer attention weights; we DO
reorder the bundle so the highest-salience evidence appears at
rank 0, which benefits any downstream LLM consumer via
prompt-position attention (a well-known property of transformer
attention under typical positional encoding regimes). The proxy
metric is auditable; the attention claim is not.

Permitted phrasings: *"W15 places the round-2 specific-tier claim
at rank 0 of the kept bundle in 8/8 R-62-tightbudget cells (the
proxy attention signal)"*, *"the W15 salience pack benefits any
downstream LLM consumer via prompt-position attention shaping under
typical transformer positional encoding regimes"*, *"the
``position_of_first_causal_claim`` metric is the load-bearing
auditable W15 signal"*. Forbidden: *"W15 manipulates attention
weights"*, *"W15 proves attention shaping"*, *"the salience pack is
an attention-mechanism intervention"*. The honest reading is: W15
is a context-packing intervention, not an attention intervention;
the prompt-position attention benefit follows from the packing,
not from any model-internal change.

### "W15 solves multi-agent context" or "the salience pack is universal"

> *"The W15 attention-aware packer solved multi-agent context."*

Forbidden. The honest reading on R-62-tightbudget at n=8 × 5 seeds
is:

* the W15 method achieves ``accuracy_full = 1.000`` on every seed;
* ``capsule_layered_fifo_packed`` (the load-bearing FIFO baseline)
  ties FIFO at ``accuracy_full = 0.000``;
* ``attention_minus_fifo_packed = +1.000`` strict separation
  on every seed.

This is a strong result, but it is **not** "multi-agent context
solved." Permitted phrasings: *"clears bar 12 of the
SDK-v3.16-anchored success criterion"*, *"first decoder-side
context-packing strict gain in the programme"*, *"closes the
W15-Λ-budget gap on R-62-tightbudget under the named bench
property"*. Forbidden: *"solves multi-agent context"*, *"the W15
packer is universal"*, *"W15-1 holds for any decoder budget"*. The
W15-1 win is conditional on (a) the bench property holding, (b)
``T_decoder`` below the union token sum, AND (c) round-2 carrying
a specific-tier disambiguator with no ``service=`` token; if any
of the three is removed, W15-Λ-budget or W15-Λ-degenerate fires
and the result collapses. The W15 layer is *one of seven*
structural axes the programme has identified; "multi-agent context
solved" requires resolving every named limit theorem on every axis,
which the programme has not done.

### "W18 broke the symmetric-corroboration wall" or "we solved ambiguity resolution"

> *"W18 broke the symmetric-corroboration wall."*

Forbidden as a *general* claim. The W18-1 win is *strongly
conditional* on the R-65-COMPAT bench property: round-2
specific-tier disambiguator carries a relational-compound mention
of *every* gold service AND *no* decoy service. Permitted
phrasings: *"clears bar 15 of the SDK-v3.19-anchored success
criterion"*, *"first capsule-native multi-agent-coordination
method that crosses the symmetric-corroboration wall on a regime
where the wall actually applies"*, *"closes the W18-Λ-sym
extension of W17-Λ-symmetric on R-65-COMPAT under the named
bench property"*. Forbidden: *"W18 broke the symmetric wall"*
(unqualified), *"we solved ambiguity resolution"*, *"W18-1 holds
for any decoder bundle"*, *"the relational scorer is universal"*.

The W18-1 win is conditional on:
* (a) symmetric-corroboration round-1 (R-64-SYM bench shape),
* (b) round-2 disambiguator carrying a closed-vocabulary
  relational-compound mention of every gold service AND no decoy
  service (R-65-COMPAT specifically — not R-65-NO-COMPAT,
  -CONFOUND, or -DECEIVE),
* (c) the relational-mention convention being inside the
  closed-vocabulary closure the W18 exact-match scorer reads.

If any condition fails, W18 ties FIFO or fails by construction:
* **W18-Λ-no-compat** (no signal): abstain → tie FIFO at 0.000.
* **W18-Λ-confound** (symmetric signal): abstain → tie FIFO at 0.000.
* **W18-Λ-deceive** (adversarial signal): trust evidence → fail at 0.000.
* **W18-Λ-real** (free-form natural-language mentions outside the
  closure): the closed-form scorer misses by construction.

The W18 method is the *fifteenth* of fifteen named structural
axes the programme has identified; "ambiguity resolution solved"
requires resolving every named limit theorem on every axis,
which the programme has *not* done. The W18-Λ-deceive falsifier
in particular names a structural limit no closed-form scorer
that *trusts* its evidence can escape (the named research move
beyond it is W18-C-OUTSIDE — an outside-information axis to
detect deceptive round-2 mentions, conjectural).

### "the relational scorer reads transformer attention" or "W18 is a learned model"

> *"The W18 method uses a small learned compatibility model that
> reads transformer attention to break the symmetric ambiguity."*

Forbidden. The W18 method is a *deterministic, training-free,
closed-form* bundle-relational scorer:
* It tokenises the round-2 disambiguator's payload via a
  closed-form regex-style splitter
  (:func:`_disambiguator_payload_tokens`).
* It scores each admitted service tag via an O(|union| ·
  |tokens|) match loop with contiguous-subsequence semantics for
  compound targets
  (:func:`_relational_compatibility_score`).
* The strict-asymmetric branch fires *iff* at least one but not
  all admitted tags have positive score; otherwise the W18 method
  abstains.

There is **no learned model**, no transformer attention reading,
no embedding lookup. A learned variant is the named
**W18-C-LEARNED** conjecture, conjectural and out of scope for
SDK v3.19. Permitted phrasings: *"closed-form bundle-relational
scorer"*, *"deterministic training-free disambiguator"*, *"the
W18 scorer reads payload bytes via a closed-form tokeniser +
contiguous-subsequence scorer"*. Forbidden: *"W18 reads attention
weights"*, *"the W18 model"*, *"the W18 embedding"*.

### "W16 solves multi-agent context end-to-end" or "the composition is universal"

> *"The W14+W15 composition solved multi-agent context end-to-end."*

Forbidden. The honest reading on R-63-COMPOSED-TIGHT at n=8 × 5
seeds (synthetic) is:

* the composed method (W14 structured prompt + W15 attention-aware
  packer over a magnitude-filter-simulated producer) achieves
  ``accuracy_full = 1.000``;
* every non-composed baseline (W14-only-budgeted, W15-only-without-
  W14, substrate FIFO) ties at ``accuracy_full = 0.000``;
* ``composed - fifo_packed_layered = +1.000`` strict separation on
  every seed.

This is a strong synthetic result, but it is **not** "multi-agent
context solved end-to-end." Permitted phrasings: *"clears bar 13
of the SDK-v3.17-anchored success criterion"*, *"first end-to-end
W14+W15 composition strict gain in the programme"*, *"closes the
W16-Λ-compose gap on R-63-COMPOSED-TIGHT under the named bench
property"*. Forbidden: *"solves multi-agent context end-to-end"*,
*"the W16 composition is universal"*, *"W16-1 holds for any
producer + any decoder budget"*. The W16-1 win is conditional on
(a) the comparable-magnitude multi-hypothesis events, (b) the
structured producer protocol, (c) ``T_decoder`` strictly between
the round-2 disambiguator's token cost and the admitted union's
token sum, AND (d) the asymmetric corroboration shape (decoys ≥ 2
distinct roles, golds = 1 distinct role); if any condition is
removed, W16-Λ-compose / W16-Λ-degenerate / W15-C-SYMMETRIC fires
and the result collapses.

The W14+W15 composition is *one of eight* structural axes the
programme has identified; "multi-agent context solved end-to-end"
requires resolving every named limit theorem on every axis, which
the programme has not done.

### "W16 demonstrates real-LLM transfer is solved" or "the replay path is a real probe"

> *"The W16-Λ-real-replay anchor proves the composition transfers
> to real LLMs."*

Forbidden. The W16-Λ-real-replay anchor is a *measurement* over
**recorded** Phase-61 ``qwen2.5:14b-32k`` bytes — not a fresh live
LLM probe. The Mac-1 endpoint at 192.168.12.191:11434 was offline
(``HTTP=000`` connection refused) at SDK v3.17 milestone capture
time; the Phase-61 capture from SDK v3.15 is the source of truth.

Permitted phrasings: *"the recorded qwen2.5:14b-32k bytes show the
composed method delivers a strict +0.500 gain over FIFO-packed-W14
under T_decoder=14"*, *"the W16-Λ-real-replay anchor confirms the
composition recovers the W14-only loose-budget accuracy under
tight budget pressure on recorded real-LLM bytes"*, *"the budget
band where the gain holds on the recorded stream is T_decoder ∈
[13, 16]"*. Forbidden: *"the composition transfers to a fresh
live LLM"* (untested; W16-C-LIVE-OLLAMA conjectural), *"the replay
result is a real-time win"* (replay is offline replay over
byte-stable JSON), *"W16 closes the model-side judgment gap"* (the
1/8 model-side failure persists on the recorded bytes).

The replay path's contribution is *bounding* the W16 result by the
empirical envelope of the prior milestone's real-LLM probe — it is
honest measurement, not an extrapolation. Treat it the way you
treat the W4-1 lifecycle audit: a tool that surfaces what is true
about the recorded run, not a substitute for a fresh probe.

### "W17 solves multi-agent context" or "the magnitude-hint is universal"

> *"The W17 magnitude-hinted prompt solves the producer-side
> ambiguity-erasure problem."*

Forbidden as stated. The W17 magnitude-hinted prompt closes the
*relative-magnitude* failure mode that the W14 structured prompt
left open on R-61-OLLAMA-A's ``slow_query_archival`` scenario,
producing an 8/8 bench-property hold-rate AND
``capsule_attention_aware = 1.000`` on a fresh live qwen2.5:14b-32k
probe at ``T_decoder = 14``. **But the win is conditional on three
things**: (a) the asymmetric-corroboration bench property
(W17-Λ-symmetric is the named wall when this is absent),
(b) the magnitude-hint table being calibrated to the synthetic
extractor's threshold values (operational definitions, not answer
hints — both gold AND decoy magnitudes are above every threshold),
AND (c) the live endpoint reachable.

Permitted phrasings: *"the W17 magnitude-hinted prompt closes the
1/8 R-61-OLLAMA-A model-side judgment miss"*, *"the magnitude-hint
extension is the load-bearing improvement on the fresh live
probe"*, *"the W17-1 anchor produces +1.000 strict gain over both
substrate FIFO and the FIFO-packed-W14H-only baseline on a fresh
live qwen2.5:14b-32k Mac-1 probe"*. Forbidden: *"W17 solves
multi-agent context"*, *"the magnitude-hint extension makes the
W14 protocol universal"*, *"W17 transfers to every benchmark
family"* (W17-C-CROSS-BENCH is conjectural).

### "W17-Λ-symmetric is just another conjecture"

Forbidden. **W17-Λ-symmetric is a *negative theorem*, not a
conjecture.** It is constructed: every capsule strategy in the
SDK ties FIFO at 0.000 on R-64-SYM under both
``T_decoder ∈ {None, 24}`` by direct measurement on n=8
saturated. The structural argument is the asymmetric-oracle
property of ``services_correct`` set-equality: when the bipartite
``(role × tag, kind, magnitude)`` multiset is symmetric for gold
and decoy, no service-blind admission AND no closed-form salience
packer can prefer one over the other.

Permitted phrasings: *"W17-Λ-symmetric is the first explicit
symmetric-corroboration limit theorem in the programme"*,
*"W17-Λ-symmetric discharges the prior W15-C-SYMMETRIC and
W16-C-SYMMETRIC conjectures as a negative theorem"*,
*"W17-Λ-symmetric names the next research frontier
(W17-C-DISAMBIGUATOR, conjectural)"*. Forbidden: *"the
symmetric-corroboration wall is still open"* (it is closed as a
negative theorem on the closed-form capsule surface; only the
disambiguator escape route is open), *"W17-Λ-symmetric is a
conditional conjecture"* (it is proved-empirical + structural
sketch).

### "The W17-C-XMODEL probe is saturated to 1.000 on 35B"

Forbidden. The R-64-LIVE-XMODEL fresh probe achieves
``capsule_attention_aware = 0.750`` on qwen3.5:35b — bench
property holds in 8/8 (transfer of the magnitude-hint extension
across the 14B → 36B-MoE jump is byte-for-byte on the
bench-property axis), but ``accuracy_root_cause = 0.750`` (the
35B emits a different specific-tier kind on one scenario than
the 14B does). The strict-gain claim against FIFO-pack and
substrate FIFO holds at +0.750 (well above the 0.50 strong-bar
threshold), but the saturated full-correctness clause (1.000 on
35B) is **not** demonstrated.

Permitted phrasings: *"the W17 magnitude-hint extension transfers
to qwen3.5:35b on the bench-property axis"*, *"the strict-gain
claim against substrate FIFO and FIFO-packed-W14H-only holds on
the 35B at +0.750"*, *"the 0.250 gap to 1.000 on 35B is a
model-class-specific specific-tier judgment artifact"*.
Forbidden: *"W17 transfers fully to 35B"*, *"the cross-model
result saturates"*, *"35B closes the gap"* (one specific-tier
judgment artifact remains; W17-C-XMODEL is proved-conditional +
empirical-research, not a saturated full-correctness claim).

### "The W17 magnitude-hint table is answer-leaking"

Forbidden. The :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS`
table values (``LATENCY_SPIKE p95_ms ≥ 1000``,
``ERROR_RATE_SPIKE error_rate ≥ 0.10``,
``FW_BLOCK_SURGE count ≥ 5``) are *operational definitions* —
the same numeric values the synthetic
``MagnitudeFilteringExtractor`` uses (Phase-61 calibration
anchors). They are below *every* gold AND decoy magnitude in
the R-61 / R-64 banks. The magnitude-hint extension does NOT
tell the LLM which service is gold; it removes the LLM's
*relative* magnitude judgment loophole (the failure mode that
produced the slow_query_archival miss).

Permitted phrasings: *"the magnitude-hint table is calibrated to
operational definitions, not to answer hints"*, *"the threshold
values are below every gold and decoy magnitude in the R-61 / R-64
banks"*, *"the W17 extension removes the LLM's relative-magnitude
judgment loophole, not its discrimination ability"*. Forbidden:
*"the magnitude-hint tells the LLM which service is gold"* (it
does not — gold and decoy both exceed every threshold), *"W17 is
prompt-engineering the answer"* (it is enforcing operational
discipline; the synthetic side confirms the same downstream
answer with and without the magnitude-hint, W17-3).

### "W16 makes the W14 or W15 layers obsolete"

Forbidden. W16 is *not* a new mechanism — it is the demonstration
that the existing W14 and W15 layers compose on a single regime
where both are individually load-bearing. On R-63-naive-tight the
composed pipeline ties FIFO at 0.000 (W16-Λ-compose), so neither
layer alone produces the win; on R-63-COMPOSED-TIGHT the
composition ties identical to W14 + W15 stacked. Removing the W14
layer (replacing structured prompt with naive prompt) collapses
the result to 0.000; removing the W15 layer (replacing salience
pack with FIFO pack at the same T_decoder) collapses it to 0.000.

Permitted phrasings: *"W14 + W15 are jointly necessary on R-63"*,
*"the composition recovers correctness when both upstream emission
and downstream retention bottlenecks are present"*, *"W16 is the
coupling statement; it does not subsume the prior layers"*.
Forbidden: *"W16 replaces W14"*, *"W16 makes W15 dispensable"*,
*"the composition is a new mechanism"*. The runtime contract is
unchanged; the SDK ships no new W16 class — the composition is
demonstrated by composing existing ``StructuredProducerProtocol``
+ ``AttentionAwareBundleDecoder`` calls in
``vision_mvp.experiments.phase63_composed_real_llm``.

### "W15 makes the W14 producer protocol obsolete"

Forbidden. W15 is a *decoder-side* intervention; W14 is a
*producer-side* intervention. They compose additively:
W15-C-COMPOSE-W14 conjectures that running W15 over a W14-emitted
stream on R-61-ollama-structured may close the 1/8 model-error
failure that W14 alone leaves, but this is a conjecture not yet
empirically verified. The W15 layer does not refute W14; it adds
an orthogonal axis. On any regime where the producer's emission
stream is the bottleneck (R-61-ollama-naive, R-13-Λ-real), W14 is
load-bearing and W15 has no influence on the producer side.

### "Solved real-LLM transfer" or "the structured prompt closes the W13-Λ-real gap"

> *"The W14 prompt protocol solved real-LLM transfer."*

Forbidden. The honest reading on R-61-ollama-structured at n=8 is:

* the bench property holds in 7/8 scenarios (one model-side
  judgment failure);
* the cross-round capsule pipeline achieves
  ``accuracy_full = 0.500``;
* ``layered − fifo = +0.500`` at *exactly* the R-61-OLLAMA-A
  threshold;
* W13 ties W12 ties W11 because the real LLM emits canonical kinds
  (no drift to widen).

This is a strong real-transfer result, but it is **not** "real-LLM
transfer solved." Permitted phrasings: *"clears the R-61-OLLAMA-A
tier"*, *"first real-LLM strict gain ≥ 0.50 over substrate FIFO in
the programme"*, *"W14 closes the W13-Λ-real producer-side gap on
the redesigned comparable-magnitude events"*. Forbidden: *"solves
real-LLM transfer"*, *"the structured prompt is universal"*,
*"W14-1 holds for any LLM"*. The W14-1 win is conditional on (a)
comparable-magnitude events, (b) structured prompt, (c) the cross-
round capsule pipeline; if any of the three is removed, W14-Λ-prompt
fires and the result collapses to 0.000.

### "W14 makes the W13 normaliser obsolete"

Forbidden. W13's contribution is *structurally invisible* on
R-61-ollama because the real LLM emits canonical kinds (zero drift).
On a *different* model class (e.g. qwen3.5:35b under W14-C4) or
under a *different* prompt, the drift channel may reopen and W13's
closure-widening will be load-bearing again. The W13 layer is
dormant on this regime, not refuted.

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

### Labelling the W12-1 win as unconditional

> *"Robust multi-round bundle decoding solves real-LLM multi-agent
> context."*

Forbidden without the conditions. The defensible W12-1 reading
names the bench properties **and** the closure contract:

* **R-58 delayed-causal-evidence shape.** R-59 inherits the four
  R-58 properties; without them the W11 contradiction-aware drop
  cannot fire even after normalisation.
* **Bounded producer-noise channel.** ``synonym_prob`` and
  ``svc_token_alt_prob`` must be set such that *every* drifted
  ``claim_kind`` is in :data:`CLAIM_KIND_SYNONYMS` and *every*
  drifted ``service=`` token matches a pattern in
  :data:`_SERVICE_TAG_REWRITES`. The closure property is mechanically
  verified by ``NoisyExtractorTests::test_noisy_variants_all_in_synonym_table``.
* **Closed-vocabulary normalisation table fits the benchmark
  family.** The default ``CLAIM_KIND_SYNONYMS`` is fitted to the
  closed-vocabulary incident-triage claim grammar; other benchmark
  families need their own tables.
* **Round-N admission not budget-starved** (inherits W11-4).

Permitted: "On the Phase-59 default config (K_auditor=8,
n_eval=12, ``synonym_prob=0.50, svc_token_alt_prob=0.30``,
synthetic-noisy-LLM extractor; bench property holds 12/12),
``RobustMultiRoundBundleDecoder`` achieves ``accuracy_full = 1.000``
against every un-normalised method including W11
``MultiRoundBundleDecoder`` at 0.000 — a +1.000 strict separation
under the named bounded-noise channel (W12-1)."

Forbidden: "W12 solves real-LLM multi-agent context."
Forbidden: "Robust multi-round bundle decoder beats W11."
(without the bench-shape conditions named).
Forbidden: "Real LLMs satisfy the R-59 bench property out of the
box." (the synthetic noisy extractor is a *calibrated approximation*,
not an empirical real-LLM measurement; the ``ollama`` opt-in mode
is the W12-C2 next data point.)

### Labelling the synthetic-noisy-LLM extractor a "real LLM"

> *"Phase-59 evaluates Wevra on a real LLM."*

Forbidden without the mode disclosure. The Phase-59 default mode
``synthetic_noisy_llm`` is a *deterministic in-process synthetic
extractor* whose noise channel is calibrated against Phase-53 14B /
35B empirical kind-drift histograms but is *not itself a real LLM
measurement*. The ``ollama`` opt-in mode is the real-LLM path; when
used, the report's ``extractor_stats`` block records
``llm_mode='ollama'``, ``n_real_calls``, ``n_failed_calls``, and
``n_synthetic_fallbacks``. Honest phrasings:

* "Phase-59 default uses a *calibrated synthetic-noisy-LLM
  extractor* whose drift channel mimics Phase-53 14B/35B
  parser_role_response distributions; this is the W12-1 anchor."
* "Phase-59 ``--llm-mode ollama`` is the opt-in real-LLM extension
  path; the W12-C2 conjecture targets that mode."

Forbidden: "Phase-59 measures real-LLM behaviour." (without naming
the LLM mode and the calibration provenance).

### Labelling SDK v3.13 the "synthetic→real-LLM transfer is closed"

> *"SDK v3.13 closes the synthetic→real-LLM transfer gap."*

Forbidden. The honest reading is two-layered:

* SDK v3.13 closes the synthetic→real-LLM transfer gap *under the
  bounded-producer-noise channel* the synthetic noisy extractor
  models. The closure property (every variant the extractor can
  emit is in the normalisation table) is mechanically verified.
* SDK v3.13 does **not** measure transfer to a real Ollama-served
  LLM. The W12-C2 conjecture targets that next move; until it is
  measured (with the ``ollama`` opt-in mode, on Mac 1 or Mac 2),
  the *real* real-LLM transfer reading is open. The synthetic side
  of the bound is the *honest cap* on the SDK v3.13 advance; over-
  claiming is the failure mode this section guards against.

Permitted: "SDK v3.13 closes the synthetic→synthetic-noisy-LLM
transfer gap by adding a closed-vocabulary normalisation layer
ahead of the W11 multi-round bundle decoder; the closure property
on the noise channel is the load-bearing premise; the real-Ollama
transfer (W12-C2) is the next data point."

### Labelling the W13-1 win as unconditional

> *"Layered open-world normalisation solves real-LLM multi-agent
> context."*

Forbidden without the conditions. The defensible W13-1 reading
names the bench properties **and** the closure contract **and** the
honest real-LLM caveat:

* **R-58 delayed-causal-evidence shape.** R-60-wide inherits R-58's
  four properties; without them the W11 contradiction-aware drop
  cannot fire even after layered normalisation.
* **Bounded producer-noise channel inside the heuristic closure.**
  Every variant the wide-OOV extractor emits must be in
  :data:`HEURISTIC_RESCUABLE_OOV_KINDS` AND must match at least one
  pattern in :data:`_HEURISTIC_KIND_RULES`. The closure-membership
  property is mechanically verified by
  ``W13ClosureTests::test_every_wide_oov_variant_outside_w12_inside_w13``.
* **Heuristic abstraction rules fit the benchmark family.** The
  default :data:`_HEURISTIC_KIND_RULES` is fitted to the closed-
  vocabulary incident-triage claim grammar; other benchmark
  families need their own rule sets (W13-C1).
* **Round-N admission not budget-starved** (inherits W11-4).

Permitted: "On the Phase-60 default config (K_auditor=8, n_eval=12,
``synthetic_wide_oov_llm`` extractor at ``wide_oov_prob=0.50``;
bench property holds 12/12 after layered normalisation),
``LayeredRobustMultiRoundBundleDecoder`` achieves
``accuracy_full = 1.000`` against every fixed-vocabulary method
including W12 ``RobustMultiRoundBundleDecoder`` at 0.000 — a
+1.000 strict separation under the named bounded-OOV-in-heuristic-
closure channel (W13-1). On real Ollama 14B (R-60-ollama), the
bench property does not hold and the W13 advance is structurally
invisible — see § 1.4 of the success criterion (R-60-OLLAMA-C
honest negative)."

Forbidden: "W13 solves real-LLM multi-agent context."
Forbidden: "Layered normalisation always beats W12."
(Without the bench-shape + closure-membership conditions named.)
Forbidden: "Real Ollama 14B drifts kinds and W13 rescues the run."
(The empirical observation is the *opposite*: real Ollama 14B
emits canonical kinds at temperature 0; the W13 advance is on
synthetic wide-OOV, not on real-LLM drift.)
Forbidden: "The synthetic→real-LLM transfer is closed."
(R-60-ollama is R-60-OLLAMA-C honest-null; the transfer story has
five layers and the real-LLM gate is event-shape / prompt-side
discipline, not normalisation.)

### Labelling the heuristic abstraction layer "open-world generalisation"

> *"Wevra now generalises to open-world LLM drift."*

Forbidden without the closure boundary disclosure. The W13-1 result
is *closure widening*, not *closure elimination*:

* The heuristic rule set has a *finite predicate union*. Inputs
  whose surface form witnesses none of the patterns escape the
  closure.
* The W13-4 falsifier (R-60-cosmic) is the named structural limit:
  truly arbitrary OOV (XYZZY_QQQQ, COSMIC_RAY_FLIP, …) ties
  ``LayeredRobustMultiRoundBundleDecoder`` at FIFO 0.000.
* The W13 method *widens* the W12 closure on the kinds the
  benchmark family's heuristic rules cover; it does not generalise
  to arbitrary LLM drift.

Permitted: "W13 widens the W12 fixed-vocabulary closure on the
incident-triage benchmark family by adding a small set of regex-
predicate abstraction rules; the new closure strictly contains the
W12 closure on R-60-wide (proved-empirical) and is bounded above
by the predicate union (W13-4)."

Forbidden: "W13 normalises arbitrary LLM drift."
Forbidden: "W13 generalises to any benchmark family."
(W13-C1 is conjectural for non-incident-triage families.)

### Labelling the R-60-ollama probe a "real-LLM win"

> *"SDK v3.14 wins on real Ollama 14B."*

Forbidden. The honest reading is the four-tier R-60-ollama grading
(§ 1.4 of the success criterion):

* SDK v3.14's R-60-ollama observation lands at **R-60-OLLAMA-C
  (null real transfer; honest negative)**: real Ollama 14B at
  temperature 0 on the calibrated incident-triage prompt does not
  drift kinds AND filters low-magnitude decoy events. The bench
  property holds in 0/4 scenarios; W12 / W13 normalisation has
  nothing to rescue.
* The W13 advance is on R-60-wide synthetic, NOT on R-60-ollama.
* The R-60-ollama probe is a *measurement* anchor; it falsified the
  conjecture (W12-C2) that real Ollama would emit non-trivial
  drift on this prompt. It does NOT falsify the W13 method itself
  on R-60-wide.

Permitted phrasings:

* "SDK v3.14 clears the strong success bar of
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` § 1.1 (R-60
  anchor) on the Phase-60 wide-OOV synthetic regime; the R-60-
  ollama probe lands at the R-60-OLLAMA-C tier (honest null real
  transfer) — the milestone is therefore strong-success on R-60-
  wide AND a partial-success / honest-null on R-60-ollama."
* "Real Ollama 14B at default settings emits canonical kinds and
  filters low-magnitude decoy events; the synthetic→real-LLM
  transfer story is gated by event-shape design + prompt-side
  discipline (W13-Λ-real), not by normalisation. SDK v3.14 measures
  this honestly without overclaiming."

Forbidden: "SDK v3.14 closes synthetic→real-LLM transfer."
Forbidden: "Real Ollama 14B emits drift; W13 rescues it."

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
- **2026-04-26 (SDK v3.14).** Adds W13 rules: forbidden phrases
  "layered open-world normalisation solves real-LLM multi-agent
  context" without the closure-membership conditions; "Wevra now
  generalises to open-world LLM drift" without the W13-4 closure
  boundary disclosure; "SDK v3.14 wins on real Ollama 14B" without
  the R-60-OLLAMA-C honest-null tier disclosure; "we solved
  multi-agent context" still forbidden after R-60-wide — the
  strongest cross-regime win now spans **seven** named regimes
  (R-54..R-58 + R-59 noisy + R-60-wide) AND a real-LLM
  *measurement* (R-60-ollama, honest-null), but is conditional on
  bounded drift inside a finite predicate-set closure; "W12 was
  wrong" — W12 is *unchanged* and still wins on R-58 / R-59 /
  R-60-clean / R-60-synonym, W13 is a strict additive widening
  layer that adds R-60-wide, not a refutation.

- **2026-04-26 (SDK v3.13).** Adds W12 rules: forbidden phrases
  "robust multi-round bundle decoding solves real-LLM multi-agent
  context" without the bench-shape + closure-property conditions;
  "Phase-59 evaluates Wevra on a real LLM" without naming the
  LLM mode (synthetic_noisy_llm vs ollama); "SDK v3.13 closes the
  synthetic→real-LLM transfer gap" without the bounded-producer-
  noise-channel disclosure; "we solved multi-agent context" still
  forbidden after the Phase-59 result — the strongest cross-regime
  win now spans **six** named regimes (R-54..R-58 + R-59 default)
  AND a real-LLM-shaped stream, but is conditional on the closure
  property and on a synthetic noise channel; "W11 was wrong" —
  W11 is *unchanged* and still wins on R-58 + R-59-clean, W12 is a
  strict additive layer that adds R-59-noisy, not a refutation.

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
