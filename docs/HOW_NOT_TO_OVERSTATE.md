# How not to overstate this

> Canonical do-not-overstate rules for the Context Zero / CoordPy
> programme. Every milestone note, paper draft, README claim, or
> README-of-README must satisfy these rules. Last touched: SDK v3.43 (W42 family — final release of the v3.4x line) 2026-05-03. Earlier: SDK v3.42 (W41 family) 2026-05-03. Earlier: SDK v3.38 (W37 family) 2026-05-02. Earlier: SDK v3.37 (W36 family) 2026-05-02. Earlier: SDK v3.36 (W35 family) 2026-05-02. Earlier: SDK v3.35 (W34 family) 2026-05-01. Earlier: SDK v3.34 (W33 family) 2026-05-01. Earlier: SDK
> v3.33 (W32 family) 2026-05-01.

The programme has a long history of moves where a candidate result
was written up too strongly and later had to be sharpened or
retracted. This document is the canonical rule-book that prevents
that; reviewers should reject any text that violates it.

## Status vocabulary (definitions)

These are the only labels permitted on theorem-style claims in
this repo. Unlabelled claims are forbidden.

- **proved** — Mathematical proof, or proof-by-inspection of code
  that a reviewer can read in under 30 lines. The proof is in
  `docs/CAPSULE_FORMALISM.md` or `docs/archive/pre-coordpy-theory/PROOFS.md` or in the
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

### "Public install is live everywhere" or "all CoordPy APIs are stable"

> *"Just `pip install coordpy`; everything in the repo is stable."*

Forbidden unless it is true at the time of writing.  Public-facing
release text must distinguish:

* the **stable released surface** (`vision_mvp.coordpy`, the public
  CLIs, and the named on-disk schema contracts),
* the **experimental included surface**
  (`vision_mvp.coordpy.__experimental__`), and
* the **out-of-scope next-programme work**
  (`W42-C-NATIVE-LATENT`, `W42-C-MULTI-HOST`).

Permitted phrasing: *"Install from a clone with `pip install -e .`
today; use `pip install coordpy` / `pipx install coordpy` once the
package is published.  Treat `vision_mvp.coordpy` as the stable SDK
surface and `vision_mvp.coordpy.__experimental__` as non-stable
research API."*

Forbidden phrasing: *"Everything in CoordPy is stable"*, *"`pip
install coordpy` is available now"* (unless it is), *"the W22..W42
ladder is part of the stable SDK contract"*.

### "Paradigm shift" without a stated reading

> *"This is a paradigm shift."*

If you write the phrase "paradigm shift" anywhere in this repo,
you must immediately specify the reading: under which
quantitative bar, on which bench, at which $n$. The phrase
"paradigm-shift candidate" is permitted only when followed by
"under W3-Cn" where W3-Cn is a named conjecture in
`THEOREM_REGISTRY.md`. The strict reading W3-C7 is **retracted**.

Last touched: SDK v3.43 (W42 family — final release of the v3.4x line) 2026-05-03.

### "Solves" without a defining gate

> *"CoordPy solves context."*

If you write "solves" or "closes" about any open problem, you
must name the *defining gate* and state which side of the gate
you are on. "Solves the bounded-context problem" is forbidden;
"closes the per-agent O(log N) bound on the routing-only setting
of Phases 1–10" is permitted because the bound is named.

### "W35 is native latent" or "W35 found the transformer trust subspace"

> *"W35 implements native latent trust transfer."*

Forbidden.  W35 implements an **audited capsule-layer trust-subspace
proxy**.  Its basis entries are derived from controller-visible
signals: W21 probe top_sets, W33 EWMA trust, W34 live-attestation /
response-feature state, top-set stability, and host health.  It does
not read hidden states, transplant KV cache, inspect attention weights,
or project embeddings.

Permitted phrasing: *"W35 is a trust-subspace dense-control proxy that
turns some W34 NO_CONSENSUS abstentions into verified reroutes when a
stable high-margin basis direction exists."*

Forbidden phrasing: *"W35 closes W33-C-NATIVE-LATENT"*, *"W35 proves
native latent transfer"*, *"W35 discovers a transformer-internal trust
subspace"*, *"W35 solves context for multi-agent teams."*

The honest reading is narrower: on R-82-TRUST-SUBSPACE-SHIFT, W35
improves correctness over W34 by +0.3125 while preserving trust
precision at 1.000, using one visible token/cell to carry roughly
13k bits of controller-verified structured state.  The hard falsifier
W35-L-ALL-BASIS-COMPROMISED remains: if every basis direction moves
together to the wrong answer, the capsule-layer proxy has no
independent signal and cannot recover.

### "W36 is native latent" or "W36 proves multi-host release readiness"

> *"W36 implements native latent host trust transfer."*

Forbidden.  W36 implements an **audited capsule-layer host-diverse
trust-subspace guard**.  It checks whether W35 dense-control support
is independently attested by distinct registered healthy hosts.  It
does not read hidden states, transplant KV cache, inspect attention
weights, project embeddings, or prove that a transformer-internal
trust subspace has been found.

Permitted phrasing: *"W36 hardens W35 by rejecting, rerouting, or
abstaining when dense-control support is not host-diverse and
verifiable."*

Forbidden phrasing: *"W36 closes W33-C-NATIVE-LATENT"*, *"W36 gives
true three-host evidence"*, *"W36 makes the repo release-ready by
itself"*, *"W36 solves context for multi-agent teams."*

The honest reading is narrower: on R-83-HOST-DIVERSE-RECOVER, W36
improves over W35 by +0.3125 correctness and restores trust precision
to 1.000.  On R-83-HOST-SPOOFED-CONSENSUS, W36 improves trust
precision by abstaining but does not recover correctness.  On
R-83-NO-LIVE-ATTESTATION, W36 is a correctness-destroying abstention
guard.  Mac 2 (`192.168.12.248`) still times out, so the live result
is two-reachable-host evidence only.

### "W37 is native latent" or "W37 closes the multi-host conjecture"

> *"W37 transplants per-host trust state across the transformer."*

Forbidden.  W37 maintains a **closed-form, zero-parameter,
per-(host, oracle, top_set) EWMA over anchored historical
observations**.  It does not access hidden states, KV cache, attention
weights, or embeddings.  The "trajectory" is a sealed tuple of bytes
sealed under manifest-v7; the EWMA update is the same first-order
``(1-alpha) * prev + alpha * obs`` recurrence used in W32 and W33.

Permitted phrasing: *"W37 hardens W36 by allowing a single-host
recovery cell to be safely rerouted iff its (host, oracle, top_set)
trajectory has been cross-host anchored above threshold across
historical cells with at least ``min_trajectory_anchored_hosts``
distinct anchor hosts; without anchored trajectory, W37 preserves W36
abstention."*

Forbidden phrasing: *"W37 closes W37-C-NATIVE-LATENT"*, *"W37 gives
true three-host evidence"*, *"W37 makes the repo release-ready by
itself"*, *"W37 solves context for multi-agent teams"*, *"W37 closes
W37-L-MULTI-HOST-COLLUSION-CAP"*.

The honest reading is narrower: on R-84-SINGLE-HOST-TRAJECTORY-RECOVER,
W37 improves over W36 by +0.500 correctness while preserving trust
precision at 1.000.  On the four named falsifiers (no history,
poisoned trajectory, disagreement, trivial), W37 preserves W36
behavior.  Mac 2 still times out (30th milestone), so the live
trajectory probe is two-reachable-host evidence only.  Two registered
hosts emitting a coordinated wrong top_set across enough cells can
defeat W37 at the capsule layer (W37-L-MULTI-HOST-COLLUSION-CAP).

### "W38 closes the multi-host blocker" or "W38 closes W37-L-MULTI-HOST-COLLUSION-CAP"

> *"W38 finally beats two-host collusion at the capsule layer."*

Forbidden.  W38 **bounds** (does not close)
``W37-L-MULTI-HOST-COLLUSION-CAP``.  The bound is conditional on the
existence of an uncompromised disjoint registered consensus reference.
W38 raises the capsule-layer adversary bar from "compromise 2 of N
trajectory hosts" to "compromise 2 of N trajectory hosts AND the
disjoint registered consensus reference".  When that reference is
itself compromised in lock-step, W38 cannot recover; this is the new
proved-conditional ``W38-L-CONSENSUS-COLLUSION-CAP`` limitation
theorem.

W38 also does not access hidden states, KV cache, attention weights,
or embeddings.  The ``ConsensusReferenceProbe`` is a controller-pre-
registered audited capsule-layer probe carrying a sealed tuple of
bytes; it is NOT a runtime ground-truth oracle and NOT a
transformer-internal projection.  The disjoint topology is enforced
*mechanically* in two places: the registry's ``__post_init__`` raises
``DisjointTopologyError`` when ``consensus_host_ids ∩
trajectory_host_ids != ∅``, and the verifier rejects envelopes
claiming an overlapping topology
(``w38_disjoint_topology_violation``).

Permitted phrasing: *"W38 wraps W37 with a disjoint cross-source
consensus-reference cross-check; when W37 reroutes on a colluded
trajectory and an uncompromised disjoint reference disagrees, W38
abstains via DIVERGENCE_ABSTAINED.  When the disjoint reference is
itself compromised, W38 cannot recover at the capsule layer."*

Forbidden phrasing: *"W38 closes W37-L-MULTI-HOST-COLLUSION-CAP"*,
*"W38 closes W38-L-CONSENSUS-COLLUSION-CAP"*, *"W38 closes
W38-C-NATIVE-LATENT"*, *"W38 gives true 3-host evidence"*, *"W38
makes the repo release-ready by itself"*, *"W38 solves context for
multi-agent teams"*.

The honest reading is narrower: on R-85-COLLUDED-CROSS-HOST-
TRAJECTORY, W38 improves over W37 by +0.500 trust precision while
preserving correctness.  On the four named falsifiers (trivial,
no-collusion, consensus-also-compromised, no-consensus-reference),
W38 preserves W37 behavior or honestly fails.  Mac 2 still times out
(31st milestone), so the live consensus probe sourced its disjoint
consensus host from a different model class on the same physical
host as one trajectory host -- a defensible weak proxy for capsule-
layer disjointness, NOT a true 3-physical-host disjoint topology.

### "W41 solves context for multi-agent teams" or "W41 closes W41-L-COMPOSITE-COLLUSION-CAP"

> *"W41 finally solves context for multi-agent teams."*

Forbidden.  W41 is an **integration** milestone, not a "solving"
milestone.  W41 jointly binds the W21..W40 trust-adjudication
chain and the W7..W11 cross-role / multi-round bundle decoder
family into a single auditable end-to-end path under a manifest-
v11 envelope, but does NOT close native-latent transfer, does
NOT close ``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP``, and does
NOT close its own ``W41-L-COMPOSITE-COLLUSION-CAP`` limitation
theorem.

W41 also does not access hidden states, KV cache, attention
weights, or embeddings.  The cross-axis classification is a
closed-form, zero-parameter, deterministic mechanical decision
over the W40 projection branch + the inner answer's services
field; the eight named integrated branches are mechanically
classifiable from per-axis branches alone.

Permitted phrasing: *"W41 is the first capsule-native end-to-end
integrated synthesis of the W21..W40 trust-adjudication chain
and the W7..W11 cross-role / multi-round bundle decoder family,
with one manifest-v11 envelope binding both axes plus a cross-
axis witness, and a measured cross-axis branch distribution that
lets researchers distinguish which axis is load-bearing on each
cell."*

Forbidden phrasing: *"W41 solves context for multi-agent teams"*,
*"W41 closes W41-L-COMPOSITE-COLLUSION-CAP"*, *"W41 closes
W41-C-NATIVE-LATENT"*, *"W41 gives true 3-host evidence"*, *"W41
makes the repo release-ready by itself"*.

The honest reading is narrower: on R-88-BOTH-AXES, R-88-TRUST-
ONLY-SAFETY, and R-88-INSUFFICIENT-RESPONSE-SIGNATURE, W41
preserves W40 byte-for-byte on the answer field while adding 1
visible token/cell of cross-axis classification overhead and
~15.5k structured bits/cell of cross-axis state under manifest-
v11.  On R-88-COMPOSITE-COLLUSION (the new W41-L-COMPOSITE-
COLLUSION-CAP regime), W41 cannot recover when the adversary
coordinates BOTH the producer-side admission AND the trust-side
W40 ratification on the wrong set; this is the new proved-
conditional limitation theorem.

### "192.168.12.101 is a Mac with a hung Ollama listener" (RETRACTED at W41)

> *"`.101` is a Mac running Ollama with a hung HTTP listener (TCP-up + HTTP-broken)."*

**Retracted at the W41 milestone (2026-05-03).** The W37 / W38 /
W39 / W40 milestones described ``192.168.12.101`` as a third Mac
with a hung Ollama HTTP listener.  Re-probing at the W41
milestone shows that ``.101`` is an **Apple TV / AirPlay
receiver**: ``HTTP/1.1 403 Forbidden`` with header ``Server:
AirTunes/860.7.1`` on port 5000; locally-administered MAC
``36:1c:eb:dc:9a:04`` (the second nibble of the first byte is
``6`` => locally administered); no Mac mDNS hostname.  Port
11434 returning "Empty reply from server" is the device closing
the connection on an unrecognised port, NOT a hung Ollama
listener.  No Ollama instance has ever been running on ``.101``
in this network.

The earlier "third physical host candidate" framing was a
network-layer mis-identification.  Recorded as **W41-INFRA-1** in
``docs/RESULTS_COORDPY_W41_INTEGRATED_SYNTHESIS.md §4.1``.  The
honest live multi-host topology going forward is the two-Mac
pair (``localhost`` + ``192.168.12.191``).
``192.168.12.248`` is recorded as gone (per user instruction).

The previous milestone's "two-reachable-host evidence" anchors
remain valid (those used ``localhost`` + ``.191`` directly via
the W39-INFRA-1 fallback path, which correctly avoided ``.101``).

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

### "W19 broke the deceptive-ambiguity wall" or "we solved adversarial ambiguity"

> *"W19 broke the deceptive-ambiguity wall."*

Forbidden as a *general* claim. The W19-1 win is *strongly
conditional* on the R-66-DECEIVE-NAIVE / R-66-CONFOUND-RESOLVABLE
bench property: the bundle carries at least one *independent
asymmetric witness* — a specific-tier handoff OTHER than the
canonical primary disambiguator whose tokenised payload mentions a
service tag asymmetrically across the candidate set. Permitted
phrasings: *"clears bar 16 of the SDK-v3.20-anchored success
criterion"*, *"first capsule-native multi-agent-coordination
method that resolves bundle-internal contradiction between a
deceptive primary and a witness-corroborated alternative"*,
*"closes the W18-Λ-deceive wall on R-66-DECEIVE-NAIVE under the
named bench property"*. Forbidden: *"W19 solved adversarial
ambiguity"* (unqualified), *"the bundle-contradiction scorer is
universal"*, *"W19-1 holds for any deceptive primary"*, *"the
trust scorer escapes deception in general"*.

The W19-1 win is conditional on:
* (a) symmetric-corroboration round-1 (R-65 / R-66 bench shape),
* (b) the bundle carrying at least one *independent asymmetric
  witness* — i.e. a specific-tier handoff from a non-canonical
  producer role under a synonym kind whose payload mentions a
  service tag asymmetrically (R-66-DECEIVE-NAIVE,
  R-66-CONFOUND-RESOLVABLE, R-66-CORROBORATED — not
  R-66-DECEIVE-TOTAL or R-66-OUTSIDE-REQUIRED),
* (c) the secondary witness's relational-mention convention being
  inside the closed-vocabulary closure the W19 exact-match scorer
  reads (the same closure that bounds W18 / W13 / W12),
* (d) the canonical-role-for-kind table identifying the canonical
  primary correctly — i.e. the primary's
  ``(source_role, claim_kind)`` pair is in
  :data:`_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND`.

If any condition fails, W19 ties FIFO or fails by construction:
* **W19-Λ-total** (no asymmetric witness anywhere in the bundle):
  W19 reduces to W18 byte-for-byte → primary-trusted branch →
  picks decoy → fails at 0.000.
* **W19-Λ-outside** (witnesses are symmetric across primary's
  named set and the complement): W19 abstains via
  ``W19_BRANCH_ABSTAINED_SYMMETRIC`` → ties FIFO at 0.000.
* **W19-Λ-real** (free-form natural-language witnesses outside the
  closure): the closed-form scorer misses by construction.

The W19 method is the *sixteenth* of sixteen named structural
axes the programme has identified; "adversarial ambiguity solved"
requires resolving every named limit theorem on every axis, which
the programme has *not* done. The W19-Λ-total and W19-Λ-outside
falsifiers in particular name structural limits no closed-form
*bundle-only* scorer can escape (the named research move beyond
both is W19-C-OUTSIDE — an outside-information axis with access
to service-graph topology / prior reliability / cross-incident
historical evidence; conjectural).

### "the W19 trust scorer is a learned model" or "W19 reads attention"

> *"The W19 method uses a small learned trust model that reads
> transformer attention to detect deception."*

Forbidden. The W19 method is a *deterministic, training-free,
closed-form* bundle-contradiction scorer:
* It identifies the canonical primary disambiguator via a closed-
  form sort with a hardcoded canonical-role-for-kind tiebreak
  (:func:`_w19_canonical_primary_index`,
  :data:`_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND`).
* It counts independent asymmetric witnesses per admitted tag via
  an O(|union| · |tokens| · |admitted_tags|) match loop, excluding
  the primary and deduplicating by
  ``(source_role, claim_kind, payload_sha)``
  (:func:`_w19_witness_counts`).
* It applies a deterministic branch decision: invert when
  ``max_aw(complement) > max_aw(named_set)``; refine when W18
  abstains AND a unique strict-max subset exists; otherwise fall
  through to W18.

There is **no learned model**, no transformer attention reading,
no embedding lookup. A learned variant is the named
**W19-C-LEARNED** conjecture, conjectural and out of scope for
SDK v3.20. Permitted phrasings: *"closed-form bundle-
contradiction scorer"*, *"deterministic training-free trust
scorer"*, *"the W19 scorer counts asymmetric witnesses excluding
the canonical primary"*. Forbidden: *"W19 reads attention
weights"*, *"the W19 trust model"*, *"the W19 learned scorer"*.

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

> *"The CoordPy runtime is fully capsule-native."*

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

> *"CoordPy runs are fully reproducible."*

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
`papers/coordpy_capsule_native_runtime.md`, and a milestone note,
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

> *"CoordPy solves context for multi-agent teams."*

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
`docs/archive/coordpy-milestones/RESULTS_COORDPY_TEAM_COORD.md` § "Cross-seed result":

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

> *"CoordPy ships capsule-native multi-agent coordination in production."*

Forbidden. The TEAM_HANDOFF / ROLE_VIEW / TEAM_DECISION capsule
kinds ship in the SDK's closed vocabulary, but the **CoordPy product
runtime** (the ``RunSpec`` → ``RUN_REPORT`` path, ``coordpy-ci``,
``coordpy-capsule verify``) does not seal any of them. They are
emitted only by ``TeamCoordinator`` — the multi-agent coordination
*research slice* (``vision_mvp/coordpy/team_coord.py``). The honest
phrasing is: "SDK v3.5 ships a multi-agent capsule coordination
research slice that runs side-by-side with the CoordPy single-run
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
  ``vision_mvp.coordpy.team_coord``) is born as a capsule and has
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

> *"Phase-59 evaluates CoordPy on a real LLM."*

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

> *"CoordPy now generalises to open-world LLM drift."*

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

### "W22 solves wire-cost" or "the latent digest is a real KV cache"

> *"W22 solves the wire-cost half of multi-agent coordination."*

Forbidden as a *general* claim. The W22-1 win is *strongly
conditional* on the R-69-CACHE-FANOUT bench property: at least
two cells share an OutsideQuery + oracle_id pair (else the cache
cannot hit) AND the inner W21 fires
``W21_BRANCH_QUORUM_RESOLVED`` on every cell (else there is
nothing to compress) AND the controller's verifier schema CID
matches the producer's signed CID. Permitted phrasings: *"clears
the post-W21 efficiency bar of the SDK-v3.23-anchored success
criterion (R-69)"*, *"the first capsule-native multi-agent-
coordination method that combines explicit-capsule passing with
audited proxies for the LatentMAS direction"*, *"closes the
wire-cost half of W21-C-CALIBRATED-TRUST on R-69-CACHE-FANOUT
under the named bench property"*. Forbidden: *"W22 solved the
wire-cost concern"* (unqualified), *"W22 is a KV cache"*, *"W22
implements latent state transfer between LLM agents"*. The W22
``SharedReadCache`` is a CAPSULE-LAYER proxy for the LatentMAS
shared-KV-read direction; it does NOT touch transformer-internal
KV caches, embedding tables, attention weights, or any model-
internal state. The "shared cache" stores raw oracle reply bytes,
content-addressed by the OutsideQuery + oracle_id CID; the cost
metric is wire-side oracle reply tokens not paid because the
entry was already in the cache — measured at the capsule
boundary.

The W22-1 win is conditional on:
* (a) the cache-hit-rate condition (cross-cell OutsideQuery
  overlap),
* (b) the inner W21 trigger condition (``W21_BRANCH_QUORUM_RESOLVED``),
* (c) the verifier-side schema CID match.

If any condition fails, W22 reduces to W21 byte-for-byte or
abstains by construction:
* **W22-Λ-no-cache** (no repeated reads): cache_tokens_saved = 0;
  the digest still compresses but no wire-side savings.
* **R-69-POISONED-DIGEST** (tampered envelope): controller fires
  ``hash_mismatch`` → W22 falls through to W21 baseline.
* **R-69-SCHEMA-DRIFT** (verifier registered with different
  schema CID): controller fires ``schema_cid_mismatch`` →
  fall through.
* **R-69-NO-TRIGGER** (W21 abstained): W22 fires
  ``W22_BRANCH_NO_TRIGGER`` and reduces to W21 byte-for-byte.

The natural extensions are conjectural and must be labelled
that way:
* **W22-C-CACHE-AMPLIFICATION** — when a probabilistic LLM
  oracle is in the registry, the cache freezes the LLM's first
  reply across all matching cells; bad first replies poison many
  cells (observed empirically on mixtral:8x7b live regime, 1/4
  drifts).
* **W22-C-LATENT-KV-NATIVE** — the W22 typed-envelope +
  verification surface as an integration boundary for true
  cross-host KV reuse between MLX-distributed LLM servers (not
  implemented; remains conjectural pending Mac-2 return).

### "We've implemented LatentMAS" or "this is collective KV pooling"

> *"This implements the LatentMAS direction."*

Forbidden as an *unqualified* claim. The W22 family implements
**capsule-layer audited proxies** for *some* of the LatentMAS
idea families:

* **schema-passing** → ``SchemaCapsule`` (closed-vocabulary type
  schema, content-addressed, shared once per session by CID).
* **delta execution** → ``LatentDigestEnvelope`` (one typed
  envelope per cell summarising the W21 vote outcome instead of
  replaying the verbose audit).
* **shared-read cache** → ``SharedReadCache`` +
  ``CachingOracleAdapter`` (CID-keyed write-once-read-many; oracle
  replies cached at the capsule boundary).
* **controller-verified side channel** → ``verify_latent_digest``
  (hash-chained, schema-versioned, parent-CID-sealed; rejection
  on any failure).

These are **honest proxies at the capsule layer**, NOT runtime
KV-cache transplants. Permitted phrasings: *"capsule-layer proxy
for the LatentMAS shared-KV-read direction"*, *"audited
controller-verified compact handoff abstraction"*, *"the closest
honest implementation of the LatentMAS idea family this repo can
verify end-to-end"*. Forbidden: *"we implemented LatentMAS"* (no
runtime KV manipulation), *"collective KV pooling"* (the cache
stores oracle reply bytes, not model-internal KV pairs),
*"latent hidden-state transfer"* (no model state is transferred;
only typed audit summaries).

If a future programme builds true KV-sharing between Apple-Silicon-
distributed MLX servers, the W22 typed-envelope + verification
surface is a useful integration-boundary anchor — but the W22
family as shipped does **not** make that claim.

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
  context" without the closure-membership conditions; "CoordPy now
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
  "Phase-59 evaluates CoordPy on a real LLM" without naming the
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

### W23 forbidden moves

#### "W23 implements cross-cell KV sharing" or "the session digest is a real shared KV cache"

> *"W23 implements cross-cell KV sharing between LLM agents."*

Forbidden as an unqualified claim. The W23
``SessionDigestEnvelope`` is a hash-chained capsule-layer summary;
it does **not** touch transformer-internal KV caches,
embedding tables, attention weights, or any model-internal state.
The "cross-cell state" is a SHA-256-addressed, schema-versioned,
parent-CID-sealed running summary of W21/W22 vote outcomes — a
controller-side audit object, not a runtime KV transplant.

Permitted phrasings: *"capsule-layer proxy for the LatentMAS
*cross-cell latent state-sharing* direction"*, *"the smallest
honest cross-cell state-sharing mechanism this repo can validate
end-to-end"*, *"clears the post-W22 efficiency bar of the SDK-
v3.24-anchored success criterion (R-70)"*, *"the first capsule-
native multi-agent-coordination method that combines explicit-
capsule passing with audited proxies for the LatentMAS direction
at the **cross-cell** session layer"*. Forbidden: *"W23 implements
cross-cell KV sharing"*, *"W23 is a real shared KV cache between
agents"*, *"the session digest is a hidden-state transfer"*.

#### "the super-token IS embedding-level steganography" or "W23 implements super-token side channels"

> *"W23 implements the LatentMAS super-token side channel."*

Forbidden as an unqualified claim. The
``SuperTokenReferenceEnvelope`` is a single-visible-token CID
prefix verified through a controller-side
``SuperTokenRegistry`` — at most one whitespace token per cell;
at most ``hex_prefix_len`` (default 16) characters of payload;
the registry is enumerable / auditable; tampering yields a
``hash_mismatch`` or ``unknown_super_token`` rejection. **No
embedding-level intervention happens. No transformer-internal
state is modified.**

Permitted phrasings: *"a bounded, audited proxy for the LatentMAS
*super-token side channel* idea"*, *"single-visible-token CID-
prefix reference verified through a controller-side registry"*,
*"the smallest honest dense-control-payload experiment this repo
can validate end-to-end"*. Forbidden: *"W23 implements
embedding-level steganography"*, *"the super-token bypasses the
explicit context channel"*, *"W23 is a covert channel"*. The
super-token is **not covert**: every reference is registered, the
registry is enumerable, and the verifier rejects unknown tokens.

#### "W23 mitigates probabilistic LLM oracles" or "we've solved cache amplification"

> *"W23 mitigates probabilistic LLM oracles."*

Forbidden as an unqualified claim. The W23-2 mitigation
(``QuorumKeyedSharedReadCache`` with
``CACHE_FRESHNESS_PER_CELL_NONCE`` on the flipping oracle)
strictly improves over W22 baseline by **+0.125** on the
**synthetic** R-70-AMPLIFIED-LLM regime — where the
``FlippingProbabilisticOracle`` deterministically returns a
decoy-asymmetric reply on consult #1 and gold-asymmetric replies
afterwards. On the **live** mixtral 8x7b probe at n=4, the same
mitigation does NOT strictly improve overall accuracy (all four
strategies tie at ``acc_full = 0.750``) — the live LLM's drift
pattern is approximately symmetric across cells at that n.

Permitted phrasings: *"empirically discharges
W22-C-CACHE-AMPLIFICATION as mitigable on the synthetic regime
at +0.125 strict gain"*, *"the W23 quorum-keyed cache changes
half of cells' outcomes on live mixtral n=4 but the mitigation
advantage is not strict per-probe at this n
(W23-C-MITIGATION-LIVE-VARIANCE)"*, *"the synthetic mitigation
preserves cross-cell wire savings on deterministic oracles while
mitigating amplification on probabilistic ones"*. Forbidden:
*"W23 solves cache amplification"*, *"the quorum-keyed cache
mitigates any LLM probabilism"*, *"W23-2 transfers to live LLMs
at n=4"*.

The W23-2 win is conditional on:
* (a) at least one registered oracle's drift pattern follows the
  flipping (first-sample-decoy → later-sample-gold) shape;
* (b) the bench has ≥ 2 cells sharing identical OutsideQuery +
  oracle_id pairs;
* (c) the per-cell nonce successfully fragments the cache key.

The synthetic flipping oracle satisfies (a) by construction; live
LLMs may or may not — the live-LLM transfer is empirical-research,
not strict-empirical.

#### "the cross-host proxy validates two-host execution"

> *"The CrossHostProducerDecoderProxy validates two-host execution."*

Forbidden as an unqualified claim. The
``CrossHostProducerDecoderProxy`` is a **within-process**
JSON-canonical serialisation round-trip; it forces the producer
and decoder to communicate ONLY through canonical bytes (no
shared Python references) so any latent shared state would
surface as a serialisation failure. This validates the
**wire-encoding contract** — bytes round-trip cleanly, the
W23 envelopes survive serialisation, the producer/decoder
interface respects the JSON-canonical encoding boundary.

It does NOT validate:
* socket-level latency, partition tolerance, or RTT contributions
  to the visible-tokens-to-decider metric;
* true two-host execution between Mac 1 and Mac 2;
* network-level adversarial conditions (dropped messages,
  out-of-order delivery, byzantine tampering on the wire).

Mac 2 has been ARP-incomplete for 17 milestones in a row. **No
true two-host execution happened in SDK v3.24.** When Mac 2
returns the same proxy interface drops in over a real socket
with no W23 code changes — but until that happens, every
"cross-host" claim must be qualified as *within-process
structural simulation*, not *cross-host execution*.

Permitted phrasings: *"within-process producer/decoder
host-split simulation"*, *"the wire-format contract is mechanically
validated by JSON-canonical round-trip on every cell"*, *"the
honest fallback for the unreachable Mac 2"*, *"the W23 envelopes
are wire-compatible with cross-host deployment when Mac 2 returns,
no W23 code changes required"*. Forbidden: *"W23 implements
cross-host KV sharing"*, *"the proxy validates two-host
execution"*, *"W23 runs on two Macs"*.

#### Labelling the SDK v3.24 result "we solved multi-agent context"

> *"SDK v3.24 solves multi-agent context."*

Forbidden. SDK v3.24 ships three honest cross-cell-axis advances
(efficiency, cache-amplification mitigation, dense-control side
channel) and the first empirical demonstration in the programme
that a named conjecture's weakness is mitigable. This clears the
post-W22 efficiency bar at +6.67 % to +25.45 % savings AND the
W22-C-CACHE-AMPLIFICATION mitigation bar at +0.125 strict gain
on the synthetic regime. But "solved" remains forbidden:

* The W23-1 win is *strongly conditional* on cross-cell session
  continuity (W23-Λ-no-delta is the named limit).
* The W23-2 mitigation does not strictly transfer to live LLMs at
  n=4 (W23-C-MITIGATION-LIVE-VARIANCE conjectural).
* The W23-3 trust boundary holds on the falsifiers tested but
  does not extend to embedding-level steganography or
  cryptographically-signed envelopes (out of scope).
* Mac 2 is unreachable; no true two-host execution validated.

Permitted phrasings:
* "SDK v3.24 clears the post-W22 cross-cell efficiency bar at
  +6.67 % (delta) to +25.45 % (super-token) on R-70-DELTA-FANOUT
  with chain-verified correctness ratification."
* "On R-70-AMPLIFIED-LLM, the W23 quorum-keyed cache empirically
  discharges W22-C-CACHE-AMPLIFICATION as mitigable at +0.125
  strict gain — the first empirical demonstration in the
  programme that a named conjecture's weakness is mitigable."
* "Three regimes anchored, the W23 conditional advance is sharp
  and falsifiable. We have not solved multi-agent context;
  we have made the strongest cross-regime conditional advance
  on the cross-cell axis to date."

### W24 forbidden moves

#### "W24 implements bounded-context summarisation in the LLM" or "the compact envelope is a real LLM context window manager"

> *"W24 implements bounded-context summarisation inside the LLM."*

Forbidden as an unqualified claim. The W24
``SessionCompactEnvelope`` is a hash-chained capsule-layer summary;
the visible-token cost on the wire collapses to a single
``<compact_ref:DDDD>`` token per cell that the controller resolves
through its own state. **Nothing inside the LLM changes** — no
attention head is reshaped, no KV cache is folded, no model
state is truncated. The "bounded window" lives at the **capsule
layer**.

Permitted phrasings: *"capsule-layer bounded-window summary
proxy for the LatentMAS *running summary* direction"*, *"a
fixed-size envelope that folds the last (compact_window - 1)
prior cell digests into one CID + bounded text"*, *"the smallest
honest bounded-context-summary mechanism this repo can validate
end-to-end"*. Forbidden: *"W24 manages the LLM context
window"*, *"the compact envelope is a real bounded summary inside
the model"*, *"W24 proves bounded context"*.

#### "W24 implements intra-cell self-consistency in the LLM" or "the resample-quorum is a real test-time mixture"

> *"W24 implements intra-cell self-consistency inside the LLM."*

Forbidden as an unqualified claim. The W24
``ResampleQuorumCachingOracleAdapter`` consults the wrapped oracle
``sample_count`` times *at the protocol layer* and returns the
majority verdict. **No change to the LLM's decoding strategy or
sampling temperature happens.** The "resample" is a closed-form
controller-side aggregation of M independent oracle consults,
not an in-model intervention.

Permitted phrasings: *"capsule-layer intra-cell resample-quorum
mitigation"*, *"M-sample majority-vote on the wrapped oracle"*,
*"closed-form policy on the oracle adapter"*. Forbidden: *"W24
implements self-consistency decoding"*, *"the W24 method is a
test-time mixture-of-samples on the LLM"*, *"W24 changes the LLM's
sampler"*.

#### "W24 mitigates probabilistic LLMs in general" or "we've solved live LLM variance"

> *"W24 mitigates live LLM probabilism."*

Forbidden as an unqualified claim. The W24-2 mitigation
strictly improves over W23 PER_CELL_NONCE by **+0.500** on the
**synthetic** R-71-INTRA-CELL-FLIP regime (where the
``IntraCellFlippingOracle`` deterministically returns
decoy-asymmetric on consult #1 and gold-asymmetric on #2..M
within each cell, in isolation as the only registered oracle). On
the **live** mixtral 8x7b probe at n=4, the same mitigation
strictly improves over W23 PER_CELL_NONCE by **+0.250** — the
live LLM does not perfectly match the synthetic pattern.

Permitted phrasings: *"empirically discharges
W23-C-MITIGATION-LIVE-VARIANCE on the intra-cell drift axis at
+0.500 strict gain (synthetic) / +0.250 strict gain (live mixtral
n=4)"*, *"the W24 resample-quorum is bounded by the LLM's
intra-cell drift pattern's similarity to the synthetic
oracle"*, *"the live mitigation is non-trivially measurable but
not saturated at the synthetic rate"*. Forbidden: *"W24 solves
live LLM variance"*, *"the resample-quorum mitigates any LLM
probabilism"*, *"W24-2 transfers fully to live LLMs"*.

The W24-2 win is conditional on:
* (a) at least one consult per cell follows a non-uniform drift
  pattern (i.e. some samples are reliably bad, others reliably
  good);
* (b) the inner oracle's behaviour is sample-count-sensitive (M
  consults yield distinguishable replies);
* (c) the cache-key freshness policy permits resampling within
  one cell without short-circuiting on cache hit.

The synthetic IntraCellFlippingOracle satisfies (a) and (b) by
construction; (c) is satisfied by the per-cell fresh oracle/
cache instances on the bench. Live LLMs at temperature=0 may or
may not satisfy (a) — the live-LLM transfer is
empirical-research, not strict-empirical.

#### "the cross-process wire validates two-host execution"

> *"The CrossProcessProducerDecoderWire validates two-host execution."*

Forbidden as an unqualified claim. The
``CrossProcessProducerDecoderWire`` is a **real
cross-PROCESS** JSON-canonical pipe (Python subprocess + stdin/
stdout); it forces the producer and decoder to communicate ONLY
through OS-level pipes (no shared Python references) so any
latent shared state would surface as a serialisation failure or
an empty subprocess reply. This is **strictly stronger** than
the W23 within-process round-trip — bytes traverse a real OS
pipe, the subprocess can be killed mid-session, and the wire
reports a real failure (not a Python exception in the same
process).

It does NOT validate:
* network-level latency, partition tolerance, or RTT contributions
  to the visible-tokens-to-decider metric across two machines;
* true two-host execution between Mac 1 and Mac 2;
* network-level adversarial conditions (dropped messages,
  out-of-order delivery, byzantine tampering on a real network).

Mac 2 has been ARP-incomplete for 18 milestones in a row. **No
true two-host execution happened in SDK v3.25.** When Mac 2
returns the same JSON-canonical interface drops in over a real
socket with no W24 code changes — but until that happens, every
"cross-host" claim must be qualified as *cross-PROCESS* (real OS
pipe), not *cross-HOST* (real network socket between machines).

Permitted phrasings: *"real cross-process producer/decoder wire
via Python subprocess + stdin/stdout pipes"*, *"the wire-format
contract is validated end-to-end by real OS-level
serialisation/deserialisation"*, *"strictly stronger
cross-process honesty than the W23 within-process round-trip"*,
*"the strongest cross-process honesty this repo can validate
end-to-end on Mac-1 alone"*. Forbidden: *"W24 implements
cross-host KV sharing"*, *"the wire validates two-host
execution"*, *"W24 runs on two Macs"*.

#### Labelling the SDK v3.25 result "we solved multi-agent context"

> *"SDK v3.25 solves multi-agent context."*

Forbidden. SDK v3.25 ships three honest cross-cell-axis advances
(bounded-window efficiency, intra-cell mitigation, real
cross-process honesty) and the first programme-internal
demonstration that the live-LLM mitigation transfer is
non-trivially measurable on a fresh probe. This clears the
post-W23 efficiency bar at +18 % to +20 % savings AND the
W23-C-MITIGATION-LIVE-VARIANCE mitigation bar at +0.500 synthetic
/ +0.250 live strict gain. But "solved" remains forbidden:

* The W24-1 win is *strongly conditional* on cross-cell session
  continuity ≥ ``compact_window`` (W24-Λ-no-compact is the named
  limit).
* The W24-2 mitigation transfers partially to live mixtral
  (+0.250 not +0.500); a live LLM whose intra-cell drift is
  unbiased symmetric across samples would produce
  ``E[mitigation] = 0`` (W24-C-LIVE-VARIANCE-COMPLETE
  conjectural).
* The W24-3 trust boundary holds on the falsifiers tested but
  does not extend to network-level adversarial conditions or
  cryptographically-signed envelopes (out of scope).
* Mac 2 is unreachable; no true two-host execution validated.

Permitted phrasings:
* "SDK v3.25 clears the post-W23 bounded-window efficiency bar
  at +18.0 % (loose) / +20.5 % (tight) on R-71-LONG-SESSION
  with compact-verified correctness ratification."
* "On R-71-INTRA-CELL-FLIP, the W24 resample-quorum empirically
  discharges W23-C-MITIGATION-LIVE-VARIANCE on the intra-cell
  drift axis at +0.500 strict gain on synthetic AND +0.250
  strict gain on live mixtral n=4 — the first programme-internal
  demonstration that the live-LLM mitigation transfer is
  non-trivially measurable on a fresh probe."
* "Three regimes anchored, the W24 conditional advance is sharp
  and falsifiable. We have not solved multi-agent context;
  we have made the strongest cross-regime conditional advance
  on the bounded-window-efficiency + intra-cell-mitigation +
  real-cross-process axes to date."

## SDK v3.30 / W29 — geometry / Grassmannian / factoradic / Lehmer / mixed-curvature / causal-validity vocabulary

W29 introduces new vocabulary that *sounds* model-internal: "geometry
partition", "subspace basis", "Grassmannian-style", "factoradic
routing", "Lehmer code", "mixed-curvature compartments", "causal-
validity gate". Each of these is a **capsule-layer audited proxy**.
None of them claim transformer-internal machinery.

What you may say (W29 honest scope):

* "On R-76-XHOST-DRIFT, the W29 geometry-partitioning step routes
  cells to per-partition inner W28 stacks with structurally
  different oracle topologies; the partition_id is a deterministic
  3-state label (LINEAR / HIERARCHICAL / CYCLIC) computed from the
  W27 input signature CID and a bounded signature history."
* "On R-76-XHOST-DRIFT, the W29 layer **strictly improves
  correctness** over both W27 and W28 baselines: 0.500 → 0.750,
  Δ = +0.250 across 5/5 seeds, with bounded overhead (mean 0.75
  tokens/cell, max 1 token/cell) and trust precision 1.000."
* "The W29 envelope's audited orthonormal subspace basis is a
  **Grassmannian-style proxy at the capsule layer** — every basis
  vector is a deterministic projection over the closed-vocabulary
  tag space; the verifier checks orthogonality (Gram off-diagonal
  ≤ 1e-4), dimension, finiteness, and content-address."
* "The W29 envelope's factoradic Lehmer-code routing index is
  bounded ``0 ≤ idx < K!`` and audit-friendly; the verifier
  decodes the index and checks it matches the registered
  consumer-permutation for the partition."
* "The W29 envelope's causal-validity signature is SHA-256 over
  (parent W28 ratification CID || sorted predecessor CIDs); it is
  a **structural commitment** the bus enforces at admission, NOT
  a temporal-ordering proof at the model layer."
* "On R-76-CROSS-HOST-LIVE the same +0.250 correctness gain holds
  on the live two-host LLM topology; 16 cross-host probe calls;
  710 LAN bytes; trust precision 1.000."
* "W29 empirically discharges W28-C-CROSS-HOST-VARIANCE on the
  magnitude axis — the first synthetic regime in the programme
  where the synthesis (W21 × W27 × W28 × W29) strictly improves
  correctness over the prior best baseline."

What you must NOT say (W29 honest scope):

* "W29 implements transformer-internal subspace projection." It
  does not. The subspace lives at the capsule layer; it is an
  audited proxy.
* "W29 implements Riemannian mixed-curvature manifolds." It does
  not. The "geometry partition" is a structural label
  (linear / hierarchical / cyclic). Mixed-curvature is invoked as
  *design metaphor only* in the module docstring.
* "W29 implements a learned manifold." It does not. Both the
  basis (``compute_structural_subspace_basis``) and the partition
  classifier (``classify_partition_id_for_cell``) are pure
  deterministic functions.
* "W29 solves transformer-internal cross-host KV sharing." It
  does not. The cross-host variance witness records LLM-probe
  disagreement on a content-addressed envelope; it does NOT share
  hidden states across hosts.
* "W29 amplifies cram-factor 8× over W28." Measured ratio on
  R-76-CHAIN-SHARED is 2.30× — the pre-committed H7 ≥ 8× bar was
  MISSED. Mechanism is real; magnitude is below bar; becomes
  named open conjecture **W29-C-CRAM-AMPLIFICATION**.
* "W29 hits 0.95 absolute correctness on R-76-XHOST-DRIFT."
  Measured 0.750 — the pre-committed H6 absolute bar was missed.
  The LOAD-BEARING Δ ≥ 0.10 axis IS met cleanly (Δ = +0.250).
* "W29 brings up Mac 2." It does not. 192.168.12.248 remains
  ARP-incomplete (24th milestone in a row).

The honest summary one may emit:

* "On R-76-XHOST-DRIFT (the first synthetic regime in the programme
  where W27 alone makes correctness mistakes), W29's geometry-
  partitioning + per-partition inner W28 dispatch strictly improves
  correctness over both W27 and W28 baselines by Δ = +0.250 across
  5/5 seeds at trust precision 1.000, on real LLM bytes too. We
  have not solved multi-agent context; we have empirically
  discharged the named open conjecture
  **W28-C-CROSS-HOST-VARIANCE** on the magnitude axis, on a regime
  where the synthesis (old explicit-capsule trust line × new
  dense-control line) is the load-bearing reason correctness
  improves. The next true wall — whichever regime makes the
  structural classifier's three-way split insufficient to
  discriminate hard cells — is the named open frontier
  **W29-C-NATIVE-LATENT** (architecture-dependent: true
  transformer-internal subspace projection vs the W29 audited
  proxy)."


## SDK v3.31 / W30 — calibrated geometry-aware dense control + multi-stride basis history + per-partition calibration prior + cross-host disagreement-routing + ancestor-chain causal binding

W30 introduces new vocabulary that *sounds* learned or model-internal:
"calibration prior", "running-mean update", "multi-stride history",
"disagreement-routing", "ancestor-chain causal binding".  Each of
these is a **capsule-layer audited proxy**.  None of them claim a
learned model, transformer-internal manipulation, or temporal-
ordering proofs at the model layer.

What you may say (W30 honest scope):

* "On R-77-CHAIN-CRAM, the W30 multi-stride basis-history
  accumulator amplifies cram-factor by 8.74× over W28 AND 3.80×
  over W29 across 5/5 seeds at stride=28, window=12 — the smallest
  stride that achieves the W29 H7 8× bar that W29 itself missed at
  2.30×.  This is the W29-C-CRAM-AMPLIFICATION empirical discharge
  on the magnitude axis."
* "On R-77-CALIBRATION-PRIOR, the W30 per-partition calibration
  prior strictly improves correctness over W29 by Δ = +0.250
  across 5/5 seeds at trust precision 1.000.  The calibrated priors
  (0.95, 0.95, 0.30) reroute CYCLIC cells (low historical
  agreement) to LINEAR (high agreement, full oracle).  This is the
  W29-C-PARTITION-CALIBRATION empirical discharge AND a sharpening
  of W21-C-CALIBRATED-TRUST."
* "On R-77-XHOST-DISAGREE, the W30 cross-host disagreement-routing
  reroutes cells with witnessed disagreement to the high-trust
  partition; Δ = +0.250 across 5/5 seeds.  This sharpens
  W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE on the synthetic axis."
* "The W30 envelope's calibration vector is a vector of floats in
  [0, 1] registered at construction time; the running-mean update
  is closed-form arithmetic over observed agreement-rate samples.
  Not a learned model."
* "The W30 envelope's basis history is a content-addressed CID over
  the last K cells' W29 basis CIDs; the verifier checks every CID
  is in the registered set AND the history_cid recomputes byte-
  for-byte."
* "The W30 envelope's ancestor chain is a sorted tuple of registered
  W29 partition CIDs over the last ancestor_window cells; the chain
  CID is SHA-256 over canonical bytes; the controller's bus
  enforces structural commitment to the declared ancestor set at
  admission time."
* "W30 discharges W29-C-CRAM-AMPLIFICATION AND W29-C-PARTITION-
  CALIBRATION in one milestone.  The next true wall is
  W29-C-NATIVE-LATENT (architecture-dependent)."

What you must NOT say (W30 honest scope):

* "W30 implements a learned trust model."  It does not.  The
  calibration vector is registered at construction time; the
  running-mean update is closed-form (``new = (prev * n + obs) /
  (n + 1)``).  A learned variant is the named conjecture
  **W30-C-PRIOR-LEARNING** (open).
* "W30 implements transformer-internal subspace projection."  It
  does not.  The basis history is a capsule-layer accumulator over
  W29's deterministic basis CIDs; an honest **proxy** for the
  LatentMAS shared-substrate direction, not a runtime KV
  transplant.
* "W30 proves temporal ordering at the model layer."  It does not.
  The ancestor chain is a sorted tuple of CIDs sealed by SHA-256;
  it commits the bus to the declared ancestor set, not to a
  particular execution sequence at the model layer.
* "W30 solves cross-host disagreement reduction."  Not in general.
  The H8 strict gain is on **synthetic** disagreement (constructed
  via a deterministic ``_SyntheticDisagreementProbe``); the
  live-LLM extension to a regime where two reachable LLMs
  systematically disagree at temperature 0 remains the named open
  **W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE**.
* "W30 amplifies cram-factor 8× on every regime."  It amplifies at
  the *registered* stride and ancestor_window; the linear scaling
  with stride is honest mechanism, but the magnitude depends on
  the bench parameters.  The H6 bar specifies the smallest stride
  at which the 8× ratio holds.
* "W30 brings up Mac 2."  It does not.  192.168.12.248 remains
  ARP-incomplete (25th milestone).
* "W30 closes W29-C-NATIVE-LATENT."  It does not.  W30 is a
  capsule-layer mechanism extension; the architecture-dependent
  native-latent question is still open.
* "W30 solves multi-agent context."  Three named conjectures
  discharged + one sharpened in one milestone is a strong step,
  but real multi-agent reality has more axes than any 5-seed
  synthetic bench can test.

The honest summary one may emit:

* "On R-77-CHAIN-CRAM, R-77-CALIBRATION-PRIOR, and R-77-XHOST-
  DISAGREE (three pre-committed regimes), the W30 calibrated
  geometry-aware dense control mechanism simultaneously discharges
  W29-C-CRAM-AMPLIFICATION (cram_ratio_w30/w28 = 8.74×, cram_ratio
  w30/w29 = 3.80×, both across 5/5 seeds) AND W29-C-PARTITION-
  CALIBRATION (Δ correctness = +0.250 across 5/5 seeds, trust
  precision 1.000) AND sharpens W29-C-CROSS-HOST-VARIANCE-LIVE-
  MAGNITUDE on the synthetic axis (Δ correctness = +0.250 across
  5/5 seeds). We have not solved multi-agent context; we have
  empirically discharged three named open conjectures in a single
  milestone, on a regime where the synthesis (W29 geometry-
  partitioning + W30 calibration + multi-stride history +
  disagreement-routing + ancestor chain) is the load-bearing reason
  correctness AND structured-bits-per-token AND trust precision all
  improve simultaneously.  The next true wall — the regime where
  the audited capsule-layer proxy is insufficient and real
  transformer-internal subspace projection is required — is the
  named open frontier **W30-C-NATIVE-LATENT**."

### W31 forbidden moves

W31 introduces another wave of vocabulary that *sounds* learned or
model-internal: "online learning", "running-mean update inside the
orchestrator", "adaptive threshold", "sealed prior trajectory",
"manifest CID".  Each of these is a **capsule-layer audited proxy**.
None of them claim a learned model in the deep-learning sense (zero
parameters, zero gradients, zero training step), transformer-
internal manipulation, or temporal-ordering proofs at the model
layer.

What you may say (W31 honest scope):

* "On R-78-NONSTATIONARY-PRIOR, the W31 closed-form running-mean
  update inside the orchestrator drives the per-partition
  calibration prior down on observed CYCLIC failures and the
  clipped-median adaptive threshold (0.8) makes the W30 reroute
  fire — strictly improving correctness over the W30 uniform-priors
  baseline by Δ = +0.125 across 5/5 seeds at trust precision 1.000.
  This is the W30-C-PRIOR-LEARNING empirical discharge on the
  magnitude axis."
* "On R-78-ADAPTIVE-THRESHOLD vs R-78-FROZEN-THRESHOLD, the
  adaptive axis isolates: with the same online-learned prior,
  clipped-median adaptive threshold yields Δ = +0.125 vs Δ = 0 for
  frozen 0.5 threshold; the difference (+0.125 ≥ +0.05) attributes
  the gain to the adaptive contribution."
* "On R-78-MANIFEST-TAMPER, the W31 manifest CID + cross-cell
  ``registered_prior_trajectory_cid`` check together detect five
  named tampers per cell-position; rejection rate = 1.000 across
  65 named tampers including the cross-cell trajectory swap that
  self-consistently recomputes the manifest CID."
* "On R-78-XLLM-LIVE, gemma2:9b on localhost vs qwen2.5:14b on
  192.168.12.191 disagree on 2/8 = 0.250 of structured-decision
  prompts at temperature 0, reproducible byte-for-byte across two
  runs.  This is the first measured live cross-architecture LLM
  disagreement at temp 0 in the programme; it sharpens
  W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE on the
  infrastructure-discharge axis."

What you must NOT say (W31 honest scope):

* "W31 implements a learned trust model."  It does not.  The
  online running-mean update is closed-form arithmetic
  (``new = (prev * n + obs) / (n + 1)``) shipped in W30 and now
  fired inside the W31 orchestrator on every cell.  Zero
  parameters; zero gradients; zero training step.  A truly-learned
  variant remains the named open conjecture
  **W31-C-LONG-WINDOW-CONVERGENCE** at the trajectory-window
  scaling axis (open).
* "W31 implements transformer-internal latent control."  It does
  not.  The sealed prior trajectory is a capsule-layer accumulator
  over W30's deterministic per-cell agreement signal; an honest
  **proxy** for the LatentMAS online-calibration direction, not a
  runtime hidden-state transplant.  W31 does not touch transformer
  KV caches, hidden states, attention weights, or embedding
  tables.
* "W31 proves temporal ordering at the model layer."  It does not.
  The prior trajectory is a sealed tuple of
  ``(cell_idx, partition_id, observed_agreement, prior_after)``
  bytes; the controller's bus enforces structural commitment to
  the declared trajectory, not to a particular execution sequence
  at the model layer.
* "W31 solves the live cross-host disagreement → strict correctness
  improvement axis."  Not yet.  The S1 result records
  2/8 = 0.250 cross-architecture disagreement at temp 0 (the
  FIRST live disagreement signal in the programme), but the full
  mechanism integration where W31 reroutes the disagreed cell to
  a registered high-trust partition AND the gold-correctness
  label correlates with one or the other model's answer is the
  named open **W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE**
  (gold-correlation axis).
* "W31 brings up Mac 2."  It does not.  192.168.12.248 remains
  ARP-incomplete (26th consecutive milestone).
* "W31 closes W30-C-NATIVE-LATENT."  It does not.  W31 is a
  capsule-layer mechanism extension; the architecture-dependent
  native-latent question carries forward as
  **W31-C-NATIVE-LATENT**.
* "W31 solves multi-agent context."  One named conjecture
  discharged (W30-C-PRIOR-LEARNING) + one sharpened
  (W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE on the
  infrastructure-discharge axis) is a strong step, but real
  multi-agent reality has more axes than any 5-seed synthetic
  bench + 8-prompt live probe can test.

The honest summary one may emit:

* "On R-78-NONSTATIONARY-PRIOR, R-78-ADAPTIVE-THRESHOLD vs
  R-78-FROZEN-THRESHOLD, and R-78-MANIFEST-TAMPER (three
  pre-committed regimes), the W31 online self-calibrated geometry-
  aware dense control mechanism simultaneously discharges
  W30-C-PRIOR-LEARNING (Δ correctness w31 vs w30 = +0.125 across
  5/5 seeds, trust precision 1.000) AND isolates the adaptive-
  threshold contribution (+0.125 ≥ +0.05 over frozen 0.5
  threshold) AND closes the cross-component swap avenue (manifest
  CID + cross-cell trajectory CID check, 65/65 = 1.000 rejection
  rate).  The R-78-XLLM-LIVE probe also records the FIRST measured
  live cross-architecture LLM disagreement at temperature 0 in the
  programme (gemma2:9b vs qwen2.5:14b, 2/8 = 0.250 disagreement
  rate, reproducible byte-for-byte).  We have not solved multi-
  agent context; we have empirically discharged one named open
  conjecture (W30-C-PRIOR-LEARNING) and sharpened another
  (W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE on the
  infrastructure-discharge axis) on a milestone where the
  synthesis (W30 calibration + W31 closed-loop online learning +
  adaptive threshold + manifest CID + first live cross-arch
  disagreement) is the load-bearing reason correctness AND audit
  precision AND infrastructure-completeness all improve
  simultaneously.  The next true wall — the regime where the
  audited capsule-layer proxy is insufficient and real
  transformer-internal subspace projection is required — remains
  the named open frontier **W30-C-NATIVE-LATENT** /
  **W31-C-NATIVE-LATENT**, plus the gold-correlation axis of
  **W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE**."

### W32 forbidden moves

W32 introduces another wave of vocabulary that *sounds* learned or
model-internal: "long-window convergence", "EWMA prior accumulator",
"Page CUSUM change-point detector", "gold-correlation routing",
"manifest-v2 CID".  Each of these is a **capsule-layer audited
proxy**.  None of them claim a learned model in the deep-learning
sense (zero parameters, zero gradients, zero training step),
transformer-internal manipulation, or a runtime ground-truth
observation; the gold-correlation map is a **registered closed-
vocabulary table**, NOT a runtime ground-truth oracle.

What you may say (W32 honest scope):

* "On R-79-LONG-WINDOW (sweep over long_window ∈ {16, 32, 64, 128}
  on the prefix-then-shift drift_recover regime), the W32 EWMA +
  Page CUSUM mechanism achieves byte-for-W31-equal correctness
  across 5/5 seeds × 4/4 windows = 20/20 cell-window pairs at
  trust precision 1.000; zero degradation as window grows.  This
  is the W31-C-LONG-WINDOW-CONVERGENCE empirical discharge on the
  **scaling-stability axis**."
* "On R-79-DRIFT-RECOVER (the multi-shift load-bearing regime),
  the strict-gain bar Δ ≥ +0.10 is honestly-null: Δ = 0.000 across
  5/5 seeds.  The bar is bounded above by the **W32-L-CYCLE-CAP
  limitation theorem** (max strict gain = ``min(c_p / 4, c_s) / N
  ≤ 0.0625`` on cycle-capped dispatcher regimes); the mechanism
  is empirically validated by ``n_change_points = 1`` firing
  exactly at the shift boundary (cell 61) across 5/5 seeds."
* "On R-79-MANIFEST-V2-TAMPER, the W32 manifest-v2 CID +
  cross-cell convergence_state_cid check together detect five
  named tampers per ratified cell; rejection rate = 1.000 across
  1525 named tampers (5/5 seeds × 61 ratified cell-positions × 5
  tampers).  The manifest-v2 CID closes cross-component swap
  avenues that the W31 manifest CID alone cannot detect (the W31
  manifest does NOT include convergence_state_cid)."
* "On R-79-XLLM-LIVE-GOLD, gemma2:9b on localhost vs qwen2.5:14b
  on 192.168.12.191 agree on 19/20 = 0.950 of gold-verifiable
  structured-decision prompts at temperature 0.  This is the FIRST
  measured live cross-architecture LLM gold-verifiable agreement
  at temp 0 in the programme; combined with W31's R-78-XLLM-LIVE
  result (0.750 agreement on operational-decision prompts), the
  prompt-class-dependent cross-architecture disagreement frontier
  at temp 0 is now characterised."

What you must NOT say (W32 honest scope):

* "W32 implements a learned trust model."  It does not.  EWMA +
  CUSUM are closed-form arithmetic with zero parameters.
* "W32 implements transformer-internal latent control."  It does
  not.  EWMA + CUSUM accumulators live at the capsule layer; an
  honest **proxy** for the LatentMAS long-window-convergent
  direction, not a runtime hidden-state transplant.
* "The W32 gold-correlation map observes ground truth at
  runtime."  It does not.  The map is a *registered closed-
  vocabulary table*; the controller registers it up-front; the
  W32 layer at runtime only reads from the map, never writes to
  it.  If the map is wrong, the W32-Λ-mis-correlated-gold
  falsifier fires (gate-bounded on synthetic; will fire on regimes
  with real cross-host disagreement).
* "W32 strictly improves correctness over W31 on all long-window
  regimes."  It does not.  On cycle-capped dispatcher regimes
  (which is the available R-79 bench infrastructure), the strict
  gain is bounded above by 0.0625 per the W32-L-CYCLE-CAP
  limitation theorem.  The strict-gain claim is honestly retained
  as **W32-C-LONG-WINDOW-STRICT-GAIN** on a regime that exceeds
  the cycle-cap (single-partition or low-cycle-window dispatcher).
* "W32 brings up Mac 2."  It does not.  192.168.12.248 remains
  ARP-incomplete (**27th consecutive milestone**, ping 100% packet
  loss).
* "W32 closes W31-C-NATIVE-LATENT."  It does not.  W32 is a
  capsule-layer mechanism extension; the architecture-dependent
  native-latent question carries forward as
  **W32-C-NATIVE-LATENT**.
* "W32 closes the live cross-host gold-correlation axis."  It does
  not.  The S1 result records 19/20 = 0.950 agreement on gold-
  verifiable prompts at temp 0; the unique disagreement (D5: TCP
  three-way handshake) has neither host correct.  The
  gold-correlation axis remains open as
  **W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE**.
* "W32 solves multi-agent context."  Five rivets tightened in one
  milestone (scaling-stability discharge of W31-C-LONG-WINDOW-
  CONVERGENCE, Page CUSUM change-point detection, manifest-v2
  cross-component tamper detection, gold-correlation routing
  infrastructure, first live cross-architecture LLM gold-
  verifiable agreement at temp 0) is a strong step, but real
  multi-agent reality has more axes than any 5-seed × 4-window
  synthetic sweep + 20-prompt live probe can test.

The honest summary one may emit:

* "On R-79-LONG-WINDOW, R-79-MANIFEST-V2-TAMPER, R-79-DRIFT-RECOVER
  (with W32-L-CYCLE-CAP limitation theorem), and R-79-XLLM-LIVE-GOLD
  (four pre-committed regimes), the W32 long-window convergent
  online geometry-aware dense control mechanism simultaneously
  discharges W31-C-LONG-WINDOW-CONVERGENCE on the scaling-stability
  axis (W32 ≥ W31 byte-for-byte across 5/5 seeds × 4/4 windows AT
  trust precision 1.000) AND closes the cross-component swap
  avenue beyond the W31 manifest CID (manifest-v2 CID + cross-cell
  convergence_state_cid check, 1525/1525 = 1.000 rejection rate)
  AND surfaces the W32-L-CYCLE-CAP limitation theorem (Δ_max ≤
  0.0625 on cycle-capped dispatcher regimes, structurally bounded)
  AND records the FIRST measured live cross-architecture LLM
  gold-verifiable agreement at temperature 0 in the programme
  (gemma2:9b vs qwen2.5:14b on 19/20 = 0.950 of gold-verifiable
  prompts, the honest converse of W31's 6/8 = 0.750 agreement on
  operational prompts).  We have not solved multi-agent context;
  we have empirically discharged one named open conjecture
  (W31-C-LONG-WINDOW-CONVERGENCE on the scaling-stability axis),
  sharpened another (W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE
  on the prompt-class-dependent agreement frontier), and proved a
  named limitation theorem (W32-L-CYCLE-CAP) that makes explicit
  the structural bound on strict-gain claims under cycle-capped
  dispatcher regimes — a load-bearing honest-scope distinction.
  The next true wall — the regime where the audited capsule-layer
  proxy is insufficient and real transformer-internal subspace
  projection is required — remains the named open frontier
  **W32-C-NATIVE-LATENT**, plus the gold-correlation axis of
  **W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE** (where the
  cross-architecture disagreement on gold-verifiable prompts at
  temp 0 systematically aligns with the gold-correctness label),
  the strict-gain axis of **W32-C-LONG-WINDOW-STRICT-GAIN** (which
  requires a regime that exceeds the W32-L-CYCLE-CAP limitation
  theorem), and the EWMA-tracked-trust integration axis of
  **W32-C-OLD-LINE-EWMA-TRUST**."

### W33 forbidden moves

W33 introduces another wave of vocabulary that *sounds* like a
learned trust model: "trust-EWMA-tracked", "per-oracle agreement
signal", "oracle-trust-state CID", "trust-trajectory CID", "anchor-
oracle reference", "manifest-v3 CID".  Each of these is a **capsule-
layer audited proxy** for the multi-oracle adjudication line that
W21 opened in 2025.  None of them claim a learned trust model in
the deep-learning sense (zero parameters, zero gradients, zero
training step), transformer-internal hidden-state manipulation, or
a runtime ground-truth oracle; the per-oracle agreement signal is a
**deterministic top-set comparison** against a registered closed-
vocabulary anchor.

What you may say (W33 honest scope):

* "On R-80-COMPROMISED-SHIFT (a three-phase oracle-compromise
  regime: K1 = 3N/8 calibration / K2 = 5N/8 single compromise /
  K3 = N double compromise), W33 EWMA-tracks each oracle's per-cell
  agreement against a registered anchor-oracle reference; oracles
  whose EWMA falls below the trust threshold are excluded from the
  effective tally; the resulting trust precision is **1.000 across
  5/5 seeds × 16 cells/seed vs W21 = 0.625 ⇒ Δ = +0.375**, with
  zero correctness regression and max overhead 1 token/cell.  This
  jointly discharges **W21-C-CALIBRATED-TRUST** AND **W32-C-OLD-
  LINE-EWMA-TRUST** in a single milestone."
* "On R-79-SINGLE-PARTITION (a prefix-then-shift regime over a
  single-partition signature space whose effective signature
  diversity exceeds the W32-L-CYCLE-CAP cycle-capped Δ_max ≤ 0.0625
  bound by construction), Δ(W32 - W31) = +0.100 across 5/5 seeds ×
  80 cells.  This is the **W32-C-LONG-WINDOW-STRICT-GAIN empirical
  discharge** on a regime that the cycle-cap does not bound."
* "On R-80-MANIFEST-V3-TAMPER, the W33 manifest-v3 CID + cross-
  component CID checks together detect five named tampers per
  ratified cell; rejection rate = 1.000 across 400 named tampers
  (5/5 seeds × 16 cells × 5 tampers).  The manifest-v3 CID closes
  cross-component swap avenues that the W21/W32 manifests alone
  cannot detect (the W33 manifest binds w21_oracle_cid +
  oracle_trust_state_cid + trust_trajectory_cid +
  anchor_oracle_set_cid + route_audit_cid_v3 + w32_long_window_cid
  together)."

What you must NOT say (W33 honest scope):

* "W33 implements a learned trust model."  It does not.  The
  per-oracle EWMA accumulator is closed-form arithmetic with zero
  parameters; the agreement signal is a deterministic top-set
  comparison.
* "W33 implements transformer-internal trust subspace projection."
  It does not.  The W33 trust state lives at the capsule layer; an
  honest **proxy** for the LatentMAS cross-agent-trust direction,
  not a runtime hidden-state transplant.  The architecture-
  dependent native-trust question carries forward as
  **W33-C-NATIVE-LATENT**.
* "W33 observes runtime ground truth."  It does not.  The anchor-
  oracle reference is a *registered subset of the same oracle
  registrations the controller already trusts*; the W33 layer
  derives the agreement signal from the controller's own probes,
  never from out-of-band ground truth.  If the anchor itself
  becomes compromised, the **W33-Λ-mis-trust-shift** falsifier
  documents the failure mode.
* "W33 closes the live cross-host trust-magnitude axis."  It does
  not.  The S1 live probe (mixtral:8x7b vs qwen3.5:35b) is honestly
  null on infrastructure (qwen3.5:35b not actually loaded on the
  remote host; mixtral past-token-budget at temp 0).  Two named
  infrastructure-fix items (W33-INFRA-1, W33-INFRA-2) are recorded;
  **W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE** remains an open
  conjecture.
* "W33 brings up Mac 2."  It does not.  192.168.12.248 remains
  ARP-incomplete (**28th consecutive milestone**, ping 100% packet
  loss).
* "W33 strictly improves trust precision on every multi-oracle
  regime."  It does not.  On regimes where no oracle is
  compromised (R-80-NO-TRUST-SHIFT) or where the trust threshold
  is pinned at 1.0 (R-80-FROZEN-TRUST-THRESHOLD), the gate never
  fires and Δ = 0; the falsifiers W33-Λ-no-trust-shift and
  W33-Λ-frozen-threshold document this.
* "W33 solves multi-oracle adjudication."  Three rivets tightened
  in one milestone (W21-C-CALIBRATED-TRUST + W32-C-OLD-LINE-EWMA-
  TRUST joint discharge, W32-C-LONG-WINDOW-STRICT-GAIN discharge
  on a single-partition regime, manifest-v3 cross-component tamper
  detection at 1.000 reject rate) is a strong step, but real
  multi-oracle reality has more axes than any 5-seed × 16-cell
  synthetic sweep can test.

The honest summary one may emit:

* "On R-80-COMPROMISED-SHIFT, R-80-MANIFEST-V3-TAMPER, R-80-TRIVIAL-
  W33, and R-79-SINGLE-PARTITION (four pre-committed regimes), the
  W33 trust-EWMA-tracked multi-oracle adjudication mechanism
  simultaneously discharges **W21-C-CALIBRATED-TRUST** AND
  **W32-C-OLD-LINE-EWMA-TRUST** AND **W32-C-LONG-WINDOW-STRICT-
  GAIN** (a joint three-conjecture discharge in a single milestone)
  AND closes the cross-component swap avenue beyond the W21 / W32
  manifests (manifest-v3 CID + cross-component CID check, 400/400
  = 1.000 rejection rate).  We have not solved multi-oracle
  adjudication; we have empirically discharged three named open
  conjectures across two research lines (the OLD W21 multi-oracle
  line AND the NEW W32 long-window-convergent line).  The next
  true wall — the regime where the audited capsule-layer trust
  proxy is insufficient and real transformer-internal trust
  subspace projection is required — remains the named open
  frontier **W33-C-NATIVE-LATENT**, plus the live cross-host
  trust-magnitude axis of **W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE**
  (currently infrastructure-bounded), the multi-host topology axis
  of **W33-C-MULTI-HOST**, and the latent cross-agent-trust axis
  of **W33-C-LATENT-CROSS-AGENT-TRUST**."

### W34 forbidden moves

W34 introduces another wave of vocabulary that *sounds* like
runtime LLM hidden-state inspection: "live oracle attestation",
"response-feature signature", "native-latent audited proxy",
"host-aware EWMA decay", "manifest-v4 CID", "preflight ``/api/tags``
discipline".  Each of these is a **capsule-layer audited proxy**
for the live-aware multi-anchor trust mechanism.  None of them
claims a learned feature-signature model in the deep-learning sense
(zero parameters, zero gradients, zero training step), transformer-
internal hidden-state manipulation, runtime KV transplant, or
out-of-band live ground-truth.  The response-feature signature is
a **closed-form deterministic SHA-256 prefix** over (first-token-
class, length-bucket, structural-hash); the multi-anchor consensus
is a **deterministic intersection** of registered anchor probes'
top_sets; the host-aware EWMA decay is a **closed-form
multiplicative scalar** in [0.5, 1.0].

What you may say (W34 honest scope):

* "On R-81-DOUBLE-ANCHOR-COMPROMISE (a three-phase compromise
  regime where the W33 single-anchor itself is compromised in the
  final phase), the W34 multi-anchor consensus reference correctly
  collapses to NO_CONSENSUS when the anchors disagree, and W34
  abstains where W33 commits to wrong; trust precision rises from
  0.625 to **1.000 across 5/5 seeds × 16 cells/seed ⇒ Δ = +0.375**,
  with zero correctness regression and max overhead 1 token/cell.
  This **closes the W33 single-anchor fragility** at the capsule
  layer."
* "On R-81-MANIFEST-V4-TAMPER, the W34 manifest-v4 CID + cross-
  component CID checks together detect five named tampers per
  ratified cell; rejection rate = **1.000 across 400 named tampers**
  (5/5 seeds × 16 cells × 5 tampers).  The manifest-v4 CID closes
  cross-component swap avenues that the W33 manifest-v3 alone
  cannot detect."
* "On R-81-RESPONSE-FEATURE-SIGNATURE, the closed-form 64-bit
  signature is byte-stable across 10 fixtures × 3 runs = 30/30
  byte-equal calls."
* "The W34 milestone closes the two named W33 infrastructure
  follow-ups via load-bearing mechanism in the live xLLM pilot:
  preflight ``/api/tags`` (W33-INFRA-1) — *honest empirical
  correction* recorded that the W33 'qwen3.5:35b not loaded'
  diagnosis was wrong; the model IS loaded — and chat-template +
  ``num_predict=4`` + stop tokens (W33-INFRA-2)."

What you must NOT say (W34 honest scope):

* "W34 implements a learned feature-signature model."  It does
  not.  ``compute_response_feature_signature`` is a closed-form
  SHA-256 hex prefix; zero parameters, zero gradients, zero
  training step.
* "W34 implements transformer-internal hidden-state projection."
  It does not.  The response-feature signature lives at the
  capsule layer; an honest **proxy** for native-latent (the
  architecture-dependent direction), not a runtime hidden-state
  transplant.  The W34 audited proxy detects feature-class
  shifts (one-word ↔ chain-of-thought, alpha ↔ digit, short ↔
  long) but cannot probe the model's hidden subspaces.
  ``W33-C-NATIVE-LATENT`` carries forward.
* "W34 solves the live cross-host trust-magnitude axis."  It does
  not.  The S1 live probe (5 host+model pairs × 13 prompts at
  temp 0) records best-effort evidence with the corrected infra
  discipline; the agreement-magnitude question is independent of
  the infra question and is honestly null on any prompt class
  where the available LLMs at temp 0 happen to agree.
  ``W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE`` and
  ``W34-C-CROSS-HOST-LIVE-MULTI-ANCHOR`` carry forward.
* "W34 brings up Mac 2."  It does not.  192.168.12.248 remains
  ARP-incomplete (**29th consecutive milestone**, ping 100%
  packet loss; port 11434 unreachable).
* "W34 strictly improves trust precision on every regime."  It
  does not.  On regimes where no anchor is compromised
  (R-81-NO-ANCHOR-DISAGREEMENT) or where the host-decay factor
  is pinned at 1.0 (R-81-FROZEN-HOST-DECAY), the multi-anchor
  consensus matches single-anchor and the host-aware decay never
  fires; Δ = 0.  The falsifiers W34-Λ-no-anchor-disagreement and
  W34-Λ-frozen-host-decay document this.
* "W34 defeats double-anchor compromise."  It does not — only
  *single*-anchor compromise.  The new
  **W34-L-MULTI-ANCHOR-CAP** limitation theorem (proved by
  inspection) names the structural ceiling: when all K anchors
  are simultaneously compromised at the capsule layer, no
  multi-anchor mechanism (including W34) can recover.  Native-
  latent (architecture-dependent) is required to break this.
* "W34 solves multi-anchor adjudication."  Two rivets closed in
  one milestone (W33 single-anchor fragility via multi-anchor
  consensus + manifest-v4 cross-component tamper detection at
  1.000 reject rate; W33-INFRA-1 + W33-INFRA-2 jointly closed)
  is a strong step, but real multi-anchor reality has more axes
  than any 5-seed × 16-cell synthetic sweep can test.

The honest summary one may emit:

* "On R-81-DOUBLE-ANCHOR-COMPROMISE, R-81-MANIFEST-V4-TAMPER,
  R-81-TRIVIAL-W34, R-81-RESPONSE-FEATURE-SIGNATURE,
  R-81-NO-ANCHOR-DISAGREEMENT, R-81-FROZEN-HOST-DECAY (six pre-
  committed regimes), the W34 live-aware multi-anchor adjudication
  mechanism closes the W33 single-anchor fragility AND closes the
  cross-component swap avenue beyond the W33 manifest-v3 (manifest-
  v4 CID, 400/400 = 1.000 rejection rate) AND adds an audited
  proxy step toward native-latent (response-feature signature,
  byte-stable) AND closes two named infrastructure follow-ups
  (W33-INFRA-1 + W33-INFRA-2).  We have not solved multi-anchor
  adjudication; we have empirically closed a structural fragility
  in the W33 trust mechanism + closed two named infra follow-ups
  + tightened the trust boundary by 14 more enumerated failure
  modes (cumulative 84 across W22 + W29 + W30 + W31 + W32 + W33
  + W34) + proved a small but sharp limitation theorem
  (W34-L-MULTI-ANCHOR-CAP).  The next true wall — the regime
  where the audited capsule-layer multi-anchor proxy is
  insufficient and real transformer-internal hidden-state-level
  cross-agent trust evidence is required — remains the named open
  frontier **W33-C-NATIVE-LATENT**, plus the live cross-host
  trust-magnitude axis of **W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE**
  AND **W34-C-CROSS-HOST-LIVE-MULTI-ANCHOR**, the multi-host
  topology axis of **W33-C-MULTI-HOST** AND **W34-C-MULTI-HOST**,
  and the latent cross-agent-trust axis of
  **W33-C-LATENT-CROSS-AGENT-TRUST**."
