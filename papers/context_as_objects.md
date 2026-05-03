# Context as Objects: Capsule-Native Coordination for Multi-Agent Teams

> Main paper draft for the Context Zero programme.
> Updated through SDK v3.39, 2026-05-02 (W22 → W38 cumulative
> trust + dense-control + live-aware multi-anchor / trust-subspace /
> host-diverse / cross-host trajectory / disjoint-consensus-reference
> guard ladder summarised in § 14.2 and § 17).
>
> **Latest milestone marker (SDK v3.39 / W38, 2026-05-02).** The
> programme now has **thirty-five** coupled research axes.  W38 wraps
> W37's anchor-cross-host basis-trajectory ratification with a
> controller-pre-registered ``ConsensusReferenceProbe`` whose host
> topology is *mechanically disjoint* from W37's trajectory hosts (the
> ``DisjointConsensusReferenceRegistry`` raises
> ``DisjointTopologyError`` otherwise; the verifier additionally
> rejects envelopes claiming an overlapping topology).  When W37
> chooses to reroute on a trajectory-anchored top_set and the disjoint
> consensus reference disagrees within ``divergence_margin_min``
> (Jaccard), W38 abstains via ``CONSENSUS_DIVERGENCE_ABSTAINED``.  The
> envelope binds a manifest-v8 CID over five components
> (parent_w37_cid, consensus_reference_state_cid,
> divergence_audit_cid, consensus_topology_cid, consensus_probe_cid).
> On R-85-COLLUDED-CROSS-HOST-TRAJECTORY, W38 raises trust precision
> over W37 from 0.500 to **1.000**
> (**Δ_trust_precision_w38_w37 = +0.500**, min and max equal across
> 5/5 seeds), abstains via DIVERGENCE on 8 cells/seed, and adds one
> visible-token overhead/cell while carrying about **9.07k structured
> bits per visible W38 token**.  On the four named falsifiers
> (R-85-TRIVIAL-W38, R-85-NO-COLLUSION-CONSENSUS-AGREES,
> R-85-CONSENSUS-ALSO-COMPROMISED, R-85-NO-CONSENSUS-REFERENCE) W38
> preserves W37 behavior and trust precision exactly.  W38 adds 14
> mechanically tested verifier failure modes, bringing the cumulative
> W22..W38 trust boundary to **140 enumerated failure modes**.  A new
> proved-conditional limitation theorem
> **W38-L-CONSENSUS-COLLUSION-CAP** is recorded: when the disjoint
> consensus reference is itself compromised in lock-step with the
> colluding trajectory hosts, W38 cannot recover at the capsule layer
> (the W38 analog of W34-L-MULTI-ANCHOR-CAP and
> W37-L-MULTI-HOST-COLLUSION-CAP); closure requires native-latent
> evidence outside the capsule layer or a 3+-host disjoint topology.
> W38 is explicitly NOT native latent transfer, NOT transformer-
> internal hidden-state projection, and NOT a KV-cache transplant; it
> is an audited capsule-layer cross-source consensus-reference proxy
> with mechanical disjoint-topology enforcement.  Mac 2 still times
> out (31st milestone in a row); bounded W38 3-host live consensus
> probe across local ``gemma2:9b``, remote ``qwen2.5:14b`` (trajectory
> pair), and remote ``qwen2.5-coder:14b`` (disjoint consensus host on
> the same physical host as a defensible weak proxy for capsule-layer
> disjointness) produced **8/8 responsive on all 3 hosts, 7/8
> trajectory-pair agreements (the one disagreement is a
> ``num_predict=4`` truncation), 7/8 cross-source consensus
> agreements, 8/8 consensus-gold correlation** at temperature 0.
> Versioning reconciliation: ``vision_mvp.__version__`` and
> ``pyproject.toml`` ``project.version`` are now both ``0.5.12`` (the
> lingering 0.5.9 vs 0.5.11 misalignment from earlier milestones is
> closed).  See
> ``docs/RESULTS_WEVRA_W38_DISJOINT_CONSENSUS_REFERENCE.md`` and
> ``docs/SUCCESS_CRITERION_W38_DISJOINT_CONSENSUS_REFERENCE.md``.

> **Previous milestone marker (SDK v3.38 / W37, 2026-05-02).** The
> programme has **thirty-four** coupled research axes.  W37 wraps
> W36's host-diverse trust-subspace guard with a closed-form,
> zero-parameter, per-(host, oracle, top_set) EWMA over *anchored*
> historical observations: a single-host current cell can be safely
> rerouted iff its (host, oracle, top_set) entry has been cross-host
> anchored above threshold across at least
> ``min_anchored_observations`` historical cells with at least
> ``min_trajectory_anchored_hosts`` distinct anchor hosts.  W37 is
> explicitly cross-cell where W36 was per-cell; its envelope binds a
> manifest-v7 CID over (parent_w36_cid, cross_host_trajectory_state_cid,
> trajectory_audit_cid, trajectory_topology_cid).  On
> R-84-SINGLE-HOST-TRAJECTORY-RECOVER, W37 improves over W36 from
> 0.500 to **1.000** correctness (**Δ_correctness_w37_w36 = +0.500**,
> min and max equal across 5/5 seeds) at trust precision **1.000**
> with one visible-token overhead/cell.  On the four named falsifiers
> (R-84-TRIVIAL-W37, R-84-NO-TRAJECTORY-HISTORY,
> R-84-POISONED-TRAJECTORY, R-84-TRAJECTORY-DISAGREEMENT) W37
> preserves W36 behavior and trust precision exactly.  The W37
> envelope carries about **29.5k structured bits per visible W37
> token** on the load-bearing regime and adds 14 mechanically tested
> verifier failure modes, bringing the cumulative W22..W37 trust
> boundary to **126 enumerated failure modes**.  A new
> proved-conditional limitation theorem
> **W37-L-MULTI-HOST-COLLUSION-CAP** is recorded: two registered
> hosts emitting a coordinated wrong top_set across enough cells can
> cross the anchored thresholds and cannot be separated at the capsule
> layer.  W37 is explicitly NOT native latent transfer, NOT
> transformer-internal hidden-state projection, and NOT a KV-cache
> transplant.  Mac 2 still times out (30th milestone in a row);
> bounded live cross-host trajectory probe across local `gemma2:9b`
> and remote `qwen2.5:14b` produced **8/8 responsive probes, 8/8
> cross-host anchored agreements, and 8/8 gold-correlated agreements**
> at temperature 0.  See
> ``docs/RESULTS_WEVRA_W37_CROSS_HOST_BASIS_TRAJECTORY.md`` and
> ``docs/SUCCESS_CRITERION_W37_CROSS_HOST_BASIS_TRAJECTORY.md``.

> **Previous milestone marker (SDK v3.37 / W36, 2026-05-02).** W36
> hardens the W35 trust-subspace proxy at the host/live boundary by
> requiring dense-control support to be independently attested by at
> least two distinct registered healthy hosts.  On R-83-HOST-DIVERSE-
> RECOVER, W36 improves over W35 from 0.625 to **0.9375** correctness
> (**+0.3125**) across 5/5 seeds and restores trust precision from
> 0.6667 to **1.000** with one visible-token overhead/cell.
>
> **Previous milestone marker (SDK v3.36 / W35, 2026-05-02).** W35 is
> the first milestone where the old explicit capsule line and the newer
> dense-control / geometry-aware line become one mechanism rather than
> parallel evidence.  W35 wraps W34's live-aware multi-anchor
> abstention path with a controller-verified trust-subspace dense
> proxy over W21 probe top_sets, W33 EWMA trust, W34 live-attestation
> / response-feature state, top-set stability, and host health.  On
> R-82-TRUST-SUBSPACE-SHIFT, W34 abstains on 6 disputed cells; W35
> safely reroutes 5/6 through the stable basis direction, raising
> correctness from 0.625 to **0.9375** (**+0.3125**) across 5/5
> seeds, with trust precision preserved at **1.000**.
>
> **Previous milestone marker (SDK v3.35 / W34, 2026-05-01).** W34 closes
> the W33 *single-anchor fragility* — the W33 trust-EWMA mechanism
> uses an anchor oracle reference to derive its per-oracle agreement
> signal; if the anchor itself becomes compromised, every honest
> non-anchor oracle's agreement signal drops against the wrong
> reference and the (compromised) anchor remains trusted.  W34
> wraps W33 with a *multi-anchor consensus reference* (the
> intersection of K registered anchors when at least
> ``anchor_quorum_min`` agree); when the intersection is empty
> (anchors disagree), W34 abstains — anchor disagreement is itself
> a trust signal.  W34 also adds an audited proxy step toward
> native-latent (a closed-form 64-bit response-feature signature
> over first_token_class + length_bucket + structural_hash; NOT a
> transformer-internal hidden-state projection), a closed-form
> host-aware EWMA decay, a content-addressed
> ``LiveOracleAttestation`` per probe, and a manifest-v4 CID over
> four component CIDs.  Measured **+0.375 trust-precision strict
> gain over W33 single-anchor across 5/5 seeds at min trust
> precision = 1.000** on R-81-DOUBLE-ANCHOR-COMPROMISE.  The
> milestone also closes two named W33 infrastructure follow-ups:
> **W33-INFRA-1** (preflight ``/api/tags`` discipline; an honest
> empirical correction — qwen3.5:35b on 192.168.12.191 IS in fact
> loaded; the W33 diagnosis was wrong, the real W33 infra failure
> was timeout + chat-template) and **W33-INFRA-2** (chat-template
> + ``num_predict=4`` + stop tokens for one-word probes).  Trust
> boundary tightened to **84 cumulative enumerated failure modes**
> across W22 + W29 + W30 + W31 + W32 + W33 + W34.  New
> **W34-L-MULTI-ANCHOR-CAP limitation theorem** (proved by
> inspection): when all K anchors are simultaneously compromised
> at the capsule layer, no multi-anchor mechanism (including W34)
> can recover; native-latent (architecture-dependent) is required
> to break this.  Mac 2 still ARP-incomplete (29th milestone).
> SDK_VERSION ``wevra.sdk.v3.35``; pyproject ``0.5.8``; 753/753
> tests pass (48 W34 unit + 494 phase69-81 + 211 wider wevra).
> See ``docs/RESULTS_WEVRA_W34_LIVE_AWARE_MULTI_ANCHOR.md`` and
> ``docs/SUCCESS_CRITERION_W34_LIVE_AWARE_MULTI_ANCHOR.md``.
>
> **Position in the research arc**: W35 changes the synthesis from
> "three separated lines" to "one composed audited-proxy stack"; W36
> adds the first host-diverse guard around that stack.  The
> OLD explicit capsule line for multi-oracle adjudication (W19..W22 →
> W21/W22/W33/W34) supplies the probes and trust evidence; the NEW
> dense-control / geometry-aware line (W29..W32) supplies the idea of
> controller-side dense state and projection; the live-aware
> multi-anchor line (W33..W34) supplies the abstention boundary; W36
> supplies the host-attestation boundary.  The composed stack is now
> W36 wraps W35 wraps W34 wraps W33 wraps W21 wraps the capsule-native
> runtime.  It is still an audited capsule-layer proxy.  It does not
> claim transformer-internal hidden-state projection or runtime KV
> transplant.  The deeper architecture-dependent walls
> (W33-C-NATIVE-LATENT, systematic W33-C-CROSS-HOST-LIVE-TRUST-
> MAGNITUDE, and true multi-host W34/W35/W36-C-MULTI-HOST) remain the
> next frontier.
>
> Updated through SDK v3.28, 2026-04-30 (W22 → W27 cross-cell
> efficiency ladder summarised in § 14.2; W27 — multi-chain
> salience-keyed dense-control fanout + per-signature scoping —
> is the first capsule-native method that simultaneously improves
> both efficiency AND correctness over the prior best).
> This file is intended to be the primary publication-grade paper
> draft for the programme's multi-agent context thesis. It is not a
> milestone diary. It is a paper-shaped synthesis of the system,
> theory, benchmarks, strongest positive results, strongest negative
> results, and the current open frontier.

## Abstract

Multi-agent LLM systems usually treat context as text: prompts, JSON
records, message logs, tool traces, and free-form summaries passed
between roles. That design conflates several distinct problems:
preserving ambiguity, normalizing producer drift, admitting evidence
under budget, decoding evidence jointly, carrying information across
rounds, and deciding which evidence a downstream model should spend
its limited context window on. We argue that this is the wrong
abstraction. The unit of context should be a typed, immutable,
lifecycle-bounded object with explicit budget and provenance.

We implement this view in **Wevra**, a capsule-native runtime and
research harness produced by the **Context Zero** programme. A
capsule is a content-addressed object satisfying six invariants:
identity, type, lifecycle, budget, provenance, and immutability. The
runtime first makes capsules load-bearing in execution rather than
merely post-hoc metadata: capsules are sealed from run setup through
the LLM byte boundary, artifacts are content-addressed at write time,
and lifecycles are mechanically audited. We then extend the same
contract to multi-agent coordination, where agents exchange typed
handoff capsules rather than raw messages.

The paper's main scientific contribution is a decomposition of the
multi-agent context problem into nine coupled structural axes:
(1) admission under budget, (2) intra-round bundle decoding,
(3) cross-round bundle decoding, (4) fixed-vocabulary normalization
of producer drift, (5) layered open-world normalization,
(6) producer-side ambiguity preservation, (7) decoder-side context
packing under bounded token budgets, (8) end-to-end producer-plus-
decoder composition, and (9) bundle-relational compatibility
disambiguation under symmetric corroboration. Across SDK v3.8-v3.19
we build a progressive benchmark ladder, R-54 through R-65, that
isolates these axes one by one. Admission alone wins in some regimes
but has a named ceiling. Bundle-aware decoding crosses that ceiling.
Cross-round decoding crosses a further temporal ceiling. Fixed-table
normalization closes a bounded producer-drift gap; layered
normalization closes a wider but still finite open-world gap.
Structured producer protocols close the first real-LLM upstream
ambiguity-erasure gap. Attention-aware capsule context packing closes
a downstream bounded-context gap. End-to-end composition then yields
the first fresh live real-LLM strict +1.000 advance over the
strongest non-composed baseline in the programme. A bundle-relational
compatibility disambiguator finally crosses the symmetric-corroboration
wall on a regime where the wall actually applies.

The strongest *live* positive result in the paper is SDK v3.18's W17
family: on a fresh live `qwen2.5:14b-32k` probe, a magnitude-hinted
structured producer protocol plus attention-aware capsule packing
yields `accuracy_full = 1.000`, while both substrate/FIFO and the
strongest non-composed baseline remain at 0.000 on the same live
benchmark. The same producer-side intervention transfers partially
across model class to a fresh live `qwen3.5:35b` MoE probe,
preserving the benchmark property 8/8 and yielding a +0.750 strict
gain. The strongest *closed-form-disambiguation* positive result is
SDK v3.19's W18 family: on the synthetic R-65-COMPAT regime (every
gold service AND the decoy mentioned by ≥ 2 distinct routed roles
via generic-noise kinds with comparable magnitudes — symmetric-
corroboration; round-2 specific-tier disambiguator carries a
relational-compound mention of every gold service AND no decoy
service), every closed-form salience scorer in the SDK ties FIFO at
0.000 (W17-Λ-symmetric extends verbatim); the new
:class:`RelationalCompatibilityDisambiguator` achieves
`accuracy_full = 1.000` at both `T_decoder ∈ {None, 24}`, +1.000
strict separation, stable across 5/5 alternate `bank_seed` values.
The strongest *negative* result is equally important: SDK v3.18
proves the programme's first explicit **symmetric-corroboration
limit theorem**, and SDK v3.19 names three further structural
limits — W18-Λ-no-compat (no relational signal → abstain), W18-Λ-
confound (symmetric relational signal → abstain), and W18-Λ-deceive
(adversarial relational signal → trust evidence and fail). The W18-
Λ-deceive falsifier names the structural limit *no closed-form
bundle-relational scorer that trusts its evidence can escape*
without an outside-information axis (W18-C-OUTSIDE, conjectural).

The paper does **not** claim that multi-agent context is solved in an
unqualified sense. The strongest honest claim is narrower and more
useful: multi-agent context becomes tractable when evidence is
represented as typed objects and when the runtime explicitly
separates producer-side ambiguity preservation, normalization,
admission, intra-round decoding, cross-round decoding, and
decoder-side bounded-context packing. The next open problem is no
longer "can capsules help?" It is "what richer semantics or learned
disambiguators are required once corroboration, magnitude, round
structure, normalization, and bounded packing are all exhausted?"

After this paper's main draft, the programme advanced four further
layers along exactly that axis (§ 14.2): a bundle-relational
compatibility disambiguator (W18, SDK v3.19), a bundle-contradiction-
aware trust-weighted disambiguator (W19, SDK v3.20), a single-source
outside-witness acquisition disambiguator (W20, SDK v3.21), and a
trust-weighted multi-source quorum adjudicator (W21, SDK v3.22). Each
layer crosses the prior layer's named structural wall on a regime
where it actually applies, ships ≥ 2 named falsifiers that make its
conditionality sharp, and preserves bounded-context efficiency
byte-for-byte. The W21 milestone (the first capsule-native multi-
agent-coordination method that crosses the **W20-Λ-compromised** wall
via multi-source adjudication under partial oracle compromise) sharpens
the post-paper deeper wall: the escape is bounded above by the
*integrity of the registered oracle set*, not by a richer scoring
rule. **Live LLM transfer (W21-Λ-real / W21-C-LIVE-WITH-REGISTRY) is
empirically partially discharged on Mac-1 mixtral 8x7b at +1.000 over
W20 in the registry-anchored regime; in the harder coalition regime
where the LLM's vote is required for quorum, cross-model split is
sharp (mixtral 8x7b: +0.750; gemma2:9b: +0.000)** — scale + general
knowledge matter for the live W21-Λ-real escape.

## 1. Introduction

Context is the central systems problem in multi-agent LLM workflows.
The default engineering response is prompt-centric: gather more text,
compress it, summarize it, and deliver the result to the next model
call. That framing makes context look like a window-management
problem. In practice it is several distinct problems at once:

1. A producer must preserve relevant ambiguity rather than collapse
   it upstream.
2. A bounded local view must admit the right evidence under budget.
3. A decoder must interpret admitted evidence jointly rather than
   item-by-item.
4. Evidence that arrives in different rounds may need to be decoded
   together.
5. Producer outputs may drift lexically or structurally away from the
   canonical surface expected by downstream logic.
6. Even after admission and decoding, a downstream consumer may still
   receive the wrong subset if the bundle is packed naively under a
   strict token budget.

Most agent systems blur these subproblems inside raw prompts, message
logs, and free-form tool traces. As a result, failures become hard to
classify. A wrong answer may come from upstream producer collapse,
lexical drift, insufficient admission, missing temporal composition,
context-window truncation, or genuine semantic ambiguity. When these
are not separated, the diagnosis tends to be vague: "the model was
confused", "the prompt was weak", "the retriever missed something",
or "the context was too large".

This paper advances a different thesis:

> **Context in multi-agent systems should be treated as an object, not**
> **as untyped text.**

The right unit of context is a typed, immutable, provenance-carrying
object whose lifecycle and budget are explicit. We call these objects
**capsules**. A capsule is not just an envelope around bytes. It is
part of a contract: it has a closed-vocabulary kind, a content-derived
identity, declared parents, explicit budgets, and a lifecycle state.
Once that contract is in place, the context problem becomes
decomposable, auditable, and falsifiable.

### 1.1 Core claim

The strongest current claim of the Context Zero programme is not that
all multi-agent context problems are solved. It is that the context
problem can be turned from a vague prompt-engineering complaint into a
stack of explicit, benchmarked subproblems:

- producer-side ambiguity preservation,
- normalization of producer drift,
- admission under role-local budget,
- intra-round bundle decoding,
- cross-round bundle decoding,
- and downstream bundle packing under bounded decoder context.

Each of these axes can be isolated, attacked, and bounded. The paper's
real contribution is this decomposition plus the evidence that it is
load-bearing.

### 1.2 Why Wevra matters

The codebase contribution is **Wevra**, the first product produced by
the Context Zero programme. Wevra is not the whole programme and it is
not claimed to be a universal agent platform. Its scientific value is
that it makes the thesis executable:

- capsules are load-bearing at runtime rather than reconstructed after
  the fact;
- team coordination is implemented through typed handoffs and bounded
  role views rather than prompt blobs;
- the benchmark family is wired directly to the same runtime objects;
- every positive result is paired with a named falsifier or limit;
- and many invariants are mechanically checked on every run.

The code is not merely an implementation appendix to the theory. It
is the evidence for the theory.

### 1.3 Main contributions

This paper makes six main contributions.

1. **Capsule-native execution.** We show that a runtime can use typed,
   content-addressed objects as its execution contract all the way to
   the LLM byte boundary.
2. **Capsule-native team coordination.** We extend the same contract
   from single-run execution to between-agent coordination.
3. **A benchmark ladder for multi-agent context.** We define a
   sequence of named regimes, R-53 through R-64, each created to
   expose a specific structural failure mode.
4. **Positive and negative theorem/result pairs.** Each method class
   is paired with its own named win and its own named falsifier or
   limit.
5. **A decomposition of the context problem into structural axes.**
   This is what makes the programme cumulative rather than a sequence
   of disconnected benchmark wins.
6. **The first fresh live end-to-end strict real-LLM win under
   bounded context.** SDK v3.18 provides the first result in the
   programme where producer-side and decoder-side interventions are
   both load-bearing and jointly beat the strongest weaker baseline on
   a fresh live real-LLM stream.

### 1.4 What this paper does not claim

This paper does **not** claim that multi-agent context has been
solved universally. The strongest current results remain conditional
on named benchmark properties, structured forms of drift, and bounded
semantic surfaces. Most importantly, the programme now has a named
**symmetric ambiguity wall**: once corroboration, magnitude, round
structure, normalization, and packing are all symmetric, the current
closed-form method family fails by construction.

That negative theorem is not a weakness of the paper. It is evidence
that the programme is now precise enough to identify what remains
unsolved.

### 1.5 Paper roadmap

The paper is organized as follows. Sections 2 through 5 define the
setting, the capsule contract, the runtime, and the team-level object
model. Section 6 introduces the nine structural axes that emerged
from the programme. Sections 7 through 9 describe the benchmark
ladder, the method families, and the evaluation discipline. Section 10
reports the main empirical and theorem-level results. Sections 11
through 17 synthesize those results into the strongest current thesis,
explain what has and has not been solved, position the work relative
to adjacent literatures, and state the current limitations plainly.
The appendices then summarize the claim taxonomy, map milestones to
paper contributions, and list the remaining paper-production steps for
a venue submission pass.

## 2. Setting and Problem Definition

We consider a multi-agent team with roles
$R = \{r_1, \ldots, r_m\}$ collaborating over one or more rounds.
Each role emits evidence items about a shared task. Evidence may be
causal, generic, misleading, redundant, or lexically drifted. One or
more deciding roles operate under bounded budgets and must produce a
team-level decision.

The paper focuses on the following failure modes:

- **Producer collapse:** the producer suppresses alternative
  hypotheses too early.
- **Producer drift:** the producer emits semantically useful but
  lexically drifted kinds or payloads.
- **Admission overload:** the deciding role cannot admit everything.
- **Single-round insufficiency:** round-local evidence is not enough.
- **Decoder-budget failure:** the right evidence is admitted but
  cannot fit into the final decoder context.
- **Symmetric ambiguity:** even with preserved evidence and bounded
  packing, gold and decoy remain indistinguishable under the current
  feature surface.

The programme's claim is that these must be modeled separately. A
single large prompt hides the distinction.

### 2.1 Formal decomposition

Let a scenario unfold over rounds $t = 1, \dots, T$. At each round,
producer role $r_i$ observes a multiset of raw events
$E_{i,t} = \{e_{i,t,1}, \dots, e_{i,t,n}\}$. A producer protocol
$P_i$ maps those events into a multiset of emitted handoff payloads
$H_{i,t}$. A normalizer $N$ maps emitted payloads into a canonical
surface. An admission policy $\pi_r$ selects a bounded subset
$A_{r,t} \subseteq \bigcup_i N(H_{i,t})$ for a deciding role $r$. A
decoder $D$ maps either a single-round view $A_{r,t}$ or a union of
multiple views $\bigcup_t A_{r,t}$ to an answer $\hat{y}$.

The decomposition studied in this paper is therefore:

- **Producer problem:** which informative hypotheses survive into
  $H_{i,t}$?
- **Normalization problem:** which emitted payloads are mapped
  correctly into the canonical surface?
- **Admission problem:** which canonicalized items survive under the
  role-local budget?
- **Decoding problem:** how should the surviving bundle be
  interpreted jointly?
- **Packing problem:** if a downstream consumer has its own strict
  token budget, which subset and ordering of the decoded bundle should
  be kept?

The standard prompt-centric stack typically fuses all of these into
one lossy transformation. The capsule-native view keeps them distinct
and therefore falsifiable.

### 2.2 What counts as "context" here

The paper uses "context" in a narrower and stronger sense than most
prompt-engineering discussions. Context here is not "all bytes seen by
the system." It is the typed, bounded, provenance-aware information
that a role or decoder is permitted to condition on at a particular
decision point. That distinction matters because:

- it is possible for the system to have seen information that a given
  role should not read directly;
- it is possible for preserving *too much* context to be harmful,
  because it destroys the downstream token budget or drowns decisive
  evidence;
- and it is possible for "better context" to mean **less text** but
  **more structure**.

## 3. Capsule Contract

The shared object model is the **capsule contract**. Each capsule
satisfies six invariants:

1. **Identity (C1).** The capsule identifier is a SHA-256 hash of a
   canonicalized representation of kind, payload, budget, and parent
   set.
2. **Typed kind (C2).** Every capsule belongs to a closed vocabulary.
3. **Lifecycle (C3).** Capsules move through
   `PROPOSED -> ADMITTED -> SEALED [-> RETIRED]`.
4. **Budget (C4).** Capsules carry explicit limits such as token
   count, byte size, witness count, round count, or parent count.
5. **Provenance (C5).** Capsules form a DAG stored in a hash-chained
   ledger.
6. **Immutability (C6).** Sealed capsules are frozen.

None of these ingredients is new in isolation. The novelty is that
the same contract is used all the way from runtime execution to team
coordination to evaluation.

### 3.1 Why the contract matters for science

The contract is what lets the programme attach real negative theorems
to system behavior. Several of the strongest limits in the paper
depend directly on the contract:

- if a producer never emits a handoff, no downstream method can
  recover it because the missing evidence has no CID and never enters
  the ledger;
- if a decoder consumes only a bounded packed subset, then dropped
  capsules are not just "ignored information" but a typed excluded
  set;
- if a benchmark property fails, the failure is visible at the object
  level rather than inferred heuristically from text.

Without the contract, many of the paper's strongest negative results
would be soft observations rather than crisp limits.

### 3.2 Scope and cost of the closed vocabulary

The closed vocabulary is simultaneously a strength and a cost.

It is a strength because:

- it makes audit possible,
- it makes theorem statements concrete,
- it makes normalization and decoding surfaces explicit,
- and it prevents silent drift in the meaning of kinds.

It is a cost because:

- every table has a finite closure,
- domain transfer is not automatic,
- and eventually the system hits a wall where richer semantics are
  required.

The later layers, especially W12, W13, and W17, are best read as
systematic explorations of that tradeoff.

## 4. Capsule-Native Execution

The first stage of the programme made capsules load-bearing inside one
Wevra run. The execution spine includes:

`PROFILE -> READINESS_CHECK -> SWEEP_SPEC -> SWEEP_CELL -> PROVENANCE -> ARTIFACT -> RUN_REPORT`

and then extends inward:

`PROMPT -> LLM_RESPONSE -> PARSE_OUTCOME -> PATCH_PROPOSAL -> TEST_VERDICT`

Key established runtime results include:

- lifecycle/execution correspondence for the run spine and inner loop;
- content-addressing at write time for substantive artifacts;
- deterministic DAG replay;
- mechanical lifecycle audit;
- and a sharp impossibility theorem for authenticating
  meta-artifacts inside the primary ledger, together with a detached
  witness construction.

### 4.1 Runtime as scientific instrument

The runtime should not be understood only as engineering. It is also
the measurement instrument for the research programme. Because every
load-bearing artifact is a typed capsule:

- the exact producer prompt can be referenced and hashed,
- the exact model output can be referenced and hashed,
- parser outcomes become typed witnesses,
- and later benchmark runs can be compared at the object level rather
  than only at the answer level.

This is part of why the programme can draw stronger conclusions than a
typical benchmark-only paper. The runtime is not external to the
claim; it is what makes the claim inspectable.

### 4.2 Why deterministic replay matters

Several later arguments rely on the distinction between:

- a fresh live run,
- a byte-stable replay of a live run,
- and a synthetic or synthetic-real-shaped generator.

Deterministic replay at the capsule DAG level is what lets the paper
state these distinctions precisely. W16 can make a replay-based
composition claim without pretending it is a fresh live claim, and
W17 can then say exactly what changed when the result moved from
replay to live.

## 5. Team-Level Capsule Coordination

At the team layer, the core coordination objects are:

- `TEAM_HANDOFF`: a typed handoff from one role to another;
- `ROLE_VIEW`: a bounded admitted view for one role in one
  coordination step;
- `TEAM_DECISION`: the team-level decision.

The team layer introduces mechanically checked lifecycle invariants
T-1..T-7. It also enables the key separation used throughout the
paper:

- what the producer emits,
- what the deciding role admits,
- what the decoder reads,
- and what the final answer asserts

can all be benchmarked independently.

### 5.1 Local views as bounded-context objects

The `ROLE_VIEW` object is where the programme's bounded-context thesis
becomes operational. A role view is not just a slice of the ledger. It
is the exact set of capsules the role is allowed to condition on,
subject to:

- parent constraints,
- target-role constraints,
- cardinality limits,
- token limits,
- and later, decoder-side packing limits on downstream consumption.

In other words, the role view is the first explicit formal object in
the codebase that stands in for "the minimum sufficient context for
this role at this time." The W15 layer sharpens that further by
introducing the distinction between admitted role view and decoder-fed
packed subset.

### 5.2 Why team-level audit matters

The programme's strongest multi-agent claims are not just about
accuracy. They are also about **well-formedness**. A team-level answer
is only meaningful if the coordination round itself is structurally
sound. The T-1..T-7 audit makes that explicit. This is why almost
every benchmark table in the programme reports audit preservation
alongside correctness.

## 6. Structural Axes of the Context Problem

The best way to understand the programme is as a ladder of structural
axes. Each later axis only became visible because the earlier ones
were isolated first.

### 6.1 Axis 1: Admission under budget

Question: what should a bounded role-local view admit?

Main results:

- **W7-2 / R-54.** Buffered cohort coherence beats FIFO when the gold
  tag has strict plurality.
- **W8-1 / R-55.** Cross-role corroboration beats single-tag
  plurality when decoys have raw plurality but gold has stronger
  distinct-role support.
- **W9-1 / R-56.** Multi-service corroboration beats single-tag
  corroboration on a multi-service gold regime.

Named limits:

- **W7-1.** FIFO is unbeatable when the producer already emits fewer
  candidates than the budget cap.
- **W9-4.** Service-blind admission fails when the decoy is also
  sufficiently corroborated.

Interpretation:

Admission alone can solve some hard context problems, but only when
the distinguishing structure is already visible at the level of what
gets admitted.

### 6.2 Axis 2: Intra-round bundle decoding

Question: once a single-round bundle is admitted, can bundle-aware
decoding solve a regime admission cannot solve?

Main result:

- **W10-1 / R-57.** Bundle-aware decoding strictly beats every
  admission policy on a decoder-forcing regime.

Named limit:

- **W10-Λ.** Admission alone is structurally insufficient on R-57.

Interpretation:

This is the first point where the programme proves that the meaning of
the **bundle** matters, not just the membership of the admitted set.

### 6.3 Axis 3: Cross-round decoding

Question: can evidence from multiple rounds be decoded jointly when
early evidence carries service inventory and later evidence carries
causal specificity?

Main result:

- **W11-1 / R-58.** Multi-round bundle decoding beats every
  single-round method.

Named limits:

- **W11-Λ.** Single-round methods are structurally insufficient.
- **W11-4.** Round-level budget starvation is a sharp falsifier.

Interpretation:

The programme now has a precise example of a context problem that is
not about retrieving the right item but about carrying the right
information across time.

### 6.4 Axis 4: Fixed-vocabulary normalization

Question: if producer drift stays inside a known synonym closure, can
normalization restore the cross-round decoder?

Main result:

- **W12-1 / R-59.** Fixed-table normalization plus multi-round
  decoding closes the synthetic-to-real-shaped gap.

Named limit:

- **W12-4.** Out-of-vocabulary drift defeats the fixed table.

Interpretation:

This is the first place where the programme converts a familiar LLM
"robustness" issue into a named structural axis with both a strict
positive result and a strict falsifier.

### 6.5 Axis 5: Layered open-world normalization

Question: can a layered heuristic normalizer widen the closure beyond
exact lookup while preserving backward compatibility?

Main result:

- **W13-1 / R-60-wide.** Layered normalization strictly beats the
  fixed table on a synthetic open-world regime.

Named limits:

- **W13-4.** Cosmic OOV is a sharp finite-closure wall.
- **W13-Λ-real.** On the first real-Ollama probe, the model emits
  canonical kinds and filters away the intended ambiguity upstream, so
  normalization becomes structurally invisible.

Interpretation:

W13 matters because it teaches the programme not to chase the wrong
next move. If the real producer is already canonicalizing or
compressing aggressively, then downstream normalization is not the
bottleneck.

### 6.6 Axis 6: Producer-side ambiguity preservation

Question: if the real producer compresses away ambiguity, can prompt
and protocol design preserve the hard event shape?

Main result:

- **W14-1 / R-61.** Structured producer protocol yields the first
  real-LLM strict gain on the cross-round stack.

Named limit:

- **W14-Λ-prompt.** If the producer does not emit the necessary
  ambiguous evidence, no downstream capsule method can recover it.

Interpretation:

The W14 layer is the first time the programme attacks the bottleneck
before the capsule coordination pipeline even starts.

### 6.7 Axis 7: Decoder-side bounded-context packing

Question: even if the right evidence is emitted and admitted, can
naive decoder-side packing destroy the advantage under a strict token
budget?

Main result:

- **W15-1 / R-62-tightbudget.** Attention-aware capsule context
  packing strictly beats FIFO packing under bounded `T_decoder`.

Named limits:

- **W15-Λ-budget.** FIFO packing ties FIFO when the decisive
  disambiguator falls past the token budget.
- **W15-Λ-degenerate.** If the token budget is effectively
  unbounded, the W15 gain disappears.

Interpretation:

This is one of the programme's most important conceptual moves
because it turns "minimum sufficient context" from a slogan into a
measurable object.

### 6.8 Axis 8: End-to-end composition

Question: can producer-side ambiguity preservation and decoder-side
packing be jointly necessary on the same regime?

Main results:

- **W16-1 / R-63 synthetic.** W14+W15 composition yields the first
  strict end-to-end composition gain.
- **W16-Λ-real-replay.** The same composition yields a strict gain on
  recorded real-LLM bytes.
- **W17-1 / R-64 live.** Magnitude-hinted producer protocol plus W15
  packing yields the first fresh live end-to-end strict +1.000
  advance over the strongest non-composed baseline.

Named limits:

- **W16-Λ-compose.** When producer collapse and decoder-budget
  pressure both fire, neither layer alone is sufficient.
- **W17-Λ-no-hint.** The legacy structured producer prompt preserves
  only a 7/8 + 0.500 envelope on the live axis.
- **W17-Λ-naive.** Naive prompt plus tight decoder budget yields
  total joint failure.

Interpretation:

This is the strongest evidence so far for the original Context Zero
thesis: the context problem is neither purely upstream nor purely
downstream.

### 6.9 The symmetric-ambiguity wall and the bundle-relational compatibility move (axis 9)

The strongest current negative theorem the programme had as of
SDK v3.18 was:

- **W17-Λ-symmetric.** When gold and decoy are symmetrically
  corroborated under comparable magnitudes, every closed-form
  salience scorer in the SDK ties FIFO at 0.000, even though the
  root-cause label itself can still be correct.

SDK v3.19 takes the smallest move beyond this wall on a regime
where the wall actually applies. The structural observation is
that every prior decoder layer (W11..W17) consumed only
*closed-vocabulary* fields of the admitted bundle: ``claim_kind``,
service tag, bipartite role × tag corroboration, operational
magnitudes, round index. None consumed the *relational text* of
the round-2 specific-tier disambiguator's payload — the substring
``relation=A_B_join`` in
``"deadlock relation=orders_payments_join wait_chain=2"``. On
R-64-SYM only the deadlock scenarios carried such a relational
mention; on the others, no asymmetric channel existed. R-65-COMPAT
is the *consistent* relational-mention regime: every scenario
family (deadlock / pool / disk / slow_query) carries a closed-
vocabulary relational compound naming the gold services in
round-2.

The W18 :class:`RelationalCompatibilityDisambiguator` is a
deterministic, training-free, closed-form scorer:

  1. Tokenise the round-2 disambiguator's payload (lower-cased,
     split on non-identifier chars, compound identifiers preserved).
  2. Score each admitted service tag in the union of admitted
     handoffs by direct-match + contiguous-subsequence compound
     match against the tokens.
  3. Apply the strict-asymmetric branch: keep positive-score tags
     iff at least one but not all admitted tags have positive
     score; otherwise abstain (fall through to the W15 inner
     answer byte-for-byte).

The W18 method clears the strong success bar on R-65-COMPAT:

- **W18-1 / R-65-COMPAT.** W18 = 1.000; every other capsule
  strategy = 0.000. **+1.000 strict separation** at both
  ``T_decoder = None`` (loose) AND ``T_decoder = 24`` (tight),
  stable across 5/5 alternate ``bank_seed`` values. The W18
  method consumes only the W15-packed bundle; ``tokens_kept_sum``
  is byte-for-byte identical to W15's. Bounded-context honesty
  preserved.

The conditionality is sharp on three named falsifiers:

- **W18-Λ-no-compat / R-65-NO-COMPAT.** Round-2 carries no
  service-tag mention; W18 abstains; ties FIFO at 0.000.
- **W18-Λ-confound / R-65-CONFOUND.** Round-2 mentions BOTH gold
  AND decoy; W18 abstains (every admitted tag has positive
  score); ties FIFO at 0.000.
- **W18-Λ-deceive / R-65-DECEIVE.** Round-2 mentions DECOY but
  NOT gold; W18 trusts its evidence and picks decoy; fails at
  0.000.

The named structural limit *no closed-form bundle-relational
scorer that trusts its evidence can escape* is W18-Λ-deceive.
The natural research move beyond it — an outside-information axis
to detect adversarial round-2 mentions — is named
**W18-C-OUTSIDE** and remains conjectural.

The W18 result strengthens the original thesis on the relational-
compatibility axis without retracting any prior axis. It is *not*
"ambiguity resolution solved": the W18-Λ-deceive falsifier names
a genuine structural limit; the W18-Λ-real conjecture names the
real-LLM transfer condition (closed-vocabulary relational
compounds emitted at the round-2 boundary). The defensible reading
is that *one named structural axis the prior milestone left
explicit* is now broken on a regime where it actually applies,
while the deeper adversarial-relational axis remains open.

## 7. Benchmark Ladder: R-53 to R-65

The benchmark family is cumulative. Each regime was built because the
previous one exposed a specific failure mode.

| Regime | Purpose | Main winner | Main limit |
| --- | --- | --- | --- |
| R-53 | Show when FIFO is unbeatable | none | low-surplus ceiling |
| R-54 | Gold-plurality admission regime | W7 | streaming instability |
| R-55 | Cross-role corroboration beats plurality | W8 | corroborated decoy |
| R-56 | Multi-service corroboration | W9 | corroborated multi-service decoy |
| R-57 | Decoder-forcing | W10 | admission ceiling |
| R-58 | Delayed-causal-evidence across rounds | W11 | single-round ceiling |
| R-59 | Synthetic-real-shaped producer drift | W12 | fixed-table OOV wall |
| R-60 | Open-world drift + first real-Ollama probe | W13 | closure wall; upstream erasure |
| R-61 | Producer-side ambiguity preservation | W14 | producer compression |
| R-62 | Decoder-side bounded context packing | W15 | FIFO packing wall |
| R-63 | End-to-end composition | W16 | joint producer+decoder failure |
| R-64 | Fresh live composition + symmetric wall | W17 | symmetric ambiguity |
| R-65 | Relational-compatibility disambiguation under symmetric corroboration | W18 | adversarial-relational round-2 (W18-Λ-deceive) |

### 7.1 Benchmark design principles

The benchmark ladder obeys four design rules.

1. **Each regime is motivated by the previous regime's strongest
   failure mode.**
2. **Each regime has a named bench property.**
3. **Each regime has at least one named falsifier.**
4. **Each new regime preserves or checks earlier anchors.**

This is what keeps the programme from turning into arbitrary benchmark
shopping.

### 7.2 Regime families in plain language

The ladder can also be read narratively:

- **R-53** asks when the producer is already so clean that structure
  cannot help.
- **R-54/R-55/R-56** ask when better admission rescues bounded local
  context.
- **R-57** asks when decoding matters more than admission.
- **R-58** asks when time matters.
- **R-59/R-60** ask when producer drift becomes the bottleneck.
- **R-61** asks what happens when the producer erases ambiguity
  itself.
- **R-62** asks what happens when the decoder cannot afford to read
  the full admitted union.
- **R-63** asks whether those two difficulties interact.
- **R-64** asks what happens when every current asymmetry is removed.
- **R-65** asks whether a single new information channel — the
  round-2 disambiguator's payload text — is enough to break the
  symmetric wall when the channel itself is asymmetric (R-65-COMPAT)
  AND what happens when the channel is silent (NO-COMPAT),
  symmetric (CONFOUND), or adversarial (DECEIVE).

This narrative is the actual research arc of the programme.

### 7.3 Why the ladder is not benchmark shopping

A fair concern for any long-running benchmark programme is that later
regimes may simply encode the method designer's preferred answer. The
paper tries to make that accusation difficult in four ways.

First, each new regime is motivated by a concrete failure mode exposed
by the previous strongest method. The regime does not appear from a
blank slate. It is a response to an identified ceiling. Second, later
regimes preserve earlier anchors rather than replacing them. A method
that wins only on the new regime but regresses badly on earlier ones
does not count as a clean programme advance. Third, each regime has a
named falsifier. The falsifier is not rhetorical. It is an executable
counter-regime in which the new method is expected to tie or fail.
Fourth, the success bars are pre-committed in the repository rather
than post-hoc descriptions written after a favorable run.

This design does not make the ladder bias-free in some impossible
absolute sense. It does, however, convert the relevant bias questions
into visible artifacts: reviewers can inspect which failure motivated
which regime, whether the stated falsifier really binds, and whether
the new method preserves earlier anchors. That is substantially more
scientifically useful than a single monolithic benchmark with no
internal causal story.

## 8. System Design and Method Families

### 8.1 Runtime layer

The runtime contribution is the capsule-native execution contract.
This layer supplies:

- immutable typed objects,
- a hash-chained provenance surface,
- deterministic replay,
- lifecycle audit,
- and content-addressed artifacts.

By itself, this does not solve multi-agent context, but it makes
later claims auditable.

### 8.2 Team coordination layer

The team layer contributes typed handoffs, role views, and team
decisions. The deciding role never sees an arbitrary prompt blob; it
sees a bounded, typed object set.

#### 8.2.1 Why this differs from ordinary message passing

At first glance, typed handoffs may look like ordinary structured
messages. The difference is that capsules are:

- content-addressed,
- lifecycle-bounded,
- provenance-linked,
- budgeted,
- and manipulated by the same runtime contract that governs the rest
  of the system.

That is what allows the programme to make cross-layer claims ordinary
message schemas typically do not support.

### 8.3 Admission methods

The admission family evolves from:

- FIFO and fixed priority,
- to buffered cohort coherence,
- to cross-role corroboration,
- to multi-service corroboration.

Each method is small, deterministic, and interpretable. Each also has
its own named limit. The programme does not hide the fact that
admission is only one part of the story.

#### 8.3.1 Why the admission ladder matters

The admission-side ladder is sometimes misread as a sequence of
slightly fancier heuristics. That is not the right reading. Each step
isolates a different informational pattern:

- **plurality**,
- **cross-role corroboration**,
- **multi-service corroboration**.

Those are different structural features of the evidence stream. W7,
W8, and W9 matter because they show exactly which kinds of structure
admission can exploit before decoding becomes necessary.

### 8.4 Decoder methods

The decoder family evolves from:

- single-round priority decoding,
- to bundle-aware intra-round decoding,
- to contradiction-aware cross-round decoding.

The decisive step is that the decoder operates on bundles of capsules
rather than on isolated items. This is where the argument starts to
look like a real context solution rather than a better filter.

#### 8.4.1 From item scoring to bundle interpretation

The decoder-side move changes the scientific object of study. The
question is no longer "which items are best?" but "what does this
bundle mean jointly?" That distinction is what allows the programme to
state W10-Λ and W11-Λ as ceilings on admission or single-round
reasoning rather than as failures of one heuristic.

### 8.5 Normalization methods

Normalization evolves from:

- no normalization,
- to fixed closed-vocabulary synonym tables,
- to layered heuristic abstraction rules.

The key scientific point is not that a table got larger. It is that
the programme turned closure and closure-failure into named research
objects with positive and negative results.

#### 8.5.1 Why fixed vs layered normalization is a research axis

Normalization is often treated as preprocessing. In this paper it is a
first-class research axis because:

- its closure can be measured,
- its failure modes can be named,
- it interacts with decoder correctness directly,
- and its real-LLM relevance depends on what the producer actually
  emits.

The W13-Λ-real observation is especially important because it shows
that improving a downstream normalizer may be scientifically
irrelevant if the producer is not generating the ambiguity one hoped
to normalize.

### 8.6 Producer-side protocol

The W14/W17 producer protocol family is the first layer that acts
before the capsule pipeline receives evidence. Its purpose is to stop
the model from erasing the hard event shape too early.

W14 contributes:

- observation/diagnosis separation,
- per-tier kind whitelists,
- one-claim-per-event discipline.

W17 adds:

- explicit operational threshold tables,
- anti-relative-magnitude instructions,
- and a fresh live comparison showing that these additions, not mere
  rerunning, close the remaining 1/8 miss.

#### 8.6.1 Protocol design as scientific intervention

The producer protocol is not merely "better prompting." In the
programme it functions as a controlled intervention on the producer's
emission distribution. That distinction matters:

- a good benchmark should not hard-code the answer into the prompt;
- but it may legitimately constrain the producer to emit one claim per
  event, or to separate observation from diagnosis, if that is exactly
  the phenomenon under study.

W14 and W17 are therefore best understood as *protocol* results, not
prompt-aesthetics results.

### 8.7 Decoder-side packing

The W15 packer turns bounded-context efficiency into an explicit axis
rather than a vague hope. The packer does not claim to manipulate
transformer attention weights directly. Instead it optimizes
prompt-facing evidence order and retention under budget using a
closed-form salience score plus hypothesis preservation.

#### 8.7.1 Why token efficiency is not cosmetic

Many systems can improve accuracy by reading more. That is not the bar
in this paper. The W15 layer matters precisely because it creates a
regime where:

- the right evidence exists,
- the system has already admitted it,
- but a bounded downstream consumer still fails unless the bundle is
  packed well.

That is the first place where "minimum sufficient context" becomes a
measurable systems property rather than a slogan.

### 8.8 End-to-end composition

W16 and W17 are the first layers that prove two different parts of
the system must work together on the same cell. This is the strongest
evidence so far for the original Context Zero thesis.

#### 8.8.1 Why composition is the real milestone

The W16/W17 composition results are important because they rule out a
common failure mode of layered research programmes: each layer works
on its own benchmark but they do not matter together. W16-Λ-compose
and W17-1 show the opposite:

- the layers can fail jointly,
- the layers can help jointly,
- and the resulting strict gains are larger than the gains of the
  isolated pieces.

That is the closest the programme has come so far to an actual
end-to-end context solution story.

### 8.9 What remains outside the current family

The paper is explicit that the current methods remain largely
closed-form. The next frontier likely requires:

- richer semantic compatibility scoring,
- learned or distilled disambiguation,
- or domain transfer beyond incident-style coordination.

That is not an omission; it is the precise consequence of
W17-Λ-symmetric.

## 9. Experimental Methodology

### 9.1 Evaluation discipline

The programme uses pre-committed success bars in
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`. A milestone is not
counted as a real advance unless it clears a named bar with:

- a code anchor,
- a benchmark property,
- a named falsifier,
- non-regression on earlier anchors,
- and audit preservation.

This discipline exists because earlier conjectures in the programme
were overstated and later retracted. The paper benefits from that
history rather than hiding it.

#### 9.1.1 Why the retractions matter

The earlier retractions are not embarrassing details to omit. They are
what justify the later discipline. Several later success bars exist
specifically because earlier claims were too easy to state and too
weak to survive replication.

### 9.2 Metrics

The key metrics are:

- `accuracy_full`,
- `accuracy_services`,
- `accuracy_root_cause`,
- audit pass rate,
- admitted capsule counts,
- token/context usage,
- ambiguity survival rate,
- decoy corroboration rate,
- normalization/OOV statistics,
- and cross-seed or cross-bank stability.

For W15-W17, the programme also uses bundle-facing proxy metrics such
as:

- `position_of_first_causal_claim`,
- `tokens_kept_sum`,
- `hypothesis_count_kept`,
- and `n_dropped_budget`.

These are not raw transformer attention weights. They are explicit,
auditable proxies for prompt-facing evidence placement under budget.

#### 9.2.1 Accuracy alone is not enough

The later layers reject a pure-accuracy notion of success. A method
that matches accuracy only by reading materially more context or by
relying on an effectively unbounded decoder budget does not count as a
strong advance in the programme's sense. This is why W15 and later
success bars include decoder-side token budgets explicitly.

### 9.3 Real-LLM evaluation

The real-LLM work in the paper is deliberately conservative. When a
probe is live, it is called live. When it is recorded replay, it is
called replay. When it is synthetic-real-shaped, it is called that.
This matters because the live-vs-replay distinction is one of the
paper's key honesty constraints.

#### 9.3.1 Replay is still scientifically useful

Replay is not a rhetorical fallback. It serves a specific scientific
role:

- it isolates downstream method changes while holding producer bytes
  constant;
- it enables exact comparison between composed and non-composed
  decoders;
- and it provides a bridge between synthetic and live evidence.

The paper therefore treats replay as a legitimate evidence class, but
never as a substitute for fresh live claims.

### 9.4 Reproducibility and artifact discipline

The programme's reproducibility stance is intentionally stronger than
"the code runs on our machine." Each milestone writes durable JSON
artifacts into `docs/data/`, keeps explicit result notes in `docs/`,
and ties named theorem/result families to code anchors and bench
properties. This paper benefits directly from that discipline because
its strongest claims are not reconstructed from memory. They are read
off of preserved benchmark families, CLI entrypoints, and checked
artifacts.

The practical reproducibility story has four components:

1. **Deterministic synthetic anchors.** Synthetic regimes are designed
   to be rerunnable in CI and under seed sweeps.
2. **Byte-stable replay where live endpoints are not the point under
   study.** Replay is used only when holding producer bytes fixed is
   the scientifically relevant move.
3. **Explicit live capture when the claim is about live behavior.**
   The v3.18 claims are careful about this distinction.
4. **A public theory registry and success-bar registry.** These
   registries ensure that claims cannot quietly change type between
   conjecture, empirical observation, and stronger theorem-style
   statement.

No reproducibility regime is perfect. Real model endpoints can change;
local hardware can differ; and some live probes necessarily depend on
service availability. But the paper's main factual claims are all
backed either by deterministic anchors, stable replay artifacts, or
explicit live captures whose status is recorded rather than blurred.

## 10. Main Results

Before discussing the layers one by one, Table 1 summarizes the
programme's strongest current reading.

| Layer | Representative regime | Strongest method | Strongest result | Named limit |
| --- | --- | --- | --- | --- |
| Admission | R-55 / R-56 | W8 / W9 | strict gains over FIFO when corroboration structure exists | corroborated decoy |
| Intra-round decoding | R-57 | W10 | strict gain where admission alone is insufficient | admission ceiling |
| Cross-round decoding | R-58 | W11 | strict gain when delayed evidence matters | single-round ceiling |
| Fixed normalization | R-59 | W12 | synthetic-real-shaped transfer under bounded closure | fixed OOV wall |
| Layered normalization | R-60 | W13 | wider synthetic open-world closure | cosmic-OOV wall |
| Producer protocol | R-61 / R-64 | W14 / W17 | live ambiguity-preservation gain | producer compression |
| Decoder packing | R-62 | W15 | bounded-context strict gain under tight decoder budget | FIFO packing wall |
| Composition | R-63 / R-64 | W16 / W17 | live end-to-end strict gain over strongest non-composed baseline | symmetric ambiguity |

### 10.1 Runtime layer

The runtime story is already stronger than a standard agent harness:

- capsules govern execution to the LLM byte boundary,
- artifacts are content-addressed at creation time,
- lifecycle audit is mechanical,
- deterministic replay exists,
- and meta-artifact authentication has a sharp impossibility theorem
  plus a constructive workaround.

This gives the team results a trustworthy substrate.

#### 10.1.1 Why this is more than infrastructure

Without this runtime, many later failures would remain hard to
classify. In a conventional agent stack, the difference between
"producer never emitted decoy" and "downstream decoder dropped decoy"
can be surprisingly difficult to pin down. In Wevra, those are
different object-level states.

### 10.2 Admission layer

The admission results establish that structure can beat FIFO, but only
in the right regimes:

- W7 wins under gold plurality,
- W8 wins under cross-role corroboration,
- W9 wins under multi-service corroboration.

The admission layer is therefore real but limited.

#### 10.2.1 What the admission ladder actually proves

The admission ladder proves that bounded-context coordination already
contains real structure before learned semantics or bundle
interpretation appear. The later decoder wins do not erase W7-W9;
they explain where their limits are.

### 10.3 Decoder layer

The decoder results establish that downstream bundle interpretation is
not reducible to better admission:

- W10 crosses the admission ceiling,
- W11 crosses the single-round ceiling.

This is where the programme first demonstrates that the meaning of the
bundle, not just its membership, matters.

#### 10.3.1 Why bundle semantics changed the programme

Before W10, the strongest story was "choose the right evidence."
After W10 and W11, the story becomes "choose the right evidence, then
interpret it jointly, possibly across time." That is much closer to
what researchers informally mean when they say that an agent team has
a context problem.

### 10.4 Normalization layer

The normalization results show that real or real-shaped producer drift
must be handled explicitly:

- W12 wins under bounded fixed-table closure,
- W13 widens the closure,
- W13-Λ-real shows that normalization is not always the active
  bottleneck on real producers.

#### 10.4.1 The hidden value of the null real result

The W13-Λ-real observation is one of the most important negative
results in the programme because it prevents a wasted research path.
It shows that the right next move after W13 was not "add more
synonyms." It was "fix the producer-side event shape." That kind of
negative result is exactly what a serious paper should include.

### 10.5 Producer-side protocol layer

The W14/W17 story is one of the paper's central contributions.

W14 shows that:

- ambiguity can be erased upstream,
- no downstream method can recover missing emitted evidence,
- and structured prompts can restore the needed event shape.

W17 then shows:

- the remaining model-side relative-magnitude miss can be closed,
- the fresh live gain can be doubled from +0.500 to +1.000,
- and the result transfers partially across model class.

#### 10.5.1 Why W17 is more than a prompt tweak

The W17 layer is not just a slightly better prompt. It closes a
specific model-side loophole: the model was comparing event magnitudes
relatively instead of against the operational thresholds that define
the benchmark property. The contribution is therefore structural: it
changes which events qualify to become capsules in the first place.

### 10.6 Decoder-side packing layer

W15 shows that even when the right evidence exists in the admitted
union, a bounded decoder context can still destroy the win. This is
one of the strongest parts of the paper because it makes the phrase
"minimum sufficient context" measurable rather than philosophical.

#### 10.6.1 Why the packing result is central to the original goal

The original goal of the programme was never just "better
coordination." It was **bounded** coordination: per-agent minimum
sufficient context. W15 is therefore central because it is the first
place where the programme measures not only whether the right answer
is obtained but whether the answer survives a strict decoder context
limit.

### 10.7 Composition layer

W16 and W17 together provide the strongest end-to-end story in the
programme:

- producer-side preservation matters,
- decoder-side packing matters,
- both can be jointly necessary,
- and together they yield the first fresh live strict win.

#### 10.7.1 Why W16 and W17 are the paper's practical center

If the paper had to pick one main result for a broad audience, it
would be W17: fresh live producer-side ambiguity preservation plus
decoder-side packing yields the first strict +1.000 gain over the
strongest weaker baseline on a live model stream.

### 10.8 Symmetric ambiguity wall

The new negative theorem W17-Λ-symmetric is as important as the fresh
live win. It shows that the current method family still depends on
asymmetry in the evidence pattern. Once that asymmetry disappears, the
closed-form capsule-native methods in the SDK stop being sufficient.

That is not a weakness of the paper. It is the cleanest statement yet
of what remains unsolved.

#### 10.8.1 Why the symmetric wall is scientifically valuable

The symmetric wall gives the next paper its problem statement. Without
it, the programme could keep stacking conditional wins without ever
identifying the deeper semantic bottleneck. W17-Λ-symmetric stops that
drift. It says, in effect: the next step is not another packing tweak;
it is a richer ambiguity-resolution method.

## 11. Result-by-Result Synthesis

It is useful to summarize the cumulative story as a progression of
questions and answers.

### 11.1 Can context be made into a runtime object?

Yes. The W3 runtime results show that capsules can govern execution,
not just describe it after the fact.

### 11.2 Can the same object model govern team coordination?

Yes. The W4 layer shows that typed handoffs and role views can be
made load-bearing between agents.

### 11.3 Can structure beat FIFO at all?

Yes, but conditionally. W7-W9 prove several admission-side wins with
named limits.

### 11.4 Can decoding beat admission?

Yes. W10 proves the first decoder-side strict separation from every
admission policy.

### 11.5 Can time matter structurally?

Yes. W11 proves that some regimes require cross-round reasoning.

### 11.6 Can synthetic drift results transfer?

Partly. W12 and W13 show that transfer is possible under bounded or
layered closure assumptions, but W13-Λ-real shows that real producer
behavior can make normalization irrelevant by collapsing ambiguity
upstream.

### 11.7 Can real models be made to preserve the hard event shape?

Yes. W14 and W17 show that prompt/protocol design can recover the
necessary ambiguity on real models, and that operational threshold
language can close the remaining model-side relative-magnitude miss.

### 11.8 Does bounded decoder context really matter?

Yes. W15 shows that downstream context packing is not cosmetic; it can
be the difference between perfect success and total failure.

### 11.9 Do the producer and decoder layers compose?

Yes. W16 and W17 show the first end-to-end composition results, first
on replayed bytes and then on a fresh live real-LLM probe.

### 11.10 What remains unsolved?

The symmetric wall. W17-Λ-symmetric shows that the current closed-form
surface cannot resolve true symmetry.

## 12. What Has Actually Been Solved?

The programme has **not** solved all of multi-agent context. What it
has solved is the decomposition and several substantial subclasses:

1. it solved several admission-sensitive subclasses;
2. it solved decoder-forcing subclasses that admission cannot solve;
3. it solved cross-round subclasses that single-round decoding cannot
   solve;
4. it solved bounded producer-drift subclasses under explicit
   normalization assumptions;
5. it solved a producer-side ambiguity-preservation subclass on a real
   model;
6. it solved a decoder-budget subclass where bounded context itself is
   the main difficulty;
7. and it solved the first end-to-end live composition subclass where
   both producer-side and decoder-side layers are jointly necessary.

That is a substantial research result. It is also still conditional.

### 12.1 Why this still counts as progress toward "solving context"

The phrase "solve context" can be misleading if read as a universal
claim. In this paper, progress means something more disciplined:
showing that more of the difficulty is now structurally understood,
measurable, and addressable. By that standard, the programme has made
substantial progress. It has turned multiple previously blurred
failure modes into explicit theorem/result pairs and has demonstrated
end-to-end live improvement on at least one real model stream.

## 13. The Strongest Current Thesis After SDK v3.24

The strongest current thesis is:

> **Multi-agent context becomes tractable when the system is designed**
> **as a layered capsule-native coordination pipeline in which**
> **producer-side ambiguity preservation, normalization, admission,**
> **bundle-aware decoding, cross-round decoding, and bounded-context**
> **packing are each explicit, audited, and benchmarked.**

This is much stronger than "better prompts help" and much more
precise than "context windows are too small."

It is also stronger than the runtime-only claim. The runtime matters
because it lets the programme audit the evidence flow. But the real
scientific content is the layered decomposition above.

### 13.1 The shortest honest thesis statement

If the entire programme had to be reduced to one sentence, it would
be:

> Multi-agent context is not a single bottleneck but a layered object
> pipeline; solving it means making each layer explicit, bounded, and
> empirically accountable.

## 14. Why the Symmetric Wall Matters

The symmetric wall is not a side note. It is the next real frontier.

The current methods all rely on some asymmetry:

- a gold plurality,
- a corroboration asymmetry,
- a causal-tier asymmetry,
- a round asymmetry,
- a producer-side preservation asymmetry,
- or a packing asymmetry.

W17-Λ-symmetric shows that once the evidence becomes fully symmetric
under the current feature surface, the closed-form strategy family in
the SDK can no longer separate gold from decoy. That means the next
class of methods must add some richer semantic or learned
disambiguation capability.

This is exactly where the next paper or next major section of the
programme should go.

### 14.1 What the next method probably looks like

The symmetric wall strongly suggests that the next successful method
will have to go beyond the current closed-form surface. Likely
directions include:

- learned compatibility scoring across bundles,
- distilled semantic disambiguators,
- or richer narrative-level hypothesis scoring.

The current paper does not attempt to solve that wall prematurely. It
earns the right to ask the question precisely.

### 14.2 Subsequent escape ladder (SDK v3.19 → v3.20 → v3.21 → v3.22)

After this paper's main draft, the programme advanced four further
layers along exactly the axis named above. Each layer is documented
in its own milestone results note; the canonical references are:

- **W18 — Bundle-relational compatibility disambiguator (SDK v3.19,
  *RESULTS_WEVRA_RELATIONAL_DISAMBIGUATOR.md*).** The first capsule-
  native method that crosses the **W17-Λ-symmetric** wall on a
  regime where it actually applies (R-65-COMPAT). A closed-form
  scorer reads the round-2 disambiguator's payload text — the
  channel every prior decoder ignored — and projects W11 / W15's
  answer through a strict-asymmetric branch. Three named falsifiers
  (R-65-NO-COMPAT, R-65-CONFOUND, R-65-DECEIVE) make the W18-1
  conditionality sharp.
- **W19 — Bundle-contradiction-aware trust-weighted disambiguator
  (SDK v3.20, *RESULTS_WEVRA_DECEPTIVE_AMBIGUITY.md*).** The first
  capsule-native method that crosses the **W18-Λ-deceive** wall on
  the bundle-resolvable case (R-66-DECEIVE-NAIVE,
  R-66-CONFOUND-RESOLVABLE). A closed-form scorer counts independent
  asymmetric witnesses *excluding* the canonical primary and inverts
  W18's projection when witnesses contradict the primary. Two named
  falsifiers (R-66-DECEIVE-TOTAL = no asymmetric witness anywhere;
  R-66-OUTSIDE-REQUIRED = witnesses are themselves symmetric) make
  the **W19-Λ-total** and **W19-Λ-outside** walls explicit. The
  natural escape from both walls — *outside information* — is named
  W19-C-OUTSIDE.
- **W20 — Outside-witness acquisition disambiguator (SDK v3.21,
  *RESULTS_WEVRA_OUTSIDE_INFORMATION.md*).** The first capsule-
  native method that crosses the **W19-Λ-outside** wall on a regime
  where it actually applies (R-67-OUTSIDE-RESOLVES). A typed
  ``OutsideWitnessOracle`` Protocol + a deterministic
  ``ServiceGraphOracle`` add an evidence-acquisition step (one
  query per cell, bounded by ``max_response_tokens``) that the
  bundle-only scorer cannot see. Three named falsifiers
  (R-67-OUTSIDE-NONE = no signal; R-67-OUTSIDE-COMPROMISED =
  adversarial signal; R-67-JOINT-DECEPTION = jointly compromised)
  make **W20-Λ-none / W20-Λ-compromised / W20-Λ-joint-deception**
  explicit. Live LLM transfer (W20-Λ-real): mixtral 8x7b on Mac-1
  achieves +0.750 over W19; smaller / coding-specialised models
  trust the deceptive primary and tie FIFO. The natural escape
  from W20-Λ-compromised — *multi-oracle aggregation* — is named
  W20-C-MULTI-ORACLE.
- **W21 — Trust-weighted multi-oracle adjudicator (SDK v3.22,
  *RESULTS_WEVRA_MULTI_ORACLE_ADJUDICATION.md*).** The first
  capsule-native method that crosses the **W20-Λ-compromised**
  wall on a regime where it actually applies (R-68-MULTI-MAJORITY).
  A registered set of N typed oracles with prior trust weights;
  the W21 scorer issues one bounded query per oracle per cell,
  counts per-tag votes across non-abstaining replies, and
  projects only when ≥ ``quorum_min`` oracles agree on a
  non-empty proper asymmetric subset. Three named falsifiers
  (R-68-MULTI-NO-QUORUM = oracles disagree; R-68-MULTI-ALL-
  COMPROMISED = jointly compromised registered set;
  R-68-MULTI-PARTIAL = sub-quorum honest signal) make
  **W21-Λ-no-quorum / W21-Λ-all-compromised / W21-Λ-partial**
  explicit. The deeper wall is now sharper: the W21 escape is
  bounded above by the *integrity of the registered oracle set*,
  not by a richer scoring rule. The conditional W21-C-PARTIAL-
  RECOVERY (with ``quorum_min = 1`` on R-68-MULTI-PARTIAL) is
  empirically discharged at 1.000 — the quorum-strictness trade-
  off is real. Live LLM transfer (W21-Λ-real / W21-C-LIVE-WITH-
  REGISTRY): a four-oracle live registry pairing two deterministic
  registry oracles with mixtral 8x7b achieves +1.000 over W20
  (registry-anchored regime, partially discharging
  W20-C-LIVE-WITH-REGISTRY). On the harder coalition regime (LLM
  vote required for quorum), cross-model split is sharp:
  mixtral 8x7b achieves +0.750; gemma2:9b lands decoy tokens
  through the closure and ties FIFO at 0.000. **Scale + general
  knowledge matter for the W21-Λ-real escape on the LLM-vote-
  required regime**.

**Cross-cell efficiency ladder (SDK v3.23 → v3.28).** After
W21 the programme advances on a different axis: not "how to escape
a stronger semantic wall" but "how to amortise the cost of the
already-working capsule-native pipeline across cells, agents, and
salience signatures":

- **W22 — Capsule + audited latent-state-sharing hybrid (SDK v3.23,
  *RESULTS_WEVRA_CAPSULE_LATENT_HYBRID.md*).** A typed
  ``LatentDigestEnvelope`` per cell carrying the W21 quorum result
  + projected subset, signed at construction; a
  ``SharedReadCache`` reuses identical-query oracle replies across
  cells. The first capsule-native method that combines explicit-
  capsule passing with audited proxies for the LatentMAS
  latent-state-sharing direction.
- **W23 — Cross-cell delta + super-token reference (SDK v3.24,
  *RESULTS_WEVRA_W23_CROSS_CELL_DELTA.md*).** A
  ``SessionDigestEnvelope`` (hash-chained running cross-cell state)
  + ``SessionDeltaEnvelope`` (per-cell delta) +
  ``SuperTokenReferenceEnvelope`` (single-token CID dense-control
  reference). The first capsule-native method to amortise running
  state via O(1) visible-token references per cell.
- **W24 — Bounded-window session compaction + intra-cell
  resample-quorum + cross-process wire (SDK v3.25,
  *RESULTS_WEVRA_W24_SESSION_COMPACTION.md*).** A
  ``MultiCellSessionCompactor`` folds the last
  ``compact_window - 1`` cell digests into one fixed-size
  ``SessionCompactEnvelope``; a ``ResampleQuorumCachingOracleAdapter``
  mitigates intra-cell drift on probabilistic LLM oracles; a
  ``CrossProcessProducerDecoderWire`` round-trips JSON envelopes
  through a real Python subprocess (real cross-*process*, not
  cross-*host*). −18 % over W23 on R-71-LONG-SESSION.
- **W25 — Shared-fanout dense-control + cross-agent state reuse
  (SDK v3.26, *RESULTS_WEVRA_W25_SHARED_FANOUT.md*).** One producer
  computes one ``FanoutEnvelope`` for K named consumers, each
  consumer resolves via 1 ``<fanout_ref:DDDD>`` token. −69.87 %
  over W24 on R-72-FANOUT-SHARED at K=3.
- **W26 — Chain-persisted dense-control fanout + per-consumer
  projections (SDK v3.27,
  *RESULTS_WEVRA_W26_CHAIN_PERSISTED_FANOUT.md*).** A two-tier
  envelope hierarchy (``ChainAnchorEnvelope`` +
  ``ChainAdvanceEnvelope``) amortises the producer's per-cell
  salience-token cost across cells via 1-token chain-advance
  references; per-consumer ``ProjectionSlot`` map enforces
  controller-verified scope. −68.79 % over W25, −90.60 % over W24
  on R-73-CHAIN-SHARED at K=3, scaling to −92.23 % over W24 at
  K=10. W25-C-K-SCALING discharged at K∈{3,5,8,10}.
- **W27 — Multi-chain salience-keyed dense-control fanout +
  per-signature scoping (SDK v3.28,
  *RESULTS_WEVRA_W27_MULTI_CHAIN_PIVOT.md*).** The first capsule-
  native method that *simultaneously* improves both efficiency AND
  correctness over the prior best (W26) on a regime where the
  prior best architecturally limits correctness. A bounded pool
  of independent W26 stacks keyed by
  :func:`compute_input_signature_cid` over canonical input
  handoffs; the audited ``MultiChainPersistedFanoutDisambiguator``
  ships :func:`verify_salience_signature` (4 enumerated failure
  modes) and :func:`verify_chain_pivot` (8 failure modes). On
  R-74-XORACLE-RECOVER (1 producer + K=3 consumers, 16 cells, 2
  signatures, partial ServiceGraphOracle on the W26 baseline):
  **−76.27 % over W26 AND +0.500 correctness over W26**, stable
  across 5/5 seeds. Discharges **W26-C-DIVERGENCE-RECOVERY** in
  the per-signature scoping direction. Four named falsifiers
  (W27-Λ-single-signature / -pool-exhausted / -pivot-tampered /
  -signature-drift) make the W27-1 conditionality sharp.

The post-paper four-layer escape ladder (W18 → W19 → W20 → W21)
discharges, in order: the symmetric-corroboration wall, the
bundle-deceive wall (bundle-resolvable case), the bundle-outside
wall (outside-resolvable case), and the single-oracle wall
(majority-honest case). Each layer adds one structurally-distinct
move (relational scoring → trust-weighted contradiction → outside-
witness acquisition → multi-source quorum); each layer is
*conditional* on the next regime's named bench property; each
layer ships ≥ 2 named falsifiers that make its conditionality
sharp; each layer is *empirically validated* on a fresh synthetic
anchor at 1.000 strict gain over the prior strongest method, and
(for W17, W20, W21) on a *fresh live LLM probe* that materially
crosses the prior wall. The strongest current thesis after SDK
v3.22 is therefore:

> **Multi-agent context becomes tractable when the system is**
> **designed as a layered capsule-native coordination pipeline in**
> **which producer-side ambiguity preservation, normalization,**
> **admission, bundle-aware decoding, cross-round decoding,**
> **bounded-context packing, bundle-relational disambiguation,**
> **trust-weighted bundle-contradiction handling,**
> **outside-witness acquisition, AND trust-weighted multi-source**
> **quorum adjudication are each explicit, audited, benchmarked,**
> **and bounded above by named structural walls.** The deeper
> walls (W21-Λ-all-compromised, W21-Λ-no-quorum) are *named* and
> *proved-empirical*; the natural escapes (W21-C-CALIBRATED-TRUST
> via prior calibration; W22 via cross-source consistency
> detection) are *conjectural*.

## 15. Limitations

This paper has several important limitations.

1. **The strongest positive results remain benchmark-conditional.**
   That is by design, but it matters.
2. **The current family is still domain-specific.** Most strong
   results are in incident-style coordination regimes.
3. **The current semantic surface is still largely closed-form.**
   The next wall likely requires richer learned semantics.
4. **Cross-model transfer is only partial.** The 35B result is a real
   positive but not yet saturation.
5. **The product boundary is narrower than the research surface.**
   Many of the strongest methods remain research-grade and opt-in.

These are not weaknesses to hide. They are the reason the paper is
credible.

### 15.1 Why the limitations section is part of the contribution

One of the programme's distinguishing features is that limitations are
named as part of the method story, not left to reviewer inference.
This has two benefits:

- it makes the positive claims stronger, because they are bounded;
- and it turns the next research agenda into a direct continuation of
  the current one.

### 15.2 Internal validity threats

The main internal-validity question is whether later regimes merely
mirror the engineering intuitions of the current method family. The
paper's answer is incomplete but concrete: named falsifiers, anchor
preservation, and explicit ceilings reduce that risk, but they do not
eliminate it. Reviewers should still inspect whether a given regime
overfits a specific closed-form scoring rule or prompt protocol.

Another internal-validity issue is the mixture of evidence classes.
Some claims are purely deterministic synthetic statements; others are
live empirical observations against model endpoints. The paper tries to
avoid category mistakes by naming these evidence classes explicitly.
Still, a reader should not treat a synthetic stability result and a
live endpoint result as equally robust in the same sense.

### 15.3 External validity threats

The programme currently lives in a particular family of tasks:
incident-style, multi-service, causally entangled coordination. That
family is broad enough to expose real context failures but narrow
enough that some of the current semantic surfaces are meaningful. It
remains an open question how much of the W7-W17 ladder transfers to:

- product-planning teams,
- long-horizon software agents,
- scientific assistants,
- negotiation or dialogue-heavy teams,
- or workflows where the relevant evidence is not naturally organized
  as service-root-cause hypotheses.

This is why the paper presents the current thesis as a programme
result, not a universal theorem about all multi-agent cognition.

### 15.4 Why the remaining wall matters more than another incremental win

The symmetric-corroboration wall is not just the next benchmark. It is
the point at which the current feature surface no longer carries enough
information to decide correctly. That matters because it distinguishes
between two different futures for the programme.

In one future, a modestly richer but still largely interpretable
semantic compatibility surface breaks the wall and preserves bounded
context efficiency. In the other, symmetry can only be broken by
methods that effectively reintroduce large opaque model calls over
nearly the whole bundle, at which point the object model still helps
auditing but no longer supplies the main disambiguation power. The
paper does not know yet which future is correct, but it now states the
choice sharply enough to study.

## 16. Related Work and Positioning

The programme sits at the intersection of several literatures:

- content-addressed and tamper-evident object systems,
- event-sourcing and provenance-aware execution,
- exact-memory and bounded-context systems,
- multi-agent coordination and blackboard-style architectures,
- retrieval and memory systems for LLMs,
- prompt/protocol design for structured extraction,
- and evaluation/runtime harnesses for LLM systems.

The distinct contribution here is not merely that Wevra has a ledger
or that it uses typed objects. It is that the paper uses one object
model to unify:

- runtime execution,
- team coordination,
- theorem/limit statements,
- and benchmarked empirical advances.

The final submission version should include a proper bibliography and
explicit positioning against adjacent systems, memory papers, and
multi-agent reasoning papers. This draft intentionally avoids
inventing loose citations without a proper reference pass.

### 16.1 Memory, retrieval, and long-context systems

Many current systems papers frame the context problem as retrieval or
memory management: decide which past messages, tool outputs, or
documents to place in front of the next model call. That literature is
highly relevant here, but the present paper differs in two important
ways.

First, it does not assume that "the right context" already exists as a
stable set of textual units waiting to be ranked. The producer may have
collapsed useful ambiguity before retrieval even begins. Second, the
paper separates admission, decoding, and packing into different axes.
This makes it possible to say that one system failed because the right
object never survived the producer, while another failed because the
right object survived but fell out of the bounded decoder bundle.

### 16.2 Multi-agent prompting and role specialization

There is a growing literature on prompt-based multi-agent systems in
which different agents critique, debate, verify, or specialize. The
paper is aligned with that literature in spirit but differs in method.
Most prompt-centric multi-agent work takes the message exchange itself
as the central object. Here, the message exchange is downstream of a
typed object model with explicit lifecycle and provenance. That shift
is what lets the paper connect protocol design, normalizer design,
decoder design, and bounded-context packing inside one cumulative
programme.

### 16.3 Formal methods, provenance, and systems audit

The capsule contract also places the paper near provenance-aware and
auditable systems work. However, ordinary provenance systems typically
stop at traceability: which component produced which artifact? The
present paper uses provenance as part of a tighter execution contract.
The same typed object surface that makes audit possible also serves as
the unit of coordination and the unit of scientific evaluation. That is
why the paper is not only a systems-audit paper, not only a runtime
paper, and not only an agent paper.

### 16.4 How this paper differs from a runtime paper

A systems reader may initially see Wevra as a runtime paper with
benchmark appendices. That is not the right reading. The runtime is
necessary, but the main scientific object is the decomposition of
context across the benchmark ladder. The runtime is what makes the
decomposition executable.

### 16.5 How this paper differs from a prompting paper

A reader coming from prompt engineering may initially see W14 or W17
as prompt wins. That is also not the right reading. The producer
protocol layers matter because they sit inside a broader object-level
stack with named downstream limits. Their meaning comes from that
stack, not from prompt craft alone.

## 17. Discussion: What Would Count as Truly Solving Context?

In this repo, a serious claim that context is solved would require at
least:

1. strong results across several benchmark families rather than one;
2. fresh live real-LLM wins, not replay only;
3. explicit bounded-context efficiency, not just accuracy;
4. robustness beyond hand-built asymmetries;
5. and a convincing story for what happens at the symmetric wall.

The programme is not there yet. But it is now far beyond the stage of
"interesting intuition." It has:

- an executable runtime contract,
- a growing theorem registry,
- multiple strict positive separations,
- multiple named negative theorems,
- and the first fresh live end-to-end win.

That is enough to support a serious publication.

### 17.1 What a skeptical reviewer should believe

After reading the paper, a skeptical reviewer should at least accept
the following:

1. The programme has an unusually explicit object-level runtime and
   coordination surface.
2. The benchmark ladder is cumulative and not arbitrary.
3. Several strong positive results are real, not cherry-picked.
4. Several strong negative results are also real, and they sharpen the
   frontier rather than weakening the paper.
5. The live end-to-end result is strong enough to justify a main paper
   even though the overall thesis remains conditional.

## 18. Conclusion

This paper has one central message:

> **Context in multi-agent LLM systems is not primarily a prompt-size**
> **problem. It is an object-level coordination problem.**

Capsules provide the object model. Wevra provides the executable
runtime. The benchmark ladder R-53 through R-64 turns the context
problem into a sequence of explicit, falsifiable subproblems.

The strongest current result is not "we solved context." It is:

- admission helps and has a named ceiling,
- decoding helps beyond that and has a named ceiling,
- cross-round reasoning helps beyond that and has a named ceiling,
- normalization helps beyond that and has a named ceiling,
- producer-side ambiguity preservation helps beyond that,
- decoder-side bounded-context packing helps beyond that,
- producer-decoder composition helps beyond that,
- and symmetric ambiguity is the next named wall.

That decomposition is the real scientific contribution. It turns
"context" from a vague complaint into a structured research programme
with executable evidence, honest limits, and a clear next frontier.

### 18.1 Final take-away

The deepest contribution of the paper is not any single +1.000 result.
It is the fact that the programme can now say, with code and tests
behind it, **which layer failed**:

- the producer,
- the normalizer,
- the admission policy,
- the single-round decoder,
- the cross-round decoder,
- the bounded-context packer,
- or the semantic surface itself.

That is what "context as objects" finally buys.

## Appendix A. Claim Taxonomy

The programme uses an explicit taxonomy:

- **proved**
- **proved-conditional**
- **mechanically-checked**
- **empirical**
- **conjectural**
- **retracted**

This taxonomy is essential. Many agent-system papers blur these
statuses; this programme explicitly does not.

## Appendix B. Milestone-to-Paper Map

The current paper incorporates the following layers:

- **W3 family:** capsule contract and capsule-native runtime
- **W4 family:** team-level capsule coordination
- **W7-W9 families:** admission-side coordination ladder
- **W10 family:** intra-round bundle decoding
- **W11 family:** cross-round bundle decoding
- **W12 family:** fixed-vocabulary normalization under bounded
  producer drift
- **W13 family:** layered open-world normalization and real-Ollama
  null result
- **W14 family:** producer-side ambiguity preservation
- **W15 family:** decoder-side bounded-context packing
- **W16 family:** end-to-end producer+decoder composition
- **W17 family:** fresh live composition, magnitude-hinted protocol,
  cross-model live transfer, and symmetric-corroboration wall
- **W18 family:** bundle-relational compatibility disambiguation
  under symmetric corroboration (R-65, SDK v3.19)
- **W19 family:** bundle-contradiction-aware trust-weighted
  disambiguation under deceptive / confounded round-2 evidence
  (R-66, SDK v3.20)
- **W20 family:** outside-witness acquisition under bundle-only
  insufficiency (R-67, SDK v3.21)
- **W21 family:** trust-weighted multi-oracle adjudication under
  partial oracle compromise (R-68, SDK v3.22)
- **W22 family:** capsule + audited latent-state-sharing hybrid —
  schema-passing (`SchemaCapsule`), delta execution
  (`LatentDigestEnvelope`), shared-read cache (`SharedReadCache`
  + `CachingOracleAdapter`), and controller-side verification
  (`verify_latent_digest`); the first capsule-native multi-agent
  coordination method that *combines* explicit-capsule passing
  with audited proxies for the LatentMAS direction (collective
  KV pooling / latent hidden-state transfer / super-token side
  channels). On R-69-CACHE-FANOUT the W22 method strictly
  reduces visible-tokens-to-decider by 14.51-16.09 % synthetic
  and 39.08 % live-mixtral while ratifying W21 correctness
  byte-for-byte. Three named falsifiers (W22-Λ-no-cache,
  R-69-POISONED-DIGEST, R-69-SCHEMA-DRIFT) and one backward-
  compat anchor (R-69-NO-TRIGGER) make the conditionality sharp.
  Newly named conjecture **W22-C-CACHE-AMPLIFICATION** (the
  cache freezes a probabilistic LLM oracle's first reply across
  matching cells) emerges from the live mixtral 8x7b probe.
  (R-69, SDK v3.23)
- **W23 family:** capsule-native cross-cell delta execution +
  quorum-keyed cache + super-token reference — hash-chained
  cross-cell session digest (`SessionDigestEnvelope`), per-cell
  delta (`SessionDeltaEnvelope`), single-visible-token CID-prefix
  reference (`SuperTokenReferenceEnvelope` +
  `SuperTokenRegistry`), per-oracle freshness policy
  (`QuorumKeyedSharedReadCache` +
  `QuorumKeyedCachingOracleAdapter`), within-process
  producer/decoder host-split proxy
  (`CrossHostProducerDecoderProxy`), and three new controller-side
  verification functions (`verify_session_digest_chain`,
  `verify_session_delta`, `verify_super_token_reference`). The
  first capsule-native multi-agent coordination method that
  *combines* explicit-capsule passing with audited proxies for the
  LatentMAS *cross-cell* state-sharing + super-token side channel
  + cache-amplification mitigation directions. On R-70-DELTA-FANOUT
  the W23 method strictly reduces visible-tokens-to-decider over
  the W22 baseline by 6.67-7.53 % (delta path) and 25.45-28.77 %
  (super-token path) while ratifying W22 correctness byte-for-byte;
  stable across 5/5 seeds. On R-70-AMPLIFIED-LLM the W23
  quorum-keyed cache **empirically discharges**
  W22-C-CACHE-AMPLIFICATION at +0.125 strict mitigation advantage
  over W22 (the first empirical demonstration in the programme
  that a named conjecture's weakness is mitigable). Three named
  falsifiers (R-70-NO-DELTA, R-70-SUPER-TOKEN-TAMPERED,
  R-70-CHAIN-BROKEN) make the W23-1 / W23-3 conditionality sharp.
  Live mixtral 8x7b probe (n=4) shows visible-tokens savings
  +2.84 % delta, +11.37 % super-token, with chain verification
  rate 0.750 and super-token verification rate 1.000; the
  synthetic mitigation does not strictly transfer at this n
  (newly named conjecture **W23-C-MITIGATION-LIVE-VARIANCE**).
  Mac 2 unreachable (17th milestone in a row); the
  `CrossHostProducerDecoderProxy` validates the wire-encoding
  contract via JSON-canonical round-trip on every cell (≈ 1.79
  KB/cell on R-70-DELTA-FANOUT n=8), but no true two-host
  execution validated. (R-70, SDK v3.24)

- **W28 family:** ensemble-verified cross-model multi-chain pivot
  ratification — `EnsemblePivotRatificationEnvelope` (signature_cid
  + probe_votes + quorum + ratified flag), 11 enumerated failure
  modes in `verify_ensemble_pivot_ratification`, three built-in
  probe types (`DeterministicSignatureProbe`,
  `OracleConsultationProbe`, `LLMSignatureProbe`), the first
  cross-host live LLM evidence in 23 milestones (localhost gemma2:9b
  + 192.168.12.191 qwen2.5:14b; 5592 LAN bytes; 128 cross-host probe
  calls). On R-75-CROSS-HOST-LIVE the trust-weighted ensemble
  ratifies 10/16 cells with trust precision 1.000. The first
  capsule-native synthesis of the explicit-capsule trust line
  (W21 multi-oracle adjudication) and the dense-control line (W27
  multi-chain salience-keyed pool). Six named falsifiers
  (W28-Λ-single-probe / -coordinated-drift / -trust-zero / -spoofed-probe
  / -quorum-tampered / -pool-exhausted-passthrough). Newly named
  conjecture **W28-C-CROSS-HOST-VARIANCE** (live cross-host
  variance reduction on a regime where W27 itself fails) emerges
  from the synthetic R-75 banks all having W27 correctness = 1.000.
  (R-75, SDK v3.29)
- **W29 family:** geometry-partitioned product-manifold dense
  control + audited subspace-basis payload + factoradic Lehmer
  routing index + causal-validity gate + cross-host variance witness
  — `GeometryPartitionedRatificationEnvelope` (partition_id +
  factoradic_route_index + audited orthonormal subspace basis CID +
  causal_validity_signature + cross_host_variance_witness_cid +
  parent W28 ratification CID + declared predecessors), 14
  enumerated failure modes in
  `verify_geometry_partition_ratification` (partition-id forgery,
  basis-orthogonality forgery, factoradic-routing forgery,
  causal-validity forgery, partition-cid forgery), structural
  classifier `classify_partition_id_for_cell` returning one of
  LINEAR / HIERARCHICAL / CYCLIC keyed by signature_history,
  per-partition inner W28 dispatch via
  `inner_per_partition: dict[int, EnsembleVerifiedMultiChainOrchestrator]`.
  The new vocabulary (geometry, Grassmannian-style, factoradic,
  Lehmer, mixed-curvature, causal-validity) is added at the
  **capsule layer as audited proxy** — explicitly NOT a transformer-
  internal subspace projection, NOT a Riemannian curvature, NOT a
  learned manifold. On **R-76-XHOST-DRIFT** the W29 method strictly
  improves correctness over BOTH W27 and W28 baselines:
  ``correctness_w27 = correctness_w28 = 0.500`` and
  ``correctness_w29 = 0.750``, **Δ = +0.250 across 5/5 seeds**,
  ``trust_precision = 1.000``, ``mean overhead = 0.75 tokens/cell``.
  Same +0.250 gain holds on the live two-host LLM topology
  (localhost gemma2:9b + 192.168.12.191 qwen2.5:14b); 16 cross-host
  probe calls; 710 LAN bytes; trust precision 1.000. **Empirically
  discharges W28-C-CROSS-HOST-VARIANCE on the magnitude axis** —
  the first capsule-native multi-agent-coordination method to
  demonstrate the synthesis (W21 × W27, sealed by W28, geometry-
  partitioned by W29) strictly improves correctness on a regime
  where the prior best (W28) makes correctness mistakes. Five
  named falsifiers (W29-Λ-trivial-partition, W29-Λ-non-orthogonal-
  basis, W29-Λ-coordinated-drift-cross-host, plus W28-Λ-quorum-
  tampered and W28-Λ-pool-exhausted-passthrough inherited).
  Newly named conjectures **W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE**
  (live LLMs systematically disagree),
  **W29-C-CRAM-AMPLIFICATION** (richer payload that hits 8×;
  measured 2.30× on R-76-CHAIN-SHARED — H7 missed),
  **W29-C-PARTITION-CALIBRATION** (calibrated per-partition trust
  priors), and **W29-C-NATIVE-LATENT** (architecture-dependent —
  true transformer-internal subspace projection vs the W29 audited
  proxy; retained as the next true wall). 38/38 W29 unit tests +
  935/935 wider regression pass. Mac 2 (192.168.12.248) still
  ARP-incomplete (24th milestone). (R-76, SDK v3.30)
- **W30 family:** calibrated geometry-partitioned dense control +
  multi-stride basis-history + per-partition calibration prior +
  cross-host disagreement-routing + ancestor-chain causal binding —
  ``CalibratedGeometryRatificationEnvelope`` (BasisHistory CID +
  PartitionCalibrationVector + AncestorChain CID + W29 parent
  partition CID + disagreement_route metadata + calibrated_cid),
  14 disjoint enumerated failure modes in
  ``verify_calibrated_geometry_ratification`` (basis-history-CID
  forgery, calibration-vector-out-of-range, ancestor-chain-CID
  forgery, disagreement-route-target forgery, calibrated-CID
  forgery), an optional ``partition_classifier_hook`` on the W29
  ``GeometryPartitionedOrchestrator`` that the W30
  ``CalibratedGeometryOrchestrator`` uses to inject two
  closed-form audited overrides — a per-partition calibration
  prior (deterministic running mean over observed agreement,
  **NOT** a learned model: ``(prev * n + obs) / (n + 1)``) and a
  cross-host disagreement-routing override (peeks at the inner
  W28's last-result ``cross_host_disagreement_count`` from the
  *previous* cell). The new vocabulary (calibration prior,
  disagreement-routing, basis-history, ancestor-chain) is added
  at the **capsule layer as audited proxy** — explicitly NOT a
  learned partition classifier, NOT transformer-internal
  calibration, NOT a true Bayesian posterior. On
  **R-77-CHAIN-CRAM** at ``stride = 28``, ``ancestor_window =
  12`` the W30 envelope packs ``cram_w30/w28 = 8.74×`` AND
  ``cram_w30/w29 = 2.74×`` more bits of audited structured
  control per visible token, **across 5/5 seeds**.
  **Empirically discharges W29-C-CRAM-AMPLIFICATION** (8.74× ≥
  the pre-committed 8.0× bar that W29 missed at 2.30×). On
  **R-77-CALIBRATION-PRIOR** the calibration override yields
  ``Δ(W30 − W29) = +0.250 correctness gain`` across 5/5 seeds
  (**discharges W29-C-PARTITION-CALIBRATION on the closed-form-
  prior axis**). On **R-77-XHOST-DISAGREE** the disagreement-
  routing override yields ``Δ(W30 − W29) = +0.250 correctness
  gain`` across 5/5 seeds (**sharpens
  W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE on the
  capsule-layer-routing axis**). ``trust_precision = 1.000`` on
  every ratifying sub-bank; tamper rejection 60/60 across five
  named modes. Three named falsifiers
  (W30-Λ-trivial-calibration, W30-Λ-non-calibratable,
  W30-Λ-degenerate-history) all empirically confirmed. Newly
  named open conjectures **W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE**
  (regime where live LLMs systematically disagree at temp 0),
  **W30-C-NATIVE-LATENT** (architecture-dependent — true
  transformer-internal subspace projection vs the W30 audited
  proxy; retained as the next true wall),
  **W30-C-MULTI-HOST** (3+ host topology, blocked on Mac 2 ARP),
  **W30-C-PRIOR-LEARNING** (true learned per-partition prior
  vs the W30 deterministic running mean — out of scope as a
  capsule-layer mechanism). 36/36 W30 unit tests + 357/357
  focused regression pass. Mac 2 (192.168.12.248) still ARP-
  incomplete (25th milestone). (R-77, SDK v3.31)
- **W31 family:** online self-calibrated geometry-aware dense
  control + sealed prior trajectory + adaptive threshold + W31
  manifest CID + first measured live cross-architecture LLM
  disagreement at temperature 0 —
  ``OnlineCalibratedRatificationEnvelope`` (``PriorTrajectoryEntry``
  tuple sealed by SHA-256 of canonical bytes carrying
  (cell_idx, partition_id, observed_agreement, prior_after);
  threshold trajectory sealed by SHA-256; ``manifest_cid`` over
  six component CIDs; outer ``w31_cid`` over the canonical
  envelope bytes), 14 disjoint enumerated failure modes in
  ``verify_online_calibrated_ratification`` (cumulative 42 across
  W29 + W30 + W31; the load-bearing addition is the
  ``registered_prior_trajectory_cid`` cross-cell check that
  catches a swap whose components are individually self-consistent
  AND whose manifest CID has been recomputed self-consistently).
  The W31 ``OnlineCalibratedOrchestrator`` adds an
  ``online_enabled`` closed-loop running-mean update on every cell
  (driven by the deterministic per-cell agreement signal
  ``ratified AND no cross-host disagreement`` ⇒ 1.0; else 0.0)
  fed into the existing
  ``update_partition_calibration_running_mean`` primitive shipped
  in W30, an ``adaptive_threshold`` clipped-median update on the
  live calibration vector (bounded to ``[0.20, 0.80]`` via
  registered constants), and a sealed trajectory truncated to
  ``trajectory_window``. The new vocabulary (online running-mean
  / adaptive threshold / sealed trajectory / manifest CID) is
  added at the **capsule layer as audited proxy** — **NOT** a
  learned model in the deep-learning sense (zero parameters,
  zero gradients, zero training step), **NOT** transformer-
  internal subspace projection, **NOT** a runtime KV transplant.
  On **R-78-NONSTATIONARY-PRIOR** (inverted partition layout)
  the W30 baseline with uniform priors gets
  ``correctness_ratified_rate_w30 = 0.750`` (no reroute fires);
  W31 with online learning observes per-cell failures on CYCLIC,
  drops the prior to 0.6667, and the clipped-median adaptive
  threshold (0.8) fires the W30 reroute on subsequent CYCLIC
  cells — strictly improving correctness to 0.875,
  ``Δ(W31 − W30) = +0.125`` across 5/5 seeds at trust precision
  1.000. **Empirically discharges W30-C-PRIOR-LEARNING.** On
  **R-78-ADAPTIVE-THRESHOLD vs R-78-FROZEN-THRESHOLD** the
  adaptive contribution isolates at +0.125 across 5/5 seeds
  (frozen yields 0). On **R-78-MANIFEST-TAMPER** the manifest
  CID + cross-cell prior_trajectory_cid check together yield
  **65/65 = 1.000 tamper rejection rate** across five named
  tampers per cell-position. Three named falsifiers all
  empirically confirmed (W31-Λ-trivial-online ⇒ byte-for-W30
  passthrough; W31-Λ-no-drift ⇒ no help on stationary regime;
  W31-Λ-frozen-threshold ⇒ no adaptive contribution at fixed
  0.5 threshold). **First measured live cross-architecture LLM
  disagreement at temperature 0 in the programme** (28th
  milestone): gemma2:9b on localhost vs qwen2.5:14b on
  192.168.12.191 disagree on **2/8 = 0.250 of structured-
  decision prompts at temp 0**, reproducible byte-for-byte
  across two runs. **Sharpens W30-C-CROSS-HOST-VARIANCE-LIVE-
  MAGNITUDE-LIVE on the infrastructure-discharge axis.** Newly
  named open conjectures **W31-C-CROSS-HOST-VARIANCE-LIVE-
  MAGNITUDE-LIVE** (gold-correlation axis),
  **W31-C-NATIVE-LATENT** (architecture-dependent — true
  transformer-internal subspace projection vs the W31 audited
  proxy; retained as the next true wall),
  **W31-C-MULTI-HOST** (3+ host topology, hardware-bounded),
  **W31-C-LONG-WINDOW-CONVERGENCE** (longer trajectory windows).
  41/41 W31 unit tests + 437/437 phase69-78 regression + 68/68
  wider wevra suite pass. Mac 2 (192.168.12.248) still ARP-
  incomplete (26th milestone). (R-78, SDK v3.32)
- **W32 family:** long-window convergent online geometry-aware
  dense control + EWMA-tracked per-partition prior + Page two-
  sided CUSUM change-point detector + GoldCorrelationMap +
  manifest-v2 CID. The first capsule-native multi-agent-
  coordination method that demonstrates *long-window scaling
  stability* of the W31 online prior on a regime where the W31
  baseline has no convergence guarantee. The W32
  ``LongWindowConvergentOrchestrator`` wraps the W31 orchestrator
  and adds three closed-form primitives: an EWMA running update
  ``ewma_new = (1−α)·ewma_prev + α·observation`` written *back*
  into the W30 calibration vector (load-bearing routing change,
  not just an audit trail); a Page two-sided CUSUM detector
  ``cusum_pos / cusum_neg`` bounded by ``cusum_max`` with named
  change-point firings; and a registered
  ``GoldCorrelationMap : (partition_id → score)`` table. The
  manifest-v2 CID seals all of the above into a single SHA-256
  over four component CIDs (``w31_online_cid``,
  ``convergence_state_cid``, ``gold_correlation_cid``,
  ``route_audit_cid_v2``); the verifier
  :func:`verify_long_window_convergent_ratification` enumerates
  14 failure modes disjoint from W29's 14 + W30's 14 + W31's 14
  (cumulative **56 across W29+W30+W31+W32**). On
  R-79-LONG-WINDOW (4 windows × 5 seeds, all 0.969 byte-equal
  to W31 baseline), W32 is byte-for-W31 stable across long
  windows — *long-window scaling stability* is empirically
  established. **Empirically discharges
  W31-C-LONG-WINDOW-CONVERGENCE on the scaling-stability axis.**
  On R-79-MANIFEST-V2-TAMPER, 1525/1525 = 1.000 cross-component
  manifest-v2 + cross-cell route-audit-v2 tamper rejection
  across five named tampers per cell-position. A pre-committed
  hard gate H6 (strict gain ≥ +0.10 over W31 on a long-window
  regime) is **honestly null**: the **W32-L-CYCLE-CAP**
  limitation theorem proves that on cycle-capped dispatcher
  regimes ``Δ_max ≤ min(c_p/4, c_s)/N ≤ 0.0625``, so the bar
  *cannot* be cleared on the available bench by any
  EWMA-on-prior method. The remaining gap is converted to the
  open conjecture **W32-C-LONG-WINDOW-STRICT-GAIN** (requires a
  regime exceeding the cycle cap). Four named falsifiers
  (W32-Λ-trivial-long-window ⇒ byte-for-W31 passthrough;
  W32-Λ-no-change-point ⇒ stable-history regime never fires
  CUSUM; W32-Λ-frozen-ewma at ``α = 1.0`` *outperforms* W31 by
  +0.016 on the available regime, an honest empirical
  correction over the predicted-null falsifier;
  W32-Λ-mis-correlated-gold ⇒ gate-bounded, never opens on
  synthetic banks). Live cross-architecture LLM gold-verifiable
  pilot (gemma2:9b on localhost vs qwen2.5:14b on
  192.168.12.191, temp 0, 20 prompts, byte-reproducible across
  two runs): **19/20 = 0.950 agreement**, sole disagreement
  D5 (TCP handshake) has neither host correct against gold —
  the first measured live cross-architecture LLM gold-verifiable
  agreement at temp 0 in the programme. **Sharpens
  W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE on the
  prompt-class-dependent disagreement frontier.** Newly named
  open conjectures **W32-C-LONG-WINDOW-STRICT-GAIN**,
  **W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE**,
  **W32-C-NATIVE-LATENT** (architecture-dependent — true
  transformer-internal subspace projection vs the W32 audited
  proxy; out of capsule-layer scope), **W32-C-MULTI-HOST**
  (3+ host topology, hardware-bounded), and
  **W32-C-OLD-LINE-EWMA-TRUST** (W21 EWMA-tracked-trust
  integration; primitives ship in W32 but the W21 integration
  is not yet built). 45/45 W32 unit tests + 414/414 phase69-79
  regression + 77/77 wider wevra suite = 536 tests pass. Mac 2
  (192.168.12.248) still ARP-incomplete (27th milestone).
  (R-79, SDK v3.33)

The post-W21 efficiency-and-coordination ladder (W22 → W31)
discharges, in order, a different family of open conjectures —
one per layer — concerning amortisation of the working
capsule-native pipeline across cells, agents, salience
signatures, host topologies, geometric partitions, and prior
calibration. The W32 layer is the first one of the post-W21
ladder that adds *no new structurally-distinct routing move*:
W32 reuses the W31 routing surface and proves *long-window
scaling stability* of W31's online prior under EWMA + CUSUM
+ gold-correlation, sealed by a manifest-v2 CID. Strict gain
on the same regime is honestly null; the remaining wall is
the **W32-L-CYCLE-CAP** limitation theorem. The natural
escapes from W32-L-CYCLE-CAP are *named*
(W32-C-LONG-WINDOW-STRICT-GAIN on regimes exceeding the cycle
cap; W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE on regimes where
LLMs disagree on gold-verifiable prompts) but currently
*conjectural*; the W32 release deliberately does not claim
strict gain where the regime cannot support it.

## Appendix C. Submission Pass Still Needed

This Markdown manuscript is now substantially beyond a skeleton, but a
submission-quality camera-ready pass would still need production work
of the normal kind:

1. a BibTeX-backed bibliography and in-text citation layer;
2. a figure set with final captions and cross-references;
3. compact venue-shaped tables distilled from the full benchmark
   artifacts;
4. theorem statements rewritten into the exact style of the target
   venue;
5. a reproducibility appendix listing exact scripts, flags, seeds, and
   artifact paths;
6. a pruning pass that moves some implementation detail from the main
   body into appendices.

Those are paper-production tasks, not missing scientific content.

## Appendix D. Recommended figure and table set

The current manuscript would benefit most from the following figures
and tables:

1. **Figure 1: capsule-native runtime overview.**
   Show the path from setup to sealed artifacts to the LLM byte
   boundary.
2. **Figure 2: structural-axis ladder.**
   Show the eight axes and the benchmark that isolates each one.
3. **Figure 3: benchmark ladder timeline.**
   Present R-53 through R-64 as a cumulative sequence of ceilings and
   crossings.
4. **Figure 4: W14-W15-W16-W17 composition diagram.**
   Show producer protocol, normalization, admission, decoding, packing,
   and the bounded decoder input bundle.
5. **Figure 5: symmetric wall.**
   Show why gold and decoy become indistinguishable under the current
   feature surface.
6. **Table 1: regime summary.**
   Keep the ladder summary from Section 7.
7. **Table 2: strongest result by layer.**
   Keep the synthesis table from Section 10.
8. **Table 3: live real-LLM composition results.**
   Present the v3.18 qwen2.5 and qwen3.5 comparisons compactly.

## References

Bibliography intentionally omitted from this Markdown draft.
Replace this section in the final submission version with a real
BibTeX-backed bibliography after the venue-targeting pass.
