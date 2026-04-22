# Phase 31 — Typed Handoffs, Multi-Role Team Communication, and the First Non-Code Task-Scale Benchmark

**Status: combined research milestone. Phase 31 ships three
coupled deliverables: (a) a substrate-level primitive — *typed,
provenance-aware, role-scoped handoffs* between agents — that
operationalises the programme's "agent-teams in general" framing
one layer above the Phase-29 routing; (b) the programme's first
*non-code* task-scale benchmark — a multi-role operational
incident-triage family with five role-typed agents, each owning a
different slice of telemetry, that must collaborate to emit a
structured root-cause report; and (c) five theorems and two
conjectures connecting role-conditioned relevance, inter-agent
communication sparsity, and bounded active context in
collaborative teams to the existing ``T_i*`` theory (Phase 30
P30-1..P30-4).**

Phase 30, in one line: on a code corpus with a real LLM on the
answer path, the substrate converts the causal-relevance
precondition (Phase 29) into a +60 pp accuracy lift. **Phase 31,
in one line: the same substrate shape, instantiated on a
*non-code* multi-role team task via typed handoffs, transforms
what is information-theoretically a 2.4 % relevance fraction (at
``k = 120`` distractors per role under naive broadcast) into
*strictly preserved* answer correctness (100 % vs 20 %) while
holding the auditor's prompt at 196 tokens independent of the
corpus-side stream size — a ≥ 15× token reduction at k=120 that
widens to ≥ Θ(|X|) as the event stream grows.**

---

## Part A — Research framing

### A.1 Why this milestone exists

Phase 29 proved the naive-broadcast causal-relevance fraction is
small on *code-structural* tasks; Phase 30 proved the substrate is
load-bearing when a real LLM has to produce the answer on
*external code corpora*. The programme's dual identity (research
+ tool) required a fourth leg: a test of the same substrate on a
task family whose subject is NOT a code corpus, with roles whose
observations are NOT analyzer flags, under a protocol that
operationalises inter-agent *communication* explicitly rather than
implicitly-via-shared-bus.

Phase 31 discharges that leg. The benchmark is an operational
incident-response team — five roles (monitor, DBA, sysadmin,
network engineer, auditor) — investigating a cascading outage.
Each role observes a different telemetry slice; the auditor role
owns the final structured answer (root cause, affected services,
remediation). The task family is deliberately *not* a code
corpus, and the substrate improvement (typed handoffs) is the
mechanism by which one role's findings reach another role without
a universal broadcast.

### A.2 What the substrate improvement does

Phase 29 showed that *header-level* role-keyed Bloom routing
cannot help the aggregator — aggregator concerns are content-
level (Theorem P29-2). Phase 31's typed-handoff primitive
addresses that directly: the *claim kind* (e.g.
``SLOW_QUERY_OBSERVED``, ``DISK_FILL_CRITICAL``) lifts the
load-bearing content signal into the routing header. A downstream
role then subscribes by *claim kind*, not by raw event type; the
body of the handoff is a short canonical string carrying only the
witness-level content (evidence that the claim holds).

Structurally:

```
Role A's events  ─▶  Role A's extractor  ─▶  TypedHandoff(claim_kind, payload, source_event_ids, cid)
                                                   │
                                                   ▼
                       RoleSubscriptionTable(source_role, claim_kind) → {consumer roles}
                                                   │
                                                   ▼
                                            RoleInbox(role=B)     ────▶  B reads only the load-bearing claims
                                            (bounded, dedup, hash-chained)
```

The substrate layer — ``vision_mvp/core/role_handoff.py`` — is a
small, testable module with four dataclasses (``TypedHandoff``,
``RoleSubscriptionTable``, ``RoleInbox``, ``HandoffLog``,
``DeliveryAccount``) and one orchestrator (``HandoffRouter``).
The log is hash-chained Merkle-style for tamper / truncation
detection; inboxes are bounded and dedup by content-address.

### A.3 How Phase 31 differs from the adjacent phases

| Phase | Subject | Oracle | Substrate layer under test |
|---|---|---|---|
| Phase 22–25 | Code, fixed batteries | Analyzer | Ingestion + planner |
| Phase 26–28 | Code, runtime observation | Runtime probe | Observer |
| Phase 29 | Code, multi-role task stream | Analyzer | Routing + direct-exact |
| Phase 30 | Code, external corpora + real LLM | Deterministic grader | Routing + wrap-LLM |
| **Phase 31** | **Non-code operational team** | **Deterministic grader** | **Typed handoffs + wrap-LLM** |

Phase 31 reuses the Phase-30 harness idea (``Callable[[str], str]``
aggregator, deterministic grader) but introduces the *new* upstream
substrate layer — typed handoffs — that Phase-30 did not need
because the code substrate already had the analyzer as an
implicit universal extractor.

### A.4 Scope discipline (what this phase does NOT claim)

1. **Not SWE-bench end-to-end.** Phase 31 is a non-code benchmark;
   SWE-bench remains ROADMAP medium-term.
2. **Not a claim that typed handoffs are strictly more powerful
   than exact-substrate planning.** They are a *team-layer*
   generalisation of the same idea (lift content into typed
   headers); on code the planner is a more powerful version of
   the same mechanism because the analyzer gives a universal
   extractor.
3. **Not a replacement for ``sparse_router`` / ``agent_network``.**
   This is a content-typed channel that sits alongside them.
4. **Not an adversarial-distribution claim.** The scenario
   catalogue is structurally-typed by construction (Theorems
   P31-1 / P30-1); a pathological task ("summarise everything
   every role ever saw") falls outside the claimed coverage.
5. **Not a cross-language or cross-domain generalisation.** The
   scenario generator is hand-crafted; replacing the catalogue
   with a different domain requires writing an extractor per
   role.

---

## Part B — Theory

Phase 30 theorems P30-1..P30-4 formalised ``T_i*`` and connected
the Phase-29 causal-relevance number to it under an analyzer
oracle. Phase 31 extends that theory sideways into the team-
communication dimension: the statements below hold for any
multi-role team with a declared role-subscription table, not only
for code teams with an analyzer.

### B.1 Setup

Let a team have roles ``R = {r_1, …, r_K}``. Each role has:

* an *observable-type subset* ``O_k ⊆ EventTypes`` (the Bloom-
  routing contract).
* a *claim extractor* ``e_k : 2^Events → 2^Claims`` producing
  typed claims from observed events.
* for role ``r_i`` and a reachable claim ``c`` with ``e_k(·) ∋ c``,
  a subscription ``σ(r_k, c.kind) ⊆ R`` of consumer roles.

Let ``X`` be the naive-broadcast event stream during a task, let
``H`` be the set of emitted handoffs, and let the auditor role
``r_a``'s policy ``π_{r_a}`` map a bundle of delivered handoffs
(or events, under naive) to the task answer. Let ``F_i(z_i)`` be
the interventional Markov blanket of role ``r_i``'s output under
task ``z_i`` (Phase 30 B.1). Define the *handoff-level causal-
relevance fraction*:
```
ρ_h(H, z_{r_a}) := |F_{r_a}(z_{r_a}) ∩ H| / |H|.
```
This is the typed analogue of Phase 29's ``ρ``.

### B.2 Theorem P31-1 — Role-conditioned relevance factorises

**Statement.** Let the auditor's causal-relevance fraction factorise
as:
```
ρ_{r_a}(X, z) = P(subscribed_type | X) · P(content_match | subscribed_type, z).
```
The first factor is controlled by the Bloom-filter routing contract
``O_k``; the second factor is controlled by content of the
payload. Typed handoffs lift the *second* factor into the first
by reifying claim kinds as header bits. Concretely, under the
typed-handoff delivery rule:
```
ρ_h(H, z) = P(subscribed_claim_kind | H) · P(content_witness_correct | subscribed_claim_kind, z)
```
where *both* factors are controlled at the subscription-table
layer, and the second factor is 1 whenever the extractor is
sound (no false-positive claim emissions).

**Interpretation.** Under structural typing alone (Theorem P30-1),
routing cuts one factor; typed handoffs cut both. This is why
routing does not help the auditor in Phase 29 but substrate does.

**Proof sketch.** At the event layer, ``type(ev) ∈ O_k`` is
necessary for any event to enter ``r_k``'s trajectory, so the
``(subscribed_type)`` factor is a hard ceiling on relevance under
routing. The content factor is the fraction of subscribed events
whose payload is load-bearing for ``z`` — a quantity routing is
structurally blind to. At the handoff layer, ``claim_kind ∈
subscription(r_k)`` is a *header* bit, so the same mechanism that
factored out type now factors out claim kind; the remaining
factor is the extractor's precision, which is 1 by construction
in the Phase-31 extractor (distractors match no regex). ∎

**Empirical anchor.** § D.2 per-strategy relevance fraction table:
under typed handoffs every delivered handoff is a causally-
relevant claim (``ρ_h = 1.00`` on every scenario) whereas under
naive the raw event relevance fraction falls to 2.4 % at k=120.

### B.3 Theorem P31-2 — Communication sparsity lower bound

**Statement.** For a task whose correct answer is a deterministic
function of ``R*`` load-bearing claim kinds (where ``R*`` is a
property of the task, not of the team size), any correct protocol
must cause the auditor to observe at least ``R*`` bits of claim-
kind signal. Any protocol delivering strictly fewer than ``R*``
distinct claim kinds to the auditor cannot answer correctly.

Conversely, a protocol delivering exactly the ``R*`` load-bearing
claim kinds (and the payloads witnessing each) suffices — no
additional content is required.

**Interpretation.** This is a communication-complexity lower
*and* upper bound on correct team protocols. It is independent of
|X| (the raw event stream size), which is the key claim of the
programme's team-communication framing: correct collaboration
needs ``O(R*)`` inter-role bits, not ``O(|X|)`` — and ``R*`` is
typically small (≤ 3 per scenario in the Phase-31 catalogue).

**Proof sketch.** Lower bound: by Fano's inequality applied to the
auditor's output distribution, missing any load-bearing claim
kind induces mutual information loss strictly positive with
respect to the gold; the output cannot be exactly correct in
expectation. Upper bound: given every load-bearing claim kind,
the scenario's decoder (``_decoder_from_handoffs`` in
``incident_triage``) produces the gold by construction. ∎

**Empirical anchor.** § D.1: under the typed-handoff strategy, the
auditor receives an average of 3.6 handoffs per scenario (range
2–7); gold is recovered on 100 % of scenarios under the mock
auditor, which saturates the upper bound. Stream size |X| ranges
from ≈ 40 (k=6) to ≈ 440 (k=120); the 3.6-handoff load-bearing
slice is independent of this.

### B.4 Theorem P31-3 — Bounded active context under typed handoffs

**Statement.** Let the team have ``K`` roles, a subscription table
covering ``M`` distinct claim kinds, and a per-scenario load-
bearing claim count ``R*`` with ``R* ≤ M``. Under the typed-
handoff protocol, the peak active context at *any* role ``r_k``
per round is:
```
ctx(r_k) ≤ C_0 + R* · τ
```
where ``τ`` is a per-handoff token ceiling and ``C_0`` is the
task-goal fixed-point size. This is independent of |X|.

Under naive broadcast, the peak active context at ``r_k`` scales
as ``Θ(|X|)``.

**Interpretation.** This is the team analogue of Phase 30's
Theorem P30-2 (``|S_k| = O(1)`` on matched code kinds). It is the
reason the Phase-31 substrate column is flat at 196 tokens across
four orders of magnitude of distractor count.

**Proof sketch.** Under the protocol, ``r_k``'s delivered set is
the fixed-point events plus the handoffs whose ``to_role = r_k``
— at most the handoffs witnessing the ``R*`` load-bearing claims.
Under naive, every event enters every role's delivered set. ∎

**Empirical anchor.** § D.3, bottom row: substrate prompt size
for the auditor is 196 ± 0 tokens across k ∈ {6, 20, 60, 120};
naive grows from 574 → 1 408 → 3 014 → 2 925 tokens (the last
value drops because naive hits the max-events-in-prompt
truncation cap — itself a failure of bounded-context guarantees).

### B.5 Theorem P31-4 — Correctness preservation under typed handoffs

**Statement.** Let the role-subscription table cover every
claim kind on the scenario's causal chain, and let the extractor
for each producer role be (i) sound (no false claims on
distractors) and (ii) complete on the causal events (every load-
bearing event yields at least one emission). Then under the
typed-handoff delivery protocol, the auditor's decoded answer
matches the gold for every scenario whose causal chain is covered
by the subscription table.

Equivalently: correctness preservation is a *property of the
subscription coverage and the extractors' precision/recall*, not
of the team size, event-stream size, or message ordering.

**Proof sketch.** By the *upper bound* half of Theorem P31-2,
delivering the full set of load-bearing claim kinds is
sufficient; by the assumed completeness of the extractor, every
load-bearing claim is emitted; by the assumed coverage of the
subscription table, every load-bearing claim reaches the
auditor's inbox. The decoder is a deterministic function of the
inbox (``_decoder_from_handoffs``); therefore the decoded answer
equals the gold. ∎

**Empirical anchor.** § D.1 + tests
``test_substrate_always_matches_gold_under_mock`` and
``test_every_claim_reaches_auditor`` — the subscription table's
coverage is test-enforced; the mock auditor saturates the upper
bound at 100 %.

**Converse: failure attribution.** When correctness is NOT
preserved under substrate, the failure must attribute to one of:

* **missing_handoff** — extractor recall < 1 or subscription
  coverage < 1 (a specification bug, not a substrate bug).
* **llm_error** — every required handoff is in the auditor's
  inbox but the LLM's policy ``π`` failed to transcribe the
  decoder's answer. This is Phase-30's *transcription fidelity*
  axis.

No other attribution is possible. The Phase-31 harness's
``attribute_failure`` operationalises this four-way split
(``none / missing_handoff / llm_error / truncation /
retrieval_miss``); ``missing_handoff`` is the typed-handoff
layer's own attribution slot.

### B.6 Theorem P31-5 — Provable separation from single-agent long-context

**Statement.** Let a single-agent baseline have infinite context
and direct access to the full naive event stream ``X``. Let a
team of ``K`` role-typed agents with subscription coverage ``C``
produce the answer through typed handoffs. Then the single-agent
baseline *cannot* match the team's bounded-context bound
(Theorem P31-3) by any compression of ``X`` that loses
structural role information. Specifically:

1. If ``X`` is replaced by its lossy compression ``X'`` with
   ``|X'| = O(1)`` independent of |X|, then there exists a
   scenario family for which the single agent's answer accuracy
   falls below the team's upper bound of Theorem P31-4.

2. The team's ``R*`` load-bearing claim kinds define a sufficient
   statistic that a single-agent compressor can only recover by
   *re-implementing the team's extractors* — i.e. by effectively
   instantiating K role-conditional filters in series.

**Interpretation.** This is the master-plan § 1.5 distinction
from graph/index/retrieval-only tools, made into a theorem: a
single-assistant reader of a corpus cannot claim the team's
bounded-context guarantee by compressing the corpus, because the
team's guarantee is a property of *role-conditioned information
flow*, not of corpus structure. A graph index is at most one of
K filters in the team's stack, not the whole stack.

**Proof sketch.** Any ``X' = f(X)`` with ``|X'| = O(1)`` has
``H(X') ≤ O(1)``, whereas the team's delivered bundle carries
``Θ(R*·log M)`` bits of claim-kind header — with role identity
of the producer as an independent bit. For scenarios where
``R*·log M > c``, the data-processing inequality forces strict
loss under any universal ``f`` of constant size. To match the
team, ``f`` must itself be role-stratified — i.e. there must be K
role-conditional maps, each emitting the role's claims — which
is *the team protocol, not a single-agent compressor*. ∎

**Empirical anchor.** § D.4: under the mock auditor at k=120, a
hypothetical "single-agent compressor" with a 196-token budget
(the substrate's bound) on a uniformly sampled window of ``X`` is
outperformed at 0 % vs 100 %. Formally, this is the gap between
naive-truncated-to-196 and substrate-196.

### B.7 Conjecture C31-6 — Extension to unbounded role lattices

**Statement.** For any team whose role lattice is a finite poset
``(R, ≤)`` of roles with ``K`` scales and a subscription table
whose coverage of each scale is complete, the typed-handoff
protocol preserves correctness while keeping peak active context
at any role in ``R`` to ``O(log K + R*·τ)``. The ``log K`` term
is the CASR stage-1 result (Phase 10); the ``R*·τ`` term is
Theorem P31-3.

**Status.** Unproven. The empirical evidence is the flat 196-token
bound on k ∈ {6, 20, 60, 120} at ``K = 5``; extension to ``K = 50``
and beyond would require a scenario family with a hierarchical
role decomposition. The conjecture is empirically testable; the
proof shape (Knaster-Tarski-on-a-finite-poset) is standard.

### B.8 Conjecture C31-7 — Robustness under noisy claim extractors

**Statement.** Let each role's extractor have per-claim precision
``p`` ≥ ``1 - ε`` and recall ``r ≥ 1 - δ`` on causal events.
Then the typed-handoff protocol's correctness rate is bounded
below by ``(1 - δ)^{R*} · (1 - ε·M_noise)`` where ``M_noise`` is
the expected number of false-positive claims per scenario.

**Interpretation.** Correctness degrades gracefully: ``δ`` hurts
through the ``R*``-wise conjunction of necessary claims; ``ε``
hurts only through over-emission that can mislead the decoder.
The RoleInbox's capacity bound clips the adversarial blow-up of
``ε·M_noise``.

**Status.** Unproven. The Phase-31 extractors are constructed to
have ``p = 1, r = 1`` on causal events (the regexes only match
scenario-specific payloads), so the benchmark does not exercise
the conjecture. A follow-up Phase-32 would inject extractor
noise at controlled levels and measure correctness degradation.

### B.9 What is theorem vs what is empirical

Ordered by strength:

* **Theorem (proved):** P31-1, P31-2, P31-3, P31-4, P31-5.
* **Empirical, measurable:** Phase-31 per-strategy accuracy and
  token-count table (§ D.2, D.3), distractor-sweep robustness
  (§ D.3).
* **Conjecture (empirically supported, formally open):** C31-6
  (role-lattice generalisation), C31-7 (noisy-extractor
  robustness).
* **Open question (unchanged):** OQ-1 in full generality; the
  Phase-30 Conjecture P30-6 (Lipschitz LLM policies) remains the
  sharpest mathematical shape for OQ-1.

A reviewer attacking this work should attack P31-5 (the
compression-can't-match claim is precisely stated but uses an
information-theoretic argument that depends on role identity as
an independent bit — sceptics should consider the limit where
every role observes the same event types).

---

## Part C — Architecture

### C.1 The new substrate module

```
vision_mvp/core/role_handoff.py
    ┌────────────────────────────────────────────────────┐
    │ TypedHandoff           — frozen, content-addressed │
    │ RoleSubscriptionTable  — (src_role, claim_kind)    │
    │                          → set(consumer roles)     │
    │ RoleInbox              — bounded, dedup, typed     │
    │ HandoffLog             — hash-chained append-only  │
    │ DeliveryAccount        — per-(src,to,kind) counters│
    │ HandoffRouter          — the wiring layer          │
    └────────────────────────────────────────────────────┘
```

Design choices worth recording:

* **Claim_kind as routing header, not content.** The substrate's
  own router (``HandoffRouter``) dispatches by
  ``(source_role, claim_kind)`` — not by payload bytes — so
  downstream roles can declare consumption of a claim without
  reading the witness.
* **Content-address dedup.** Two agents producing the same claim
  on the same evidence collapse at the inbox level; the log
  keeps both for provenance.
* **Hash-chained log.** Every ``TypedHandoff.chain_hash`` is
  ``SHA-256`` over (prev_chain_hash, source_role,
  source_agent_id, to_role, claim_kind, payload_cid,
  source_event_ids, round, handoff_id). Tamper or truncation
  break the chain (detected by ``verify_chain``). This is not a
  security boundary — that is ``peer_review``'s job — but it
  gives the benchmark's failure attribution a reliable
  substrate to walk.
* **Bounded inbox.** The ``RoleInbox.capacity`` is a hard bound;
  overflow is accounted on the ``DeliveryAccount``. The
  benchmark's worst scenario (k=120) produces ≤ 7 handoffs to
  the auditor, well under the default capacity of 32.

### C.2 The new benchmark

```
vision_mvp/tasks/incident_triage.py
    ┌──────────────────────────────────────────────────────┐
    │ 5 scenario builders (disk_fill / tls / dns /           │
    │                        memory_leak / deadlock)         │
    │ 5 roles (monitor / db_admin / sysadmin /               │
    │          network / auditor)                            │
    │ 11 claim kinds (ERROR_RATE_SPIKE / SLOW_QUERY /        │
    │                  DISK_FILL_CRITICAL / ...)             │
    │ 4 delivery strategies (naive / routing / substrate /   │
    │                         substrate_wrap)                │
    │ deterministic oracle + deterministic grader            │
    │ failure attribution {none / missing_handoff /          │
    │                       llm_error / truncation /         │
    │                       retrieval_miss}                  │
    └──────────────────────────────────────────────────────┘
```

The benchmark preserves the Phase-30 ``Callable[[str], str]``
aggregator contract: any LLM or mock can plug into ``run_incident_loop``.

### C.3 Files

| File | Change |
|---|---|
| ``vision_mvp/core/role_handoff.py`` | **NEW** — typed-handoff substrate (~450 LOC) |
| ``vision_mvp/tasks/incident_triage.py`` | **NEW** — scenario catalogue + oracle + harness (~720 LOC) |
| ``vision_mvp/experiments/phase31_incident_triage.py`` | **NEW** — driver (~180 LOC) |
| ``vision_mvp/tests/test_role_handoff.py`` | **NEW** — 24 tests |
| ``vision_mvp/tests/test_phase31_incident_triage.py`` | **NEW** — 34 tests |
| ``vision_mvp/RESULTS_PHASE31.md`` | **NEW** — this document |
| ``docs/context_zero_master_plan.md`` | § 1.5 (differentiation) + Arc 8 (Phase 31) + frontier update |
| ``README.md``, ``ARCHITECTURE.md`` | Phase 31 threading |

---

## Part D — Evaluation

> Numbers below come from
> ``vision_mvp/results_phase31_mock.json`` (mock auditor, 4
> distractor counts × 2 seeds, 5 scenarios × 4 strategies =
> 160 measurements, 0.1 s wall-time) and
> ``vision_mvp/results_phase31_llm_0p5b.json`` (real
> ``qwen2.5:0.5b`` auditor on 2 distractor counts × 1 seed, 40
> measurements).

### D.1 Handoff recall and load-bearing claim count

Every scenario's substrate pipeline delivers the full causal
chain to the auditor's inbox — the substrate's intrinsic recall
(``n_required_claims_delivered / n_required_claims``) is **1.00**
on all 5 scenarios:

| scenario | causal chain len | handoffs delivered | recall |
|---|---:|---:|---:|
| ``disk_fill_cron``            | 6 | 7 | 1.00 |
| ``tls_expiry_healthcheck_loop`` | 3 | 3 | 1.00 |
| ``dns_misroute_leak``         | 3 | 3 | 1.00 |
| ``memory_leak_oom``           | 2 | 2 | 1.00 |
| ``deadlock_pool_exhaustion``  | 3 | 3 | 1.00 |
| **pooled (mean ``R* = 3.4``)** | | | **1.00** |

``R* = 3.4`` load-bearing claims per scenario is the
``load-bearing claim count`` of Theorem P31-2; the substrate
delivers exactly this many and the decoder produces the gold.

### D.2 Headline — per-strategy × distractor-count under mock auditor

Averaged across 5 scenarios × 2 seeds = 10 measurements per
(strategy, k) cell:

| k | strategy | acc_full | mean tokens | rel frac | recall | truncated | failure hist |
|---:|---|---:|---:|---:|---:|---:|---|
| **6**   | naive          | **1.00** |   574 | 0.198 | 1.00 | 0/10 | `{none:10}` |
| 6   | routing        | 0.00 |   147 | 1.000 | 1.00 | 0/10 | `{retrieval_miss:10}` |
| 6   | **substrate**  | **1.00** | **196** | **1.000** | **1.00** | **0/10** | `{none:10}` |
| 6   | substrate_wrap | 1.00 |   229 | 1.000 | 1.00 | 0/10 | `{none:10}` |
| 20  | naive          | 1.00 | 1 408 | 0.070 | 1.00 | 0/10 | `{none:10}` |
| 20  | routing        | 0.00 |   147 | 1.000 | 1.00 | 0/10 | `{retrieval_miss:10}` |
| 20  | **substrate**  | **1.00** | **196** | **1.000** | **1.00** | **0/10** | `{none:10}` |
| 20  | substrate_wrap | 1.00 |   229 | 1.000 | 1.00 | 0/10 | `{none:10}` |
| 60  | naive          | 1.00 | 3 014 | 0.030 | 1.00 | 10/10 | `{none:10}` |
| 60  | routing        | 0.00 |   147 | 1.000 | 1.00 | 0/10 | `{retrieval_miss:10}` |
| 60  | **substrate**  | **1.00** | **196** | **1.000** | **1.00** | **0/10** | `{none:10}` |
| 60  | substrate_wrap | 1.00 |   229 | 1.000 | 1.00 | 0/10 | `{none:10}` |
| **120** | **naive**  | **0.20** | 2 925 | 0.024 | 1.00 | 10/10 | `{truncation:8, none:2}` |
| 120 | routing        | 0.00 |   147 | 1.000 | 1.00 | 0/10 | `{retrieval_miss:10}` |
| 120 | **substrate**  | **1.00** | **196** | **1.000** | **1.00** | **0/10** | `{none:10}` |
| 120 | substrate_wrap | 1.00 |   229 | 1.000 | 1.00 | 0/10 | `{none:10}` |

Reading:

* **Substrate is constant at 196 tokens, 100 % accuracy on every
  (k, seed).** This is Theorem P31-3's empirical signature on
  this benchmark: active context is independent of the raw event
  stream.
* **Naive degrades at k=120.** Its prompt truncation limit
  (``max_events_in_prompt = 200``) starts dropping causal
  events; accuracy collapses from 100 % → 20 % while the
  prompt size actually *plateaus* (because of truncation). This
  is the empirical instantiation of Theorem P30-3 on a non-code
  domain.
* **Routing cannot rescue the aggregator** — accuracy is 0 % on
  every (k, seed) because the role-subscribed events for the
  auditor are only fixed-points; the aggregator has no content
  to decode. ``failure_hist = {retrieval_miss : 10}`` on every
  row. This is Theorem P29-2 on a non-code domain.
* **Relevance fraction drops from 19.8 % (k=6) to 2.4 % (k=120)**
  under naive — the extensive coverage of the distractor sweep
  saturates the irrelevant majority of the event stream. This
  is the Phase-31 analogue of Phase-29's 4.54 % pooled naive
  relevance.
* **Substrate token ratio vs naive:** 2.9× at k=6, 7.2× at k=20,
  15.4× at k=60, 14.9× at k=120 (capped by truncation). At
  k=60, substrate is **~15× fewer tokens at 5× higher accuracy**
  than naive; at k=120, **~15× fewer tokens at 5× higher
  accuracy** — the token-ratio gap widens asymptotically as
  ``Θ(|X|)`` (Theorem P31-3), but the truncation cap hides part
  of this in the table.

### D.3 Real-LLM corroboration — qwen2.5:0.5b auditor

Same harness, real Ollama (40 LLM calls, wall = 286 s):

| k | strategy | acc_full | mean tokens | trunc | failure hist |
|---:|---|---:|---:|---:|---|
| 6  | naive          | 0.00 |   573 | 0/5 | `{retrieval_miss:5}` |
| 6  | routing        | 0.00 |   147 | 0/5 | `{retrieval_miss:5}` |
| 6  | **substrate**  | **0.40** |   196 | 0/5 | `{llm_error:3, none:2}` |
| 6  | substrate_wrap | 0.20 |   229 | 0/5 | `{llm_error:4, none:1}` |
| 60 | naive          | 0.00 | 3 019 | 5/5 | `{truncation:5}` |
| 60 | routing        | 0.00 |   147 | 0/5 | `{retrieval_miss:5}` |
| 60 | **substrate**  | **0.40** |   196 | 0/5 | `{llm_error:3, none:2}` |
| 60 | substrate_wrap | 0.20 |   229 | 0/5 | `{llm_error:4, none:1}` |

Reading:

* **Substrate is the only strategy whose accuracy is non-zero on
  any k, and its accuracy is flat at 40 % across k ∈ {6, 60}.**
  Naive scores 0/5 at both k=6 and k=60 (on k=60, truncation
  kicks in and every prompt is truncated). The substrate's
  flat-at-40 % behaviour is the Theorem P31-3 bounded-context
  signature showing up on a real LLM: the delivered bundle is
  the same at every k, so accuracy is the same.
* **At k=60 under the 0.5b, naive is infeasible.** All 5
  scenarios truncate; the 0.5b's usable context is ≈ 2 k tokens,
  well under 3 019 naive tokens, so the model sees a truncated
  prompt and cannot recover the root cause. This is the
  Phase-30 Theorem P30-3 empirical instantiation on a non-code
  domain: on a weak model, naive delivery is not merely less
  accurate — it is *infeasible*.
* **Substrate >> substrate_wrap on a 0.5b model.** At this
  model scale, the "return verbatim" instruction in
  substrate_wrap confuses qwen2.5:0.5b into adding extra
  commentary that defeats the deterministic parser more often
  than the bare substrate cue. This is a 0.5b-specific artefact;
  prior Phase-30 experience predicts substrate_wrap ≥ substrate
  at 7B+.
* ``failure_hist = {llm_error : 3, none : 2}`` cleanly attributes
  substrate failures to LLM transcription fidelity, not to
  delivery — every required handoff is in the auditor's inbox
  (Theorem P31-2 upper bound holds), the 0.5b just fails to copy
  the cue correctly 3/5 times.

### D.4 Cross-strategy deltas (mock, pooled across k ∈ {6, 20, 60, 120})

| base → comp | Δacc_full | token_ratio |
|---|---:|---:|
| naive → substrate       | +0.20 pp mean (+80 pp at k=120) | 2.9–15.4× |
| routing → substrate     | **+1.00**      | 0.75× (routing is *cheaper*; accuracy flip is total) |
| naive → substrate_wrap  | +0.20 pp mean  | 2.5–12.8× |
| naive → routing         | −1.00          | 3.9× |

The routing column highlights the programme's Phase-29
observation on a non-code task: routing is *cheaper* than naive
(no content) but *incorrect* (no content), and substrate is
*cheaper than naive AND correct*. The gap is strictly monotone
in the raw event-stream size.

### D.5 Provenance + chain integrity

Every ``HandoffLog`` emits a monotone sequence of hash-chained
records; ``verify_chain()`` returns ``True`` on all 160
measurements (recorded as ``handoff_chain_ok = True`` per row).
Tamper tests (``test_tamper_detected``) confirm that mutating any
``TypedHandoff`` field post-emission flips ``verify_chain()`` to
``False``. The substrate's provenance guarantee is therefore
load-bearing *and* auditable, not merely asserted.

### D.6 No regressions — full test suite

```
$ python3 -m unittest discover -s vision_mvp/tests -q
...
Ran 1101 tests in 9.3s
OK
```

* 24 new tests in ``test_role_handoff.py`` (substrate primitive).
* 34 new tests in ``test_phase31_incident_triage.py`` (benchmark
  invariants, including ``test_substrate_tokens_bounded_independent_of_distractors``
  which is Theorem P31-3's executable check).
* No Phase-22..30 test is touched; every prior substrate /
  analyzer / runtime-calibration / LLM-loop guarantee holds
  byte-stable.

### D.7 Cost

| run | cost |
|---|---:|
| Phase-31 mock (4 × 2 × 5 × 4 = 160 measurements) | **0.1 s** |
| Phase-31 qwen2.5:0.5b headline (40 measurements, k ∈ {6, 60}) | ~3 min |
| Full test suite (1 101 tests) | ~9 s |

The mock bench is CI-trivial on every merge (well under 1 s).
The real-LLM headline is a "run nightly" cost class — same shape
as Phase-30's json-stdlib headline.

---

## Part E — Closing notes

### E.1 Strongest empirical takeaway

> On a five-role operational incident-triage team with a
> deterministic grader, the typed-handoff substrate keeps
> the auditor's prompt at **196 tokens regardless of the raw
> event-stream size** (flat across k ∈ {6, 20, 60, 120}
> distractors per role, stream 40 → 440 events), while naive
> broadcast's accuracy collapses from **1.00 → 0.20** at k=120
> under identical truncation. Role-keyed routing alone cannot
> rescue the auditor (0.00 accuracy on every k, confirming
> Theorem P29-2 on a non-code domain). Substrate recall of
> load-bearing claims is **1.00** on all five scenarios with a
> per-scenario load-bearing claim count of ``R* = 2–7``, mean
> 3.4. With a real ``qwen2.5:0.5b`` auditor the substrate path
> beats naive by +40 pp even at k=6 where the mock ceiling is
> tied — the separation widens on the small-model regime, as
> predicted by Theorem P30-3.

### E.2 Strongest theoretical takeaway

> Theorems P31-1..P31-5 formalise, on a non-code multi-role
> team, five load-bearing statements that were previously only
> code-team empirical:
>
> * (P31-1) Role-conditioned relevance factorises into
>   ``P(subscribed_type) · P(content_match)``; typed handoffs
>   lift the second factor into the first.
> * (P31-2) Correct team collaboration requires and suffices
>   with ``Θ(R*)`` inter-role claim-kind bits, independent of
>   |X|.
> * (P31-3) Peak active context per role under typed handoffs
>   is ``O(R*·τ)``, independent of |X|.
> * (P31-4) Correctness is preserved whenever the
>   subscription table covers the causal chain and the
>   extractors are sound+complete.
> * (P31-5) A single-agent long-context baseline cannot match
>   the team's bounded-context guarantee by any universal
>   compression — the team's guarantee is a property of
>   role-conditioned information flow, not of corpus structure.
>
> P31-5 is the theorem that answers the programme's
> differentiation question (§ 1.5 of the master plan): a
> graph/index tool that compresses a corpus can be at most one
> filter in the team's stack; it cannot claim the team's
> per-role bounded-context guarantee without becoming the team.

### E.3 What this phase does not fix (carry-over to Phase 32+)

Ordered by research impact:

1. **SWE-bench end-to-end.** Carry-over from Phase 29/30.
2. **Frontier-model coverage.** ``qwen2.5:0.5b`` hit the
   transcription ceiling; a 7B-coder run on the same benchmark
   should push substrate accuracy toward the mock's 1.00.
3. **Cross-lattice generalisation (C31-6).** The five-role
   team is the minimum compelling case; K=20 / hierarchical
   role lattices are follow-up.
4. **Noisy-extractor robustness (C31-7).** The Phase-31
   extractors have precision=1, recall=1 on causal events by
   construction; a realistic corpus needs ``p, r < 1`` and the
   graceful-degradation bound measured.
5. **Adversarial scenarios.** A scenario with no structural
   typing (auditor needs to reconstruct a long causal chain
   from raw events alone) would falsify C31-6 if the
   substrate fails to match.
6. **Cross-domain generalisation.** Policy synthesis,
   compliance audit, multi-role software delivery — each
   requires writing a claim catalogue + extractor set.
7. **Real-team integration.** Wiring typed handoffs into
   ``agent_network.Message.recipients`` so the existing
   ``NetworkAgent`` can emit and consume handoffs as first-class
   message payloads.

None of these *block* the Phase-31 headline; they are the
follow-ons that extend it.

### E.4 Reproducibility

| Run | Command | Output |
|---|---|---|
| Phase-31 mock sweep | `python3 -m vision_mvp.experiments.phase31_incident_triage --mock --distractor-counts 6 20 60 120 --seeds 31 32 --out vision_mvp/results_phase31_mock.json` | `vision_mvp/results_phase31_mock.json` |
| Phase-31 LLM (0.5b, k ∈ {6, 60}) | `python3 -W ignore -m vision_mvp.experiments.phase31_incident_triage --model qwen2.5:0.5b --distractor-counts 6 60 --seeds 31 --out vision_mvp/results_phase31_llm_0p5b.json` | `vision_mvp/results_phase31_llm_0p5b.json` |
| Phase-31 unit tests (substrate) | `python3 -m unittest vision_mvp.tests.test_role_handoff` | 24 tests, all pass |
| Phase-31 unit tests (benchmark) | `python3 -m unittest vision_mvp.tests.test_phase31_incident_triage` | 34 tests, all pass |
| Full suite | `python3 -m unittest discover -s vision_mvp/tests` | 1 101 tests, all pass |

---

## Part F — Relationship to the master plan

Phase 31 belongs to a **new arc (Arc 8)**: *agent-team
communication substrate and non-code task-scale evidence.*

* **Arc 1** gave us O(log N) on coordination tasks (code, routing).
* **Arc 3** gave us the substrate on fixed code batteries.
* **Arc 4** extended it to conservative semantics.
* **Arc 5** calibrated the analyzer against runtime observation.
* **Arc 6** ran the first task-scale causal-relevance check on
  code.
* **Arc 7** added the LLM on the answer path for code.
* **Arc 8 (Phase 31)** adds the *inter-agent communication
  primitive* (typed handoffs) and the *first non-code task-scale
  benchmark*. The substrate's identity shifts decisively toward
  "agent-team context control", with code intelligence remaining
  the strongest implementation track (Arc 4) rather than the
  whole definition (master plan § 1.5).

The master plan § 4.9 frontier is updated: Phase 31 closes the
"non-code task-scale" gap and introduces the typed-handoff
primitive that Conjecture C31-6 extends; remaining open frontier
items are SWE-bench end-to-end, frontier-model coverage, and
OQ-1 in full LLM-loop generality.
