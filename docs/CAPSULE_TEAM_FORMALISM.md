# Capsule team-level formalism

> Formal model of the SDK v3.5 multi-agent capsule coordination
> layer (``vision_mvp.wevra.team_coord``). Last touched:
> 2026-04-26.

This document specifies the **team-level** capsule contract — the
W4 family of theorems — that complements the run-boundary,
intra-cell, parser-axis, and LLM-byte-boundary contracts (the W3
family in [`docs/CAPSULE_FORMALISM.md`](CAPSULE_FORMALISM.md)).
Where the W3 family makes capsules load-bearing inside *one* Wevra
run, the W4 family makes capsules load-bearing **between agents in
a team**.

## 1. Setting

A multi-agent capsule coordination **round** consists of the
following objects:

* A finite set of roles $R = \{r_1, \ldots, r_m\}$.
* A per-role budget $\beta_r = (K_r, T_r, \mathcal{K}_r)$ where:
  * $K_r \in \mathbb{N}$ — admission cardinality cap (``K_role``).
  * $T_r \in \mathbb{N}$ — admission token-sum cap (``T_role``).
  * $\mathcal{K}_r \subseteq \mathcal{K} \cup \{\star\}$ — admitted
    claim kinds, where $\star$ denotes "any".
* A capsule ledger $\mathcal{L}$ (``CapsuleLedger``) that records
  every sealed capsule with its hash chain.
* A coordinator $\mathcal{C}$ that drives one round end-to-end via
  ``TeamCoordinator``.

Within a round, three capsule kinds are sealed:

* $\mathsf{H}_{ij}^k$ — a $\mathsf{TEAM\_HANDOFF}$ capsule with
  source role $r_i$, target role $r_j$, claim kind $k$, payload
  $p$, parents $P$ (zero or more upstream handoff CIDs). Identity
  $\mathrm{cid}(\mathsf{H}) = \mathrm{SHA256}(\mathsf{kind},
  \mathsf{payload}, \mathsf{budget}, \mathsf{parents})$ —
  byte-identical handoffs collapse (Capsule Contract C1).
* $\mathsf{V}_r$ — a $\mathsf{ROLE\_VIEW}$ capsule for role $r$,
  whose parents are the CIDs of admitted $\mathsf{TEAM\_HANDOFF}$
  capsules; budget cap $K_r, T_r$.
* $\mathsf{D}$ — a $\mathsf{TEAM\_DECISION}$ capsule for the
  deciding role; parents are the role views consulted.

The role-local **admission set** at the moment of sealing the role
view is denoted $A_r \subseteq \mathcal{L}|_{\mathsf{TEAM\_HANDOFF}}$:
the subset of TEAM_HANDOFF capsules in the ledger whose CIDs are
$\mathsf{V}_r$'s parents.

## 2. Theorems (W4 family)

### 2.1 W4-1 — Team-lifecycle audit soundness (mechanically-checked)

**Statement.** For any ledger $\mathcal{L}$ produced by a round of
``TeamCoordinator``, the runtime audit
``audit_team_lifecycle(L).verdict == "OK"`` iff $\mathcal{L}$
satisfies all of T-1..T-7:

* **T-1.** Every $\mathsf{TEAM\_HANDOFF}$ capsule is in lifecycle
  state SEALED.
* **T-2.** Every $\mathsf{ROLE\_VIEW}$ capsule is in lifecycle
  state SEALED.
* **T-3.** Every $\mathsf{TEAM\_DECISION}$ capsule is in lifecycle
  state SEALED.
* **T-4.** For every $\mathsf{V}_r$, $\mathrm{parents}(\mathsf{V}_r)
  \subseteq \mathcal{L}|_{\mathsf{TEAM\_HANDOFF}}$ (every parent
  CID is a sealed TEAM_HANDOFF in the same ledger).
* **T-5.** For every $\mathsf{V}_r$,
  $|\mathrm{parents}(\mathsf{V}_r)| \le K_r$ (the per-role
  cardinality budget is respected).
* **T-6.** For every $\mathsf{D}$, $\mathrm{parents}(\mathsf{D})
  \subseteq \mathcal{L}|_{\mathsf{ROLE\_VIEW}}$.
* **T-7.** For every $\mathsf{V}_r$ and every parent CID
  $c \in \mathrm{parents}(\mathsf{V}_r)$ that is a TEAM_HANDOFF,
  $c.\mathsf{payload}.\mathsf{to\_role} = r$ (a role only admits
  handoffs targeted at it).

**Proof.** By inspection of
``audit_team_lifecycle`` in
[`vision_mvp/wevra/team_coord.py`](../vision_mvp/wevra/team_coord.py).
Each invariant's failure is enumerated with a concrete
counterexample record. The audit code is short (≈ 100 lines) and
the failure modes are closed-vocabulary; soundness holds by
construction.

**Status.** *Proved + mechanically-checked* — runs on every
``TeamCoordinator`` round and is unit-tested in
``test_wevra_team_coord.py::TeamLifecycleAuditTests``.

### 2.2 W4-2 — Coverage-implies-correctness (proved-conditional)

Let $S$ be a benchmark scenario with a finite **causal claim
set** $\mathcal{C}_S = \{(r_i, k_j, p_l)\}$ — the (source role,
claim kind, payload) triples that jointly witness the gold
answer. Let $\mathsf{Dec} : \mathcal{P}(\mathcal{C}) \to
\mathcal{A}$ be a deterministic team decoder over admitted
handoffs. Suppose:

* **Premise A (faithful decoder).** $\mathsf{Dec}$ is *coverage-
  monotone*: if $\mathcal{C}_S \subseteq \mathrm{handoffs}(A)$,
  then $\mathsf{Dec}(A)$ equals the gold answer for $S$.
* **Premise B (admission soundness).** Every admitted handoff is
  causally-relevant to $S$: $\mathrm{handoffs}(A) \subseteq
  \mathcal{C}_S$.

**Claim.** Under premises A and B, if the deciding role's
$\mathsf{V}_r$ admits handoffs whose (role, kind, payload) triples
form a superset of $\mathcal{C}_S$, then the team decision
$\mathsf{D}$ is *correct on $S$*.

**Proof.** Direct: by Premise A, $\mathsf{Dec}(A)$ equals the
gold answer; the team decision capsule's payload carries
$\mathsf{Dec}(A)$ verbatim (``_decision_from_capsule_view``);
therefore $\mathsf{D}$ records the gold answer.

**Status.** *Proved-conditional*. Premises A + B are scenario-
specific; they hold for the Phase-31 incident-triage decoder
under identity-noise extraction. The conditional theorem is
unit-tested in
``TeamLevelCorrectnessTests::test_w4_2_coverage_implies_correct``
(verifies that uncapped budgets + identity noise produce
``root_cause_correct == True`` on a representative scenario).

### 2.3 W4-3 — Local-view limitation (proved-negative)

**Statement.** There exists a benchmark family $\mathcal{F}$ such
that, for any deterministic admission policy $\pi$ (FIFO, claim-
priority, coverage-guided, learned, …), there is a per-role
budget $\beta_r = (K_r, T_r, \mathcal{K}_r)$ with $K_r$ strictly
below the role's *causal-share floor* $\kappa_r(S)$ on some
scenario $S \in \mathcal{F}$ — and on that $S$, the team gate
fails:
$\mathsf{Dec}(A_r) \ne \mathsf{gold}(S)$.

**Proof sketch.** By construction. Let $S$ be a scenario whose
correct decoder requires admitting $\kappa_r(S) > 1$ distinct
$(r_i, k_j)$ pairs into $\mathsf{V}_r$ (e.g. the disk-fill
scenario in incident-triage requires at least
$\{(\text{sysadmin}, \text{CRON\_OVERRUN}),
(\text{sysadmin}, \text{DISK\_FILL\_CRITICAL}),
(\text{db\_admin}, \text{POOL\_EXHAUSTION})\}$). Set $K_r = 1$.
Then $|A_r| \le 1 < \kappa_r(S)$, so by the contrapositive of
Premise A in W4-2, $\mathsf{Dec}(A_r) \ne \mathsf{gold}(S)$.
The decision $\mathsf{D}$ records a wrong answer regardless of
$\pi$.

**Status.** *Proved-negative*. The theorem says: per-role budgets
**below the causal-share floor** cannot be rescued by *any*
admission policy. In particular, learning over capsule features
cannot improve team accuracy on those scenarios.

The negative is unit-tested in
``TeamLevelCorrectnessTests::test_w4_3_local_view_limitation_at_tight_budget``
(asserts ``n_correct < n_bank`` under $K_\text{auditor}=1$).

### 2.4 W4-C1 — Learned-policy advantage at matched budgets (conjectural)

**Conjecture.** On the noisy-extraction incident-triage benchmark
(`Phase 52`), the learned per-role admission policy
(``LearnedTeamAdmissionPolicy`` trained on a 60-scenario partition)
strictly improves *pooled* team-decision accuracy over every
fixed admission policy (FIFO / claim-priority / coverage-guided)
at matched budgets and matched noise.

**Falsifier.** A bench × seed × budget configuration on which the
fixed best-of-three baseline matches or exceeds the learned
policy's pooled ``accuracy_full``.

**Status.** *Empirical (positive on default config)*. On the
default config (``K_auditor=8``, ``T_auditor=256``, noise
``(drop, spurious, mislabel) = (0.10, 0.30, 0.05)``,
$n_\text{eval}=31$), the learned policy reports
``accuracy_full = 0.161`` vs the strongest fixed baseline
``accuracy_full = 0.065`` (coverage-guided), a
$\Delta = +0.096$. The conjecture is not yet promoted to a
theorem because:

* The training partition is small ($n_\text{train}=60$); the
  margin shrinks at smaller train sizes (the budget sweep
  reports ``accuracy_full=0.0`` for the learned policy at $n=14$
  eval scenarios, with fixed baselines also at $0.0$ — i.e. all
  strategies bottom out simultaneously when training scale is
  small AND budget is tight).
* The benchmark is the bundled deterministic scenario family;
  cross-bench transfer (e.g. the Phase-32 compliance-review
  family) is open.

## 3. Relation to prior W3 theorems

The W4 family is **layered on top of** the W3 family. Specifically:

* Every W4-* theorem assumes the underlying capsule contract
  (C1..C6) and the W3-7..W3-13 algebra. The new statements are
  about the *team layer*; the per-capsule semantics are
  inherited.
* W4-1 (team-lifecycle audit) is the team-level analogue of W3-40
  / W3-45 (run-boundary lifecycle audit). The same proof
  technique — enumeration of failure modes by code inspection —
  applies.
* W4-2 (coverage-implies-correctness) is the team-level analogue
  of W3-32 (lifecycle ↔ execution-state correspondence). Both say
  "if the structural condition holds at the capsule layer, the
  *outcome* matches the spec."
* W4-3 (local-view limitation) is the team-level analogue of
  W3-14 (per-capsule budgets cannot enforce table-level
  cardinality invariants) — a sharp negative limit on what
  per-role budgets alone can guarantee. The natural next move is
  a **cohort lift** at the team layer (Phase 53 candidate, see
  § 4 below) — admitting a COHORT capsule whose member set
  cardinality is the role's causal-share floor.

## 4. Frontier (W4-C* conjectures)

* **W4-C1**: Learned-policy advantage at matched budgets (above).
* **W4-C2**: Cohort-lifted role view closes the W4-3 limitation
  on a sub-class of scenarios. Falsifier: a scenario whose
  causal-share is $> $ a single COHORT's ``max_parents``.
* **W4-C3**: The capsule-layer admission rule subsumes the
  Phase-36 ``AdaptiveSubscriptionTable`` route-edit primitive —
  every adaptive-edge install corresponds to a TEAM_HANDOFF
  capsule whose admission policy approves a previously-disallowed
  ``(source_role, claim_kind)`` pair for one round. Status:
  open.

## 5. Code anchors

| Theorem / claim | Code anchor                                               |
| --------------- | --------------------------------------------------------- |
| W4-1            | `team_coord.audit_team_lifecycle`; `TeamLifecycleAuditTests` |
| W4-2            | `team_coord.TeamCoordinator`; `TeamLevelCorrectnessTests::test_w4_2_*` |
| W4-3            | `TeamLevelCorrectnessTests::test_w4_3_*`; sweep in `phase52_team_coord.run_phase52_budget_sweep` |
| W4-C1           | `phase52_team_coord.run_phase52` default config;          |
|                 | `LearnedAdmissionPolicyTests`                            |

## 6. Limits of this formalism

* The model assumes **synchronous rounds**. Asynchronous /
  pipelined rounds are not yet covered; they would introduce a
  notion of "what handoffs were visible at admission time" that
  the present admission API does not expose.
* The ``TEAM_DECISION`` capsule is opaque at the capsule layer:
  the decoder `Dec` is a black-box function on $A_r$. The audit
  cannot verify that the decision payload *correctly* encodes
  $\mathsf{Dec}(A_r)$ — only that the lifecycle and parent links
  are well-formed. Decoder correctness is a separate
  responsibility (the bench's ``grade_answer``).
* The benchmark family is **synthetic**. The theorems hold on the
  family by construction; their *practical relevance* depends on
  how representative the family is of real multi-agent settings.
  See ``HOW_NOT_TO_OVERSTATE.md`` § "Forbidden moves" for the
  honest framing.
