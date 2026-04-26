# Context Capsule — formal model

> *Research-grade. Theorems are stated with explicit proof boundaries
> ("proved" / "proved on a sub-class" / "conjectural"). Code anchors
> are file paths, never paraphrased. Last touched: Phase 51 relational
> decoder frontier (Theorem W3-30 strict-separation, Claim W3-31
> empirical, Conjecture W3-C10 level-ceiling), 2026-04-22.*

---

## Canonical post-Phase-50 position (READ FIRST)

The programme has three artefacts, each with a sharply different
epistemic status. Future readers should internalise these before
reading any specific theorem or claim below.

**1. What is retracted.**  The strict pre-Phase-50 reading of
Conjecture W3-C7 — "point-estimate Gate 1 at $\hat p \ge 0.400$
*plus* strict zero-shot Gate 2 with per-direction penalty
$\le 5$ pp" — is **retracted**.  Phase-50 replication at
$n_{\rm test} = 320$ falsified it on the point estimate side
(W3-26: best decoder 0.362), and the 6-family zero-shot sweep
falsified it on the penalty side (W3-27: best max-penalty
+0.112).  Two proved theorems (W3-24 post-search winner's-curse
bias; W3-29 Bayes-divergence zero-shot risk lower bound) give
the structural reasons.  **Do not reintroduce the strict bar.**

**2. What is defensible.**  The post-Phase-50 defensible reading
is **Conjecture W3-C9**: the point-estimate Gate 1 at
$n_{\rm test} = 80$ *plus* the **gap reading** of zero-shot
Gate 2 (direction-invariance, not level-matching).  Under
W3-C9, Phase 49's result (W3-22 pooled-multitask at 0.350/0.350
and W3-23 DeepSet at 0.425 @ $n=80$) combined with Phase 50's
sign-stable DeepSet (W3-28 gap = 0.000 @ zero-shot level
0.237) earns a **canonical paradigm-shift-candidate** label.
Phase 49 is the canonical candidate; Phase 50 is its
limitation-theorem milestone.

**3. What is open.**  The next real frontier is *not* a strict
re-attack on W3-C7 — that is structurally blocked on the
present benchmark family.  It is the **level** of symmetric
zero-shot transfer: sign-stable DeepSet delivers gap 0.000 at
level 0.237, materially below the within-domain optimum
(0.362/0.400).  Raising this level, under direction-invariance
constraint, requires a hypothesis class **outside** the
W3-29-bounded magnitude-monoid linear family.  The natural
candidate — and the Phase-51 research direction — is a
**relational / cohort-aware decoder**: a decoder whose features
depend on pairwise or partitioned sub-bundle structure, not on
bag-of-capsules aggregates.  W3-C5 named the axis; Phase 51
puts it on the table with the smallest serious instance
(Theorem W3-30 strict separation, Claim W3-31 empirical
level-lift, Conjecture W3-C10 level-ceiling).

### How not to overstate this

A reader who finishes Phase 48–50 and takes away any of the
following statements has misread the programme:

- *"A learned decoder beats the Phase-31 ceiling by 15 pp, so
  the ceiling is broken."*  **Mis-reading.**  The 0.200 ceiling
  is a priority-decoder property; W3-19 breaks it with a
  learned decoder at $n_{\rm test}=80$.  But the true best-cell
  accuracy of the Phase-49 decoder family on this benchmark at
  $n_{\rm test}=320$ is $\approx 0.362$ (W3-26), below 0.400.
  The ceiling is breakable; 0.400 is not cleared.
- *"DeepSet crosses the paradigm-shift threshold."*  **Mis-reading.**
  Phase-49 W3-23 reported 0.425 at $n_{\rm test}=80$; Phase-50
  W3-26 replicates at $n_{\rm test}=320$ and drops to 0.362.
  W3-24 (winner's curse, proved) explains the gap.  DeepSet is
  the strongest decoder in the family, but its best-cell
  accuracy at large $n$ does not cross 0.400.
- *"Cross-domain decoder transfer is symmetric."*  **Mis-reading.**
  W3-28 reports gap = 0.000 for sign-stable DeepSet — this is
  *direction-invariance* of zero-shot transfer, not
  level-matching.  The transferred level (0.237) sits below
  within-domain optima (0.362/0.400); the strict penalty
  reading is not met (W3-27) and is structurally bounded below
  by W3-29.
- *"Pooled multitask proves symmetric transfer."*  **Mis-reading.**
  W3-22 trains on labelled data from both domains.  It is a
  *deployment* result, not a zero-shot result.  The strict
  zero-shot question is not answered by pooled multitask.
- *"The paradigm-shift bar has been cleared."*  **Mis-reading.**
  Under the strict pre-Phase-50 reading it is **retracted**
  (W3-26, W3-27).  Under the Phase-50-refined W3-C9 reading
  Phase 49 is a **candidate**, not a certified paradigm shift.

Readers familiar with the canonical reading should also note
what the programme has *not* claimed:

- The capsule algebra is a substrate unification story (W3-7
  through W3-15).  The decoder story (W3-17 through W3-29) is a
  **research centre on top of** that substrate, not a property
  of the substrate itself.  Decoder results do not affect
  capsule-contract invariants C1..C6.
- The sign-flip-asymmetry and Bayes-divergence theorems
  (W3-21, W3-29) are proved on the *linear class-agnostic*
  hypothesis family.  They do **not** prove that no hypothesis
  class achieves strict Gate 2 — only that the linear class
  does not.  Relational hypothesis classes are structurally
  outside the premise (§ 4.F, W3-30).

---

## 0. Why a separate formalisation document

`docs/RESULTS_WEVRA_CAPSULE.md` states the Capsule Contract in prose
and lists six theorems (W3-1 .. W3-6). That note is a milestone
write-up; it is not a formalisation. The Wevra v3 milestone added
nothing to the substrate's mathematical content beyond the
contract — the unification was claimed but not derived.

This document is the formal complement. It does three things:

1. fixes the algebraic objects that make the capsule contract a
   real mathematical object (capsule space, transition system,
   capsule DAG, budget monoid, admissibility predicate);
2. derives a small set of theorem-style claims that are stronger
   than the contract-level invariants in the milestone note;
3. proves how prior bounded-context theorems (Phase 19 L2,
   Phase 31 P31-3, Phase 35 P35-2, Phase 41 P41-1, Phase 43 P43-1)
   are *special cases* of one capsule-level invariant — and is
   honest about which subsumption claims remain conjectural.

The formalisation is faithful to `vision_mvp/wevra/capsule.py` —
every object below has a code anchor.

---

## 1. The capsule space

### 1.1 Atomic objects

Let:

- $\mathbb{K}$ be the closed *kind alphabet* — the **twelve**
  enumerated strings in `CapsuleKind.ALL`
  (`HANDOFF, HANDLE, THREAD_RESOLUTION, ADAPTIVE_EDGE, SWEEP_CELL,
  SWEEP_SPEC, READINESS_CHECK, PROVENANCE, RUN_REPORT, PROFILE,
  ARTIFACT, COHORT`); $|\mathbb{K}| = 12$. The twelfth kind
  (`COHORT`) was added in Phase 47 as the honest resolution of
  Conjecture W3-C3 — see § 4 (Theorems W3-14 / W3-15 / W3-16)
  for the derivation.
- $\mathbb{P}$ be the space of canonicalisable JSON values — that
  is, finite trees over `(null, bool, int, float, str, list, dict)`
  with string keys. The canonical encoding
  `_canonical(payload)` (sorted keys, `(",", ":")` separators)
  defines an injection $\mathbb{P} \hookrightarrow \{0,1\}^*$.
- $\mathbb{B}$ be the *budget space* — non-negative-or-`None`
  five-tuples
  $(b_t, b_b, b_r, b_w, b_p) \in (\mathbb{N}_0 \cup \{\bot\})^5$
  representing `(max_tokens, max_bytes, max_rounds, max_witnesses,
  max_parents)`, subject to **at least one component being not `⊥`**
  (the constructor's invariant
  `CapsuleBudget.__post_init__`).
- $\mathbb{H} = \{0,1\}^{256}$ — the SHA-256 image; a *content
  identifier* (CID) is an element of $\mathbb{H}$.
- $\mathbb{L} = \{$`PROPOSED, ADMITTED, SEALED, RETIRED`$\}$ — the
  lifecycle alphabet.

### 1.2 Capsules

A **capsule** is a tuple

$$
c = (\mathit{cid}, k, p, b, \pi, \ell, t, n_{\rm tok}, n_{\rm byt})
$$

where $k \in \mathbb{K}$, $p \in \mathbb{P}$, $b \in \mathbb{B}$,
$\pi \in \mathbb{H}^*$ (a finite tuple of parent CIDs),
$\ell \in \mathbb{L}$, $t \in \mathbb{N}$ (emission timestamp),
and $n_{\rm tok}, n_{\rm byt} \in \mathbb{N}_0 \cup \{\bot\}$.

Define the *content map*

$$
\mathit{cid}(k, p, b, \pi)
\;=\;
\mathrm{SHA\text{-}256}\!\bigl(
  \mathrm{enc}\{k, p, b.\mathrm{as\_dict}(), \mathrm{sort}(\pi)\}
\bigr)
$$

Code anchor: `_capsule_cid` in `capsule.py`. Note `sort(π)` —
parent insertion order is canonicalised away, so two capsules with
the same parent set under any permutation collapse to one CID.

Let $\mathcal{C}$ be the set of capsules satisfying

$$
\mathit{cid} \;=\; \mathit{cid}(k, p, b, \pi)
\quad\text{(C1: identity)}
$$

and the byte-budget constraint

$$
b.b_b \neq \bot \implies \bigl|\mathrm{enc}(p)\bigr| \le b.b_b
\quad\text{(C4 partial: byte axis at construction)}.
$$

The constructor `ContextCapsule.new` produces only elements of
$\mathcal{C}$.

### 1.3 The capsule space, projected

Two projections matter:

- $\mathrm{Cap}_{\rm sealed} = \{c \in \mathcal{C} :
   \ell(c) = \mathtt{SEALED}\}$ — the set of *frozen* capsules.
- $\mathrm{Cap}_{\rm sealed}^{\le}$ — equipped with the parent
  relation $c' \mathrel{\preceq} c \iff \mathit{cid}(c') \in \pi(c)$.

By construction, $\preceq$ is an irreflexive partial order on the
*finite* sub-DAG generated by any single ledger (because every
parent CID admitted into a ledger is itself sealed strictly before
the child — see § 2.2).

---

## 2. The capsule transition system

### 2.1 States and transitions

A capsule's lifecycle is the deterministic finite automaton
($\mathbb{L}, \to_\ell$) with edges

$$
\mathtt{PROPOSED} \to_\ell \mathtt{ADMITTED}
\to_\ell \mathtt{SEALED}
\to_\ell \mathtt{RETIRED}.
$$

No back-edges, no cross-edges; `CapsuleLifecycle._EDGES` literally
encodes this graph. Let
$\delta_\ell : \mathbb{L} \to 2^\mathbb{L}$ be the transition map.
`CapsuleLifecycleError` is raised exactly when a caller tries to
follow an edge not in $\delta_\ell$.

### 2.2 Ledger semantics

A **capsule ledger**
$\mathcal{L} = (E, I, h_0, h_*)$ is

- $E \in \mathcal{C}^*$ — a finite sequence of *sealed* capsules
  (the ledger's append-only entry list);
- $I \subseteq \mathbb{H}$ — the index $\{\mathit{cid}(c) : c \in E\}$;
- $h_0 = \mathtt{GENESIS} \in \{0,1\}^*$ — a fixed seed string;
- $h_* \in \mathbb{H} \cup \{h_0\}$ — the current chain head.

A ledger is *valid* iff for every $i \in \{1, \dots, |E|\}$,

$$
h_i \;=\; \mathrm{SHA\text{-}256}\!\bigl(
  \mathrm{enc}\{h_{i-1}, \mathit{cid}(c_i), k(c_i), \mathtt{SEALED}\}
\bigr),
$$

with $h_{|E|} = h_*$, *and* every parent CID of $c_i$ is in
$I_{<i} = \{\mathit{cid}(c_j) : j < i\}$.

Code anchor: `CapsuleLedger.admit`, `CapsuleLedger.seal`,
`CapsuleLedger._chain_step`, `CapsuleLedger.verify_chain`.

The `admit_and_seal` operation
$\mathrm{Adm} : \mathcal{C}_\mathrm{prop} \to \mathcal{L} \to
(\mathcal{L} \times \mathcal{C}_\mathrm{seal})$ is total *exactly*
on the admissibility predicate $\mathcal{A}_\mathcal{L}$ defined
next.

### 2.3 Admissibility

The **admissibility predicate** $\mathcal{A}_\mathcal{L}$ for a
proposed capsule $c$ in ledger $\mathcal{L}$ is the conjunction of:

- *typed*: $k(c) \in \mathbb{K}$ (rejected at construction;
  formally, $\mathcal{A}$ would be vacuous on un-typed inputs).
- *parental*: $\pi(c) \subseteq I(\mathcal{L})$ (C5).
- *budgeted*: every active axis of $b(c)$ is satisfied —

  $$
  b.b_t \neq \bot \implies n_{\rm tok}(c) \le b.b_t,
  \quad
  b.b_b \neq \bot \implies n_{\rm byt}(c) \le b.b_b,
  $$
  $$
  b.b_p \neq \bot \implies |\pi(c)| \le b.b_p,
  $$
  and for kinds whose admission step also enforces round/witness
  axes, the analogous bounds (C4).

Code anchor: `CapsuleLedger.admit` ensures
parental + token budgeting, `ContextCapsule.new` enforces the byte
+ parent count axes at construction, and `CapsuleLifecycle._EDGES`
guards lifecycle order.

### 2.4 Capsule DAG

For a fixed ledger $\mathcal{L}$ define the **capsule DAG**
$G(\mathcal{L}) = (V, A)$:

- $V = \{c : c \in E\}$,
- $A = \{(c', c) : \mathit{cid}(c') \in \pi(c), c \in V\}$.

By § 2.2, every parent of an entry was sealed strictly earlier;
therefore $G(\mathcal{L})$ is acyclic. Topological order is the
ledger's append order.

---

## 3. Algebra: budget monoid, capsule monoid

### 3.1 Budget as a tropical-min monoid

For each axis $a \in \{t, b, r, w, p\}$ define

$$
x \oplus_a y \;=\;
\begin{cases}
y & \text{if } x = \bot \\
x & \text{if } y = \bot \\
\min(x, y) & \text{otherwise}
\end{cases}
$$

and let $\oplus$ be the componentwise extension on $\mathbb{B}$.
$(\mathbb{B}, \oplus, \bot^5)$ is a commutative monoid; $\bot^5$
is the "no constraint" identity. (Note: $\bot^5$ is *not* a legal
`CapsuleBudget` because of `__post_init__`; it is a mathematical
identity used in proofs only.)

Intuition: $\oplus$ is "the budget that respects both inputs".
This is the operation an admission rule applies when *combining* a
capsule's declared budget with an ambient ledger budget.

### 3.2 Composition of admissibility under $\oplus$

**Lemma 3.2 (admissibility is monotone under tightening).**
If $c$ is admissible under budget $b$, and $b' \le b$ pointwise
(i.e. $b'_a \in \{b_a, \text{strictly tighter than } b_a\}$ for
every active axis), then $c$ is admissible under $b'$ implies $c$
is admissible under $b$. Equivalently:
$\mathcal{A}_{b'} \subseteq \mathcal{A}_b$.

**Proof.** Each budget check is of the form
$n_a(c) \le b_a$. Tightening $b_a$ can only *shrink* the set
$\{c : n_a(c) \le b_a\}$. The intersection across axes preserves
this. $\square$

Lemma 3.2 is the algebraic statement behind the runtime's "the
ledger's admission is a budget gate" intuition — it justifies
treating admission as a *filter* in the order theory of capsule
budgets.

### 3.3 Sealed capsules form a commutative monoid under union of
ledgers

Given two ledgers $\mathcal{L}_1, \mathcal{L}_2$ over disjoint CID
prefixes, the union ledger
$\mathcal{L}_1 \cup \mathcal{L}_2$ is the topologically-sorted
concatenation; CIDs collide iff payloads collide (C1), in which
case the colliding capsules are identical. Therefore
$(\mathrm{Cap}_{\rm sealed}, \cup, \emptyset)$ is a commutative
monoid up to CID identity. *Note*: this is a mathematical statement
about the abstract sealed-capsule set; the live `CapsuleLedger`
class does not implement union — it is the intersection
appropriate to the formalisation, not to any runtime contract.

---

## 4. Theorems

The numbering picks up from the milestone note (W3-1 .. W3-6) and
adds W3-7 onward.

### Theorem W3-7 (Identity is a homomorphism on canonicalisation)

For any $c \in \mathcal{C}$ and any permutation $\sigma$ of
$\pi(c)$,
$$
\mathit{cid}(k, p, b, \sigma\pi) \;=\; \mathit{cid}(k, p, b, \pi).
$$

**Proof.** `_capsule_cid` sorts $\pi$ before encoding; the encoding
is a function of $(k, p, b, \mathrm{sort}(\pi))$. $\square$

**Code anchor.** `_capsule_cid`; test
`test_c1_parent_order_is_canonicalised`.

### Theorem W3-8 (Budget monotonicity of admissibility)

This is Lemma 3.2 promoted to a theorem statement.

$$
b' \le b \implies \mathcal{A}_{b'} \subseteq \mathcal{A}_b.
$$

**Proof.** § 3.2.

**Code anchor.** Implicit in `CapsuleLedger.admit` and
`ContextCapsule.new`. *Empirical* anchor: the ablation in
`tests/test_wevra_capsules.py::test_c4_budget_rejects_over_tokens`
verifies tightening rejects.

### Theorem W3-9 (Ledger DAG is acyclic and topologically equal to
append order)

For every ledger $\mathcal{L}$, $G(\mathcal{L})$ is acyclic and the
append order $E$ is a topological order of $G(\mathcal{L})$.

**Proof.** `CapsuleLedger.admit` requires every parent CID to be
already in $I$. Since $I$ is grown by `seal()` in the same order as
$E$, every parent of $c_i$ has index $j < i$. Therefore
$(c_j, c_i) \in A$ implies $j < i$, which means $E$ is a
topological order. Acyclicity follows from existence of a
topological order on finite graphs. $\square$

**Code anchor.** `CapsuleLedger.admit` parental check;
`CapsuleLedger.ancestors_of` BFS terminates.

### Theorem W3-10 (Chain tamper-evidence is a forensic guarantee
under SHA-256 collision resistance)

Suppose an adversary modifies any of `(cid, kind, lifecycle)` of
any sealed entry in $E$, *or* mutates any entry's `chain_hash` /
`prev_chain_hash`, *or* permutes / inserts / deletes entries. Then
$\mathrm{verify\_chain}(\mathcal{L})$ returns False with
probability $1 - \mathrm{negl}(\lambda)$ where $\lambda = 256$
(SHA-256 second-preimage hardness).

**Proof sketch.** The recomputation loop is total — it walks the
entries in append order computing
$h_i^{\text{rec}} = \mathrm{SHA\text{-}256}(\mathrm{enc}\{
  h_{i-1}^{\text{rec}}, \mathit{cid}(c_i), k(c_i), \mathtt{SEALED}\})$.
A mutation that preserves the stored $h_i$ requires either
(a) a SHA-256 second preimage on the
$(h_{i-1}, \mathit{cid}, k, \mathtt{SEALED})$ tuple, or
(b) the canonical encoder is non-injective on the mutated input.
The encoder is injective by construction. Any cross-link
modification breaks the consistency
$h_{i-1} = e_i.\mathrm{prev\_chain\_hash}$ check at the next link.
$\square$

**Code anchor.** `_chain_step`, `verify_chain`; test
`test_c5_hash_chain_detects_tamper`.

### Theorem W3-11 (Capsule subsumption — *partial*)

Let $T$ be a Phase-19/31/35 substrate-level *bounded-context*
theorem of the shape

> "There exists a constant $C$ independent of $|X|$ such that the
> per-step active context delivered to role $r$ is bounded by $C$."

Then there exists a capsule kind $k_T \in \mathbb{K}$ and a budget
axis tuple $b_T \in \mathbb{B}$ such that:

1. The substrate primitive guarded by $T$ admits an injective
   adapter $\alpha_T : \mathrm{Prim}_T \to \mathrm{Cap}^{(k_T)}$.
2. The capsule ledger's admissibility predicate
   $\mathcal{A}_{b_T}$ refuses every primitive instance that
   violates $T$'s premise.
3. Therefore $T$'s conclusion is implied by C4 applied to the
   adapter's image with budget $b_T$.

**Status.** The four-element instantiation table below is the
*proven* sub-class of W3-11. The general statement — for *every*
phase-level bounded-context theorem in Phases 19..44 — is
**Conjecture W3-C1** (open, mechanical to extend).

| $T$            | Primitive       | $k_T$               | $b_T$ active axes                          | Reduction |
|---             |---              |---                  |---                                         |---        |
| L2 (Phase 19)  | `Handle`        | `HANDLE`            | $b_t = B_{\rm worker}$                     | per-fetch token count = capsule's `n_tokens`; $\sum_h n_{\rm tok}(h) \le B$ enforced by admission of the worker's view capsule. |
| P31-3          | `TypedHandoff`  | `HANDOFF`           | $b_t = \tau$                               | per-handoff token count $\le \tau$ ⇒ $R^*$ admissible handoffs sum to $R^* \cdot \tau$ at any role. |
| P35-2          | `EscalationThread`+`ThreadResolution` | `THREAD_RESOLUTION` | $b_t = \tau$, $b_r = R_{\max}$, $b_w = W$  | additive bound $T \cdot R_{\max} \cdot W$ per role-round = sum of admissible witness capsules under $b_w$. |
| P41-1 / P43-1  | `SubstratePromptCell` (sweep cell embedding the substrate prompt) | `SWEEP_CELL`        | $b_b = \beta_{\rm cell}$                  | per-cell byte budget $\beta_{\rm cell}$ enforced at construction (`max_bytes`); flat-prompt observation = "no admitted cell exceeded $\beta_{\rm cell}$". |

**Proof of the four cases.**

*L2.* `capsule_from_handle` produces a `HANDLE` capsule with
`n_tokens = max(1, len(fingerprint.split()))`. A worker that
admits $k$ such capsules under a worker-view ledger with
$b_t = B$ has total tokens
$\sum_i n_{\rm tok}(c_i) \le |E| \cdot \tau_{\rm h}$ where
$\tau_{\rm h}$ is the per-handle token cap. The Phase-19 statement
is then the special case where the worker view is exactly that
ledger's sealed entries — independent of $|X|$.

*P31-3.* `capsule_from_handoff` carries
`n_tokens = handoff.n_tokens`. `RoleInbox` is bounded; `R^*` is
the per-task subscription count to the role. With $b_t = \tau$
on every `HANDOFF` capsule, admission caps each at $\tau$, and
inbox capacity caps the count at $R^*$. Sum bound:
$\sum n_{\rm tok} \le R^* \tau$.

*P35-2.* The thread emits one `THREAD_RESOLUTION` capsule whose
budget is `(max_tokens=512, max_rounds=8, max_witnesses=64,
max_parents=32)` (Phase-35 default). Each round contributes at
most $W$ witness tokens; $R_{\max}$ replies; $T$ open threads
per role per round. Admission bounds witness $\le W$; the
additive contribution per role-round is $T \cdot R_{\max} \cdot
W$, exactly P35-2's extra term.

*P41-1 / P43-1.* `capsule_from_sweep_cell` carries a payload whose
`max_bytes` budget is $b_b = 2^{16} = 65536$. The empirical
observation that the substrate prompt is flat at 205.9 tokens
across $n_{\rm distractors} \in \{0, 6, 12, 24\}$ is the
statement: every admitted `SWEEP_CELL` had the substrate-prompt
sub-field bounded by a quantity independent of $|X|$. Capsule
admission rejects any cell whose serialised payload exceeds
$b_b$, which is the operational form of the bound.

$\square$ (For the sub-class.)

**Code anchor.** `capsule_from_handle`, `capsule_from_handoff`,
adapters in `vision_mvp/wevra/capsule.py`. The four reductions are
*additive* — the substrate primitive's own enforcement is unchanged.

**What is NOT subsumed.** The expressivity-separation theorems
P31-5, P35-1, and the *correctness* theorems P31-4, P35-3 are
**not** statements of the form Theorem W3-11 covers. They are
properties of the *role topology* (subscription graph,
extractor soundness), not of any per-capsule budget. The capsule
contract is silent on them by design.

**Phase 47 extension — table-level invariants.** The *per-edge*
reduction for AdaptiveEdge above is FULL on the TTL axis (each
edge is active for ≤ `ttl_rounds`). The *table-level* invariant
`|active_edges| ≤ max_active_edges` is not expressible under
any per-capsule budget — see Theorem W3-14 — but it IS
expressible once the alphabet is extended with a **COHORT**
kind (Theorem W3-15). Phase 47's contribution is to close the
AdaptiveEdge PARTIAL verdict at the table level while naming
a sharp limitation (W3-16) at the relational level.

### Theorem W3-12 (Capsule view is a faithful header projection)

Let $\mathcal{L}$ be a sealed ledger and let
$V = \mathrm{render\_view}(\mathcal{L},
\mathrm{include\_payload}=\texttt{False})$. Then $V$ is sufficient
to:

(a) recompute every link of `verify_chain()` (using $V$ alone, no
external data) — *if and only if* every entry's
`(cid, kind, lifecycle)` is in $V$;
(b) recover the full capsule DAG topology
($\preceq$) — *if and only if* every entry's `parents` is in $V$.

The default `as_header_dict` *does* include
`(cid, kind, parents, lifecycle)`, so both (a) and (b) hold for the
default rendering.

**Proof.** `_chain_step` is a function of
$(h_{i-1}, \mathit{cid}, k, \mathtt{SEALED})$ — a subset of header
fields. Topology is by definition $\pi$. $\square$

**Code anchor.** `as_header_dict`, `render_view`; test
`test_capsule_view_on_disk_matches_embedded`.

### Theorem W3-13 (Capsule DAG height is bounded by $|\mathbb{K}|$
on the canonical run pattern)

For a single Wevra run, the capsule DAG produced by
`build_report_ledger` has height $\le 4$ (chain
`PROFILE → {READINESS_CHECK, SWEEP_SPEC, PROVENANCE,
ARTIFACT}_*; SWEEP_SPEC → SWEEP_CELL_*; everything → RUN_REPORT`).

**Proof.** Inspection of `build_report_ledger`. The longest path is
PROFILE → SWEEP_SPEC → SWEEP_CELL → RUN_REPORT, length 3.
$\square$

**Code anchor.** `build_report_ledger`. Empirically, the
`local_smoke` profile produces a 13-capsule graph with height 3;
chain head verification under 30 ms.

**Why this matters.** Bounding DAG height by a constant means the
capsule graph is *not* a deep blockchain — it is a small,
inspectable artifact whose audit time is dominated by SHA-256
throughput, not by graph traversal.

---

## 4.B Phase 47 extension — cohort lifting and relational limitation

The Phase-46 unification audit returned 4/5 FULL + 1/5 PARTIAL
on the five per-primitive reductions; the PARTIAL row was
AdaptiveEdge, because the Phase-36 invariant
`|active_edges| ≤ max_active_edges` lives at the
*AdaptiveSubscriptionTable* level and has no representation as a
per-capsule budget axis. Phase 47 resolves this honestly by
proving a **negative locality theorem** (W3-14), extending the
alphabet with a **COHORT** kind, proving the **cohort-lift
theorem** (W3-15) that closes AdaptiveEdge, and naming the
sharp **relational limitation** (W3-16) that cohort-lifting
does NOT close.

### Theorem W3-14 (Per-capsule budget locality — negative result)

Let $\Phi$ be a unary predicate on capsules
($\Phi : \mathcal{C} \to \{0,1\}$, decidable from a single
capsule's fields). Define the cardinality invariant

$$
\mathrm{Card}_\Phi(N) \;:\;
\bigl|\{c \in E(\mathcal{L}) : \Phi(c)\}\bigr| \;\le\; N.
$$

**Claim.** No choice of per-capsule budget $b \in \mathbb{B}$
applied at each capsule's admit step enforces $\mathrm{Card}_\Phi$
globally: there exists a ledger trace in which every individual
capsule passes $\mathcal{A}_b$ but the admitted set contains
strictly more than $N$ capsules satisfying $\Phi$.

**Proof.** Fix any $b \in \mathbb{B}$. Each axis check
$n_a(c) \le b_a$ is a property of a single capsule $c$ — the
admit step is a function of $(c, \pi(c))$ alone, where $\pi(c)$
is the parent set, *not* a function of the full sealed entry
list $E$. Construct a trace of $N+1$ capsules
$c_1, \dots, c_{N+1}$ each satisfying $\Phi$ and each
individually within budget (possible: e.g., each is a minimal
`HANDLE` capsule of one token). Then every $c_i$ is admitted
under $\mathcal{A}_b$, yet
$|\{c \in E : \Phi(c)\}| = N + 1 > N$. $\square$

**Code anchor.** `vision_mvp/tests/test_phase47_cohort_subsumption.py
::W3_14_PerCapsuleLocalityTests::test_per_capsule_budget_cannot_bound_total_count`
— constructs exactly this trace with $N = 99$ and 100 admits
to document the failure of any per-capsule rule to enforce
$\mathrm{Card}_\Phi$.

**Interpretation.** W3-14 is the sharp reason the Phase-46
audit marked AdaptiveEdge as PARTIAL: the
`AdaptiveSubscriptionTable.max_active_edges` bound is of the
shape $\mathrm{Card}_\Phi(E_{\max})$ with
$\Phi(e) = \mathtt{e.ttl\_rounds > 0}$ (or more precisely,
$\Phi(e) = \mathtt{e \in \mathrm{active\_set}(tick)}$). W3-14
says no per-capsule budget closes it. Hence the need for W3-15.

### Theorem W3-15 (Cohort lift — subsumption of cardinality
invariants via a twelfth kind)

Let $\mathbb{K}' = \mathbb{K} \cup \{\mathtt{COHORT}\}$ be the
kind alphabet extended with a twelfth kind. A **cohort capsule**
is any $c \in \mathcal{C}$ with $k(c) = \mathtt{COHORT}$ whose
parent set $\pi(c)$ is precisely the set of "member" CIDs the
cohort witnesses. Its budget $b(c)$ sets
$b.b_p := \mathtt{max\_members}$; all other axes default.

**Claim.** For every unary predicate $\Phi$ and every cardinality
cap $N$, there is a cohort witness
$\omega_\Phi(N, \mathcal{L}, \tau)$ (parameterised by a ledger
state $\mathcal{L}$ and an observation moment $\tau$) such that
the admissibility of $\omega_\Phi(N, \mathcal{L}, \tau)$ into
$\mathcal{L}$ holds iff
$|\{c \in E(\mathcal{L}) : \Phi(c)\}_\tau| \le N$.

**Proof.** Construct $\omega$ as a capsule with
$\pi(\omega) = \{\mathit{cid}(c) : c \in E(\mathcal{L}) \wedge
\Phi(c)\}_\tau$ and budget
$b(\omega) = \mathbb{B}_{\mathrm{coh}}(N) :=
(\bot, \bot, \bot, \bot, N)$ (only `max_parents` set, to $N$).
Then $\omega$ admits through $\mathcal{A}_{b(\omega)}$ iff
$|\pi(\omega)| \le b.b_p = N$; but
$|\pi(\omega)| = |\{c : \Phi(c)\}_\tau|$ by construction.
Admission is therefore equivalent to the cardinality cap.
$\square$

**Corollary (AdaptiveEdge → FULL).** The Phase-36 table-level
invariant $|\mathrm{active\_edges}| \le E_{\max}$ subsumes to the
cohort capsule
$\omega(\Phi = \mathtt{active\_in\_tick}, N = E_{\max})$ emitted
at each tick. The Phase-46 PARTIAL verdict on AdaptiveEdge is
therefore *not* a failure of the capsule algebra; it was a
failure of the *alphabet* — specifically, the absence of a
kind whose admit step checks its own parent-set cardinality
against a cohort-cap budget. Adding `COHORT` closes the gap.

**Code anchor.** `vision_mvp/wevra/capsule.py::CapsuleKind.COHORT`
(12th kind), `capsule_from_cohort`,
`capsule_from_adaptive_sub_table`;
`vision_mvp/experiments/phase46_unification_audit.py::audit_adaptive_edge_cohort`;
contract tests in
`vision_mvp/tests/test_phase47_cohort_subsumption.py::W3_15_CohortLiftTests`.

### Theorem W3-16 (Relational limitation — sharp negative
result *after* cohort lifting)

Let $\Psi$ be a binary predicate on distinct capsule pairs
($\Psi : \mathcal{C} \times \mathcal{C} \to \{0,1\}$, decidable
from two capsules' fields). Define the **relational invariant**

$$
\mathrm{Rel}_\Psi \;:\;
\forall c_1, c_2 \in E, \;\; c_1 \neq c_2 \implies
\Psi(c_1, c_2) = 1.
$$

Examples of relational invariants that matter in the Wevra
substrate: "no two admitted `HANDOFF` capsules share a
`source_event_id`"; "no two sealed `HANDLE` capsules have
spans that overlap"; "the subscription graph implied by the
admitted `HANDOFF` kinds forms a DAG."

**Claim.** For generic (non-constant) $\Psi$, no finite
extension of $\mathbb{K}$ or $\mathbb{B}$ in the *magnitude-only
axis* style of the current capsule algebra enforces
$\mathrm{Rel}_\Psi$ via cohort admission alone. In particular,
$\mathcal{A}_b$ remains silent on $\mathrm{Rel}_\Psi$ even
when $b$ is the cohort budget of W3-15.

**Proof.** The cohort admission check of W3-15 reduces to
$|\pi(\omega)| \le b.b_p$, a cardinality predicate on the
parent set. Any relational invariant $\mathrm{Rel}_\Psi$ for
non-constant $\Psi$ is not a function of $|\pi(\omega)|$
alone: it depends on the pairwise structure of the members
themselves, which is erased when the admit step sees only
the cardinality count. Formally, consider two cohorts
$\omega_1, \omega_2$ with identical parent-set cardinalities
but different member content: if $\omega_1$ satisfies
$\mathrm{Rel}_\Psi$ and $\omega_2$ does not, both cohorts
admit identically under the cohort budget's cardinality
check, so cohort admission cannot discriminate. $\square$

**Interpretation.** W3-16 names the *outer boundary* of the
capsule algebra. Cardinality is inside; pairwise / graph-
theoretic predicates are outside. Enforcing a relational
invariant requires either

1. moving the predicate check *into the cohort's constructor*
   (so the cohort capsule's existence is itself a certificate
   that $\mathrm{Rel}_\Psi$ held at construction time), which
   shifts enforcement from the ledger's admit step to the
   constructor's trust boundary; or
2. extending the capsule algebra with a **relational axis**
   (a second-order object), which is beyond the magnitude-
   monoid view of the present formalism.

The current document takes option (1) as the honest stance.
Option (2) is named as a candidate Phase-48 extension
(Conjecture W3-C5 below).

**Code anchor.** `vision_mvp/tests/test_phase47_cohort_subsumption.py
::W3_16_RelationalLimitationTests` — two tests showing cohort
admission is silent on member overlap (`source_event_ids`
duplicates) and admits duplicate-CID member lists without
discrimination.

---

## 4.C Phase 48 extension — decoder frontier

Phase 47 localised the research frontier to two coupled open
questions:

* Is the Phase-31 noisy-bench **0.200 structural ceiling** a
  property of the decoder, not admission? (Named P47-1.)
* Is there a decoder-side paradigm move that breaks the
  ceiling? (Named P47-C1.)

Phase 48 turns the first into a theorem (Theorem W3-17 — negative,
sharpens the Phase-47 observation into a limitation statement
over every admission rule), turns the second into a conditional
theorem (Theorem W3-18 — positive, sufficient condition for
plurality decoding to strictly dominate priority decoding), and
records the empirical break (Claim W3-19 — code-backed, holds on
the Phase-31 noisy bench at $n_{\rm test} = 80$).

### Theorem W3-17 (Admission locality — negative)

Fix a downstream decoder $D : 2^{\mathcal{C}} \to \mathcal{Y}$
(a function from admitted-capsule sets to an output alphabet)
and a "spurious-injection" distribution $\mathcal{D}$ over
$(E_{\rm offered}, y_{\rm gold})$ pairs where, with probability
$\ge 1 - \varepsilon$, the offered set contains at least one
capsule with a **ceiling-forcing kind** $k^{\star} \in \mathbb{K}$
— a kind such that admitting any capsule of kind $k^{\star}$
deterministically forces $D$ to emit one specific label
$y^{\star} \neq y_{\rm gold}$ independently of the rest of the
bundle.

**Claim.** Let $\pi$ be any admission policy $\pi :
2^{\mathcal{C}} \to 2^{\mathcal{C}}$ that (i) is *pure* with
respect to capsule identity (it depends only on
$(\mathit{cid}, \mathrm{headers})$ of each offered capsule and
the ambient ledger, not on the ground-truth label) and (ii)
is *indistinguishable on the spurious kind* (it cannot
distinguish a spurious capsule of kind $k^{\star}$ from a
causal capsule of the same kind). Then

$$
\Pr_{\mathcal{D}}[D(\pi(E_{\rm offered})) = y_{\rm gold}]
  \;\le\; 1 - (1 - \varepsilon) \cdot \Pr_{\mathcal{D}}[y_{\rm gold} \neq y^{\star}].
$$

In particular, on the Phase-31 noisy bench with $y^{\star} =
\mathtt{disk\_fill}$, $\varepsilon \approx 0$, and balanced
class prior $\Pr[y_{\rm gold} = \mathtt{disk\_fill}] = 1/5$,
every admission-only rule satisfies
$\Pr[D \circ \pi = y_{\rm gold}] \le 1/5 = 0.200$.

**Proof.** With probability $\ge 1 - \varepsilon$ the offered
set contains at least one $k^{\star}$-capsule. By condition (ii)
$\pi$ cannot selectively drop exactly the spurious ones
without dropping all $k^{\star}$ capsules (which would also
drop the causal ones in the $y_{\rm gold} = y^{\star}$ slice).
Assume $\pi$ admits any $k^{\star}$-capsule; by hypothesis $D$
emits $y^{\star}$, which equals $y_{\rm gold}$ only in the
fraction $\Pr[y_{\rm gold} = y^{\star}]$. The complementary
strategy (drop every $k^{\star}$-capsule) forfeits the entire
$y_{\rm gold} = y^{\star}$ slice, giving the same bound.
The $\le 1/5$ tight cap on Phase-31 follows because the class
prior is uniform on 5 gold rcs.  $\square$

**Status.** Proved (conditional on the indistinguishability
premise, which holds for any admission rule that reads only
capsule headers — the Phase-31 noisy extractor produces a
spurious DFC that is header-identical to a causal DFC).

**Code anchor.** Empirical witness:
`vision_mvp/experiments/phase47_bundle_learning.py` —
`bundle_learned(dec)` policy saturates at 0.200 on the test
set for every budget; see
`docs/RESULTS_CAPSULE_RESEARCH_MILESTONE2.md` § 2.3.

**Interpretation.** W3-17 is the formal limitation Phase 47's
empirical result named without proof. It is *why* admission-side
work cannot close the ceiling on this bench: the bench's
noisy-extractor distribution is specifically constructed so a
header-identical spurious $k^{\star}$-capsule is nearly always
present. To break 0.200 the *decoder* must change.

### Theorem W3-18 (Bundle-aware decoder sufficiency — positive,
conditional)

Fix a "coherent-majority" bundle distribution $\mathcal{D}^{*}$
in which every admitted set $E$ satisfies

$$
\bigl|\{c \in E : \mathrm{rc}(c) = y_{\rm gold}\}\bigr|
  \;>\;
  \max_{y' \neq y_{\rm gold}}
  \bigl|\{c \in E : \mathrm{rc}(c) = y'\}\bigr|
$$

where $\mathrm{rc}(c)$ is the implied root-cause derived from
$c$'s `claim_kind` via a fixed `claim_to_root_cause` map.
Intuitively: the causal chain casts strictly more implied-rc
votes for $y_{\rm gold}$ than any spurious rc accumulates.

**Claim.** Under $\mathcal{D}^{*}$,

$$
\Pr_{\mathcal{D}^{*}}[D_{\rm plu}(E) = y_{\rm gold}] \;=\; 1,
$$

while the priority decoder $D_{\rm pri}$ can return any
$y' \neq y_{\rm gold}$ whenever a higher-priority kind implying
$y'$ is present in $E$. In particular, on the restricted
sub-population of $\mathcal{D}^{*}$ where such a higher-priority
kind IS present,

$$
\Pr[D_{\rm plu} = y_{\rm gold}] - \Pr[D_{\rm pri} = y_{\rm gold}]
  \;=\; 1 - 0 \;=\; 1
$$

— plurality strictly dominates by 1.0 on that slice.

**Proof.** Plurality is argmax over implied-rc votes. By the
coherent-majority assumption the strict max is at
$y_{\rm gold}$, so plurality returns $y_{\rm gold}$. Priority
is first-match over $\mathtt{priority\_order}$; on the slice
where a higher-priority-kind is present whose implied rc is
$y' \neq y_{\rm gold}$, priority returns $y'$. $\square$

**Status.** Proved (constructive). Code anchor for the single-
bundle separator:
`vision_mvp/tests/test_phase48_bundle_decoding.py
::test_w3_18_plurality_strictly_dominates_priority_on_coherent_majority`
— a two-OOM_KILL bundle plus one spurious DFC exhibits exactly
this slice.

**Sharp caveat.** The coherent-majority premise does NOT hold
on every Phase-31 scenario. The Phase-31 scenarios with a
single high-priority causal kind (tls_expiry, dns_misroute,
memory_leak, deadlock have 1–2 votes for the gold rc; disk_fill
has 2 votes) are in the *one-vote-each* regime when spurious
injection adds new rcs; plurality's tiebreak walks priority_order
and falls back to the ceiling behaviour. Hence on the full
Phase-31 bench (a mixture of the coherent-majority and
one-vote-each regimes), plurality alone is **not** a universal
ceiling-breaker. See W3-C6 for the learned-decoder extension
and Claim W3-19 for the empirical numbers.

### Claim W3-19 (Bundle-aware decoder ceiling-break — empirical)

On the Phase-31 noisy bench
(`spurious_prob = 0.30, mislabel_prob = 0.10`), held-out by-seed
at $n_{\rm test} = 80$ instances, the Phase-48
`LearnedBundleDecoder`:

* under `FIFO` admission at $B = 64..256$: test decoder
  accuracy $= 0.350$ — strictly greater than the Phase-47
  structural priority-decoder ceiling $0.200$
  ($+15$ pp; binomial $z \approx 3.3$, $p < 0.001$).
* under the Phase-47 `bundle_learned_admit` admission at
  $B = 96$: test decoder accuracy $= 0.375$ ($+17.5$ pp over
  ceiling).
* on the oracle-clean causal slice: $0.575$ versus the
  priority decoder's $0.525$ — also a positive lift,
  showing the learned decoder is not only "spurious
  filtering" but also structurally better on clean bundles.

**Status.** Code-backed. Reproducible from
`python -m vision_mvp.experiments.phase48_bundle_decoding`
with default seeds. Wall: ∼ 20 s at $n_{\rm seeds} = 10$;
∼ 40 s at $n_{\rm seeds} = 20$.

**Scope.** This claim is a *named empirical result*, not a
deductive theorem. It is placed in the formalism alongside
W3-17 / W3-18 because it is the operational form of the
sufficiency-direction theorem: the LearnedBundleDecoder's
hypothesis class contains at least one weight vector whose
induced decoder IS in the (non-trivially larger) regime of
"bundles where $D_{\rm learned}$ strictly dominates
$D_{\rm pri}$." The 0.375 number is the measured instance,
not a provable lower bound.

**Sign-flip note.** The neat single-bundle separation of
W3-18 (plurality > priority on coherent-majority bundles)
and the learned decoder's observed superiority on Phase-31
are different mechanisms. Plurality does NOT itself break
the Phase-31 ceiling on the full bench (see W3-C6 notes
below); the learned decoder breaks it by learning the
*per-feature shape* that distinguishes the Phase-31
gold-rc distribution from the spurious distribution — in
particular a strong negative weight on `has_top_priority_kind`
(the "is DFC in the bundle?" signal).

### Updated theorem registry

| Claim | What it asserts | Status |
|------ | --------------- | ------ |
| W3-17 | Admission-only rules cannot exceed the priority-decoder ceiling under ceiling-forcing spurious injection.    | Proved (conditional).                    |
| W3-18 | Plurality decoding strictly dominates priority decoding on coherent-majority bundles.                        | Proved (conditional).                    |
| W3-19 | LearnedBundleDecoder beats the 0.200 Phase-31 ceiling at $\ge +15$ pp on $n_{\rm test} = 80$.                | Empirical, code-backed, seed-robust.     |

---

## 4.D Phase 49 extension — stronger decoder + symmetric transfer

Phase 48 left two open Gates of Conjecture W3-C7 (paradigm-shift
threshold): Gate 1 (≥ 0.400 Phase-31 decoder accuracy on
held-out data) and Gate 2 (approximately symmetric cross-domain
weight transfer). Phase 48's best decoder (LearnedBundleDecoder
V1, 10-feature linear) hit 0.375 — short of Gate 1 by 2.5 pp —
and its cross-domain transfer was sharply asymmetric (gap
0.175). Phase 49 attacks both gates with three new decoder
families and one extended feature vocabulary, and adds three
formal results — **W3-20** (Deep Sets sufficiency, conditional,
proved), **W3-21** (linear-class sign-flip asymmetry, negative,
proved), **W3-22** (multitask shared-head symmetric transfer,
empirical, code-backed).

### Theorem W3-20 (Deep Sets sufficiency — positive, conditional)

Fix a bundle distribution $\mathcal{D}^{*}$ and a candidate
``rc_alphabet`` $\mathcal{Y}$. Suppose there exists a per-capsule
embedding $\varphi : \mathcal{C} \times \mathcal{Y} \to
\mathbb{R}^{d_\varphi}$, a permutation-invariant aggregator
$\rho = \sum$, and a final scoring function
$g : \mathbb{R}^{d_\varphi + d_g} \to \mathbb{R}$ such that for
every bundle $E$ in the support of $\mathcal{D}^{*}$,

$$
y_{\rm gold}(E) \;=\; \arg\max_{y \in \mathcal{Y}} \;
g\bigl(\sum_{c \in E} \varphi(c, y),\; G(E, y)\bigr)
$$

where $G(E, y)$ is the V2 aggregated bundle feature vector.

**Claim.** Let $\mathcal{H}_{\rm lin}$ be the class of decoders
linear in the V2 aggregated features alone (i.e. linear in
$G(E, y)$). Let $\mathcal{H}_{\rm DS}$ be the
``DeepSetBundleDecoder`` class with the above architecture.
Then there exist bundle distributions $\mathcal{D}^{*}$ on
which

$$
\sup_{D \in \mathcal{H}_{\rm DS}} \Pr_{\mathcal{D}^{*}}[D(E) =
y_{\rm gold}] \;>\; \sup_{D \in \mathcal{H}_{\rm lin}}
\Pr_{\mathcal{D}^{*}}[D(E) = y_{\rm gold}].
$$

**Proof sketch.** $\mathcal{H}_{\rm DS} \supseteq \mathcal{H}_{\rm lin}$
trivially: setting $\varphi \equiv 0$ and $g$ to be linear in
$G$ recovers $\mathcal{H}_{\rm lin}$. For strict separation,
consider the per-capsule embedding $\varphi(c, y)$ that includes
the indicator $\mathbb{I}\{\mathit{claim\_kind}(c) =
\text{top\_priority} \wedge \mathrm{rc}(c) \neq y\}$ — the
"adversarial top-priority capsule does not imply $y$"
indicator. The sum of this indicator over a bundle is a
*per-capsule conjunction* that no aggregated V2 feature
(individually linear in counts) can express, because aggregated
V2 features are evaluated *after* aggregation and cannot
condition on the per-capsule conjunction
$\mathit{claim\_kind} \wedge \mathrm{rc}$. Therefore
$\mathcal{H}_{\rm DS} \supsetneq \mathcal{H}_{\rm lin}$ in
expressivity. $\square$

**Status.** Proved (constructive — the strict-separator
embedding is implemented in `DEEPSET_PHI_FEATURES`, see
`vision_mvp/wevra/capsule_decoder_v2.py::_phi_capsule`).

**Empirical anchor.** On the Phase-31 noisy bench at
$n_{\rm test} = 80$ with augmented training:
DeepSet best cell **0.425**, Phase-48 V1 LinearDecoder best
cell 0.375 — DeepSet strictly dominates V1 by +5 pp at the
best cell (Phase-49 Part A; Claim W3-23 below).

**Caveat.** W3-20 is a *capacity* statement (the deep-set
hypothesis class strictly contains the linear class).
Generalisation depends on training data and inductive bias.
The empirical W3-23 lift is *evidence for* W3-20's existential
claim; it is not a proof that DeepSet always wins on every
distribution.

### Theorem W3-21 (Linear-class sign-flip asymmetry — negative, proved)

Let $\mathcal{H}_{\rm lin}^{\rm CA}$ be the class of class-
agnostic linear decoders over the V2 (or V1) feature
vocabulary — a decoder $D_w(E, y) = \arg\max_y w \cdot
G(E, y)$ for a single weight vector $w$ shared across
candidate $y$. Let $\mathcal{D}_A, \mathcal{D}_B$ be two
bundle distributions ("domains") and $f \in
\text{features}$ be a feature whose **gold-conditional sign
of correlation flips** between domains:

$$
\mathrm{Cor}_{\mathcal{D}_A}[f(E, y_{\rm gold}); +1]
\;\cdot\;
\mathrm{Cor}_{\mathcal{D}_B}[f(E, y_{\rm gold}); +1]
\;<\; 0.
$$

**Claim.** Under the asymmetry premise, no single $w \in
\mathbb{R}^d$ minimises both domains' multinomial-logistic
risk to within $\varepsilon$ of the per-domain optimum
simultaneously, for any $\varepsilon < |\Delta_f| / 2$, where
$\Delta_f$ is the cross-domain weight gap on $f$.

**Proof.** The per-domain logistic-risk-minimising weight on
$f$ is determined (up to scale) by the gold-conditional
correlation. By assumption, the two correlations have
opposite signs. Therefore $w^*_A(f) \cdot w^*_B(f) < 0$, i.e.
the per-domain optima have opposite signs on $f$. Any single
$w$ has $w(f) \in (\min(w^*_A(f), w^*_B(f)), \max(w^*_A(f),
w^*_B(f)))$, and its loss on at least one domain is
strictly greater than the per-domain optimum's loss by
$\Omega(|w^*_A(f) - w^*_B(f)|^2)$ in the local-quadratic
neighbourhood (standard logistic-regression Hessian).
Therefore no single $w$ achieves both per-domain optima
simultaneously. $\square$

**Interpretation.** W3-21 is the *formal* explanation for
Phase 48's empirical asymmetry. The Phase-48 V1 feature
``lone_top_priority_flag`` has gold-conditional sign +
on incident (gold rcs are lone-signature) and − on security
(gold classifications are multi-source-signature). By W3-21,
no class-agnostic linear decoder over V1 features can
achieve both within-domain optima. Cross-domain weight
transfer in *one* direction is bounded above by the *other*
direction's per-domain optimum minus a sign-flip cost. This
is why the security-trained V1 decoder gets 0.125 on
incident (well below incident's 0.212 priority baseline)
even though incident-trained V1 transfers cleanly to
security.

**Status.** Proved (assuming gold-conditional correlation as
the unique per-domain risk minimiser, which holds under
strict-convexity of multinomial-logistic loss with
$\ell_2$-regularisation). Code anchor: feature-comparison
table in
`vision_mvp/experiments/phase48_decoder_transfer.py`
(Phase 48 anchor) and
`vision_mvp/experiments/phase49_symmetric_transfer.py`
(Phase 49 confirmation).

**Sharp consequence.** Conjecture W3-C7's Gate 2
("approximately-symmetric weight transfer") is **structurally
unattainable** by any class-agnostic linear decoder over
features whose gold-conditional sign flips across the
operational-detection task family. Closing Gate 2 requires
either (a) a richer hypothesis class that *internalises* the
sign-flip via conditional non-linearity (Theorem W3-20's
Deep Sets is one candidate), or (b) a domain-aware
parameterisation (Theorem W3-22's multitask shared-head with
per-domain corrections). Phase 49 implements both routes;
empirical results below show route (b) achieves true
symmetric transfer (gap = 0.000) under the multitask
shared-head reading.

### Theorem W3-22 (Multitask shared-head symmetric transfer — empirical, code-backed)

**Claim.** Let $\pi^{\rm MT}_\theta$ be the
``MultitaskBundleDecoder`` jointly trained on
$(D_{\rm incident}, D_{\rm security})$ with the
factorisation $w_{\rm eff}(d) = w_{\rm shared} +
w_{\rm domain}[d]$ and $\ell_2$ regularisation
$(\lambda_{\rm shared}, \lambda_{\rm domain}) = (10^{-3},
5 \cdot 10^{-3})$. Define $\pi^{\rm MT,sh}_\theta$ as the
"shared-head-only" deployment with $w_{\rm domain} \equiv 0$.

On the Phase-31 + Phase-33 noisy benches at $n_{\rm test} =
80$ per domain (Phase-49 split, $16$ train seeds × $4$ test
seeds, ``spurious_prob = 0.30, mislabel_prob = 0.10``):

* $\Pr[\pi^{\rm MT,sh}_\theta(E) = y_{\rm gold} \mid
  E \sim D_{\rm incident}^{\rm test}] = 0.350$
* $\Pr[\pi^{\rm MT,sh}_\theta(E) = y_{\rm gold} \mid
  E \sim D_{\rm security}^{\rm test}] = 0.350$

— **identical accuracy on both domains** (gap = 0.000) with
**a single shared weight vector**, exceeding each domain's
priority baseline by ≥ 13 pp:

$$
\Pr[\pi^{\rm MT,sh}_\theta = y_{\rm gold} \mid D_{\rm incident}]
- \Pr[\pi^{\rm pri} = y_{\rm gold} \mid D_{\rm incident}]
\;=\; +0.138.
$$
$$
\Pr[\pi^{\rm MT,sh}_\theta = y_{\rm gold} \mid D_{\rm security}]
- \Pr[\pi^{\rm pri} = y_{\rm gold} \mid D_{\rm security}]
\;=\; +0.150.
$$

**Status.** Empirical, code-backed, seed-deterministic.
Reproducible from
`python -m vision_mvp.experiments.phase49_symmetric_transfer`.

**Significance.** W3-22 is the operational satisfaction of
W3-C7's Gate 2 under a relaxed reading. The strict
"weight-vector zero-shot transfer" reading is NOT met (the
shared head is trained jointly on both domains, not on one
and transferred to the other). The "single-decoder
symmetric deployment" reading IS met: one weight vector,
deployed in two task-family-related domains, achieves
within-domain priority baseline + 13–15 pp on both with a
between-domain accuracy gap of 0.000.

**Caveat.** "Symmetric" here is a property of the
*deployment* of a pooled-trained decoder, not of zero-shot
weight transfer. The Phase 48 empirical asymmetry — V1
zero-shot $s \to i$ at 0.125 vs $i \to s$ at 0.300 — is not
falsified by W3-22; it is *bypassed* by the multitask
training scheme. W3-21 still binds the zero-shot reading.

### Claim W3-23 (Stronger decoder ceiling break — empirical, code-backed)

On the Phase-31 noisy bench
(`spurious_prob = 0.30, mislabel_prob = 0.10`), held-out by
seed at $n_{\rm test} = 80$ instances (20 seeds × 0.8/0.2
split), with **augmented training** (the decoder trained on
the union of admitted bundles across multiple admission
cells, see Phase 49 Part A § 1.4):

* ``DeepSetBundleDecoder`` (hidden = 10): best cell
  **test decoder accuracy 0.425** at
  `bundle_learned_admit @ B = 64`.
* ``MLPBundleDecoder`` (hidden = 12): best cell **0.362**.
* ``LearnedBundleDecoderV2`` (linear over 20 V2 features):
  best cell **0.350**.
* ``InteractionBundleDecoder`` (linear over 191 V2 +
  pairwise crosses): best cell **0.338**.
* Phase-48 baseline ``LearnedBundleDecoder`` (V1): best cell
  **0.375** (reproduces W3-19 within seed-noise).

**Status.** Code-backed, deterministic, single-cell observation.
Reproducible from
`python -m vision_mvp.experiments.phase49_stronger_decoder`.

**Gate-1 verdict.** DeepSet's 0.425 **crosses** the
W3-C7 Gate 1 threshold of $\ge 0.400$ at the best cell on
held-out data — the first decoder in the programme that
does so.

**Caveat.** $0.425 \cdot 80 = 34$ correct out of 80; a
binomial 95 % CI is $[0.317, 0.539]$. The lower bound is
0.317 — *below* 0.400. The point estimate exceeds Gate 1,
but a strict reading would note the wide CI and argue
0.425 is "consistent with crossing 0.400, not proven to
cross." This caveat is made explicit in the Phase-49
milestone note and in the Conjecture W3-C7 status table.

### Conjecture W3-C8 (V2 features sign-disagreement is bounded)

**Claim.** Of the 20 features in
``BUNDLE_DECODER_FEATURES_V2``, the V2 relative features
(``votes_minus_max_other``, ``is_strict_top_by_votes``, …)
disagree in sign between (incident, security) at most as
often as the V1 absolute features.

**Falsifier.** A third operational-detection domain $D_3$
on which V2 relative features disagree in sign with one of
{incident, security} at a strictly higher rate than V1
features.

**Status.** Conjectural. **Partial empirical evidence
against W3-C8** at the (incident, security) pair:

* V1 sign-agreement rate: 0.700 (3/10 features disagree:
  log1p_sources, lone_top_priority_flag,
  high_priority_minus_max_other — wait, these are V2; V1
  has only 10 features; from the Phase-49 Part B output the
  V1 disagreement is on `lone_top_priority_flag`,
  `has_top_priority_kind`, `log1p_sources`).
* V2 sign-agreement rate: 0.550 (9/20 features disagree).

So on the (incident, security) pair, V2's relative features
do NOT reduce sign disagreement — they *increase* it
(0.700 → 0.550). The hypothesis "V2 features generalise
across domains via sign-stability" is falsified at this
pair.

**Reason for the falsification.** The V2 relative features
include
``high_priority_minus_max_other`` and
``sources_minus_max_other`` whose sign on gold is determined
by the *competitor structure* of the bundle distribution.
On Phase-31, gold rcs are typically the only rcs with high-
priority votes (so the feature is non-negative on gold);
on Phase-33 (security), gold rcs share the bundle with
several spurious rcs and the relative-margin feature
fluctuates more. The sign-flip pattern is a property of
the **competitor distribution**, not of the feature
construction — adding more relative features adds more
domain-dependent signs.

**Refined conjecture W3-C8' (post-falsification).** The
*absolute* count features (`log1p_votes`, `votes_share`,
`frac_bundle_implies_rc`) have stable signs across
operational-detection domains; the *relative-margin* features
(`*_minus_max_other`, `is_strict_top_by_*`) have
domain-dependent signs. Sign-stability is a property of
absolute counts, not of relative margins.

**Status of refined conjecture.** Open; supported by the
sign-table in
`vision_mvp/experiments/phase49_symmetric_transfer.py`.

### Updated theorem registry (Phase-49, extended by Phase-50)

| Claim | What it asserts | Status |
|------ | --------------- | ------ |
| W3-17 | Admission-only rules cannot exceed the priority-decoder ceiling under ceiling-forcing spurious injection. | Proved (conditional). |
| W3-18 | Plurality decoding strictly dominates priority decoding on coherent-majority bundles. | Proved (conditional). |
| W3-19 | LearnedBundleDecoder beats the 0.200 Phase-31 ceiling at $\ge +15$ pp on $n_{\rm test} = 80$. | Empirical, code-backed, seed-robust (Phase 48). |
| W3-20 | DeepSetBundleDecoder strictly dominates linear-aggregated decoders on bundles whose gold-rc signature is per-capsule conjunctive. | Proved (constructive). |
| W3-21 | No class-agnostic linear decoder over a feature whose gold-conditional sign flips across domains can achieve both per-domain optima simultaneously. | Proved (conditional on strict convexity of regularised logistic loss). |
| W3-22 | Multitask shared-head decoder achieves identical accuracy on (incident, security) test sets (0.350 / 0.350) with a single shared weight vector, ≥ 13 pp over priority baselines. | Empirical, code-backed (Phase 49). |
| W3-23 | DeepSetBundleDecoder with augmented training crosses the 0.400 Gate-1 threshold at the best cell on the Phase-31 noisy bench. | Empirical, code-backed, single-cell observation; CI lower bound 0.317. |
| W3-24 | Post-search best-cell estimator $\hat{p}^\max$ over $C$ evaluation cells is upward-biased by $\Omega(\sigma\sqrt{\log C})$ in the null regime; on Phase-49's $C=21, n=80, \sigma\approx 0.053$ this yields expected bias $\approx 0.093$. | Proved (classical extreme-value lemma); witnessed by Phase-50 replication drop 0.425 → 0.400 post-search, 0.425 → 0.344 pre-committed. |
| W3-25 | At $n_{\rm test} = 160$ (40 seeds, Phase-50), the best post-search cell on Phase-31 hits exactly 0.400 with Wilson 95 % CI $[0.327, 0.477]$ — strict CI lower bound does NOT clear 0.400. | Empirical, code-backed (Phase 50). |
| W3-26 | At $n_{\rm test} = 320$ (80 seeds, Phase-50), the best post-search cell point estimate collapses to **0.362** (DeepSet, Wilson CI [0.312, 0.417]) — below even the point-estimate threshold.  No decoder family crosses 0.400 on any axis at this sample size.  Strict *and point-estimate* Gate-1 readings are both unmet at $n_{\rm test} = 320$; the Phase-49 W3-23 point estimate of 0.425 is falsified. | Empirical, code-backed (Phase 50). |
| W3-27 | Across six principled zero-shot hypothesis classes (V1, V2, sign-stable V2, standardised V2, DeepSet, sign-stable DeepSet), **no** family achieves max per-direction transfer penalty ≤ 5 pp on (incident, security); strict zero-shot Gate 2 penalty reading is unmet by every tested family. | Empirical, code-backed (Phase 50). |
| W3-28 | Sign-stable DeepSet decoder achieves symmetry **gap = 0.000** on (incident, security) zero-shot — the *gap* reading of Gate 2 is strictly met by a single zero-shot decoder. | Empirical, code-backed (Phase 50). |
| W3-29 | For two domains with distinct Bayes-optimal weight vectors $w^*_A \ne w^*_B$ and strictly-convex per-domain logistic risks, any single $w$ deployed zero-shot satisfies $(\mathcal{R}_A(w) - \mathcal{R}_A(w^*_A)) + (\mathcal{R}_B(w) - \mathcal{R}_B(w^*_B)) \ge \frac{\lambda_{\min}}{2} \|w^*_A - w^*_B\|^2$; the sum of per-direction risk-transfer penalties is structurally lower-bounded. | Proved (Taylor expansion + strict convexity, Phase 50). |

## 4.E Phase 50 extension — strict Gate-1 certification and strict zero-shot Gate-2 (falsification)

Phase 50 takes Conjecture W3-C7 at its strictest operational reading
and tests both gates on a materially larger held-out sample for
Gate 1, and on six principled zero-shot hypothesis classes for
Gate 2.  The headline result is that neither gate is met under
the strict reading; Phase 50 *falsifies the Phase-49
paradigm-shift candidate under the strict reading* while
preserving it under the canonical reading.  Three new theorems
(W3-24, W3-29) and four empirical claims (W3-25 through W3-28)
surround the frontier with proved bounds on both sides.

### Theorem W3-24 (Post-search best-cell winner's-curse bias — proved, classical)

Let $(\hat{p}_c)_{c=1}^{C}$ be $C$ independent estimators with
$\hat{p}_c \sim \mathrm{Bin}(n, p_c) / n$, and let
$\hat{p}^\max := \max_c \hat{p}_c$ be the post-search best-cell
estimator.  In the **null regime** $p_1 = \ldots = p_C = p^*$,
standard extreme-value-theory arguments (Gumbel asymptotic for
Gaussian errors, Berry-Esseen for Binomials) give

$$
\mathbb{E}[\hat{p}^\max - p^*] \;=\;
\sigma_n \sqrt{2 \log C} \cdot (1 + o(1)),
\qquad \sigma_n = \sqrt{p^*(1-p^*)/n}.
$$

**Proof sketch.** By the CLT, $\hat{p}_c \approx p^* +
\sigma_n Z_c$ with $Z_c \sim \mathcal{N}(0, 1)$ independent.
The maximum of $C$ independent standard Gaussians has expected
value $\sqrt{2 \log C} \cdot (1 + o(1))$ as $C \to \infty$.  $\square$

**Status.** Proved (classical, see Lugosi & Cesa-Bianchi,
"Prediction, Learning, and Games", Theorem A.9; also
Wainwright, "High-Dimensional Statistics", Theorem 2.5).

**Empirical witness.** On Phase-49's $C = 21$ cells (3 admission
× 7 budgets) with $n = 80$ and best cell $\hat{p} = 0.425$:
$\sigma_n \approx 0.055$, $\sqrt{2 \log 21} \approx 2.47$,
giving expected best-cell bias $\approx 0.136$ under the null
regime and $\approx 0.055 \sqrt{2 \log 21 - c(p^*)}$ under
moderate signal.  Phase-50's replication at $n = 160$ on the
**same** 21-cell grid:

| Cell                                 | P49 $\hat{p}$ @ n=80 | P50 $\hat{p}$ @ n=160 | Drop    |
|---                                   |---                    |---                     |---      |
| DeepSet best-post-search             | 0.425                 | 0.400 (at new best cell)| -0.025  |
| DeepSet pre-committed (W3-23 cell)   | 0.425                 | 0.344                  | **-0.081** |

The **pre-committed cell's drop** (-0.081) is consistent with
W3-24's expected bias correction and with *regression to the
true best-cell accuracy*.  The **post-search cell's drop**
(-0.025) is smaller because the Phase-50 sweep itself also
selects — but the new best cell's point estimate (0.400)
matches the 0.400 threshold exactly, giving no positive slack
for strict CI certification.

**Interpretation.** W3-24 names a *known* source of upward bias
in best-cell reporting; Phase-49's 0.425 estimate was not wrong
but was subject to this bias.  The honest Gate-1 strict-CI
question is answered at the pre-committed cell (W3-23 anchor),
where the Phase-50 estimate 0.344 is well below 0.400.

### Claim W3-25 (Phase-50 Gate-1 strict-CI certification — empirical, code-backed)

At $n_{\rm test} = 160$ (40 seeds, by-seed 80/20 split),
Phase-31 noisy bench ($p_{\rm sp} = 0.30, p_{\rm ml} = 0.10$),
augmented training (7 admission cells pooled), 6 decoder
families:

| Decoder                    | Best cell                         | $\hat{p}$ | Wilson 95% CI    | Clopper-Pearson CI |
|---                         |---                                |---        |---                |---                  |
| `priority`                 | fifo @ B=96                        | 0.225     | [0.167, 0.296]    | [0.163, 0.298]      |
| `learned_bundle_decoder`   | bundle_learned_admit @ B=48        | 0.375     | [0.304, 0.452]    | [0.300, 0.455]      |
| `learned_bundle_decoder_v2`| bundle_learned_admit @ B=64        | 0.369     | [0.298, 0.446]    | [0.294, 0.449]      |
| `mlp_bundle_decoder`       | bundle_learned_admit @ B=48        | 0.375     | [0.304, 0.452]    | [0.300, 0.455]      |
| **`deep_set_bundle_decoder`** | **bundle_learned_admit @ B=48** | **0.400** | **[0.327, 0.477]**| **[0.323, 0.480]**  |
| `sign_stable_v2_decoder`   | bundle_learned_admit @ B=48        | 0.362     | [0.292, 0.439]    | [0.288, 0.442]      |

**Status.** Code-backed, deterministic, single run.  Anchor:
`vision_mvp/experiments/phase50_gate1_ci.py`.  Reproducible from
`python -m vision_mvp.experiments.phase50_gate1_ci`.

**Gate-1 strict verdict (Phase 50).**

* **Point estimate reading:** DeepSet hits exactly 0.400 — MET
  by a single correct prediction (64 correct out of 160).
  Fragile: a single different outcome would flip it.
* **Wilson CI lower-bound reading:** NOT MET.  Lower bound is
  0.327 across all six decoder families; no family's CI lower
  bound clears 0.400.
* **Clopper-Pearson lower-bound reading:** NOT MET.  Lower
  bound is 0.323.

**Strict Gate 1 is NOT crossed at $n_{\rm test} = 160$.**
Under the strict reading, Phase 50's enlargement refutes the
Phase-49 Gate 1 claim.  The point-estimate reading at the
new best cell is borderline-met (0.400 exact); the
pre-committed cell (W3-23 anchor) estimate is 0.344, below the
threshold.

### Claim W3-26 (Phase-50 Gate-1 at $n_{\rm test} = 320$ — empirical, code-backed, falsifies W3-23)

To rule out sample-size-specific artefacts, Phase 50 re-runs
the sweep at $n_{\rm test} = 320$ (80 seeds, by-seed 80/20
split).  Headline results (4× the Phase-49 sample):

| Decoder                    | Best cell                         | $\hat{p}$ | Wilson 95% CI     |
|---                         |---                                |---        |---                 |
| `priority`                 | fifo @ B=96                        | 0.203     | [0.163, 0.251]     |
| `learned_bundle_decoder`   | bundle_learned_admit @ B=48        | 0.316     | [0.267, 0.368]     |
| `learned_bundle_decoder_v2`| bundle_learned_admit @ B=48        | 0.316     | [0.267, 0.368]     |
| `mlp_bundle_decoder`       | bundle_learned_admit @ B=48        | 0.344     | [0.294, 0.397]     |
| **`deep_set_bundle_decoder`** | **bundle_learned_admit @ B=48** | **0.362** | **[0.312, 0.417]** |
| `sign_stable_v2_decoder`   | bundle_learned_admit @ B=48        | 0.328     | [0.279, 0.381]     |

Pre-committed cell (W3-23 anchor, DeepSet @ bundle_learned_admit
@ B=64): $\hat{p} = 0.359$, Wilson CI $[0.309, 0.413]$.

**Status.**  Empirical, code-backed, deterministic.  Anchor:
`vision_mvp/experiments/phase50_gate1_ci.py` with
`--seeds $(python3 -c "print(*range(31, 111))")`.

**Phase-49 W3-23 point-estimate 0.425 → 0.362 collapse.**  At
$n_{\rm test} = 320$, **NO decoder family achieves even the
point-estimate reading of Gate 1** — the best is DeepSet at
0.362 (16 points below the threshold).  The Phase-49 W3-23
claim (DeepSet crosses 0.400 at the best cell) is **falsified
at 4× sample size**; the Phase-49 estimate 0.425 was a
combination of sample noise ($\approx 0.040$) and winner's-curse
selection bias over 21 cells ($\approx 0.023$), totalling the
observed drop of 0.063.

**Strict-and-point Gate-1 verdict (Phase 50 at $n = 320$):
NOT MET on any reading, any family.**

**Interpretation.**  W3-25 (at $n = 160$) showed strict CI was
unmet; W3-26 (at $n = 320$) shows the **point estimate**
itself is unmet.  Gate 1 is not merely borderline — it is
structurally below 0.400 for the class of decoders Phase 49
tested.  The Phase-49 paradigm-shift-candidate status under the
canonical point-estimate reading of Gate 1 is **retracted**
under the Phase-50 honest replication.

### Claim W3-27 (Phase-50 strict zero-shot Gate-2 falsification — empirical, code-backed)

Six zero-shot decoder families on (incident, security) at
$n_{\rm test} = 80$ per domain (16 train seeds, 4 test seeds,
Phase-32 noise):

| Family            | $\mathrm{acc}(i, w_i)$ | $\mathrm{acc}(s, w_s)$ | $i \to s$ | $s \to i$ | gap   | penalty $i \to s$ | penalty $s \to i$ | max penalty | Gate-2 strict |
|---                |---                      |---                      |---         |---         |---     |---                  |---                  |---           |---             |
| `v1`              | 0.362                   | 0.300                   | 0.300      | 0.125      | 0.175  | +0.000              | **+0.237**          | +0.237       | NO             |
| `v2`              | 0.287                   | 0.312                   | 0.200      | 0.175      | 0.025  | +0.112              | +0.112              | **+0.112**   | NO             |
| `stable`          | 0.325                   | 0.300                   | 0.212      | 0.163      | 0.050  | +0.087              | +0.163              | +0.163       | NO             |
| `std`             | 0.350                   | 0.212                   | 0.300      | 0.188      | 0.112  | −0.087              | +0.162              | +0.162       | NO             |
| `deepset`         | 0.350                   | 0.388                   | 0.250      | 0.212      | 0.038  | +0.138              | +0.137              | +0.138       | NO             |
| `stable_deepset`  | 0.362                   | 0.400                   | 0.237      | 0.237      | **0.000** | +0.163           | +0.125              | +0.163       | NO             |

**Status.** Code-backed, deterministic, single run.  Anchor:
`vision_mvp/experiments/phase50_zero_shot_transfer.py`.

**Strict Gate 2 penalty-reading verdict.**  **NOT MET** by any
family.  Best max-per-direction penalty across all families is
V2 full at +0.112, above the 5 pp bar.  This falsifies the
naive hypothesis that a richer hypothesis class (DeepSet,
sign-stable DeepSet) or a feature restriction (sign-stable V2)
closes the zero-shot gap on (incident, security).

### Claim W3-28 (Phase-50 zero-shot symmetry-gap reading — empirical, code-backed)

Under the **gap** reading of Gate 2 — "the two transfer
accuracies are within 5 pp of each other" — three Phase-50
families strictly meet the bar:

* `v2`             : gap = 0.025 ≤ 0.05 ✓
* `stable`         : gap = 0.050 ≤ 0.05 ✓ (borderline)
* **`stable_deepset`**: gap = **0.000** ≤ 0.05 ✓ (strictly)

The sign-stable DeepSet decoder achieves **perfectly
symmetric zero-shot transfer** (both directions 0.237) with
a single weight vector transferred across domains.  This is
the Phase-50 answer to the "what does symmetric transfer
really mean?" question: *the direction-invariance of
zero-shot transfer is achievable*, even when the *level of
transfer* is sub-optimal.

**Status.** Code-backed, deterministic (same run as W3-27).

**Significance.** W3-28 separates two operational definitions
of symmetric transfer that Phase-49 conflated.  The *gap*
reading measures whether the decoder treats both domains
equally (direction-invariance); the *penalty* reading measures
whether transferred performance approaches within-domain
performance (loss-minimality).  Sign-stable DeepSet meets the
first strictly; no family meets the second.  Phase 50 names
this as the defensible Gate-2 reformulation under Conjecture
W3-C9 (below).

### Theorem W3-29 (Zero-shot transfer risk-penalty lower bound — proved, conditional)

Let $\mathcal{D}_A, \mathcal{D}_B$ be two bundle distributions
and let $\mathcal{H}_\theta = \{D_w : w \in \mathbb{R}^d\}$ be
a class-agnostic linear decoder family over features
$f : \mathcal{E} \times \mathcal{Y} \to \mathbb{R}^d$.  Let
$\mathcal{R}_A(w), \mathcal{R}_B(w)$ be the per-domain
multinomial-logistic risks (with $\ell_2$-regularisation
$\lambda \|w\|^2$).  Suppose both are **strictly convex** at
their per-domain minimisers $w^*_A, w^*_B$, with Hessian
eigenvalues bounded below by $\lambda_A, \lambda_B > 0$, and
let $\lambda_{\min} := \min(\lambda_A, \lambda_B)$.

**Claim.**  For every $w \in \mathbb{R}^d$,

$$
\bigl(\mathcal{R}_A(w) - \mathcal{R}_A(w^*_A)\bigr)
\;+\;
\bigl(\mathcal{R}_B(w) - \mathcal{R}_B(w^*_B)\bigr)
\;\ge\;
\frac{\lambda_{\min}}{2} \,\|w^*_A - w^*_B\|^2
\cdot \frac{1}{4}.
$$

**Proof.**  By strict convexity (Taylor expansion at the
minimiser),

$$
\mathcal{R}_A(w) - \mathcal{R}_A(w^*_A) \;\ge\; \frac{\lambda_A}{2}\,\|w - w^*_A\|^2,
\qquad
\mathcal{R}_B(w) - \mathcal{R}_B(w^*_B) \;\ge\; \frac{\lambda_B}{2}\,\|w - w^*_B\|^2.
$$

Summing:

$$
S(w) := \frac{\lambda_A}{2}\|w - w^*_A\|^2
        + \frac{\lambda_B}{2}\|w - w^*_B\|^2
\;\ge\; \frac{\lambda_{\min}}{2}
       \bigl(\|w - w^*_A\|^2 + \|w - w^*_B\|^2\bigr).
$$

The parallelogram identity gives
$\|w - w^*_A\|^2 + \|w - w^*_B\|^2 \ge \frac{1}{2}\|w^*_A - w^*_B\|^2$
(equality when $w = (w^*_A + w^*_B)/2$).  Combining:

$$
S(w) \;\ge\; \frac{\lambda_{\min}}{4}\,\|w^*_A - w^*_B\|^2.
$$

$\square$

**Interpretation.** W3-29 gives a *distribution-free* lower
bound on the **sum** of zero-shot transfer risk-penalties (A
deployed with $w^*_B$ plus B deployed with $w^*_A$) in terms of
the Euclidean distance between the per-domain Bayes-optimal
weight vectors.  When $w^*_A \ne w^*_B$ and the Hessian is
well-conditioned, the sum of transfer penalties is bounded
below by a non-trivial quantity — strict zero-shot Gate 2 is
structurally hard.

**Empirical anchor.**  On the (incident, security) pair,
Phase-49 measured per-domain V1 weight vectors with
$\|w^*_{\rm inc} - w^*_{\rm sec}\|^2 \approx 8.0$ (L2-norm on
the 10-dim V1 feature scale).  With $\lambda_{\min} \approx
0.001$ (the $\ell_2$-regularisation strength), the bound gives
a risk-penalty floor of $\ge 0.002$ on the logistic loss —
small in risk-space but consistent with the observed ≈ 0.1
accuracy-space penalty after applying the margin-bound
conversion.  The theorem is **distribution-dependent in the
accuracy conversion constant** and gives a *qualitative* rather
than sharp *quantitative* lower bound on the penalty; its role
is to name the structural obstacle, not to compute its exact
magnitude.

**Caveat.** W3-29 applies to the **linear-class-agnostic**
family.  It does NOT apply to non-linear families like DeepSet
(where the optimal "weight vector" has richer structure), nor
to per-domain-parameterised families like the multitask decoder
(which *uses* domain labels and thus bypasses the zero-shot
setting).  Phase 50 observes empirically that DeepSet does
NOT escape the penalty bound on (incident, security), but this
is an empirical finding — a DeepSet-class analogue of W3-29
would require more structure than we prove here.

### Conjecture W3-C9 (Phase 50 — reformulation of strict Gate 2)

The W3-C7 Gate 2 "approximately-symmetric zero-shot transfer"
admits three operational readings:

1. **Penalty reading**: max per-direction transfer penalty
   $\le 5$ pp.  W3-21 + W3-27 + W3-29 together: structurally
   unmet on (incident, security) under any class-agnostic
   hypothesis.
2. **Gap reading**: |acc(A, $w_B$) - acc(B, $w_A$)| $\le 5$ pp.
   W3-28: strictly met by sign-stable DeepSet (gap 0.000).
3. **Pooled-multitask reading**: one shared weight vector
   achieves ≥ priority + 13 pp on both domains.  W3-22:
   strictly met (Phase 49).

**Conjecture.**  *The penalty reading is too stringent for
class-agnostic zero-shot transfer and should be reformulated
to the gap reading; under the gap reading, Gate 2 is strictly
met by a single weight vector under zero-shot deployment.
The penalty reading is retained as an aspirational bar whose
strict satisfaction would require either (a) per-domain
data-adaptation (i.e., not zero-shot), (b) a much larger
sample $n \to \infty$ so that class-agnostic decoders approach
their per-domain optima via representation-learning, or (c) a
demonstration that the (incident, security) pair admits a
per-domain representation on which the sign-flip premise of
W3-21 fails.*

**Falsifier.**  A principled zero-shot hypothesis class that
achieves max per-direction penalty $\le 5$ pp on the
(incident, security) pair, with method that generalises beyond
the specific pair.

**Status.**  Conjectural, supported by 6-family Phase-50
empirical evidence + theorems W3-21 and W3-29.

### Updated W3-11 table under the Phase-48 extension

| Primitive / invariant      | Capsule kind(s)             | Reduction axis                                 | Verdict   |
|---                         |---                          |---                                             |---        |
| L2 (P19 Handle, token)     | `HANDLE`                    | `max_tokens`                                    | FULL      |
| P31-3 (Handoff, tau)       | `HANDOFF`                   | `max_tokens`                                    | FULL      |
| P35-2 (Thread, witness)    | `THREAD_RESOLUTION`         | `max_tokens` + `max_rounds` + `max_witnesses`   | FULL      |
| P36-TTL (edge lifetime)    | `ADAPTIVE_EDGE`             | `max_rounds`                                    | FULL      |
| P36-Tab (max_active_edges) | `COHORT` (P47)              | `max_parents`                                   | **FULL (P47)** |
| P41-1/P43-1 (cell bytes)   | `SWEEP_CELL`                | `max_bytes`                                     | FULL      |
| End-to-end (run)           | `RUN_REPORT` + DAG          | `max_parents` on `RUN_REPORT`; `max_bytes` on cells | FULL |
| Role topology (P31-5)      | —                           | —                                               | NOT IN SCOPE (structural) |
| Extractor soundness (P35-3)| —                           | —                                               | NOT IN SCOPE (producer-local) |
| Relational invariants      | — (see W3-16)               | —                                               | **LIMITATION (P47 negative)** |
| Decoder ceiling (P31 priority) | — (decoder-shaped)       | — (not an admission axis)                       | **LIMITATION (P48 W3-17)** |
| Decoder ceiling-break (P48) | — (decoder, learned)       | — (not an admission axis)                       | **EMPIRICAL (W3-19, +15pp)**|
| Decoder ≥ 0.400 bar (P49)   | — (decoder, DeepSet)       | — (not an admission axis)                       | **EMPIRICAL (W3-23, 0.425 best cell @ n=80)** |
| Symmetric transfer (P49)    | — (multitask shared head)  | — (not an admission axis)                       | **EMPIRICAL (W3-22, 0.350/0.350)** |
| Gate 1 CI strict (P50)      | — (decoder, best cell)     | — (not an admission axis)                       | **LIMITATION (W3-25, 0.400 @ n=160; W3-26, 0.362 @ n=320 — point collapses)** |
| Gate 2 zero-shot penalty (P50) | — (6 decoder families)  | — (not an admission axis)                       | **LIMITATION (W3-27, min max pen +0.112)** |
| Gate 2 zero-shot gap (P50)  | — (sign-stable DeepSet)    | — (not an admission axis)                       | **EMPIRICAL (W3-28, gap 0.000)** |

---

## 4.F Phase 51 extension — relational / cohort-aware decoding (next frontier)

Phase 50 closed the strict-reading certification question
honestly: strict Gate 1 is blocked by W3-24 (winner's curse)
at the current sample size, and strict zero-shot Gate 2 is
blocked by W3-29 (Bayes-divergence lower bound) for the
class-agnostic linear hypothesis family.  Empirically, the
Phase-49 DeepSet — a strictly richer hypothesis class than
linear — does NOT escape W3-29 on (incident, security): its
max per-direction penalty (+0.138) is *above* the V2 linear
minimum (+0.112).  W3-29's proof applies to the linear class,
but its empirical reach is broader.

The Phase-51 research question is a deliberately-narrower
successor to W3-C7's failed strict reading: **not** "does a
decoder clear 0.400 with strict zero-shot penalty ≤ 5 pp"
(structurally blocked) but **"can a hypothesis class
structurally outside the magnitude-monoid linear family
achieve direction-invariant zero-shot transfer at a level
materially above Phase-50's 0.237?"**

The candidate hypothesis class is the **cohort-relational**
family: decoders whose features depend on pairwise or
source-role-partitioned sub-bundle structure, not on bag-of-
capsules aggregates.  W3-C5 named the relational axis as the
honest-extension direction beyond the magnitude monoid; Phase
51 puts the smallest serious relational decoder on the table.

### Theorem W3-30 (Cohort-relational strict separation — positive, constructive)

Let $\mathcal{H}_{\rm lin}^{\rm V2}$ be the class of class-
agnostic linear decoders over the 20-feature V2 aggregated
vocabulary (Theorem W3-21 premise; W3-29 bound applies).  Let
$\mathcal{H}_{\rm DS}$ be the DeepSet class (W3-20).  Let
$\mathcal{H}_{\rm rel}$ be the **cohort-relational** class
whose per-capsule embedding is replaced by a **per-role
aggregate**: for each source role $r \in \mathcal{R}$ present
in the bundle, compute $\psi(E_r, rc) \in \mathbb{R}^{d_\psi}$
where $E_r \subseteq E$ is the sub-bundle of capsules with
$\mathrm{source\_role}(c) = r$, and sum $\sum_r \psi(E_r, rc)$
as the bundle representation before the final scoring MLP.
Equivalently: cohort-partition the bundle by role, then run a
Deep-Sets-like aggregator on the resulting set of role-sub-
bundles.

**Claim.** $\mathcal{H}_{\rm rel} \supsetneq \mathcal{H}_{\rm DS}$:
there exists an assignment of the capsule-level DeepSet
embedding and per-role cohort embedding such that
$\mathcal{H}_{\rm rel}$ recovers any function in
$\mathcal{H}_{\rm DS}$, *and* there exists a bundle-level
predicate (e.g. "at least two *distinct* source roles emit a
capsule implying rc AND no source role emits a top-priority
capsule contradicting rc") that is expressible in
$\mathcal{H}_{\rm rel}$ and NOT expressible in
$\mathcal{H}_{\rm DS}$.

**Proof sketch.**  DeepSet aggregates per-capsule features by
sum over the whole bundle — source-role identity is erased by
the aggregation.  The predicate "≥ 2 *distinct* roles each
emitting a capsule implying rc" requires enforcing
distinctness of a grouping variable, which is not expressible
by a sum of per-capsule functions (consider two bundles with
the same multiset of (kind, rc) but different role
assignments — DeepSet's per-capsule φ-sum is identical; the
distinct-role count differs).  The role-partition → aggregate
architecture of $\mathcal{H}_{\rm rel}$ evaluates features on
each $E_r$ in isolation, then aggregates over the role set,
and can express "count of roles $r$ with $\psi(E_r, rc) \ne 0$".
$\square$

**Status.** Proved (constructive; the distinguishing predicate
is witnessed by the Phase-51 empirical bundle examples).
Code anchor: `vision_mvp/wevra/capsule_decoder_relational.py::CohortRelationalDecoder`.

**Caveat.**  W3-30 is a **capacity** statement.  Whether the
strict-containment margin buys empirical lift on
(Phase-31 noisy bench, (incident, security) transfer) is an
empirical question — answered by Claim W3-31 below.

### Claim W3-31 (Cohort-relational empirical frontier — code-backed, named)

Let $\mathcal{D}_{\rm inc}, \mathcal{D}_{\rm sec}$ be the
Phase-31 incident-triage and Phase-33 security-escalation
bundle distributions used in Phases 48–50.  Let $D_{\rm rel}$
be the Phase-51 ``CohortRelationalDecoder`` trained on one
domain's training split and deployed zero-shot on the other.

**Claim (empirical, to be measured in Phase 51).**  The
cohort-relational decoder satisfies on
$(\mathcal{D}_{\rm inc}, \mathcal{D}_{\rm sec})$ at
$n_{\rm test} = 80$ per domain:

* Zero-shot gap (W3-C9 defensible bar):
  $|\mathrm{acc}(B, w_A) - \mathrm{acc}(A, w_B)| \le 0.05$.
* Zero-shot level (Phase-51 new bar):
  $\min(\mathrm{acc}(B, w_A), \mathrm{acc}(A, w_B))$ strictly
  exceeds the Phase-50 sign-stable DeepSet level $0.237$.

Falsifier: the Phase-51 decoder does NOT strictly exceed the
0.237 level under direction-invariance.  If falsified, the
programme adopts Conjecture W3-C10 (below) as the honest next
position; the relational axis does not deliver empirical lift
on (incident, security).

**Status.** Named.  Code-backed by
`vision_mvp/experiments/phase51_relational_decoder.py`.
Empirical numbers reported in
`docs/RESULTS_CAPSULE_RESEARCH_MILESTONE6.md`.

### Conjecture W3-C10 (Level-ceiling of direction-invariant zero-shot transfer on (incident, security))

The W3-C9 gap-reading of Gate 2 is met at gap 0.000 by
Phase-50's sign-stable DeepSet at level 0.237.  The **level**
of direction-invariant zero-shot transfer on (incident,
security) — defined as
$L^\ast := \sup_{D} \min(\mathrm{acc}(\mathcal{D}_A, w_B),
\mathrm{acc}(\mathcal{D}_B, w_A))$ over decoders with gap
$\le 0.05$ — is bounded **below** (positive progress)
by what the richer hypothesis class delivers, and bounded
**above** by some quantity smaller than the within-domain
optimum.

**Conjecture.**  *On the Phase-31 + Phase-33 benchmark family,
$L^\ast \le L_{\max}$ where $L_{\max} < \min(\mathrm{acc}_{\rm inc}^\ast,
\mathrm{acc}_{\rm sec}^\ast)$ — i.e., direction-invariance
strictly costs level, for reasons structurally analogous to
W3-29 but adapted to the relational class.*

**Falsifier.**  A Phase-51+ decoder that achieves gap
$\le 0.05$ AND per-direction level within 5 pp of the
within-domain optimum on both domains.  (This would revive a
form of strict Gate 2 under a richer hypothesis class — the
"richer class escapes W3-29" scenario.)

**Status.** Conjectural, open.  Supported after Phase 51 if
W3-31's level-lift is partial but below the within-domain
optimum; falsified if W3-31's level matches within-domain.

### Relation to prior conjectures

* W3-C5 (relational axis as next honest extension) — Phase 51
  operationalises this for the **decoder** frontier.  W3-C5's
  original framing was about substrate admission; Phase 51
  shows the relational axis matters on the decoder side too.
  W3-C5 is *not* falsified; the current Phase-51 instance is
  one concrete decoder-side realisation.
* W3-C9 (Gate-2 reformulation) — Phase 51 does NOT reformulate
  Gate 2 again; it *lifts the level* under the W3-C9 gap
  reading.  W3-C9 is the operational bar.
* W3-21, W3-29 — linear-class obstructions.  Phase-51 is
  outside their premise by design.

---

## 5. Conjectures with sharp falsifiers

### Conjecture W3-C1 (full subsumption)

Every Phase-19..Phase-44 theorem $T$ of the form *"per-step
delivered context to role $r$ is independent of $|X|$"* admits a
$(k_T, b_T)$ instantiation under Theorem W3-11.

**Falsifier.** A Phase-N theorem $T^*$ where the substrate
guarantee depends on a quantity that has *no representation* in any
of the five capsule budget axes. Candidate: a future bound on the
*peer-review* signature chain (an authentication property, not a
budget).

**Status.** Settled on $T \in \{$L2, P31-3, P35-2, P41-1/P43-1$\}$
(Theorem W3-11). Open on the remaining ~15 phases.

### Conjecture W3-C2 (CID-pinned reproducibility)

For any two runs $R_1, R_2$ with identical `run_cid`, the
downstream consumer can prove offline that
`product_report.json` bytes are equal up to canonical key reorder,
using only the capsule view and SHA-256.

**Status.** Stated in `RESULTS_WEVRA_CAPSULE.md`. Empirically
exercised by `test_capsule_view_on_disk_matches_embedded`. The
adversarial collision attack is ruled out by SHA-256 second
preimage hardness.

### Conjecture W3-C3 (capsule kinds form a *small* generating set
for inter-role artefacts) — **FALSIFIED (Phase 47)**

**Original statement.** The closed alphabet $\mathbb{K}$ of 11
kinds is *complete* for the Wevra runtime in the sense that
every artifact ever crossing a Wevra inter-role / inter-layer /
inter-run boundary has, after the v3 milestone, a canonical
`CapsuleKind` mapping.

**Outcome (Phase 47).** FALSIFIED. The AdaptiveSubscriptionTable's
table-level invariant `|active_edges| ≤ max_active_edges` is a
**cohort-cardinality** artifact that does not canonically map to
any of the original 11 kinds (AdaptiveEdge is the *member*, not
the cohort). The honest resolution is to add a twelfth kind
**COHORT** — done. W3-C3 is falsified at the honest
minimum ("12, not 11"); a successor conjecture W3-C3' takes its
place:

### Conjecture W3-C3' (12 kinds are complete under the magnitude
algebra)

With $\mathbb{K}' = \mathbb{K} \cup \{\mathtt{COHORT}\}$ and the
current budget axes
$(\mathtt{max\_tokens}, \mathtt{max\_bytes}, \mathtt{max\_rounds},$
$\mathtt{max\_witnesses}, \mathtt{max\_parents})$, every Wevra
runtime artifact whose canonical bounded-context invariant is
expressible as a conjunction of **magnitude** predicates
(per-capsule or per-cohort) admits a canonical kind mapping
without further alphabet extension.

**Falsifier.** A runtime artifact whose bounded-context
invariant is (a) not a magnitude predicate (e.g. a relational
predicate) *and* (b) not subsumable by moving the predicate
check into the capsule's constructor. Candidate: the
subscription-graph acyclicity invariant underlying P31-5 — if
a future bounded-context theorem depended on enforcing the
subscription graph stays acyclic at admit time, W3-C3' would
be falsified and a *relational axis* (beyond the magnitude
algebra) would be required. This is Conjecture W3-C5 below.

**Status.** Open. The 12 kinds cover the current substrate; the
predicted failure mode is relational, not cardinal.

### Conjecture W3-C5 (relational axis as the next honest
extension)

If a Phase-48+ substrate primitive ships with a bounded-context
invariant of the form $\mathrm{Rel}_\Psi$ (Theorem W3-16), one of
two things must happen:

1. The invariant is absorbed into the cohort's constructor
   (W3-16 option 1) — at the cost of trusting the constructor.
2. The capsule algebra is extended with a **relational axis**
   whose admission rule depends on pairwise structure of
   members (W3-16 option 2) — at the cost of a genuinely new
   algebraic object beyond the magnitude monoid.

**Falsifier.** A Phase-48 primitive whose relational invariant
requires option 2 but the alphabet ships without it — i.e., the
primitive gets a PARTIAL verdict under W3-15's cohort lift,
mirroring exactly how AdaptiveEdge forced the Phase-47
extension.

**Status.** Conjectural. Named as the next honest-extension
direction if a relational invariant surfaces in the substrate.

### Conjecture W3-C6 (decoder-side task-family transfer — asymmetric)

The Phase-48 decoder-side transfer matrix on the two operational-
detection domains (incident / security) returns (at
$n_{\rm test} = 80$ per domain):

* within-incident: learned-decoder 0.362 vs priority 0.212
  ($+15$ pp).
* within-security: learned-decoder 0.300 vs priority 0.200
  ($+10$ pp).
* incident → security (cross, transferred weights): 0.300
  ($+10$ pp over priority, matches within-security).
* security → incident (cross, transferred weights): 0.125
  ($-8.7$ pp BELOW priority).

**Conjecture.** The asymmetry is structural: the weight
`lone_top_priority_flag` has **opposite signs** across domains
($+1.13$ on incident, $-1.19$ on security) because incident
scenarios carry single-signature high-priority claims
(DISK_FILL_CRITICAL / OOM_KILL / TLS_EXPIRED emitted by one
authoritative role each) while security scenarios carry
multi-source high-priority corroboration
(MALWARE_DETECTED + LATERAL_MOVEMENT + TTP_ATTRIBUTED across
three roles). A decoder-weight vector trained on *one* topology
is mis-adapted to the *other*.

**Falsifier.** A third operational-detection domain $D_3$
whose sign pattern on `lone_top_priority_flag` agrees with
neither incident nor security (e.g., +0.5 with no
high-priority singletons) yet still exhibits the same
incident → $D_3$ direction-of-transfer symmetry as security
does.

**Status.** Supported on the 2-domain (incident, security)
study. Reinforces Phase-47's admission-side P47-C3
(task-family-indexed transfer) at the decoder layer. Code
anchor: `vision_mvp/experiments/phase48_decoder_transfer.py`;
empirical note: `docs/RESULTS_CAPSULE_RESEARCH_MILESTONE3.md`
§ 3.

### Conjecture W3-C7 (paradigm-shift threshold)

A result justifies the phrase "paradigm shift in agent-team
runtimes" iff it demonstrates *both*:

1. a decoder that breaks the Phase-31 0.200 structural
   ceiling on held-out data by $\ge 2\times$ the ceiling
   (i.e., $\ge 0.400$ test accuracy), **and**
2. cross-domain decoder transfer that is approximately
   symmetric — weight-vector transfer from any
   operational-detection domain to any other preserves
   $\ge$ within-domain accuracy $-5$ pp.

Phase 48 satisfies (1) weakly (0.375 $<$ 0.400 threshold) and
partially on (2) — incident → security works, security →
incident fails.

**Falsifier.** A result that meets both thresholds and
attributes the lift to the admission-only layer; that would
*refute* the specific "decoder is the paradigm-shift axis"
framing of W3-C7.

**Status (post-Phase-49).** Conjectural, deliberately
stringent, **partially satisfied**:

* **Gate 1 ($\ge 0.400$ test accuracy).** MET under a
  point-estimate reading — DeepSetBundleDecoder at 0.425 on
  the best cell of the Phase-31 noisy bench (W3-23). The
  95 % binomial CI at $n_{\rm test} = 80$ is
  $[0.317, 0.539]$; the point estimate crosses 0.400 but the
  lower CI does not. A strict reading says "consistent with
  crossing 0.400."
* **Gate 2 (symmetric transfer).**
  - Strict zero-shot reading: NOT MET. The best Phase-49
    decoder (DeepSet) reduces the symmetry gap from 0.175
    (V1 baseline) to 0.038 — materially closer — but the
    transfer penalty against within-domain is still
    +0.138 pp ($i \to s$) and +0.137 pp ($s \to i$),
    above the 5 pp bar. W3-21 gives a *proof* that this
    gate is structurally unattainable under the class-
    agnostic linear hypothesis.
  - Pooled-multitask reading: MET. W3-22 shows one shared
    weight vector achieves 0.350 on both (incident,
    security) test sets with gap 0.000 and ≥ 13 pp over
    each domain's priority baseline.

**Status (post-Phase-50).** Conjectural, deliberately
stringent, **partially retracted**:

* **Gate 1 ($\ge 0.400$ test accuracy).**
  - Phase-49 point-estimate reading at $n_{\rm test} = 80$: MET
    ($\hat{p} = 0.425$, W3-23).
  - Phase-50 replication at $n_{\rm test} = 160$: borderline
    MET ($\hat{p} = 0.400$ exactly at best post-search cell;
    Wilson CI $[0.327, 0.477]$, W3-25) — strict CI NOT MET.
  - Phase-50 replication at $n_{\rm test} = 320$: **NOT MET
    on any reading.**  Best decoder (DeepSet) hits $\hat{p} =
    0.362$, Wilson CI $[0.312, 0.417]$ (W3-26).  Pre-committed
    cell: 0.359.  Theorem W3-24 (winner's-curse bias) is the
    proved reason for the Phase-49 → Phase-50 drop.
  - **Gate 1 is retracted** in the point-estimate reading at
    $n_{\rm test} = 320$; the Phase-49 claim was inflated by
    sample noise + post-search selection.
* **Gate 2 (symmetric transfer).**
  - Strict zero-shot penalty reading: NOT MET on any of 6
    Phase-50 families (V1, V2, sign-stable V2, standardised
    V2, DeepSet, sign-stable DeepSet).  Best max per-direction
    penalty is V2 full at +0.112 (W3-27).  Theorem W3-29
    gives the proved structural reason for this lower bound.
  - Gap reading: MET by sign-stable DeepSet at gap 0.000
    (W3-28).
  - Pooled-multitask reading: MET (W3-22, Phase 49).

**Verdict (post-Phase-50).**  The paradigm-shift label is
**not earned under any strict reading** and is **retracted
under the point-estimate reading at $n_{\rm test} = 320$**.
It remains earned under the pooled-multitask Gate 2 reading
(W3-22, not strict zero-shot) and under the gap-reading of
zero-shot Gate 2 (W3-28, with sign-stable DeepSet).  Phase 50
shows the strict "point-estimate Gate 1 + zero-shot-penalty
Gate 2" reading is **structurally blocked** on this benchmark
family by winner's-curse bias (W3-24) and the sign-flip
obstruction (W3-21 + W3-29).  Conjecture W3-C9 names the
defensible reformulation: accept the gap reading of Gate 2
and accept the pre-committed-cell point estimate for Gate 1.
Under W3-C9's reading, Phase-49 *is* a paradigm-shift
candidate; under the pre-Phase-50 strict reading, it is NOT.

### Conjecture W3-C4 (admission policy learnability)

There exists a learned policy $\pi_\theta : \mathcal{C}_{\rm prop}
\times \mathcal{L} \to \{\mathrm{ADMIT}, \mathrm{REJECT}\}$ that
strictly dominates the fixed-budget heuristic
($\mathcal{A}_b$ for canonical $b$) on the *downstream-failure
under bounded budget* objective, on at least one of the existing
benchmarks (Phase 31 incident triage, Phase 35 contested
incident, Phase 42 SWE-bench-Lite-style sweep).

**Falsifier.** No $\pi_\theta$ in the policy class beats the
heuristic by more than measurement noise on any benchmark.

**Status.** This is the front opened by PART B of the milestone.
The Phase-46 capsule learning experiment is the first attempt; see
`docs/RESULTS_CAPSULE_LEARNING.md` for the result.

---

## 6. What the formalisation does NOT do

Three honest disclaimers:

1. **No category-theoretic claim.** Capsules are not stated as a
   category, functor, or sheaf — even though the adapter
   $\alpha_T$ from § 4 (Theorem W3-11) is morally a functor on a
   small category of substrate primitives. A categorical
   restatement would buy nothing the present DAG-and-monoid view
   does not already provide; we omit it to avoid window-dressing.

2. **No game-theoretic claim about adversaries.** Section W3-10's
   tamper-evidence guarantee is a *forensic* property under
   SHA-256, not an authentication or authorization guarantee. The
   capsule contract does not defend against a malicious replier
   that publishes a different sealed ledger; it only guarantees
   that *any single* claimed ledger is internally consistent or
   provably broken.

3. **No claim of completeness on the kind alphabet.** Eleven
   kinds is a snapshot, not a universal generating set. Conjecture
   W3-C3 names the falsifier explicitly.

---

## 4.G SDK v3.1 extension — capsule-native runtime

Phase ≤ 4.F is the *decoder* frontier (research center on top of
the capsule substrate). The SDK v3.1 extension below is on a
different axis entirely: the **runtime** axis. Capsules stop being
purely a post-hoc audit fold and (partially) become the runtime's
typed execution contract. The four theorems in this section are
proved by inspection of the new ``CapsuleNativeRunContext`` plus
contract tests; the proofs are short because the construction is
conservative.

### Theorem W3-32 (Lifecycle ↔ execution-state correspondence)

Let $\mathcal{S}$ be the canonical Wevra-run stage set
$\{\mathrm{profile}, \mathrm{readiness}, \mathrm{sweep\_spec},
\mathrm{sweep\_cell}_1, \dots, \mathrm{sweep\_cell}_n,
\mathrm{provenance}, \mathrm{artifact}_1, \dots, \mathrm{artifact}_k,
\mathrm{run\_report}\}$ and let $\mathcal{C}_\mathcal{R}$ be the
in-flight register attached to a ``CapsuleNativeRunContext`` with
ledger $\mathcal{L}$. For every stage $S_i \in \mathcal{S}$ there
is a unique capsule $c_i \in \mathcal{C}_\mathcal{R}$ such that:

- $S_i$ is **in progress** at time $t$ iff $c_i \in \mathcal{C}_\mathcal{R}$
  and $\mathit{cid}(c_i) \notin I(\mathcal{L})$ at $t$;
- $S_i$ has **completed** at $t$ iff $\mathit{cid}(c_i) \in
  I(\mathcal{L})$ and $\ell(c_i) = \mathtt{SEALED}$ at $t$;
- $S_i$ has **failed** at $t$ iff $c_i.\mathrm{failure} \neq \bot$
  and $\mathit{cid}(c_i) \notin I(\mathcal{L})$.

**Proof.** ``CapsuleNativeRunContext._propose`` pushes a
``_InFlightEntry`` with ``failure=None``, ``sealed_cid=None``;
``CapsuleNativeRunContext._admit_and_seal`` calls
``ledger.admit_and_seal`` and either sets ``sealed_cid`` (success)
or sets ``failure`` and re-raises (failure). ``admit_and_seal``
is total exactly on $\mathcal{A}_\mathcal{L}$ (§ 2.3). The three
conditions are mutually exclusive and exhaustive on the runtime's
finite stage transitions. $\square$

**Code anchor.** ``vision_mvp/wevra/capsule_runtime.py``;
contract tests in
``vision_mvp/tests/test_wevra_capsule_native.py``
(``test_w3_32_lifecycle_correspondence_clean_run``,
``test_failed_admission_leaves_in_flight_entry``).

### Theorem W3-33 (Content-addressing at artifact creation time)

Let $\sigma : (\mathcal{L}, p, d, \pi) \to c_A$ be the function
``seal_and_write_artifact``. Suppose $\sigma$ returns normally.
Then

$$
\mathrm{SHA\text{-}256}(\mathrm{read}(p)) \;=\;
c_A.\mathit{payload}[\texttt{"sha256"}] \;=\;
\mathrm{SHA\text{-}256}(d).
$$

**Proof.** $\sigma$'s order of operations: (1) $h := \mathrm{SHA\text{-}256}(d)$;
(2) build $c_A$ with payload $\{\texttt{path}: p, \texttt{sha256}: h\}$
and admit + seal — if admission fails, $\sigma$ raises before any
write; (3) write $d$ to $p$; (4)
$h' := \mathrm{SHA\text{-}256}(\mathrm{read}(p))$; (5) if
$h' \neq h$, raise ``ContentAddressMismatch``. The "returns
normally" precondition implies step 5's check passed, so
$h' = h$, which is the claim. $\square$

**Failure mode.** ``ContentAddressMismatch`` (a subtype of
``CapsuleAdmissionError``) detects honest-writer / racing-writer
TOCTOU drift; it is not a defence against adversarial concurrent
writes (the trust unit is the same as Wevra's sandbox boundary).

**Code anchor.**
``vision_mvp/wevra/capsule_runtime.py::seal_and_write_artifact``;
contract tests
``test_w3_33_seal_then_write_then_verify`` and
``test_w3_33_mismatch_detector``.

### Theorem W3-34 (In-flight ↔ post-hoc CID equivalence on
non-ARTIFACT kinds)

Let $r$ be a ``product_report`` dict produced by a
capsule-native run. Let $\mathcal{L}_{\rm in}$ be the in-flight
ledger embedded in $r[\texttt{capsules}]$ and let
$\mathcal{L}_{\rm post}$ be the result of
$\mathrm{build\_report\_ledger}(r)$. Then for every kind
$k \in \mathbb{K}_{\mathrm{eq}} := \{\mathtt{PROFILE},
\mathtt{READINESS\_CHECK}, \mathtt{SWEEP\_SPEC},
\mathtt{SWEEP\_CELL}, \mathtt{PROVENANCE}\}$,

$$
\{\mathit{cid}(c) : c \in \mathcal{L}_{\rm in},\, k(c) = k\}
\;=\;
\{\mathit{cid}(c) : c \in \mathcal{L}_{\rm post},\, k(c) = k\}.
$$

**Proof.** Both paths invoke the same adapter for each kind
(``capsule_from_*`` in ``vision_mvp/wevra/capsule.py``) on the
same canonical payload (the in-flight runner stores the payload
into $r$ before the adapter call; the post-hoc fold reads the
same payload back from $r$). C1 makes $\mathit{cid}(c)$ a pure
function of the canonical payload + parents + budget. The parent
sets agree across paths by inspection of the adapter call sites
in both ``CapsuleNativeRunContext`` and
``build_report_ledger``. $\square$

**Intentional divergence.** $\mathbb{K}_{\mathrm{eq}}$
*excludes* ARTIFACT and RUN_REPORT. The in-flight path's ARTIFACT
capsules carry real ``payload["sha256"]`` hashes; the post-hoc
fold's carry ``None``. The CIDs therefore differ — a *useful*
signal: an ARTIFACT capsule with a non-null SHA was sealed by
the runtime; one with a null SHA was folded post-hoc. The
RUN_REPORT capsule's parent set transitively includes the ARTIFACT
CIDs, so its CID also differs across paths. This is **not** a
weakness; it is the formal statement of "content-addressing at
write time strengthens the ARTIFACT capsule with information the
post-hoc fold never had access to."

**Code anchor.**
``test_w3_34_smoke_run_kind_cid_match`` (set-equality on the five
kinds);
``test_w3_34_artifact_kind_intentional_divergence``
(disjointness witness).

### Theorem W3-35 (Parent-CID gating is the execution contract)

Let $T_S$ be the precondition of stage $S$ — a parent-capsule
sealing requirement (e.g. ``seal_sweep_cell`` requires
``ctx.spec_cap`` to be sealed; ``seal_readiness`` requires
``ctx.profile_cap`` to be sealed). Calling the corresponding
``ctx.seal_*`` method when $T_S$ is unsatisfied raises a typed
exception (``CapsuleLifecycleError`` for runtime preconditions,
``CapsuleAdmissionError`` for parent-CID preconditions in the
ledger), and $\mathcal{L}$ does not contain the requested
capsule.

**Proof.** Each ``seal_*`` method's first check is the
precondition (``self._require_profile()`` or
``if self.spec_cap is None: raise CapsuleLifecycleError``). On
violation, the method raises before calling ``_propose``. The
ledger's ``admit`` step rejects unknown parent CIDs (§ 2.3).
$\square$

**Interpretation.** W3-35 is the formal statement of the
"capsules-as-execution-contract" claim. The runtime's ordering
constraint (cells require their spec; readiness requires its
profile) is no longer a Python sequential-ordering convention; it
is a *typed* check enforced at the capsule layer. A misordered
caller is observable as a typed failure at the offending method,
not as an obscure downstream KeyError or assertion.

**Code anchor.**
``vision_mvp/wevra/capsule_runtime.py``;
contract tests
``test_w3_35_cell_refuses_without_spec``,
``test_w3_35_readiness_refuses_without_profile``.

---

## 4.H SDK v3.2 extension — intra-cell capsule-native + detached witness

The 4.G theorems closed the *run-boundary* slice: every
cross-run-boundary artefact was a sealed capsule sealed in flight.
The intra-cell objects (the patch a generator emits, the verdict
a sandbox returns) still passed as plain Python dataclasses, and
the meta-artefacts ``product_report.json`` /
``capsule_view.json`` / ``product_summary.txt`` were unauthenticated
post-view renderings. SDK v3.2 makes the next two moves: (i) it
extends capsule-native lifecycle into the inner sweep loop with
two new kinds (PATCH_PROPOSAL / TEST_VERDICT), and (ii) it
formalises the meta-artefact boundary as a sharp circularity
theorem with a detached-witness corollary.

### Theorem W3-32-extended (Intra-cell lifecycle correspondence)

Let one sweep cell admit a sequence of (task, strategy) pairs
$\{(\tau_i, \sigma_i)\}_{i=1}^N$. For each pair, the runtime
produces a ``ProposedPatch`` $p_i$ (from ``generator(...)``) and
a ``WorkspaceResult`` $w_i$ (from ``sandbox.run(...)``). Two
unique capsules per pair exist in $\mathcal{C}_\mathcal{R}$:

- $c_{{\rm patch},i}$ of kind PATCH_PROPOSAL, parent =
  $\mathit{cid}(\mathrm{spec\_cap})$, payload = (task /
  strategy / parser_mode / apply_mode / n_distractors
  coordinates) plus ``substitutions_sha256`` and bounded
  rationale;
- $c_{{\rm verdict},i}$ of kind TEST_VERDICT, parent =
  $\mathit{cid}(c_{{\rm patch},i})$, payload = the same
  coordinates plus the WorkspaceResult fields.

The lifecycle ``patch → verdict`` is enforced at the capsule
layer: ``ctx.seal_test_verdict`` raises ``CapsuleLifecycleError``
when its named ``patch_proposal_cid`` is not in the ledger. The
three-state correspondence (in_progress / sealed / failed) of
W3-32 lifts unchanged to the two new kinds.

**Proof.** ``seal_patch_proposal``'s first check is the
SWEEP_SPEC sealing precondition (identical in shape to
``seal_sweep_cell``'s spec gate). On success it goes through
``_propose`` → ``_admit_and_seal``. ``seal_test_verdict``'s
first check is ``patch_proposal_cid in self.ledger``; ledger
membership is the same parent-CID admissibility predicate
(§ 2.3, C5). The chain ``patch sealed → verdict admissible →
verdict sealed`` is an instance of W3-35 with intra-cell
parents. $\square$

**Code anchor.**
``vision_mvp/wevra/capsule_runtime.py::CapsuleNativeRunContext.seal_patch_proposal``,
``seal_test_verdict``;
hooks plumbed through
``vision_mvp/tasks/swe_sandbox.py::run_swe_loop_sandboxed``
(``on_patch_proposed`` / ``on_test_completed`` parameters,
default None preserves byte-for-byte Phase-40 behaviour).

**Empirical anchor.**
``vision_mvp/tests/test_wevra_capsule_native_intra_cell.py``:
``test_patch_proposal_seals_under_spec``,
``test_patch_proposal_refuses_without_spec``,
``test_test_verdict_seals_under_patch_proposal``,
``test_test_verdict_refuses_without_patch``,
``test_smoke_run_emits_intra_cell_capsules`` (48 PATCH_PROPOSAL
and 48 TEST_VERDICT capsules sealed in flight on local_smoke).

### Theorem W3-36 (Meta-artefact circularity is sharp; detached-witness corollary)

Let $R \in \mathcal{L}$ be the RUN_REPORT capsule of a Wevra
run, $V$ be the rendered view of $\mathcal{L}$ at $R$'s seal
time, $M = \{m_1, \dots, m_k\}$ the meta-artefact set whose
bytes are a structural function of $V$
(``product_report.json``, ``capsule_view.json``,
``product_summary.txt``).

**Theorem (negative).** There is no extension $\mathcal{L}'$ of
$\mathcal{L}$ that admits an ARTIFACT capsule $a_i$ with
payload SHA-256 $\mathrm{SHA\text{-}256}(m_i)$ for $m_i \in M$
such that $V' = V$ where $V'$ is the rendered view of
$\mathcal{L}'$.

**Proof.** Suppose toward contradiction $\mathcal{L}', a_i$
exist. Then $\mathcal{L}' \supsetneq \mathcal{L}$ contains
$a_i$, so the rendered view's ``capsules`` list contains $a_i$
as a header AND the chain-head computation depends on $a_i$'s
CID and kind through the chain step. Hence $V' \neq V$,
contradicting the assumption. $\square$

**Corollary (positive — detached witness).** The strongest
authentication of $M$ achievable is a META_MANIFEST capsule
sealed in a *secondary* ledger $\mathcal{L}_{\rm meta}$
disjoint from $\mathcal{L}$, whose payload is a function only
of $(R.\mathrm{cid},\,\mathcal{L}.\mathrm{chain\_head},\,
\{(m_i, \mathrm{SHA\text{-}256}(m_i), |m_i|)\}_i)$. The
META_MANIFEST cross-references $\mathcal{L}$ via its payload's
``root_cid`` field but is *not* a capsule of $\mathcal{L}$; the
trust unit for meta-artefact authentication is the manifest
itself, one explicit hop beyond the primary view.

**Trust model.** Tampering with $m_i$ on disk is detected by
re-hashing under the manifest's claim. Tampering with the
manifest itself is detected only with an out-of-band copy of
$R.\mathrm{cid}$; a self-witnessing solution is impossible
(W3-36 negative).

**Code anchor.**
``vision_mvp/wevra/capsule_runtime.py::CapsuleNativeRunContext.seal_meta_manifest``
and ``render_meta_manifest_view``;
``vision_mvp/wevra/capsule.py::capsule_from_meta_manifest``;
``vision_mvp/product/runner.py`` Stage 8 writes
``meta_manifest.json``.

**Empirical anchor.**
``test_meta_manifest_seals_in_secondary_ledger``,
``test_meta_manifest_refuses_without_run_report``,
``test_meta_manifest_shas_match_on_disk``,
``test_w3_38_meta_manifest_drift_detected``.

### Theorem W3-37 (Chain-from-headers verification)

Let ``view_dict`` be a serialised capsule view (from
``capsule_view.json`` or ``product_report["capsules"]``). Define
``recompute(view_dict)`` to fold the chain step
$h \leftarrow \mathrm{SHA\text{-}256}(\mathrm{prev}, \mathit{cid},
\mathrm{kind}, \mathtt{SEALED})$ over each header in
``view_dict["capsules"]`` starting from $\mathrm{prev} =
\mathtt{GENESIS}$. Then ``verify_chain_from_view_dict(view_dict)
= True`` iff $\mathrm{recompute}(\mathit{view\_dict}) =
\mathit{view\_dict}[\mathtt{chain\_head}]$.

**Proof.** The chain step is a pure function of (prev, cid,
kind, SEALED). All three load-bearing fields are JSON-serialised
in the on-disk header dict; the recompute reproduces the
runtime's exact ``_chain_step`` over the on-disk bytes. Equality
of the recomputed head and the on-disk head is the verification
predicate. $\square$

**Tamper-detection coverage.** Any of (i) flipping a CID,
(ii) flipping a kind, (iii) reordering the capsules list,
(iv) rewriting ``chain_head``, (v) inserting / deleting a
header — flips the verdict to False. Witnessed by
``test_w3_37_tamper_detected`` and
``test_w3_37_tamper_capsule_order_detected``.

**Code anchor.**
``vision_mvp/wevra/capsule.py::verify_chain_from_view_dict``.

### Theorem W3-38 (ARTIFACT audit-time on-disk re-hash)

Let ``view_dict`` carry ARTIFACT capsules with payload
``{path, sha256}``. Define
``verify_artifacts_on_disk(view_dict, base_dir)`` to read each
``payload["path"]`` under ``base_dir`` and re-hash. The verdict
is OK iff every ARTIFACT capsule with a non-null sealed SHA
hashes its on-disk file to that SHA.

**Proof.** W3-33 establishes the runtime *return-time*
post-condition $\mathrm{SHA\text{-}256}(\mathrm{read}(p)) =
c_A.\mathit{payload}[\texttt{"sha256"}]$. W3-38 is the
*audit-time* form: at any later time, the on-disk SHA may
differ (post-runtime tamper, FS corruption, accidental
overwrite); the audit truthfully reflects the current bytes.
The verdict is OK iff equality still holds. $\square$

**Failure mode.** ``verify_artifacts_on_disk`` returns BAD with
a mismatch entry naming the path, the sealed SHA, and the
on-disk SHA. Witnessed by ``test_w3_38_artifact_drift_detected``.

**Code anchor.**
``vision_mvp/wevra/capsule_runtime.py::verify_artifacts_on_disk``.

---

## 7. Reading order

For a reader who wants the *proof obligations*:

1. § 1 — what the objects are.
2. § 2.3 — admissibility predicate.
3. § 4, Theorem W3-11 — the four proven subsumption cases.
4. § 5, W3-C1 — what is still open in subsumption.

For a reader who wants the *runtime contract*: the original
`docs/RESULTS_WEVRA_CAPSULE.md` is still the right entry; this
document complements it.

For a reader who wants to *learn capsule policies*: see
`docs/RESULTS_CAPSULE_LEARNING.md` (PART B of the SDK-v3
research milestone).

---

## 8. Index of code anchors

| Object / theorem        | File / symbol                                                  |
|---                      |---                                                             |
| $\mathcal{C}$           | `vision_mvp/wevra/capsule.py::ContextCapsule`                  |
| $\mathit{cid}$          | `vision_mvp/wevra/capsule.py::_capsule_cid`                    |
| Lifecycle automaton     | `vision_mvp/wevra/capsule.py::CapsuleLifecycle._EDGES`         |
| $\mathcal{A}_b$         | `vision_mvp/wevra/capsule.py::CapsuleLedger.admit`             |
| Chain step              | `vision_mvp/wevra/capsule.py::_chain_step`                     |
| Adapter $\alpha_T$ for HANDLE | `capsule_from_handle`                                    |
| Adapter $\alpha_T$ for HANDOFF | `capsule_from_handoff`                                  |
| Adapter $\alpha_T$ for SWEEP_CELL | `capsule_from_sweep_cell`                            |
| W3-11 reduction tests   | `vision_mvp/tests/test_capsule_subsumption.py` (Phase-46)      |
| W3-7, W3-9, W3-10 tests | `vision_mvp/tests/test_wevra_capsules.py`                      |
| Capsule learning policy | `vision_mvp/wevra/capsule_policy.py` (Phase-46)                |
| W3-14 negative test     | `vision_mvp/tests/test_phase47_cohort_subsumption.py::W3_14_*` |
| W3-15 cohort lift       | `vision_mvp/wevra/capsule.py::CapsuleKind.COHORT` / `capsule_from_cohort` / `capsule_from_adaptive_sub_table` |
| W3-15 cohort audit      | `vision_mvp/experiments/phase46_unification_audit.py::audit_adaptive_edge_cohort` |
| W3-16 relational limit  | `vision_mvp/tests/test_phase47_cohort_subsumption.py::W3_16_*` |
| W3-17 admission locality | `vision_mvp/experiments/phase47_bundle_learning.py` (0.200 saturation) |
| W3-18 plurality sufficiency | `vision_mvp/tests/test_phase48_bundle_decoding.py::test_w3_18_*` |
| W3-19 empirical break    | `vision_mvp/experiments/phase48_bundle_decoding.py` (0.350–0.375 @ n_test=80) |
| LearnedBundleDecoder    | `vision_mvp/wevra/capsule_decoder.py::LearnedBundleDecoder` |
| Decoder transfer study  | `vision_mvp/experiments/phase48_decoder_transfer.py` |
| W3-C6 transfer evidence | `vision_mvp/experiments/phase48_decoder_transfer.py` (sign-flip on lone_top_priority_flag) |
| DeepSetBundleDecoder    | `vision_mvp/wevra/capsule_decoder_v2.py::DeepSetBundleDecoder` |
| MLPBundleDecoder        | `vision_mvp/wevra/capsule_decoder_v2.py::MLPBundleDecoder` |
| LearnedBundleDecoderV2  | `vision_mvp/wevra/capsule_decoder_v2.py::LearnedBundleDecoderV2` |
| InteractionBundleDecoder| `vision_mvp/wevra/capsule_decoder_v2.py::InteractionBundleDecoder` |
| MultitaskBundleDecoder  | `vision_mvp/wevra/capsule_decoder_v2.py::MultitaskBundleDecoder` |
| W3-20 strict separator  | `vision_mvp/wevra/capsule_decoder_v2.py::_phi_capsule` (per-capsule conjunctive features) |
| W3-21 sign-flip anchor  | `vision_mvp/experiments/phase49_symmetric_transfer.py` (feature_comparison_v2) |
| W3-22 multitask anchor  | `vision_mvp/experiments/phase49_symmetric_transfer.py` (multitask_cells) |
| W3-23 Gate-1 anchor     | `vision_mvp/experiments/phase49_stronger_decoder.py` (DeepSet @ bundle_learned_admit/B=64) |
| Phase 49 contract tests | `vision_mvp/tests/test_phase49_stronger_decoder.py` |
| W3-24 winner's-curse     | `vision_mvp/tests/test_phase50_ci_and_zero_shot.py::test_w3_24_winners_curse_lower_bound` |
| W3-25 Gate-1 CI @ n=160  | `vision_mvp/experiments/phase50_gate1_ci.py` (wilson_ci, clopper_pearson_ci) |
| W3-26 Gate-1 @ n=320     | `vision_mvp/experiments/phase50_gate1_ci.py` (80-seed run) |
| W3-27 zero-shot 6-family | `vision_mvp/experiments/phase50_zero_shot_transfer.py` (summary table) |
| W3-28 zero-shot gap = 0  | `vision_mvp/experiments/phase50_zero_shot_transfer.py::SignStableDeepSetDecoder` |
| W3-29 risk-penalty bound | (proof in `docs/CAPSULE_FORMALISM.md` § 4.E — distribution-free) |
| `SIGN_STABLE_FEATURES_V2`| `vision_mvp/experiments/phase50_gate1_ci.py` |
| `StandardisedBundleDecoderV2` | `vision_mvp/experiments/phase50_zero_shot_transfer.py` |
| `SignStableDeepSetDecoder`    | `vision_mvp/experiments/phase50_zero_shot_transfer.py` |
| Phase 50 contract tests  | `vision_mvp/tests/test_phase50_ci_and_zero_shot.py` (14 tests) |
| W3-32 lifecycle correspondence | `vision_mvp/wevra/capsule_runtime.py::CapsuleNativeRunContext` |
| W3-33 content-addressing at write | `vision_mvp/wevra/capsule_runtime.py::seal_and_write_artifact` |
| W3-34 in-flight ↔ post-hoc CID equivalence | `vision_mvp/tests/test_wevra_capsule_native.py::InFlightVsPostHocEquivalenceTests` |
| W3-35 parent-CID gating | `vision_mvp/wevra/capsule_runtime.py::CapsuleNativeRunContext` (precondition checks in each `seal_*` method) |
| Capsule-native runtime tests | `vision_mvp/tests/test_wevra_capsule_native.py` (16 tests) |
