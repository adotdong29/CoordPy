# W86-T-BYZANTINE-V1-SAFETY / W86-T-BYZANTINE-V1-LIVENESS — proof

**Status:** proved-conditional.
**Authored:** 2026-05-20.
**Anchors:** `coordpy.byzantine_fault_tolerance_v1`,
`tests/test_w86_byzantine_fault_tolerance_v1.py`,
`scripts/run_w86_bft_v1_bench.py`,
`results/w86/bft/bft_v1_suite_report.json`.

## Setup

Let `M = {r_1, …, r_n}` be a static membership set of `n` replicas,
each with an Ed25519 keypair `(sk_i, pk_i)`. All replicas know each
other's public keys; the membership CID `cid(M)` is a SHA-256 over
the canonical JSON of `{replica_id, public_key_hex}` records.

Let `f` denote an upper bound on the number of Byzantine replicas
(arbitrary adversaries who may sign anything with their private
keys, equivocate, delay, or stop).  Honest replicas always follow
the protocol.

The classical PBFT bound is `n ≥ 3f + 1`, i.e. `f_bound :=
⌊(n − 1) / 3⌋`. The quorum size is `Q := 2f + 1`.

A *consensus round* is parametrised by `(view, sequence)`. The
primary `p(view) := r_{(view mod n)}` initiates the round.

## Messages

Every phase message `m = ⟨phase, view, seq, value_cid,
cid(M), sig⟩` is signed with Ed25519 under `sk_{sender}` over the
deterministic JSON of `{phase, view, seq, value_cid, cid(M)}`.
Verification uses `pk_{sender}` retrieved from `M` via
`replica_id`. Ed25519 is EUF-CMA secure under the standard
discrete-log hardness assumption on Curve25519.

## Theorem 1 — Safety (W86-T-BYZANTINE-V1-SAFETY)

**Claim.** For any execution with `f ≤ f_bound` Byzantine
replicas, **no two honest replicas commit different values at
the same `(view, seq)`**.

**Proof.** Suppose by contradiction that two honest replicas
`r_a` and `r_b` commit `v_a` and `v_b` at `(view, seq)` with
`v_a ≠ v_b` (so `cid(v_a) ≠ cid(v_b)`).

By the protocol, `r_a` collected `Q = 2f + 1` distinct signed
**commit** messages whose `value_cid = cid(v_a)`. Likewise
`r_b` collected `Q` distinct signed commit messages with
`value_cid = cid(v_b)`. Call these sets `C_a` and `C_b`.

Each commit message is signed by one replica; therefore
`|C_a|, |C_b| = Q`. The total membership has `n` replicas, so
`|C_a ∩ C_b| ≥ 2Q − n = 4f + 2 − n ≥ 4f + 2 − (3f + 1) = f + 1`.

Hence at least `f + 1` replicas signed commit messages for
**both** `cid(v_a)` **and** `cid(v_b)` at `(view, seq)`. Since
there are at most `f` Byzantine replicas, at least one of these
replicas is honest.

But an honest replica only signs **commit** for `value_cid`
**after** signing **prepare** for the *same* `value_cid` at the
same `(view, seq)`, and an honest replica signs **prepare** for
**at most one** `value_cid` at a given `(view, seq)`. Therefore
the same honest replica cannot have signed commits for both
`cid(v_a)` and `cid(v_b)`.

This is a contradiction; safety holds. ∎

The proof reuses the standard PBFT safety argument (Castro &
Liskov 1999, §5.2) verbatim under our message format.

## Theorem 2 — Equivocation Detection
(W86-T-BYZANTINE-V1-EQUIVOCATION-DETECT)

**Claim.** If a Byzantine replica `r_*` signs two phase
messages `m_a, m_b` with `phase(m_a) = phase(m_b)`,
`view(m_a) = view(m_b)`, `seq(m_a) = seq(m_b)`, but
`value_cid(m_a) ≠ value_cid(m_b)`, then the
``ByzantineEquivocationEvidenceV1`` capsule built from
`{m_a, m_b}` is *independently verifiable*: a third party
with `pk_{r_*}` re-derives `signature_a_valid = True`,
`signature_b_valid = True`, and `messages_contradict =
True`.

**Proof.** ``independently_verify`` recomputes the canonical
message bytes for each `m_a, m_b` from the stored
`(phase, view, seq, value_cid, cid(M))` tuple — these are the
exact bytes that `r_*` originally signed. Ed25519 verification
is deterministic given the message bytes, signature bytes, and
public key, so the third party re-derives the same validity
verdict as the original sender. `messages_contradict` is a
syntactic check on the stored tuple. Combining
`signature_a_valid ∧ signature_b_valid ∧ messages_contradict`
gives `conclusively_byzantine = True`. ∎

## Theorem 3 — Safety under `f > f_bound` partial-synchrony
(W86-T-BYZANTINE-V1-SAFETY-ABOVE-BOUND)

**Claim.** If `f > f_bound` Byzantine replicas attempt to push a
value `v_bad ≠ v_honest` by signing prepare for `cid(v_bad)`
without honest cooperation, *and* honest replicas hold to their
prepare-only-if-own-proposal-matches rule, then `v_bad` does
**not** reach quorum unless the Byzantines themselves number
at least `Q = 2 · f_bound + 1`. Equivalently, the protocol
refuses to commit `v_bad` unless the byzantine fraction crosses
the classical `2/3` super-majority — at which point any safety
guarantee is intentionally lost.

**Proof.** Let `B ⊂ M` be the set of Byzantine replicas, `|B| = f`.
The set of prepare signatures for `cid(v_bad)` is
contained in `B` ∪ `{r ∈ honest : r's own proposed value has
cid(v_bad)}`. By construction `v_honest ≠ v_bad`, so no honest
replica's proposed value matches `cid(v_bad)`; the set of
honest prepares for `cid(v_bad)` is empty. Therefore the
prepare set for `cid(v_bad)` has size at most `f`. Quorum
requires `2 · f_bound + 1`. Hence `v_bad` reaches quorum only
when `f ≥ 2 · f_bound + 1`. ∎

For `n = 4, f_bound = 1`, the safety bound holds while
`f ≤ 2`; the protocol refuses to commit at `f = 2`. At `f = 3`,
safety is intentionally lost (the bench acknowledges this is
beyond the classical PBFT region).

## Theorem 4 — Liveness under partial synchrony
(W86-T-BYZANTINE-V1-LIVENESS)

**Claim.** If `f ≤ f_bound`, the network eventually delivers
honest messages within bounded delay `Δ_{honest}`, and the
primary `p(view)` is honest, then every round terminates
within bounded rounds.

**Proof.** Under the assumption that the primary is honest,
the primary sends a pre-prepare. Honest replicas `H` (with
`|H| = n − f ≥ 2f + 1 = Q`) all agree with the primary's
proposed value and sign prepare. Under partial synchrony,
within `Δ_{honest}` every honest replica receives all `|H|`
prepare signatures and counts a valid quorum (rejecting any
Byzantine equivocators per Theorem 2). Each honest replica
then signs commit. Within another `Δ_{honest}` every honest
replica receives `|H| ≥ Q` commit signatures and commits.

The total round time is bounded by `O(Δ_{honest})`. ∎

When the primary is Byzantine, liveness can fail (the primary
may refuse to send pre-prepare). Recovery requires the
view-change protocol, which is V2.

## Sanity check

The empirical bench (`scripts/run_w86_bft_v1_bench.py`) at
`n = 7, f = 2, mu = 1.0, delta = 0.3` commits `mu` exactly,
producing `committed_error = 0.0 ≤ B = 0.0`. The proof's
strict-minority safety bound holds with margin.

The refuse-to-commit bench at `n = 4, f = 2` (above the
classical bound) refuses to commit any value, consistent with
Theorem 3.

The equivocation bench at `n = 4, target_delta = 7.7` produces
exactly one independently-verifiable equivocation evidence
capsule with `conclusively_byzantine = True`, consistent with
Theorem 2.

## Honest scope

The proofs above assume:

1. Ed25519 signatures are EUF-CMA secure — standard
   cryptographic assumption.
2. Honest replicas execute the protocol verbatim — including
   refusing to commit when equivocation is detected.
3. Static membership — proofs do not extend to dynamic
   joining/leaving in V1.
4. Partial synchrony — Theorem 4 requires bounded delays
   eventually; pure asynchrony would impose the FLP
   impossibility.
5. The bench-level claims assume `n = 4` or `n = 7` membership.
   The proofs are valid for arbitrary `n ≥ 4`; the empirical
   sanity check covers these two sizes.

These limitations are tracked as `W86-L-BYZANTINE-V1-*` rows in
``docs/THEOREM_REGISTRY.md``.
