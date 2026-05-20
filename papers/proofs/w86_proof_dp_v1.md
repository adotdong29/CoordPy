# W86-T-DP-V1-INTEGRITY-COMPOSITION — proof

**Status:** proved-conditional.
**Authored:** 2026-05-20.
**Anchors:** `coordpy.differential_privacy_v1`,
`tests/test_w86_differential_privacy_v1.py`,
`scripts/run_w86_dp_v1_bench.py`,
`results/w86/dp/dp_v1_bench_report.json`.

## Setup

Let `V` be a numeric value on which we want to commit *both*:

1. **Differential Privacy** with budget `ε > 0` (and optionally
   `δ ∈ [0, 1)`) via the Laplace or Gaussian mechanism;
2. **Integrity** via a content-addressed Merkle anchor over the
   committed bytes.

Naively combining them is dangerous:

* If the integrity anchor commits to the *cleartext* `V`, an
  observer with the anchor + the perturbed value `V'` can
  recover bits about `V` (anchor leakage), defeating DP.
* If DP is applied *after* the anchor is computed, the anchor
  no longer verifies the bytes that ship downstream.

V1 solves this by anchoring the DP capsule itself:

```
1. ε,δ,sensitivity → MechanismParams                 (CID m)
2. V → +noise(m) → V'                                (perturbed)
3. DPCapsuleV1 = {V', m, noise_seed_cid, ε, δ}       (CID c)
4. integrity_anchor = SHA256({c, tenant_id})         (CID a)
```

The anchor commits to `c` (the DP capsule's CID), not to `V`.

## Theorem 1 — DP guarantee (W86-T-DP-V1-LAPLACE-PURE)

**Claim.** For any two adjacent inputs `V_1, V_2` with
`|V_1 − V_2| ≤ Δ` (sensitivity) and any measurable event `E`:

`Pr[V' ∈ E | V_1] / Pr[V' ∈ E | V_2] ≤ exp(ε)`.

**Proof.** The Laplace mechanism's pdf is
`p(x; μ, b) = exp(−|x − μ| / b) / (2b)` with `b = Δ / ε`.
For any output `V' = y`:

```
p(y | V_1) / p(y | V_2)
  = exp(−|y − V_1| / b) / exp(−|y − V_2| / b)
  = exp((|y − V_2| − |y − V_1|) / b)
  ≤ exp(|V_1 − V_2| / b)
  ≤ exp(Δ / b)
  = exp(ε).
```

For an event `E`, the ratio of probabilities is bounded by the
pointwise ratio bound, so `Pr[E | V_1] ≤ exp(ε) · Pr[E | V_2]`.
∎

This is the classical Laplace-mechanism guarantee (Dwork et al.
2006, Theorem 3.6). The Gaussian mechanism with σ as in
`DPMechanismParamsV1.noise_scale()` gives `(ε, δ)`-DP by Dwork
et al. 2014, Theorem A.1.

## Theorem 2 — Basic composition (W86-T-DP-V1-COMPOSITION)

**Claim.** If `n` queries `Q_1, …, Q_n` each satisfy `(ε_i, δ_i)`-DP
on adjacent datasets, then their sequential composition satisfies
`(Σε_i, Σδ_i)`-DP.

**Proof.** Apply Theorem 3.16 of Dwork & Roth 2014 (sequential
composition for `(ε, δ)`-DP). ∎

`DPBudgetTrackerV1` enforces `Σε_i ≤ ε_total` and `Σδ_i ≤ δ_total`
strictly: any request that would exceed the total is refused
(``request_spend`` returns False, the refusal is recorded as a
``DPBudgetBreachEventV1``).

## Theorem 3 — Integrity-anchor composition with DP
(W86-T-DP-V1-INTEGRITY-COMPOSITION)

**Claim.** Anchoring the DP capsule by hashing its CID (rather
than the original `V`) preserves DP. Specifically, the joint
distribution `(c, a) = (DPCapsule.cid(), anchor(c, tid))`
satisfies the same `(ε, δ)`-DP guarantee as `(V', m, noise_seed_cid,
ε, δ)` alone.

**Proof.** The anchor `a = H({c, tid})` is a deterministic
function of `c` and the tenant id `tid`. The post-processing
theorem of DP (Dwork & Roth 2014, Proposition 2.1) states: if
`M(V)` is `(ε, δ)`-DP and `f` is any function, then
`f(M(V))` is also `(ε, δ)`-DP.

Define `M(V) = c = DPCapsuleV1.cid(V, m, noise_seed)`. `M` is
`(ε, δ)`-DP by Theorem 1 (the noise is on `V'`, the capsule
includes `V'` only, no cleartext `V`), and SHA-256 hashing is
a deterministic function. Hence `f(M(V)) = a` is also
`(ε, δ)`-DP. ∎

Note: the anti-cheat clause "Do not store the un-perturbed
payload" is enforced syntactically — `DPCapsuleV1` carries
`perturbed_value`, `mechanism_params_cid`, `noise_seed_cid`,
`ε_spent`, `δ_spent`, `label` but NEVER a cleartext field.
The bench reports `raw_value_not_in_capsule_dict = True`.

## Theorem 4 — PII redactor span-CID is non-leaky
(W86-T-DP-V1-REDACTION-AUDIT-NOT-LEAKY)

**Claim.** `RedactionEventV1.spans_redacted_cid` is a SHA-256 of
the *span tuples* `[(start_i, end_i)]`, not of the redacted text.
Hence an adversary with the redaction event and the redacted
output cannot recover the original text characters at the
redacted positions, beyond what the redacted text already leaks
(token boundary, length).

**Proof.** The CID is computed over the deterministic JSON of
`{pattern_name, spans}` only. Under SHA-256 preimage resistance
(standard hardness assumption), an adversary cannot recover any
of the original byte values at the redacted positions from
`spans_redacted_cid` alone, nor from `spans_redacted_cid` plus
the redacted text. ∎

## Sanity checks

* Test `test_dp_capsule_does_not_store_raw_value` enumerates
  the capsule's dict keys and confirms no `raw_value`,
  `original_value`, or `cleartext_value` field exists.
* Test `test_budget_tracker_refuses_overflow` requests ε=0.5,
  ε=0.4, then ε=0.3 against a 1.0 total; the third call is
  refused. This is the literal "over-budget queries are
  refused with explicit audit" requirement.
* Test `test_utility_curve_monotonic_ε_increasing` confirms
  that on a fixed `(V, sensitivity)`, doubling ε halves the
  Laplace noise scale, halving the mean absolute error. The
  bench reports points at ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0} with
  1000 samples each — the "utility-vs-privacy curve" the DoD
  asks for.

## Honest scope

The proofs assume:

1. SHA-256 preimage resistance (standard cryptographic
   assumption).
2. Floating-point underflow / catastrophic cancellation does
   not break the Laplace cdf bound (`fp64` arithmetic is
   adequate at the bench's noise scales).
3. The noise seed is generated from a CSPRNG
   (`secrets.token_bytes`); the seed itself is content-
   addressed but the bytes are NOT stored in the capsule.
4. Basic composition; Rényi DP / advanced composition is V2
   work and would tighten the bound at high `n`.
5. Numeric DP only; categorical DP is V2 (would use the
   exponential mechanism).

These limitations are tracked as `W86-L-DP-V1-*` rows in
``docs/THEOREM_REGISTRY.md``.
