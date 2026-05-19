# Theorem W84-T-INTEGRITY-FILTERING-VARIANCE-OPTIMAL (proved)

## Statement

Let `H` denote a set of `h` honest witnesses (each reporting
`y_i = μ + noise_i`, `noise_i ~ N(0, σ² I_d)` iid). Let `T`
denote a set of `t` tampered witnesses, each reporting
`y_j = μ + tamper_j` with `tamper_j ∈ ℝ^d` being any
deterministic offset, possibly chosen adversarially.

**Assumption A1 (verifier correctness).** The integrity
verifier outputs `BAD_SIGNATURE` for every tampered witness
and `OK` for every honest one. (This corresponds to the W82
cryptographic-state-integrity property:
`integrity_verdict == OK` iff the witness's payload bytes were
not altered after signing.)

**Assumption A2 (no cross-contamination).** `tamper_j` is
independent of `noise_i` for all `i, j` (the tampered
witnesses do not see the honest noise before deciding their
tampers).

**Theorem.** The hard-drop integrity-trust-coupled consensus
(W83 `integrity_trust_coupled_consensus_v1`) — which drops all
witnesses with `BAD_SIGNATURE` and unweighted-averages the
rest — achieves

```
E[||consensus_filtered - μ||_2^2] = d σ² / h
```

regardless of the choice of `tamper_j` values. In particular,
no adversarial choice of tamper values can increase the
filtered consensus's mean error.

## Proof

By Assumption A1, the verifier flags every `j ∈ T` as
`BAD_SIGNATURE`. The hard-drop consensus drops all such
witnesses. The post-filter consensus is exactly:

```
consensus_filtered = (1/h) ∑_{i ∈ H} y_i
```

— the unweighted mean of the honest witnesses.

By Theorem W84-T-HONEST-WITNESS-CONSENSUS-ERROR-BOUND
(see `W84_honest_witness_consensus_error_bound.md`), the
expected squared L2 error of the unweighted mean of `h` iid
Gaussian honest witnesses is exactly `d σ² / h`.

The `tamper_j` values do not enter the formula. Therefore no
choice of tamper values changes the expected error.

QED.

## Compare to the un-filtered (naive) consensus

Without integrity filtering:

```
consensus_unfiltered = (1/(h+t)) (∑_H y_i + ∑_T y_j)
                     = μ + (1/(h+t)) (∑_H noise_i + ∑_T tamper_j)
```

```
E[||consensus_unfiltered - μ||²]
  = (1/(h+t)²) E[||∑ noise + ∑ tamper||²]
  = (1/(h+t)²) (E[||∑ noise||²] + ||∑ tamper||²)    (A2)
  = (1/(h+t)²) (h σ² d + ||∑ tamper||²)
```

This is monotone-increasing in `||∑ tamper||²`. An adversary
choosing `tamper_j` can drive the unfiltered consensus error
arbitrarily high (in O(t² B²) for a per-tamper magnitude
bound B). The filtered consensus is impervious to this attack.

The conclusion: integrity-trust-coupled hard-drop is
*adversarially robust* on this model, and the filtered error
is exactly the honest-witness noise floor.

## Empirical sanity check

`tests/test_w84_analytical_bounds.py::
test_w84_proved_integrity_filtering_variance_optimal_holds`
constructs a regime with `h = 16` honest witnesses (σ = 1) and
`t = 8` tampered witnesses (each `tamper_j` drawn uniformly
from `[5, 10]^d`, i.e., a large adversarial bias) at
`d = 4`. The filtered consensus's measured mean squared error
across 200 Monte Carlo trials is asserted to be within ±10% of
`d σ² / h = 4 / 16 = 0.25`.

In the same regime, the unfiltered consensus's mean squared
error is also computed and asserted to be SIGNIFICANTLY higher
than the filtered bound — demonstrating that the adversary's
choice has a real, measurable effect on the unfiltered path
but no effect on the filtered path.

## What this proves

A robustness *exactness* claim, not just a bound: the filtered
consensus achieves exactly the honest-witness noise floor,
regardless of the tamper adversary. This is the W83
integrity-trust-coupled consensus's mathematical guarantee.

## What it does NOT prove

The proof assumes the verifier catches every tamper (A1). If
the verifier has a false-negative rate `p > 0`, the filtered
consensus carries `p · t` surviving tampered witnesses in
expectation and the bound becomes
`E[||consensus_filtered - μ||²] ≤ d σ²/(h + (1-p)t) +
||(1-p) ∑ tamper||²/(h + (1-p)t)²`. The exact bound under
imperfect verification is a separate theorem (V2; the W83 V1
verifier is conservatively-correct under the W82
cryptographic-state-integrity scheme, so p ≈ 0 in
practice).
