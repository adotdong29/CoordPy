# Theorem W84-T-HONEST-WITNESS-CONSENSUS-ERROR-BOUND (proved)

## Statement

Let `μ ∈ ℝ^d` be a true value. Let `h` honest witnesses report
values `y_i = μ + noise_i` for `i = 1, ..., h`, where each
`noise_i ~ N(0, σ² I_d)` independently. Let `consensus = (1/h)
∑_{i=1}^h y_i` be the unweighted mean of the honest witnesses.

**Assumptions.**

* (A1) Each `noise_i ~ N(0, σ² I_d)`, i.e., centred isotropic
  Gaussian.
* (A2) The noise vectors are mutually independent (iid).
* (A3) No adversarial witnesses (every reported value is
  `μ + noise_i`; no biased reports).

**Theorem.** Under (A1)-(A3), the expected squared L2 error of
the consensus is exactly

```
E[||consensus - μ||_2^2] = d · σ² / h
```

## Proof

```
consensus - μ = (1/h) ∑_i (μ + noise_i) - μ
              = (1/h) ∑_i noise_i
```

The sum of `h` iid Gaussian noise vectors has distribution
`N(0, h σ² I_d)` (sum of variances). Dividing by `h`:

```
(1/h) ∑ noise_i ~ N(0, (σ²/h) I_d)
```

The expected squared L2 norm of a centred Gaussian
`x ~ N(0, Σ)` is `E[||x||²] = tr(Σ)`. With
`Σ = (σ²/h) I_d`, `tr(Σ) = d σ²/h`. Therefore:

```
E[||consensus - μ||²] = d σ² / h
```

QED.

## Empirical sanity check

`tests/test_w84_analytical_bounds.py::
test_w84_proved_honest_consensus_error_bound_holds` samples
`h = 32` honest witnesses with `μ = 0_d`, `σ = 1`, `d = 4`,
averages 200 Monte Carlo trials, and asserts the measured
mean squared error is within ±10% of the theoretical
`d σ² / h = 4/32 = 0.125`.

The bound is the proof's conclusion. The test fails the build
if the proof's predicted value is ever empirically violated by
more than the stated tolerance (the 10% margin absorbs the
Monte Carlo standard error at `n_trials = 200`).

## What this proves

The exact (not "within ε") expected squared error of
unweighted averaging over `h` iid Gaussian honest witnesses.
This is the noise floor of *any* unweighted-mean consensus
mechanism over honest witnesses.

## What it does NOT prove

The bound assumes:
* witnesses are iid Gaussian (no heavy tails);
* no adversarial bias (each witness reports `μ + noise`,
  not `μ + bias + noise`);
* uniform weighting (no trust-weighted variant).

The trust-weighted variant in
`coordpy/integrity_trust_coupled_consensus_v1.py` achieves
this bound when all surviving witnesses have equal trust.
Lower error than `d σ²/h` is impossible under the iid Gaussian
honest model — the bound is tight.
