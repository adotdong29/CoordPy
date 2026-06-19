# W84-T-TRUST-WEIGHTED-CONSENSUS-BOUND — proof

**Status:** proved-conditional.
**Authored:** 2026-05-19.
**Anchors:** `coordpy.adversarial_consensus_repair_v1`,
`tests/test_w81_adversarial_consensus_repair.py`,
`tests/test_w84_analytical_bounds.py::test_w84_consensus_bound_holds_on_w81_bench`.

## Statement

Let $W = \{w_1, \dots, w_n\}$ be a set of $n$ scalar witnesses
arranged around an unknown ground-truth value $\mu \in \mathbb{R}$.
Suppose:

* **(Honest noise).** Honest witnesses $h \subseteq W$ with
  $|h| = n - f$ are iid Gaussian: $w_i \sim \mathcal{N}(\mu,
  \sigma^2)$ for $i \in h$.
* **(Adversarial bias bound).** Corrupt witnesses $c = W \setminus h$
  with $|c| = f$ have arbitrary, but bounded, bias:
  $|w_i - \mu| \leq B$ for $i \in c$, with $B \geq 0$ a constant
  declared at consensus time.
* **(Trust weighting).** The W81
  ``adversarial_consensus_repair_v1`` controller assigns each
  witness a non-negative trust weight $t_i$ with $\sum_i t_i = 1$.
  The controller hard-drops corruption-suspect witnesses (its
  ``corruption_penalty`` parameter); call $\hat{c} \subseteq c$
  the set of *detected* corrupt witnesses. We do NOT require that
  the controller detects all of $c$.
* **(Strict-minority constraint).** $f < n / 2$.

Let $\hat{\mu} = \sum_i t_i w_i$ be the trust-weighted consensus.
Then

$$
\mathbb{E}[(\hat{\mu} - \mu)^2]
\;\leq\;
\sigma^2 \cdot \sum_{i \in h} t_i^2
\;+\;
B^2 \cdot \left(\sum_{i \in c \setminus \hat{c}} t_i\right)^2.
\quad (\dagger)
$$

In words: the expected squared error is at most the variance of
the trust-weighted honest mean (first term) plus a bias squared
from the *undetected* corrupt witnesses (second term).

## Proof

Decompose the trust-weighted estimate around $\mu$:

$$
\hat{\mu} - \mu
\;=\;
\sum_{i \in h} t_i (w_i - \mu)
\;+\;
\sum_{i \in c \setminus \hat{c}} t_i (w_i - \mu).
$$

(The sum over $\hat{c}$ vanishes because hard-dropped witnesses
get $t_i = 0$ by construction in
``adversarial_consensus_repair_v1``.)

Let $X = \sum_{i \in h} t_i (w_i - \mu)$ and
$Y = \sum_{i \in c \setminus \hat{c}} t_i (w_i - \mu)$.

Then $(\hat{\mu} - \mu)^2 = X^2 + 2 X Y + Y^2$, and

$$
\mathbb{E}[(\hat{\mu} - \mu)^2]
\;=\;
\mathbb{E}[X^2] + 2 \mathbb{E}[X Y] + \mathbb{E}[Y^2].
$$

* **Bound on $\mathbb{E}[X^2]$.** $X$ is a weighted sum of $n - f$
  independent zero-mean Gaussians with variance $\sigma^2$; therefore
  $\mathbb{E}[X^2] = \sigma^2 \sum_{i \in h} t_i^2$.
* **Bound on $\mathbb{E}[Y^2]$.** $Y$ is a fixed weighted sum
  (adversary chooses bias deterministically) so
  $Y^2 = \big(\sum_{i \in c \setminus \hat{c}} t_i (w_i - \mu)\big)^2 \leq B^2 \big(\sum_{i \in c \setminus \hat{c}} t_i\big)^2$
  by Cauchy–Schwarz and the $|w_i - \mu| \leq B$ constraint.
* **The cross term.** $\mathbb{E}[XY] = Y \cdot \mathbb{E}[X] = 0$
  because $X$ is zero-mean and $Y$ is deterministic given the
  adversary's choice. (The adversary can correlate $Y$ to the
  trust weights $t$; but $t$ is chosen *before* seeing $w_i$
  values for $i \in h$, so $X$ remains zero-mean conditionally.)

Combining: $(\dagger)$ holds. $\square$

## Remarks

* The bound is **tight** when the adversary realises the maximum-
  bias choice and the controller fails to drop any of $c$: in that
  worst case $Y^2 = B^2 \big(\sum_{i \in c} t_i\big)^2$, and the
  bound becomes $\sigma^2 \sum_h t_i^2 + B^2 (\sum_c t_i)^2$.
* The bound **collapses to $\sigma^2 / n$** when $t_i = 1/n$ and
  $\hat{c} = c$ — the W81 hard-drop-on-corruption behaviour
  saturates this best case.
* The strict-minority constraint $f < n/2$ is **not strictly
  required** by the bound itself (the bound holds for any $f$); it
  is required by the W81 controller's *detection* heuristic, which
  assumes corruption is a minority. The bound makes no use of
  detection rate beyond the explicit $|c \setminus \hat{c}|$
  appearance.
* The proof's adversary model is **strong**: the adversary picks
  the values $w_i$ for $i \in c$ adversarially, knowing the trust
  weights but not the honest realisations.

## Empirical sanity check

The W81 bench
``tests/test_w81_adversarial_consensus_repair.py`` measures
empirical mean error on $n=7$, $f=2$ over 100 seeds. The bound
applied at $\sigma = 0.20$, $B = 0.40$, $t_i \approx 1/n$
predicts a worst-case MSE of

$$
0.04 \cdot 7 \cdot (1/7)^2 + 0.16 \cdot (2/7)^2
\;\approx\; 0.0057 + 0.0131 \;\approx\; 0.019,
$$

so RMSE $\lesssim 0.137$. The bench's empirical RMSE is well
below this on 100 seeds; the empirical sanity test
``test_w84_consensus_bound_holds_on_w81_bench`` confirms this
non-violation.
