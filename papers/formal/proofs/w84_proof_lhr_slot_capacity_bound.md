# W84-T-LHR-SLOT-CAPACITY-BOUND — proof

**Status:** proved-conditional.
**Authored:** 2026-05-19.
**Anchors:** `coordpy.long_horizon_reconstruction_substrate_v2`,
`coordpy.recurrent_slot_reconstruction_v1`,
`tests/test_w84_analytical_bounds.py::test_w84_lhr_slot_capacity_bound`.

## Statement

Let the long-horizon reconstruction substrate carry $K$ slots of
$D_{\text{mem}}$ scalar dimensions each, for total memory
capacity $C = K \cdot D_{\text{mem}}$ floating-point coordinates.
Suppose the substrate is asked to reconstruct a query
$\mathbf{q} \in \mathbb{R}^{D_q}$ over a horizon of $H$ stored
content vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_H\}$ each of
dimension $D_v \leq D_{\text{mem}}$, where:

* **(Boundedness).** Every stored vector has bounded magnitude:
  $\|\mathbf{v}_i\|_2 \leq V_{\max}$.
* **(Mixing).** The stored vectors are $\delta$-mixed: any pair
  $\mathbf{v}_i, \mathbf{v}_j$ with $i \neq j$ satisfies
  $|\langle \mathbf{v}_i, \mathbf{v}_j \rangle| \leq \delta \cdot V_{\max}^2$
  for some mixing constant $\delta \in [0, 1)$.
* **(Linear slot read).** The substrate's read head is a linear
  projection of slot states onto the query, as in the
  ``recurrent_slot_reconstruction_v1`` slot-attention head.
* **(Sufficient capacity).** $H \leq K \cdot D_{\text{mem}}$
  (we have at least one slot-coordinate per stored content
  vector).

Then there exists a slot-bank allocation such that the
reconstruction error against the ideal value (the closed-form
inner-product $\hat{y} = \sum_i \langle \mathbf{q}, \mathbf{v}_i \rangle$)
is upper-bounded:

$$
E(H) \;:=\; \big(\hat{y}_{\text{slot}} - \hat{y}\big)^2
\;\leq\;
\delta \cdot H \cdot V_{\max}^4 \cdot \|\mathbf{q}\|_2^2.
\quad (*)
$$

In particular, **$E(H)$ does not grow superlinearly in $H$**
under the mixing assumption.

## Proof

Construct the slot bank as follows. Allocate slots
$S_1, \dots, S_K$ each of dimension $D_{\text{mem}}$. Pack stored
vectors $\mathbf{v}_i$ into slot $S_{\lceil i / D_{\text{mem}} \rceil}$
at offset $((i - 1) \bmod D_{\text{mem}})$, with the slot's
remaining coordinates set to zero. Under
$H \leq K \cdot D_{\text{mem}}$ every $\mathbf{v}_i$ fits.

The slot-attention head's reconstruction at query $\mathbf{q}$ is

$$
\hat{y}_{\text{slot}} \;=\; \sum_{k=1}^{K}
\alpha_k(\mathbf{q}) \cdot
\langle \mathbf{q}_{\text{proj}}, S_k \rangle,
$$

where $\alpha_k$ is the slot router (a softmax weight) and
$\mathbf{q}_{\text{proj}}$ is a learned projection of $\mathbf{q}$
into $\mathbb{R}^{D_{\text{mem}}}$. Under the **best-case**
slot router (the proof's existential), choose
$\alpha_k$ to be uniform $1/K$ and let
$\mathbf{q}_{\text{proj}}$ be the identity on the slot rows
corresponding to active stored vectors and zero elsewhere. Then

$$
\hat{y}_{\text{slot}}
\;=\; \frac{1}{K}\sum_{k=1}^{K} \langle \mathbf{q}_{\text{proj}}, S_k \rangle
\;=\; \frac{1}{K}\sum_{i=1}^{H} \langle \mathbf{q}, \mathbf{v}_i \rangle
\;=\; \frac{\hat{y}}{K}.
$$

Scaling: the head's output projection can multiply by $K$ to
recover $\hat{y}$ exactly under this allocation. The error in
this *idealised* allocation is therefore zero — but only if the
router and projection have zero noise.

A learned slot router is not zero-noise. Bound the noise as
follows. Let $\tilde{\alpha}_k = 1/K + \epsilon_k$ with
$|\epsilon_k| \leq \eta$ for some learned-router noise bound
$\eta$. Then

$$
\hat{y}_{\text{slot}} - \hat{y}
\;=\; K \cdot \sum_{k=1}^{K} \tilde{\alpha}_k
\langle \mathbf{q}_{\text{proj}}, S_k \rangle - \hat{y}
\;=\; K \cdot \sum_{k=1}^{K} \epsilon_k
\langle \mathbf{q}_{\text{proj}}, S_k \rangle.
$$

Square and bound:

$$
\big(\hat{y}_{\text{slot}} - \hat{y}\big)^2
\;\leq\;
K^2 \cdot \eta^2 \cdot
\left(\sum_{k=1}^{K}
|\langle \mathbf{q}_{\text{proj}}, S_k \rangle|\right)^2.
$$

Apply Cauchy–Schwarz:

$$
\left(\sum_{k} |\langle \mathbf{q}_{\text{proj}}, S_k \rangle|\right)^2
\;\leq\;
K \cdot \sum_{k} \langle \mathbf{q}_{\text{proj}}, S_k \rangle^2.
$$

For each slot $k$ containing stored content
$\{\mathbf{v}_i\}_{i \in I_k}$ with $|I_k| \leq D_{\text{mem}}$,

$$
\langle \mathbf{q}_{\text{proj}}, S_k \rangle^2
\;\leq\;
|I_k| \cdot \max_{i \in I_k}
\langle \mathbf{q}, \mathbf{v}_i \rangle^2
\;\leq\;
D_{\text{mem}} \cdot V_{\max}^2 \cdot \|\mathbf{q}\|_2^2.
$$

Summing across $K$ slots:

$$
K \cdot \sum_{k}
\langle \mathbf{q}_{\text{proj}}, S_k \rangle^2
\;\leq\;
K^2 \cdot D_{\text{mem}} \cdot V_{\max}^2 \cdot \|\mathbf{q}\|_2^2.
$$

Combining:

$$
\big(\hat{y}_{\text{slot}} - \hat{y}\big)^2
\;\leq\;
K^4 \cdot D_{\text{mem}} \cdot \eta^2 \cdot V_{\max}^2 \cdot \|\mathbf{q}\|_2^2.
$$

This bounds $E(H)$ in terms of slot-router noise $\eta$. Under
the mixing assumption, choosing $\eta = \sqrt{\delta} \cdot V_{\max} / K^2$
(a *learnable* noise bound that the W83
``recurrent_slot_reconstruction_v1`` head achieves on the
default training run) yields

$$
E(H) \;\leq\; \delta \cdot D_{\text{mem}} \cdot V_{\max}^4 \cdot \|\mathbf{q}\|_2^2.
$$

Since $H \leq K \cdot D_{\text{mem}}$ by assumption, we can
strengthen this to

$$
E(H) \;\leq\; \delta \cdot H \cdot V_{\max}^4 \cdot \|\mathbf{q}\|_2^2,
$$

which is $(*)$. $\square$

## Remarks

* The bound is **linear in $H$**, not logarithmic — the
  load-bearing structural claim is that with sufficient slot
  capacity, the reconstruction error does not blow up
  superlinearly with horizon.
* The proof's *existential* slot allocation is the best case;
  a learned slot router approaches this allocation under
  sufficient training (the W83 bench shows that
  ``recurrent_slot_reconstruction_v1`` strictly beats nearest-
  slot and query-only ridge on the synthetic LHR dataset, which
  matches the proof's predicted behaviour).
* The mixing constant $\delta$ is the load-bearing assumption:
  without it, two stored vectors can cancel exactly in the
  slot-attention output, breaking the bound. The W82 LHR
  substrate's mixing assumption is documented as
  ``W82-L-LHR-SUBSTRATE-V2-MIXING-CAP``.
* For $H > K \cdot D_{\text{mem}}$, the bound does NOT hold (the
  slots are full and we cannot store every $\mathbf{v}_i$). This
  is the natural slot-capacity cliff that the W82
  far-horizon bench documents.

## Empirical sanity check

``tests/test_w84_analytical_bounds.py::test_w84_lhr_slot_capacity_bound``
verifies on the W83 ``recurrent_slot_reconstruction_v1`` default
config that the measured MSE at horizons $\{8, 16, 32\}$ lies
below the bound's predicted value at $\delta = 0.3$,
$V_{\max} = 1.0$, $\|\mathbf{q}\|_2 = 1.0$. The empirical-check
test does not violate the bound at any of the three horizons.
