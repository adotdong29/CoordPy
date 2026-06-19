# W84-T-REPLAY-FROM-KV-EXACT — proof

**Status:** proved-conditional.
**Authored:** 2026-05-19.
**Anchors:** `coordpy.controlled_runtime_substrate_v1`,
`coordpy.streaming_substrate_intercept_v1`,
`tests/test_w79_controlled_runtime_substrate.py`,
`tests/test_w84_analytical_bounds.py::test_w84_replay_from_kv_exact_byte_identity`.

## Statement

Let $\mathcal{R}$ be a controlled transformer-class runtime
satisfying:

* **(Causal attention).** Layer $L$'s attention mask is the
  standard lower-triangular causal mask:
  $\text{mask}[t, s] = -\infty$ for $s > t$, $0$ otherwise.
* **(Content-addressed KV writes/reads).** At each forward step
  the runtime appends the new tokens' $(K, V)$ projections to a
  layer-wise cache; the cache is read at the *exact* positions
  the source forward wrote them.
* **(Deterministic float arithmetic).** All arithmetic is fp32
  IEEE-754 with deterministic associativity (no Tensor-Core
  fast-paths; no GPU `__shfl_*` cross-lane sums; no fused
  multiply-add reordering across passes).
* **(Frozen weights).** The runtime's parameters are read-only
  between the source forward and the replay-from-KV.
* **(No injection).** No hidden-state, attention-bias, or
  prefix-state injection is applied at either forward.

Let $T_0$ be the original forward over a prompt of length
$N_0$, producing logits $L_0 \in \mathbb{R}^{N_0 \times V}$
(where $V$ is the vocab size) and KV cache $K_0$. Let
$T_1$ be the replay-from-KV forward over new tokens of length
$N_1$, starting from $K_0$, producing logits
$L_1 \in \mathbb{R}^{N_1 \times V}$. Let $T_2$ be the
**recompute** forward over the same $(N_0 + N_1)$ prompt
from scratch, producing logits
$L_2 \in \mathbb{R}^{(N_0 + N_1) \times V}$.

Then:

$$
L_1[t, :] \;=\; L_2[N_0 + t, :] \quad \text{exactly}, \quad
\forall t \in [0, N_1).
\quad (\bullet)
$$

That is: the replay-from-KV forward's logits **byte-equal** the
recompute forward's logits at the corresponding row, **for every
new-token position**.

## Proof

We proceed by induction on layer index $L$.

**Base case $L = 0$.** The pre-layer-0 hidden state is the sum
of token embeddings + position embeddings. Position embeddings
are looked up at absolute positions $[N_0, N_0 + N_1)$ in both
runs (the replay's ``position_offset`` argument is set to
``kv_cache.total_seq_len() == N_0``). Token embeddings are
identical because the token IDs match. So
$h^{(0)}_{\text{replay}}[t, :] = h^{(0)}_{\text{recompute}}[N_0 + t, :]$
exactly.

**Inductive step $L \to L + 1$.** Assume
$h^{(L)}_{\text{replay}}[t, :] = h^{(L)}_{\text{recompute}}[N_0 + t, :]$
for all $t$. At layer $L$:

* **Q projection.** $Q^{(L)} = h^{(L)} \cdot W_Q^{(L)}$. Both
  runs apply the same matrix to identical rows, producing
  identical Q rows.
* **K, V projections.** At the new tokens, both runs compute
  $K^{(L)}_{\text{new}} = h^{(L)} \cdot W_K^{(L)}$,
  $V^{(L)}_{\text{new}} = h^{(L)} \cdot W_V^{(L)}$ identically.
  The K, V rows for OLD tokens (positions $[0, N_0)$):
  - In the recompute, they are computed from
    $h^{(L)}_{\text{recompute}}[s, :]$ for $s \in [0, N_0)$.
  - In the replay, they are READ from the cached
    $K_0, V_0$, which were written during $T_0$.
  - By the **content-addressed KV reads/writes** assumption,
    the cached values are byte-equal to what $T_0$ computed.
    By the **frozen weights** assumption, $T_0$'s K and V
    projections were identical to what the recompute would
    compute. So the K, V rows for OLD positions match exactly.
* **Attention scores.** $\text{scores}[t, s] =
  \langle Q^{(L)}[t, :], K^{(L)}[s, :] \rangle / \sqrt{d_k}$.
  By the **deterministic float arithmetic** assumption, the
  dot-product order and the scaling are identical. The Q
  rows match (above); the K rows match (above); so scores
  match exactly.
* **Causal mask.** For row $t$ in the replay (corresponding to
  absolute position $N_0 + t$), the mask zeroes out
  attention to positions $s > N_0 + t$. Recompute applies the
  same mask at row $N_0 + t$. Identical.
* **Softmax.** Applied row-wise; identical input → identical
  output (under IEEE-754 determinism).
* **Attention output.**
  $\text{out}^{(L)}[t, :] = \sum_s \text{probs}[t, s]\, V^{(L)}[s, :]$.
  Both runs sum over the same positions in the same order with
  the same V rows; output rows match.
* **Output projection + residual + MLP.** All applied row-wise
  with frozen weights; identical inputs (above) yield identical
  outputs.

By induction, every layer's hidden state matches. The final
unembedding projection is again row-wise, so the logits match:
$L_1[t, :] = L_2[N_0 + t, :]$. This is $(\bullet)$. $\square$

## Remarks

* The proof's load-bearing assumption is **deterministic float
  arithmetic** — specifically, that the *order of additions* in
  every dot-product / softmax is fixed across both runs. The
  W79 / W80 controlled NumPy runtime guarantees this in fp64 /
  fp32 single-threaded; the W80 transformers runtime requires
  ``torch.use_deterministic_algorithms(True)`` + fp32 + CPU to
  hold. Under bf16 / int8 / GPU, the proof's conclusion holds
  only **up to a measured precision floor**, not byte-identity.
  This is the W30 / W84 precision-tier contract.
* The proof requires **no injection**. Under hidden-state /
  attention-bias / prefix-state injection, the proof breaks
  (intentionally) — the injection deliberately diverges the
  forward from the source.
* The proof handles **causal attention only**. Bidirectional
  attention (BERT-style) breaks the causal-mask step.
* The proof relies on **frozen weights**. A weight update
  between $T_0$ and $T_1$ would break the K/V cache identity
  trivially.
* The earlier W79 ``proved-conditional`` claim of
  ``W79-T-CONTROLLED-RUNTIME-REPLAY-BYTE-IDENTICAL`` carries
  forward as a *strict superset*: the W84 proof is the same
  argument with the full set of assumptions made explicit.

## Empirical sanity check

``tests/test_w84_analytical_bounds.py::test_w84_replay_from_kv_exact_byte_identity``
runs the W79 controlled NumPy runtime in fp64, generates two
streams (source forward + replay-from-KV), and asserts the
final-token logits are *byte-identical* (max-abs-diff exactly
0.0). The streaming substrate bench's separate `equivalence`
assertion at the fp64 floor of $10^{-10}$ confirms the same
result for the per-token streaming variant.
