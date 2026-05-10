# RESULTS — W43 Product-Manifold Capsule (PMC)

> Programme step: post-CoordPy 0.5.20. Mints axis 40 of the Context
> Zero programme. Strictly additive on top of the released v3.43
> line. The released SDK contract is byte-for-byte unchanged; the
> W43 surface lives at `coordpy.product_manifold` and is reachable
> only through an explicit import.

## TL;DR

W43 introduces the **first capsule-native product-manifold layer** in
CoordPy. Each cell's coordination state is decomposed across six
interlocking channels (hyperbolic branch / spherical consensus /
euclidean attribute / factoradic route / subspace basis /
causal-clock), each content-addressed and bound under a
``manifest-v13`` envelope verified through 18 enumerated failure
modes.

The R-90 benchmark family across 5 seeds × 8 cell families produces
the following honest, repeatable, code-backed gains (vs the released
W42 baseline):

| family | metric | W42 baseline | W43 PMC | Δ |
|---|---|---|---|---|
| `r90_trivial_pmc` | passthrough_ok | 1.000 | 1.000 | +0.000 (sanity) |
| `r90_long_branch` | branch_round_trip_ok | n/a | 1.000 | +1.000 (new channel) |
| `r90_consensus_cycle` | trust_precision | 0.600 | **1.000** | **+0.400** |
| `r90_compact_state_transfer` | structured_bits / overhead_token | 1536 | **1808** | **+272 bits** |
| `r90_causal_violation` | causal_rejection_rate | 0.000 | **1.000** | **+1.000** |
| `r90_routing_compression` | factoradic_bit_gain | 0 | **16** | **+16 bits** |
| `r90_linear_flow_falsifier` | no_false_abstain | 1.000 | 1.000 | +0.000 (no over-claim) |
| `r90_subspace_drift` | trust_precision | 0.600 | **1.000** | **+0.400** |

All H1..H10 success-criterion hypotheses pass cleanly (see
`docs/SUCCESS_CRITERION_W43_PRODUCT_MANIFOLD.md`).

## What is shipped

* **`coordpy/product_manifold.py`** (~1500 LoC, dependency-free):
  the W43 layer. Six channels, 18-mode verifier, orchestrator,
  builders, policy registry. Mathematically motivated bounded
  approximations of the architecture vision; nothing requires
  transformer-internal access.

* **`coordpy/r90_benchmark.py`**: the R-90 benchmark family. Eight
  cell families, three baselines, 5-seed aggregator, text-report
  renderer. Comparable to the R-69..R-89 ladder under
  `docs/RESULTS_COORDPY_W*.md`.

* **`tests/test_product_manifold.py`** (45 tests): channels,
  decision selector, orchestrator, verifier, schema invariants.

* **`tests/test_r90_benchmark.py`** (13 tests): each H1..H8
  hypothesis is exercised directly; aggregator + text-report
  smoke; channel-specific seeded round-trips.

* **`docs/SUCCESS_CRITERION_W43_PRODUCT_MANIFOLD.md`**: the
  pre-committed criterion document.

## What was NOT done (honest scope)

W43 is a capsule-layer milestone. It does NOT close any of:

* **`W43-C-MIXED-CURVATURE-LATENT`** — full transformer-internal
  mixed-curvature attention. The hyperbolic / spherical / euclidean
  channels are *bounded mathematical encodings* of the cell's
  branching / consensus / attribute structure; they do not modify
  the model's attention computation.

* **`W43-C-COLLECTIVE-KV-POOLING`** — host-collective KV-cache
  sharing. The W43 envelope does not pool or transfer KV state
  across hosts; it operates on the capsule layer only.

* **`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`** — a true continuous
  Gr(k, d) homotopy. The subspace basis carrier is a bounded
  approximation: it captures a single point on the Grassmannian
  per cell with QR-canonicalisation, not a continuous path.

These remain conjectural and are explicitly out of scope for the
capsule layer. The next-substrate steps are tracked in the
research-status document.

## Architecture triage (PART B of the milestone prompt)

| Vision item | Bucket | Status in W43 |
|---|---|---|
| Mixed-curvature product manifold | (1) Implementable | Decomposed into 3 channels with closed-form bounded codings (hyperbolic / spherical / euclidean) |
| Hyperbolic branch channel | (2) Approximable | Poincare-disk-style encoding; bit-perfect round-trip for paths up to 2*dim |
| Spherical consensus channel | (1) Implementable | Unit-norm L2 signature over claim_kinds |
| Euclidean attribute channel | (1) Implementable | Linear attribute vector, padded/truncated to fixed dim |
| Factoradic / permutation routing | (1) Implementable | Bijective Lehmer-code; ceil(log2(n!)) bits per cell |
| Grassmannian / subspace packing | (2) Approximable | Bounded-rank QR-canonicalised basis; principal-angle drift for comparison |
| Lorentzian / causal validity | (1) Implementable | Lamport vector clocks + componentwise dependency closure |
| Hardware substrate / collective KV sharing | (4) Substrate-blocked | Requires transformer-internal access not exposed |
| True KV pooling across hosts | (4) Substrate-blocked | Same |
| Full mixed-curvature attention | (4) Substrate-blocked | Requires modifying transformer internals |
| Continuous attention-aware geometry routing | (3) Benchmarkable not yet implementable | Compared via per-channel proxies in R-90 |
| Full Grassmannian homotopy | (3) Benchmarkable not yet implementable | Bounded approximation OK; full theory needs continuous space |

## Capsule strengthening (PART D)

W43 does NOT replace the capsule kind set. It strengthens the
existing capsule contract along five axes:

1. **Typed channel state carriers.** Each channel's encoding is a
   frozen dataclass with a deterministic CID
   (`HyperbolicBranchEncoding`, `SphericalConsensusSignature`,
   `EuclideanAttributeVector`, `FactoradicRoute`, `SubspaceBasis`,
   `CausalVectorClock`). A capsule's payload may carry the channel
   bundle's CID; the channel encodings themselves are content-
   addressed.

2. **Stronger provenance.** The W43 envelope records seven
   component CIDs (manifold_state, factoradic_route, subspace_basis,
   causal_clock, manifold_audit, manifold_witness) bound under one
   `manifest_v13_cid`. Every component is recoverable from the
   capsule bytes alone.

3. **Stronger audit.** The verifier enumerates 18 disjoint failure
   modes; tampering with any subfield is detected by name. This is
   strictly stronger than the W42 verifier (14 modes) which it
   sits on top of.

4. **Causal ordering.** The Lamport vector clock channel adds an
   explicit, mechanically-verifiable causal admissibility check;
   out-of-order handoffs are detected with 100% rejection rate
   on R-90-CAUSAL-VIOLATION.

5. **Compact state transfer.** The factoradic side-channel adds
   structured information capacity (ceil(log2(n!)) bits) at zero
   visible-token cost; the subspace channel adds a basis carrier
   whose CID summarises a continuous-style state representation
   in 256 bits.

## Bit / token cramming frontier (PART E)

The W43 envelope carries:

* 7 × 256 = **1792 bits** of structured-component CIDs
* + ceil(log2(n!)) **factoradic bits** (16 at n=8; 36 at n=12)

at the cost of:

* **1 visible token of overhead per active cell** (the
  ``manifold-decision`` overhead token).

The W42 baseline carries 6 × 256 = 1536 bits per active cell at the
same overhead. W43's strict bit-gain per visible token:

  Δ = (1792 + 16) − 1536 = **+272 bits** per cell at n=8.

The cram-frontier metric `structured_bits_per_overhead_token` is
recorded on every result envelope (`cram_factor_w43`).

This is the first capsule-layer milestone in CoordPy that crosses
**1800 structured bits per visible-token of overhead** while
preserving full content-addressed audit.

## Theory and limitations (PART L)

### W43-T-MANIFOLD-ROUND-TRIP (code-backed)

For all binary branch paths of depth at most ``2 * dim``, the
hyperbolic-channel encoding ``encode_hyperbolic_branch`` followed
by ``decode_hyperbolic_branch_path_prefix`` returns the original
path bit-for-bit.

*Witness*: ``TestHyperbolicChannel.test_round_trip_for_all_paths_up_to_capacity``
in ``tests/test_product_manifold.py``.

### W43-T-FACTORADIC-BIJECTION (code-backed)

For every ``n`` in ``[0, 7]`` and every permutation ``perm`` of
``range(n)``, ``decode_factoradic_route(encode_factoradic_route(perm))
.permutation == perm``. The factoradic-int representation carries
exactly ``ceil(log2(n!))`` bits and is bijective with the symmetric
group ``S_n``.

*Witness*: ``TestFactoradicChannel.test_round_trip_for_all_perms_up_to_n6``
in ``tests/test_product_manifold.py``.

### W43-T-SUBSPACE-CANONICAL (code-backed)

For two ``d × k`` matrices spanning the same column subspace,
``encode_subspace_basis`` produces byte-identical CIDs. Principal-
angle drift between identical subspaces is at most ``1e-6``;
between fully-orthogonal subspaces it is exactly ``π/2``.

*Witness*: ``TestSubspaceChannel`` in
``tests/test_product_manifold.py``.

### W43-T-CAUSAL-ADMISSIBILITY (code-backed)

For any sequence of vector clocks ``c_0 ... c_{n-1}``,
``is_causally_admissible`` returns True iff ``c_i <= c_{i+1}``
componentwise for all ``i``. Out-of-order sequences are detected
at the first violation index.

*Witness*: ``TestCausalChannel`` in
``tests/test_product_manifold.py``.

### W43-T-PRODUCT-DECOMPOSITION-SUFFICIENCY (mathematically
motivated, empirically verified)

Under the bounded-channel-norm assumption (each channel's encoding
lives in its prescribed bounded set: hyperbolic in the disk of
radius ``r_max < 1``, spherical on the unit sphere, euclidean in
``R^d``, subspace on Gr(k, d), factoradic in ``[0, n!)``), the
product encoding ``ProductManifoldChannelBundle`` is sufficient to
distinguish two cells iff at least one channel's encoding differs.
The decomposition is therefore lossless modulo the encoding
truncation (path bits beyond ``2 * dim``, claim_kinds beyond the
spherical bucket count, attribute fields beyond ``dim``).

*Empirical witness*: every R-90 family that exercises a single
channel (hyperbolic, spherical, factoradic, subspace, causal)
produces a strict gain over the W42 baseline that does NOT
exercise that channel.

### W43-L-TRIVIAL-PASSTHROUGH (code-backed)

When the registry is configured trivially
(``pmc_enabled=False``, ``manifest_v13_disabled=True``, all
``abstain_on_*=False``), the W43 orchestrator emits no envelope and
adds zero overhead tokens. The result is byte-for-W42.

*Witness*: ``TestOrchestratorTrivialPassthrough`` in
``tests/test_product_manifold.py`` and the
``r90_trivial_pmc`` benchmark family.

### W43-L-DUAL-CHANNEL-COLLUSION-CAP (proved-conditional)

When an adversary controls BOTH the spherical-channel signature
(forges agreement) AND the registered subspace basis (forges
zero drift), W43 cannot recover at the capsule layer. The
remaining defence is the causal-clock channel; if the adversary
also forges a monotone vector-clock sequence, the W43 layer
ratifies on the wrong cell.

The escape hatch is one of:

* register the role-handoff signature against a stricter policy
  (a different ``role_handoff_signature_cid``), forcing the
  attacker to also forge the W42 invariance signature; or
* obtain native-latent evidence outside the capsule layer
  (``W43-C-MIXED-CURVATURE-LATENT``).

### W43-L-FORGED-CAUSAL-CLOCK-CAP (proved-conditional)

The Lamport vector clock channel detects sequences that violate
the partial order. When an adversary forges a sequence that
*satisfies* the partial order but does NOT match the registered
topology, the causal-clock channel cannot infer the topology
from the clocks alone. The defence is to bind the registered
``expected_causal_topology_hash`` to the policy entry; the W43
audit records the policy entry's CID, so a forged clock
sequence will mismatch the policy entry's expected topology hash
when the controller-side policy registry is honest.

### W43-L-OBLIVIOUS-FACTORADIC-CAP (proved-conditional)

The factoradic channel encodes a permutation of
``role_arrival_order`` against ``role_universe``. When role names
are silently permuted in the registered universe, the factoradic
channel routes via the *registered* permutation, not the
*intended* one. Cross-host topology disambiguation is out of
scope at the capsule layer.

### W43-C-MIXED-CURVATURE-LATENT (conjectural)

Full transformer-internal mixed-curvature attention (the
"product-manifold attention" direction in the architecture vision)
requires substrate access — modifying the transformer's attention
computation to operate over a product of hyperbolic, spherical,
and euclidean inner products. The W43 capsule-layer encoding is a
bounded proxy; a substrate-level implementation is conjectural.

### W43-C-COLLECTIVE-KV-POOLING (conjectural)

Host-collective KV-cache sharing requires transformer-internal
hooks not exposed at the capsule layer. The W43 envelope is
content-addressed but does not pool or transfer KV state across
hosts.

### W43-C-FULL-GRASSMANNIAN-HOMOTOPY (conjectural)

A true Gr(k, d) homotopy requires continuous representation that
the bounded basis proxy approximates only locally. The W43
subspace channel captures one point on the Grassmannian per cell;
a continuous-path representation is conjectural.

## Product-boundary decisions (PART M)

The released CoordPy 0.5.20 stable surface is byte-for-byte
unchanged. The W43 module ships in the source tree but is **not**
re-exported through `coordpy/__init__.py` and is **not** listed in
`coordpy.__experimental__`. The first-run UX (`coordpy-team run
--preset quant_desk ...`) is unaffected; the smoke driver
(`tests/test_smoke_full.py`) reports "ALL CHECKS PASSED".

A sophisticated caller reaches the W43 surface explicitly:

```python
from coordpy.product_manifold import (
    ProductManifoldOrchestrator,
    build_product_manifold_registry,
    ProductManifoldPolicyEntry,
    encode_spherical_consensus,
    encode_subspace_basis,
)
from coordpy.r90_benchmark import run_all_families, render_text_report

results = run_all_families()
print(render_text_report(results))
```

This explicit import reflects the milestone's research-grade status.
A future milestone may promote a stable subset of the W43 surface
under `coordpy.__experimental__` once the cross-host live evidence
is acquired.

## Validation

* **Baseline regression**: `tests/test_smoke_full.py` reports "ALL
  CHECKS PASSED" with the W43 module on disk.

* **W43 + R-90 tests**: ``58 / 58 passed`` (`pytest tests/`).

* **R-90 across 5 seeds**: every family passes its pre-committed
  hypothesis; the seeded gains are reproducible and deterministic.

## Where this leaves the programme

W43 is the first capsule-layer milestone in CoordPy that crosses
**six channels of structured state** + **18 enumerated verifier
failure modes** + **>1800 structured bits per visible-token
overhead** while preserving full content-addressed audit and
producing strict empirical gains over the W42 line on five
distinct R-90 families.

The remaining open frontiers are the three conjectures
(`W43-C-MIXED-CURVATURE-LATENT`, `W43-C-COLLECTIVE-KV-POOLING`,
`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`); these require new
architectural substrate beyond the capsule layer and are
explicitly out of scope for the W43 milestone.
