# Pre-committed success criterion — SDK v3.30 / W29
# Geometry-partitioned product-manifold dense control + cross-host variance discharge

> Pre-commit doc.  Authored **before** the W29 mechanism is implemented
> or any R-76 number is observed; defines the bar W29 must clear and
> the named falsifiers it must visibly survive.  Written to be falsifiable.
>
> Cross-references:
>
> * `docs/SUCCESS_CRITERION_W28_ENSEMBLE_VERIFIED_MULTI_CHAIN.md` — the
>   pre-committed bar for W28 (W21 × W27 synthesis + ensemble
>   ratification + 11 failure modes).  W29's bar is strictly stronger:
>   it must include W28's bar AND add new geometry-aware machinery AND
>   discharge the **W28-C-CROSS-HOST-VARIANCE** open conjecture on a
>   regime where W27 alone makes correctness mistakes.
> * `docs/RESULTS_COORDPY_W28_ENSEMBLE_VERIFIED_MULTI_CHAIN.md` — the
>   measured W28 milestone result (8/8 hard, 3/4 soft, 1 honestly null;
>   the null becomes W28-C-CROSS-HOST-VARIANCE which W29 must discharge).
> * `docs/THEOREM_REGISTRY.md` — registry where W29 named claims will be
>   added on success.
> * `docs/HOW_NOT_TO_OVERSTATE.md` — soundness guardrails; W29 is
>   capsule-layer audited proxy, not transformer-internal subspace
>   projection.
>
> Last touched: 2026-04-30 (pre-commit, before any W29 code is written).

---

## 1.  Position relative to W28

W28 (SDK v3.29) was the strongest capsule-native multi-agent-coordination
result the programme had shipped.  It composed the explicit-capsule
trust line (W21 trust-weighted multi-oracle adjudication) with the
dense-control line (W27 multi-chain salience-keyed pool) inside one
decision via an ensemble probe quorum, added 11 enumerated trust-
boundary failure modes, and produced the first cross-host live
evidence in 23 milestones (localhost gemma2:9b + 192.168.12.191
qwen2.5:14b; 5592 LAN bytes; 128 cross-host probe calls; 16/16 cells
correct; 10/16 ratified, 6/16 below quorum).  Its stable-vs-experimental
boundary tightened (`__experimental__` tuple; SDK v3.29; pyproject 0.5.2).

The honestly-stated remaining gaps after W28 (per the W28 results note's
own §10 and §6, plus the master plan post-W28 audit):

* **G1 — `W28-C-CROSS-HOST-VARIANCE` is open.**  The variance-reduction
  *magnitude* on a regime where W27 alone makes correctness mistakes is
  unmeasured.  R-75 banks all have W27 correctness = 1.000 — they are
  structurally incapable of exhibiting ε > 0 cross-host variance gain.
* **G2 — Mac 2 still not participating.**  192.168.12.248 ARP-incomplete
  for 23 (now 24) consecutive milestones.  Two reachable hosts only.
* **G3 — Transformer-internal KV / hidden-state sharing not
  demonstrated.**  Every dense payload is still an audited proxy at the
  capsule layer.  Latent vocabulary in the paper (W22 `LatentDigest`
  family) is explicitly framed as *not* a transformer KV cache.
* **G4 — Geometry / mixed-curvature / Grassmannian / factoradic / Lehmer
  vocabulary is NOT YET in the paper or the master plan.**  These are
  net-new directions for SDK v3.30, not extensions of an existing
  commitment.  Any new claim along these axes must therefore be
  framed honestly: capsule-layer geometry proxy, not model-internal
  subspace projection.
* **G5 — The W21/W27 synthesis is operational in W28 but the headline
  efficiency gain on a regime where W27 ITSELF FAILS is missing.**
  W28-Λ-coordinated-drift confirms the synthesis cannot help when probes
  drift in a correlated way; the regime where the synthesis DOES help
  in correctness terms is open.
* **G6 — Density / cram-factor is not yet a measured metric.**  The
  programme has reduced visible tokens at every step (W23..W28) but has
  not yet asked: *how many bits of structured, audited control are
  packed per visible token?*  This is the natural cram-factor metric
  the dense-control line should report.

W29 must close G1 (a regime where W27 alone makes mistakes; W28 alone
also misses; W29 strictly improves correctness with bounded overhead);
explicitly add the geometry/factoradic/subspace-basis/causal-validity
mechanism and label it as audited proxy (G3 honesty); make G2 progress
if hardware permits, else honestly-null; report a cram-factor headline
(G6); and make G5 concrete with a named regime where the synthesis IS
the load-bearing reason correctness improves.

W29 does NOT claim transformer-internal KV sharing.  W29 does NOT claim
"we solved context."  W29 is the next step on the honest dense-control
arc, with explicit Grassmannian / factoradic / mixed-curvature proxy
machinery added at the capsule layer.

---

## 2.  Hard gates (must all pass)

### H1 — Real mechanism beyond W28 with ≥ 12 enumerated failure modes

The W29 layer must add a NEW content-addressed envelope class
``GeometryPartitionedRatificationEnvelope`` and a NEW pure verifier
``verify_geometry_partition_ratification`` enumerating **at least 12
failure modes** that did NOT exist in any W22..W28 verifier.

Required new failure modes (explicit list — verifier must enumerate
*at least* these, may add more):

1. ``empty_partition_envelope``  — None envelope passed.
2. ``schema_version_unknown``    — schema_version mismatch with W29 schema.
3. ``schema_cid_mismatch``       — schema_cid != registered.
4. ``w28_parent_cid_mismatch``   — parent W28 ratification_cid is not the
   one being claimed by this partition envelope.
5. ``partition_id_unregistered`` — partition_id not in the registered
   partition table (must be in {0=linear, 1=hierarchical, 2=cyclic}).
6. ``subspace_basis_dim_mismatch`` — basis_dim != registered;
   ``len(basis_vectors) != basis_dim``; per-vector length != ambient_dim.
7. ``subspace_basis_non_orthogonal`` — Gram matrix off-diagonal entries
   exceed orthogonality tolerance (default 1e-4).
8. ``subspace_basis_nan_inf`` — any NaN / Inf in any basis coefficient.
9. ``factoradic_index_out_of_range`` — index < 0 OR index ≥ K!.
10. ``factoradic_route_inverse_mismatch`` — decoding the index back to
    a permutation does not match the registered consumer order's
    permutation rank for this partition.
11. ``causal_predecessor_unregistered`` — a predecessor CID claimed by
    the envelope is not in the controller's registered predecessor set.
12. ``causal_validity_signature_invalid`` — recomputing
    ``SHA-256(parent_w28_cid || sorted(predecessor_cids))`` does not
    match the envelope's claim.
13. ``cross_host_variance_witness_unsealed`` — when the registered
    probe table has cross-host probes AND the cell witnessed
    cross-host disagreement, the envelope must carry a non-empty
    ``cross_host_variance_witness_cid``.
14. ``partition_cid_hash_mismatch`` — recomputing the partition_cid
    over canonical bytes does not match the envelope's stored
    ``partition_cid``.

The verifier MUST be a pure function (no side effects); soundness MUST
hold by inspection.  Every failure mode MUST be unit-tested.

### H2 — No regression on R-76-TRIVIAL-PARTITION at K=1

With a partition table containing only ``partition_id = 0`` (linear),
``basis_dim = 0`` (no basis carried), ``factoradic_route_index = 0``
(K=1 single consumer), and ``causal_predecessor_cids = ()``, the W29
envelope's wire-token cost MUST equal **0** and W29 MUST reduce to W28
**byte-for-byte** across **5/5 seeds**.  This is the W29-Λ-trivial-partition
falsifier and the strict backward-compatibility anchor.

Strict measurement:

* ``mean_total_w29_visible_tokens == mean_total_w28_visible_tokens``
  for every seed in {11, 17, 23, 29, 31}.
* ``correctness_ratified_rate_w29 == correctness_ratified_rate_w28``
  byte-for-byte.
* Every cell in this bank yields a
  ``w29_decoder_branch == "trivial_partition_passthrough"`` audit record.

### H3 — Trust boundary sound

Tampered envelopes MUST be rejected.  For at least 4 of the 14
enumerated failure modes:

* one named tampering pass on the bench (e.g. flip
  ``factoradic_route_index`` to ``K!``; corrupt one basis coefficient
  with NaN; replace ``causal_validity_signature`` with garbage;
  unregister a partition_id);
* the controller verifier MUST reject with the expected reason on
  ≥ **95%** of attempted tampers across **5/5 seeds**.

Mechanically asserted: every named failure mode in H1 must be covered
by a unit test in ``test_phase76_geometry_partitioned.py``.

### H4 — Honest scope of new mechanism stated in module docstring

The new W29 module-level docstring MUST state explicitly:

* W29 does NOT touch transformer KV caches, hidden states, attention
  weights, or any model-internal state.  The "subspace basis" lives at
  the **capsule layer**; it is an honest **proxy** for the LatentMAS
  Grassmannian / shared-substrate direction, not a runtime KV
  transplant.
* The "geometry partition" is a **structural label** (linear /
  hierarchical / cyclic) over the W27 cell branch + signature
  history; it is NOT a learned manifold and NOT a curvature in any
  Riemannian sense.  Mixed-curvature is invoked as design metaphor
  only.
* The "factoradic routing index" is a Lehmer-code permutation index
  over the registered consumer order; it is bounded (0 ≤ idx < K!) and
  audit-friendly; it is NOT a learned routing function.
* The "causal validity signature" is a SHA-256 over the parent W28
  CID and the registered predecessor CIDs; it does NOT prove temporal
  ordering at the model layer; it is a **structural commitment** that
  the bus enforces at admission time.
* W29 does NOT claim "we solved context."  The original
  `SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` bar is unchanged.

The docstring MUST also enumerate the named falsifiers below.

### H5 — At least three named falsifiers, all empirically observed

Three named W29 falsifiers MUST be visible in the R-76 family
artifacts (each a distinct pre-committed sub-bank):

* **W29-Λ-trivial-partition** (no help) — partition table size 1,
  basis_dim = 0, K = 1 ⇒ W29 = W28 byte-for-byte.  H2 anchor.
* **W29-Λ-non-orthogonal-basis** (unsafe) — a producer whose subspace
  basis vectors are not pairwise orthogonal must be rejected by
  ``subspace_basis_non_orthogonal`` on every cell;
  ``correctness_ratified_rate_w29`` MUST therefore be 0 with
  ``n_unrejected = 0``.
* **W29-Λ-coordinated-drift-cross-host** — cross-host probes that
  drift identically (same prompt + temperature 0 ⇒ same reply across
  hosts) cannot reduce variance even with cross-host telemetry
  on; the cross_host_variance_witness must record zero observed
  disagreement on this bank, AND
  ``correctness_ratified_rate_w29 == correctness_ratified_rate_w28``
  on this bank.

### H6 — Strict cross-host-variance discharge: W29 > W28 > W27 on a regime where W27 alone makes mistakes

This is the **load-bearing gate** for the W28-C-CROSS-HOST-VARIANCE
discharge.

The R-76-XHOST-DRIFT sub-bank MUST be constructed such that:

* W27 alone (no ensemble layer) yields
  ``correctness_ratified_rate_w27 ≤ 0.75`` (W27 makes correctness
  mistakes on this regime by construction).
* W28 alone (W27 + single deterministic ensemble probe) yields
  ``correctness_ratified_rate_w28 ≤ 0.85`` (W28's deterministic local
  probe cannot detect the regime's drift pattern; the local probe
  ratifies because the signature is locally recomputable, but the
  underlying answer is wrong because the W27 inner stack's oracle
  misses on a fraction of cells).
* W29 (W28 + geometry partitioning + cross-host probes + variance
  witness) yields ``correctness_ratified_rate_w29 ≥ 0.95`` AND
  ``correctness_ratified_rate_w29 - correctness_ratified_rate_w28 ≥ 0.10``
  AND ``trust_precision_w29 ≥ 0.95`` across **5/5** seeds.

Note: the regime is a synthetic construction where the oracle is
*partially scoped* (analogue of phase74's ``xoracle_recover``) AND
the partition gate routes the unsupported cells to a separate
ensemble stack.  The honest framing is "W29's structural extra
information about cells corrects W27's blind spot when ensemble
ratification alone is insufficient."  This is NOT a claim that LLM
ensembles solve everything — the regime is deliberately constructed
to be hard for W27 and easy for W29's structural awareness.

### H7 — Cram-factor headline

Define ``cram_factor`` as

  cram_factor = (bits of structured, audited control) / (visible-token cost)

where the bits-of-control numerator is computed from canonical bytes
of the partition envelope payload (partition_id + factoradic_index +
basis_vectors + causal_validity_signature + cross_host_variance_witness)
treating every byte as 8 bits, and the visible-token denominator is
``max(1, n_w29_wire_tokens)`` (per-cell average across the bank).

H7 requires:

* ``cram_factor_w29 ≥ 8 × cram_factor_w28`` on R-76-CHAIN-SHARED at K=3.
  (W29 packs at least 8× more structured bits per visible token than
  W28 does in the same regime.)  Honest interpretation: W29 piggy-backs
  the entire partition / factoradic / basis / causal-validity / variance
  payload onto the same single ``<partition_ref:DDDD>`` token already
  charged at the wire level.
* The headline cram-factor MUST be emitted in every R-76 results JSON
  artifact with a ``cram_factor_w29`` and ``cram_factor_w28`` field.

### H8 — Old-line strengthening clause: discharge or sharpen one earlier conjecture

W29 MUST either fully discharge **W28-C-CROSS-HOST-VARIANCE** (load-bearing
new milestone — H6 anchor) OR sharpen at least one of:

* **W21-C-CALIBRATED-TRUST** — calibrated trust priors strictly
  outperform uniform.  W29's per-partition trust prior table is the
  natural place to land calibrated priors (one set per partition_id).
* **W22-C-CACHE-AMPLIFICATION** — cache freezes a probabilistic LLM
  oracle's first reply across matching cells.  W29's cross-host
  variance witness can detect when the cache amplification IS the
  source of disagreement.
* **W26-C-K-SCALING** or **W27-C-MULTI-SIGNATURE-SCALING** — K → ∞
  asymptote.  W29's bench can measure at K∈{3, 5, 8} so the partition
  decision is not over a single (producer, K consumers) topology.
* **W4-C2** (cohort-lifted role view) — long-pending; W29's partition
  envelope is a natural cohort-lift carrier.

The discharge MUST cite the specific empirical anchor.

### H9 — Release-readiness clause

* SDK_VERSION bumped to ``coordpy.sdk.v3.30``.
* ``__experimental__`` tuple updated to include every W29 symbol.
* ``pyproject.toml`` version bumped to ``0.5.3``.
* CHANGELOG entry added.
* ``ARCHITECTURE.md`` (or its equivalent) reflects the new
  stable-vs-experimental boundary; W29 is in experimental.
* Public-facing summary in `README.md` / `docs/START_HERE.md`
  acknowledges the W29 milestone with the load-bearing claim plus
  honest scope.
* The stable runtime contract (RunSpec → run report) is BYTE-FOR-BYTE
  unchanged (no W29 code on the stable path).

### H10 — Focused regression green

* All W22..W28 regression (``test_phase69`` through ``test_phase75``)
  remains 254/254 passing byte-for-byte.
* All wider regression ``test_coordpy_*`` remains green.
* New ``test_phase76_geometry_partitioned.py`` MUST cover every
  enumerated H1 failure mode + H2 byte-equivalence + H3 tamper-rejection
  + H5 falsifiers + H7 cram-factor + the disabled / no-trigger paths.

---

## 3.  Soft gates (must report honestly; null-acceptable with explanation)

### S1 — Cross-host live evidence with ≥ 2 different model families on R-76-XHOST-DRIFT

Same intent as W28's S1: live LLM probes from different model families
on different reachable hosts, this time as part of the cross-host
variance witness on a regime where W27 alone makes correctness
mistakes.  Status:

* PASS if the run records ``n_cross_host_probe_calls > 0`` AND the
  cross_host_variance_witness fires on at least one cell AND the
  resulting W29 correctness clears the H6 bar.
* HONESTLY-NULL if both reachable hosts are present but the LLMs
  agree on every cell (no disagreement to witness).  Report the
  agreement rate and label the gap.
* HONESTLY-NULL if Mac 2 (192.168.12.248) is still ARP-incomplete
  AND the live ensemble is single-host (then S1/S2 reduce to
  best-effort probes; the gap is hardware, not mechanism).

### S2 — Mac 2 returning OR honest fallback

* PASS if 192.168.12.248 is reachable AND a backend on it
  participates in the R-76-XHOST-DRIFT ensemble.
* HONESTLY-NULL otherwise.  When null, the bench MUST honestly
  report Mac 2 ARP status and continue with the strongest available
  topology (localhost + 192.168.12.191).

### S3 — Trust precision = 1.000 on the cross-host bench

Across the R-76-CROSS-HOST-LIVE bank, ``trust_precision_w29`` (cells
ratified ∧ correct / cells ratified) MUST be 1.000.  Allows
under-coverage (some cells unratified) but not false ratification.

### S4 — Token-overhead bound ≤ 2 tokens/cell

For any R-76 sub-bank, the W29 layer's per-cell visible-token cost
MUST satisfy ``mean_overhead_w29_vs_w28_per_cell ≤ 2.0`` AND
``max_overhead_w29_vs_w28_per_cell ≤ 2``.  (W28's S4 was identical;
W29 inherits because the partition envelope rides one extra wire
token at most.)

### S5 — At least one earlier conjecture sharpened or discharged

Reported per the H8 menu.  PASS / PARTIAL / NULL with explanation.

---

## 4.  Verdict rule

* **Strong success**: 10/10 hard gates met AND ≥ 4/5 soft gates
  PASS or honestly-null with explanation.
* **Partial-strong success**: 10/10 hard gates met AND any soft gates
  fall through PASS / honestly-null distribution.
* **Partial success**: 8-9 hard gates met, OR any one hard gate fails
  in a way that does not invalidate the mechanism (e.g. H6's
  ε > 0.10 misses by less than 0.02 but every other H gate is green
  and the underlying W29 mechanism is sound).
* **Failure**: any of H1, H3, H4, H10 fail OR ≤ 7 hard gates met.

If the verdict is "Failure", DO NOT bump the SDK version; instead
write a results-note explaining what was learned and which named
falsifier landed.  The infrastructure-discharge soft gate
``W29-C-CROSS-HOST-VARIANCE`` is the named open-conjecture follow-up
in the failure case.

---

## 5.  Named theorem-style claims to be evaluated

* **W29-1 (proved + mechanically-checked)** — Trust-boundary soundness:
  ``verify_geometry_partition_ratification`` rejects every enumerated
  tampering mode.
* **W29-2 (proved + empirical)** — Trivial-partition byte-for-byte
  reduction: at K=1, basis_dim=0, partition_table size 1, W29's
  per-cell visible-token cost equals W28's byte-for-byte.
* **W29-3 (proved-conditional + empirical)** — Cross-host-variance
  discharge magnitude: on R-76-XHOST-DRIFT,
  ``correctness_ratified_rate_w29 - correctness_ratified_rate_w28 ≥ 0.10``
  with ``trust_precision_w29 ≥ 0.95`` across 5/5 seeds.
  **This is the W28-C-CROSS-HOST-VARIANCE empirical discharge.**
* **W29-4 (proved + empirical)** — Cram-factor amplification:
  ``cram_factor_w29 ≥ 8 × cram_factor_w28`` on R-76-CHAIN-SHARED.
* **W29-Λ-trivial-partition** (proved-empirical) — H2 anchor.
* **W29-Λ-non-orthogonal-basis** (proved-empirical) — unsafe-without-
  verification.  H5 anchor.
* **W29-Λ-coordinated-drift-cross-host** (proved-empirical) —
  cross-host telemetry cannot fix coordinated drift; H5 anchor.
* **W29-C-PARTITION-CALIBRATION** (conjectural, open) — per-partition
  trust priors calibrated from held-out per-partition agreement
  strictly outperform uniform priors.  Direct analogue of
  W21-C-CALIBRATED-TRUST and W28-C-CALIBRATED-TRUST.
* **W29-C-NATIVE-LATENT** (conjectural, open) — true transformer-
  internal subspace projection (Grassmannian-style hidden-state share)
  strictly outperforms the W29 audited proxy.  Architecture-dependent;
  retained as the next true wall.

---

## 6.  Pre-commit checksum

This file is intentionally written **before** the W29 mechanism exists
in code.  Any post-hoc edit to lower a gate is a violation of the
pre-commit discipline.  If a gate cannot be met, the verdict is
honestly recorded; the gate is NOT redefined.

End of pre-commit success criterion.
