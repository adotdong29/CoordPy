# Pre-committed success criterion — SDK v3.31 / W30
# Calibrated geometry-aware dense control + multi-stride basis history +
# per-partition calibration prior + cross-host disagreement-routing +
# ancestor-chain causal binding

> Pre-commit doc.  Authored **before** the W30 mechanism is implemented
> or any R-77 number is observed; defines the bar W30 must clear and
> the named falsifiers it must visibly survive.  Written to be falsifiable.
>
> Cross-references:
>
> * `docs/SUCCESS_CRITERION_W29_GEOMETRY_PARTITIONED.md` — pre-committed
>   bar for W29 (10 hard / 5 soft gates).  W30's bar must include the
>   W29 surface AND add three new mechanism axes (multi-stride basis
>   history, per-partition calibration prior, cross-host disagreement-
>   routing) AND discharge the still-open W29-C-CRAM-AMPLIFICATION
>   and W29-C-PARTITION-CALIBRATION conjectures on a regime where the
>   prior best (W29) misses by construction.
> * `docs/RESULTS_WEVRA_W29_GEOMETRY_PARTITIONED.md` — measured W29 result
>   (8/10 hard PASS, 2/10 PARTIAL — H6 absolute 0.75 < 0.95, H7 cram
>   2.30× < 8×).  Both PARTIALs become named follow-up conjectures the
>   W30 bar must address head-on.
> * `docs/THEOREM_REGISTRY.md` — registry where W30 named claims will be
>   added on success.
> * `docs/HOW_NOT_TO_OVERSTATE.md` — soundness guardrails; W30 is
>   capsule-layer audited proxy, not transformer-internal subspace
>   projection.  W30 vocabulary additions must satisfy the same
>   honest-scope language as W29.
>
> Last touched: 2026-05-01 (pre-commit, before any W30 code is written).

---

## 1.  Position relative to W29

W29 (SDK v3.30) was the first capsule-native multi-agent-coordination
method to *strictly improve correctness* over both W27 AND W28
baselines on a regime where W27 alone makes correctness mistakes
(R-76-XHOST-DRIFT: +0.250 across 5/5 seeds + on the live two-host LLM
topology).  It introduced the geometry / Grassmannian / factoradic /
Lehmer / mixed-curvature / causal-validity vocabulary at the capsule
layer as audited proxy, with 14 enumerated trust-boundary failure
modes.  Two pre-committed gates were honestly missed:

* **Gap A — H7 (cram-factor 8×) MISSED at 2.30×.**  W29's structured
  payload size on R-76-CHAIN-SHARED is bounded by JSON serialisation
  of the basis (16 floats × 4 dp + overhead) and a handful of hashes
  (~810 bytes total) divided by 1 wire token, yielding a 2.30×
  amplification over W28 — short of the pre-committed 8× bar.  Became
  the named open conjecture **W29-C-CRAM-AMPLIFICATION**.
* **Gap B — H6 absolute 0.95 MISSED at 0.750.**  The load-bearing Δ ≥
  0.10 axis was met cleanly (Δ = +0.250); the absolute 0.95 bar would
  require widening the CYCLIC routing fraction (a benchmark-engineering
  follow-up) AND addressing the W22 cache-amplification effect that
  remains on the partial-oracle partition's cold-start GOLD_A cells.

Three additional W29-named open conjectures inherit forward:

* **W29-C-PARTITION-CALIBRATION** — per-partition trust priors
  calibrated from held-out per-partition agreement strictly outperform
  uniform priors.  W29 registered the field (`PartitionRegistration.
  trust_prior`) but every builder hard-codes 1.0; no calibration data
  flows into the verifier today.  Discharge surface is open.
* **W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE** — on a regime where
  cross-host LLM probes themselves disagree (not just the partial
  oracle abstaining), the W29 cross-host variance witness fires
  non-empty on > 0 cells AND the trust gate reduces false
  ratifications.  Infrastructure discharged in W29; magnitude on a
  REAL cross-LLM disagreement regime remained for the next milestone.
* **W29-C-NATIVE-LATENT** — true transformer-internal subspace
  projection (Grassmannian-style hidden-state share) strictly
  outperforms the W29 audited proxy.  Architecture-dependent; retained
  as the next true wall.

W30's job is to:

1. **Discharge W29-C-CRAM-AMPLIFICATION** on the magnitude axis by
   amplifying the W29 envelope's structured-bits numerator without
   growing the wire-token denominator (multi-stride basis history).
2. **Discharge W29-C-PARTITION-CALIBRATION** on the discharge axis by
   making per-partition trust priors load-bearing AND demonstrating
   strict improvement over uniform priors on a regime where the
   per-partition agreement-rate signal is informative.
3. **Sharpen W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE** by
   constructing a regime where cross-host probes systematically
   disagree (a synthetic + live evidence pair) AND demonstrating that
   disagreement-routed adjudication strictly improves correctness on
   that regime.
4. **Strengthen the trust boundary** with an ancestor-chain causal
   binding: W30 extends W29's predecessor-CID set to a sorted
   ancestor-chain of the last K cells, sealed in the W30 envelope.
5. **Preserve the W29 byte-for-W28 trivial path** AND add a new
   trivial-calibration anchor where W30 reduces to W29 byte-for-byte.

W30 does NOT claim transformer-internal KV sharing.  W30 does NOT
claim "we solved context."  W30 does NOT claim a learned model.
W30 is the next step on the honest dense-control arc, with calibrated
trust + multi-stride history + disagreement-routing + ancestor-chain
machinery added at the capsule layer.

---

## 2.  Hard gates (must all pass)

### H1 — Real mechanism beyond W29 with ≥ 14 enumerated failure modes

The W30 layer must add a NEW content-addressed envelope class
``CalibratedGeometryRatificationEnvelope`` and a NEW pure verifier
``verify_calibrated_geometry_ratification`` enumerating **at least
14 NEW failure modes** that did NOT exist in any W22..W29 verifier.

Required new failure modes (explicit list — verifier must enumerate
*at least* these, may add more):

1. ``empty_calibrated_envelope``        — None envelope passed.
2. ``schema_version_unknown``           — schema_version mismatch with W30 schema.
3. ``schema_cid_mismatch``              — schema_cid != registered.
4. ``w29_parent_cid_mismatch``          — env.w29_partition_cid not the
   one registered.
5. ``basis_history_cid_mismatch``       — recomputing the basis_history_cid
   over canonical bytes does not match the envelope's stored value.
6. ``basis_history_stride_mismatch``    — len(basis_cid_history) != registered_stride
   when stride > 0; or stride < 0; or basis history contains a
   non-hex CID.
7. ``basis_history_contains_unregistered_cid`` — at least one CID in
   the basis history is not in the controller's registered basis_cid set.
8. ``calibration_cid_mismatch``         — recomputing calibration_cid does
   not match envelope's stored value.
9. ``calibration_vector_dim_mismatch``  — calibration_vector length !=
   number of registered partitions.
10. ``calibration_vector_out_of_range`` — any prior < 0.0 OR > 1.0 OR
    NaN/Inf.
11. ``ancestor_chain_cid_mismatch``     — recomputing ancestor_chain_cid
    over the sorted ancestor CIDs does not match the envelope's stored
    value.
12. ``ancestor_chain_unregistered_cid`` — at least one ancestor CID is
    not in the controller's registered ancestor set.
13. ``disagreement_route_unsealed``     — when the controller observed
    cross-host disagreement AND the route flag is True, the envelope's
    ``disagreement_route_target_partition_id`` must be a registered
    partition_id.
14. ``calibrated_cid_hash_mismatch``    — recomputing the
    calibrated_cid over canonical bytes does not match the envelope's
    stored ``calibrated_cid``.

The verifier MUST be a pure function (no side effects); soundness MUST
hold by inspection.  Every failure mode MUST be unit-tested.

### H2 — No regression on R-77-TRIVIAL-CALIBRATION

With a registry whose ``calibration_stride = 0`` (no basis history
carried), ``calibration_vector = ()`` (no per-partition prior), and
``ancestor_window = 0`` (no ancestor chain), the W30 envelope's
wire-token cost MUST equal **0** and W30 MUST reduce to W29
**byte-for-byte** across **5/5 seeds**.  This is the
W30-Λ-trivial-calibration falsifier and the strict
backward-compatibility anchor.

Strict measurement:

* ``mean_total_w30_visible_tokens == mean_total_w29_visible_tokens``
  for every seed in {11, 17, 23, 29, 31}.
* ``correctness_ratified_rate_w30 == correctness_ratified_rate_w29``
  byte-for-byte.
* Every cell in this bank yields a
  ``w30_decoder_branch == "trivial_calibration_passthrough"`` audit record.

### H3 — Trust boundary sound

Tampered envelopes MUST be rejected.  For at least 5 of the 14
enumerated failure modes:

* one named tampering pass on the bench (e.g. corrupt one entry in the
  basis-history list with ``"DEADBEEF"``; flip ``calibration_vector[0]``
  to ``2.0``; replace ``ancestor_chain_cid`` with a random hex; set
  ``disagreement_route_target_partition_id`` to ``99``; corrupt the
  ``calibrated_cid`` byte-for-byte);
* the controller verifier MUST reject with the expected reason on
  ≥ **95%** of attempted tampers across **5/5 seeds**.

Mechanically asserted: every named failure mode in H1 must be covered
by a unit test in ``test_phase77_calibrated_dense_control.py``.

### H4 — Honest scope of new mechanism stated in module docstring

The new W30 module-level docstring MUST state explicitly:

* W30 does NOT touch transformer KV caches, hidden states, attention
  weights, or any model-internal state.  The "basis history" is a
  capsule-layer accumulator over the last K cells' deterministic
  basis CIDs; the "calibration vector" is a closed-form running mean
  over per-partition agreement-rate observations; both are honest
  audited proxies for the LatentMAS calibration / shared-substrate
  direction, not runtime KV transplants.
* The "calibration prior" is a vector of floats in [0, 1] with one
  entry per registered partition; it is computed by a deterministic
  closed-form Bayesian-style update over a held-out audit set; it is
  NOT a learned model.
* The "ancestor chain" is a sorted tuple of registered ancestor
  cell CIDs; the chain CID is SHA-256 over canonical bytes; it does
  NOT prove temporal ordering at the model layer; it is a structural
  commitment that the bus enforces at admission time.
* The "disagreement-routing" is a closed-form decision: when the
  controller observed cross-host probe disagreement on the cell, the
  W30 layer ROUTES the cell to a registered "high-trust"
  partition_id (default CYCLIC); the route target is sealed in the
  envelope and verified.
* W30 does NOT claim "we solved context."  The original
  `SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` bar is unchanged.

The docstring MUST also enumerate the named falsifiers below.

### H5 — At least three named falsifiers, all empirically observed

Three named W30 falsifiers MUST be visible in the R-77 family
artifacts (each a distinct pre-committed sub-bank):

* **W30-Λ-trivial-calibration** (no help) — ``calibration_stride = 0``,
  ``calibration_vector = ()``, ``ancestor_window = 0`` ⇒ W30 = W29
  byte-for-byte.  H2 anchor.
* **W30-Λ-non-calibratable** (no calibration data) — when no
  per-partition agreement history is registered (all priors uniform
  at 1.0), the calibrated trust gate cannot improve over the W29
  uniform gate; ``correctness_ratified_rate_w30 ==
  correctness_ratified_rate_w29`` on this bank.
* **W30-Λ-degenerate-history** (no cram help) — when ``calibration_stride
  = 1`` (the basis history holds only the current cell's basis CID),
  the cram-factor numerator does NOT amplify because only one CID is
  in the history; ``cram_factor_w30 / cram_factor_w29 ≤ 1.20`` on
  this bank.

### H6 — Strict cram-factor amplification: discharge of W29-C-CRAM-AMPLIFICATION

This is one of two **load-bearing gates** for the discharge of W29's
two PARTIAL gates.

The R-77-CHAIN-CRAM sub-bank MUST be constructed such that:

* ``calibration_stride ≥ 24`` (the W30 envelope carries a basis-history
  CID over the last ≥24 cells' basis CIDs).  The bench MUST report
  the smallest stride at which the 8× ratio holds; the pre-committed
  default headline parameter is ``calibration_stride = 28``.
* ``ancestor_window ≥ 8`` (the W30 envelope carries a sorted
  ancestor-chain CID over the last ≥8 cells' W29 partition CIDs).
  Pre-committed default is ``ancestor_window = 12``.
* ``calibration_vector_dim = 3`` (one calibration prior per
  registered partition).
* The pre-commit floor is justified by the linear scaling of the
  basis-history payload with stride: at stride=8 the W30 envelope
  packs ~1230 bytes / 1 wire token = 4.23 × W28; reaching the 8×
  bar requires the multi-stride accumulator to be deeper.  This is
  honest mechanism scaling, not gating: the W30 envelope's payload
  grows linearly with the registered ``calibration_stride``, so the
  H6 bar specifies the smallest stride at which the load-bearing
  W29-C-CRAM-AMPLIFICATION discharge holds.

Headline:

* ``cram_factor_w30 / cram_factor_w28 ≥ 8.0`` across **5/5 seeds**
  (the original W29 H7 bar, **discharged** by the W30 multi-stride
  history accumulator).
* ``cram_factor_w30 / cram_factor_w29 ≥ 2.0`` across 5/5 seeds (the
  W30-over-W29 amplification ratio).
* Wire-token cost MUST stay bounded: ``mean_overhead_w30_vs_w29_per_cell
  ≤ 1.0`` AND ``max_overhead_w30_vs_w29_per_cell ≤ 1``.

This is the **W29-C-CRAM-AMPLIFICATION empirical discharge**.  The
honest framing: W30 packs strictly more audit-friendly structured-
control bits per visible token than W28 by accumulating the prior
cells' basis CIDs into one envelope-side history blob — at the same
single ``<calibrated_ref:DDDD>`` wire token.

### H7 — Per-partition calibration strictly beats uniform: discharge of W29-C-PARTITION-CALIBRATION

This is the second **load-bearing gate**.

The R-77-CALIBRATION-PRIOR sub-bank MUST be constructed such that:

* W29 with uniform priors (1.0, 1.0, 1.0) yields
  ``correctness_ratified_rate_w29 ≤ 0.65`` (W29 makes false
  ratifications on cells whose true partition has historically poor
  agreement — the W21-C-CALIBRATED-TRUST signal);
* W30 with calibrated priors (computed from a held-out per-partition
  agreement audit set, e.g. (0.95, 0.95, 0.30) reflecting that
  CYCLIC partition's historical agreement is poor) yields
  ``correctness_ratified_rate_w30 ≥ 0.80`` AND
  ``correctness_ratified_rate_w30 - correctness_ratified_rate_w29
  ≥ +0.10`` AND ``trust_precision_w30 ≥ 0.95`` across **5/5 seeds**.

This is the **W29-C-PARTITION-CALIBRATION empirical discharge**
(direct analogue of W21-C-CALIBRATED-TRUST and W28-C-CALIBRATED-TRUST).

### H8 — Cross-host disagreement-routing strict gain

The R-77-XHOST-DISAGREE sub-bank MUST exhibit a regime where
cross-host probes systematically disagree (synthetic: two LLM probes
where one returns a deterministic decoy on a fraction of cells; live:
a regime crafted to surface gemma2 vs qwen2.5 architectural
disagreement at temperature 0).

* W29 baseline (no disagreement-routing) yields
  ``correctness_ratified_rate_w29 ≤ 0.75`` because the partition
  classifier alone does not exploit the disagreement signal — it
  routes by signature history not by per-cell disagreement.
* W30 with disagreement-routing (when ≥1 cross-host probe pair
  disagrees on a cell, route the cell to the registered
  high-trust partition_id, default CYCLIC) yields
  ``correctness_ratified_rate_w30 - correctness_ratified_rate_w29
  ≥ +0.10`` AND ``trust_precision_w30 ≥ 0.95`` across **5/5 seeds**.

### H9 — Release-readiness clause

* SDK_VERSION bumped to ``wevra.sdk.v3.31``.
* ``__experimental__`` tuple updated to include every W30 symbol.
* ``pyproject.toml`` version bumped to ``0.5.4``.
* CHANGELOG entry added.
* ``docs/RESEARCH_STATUS.md`` reflects the new milestone.
* W30 is in experimental; the stable runtime contract (RunSpec → run
  report) is BYTE-FOR-BYTE unchanged (no W30 code on the stable path).
* Public-facing summary in `README.md` / `docs/START_HERE.md`
  acknowledges the W30 milestone with the load-bearing claim plus
  honest scope.

### H10 — Focused regression green

* All W22..W29 regression (``test_phase69`` through ``test_phase76``)
  remains passing byte-for-byte.
* All wider regression ``test_wevra_*`` remains green.
* New ``test_phase77_calibrated_dense_control.py`` MUST cover every
  enumerated H1 failure mode + H2 byte-equivalence + H3 tamper-rejection
  + H5 falsifiers + H6 cram-factor + H7 calibration prior + H8
  disagreement-routing.

---

## 3.  Soft gates (must report honestly; null-acceptable with explanation)

### S1 — Cross-host live evidence on R-77-XHOST-DISAGREE

Live LLM probes from two hosts on the disagreement-driven regime.
Status:

* **PASS** if the run records ``n_cross_host_probe_calls > 0`` AND
  the controller observes ≥ 1 cell with cross-host disagreement AND
  the resulting W30 correctness clears the H8 bar on the live data.
* **HONESTLY-NULL** if both reachable hosts are present but the LLMs
  agree on every cell (no disagreement to exploit).  Report the
  agreement rate and label the gap.
* **HONESTLY-NULL** if Mac 2 (192.168.12.248) is still ARP-incomplete
  AND the live ensemble is single-host (then S1/S2 reduce to
  best-effort probes; the gap is hardware, not mechanism).

### S2 — Mac 2 returning OR honest fallback

* PASS if 192.168.12.248 is reachable AND a backend on it
  participates in the R-77-XHOST-DISAGREE / R-77-CROSS-HOST-LIVE
  ensemble.
* HONESTLY-NULL otherwise.  When null, the bench MUST honestly
  report Mac 2 ARP status and continue with the strongest available
  topology (localhost + 192.168.12.191).

### S3 — Trust precision = 1.000 on the cross-host bench

Across the R-77-CROSS-HOST-LIVE bank, ``trust_precision_w30`` (cells
ratified ∧ correct / cells ratified) MUST be 1.000.  Allows
under-coverage (some cells unratified) but not false ratification.

### S4 — Token-overhead bound ≤ 2 tokens/cell (cumulative)

For any R-77 sub-bank, the W30 layer's per-cell visible-token cost
MUST satisfy:

* ``mean_overhead_w30_vs_w29_per_cell ≤ 1.0`` AND
* ``max_overhead_w30_vs_w29_per_cell ≤ 1`` AND
* ``mean_overhead_w30_vs_w28_per_cell ≤ 2.0``.

(W30's envelope rides one extra wire token at most beyond W29's
single token; cumulative overhead over W28 is therefore ≤ 2 tokens.)

### S5 — At least one earlier conjecture sharpened or discharged

Reported per the H6 / H7 menu.  PASS / PARTIAL / NULL with
explanation.  At minimum, **W29-C-CRAM-AMPLIFICATION** AND
**W29-C-PARTITION-CALIBRATION** are the natural targets; W30 should
also at least *sharpen* W21-C-CALIBRATED-TRUST (per-partition
calibrated priors are the natural land for the W21 conjecture).

---

## 4.  Verdict rule

* **Strong success**: 10/10 hard gates met AND ≥ 4/5 soft gates
  PASS or honestly-null with explanation.
* **Partial-strong success**: 10/10 hard gates met AND any soft gates
  fall through PASS / honestly-null distribution.
* **Partial success**: 8-9 hard gates met, OR any one hard gate fails
  in a way that does not invalidate the mechanism (e.g. H6 misses
  the 8× bar by less than 1.5×, or H7 misses Δ ≥ 0.10 by less than
  0.03 but every other H gate is green and the underlying W30
  mechanism is sound).
* **Failure**: any of H1, H3, H4, H10 fail OR ≤ 7 hard gates met.

If the verdict is "Failure", DO NOT bump the SDK version; instead
write a results-note explaining what was learned and which named
falsifier landed.  The honest follow-up conjectures
(W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE, W30-C-NATIVE-LATENT)
remain open for SDK v3.32.

---

## 5.  Named theorem-style claims to be evaluated

* **W30-1 (proved + mechanically-checked)** — Trust-boundary soundness:
  ``verify_calibrated_geometry_ratification`` rejects every enumerated
  tampering mode.
* **W30-2 (proved + empirical)** — Trivial-calibration byte-for-W29
  reduction: at ``calibration_stride = 0``, no calibration vector,
  no ancestor window, W30's per-cell visible-token cost equals
  W29's byte-for-byte.
* **W30-3 (proved-conditional + empirical)** — Cram-factor amplification
  discharge: on R-77-CHAIN-CRAM with the pre-committed default
  ``calibration_stride = 28`` and ``ancestor_window = 12``,
  ``cram_factor_w30 / cram_factor_w28 ≥ 8.0`` AND
  ``cram_factor_w30 / cram_factor_w29 ≥ 2.0`` across 5/5 seeds.
  The headline framing: the W30 multi-stride basis-history accumulator
  honestly amplifies the W29 envelope's structured payload by packing
  K cells' worth of audited basis CIDs into one wire token, scaling
  linearly with ``calibration_stride``.  **This is the
  W29-C-CRAM-AMPLIFICATION empirical discharge** at the smallest
  stride that achieves the 8× bar.
* **W30-4 (proved-conditional + empirical)** — Per-partition calibration
  discharge: on R-77-CALIBRATION-PRIOR with calibrated priors
  (e.g., (0.95, 0.95, 0.30)) reflecting historical per-partition
  agreement-rate, ``correctness_ratified_rate_w30 -
  correctness_ratified_rate_w29 ≥ +0.10`` AND ``trust_precision_w30
  ≥ 0.95`` across 5/5 seeds.  **This is the W29-C-PARTITION-CALIBRATION
  empirical discharge** AND a sharpening of W21-C-CALIBRATED-TRUST.
* **W30-5 (proved-conditional + empirical)** — Disagreement-routing
  strict gain: on R-77-XHOST-DISAGREE, the W30 layer's
  disagreement-routed adjudication strictly improves correctness
  over W29 baseline by Δ ≥ +0.10 across 5/5 seeds.
* **W30-Λ-trivial-calibration** (proved-empirical) — H2 anchor.
* **W30-Λ-non-calibratable** (proved-empirical) — H5 anchor.
* **W30-Λ-degenerate-history** (proved-empirical) — H5 anchor.
* **W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE** (conjectural, open) —
  on a regime where cross-host LLM probes systematically disagree
  AND the disagreement-routing path strictly improves correctness on
  live LLM bytes (not just synthetic), sharpened from W29's named
  conjecture.  Status: subject to live LLM disagreement availability;
  if the two reachable LLMs agree at temp 0 on the bench, this
  remains open.
* **W30-C-NATIVE-LATENT** (conjectural, open) — true transformer-
  internal subspace projection (Grassmannian-style hidden-state
  share) strictly outperforms the W30 audited proxy.  Architecture-
  dependent; retained as the next true wall.
* **W30-C-MULTI-HOST** (conjectural, open) — adding a third reachable
  host (when Mac 2 returns) strictly improves the disagreement-
  routing signal-to-noise on a regime where a 2-host majority is
  insufficient.  Status: hardware-bounded.

---

## 6.  Pre-commit checksum

This file is intentionally written **before** the W30 mechanism exists
in code.  Any post-hoc edit to lower a gate is a violation of the
pre-commit discipline.  If a gate cannot be met, the verdict is
honestly recorded; the gate is NOT redefined.

End of pre-commit success criterion.
