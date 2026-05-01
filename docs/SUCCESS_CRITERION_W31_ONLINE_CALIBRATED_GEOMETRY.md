# Pre-committed success criterion — SDK v3.32 / W31
# Online self-calibrated geometry-aware dense control + sealed prior
# trajectory + adaptive threshold + cross-architecture live disagreement-
# routing + W31 manifest CID

> Pre-commit doc.  Authored **before** the W31 mechanism is implemented
> or any R-78 number is observed; defines the bar W31 must clear and
> the named falsifiers it must visibly survive.  Written to be falsifiable.
>
> Cross-references:
>
> * `docs/SUCCESS_CRITERION_W30_CALIBRATED_GEOMETRY.md` — pre-committed
>   W30 bar (10 hard / 5 soft gates).  W30 was a STRONG SUCCESS:
>   cram-factor 8.74× discharged W29-C-CRAM-AMPLIFICATION,
>   per-partition calibration prior discharged W29-C-PARTITION-
>   CALIBRATION at +0.250 trust-precision 1.000, disagreement-routing
>   sharpened W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE at +0.250
>   trust-precision 1.000.  Three new W30-named open conjectures
>   inherit forward to W31.
> * `docs/RESULTS_WEVRA_W30_CALIBRATED_GEOMETRY.md` — measured W30
>   result.
> * `docs/THEOREM_REGISTRY.md` — registry where W31 named claims will
>   be added on success.
> * `docs/HOW_NOT_TO_OVERSTATE.md` — soundness guardrails; W31 is
>   capsule-layer audited proxy, not transformer-internal subspace
>   projection.  W31 vocabulary additions must satisfy the same
>   honest-scope language as W29/W30.
>
> Last touched: 2026-05-01 (pre-commit, before any W31 code is written).

---

## 1.  Position relative to W30

W30 (SDK v3.31) was the first capsule-native multi-agent-coordination
method to **simultaneously discharge** two pre-committed open
conjectures (W29-C-CRAM-AMPLIFICATION on the magnitude axis,
W29-C-PARTITION-CALIBRATION on the discharge axis) and **sharpen** a
third (W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE on the synthetic
axis), in one coherent mechanism extension on top of W29.  But W30
honestly carried four named open conjectures forward:

* **W30-C-PRIOR-LEARNING** — learning the per-partition calibration
  prior from held-out per-partition agreement-rate observations (via
  repeated calls to `update_partition_calibration_running_mean`)
  strictly outperforms a hand-set prior on a regime where the
  agreement-rate distribution is non-stationary.  The closed-form
  running-mean update is shipped in W30 but **never fires inside the
  orchestrator** — every benchmark hand-sets the prior.  Discharge
  surface is open.
* **W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE** — on a regime
  where the cross-host LLM probes themselves systematically disagree
  at temperature 0 (e.g., gemma2 vs qwen2.5 architectural
  disagreement on prompts crafted to surface family-specific bias),
  the W30 cross-host variance witness fires non-empty AND the
  disagreement-routed adjudication strictly improves correctness on
  live LLM bytes (not just synthetic).
* **W30-C-NATIVE-LATENT** — true transformer-internal subspace
  projection (Grassmannian-style hidden-state share) strictly
  outperforms the W30 audited proxy.  Architecture-dependent;
  retained as the next true wall.
* **W30-C-MULTI-HOST** — adding a third reachable host (when Mac 2
  returns) strictly improves the disagreement-routing signal-to-noise
  on a regime where a 2-host majority is insufficient.  Status:
  hardware-bounded.

W31's job is to:

1. **Discharge W30-C-PRIOR-LEARNING** on the magnitude axis by
   running the closed-form running-mean update *inside the
   orchestrator* on every ratified cell, then demonstrating strict
   correctness improvement over a hand-set prior on a regime where
   the per-partition agreement-rate distribution is non-stationary
   (the "drift regime").
2. **Add an adaptive threshold** that responds to the running prior
   distribution by closed-form (clipped median) so that no single
   threshold value is hard-coded.
3. **Seal the prior trajectory** via a content-addressed CID over
   the canonical bytes of the per-cell sequence
   `(cell_idx, partition_id, observed_agreement, prior_after)`.
   Tampering on any trajectory entry is detected by the verifier.
4. **Add a W31 manifest CID** that hashes
   `(basis_history_cid, calibration_cid, ancestor_chain_cid,
     prior_trajectory_cid, threshold_trajectory_cid,
     route_audit_cid)` so cross-component swaps are detected.
5. **Sharpen W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE** by
   constructing a small live cross-architecture probe (gemma2:9b on
   localhost vs qwen2.5:14b on 192.168.12.191) on prompts crafted
   for surface-level family-specific disagreement at temperature 0.
   The bar is null-acceptable with explanation: if the two LLMs
   honestly agree on every cell at temp 0, the witness is null and
   the sharpening is one-sided (synthetic + infrastructure-ready).
6. **Preserve the W30 byte-for-byte trivial path** AND add a new
   W31 trivial-online anchor where W31 reduces to W30 byte-for-byte
   when `online_enabled=False` AND `manifest_disabled=True` AND
   `trajectory_window=0`.

W31 does NOT claim transformer-internal KV sharing.  W31 does NOT
claim "we solved context."  W31 does NOT claim a *learned model* in
the deep-learning sense — the running-mean update is closed-form
arithmetic.  W31 is the next step on the honest dense-control arc,
with online prior-learning + adaptive threshold + sealed trajectory
machinery added at the capsule layer.

---

## 2.  Hard gates (must all pass)

### H1 — Real mechanism beyond W30 with ≥ 14 enumerated failure modes

The W31 layer must add a NEW content-addressed envelope class
``OnlineCalibratedRatificationEnvelope`` and a NEW pure verifier
``verify_online_calibrated_ratification`` enumerating **at least
14 NEW failure modes** that did NOT exist in any W22..W30 verifier.

Required new failure modes (explicit list — verifier must enumerate
*at least* these, may add more):

1. ``empty_w31_envelope``                       — None envelope passed.
2. ``w31_schema_version_unknown``               — schema_version mismatch
   with W31 schema.
3. ``w31_schema_cid_mismatch``                  — schema_cid != registered.
4. ``w30_parent_cid_mismatch``                  — env.w30_calibrated_cid
   not the one registered.
5. ``prior_trajectory_cid_mismatch``            — recomputing
   prior_trajectory_cid over canonical bytes does not match the
   envelope's stored value.
6. ``prior_trajectory_length_mismatch``         — len(trajectory) > the
   registered ``trajectory_window`` cap; or trajectory contains
   non-monotone cell indices.
7. ``prior_trajectory_unregistered_partition``  — at least one
   `partition_id` in the trajectory is not in the controller's
   registered partition set.
8. ``prior_trajectory_observed_out_of_range``   — at least one
   `observed_agreement` < 0 OR > 1 OR NaN/Inf.
9. ``prior_trajectory_prior_after_out_of_range`` — at least one
   `prior_after` < 0 OR > 1 OR NaN/Inf.
10. ``threshold_trajectory_cid_mismatch``       — recomputing
    threshold_trajectory_cid does not match envelope's stored value.
11. ``threshold_trajectory_value_out_of_range``  — any threshold < 0 OR
    > 1 OR NaN/Inf.
12. ``threshold_trajectory_length_mismatch``     — len(threshold_trajectory)
    != len(prior_trajectory).
13. ``manifest_cid_mismatch``                    — recomputing the W31
    manifest CID over (basis_history_cid, calibration_cid,
    ancestor_chain_cid, prior_trajectory_cid,
    threshold_trajectory_cid, route_audit_cid) does not match the
    envelope's stored value.
14. ``w31_outer_cid_mismatch``                   — recomputing the outer
    W31 ``w31_cid`` over canonical bytes does not match the
    envelope's stored ``w31_cid``.

The verifier MUST be a pure function (no side effects); soundness MUST
hold by inspection.  Every failure mode MUST be unit-tested.

### H2 — No regression on R-78-TRIVIAL-W31

With a registry whose ``online_enabled = False``, ``manifest_disabled
= True``, and ``trajectory_window = 0`` (no online learning, no
manifest), the W31 envelope's wire-token cost MUST equal **0** and
W31 MUST reduce to W30 **byte-for-byte** across **5/5 seeds**.  This
is the W31-Λ-trivial-online falsifier and the strict
backward-compatibility anchor.

Strict measurement:

* ``mean_total_w31_visible_tokens == mean_total_w30_visible_tokens``
  for every seed in {11, 17, 23, 29, 31}.
* ``correctness_ratified_rate_w31 == correctness_ratified_rate_w30``
  byte-for-byte.
* Every cell in this bank yields a
  ``w31_decoder_branch == "trivial_online_passthrough"`` audit record.

### H3 — Trust boundary sound

Tampered W31 envelopes MUST be rejected.  For at least 5 of the 14
enumerated failure modes:

* one named tampering pass on the bench (e.g. corrupt one
  `observed_agreement` to `2.0`; flip one `partition_id` to an
  unregistered one; replace `prior_trajectory_cid` with a random hex;
  flip `threshold_trajectory_cid`; corrupt the `manifest_cid`
  byte-for-byte; corrupt the outer `w31_cid`);
* the controller verifier MUST reject with the expected reason on
  ≥ **95%** of attempted tampers across **5/5 seeds**.

Mechanically asserted: every named failure mode in H1 must be covered
by a unit test in ``test_phase78_online_calibrated.py``.

### H4 — Honest scope of new mechanism stated in module docstring

The new W31 module-level docstring MUST state explicitly:

* W31 does NOT touch transformer KV caches, hidden states, attention
  weights, or any model-internal state.  The "online learning" is a
  closed-form running-mean update over a deterministic per-cell
  agreement signal; the "adaptive threshold" is closed-form (clipped
  median of the prior vector); the "sealed trajectory" is a
  content-addressed CID over canonical bytes.  Both are honest
  audited proxies for the LatentMAS online-calibration / shared-
  substrate direction, not runtime KV transplants.
* The "online learned" calibration prior is NOT a learned model in
  the deep-learning sense.  It is a closed-form Bayesian-style
  running mean over a per-cell agreement-rate observation; the
  observation itself is a deterministic boolean signal (cell ratified
  AND no cross-host disagreement); the update has zero parameters,
  zero gradients, zero training step.
* The "adaptive threshold" is a closed-form clipped median of the
  current calibration vector, bounded to ``[threshold_min,
  threshold_max]`` (registered constants).  It is not a hyperparameter
  search; it is one line of arithmetic.
* The "manifest CID" is SHA-256 over the concatenation of the
  underlying component CIDs.  It does NOT add new entropy; it adds
  cross-component tamper detection.
* The "prior trajectory" is a sealed tuple of
  `(cell_idx, partition_id, observed_agreement, prior_after)` bytes;
  it does NOT prove temporal order at the model layer; it does prove
  the controller's bus saw exactly that sequence of online updates.
* W31 does NOT claim "we solved context."  The original
  `SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` bar is unchanged.

The docstring MUST also enumerate the named falsifiers below.

### H5 — At least three named falsifiers, all empirically observed

Three named W31 falsifiers MUST be visible in the R-78 family
artifacts (each a distinct pre-committed sub-bank):

* **W31-Λ-trivial-online** (no help) — ``online_enabled = False``,
  ``manifest_disabled = True``, ``trajectory_window = 0`` ⇒ W31 = W30
  byte-for-byte.  H2 anchor.
* **W31-Λ-no-drift** (no online help when stationary) — when the
  per-partition agreement-rate distribution is stationary across the
  session, the online-learned prior converges to the same hand-set
  prior, and ``correctness_ratified_rate_w31 ==
  correctness_ratified_rate_w30`` on this bank.
* **W31-Λ-frozen-threshold** (no adaptive help) — when
  ``adaptive_threshold = False``, the threshold trajectory is a
  constant; the adaptive override never fires; W31 reduces to "online
  prior + frozen threshold."  Used to attribute the H6 gain to the
  online-prior axis vs the adaptive-threshold axis.

### H6 — Online prior strictly beats hand-set on drift regime: discharge of W30-C-PRIOR-LEARNING

This is the **load-bearing gate**.

The R-78-NONSTATIONARY-PRIOR sub-bank MUST be constructed such that:

* The per-partition agreement-rate distribution shifts mid-session.
  Concretely: the bench is divided into two regimes; in regime A
  (cells 0..N/2), CYCLIC has high agreement; in regime B (cells
  N/2..N), CYCLIC has low agreement.  The hand-set W30 prior
  (calibrated on regime A) under-fires in regime B; the
  online-learned W31 prior (updated cell-by-cell) tracks the shift
  and reroutes correctly in regime B.
* W30 baseline (hand-set prior calibrated on regime A) yields
  ``correctness_ratified_rate_w30 ≤ 0.65``.
* W31 with online learning + adaptive threshold yields
  ``correctness_ratified_rate_w31 ≥ 0.80`` AND
  ``correctness_ratified_rate_w31 - correctness_ratified_rate_w30
  ≥ +0.10`` AND ``trust_precision_w31 ≥ 0.95`` across **5/5 seeds**.

This is the **W30-C-PRIOR-LEARNING empirical discharge**.  Honest
framing: the running-mean update inside the orchestrator turns the
W30 calibration prior from a *static registration* into an
*online-tracked statistic* that responds to the regime shift, at no
extra wire-token cost.

### H7 — Adaptive threshold strictly contributes vs frozen threshold

The R-78-ADAPTIVE-THRESHOLD sub-bank MUST construct a regime where
the prior distribution is such that a fixed 0.5 threshold either
reroutes too aggressively (every cell) or too conservatively
(no cells), but a clipped-median adaptive threshold reroutes the
right subset.

* W31 with adaptive threshold yields
  ``correctness_ratified_rate_w31 ≥ 0.80``;
* W31 with frozen threshold (same online-learned prior) yields
  ``correctness_ratified_rate_w31_frozen < correctness_ratified_rate_w31_adaptive``
  by at least **+0.05** across 5/5 seeds.

This isolates the contribution of the adaptive-threshold axis and
prevents H6's gain from being attributed entirely to online learning.

### H8 — Manifest CID detects cross-component tampering

The R-78-MANIFEST-TAMPER sub-bank MUST exhibit a regime where:

* the controller swaps the `prior_trajectory_cid` from a different
  envelope (cross-component swap that individually each component CID
  remains internally consistent);
* the W30-style per-component checks alone would NOT detect this swap
  (each component is internally consistent);
* the W31 manifest CID DOES detect the swap, returning
  ``manifest_cid_mismatch`` on **5/5 seeds**.

This validates that the manifest CID adds real cross-component tamper
detection beyond the union of per-component CIDs.

### H9 — Release-readiness clause

* SDK_VERSION bumped to ``wevra.sdk.v3.32``.
* ``__experimental__`` tuple updated to include every W31 symbol.
* ``pyproject.toml`` version bumped to ``0.5.5``.
* CHANGELOG entry added.
* ``docs/RESEARCH_STATUS.md`` reflects the new milestone.
* W31 is in experimental; the stable runtime contract (RunSpec → run
  report) is BYTE-FOR-BYTE unchanged (no W31 code on the stable path).
* Public-facing summary in `README.md` / `docs/START_HERE.md`
  acknowledges the W31 milestone with the load-bearing claim plus
  honest scope.

### H10 — Focused regression green

* All W22..W30 regression (``test_phase69`` through ``test_phase77``)
  remains passing byte-for-byte.
* All wider regression ``test_wevra_*`` remains green.
* New ``test_phase78_online_calibrated.py`` MUST cover every
  enumerated H1 failure mode + H2 byte-equivalence + H3 tamper-rejection
  + H5 falsifiers + H6 drift discharge + H7 adaptive threshold +
  H8 manifest tamper detection.

---

## 3.  Soft gates (must report honestly; null-acceptable with explanation)

### S1 — Cross-architecture live evidence on R-78-XLLM-LIVE

Live LLM probes from two reachable hosts on a cross-architecture
disagreement-driven regime (gemma2:9b on localhost + qwen2.5:14b on
192.168.12.191).  Status:

* **PASS** if the run records ``n_cross_host_probe_calls > 0`` AND
  the controller observes ≥ 1 cell with cross-architecture LLM
  disagreement AND the resulting W31 correctness clears the H6 bar
  on the live data.
* **HONESTLY-NULL** if both reachable hosts are present but the LLMs
  agree on every cell at temp 0.  Report the agreement rate and label
  the gap.
* **HONESTLY-NULL** if Mac 2 (192.168.12.248) is still ARP-incomplete
  AND the live ensemble is two-host (then S1 reduces to best-effort
  probes; the gap is hardware, not mechanism).

### S2 — Mac 2 returning OR honest fallback

* PASS if 192.168.12.248 is reachable AND a backend on it
  participates in the R-78-XLLM-LIVE ensemble.
* HONESTLY-NULL otherwise.  When null, the bench MUST honestly
  report Mac 2 ARP status and continue with the strongest available
  topology (localhost + 192.168.12.191).

### S3 — Trust precision = 1.000 on the cross-host bench

Across the R-78-XLLM-LIVE bank, ``trust_precision_w31`` (cells
ratified ∧ correct / cells ratified) MUST be 1.000.  Allows
under-coverage (some cells unratified) but not false ratification.

### S4 — Token-overhead bound ≤ 1 token/cell vs W30

For any R-78 sub-bank, the W31 layer's per-cell visible-token cost
MUST satisfy:

* ``mean_overhead_w31_vs_w30_per_cell ≤ 1.0`` AND
* ``max_overhead_w31_vs_w30_per_cell ≤ 1`` AND
* ``mean_overhead_w31_vs_w28_per_cell ≤ 3.0``.

(W31's envelope rides one extra wire token at most beyond W30's
single token; cumulative overhead over W28 is therefore ≤ 3 tokens.)

### S5 — At least one earlier conjecture sharpened or discharged

Reported per the H6 menu.  PASS / PARTIAL / NULL with explanation.
At minimum, **W30-C-PRIOR-LEARNING** is the natural target; W31
should also at least *sharpen* W30-C-CROSS-HOST-VARIANCE-LIVE-
MAGNITUDE-LIVE on the cross-architecture infrastructure axis.

---

## 4.  Verdict rule

* **Strong success**: 10/10 hard gates met AND ≥ 4/5 soft gates
  PASS or honestly-null with explanation.
* **Partial-strong success**: 10/10 hard gates met AND any soft gates
  fall through PASS / honestly-null distribution.
* **Partial success**: 8-9 hard gates met, OR any one hard gate fails
  in a way that does not invalidate the mechanism (e.g. H6 misses
  the +0.10 bar by less than 0.03, or H7 misses the +0.05 bar by less
  than 0.02 but every other H gate is green and the underlying W31
  mechanism is sound).
* **Failure**: any of H1, H3, H4, H10 fail OR ≤ 7 hard gates met.

If the verdict is "Failure", DO NOT bump the SDK version; instead
write a results-note explaining what was learned and which named
falsifier landed.  The honest follow-up conjectures
(W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE,
W31-C-NATIVE-LATENT, W31-C-MULTI-HOST) remain open for SDK v3.33.

---

## 5.  Named theorem-style claims to be evaluated

* **W31-1 (proved + mechanically-checked)** — Trust-boundary soundness:
  ``verify_online_calibrated_ratification`` rejects every enumerated
  tampering mode.
* **W31-2 (proved + empirical)** — Trivial-online byte-for-W30
  reduction: at ``online_enabled = False``,
  ``manifest_disabled = True``, ``trajectory_window = 0``, W31's
  per-cell visible-token cost equals W30's byte-for-byte.
* **W31-3 (proved-conditional + empirical)** — Online prior-learning
  discharge: on R-78-NONSTATIONARY-PRIOR with the regime-shift
  schedule, the closed-form running-mean update inside the
  orchestrator yields
  ``correctness_ratified_rate_w31 - correctness_ratified_rate_w30
  ≥ +0.10`` AND ``trust_precision_w31 ≥ 0.95`` across 5/5 seeds.
  **This is the W30-C-PRIOR-LEARNING empirical discharge**.
* **W31-4 (proved-conditional + empirical)** — Adaptive-threshold
  contribution: on R-78-ADAPTIVE-THRESHOLD, the clipped-median
  adaptive threshold strictly outperforms the frozen threshold
  (same online-learned prior) by Δ ≥ +0.05 across 5/5 seeds.
* **W31-5 (proved-conditional + empirical)** — Manifest cross-
  component tamper detection: on R-78-MANIFEST-TAMPER, swapping
  ``prior_trajectory_cid`` from a different envelope is rejected
  with reason ``manifest_cid_mismatch`` on 5/5 seeds.
* **W31-Λ-trivial-online** (proved-empirical) — H2 anchor.
* **W31-Λ-no-drift** (proved-empirical) — stationary regime ⇒ no help.
* **W31-Λ-frozen-threshold** (proved-empirical) — frozen threshold
  isolates the adaptive contribution.
* **W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE** (conjectural,
  open) — sharpened from W30; subject to the live LLM disagreement
  distribution.  If the two reachable LLMs agree at temp 0 on the
  bench, this remains open.
* **W31-C-NATIVE-LATENT** (conjectural, open) — true transformer-
  internal subspace projection (Grassmannian-style hidden-state
  share) strictly outperforms the W31 audited proxy.  Architecture-
  dependent; retained as the next true wall.
* **W31-C-MULTI-HOST** (conjectural, open) — adding a third reachable
  host (when Mac 2 returns) strictly improves the disagreement-
  routing signal-to-noise on a regime where a 2-host majority is
  insufficient.  Status: hardware-bounded.

---

## 6.  Pre-commit checksum

This file is intentionally written **before** the W31 mechanism exists
in code.  Any post-hoc edit to lower a gate is a violation of the
pre-commit discipline.  If a gate cannot be met, the verdict is
honestly recorded; the gate is NOT redefined.

End of pre-commit success criterion.
