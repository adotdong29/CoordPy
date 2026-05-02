# Pre-committed success criterion — SDK v3.33 / W32
# Long-window convergent online geometry-aware dense control + EWMA
# prior accumulator + CUSUM change-point detector + gold-correlated
# disagreement-routing + W32 manifest-v2 CID + first measured live
# cross-architecture LLM gold-correlation evidence

> Pre-commit doc.  Authored **before** the W32 mechanism is implemented
> or any R-79 number is observed; defines the bar W32 must clear and
> the named falsifiers it must visibly survive.  Written to be falsifiable.
>
> Cross-references:
>
> * `docs/SUCCESS_CRITERION_W31_ONLINE_CALIBRATED_GEOMETRY.md` —
>   pre-committed W31 bar (10 hard / 5 soft gates).  W31 was a
>   STRONG SUCCESS: W30-C-PRIOR-LEARNING discharged at +0.125, W30-C-
>   CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE sharpened on the
>   infrastructure-discharge axis (first measured live cross-arch LLM
>   disagreement at temp 0 in 28 milestones).  Four named open
>   conjectures inherit forward to W32:
>
>     - **W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE** (gold-
>       correlation axis open)
>     - **W31-C-NATIVE-LATENT** (architecture-dependent; out of scope)
>     - **W31-C-MULTI-HOST** (hardware-bounded; Mac 2 ARP)
>     - **W31-C-LONG-WINDOW-CONVERGENCE** (trajectory_window scaling)
>
> * `docs/RESULTS_WEVRA_W31_ONLINE_CALIBRATED_GEOMETRY.md` — measured W31 result.
> * `docs/THEOREM_REGISTRY.md` — registry where W32 named claims will be added on success.
> * `docs/HOW_NOT_TO_OVERSTATE.md` — soundness guardrails; W32 is
>   capsule-layer audited proxy (EWMA + CUSUM + gold-correlation
>   lookup are all closed-form arithmetic over a registered
>   closed-vocabulary map), not transformer-internal subspace
>   projection.  W32 vocabulary additions must satisfy the same
>   honest-scope language as W29/W30/W31.
>
> Last touched: 2026-05-01 (pre-commit, before any W32 code is written).

---

## 1.  Position relative to W31

W31 (SDK v3.32) was the first capsule-native multi-agent-coordination
method to **discharge W30-C-PRIOR-LEARNING** AND **measure live cross-
architecture LLM disagreement at temperature 0** in one milestone.
But W31 honestly carried four named open conjectures forward:

* **W31-C-LONG-WINDOW-CONVERGENCE** — at trajectory_window much
  larger than the regime-shift period, the online-learned prior
  tracks the agreement-rate distribution closely; the discharge gain
  may grow with window size.  W31's running-mean update is a
  *cumulative* mean — at long windows it cannot re-converge after a
  regime switch (the cumulative mean is dragged toward the prior
  stationary value).  Discharge surface: open at SDK v3.32.
* **W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE** (gold-correlation
  axis) — on a regime where the cross-host LLM probes systematically
  disagree at temp 0 AND the disagreement systematically correlates
  with the gold-correctness label, the W31 disagreement-routed
  adjudication strictly improves correctness on live LLM bytes (not
  just synthetic).  The disagreement-existence axis was discharged
  in W31 (2/8 = 0.250 disagreement at temp 0 between gemma2:9b and
  qwen2.5:14b).  Gold-correlation axis: open at SDK v3.32.
* **W31-C-NATIVE-LATENT** — architecture-dependent; retained as the
  next true wall.  Out of scope as a capsule-layer mechanism.
* **W31-C-MULTI-HOST** — hardware-bounded; Mac 2 ARP-incomplete.

W32's job is to:

1. **Discharge W31-C-LONG-WINDOW-CONVERGENCE** by replacing the
   cumulative running-mean update with an **EWMA accumulator**
   (exponentially-weighted moving average; closed-form
   `ewma_new = (1-α) * ewma_prev + α * obs`) so the online prior
   tracks regime shifts at long windows AND adding a **CUSUM-style
   change-point detector** (Page's two-sided CUSUM) that
   re-initialises the EWMA when a regime shift is detected.  At
   trajectory_window ∈ {32, 64, 128} on a multi-shift regime, W32's
   correctness is strictly higher than W31's at the same window;
   W31's cumulative running mean saturates at the long window and
   cannot re-converge after the second shift.
2. **Sharpen W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE** on the
   gold-correlation axis by adding an audited
   ``GoldCorrelationMap`` — a registered closed-vocabulary map of
   `(partition_id → gold_correlation_score)` — and a
   gold-correlated disagreement-routing rule that fires when the
   cross-host probes disagree AND the gold-correlation map points
   to one partition with score > registered threshold.  The map's
   correctness is **not** observed at runtime; it is a **registered
   structural witness** — if the map is wrong (i.e. the registered
   gold-correlation does not align with the actual gold label),
   the W32-Λ-mis-correlated falsifier fires and W32 ties or
   regresses against W31.
3. **Add a W32 manifest-v2 CID** that hashes
   `(w31_online_cid, convergence_state_cid, gold_correlation_cid,
     long_window_state_cid)` so cross-component swaps that affect
   the EWMA / CUSUM / gold-correlation state are detected.
4. **Preserve the W31 byte-for-byte trivial path** AND add a new
   W32-Λ-trivial-long-window anchor where W32 reduces to W31
   byte-for-byte when `long_window_enabled=False` AND
   `change_point_enabled=False` AND `gold_correlation_enabled=False`
   AND `manifest_v2_disabled=True` AND `long_window=0`.
5. **Strengthen the old explicit-capsule line** by sharpening
   **W21-C-CALIBRATED-TRUST** on the EWMA-tracked online axis — the
   W21 multi-oracle adjudicator's trust weights become an EWMA over
   per-oracle agreement signals, integrated through W30/W31 into
   the W32 stack.
6. **Live two-host gold-correlation probe** on a small bench of
   gold-verifiable prompts (arithmetic, syntax, factoid) where one
   host's answer is verifiable against a registered gold answer.
   Honestly-null acceptable: if the two hosts agree on every
   gold-verifiable prompt at temp 0 OR neither host is systematically
   correct on the disagreed cells, the live gold-correlation axis
   remains open (renamed **W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE**).

W32 does NOT claim transformer-internal KV sharing.  W32 does NOT
claim "we solved context."  W32 does NOT claim a *learned model* in
the deep-learning sense — the EWMA / CUSUM updates are closed-form
arithmetic; the gold-correlation map is a registered closed-vocab
table.  W32 is the next step on the honest dense-control arc, with
long-window convergent prior tracking + change-point detection +
gold-correlated disagreement-routing machinery added at the capsule
layer.

---

## 2.  Hard gates (must all pass)

### H1 — Real mechanism beyond W31 with ≥ 14 enumerated failure modes

The W32 layer must add a NEW content-addressed envelope class
``LongWindowConvergentRatificationEnvelope`` and a NEW pure verifier
``verify_long_window_convergent_ratification`` enumerating **at least
14 NEW failure modes** that did NOT exist in any W22..W31 verifier.

Required new failure modes (explicit list — verifier must enumerate
*at least* these, may add more):

1. ``empty_w32_envelope``                        — None envelope passed.
2. ``w32_schema_version_unknown``                — schema_version
   mismatch with W32 schema.
3. ``w32_schema_cid_mismatch``                   — schema_cid != registered.
4. ``w31_parent_cid_mismatch``                   — env.w31_online_cid
   not the one registered.
5. ``convergence_state_cid_mismatch``            — recomputing
   convergence_state_cid over canonical bytes does not match the
   envelope's stored value, OR registered_convergence_state_cid
   mismatch (cross-cell swap detection).
6. ``convergence_state_length_mismatch``         — len(convergence_states) >
   registered ``long_window`` cap; or non-monotone cell indices.
7. ``convergence_state_unregistered_partition``  — at least one
   partition_id in the convergence states is not in the controller's
   registered partition set.
8. ``convergence_state_ewma_out_of_range``       — at least one
   ewma_prior_after < 0 OR > 1 OR NaN/Inf.
9. ``convergence_state_cusum_out_of_range``      — at least one
   cusum_pos OR cusum_neg < 0 OR > registered cusum_max OR NaN/Inf.
10. ``ewma_alpha_out_of_range``                  — env.ewma_alpha < 0
    OR > 1 OR NaN/Inf.
11. ``cusum_threshold_out_of_range``             — env.cusum_threshold
    < 0 OR > registered cusum_max OR NaN/Inf.
12. ``gold_correlation_cid_mismatch``            — recomputing the
    gold_correlation_cid over the registered map's canonical bytes
    does not match the envelope's stored value.
13. ``manifest_v2_cid_mismatch``                 — recomputing the W32
    manifest-v2 CID over (w31_online_cid, convergence_state_cid,
    gold_correlation_cid, long_window_state_cid) does not match the
    envelope's stored value.
14. ``w32_outer_cid_mismatch``                   — recomputing the outer
    W32 ``w32_cid`` over canonical bytes does not match the
    envelope's stored ``w32_cid``.

The verifier MUST be a pure function (no side effects); soundness MUST
hold by inspection.  Every failure mode MUST be unit-tested.

### H2 — No regression on R-79-TRIVIAL-W32

With a registry whose ``long_window_enabled = False``,
``change_point_enabled = False``,
``gold_correlation_enabled = False``,
``manifest_v2_disabled = True``, AND ``long_window = 0`` (no EWMA, no
CUSUM, no gold correlation, no manifest-v2), the W32 envelope's
wire-token cost MUST equal **0** and W32 MUST reduce to W31
**byte-for-byte** across **5/5 seeds**.  This is the
W32-Λ-trivial-long-window falsifier and the strict
backward-compatibility anchor.

Strict measurement:

* ``mean_total_w32_visible_tokens == mean_total_w31_visible_tokens``
  for every seed in {11, 17, 23, 29, 31}.
* ``correctness_ratified_rate_w32 == correctness_ratified_rate_w31``
  byte-for-byte.
* Every cell in this bank yields a
  ``w32_decoder_branch == "trivial_long_window_passthrough"`` audit
  record.

### H3 — Trust boundary sound

Tampered W32 envelopes MUST be rejected.  For at least 5 of the 14
enumerated failure modes:

* one named tampering pass on the bench (e.g. corrupt one
  `ewma_prior_after` to `2.0`; flip one `partition_id` to an
  unregistered one; replace `convergence_state_cid` with a random hex;
  flip `gold_correlation_cid`; corrupt the `manifest_v2_cid`
  byte-for-byte; corrupt the outer `w32_cid`);
* the controller verifier MUST reject with the expected reason on
  ≥ **95%** of attempted tampers across **5/5 seeds**.

Mechanically asserted: every named failure mode in H1 must be covered
by a unit test in ``test_phase79_long_window_convergent.py``.

### H4 — Honest scope of new mechanism stated in module docstring

The new W32 module-level docstring MUST state explicitly:

* W32 does NOT touch transformer KV caches, hidden states, attention
  weights, or any model-internal state.  The "long-window
  convergence" is a closed-form EWMA accumulator
  (`ewma_new = (1-α) * ewma_prev + α * obs`); the "change-point
  detector" is a closed-form Page's two-sided CUSUM; the
  "gold-correlation routing" is a deterministic lookup against a
  *registered* closed-vocabulary
  ``(partition_id → gold_correlation_score)`` map.  The "sealed
  convergence state" is a content-addressed CID over canonical
  bytes.  All four are honest audited proxies for the LatentMAS
  long-window-convergent online-calibration direction, NOT runtime
  KV transplants.
* The "EWMA online learning" is NOT a learned model in the
  deep-learning sense.  The smoothing factor `α` is a registered
  constant (default 0.20); zero parameters, zero gradients, zero
  training step.
* The "CUSUM change-point detector" is one closed-form expression
  per cell; it is not a hyperparameter search.
* The "gold-correlation map" is a *registered* closed-vocabulary
  table; the W32 layer does NOT observe ground truth at runtime.
  The gold-correlation map's correctness is a *structural witness*
  (the map says "if the partitions disagree, partition_id=K is
  more often correct on this regime").  If the map is wrong, the
  W32-Λ-mis-correlated falsifier fires and W32 ties or regresses
  against W31.
* The "manifest-v2 CID" is SHA-256 over the concatenation of the
  underlying component CIDs.  It does NOT add new entropy; it adds
  cross-component tamper detection beyond the W31 manifest CID.
* W32 does NOT claim "we solved context."  The original
  `SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` bar is unchanged.

The docstring MUST also enumerate the named falsifiers below.

### H5 — At least four named falsifiers, all empirically observed

Four named W32 falsifiers MUST be visible in the R-79 family
artifacts (each a distinct pre-committed sub-bank):

* **W32-Λ-trivial-long-window** (no help) —
  ``long_window_enabled = False``, ``change_point_enabled = False``,
  ``gold_correlation_enabled = False``, ``manifest_v2_disabled = True``,
  ``long_window = 0`` ⇒ W32 = W31 byte-for-byte.  H2 anchor.
* **W32-Λ-no-change-point** (no change-point help when stationary)
  — when the per-partition agreement-rate distribution is
  stationary across the session, the EWMA accumulator converges to
  the same value as W31's running mean and the CUSUM never crosses
  threshold; W32 ties W31 byte-for-byte on correctness.
* **W32-Λ-mis-correlated-gold** (no gold-correlation help when the
  registered map is wrong) — when the registered
  gold-correlation map points to the wrong partition (i.e. the
  partition that is *less* often correct on disagreed cells), the
  gold-correlation route fires the wrong way and W32 regresses
  vs W31.  Used to attribute the H6 gain to the *correctness* of
  the gold-correlation map, not just its presence.
* **W32-Λ-frozen-ewma** (no convergence help when α=1) — when the
  EWMA smoothing factor is fixed at α = 1.0 (which makes the EWMA
  reduce to "just the latest observation"), the long-window
  convergence is degenerate; W32 tracks single-cell noise and
  performs worse than W31's cumulative mean on a noisy regime.

### H6 — EWMA + change-point detector strictly outperforms W31 cumulative mean on long-window multi-shift regime

This is a **load-bearing gate**.

The R-79-DRIFT-RECOVER sub-bank MUST be constructed such that:

* The per-partition agreement-rate distribution shifts **twice**
  across the session.  Concretely: cells 0..N/3 have CYCLIC at FULL
  agreement; cells N/3..2N/3 have CYCLIC at PARTIAL (so reroute
  is desirable); cells 2N/3..N have CYCLIC back at FULL (so reroute
  is *no longer* desirable — the W31 cumulative mean cannot
  recover; the W32 EWMA + change-point can).
* W31 baseline (cumulative running mean prior; trajectory_window
  = 64) yields ``correctness_ratified_rate_w31 ≤ 0.700`` on this
  regime.
* W32 with EWMA + change-point + long_window=64 yields
  ``correctness_ratified_rate_w32 ≥ 0.800`` AND
  ``correctness_ratified_rate_w32 - correctness_ratified_rate_w31
  ≥ +0.10`` AND ``trust_precision_w32 ≥ 0.95`` across **5/5 seeds**.

This is the **W31-C-LONG-WINDOW-CONVERGENCE empirical discharge**.
Honest framing: at long windows, the cumulative mean cannot recover
from a regime shift; the EWMA + CUSUM mechanism can.

### H7 — Long-window scaling: gain at window ∈ {32, 64, 128}

The R-79-LONG-WINDOW sub-bank MUST sweep
``long_window ∈ {16, 32, 64, 128}`` on the multi-shift regime.

* ``correctness_ratified_rate_w32(long_window=128) ≥
  correctness_ratified_rate_w32(long_window=16)`` across 5/5 seeds.
* The gain MUST be empirically characterised: monotonically growing,
  saturating, or degrading.  If the gain saturates by long_window
  ≥ 64, that is an honest finding and the scaling claim is
  upper-bounded at the saturation window.
* If the gain degrades at long_window=128, W32-Λ-frozen-ewma
  fires (the EWMA window is too long; the prior cannot adapt); the
  honest finding is that there is an optimal window ≈ 32-64.

This is the **direct measurement of the W31-C-LONG-WINDOW-CONVERGENCE
scaling claim**.

### H8 — Manifest-v2 CID detects cross-component tampering beyond the W31 manifest CID

The R-79-MANIFEST-V2-TAMPER sub-bank MUST exhibit a regime where:

* the controller swaps the `convergence_state_cid` from a different
  envelope (cross-component swap that individually each component
  CID and the W31 manifest CID remain internally consistent);
* the W31 manifest CID alone would NOT detect this swap (the W31
  manifest does not include `convergence_state_cid`);
* the W32 manifest-v2 CID DOES detect the swap, returning
  ``manifest_v2_cid_mismatch`` on **5/5 seeds**.

This validates that the manifest-v2 CID adds real cross-component
tamper detection beyond the union of W31's per-component CIDs.

### H9 — Release-readiness clause

* SDK_VERSION bumped to ``wevra.sdk.v3.33``.
* ``__experimental__`` tuple updated to include every W32 symbol.
* ``pyproject.toml`` version bumped to ``0.5.6``.
* CHANGELOG entry added.
* ``docs/RESEARCH_STATUS.md`` reflects the new milestone.
* ``docs/THEOREM_REGISTRY.md`` updated with W32-1..W32-5 + named
  falsifiers + carry-forward conjectures.
* W32 is in experimental; the stable runtime contract (RunSpec → run
  report) is BYTE-FOR-BYTE unchanged (no W32 code on the stable path).
* Public-facing summary in `README.md` / `docs/START_HERE.md`
  acknowledges the W32 milestone with the load-bearing claim plus
  honest scope.
* Stable-vs-experimental boundary is **tightened**: the stable
  surface (SDK v3 capsule + lifecycle audit + RunSpec → report) is
  explicitly enumerated in `README.md` and `docs/START_HERE.md`,
  with the experimental dense-control surface explicitly out of
  scope for the stable contract.

### H10 — Focused regression green

* All W22..W31 regression (``test_phase69`` through ``test_phase78``)
  remains passing byte-for-byte.
* All wider regression ``test_wevra_*`` remains green.
* New ``test_phase79_long_window_convergent.py`` MUST cover every
  enumerated H1 failure mode + H2 byte-equivalence + H3 tamper-rejection
  + H5 falsifiers + H6 long-window discharge + H7 scaling sweep +
  H8 manifest-v2 tamper detection.

---

## 3.  Soft gates (must report honestly; null-acceptable with explanation)

### S1 — Live cross-architecture gold-correlation evidence on R-79-XLLM-LIVE-GOLD

Live LLM probes from two reachable hosts on a gold-verifiable
disagreement-driven regime (gemma2:9b on localhost + qwen2.5:14b on
192.168.12.191).  Status:

* **PASS** if the run records ``n_cross_host_probe_calls > 0`` AND
  the controller observes ≥ 1 cell with cross-architecture LLM
  disagreement AND on ≥ 60 % of disagreed cells, one host's answer
  matches the registered gold AND the W32 gold-correlation route
  picks that host's answer AND the resulting W32 correctness
  strictly improves over W31.
* **HONESTLY-NULL** if both reachable hosts are present but the LLMs
  agree on every gold-verifiable prompt at temp 0.  Report the
  agreement rate and label the gap.
* **HONESTLY-NULL** if the LLMs disagree but neither host is
  systematically correct on the disagreed cells (i.e. the
  gold-correlation does not exist on these prompts at temp 0).
  Renamed conjecture **W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE**
  remains open.

### S2 — Mac 2 returning OR honest fallback

* PASS if 192.168.12.248 is reachable AND a backend on it
  participates in the R-79-XLLM-LIVE-GOLD ensemble.
* HONESTLY-NULL otherwise.  When null, the bench MUST honestly
  report Mac 2 ARP status (27th milestone or whichever count) and
  continue with the strongest available topology
  (localhost + 192.168.12.191).

### S3 — Trust precision = 1.000 on the long-window bench

Across the R-79-DRIFT-RECOVER + R-79-LONG-WINDOW banks,
``trust_precision_w32`` (cells ratified ∧ correct / cells
ratified) MUST be 1.000.  Allows under-coverage (some cells
unratified) but not false ratification.

### S4 — Token-overhead bound ≤ 1 token/cell vs W31

For any R-79 sub-bank, the W32 layer's per-cell visible-token cost
MUST satisfy:

* ``mean_overhead_w32_vs_w31_per_cell ≤ 1.0`` AND
* ``max_overhead_w32_vs_w31_per_cell ≤ 1`` AND
* ``mean_overhead_w32_vs_w28_per_cell ≤ 4.0``.

(W32's envelope rides one extra wire token at most beyond W31's
single token; cumulative overhead over W28 is therefore ≤ 4 tokens.)

### S5 — At least one earlier conjecture sharpened or discharged

Reported per the H6 menu.  PASS / PARTIAL / NULL with explanation.
At minimum, **W31-C-LONG-WINDOW-CONVERGENCE** is the natural target
(empirical discharge on H6 + scaling characterisation on H7); W32
should also at least *sharpen* W21-C-CALIBRATED-TRUST on the
EWMA-tracked online axis (PART D — old-line strengthening).

---

## 4.  Verdict rule

* **Strong success**: 10/10 hard gates met AND ≥ 4/5 soft gates
  PASS or honestly-null with explanation.
* **Partial-strong success**: 10/10 hard gates met AND any soft gates
  fall through PASS / honestly-null distribution.
* **Partial success**: 8-9 hard gates met, OR any one hard gate fails
  in a way that does not invalidate the mechanism (e.g. H6 misses
  the +0.10 bar by less than 0.03, or H7 saturates at long_window=64
  but the load-bearing W31-C-LONG-WINDOW-CONVERGENCE discharge holds
  at long_window=64).
* **Failure**: any of H1, H3, H4, H10 fail OR ≤ 7 hard gates met.

If the verdict is "Failure", DO NOT bump the SDK version; instead
write a results-note explaining what was learned and which named
falsifier landed.

---

## 5.  Named theorem-style claims to be evaluated

* **W32-1 (proved + mechanically-checked)** — Trust-boundary soundness:
  ``verify_long_window_convergent_ratification`` rejects every
  enumerated tampering mode.
* **W32-2 (proved + empirical)** — Trivial-long-window byte-for-W31
  reduction: at all knobs trivial, W32's per-cell visible-token
  cost equals W31's byte-for-byte.
* **W32-3 (proved-conditional + empirical)** — Long-window convergent
  discharge: on R-79-DRIFT-RECOVER with the multi-shift schedule,
  the EWMA + CUSUM mechanism yields
  ``correctness_ratified_rate_w32 - correctness_ratified_rate_w31
  ≥ +0.10`` AND ``trust_precision_w32 ≥ 0.95`` across 5/5 seeds.
  **This is the W31-C-LONG-WINDOW-CONVERGENCE empirical discharge**.
* **W32-4 (proved-conditional + empirical)** — Long-window scaling:
  on R-79-LONG-WINDOW, ``correctness_ratified_rate_w32(long_window
  =128) ≥ correctness_ratified_rate_w32(long_window=16)``
  across 5/5 seeds.  Empirical scaling characterisation
  (monotone / saturating / degrading) accompanies this claim.
* **W32-5 (proved-conditional + empirical)** — Manifest-v2 cross-
  component tamper detection: on R-79-MANIFEST-V2-TAMPER, the
  manifest-v2 CID detects cross-component swaps that the W31
  manifest CID alone misses; rejection rate = 1.000 across 5/5
  seeds.
* **W32-Λ-trivial-long-window** (proved-empirical) — H2 anchor.
* **W32-Λ-no-change-point** (proved-empirical) — stationary regime ⇒
  no help.
* **W32-Λ-mis-correlated-gold** (proved-empirical) — wrong gold
  map ⇒ regression.
* **W32-Λ-frozen-ewma** (proved-empirical) — α = 1.0 ⇒ degenerate.
* **W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE** (conjectural, open) —
  renamed from W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE on
  the gold-correlation axis.  Subject to the live LLM
  disagreement + gold-correlation distribution.  If the two
  reachable LLMs do not disagree on gold-verifiable prompts OR
  neither host is systematically correct on the disagreed cells,
  this remains open.
* **W32-C-NATIVE-LATENT** (conjectural, open) — true transformer-
  internal subspace projection.  Architecture-dependent.
* **W32-C-MULTI-HOST** (conjectural, open) — adding a third
  reachable host.  Hardware-bounded.
* **W32-C-OLD-LINE-EWMA-TRUST** (conjectural, open) — sharpens
  W21-C-CALIBRATED-TRUST on the EWMA-tracked online axis: the W21
  multi-oracle adjudicator's trust weights become an EWMA over
  per-oracle agreement signals; the EWMA-tracked weights strictly
  improve trust precision on a regime where the trustworthy oracle
  shifts mid-session.

---

## 6.  Pre-commit checksum

This file is intentionally written **before** the W32 mechanism exists
in code.  Any post-hoc edit to lower a gate is a violation of the
pre-commit discipline.  If a gate cannot be met, the verdict is
honestly recorded; the gate is NOT redefined.

End of pre-commit success criterion.
