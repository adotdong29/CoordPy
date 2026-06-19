# Pre-committed success criterion — SDK v3.34 / W33
# Trust-EWMA-tracked multi-oracle adjudication +
# single-partition long-window strict-gain regime +
# fresh live cross-architecture LLM disagreement evidence on
# trust-calibration prompts

> Pre-commit doc.  Authored **before** the W33 mechanism is implemented
> or any R-80 number is observed; defines the bar W33 must clear and
> the named falsifiers it must visibly survive.  Written to be
> falsifiable.
>
> Cross-references:
>
> * `docs/SUCCESS_CRITERION_W32_LONG_WINDOW_CONVERGENT.md` —
>   pre-committed W32 bar (10 hard / 5 soft).  W32 was a PARTIAL
>   SUCCESS: W31-C-LONG-WINDOW-CONVERGENCE discharged on the
>   scaling-stability axis, W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-
>   LIVE sharpened on the prompt-class-dependent agreement frontier
>   (gold-verifiable agree at 0.950, operational disagree at 0.250).
>   Five named open conjectures inherit forward to W33:
>
>     - **W32-C-LONG-WINDOW-STRICT-GAIN** (cycle-cap-bounded; needs
>       a regime that exceeds the cycle cap)
>     - **W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE** (gold-correlation
>       axis open; current LLMs at temp 0 honestly null)
>     - **W32-C-OLD-LINE-EWMA-TRUST** (W21 EWMA-tracked-trust
>       integration; primitives ship in W32 but the W21 integration
>       is not yet built)
>     - **W32-C-NATIVE-LATENT** (architecture-dependent; out of scope)
>     - **W32-C-MULTI-HOST** (hardware-bounded; Mac 2 ARP)
>
> * `docs/RESULTS_COORDPY_W32_LONG_WINDOW_CONVERGENT.md` — measured W32 result.
> * `docs/THEOREM_REGISTRY.md` — registry where W33 named claims will
>   be added on success.
> * `docs/HOW_NOT_TO_OVERSTATE.md` — soundness guardrails; W33 is
>   capsule-layer audited proxy (per-oracle EWMA + trust-trajectory
>   are closed-form arithmetic over a registered closed-vocabulary
>   oracle list), not transformer-internal subspace projection or a
>   learned trust model.  W33 vocabulary additions must satisfy the
>   same honest-scope language as W29/W30/W31/W32.
>
> Last touched: 2026-05-01 (pre-commit, before any W33 code is written).

---

## 1.  Position relative to W32

W32 (SDK v3.33) discharged W31-C-LONG-WINDOW-CONVERGENCE on the
**scaling-stability axis** (4 windows × 5 seeds = 20/20 byte-equal
correctness with zero degradation as window grows) AND measured the
first live cross-architecture LLM gold-verifiable agreement at
temperature 0 in the programme (19/20 = 0.950).  But W32 honestly
carried five named open conjectures forward (see §0 cross-reference).

W32's H6 +0.10 strict-gain bar was honestly null per the
**W32-L-CYCLE-CAP limitation theorem**: the W29 dispatcher's
cycle_window=8 with 3 partitions caps c_p / N at ~0.25, bounding the
maximum strict gain by min(c_p/4, c_s) / N ≤ 0.0625 — below the
+0.10 bar by structural mathematical bound, not by mechanism failure.

W32 also shipped EWMA + Page CUSUM + GoldCorrelationMap primitives at
the capsule layer.  But the **OLD explicit-capsule line** — the W21
TrustWeightedMultiOracleDisambiguator — never received the EWMA-
tracked-trust integration that the W32 primitives now make trivial
to wire.  W32-C-OLD-LINE-EWMA-TRUST was named in the W32 results note
as the new old-line-strengthening conjecture.

W33's job is to:

1. **Discharge W32-C-OLD-LINE-EWMA-TRUST** by integrating the W32
   EWMA primitive with the W21 multi-oracle adjudicator.  Build a
   `TrustEWMATrackedMultiOracleOrchestrator` (W33) that wraps a
   `TrustWeightedMultiOracleDisambiguator` (W21) and maintains a
   per-oracle EWMA over observed quorum-agreement.  When an oracle's
   EWMA-tracked trust falls below a registered threshold, its vote
   is *down-weighted* (or excluded) from the quorum count; when its
   EWMA recovers, it counts again.  Closed-form, deterministic, no
   learned model in the deep-learning sense; uses the W32
   `update_ewma_prior` primitive byte-for-byte.
2. **Discharge W32-C-LONG-WINDOW-STRICT-GAIN** by constructing a
   single-partition long-window benchmark **R-80-SINGLE-PARTITION**
   that **exceeds the W32-L-CYCLE-CAP**.  All cells route through a
   single partition (signature_period=1 ⇒ every cell maps to the
   same partition) with a prefix-then-shift drift_recover regime.
   The W31 cumulative running mean cannot re-converge; the W32 EWMA
   can.  At c_p / N = 1.0, the cycle cap no longer bounds the gain.
3. **Sharpen W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE** on the
   trust-calibration prompt-class axis with a fresh live probe on a
   bench of trust-calibration prompts (where one host is
   architecturally more likely to be correct, e.g. a reasoning-
   intensive prompt class where qwen3.5:35b might disagree with
   gemma2:9b at temp 0 AND have higher accuracy).
4. **Add a W33 manifest-v3 CID** that hashes the W32 manifest-v2 CID
   plus new components: `oracle_trust_state_cid` (the per-oracle
   EWMA trust state) and `trust_trajectory_cid` (sealed sequence of
   per-cell trust observations).  Cross-component swap detection
   that the W32 manifest-v2 CID alone cannot catch.
5. **Preserve the W32 byte-for-byte trivial path** AND add a new
   W33-Λ-trivial-trust-ewma anchor where W33 reduces to W21
   byte-for-byte when `trust_ewma_enabled = False` AND
   `manifest_v3_disabled = True` AND `trust_trajectory_window = 0`.
6. **Strengthen the strict-gain claim** by the new R-80-SINGLE-
   PARTITION bench at long_window=64 over a prefix=48 + shift=16
   drift_recover regime, where Δ(W32 - W31) ≥ +0.10 across 5/5 seeds.

W33 does NOT claim transformer-internal KV sharing.  W33 does NOT
claim "we solved context."  W33 does NOT claim a *learned trust
model* in the deep-learning sense — the per-oracle EWMA trust update
is the closed-form W32 `update_ewma_prior` primitive applied to the
W21 quorum-agreement signal.  W33 is the next step on the honest
dense-control + multi-oracle-adjudication arc, with EWMA-tracked
per-oracle trust + single-partition long-window strict-gain regime
+ W33 manifest-v3 CID machinery added at the capsule layer.

---

## 2.  Hard gates (must all pass)

### H1 — Real mechanism beyond W32 with ≥ 14 enumerated failure modes

The W33 layer must add a NEW content-addressed envelope class
``TrustEWMARatificationEnvelope`` and a NEW pure verifier
``verify_trust_ewma_ratification`` enumerating **at least 14 NEW
failure modes** that did NOT exist in any W22..W32 verifier.

Required new failure modes (explicit list — verifier must enumerate
*at least* these, may add more):

1. ``empty_w33_envelope``                        — None envelope passed.
2. ``w33_schema_version_unknown``                — schema_version
   mismatch with W33 schema.
3. ``w33_schema_cid_mismatch``                   — schema_cid != registered.
4. ``w32_parent_cid_mismatch``                   — env.w32_cid not
   the one registered (or empty when W33 is wired without W32 inner).
5. ``oracle_trust_state_cid_mismatch``           — recomputing
   oracle_trust_state_cid over canonical bytes does not match the
   envelope's stored value, OR registered_oracle_trust_state_cid
   mismatch (cross-cell swap detection).
6. ``oracle_trust_state_unregistered_oracle``    — at least one
   oracle_id in the trust state is not in the controller's
   registered oracle set.
7. ``oracle_trust_state_ewma_out_of_range``      — at least one
   ewma_trust_after < 0 OR > 1 OR NaN/Inf.
8. ``trust_trajectory_cid_mismatch``             — recomputing the
   trust_trajectory_cid over canonical bytes does not match the
   envelope's stored value.
9. ``trust_trajectory_length_mismatch``          — len(trust_trajectory)
   > registered ``trust_trajectory_window`` OR non-monotone cell indices.
10. ``trust_trajectory_unregistered_oracle``     — at least one
    oracle_id in the trust trajectory is not registered.
11. ``trust_trajectory_observed_out_of_range``   — at least one
    observed_quorum_agreement < 0 OR > 1 OR NaN/Inf.
12. ``trust_threshold_out_of_range``             — env.trust_threshold
    < 0 OR > 1 OR NaN/Inf.
13. ``manifest_v3_cid_mismatch``                 — recomputing the
    W33 manifest-v3 CID over (w32_cid, oracle_trust_state_cid,
    trust_trajectory_cid, trust_route_audit_cid) does not match the
    envelope's stored value.
14. ``w33_outer_cid_mismatch``                   — recomputing the outer
    W33 ``w33_cid`` over canonical bytes does not match the
    envelope's stored ``w33_cid``.

The verifier MUST be a pure function (no side effects); soundness MUST
hold by inspection.  Every failure mode MUST be unit-tested.

### H2 — No regression on R-80-TRIVIAL-W33

With a registry whose ``trust_ewma_enabled = False``,
``manifest_v3_disabled = True``, AND ``trust_trajectory_window = 0``,
the W33 envelope's wire-token cost MUST equal **0** and W33 MUST
reduce to W21 (or W32) **byte-for-byte** across **5/5 seeds**.  This
is the W33-Λ-trivial-trust-ewma falsifier and the strict
backward-compatibility anchor.

### H3 — Trust boundary sound

Tampered W33 envelopes MUST be rejected.  For at least 5 of the 14
enumerated failure modes, mechanically check that the verifier rejects.
Rejection rate **= 1.000** on R-80-MANIFEST-V3-TAMPER across all 5
named tampers per ratified cell × 5 seeds.

### H4 — Honest scope stated in module docstring

The W33 module docstring MUST explicitly state:

* The W33 EWMA-tracked-trust update is the W32 `update_ewma_prior`
  primitive applied to the W21 quorum-agreement signal.  Closed-form,
  deterministic, no learned model in the deep-learning sense.
* W33 does NOT claim transformer-internal KV sharing.
* W33 does NOT claim "we solved context."
* The trust-trajectory is **sealed** but does NOT prove temporal
  ordering at the model layer.

### H5 — At least 4 named falsifiers, all empirically observed

* **W33-Λ-trivial-trust-ewma** — at the trivial registry path, W33
  reduces to W21 byte-for-byte (R-80-TRIVIAL-W33).
* **W33-Λ-no-trust-shift** — when the regime has no trust shift
  (every oracle agrees with quorum throughout), every EWMA stays at
  1.0; W33 ties W21 byte-for-byte (R-80-NO-TRUST-SHIFT).
* **W33-Λ-frozen-threshold** — when ``trust_threshold = 0.0`` (every
  EWMA-tracked oracle counts regardless), W33 ties W21
  byte-for-byte (R-80-FROZEN-TRUST-THRESHOLD).
* **W33-Λ-mis-trust-shift** — when an honest oracle is mis-classified
  as a "trust-shifted" oracle (its EWMA dropped because of a
  *transient* disagreement, not a true compromise), W33 may
  regress vs W21 by losing a critical vote (R-80-MIS-TRUST-SHIFT).

### H6 — Trust-EWMA strictly outperforms W21 fixed-trust on R-80-COMPROMISED-SHIFT

On a regime where one oracle is consistently wrong on a calibration
prefix (cells 0..K) AND a second oracle becomes compromised on the
shift (cells K..N), W33 with EWMA-tracked trust must achieve:

* ``trust_precision_w33 >= trust_precision_w21 + 0.20`` on R-80-
  COMPROMISED-SHIFT, AND
* ``correctness_ratified_rate_w33 >= correctness_ratified_rate_w21``
  byte-for-byte, AND
* ``mean_total_w33_visible_tokens <= mean_total_w21_visible_tokens
  + 1.0`` per cell (max overhead 1 token/cell).

Across 5/5 seeds on bank_seed ∈ {11, 17, 23, 29, 31}.

### H7 — Long-window strict-gain on R-80-SINGLE-PARTITION

On a single-partition regime that **exceeds the W32-L-CYCLE-CAP**
(c_p / N > 0.4), the W32 EWMA + CUSUM mechanism strictly improves
over W31's cumulative running mean by Δ ≥ +0.10 correctness
ratification rate.  Across 5/5 seeds on bank_seed ∈ {11, 17, 23, 29, 31}.

This is the **W32-C-LONG-WINDOW-STRICT-GAIN empirical discharge**.

### H8 — Manifest-v3 CID detects cross-component tampering

Reject rate **= 1.000** on R-80-MANIFEST-V3-TAMPER across 5 named
tampers per ratified cell × 5 seeds.

### H9 — Release-readiness clause

* SDK_VERSION bumped to ``coordpy.sdk.v3.34``.
* ``__experimental__`` tuple updated with W33 symbols.
* pyproject.toml ``0.5.7``.
* CHANGELOG entry added.
* W33 in ``__experimental__``; stable runtime contract byte-for-byte
  unchanged.

### H10 — Focused regression green

* All W33 unit tests pass.
* phase69-79 regression suite passes.
* phase80 unit tests pass (the W33 family bench).
* Wider coordpy suite passes.

---

## 3.  Soft gates (best-effort)

### S1 — Cross-architecture live trust-calibration evidence

Run a live two-host probe on a bench of trust-calibration prompts at
temperature 0 (e.g. multi-step reasoning where the larger model is
expected to be more often correct on disagreed cells).  Honestly-null
acceptable: if both hosts agree on every prompt at temp 0 OR the
disagreement is gold-correlation null on the available LLMs, the
trust-calibration live axis remains open as renamed
**W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE**.

Best-effort: pair the largest reachable model on each host
(e.g. ``mixtral:8x7b`` on localhost vs ``qwen3.5:35b`` on
192.168.12.191) instead of the W32 pair (gemma2:9b vs qwen2.5:14b).

### S2 — Mac 2 returning OR honest fallback

192.168.12.248 ARP-incomplete continues — record honest fallback
(28th milestone in a row).

### S3 — Trust precision = 1.000 on R-80-COMPROMISED-SHIFT

W33 trust precision = 1.000 on the bench where it is supposed to win.

### S4 — Token-overhead bound ≤ 1 token/cell vs W21 / W32

``max_overhead_w33_per_cell <= 1``,
``mean_overhead_w33_per_cell <= 1.0``.

### S5 — At least one earlier conjecture sharpened or discharged

* **W21-C-CALIBRATED-TRUST** discharged via the W33 EWMA-tracked-trust
  mechanism.
* **W32-C-OLD-LINE-EWMA-TRUST** discharged via the integration of
  W32 EWMA primitives into the W21 multi-oracle line.
* **W32-C-LONG-WINDOW-STRICT-GAIN** discharged on the single-
  partition regime that exceeds the cycle cap.

---

## 4.  Verdict rule

* **Strong success**: 10/10 hard gates pass + 4/5 soft gates pass.
* **Partial success**: 8-9/10 hard gates pass.
* **Failure**: < 8/10 hard gates.

W33 must clear at least 8/10 to be considered an honest milestone;
must clear all 10 to be considered a strong success.

---

## 5.  Honest scope (what W33 does NOT claim)

* W33 does NOT claim "we solved context."  The original
  `SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` bar is unchanged.
* W33 does NOT claim a learned trust model.  EWMA + trust-threshold
  are closed-form arithmetic; zero parameters, zero gradients, zero
  training step.
* W33 does NOT claim transformer-internal latent control.  The
  per-oracle EWMA trust state lives at the capsule layer; an honest
  proxy for online trust calibration, not a runtime hidden-state
  transplant.
* W33 does NOT close `W32-C-NATIVE-LATENT` (architecture-dependent;
  the next true wall) or `W32-C-MULTI-HOST` (hardware-bounded; Mac 2
  ARP).
* W33 does NOT claim that the trust-trajectory proves temporal
  ordering at the model layer.  The trajectory is a *sealed* tuple
  of (cell_idx, oracle_id, observed_quorum_agreement, ewma_trust_
  after) bytes; it proves byte-stable replay but not that the
  underlying decisions actually executed in that order at the model
  layer.

---

## 6.  Implementation guardrails

* Reuse the W32 `update_ewma_prior` primitive verbatim — no new EWMA
  math.
* The W33 envelope must NEST the W32 envelope's CID, NOT replace it
  — the cumulative trust boundary across W22..W33 is now ≥ 70
  enumerated failure modes.
* The trust-trajectory window must be configurable; default 16
  matches W31 trajectory_window; verifier rejects longer trajectories.
* The trust-threshold must be configurable in [0, 1]; default 0.5.
* The W33 trust-state update must fire on EVERY cell where the W21
  inner ratifies a quorum (regardless of whether W21 commits or
  abstains).

---

## 7.  Cross-architecture live probe protocol (S1, best-effort)

Topology probe at the start of the W33 milestone:

1. Discover reachable hosts:
   * localhost — first, pick the largest available model.
   * 192.168.12.191 — second, pick the largest available model.
   * 192.168.12.248 — third, ARP-check; expected to fail (28th
     consecutive milestone).
2. If two hosts reachable, pair the largest model on each and run
   a 20-prompt trust-calibration bench at temperature 0:
   * 5 multi-step reasoning prompts (where larger is expected to win).
   * 5 specialised-knowledge prompts (where domain priors differ).
   * 5 ambiguous-syntax prompts (where tokenisation may differ).
   * 5 closed-vocab factoid prompts (the W32 baseline).
3. Record per-prompt: agreement, who-correct on disagreement.
4. Assemble the prompt-class-dependent agreement frontier for each
   prompt class.
5. If gold-correlation regime found, register a W33 GoldCorrelationMap
   variant (per-prompt-class) and run R-80-XLLM-LIVE-TRUST for the
   strict-gain axis.

Honestly-null OK: if no class has gold-correlation, record null and
keep W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE open.
