# Success criterion — CoordPy SDK v3.35 / W34
# Live-aware multi-anchor adjudication + native-latent audited
# response-feature proxy + W34 manifest-v4 CID + live cross-host
# pre-flight discipline

**Pre-committed**: 2026-05-01, before any W34 implementation code.
**Target**: SDK v3.35 / W34 family.
**Position relative to W33**: W33 (SDK v3.34) was a STRONG SUCCESS
(10/10 hard gates passed) that jointly discharged
W21-C-CALIBRATED-TRUST + W32-C-OLD-LINE-EWMA-TRUST +
W32-C-LONG-WINDOW-STRICT-GAIN.  W33 carried four open conjectures
forward:

* **W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE** — open; the live xLLM
  pilot was bounded by infra (the W33 milestone diagnosed
  qwen3.5:35b on 192.168.12.191 as "model not loaded", but a fresh
  ``/api/tags`` curl on 2026-05-01 confirms the model IS in fact
  loaded — the real W33 infra failure was timeout-budget /
  prompt-template, NOT model availability).
* **W33-C-NATIVE-LATENT** — open; the strongest dense-control
  story remains capsule-layer audited proxy.
* **W33-C-MULTI-HOST** — open; Mac 2 still ARP-incomplete (29th
  milestone in a row).
* **W33-C-LATENT-CROSS-AGENT-TRUST** — open; deepest
  trust/semantics wall.

Plus two named infrastructure follow-ups inherited from W33:

* **W33-INFRA-1** — pre-flight ``/api/tags`` model verification.
  *Honest correction recorded for W34*: the W33 milestone called
  this "model not loaded on remote host" but the fresh probe shows
  the model IS loaded.  The real infra blocker on the W33 run was
  120 s timeout exhaustion combined with 35B-MoE prompt-template
  mismatch, NOT model absence.  W34 must close both axes.
* **W33-INFRA-2** — strict token-budget / chat-template for
  one-word probes.

W34 must aim at all of them at once and pre-commit a falsifiable
bar.

---

## 1. Mechanism (W34, pre-committed)

W34 wraps a :class:`TrustEWMATrackedMultiOracleOrchestrator` (W33)
with a :class:`LiveAwareMultiAnchorOrchestrator` that:

1. **Multi-anchor consensus reference** instead of a single anchor.
   The registry registers ``K ≥ 1`` anchor oracle IDs.  The W34
   reference is the *intersection* of anchor probes' ``top_set``s
   when the anchors agree (with a registered ``anchor_quorum_min``
   in {1, 2, ..., K}); when fewer anchors agree than
   ``anchor_quorum_min``, W34 emits a
   ``W34_BRANCH_MULTI_ANCHOR_NO_CONSENSUS`` marker and reroutes the
   reference to the W21-quorum-resolved top_set (preserving W33
   behavior on regimes where the multi-anchor consensus is not
   available).

2. **Live attestation envelope** for live LLM oracles: a content-
   addressed :class:`LiveOracleAttestation` that records (host_id,
   model_id, response_feature_signature, latency_ms_bucket,
   preflight_ok).  Attached as a sealed sub-envelope to the W34
   envelope; verified via :func:`verify_live_oracle_attestation`
   against a registered live-oracle topology.

3. **Response-feature signature** — a 64-bit deterministic hash
   over (first_token_class, length_bucket, structural_hash) of an
   LLM response.  This is the **W34 audited proxy** for native-
   latent — closed-form, zero parameters, zero gradients, zero
   training step; an honest approximation of a transformer-internal
   subspace projection at the capsule layer.  Explicit gap: NOT a
   transformer-internal hidden-state projection; the W34 proxy
   detects feature-class shifts (e.g. the model went from emitting
   one-word answers to chain-of-thought) but cannot probe hidden
   states.

4. **Host-aware EWMA decay** — a closed-form ``host_decay_factor``
   in [0.5, 1.0] applied per-host: when a host fails preflight or
   times out, its oracles' EWMA-tracked trust decays *faster*
   (multiplicative) than the W33 baseline.  Bounded; closed-form;
   zero parameters.

5. **Pre-flight discipline** (closes W33-INFRA-1 + W33-INFRA-2):

   * ``/api/tags`` check before the first probe to confirm model
     availability;
   * one-word-probe template that uses ``/api/chat`` with
     ``messages=[{role: system, content: STRICT_ONE_TOKEN_RULE},
     {role: user, content: prompt}]`` and ``num_predict=4``,
     ``options={stop=["\n", ".", " "], temperature=0.0, seed=0}``;
   * adaptive timeout: small models 30 s, medium 60 s, large
     (>= 30B) 240 s — closes the qwen3.5:35b false negative.

6. **W34 manifest-v4 CID** — SHA-256 over (parent_w33_cid,
   live_attestation_cid, multi_anchor_cid, host_topology_cid).
   The W22..W33 manifest CIDs do NOT include
   ``live_attestation_cid`` or ``multi_anchor_cid``; the W34
   manifest-v4 does — closing cross-component swap avenues that
   the W33 manifest-v3 alone cannot detect.

W34 charges at most **2 visible tokens / cell** on the wire (one
``<w34_ref:DDDD>`` + one optional ``<w34_attest:DDDD>`` when live
attestation is present).  The trivial-W34 path
(``multi_anchor_quorum_min=1`` AND ``live_attestation_disabled=True``
AND ``manifest_v4_disabled=True``) reduces to W33 byte-for-byte —
the H2 byte-for-W33 anchor.

W34 does NOT touch transformer KV caches, embedding tables, or
attention weights.  W34 does NOT learn a feature-signature model.
W34 does NOT prove temporal ordering at the model layer.  W34 is
explicitly a **capsule-layer audited proxy** for live-aware
multi-anchor adjudication.

---

## 2. Pre-committed hard gates

| Gate | Description | Pass iff |
|---|---|---|
| **H1** | Real mechanism beyond W33 with ≥ 14 enumerated failure modes in ``verify_live_aware_multi_anchor_ratification``, **disjoint from W22..W33** sets (cumulative 84 across W22+W29+W30+W31+W32+W33+W34) | 14 distinct ``LatentVerificationOutcome.reason`` strings; no overlap with prior 70 |
| **H2** | Trivial-W34 byte-for-W33 invariant on R-81-TRIVIAL-W34 (``multi_anchor_quorum_min=1`` AND ``live_attestation_disabled=True`` AND ``manifest_v4_disabled=True``) | ``mean_total_w34_visible_tokens == mean_total_w33_visible_tokens`` byte-for-byte across 5/5 seeds; every cell yields ``W34_BRANCH_TRIVIAL_MULTI_ANCHOR_PASSTHROUGH`` |
| **H3** | Manifest-v4 cross-component tamper detection on R-81-MANIFEST-V4-TAMPER, reject rate ≥ 95 % across 5 named tampers per ratified cell | 1.000 reject rate across 5/5 seeds × N cells × 5 tampers |
| **H4** | Honest scope of new mechanism stated in module docstring (NOT learned model, NOT transformer-internal, NOT runtime KV transplant; the response-feature signature is the audited proxy with explicit gap to native-latent stated) | Module docstring contains all four disclaimers verbatim |
| **H5** | At least 5 named falsifiers, all empirically observed, disjoint from W33's 4 | W34-Λ-trivial-multi-anchor + W34-Λ-no-anchor-disagreement + W34-Λ-anchor-betrays + W34-Λ-frozen-host-decay + W34-Λ-mis-feature-signature; every falsifier observed empirically |
| **H6** | **Multi-anchor strict trust-precision gain** on R-81-DOUBLE-ANCHOR-COMPROMISE — a regime where the W33 single-anchor itself becomes compromised mid-session: ``Δ_trust_precision_w34_w33 ≥ +0.10`` AND no correctness regression AND ≤ 2 token/cell overhead AND trust precision = 1.000 across 5/5 seeds × ≥ 16 cells | Δ_trust_prec ≥ 0.10; min trust prec = 1.000 |
| **H7** | **Pre-flight infra discipline closes W33-INFRA-1 + W33-INFRA-2** on R-81-XLLM-PREFLIGHT: when the W34 preflight fails for a host, that host's oracles are explicitly skipped (not silently timed out); when ``/api/tags`` confirms availability and the chat-template is used with ``num_predict=4`` + stop tokens, at least 1 host returns valid one-word responses on ≥ 80 % of one-word probes | Pre-flight check fires; valid-response rate ≥ 80 % on at least one host |
| **H8** | **Native-latent audited proxy is byte-stable** on R-81-RESPONSE-FEATURE-SIGNATURE: closed-form deterministic ``response_feature_signature`` reproducible byte-for-byte across 2/2 runs of the same prompt at temp 0; signature length = 16 hex chars (64 bits) | byte-equal across 2 runs |
| **H9** | Release-readiness clause | SDK_VERSION bumped to coordpy.sdk.v3.35; pyproject.toml 0.5.8; ``__experimental__`` extended with W34 symbols; CHANGELOG entry; stable runtime contract byte-for-byte unchanged |
| **H10** | Focused regression green | ≥ 25 W34 unit tests + all phase69-81 + wider coordpy suite all PASS |

**Hard-gate aggregate**: 10/10 PASS = STRONG SUCCESS; 8-9/10 PASS = PARTIAL SUCCESS; ≤ 7/10 PASS = FAILURE.

---

## 3. Pre-committed soft gates

| Gate | Description | Pass iff |
|---|---|---|
| **S1** | Cross-architecture live trust-calibration evidence on R-81-XLLM-LIVE-MULTI-ANCHOR with the W34 preflight discipline | At least one host on at least one prompt class returns gold-correlated agreement byte-for-byte across two independent runs at temp 0; if honestly null, ``W34-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE`` is the new live-magnitude conjecture |
| **S2** | Mac 2 returning OR honest fallback | 192.168.12.248 reachable AND fresh ``/api/tags`` succeeds (PASS), or honestly recorded as 29th milestone with explicit Mac 2 still-unavailable note (HONESTLY-NULL acceptable) |
| **S3** | Trust precision = 1.000 on R-81-DOUBLE-ANCHOR-COMPROMISE across 5/5 seeds | min_trust_prec_w34 = 1.000 |
| **S4** | Token overhead bound ≤ 2 token/cell vs W33 | max_overhead_w34 ≤ 2; mean ≤ 2 |
| **S5** | At least one earlier conjecture sharpened OR discharged | Either ``W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE`` discharged on a corrected-infra regime, OR a new explicit limitation theorem is proved (``W34-L-MULTI-ANCHOR-CAP``: when all K anchors are simultaneously compromised, no multi-anchor mechanism can recover at the capsule layer) |
| **S6** | At least one infra blocker closed | Either W33-INFRA-1 closed (preflight fires correctly), OR W33-INFRA-2 closed (chat-template + num_predict=4 yields valid one-word responses), OR both |

**Soft-gate aggregate**: 4-6/6 PASS, 0-2/6 honestly-null with full disclosure.

---

## 4. Verdict rule

* **STRONG SUCCESS** = 10/10 hard gates PASS.
* **PARTIAL SUCCESS** = 8-9/10 hard gates PASS, with explicit
  honest-null dispositions on the missing gates AND at least one
  named open conjecture inheriting forward.
* **FAILURE** = ≤ 7/10 hard gates PASS.

---

## 5. Pre-committed falsifiers

* **W34-Λ-trivial-multi-anchor** — when ``multi_anchor_quorum_min=1``
  AND ``live_attestation_disabled=True`` AND
  ``manifest_v4_disabled=True``, W34 = W33 byte-for-byte.  Predicts
  ``mean_total_w34_visible_tokens == mean_total_w33_visible_tokens``.

* **W34-Λ-no-anchor-disagreement** — when all K registered anchors
  always agree throughout the session, the W34 multi-anchor branch
  ties W33 single-anchor byte-for-byte (both reduce to the unique
  agreed reference).

* **W34-Λ-anchor-betrays** — when the W33 single-anchor itself
  becomes compromised mid-session (the original W33 design has no
  defense against this), W33 single-anchor commits to the wrong
  detrust direction; W34 with ``K ≥ 2`` and ``anchor_quorum_min ≥ 2``
  recovers because the second anchor still agrees with the
  consortium and detrusts the betraying anchor.  Predicts
  Δ_trust_prec_w34_w33 ≥ +0.10.

* **W34-Λ-frozen-host-decay** — when ``host_decay_factor=1.0``
  (no decay), the host-aware EWMA decay never fires; W34 ties W33
  on regimes where some host is unresponsive (the unresponsive
  host's oracles still get full agreement signal because abstain
  → 1.0 in W33).

* **W34-Λ-mis-feature-signature** — when two distinct LLM responses
  collide on the W34 response-feature signature (constructed
  adversarially: two prompts whose responses share first_token_class
  + length_bucket + structural_hash but differ semantically), the
  W34 mechanism detects no feature-class shift and is no worse than
  W33 (i.e. does not regress).

---

## 6. Discharges targeted

Empirically:
* **W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE** — if S1 succeeds with
  byte-equal cross-host agreement on at least one prompt class
  with gold-correlation evidence, this conjecture is discharged on
  the live-magnitude axis.  Otherwise it remains open as honestly
  null with the corrected-infra evidence.
* **W33-INFRA-1** — if H7 passes, this is closed (preflight fires
  correctly; qwen3.5:35b false-negative removed).
* **W33-INFRA-2** — if H7 passes, this is closed (chat-template +
  ``num_predict=4`` + stop tokens yields valid one-word responses).

Theoretically:
* **W34-L-MULTI-ANCHOR-CAP** — limitation theorem: when all K
  anchors are simultaneously compromised at the capsule layer, no
  multi-anchor mechanism (including W34) can recover.  Proof by
  inspection: the only signal available at the capsule layer is the
  agreement between probes; if all K anchors agree on the wrong
  reference, the EWMA update converges to high agreement on the
  wrong direction.  Native-latent (architecture-dependent) is
  required to break this.

---

## 7. Bench requirements

Phase 81 / R-81 benchmark family:
* **R-81-TRIVIAL-W34** — H2 anchor; trivial registry; W34 = W33 byte-for-byte.
* **R-81-DOUBLE-ANCHOR-COMPROMISE** — H6 main load-bearing claim; the
  W33 single-anchor itself becomes compromised mid-session; W34 with
  K=2 + anchor_quorum_min=2 recovers.
* **R-81-NO-ANCHOR-DISAGREEMENT** — W34-Λ-no-anchor-disagreement falsifier; all anchors always agree.
* **R-81-ANCHOR-BETRAYS** — W34-Λ-anchor-betrays observation; same regime as R-81-DOUBLE-ANCHOR-COMPROMISE but with K=1 (which fails — the W33 baseline behavior).
* **R-81-FROZEN-HOST-DECAY** — W34-Λ-frozen-host-decay falsifier;
  ``host_decay_factor=1.0``.
* **R-81-MIS-FEATURE-SIGNATURE** — W34-Λ-mis-feature-signature falsifier; collision construction.
* **R-81-MANIFEST-V4-TAMPER** — H3 cross-component tamper detection;
  5 named tampers per cell.
* **R-81-XLLM-PREFLIGHT** — H7 infra blocker closure; live ``/api/tags`` check + adaptive timeout + chat-template + ``num_predict=4``.
* **R-81-RESPONSE-FEATURE-SIGNATURE** — H8 native-latent audited proxy byte-stability; same prompt at temp 0 reproduces signature exactly across 2 runs.
* **R-81-XLLM-LIVE-MULTI-ANCHOR** — S1 best-effort live cross-host evidence with corrected infra.

---

## 8. Honest scope (what W34 does NOT claim)

* W34 does NOT claim "we solved context."
* W34 does NOT claim a learned feature-signature model.  Closed-form deterministic.
* W34 does NOT claim transformer-internal latent control.  The response-feature signature is the audited proxy at the capsule layer.
* W34 does NOT prove temporal ordering at the model layer.
* W34 does NOT close ``W33-C-NATIVE-LATENT`` (architecture-dependent; the next true wall) unless on a small ``W34-L-MULTI-ANCHOR-CAP`` limitation theorem.
* W34 does NOT close ``W33-C-MULTI-HOST`` unless Mac 2 returns.
* W34 does NOT commit to discharging ``W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE`` if the available LLMs at temp 0 agree on every gold-verifiable prompt or if no prompt class is gold-correlated.
* W34 does NOT touch transformer KV caches, embedding tables, attention weights.
