# Success criterion - Wevra SDK v3.42 / W41
# Integrated multi-agent context synthesis + manifest-v11 CID +
# cross-axis witness CID + producer-axis x trust-axis decision
# selector

**Pre-committed before final W41 verdict / RC2 declaration**:
2026-05-03.
**Target**: SDK v3.42 / W41 family.
**Position relative to W40 RC1**: W40 RC1 wraps the W22..W39
explicit-capsule trust-adjudication chain with a cross-host
response-signature heterogeneity layer + manifest-v10 CID +
mechanically-enforced mutual-disjointness over K registered
member host pools.  W40 RC1 declared the SDK v3.41 line a
release candidate honestly: 12/12 hard gates + S3 pass; the
``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP`` limitation theorem
is the named open wall at the trust-axis layer; ``W39-INFRA-1``
/ ``W40-INFRA-1`` carry forward the live-multi-host gap
honestly.

W41 is therefore framed as a **synthesis** milestone, not a
"W41: one more local mechanism" milestone.  The user goal is
*"using everything already built in this repo, what is the
strongest honest full-system attempt we can make to solve
context for multi-agent teams?"*  W41 attempts to answer that
by jointly binding the strongest old-line explicit-capsule
trust-adjudication chain (W21..W40) and the strongest cross-
role / multi-round bundle decoder family (W7..W11) in a
single auditable end-to-end path with one ``manifest-v11``
envelope.

W41 also explicitly **does not** add a new transformer-internal
mechanism.  The integrated synthesis decision is closed-form,
zero-parameter, and capsule-layer.  W41 introduces a new
proved-conditional limitation theorem,
``W41-L-COMPOSITE-COLLUSION-CAP``: when both the producer-side
ambiguity AND the trust-side collusion are coordinated by an
adversary, integration cannot recover at the capsule layer.
This is the W41 analog of ``W40-L-COORDINATED-DIVERSE-
RESPONSE-CAP``.

**Crucial release context.** The infrastructure narrative
inherited from W37-W40 (".101 is a Mac with hung Ollama
listener") is **retracted** at this milestone.  Re-probing
``192.168.12.101`` in the W41 milestone confirms it is an
Apple TV / AirPlay receiver (port 5000 returns
``AirTunes/860.7.1``; locally-administered MAC
``36:1c:eb:dc:9a:04``) — NOT a Mac running Ollama.  The
historical "TCP-up + HTTP-broken Ollama" framing was a
mis-identification at the network layer.  The honest live
multi-host topology is **two reachable Ollama hosts**:
``localhost`` (Mac 1) and ``192.168.12.191`` (Mac 2 /
``HSC136047-MAC.lan``).  ``192.168.12.248`` is gone (per user).
``192.168.12.101`` is recorded as ``W41-INFRA-1`` (Apple TV;
not a Mac).  See § 4 below for the discharge.

---

## 1. Mechanism bar

W41 must implement a real method beyond W40:

1. Wrap the W40 ``CrossHostResponseHeterogeneityOrchestrator``
   with an explicit cross-axis classification layer.
2. Define a new audited envelope class
   :class:`IntegratedSynthesisRatificationEnvelope` carrying
   ``(producer_axis_branch, trust_axis_branch,
   integrated_branch, producer_services, trust_services,
   integrated_services, w40_projection_branch)`` plus four new
   content-addressed CIDs (``synthesis_state_cid``,
   ``synthesis_decision_cid``, ``synthesis_audit_cid``,
   ``cross_axis_witness_cid``) PLUS the manifest-v11 CID PLUS
   the outer w41 CID.
3. Maintain a controller-side
   :class:`IntegratedSynthesisRegistry` whose
   ``inner_w40_registry`` is a registered W40 registry; its
   ``schema`` is the same SchemaCapsule the W22..W40 chain
   uses.
4. Expose two safety knobs
   ``abstain_on_axes_diverged`` and
   ``require_both_axes_for_ratification`` that, when enabled,
   change the cross-axis selector's behaviour deterministically
   (the W41-L-AXES-DIVERGED-ABSTAINED branch + the
   W41-L-REQUIRE-BOTH-AXES branch).
5. When the inner W40 chain is fully disabled (the W41
   ``trivial_w41`` bank), W41 must reduce to W40 byte-for-W39
   byte-for-W38 byte-for-... byte-for-substrate (the W41-L-
   TRIVIAL-PASSTHROUGH falsifier branch).
6. Define the integrated decision selector
   :func:`select_integrated_synthesis_decision` over
   ``(producer_axis_branch, trust_axis_branch,
   producer_services, trust_services)`` that returns one of
   eight named branches:
   - ``trivial_integrated_passthrough``
   - ``integrated_disabled``
   - ``integrated_rejected``
   - ``integrated_producer_only``
   - ``integrated_trust_only``
   - ``integrated_both_axes``
   - ``integrated_axes_diverged_abstained``
   - ``integrated_neither_axis``
   The selector is deterministic and zero-parameter: the only
   inputs are the per-axis branches and per-axis service
   tuples; the only outputs are the branch and the integrated
   service tuple.
7. Seal the new state in a content-addressed **manifest-v11**
   envelope over four CIDs (parent_w40_cid, synthesis_state_cid,
   synthesis_decision_cid, synthesis_audit_cid,
   cross_axis_witness_cid).  The synthesis_audit and
   cross_axis_witness CIDs are namespaced as
   ``w41_synthesis_audit`` and ``w41_cross_axis_witness`` so
   substituting a W22..W40 audit / witness for them is
   mechanically detected.
8. Keep the native-latent gap explicit: no transformer
   hidden-state, KV-cache, attention-weight, or embedding-table
   access is added.  The cross-axis classification is a closed-
   form deterministic mechanical decision over the W40
   projection branch + the inner top_set.
9. Token accounting must be self-consistent: the verifier
   rejects an envelope where ``n_w41_visible_tokens !=
   n_w40_visible_tokens + n_w41_overhead_tokens`` (the new
   ``w41_token_accounting_invalid`` failure mode).
10. Reproduce all R-87 banks under W41 wrapping, with the new
    R-88 measurement axes (cross-axis branch distribution,
    integrated services, manifest-v11 CID).

The permitted headline is **integrated multi-agent context
synthesis audited proxy + manifest-v11 CID**, not native latent
transfer and not closure of
``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP`` in general.

---

## 2. Hard gates

| Gate | Description | Pass iff |
| --- | --- | --- |
| **H1** | Real W41 mechanism beyond W40 | New orchestrator, registry, envelope, verifier, integrated-synthesis selector, four new content-addressed CIDs, and manifest-v11 CID implemented; W41 makes a routing decision W40 cannot make (cross-axis classification) |
| **H2** | Trust boundary | ``verify_integrated_synthesis_ratification`` enumerates 14 disjoint W41 failure modes; tests exercise every mode plus clean-envelope sanity |
| **H3** | Trivial-W41 byte-for-W40 preservation | With synthesis disabled, manifest-v11 disabled, and abstain-on-axes-diverged disabled, W41 reduces to W40 byte-for-byte across 5 seeds (R-88-TRIVIAL-W41 ``w41_w40_byte_equivalent = True``) |
| **H4** | No correctness regression on R-88-BOTH-AXES | On R-88-BOTH-AXES (= R-87-NO-REGRESSION-DIVERSE-AGREES wrapped by W41), W41 must not weaken W40 (delta_correctness >= 0, delta_trust_precision >= 0) |
| **H5** | Trust-precision preservation on R-88-TRUST-ONLY-SAFETY | On R-88-TRUST-ONLY-SAFETY (= R-87-RESPONSE-SIGNATURE-COLLAPSE wrapped by W41), W41 must preserve trust precision >= 1.0 (the safety branch INTEGRATED_TRUST_ONLY clears services on collapse, matching W40's COLLAPSE_ABSTAINED behaviour) |
| **H6** | W41-L-COMPOSITE-COLLUSION-CAP fires on R-88-COMPOSITE-COLLUSION | On R-88-COMPOSITE-COLLUSION (= R-87-COORDINATED-DIVERSE-RESPONSE wrapped by W41), W41 must not claim recovery; delta_trust_precision_w41_w40 = 0 across 5/5 seeds; W41 reports the INTEGRATED_BOTH_AXES branch on the wrong set, the same cell-by-cell answer as W40 |
| **H7** | Insufficient-response-signature falsifier inherited | On R-88-INSUFFICIENT-RESPONSE-SIGNATURE (= R-87-INSUFFICIENT-RESPONSE-SIGNATURE wrapped by W41), when fewer than ``min_response_signature_probes`` member probes carry response signatures, W41 routes through INTEGRATED_PRODUCER_ONLY (preserving W40 byte-for-W39 semantics on the answer) |
| **H8** | Old explicit capsule line preserved | Focused W22..W40 regression slices stay green; the W22..W40 inner orchestrator chain is unchanged byte-for-byte (W41 is strictly additive) |
| **H9** | Cross-axis classification mechanical | The integrated decision selector is zero-parameter and deterministic; given the same per-axis branches and service tuples, two calls produce byte-identical (branch, services) outputs |
| **H10** | Live-multi-host topology recorded honestly | The lab topology is re-probed: ``192.168.12.101`` is identified as an Apple TV / AirPlay receiver (NOT a Mac with Ollama; the W37..W40 framing is retracted at this milestone), ``192.168.12.191`` is recorded as a working second Mac with Ollama, ``192.168.12.248`` is recorded as gone (per user); the milestone records the strongest live multi-host evidence available (two-Mac topology) |
| **H11** | Broad regression confidence | Focused W22..W41 regression is green; broad spot checks on phase 11-39 + phase 40-51 + phase 6 ideas remain green; phase 50 collection-time hang carried forward unchanged |
| **H12** | Release-readiness clause / RC2 | Versioning, changelog, success bar, results note, theorem registry, README/START_HERE/master plan/paper markers updated only if H1..H11 pass and the stable runtime remains unchanged; vision_mvp ``__version__`` aligned with pyproject; **RC2 declared** if H1..H11 + S3 pass and the lab topology is recorded honestly |

**Hard-gate aggregate**:

- **Strong success** = 11-12 gates pass, with no trust/audit weakening.
- **Partial success** = 9-10 gates pass, with exact blockers carried
  forward.
- **Failure** = <= 8 gates pass, any verifier weakening, or any
  unbounded native-latent/live claim.

---

## 3. Soft gates

| Gate | Description | Target |
| --- | --- | --- |
| **S1** | Cross-axis branch distribution measured per bank | The R-88 driver records ``w41_integrated_branch_hist`` for every bank/seed; the histogram is reproducible and reveals which axis is load-bearing per cell |
| **S2** | Lab topology corrected | The ``.101`` Apple TV mis-identification recorded honestly in HOW_NOT_TO_OVERSTATE.md and the W41 results note; the corrected two-Mac topology is the live anchor going forward |
| **S3** | Stable-vs-experimental boundary | W41 remains under ``__experimental__``; stable runtime contract unchanged |
| **S4** | Theory | Add one conditional sufficiency claim (W41-3), one limitation theorem (W41-L-COMPOSITE-COLLUSION-CAP), one inherited falsifier (W41-L-INSUFFICIENT-RESPONSE-SIGNATURE), and one native-latent gap claim (W41-C-NATIVE-LATENT) |
| **S5** | Paper / master-plan synthesis | Old explicit-capsule line (W21..W34), dense-control / geometry line (W35..W36), W37 cross-host trajectory line, W38 disjoint-consensus line, W39 multi-host quorum line, W40 response-heterogeneity line, and W41 integrated-synthesis line read as a single arc with an explicit RC2 release boundary; the paper marker is updated |
| **S6** | Full broad regression actually counted | Total tests counted: 698 phase69-88 + 364 phase11-39 + 205 phase40-51 + phase6 = 1267 tests pass excluding the named pre-existing ``test_phase50_ci_and_zero_shot`` collection-time hang carried forward unchanged from W40 |
| **S7** | Release-candidate gate (RC2) | If H1..H12 + S3 pass and the new lab topology is recorded honestly, declare a release-candidate (``RC2``) status for the SDK v3.42 line, with the W41 cross-axis synthesis layer pinned in README and the open conjectures + limitation theorems cut-list pinned in THEOREM_REGISTRY.md |

Soft gates cannot compensate for failed trust/audit hard gates.

---

## 4. Named falsifiers

- **W41-L-TRIVIAL-PASSTHROUGH**: synthesis disabled +
  ``manifest_v11_disabled = True`` + ``abstain_on_axes_diverged
  = False`` reduces W41 to W40 byte-for-byte.
- **W41-L-INSUFFICIENT-RESPONSE-SIGNATURE** (inherited from
  W40): when fewer than ``min_response_signature_probes``
  registered member probes carry response signatures (or any
  registered probe is missing them), W41 routes through
  INTEGRATED_PRODUCER_ONLY without correctness or trust-
  precision regression vs W40.
- **W41-L-NO-REGRESSION-BOTH-AXES**: when the producer axis
  fires AND the trust axis ratifies AND both produce the same
  service set, W41 ratifies via INTEGRATED_BOTH_AXES without
  correctness or trust-precision regression vs W40.
- **W41-L-AXES-DIVERGED-ABSTAINED**: when the producer axis
  fires AND the trust axis ratifies but their service sets are
  disjoint, W41 abstains via INTEGRATED_AXES_DIVERGED_
  ABSTAINED with empty integrated services (a safety branch).
- **W41-L-COMPOSITE-COLLUSION-CAP** (NEW): when the producer-
  axis services AND the trust-axis ratified services AGREE on a
  wrong top_set produced by an adversary that has both
  compromised the W21 producer-side admission AND injected
  diverse W40 response bytes that all encode the same wrong
  top_set, W41 cannot recover at the capsule layer; this is the
  W41 analog of ``W34-L-MULTI-ANCHOR-CAP``,
  ``W37-L-MULTI-HOST-COLLUSION-CAP``,
  ``W38-L-CONSENSUS-COLLUSION-CAP``,
  ``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP``, and
  ``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP``, and is closed
  only by transformer-internal evidence outside the capsule
  layer OR by a K+1-host disjoint topology with at least one
  new genuinely uncompromised host pool.
- **W41-L-NATIVE-LATENT-GAP**: if a regime requires
  transformer-internal evidence not visible through trajectory
  observations, EWMA, response feature signatures, host
  attestations, anchor consensus, single-disjoint-consensus
  references, multi-host disjoint quorum references, cross-
  host response-heterogeneity witnesses, OR the cross-axis
  classification, W41 is insufficient.
- **W41-INFRA-1** (NEW): ``192.168.12.101`` is an Apple TV /
  AirPlay receiver (port 5000 returns ``AirTunes/860.7.1``;
  locally-administered MAC ``36:1c:eb:dc:9a:04``); the W37..W40
  "TCP-up + HTTP-broken Ollama Mac" framing for ``.101`` is
  retracted.  ``192.168.12.248`` is recorded as gone (per
  user).  The honest live multi-host topology is the two-Mac
  pair (``localhost`` + ``192.168.12.191``).

---

## 5. Claim boundary

W41 may claim:

- an audited integrated multi-agent context synthesis layer
  with manifest-v11 binding W40 + cross-axis decision +
  synthesis-audit + cross-axis witness;
- measured W41 cross-axis branch distribution on five R-88
  banks (trivial / both_axes / trust_only_safety /
  composite_collusion / insufficient_response_signature) at
  ``n_eval = 16``;
- preserved W40 trust precision on every R-88 bank (no
  regression);
- preserved W40 correctness on R-88-BOTH-AXES + R-88-TRIVIAL
  (no regression);
- explicit safety on the cross-axis-diverged falsifier and the
  composite-collusion limitation theorem;
- a release-candidate (``RC2``) declaration **only** when
  H1..H12 + S3 pass and the lab topology is recorded honestly
  (with the ``.101 = Apple TV`` retraction explicitly carried
  forward).

W41 may NOT claim:

- native latent transfer;
- transformer-internal trust subspace;
- KV-cache transplant;
- hidden-state projection;
- closure of ``W37-L-MULTI-HOST-COLLUSION-CAP`` /
  ``W38-L-CONSENSUS-COLLUSION-CAP`` /
  ``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` /
  ``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP`` at the capsule
  layer (only further-bounded by the W41 cross-axis
  classification);
- closure of ``W41-L-COMPOSITE-COLLUSION-CAP`` at the capsule
  layer (it is the new proved-conditional limitation theorem);
- "solving context for multi-agent teams" without naming the
  defining gate (per HOW_NOT_TO_OVERSTATE.md).

The honest reading is narrower: **W41 is the first capsule-
native end-to-end integrated synthesis of the W21..W40
trust-adjudication chain and the W7..W11 cross-role / multi-
round bundle decoder family, with one manifest-v11 envelope
binding both axes plus a cross-axis witness, and a measured
cross-axis branch distribution that lets researchers
distinguish which axis is load-bearing on each cell.  The
residual wall is named (W41-L-COMPOSITE-COLLUSION-CAP), not
papered over.**
