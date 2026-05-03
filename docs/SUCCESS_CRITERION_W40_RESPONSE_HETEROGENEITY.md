# Success criterion - CoordPy SDK v3.41 / W40
# Cross-host response-signature heterogeneity ratification +
# manifest-v10 CID + cross-host response-text Jaccard divergence
# guard

**Pre-committed before final W40 verdict / RC1 declaration**:
2026-05-03.
**Target**: SDK v3.41 / W40 family.
**Position relative to W39**: W39 wraps W38 with a K-of-N
mutually-disjoint quorum of disjoint consensus-reference probes
(each on a physically distinct host pool, mechanically disjoint
from the W37 trajectory hosts AND mutually disjoint from every
other quorum probe's host pool).  W39 raises the capsule-layer
adversary bar from "compromise 2 of N trajectory hosts AND the
single disjoint registered consensus reference" to "compromise
2 of N trajectory hosts AND ``quorum_min`` of the K mutually-
disjoint registered consensus references, each on a physically
distinct host pool".

W39's deepest open wall is **W39-L-FULL-DISJOINT-QUORUM-
COLLUSION-CAP**: when **all** K registered disjoint quorum
probes are themselves compromised in lock-step with the
colluding trajectory hosts (each emits the same wrong top_set),
W39 cannot recover at the capsule layer.  W39's named open
conjecture **W39-C-NATIVE-LATENT** states that true
transformer-internal trust-state projection may separate regimes
where all capsule-visible host/probe/evidence/trajectory/
consensus/quorum signals are either absent or coordinated.
W39's open conjecture **W39-C-LIVE-TRUNCATION-RECOVERY** (NEW
in W39, empirical-suggestive on a single live ``h2o`` probe)
states that the W39 quorum can recover from trajectory-pair-
only truncation errors when quorum members are not subject to
the same ``num_predict``/stop-token regime.

W40 must therefore be judged as a **response-heterogeneity-
bounding** milestone aimed directly at the
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` adversary bar,
**not** another sideways research axis and **not** another
narrow benchmark-only win.  W40 adds a new evidence axis
**orthogonal to top_set**: the per-member response **text bytes**
themselves.  Even if K colluders coordinate their declared
top_set, naturally-independent K probes should produce
heterogeneous response text bytes (different paraphrases,
different surface forms, different tokenisations).  If the K
member probes' response text bytes collapse to homogeneity --
all K identical, or all K with mean pairwise Jaccard divergence
below a registered threshold -- W40 abstains via
``RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED`` even when W39 would
have ratified.

W40 raises the capsule-layer adversary bar from "compromise 2
of N trajectory hosts AND ``quorum_min`` of the K mutually-
disjoint registered consensus references" to "compromise 2 of
N trajectory hosts AND ``quorum_min`` of the K mutually-
disjoint registered consensus references AND inject K diverse
response text bytes that all encode the same wrong top_set".

When the adversary diversifies response text bytes while
holding the wrong top_set (the **coordinated-diverse-response**
attack), W40 cannot recover; this is the new
**W40-L-COORDINATED-DIVERSE-RESPONSE-CAP** limitation theorem
(the W40 analog of ``W34-L-MULTI-ANCHOR-CAP``,
``W37-L-MULTI-HOST-COLLUSION-CAP``,
``W38-L-CONSENSUS-COLLUSION-CAP``, and
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP``).  It is closed
only by transformer-internal evidence outside the capsule layer
or by a K+1-host disjoint quorum topology with a new genuinely
uncompromised pool.

W40 also encodes the second axis of release readiness: the
W39-INFRA-1 finding (``192.168.12.101`` Ollama HTTP listener
hung after first model-load contact under one-word probe budget)
must be re-probed; if not restorable in this environment, the
release boundary records ``.101`` honestly as a TCP-up + HTTP-
broken bound rather than as a fully-discharged third physical
host.  This success criterion is the explicit release-readiness
gate for SDK v3.41 / RC1.

The W40 mechanism is still a closed-form, zero-parameter,
audited capsule-layer proxy.  It does **not** read transformer
hidden states, transplant KV cache, claim native latent
transfer, or use a learned heterogeneity model.  Its
heterogeneity score is a closed-form Jaccard divergence over
canonical sorted token bags, computed pair-wise across the K
member probes' response text bytes and reduced via mean.

---

## 1. Mechanism bar

W40 must implement a real method beyond W39:

1. Wrap W39 with a **cross-host response-heterogeneity** layer.
2. Define a new audited probe class
   :class:`MultiHostResponseHeterogeneityProbe` carrying K
   :class:`ResponseSignatureProbe` member probes, each tagged
   with ``(member_index, host_ids, oracle_ids,
   response_token_bag, response_signature_cid, cell_idx)``.
   ``response_signature_cid`` is SHA-256 over the canonical
   normalized response text bytes; ``response_token_bag`` is
   the canonical sorted whitespace token bag.  The probe is a
   controller-pre-registered audited capsule-layer artefact,
   **not** a runtime ground-truth oracle and **not** a learned
   heterogeneity model.
3. Maintain a controller-side
   :class:`CrossHostResponseHeterogeneityRegistry` whose K
   registered member host pools are (a) mechanically disjoint
   from the W37 trajectory hosts AND from the W38 single
   consensus reference (the W38/W39 precondition) AND (b)
   mutually disjoint from one another (the W39 precondition,
   inherited).  Building a registry whose any two registered
   member host pools have non-empty intersection raises
   :class:`MutuallyDisjointTopologyError` at construction time;
   the verifier additionally rejects an envelope whose declared
   topology has overlapping pool pair
   (``w40_response_mutual_disjointness_violation``).
4. Expose a single new safety knob
   ``allow_response_signature_collapse_abstain`` that, when
   enabled, lets W40 convert a W39 ``RATIFIED`` decision into a
   ``RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED`` decision iff the
   mean pairwise Jaccard divergence over the K member probes'
   ``response_token_bag``s falls strictly below
   ``response_text_diversity_min``.
5. When fewer than ``min_response_signature_probes`` member
   probes are registered for the current cell, OR when any
   member probe is missing its ``response_signature_cid`` /
   ``response_token_bag``, W40 must reduce to W39 byte-for-byte
   (the ``RESPONSE_SIGNATURE_INSUFFICIENT`` falsifier branch).
6. When at least ``min_response_signature_probes`` member
   probes are registered AND their pairwise mean Jaccard
   divergence is at or above ``response_text_diversity_min``,
   W40 must ratify the W39 reroute (the
   ``RESPONSE_SIGNATURE_DIVERSE`` branch).
7. When the W39 layer did not RATIFY (i.e. W37/W38 did not
   reroute, or W39 itself abstained / fell through), W40 must
   not change the answer (the
   ``RESPONSE_SIGNATURE_NO_TRIGGER`` branch).
8. Seal the new state in a content-addressed **manifest-v10**
   envelope over six CIDs (parent_w39_cid,
   response_signature_state_cid, response_signature_audit_cid,
   response_signature_topology_cid,
   response_signature_decision_cid,
   response_heterogeneity_witness_cid).
9. Keep the native-latent gap explicit: no transformer
   hidden-state, KV-cache, attention-weight, or embedding-table
   access is added.  The cross-host response-heterogeneity is K
   audited capsule-layer probes, NOT a runtime ground-truth
   oracle and NOT a transformer-internal trust-state projection.
10. Mechanically enforce **mutual** disjointness of the K
    registered member host pools (inherited from W39); the
    W40 registry raises :class:`MutuallyDisjointTopologyError`
    at construction time; the verifier rejects an envelope
    claiming an overlapping pool pair.
11. Reproduce W39-INFRA-1 in the live xLLM probe path; record
    ``192.168.12.101`` honestly as TCP-up + HTTP-broken if the
    Ollama HTTP listener still returns "Empty reply from server"
    after first model-load contact.
12. Keep the W39 multi-host disjoint quorum probe-provider path
    intact: W40 only acts on W39's RATIFIED branch and is a
    pure superset of W39 in terms of safety branches.

The permitted headline is **cross-host response-signature
heterogeneity audited proxy + manifest-v10 CID**, not native
latent transfer and not closure of
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` in general.

---

## 2. Hard gates

| Gate | Description | Pass iff |
| --- | --- | --- |
| **H1** | Real W40 mechanism beyond W39 | New orchestrator, registry, envelope, verifier, response-heterogeneity probe, response-decision selector, mechanical mutual-disjointness enforcement (inherited from W39), and manifest-v10 CID implemented; W40 makes a routing decision W39 cannot make |
| **H2** | Trust boundary | ``verify_cross_host_response_heterogeneity_ratification`` enumerates 14 disjoint W40 failure modes; tests exercise every mode plus clean-envelope sanity |
| **H3** | Trivial-W40 old-regime preservation | With response-heterogeneity disabled and manifest-v10 disabled, W40 reduces to W39 byte-for-byte across 5 seeds |
| **H4** | Load-bearing response-signature-collapse mitigation | On R-87-RESPONSE-SIGNATURE-COLLAPSE, W40 improves trust precision over W39 by at least +0.20 and restores trust precision to >= 0.95 with no correctness regression below W39, adding <= 1 visible token/cell |
| **H5** | No-regression-diverse-agrees falsifier | On R-87-NO-REGRESSION-DIVERSE-AGREES, W40 must not weaken W39 (delta_correctness >= 0, delta_trust_precision >= 0) |
| **H6** | Coordinated-diverse-response limitation theorem | On R-87-COORDINATED-DIVERSE-RESPONSE, W40 must not claim recovery; the W40-L-COORDINATED-DIVERSE-RESPONSE-CAP limitation theorem fires (W40 = W39 within 1 visible token); delta_trust_precision = 0 across 5/5 seeds |
| **H7** | Insufficient-response-signature falsifier | On R-87-INSUFFICIENT-RESPONSE-SIGNATURE, when fewer than ``min_response_signature_probes`` registered probes carry response signatures (or any probe is missing them), W40 reduces to W39 byte-for-byte (delta = 0); the W40 audit envelope still records the RESPONSE_SIGNATURE_INSUFFICIENT branch |
| **H8** | Old explicit capsule line preserved | Focused W33, W34, W35, W36, W37, W38, W39 regression slices stay green; W40 composes W21/W33/W34/W35/W36/W37/W38/W39 rather than bypassing them |
| **H9** | Mutual-disjointness mechanically enforced | Building a W40 registry whose any two registered member host pools have non-empty intersection raises ``MutuallyDisjointTopologyError`` at construction time; the verifier additionally rejects an envelope whose registered topology claims an overlapping pool pair (``w40_response_mutual_disjointness_violation``) |
| **H10** | W39-INFRA-1 re-probed | The lab topology is re-probed: ``192.168.12.101`` is recorded honestly (TCP-up + HTTP-broken if not restorable, or restored and used in the live xLLM probe if recoverable); ``192.168.12.248`` is recorded as still ARP-incomplete; the milestone records the strongest live multi-host evidence available |
| **H11** | Broad regression confidence | Full ``pytest vision_mvp/tests -q`` runs to completion at least once during the milestone with the result counted (or honestly carried forward with named exclusions); focused W22..W40 regression is green |
| **H12** | Release-readiness clause / RC1 | Versioning, changelog, success bar, results note, theorem registry, README/START_HERE/master plan/paper markers updated only if H1..H11 pass and the stable runtime remains unchanged; vision_mvp ``__version__`` aligned with pyproject; **RC1 declared** if H1..H11 + S3 pass and the lab topology + W39-INFRA-1 are recorded honestly |

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
| **S1** | Stronger live multi-host evidence | Bounded live xLLM response-heterogeneity probe observes either cross-architecture response heterogeneity (different model classes produce naturally diverse one-word answers) OR honest collapse (e.g. all hosts at temperature 0 + ``num_predict=4`` produce byte-identical canonical answers); ``192.168.12.101`` is re-probed |
| **S2** | W39-INFRA-1 re-probed and bounded | The historical ``192.168.12.101`` host is re-probed; the milestone records whether it is now restorable (W39-INFRA-1 closed) or remains TCP-up + HTTP-broken (W40-INFRA-1 carries forward) |
| **S3** | Stable-vs-experimental boundary | W40 remains under ``__experimental__``; stable runtime contract unchanged |
| **S4** | Theory | Add one conditional sufficiency claim (W40-3), one limitation theorem (W40-L-COORDINATED-DIVERSE-RESPONSE-CAP), one conditional discharge (W39-C-LIVE-TRUNCATION-RECOVERY -> W40-RESPONSE-HETEROGENEITY captures budget-divergent paraphrases as one instance), and one native-latent gap claim (W40-C-NATIVE-LATENT) |
| **S5** | Paper/master-plan synthesis | Old explicit-capsule line, dense-control/geometry line, W38 disjoint-consensus line, W39 multi-host quorum line, and W40 response-heterogeneity line read as a single arc with an explicit RC1 release boundary; the paper marker is updated |
| **S6** | Full broad regression actually counted | ``pytest vision_mvp/tests -q`` completes successfully and the count is recorded in the milestone report (excluding any named pre-existing collection-time hang carried forward unchanged) |
| **S7** | Release-candidate gate (RC1) | If H1..H12 + S3 pass and the new lab topology is recorded honestly, declare a release-candidate (``RC1``) status for the SDK v3.41 line, with a final "stable vs experimental" cut-list pinned in README, and a final "open conjectures + limitation theorems" cut-list pinned in THEOREM_REGISTRY.md |

Soft gates cannot compensate for failed trust/audit hard gates.

---

## 4. Named falsifiers

- **W40-L-TRIVIAL-PASSTHROUGH**: response-heterogeneity disabled
  + ``allow_response_signature_collapse_abstain=False`` +
  manifest-v10 disabled reduces W40 to W39 byte-for-byte.
- **W40-L-INSUFFICIENT-RESPONSE-SIGNATURE**: when fewer than
  ``min_response_signature_probes`` registered member probes
  carry response signatures (or any registered probe is missing
  them), W40 cannot detect heterogeneity collapse and must
  preserve W39 behavior byte-for-byte.
- **W40-L-NO-REGRESSION-DIVERSE-AGREES**: when the K registered
  member probes carry diverse response signatures (mean pairwise
  Jaccard divergence at or above
  ``response_text_diversity_min``) AND W39 ratified, W40 must
  ratify without correctness or trust-precision regression vs
  W39.
- **W40-L-COORDINATED-DIVERSE-RESPONSE-CAP**: when the K
  registered member probes' response signatures are *injected*
  to be diverse (mean pairwise Jaccard divergence at or above
  ``response_text_diversity_min``) but all encode the same wrong
  top_set in lock-step (the W40 analog of the
  W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP attack with the
  additional adversary requirement of injecting diverse
  response bytes), W40 cannot recover at the capsule layer; this
  is the W40 analog of ``W34-L-MULTI-ANCHOR-CAP``,
  ``W37-L-MULTI-HOST-COLLUSION-CAP``,
  ``W38-L-CONSENSUS-COLLUSION-CAP``, and
  ``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` and is closed
  only by transformer-internal evidence which W40 does not
  access OR by a K+1-host disjoint topology with a new
  genuinely uncompromised pool.
- **W40-L-MUTUAL-DISJOINTNESS-VIOLATION**: a registered W40
  member-host topology whose any two registered host pools
  have non-empty intersection is rejected at registration time
  (and any envelope claiming such a topology is rejected by
  the verifier), inherited from W39's mutual-disjointness
  precondition.
- **W40-L-NATIVE-LATENT-GAP**: if a regime requires
  transformer-internal evidence not visible through trajectory
  observations, EWMA, response feature signatures, host
  attestations, anchor consensus, single-disjoint-consensus
  references, multi-host disjoint quorum references, OR
  cross-host response-heterogeneity witnesses, W40 is
  insufficient.

---

## 5. Claim boundary

W40 may claim:

- an audited cross-host response-signature heterogeneity
  ratification proxy with mutually-disjoint physical-host
  topology enforcement (inherited from W39);
- measured W40-over-W39 trust-precision gain on a regime where
  W39's K-of-N disjoint quorum is itself compromised in
  lock-step on top_set AND on response signature bytes (the
  "naive" full-quorum-collusion attack: all K colluders push
  identical wrong response bytes);
- preserved W39 behavior on the trivial path, the
  insufficient-response-signature falsifier, and the
  no-regression-diverse-agrees regime;
- explicit safety on the coordinated-diverse-response
  limitation theorem and the mutual-disjointness-violation
  falsifier;
- broader regression confidence if the full ``vision_mvp/tests``
  suite is run end-to-end during the milestone (excluding the
  named pre-existing ``test_phase50_ci_and_zero_shot``
  collection-time hang carried forward unchanged from W39);
- honest re-probing of the W39-INFRA-1 finding (``.101``
  re-probed; restored or carried forward as TCP-up + HTTP-
  broken); ``.248`` honestly recorded as still ARP-incomplete;
- a release-candidate (``RC1``) declaration **only** when
  H1..H12 + S3 pass and the lab topology is recorded honestly.

W40 may not claim:

- native latent transfer;
- transformer-internal trust subspace;
- KV-cache transplant;
- hidden-state projection;
- closure of ``W37-L-MULTI-HOST-COLLUSION-CAP`` at the capsule
  layer (only further-bounded);
- closure of ``W38-L-CONSENSUS-COLLUSION-CAP`` at the capsule
  layer (only further-bounded);
- closure of ``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` at
  the capsule layer (only further-bounded by raising the
  adversary bar from a K-of-N mutually-disjoint quorum to a
  K-of-N mutually-disjoint quorum WITH heterogeneous response
  signatures);
- closure of the new
  ``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP`` limitation
  theorem at the capsule layer;
- a true K+1-host live disjoint topology while only one
  off-cluster physical host is reachable for live inference
  AND ``192.168.12.101`` Ollama HTTP listener remains hung
  under the one-word probe budget;
- solved context for multi-agent teams;
- production release independent of the explicit RC1 gate;
  RC1 is a release-candidate not a final release.
