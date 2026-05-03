# Success criterion - CoordPy SDK v3.43 / W42
# Cross-role-invariant synthesis + manifest-v12 CID +
# role-handoff-signature axis + composite-collusion bounding

**Pre-committed before final W42 verdict / RC3 declaration**:
2026-05-03.
**Target**: SDK v3.43 / W42 family.
**Position relative to W41 RC2**: W41 RC2 wraps the W22..W40
explicit-capsule trust-adjudication chain with the W7..W11
multi-round bundle decoder family in a single auditable
manifest-v11 envelope, classifying each cell on a
producer-axis x trust-axis cross-axis decision selector.  W41
RC2 declared the SDK v3.42 line a release candidate honestly:
12/12 hard gates + S3 pass; the
``W41-L-COMPOSITE-COLLUSION-CAP`` limitation theorem is the
named open wall at the integrated layer (when both axes are
coordinated by an adversary on the same wrong top_set, W41
cannot recover at the capsule layer).

W42 is therefore framed as a **third-axis bounding** milestone
*and* a **release-closure** milestone.  The user goal is the
same as W41: *"using everything already built in this repo,
plus any honest infra/model improvements, can this repo now
actually earn release?"*  W42 attempts to answer that by
adding a third orthogonal evidence axis (cross-role-handoff
invariance) that mechanically raises the adversary bar from
"compromise the W21 producer-side admission AND the W40
trust-side ratification" (the W41-L-COMPOSITE-COLLUSION-CAP
adversary) to "compromise the W21 producer-side admission AND
the W40 trust-side ratification AND the controller-side
role-invariance policy registry that maps role-handoff
signatures to expected service sets" (the W42 adversary).

W42 also explicitly **does not** add a new transformer-internal
mechanism.  The role-handoff invariance axis is closed-form,
zero-parameter, and capsule-layer.  W42 introduces a new
proved-conditional limitation theorem,
``W42-L-FULL-COMPOSITE-COLLUSION-CAP``: when the adversary
also poisons the controller-side role-invariance policy
registry, W42 cannot recover at the capsule layer.  This is
the W42 analog of ``W41-L-COMPOSITE-COLLUSION-CAP``, sharper
in adversary cost.

**Crucial release context.** The W41 milestone retracted the
historical ``.101 = Mac with hung Ollama`` framing (``.101``
is an Apple TV / AirPlay receiver).  The honest live
multi-host topology is two reachable Macs: ``localhost`` and
``192.168.12.191``.  Both Macs serve Ollama at temperature 0
on closed-vocabulary prompts.  The W42 milestone re-confirms
this and adds a paraphrase-invariance live probe across a
controller-pre-registered set of K=4 paraphrases of one
gold-verifiable arithmetic prompt.

---

## 1. Mechanism bar

W42 must implement a real method beyond W41:

1. Wrap the W41 :class:`IntegratedSynthesisOrchestrator` with a
   role-handoff-invariance ratification layer.
2. Define a new audited envelope class
   :class:`RoleInvariantSynthesisRatificationEnvelope`
   carrying ``(role_handoff_signature_cid, invariance_branch,
   role_expected_services, role_invariance_score,
   integrated_services_pre_w42, integrated_services_post_w42,
   policy_entry_cid)`` plus four new content-addressed CIDs
   (``invariance_state_cid``, ``invariance_decision_cid``,
   ``invariance_audit_cid``, ``invariance_witness_cid``) PLUS
   the role_handoff_signature_cid PLUS the manifest-v12 CID
   PLUS the outer w42 CID.
3. Maintain a controller-side
   :class:`RoleInvariancePolicyRegistry` whose
   :class:`RoleInvariancePolicyEntry` entries map
   ``role_handoff_signature_cid -> expected_services``.  An
   honest controller pre-registers the canonical mapping for
   each known role-handoff signature; an unknown signature
   falls through to ``ROLE_INVARIANCE_NO_TRIGGER`` (preserving
   W41 byte-for-byte on the answer).
4. Compute the role-handoff signature CID
   :func:`compute_role_handoff_signature_cid` deterministically
   from the cell's input handoffs as
   ``SHA-256`` over canonical sorted ``(role, kind,
   payload_canonical)`` tuples, namespaced as
   ``w42_role_handoff_signature``.  The signature is
   independent of oracle responses and response signatures;
   compromising it requires controlling the upstream cell
   schema, not just the W21 oracle responses or the W40
   response bytes.
5. Expose two safety knobs ``invariance_enabled`` and
   ``abstain_on_invariance_diverged`` that, when enabled,
   change the W42 selector's behaviour deterministically (the
   W42-L-INVARIANCE-DIVERGED-ABSTAINED branch).
6. When the W41 inner chain is fully disabled (the W42
   ``trivial_w42`` bank), W42 must reduce to W41 byte-for-W40
   byte-for-... byte-for-substrate (the W42-L-TRIVIAL-
   PASSTHROUGH falsifier branch).
7. Define the invariance decision selector
   :func:`select_role_invariance_decision` over
   ``(integrated_services, expected_services,
   policy_match_found)`` that returns one of seven named
   branches:

   - ``trivial_invariance_passthrough``
   - ``invariance_disabled``
   - ``invariance_rejected``
   - ``invariance_no_trigger``
   - ``invariance_ratified``
   - ``invariance_diverged_abstained``
   - ``invariance_no_policy``

   The selector is deterministic and zero-parameter: the only
   inputs are the integrated services, the expected services,
   and a bool indicating policy match.
8. Seal the new state in a content-addressed **manifest-v12**
   envelope over six CIDs (parent_w41_cid,
   invariance_state_cid, invariance_decision_cid,
   invariance_audit_cid, invariance_witness_cid,
   role_handoff_signature_cid).  The invariance_audit and
   invariance_witness CIDs are namespaced as
   ``w42_invariance_audit`` and ``w42_invariance_witness`` so
   substituting a W22..W41 audit / witness for them is
   mechanically detected.
9. Keep the native-latent gap explicit: no transformer
   hidden-state, KV-cache, attention-weight, or embedding-
   table access is added.  The role-handoff invariance check
   is a closed-form deterministic mechanical comparison.
10. Token accounting must be self-consistent: the verifier
    rejects an envelope where ``n_w42_visible_tokens !=
    n_w41_visible_tokens + n_w42_overhead_tokens`` (the new
    ``w42_token_accounting_invalid`` failure mode).
11. Reproduce all R-88 banks under W42 wrapping with the new
    R-89 measurement axes (role-handoff signature
    distribution, role-invariance scores, manifest-v12 CID).

The permitted headline is **cross-role-invariant synthesis
audited proxy + manifest-v12 CID + composite-collusion
bounding**, not native latent transfer and not closure of
``W41-L-COMPOSITE-COLLUSION-CAP`` in general.

---

## 2. Hard gates

| Gate | Description | Pass iff |
| --- | --- | --- |
| **H1** | Real W42 mechanism beyond W41 | New orchestrator, registry, policy entry, envelope, verifier, role-handoff-signature CID, invariance decision selector, six new content-addressed CIDs, and manifest-v12 CID implemented; W42 makes a routing decision W41 cannot make (third-axis abstention on policy disagreement) |
| **H2** | Trust boundary | ``verify_role_invariant_synthesis_ratification`` enumerates 14 disjoint W42 failure modes; tests exercise every mode plus clean-envelope sanity; cumulative trust boundary across W22 + W29 + W30 + W31 + W32 + W33 + W34 + W35 + W36 + W37 + W38 + W39 + W40 + W41 + W42 = **196 enumerated failure modes** |
| **H3** | Trivial-W42 byte-for-W41 preservation | With invariance disabled, manifest-v12 disabled, and abstain-on-invariance-diverged disabled, W42 reduces to W41 byte-for-byte across 5 seeds (R-89-TRIVIAL-W42 ``w42_w41_byte_equivalent = True``) |
| **H4** | No correctness regression on R-89-ROLE-INVARIANT-AGREES | On R-89-ROLE-INVARIANT-AGREES (= R-88-BOTH-AXES wrapped by W42 with a clean role-invariance policy that agrees with the integrated services), W42 must not weaken W41 (delta_correctness >= 0, delta_trust_precision >= 0); branch distribution must include INVARIANCE_RATIFIED |
| **H5** | Strict trust-precision recovery on R-89-ROLE-INVARIANT-RECOVER | On R-89-ROLE-INVARIANT-RECOVER (= R-88-COMPOSITE-COLLUSION wrapped by W42 with a role-invariance policy that disagrees with the W41 wrong colluded set), W42 must abstain via INVARIANCE_DIVERGED_ABSTAINED on the recovery half; **trust_precision_w42 = 1.000 strictly improving over trust_precision_w41 = 0.500 across 5/5 seeds** (Δ_trust_precision_w42_w41 = +0.500) |
| **H6** | W42-L-FULL-COMPOSITE-COLLUSION-CAP fires on R-89-FULL-COMPOSITE-COLLUSION | On R-89-FULL-COMPOSITE-COLLUSION (= R-88-COMPOSITE-COLLUSION wrapped by W42 with a role-invariance policy that ALSO encodes the wrong colluded set), W42 must not claim recovery; delta_trust_precision_w42_w41 = 0 across 5/5 seeds; W42 reports the INVARIANCE_RATIFIED branch on the wrong set (the new W42-L-FULL-COMPOSITE-COLLUSION-CAP limitation theorem fires) |
| **H7** | Insufficient-invariance-policy falsifier | On R-89-INSUFFICIENT-INVARIANCE-POLICY (= R-88-BOTH-AXES wrapped by W42 with **no** policy entry registered for the cell's role-handoff signature), W42 routes through INVARIANCE_NO_POLICY (preserving W41 byte-for-W40 semantics on the answer); delta = 0 across 5/5 seeds |
| **H8** | Old explicit capsule line preserved | Focused W22..W41 regression slices stay green; the W22..W41 inner orchestrator chain is unchanged byte-for-byte (W42 is strictly additive) |
| **H9** | Role-handoff signature mechanical | The role-handoff-signature CID is closed-form deterministic (zero parameters); given the same canonical input handoffs, two calls produce byte-identical CIDs; permutation-invariant over handoff order via canonical sorting |
| **H10** | Live cross-host paraphrase-invariance probe | A bounded live probe at temperature 0 on the two-Mac topology (localhost gemma2:9b + 192.168.12.191 qwen2.5:14b) records K=4 paraphrases of a gold-verifiable closed-vocabulary arithmetic prompt; the paraphrase-invariance rate is recorded honestly (not load-bearing for the W42 closed-form mechanism, but a realism anchor) |
| **H11** | Broad regression confidence | Focused W22..W42 regression is green; broad spot checks on phase 11-39 + phase 40-51 + phase 6 ideas remain green; phase 50 collection-time hang carried forward unchanged from W41 |
| **H12** | Release-readiness clause / RC3 final release | Versioning, changelog, success bar, results note, theorem registry, README/START_HERE/master plan markers updated; vision_mvp ``__version__`` aligned with pyproject; **RC3 declared a final release** if H1..H11 + S3 + S7 pass and the strict +0.500 trust-precision gain on R-89-ROLE-INVARIANT-RECOVER holds across 5/5 seeds; release verdict explicit and forced |

**Hard-gate aggregate**:

- **Strong success** = 11-12 gates pass, with no trust/audit
  weakening AND the R-89-ROLE-INVARIANT-RECOVER strict +0.500
  gain holds across 5/5 seeds.
- **Partial success** = 9-10 gates pass, with exact blockers
  carried forward.
- **Failure** = <= 8 gates pass, any verifier weakening, or
  any unbounded native-latent/live claim.

---

## 3. Soft gates

| Gate | Description | Target |
| --- | --- | --- |
| **S1** | Role-handoff signature distribution measured per bank | The R-89 driver records the distinct role-handoff signature CIDs observed per bank/seed; the distribution is reproducible |
| **S2** | Lab topology re-confirmed | ``localhost`` + ``192.168.12.191`` two-Mac live topology re-confirmed at the W42 milestone; ``.248`` recorded as gone; ``.101`` recorded as Apple TV / AirPlay (W41-INFRA-1 carry-forward) |
| **S3** | Stable-vs-experimental boundary | W42 remains under ``__experimental__``; stable RunSpec -> run report runtime contract unchanged byte-for-byte |
| **S4** | Theory | Add one strict-gain claim (W42-3), one limitation theorem (W42-L-FULL-COMPOSITE-COLLUSION-CAP), one inherited trivial-passthrough falsifier (W42-L-TRIVIAL-PASSTHROUGH), one insufficient-policy falsifier (W42-L-INSUFFICIENT-INVARIANCE-POLICY), and one native-latent gap claim (W42-C-NATIVE-LATENT) |
| **S5** | Paper / master-plan synthesis | Old explicit-capsule line (W21..W34), dense-control / geometry line (W35..W36), trust-axis multi-host line (W37..W40), W41 integrated-synthesis line, and W42 cross-role-invariance line read as a single coherent arc with an explicit RC3 final-release boundary |
| **S6** | Full broad regression actually counted | Total tests counted across phase11-89; pre-existing ``test_phase50_ci_and_zero_shot`` collection-time hang carried forward unchanged from W41 |
| **S7** | Final-release gate (RC3) | If H1..H12 + S3 pass AND the R-89-ROLE-INVARIANT-RECOVER strict +0.500 gain holds AND the live two-Mac probe records honestly, declare ``v3.43`` a final release (NOT just RC3); the SDK v3.4x line closes with the W42 cross-role-invariance bounding result; open conjectures + limitation theorems cut-list pinned in THEOREM_REGISTRY.md |

Soft gates cannot compensate for failed trust/audit hard gates.

---

## 4. Named falsifiers

- **W42-L-TRIVIAL-PASSTHROUGH**: invariance disabled +
  ``manifest_v12_disabled = True`` +
  ``abstain_on_invariance_diverged = False`` reduces W42 to
  W41 byte-for-byte.
- **W42-L-INSUFFICIENT-INVARIANCE-POLICY**: when no policy
  entry is registered for the cell's role-handoff signature,
  W42 routes through INVARIANCE_NO_POLICY without
  correctness or trust-precision regression vs W41.
- **W42-L-NO-REGRESSION-INVARIANCE-AGREES**: when the policy
  entry's expected services match the W41 integrated
  services, W42 ratifies via INVARIANCE_RATIFIED without
  correctness or trust-precision regression vs W41.
- **W42-L-INVARIANCE-DIVERGED-ABSTAINED**: when the policy
  entry's expected services disagree with the W41 integrated
  services, W42 abstains via INVARIANCE_DIVERGED_ABSTAINED
  with empty integrated services (a safety branch).
- **W42-L-FULL-COMPOSITE-COLLUSION-CAP** (NEW): when the
  adversary controls the W21 producer-side admission AND
  injects diverse W40 response bytes that all encode the same
  wrong top_set AND poisons the controller-side
  role-invariance policy registry to register the wrong
  top_set as the expected services for the colluded
  role-handoff signature, W42 cannot recover at the capsule
  layer; this is the W42 analog of W41-L-COMPOSITE-COLLUSION-
  CAP, sharper in adversary cost.

---

## 5. End-state release verdict

This success criterion is the **final-release gate** for the
SDK v3.4x line.  The forced verdict structure:

- **Release v3.43 final**: H1..H12 + S3 + S7 pass; the SDK
  v3.43 line ships as a final release with the W42
  cross-role-invariance bounding layer pinned in README, the
  W42 mechanism + open conjectures + limitation theorems
  cut-list pinned in THEOREM_REGISTRY.md, and the
  cross-role-invariance live two-Mac probe recorded as the
  realism anchor.
- **Release v3.43 RC3 (with caveats)**: 9-10 hard gates pass;
  release ships with explicit caveats listing the failed
  gates as open release-blockers.
- **Do not release**: <= 8 hard gates pass, OR any trust/audit
  weakening detected.  Honest verdict: the W42 layer did not
  earn release; the SDK v3.42 RC2 remains the strongest
  release-candidate; the next milestone must materially
  improve the failed gate(s).

The verdict is binary on H5 (the strict +0.500 gain across
5/5 seeds): if H5 fails, the W42 mechanism does not bound
W41-L-COMPOSITE-COLLUSION-CAP, and the milestone must not
ship as a final release.
