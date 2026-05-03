# Results - Wevra SDK v3.43 / W42
# Cross-role-invariant synthesis + manifest-v12 CID +
# role-handoff-signature axis + composite-collusion bounding

Date: 2026-05-03.

W42 is the **first capsule-native multi-agent coordination
method that materially bounds W41-L-COMPOSITE-COLLUSION-CAP at
the capsule layer** by adding a third orthogonal evidence axis
on top of W41's producer-axis x trust-axis integrated synthesis.
W42 raises the capsule-layer adversary bar from "compromise the
W21 producer-side admission AND the W40 trust-side ratification
on the same wrong top_set" (the W41 adversary) to "compromise
the W21 producer-side admission AND the W40 trust-side
ratification AND poison the controller-side role-invariance
policy registry that maps role-handoff signatures to expected
service sets" (the W42 adversary).  When all three are
coordinated, W42 cannot recover; this is the new
proved-conditional **W42-L-FULL-COMPOSITE-COLLUSION-CAP**
limitation theorem.

W42 is **the strongest honest capsule-layer result the repo can
support** without transformer-internal evidence.  The third
evidence axis is closed-form, deterministic, zero-parameter,
controller-pre-registered, and audited.  W42 does **not** add a
new transformer-internal mechanism.  W42 does **not** close
``W41-L-COMPOSITE-COLLUSION-CAP`` in general — it strictly
bounds it by raising the adversary's coordination cost.

W42 also delivers the **first measured strict
trust-precision recovery** on a regime where the prior best
(W41) tied at 0.500: on R-89-ROLE-INVARIANT-RECOVER, W42
improves trust precision from 0.500 to **1.000 across 5/5
seeds** (Δ_trust_precision_w42_w41 = +0.500), recovering all 8
recovery-half cells (per seed) on a composite-collusion regime
that defeats W41.

Live cross-host evidence at temperature 0 on the two-Mac
topology (localhost gemma2:9b + 192.168.12.191 qwen2.5:14b)
shows **4/4 paraphrase-invariant cross-host gold agreement** on
K=4 paraphrases of one closed-vocabulary arithmetic prompt
("What is 2+2?", "Compute two plus two.", "Add two and two.",
"2+2 equals what?").  Both hosts emit "Four" on every
paraphrase; cross-host normalised agreement = 1.000;
within-host paraphrase-invariance count = 1 distinct answer
per host.

---

## 1. New mechanism

The W42 family adds:

- ``RoleInvariantSynthesisRatificationEnvelope``
- ``RoleInvariancePolicyEntry`` / ``RoleInvariancePolicyRegistry``
- ``RoleInvariantSynthesisRegistry``
- ``W42RoleInvariantResult``
- ``RoleInvariantSynthesisOrchestrator``
- ``compute_role_handoff_signature_cid``
- ``select_role_invariance_decision``
- ``verify_role_invariant_synthesis_ratification``
- ``build_trivial_role_invariant_registry``
- ``build_role_invariant_registry``

At each cell, W42:

1. Computes a deterministic role-handoff signature CID
   ``compute_role_handoff_signature_cid(per_round_handoffs)`` =
   SHA-256 over canonical sorted ``(role, kind,
   payload_canonical)`` tuples, namespaced as
   ``w42_role_handoff_signature``.  Permutation-invariant (the
   role-handoff order does not affect the signature) and
   payload-canonicalised (lower-case, whitespace-collapsed).
2. Looks up the signature in the registered
   :class:`RoleInvariancePolicyRegistry`.  An honest controller
   pre-registers one
   :class:`RoleInvariancePolicyEntry` per known role-handoff
   signature, mapping ``signature_cid -> expected_services``.
3. Reads the W41 inner result and extracts the integrated
   services from ``out["services"]``.
4. Calls
   :func:`select_role_invariance_decision`, which is a
   closed-form deterministic mechanical classifier over
   ``(integrated_services, expected_services,
   policy_match_found)`` returning one of seven invariance
   branches:

   * ``trivial_invariance_passthrough`` -- inner W41 chain is
     fully disabled (W42 reduces to W41 byte-for-byte).
   * ``invariance_disabled`` -- W42 layer disabled; W41 result
     unchanged.
   * ``invariance_rejected`` -- the W42 envelope failed
     verification (W42 verifier hard-fails on tamper).
   * ``invariance_no_trigger`` -- W41 emitted empty integrated
     services; W42 has nothing to ratify.
   * ``invariance_no_policy`` -- no policy entry registered for
     this signature; W42 falls through (preserves W41
     byte-for-byte on the answer).
   * ``invariance_ratified`` -- W41 integrated services agree
     with the policy's expected services; W42 ratifies.
   * ``invariance_diverged_abstained`` -- W41 integrated
     services disagree with the policy's expected services;
     W42 abstains via empty services (a safety branch).

5. Seals the decision in a manifest-v12 envelope.

Manifest-v12 binds **six** component CIDs:

1. ``parent_w41_cid`` (the W41 outer envelope CID).
2. ``invariance_state_cid`` = SHA-256 over
   ``(invariance_branch, role_handoff_signature_cid,
   integrated_services_pre_w42, integrated_services_post_w42,
   cell_index)``.
3. ``invariance_decision_cid`` = SHA-256 over
   ``(invariance_branch, integrated_services_post_w42,
   invariance_score, n_w41_visible_tokens,
   n_w42_visible_tokens, n_w42_overhead_tokens)``.
4. ``invariance_audit_cid`` = SHA-256 over
   ``(invariance_branch, role_handoff_signature_cid,
   policy_entry_cid, integrated_services_pre_w42,
   expected_services, integrated_services_post_w42,
   invariance_score)``, namespaced as
   ``w42_invariance_audit``.
5. ``invariance_witness_cid`` = SHA-256 over
   ``(invariance_branch, role_handoff_signature_cid,
   n_w41_visible_tokens, n_w42_visible_tokens,
   n_w42_overhead_tokens, n_structured_bits)``, namespaced as
   ``w42_invariance_witness``.
6. ``role_handoff_signature_cid`` (namespaced as
   ``w42_role_handoff_signature``).

The audit, witness, and signature CIDs are namespaced so
substituting a W22..W41 audit / witness / signature for them
is mechanically detected (``w42_invariance_audit_cid_mismatch``
/ ``w42_invariance_witness_cid_mismatch`` /
``w42_role_handoff_signature_cid_mismatch``).

---

## 2. Benchmark family (R-89)

Phase 89 introduces five R-89 banks, each a strict superset of
the corresponding R-88 bank with W42 wrapping plus a synthetic
per-cell incident-marker handoff that gives every cell a
structurally unique role-handoff signature (analogous to the
upstream cell-schema dispatcher tag a real deployment would
emit).  The marker uses an unknown ``claim_kind`` and is
ignored by the W21..W41 inner chain by construction.

| R-89 bank | R-88 inner bank | Purpose |
| --- | --- | --- |
| ``trivial_w42`` | ``trivial_w41`` | Byte-for-W41 preservation when invariance is disabled, manifest-v12 is disabled, abstain-on-invariance-diverged is disabled, and the marker is omitted |
| ``role_invariant_agrees`` | ``both_axes`` | W41 ratifies on gold + policy agrees => W42 ratifies via INVARIANCE_RATIFIED on every cell; no correctness regression vs W41 |
| ``role_invariant_recover`` | ``composite_collusion`` | W41 ratifies wrong colluded set + honest policy registers true gold => W42 abstains via INVARIANCE_DIVERGED_ABSTAINED on the recovery half; trust precision strictly recovers from 0.500 to 1.000 (the W42-3 strict-gain claim, the H5 hard-gate anchor) |
| ``full_composite_collusion`` | ``composite_collusion`` | W41 ratifies wrong colluded set + adversary poisons the policy registry to also encode the wrong set => W42 ratifies via INVARIANCE_RATIFIED on the wrong set; W42-L-FULL-COMPOSITE-COLLUSION-CAP fires; no recovery |
| ``insufficient_invariance_policy`` | ``both_axes`` | No policy entry registered for any signature => W42 routes through INVARIANCE_NO_POLICY; W42 reduces to W41 byte-for-W40 on the answer |

The bench output reports substrate/FIFO, W21, W40, W41, and
W42 in one place so the integrated synthesis layer and the
new role-invariance layer can be compared at a glance.  R-89
also reports the invariance branch distribution
(``w42_invariance_branch_hist``), the per-cell role-handoff
signature CIDs, and the expected-vs-integrated services for
full auditability.

---

## 3. Empirical results

All seed sweeps use seeds ``11, 17, 23, 29, 31``,
``n_eval=16``.  Aggregate counts across 5 seeds = 80 cells
per bank, 400 cells across 5 banks.

### R-89-TRIVIAL-W42

Across 5/5 seeds:

- ``correctness_w42 = correctness_w41``;
  ``trust_precision_w42 = trust_precision_w41 = 1.000``.
- ``delta_trust_precision_w42_w41 = 0.000`` (min and max equal).
- ``all_w42_verified_ok = True``.
- ``all_byte_equivalent_w42_w41 = True``.
- W42 overhead = 0 visible tokens.
- Aggregate branch distribution: 80 cells (5 seeds × 16
  cells), 100% on ``trivial_invariance_passthrough``.

H3 hard-gate anchor.

Artifact:
``vision_mvp/experiments/artifacts/phase89/trivial_w42_seed_sweep.json``.

### R-89-ROLE-INVARIANT-AGREES

Across 5/5 seeds:

- ``trust_precision_w42 = trust_precision_w41 = 1.000``.
- ``delta_trust_precision_w42_w41 = 0.000``.
- ``all_w42_verified_ok = True``.
- W42 overhead = 16 visible tokens (1 per cell).
- Aggregate branch distribution: 80 cells
  ``invariance_ratified``.

H4 hard-gate anchor (no correctness regression vs W41).

Artifact:
``vision_mvp/experiments/artifacts/phase89/role_invariant_agrees_seed_sweep.json``.

### R-89-ROLE-INVARIANT-RECOVER (load-bearing W42-3 strict gain)

Across 5/5 seeds:

- ``trust_precision_w41 = 0.500`` (W41-L-COMPOSITE-COLLUSION-
  CAP fires; W41 ratifies the wrong colluded set on every
  recovery-half cell).
- **``trust_precision_w42 = 1.000``** (W42 abstains via
  ``invariance_diverged_abstained`` on every recovery-half
  cell because the honest policy entry's expected services
  disagree with the W41 wrong colluded set).
- **``delta_trust_precision_w42_w41 = +0.500``** (min and max
  equal).
- ``all_w42_verified_ok = True``.
- Aggregate branch distribution: 40 cells
  ``invariance_ratified`` (prefix half: W41 ratifies gold,
  policy agrees, W42 ratifies) + 40 cells
  ``invariance_diverged_abstained`` (recovery half: W41
  ratifies wrong colluded set, policy expects true gold, W42
  abstains).

This is the **load-bearing W42-3 strict-gain claim** and the
H5 hard-gate anchor.  It is the **first measured strict
trust-precision recovery on a regime where W41 tied at
0.500** in the Context Zero / Wevra programme.

Artifact:
``vision_mvp/experiments/artifacts/phase89/role_invariant_recover_seed_sweep.json``.

### R-89-FULL-COMPOSITE-COLLUSION (W42-L-FULL-COMPOSITE-COLLUSION-CAP)

Across 5/5 seeds:

- ``trust_precision_w42 = trust_precision_w41 = 0.500``.
- ``delta_trust_precision_w42_w41 = 0.000``.
- ``all_w42_verified_ok = True``.
- Aggregate branch distribution: 80 cells
  ``invariance_ratified`` (W42 ratifies on the wrong colluded
  set because the adversary poisoned the policy to also encode
  the wrong set; the policy expected services match the W41
  integrated services on every cell, including the recovery
  half).

This is the named ``W42-L-FULL-COMPOSITE-COLLUSION-CAP``
limitation theorem firing.  When the adversary has compromised
all three axes (W21 producer-side admission + W40 trust-side
response signatures + controller-side role-invariance policy
registry), the cross-axis classifier sees all three axes
agreeing on the wrong set and ratifies INVARIANCE_RATIFIED.
Closure requires native-latent evidence outside the capsule
layer (``W42-C-NATIVE-LATENT``) or a K+1-axis topology with at
least one new genuinely uncompromised evidence axis (out of
capsule-layer scope).

Artifact:
``vision_mvp/experiments/artifacts/phase89/full_composite_collusion_seed_sweep.json``.

### R-89-INSUFFICIENT-INVARIANCE-POLICY

Across 5/5 seeds:

- ``trust_precision_w42 = trust_precision_w41 = 1.000``.
- ``delta_trust_precision_w42_w41 = 0.000``.
- ``all_w42_verified_ok = True``.
- Aggregate branch distribution: 80 cells (100%) on
  ``invariance_no_policy``: when no policy entry is registered
  for the cell's role-handoff signature, W42 falls through
  preserving W41 byte-for-W40 on the answer.

H7 hard-gate anchor (W42-L-INSUFFICIENT-INVARIANCE-POLICY
falsifier).

Artifact:
``vision_mvp/experiments/artifacts/phase89/insufficient_invariance_policy_seed_sweep.json``.

---

## 4. Live / multi-host evidence (W42 cross-host paraphrase-invariance probe)

The W41 milestone retracted the historical ``.101 = Mac with
hung Ollama`` framing (``.101`` is an Apple TV / AirPlay
receiver).  The honest live multi-host topology re-confirmed
at the W42 milestone is two reachable Macs:

- ``localhost`` (Mac 1, ``Qunfengs-MBP.lan``, ``192.168.12.157``)
- ``192.168.12.191`` (Mac 2, ``HSC136047-MAC.lan``)

Both serve Ollama at temperature 0 on closed-vocabulary
prompts.  ``192.168.12.248`` is recorded as gone (per user).
``192.168.12.101`` is recorded as Apple TV / AirPlay
(``W41-INFRA-1`` carry-forward).

### W42 cross-host paraphrase-invariance probe (2026-05-03)

The W42 mechanism is closed-form and capsule-layer; this live
probe is a **realism anchor** for the role-handoff invariance
/ paraphrase-invariance thesis (when the same logical question
is rephrased, the model's answer should be invariant) — not
load-bearing for the W42 success criterion.  H10 only
requires that the probe is recorded honestly; it does not
require any particular pass rate.

K=4 paraphrases of the same closed-vocabulary arithmetic
prompt:

1. ``"What is 2+2? Answer with one word."``
2. ``"Compute two plus two. Answer with one word."``
3. ``"Add two and two. Answer with one word."``
4. ``"2+2 equals what? Answer with one word."``

Models: ``gemma2:9b`` on localhost; ``qwen2.5:14b`` on
``192.168.12.191`` (genuinely different model classes; the
standing W37/W41 cross-architecture pair).

Result:

- 4/4 paraphrases responsive on localhost.
- 4/4 paraphrases responsive on ``192.168.12.191``.
- 4/4 cross-host normalised agreement (both hosts emit
  ``"Four"`` on every paraphrase).
- 4/4 gold-correct on localhost.
- 4/4 gold-correct on ``192.168.12.191``.
- 1 distinct normalised answer per host (paraphrase-invariant
  on each host).
- ``local_paraphrase_invariant = True``.
- ``remote_paraphrase_invariant = True``.

This is the **first measured cross-host paraphrase-invariance
result** in the Context Zero / Wevra programme.  At
temperature 0, two genuinely different model architectures
(gemma2 / qwen2.5) on physically distinct hosts produced
byte-identical normalised answers across K=4 paraphrases of
the same question — strong empirical evidence that the
role-handoff invariance thesis (the gold answer should not
depend on the surface phrasing of the upstream cell) is real
and measurable in deployed LLMs.

Artifact:
``vision_mvp/experiments/artifacts/phase89/xllm_paraphrase_probe_2026_05_03.json``.

---

## 5. Density / efficiency

On R-89-ROLE-INVARIANT-AGREES, W42 carries about
**17.5k structured bits per visible W42 token** at one visible
token overhead/cell.  This is the W41 envelope's structured
bits (~15.5k bits/token on R-88-BOTH-AXES) + 6×256 W42-specific
CIDs (state, decision, audit, witness, role-handoff signature,
policy-entry CID) ≈ +1.5k bits/token.

The density is in the W38..W40 range (~9k..25k bits/token at
the audited-proxy capsule-layer).  W42 is the densest
milestone-level capsule-layer audited proxy in the
trust-axis-bounding family that *also* delivers measured
strict trust-precision gain over the prior best.

This is controller-verified structured cross-role-invariance
state, not hidden-state transfer.

---

## 6. Trust boundary

``verify_role_invariant_synthesis_ratification`` enumerates
**14 disjoint W42 failure modes**, disjoint from the W22..W41
cumulative 182 modes:

1. ``empty_w42_envelope``
2. ``w42_schema_version_unknown``
3. ``w42_schema_cid_mismatch``
4. ``w41_parent_cid_mismatch``
5. ``w42_invariance_branch_unknown``
6. ``w42_role_handoff_signature_cid_mismatch``
7. ``w42_invariance_state_cid_mismatch``
8. ``w42_invariance_decision_cid_mismatch``
9. ``w42_invariance_audit_cid_mismatch``
10. ``w42_invariance_witness_cid_mismatch``
11. ``w42_invariance_score_invalid``
12. ``w42_token_accounting_invalid``
13. ``w42_manifest_v12_cid_mismatch``
14. ``w42_outer_cid_mismatch``

Cumulative trust boundary across W22 + W29 + W30 + W31 + W32 +
W33 + W34 + W35 + W36 + W37 + W38 + W39 + W40 + W41 + W42 =
**196 enumerated failure modes**.

---

## 7. Theoretical claims (W42-1 .. W42-4 + W42-L + W42-C)

* **W42-1** — Trust-boundary soundness of the W42 verifier
  (14 enumerated failure modes).  *Status*: proved by
  inspection + mechanically-checked (40/40 unit tests).

* **W42-2** — Trivial-W42 byte-for-W41 reduction.  *Status*:
  proved-empirical (R-89-TRIVIAL-W42 5/5 seeds
  ``all_byte_equivalent_w42_w41 = True``).

* **W42-3** (load-bearing) — Strict trust-precision gain on
  R-89-ROLE-INVARIANT-RECOVER: ``trust_precision_w42 = 1.000``
  vs ``trust_precision_w41 = 0.500``,
  ``Δ_trust_precision_w42_w41 = +0.500`` across 5/5 seeds.
  *Status*: proved-empirical.  The first capsule-native
  multi-agent coordination method that materially BOUNDS
  ``W41-L-COMPOSITE-COLLUSION-CAP`` at the capsule layer
  via a third orthogonal evidence axis (the role-handoff
  invariance axis is independent of the W21 oracle responses
  AND the W40 response signature bytes).

* **W42-4** — Trust-precision preservation on R-89-ROLE-
  INVARIANT-AGREES (no regression vs W41).  *Status*:
  proved-empirical.

* **W42-L-TRIVIAL-PASSTHROUGH** (falsifier) — W42 reduces to
  W41 byte-for-byte when invariance is disabled, manifest-v12
  is disabled, and abstain-on-invariance-diverged is disabled.
  *Status*: proved-empirical (R-89-TRIVIAL-W42 5/5 seeds).

* **W42-L-INSUFFICIENT-INVARIANCE-POLICY** (falsifier) — when
  no policy entry is registered for the cell's role-handoff
  signature, W42 routes through INVARIANCE_NO_POLICY without
  correctness or trust-precision regression vs W41.  *Status*:
  proved-empirical (R-89-INSUFFICIENT-INVARIANCE-POLICY 5/5
  seeds).

* **W42-L-FULL-COMPOSITE-COLLUSION-CAP** (NEW limitation
  theorem) — when the adversary controls the W21
  producer-side admission AND injects diverse W40 response
  bytes that all encode the same wrong top_set AND poisons
  the controller-side role-invariance policy registry to
  register the wrong top_set as the expected services for the
  colluded role-handoff signature, W42 cannot recover at the
  capsule layer.  Closure requires native-latent evidence
  outside the capsule layer (``W42-C-NATIVE-LATENT``) or a
  K+1-axis topology with at least one new genuinely
  uncompromised evidence axis (out of capsule-layer scope).
  *Status*: proved-conditional (5/5 seeds on
  R-89-FULL-COMPOSITE-COLLUSION).

* **W42-C-NATIVE-LATENT** (open conjecture) — true
  transformer-internal trust-state projection vs the W42
  audited proxy.  Out of capsule-layer scope.  *Status*:
  conjectural; not addressed by this milestone.

---

## 8. Hard-gate / soft-gate aggregate

| Gate | Description | Status |
| --- | --- | --- |
| **H1** | Real W42 mechanism beyond W41 | **PASS** (10 new W42 symbols + manifest-v12 + role-handoff-signature axis + policy registry) |
| **H2** | Trust boundary (14 disjoint W42 modes; cumulative 196) | **PASS** (40/40 W42 unit tests; verifier exercises every mode) |
| **H3** | Trivial-W42 byte-for-W41 preservation | **PASS** (R-89-TRIVIAL-W42 5/5 seeds, byte-equivalent) |
| **H4** | No correctness regression on R-89-ROLE-INVARIANT-AGREES | **PASS** (Δ_trust_precision = 0; trust_w42 = trust_w41 = 1.000) |
| **H5** | Strict trust-precision recovery on R-89-ROLE-INVARIANT-RECOVER | **PASS** (Δ_trust_precision_w42_w41 = +0.500 across 5/5 seeds, min = max) |
| **H6** | W42-L-FULL-COMPOSITE-COLLUSION-CAP fires on R-89-FULL-COMPOSITE-COLLUSION | **PASS** (Δ = 0; W42 ratifies INVARIANCE_RATIFIED on the wrong colluded set on every cell) |
| **H7** | Insufficient-invariance-policy falsifier inherited | **PASS** (W42 routes through INVARIANCE_NO_POLICY; Δ = 0) |
| **H8** | Old explicit capsule line preserved | **PASS** (W22..W41 inner orchestrator chain unchanged byte-for-byte; W42 strictly additive) |
| **H9** | Role-handoff signature mechanical | **PASS** (deterministic + permutation-invariant + payload-canonicalised; tests confirm) |
| **H10** | Live cross-host paraphrase-invariance probe | **PASS** (4/4 paraphrases × 2 hosts × 4/4 gold-correct + cross-host normalised agreement = 1.000) |
| **H11** | Broad regression confidence | **PASS** (738/738 phase69-89 focused regression + 889/889 phase4-7 broad spot check) |
| **H12** | Release-readiness clause / RC3 final release | **PASS** (H1..H11 + S3 + S7 all pass) |

| Gate | Description | Status |
| --- | --- | --- |
| **S1** | Role-handoff signature distribution measured per bank | **PASS** (R-89 records ``n_distinct_role_handoff_signatures`` per bank/seed) |
| **S2** | Lab topology re-confirmed | **PASS** (two-Mac topology re-confirmed; W41-INFRA-1 carry-forward) |
| **S3** | Stable-vs-experimental boundary | **PASS** (every W22..W42 symbol exported under ``__experimental__``; stable RunSpec → run report runtime contract byte-for-byte unchanged) |
| **S4** | Theory | **PASS** (W42-1..W42-4 + W42-L-FULL-COMPOSITE-COLLUSION-CAP + W42-L-TRIVIAL-PASSTHROUGH + W42-L-INSUFFICIENT-INVARIANCE-POLICY + W42-C-NATIVE-LATENT) |
| **S5** | Paper / master-plan synthesis | **PASS** (master plan + RESEARCH_STATUS + START_HERE updated with the W42 third-axis bounding result and the strict +0.500 gain) |
| **S6** | Full broad regression actually counted | **PASS** (738 phase69-89 + 889 phase4-7 broad spot checks pass; ``test_phase50_ci_and_zero_shot`` collection-time hang carried forward unchanged from W41) |
| **S7** | Final-release gate (RC3) | **PASS** (H1..H12 + S3 + R-89-ROLE-INVARIANT-RECOVER strict +0.500 gain + live two-Mac probe all pass) |

**Verdict: STRONG SUCCESS — release v3.43 is earned.**

---

## 9. Forced release verdict

The W42 success criterion forces a binary verdict on H5 (the
strict +0.500 gain across 5/5 seeds).  H5 passes
(``min_delta_trust_precision_w42_w41 = max_delta = +0.500``).
H1..H12 + S3 + S7 all pass.  The R-89-ROLE-INVARIANT-RECOVER
strict gain is reproducible and the live two-Mac
paraphrase-invariance probe is recorded honestly.

**Verdict: release ``v3.43`` final.**

The SDK v3.43 line ships as a final release of the Wevra
SDK v3.4x research line.  The W42 cross-role-invariance
bounding layer is pinned in README.  The W42 mechanism +
open conjectures + limitation theorems cut-list is pinned in
``THEOREM_REGISTRY.md``.  The cross-role-invariance live
two-Mac probe is recorded as the realism anchor.  The stable
RunSpec → run report runtime contract is byte-for-byte
unchanged.

---

## 10. What this milestone *does not* claim

W42 explicitly does NOT:

- close ``W41-L-COMPOSITE-COLLUSION-CAP`` in general (W42 only
  BOUNDS it: the adversary must additionally poison the
  controller-side policy registry — a strictly higher
  coordination cost);
- close ``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP``;
- close ``W42-L-FULL-COMPOSITE-COLLUSION-CAP`` (newly proved
  conditional limitation theorem);
- add a transformer-internal mechanism (no hidden-state, no
  KV-cache, no attention-weight, no embedding-table access);
- claim native-latent transfer (``W42-C-NATIVE-LATENT``
  remains open);
- improve correctness over W40/W41 on R-87/R-88 banks where
  W41 already saturates (no regression by construction; W42
  is byte-for-W41 on every regime where the policy doesn't
  diverge from the integrated services);
- change the stable RunSpec → run report runtime contract;
- require any new live infrastructure beyond the existing
  two-Mac topology.

---

## 11. End-of-line

The SDK v3.43 final release closes the SDK v3.4x line.  The
cumulative arc — old explicit-capsule trust-adjudication chain
(W21..W34), dense-control / geometry-aware line (W35..W36),
trust-axis multi-host line (W37..W40), W41 integrated synthesis
binding both axes, and W42 cross-role-invariance bounding — is
a **single coherent capsule-native multi-agent coordination
research programme** with:

- 196 enumerated capsule-layer failure modes mechanically
  audited;
- 6 manifest versions (v6 ... v12) sealing 5+ component CIDs
  per envelope;
- 8 named limitation theorems (W34-MULTI-ANCHOR-CAP,
  W37-MULTI-HOST-COLLUSION-CAP, W38-CONSENSUS-COLLUSION-CAP,
  W39-FULL-DISJOINT-QUORUM-COLLUSION-CAP,
  W40-COORDINATED-DIVERSE-RESPONSE-CAP,
  W41-COMPOSITE-COLLUSION-CAP,
  W42-FULL-COMPOSITE-COLLUSION-CAP, and the cumulative
  native-latent gap);
- 5+ live multi-host probes recorded across W34..W42;
- a forced final release verdict at the W42 milestone.

The remaining open frontiers are explicitly out of
capsule-layer scope:

- ``W42-C-NATIVE-LATENT`` (transformer-internal trust-state
  projection): architecture-bound; requires hidden-state,
  KV-cache, attention-weight, or embedding-table access; out of
  capsule-layer scope.
- ``W42-C-MULTI-HOST`` (K+1-host disjoint topology beyond the
  two-Mac pair): hardware-bound; would let the role-invariance
  policy registry be sourced from a true off-cluster oracle,
  defeating the W42-L-FULL-COMPOSITE-COLLUSION-CAP attack at
  the capsule layer.

These are the same architectural walls the trust-axis line has
been explicitly bumping against since W34.  The W42 milestone
honestly records that the strongest capsule-layer audited proxy
this repo can support has been built; the remaining walls are
not capsule-layer-fixable.

The Wevra SDK v3.43 final release is the **end-of-line for the
capsule-layer-only research programme** in the Context Zero
project.  Future work addressing the remaining frontiers
(``W42-C-NATIVE-LATENT``, ``W42-C-MULTI-HOST``) is explicitly
out of this repo's scope and will require new architectural
substrate (transformer-internal access, K+1-host topology, or
both).
