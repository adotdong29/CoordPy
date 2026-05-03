# Results - Wevra SDK v3.42 / W41
# Integrated multi-agent context synthesis + manifest-v11 CID +
# cross-axis witness CID + producer-axis x trust-axis decision
# selector

Date: 2026-05-03.

W41 is a **synthesis** milestone, not "W41: one more local
mechanism."  W41 jointly binds the strongest old-line explicit-
capsule trust-adjudication chain (W21..W40) AND the strongest
cross-role / multi-round bundle decoder family (W7..W11) into a
single auditable end-to-end path with one ``manifest-v11``
envelope binding both axes plus a content-addressed cross-axis
witness.

W41 raises the capsule-layer adversary bar from "compromise the
W22..W40 trust-adjudication chain" to "compromise the W22..W40
trust-adjudication chain AND coordinate the W7..W11 producer-
side admission so the cross-axis classification cannot use one
axis to overrule the other."  When the producer-side ambiguity
AND the trust-side collusion are coordinated by an adversary,
W41 cannot recover at the capsule layer.  This is the new
proved-conditional **W41-L-COMPOSITE-COLLUSION-CAP** limitation
theorem (the W41 analog of ``W34-L-MULTI-ANCHOR-CAP``,
``W37-L-MULTI-HOST-COLLUSION-CAP``,
``W38-L-CONSENSUS-COLLUSION-CAP``,
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP``, and
``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP``).

W41 does **not** close ``W40-L-COORDINATED-DIVERSE-RESPONSE-
CAP`` and does **not** close native-latent transfer.  W41 is
the strongest honest capsule-layer integrated audited proxy
this repo can support without transformer-internal evidence.

The result is also a **measurement** advance: R-88 is the first
end-to-end multi-agent context benchmark family that records
the cross-axis branch distribution per cell, letting researchers
distinguish which axis (producer-side vs trust-side vs both vs
neither) was load-bearing on each cell of each bank.

This milestone also retracts the W37..W40 ``.101`` framing.
``192.168.12.101`` is an Apple TV / AirPlay receiver
(``AirTunes/860.7.1`` banner on port 5000;
locally-administered MAC ``36:1c:eb:dc:9a:04``), NOT a Mac
running Ollama.  The honest live multi-host topology is
``localhost`` + ``192.168.12.191`` -- two reachable Macs both
serving Ollama at ``temperature = 0`` on closed-vocabulary
prompts.  See § 4 for the full retraction.

---

## 1. New mechanism

The W41 family adds:

- ``IntegratedSynthesisRatificationEnvelope``
- ``IntegratedSynthesisRegistry``
- ``W41IntegratedSynthesisResult``
- ``IntegratedSynthesisOrchestrator``
- ``select_integrated_synthesis_decision``
- ``classify_producer_axis_branch``
- ``classify_trust_axis_branch``
- ``verify_integrated_synthesis_ratification``
- ``build_trivial_integrated_synthesis_registry``
- ``build_integrated_synthesis_registry``

W41 wraps W40.  At each cell, W41:

1. Reads W40's last-cell projection branch and answer's
   ``services``.
2. Classifies the **producer axis** branch from the inner
   answer's ``services`` set:
   - non-empty -> ``producer_axis_fired``
   - empty -> ``producer_axis_no_trigger``.
3. Classifies the **trust axis** branch from the W40 projection
   branch:
   - ``response_signature_diverse`` -> ``trust_axis_ratified``
   - ``response_signature_collapse_abstained``
     -> ``trust_axis_abstained``
   - everything else -> ``trust_axis_no_trigger``.
4. Calls
   :func:`select_integrated_synthesis_decision`, which is a
   closed-form, zero-parameter, deterministic mechanical
   classifier over ``(producer_axis_branch, trust_axis_branch,
   producer_services, trust_services)`` returning one of eight
   integrated branches:

   * ``trivial_integrated_passthrough`` -- inner W40 chain is
     fully disabled (W41 reduces to W40 byte-for-byte).
   * ``integrated_disabled`` -- W41 layer disabled; W40 result
     unchanged.
   * ``integrated_rejected`` -- the W41 envelope failed
     verification (W41 verifier hard-fails on tamper).
   * ``integrated_producer_only`` -- producer fired, trust
     no_trigger; W41 ratifies on the producer set.
   * ``integrated_trust_only`` -- (a) producer no_trigger,
     trust ratified (W41 ratifies on the trust set), OR (b)
     trust abstained (W41 routes through the trust safety
     branch with empty services -- trust over-rides producer
     because trust abstention is a safety branch).
   * ``integrated_both_axes`` -- producer fired AND trust
     ratified AND service sets agree (intersection wins on
     partial overlap).
   * ``integrated_axes_diverged_abstained`` -- producer fired
     AND trust ratified but service sets are disjoint; W41
     abstains via empty services.
   * ``integrated_neither_axis`` -- both axes no_trigger;
     answer empty.

5. Seals the decision in a manifest-v11 envelope.

Manifest-v11 binds:

1. W40 parent CID (``parent_w40_cid``).
2. Synthesis state CID
   (``synthesis_state_cid`` = SHA-256 over
   ``(producer_axis_branch, trust_axis_branch,
   integrated_branch, integrated_services, cell_index)``).
3. Synthesis decision CID
   (``synthesis_decision_cid`` = SHA-256 over
   ``(integrated_branch, integrated_services,
   n_w40_visible_tokens, n_w41_visible_tokens,
   n_w41_overhead_tokens)``).
4. Synthesis audit CID
   (``synthesis_audit_cid`` = SHA-256 over
   ``(integrated_branch, producer_axis_branch,
   trust_axis_branch, producer_services, trust_services,
   integrated_services, w40_projection_branch)``,
   namespaced as ``w41_synthesis_audit``).
5. Cross-axis witness CID
   (``cross_axis_witness_cid`` = SHA-256 over
   ``(producer_axis_branch, trust_axis_branch,
   integrated_branch, n_w40_visible_tokens,
   n_w41_visible_tokens, n_w41_overhead_tokens,
   n_structured_bits)``, namespaced as
   ``w41_cross_axis_witness``).

The synthesis audit and cross-axis witness CIDs are namespaced
so substituting a W22..W40 audit / witness for them is
mechanically detected (the verifier rejects on
``w41_synthesis_audit_cid_mismatch`` /
``w41_cross_axis_witness_cid_mismatch``).

---

## 2. Benchmark family (R-88)

Phase 88 introduces five R-88 banks, each a strict superset of
the corresponding R-87 bank with W41 wrapping:

| R-88 bank | R-87 inner bank | Purpose |
| --- | --- | --- |
| ``trivial_w41`` | ``trivial_w40`` | Byte-for-W40 preservation when synthesis is disabled, manifest-v11 is disabled, abstain-on-axes-diverged is disabled |
| ``both_axes`` | ``no_regression_diverse_agrees`` | Producer fires + trust ratifies => W41 ratifies via INTEGRATED_BOTH_AXES on the gold set; no correctness regression vs W40 |
| ``trust_only_safety`` | ``response_signature_collapse`` | Producer fires + trust abstains (response collapse) => W41 routes through INTEGRATED_TRUST_ONLY safety branch with empty services; trust precision preserved |
| ``composite_collusion`` | ``coordinated_diverse_response`` | Producer fires + trust ratifies on the wrong (colluded) set with diverse response bytes => W41 ratifies via INTEGRATED_BOTH_AXES on the wrong set; W41-L-COMPOSITE-COLLUSION-CAP fires; no recovery |
| ``insufficient_response_signature`` | ``insufficient_response_signature`` | Producer fires + trust no_trigger (insufficient probes) => W41 routes through INTEGRATED_PRODUCER_ONLY; W41 reduces to W40 byte-for-W39 on the answer |

The bench output reports substrate/FIFO, W21, W40, and W41 in
one place so the trust-adjudication chain and the integrated
synthesis layer can be compared at a glance.  R-88 also reports
the cross-axis branch distribution
(``w41_integrated_branch_hist``), the per-axis branch
distributions, and the per-cell integrated services for full
auditability.

---

## 3. Empirical results

All seed sweeps use seeds ``11, 17, 23, 29, 31``, ``n_eval=16``.

### R-88-TRIVIAL-W41

Across 5/5 seeds:

- ``correctness_w41 = correctness_w40 = 1.000``.
- ``trust_precision_w41 = trust_precision_w40 = 1.000``.
- ``delta_correctness_w41_w40 = 0.000`` (min and max equal).
- ``delta_trust_precision_w41_w40 = 0.000``.
- ``all_w41_verified_ok = True``.
- ``all_byte_equivalent_w41_w40 = True``.
- W41 overhead = 0 visible tokens.
- Aggregate branch distribution: 80 cells (5 seeds × 16 cells),
  100% on ``trivial_integrated_passthrough``.

This is the W41-L-TRIVIAL-PASSTHROUGH falsifier branch and the
H3 hard-gate anchor.

Artifact:
``vision_mvp/experiments/artifacts/phase88/trivial_w41_seed_sweep.json``.

### R-88-BOTH-AXES

Across 5/5 seeds:

- ``correctness_w41 = correctness_w40 = 1.000``.
- ``trust_precision_w41 = trust_precision_w40 = 1.000``.
- ``delta_correctness_w41_w40 = 0.000``.
- ``delta_trust_precision_w41_w40 = 0.000``.
- ``all_w41_verified_ok = True``.
- W41 overhead = 16 visible tokens (1 per cell).
- Aggregate branch distribution: 40 cells
  ``integrated_producer_only`` (prefix half: producer fires,
  trust no_trigger) + 40 cells ``integrated_both_axes``
  (recovery half: producer fires AND trust ratifies via DIVERSE
  on the gold set).
- W41 carries about **15.5k structured bits per visible W41
  token** -- in the W38 (~9.07k bits/token) to W40 (~14.5k
  bits/token) range.

This is the H4 hard-gate anchor (no correctness regression vs
W40).

Artifact:
``vision_mvp/experiments/artifacts/phase88/both_axes_seed_sweep.json``.

### R-88-TRUST-ONLY-SAFETY

Across 5/5 seeds:

- ``correctness_w41 = correctness_w40 = 0.500`` (8/16 cells
  correct -- prefix half).
- ``trust_precision_w41 = trust_precision_w40 = 1.000``.
- ``delta_correctness_w41_w40 = 0.000``.
- ``delta_trust_precision_w41_w40 = 0.000``.
- ``all_w41_verified_ok = True``.
- W41 overhead = 16 visible tokens.
- Aggregate branch distribution: 40 cells
  ``integrated_producer_only`` (prefix half: producer fires,
  trust no_trigger) + 40 cells ``integrated_trust_only``
  (recovery half: producer fires AND trust abstains via
  COLLAPSE; safety branch routes through empty services).

This is the H5 hard-gate anchor (trust precision preserved at
1.000 on the safety branch).  The W41 INTEGRATED_TRUST_ONLY
safety branch is a *new* observable: it explicitly records that
the trust-side abstention took precedence over the producer-
side ratified set, which is the safety-correct behaviour.

Artifact:
``vision_mvp/experiments/artifacts/phase88/trust_only_safety_seed_sweep.json``.

### R-88-COMPOSITE-COLLUSION (W41-L-COMPOSITE-COLLUSION-CAP)

Across 5/5 seeds:

- ``correctness_w41 = correctness_w40 = 0.500`` (prefix half
  correct, recovery half wrong).
- ``trust_precision_w41 = trust_precision_w40 = 0.500``.
- ``delta_correctness_w41_w40 = 0.000``.
- ``delta_trust_precision_w41_w40 = 0.000``.
- ``all_w41_verified_ok = True``.
- Aggregate branch distribution: 40 cells
  ``integrated_producer_only`` + 40 cells
  ``integrated_both_axes`` (W41 ratifies on the wrong colluded
  set; the producer set agrees with the trust-ratified set
  because both have been compromised by the adversary).

This is the named ``W41-L-COMPOSITE-COLLUSION-CAP`` limitation
theorem firing.  When the adversary has BOTH compromised the
W21 producer-side admission AND injected diverse W40 response
bytes that all encode the same wrong top_set, the cross-axis
classifier sees both axes agreeing on a wrong set and ratifies
INTEGRATED_BOTH_AXES.  Closure requires native-latent evidence
outside the capsule layer (``W41-L-NATIVE-LATENT-GAP``) or a
K+1-host disjoint quorum topology with at least one new
genuinely uncompromised host pool (out of capsule-layer scope).

Artifact:
``vision_mvp/experiments/artifacts/phase88/composite_collusion_seed_sweep.json``.

### R-88-INSUFFICIENT-RESPONSE-SIGNATURE

Across 5/5 seeds:

- ``correctness_w41 = correctness_w40 = 0.500``.
- ``trust_precision_w41 = trust_precision_w40 = 0.500``.
- ``delta_correctness_w41_w40 = 0.000``.
- ``delta_trust_precision_w41_w40 = 0.000``.
- ``all_w41_verified_ok = True``.
- Aggregate branch distribution: 80 cells (100%) on
  ``integrated_producer_only``: when W40 returns NO_TRIGGER on
  insufficient probes, W41 routes through PRODUCER_ONLY,
  preserving the W40 byte-for-W39 behavior on the answer.

This is the H7 hard-gate anchor (W41-L-INSUFFICIENT-RESPONSE-
SIGNATURE inherited from W40).

Artifact:
``vision_mvp/experiments/artifacts/phase88/insufficient_response_signature_seed_sweep.json``.

---

## 4. Live / multi-host evidence (W41-INFRA-1)

### 4.1 Lab topology retraction

The W37 / W38 / W39 / W40 milestones described
``192.168.12.101`` as a third Mac with a hung Ollama HTTP
listener ("TCP-up + HTTP-broken Ollama").  This framing is
**retracted** at the W41 milestone.  Re-probing ``.101`` in the
W41 milestone shows:

- ``ping -c 2 -W 2 192.168.12.101``: 0% packet loss, 4-5 ms
  RTT.
- ``nc -zv 192.168.12.101 22``: TCP SSH connect succeeds (auth
  methods ``publickey``, ``password``, ``keyboard-interactive``).
- ``nc -zv 192.168.12.101 11434``: TCP connect succeeds.
- ``curl http://192.168.12.101:11434/``: "Empty reply from
  server" (the Ollama-style empty response).
- ``curl http://192.168.12.101:5000/``: ``HTTP/1.1 403
  Forbidden`` with header ``Server: AirTunes/860.7.1``.
- ARP entry: ``? (192.168.12.101) at 36:1c:eb:dc:9a:4 on en0``
  (locally-administered MAC: the second nibble of the first
  byte is ``6`` = locally administered; this is not an Apple
  Mac OUI).

The combination of the AirTunes banner on port 5000 + the
locally-administered MAC + the lack of a Mac mDNS hostname
identifies ``.101`` as an Apple TV / AirPlay receiver.  Port
11434 returning "Empty reply from server" is the device
closing the connection on an unrecognised port, NOT a hung
Ollama listener.  No Ollama instance has ever been running on
``.101`` in this network.

The earlier "third physical host candidate" framing at the W37
/ W38 milestones was a network-layer mis-identification.  The
honest correction is recorded as **W41-INFRA-1**: ``.101`` is
an Apple TV; it is NOT a Mac with Ollama; the W37..W40
"TCP-up + HTTP-broken Ollama" framing is retracted at this
milestone.  The previous milestone's "two-reachable-host
evidence" anchors remain valid (those used ``localhost`` +
``.191`` directly via the W39-INFRA-1 fallback path, which
correctly avoided ``.101``).

``192.168.12.248`` is recorded as gone (per user instruction:
"ignore .248, its gone").

### 4.2 Honest live multi-host topology

The honest live multi-host topology going forward is the
two-Mac pair:

- ``localhost`` (Mac 1, ``Qunfengs-MBP.lan``,
  ``192.168.12.157``) -- Ollama models inventoried:
  ``gemma2:9b``, ``qwen2.5:0.5b``, ``qwen2.5-coder:7b``,
  ``deepseek-r1:7b``, ``llama3.1:8b``, ``mixtral:8x7b``
  (46.7B), ``lexi-coder``, ``llama-3-8b-lexi-uncensored``.
- ``192.168.12.191`` (Mac 2, ``HSC136047-MAC.lan``) --
  Ollama models inventoried: ``qwen2.5:14b``,
  ``qwen2.5:14b-32k``, ``qwen2.5-coder:14b``,
  ``qwen2.5-coder:14b-32k``, ``qwen3.5:35b`` (36.0B MoE).

A bounded W41 sanity probe at ``temperature = 0`` +
``num_predict = 4`` on the two-Mac topology produced byte-
agreed answers across architectures:

- ``localhost gemma2:9b``: prompt "What is 2+2? Answer with
  one word." -> "Four".
- ``192.168.12.191 qwen2.5:14b``: same prompt -> "Four".

This sharpens the empirical-suggestive
``W37-C-LIVE-TRUNCATION-RECOVERY`` /
``W40-C-LIVE-RESPONSE-HETEROGENEITY`` anchors at the closed-
vocabulary one-word-prompt regime: under ``num_predict = 4``,
two cross-architecture Macs at temperature 0 produce identical
canonical answers on a deterministic gold-verifiable prompt.

### 4.3 W41 mechanism does not require live evidence

By design, the W41 mechanism is closed-form, zero-parameter,
and capsule-layer.  The cross-axis classification is computed
deterministically from the W40 projection branch + the inner
answer's services field; no live LLM inference is required.
The R-88 bench therefore decouples the W41 scientific claim
from the live multi-host bound.  The live two-Mac probe is a
**bonus realism anchor**, not load-bearing for the W41 success
criterion.

---

## 5. Trust boundary

``verify_integrated_synthesis_ratification`` enumerates 14 W41
failure modes:

1. ``empty_w41_envelope``
2. ``w41_schema_version_unknown``
3. ``w41_schema_cid_mismatch``
4. ``w40_parent_cid_mismatch`` (W41-specific, namespaced)
5. ``w41_integrated_branch_unknown``
6. ``w41_producer_axis_branch_unknown``
7. ``w41_trust_axis_branch_unknown``
8. ``w41_synthesis_state_cid_mismatch``
9. ``w41_synthesis_decision_cid_mismatch``
10. ``w41_synthesis_audit_cid_mismatch``
11. ``w41_cross_axis_witness_cid_mismatch``
12. ``w41_token_accounting_invalid``
13. ``w41_manifest_v11_cid_mismatch``
14. ``w41_outer_cid_mismatch``

Cumulative W22 + W29 + W30 + W31 + W32 + W33 + W34 + W35 + W36
+ W37 + W38 + W39 + W40 + W41 trust boundary: **182 enumerated
failure modes** (was 168 at W40 RC1; W41 added 14 cleanly,
disjoint from W22..W40's cumulative set).

---

## 6. Theory-forward claims

### W41-1 - verifier boundary

**Claim**: The W41 integrated-synthesis envelope is
mechanically verifiable against 14 disjoint failure modes.

**Status**: proved by inspection + mechanically checked (16/16
gate checks executed; 14 named failure modes enumerated).

### W41-2 - trivial reduction

**Claim**: Disabled synthesis + disabled abstain-on-axes-
diverged + disabled manifest-v11 reduces to W40 byte-for-byte.

**Status**: empirical across 5/5 seeds on R-88-TRIVIAL-W41
(``all_byte_equivalent_w41_w40 = True``).

### W41-3 - integrated-synthesis sufficiency (load-bearing)

**Claim**: When the producer axis fires and the trust axis
ratifies and the producer service set agrees with the trust
ratified service set, W41 ratifies via INTEGRATED_BOTH_AXES on
the agreed set, with neither correctness nor trust-precision
regression vs W40.  When the producer axis fires and the trust
axis abstains, W41 routes through INTEGRATED_TRUST_ONLY with
empty services (the safety branch), preserving trust precision
at the W40 level.  When the trust axis no_trigger fires, W41
falls through to INTEGRATED_PRODUCER_ONLY, preserving the W40
behaviour byte-for-byte.

**Status**: proved-conditional + empirical on R-88-BOTH-AXES,
R-88-TRUST-ONLY-SAFETY, and R-88-INSUFFICIENT-RESPONSE-
SIGNATURE (5/5 seeds each; delta_correctness_w41_w40 = 0,
delta_trust_precision_w41_w40 = 0).

### W41-4 - cross-axis classification mechanical

**Claim**: The integrated decision selector
:func:`select_integrated_synthesis_decision` is closed-form,
zero-parameter, and deterministic; given the same per-axis
branches and service tuples, it produces byte-identical
``(branch, services)`` outputs.

**Status**: proved by inspection (the selector is a closed
match on branches; no learned state).

### W41-L-COMPOSITE-COLLUSION-CAP

**Claim**: When the adversary has BOTH compromised the W21
producer-side admission AND injected diverse W40 response bytes
that all encode the same wrong top_set, the cross-axis
classifier sees both axes agreeing on a wrong set and ratifies
``integrated_both_axes`` on the wrong set; W41 cannot recover
at the capsule layer.  This is the W41 analog of
``W34-L-MULTI-ANCHOR-CAP``,
``W37-L-MULTI-HOST-COLLUSION-CAP``,
``W38-L-CONSENSUS-COLLUSION-CAP``,
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP``, and
``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP`` and is closed only
by transformer-internal evidence outside the capsule layer
(``W41-L-NATIVE-LATENT-GAP``) or a K+1-host disjoint topology
with at least one new genuinely uncompromised host pool.

**Status**: proved-conditional limitation theorem + empirical
on R-88-COMPOSITE-COLLUSION (5/5 seeds, delta = 0).

### W41-L-INSUFFICIENT-RESPONSE-SIGNATURE (inherited)

**Claim**: When fewer than ``min_response_signature_probes``
member probes carry response signatures (or any registered
member probe is missing them), W41 routes through
INTEGRATED_PRODUCER_ONLY without correctness or trust-precision
regression vs W40.

**Status**: proved by inspection + empirical on
R-88-INSUFFICIENT-RESPONSE-SIGNATURE.

### W41-L-AXES-DIVERGED-ABSTAINED

**Claim**: When the producer axis fires AND the trust axis
ratifies but their service sets are disjoint, W41 abstains via
``integrated_axes_diverged_abstained`` with empty integrated
services (a safety branch).  This branch is a defensive default
(``abstain_on_axes_diverged = True`` in the
``IntegratedSynthesisRegistry``) that can be opted out of by
setting the flag to ``False``, in which case W41 falls back to
``integrated_producer_only`` on disjoint axes.

**Status**: proved by inspection (the selector implements
this branch directly).

### W41-C-NATIVE-LATENT

**Conjecture**: True transformer-internal trust-state
projection may separate regimes where all capsule-visible
producer-axis / trust-axis / cross-axis signals are either
absent or coordinated by an adversary.  W41 narrows the
audited proxy further along the cross-axis classification axis
but does not close this.

**Status**: conjectural and architecture-dependent.

### W41-C-MULTI-HOST

**Conjecture**: A K+1-host disjoint quorum topology with at
least one new genuinely uncompromised host pool would let the
W40 quorum size be raised beyond ``quorum_min``, which would
defeat the ``W41-L-COMPOSITE-COLLUSION-CAP`` collusion attack
at the capsule layer.  Currently bounded by the lab's two-Mac
topology (``localhost`` + ``192.168.12.191``); ``.248`` is
gone; ``.101`` is an Apple TV (not a Mac); a third genuine Mac
is not available in this environment.

**Status**: conjectural; hardware-bounded; capsule-layer
mechanism unchanged.

---

## 7. Density / efficiency

On R-88-BOTH-AXES, the W41 envelope carries about **15.5k
structured bits per visible W41 token** at one visible token
overhead/cell.  This is in the W38 (~9.07k bits/token) to W40
(~14.5k bits/token) range, slightly above W40 because the W41
envelope adds four new content-addressed CIDs (state, decision,
audit, witness = 4 × 256 bits = 1024 additional bits per
envelope).  This is controller-verified structured cross-axis
state, not hidden-state transfer.

The W41 layer is strictly additive: zero token overhead on
trivial bank, exactly 1 visible token overhead per cell on
active banks (matching the W38 / W39 / W40 pattern).

---

## 8. Regression confidence

Total tests at W41 RC2:

- **698/698** phase69-88 focused W22..W41 stack regression
  (was 661/661 at W40 RC1; W41 added 37 cleanly).
- **364/364** phase11-39 broad spot check.
- **205/205** phase40-51 + phase6 broad spot check.

Cumulative: **1267 tests pass**, excluding the named pre-
existing ``test_phase50_ci_and_zero_shot`` collection-time hang
that has been carried forward unchanged since long before the
W22..W41 work began (last touched by commit ``cc2b652``).

---

## 9. Stable vs experimental boundary

W41 is exported only under ``__experimental__``.  The stable
runtime contract (``RunSpec`` -> run report, capsule
primitives, lifecycle audit) is **unchanged**.  ``SDK_VERSION``
is ``wevra.sdk.v3.42``; ``vision_mvp.__version__`` and
``pyproject.toml`` are both ``0.5.15`` (alignment maintained).

---

## 10. RC2 verdict

H1..H12 + S1, S2, S3, S4, S5, S6, S7 of the W41 success
criterion all pass.  The SDK v3.42 line is therefore declared
**RC2** (the second release-candidate of the Wevra SDK v3.4x
line).  RC2 carries forward the RC1 stable-vs-experimental
boundary: every W22..W41 symbol is exported under
``__experimental__``; the stable RunSpec -> run report runtime
contract is byte-for-byte unchanged; the open-conjectures and
limitation-theorems cut-list is pinned in
``THEOREM_REGISTRY.md``.

The honest residual gap at RC2 is:

1. ``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP`` (carried
   forward; bounded by W41-L-COMPOSITE-COLLUSION-CAP at the
   integration layer).
2. ``W41-L-COMPOSITE-COLLUSION-CAP`` (NEW; closed only by
   native-latent evidence outside the capsule layer or a
   K+1-host disjoint topology with a new uncompromised pool).
3. ``W41-L-NATIVE-LATENT-GAP`` (architecture-dependent; out
   of capsule-layer scope).
4. Live multi-host topology bounded at two Macs (``.101`` is
   AirPlay, ``.248`` gone); ``W41-C-MULTI-HOST`` carries
   forward.

These walls are named, not papered over.  W41 is the strongest
honest capsule-layer integrated synthesis the repo can support
without transformer-internal evidence.
