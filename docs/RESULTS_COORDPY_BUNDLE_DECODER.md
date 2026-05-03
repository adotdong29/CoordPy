# SDK v3.11 — bundle-aware team decoder + W10 family

> Theory-forward results note for the SDK v3.11 milestone. The
> milestone introduces the **first capsule-native multi-agent
> coordination method that crosses the admission/decoding split**:
> a bundle-aware team decoder paired with the SDK v3.10 W9 admission
> policy, anchored on the new Phase-57 *decoder-forcing* benchmark.
>
> **Bottom line.** SDK v3.10 W9 is the structural ceiling of any
> service-blind admission policy in the programme — we name this
> ceiling W10-Λ. SDK v3.11 establishes that ceiling, then crosses
> it with a small, deterministic, training-free decoder modification
> (W10-1). The win is conditional on the bench property (gold root
> cause is in a specific causal tier) and is sharp at its named
> falsifier (W10-4). Last touched: SDK v3.11, 2026-04-26.

## TL;DR — what shipped

| Question                                           | Answer                                                           |
| -------------------------------------------------- | ---------------------------------------------------------------- |
| Where did SDK v3.10 W9 hit a structural wall?      | The W9-4 falsifier: when a decoy is corroborated by ≥ 2 distinct producer roles via *any* claim_kinds, no service-blind admission policy can exclude it. |
| What is the new SDK v3.11 anchor regime?           | **R-57 / Phase-57** — multi-service-gold + corroborated-decoy via *non-causal* claim_kinds. The smallest deterministic regime where (a) admission alone is provably insufficient (W10-Λ) and (b) a decoder-side projection rule fully closes the gap (W10-1). |
| What is the new method?                            | The **joint admit-and-decode method** (W10-AD): SDK v3.10 W9 admission with `top_k = |gold|+1`, paired with the new `BundleAwareTeamDecoder` (`cck_filter=True`, `role_corroboration_floor=1`, `fallback_admitted_size_threshold=2`). |
| Headline result on R-57 default                    | `bundle_decoder` `accuracy_full = 1.000`; every service-blind admission policy in the SDK (FIFO, priority, coverage, W7-2, W8, W9) ties FIFO at `0.000`. Headline gap = **+1.000** vs every admission baseline, stable across **5/5** alternate bank seeds. |
| Backward-compat (R-53 / R-54 / R-55 / R-56)        | Bundle decoder ties best-prior on every cell of every prior regime (W10-3 — proved-empirical). The trust-admission fallback (`fallback_admitted_size_threshold=2`) is the load-bearing piece. |
| Falsifier                                          | W10-4: when the decoy is *also* mentioned via a CCK-eligible claim_kind, the bundle decoder cannot exclude it; ties FIFO at 0.000. |
| Audit                                              | T-1..T-7 OK on every cell of every regime (W10-aud). |
| Did we solve multi-agent context?                  | **No.** We sharpened the conditional. The bundle decoder closes the W10-Λ gap *only* when the gold root_cause is in a specific causal tier (data / storage / compute / edge / network). Generic-tier root_causes (`error_spike`, `latency_spike`) collapse the CCK to all-noise; the decoder cannot help there. |

## 1. The structural wall (W10-Λ)

### 1.1 Statement

**Theorem W10-Λ (admission-only structural limit, proved-empirical
+ structural sketch).** On any candidate stream where:

* `gold_services` has size `n ≥ 2`;
* every gold service is mentioned by `≥ min_corroborated_roles`
  distinct producer roles;
* at least one decoy service is *also* mentioned by
  `≥ min_corroborated_roles` distinct producer roles;
* the (role, tag) bipartite multiset for the decoy is
  observationally indistinguishable from at least one gold service's
  bipartite multiset under any service-blind aggregation;

then every service-blind admission policy in the SDK
(`FifoAdmissionPolicy`, `ClaimPriorityAdmissionPolicy`,
`CoverageGuidedAdmissionPolicy`, `CohortCoherenceAdmissionPolicy`
(W7-2), `CrossRoleCorroborationAdmissionPolicy` (W8),
`MultiServiceCorroborationAdmissionPolicy` (W9)) achieves
`accuracy_full = 0.000` under the decoder's set-equality
`services_correct` check.

### 1.2 Why

The state of every shipped admission policy is a function of the
bipartite multiset `{(source_role, service=<tag>) : candidate ∈
auditor_stream}`. The corroborated decoy is, by construction,
indistinguishable from one of the gold services on this multiset.
Therefore no service-blind admission policy can prefer one over the
other.

### 1.3 Empirical anchor

* `docs/data/phase57_decoder_K8_n10.json` — 6 admission strategies
  (FIFO, priority, coverage, W7-2, W8, W9) all score
  `accuracy_full = 0.000` and `accuracy_services = 0.000` on
  Phase-57 default; `accuracy_root_cause` ≥ 0.700 (admission can
  pick the right root_cause but not the right service set).
* `docs/data/phase57_seed_sweep.json` — 5/5 bank seeds reproduce
  the W10-Λ witness.

## 2. The decoder closes the gap (W10-1)

### 2.1 Mechanism

The `BundleAwareTeamDecoder` runs three steps:

1. **Pick root_cause** by the same priority decoder as
   `vision_mvp.tasks.incident_triage._decoder_from_handoffs`
   (`_decoded_root_cause` in `team_coord.py`).
2. **Look up the causal claim-kind set** `CCK[root_cause]` from the
   closed-vocabulary table `CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE` —
   data-tier root_causes get the data-tier causal kinds
   (`POOL_EXHAUSTION`, `DEADLOCK_SUSPECTED`, `SLOW_QUERY_OBSERVED`),
   storage-tier root_causes get storage-tier causal kinds, etc.
3. **Filter services** to those mentioned via at least one
   CCK-eligible claim_kind (with two pragmatic knobs:
   `role_corroboration_floor` for stricter contradiction-aware
   filtering and `fallback_admitted_size_threshold` for backward-
   compat trust-admission fallback).

### 2.2 Statement

**Theorem W10-1 (bundle-aware decoder sufficiency, proved-empirical
n=50 saturated).** Pairing the SDK v3.10 W9 admission policy
(`top_k = |gold|+1 = 3`, `min_corroborated_roles = 2`) with
`BundleAwareTeamDecoder(cck_filter=True,
role_corroboration_floor=1, fallback_admitted_size_threshold=2)`
achieves `accuracy_full = 1.000` on Phase-57 default
(K_auditor=8, T_auditor=256, n_eval=10, bank_seed=11), stable
across 5/5 alternate bank seeds (11, 17, 23, 29, 31).

### 2.3 Empirical anchor

* `docs/data/phase57_decoder_K8_n10.json` — `capsule_bundle_decoder`
  `accuracy_full = 1.000` against every other strategy at 0.000.
* `docs/data/phase57_seed_sweep.json` — gap = +1.000 on each of
  5/5 alternate bank seeds.

### 2.4 Why this *is* a decoder-side move (and not just admission)

The `MultiServiceCorroborationAdmissionPolicy` with `top_k=3`
admits the *full* dominant set — both gold services AND the
corroborated decoy. Service-blind admission cannot do better
under W10-Λ. The bundle decoder is what discriminates: it reads
off the chosen root_cause from admitted handoffs, looks up CCK,
and projects the admitted services through the CCK predicate.
The admission policy alone admits {api, db, cache} on a
pool_exhaustion incident; the decoder picks {api, db}.

## 3. Backward-compat (W10-3)

### 3.1 Statement

**Theorem W10-3 (backward-compat with W7-2 / W8 / W9, proved-
empirical).** On the default banks of Phase 54, Phase 55, and
Phase 56, the bundle-aware decoder paired with the SDK v3.10 W9
admission (or with the corresponding SDK v3.8/v3.9 admission)
produces the same `services` set and therefore the same
`accuracy_full` as the prior best capsule strategy.

### 3.2 Why the trust-admission fallback is load-bearing

Without `fallback_admitted_size_threshold ≥ 2`, the bundle decoder
filters out gold services on Phase 56 `slow_query_web_db`
scenarios (gold *web* is mentioned only via non-CCK kinds
LATENCY_SPIKE on monitor + FW_BLOCK_SURGE on network). The
fallback rule "if admission already produced a set of size
≤ threshold, trust it" preserves R-56 backward-compat exactly.

### 3.3 Empirical anchor

* `docs/data/phase57_cross_regime.json` (with the bundle-decoder
  evaluated on each regime via the cross-regime hook) — 1.000 on
  R-54 / R-55 / R-56 default.
* In-line backward-compat probe (run from this repo):

```text
bundle on Phase 54: 1.0
bundle on Phase 55: 1.0
bundle on Phase 56: 1.0
```

## 4. The falsifier (W10-4)

### 4.1 Statement

**Theorem W10-4 (decoy-CCK-promotion falsifier, proved-empirical
n=10 saturated).** When the decoy is mentioned via at least one
claim_kind in `CCK[gold_root_cause]` (i.e. the decoy is *also*
in the same causal tier as the gold root_cause), the bundle
decoder cannot exclude it. The W10-1 win does not hold; bundle
decoder ties FIFO at `accuracy_full = 0.000` on the
Phase-57-falsifier bank.

### 4.2 Empirical anchor

* `docs/data/phase57_falsifier_K8_n10.json` — every strategy
  including `capsule_bundle_decoder` scores `accuracy_full = 0.000`.
* Bench property: `scenarios_with_decoy_cck_promotion = 10/10`,
  `scenarios_with_decoder_forcing_property = 0/10`.

## 5. Honest scope (what the milestone does NOT claim)

* **Not** "we solved multi-agent context." SDK v3.11 sharpens the
  conditional from "admission alone is sufficient under decoy-
  storm-non-corroboration" (W9 reading) to "admission + decoder is
  sufficient under specific-tier root_cause + decoy-mentioned-only-
  via-non-causal-kinds". Both are conditional. The W10-4 falsifier
  names the next wall.
* **Not** "the bundle decoder is a strict improvement over W9 on
  every regime." On R-56 the bundle decoder *ties* W9 at 1.000 (via
  the trust-admission fallback). On R-57 the bundle decoder
  *strictly improves* over W9 by +1.000. On R-58 (hypothetical:
  decoy CCK-promoted), neither helps.
* **Not** "the CCK table is universal." It is a closed-vocabulary
  table tailored to the incident-triage benchmark family. Other
  benchmark families would require their own tier table. The
  *programme-level* claim is that "decoder-side
  compatibility/contradiction reasoning is the natural next axis
  beyond admission" — the *implementation-level* claim is the CCK
  table for incident-triage.
* **Not** "real production multi-agent teams have specific-tier
  root_causes." Many do not. When the gold root_cause is generic
  noise (`error_spike`, `latency_spike`), CCK collapses to all-
  noise and the bundle decoder is a no-op. This is the *named
  scope limit* of W10-1.

## 6. Scientific consequences

### 6.1 The strongest current CoordPy "multi-agent context" thesis

**Materially stronger than the SDK v3.10 thesis.** Before this
milestone, the strongest claim was: *"On R-56 (multi-service-gold +
single-role-decoy), W9 strictly beats every prior admission
policy."* That left admission as the only capsule-native lever the
SDK exposes. After this milestone, the strongest claim is:

> *"Capsule-native multi-agent coordination requires both an
> admission policy that respects cross-role corroboration AND a
> decoder that filters admitted services by the causal claim-kind
> set of the chosen root_cause. Each axis is necessary for some
> regime; together they cross both the W7-2 single-tag-plurality
> wall (W8) and the W9-4 corroborated-decoy wall (W10-1). The
> joint method has a sharp falsifier (W10-4) and is conditional
> on the gold root_cause's tier (W10-1's honest scope)."*

This is the **first capsule-native multi-agent claim that names
both halves of the admit-and-decode split**.

### 6.2 Where admission ends and decoding begins

W10-Λ is the formal answer: admission ends precisely where the
(role, tag) bipartite multiset stops being a *separator* between
gold and decoy. The (role, kind, tag) tripartite multiset *is*
a separator under the CCK predicate when the gold root_cause is
in a specific causal tier; this is the regime where decoding can
help.

### 6.3 What this means for the original "context for multi-agent
teams" goal

The original Context Zero thesis is *per-agent minimum-sufficient
context for multi-agent teams*. SDK v3.11 makes the structural
shape of "minimum-sufficient" sharper by adding decoding to the
levers a capsule-native runtime exposes — the per-agent admitted
set is not the *only* place context is shaped. The team decoder
projects each role's view through a closed-vocabulary
compatibility predicate; the projection is part of the contract.
This is one notch closer to a *fully* satisfying answer to "what
context does each agent need", but it is still conditional on
named bench properties — see § 5.

## 7. Pre-committed success-criterion check

Per `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1 (R-57
anchor for SDK v3.11):

| Requirement                                           | Met?     | Evidence                                                                            |
| ----------------------------------------------------- | -------- | ----------------------------------------------------------------------------------- |
| 1. Code anchor in `team_coord.py` + SDK re-export     | ✅       | `BundleAwareTeamDecoder`, `decode_admitted_role_view`, `CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE` exported from `vision_mvp.coordpy`. |
| 2. Strict gain ≥ 0.20 on R-57 vs FIFO + W9            | ✅       | `bundle_decoder − fifo = +1.000`; `bundle_decoder − multi_service = +1.000`. |
| 3. Stability ≥ 3 bank seeds                           | ✅       | 5/5 seeds (11, 17, 23, 29, 31); `phase57_seed_sweep.json`. |
| 4. No regression > 0.05 on R-53 / R-54 / R-55 / R-56  | ✅       | Bundle decoder = 1.000 on each prior regime via the trust-admission fallback. R-53 admission strategies unchanged in the SDK (additive new strategy). |
| 5. Audit T-1..T-7 preserved on every cell             | ✅       | `audit_ok_grid` = True on every capsule strategy on every regime. |
| 6. Named bench property + named falsifier             | ✅       | Bench property: `decoder_forcing_property_holds` = 10/10. Falsifier: `decoy_cck_promoted` = 10/10 on falsifier bank → all strategies 0.000. |
| 7. Admission/decoding split (SDK v3.11 only)          | ✅       | W10-Λ: `max_admission_only_accuracy_full = 0.000` on R-57. New method modifies decoder. |

**Verdict:** SDK v3.11 clears the **strong success bar** § 1.1
(R-57 anchor, all 7 conditions met).

## 8. Theorem family W10 (minted by this milestone)

| Claim   | Description                                                     | Status                            | Anchor                                                                                       |
| ------- | --------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------- |
| W10-Λ   | Service-blind admission limit on R-57                           | proved-empirical + structural     | `phase57_decoder_K8_n10.json`; `team_coord.MultiServiceCorroborationAdmissionPolicy` state shape. |
| W10-1   | Bundle-aware decoder sufficiency on R-57                        | proved-empirical (n=50 saturated) | `phase57_decoder_K8_n10.json`; `phase57_seed_sweep.json`.                                    |
| W10-2   | CCK structural correctness                                      | proved (by inspection)            | `team_coord.CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE`; `team_coord.BundleAwareTeamDecoder.decode`.  |
| W10-3   | Backward-compat with W7-2 / W8 / W9 on R-54 / R-55 / R-56       | proved-empirical (n=10 each)      | Backward-compat probe in `RESULTS_COORDPY_BUNDLE_DECODER.md` § 3.                             |
| W10-4   | Decoy-CCK-promotion falsifier ties FIFO at 0.000                | proved-empirical (n=10 saturated) | `phase57_falsifier_K8_n10.json`.                                                             |
| W10-AD  | Joint admit-and-decode method definition (W9 + bundle)          | code-anchored                     | `phase57_decoder_forcing.run_phase57` `capsule_bundle_decoder` strategy.                     |
| W10-aud | Team-lifecycle audit T-1..T-7 holds for all cells of R-57       | proved + mechanically-checked     | `phase57_decoder_K8_n10.json::audit_ok_grid`.                                                |
| W10-C1  | CCK table extends to non-incident-triage tier maps              | conjectural                       | Falsifier: a 2nd benchmark family where CCK fails to filter.                                 |
| W10-C2  | Real-LLM transfer of W10-1                                      | conjectural                       | Phase-58 candidate (real-LLM extractor).                                                     |
| W10-C3  | Multi-round bundle decoder closes W10-4                         | conjectural                       | Multi-round capsule chain not yet shipped.                                                   |

## 9. Cross-references

* Pre-committed success criterion:
  `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1
  (R-57 anchor) and § 2.6 (bench-ingredients of R-57).
* Theorem registry: `docs/THEOREM_REGISTRY.md` § "Bundle-aware
  team decoder (W10-Λ .. W10-4) — SDK v3.11".
* Code: `vision_mvp/coordpy/team_coord.py`
  (`BundleAwareTeamDecoder`, `CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE`,
  `_decoded_root_cause`); `vision_mvp/coordpy/__init__.py`
  (re-exports).
* Benchmark: `vision_mvp/experiments/phase57_decoder_forcing.py`.
* Tests: `vision_mvp/tests/test_coordpy_bundle_decoder.py`.
* Result data: `docs/data/phase57_*.json`.
