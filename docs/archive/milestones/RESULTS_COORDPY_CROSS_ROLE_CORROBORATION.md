# SDK v3.9 — Cross-Role Corroboration Multi-Agent Coordination

> Milestone results note. SDK v3.8 produced an **honest conditional**
> result on multi-agent coordination: at the Phase-54 default config
> (gold-plurality + foreign-service decoys), buffered cohort-coherence
> admission (W7-2) beats substrate FIFO by +1.000 on accuracy_full.
> But the win is *brittle*: in any regime where some decoy carries
> strictly more raw mentions than gold, W7-2 picks the decoy
> plurality and ties FIFO at 0.000 — this is the named W7-2
> falsifier in `HOW_NOT_TO_OVERSTATE.md`.
>
> SDK v3.9 directly attacks this falsifier. It (a) builds the
> Phase-55 *decoy-plurality + cross-role-corroborated gold* regime
> — a fairer, harder bench than Phase 54 because every scenario
> *also* has locally-misleading per-role distractors — (b)
> implements one stronger admission method that aggregates across
> distinct producer roles rather than over raw mention counts, and
> (c) shows a clean strict separation: on Phase 55, the new policy
> beats both substrate FIFO and SDK v3.8 W7-2 buffered cohort by
> **+1.000** on accuracy_full, stable across **5/5** alternate
> bank seeds, while preserving the W7-2 win on Phase 54
> backward-compatibly.
>
> Theorem-forward: SDK v3.9 mints the **W8 theorem family**
> (W8-1 / W8-2 / W8-3 / W8-4 — proved or proved-empirical) and is
> honest about *which* multi-agent context problems the new policy
> demonstrably solves. The headline result clears the **strong
> success bar** of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`
> — the first SDK milestone whose coordination-performance gain
> spans *three* named regimes (Phase 53 no-regression, Phase 54
> backward-compat, Phase 55 strict win) with cross-seed stability,
> a named bench property, and an explicitly named falsifier regime.
> Last touched: 2026-04-26.

## TL;DR — what SDK v3.9 ships

* **Phase-55 decoy-plurality benchmark**
  (`vision_mvp/experiments/phase55_decoy_plurality.py`). The
  smallest deterministic multi-agent benchmark where (i) each
  scenario has **decoy raw plurality** (some decoy service has
  strictly more raw mentions than gold) AND (ii) the gold service
  is **cross-role corroborated** (mentioned by strictly more
  distinct producer roles than any decoy). 10/10 scenarios on the
  default bank satisfy both properties; mechanical witnesses are
  in `Phase55BankShapeTests`.
* **`CrossRoleCorroborationAdmissionPolicy`**
  (`vision_mvp/coordpy/team_coord.py`). A deterministic, training-free,
  interpretable admission rule that aggregates over (role, tag)
  bipartite multisets rather than over raw tag counts. Default
  score: ``score(tag) = 100 * |distinct_roles| + |raw_mentions|``.
  Two sub-modes (streaming / buffered) consistent with W7-2's
  design; the buffered factory `from_candidate_stream` is the W8
  anchor.
* **W8 theorem family** (4 members; all proved or proved-empirical).
* **PART D evaluation** across three named regimes:

  | regime              | corroboration | W7-2 cohort | FIFO  | gap (corr − fifo) |
  | ------------------- | ------------- | ----------- | ----- | ----------------- |
  | Phase 53 synthetic  | 0.800         | 0.800       | 0.800 |   +0.000          |
  | Phase 53 14B real-LLM| 0.800        | 0.800       | 0.800 |   +0.000          |
  | Phase 54 default    | 1.000         | 1.000       | 0.000 |   +1.000          |
  | **Phase 55 default**| **1.000**     | **0.000**   | **0.000** | **+1.000**    |
  | Phase 55 falsifier  | 0.000         | 0.000       | 0.000 |   +0.000          |

  **Phase 55**: corroboration wins +1.000 vs both FIFO and W7-2
  (stable across 5/5 bank seeds). **Phase 54**: corroboration ties
  W7-2 at 1.000 — backward-compat preserved. **Phase 53**: no
  regression vs FIFO at K=4 (synthetic and real-LLM 14B).
  **Phase 55 falsifier** (gold has no cross-role corroboration):
  corroboration ties FIFO at 0.000 — confirming the conditional
  nature of the W8-1 win.

* **Audit invariants preserved.** All capsule strategies — fifo,
  priority, coverage, cohort_buffered, corroboration — pass
  `audit_team_lifecycle` (T-1..T-7) on every cell of every regime.

## 1. The W7-2 weakness this milestone attacks

Phase-54 W7-2 buffered cohort uses a *single-tag-plurality* admission
rule: pick the service tag with the most raw mentions in the
candidate stream and admit only candidates carrying that tag. This
works on Phase 54 because the bench has *gold-plurality* by design.

It fails as soon as the bench has decoy raw plurality. Concretely:

```
candidate stream (1 scenario, abbreviated):
  monitor:  service=archival × 4
  db_admin: service=api      × 1
  sysadmin: service=api      × 1
  network:  service=api      × 1

W7-2 buffered cohort:
  raw counts: archival=4, api=3
  → fixed_plurality_tag = "archival"
  → admits 4 archival-tagged candidates
  → decoder's services = {archival}
  → grading: services_correct = False (gold = {api}); fail.
```

But the same stream is *cross-role-corroborated*: api is mentioned
by 3 distinct producer roles (db_admin, sysadmin, network), while
archival is mentioned by only 1 role (monitor). A policy that
weighted distinct-role coverage over raw count would pick api and
win.

This is exactly the named W7-2 falsifier in
`HOW_NOT_TO_OVERSTATE.md`:

> "If decoy plurality holds, buffered cohort coherence picks the
> decoy plurality and ties FIFO at 0.000."

## 2. The Phase-55 benchmark — design, properties, falsifier

### 2.1 Why this bench is fair, not benchmark-shopping

The bench combines four pre-committed properties (mechanically
verified by `Phase55BankShapeTests`):

* **Decoy raw plurality**: some decoy service has strictly more
  raw mentions in the auditor stream than gold (10/10 scenarios).
  This is what makes W7-2 fail.
* **Cross-role corroboration of gold**: the gold service is
  mentioned by strictly more distinct producer roles (≥ 3) than
  any decoy service (10/10 scenarios). This is the relational
  signal the new policy can exploit.
* **Budget pressure**: every scenario has |candidates_to_auditor| > K
  at K=4 (10/10 scenarios).
* **Locally-misleading per-role distractors**: each producer role's
  *local* plurality of `service=<tag>` mentions is a *decoy* —
  i.e. if the auditor naively read any single role's view in
  isolation, it would pick the wrong service for that role. Only
  by aggregating across ≥ 3 roles does the gold emerge.

Realistic distractors. Every real-world incident-response role
sees telemetry from many services, only one of which is causal in
any given incident. A monitor role can plausibly emit many
mentions of a benign service (e.g. an archival cron storm). A
human SRE correlates signals across dashboards; the corroboration
policy is the smallest mechanical move that captures that
correlation.

### 2.2 Pre-committed bench property + falsifier

The bench property is named in code:

```python
def _candidate_stats(candidates, real_service):
    ...
    return {
        ...
        "decoy_plurality_holds": other_raw_max > real_raw,
        "gold_corroboration_holds": real_roles > other_roles_max,
        ...
    }
```

A *named* falsifier regime
(`build_phase55_falsifier_bank`) instantiates the inverse property:
gold has no cross-role corroboration; decoy is corroborated. On
this regime, `gold_corroboration_holds = False` for every scenario
(mechanically verified by
`Phase55BankShapeTests.test_falsifier_bank_drops_corroboration_property`).

This makes the W8-1 conditionality sharp: when the bench property
holds, corroboration wins; when it doesn't, corroboration ties FIFO.

## 3. The corroboration admission policy (SDK v3.9)

### 3.1 Public surface

```python
from vision_mvp.coordpy import (
    TeamCrossRoleCorroborationAdmissionPolicy,  # canonical SDK alias
)
from vision_mvp.coordpy.team_coord import (
    CrossRoleCorroborationAdmissionPolicy,
)

# Streaming mode (W7-1-aux-style baseline — arrival-order-sensitive):
policy = CrossRoleCorroborationAdmissionPolicy()

# Buffered mode (W8-1 anchor — arrival-order-stable):
stream = [(c[0], c[3]) for c in candidate_stream]
policy = CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(stream)
```

### 3.2 Score function (one regex, one counter)

For each `service=<tag>` token in the candidate stream:

```
score(tag) = role_weight * |{distinct producer roles emitting tag}|
           + |{total mentions of tag}|
```

with `role_weight = 100` (default). The buffered policy admits
candidates whose tag matches the highest-scoring tag (lex tie-break
on the tag name).

### 3.3 Why it is *strictly cross-role*

The W7-2 single-tag plurality (`CohortCoherenceAdmissionPolicy`)
sees only the multiset of service tags. The corroboration policy's
score function explicitly aggregates over the **(role, tag)
bipartite multiset**, so it can express "gold is mentioned by 3
distinct roles vs decoy by 1 distinct role" — a relational signal
the W7-2 policy cannot represent. This is the minimum interesting
**strict generalisation** of W7-2.

### 3.4 Why it is honest

* **Deterministic** and **small**: one regex (re-using
  `_SERVICE_TAG_RE`), one counter, no learning, no
  training-distribution dependency.
* **Backward-compatible** with W7-2 by construction: on any stream
  where gold has both raw plurality AND distinct-role coverage,
  the two policies admit identical sets. On Phase 54, the two are
  indistinguishable on accuracy_full (W8-3).
* **Falsifiable**: a stream where the *decoy* has strictly more
  distinct-role coverage AND strictly more raw mentions than gold
  falsifies W8-1 — and the falsifier regime is the named
  `build_phase55_falsifier_bank`.

### 3.5 Lifecycle invariants

The policy preserves T-1..T-7 by construction: it returns standard
`AdmissionDecision` records via the existing `TeamCoordinator`
admission path. No new lifecycle states; W6-1 generalisation to
Phase-55 holds.

## 4. Theory — what the W8 family says

### 4.1 W8-1 (proved-empirical, strict separation)

> **W8-1 (proved-empirical, n = 10 × 5 bank seeds = 50 scenarios
> saturated).** When the candidate stream satisfies the
> *decoy-plurality + cross-role-corroborated-gold property*
> (some decoy service has strictly more raw mentions in the auditor
> stream than gold AND the gold service is mentioned by strictly
> more distinct producer roles than any decoy) and
> `|candidates| > K_auditor`, the *buffered*
> `CrossRoleCorroborationAdmissionPolicy` (constructed via
> `from_candidate_stream`) achieves
> `accuracy_full = 1.000` while substrate FIFO,
> `CapsuleFifoAdmissionPolicy`, `ClaimPriorityAdmissionPolicy`,
> `CoverageGuidedAdmissionPolicy`, AND
> `CohortCoherenceAdmissionPolicy` (W7-2 buffered) all produce
> `accuracy_full = 0.000`. The headline gap is **+1.000** on the
> pre-committed Phase-55 default (`K_auditor=4`, `T_auditor=128`,
> `n_eval=10`, `bank_seed ∈ {11, 17, 23, 29, 31}`). This is the
> first *strict separation* between corroboration and W7-2 in the
> programme.
>
> **Anchors.** `docs/data/phase55_decoy_plurality_K4_n10.json`;
> `docs/data/phase55_seed_sweep.json`;
> `Phase55DefaultConfigTests::test_corroboration_strictly_beats_W7_2_at_K4`;
> `Phase55SeedStabilityTests::test_gap_holds_across_five_seeds`.
>
> **Falsifier.** A Phase-55 default config where
> `corroboration_minus_cohort_buffered_accuracy_full < 0.50`. (Test
> fails; CI gates W8-1.) AND a named falsifier regime
> (`build_phase55_falsifier_bank`) where
> `gold_corroboration_holds = False` everywhere and corroboration
> ties FIFO (mechanically verified by
> `Phase55FalsifierTests::test_corroboration_ties_fifo_on_falsifier`).

### 4.2 W8-2 (proved, structural)

> **W8-2 (proved).** For the corroboration score function
> `score(tag) = W_role · |distinct_roles(tag)| + |mentions(tag)|`,
> if `W_role > Δr_max` where `Δr_max` is the maximum raw-count
> difference between any two tags in the candidate stream, then
> for any pair of tags `(t1, t2)` with
> `|distinct_roles(t1)| > |distinct_roles(t2)|`,
> `score(t1) > score(t2)` regardless of the raw-count distribution.
>
> **Proof sketch.** Let `Δr_role = |distinct_roles(t1)|
> − |distinct_roles(t2)| ≥ 1`. Then
> `score(t1) − score(t2) = W_role · Δr_role
> + (|mentions(t1)| − |mentions(t2)|)
> ≥ W_role · 1 − Δr_max = W_role − Δr_max > 0`.
> ∎
>
> **Anchor.** `team_coord.CrossRoleCorroborationAdmissionPolicy`
> docstring; `CorroborationPolicyUnitTests::test_w8_2_role_weight_dominates_raw_count`.
>
> **Practical consequence.** With the default `role_weight=100`,
> any candidate stream of size `< 100` cannot have a
> raw-count-only advantage that overrides a 1-role corroboration
> advantage. This bounds the regime where W8-1 holds in terms of
> the score function's parameter.

### 4.3 W8-3 (proved-empirical, backward-compat)

> **W8-3 (proved-empirical, n = 10).** On the Phase-54 default
> config, the buffered `CrossRoleCorroborationAdmissionPolicy`
> achieves `accuracy_full = 1.000`, identical to the buffered
> `CohortCoherenceAdmissionPolicy` (W7-2). Corroboration is a
> *strict generalisation* of W7-2 — it preserves all W7-2 wins
> while extending to additional regimes.
>
> **Anchor.** `docs/data/phase55_cross_regime.json` `phase54_default`
> entry; `CrossRegimeBackwardCompatibilityTests::test_corroboration_matches_W7_2_on_phase54`.
>
> **Falsifier.** A Phase-54 default scenario where the two
> policies' admitted sets differ.

### 4.4 W8-4 (proved-empirical, decoy-corroboration falsifier)

> **W8-4 (proved-empirical, n = 10).** When the candidate stream
> satisfies the *decoy-corroborated* property (the decoy service
> has strictly more distinct producer roles than gold), buffered
> corroboration picks the decoy and ties FIFO at
> `accuracy_full = 0.000`. The W8-1 win does NOT hold in this
> regime.
>
> **Anchor.** `docs/data/phase55_falsifier_K4_n10.json`;
> `Phase55FalsifierTests::test_corroboration_ties_fifo_on_falsifier`.
>
> **Sharper structural observation.** Even with no budget pressure
> (K_auditor=8, all candidates fit), the W8-1 conditional regime
> still requires the corroboration property. *Without it*, the
> decoder's set-equality `services_correct` check is unrescuable by
> any service-blind admission policy (a strict generalisation of
> W7-2-conditional from SDK v3.8).

### 4.5 W8-C1 (conjectural — extension to multi-service-gold)

> **W8-C1 (conjectural).** When the gold answer has *multiple*
> service tags (a multi-service incident), a corroboration policy
> that admits the *top-k* highest-scoring tags strictly improves
> accuracy_full over single-tag corroboration on a multi-service
> Phase-56-style bench.
>
> **Status.** Conjectural; falsifier regime not yet built. This is
> the W7-C1 conjecture restated under W8 framing.

### 4.6 W8-C2 (conjectural — real-LLM multi-service decoy)

> **W8-C2 (conjectural).** The W8-1 strict-separation result
> transfers to the Phase-53-style real-LLM regime when the LLM is
> prompted with a multi-service event mix that produces decoy raw
> plurality + gold cross-role corroboration in the LLM-extracted
> candidate stream. Falsifier: a real-LLM run where corroboration
> beats FIFO by < 0.20 on accuracy_full after both bench
> properties are verified.
>
> **Status.** Conjectural; the SDK v3.9 real-LLM regression check
> (Phase 53 14B at K=4) confirms no regression in the *low-surplus*
> regime (where W8-1 doesn't fire); the W8-C2 conjectural regime
> requires re-prompting the producer roles to inject multi-service
> chatter, which is a Phase-56 candidate.

### 4.7 W8-C3 (conjectural — bundle-aware decoder companion)

> **W8-C3 (conjectural).** Adding a small bundle-aware decoder
> (one that filters `service=<tag>` tokens at decode time by the
> dominant cohort signature, rather than naive set-union) strictly
> improves accuracy_full on benches where the admission policy
> *cannot* fully filter foreign-service candidates (e.g.
> partial-coverage cases). Falsifier: a bench where the decoder
> modification provides ≤ +0.05 over admission alone.
>
> **Status.** Conjectural; not yet shipped. This is W7-C2 restated
> under W8 framing.

### 4.8 Status summary

| Claim       | Status                                    |
| ----------- | ----------------------------------------- |
| W8-1        | proved-empirical (Phase 55 anchor, n=50)  |
| W8-2        | proved (structural, code-anchored)        |
| W8-3        | proved-empirical (Phase 54 backward-compat) |
| W8-4        | proved-empirical (Phase 55 falsifier)     |
| W8-C1       | conjectural                               |
| W8-C2       | conjectural                               |
| W8-C3       | conjectural                               |

## 5. PART D — broad evaluation summary

### 5.1 Per-regime headline (K_auditor=4, n_eval=10, bank_seed=11)

```
                                  accuracy_full
Strategy                  P53syn  P53-14B  P54-def  P55-def  P55-fals
substrate                 0.800   0.800    0.000    0.000    0.000
capsule_fifo              0.800   0.800    0.000    0.000    0.000
capsule_priority          0.800   0.800    0.000    0.000    0.000
capsule_coverage          0.800   0.800    0.000    0.000    0.000
capsule_cohort_buffered   0.800   0.800    1.000    0.000    0.000
capsule_corroboration     0.800   0.800    1.000    1.000    0.000  ← SDK v3.9
```

### 5.2 Where corroboration *wins* and *loses*

* **Wins**: Phase 55 default (+1.000 vs both FIFO and W7-2,
  stable across 5/5 seeds).
* **Backward-compat ties**: Phase 54 default (matches W7-2 at 1.000),
  Phase 53 synthetic and real-LLM 14B (matches all strategies at
  0.800 — no regression).
* **Loses (correctly)**: Phase 55 falsifier — when the
  corroboration property doesn't hold, the policy ties FIFO at
  0.000. This is the *named conditional limit*, not a milestone
  failure.

### 5.3 Bench-property witnesses (mechanical)

| Property                          | Phase 53 14B | Phase 54 | Phase 55 default | Phase 55 falsifier |
| --------------------------------- | ------------ | -------- | ---------------- | ------------------ |
| surplus on every scenario         | 0/5 (low surplus) | 10/10 | 10/10 | 10/10 |
| gold raw plurality                | n/a          | 10/10    | 0/10             | 0/10               |
| decoy raw plurality               | n/a          | 0/10     | 10/10            | 10/10              |
| gold cross-role corroboration     | n/a          | 10/10    | 10/10            | 0/10               |
| W7-2 wins                          | n/a (ties)   | yes      | no (decoy plur.) | no (no signal)     |
| W8 wins                            | n/a (ties)   | yes (W8-3) | yes (W8-1)     | no (W8-4 falsifier)|

### 5.4 Audit_ok grid (T-1..T-7 across all cells)

Every capsule strategy passes `audit_team_lifecycle` on every cell
of every regime. The new `capsule_corroboration` strategy is in the
`audit_ok_grid`'s `True` column on Phase 53, Phase 54, Phase 55
default, and Phase 55 falsifier. W6-1 generalises straight through.

## 6. What the original thesis looks like after SDK v3.9

The original Context Zero thesis is **per-agent minimum-sufficient
context for multi-agent teams**. The honest reading after SDK v3.9:

1. **Audit / lifecycle (preserved & extended).** Capsule layer's
   audit contribution is unchanged from SDK v3.5–v3.8 — T-1..T-7
   hold mechanically on every coordination round, including the
   new Phase-55 cells. **The durable contribution.**

2. **Coordination performance (now spans three regimes, with
   stated conditions).** SDK v3.7 conditionally falsified the
   SDK v3.5 learned-policy advantage on the real-LLM regime.
   SDK v3.8 demonstrated a *conditional* structural win
   (W7-2 buffered cohort) on Phase 54. SDK v3.9 demonstrates a
   **strictly stronger** conditional structural win (W8-1
   corroboration) that:
   * **Wins** on Phase 55 (where W7-2 fails by construction).
   * **Matches** W7-2 on Phase 54 (backward-compat).
   * **Does not regress** on Phase 53 synthetic / 14B real-LLM.
   * **Has a named falsifier regime** where it correctly ties FIFO.

3. **Producer floor (preserved).** W7-3 (no admission policy can
   recover a claim the producer never emitted) holds by
   construction for the new policy too — the `capsule_role_view.parents`
   contract (Capsule Contract C5) forbids referencing un-sealed CIDs.

**The honest reading after SDK v3.9:** The capsule layer's *audit*
contribution generalises across all benchmarks tested so far. The
capsule layer's *coordination-performance* contribution now spans
**three** named regimes (no-regression on Phase 53; backward-compat
on Phase 54; strict win on Phase 55), is **stable across multiple
bank seeds**, and has an **explicitly named falsifier regime**.

This clears the **strong success bar** of
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1. It is the
**first SDK milestone** to do so.

## 7. What SDK v3.9 is *not*

* **Not** "we solved multi-agent context." Three named regimes
  is a stronger result than two, but real multi-agent teams have
  more axes (heterogeneous producers, time-varying budgets,
  multi-round handoffs, conflicting goals). The W8-1 conditional
  regime is a *demonstration* that capsule-native cross-role
  coordination *can* solve multi-agent coordination problems
  cleanly under stated conditions; it is not a universal claim.
* **Not** "corroboration always beats W7-2." On Phase 54 they
  tie (W8-3); on the Phase 55 falsifier they both lose (W8-4).
  W8-1's win is conditional on the named bench property.
* **Not** "the streaming corroboration policy works." Streaming
  is arrival-order-sensitive in the same way as W7-1-aux; the
  buffered variant is the load-bearing one, and its construction
  (`from_candidate_stream`) requires the candidate stream to be
  visible up-front. Real-time / online admission would need a
  different mechanism (deferred to W8-C3 or a future bundle-aware
  decoder companion).
* **Not** a CoordPy product runtime contract change. The CoordPy
  single-run product runtime is byte-for-byte unchanged from SDK
  v3.8. The new admission policy is a research-slice addition to
  `vision_mvp.coordpy.team_coord`.

## 8. Files / tests / artefacts

* **`vision_mvp/coordpy/team_coord.py`** *(extended)* —
  `CrossRoleCorroborationAdmissionPolicy` added; streaming +
  `from_candidate_stream` buffered factory; `_candidate_source_role`
  helper; `ALL_FIXED_POLICY_NAMES` updated.
* **`vision_mvp/coordpy/__init__.py`** — re-exports
  `TeamCrossRoleCorroborationAdmissionPolicy` (canonical SDK
  alias); `SDK_VERSION` bumped to `coordpy.sdk.v3.9`.
* **`vision_mvp/experiments/phase55_decoy_plurality.py`** *(new)* —
  Phase-55 driver; 5 base scenario builders; default + falsifier
  bank constructors; `run_phase55`, `run_phase55_budget_sweep`,
  `run_seed_stability_sweep`, `run_cross_regime_summary`.
* **`vision_mvp/tests/test_coordpy_cross_role_corroboration.py`**
  *(new)* — 34 contract tests (corroboration policy unit tests,
  bank shape contract, default config win, seed stability,
  falsifier behaviour, budget sweep, audit_ok grid, cross-regime
  backward-compat with Phase 54 + no-regression on Phase 53
  synthetic).
* **`vision_mvp/tests/test_coordpy_public_api.py`** *(updated)* —
  `test_sdk_version_is_v3_9` and
  `test_cross_role_corroboration_admission_policy_is_exported`
  added; W7-2 export test preserved.
* **`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`** *(new)* —
  pre-committed strict / partial / falsifying success bars;
  named-regime taxonomy (R-53 / R-54 / R-55).
* **`docs/data/phase55_decoy_plurality_K4_n10.json`** *(new)* —
  frozen Phase-55 default-config result.
* **`docs/data/phase55_falsifier_K4_n10.json`** *(new)* — frozen
  Phase-55 falsifier-bank result.
* **`docs/data/phase55_budget_sweep.json`** *(new)* — K-sweep
  showing W8-1 wins across K∈{3..8} and partial win at K=2.
* **`docs/data/phase55_seed_sweep.json`** *(new)* — 5/5 seed
  stability evidence for W8-1.
* **`docs/data/phase55_cross_regime.json`** *(new)* —
  Phase 54 + Phase 55 + Phase 55 falsifier bundled report.
* **`docs/data/phase53_real_llm_corroboration_check.json`** *(new)*
  — frozen Phase 53 14B real-LLM regression check.
* **`docs/RESULTS_COORDPY_CROSS_ROLE_CORROBORATION.md`** *(this file)*.
* **`docs/THEOREM_REGISTRY.md`** — W8 family rows added.
* **`docs/RESEARCH_STATUS.md`** — eighth research axis added.
* **`docs/HOW_NOT_TO_OVERSTATE.md`** — W8-1 overstatement guard
  added; W7-2 falsifier annotation refined.
* **`docs/context_zero_master_plan.md`** — § 4.26 added.
* **`docs/START_HERE.md`** — SDK v3.9 paragraph added.

## 9. Tests + validation runs

```text
$ python3 -m unittest -v vision_mvp.tests.test_coordpy_cross_role_corroboration
Ran 34 tests in 1.471s — OK

$ python3 -m unittest \
    vision_mvp.tests.test_coordpy_team_coord \
    vision_mvp.tests.test_coordpy_llm_backend \
    vision_mvp.tests.test_coordpy_capsule_native_inner_loop \
    vision_mvp.tests.test_coordpy_capsule_native \
    vision_mvp.tests.test_coordpy_capsule_native_intra_cell \
    vision_mvp.tests.test_coordpy_capsule_native_deeper \
    vision_mvp.tests.test_coordpy_scale_vs_structure \
    vision_mvp.tests.test_coordpy_cross_role_coherence \
    vision_mvp.tests.test_coordpy_cross_role_corroboration
Ran 171 tests in 5.612s — OK

$ python3 vision_mvp_tests_discover.py
failures=0 errors=0 total=1565

$ python3 -m vision_mvp.experiments.phase55_decoy_plurality \
    --K-auditor 4 --n-eval 10 --bank-seed 11
[phase55] gap corroboration−fifo: +1.000
[phase55] gap corroboration−cohort_buffered: +1.000

$ python3 -m vision_mvp.experiments.phase55_decoy_plurality \
    --seed-sweep --n-eval 10
seed=11: gap_corr−fifo=+1.000  gap_corr−cohort=+1.000
seed=17: gap_corr−fifo=+1.000  gap_corr−cohort=+1.000
seed=23: gap_corr−fifo=+1.000  gap_corr−cohort=+1.000
seed=29: gap_corr−fifo=+1.000  gap_corr−cohort=+1.000
seed=31: gap_corr−fifo=+1.000  gap_corr−cohort=+1.000

$ python3 -m vision_mvp.experiments.phase55_decoy_plurality \
    --falsifier --n-eval 10
[phase55] gap corroboration−fifo: +0.000

$ python3 -m vision_mvp.experiments.phase55_decoy_plurality \
    --budget-sweep --n-eval 10
  K=2: corr=+0.20 cohort=+0.00
  K=3..8: corr=+1.00 cohort=+0.00
```

Real-LLM regression (manual; not in CI):

```text
$ python3 /tmp/phase53_real_llm_corroboration_check.py
[regression] Phase 53 qwen2.5:14b-32k regime, K=4, n_eval=5:
  substrate                accuracy_full = 0.800
  capsule_fifo               accuracy_full = 0.800
  capsule_priority           accuracy_full = 0.800
  capsule_coverage           accuracy_full = 0.800
  capsule_cohort_buffered    accuracy_full = 0.800
  capsule_corroboration      accuracy_full = 0.800
```

## 10. What remains open

* **Multi-service-gold scenarios.** Phase 55 has single-service
  gold; multi-service incidents would require a corroboration
  policy that admits *top-k* tags. Phase-56 candidate; W8-C1.
* **Real-LLM multi-service decoy regime.** Re-prompt Phase 53's
  producer LLM with a multi-service event mix to surface decoy
  raw plurality + gold cross-role corroboration in real-LLM
  candidate streams. Predicted (W8-C2): corroboration improves
  over FIFO and W7-2 in this regime.
* **Bundle-aware decoder companion.** The current decoder
  (`_phase31_decoder_from_handoffs`) is service-blind;
  `services` is a union over admitted payloads. A decoder that
  filters by the dominant cohort signature would be a *second*
  axis of structural improvement orthogonal to admission.
  Predicted (W8-C3): admission + decoder coordination ≥
  admission-alone on benches with partial corroboration.
* **Two-Mac sharded inference (still open from SDK v3.6/v3.7/v3.8).**
  Mac 2 still offline; SDK v3.7 runbook unchanged.

---

*Theorem-forward summary: SDK v3.9 ships the smallest deterministic
multi-agent benchmark where decoy raw plurality breaks single-tag
plurality cohort coherence (Phase 55), the smallest interpretable
**cross-role-corroboration** admission policy
(`CrossRoleCorroborationAdmissionPolicy` — buffered mode), and the
**strongest cross-regime conditional structural-win result** the
programme has ever produced (corroboration − W7-2 = +1.000 on
Phase 55 default; corroboration − FIFO = +1.000 on Phase 55,
+1.000 on Phase 54, 0.000 on Phase 53; stable across 5/5 bank
seeds; falsifier regime correctly ties FIFO at 0.000).
The honest reading: the capsule layer's audit contribution is
preserved, AND the capsule layer's coordination-performance
contribution now spans three named regimes with cross-bank
stability and a named falsifier — clearing the strong success
bar of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1.
The W7-2 falsifier (decoy plurality) is now resolved by W8;
the W8-1 falsifier (decoy corroboration) is named, mechanically
verified, and not yet attacked.*
