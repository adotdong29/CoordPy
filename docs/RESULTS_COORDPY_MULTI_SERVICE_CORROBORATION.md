# SDK v3.10 — Multi-Service Cross-Role Corroboration

> Milestone results note. SDK v3.9 produced a **strict-separation
> conditional** result on multi-agent coordination: when the candidate
> stream satisfies the *decoy-plurality + cross-role-corroborated-gold*
> property, the buffered `CrossRoleCorroborationAdmissionPolicy`
> (W8-1) beats both substrate FIFO and SDK v3.8 W7-2 buffered cohort
> by +1.000 on `accuracy_full` (Phase 55).
>
> But W8 has a **named falsifier**: it picks the top-1 corroborated
> tag and *only* the top-1. On any *multi-service-gold* regime
> (where `gold_services = (A, B)` with both A and B causal — the
> canonical realistic case), W8 admits only candidates carrying the
> single highest-scoring tag and the decoder's set-equality
> `services_correct` check fails. This is the *W8 multi-service-gold
> falsifier*, named in `HOW_NOT_TO_OVERSTATE.md` since SDK v3.9.
>
> SDK v3.10 directly attacks that falsifier. It (a) builds the
> Phase-56 *multi-service-gold + cross-role-corroborated* regime —
> a fairer, harder bench than Phase 55 because every scenario
> has **two** gold services that each independently require
> corroboration evidence — (b) implements one stronger admission
> method that admits the **top-K cross-role-corroborated tier** rather
> than the single highest-scoring tag, gated by an
> argmax-by-role-count rule that preserves W8 backward-compat, and
> (c) shows a clean strict separation: on Phase 56,
> `MultiServiceCorroborationAdmissionPolicy` (W9) beats substrate
> FIFO, SDK v3.8 W7-2 buffered cohort, **and** SDK v3.9 W8
> corroboration by **+1.000** on `accuracy_full`, stable across
> **5/5** alternate bank seeds, while preserving the W8 win on Phase
> 55 backward-compatibly and not regressing on Phase 53 / 54.
>
> Theorem-forward: SDK v3.10 mints the **W9 theorem family**
> (W9-1 / W9-2 / W9-3 / W9-4 — proved or proved-empirical) and is
> honest about *where* the win lives (multi-service incidents with
> single-role decoy storms) and where it doesn't (decoy-corroborated
> falsifier regime, single-service gold trivially solved by W8).
> The headline result clears the **strong success bar** of
> `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1 (R-56
> anchor).
> Last touched: 2026-04-26.

## TL;DR — what SDK v3.10 ships

* **Phase-56 multi-service-corroboration benchmark**
  (`vision_mvp/experiments/phase56_multi_service_corroboration.py`).
  The smallest deterministic multi-agent benchmark where (i) every
  scenario has `gold_services` of size **2**, (ii) both gold
  services are corroborated by ≥ 2 distinct producer roles, AND
  (iii) at least one *decoy* service has raw plurality but is
  corroborated by exactly 1 producer role. 10/10 scenarios on the
  default bank satisfy all three properties; mechanical witnesses
  are in `Phase56BankShapeTests`.
* **`MultiServiceCorroborationAdmissionPolicy`**
  (`vision_mvp/coordpy/team_coord.py`). A deterministic, training-free,
  interpretable admission rule that admits the top-K cross-role-
  corroborated tags above a min-role threshold, gated by an
  *argmax-by-role-count* rule so it strictly generalises W8
  single-tag corroboration. Default: `top_k=2,
  min_corroborated_roles=2, role_weight=100`. Two sub-modes
  (streaming / buffered) consistent with W7-2 and W8 design; the
  buffered factory `from_candidate_stream` is the W9 anchor.
* **W9 theorem family** (4 members; all proved or proved-empirical)
  + 1 conjecture (W9-C1).
* **Cross-regime evaluation** across four named regimes:

  | regime              | multi_service | corroboration | W7-2 cohort | FIFO  | gap (ms − fifo) |
  | ------------------- | ------------- | ------------- | ----------- | ----- | --------------- |
  | Phase 53 synthetic  |   0.800       |   0.800       |   0.800     | 0.800 |   +0.000        |
  | Phase 54 default    |   1.000       |   1.000       |   1.000     | 0.000 |   +1.000        |
  | Phase 55 default    |   1.000       |   1.000       |   0.000     | 0.000 |   +1.000        |
  | **Phase 56 default**|   **1.000**   |   **0.000**   |   **0.000** | **0.000** | **+1.000** |
  | Phase 56 falsifier  |   0.000       |   0.000       |   0.000     | 0.000 |   +0.000        |

  **Phase 56**: multi_service wins +1.000 vs FIFO **and vs W8**
  (stable across 5/5 bank seeds — the headline result). **Phase
  55**: multi_service ties W8 at 1.000 — backward-compat preserved
  via the argmax-by-role-count gate (W9-3). **Phase 54**: ties W7-2
  at 1.000 — single-service-gold case collapses to the
  argmax-by-role-count tier of size 1 (also W9-3). **Phase 53**:
  no regression vs FIFO at K=4 (synthetic). **Phase 56 falsifier**
  (decoy promoted to ≥ 2 distinct roles): multi_service ties FIFO
  at 0.000 — confirming the conditional nature of the W9-1 win.

* **Audit invariants preserved.** All capsule strategies — fifo,
  priority, coverage, cohort_buffered, corroboration, multi_service
  — pass `audit_team_lifecycle` (T-1..T-7) on every cell of every
  regime. The new `capsule_multi_service` strategy is in the
  `audit_ok_grid`'s `True` column on Phase 53 / 54 / 55 / 56 default
  / 56 falsifier. W6-1 generalises straight through.

## 1. The W8 weakness this milestone attacks

SDK v3.9 W8 corroboration uses a *single-tag-by-corroboration-score*
admission rule: pick the service tag with the highest cross-role
corroboration score and admit only candidates carrying that tag.
This works on Phase 55 because the bench has *single-service-gold*
(`gold_services` of size 1) by design.

It fails as soon as the bench has multi-service gold. Concretely:

```
candidate stream (1 scenario, abbreviated; gold = (api, db)):
  monitor:  service=archival × 3   (decoy storm, single-role)
  monitor:  service=api      × 1   (gold A, monitor)
  monitor:  service=db       × 1   (gold B, monitor)
  db_admin: service=api      × 1   (gold A, db_admin → 2 roles)
  db_admin: service=db       × 1   (gold B, db_admin → 2 roles)

W8 buffered corroboration:
  scores: api=200+2=202, db=200+2=202, archival=100+3=103
  → top-1 (lex tie-break) = "api"
  → admits 2 api-tagged candidates
  → decoder's services = {api}
  → grading: services_correct = False (gold = {api, db}); fail.
```

A policy that admits the **top-K cross-role-corroborated** tier —
the `MultiServiceCorroborationAdmissionPolicy` shipped in this SDK
— picks `{api, db}` (both 2-role-corroborated, in the
argmax-by-role-count tier; archival excluded by min-role threshold)
and admits 4 candidates spanning both gold services. The decoder
produces `services = {api, db}` and `services_correct = True`.

This is exactly the *W8 multi-service-gold falsifier* in
`HOW_NOT_TO_OVERSTATE.md`:

> "If gold_services has size > 1, single-tag corroboration admits
> only the top-1 tag and ties FIFO at services_correct = False."

## 2. The Phase-56 benchmark — design, properties, falsifier

### 2.1 Why this bench is fair, not benchmark-shopping

The bench combines four pre-committed properties (mechanically
verified by `Phase56BankShapeTests`):

* **Multi-service gold.** Every scenario has `gold_services` of
  size 2. This is the canonical realistic incident shape: a
  pool-cascade between api + db; a deadlock between orders +
  payments; a slow-query cascade between web + db; an error spike
  hitting api + mobile; a disk fill cascading to storage + logs.
  The single-service Phase-55 case is the *easy* case; multi-
  service is the regime where the auditor must aggregate evidence
  across distinct gold tags simultaneously.
* **Cross-role corroboration of *both* gold tags**: each gold
  service is mentioned by ≥ 2 distinct producer roles
  (10/10 scenarios). This is the relational signal both the W8 and
  W9 policies can exploit — but only W9's top-K formulation can
  capture it on multiple tags simultaneously.
* **Single-role decoy storm with raw plurality**: at least one
  decoy service has strictly more raw mentions than each gold
  service AND is corroborated by exactly 1 producer role
  (10/10 scenarios). This is what makes W7-2 fail (it picks the
  decoy plurality) and why FIFO fails (the decoy storm comes early
  in the stream and consumes the budget).
* **Budget pressure**: every scenario has
  `|candidates_to_auditor| > K_auditor` at K=4 (10/10 scenarios).
* **Realistic claim_kinds**: each gold-tagged claim carries a
  causal `claim_kind` whose decoder priority maps to the gold
  `root_cause`; decoys carry plausible-but-not-causal kinds (or
  the same kinds with different service tags).

Realistic distractors: every real production incident the SRE
community catalogues involves multiple services. A decoy storm on
a single role (e.g. archival cron filling sysadmin telemetry, or
metrics-scrape latency hitting monitor) is the canonical
single-role distractor. A human SRE correlates signals across
dashboards and notices the *cross-role* pattern; the W9 admission
policy is the smallest mechanical move that captures both halves
of that correlation simultaneously.

### 2.2 Pre-committed bench property + falsifier

The bench property is named in code:

```python
def _candidate_stats(candidates, gold_services_pair, decoy_storm_service):
    ...
    return {
        ...
        "n_gold_corroborated_roles_geq2": n_gold_corroborated,
        "max_decoy_role_count": other_max,
        "multi_service_gold_property_holds": (
            n_gold_corroborated == 2 and other_max <= 1),
        "decoy_corroboration_holds": other_max >= 2,
        ...
    }
```

A *named* falsifier regime
(`build_phase56_falsifier_bank`) instantiates the inverse property:
a decoy is promoted to ≥ 2 distinct producer roles. On this regime,
`decoy_corroboration_holds = True` for every scenario (mechanically
verified by `Phase56FalsifierTests::test_decoy_corroboration_property_holds_on_falsifier_bank`).

This makes the W9-1 conditionality sharp: when the bench property
holds, multi_service wins; when it doesn't, multi_service ties FIFO.

## 3. The multi-service corroboration admission policy (SDK v3.10)

### 3.1 Public surface

```python
from vision_mvp.coordpy import (
    TeamMultiServiceCorroborationAdmissionPolicy,  # canonical SDK alias
)
from vision_mvp.coordpy.team_coord import (
    MultiServiceCorroborationAdmissionPolicy,
)

# Streaming mode (research baseline — arrival-order-sensitive):
policy = MultiServiceCorroborationAdmissionPolicy(
    top_k=2, min_corroborated_roles=2)

# Buffered mode (W9-1 anchor — arrival-order-stable):
stream = [(c[0], c[3]) for c in candidate_stream]
policy = MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
    stream, top_k=2, min_corroborated_roles=2)
```

### 3.2 Selection rule (one regex, one counter, three filters)

For each `service=<tag>` token in the candidate stream:

```
score(tag) = role_weight * |distinct_roles(tag)| + |raw_mentions(tag)|
```

Then:

1. **Min-role floor.** Drop any tag with
   `|distinct_roles(tag)| < min_corroborated_roles` (default 2).
2. **Argmax-by-role-count tier.** Of the remaining tags, take all
   tags whose distinct-role count equals the maximum distinct-role
   count among the eligible set. This is the **structural cross-
   role-corroboration tier** — the load-bearing W9 move.
3. **Top-K by score.** Among the argmax tier, take the `top_k`
   highest-scoring tags (lex tie-break on the tag name).

The buffered policy admits a candidate iff its tag is in the
resulting dominant set (or the candidate has no tag).

### 3.3 Why the argmax-by-role-count gate is load-bearing

Without it, on a Phase-55-style single-service-gold regime where
the gold has strictly more distinct producer roles than every
decoy, a *second* tag (a decoy) with sub-max role count but high
raw count could enter the top-K dominant set, breaking
backward-compat with W8 single-tag corroboration. The argmax
gate ensures W9 *collapses to W8* when only one tag has the
maximum role count — making W9 a strict generalisation of W8.

This is W9-3 (proved-empirical, mechanically tested by
`W9BackwardCompatTests::test_w9_matches_w8_on_phase55_default`).

### 3.4 Why it is honest

* **Deterministic** and **small**: one regex (re-using
  `_SERVICE_TAG_RE`), one counter over (role, tag) pairs, no
  learning, no training-distribution dependency.
* **Strictly more general than W8 by construction**: when the
  argmax-by-role-count tier has size 1, W9 admits exactly the same
  set as W8. On Phase-55 default, the two are indistinguishable on
  `accuracy_full` (W9-3).
* **Falsifiable**: a stream where a *decoy* is also corroborated
  above the threshold AND has competitive score falsifies W9-1.
  The W9-1 falsifier regime is the named
  `build_phase56_falsifier_bank`.

### 3.5 Lifecycle invariants

The policy preserves T-1..T-7 by construction: it returns standard
`AdmissionDecision` records via the existing `TeamCoordinator`
admission path. No new lifecycle states; W6-1 generalisation to
Phase-56 holds (mechanically tested by
`W9LifecycleAuditTests::test_default_bank_audit_ok_on_every_cell`).

## 4. Theory — what the W9 family says

### 4.1 W9-1 (proved-empirical, strict separation)

> **W9-1 (proved-empirical, n = 10 × 5 bank seeds = 50 scenarios
> saturated).** When the candidate stream satisfies the
> *multi-service-gold-corroborated property* (every gold service is
> mentioned by ≥ 2 distinct producer roles AND every non-gold
> service is mentioned by ≤ 1 distinct producer role) and
> `|candidates| > K_auditor`, the *buffered*
> `MultiServiceCorroborationAdmissionPolicy` (constructed via
> `from_candidate_stream` with `top_k=2, min_corroborated_roles=2`)
> achieves `accuracy_full = 1.000` while substrate FIFO,
> `CapsuleFifoAdmissionPolicy`, `ClaimPriorityAdmissionPolicy`,
> `CoverageGuidedAdmissionPolicy`,
> `CohortCoherenceAdmissionPolicy` (W7-2 buffered),
> AND `CrossRoleCorroborationAdmissionPolicy` (W8 buffered) all
> produce `accuracy_full = 0.000`. The headline gap is **+1.000**
> on the pre-committed Phase-56 default (`K_auditor=4,
> T_auditor=128, n_eval=10, bank_seed ∈ {11, 17, 23, 29, 31}`).
> This is the **first strict separation between W9 and W8** in the
> programme, and the first programme result whose strict-gain
> regime spans **four** named bench regimes (Phase 53 no-regression,
> Phase 54 backward-compat, Phase 55 backward-compat, Phase 56
> strict win) with cross-bank stability.
>
> **Anchors.** `docs/data/phase56_multi_service_K4_n10.json`;
> `docs/data/phase56_seed_sweep.json`;
> `Phase56DefaultConfigTests::test_multi_service_strictly_beats_w8_at_K4`;
> `Phase56SeedStabilityTests::test_gap_holds_across_five_seeds`.
>
> **Falsifier.** A Phase-56 default config where
> `multi_service_minus_corroboration_accuracy_full < 0.50`. (Test
> fails; CI gates W9-1.) AND a named falsifier regime
> (`build_phase56_falsifier_bank`) where
> `decoy_corroboration_holds = True` everywhere and multi_service
> ties FIFO (mechanically verified by
> `Phase56FalsifierTests::test_multi_service_ties_fifo_on_falsifier`).

### 4.2 W9-2 (proved, structural)

> **W9-2 (proved).** Let `eligible(stream) = {t : |distinct_roles(t)|
> ≥ min_corroborated_roles}` and `argmax_roles(stream) =
> {t ∈ eligible : |distinct_roles(t)| = max_{u ∈ eligible}
> |distinct_roles(u)|}`. Then for any candidate stream and any
> `min_corroborated_roles ≥ 1`:
>
> *(a) Single-role exclusion.* For any tag `t` with
> `|distinct_roles(t)| < min_corroborated_roles`,
> `t ∉ dominant_set`. ∎ *(by construction).*
>
> *(b) Argmax-tier collapse.* If `|argmax_roles(stream)| = 1`, then
> `dominant_set = argmax_roles(stream) =
> {`tag with the most distinct roles`}`, regardless of `top_k ≥ 1`.
>
> *(c) Multi-tag admission.* If `|argmax_roles(stream)| = k₀ ≤ top_k`,
> then `dominant_set = argmax_roles(stream)` (size `k₀`).
>
> **Anchor.** `team_coord.MultiServiceCorroborationAdmissionPolicy`
> docstring; `_dominant_tag_set` helper;
> `MultiServicePolicyUnitTests::test_dominant_set_collapses_to_w8_under_role_count_argmax`;
> `MultiServicePolicyUnitTests::test_dominant_set_admits_multi_service_gold`.
>
> **Practical consequence.** With the default
> `min_corroborated_roles=2`, no single-role decoy storm can ever
> enter the dominant set, regardless of raw count. With the
> default `top_k=2`, at most two tags are admitted — sufficient
> for size-2 multi-service-gold benches.

### 4.3 W9-3 (proved-empirical, backward-compat with W8 + W7-2)

> **W9-3 (proved-empirical, n = 10).** On the Phase-55 default
> config (single-service-gold; gold has strictly more distinct
> producer roles than every decoy), the buffered W9 policy achieves
> `accuracy_full = 1.000`, identical to the buffered W8
> corroboration policy. By W9-2(b), `argmax_roles(stream)` has size
> 1 and W9's dominant set equals W8's `fixed_dominant_tag` —
> mechanically tested by
> `W9BackwardCompatTests::test_w9_admits_same_set_as_w8_when_one_tag_dominates`.
>
> Similarly, on Phase-54 default, W9 ties W7-2 buffered cohort at
> 1.000 (single-service-gold case).
>
> W9 is therefore a **strict generalisation** of W8: it preserves
> all W8 wins while extending to multi-service-gold regimes.
>
> **Anchor.** `docs/data/phase56_cross_regime.json` `phase55_default`
> and `phase54_default` entries;
> `W9BackwardCompatTests::test_w9_matches_w8_on_phase55_default`.
>
> **Falsifier.** A Phase-55 default scenario where W9 and W8 admit
> different sets.

### 4.4 W9-4 (proved-empirical, decoy-corroboration falsifier)

> **W9-4 (proved-empirical, n = 10).** When the candidate stream
> satisfies the *decoy-corroborated* property (at least one decoy
> service has ≥ `min_corroborated_roles` distinct producer roles
> AND its corroboration score is competitive with at least one gold
> service's), the W9 policy admits the decoy (because it enters
> the argmax-by-role-count tier) and the auditor's `services` set
> includes the decoy → `services_correct` fails → `accuracy_full =
> 0.000`. The W9-1 win does NOT hold in this regime.
>
> **Anchor.** `docs/data/phase56_falsifier_K4_n10.json`;
> `Phase56FalsifierTests::test_multi_service_ties_fifo_on_falsifier`.
>
> **Sharper structural observation.** The W9-4 conditional regime
> reveals a deep limitation: *no service-blind admission policy*
> can solve the multi-service decoder under decoy corroboration,
> because the decoder reads the union of service tags. Under
> decoy corroboration, the only structural escape is a
> **bundle-aware decoder companion** that filters service tags at
> decode time by *which tags actually carry the priority causal
> claim_kinds*. This is W9-C1 below.

### 4.5 W9-C1 (conjectural — bundle-aware decoder companion)

> **W9-C1 (conjectural).** Adding a small bundle-aware decoder that
> filters `service=<tag>` tokens at decode time by the dominant
> *(claim_kind, role)* signature — i.e. only tag `t` is added to
> `services` if at least one admitted handoff with tag `t` carries
> a causal claim_kind for the inferred root_cause — strictly
> improves accuracy_full on the Phase-56 falsifier regime where
> admission alone admits a decoy. Falsifier: a Phase-56 falsifier
> bench where the decoder modification provides ≤ +0.20 over
> admission alone.
>
> **Status.** Conjectural; not yet shipped. This re-frames W8-C3
> under the W9 framing and pushes the structural research axis
> from *admission* into *decoding* — the two axes the SDK can
> independently improve.

### 4.6 W9-C2 (conjectural — top-K with K > 2)

> **W9-C2 (conjectural).** The W9-1 strict-separation result
> generalises to multi-service incidents with `|gold_services| ≥ 3`
> when `top_k = |gold_services|`, provided every gold service is
> corroborated by ≥ 2 distinct producer roles AND every decoy is
> corroborated by ≤ 1. Falsifier: a Phase-57 bench (3-service gold)
> where W9 with `top_k=3` beats FIFO by < 0.50.
>
> **Status.** Conjectural; not yet built.

### 4.7 W9-C3 (conjectural — real-LLM multi-service decoy)

> **W9-C3 (conjectural).** The W9-1 strict-separation result
> transfers to the Phase-53-style real-LLM regime when the LLM is
> prompted with a multi-service event mix that produces multi-
> service gold corroboration in the LLM-extracted candidate stream.
> Falsifier: a real-LLM run where multi_service beats W8 by < 0.20
> on accuracy_full after the multi-service-gold property is
> verified.
>
> **Status.** Conjectural; the SDK v3.10 real-LLM regression check
> (Phase 53 synthetic at K=4) confirms no regression in the
> low-surplus regime (where W9-1 doesn't fire); the W9-C3 conjectural
> regime requires re-prompting Phase-53's producer LLM with multi-
> service event mixes.

### 4.8 Status summary

| Claim       | Status                                        |
| ----------- | --------------------------------------------- |
| W9-1        | proved-empirical (Phase 56 anchor, n=50)      |
| W9-2        | proved (structural, code-anchored)            |
| W9-3        | proved-empirical (Phase 55 + 54 backward-compat) |
| W9-4        | proved-empirical (Phase 56 falsifier)         |
| W9-C1       | conjectural (bundle-aware decoder companion) |
| W9-C2       | conjectural (top-K for |gold| ≥ 3)            |
| W9-C3       | conjectural (real-LLM multi-service)          |

## 5. PART D — broad evaluation summary

### 5.1 Per-regime headline (K_auditor=4, n_eval=10, bank_seed=11)

```
                                   accuracy_full
Strategy                  P53syn  P54-def  P55-def  P56-def  P56-fals
substrate                 0.800   0.000    0.000    0.000    0.000
capsule_fifo              0.800   0.000    0.000    0.000    0.000
capsule_priority          0.800   0.000    0.000    0.000    0.000
capsule_coverage          0.800   0.000    0.000    0.000    0.000
capsule_cohort_buffered   0.800   1.000    0.000    0.000    0.000
capsule_corroboration     0.800   1.000    1.000    0.000    0.000
capsule_multi_service     0.800   1.000    1.000    1.000    0.000   ← SDK v3.10
```

### 5.2 Where multi_service *wins* and *loses*

* **Wins**: Phase 56 default (+1.000 vs FIFO, W7-2, **and W8**,
  stable across 5/5 seeds — the headline result).
* **Backward-compat ties**: Phase 55 default (matches W8 at 1.000
  via the argmax-by-role-count gate, W9-3); Phase 54 default
  (matches W7-2 at 1.000); Phase 53 synthetic (matches all
  strategies at 0.800 — no regression).
* **Loses (correctly)**: Phase 56 falsifier — when the bench
  property doesn't hold (decoy corroborated by ≥ 2 roles), the
  policy ties FIFO at 0.000. This is the *named conditional limit*,
  not a milestone failure.

### 5.3 Bench-property witnesses (mechanical)

| Property                          | Phase 53 | Phase 54 | Phase 55 default | Phase 56 default | Phase 56 falsifier |
| --------------------------------- | -------- | -------- | ---------------- | ---------------- | ------------------ |
| surplus on every scenario         | low      | 10/10    | 10/10            | 10/10            | 10/10              |
| gold raw plurality                | n/a      | 10/10    | 0/10             | 0/10             | 0/10               |
| decoy raw plurality               | n/a      | 0/10     | 10/10            | 10/10            | 10/10              |
| gold cross-role corroboration     | n/a      | 10/10    | 10/10            | 10/10 each gold  | 10/10 each gold    |
| multi-service gold (size 2)       | n/a      | n/a      | n/a              | 10/10            | 10/10              |
| decoy cross-role corroboration    | n/a      | n/a      | 0/10             | 0/10             | 10/10              |
| W7-2 wins                          | n/a      | yes      | no               | no               | no                 |
| W8 wins                            | n/a      | yes (W8-3) | yes (W8-1)     | no (W8 falsif.)  | no                 |
| W9 wins                            | n/a      | yes (W9-3) | yes (W9-3)     | yes (W9-1)       | no (W9-4 falsif.)  |

### 5.4 Audit_ok grid (T-1..T-7 across all cells)

Every capsule strategy passes `audit_team_lifecycle` on every cell
of every regime. The new `capsule_multi_service` strategy is in the
`audit_ok_grid`'s `True` column on Phase 53 / 54 / 55 / 56 default
/ 56 falsifier. W6-1 generalises straight through.

## 6. What the original thesis looks like after SDK v3.10

The original Context Zero thesis is **per-agent minimum-sufficient
context for multi-agent teams**. The honest reading after SDK v3.10:

1. **Audit / lifecycle (preserved & extended).** Capsule layer's
   audit contribution is unchanged from SDK v3.5–v3.9 — T-1..T-7
   hold mechanically on every coordination round, including the
   new Phase-56 cells. **The durable contribution.**

2. **Coordination performance (now spans four regimes, with
   stated conditions).** SDK v3.7 conditionally falsified the
   SDK v3.5 learned-policy advantage on the real-LLM regime.
   SDK v3.8 demonstrated a *conditional* structural win
   (W7-2 buffered cohort) on Phase 54. SDK v3.9 demonstrated a
   *strictly stronger* conditional structural win
   (W8-1 corroboration) on Phase 55. SDK v3.10 demonstrates a
   **strictly stronger still** conditional structural win
   (W9-1 multi-service corroboration) that:
   * **Wins** on Phase 56 (where W8 fails by construction).
   * **Matches** W8 on Phase 55 (backward-compat via argmax gate).
   * **Matches** W7-2 on Phase 54 (single-service-gold case).
   * **Does not regress** on Phase 53 synthetic.
   * **Has a named falsifier regime** where it correctly ties FIFO.

3. **Producer floor (preserved).** W7-3 (no admission policy can
   recover a claim the producer never emitted) holds by
   construction for the new policy too — the
   `capsule_role_view.parents` contract (Capsule Contract C5)
   forbids referencing un-sealed CIDs.

**The honest reading after SDK v3.10:** The capsule layer's *audit*
contribution generalises across all benchmarks tested so far. The
capsule layer's *coordination-performance* contribution now spans
**four** named regimes (no-regression on Phase 53; backward-compat
on Phase 54 and Phase 55; strict win on Phase 56), is **stable
across multiple bank seeds**, and has an **explicitly named
falsifier regime**.

This clears the **strong success bar** of
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1 (R-56 anchor).
It is the **second consecutive SDK milestone** to do so — and the
first whose strict-gain regime is *not* solvable by the previous
SDK's strongest method.

## 7. What SDK v3.10 is *not*

* **Not** "we solved multi-agent context." Four named regimes
  is a stronger result than three, but real multi-agent teams have
  more axes (heterogeneous producers, time-varying budgets,
  multi-round handoffs, conflicting goals, multi-service incidents
  with `|gold| ≥ 3`). The W9-1 conditional regime is a
  *demonstration* that capsule-native cross-role coordination
  *can* solve multi-service multi-agent coordination problems
  cleanly under stated conditions; it is not a universal claim.
* **Not** "multi_service always beats W8." On Phase 55 they tie
  (W9-3); on Phase 56 falsifier they both lose (W9-4). W9-1's win
  is conditional on the multi-service-gold + single-role-decoy
  property.
* **Not** "the streaming multi_service policy works." Streaming is
  arrival-order-sensitive in the same way as W7-1-aux and W8
  streaming; the buffered variant is the load-bearing one, and its
  construction (`from_candidate_stream`) requires the candidate
  stream to be visible up-front. Real-time / online admission would
  need a different mechanism (deferred to W9-C1 or a future
  bundle-aware decoder companion).
* **Not** a CoordPy product runtime contract change. The CoordPy
  single-run product runtime is byte-for-byte unchanged from SDK
  v3.9. The new admission policy is a research-slice addition to
  `vision_mvp.coordpy.team_coord` (additive surface).

## 8. Files / tests / artefacts

* **`vision_mvp/coordpy/team_coord.py`** *(extended)* —
  `MultiServiceCorroborationAdmissionPolicy` added; streaming +
  `from_candidate_stream` buffered factory; `_dominant_tag_set`
  helper; `ALL_FIXED_POLICY_NAMES` updated.
* **`vision_mvp/coordpy/__init__.py`** — re-exports
  `TeamMultiServiceCorroborationAdmissionPolicy` (canonical SDK
  alias); `SDK_VERSION` bumped to `coordpy.sdk.v3.10`.
* **`vision_mvp/experiments/phase56_multi_service_corroboration.py`**
  *(new)* — Phase-56 driver; 5 base scenario builders; default +
  falsifier bank constructors; `run_phase56`,
  `run_phase56_seed_stability_sweep`, `run_cross_regime_summary`.
* **`vision_mvp/tests/test_coordpy_multi_service_corroboration.py`**
  *(new)* — 36 contract tests (multi_service policy unit tests,
  bank shape contract, default config win, seed stability,
  falsifier behaviour, budget pressure, audit_ok grid, cross-regime
  backward-compat with Phase 55 + no-regression on Phase 53
  synthetic).
* **`vision_mvp/tests/test_coordpy_public_api.py`** *(updated)* —
  `test_sdk_version_is_v3_10` (renamed from v3_9); W8 and W9 export
  tests both preserved.
* **`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`** *(updated)* —
  R-56 named regime added; bar anchor advanced to R-56; falsifying-
  failure list extended to gate W8-1 contract test.
* **`docs/data/phase56_multi_service_K4_n10.json`** *(new)* —
  frozen Phase-56 default-config result.
* **`docs/data/phase56_falsifier_K4_n10.json`** *(new)* — frozen
  Phase-56 falsifier-bank result.
* **`docs/data/phase56_seed_sweep.json`** *(new)* — 5/5 seed
  stability evidence for W9-1.
* **`docs/data/phase56_cross_regime.json`** *(new)* —
  Phase 54 + Phase 55 + Phase 56 default + Phase 56 falsifier
  bundled report.
* **`docs/data/phase53_synthetic_w9_regression_check.json`** *(new)*
  — frozen Phase-53 synthetic regression check (W9 ties FIFO).
* **`docs/RESULTS_COORDPY_MULTI_SERVICE_CORROBORATION.md`** *(this file)*.
* **`docs/THEOREM_REGISTRY.md`** *(extended)* — W9 family rows added.
* **`docs/RESEARCH_STATUS.md`** *(extended)* — ninth research axis added.
* **`docs/HOW_NOT_TO_OVERSTATE.md`** *(extended)* — W9-1 overstatement
  guard added; W8 multi-service-gold falsifier annotation refined.
* **`docs/context_zero_master_plan.md`** *(extended)* — § 4.27 added.
* **`docs/START_HERE.md`** *(extended)* — SDK v3.10 paragraph added.

## 9. Tests + validation runs

```text
$ python3 -m unittest -v vision_mvp.tests.test_coordpy_multi_service_corroboration
Ran 36 tests in 0.642s — OK

$ python3 -m unittest \
    vision_mvp.tests.test_coordpy_team_coord \
    vision_mvp.tests.test_coordpy_llm_backend \
    vision_mvp.tests.test_coordpy_capsule_native_inner_loop \
    vision_mvp.tests.test_coordpy_capsule_native \
    vision_mvp.tests.test_coordpy_capsule_native_intra_cell \
    vision_mvp.tests.test_coordpy_capsule_native_deeper \
    vision_mvp.tests.test_coordpy_scale_vs_structure \
    vision_mvp.tests.test_coordpy_cross_role_coherence \
    vision_mvp.tests.test_coordpy_cross_role_corroboration \
    vision_mvp.tests.test_coordpy_multi_service_corroboration \
    vision_mvp.tests.test_coordpy_public_api
Ran 207 tests in ~6s — OK

$ python3 -m vision_mvp.experiments.phase56_multi_service_corroboration \
    --K-auditor 4 --n-eval 10 --bank-seed 11
[phase56] gap multi_service−fifo: +1.000
[phase56] gap multi_service−cohort_buffered: +1.000
[phase56] gap multi_service−corroboration: +1.000

$ python3 -m vision_mvp.experiments.phase56_multi_service_corroboration \
    --seed-sweep --n-eval 10
seed=11: ms-fifo=+1.000  ms-cohort=+1.000  ms-corr=+1.000
seed=17: ms-fifo=+1.000  ms-cohort=+1.000  ms-corr=+1.000
seed=23: ms-fifo=+1.000  ms-cohort=+1.000  ms-corr=+1.000
seed=29: ms-fifo=+1.000  ms-cohort=+1.000  ms-corr=+1.000
seed=31: ms-fifo=+1.000  ms-cohort=+1.000  ms-corr=+1.000

$ python3 -m vision_mvp.experiments.phase56_multi_service_corroboration \
    --falsifier --n-eval 10
[phase56] gap multi_service−fifo: +0.000
```

## 10. What remains open

* **Bundle-aware decoder companion (W9-C1).** The decoder
  (`_phase31_decoder_from_handoffs`) is service-blind; `services`
  is a union over admitted payloads. A decoder that filters service
  tags by the dominant *(claim_kind, role)* signature would be a
  **second** axis of structural improvement, orthogonal to
  admission, and would be the natural attack on the W9-4 falsifier
  regime.
* **Multi-service-gold of size ≥ 3 (W9-C2).** Phase-56 has size-2
  gold; size-3 and beyond would require `top_k = |gold|` and a
  Phase-57 bench.
* **Real-LLM multi-service-gold regime (W9-C3).** Re-prompt Phase
  53's producer LLM with a multi-service event mix to surface
  multi-service-gold corroboration in real-LLM candidate streams.
  Predicted (W9-C3): multi_service improves over W8 in this regime.
* **Multi-round handoff dynamics.** The current Phase-56 bench is
  single-round (one round of producer emissions → one auditor
  decision). A multi-round version where round-1 admission decisions
  shape round-2 producer emissions (memory / cohort handling) would
  be a deeper test of capsule-native coordination.
* **Two-Mac sharded inference (still open from SDK v3.6/v3.7/v3.8/v3.9).**
  Mac 2 still offline; runbook unchanged.

---

*Theorem-forward summary: SDK v3.10 ships the smallest deterministic
multi-agent benchmark where multi-service-gold breaks single-tag
cross-role corroboration (Phase 56), the smallest interpretable
**multi-service cross-role corroboration** admission policy
(`MultiServiceCorroborationAdmissionPolicy` — buffered mode), and
the **strongest cross-regime conditional structural-win result the
programme has ever produced** (multi_service − W8 = +1.000 on
Phase 56 default; multi_service − FIFO = +1.000 on Phase 54, Phase 55
and Phase 56; 0.000 on Phase 53 synthetic; stable across 5/5 bank
seeds; falsifier regime correctly ties FIFO at 0.000). The honest
reading: the capsule layer's audit contribution is preserved, AND
the capsule layer's coordination-performance contribution now spans
**four** named regimes with cross-bank stability and a named
falsifier — clearing the strong success bar of
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1 (R-56 anchor).
The W8 multi-service-gold falsifier is now resolved by W9; the
W9-4 falsifier (decoy corroboration) is named, mechanically
verified, and will be attacked next via the W9-C1 bundle-aware
decoder companion.*
