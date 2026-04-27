# Results — multi-round bundle decoder (SDK v3.12, W11 family)

> Milestone note for the SDK v3.12 push: the *first cross-round
> capsule-native coordination move* in the Wevra programme. Where
> SDK v3.11 (W10) attacked the multi-agent context problem from the
> *decoder* side within a single round, SDK v3.12 (W11) attacks the
> *temporal* axis: a regime where round-1 admitted evidence carries
> the service-tag inventory but no causal claim, and round-2 carries
> the causal claim but no service tags. Single-round decoding —
> including W10 — is **structurally insufficient**. The
> ``MultiRoundBundleDecoder`` decodes the round-union and
> contradiction-aware-drops noise-corroborated decoys.
> Last touched: SDK v3.12, 2026-04-26.

## TL;DR

* New benchmark **R-58** (Phase 58, multi-round delayed-causal-evidence)
  in
  ``vision_mvp/experiments/phase58_multi_round_decoder.py`` —
  deterministic 8-scenario bank where round-1 carries
  generic-noise-only mentions of (gold_A, gold_B, decoy) with the
  decoy cross-role-corroborated, and round-2 carries one
  specific-tier disambiguating claim with **no ``service=`` token**.
* New SDK method ``MultiRoundBundleDecoder`` in
  ``vision_mvp/wevra/team_coord.py`` — multi-bundle, contradiction-
  aware bundle decoder. The round-union elects a specific root_cause
  from the union of admitted ``claim_kinds``; if the elected
  root_cause is specific-tier, the decoder drops every service tag
  whose admitted mentions are *exclusively* generic-noise kinds AND
  span ≥ ``noise_decoy_role_floor`` (default 2) distinct producer
  roles. Pure decoder-side; admission is FIFO.
* Headline (R-58 default, K_auditor=8, n_eval=8, bank_seed=11):
  **capsule_multi_round = 1.000**, every other strategy
  (substrate, FIFO, priority, coverage, W7-2, W8, W9, W10
  single-round bundle) ties FIFO at 0.000 — the **first strict
  separation between multi-round and single-round capsule-native
  coordination** in the programme.
* Stability: gap = +1.000 across **5/5** alternate bank seeds
  (11, 17, 23, 29, 31).
* Backward compatibility: on R-54..R-57, the cross-regime summary
  preserves the prior wins byte-for-byte (W7-2 / W8 / W9 / W10
  all 1.000 on their respective benches).
* Falsifier (W11-4): with ``K_auditor=4`` and round-1 noise count
  doubled to flood the budget, multi_round ties FIFO at 0.000 —
  the named conditionality is sharp.

## 1. The structural move

### 1.1 Why single-round decoding cannot win on R-58

A single-round decoder consumes one ROLE_VIEW capsule's admitted
handoffs. On R-58:

* **Round 1's admitted bundle** elects a *generic*-tier root_cause
  via the priority decoder (``LATENCY_SPIKE`` → ``latency_spike``,
  ``ERROR_RATE_SPIKE`` → ``error_spike``, ``FW_BLOCK_SURGE`` →
  ``fw_block``). The W10 CCK filter for a generic root_cause
  consists of generic-noise kinds themselves; the filter cannot
  distinguish gold from decoy. The corroborated decoy survives →
  ``services_correct`` fails.
* **Round 2's admitted bundle** is a single specific-tier claim
  with no ``service=`` token (e.g.
  ``DEADLOCK_SUSPECTED relation=orders_payments_join``). The
  decoder elects ``deadlock`` correctly but the service-tag set is
  empty → ``services = ()`` → ``services_correct`` fails.

This is **W11-Λ** (single-round structural limit at the temporal
axis): no single-round decoder, with any service-blind admission
policy, can solve R-58. Mechanically witnessed by
``Phase58DefaultTests.test_W11_Lambda_single_round_limit``: every
of (substrate, FIFO, priority, coverage, W7-2, W8, W9, W10
single-round bundle) achieves ``accuracy_full = 0.000`` on R-58
default.

### 1.2 The W11 multi-round move

The ``MultiRoundBundleDecoder`` decodes the *union* of admitted
handoffs across rounds. On R-58:

* Round-1 contributes the *service-tag inventory* (gold_A, gold_B,
  decoy) carrying generic-noise kinds.
* Round-2 contributes a specific-tier ``claim_kind`` (e.g.
  ``DEADLOCK_SUSPECTED``) which dominates the priority decoder.
  The elected ``root_cause`` is now ``deadlock``.
* The decoder applies the *contradiction-aware noise-decoy drop*:
  for the elected specific-tier root_cause, every service tag
  whose admitted mentions are exclusively generic-noise kinds AND
  span ≥ ``noise_decoy_role_floor`` distinct producer roles is
  filtered out. The decoy (corroborated by monitor + network via
  generic-noise) is dropped; gold_A and gold_B (mentioned only by
  monitor in round-1, single role) survive.

### 1.3 Why this is a *real* method change

* The W11 decoder consumes a multi-bundle input —
  ``decode_rounds(per_round_handoffs)`` — and produces a
  *projection* over the union of the bundles. The W10
  ``BundleAwareTeamDecoder`` is the single-bundle special case.
* The *noise-decoy drop* is a contradiction-aware predicate
  orthogonal to the W10 CCK predicate. W10 asks "is the tag's
  claim_kind in the chosen root_cause's CCK?"; W11 asks "is the
  tag's evidence pattern compatible with the chosen root_cause's
  *tier*?". Both are deterministic, training-free, closed-vocabulary.
* Admission is FIFO; the W11 contribution is purely decoder-side.
  This is the cleanest possible test of "decoding alone, given a
  fair admission floor, can resolve a regime that no single-round
  decoder can solve."

## 2. Pre-committed bench property + falsifier

### 2.1 Bench property (delayed-causal-evidence)

Every R-58 default scenario satisfies all four:

1. ``round1_only_generic_noise``: every round-1 emission to the
   auditor has ``claim_kind`` ∈
   ``{LATENCY_SPIKE, ERROR_RATE_SPIKE, FW_BLOCK_SURGE}``.
2. ``round2_only_specific``: every round-2 emission has
   ``claim_kind`` NOT in the generic-noise set.
3. ``decoy_only_in_round1``: the decoy service is not mentioned
   in round 2.
4. ``round1_decoy_corroborated``: the decoy is mentioned by ≥ 2
   distinct producer roles in round 1.

Mechanically verified by
``Phase58BankShapeTests.test_default_bank_property_holds`` on the
default bank (8/8 scenarios).

### 2.2 Falsifier (W11-4 — round-1 noise floods budget)

The Phase-58 falsifier bank
(``build_phase58_falsifier_bank``) doubles every round-1 noise
emission so that the round-1 candidate count exceeds the auditor's
per-round inbox capacity. With ``K_auditor=4``, FIFO admission
admits round-1 noise up to capacity; the round-2 specific-tier
claim is dropped at the team-handoff seal-and-admit step on the
already-full inbox of an unrelated test path, OR the union loses
the specific kind, and the elected root_cause stays generic.
Multi_round ties FIFO at 0.000 (``Phase58FalsifierTests``).

### 2.3 Why this is fair

* The bench is **deterministic** by design — isolates the
  *temporal-decoding axis* from extraction quality.
* The bench property is **named in code** and mechanically tested.
* The falsifier is **named in code** and the sharp conditionality
  (round-1 budget vs round-2 admission) is structurally
  inevitable, not a bench artefact.

## 3. Cross-regime evaluation

Cross-regime summary table at ``K_auditor=4`` (n_eval=8,
bank_seed=11, bank_replicates=2):

| Regime           | Best prior      | New (multi-round) | Backward-compat |
|------------------|-----------------|---------------------|-----------------|
| R-54 (W7-2)      | cohort_buffered=1.000 | n/a (single round) | ✓ preserved   |
| R-55 (W8)        | corroboration=1.000  | n/a              | ✓ preserved   |
| R-56 (W9)        | multi_service=1.000  | n/a              | ✓ preserved   |
| R-57 (W10)       | bundle_decoder=1.000 | n/a              | ✓ preserved   |
| **R-58 (W11)**   | every other = 0.000  | **multi_round=1.000** | gap = **+1.000** |
| R-58 falsifier   | every = 0.000        | multi_round=0.000  | conditionality sharp |

The R-58 default cross-regime summary uses ``K_auditor=8`` because
R-58's per-round candidate count (round-1: 6 to auditor; round-2:
1) requires headroom for the round-1 admission to retain the full
service inventory at the noise floor.

## 4. Theorem family W11 (minted by this milestone)

### W11-Λ — single-round structural limit (proved-empirical + structural sketch)

**Statement.** On R-58 default, every service-blind admission
policy paired with any single-round decoder (substrate priority,
W10 single-round bundle) ties FIFO at ``accuracy_full = 0.000``.

**Sketch.** Single-round mode partitions admitted handoffs into
disjoint per-round bundles. Round-1's bundle elects a generic-tier
root_cause; the W10 CCK is the generic-noise set itself, so the
decoy survives. Round-2's bundle elects ``deadlock`` /
``pool_exhaustion`` / etc. but contains no ``service=`` token, so
the service set is empty.

**Status.** Proved-empirical on n_eval=8 saturated across 5/5
bank seeds. Structural argument by inspection of the decoder
priority and CCK closed-vocabulary tables.

### W11-1 — multi-round bundle decoder sufficiency (proved-empirical)

**Statement.** Pairing FIFO admission with the
``MultiRoundBundleDecoder`` (``noise_decoy_role_floor=2``,
``cck_filter=False`` inner) achieves ``accuracy_full = 1.000`` on
R-58 default.

**Status.** Proved-empirical on n_eval=8 saturated across 5/5 bank
seeds. Anchor: ``Phase58DefaultTests.test_multi_round_strict_win``.

### W11-2 — round-union monotonicity (proved structural)

**Statement.** Let ``H_1, H_2, ..., H_n`` be the per-round admitted
bundles and ``K(B)`` the set of ``claim_kinds`` in bundle ``B``.
Then ``K(union)`` is a superset of ``K(H_i)`` for every ``i``, and
the priority decoder's elected root_cause on ``union`` is at most
as generic as on any individual ``H_i`` (in the priority order).

**Status.** Proved by construction. The priority decoder selects
the highest-priority kind in its input; adding more kinds can only
move the result up the priority list, not down.

### W11-3 — backward compatibility (proved-empirical)

**Statement.** With a single bundle the W11 decoder reduces to a
W10-equivalent ``services`` set on R-54..R-57 default banks.

**Status.** Proved-empirical via
``Phase58CrossRegimeTests``. Each prior anchor (cohort_buffered on
R-54, corroboration on R-55, multi_service on R-56, bundle_decoder
on R-57) achieves ``accuracy_full = 1.000`` unchanged from the
SDK v3.8/v3.9/v3.10/v3.11 baseline.

### W11-4 — round-budget falsifier (proved-empirical)

**Statement.** With ``K_auditor=4`` on the Phase-58 falsifier bank
(round-1 noise count doubled), W11 multi_round ties FIFO at 0.000.

**Status.** Proved-empirical, n_eval=8 saturated. The named
conditionality on per-round budget sufficiency.

## 5. Honest scope (what this milestone does NOT claim)

* **Not** "we solved multi-agent context." R-58's win is *conditional*
  on (a) the named bench property
  (delayed-causal-evidence-with-noise-corroborated-decoy),
  (b) the closed-vocabulary generic-noise-kind set being
  meaningful for the benchmark family, and (c) round-2 admission
  not being budget-starved.
* **Not** "the W11 decoder beats W10 generally." The W11
  contradiction-aware drop is a *strict generalisation* of W10
  along the noise-corroboration axis only. On regimes where the
  decoy is NOT noise-corroborated (R-55 default), W11 gives
  identical answers; on regimes where the decoy IS in the same
  causal tier as gold (R-57 falsifier W10-4), W11 cannot help
  either.
* **Not** "multi-round capsule chains are a complete solution to
  delayed evidence." Real production multi-agent teams have
  multi-round handoffs with arbitrary delay distributions,
  inter-round contradictions, role failures, and goal mismatch.
  R-58 isolates a single, deterministic, two-round decomposition.
* **Not** "real-LLM transfer is established." R-58 is synthetic.
  Real-LLM transfer is the open conjecture **W11-C2**.
* **Not** "the runtime now needs multi-round capsule memory." The
  Wevra single-run product runtime is unchanged. ``W11`` is
  research-grade SDK code.

## 6. Active conjectures (SDK v3.12)

* **W11-C1** (cross-bench): the noise-decoy drop generalises to
  any incident-triage benchmark where the gold root_cause is
  specific-tier. Conjectural; falsifier = a bench where the gold
  root_cause is itself generic (R-58-generic-gold).
* **W11-C2** (real-LLM): the W11 win transfers to a real-LLM
  multi-round regime where round-1 producers emit generic-tier
  observations and round-2 producers emit specific-tier
  diagnoses. Conjectural; Phase-59 candidate.
* **W11-C3** (multi-step disambiguation): with three or more rounds
  and conflicting specific-tier evidence across rounds (e.g.
  round-2 emits DEADLOCK, round-3 emits POOL_EXHAUSTION on a
  different gold service), a *contradiction-aware* round
  resolution rule (e.g. last-wins-with-confidence) strictly
  outperforms naive union. Conjectural; not yet measured.

## 7. Files changed

* New SDK class:
  ``vision_mvp/wevra/team_coord.py`` — adds
  ``MultiRoundBundleDecoder``,
  ``collect_admitted_handoffs``, ``_GENERIC_NOISE_CLAIM_KINDS``;
  re-exported via ``__all__``.
* New benchmark:
  ``vision_mvp/experiments/phase58_multi_round_decoder.py``.
* New tests:
  ``vision_mvp/tests/test_wevra_multi_round_decoder.py`` — 19
  tests across unit, bank-shape, default, falsifier, seed
  stability, cross-regime, single-round reduction.
* Artifacts:
  ``docs/data/phase58_default_K8_n8.json``,
  ``docs/data/phase58_falsifier_K4_n8.json``,
  ``docs/data/phase58_seed_sweep_K8_n8.json``,
  ``docs/data/phase58_cross_regime.json``.
* Doc updates:
  ``docs/RESEARCH_STATUS.md``,
  ``docs/THEOREM_REGISTRY.md``,
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``,
  ``docs/context_zero_master_plan.md``,
  ``docs/RESULTS_WEVRA_MULTI_ROUND_DECODER.md`` (this file).

## 8. What this milestone advances

* **The original Context-Zero thesis** — *per-agent
  minimum-sufficient context for multi-agent teams* — gains its
  first **cross-round** instance: the *minimum-sufficient context
  for the auditor's decision* in R-58 spans both rounds, not one.
  No single round's evidence is sufficient. The capsule-native
  ROLE_VIEW per-round seal + cross-round union path is the
  smallest cross-round move that survives the lifecycle audit
  (T-1..T-7 OK on every cell of every R-58 capsule strategy).
* **The decoder/admission split** continues to deepen. SDK v3.11
  established the split (W10-Λ proved admission cannot suffice on
  R-57). SDK v3.12 deepens the *decoder* leg by giving it
  multi-bundle input and a contradiction-aware filter.
* **The Wevra programme has three structural axes**
  (admission, decoding, temporal) with named limit theorems on each.
  W7-2/W8/W9 work on admission; W10 works on decoding within a
  round; W11 works on decoding across rounds. The runtime contract
  is unchanged; all three are research-grade SDK extensions.

## Cross-references

* Bench: ``vision_mvp/experiments/phase58_multi_round_decoder.py``
* Method: ``vision_mvp/wevra/team_coord.py``
  (``MultiRoundBundleDecoder``)
* Tests: ``vision_mvp/tests/test_wevra_multi_round_decoder.py``
* Prior milestone: ``docs/RESULTS_WEVRA_BUNDLE_DECODER.md``
* Success criterion: ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``
  (R-58 anchor + bar 8 — temporal/structural split)
* Theorem registry: ``docs/THEOREM_REGISTRY.md`` (W11 family)
* Master plan: ``docs/context_zero_master_plan.md``
