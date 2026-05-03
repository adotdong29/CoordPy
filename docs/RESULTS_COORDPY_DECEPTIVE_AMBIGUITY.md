# CoordPy Results — bundle-contradiction-aware trust-weighted disambiguator (SDK v3.20, W19 family)

> **Milestone.** SDK v3.20 attacks the **deceptive-ambiguity wall**
> (W18-Λ-deceive, named in SDK v3.19) and the **confound wall**
> (W18-Λ-confound) in the *bundle-resolvable* case — i.e. when
> the bundle carries at least one *independent asymmetric witness*
> beyond the canonical primary disambiguator. The W19 family
> ships ``BundleContradictionDisambiguator`` — a deterministic,
> training-free witness counter that identifies a single canonical
> primary specific-tier disambiguator, counts independent
> asymmetric witnesses per admitted service tag (excluding the
> primary), and inverts or refines W18's projection when
> witnesses contradict the primary. **The first capsule-native
> multi-agent-coordination method that resolves bundle-internal
> contradiction between a deceptive primary and a witness-
> corroborated alternative — strict +1.000 over W18 on three
> R-66 sub-banks, stable across 5/5 alternate ``bank_seed``
> values, with bounded-context efficiency preserved byte-for-
> byte and full-programme regression at 555 / 555 coordpy tests.**
>
> Last touched: 2026-04-28.

## TL;DR

* **W18-Λ-deceive-extension (proved-empirical n=8 saturated × 5
  seeds + structural sketch).** On R-66-DECEIVE-NAIVE (symmetric
  round-1 corroboration; round-2 primary names DECOY ONLY;
  round-2 secondary witness names GOLD ONLY), every closed-form
  scorer in the SDK pre-W19 — including W18 itself — ties FIFO
  at ``accuracy_full = 0.000``. W18's full-disambiguator scorer
  sees positive scores on every admitted tag (the primary
  contributes decoy hits AND the secondary contributes gold
  hits; W18 concatenates both); W18 abstains; falls through to
  the empty inner W15 answer. The wall is real and structural
  for every scorer that does not distinguish the canonical
  primary from secondary witnesses.

* **W19-1 (proved-conditional + proved-empirical n=120 saturated
  across 5 seeds × 3 regimes × 8 cells).** Pairing the W18
  ``RelationalCompatibilityDisambiguator`` with the new
  ``BundleContradictionDisambiguator`` achieves
  ``accuracy_full = 1.000`` on R-66-DECEIVE-NAIVE-LOOSE
  (``T_decoder = None``) AND on R-66-DECEIVE-NAIVE-TIGHT
  (``T_decoder = 24``) AND on R-66-CONFOUND-RESOLVABLE
  (``T_decoder = None``), strictly improving over the W18
  baseline by **+1.000** on all three regimes, stable across
  **5/5** alternate ``bank_seed`` values
  (11, 17, 23, 29, 31). **First capsule-native method that
  crosses the deceptive-ambiguity wall on regimes where the
  bundle carries an independent asymmetric witness for gold.**

* **Two named falsifiers make the conditionality sharp:**
  R-66-DECEIVE-TOTAL (no asymmetric witness anywhere; W19-Λ-total:
  W19 reduces to W18 and FAILS at 0.000) and R-66-OUTSIDE-REQUIRED
  (witnesses are themselves symmetric across primary's named set
  and the complement; W19-Λ-outside: W19 abstains via
  ``W19_BRANCH_ABSTAINED_SYMMETRIC`` and ties FIFO at 0.000).

* **W19-3 (full programme regression).** On R-58 default and on
  every R-65 default bank (compat / no_compat / confound /
  deceive), W19 ties W18 byte-for-byte on the answer field. With
  ``enabled = False`` W19 reduces to W18 byte-for-byte. **All
  prior coordpy tests pass** (450 / 450 in the coordpy suite: 405
  pre-existing + 45 new W19 tests = 450 total; broader regression
  555 / 555 across the coordpy-prefix suite).

* **Bounded-context honesty preserved (W19-2).** The W19 scorer
  reads only the W18-packed bundle (which itself reads only the
  W15-packed bundle). On R-66-DECEIVE-NAIVE-TIGHT,
  ``tokens_kept_sum`` is byte-for-byte identical to W18's
  (188 / 226 tokens kept across 8 cells; same bundle, no extra
  capsule reads).

* **Honest scope.** R-66 is a *synthetic* regime; the producer
  is :class:`IdentityExtractor`. Real-LLM transfer (W19-Λ-real)
  is conjectural pending Mac-1 reachable + the LLM emitting the
  secondary-witness convention in closed-vocabulary form. The
  W19-Λ-total and W19-Λ-outside walls are real and structural
  for closed-form bundle-only scorers; the natural escape is
  outside information (W19-C-OUTSIDE, conjectural).

## 1. Result table

Pooled ``accuracy_full`` on n=8 cells per regime at
``K_auditor=12, T_auditor=256, bank_seed=11``. The full data is
in ``docs/data/phase66_cross_regime_synthetic.json``.

| Regime | Substrate FIFO | W15 attention-aware | W18 relational-compat | **W19 bundle-contradiction** | gap (W19−W18) |
| --- | ---: | ---: | ---: | ---: | ---: |
| R-66-CORROBORATED (positive anchor) | 0.000 | 0.000 | **1.000** | **1.000** | 0.000 |
| **R-66-DECEIVE-NAIVE-LOOSE** | 0.000 | 0.000 | 0.000 | **1.000** | **+1.000** |
| **R-66-DECEIVE-NAIVE-TIGHT** (`T_decoder=24`) | 0.000 | 0.000 | 0.000 | **1.000** | **+1.000** |
| **R-66-CONFOUND-RESOLVABLE** | 0.000 | 0.000 | 0.000 | **1.000** | **+1.000** |
| R-66-DECEIVE-TOTAL (W19-Λ-total) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| R-66-OUTSIDE-REQUIRED (W19-Λ-outside) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Bench-property hold-rate is 8/8 on every cell of every regime
(symmetric round-1 corroboration AND named (primary, secondary)
shape — see ``Phase66BenchPropertyTests``).

5-seed stability sweep on R-66-DECEIVE-NAIVE-LOOSE,
R-66-DECEIVE-NAIVE-TIGHT, and R-66-CONFOUND-RESOLVABLE: gap
``W19 − W18 = +1.000`` on every seed (min = mean = max = +1.000),
well above the 0.50 strong-bar threshold. Anchors:
``docs/data/phase66_seed_sweep_deceive_naive_loose.json``,
``docs/data/phase66_seed_sweep_deceive_naive_tight.json``,
``docs/data/phase66_seed_sweep_confound_resolvable.json``.

## 2. Method (W19, deterministic + closed-form + training-free)

The W19 :class:`BundleContradictionDisambiguator` is a four-stage
pipeline that wraps the W18 layer:

1. **Run inner W18.** The W18 layer (which itself wraps W15)
   produces a base answer + W15 pack stats + W18 audit
   (``compatibility`` block).
2. **Identify the canonical primary.** Walk the admitted union;
   for each specific-tier handoff, score by canonical sort
   ``(-ridx, canonical_role_match, raw_kind, source_role,
   claim_kind, payload_sha, payload)``. The canonical-role
   tiebreak (0 = match, 1 = non-match) ensures the canonical
   primary is identified even when a non-canonical role emitted
   a synonym/heuristic-rescued kind that normalises to the same
   canonical specific-tier kind.
3. **Count asymmetric witnesses per admitted tag.** For each
   service tag T in the admitted union, count specific-tier
   handoffs *other than the primary* whose tokenised payload
   mentions T (via the W18 scorer:
   :func:`_relational_compatibility_score`). Deduplicate by
   ``(source_role, claim_kind, payload_sha)`` so byte-identical
   witnesses collapse to one count.
4. **Decide the W19 branch.** Let N = W18's named set
   (positive-score tags from the full concatenated disambiguator
   text), U = admitted tag union, aw(T) = witness count for T.

   * **W19-1-inversion branch.** If ``0 < |N| < |U|`` AND
     ``max_aw(U \ N) > max_aw(N)``: invert — project to the
     high-witness tags in U \ N (the primary's evidence is
     contradicted by independent witnesses; the witnesses
     point to a different tag set).
   * **W19-1-confound branch.** If W18 abstained
     (``|N| ∈ {0, |U|}``) AND there is a unique strict-max-aw
     subset M ⊊ U with ``|M| ≥ 1``: project to M (W18 had
     symmetric/empty signal; W19 picks the witness-corroborated
     subset).
   * **W19_BRANCH_ABSTAINED_SYMMETRIC.** If W18 abstained AND
     witnesses are symmetric across U: abstain (W19-Λ-outside
     wall — closed-form bundle-only scope cannot distinguish).
   * **W19_BRANCH_PRIMARY_TRUSTED.** Otherwise fall through to
     W18's answer (W19-3 backward-compat path; also the W19-Λ-
     total path when no witnesses anywhere).

The implementation lives in
``vision_mvp/coordpy/team_coord.py``:

* :class:`BundleContradictionDisambiguator` — the W19 pipeline.
* :class:`W19TrustResult` — per-decode audit record carrying
  the projected answer, the W18 fall-through answer, the
  primary's payload, the per-tag W18 scores, the per-tag
  witness counts, and the W19 branch.
* :func:`_w19_canonical_primary_index` — closed-form
  deterministic primary identifier with canonical-role-for-kind
  tiebreak.
* :func:`_w19_witness_counts` — closed-form deterministic
  witness counter excluding the primary.
* :data:`_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND` — closed-
  vocabulary canonical-role-for-kind table for the incident-
  triage benchmark family. Hardcoded to avoid a layering
  inversion (team_coord is a peer of incident_triage, not a
  dependent).
* :data:`W19_SYMMETRIC_NOISE_KINDS` — the round-1 generic-
  noise kinds explicitly excluded from witness counting.
* :data:`W19_BRANCH_*` — closed-vocabulary branch labels.

## 3. Bench (R-66, deterministic + closed-vocabulary + named)

The Phase-66 bench (``vision_mvp/experiments/phase66_deceptive_
ambiguity.py``) ships five sub-banks plus a cross-regime synthetic
summary:

* **R-66-CORROBORATED.** Symmetric round-1 corroboration; round-2
  primary AND secondary both name gold ONLY. Both W18 (via
  primary alone) AND W19 (via primary + secondary consistency)
  recover gold; W18 = W19 = 1.000. The W19-3 backward-compat
  anchor on the deception-resolvable shape.
* **R-66-DECEIVE-NAIVE.** Same round-1; round-2 primary names
  DECOY ONLY (``relation=decoy_decoy_*``); secondary names GOLD
  ONLY (``relation=A_B_*`` under a synonym kind from monitor).
  W18 abstains on the full-set hit; W19 fires the confound-
  resolved branch via aw(gold) > aw(decoy) and projects to gold.
  **+1.000 strict separation** at both ``T_decoder = None``
  (loose) and ``T_decoder = 24`` (tight).
* **R-66-CONFOUND-RESOLVABLE.** Same round-1; round-2 primary
  names ALL three (``relation=A_B_decoy_*``); secondary names
  GOLD ONLY. W18 abstains on the full-set hit; W19 picks the
  strict-max-aw subset = {gold} via the confound-resolved
  branch. **+1.000 strict separation.**
* **R-66-DECEIVE-TOTAL.** Same round-1; round-2 primary names
  DECOY ONLY; *no* secondary witness anywhere. The bundle is
  exhausted of asymmetric signal; W19 reduces to W18 and fails.
  **W19-Λ-total falsifier.**
* **R-66-OUTSIDE-REQUIRED.** Same round-1; round-2 primary names
  DECOY ONLY; secondary names ALL three (symmetric witnesses).
  W19 abstains via ``W19_BRANCH_ABSTAINED_SYMMETRIC``.
  **W19-Λ-outside falsifier.**

Each scenario family (deadlock / pool_exhaustion / disk_fill /
slow_query_cascade) is replicated `n_replicates=2` for default
n=8 cells. The bank generator is RNG-deterministic given
``bank_seed``.

The Phase-66 secondary-witness routing extension
(:data:`_P66_SECONDARY_ROUTES`) augments the canonical
incident-triage subscription table with ``monitor → DEADLOCK_DETECTED
/ POOL_EXHAUSTED / DISK_FILL_DETECTED / SLOW_QUERY_DETECTED →
ROLE_AUDITOR``. The augmentation is Phase-66-local; the canonical
table is left byte-for-byte unchanged. The W12 / W13 layered
normaliser resolves the synonym kinds to their canonical specific-
tier kinds before the W19 scorer reads them.

## 4. Bounded-context honesty (W19-2)

On R-66-DECEIVE-NAIVE-TIGHT (``T_decoder = 24``), per
``docs/data/phase66_cross_regime_synthetic.json::r66_deceive_naive_tight.pack_stats_summary``:

| Strategy | tokens_input_sum | tokens_kept_sum | n_handoffs_kept / admitted |
| --- | ---: | ---: | ---: |
| W15 attention-aware (standalone) | 226 | 184 | 66 / 80 |
| W18 relational-compat | 226 | **188** | 66 / 80 |
| W19 bundle-contradiction | 226 | **188** | 66 / 80 |

W19's ``tokens_kept_sum`` is **byte-for-byte identical** to W18's;
W19 reads the same bundle W18 reads. The W15 pack stats are
unchanged byte-for-byte across the W18 → W19 layer addition.
Token-budget honesty preserved (mechanically verified by
``Phase66TokenEfficiencyTests::test_w19_does_not_inflate_tokens_kept``).

The strict token-budget floor (``tokens_kept ≤ T_decoder × n_cells
= 24 × 8 = 192``) is satisfied: 188 ≤ 192. ✓

## 5. Cross-regime backward-compat (W19-3)

On R-58 default (delayed-causal-evidence) and on every R-65
default bank (compat / no_compat / confound / deceive), W19
ties W18 byte-for-byte on the answer field. Mechanically verified
by ``Phase66BackwardCompatTests``:

* ``test_w19_matches_w18_on_phase58_default``: 4/4 deadlock /
  pool / disk / slow_query scenarios — W19 services = W18
  services; W19 root_cause = W18 root_cause.
* ``test_w19_matches_w18_on_phase65_default_banks``: 16 cells
  (4 sub-banks × 4 families) — W19 = W18 byte-for-byte.

Audit T-1..T-7 holds on every cell of every capsule strategy
on every (bank, ``T_decoder``, bank_seed) cell. Mechanically
verified by ``Phase66DefaultConfigTests::test_w19_audit_OK_on_every_cell``.

Full programme regression: 405 pre-existing coordpy tests + 45
new W19 tests = **450 / 450** in the targeted coordpy suites;
**555 / 555** across all `test_coordpy_*.py` files. The W19
surface is purely additive on top of W18; the SDK v3.19 runtime
contract is byte-for-byte unchanged.

## 6. Theory (the W19 family)

* **W19-Λ-deceive-extension** (proved-empirical + structural
  sketch). W18-Λ-deceive extends to R-66-DECEIVE-NAIVE for every
  closed-form bundle-relational scorer that *trusts* its
  concatenated disambiguator text. The wall is real and
  structural for every scorer pre-W19; W18's behaviour is
  *abstention* (full-set hit) rather than R-65-DECEIVE's
  strict-asymmetric pick of decoy — both yield 0.000.
* **W19-1** (proved-conditional + proved-empirical n=120
  saturated). The W19 method strictly improves over W18 by
  +1.000 on R-66-DECEIVE-NAIVE-LOOSE, R-66-DECEIVE-NAIVE-TIGHT,
  AND R-66-CONFOUND-RESOLVABLE, stable across 5/5 alternate
  bank seeds.
* **W19-2** (proved by inspection + mechanically-checked). W19
  determinism + closed-form correctness; bounded-context
  honesty (``tokens_kept_sum`` byte-for-byte identical to W18's).
* **W19-3** (proved-empirical full programme regression).
  Backward-compat with R-54..R-65 byte-for-byte; with
  ``enabled = False`` the W19 method reduces to W18 byte-for-byte.
* **W19-Λ-total / -outside** (proved-empirical n=8 saturated
  each). Two named structural limit regimes where W19 ties FIFO
  by construction. The W19-Λ-total wall (no asymmetric witness
  anywhere) bounds the bundle-only closed-form scope; the
  W19-Λ-outside wall (symmetric witnesses) bounds the same scope
  even when witnesses exist.
* **W19-C-LEARNED, W19-C-OUTSIDE, W19-Λ-real, W19-C-CROSS-BENCH
  (conjectural).** Named extension axes. W19-C-OUTSIDE is the
  natural escape from BOTH falsifier walls — outside-information
  axis (service-graph topology, prior reliability scores,
  cross-incident historical evidence).

## 7. Defensible thesis after SDK v3.20

The synthetic→real-LLM-and-bounded-context transfer story now
has **eleven layers + two named structural walls + five named
falsifier regimes** (W18-Λ-deceive's bundle-only scope is now
*partially* discharged by W19-1; the *complete* bundle-only
discharge requires both W19-Λ-total and W19-Λ-outside to also be
broken, which closed-form bundle-only scorers cannot do —
escape requires outside information).

Concretely, after SDK v3.20:

1. The first capsule-native multi-agent-coordination method
   has crossed the **deceptive-ambiguity wall** on regimes
   where the bundle carries an independent asymmetric witness
   for gold (R-66-DECEIVE-NAIVE / R-66-CONFOUND-RESOLVABLE).
2. The deeper wall — adversarial-ambiguity *without* any
   independent witness in the bundle (W19-Λ-total) AND
   adversarial-ambiguity with *symmetric* witnesses
   (W19-Λ-outside) — is *named* but *not* broken; the named
   research move beyond it (W19-C-OUTSIDE — outside-information
   axis) is conjectural.
3. The original Context-Zero thesis — *per-agent minimum-
   sufficient context for multi-agent teams* — gains its
   **first capsule-native method to resolve bundle-internal
   contradiction between primary disambiguator and witnesses**.
4. The thesis is materially stronger AND the next research
   frontier is materially clearer: outside information is
   genuinely necessary to escape the bundle-only walls (a
   structural result, not a method gap).

## 8. Honest scope (what we did NOT do)

* The R-66 advance is **synthetic only**. Real-LLM transfer is
  W19-Λ-real, conjectural pending Mac-1 reachable + the LLM's
  secondary-witness emission style observed.
* The W19-Λ-total wall remains real: when the bundle has no
  asymmetric witness, no bundle-only closed-form scorer can
  escape. The natural escape is outside information
  (W19-C-OUTSIDE, conjectural).
* The W19-Λ-outside wall remains real: when witnesses are
  symmetric, no bundle-only closed-form scorer can prefer one
  side. Same escape.
* "Multi-agent context solved" still requires resolving every
  named limit theorem on every axis. SDK v3.20 closes one more
  axis (W18-Λ-deceive bundle-resolvable case); the deeper
  scopes (W19-Λ-total / W19-Λ-outside / W19-Λ-real /
  W19-C-CROSS-BENCH) remain conjectural.
* The relational-mention closure is a **closed-vocabulary
  contract** the synthetic bench enforces. Free-form natural-
  language witnesses fall outside the W19 exact-match layer
  by construction. The natural extension is W19-C-LEARNED
  (conjectural).
* No transformer attention is read; no learned model is
  trained; no embedding lookup is performed. The W19 method is
  *deterministic, training-free, closed-form*. The
  ``W19_BRANCH_*`` audit labels expose the precise branch the
  scorer fired on every cell.

## 9. Reproducibility

```
# R-66-CORROBORATED (W19-3 + W18 ratification anchor):
python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \
    --bank corroborated --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --out -

# R-66-DECEIVE-NAIVE-LOOSE (W19-1 deceive anchor):
python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \
    --bank deceive_naive --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --out -

# R-66-DECEIVE-NAIVE-TIGHT (W19-1 + W15 composition):
python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \
    --bank deceive_naive --decoder-budget 24 \
    --K-auditor 12 --n-eval 8 --out -

# R-66-CONFOUND-RESOLVABLE (W19-1 confound anchor):
python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \
    --bank confound_resolvable --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --out -

# R-66-DECEIVE-TOTAL (W19-Λ-total falsifier):
python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \
    --bank deceive_total --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --out -

# R-66-OUTSIDE-REQUIRED (W19-Λ-outside falsifier):
python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \
    --bank outside_required --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --out -

# Cross-regime synthetic summary:
python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \
    --cross-regime-synthetic --K-auditor 12 --n-eval 8 --out -

# 5-seed stability sweep:
python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \
    --bank deceive_naive --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --seed-sweep --out -
```

Tests:

```
python3 -m pytest vision_mvp/tests/test_coordpy_bundle_contradiction.py
```

Cross-regime regression on the coordpy suite:

```
python3 -m pytest vision_mvp/tests/test_coordpy_*.py
```

Anchors:

* ``vision_mvp/coordpy/team_coord.py`` —
  :class:`BundleContradictionDisambiguator`,
  :class:`W19TrustResult`,
  :func:`_w19_canonical_primary_index`,
  :func:`_w19_witness_counts`,
  :data:`_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND`,
  :data:`W19_SYMMETRIC_NOISE_KINDS`,
  :data:`W19_BRANCH_PRIMARY_TRUSTED`,
  :data:`W19_BRANCH_INVERSION`,
  :data:`W19_BRANCH_CONFOUND_RESOLVED`,
  :data:`W19_BRANCH_ABSTAINED_NO_SIGNAL`,
  :data:`W19_BRANCH_ABSTAINED_SYMMETRIC`,
  :data:`W19_BRANCH_DISABLED`,
  :data:`W19_ALL_BRANCHES`.
* ``vision_mvp/experiments/phase66_deceptive_ambiguity.py`` —
  :func:`build_phase66_bank`,
  :func:`run_phase66`,
  :func:`run_phase66_seed_stability_sweep`,
  :func:`run_cross_regime_synthetic`,
  :data:`_P66_SECONDARY_ROUTES`,
  :data:`_BANK_EXPECTED_SHAPE`.
* ``vision_mvp/tests/test_coordpy_bundle_contradiction.py`` —
  ``W19PrimaryIndexTests``, ``W19WitnessCountsTests``,
  ``W19DecoderUnitTests``, ``Phase66BenchPropertyTests``,
  ``Phase66DefaultConfigTests``, ``Phase66SeedStabilityTests``,
  ``Phase66FalsifierTests``, ``Phase66BackwardCompatTests``,
  ``Phase66TokenEfficiencyTests``,
  ``Phase66CrossRegimeSyntheticTests``.
* ``docs/data/phase66_cross_regime_synthetic.json`` —
  full cross-regime data file.
* ``docs/data/phase66_seed_sweep_*.json`` — 5-seed stability data.
* ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` § 1.1 bar 16,
  § 2.15 R-66 ingredients, canonical phrasing for SDK v3.20.
* ``docs/THEOREM_REGISTRY.md`` W19 family.
* ``docs/HOW_NOT_TO_OVERSTATE.md`` — W19 honest-scope claims.
