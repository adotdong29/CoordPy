# SDK v3.5 — capsule-native multi-agent team coordination

> Milestone note for the SDK v3.5 research slice
> ([`vision_mvp/wevra/team_coord.py`](../vision_mvp/wevra/team_coord.py)
> + [`vision_mvp/wevra/team_policy.py`](../vision_mvp/wevra/team_policy.py)
> + [`vision_mvp/experiments/phase52_team_coord.py`](../vision_mvp/experiments/phase52_team_coord.py)).
> Last touched: 2026-04-26.

## What changed

The capsule layer was previously load-bearing inside *one* Wevra
run (run boundary → cell → parser axis → LLM byte boundary, the
W3 family). SDK v3.5 makes capsules load-bearing **between agents
in a team**. The original Context-Zero thesis — *per-agent
minimum-sufficient context for multi-agent teams* — now has a
capsule-native research slice that exercises the load-bearing
abstraction at the team boundary, not just the single-run boundary.

Three closed-vocabulary capsule kinds are added:

* **TEAM_HANDOFF** — capsule-native multi-agent handoff. Distinct
  from the substrate-adapter HANDOFF (which lifts a Phase-31
  ``TypedHandoff``); a TEAM_HANDOFF is born as a capsule and has
  no substrate twin. Payload: `(source_role, to_role, claim_kind,
  payload, round, payload_sha256, n_tokens)`. Identity is
  content-addressed; byte-identical handoffs collapse (Capsule
  Contract C1).

* **ROLE_VIEW** — per-role admitted view of a coordination round.
  Parents are the CIDs of admitted TEAM_HANDOFF capsules;
  ``max_parents`` is the role-local cardinality cap $K_r$;
  ``max_tokens`` is the role-local token cap $T_r$. A ROLE_VIEW
  *is* the role's local-view-under-budget object — the W4 theorems
  state things about ROLE_VIEW capsules, not about agent-local
  scratchpads.

* **TEAM_DECISION** — team-level decision. Parents: the ROLE_VIEW
  capsules the deciding role consulted. Payload is the structured
  team answer.

A `TeamCoordinator` orchestrates one coordination round
end-to-end; a `TeamLifecycleAudit` mechanically verifies T-1..T-7
on every finished round.

## What this gives the original thesis

Before SDK v3.5, capsule-native execution and *multi-agent team
coordination* were two stories that did not directly meet. The
substrate (`vision_mvp/core/role_handoff.py`) carried the team
story; the capsule layer carried the single-run story. SDK v3.5
makes them one story. Specifically:

* **Capsules as the actual coordination object**, not a description
  of one. The TEAM_HANDOFF capsule is what crosses role boundaries
  in the new path; a substrate ``TypedHandoff`` is no longer
  required.
* **Role-local capsule view under explicit budget.** ROLE_VIEW
  records the role's admission decisions structurally; the budget
  is enforced by the capsule layer, not by string-counting in the
  inbox.
* **Lifecycle-audited team coordination.** The same audit
  technique that proves the run-boundary lifecycle (W3-40 / W3-45)
  proves the team-boundary lifecycle (W4-1).

## Theorems (W4 family)

* **W4-1 — Team-lifecycle audit soundness.** *Proved +
  mechanically-checked.* The runtime audit returns OK iff T-1..T-7
  hold on the ledger. Anchor:
  ``team_coord.audit_team_lifecycle``;
  ``TeamLifecycleAuditTests``.
* **W4-2 — Coverage-implies-correctness.** *Proved-conditional.*
  Under (faithful decoder, sound admission), if the role view
  admits a superset of the scenario's causal claims, the team
  decision is correct. Anchor:
  ``test_w4_2_coverage_implies_correct``.
* **W4-3 — Local-view limitation.** *Proved-negative.* A per-role
  budget $K_r$ strictly below the role's causal-share floor on a
  scenario admits the wrong answer regardless of admission policy.
  Anchor: ``test_w4_3_local_view_limitation_at_tight_budget``;
  budget-sweep evidence in ``run_phase52_budget_sweep``.
* **W4-C1 — Learned-policy advantage at matched budgets.**
  *Empirical (positive on default config); conjectural otherwise.*
  See `phase52_team_coord.run_phase52`.

Full statements + proofs:
[`docs/CAPSULE_TEAM_FORMALISM.md`](CAPSULE_TEAM_FORMALISM.md).

## Benchmark — Phase 52

The reference benchmark instantiates `TeamCoordinator` on the
Phase-31 incident-triage bank under controlled noise and compares
five strategies head-to-head:

| strategy            | description                                     |
| ------------------- | ----------------------------------------------- |
| `substrate`         | Phase-31 typed-handoff baseline (no capsule layer) |
| `capsule_fifo`      | capsule-native + FIFO admission policy           |
| `capsule_priority`  | capsule-native + claim-priority admission        |
| `capsule_coverage`  | capsule-native + coverage-guided admission       |
| `capsule_learned`   | capsule-native + learned per-role admission policy |

### Cross-seed result (W4-C1 — honest empirical reading)

Eval partition: 31 scenarios (90-scenario expanded bank,
2/3 train / 1/3 eval split). Configuration:
``K_auditor=8``, ``T_auditor=256``, noise
``(drop=0.10, spurious=0.30, mislabel=0.05)``,
``train_epochs=200``, ``train_lr=0.5``,
``PYTHONHASHSEED=0`` (the noise wrapper's per-call RNG seed
depends on ``hash(scenario_id)``, so a fixed Python hash seed is
required for cross-run reproducibility).

The headline contrast is **learned vs coverage-guided**
(coverage-guided is the strongest fixed baseline on this bench).
Aggregated over ``train_seed ∈ {0, 1, …, 11}`` (12 seeds):

| metric (learned − coverage)               | mean    | seeds with gap > 0 |
| ----------------------------------------- | ------- | ------------------- |
| ``accuracy_full``                         | +0.054  | 11 / 12             |
| ``accuracy_root_cause``                   | +0.032  | 8 / 12              |
| ``accuracy_root_cause`` (median)          | +0.049  | —                   |
| ``mean_n_admitted_auditor`` saved         | +1.26 handoffs | **12 / 12**   |

Reading:

* **Budget-efficiency dominance is robust.** On every train seed
  the learned policy admits ~1.26 fewer handoffs per scenario
  than coverage-guided at matched team-level budget (12/12
  seeds, deterministic in direction; magnitude varies $\pm 0.5$).
  This is the load-bearing positive empirical signal of W4-C1.
* **Accuracy improvement is most-of-the-time, not all-the-time.**
  ``accuracy_full`` improves on 11/12 seeds (mean $+0.054$);
  ``accuracy_root_cause`` improves on 8/12 seeds (mean $+0.032$).
  The strongest single-seed result reaches gap_full=$+0.129$ and
  gap_root_cause=$+0.097$; the weakest single-seed result is
  gap_root_cause=$-0.097$ (one outlier where the learned policy
  underperforms). The conjecture's status is therefore
  *empirical-positive on the mean, not strict per-seed.*
* **Every capsule strategy reports
  ``team_lifecycle_audit.verdict == "OK"``** on every eval
  scenario (``audit_ok_rate = 1.000``) — the W4-1 mechanical
  result. The substrate baseline is excluded from the audit
  because it does not emit team-level capsules.
* **Sensitivity to noise level.** At higher noise
  (``spurious_prob = 0.50``), coverage-guided beats the learned
  policy on root_cause (mean $-0.089$ across 8 seeds) — i.e. the
  learned-policy advantage *does not survive* heavier noise. This
  is honest evidence that the W4-C1 advantage is conditional on
  the noise regime; the coverage-guided baseline is more robust
  out-of-distribution.

### Single-seed sample (illustration only)

For comparison with prior results, here is **one** representative
single-seed run (``train_seed=8``, ``PYTHONHASHSEED=0``):

| strategy            | accuracy_full | accuracy_root_cause | mean_n_admitted_auditor | audit_ok_rate |
| ------------------- | ------------- | ------------------- | ----------------------- | ------------- |
| substrate           | 0.0322        | 0.2581              | 7.871                   | n/a           |
| capsule_fifo        | 0.0322        | 0.1290              | 7.677                   | 1.000         |
| capsule_priority    | 0.0322        | 0.1290              | 7.484                   | 1.000         |
| capsule_coverage    | 0.0000        | 0.1935              | 6.871                   | 1.000         |
| **capsule_learned** | **0.1290**    | **0.3225**          | **5.000**               | 1.000         |

This is the **upper-end single seed**, not a typical seed.
Reporting single-seed numbers in isolation overstates the result;
the cross-seed table above is the canonical reading.

### Budget-sweep result (W4-3 evidence)

A coarser sweep (40-scenario bank → 14 eval; ``K_auditor ∈ {4, 6,
8, 12, 16}``) confirms the local-view-limitation direction:

* At ``K_auditor=4``, every strategy reports
  ``accuracy_full = 0.000`` — no policy admits enough
  causally-relevant claims to support a correct decision.
* At ``K_auditor=8`` and above, fixed baselines climb to
  ``accuracy_full ≈ 0.21`` (coverage-guided) while learned
  underperforms at the smaller training scale (an honest signal:
  the learned policy depends on training-data scale and is *not*
  monotonically dominant).

The sweep is *empirical evidence supporting* W4-3 (the negative
theorem), not a proof — the proof is in the W4-3 construction.

## Honest scope

* The benchmark is the synthetic Phase-31 incident-triage family.
  It is *deliberately* the kind of task the capsule layer should
  dominate on (typed claims, partial per-role evidence, fixed
  decoder, deterministic generator). External validity to a real
  production multi-agent team is open.
* The learned policy is a per-role logistic regression over six
  hand-named features. It is **not** a deep model. The result is
  intended as "modest learning over capsule features beats the
  best fixed admission rule," not "learning solves multi-agent
  context."
* The `TEAM_HANDOFF` / `ROLE_VIEW` / `TEAM_DECISION` capsule kinds
  ship in the SDK's closed vocabulary, but the **Wevra product
  runtime** (the ``RunSpec`` → ``RUN_REPORT`` path) does not seal
  any of them. They are emitted only by ``TeamCoordinator`` —
  i.e. by the multi-agent coordination *research slice*. The
  product / programme split (`docs/START_HERE.md` § 1) is
  preserved.
* The team-layer audit verifies *lifecycle* invariants only. It
  does not verify the *correctness* of the team decoder
  (``Dec(A_r) = gold(S)``); decoder correctness is a separate
  evaluation responsibility (``grade_answer``).

## What remains outside the slice

* **Real-LLM team coordination.** The benchmark is run against a
  deterministic decoder + noisy-extractor surrogate, not real LLM
  agents. The capsule-layer mechanism is independent of the
  decoder; an LLM-driven decoder would slot into the same path.
* **Asynchronous / pipelined rounds.** The current API is
  synchronous: all handoffs of round $n$ are emitted before
  ``seal_role_view``. Asynchronous coordination requires extending
  the admission API.
* **Cross-team coordination.** Multiple ``TeamCoordinator``
  instances on the same ledger work mechanically (the ``team_tag``
  field disambiguates), but no theorem yet covers cross-team
  dependencies.
* **Cohort-lifted role view.** W4-C2 (open) — admitting a COHORT
  capsule rather than individual handoffs may close the W4-3
  limitation on a sub-class of scenarios. The Phase 53 candidate.

## Reproducibility

```bash
python -m vision_mvp.experiments.phase52_team_coord --out -
python -m vision_mvp.experiments.phase52_team_coord --budget-sweep --out -

python -m pytest vision_mvp/tests/test_wevra_team_coord.py -v
```

The benchmark's report carries ``schema = phase52.team_coord.v1``
and is reproducible from default seeds.

## Cross-references

* Formal model: [`docs/CAPSULE_TEAM_FORMALISM.md`](CAPSULE_TEAM_FORMALISM.md)
* Theorem registry: [`docs/THEOREM_REGISTRY.md`](THEOREM_REGISTRY.md)
  (W4 family rows)
* Research status: [`docs/RESEARCH_STATUS.md`](RESEARCH_STATUS.md)
  (multi-agent coordination slice)
* Tests: [`vision_mvp/tests/test_wevra_team_coord.py`](../vision_mvp/tests/test_wevra_team_coord.py)
* Code: [`vision_mvp/wevra/team_coord.py`](../vision_mvp/wevra/team_coord.py),
  [`vision_mvp/wevra/team_policy.py`](../vision_mvp/wevra/team_policy.py),
  [`vision_mvp/experiments/phase52_team_coord.py`](../vision_mvp/experiments/phase52_team_coord.py)
* Substrate baseline: [`vision_mvp/core/role_handoff.py`](../vision_mvp/core/role_handoff.py),
  [`vision_mvp/tasks/incident_triage.py`](../vision_mvp/tasks/incident_triage.py)
