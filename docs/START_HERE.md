# Start Here

CoordPy is a Python-first SDK and CLI for building auditable AI
agent teams with structured, content-addressed context. If you want
the fastest path to understanding what this repo ships and how to
use it, read this page first.

## What CoordPy is

CoordPy gives you a stable runtime contract for AI agent teamwork:

* **Typed capsules instead of raw prompt strings.** Prompts,
  responses, handoffs, and reports are stored as structured,
  content-addressed objects with provenance and budget metadata.
* **A reproducible runtime.** One `RunSpec` in, one `RunReport` out,
  with a sealed capsule graph you can inspect and verify.
* **A team-coordination layer.** Agents exchange `TEAM_HANDOFF`,
  `ROLE_VIEW`, and `TEAM_DECISION` capsules instead of ad hoc text.
* **An audit surface.** `coordpy-capsule verify` can re-hash the
  report and its artifacts from disk.

This repo also includes the full experimental research ladder under
`vision_mvp.coordpy.__experimental__`, but the released product
surface is the stable SDK and CLI.

## Who it is for

CoordPy is for:

* developers building AI agent teams or LLM workflows that need
  reproducible shared context
* teams that want an audit trail instead of opaque prompt glue
* researchers who want the released system plus the benchmark and
  theorem trail behind it

If you only want the product surface, stay on the stable SDK and CLI
below. If you want the full research programme, jump to the paper and
results links in [Where to read next](#where-to-read-next).

## Install

Today, the exact install path is from a clone:

```bash
git clone https://github.com/adotdong29/context-zero.git
cd context-zero
pip install -e .
```

Once published to PyPI, the intended public install paths are:

```bash
pip install coordpy
pipx install coordpy
```

The CLI commands installed by the package are:

```bash
coordpy
coordpy-import
coordpy-ci
coordpy-capsule
```

Only required dependency is NumPy. Optional extras are available for
heavier local setups: `coordpy[scientific]`, `coordpy[dl]`,
`coordpy[heavy]`, `coordpy[crypto]`, `coordpy[docker]`,
`coordpy[dev]`.

## Minimal quickstart

Run the local smoke profile:

```bash
coordpy --profile local_smoke --out-dir /tmp/coordpy-smoke
coordpy-capsule verify --report /tmp/coordpy-smoke/product_report.json
```

Minimal Python path:

```python
from vision_mvp.coordpy import RunSpec, run

report = run(RunSpec(profile="local_smoke", out_dir="/tmp/coordpy-smoke"))
assert report["readiness"]["ready"]
print(report["summary_text"])
```

If you want the real-LLM demo path with a local Ollama endpoint:

```bash
COORDPY_OLLAMA_URL=http://localhost:11434 \
  coordpy --profile local_smoke --acknowledge-heavy --out-dir /tmp/coordpy-smoke
```

## Stable vs experimental

**Stable and released in SDK v3.43**

* `vision_mvp.coordpy` SDK surface: `RunSpec`, `run`, `RunReport`,
  `SweepSpec`, `run_sweep`, `CoordPyConfig`, `profiles`, `ci_gate`,
  `import_data`, `extensions`, capsule primitives, schema constants
* CLI surface: `coordpy`, `coordpy-import`, `coordpy-ci`,
  `coordpy-capsule`
* On-disk schemas: `coordpy.capsule_view.v1`,
  `coordpy.provenance.v1`, `phase45.product_report.v2`

**Experimental but included**

* `vision_mvp.coordpy.__experimental__`
* W22..W42 trust/adjudication and multi-agent coordination ladder
* R-69..R-89 benchmark drivers
* bounded live cross-host probes

**Out of scope for this release**

* `W42-C-NATIVE-LATENT`: transformer-internal trust transfer
* `W42-C-MULTI-HOST`: K+1-host disjoint topology beyond the current
  two-host setup

Those are next-programme architecture questions, not blockers to the
released CoordPy v3.43 line.

## Where to read next

If you want to use CoordPy:

* [`README.md`](../README.md) — product landing page, install, CLI,
  stable surface
* [`examples/`](../examples/) — short runnable examples
* [`ARCHITECTURE.md`](../ARCHITECTURE.md) — architectural overview

If you want to understand the released result:

* [`docs/RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md`](RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md) — final
  release result note
* [`docs/SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md`](SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md) —
  pre-committed success bar
* [`docs/RESEARCH_STATUS.md`](RESEARCH_STATUS.md) — current claims and
  status
* [`docs/THEOREM_REGISTRY.md`](THEOREM_REGISTRY.md) — theorem and
  conjecture index
* [`docs/HOW_NOT_TO_OVERSTATE.md`](HOW_NOT_TO_OVERSTATE.md) — claim
  boundary
* [`papers/context_as_objects.md`](../papers/context_as_objects.md) —
  main paper draft

## Historical research record

Everything below is preserved as the per-milestone audit trail. Use
the sections above for current onboarding; use the table below when
you need milestone-by-milestone history.

> **Current canonical reading.** The active scientific and product
> position is captured by a small set of files; everything else is
> historical record under [`archive/`](archive/).
>
> | Topic                                | Live doc                                                           |
> | ------------------------------------ | ------------------------------------------------------------------ |
> | One-pass orientation                 | this file (`docs/START_HERE.md`)                                   |
> | What is true *now*                   | [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)                         |
> | Theorem-by-theorem status            | [`THEOREM_REGISTRY.md`](THEOREM_REGISTRY.md)                       |
> | What may be claimed (do-not-overstate) | [`HOW_NOT_TO_OVERSTATE.md`](HOW_NOT_TO_OVERSTATE.md)               |
> | Run-boundary capsule formalism (W3)  | [`CAPSULE_FORMALISM.md`](CAPSULE_FORMALISM.md)                     |
> | Team-boundary capsule formalism (W4) | [`CAPSULE_TEAM_FORMALISM.md`](CAPSULE_TEAM_FORMALISM.md)           |
> | Long-running master plan             | [`context_zero_master_plan.md`](context_zero_master_plan.md)       |
> | Two-Mac MLX runbook                  | [`MLX_DISTRIBUTED_RUNBOOK.md`](MLX_DISTRIBUTED_RUNBOOK.md)         |
> | Latest milestone (SDK v3.43 final)   | [`RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md`](RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md) |
> | Pre-committed success bar (SDK v3.43)| [`SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md`](SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md) |
> | Previous milestone (SDK v3.42 RC2)   | [`RESULTS_COORDPY_W41_INTEGRATED_SYNTHESIS.md`](RESULTS_COORDPY_W41_INTEGRATED_SYNTHESIS.md) |
> | Pre-committed success bar (SDK v3.42)| [`SUCCESS_CRITERION_W41_INTEGRATED_SYNTHESIS.md`](SUCCESS_CRITERION_W41_INTEGRATED_SYNTHESIS.md) |
> | Previous milestone (SDK v3.41 RC1)   | [`RESULTS_COORDPY_W40_RESPONSE_HETEROGENEITY.md`](RESULTS_COORDPY_W40_RESPONSE_HETEROGENEITY.md) |
> | Pre-committed success bar (SDK v3.41)| [`SUCCESS_CRITERION_W40_RESPONSE_HETEROGENEITY.md`](SUCCESS_CRITERION_W40_RESPONSE_HETEROGENEITY.md) |
> | Previous milestone (SDK v3.40)       | [`RESULTS_COORDPY_W39_MULTI_HOST_DISJOINT_QUORUM.md`](RESULTS_COORDPY_W39_MULTI_HOST_DISJOINT_QUORUM.md) |
> | Pre-committed success bar (SDK v3.40)| [`SUCCESS_CRITERION_W39_MULTI_HOST_DISJOINT_QUORUM.md`](SUCCESS_CRITERION_W39_MULTI_HOST_DISJOINT_QUORUM.md) |
> | Previous milestone (SDK v3.39)       | [`RESULTS_COORDPY_W38_DISJOINT_CONSENSUS_REFERENCE.md`](RESULTS_COORDPY_W38_DISJOINT_CONSENSUS_REFERENCE.md) |
> | Pre-committed success bar (SDK v3.39)| [`SUCCESS_CRITERION_W38_DISJOINT_CONSENSUS_REFERENCE.md`](SUCCESS_CRITERION_W38_DISJOINT_CONSENSUS_REFERENCE.md) |
> | Previous milestone (SDK v3.38)       | [`RESULTS_COORDPY_W37_CROSS_HOST_BASIS_TRAJECTORY.md`](RESULTS_COORDPY_W37_CROSS_HOST_BASIS_TRAJECTORY.md) |
> | Pre-committed success bar (SDK v3.38)| [`SUCCESS_CRITERION_W37_CROSS_HOST_BASIS_TRAJECTORY.md`](SUCCESS_CRITERION_W37_CROSS_HOST_BASIS_TRAJECTORY.md) |
> | Previous milestone (SDK v3.37)       | [`RESULTS_COORDPY_W36_HOST_DIVERSE_TRUST_SUBSPACE.md`](RESULTS_COORDPY_W36_HOST_DIVERSE_TRUST_SUBSPACE.md) |
> | Pre-committed success bar (SDK v3.37)| [`SUCCESS_CRITERION_W36_HOST_DIVERSE_TRUST_SUBSPACE.md`](SUCCESS_CRITERION_W36_HOST_DIVERSE_TRUST_SUBSPACE.md) |
> | Previous milestone (SDK v3.36)       | [`RESULTS_COORDPY_W35_TRUST_SUBSPACE_DENSE_CONTROL.md`](RESULTS_COORDPY_W35_TRUST_SUBSPACE_DENSE_CONTROL.md) |
> | Pre-committed success bar (SDK v3.36)| [`SUCCESS_CRITERION_W35_TRUST_SUBSPACE_DENSE_CONTROL.md`](SUCCESS_CRITERION_W35_TRUST_SUBSPACE_DENSE_CONTROL.md) |
> | Previous milestone (SDK v3.35)       | [`RESULTS_COORDPY_W34_LIVE_AWARE_MULTI_ANCHOR.md`](RESULTS_COORDPY_W34_LIVE_AWARE_MULTI_ANCHOR.md) |
> | Pre-committed success bar (SDK v3.35)| [`SUCCESS_CRITERION_W34_LIVE_AWARE_MULTI_ANCHOR.md`](SUCCESS_CRITERION_W34_LIVE_AWARE_MULTI_ANCHOR.md) |
> | Previous milestone (SDK v3.34)       | [`RESULTS_COORDPY_W33_TRUST_EWMA_TRACKED.md`](RESULTS_COORDPY_W33_TRUST_EWMA_TRACKED.md) |
> | Pre-committed success bar (SDK v3.34)| [`SUCCESS_CRITERION_W33_TRUST_EWMA_TRACKED.md`](SUCCESS_CRITERION_W33_TRUST_EWMA_TRACKED.md) |
> | Previous milestone (SDK v3.33)       | [`RESULTS_COORDPY_W32_LONG_WINDOW_CONVERGENT.md`](RESULTS_COORDPY_W32_LONG_WINDOW_CONVERGENT.md) |
> | Pre-committed success bar (SDK v3.33)| [`SUCCESS_CRITERION_W32_LONG_WINDOW_CONVERGENT.md`](SUCCESS_CRITERION_W32_LONG_WINDOW_CONVERGENT.md) |
> | Previous milestone (SDK v3.32)       | [`RESULTS_COORDPY_W31_ONLINE_CALIBRATED_GEOMETRY.md`](RESULTS_COORDPY_W31_ONLINE_CALIBRATED_GEOMETRY.md) |
> | Pre-committed success bar (SDK v3.32)| [`SUCCESS_CRITERION_W31_ONLINE_CALIBRATED_GEOMETRY.md`](SUCCESS_CRITERION_W31_ONLINE_CALIBRATED_GEOMETRY.md) |
> | Previous milestone (SDK v3.31)       | [`RESULTS_COORDPY_W30_CALIBRATED_GEOMETRY.md`](RESULTS_COORDPY_W30_CALIBRATED_GEOMETRY.md) |
> | Pre-committed success bar (SDK v3.31)| [`SUCCESS_CRITERION_W30_CALIBRATED_GEOMETRY.md`](SUCCESS_CRITERION_W30_CALIBRATED_GEOMETRY.md) |
> | Previous milestone (SDK v3.30)       | [`RESULTS_COORDPY_W29_GEOMETRY_PARTITIONED.md`](RESULTS_COORDPY_W29_GEOMETRY_PARTITIONED.md) |
> | Pre-committed success bar (SDK v3.30)| [`SUCCESS_CRITERION_W29_GEOMETRY_PARTITIONED.md`](SUCCESS_CRITERION_W29_GEOMETRY_PARTITIONED.md) |
> | Previous milestone (SDK v3.29)       | [`RESULTS_COORDPY_W28_ENSEMBLE_VERIFIED_MULTI_CHAIN.md`](RESULTS_COORDPY_W28_ENSEMBLE_VERIFIED_MULTI_CHAIN.md) |
> | Previous milestone (SDK v3.28)       | [`RESULTS_COORDPY_W27_MULTI_CHAIN_PIVOT.md`](RESULTS_COORDPY_W27_MULTI_CHAIN_PIVOT.md) |
> | Previous milestone (SDK v3.27)       | [`RESULTS_COORDPY_W26_CHAIN_PERSISTED_FANOUT.md`](RESULTS_COORDPY_W26_CHAIN_PERSISTED_FANOUT.md) |
> | Previous milestone (SDK v3.26)       | [`RESULTS_COORDPY_W25_SHARED_FANOUT.md`](RESULTS_COORDPY_W25_SHARED_FANOUT.md) |
> | Previous milestone (SDK v3.25)       | [`RESULTS_COORDPY_W24_SESSION_COMPACTION.md`](RESULTS_COORDPY_W24_SESSION_COMPACTION.md) |
> | Previous milestone (SDK v3.24)       | [`RESULTS_COORDPY_W23_CROSS_CELL_DELTA.md`](RESULTS_COORDPY_W23_CROSS_CELL_DELTA.md) |
> | Previous milestone (SDK v3.23)       | [`RESULTS_COORDPY_CAPSULE_LATENT_HYBRID.md`](RESULTS_COORDPY_CAPSULE_LATENT_HYBRID.md) |
> | Previous milestone (SDK v3.22)       | [`RESULTS_COORDPY_MULTI_ORACLE_ADJUDICATION.md`](RESULTS_COORDPY_MULTI_ORACLE_ADJUDICATION.md) |
> | Previous milestone (SDK v3.21)       | [`RESULTS_COORDPY_OUTSIDE_INFORMATION.md`](RESULTS_COORDPY_OUTSIDE_INFORMATION.md) |
> | Previous milestone (SDK v3.20)       | [`RESULTS_COORDPY_DECEPTIVE_AMBIGUITY.md`](RESULTS_COORDPY_DECEPTIVE_AMBIGUITY.md) |
> | Previous milestone (SDK v3.19)       | [`RESULTS_COORDPY_RELATIONAL_DISAMBIGUATOR.md`](RESULTS_COORDPY_RELATIONAL_DISAMBIGUATOR.md) |
> | Previous milestone (SDK v3.18)       | [`RESULTS_COORDPY_LIVE_COMPOSITION.md`](RESULTS_COORDPY_LIVE_COMPOSITION.md) |
> | Previous milestone (SDK v3.17)       | [`RESULTS_COORDPY_COMPOSED_REAL_LLM.md`](RESULTS_COORDPY_COMPOSED_REAL_LLM.md) |
> | Previous milestone (SDK v3.16)       | [`RESULTS_COORDPY_ATTENTION_AWARE.md`](RESULTS_COORDPY_ATTENTION_AWARE.md) |
> | Previous milestone (SDK v3.15)       | [`RESULTS_COORDPY_PRODUCER_AMBIGUITY.md`](RESULTS_COORDPY_PRODUCER_AMBIGUITY.md) |
> | Previous milestone (SDK v3.14)       | [`RESULTS_COORDPY_OPEN_WORLD_NORMALIZATION.md`](RESULTS_COORDPY_OPEN_WORLD_NORMALIZATION.md) |
> | Previous milestone (SDK v3.13)       | [`RESULTS_COORDPY_REAL_LLM_MULTI_ROUND.md`](RESULTS_COORDPY_REAL_LLM_MULTI_ROUND.md) |
> | Previous milestone (SDK v3.12)       | [`RESULTS_COORDPY_MULTI_ROUND_DECODER.md`](RESULTS_COORDPY_MULTI_ROUND_DECODER.md) |
> | Previous milestone (SDK v3.11)       | [`RESULTS_COORDPY_BUNDLE_DECODER.md`](RESULTS_COORDPY_BUNDLE_DECODER.md) |
> | Pre-committed success bar (SDK v3.13)| [`SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`](SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md) |
> | Previous milestone (SDK v3.10)       | [`RESULTS_COORDPY_MULTI_SERVICE_CORROBORATION.md`](RESULTS_COORDPY_MULTI_SERVICE_CORROBORATION.md) |
> | Previous milestone (SDK v3.9)        | [`RESULTS_COORDPY_CROSS_ROLE_CORROBORATION.md`](RESULTS_COORDPY_CROSS_ROLE_CORROBORATION.md) |
> | Previous milestone (SDK v3.8)        | [`RESULTS_COORDPY_CROSS_ROLE_COHERENCE.md`](RESULTS_COORDPY_CROSS_ROLE_COHERENCE.md) |
> | Previous milestone (SDK v3.7)        | [`RESULTS_COORDPY_SCALE_VS_STRUCTURE.md`](RESULTS_COORDPY_SCALE_VS_STRUCTURE.md) |
> | Repo top-level                       | [`../README.md`](../README.md), [`../ARCHITECTURE.md`](../ARCHITECTURE.md), [`../CHANGELOG.md`](../CHANGELOG.md) |
> | Historical record (read-only)        | [`archive/`](archive/) — pre-CoordPy theory, older CoordPy milestones, sprint prompts |
