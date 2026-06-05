# Frontier-relevance audit — W137 (parser-neutral hard battlefield v2 + model-ladder calibration)

Supplements the W135/W136 audits; all prior classifications remain in force. Classifies each W137
asset as **active frontier arsenal**, **useful baseline-only**, or **dead/anti-pattern**, so later
milestones reuse the right pieces and avoid the wrong ones.

## Active frontier arsenal (reuse these)

- **`coordpy.parser_neutral_io_v1` (the HC1 parser-neutrality kernel).** The reusable repair for the
  W136 confound: any future generated battlefield must declare an `IoShapeV1` and pass
  `parser_neutrality_gate_v1` (dual-parser agreement). This is the durable, general fix — not a
  per-problem hack like the W136 `_reformat`. **Highest-value W137 asset.**
- **`coordpy.model_ladder_calibration_v1` (HC3 + HC4 model-ladder calibration).** The instrument that
  decides whether a minted field has real, mechanism-sensitive headroom (vs the W136 one-shot
  failure). Discrimination-across-tiers, not raw difficulty, is the admission criterion. Reusable on
  ANY future generated slate; push-button.
- **`coordpy.hard_battlefield_slate_v2` + `hard_battlefield_corpus_v2`.** The parser-neutral hard
  slate (17 templates, 4 modes) + the seed-disjoint corpus assembler with HC5 diversity and
  construction-based resistance. Reusable substrate for re-testing any mechanism on a CLEAN field.
- **The §7a/§7b/§7c earn discipline (RUNBOOK_W137) with the parsing/formatting-only EXCLUSION
  (the W136 clause) + the ≥2-model-tier same-sign condition.** The earn rule that makes "the
  mechanism beats blind reflexion on a clean hard field" falsifiable.
- **The fake-different discipline on the repaired field** (`repaired_field_mechanism_bench_v1`):
  M3 relabeled-reflexion + B0 classify `FAKE_DIFFERENT`; C0/M1/M2 classify `REAL` — the bench cannot
  reward decoration.

## Useful baseline-only (controls, not the lead)

- **A1 (self-consistency)** and **B0 (blind reflexion)** — the same-budget anchors the lead arm must
  beat by ≥+5 pp on a clean field.
- **C0 (exact-oracle complexity witness, W133 EW2)** — the complexity-mode diagnostic; W133/W134
  showed it is REAL but single-mode/single-family, so it is a control, not the lead.
- **M2 (oracle-free deployable, W134 D3)** — the deployability control; oracle-free + parser-neutral
  ⇒ no parsing rescue possible.

## Dead directions / anti-patterns (do NOT pursue)

- **The OLD W132–W135 generated WA/SE slices** as an algorithm benchmark — I/O-confounded
  (W136) AND, I/O-fixed, low-headroom. W137 does NOT rerun them. The new parser-neutral slate
  supersedes them.
- **Per-problem I/O reformatters** (the W136 `_reformat` hard-keyed on 3 ids) — superseded by the
  generic normal-form contract; do not extend the per-problem hack.
- **The learned-memory line** (differentiable/composed/live memory, constrained-policy) — KILLED in
  W124/W136 as synthetic-only at inference; only the weightless `controller_native_code_mechanism_v1`
  router is usable. Unchanged.
- **Bounded-context / compaction / prose-summary / "cram less / truncate better"** — remain explicit
  anti-patterns. The W137 mechanisms are the OPPOSITE move (richer oracle-grounded feedback on a
  clean field), not compression.
- **Date-based-resistance stronger-model hunts** — the `258b6ed7` gate is CLOSED
  (no primary-KNOWN-cutoff stronger model); W137 uses construction-based resistance instead, which is
  strictly stronger (DyCodeEval 2503.04149) and probes any model regardless of cutoff.

## Frontier target + gate

Default frontier target stays `meta/llama-3.3-70b-instruct` (the W105 retirement model; KNOWN cutoff
~Dec-2023; resistant by date AND construction). Maverick (Aug-2024 KNOWN, W113-settled) is an optional
push-button cross-scale check; W137 does NOT block on it. The frontier run is earned ONLY by §7b on
the repaired held-out eval slice; the stronger-model cutoff gate (`258b6ed7`,
`NO_CERTIFIABLE_STRONGER_MODEL`) is re-derived in code and CLOSED.
