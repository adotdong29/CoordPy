# Contamination-control framing — W116 (upstream instrument-supply ATTACK + primary-source cutoff ATTACK + upstream-admission pipeline)

> Companion to `CONTAMINATION_CONTROL_FRAMING_W113_V1.md` /
> `…_W114_V1.md` / `…_W115_V1.md`. W116 tests **certification supply**, NOT the
> contamination confound itself; the confound status is UNCHANGED (STRENGTHENED-
> not-proven, per W113). This doc records what W116's supply-side attack does and
> does not license about contamination.

## What W116 did

W116 attacked the two halves of the W114/W115 supply blocker at their sources:

* **Instrument side (Lane α)** — went one level upstream of the pinned `release_v6`
  and re-verified the LiveCodeBench frontier LIVE at FOUR authoritative surfaces
  (lite file tree / loader `ALLOWED_FILES` + `release_latest` resolution / full
  `code_generation` dataset / GitHub repo). Result: NO admissible new instrument;
  the functional frontier is conclusively 2025-04-05; `release_latest` resolves to
  `release_v6`; a "planned v7" exists only in a non-primary summary and is
  INADMISSIBLE.
* **Model side (Lane β)** — re-checked official cutoff disclosures from PRIMARY
  sources. Result: the last hypothesized tier-2 candidate
  (`mistralai/mistral-small-4-119b-2603`) is now a CONFIRMED REAL model (119B MoE,
  2026-03-16) whose official card discloses NO cutoff; Qwen3-Coder-480B +
  DeepSeek-V4-pro remain UNKNOWN; no reachable stronger-than-Maverick model has a
  primary-KNOWN cutoff ≤ 2025-01.

Verdict re-derives `NO_CERTIFIABLE_STRONGER_MODEL` (decision CID `258b6ed7…`,
byte-identical to W114/W115); $0 NIM.

## The contamination logic (unchanged from W113/W114/W115)

* A LiveCodeBench functional problem is contamination-RESISTANT **for a model** iff
  its `contest_date` is strictly after that model's training cutoff (resistance is
  MODEL-CUTOFF-RELATIVE — the W112/W113 finding).
* To certify a slice resistant for a STRONGER-than-Maverick model you need a
  primary-KNOWN cutoff AND ≥30 functional problems dated after it. On `release_v6`
  that requires a KNOWN cutoff ≤ 2025-01.
* W116 confirms — at the upstream supply level — that neither half exists: no
  admissible instrument with post-Apr-2025 functional problems, and no reachable
  stronger model with a primary-KNOWN cutoff ≤ 2025-01.

## What W116 does NOT license

* It does **NOT** prove the contamination confound. W116 tests certification SUPPLY
  (is there an instrument + a KNOWN cutoff to run a clean stronger-model pilot?), not
  the confound. The confound is STRENGTHENED-not-proven (four single-/within-model
  dissociations through W113), UNCHANGED by W116.
* It does **NOT** add a retirement or weaken W89/W105 (still exactly TWO,
  contamination-EXPOSED-HumanEval-family at 70B).
* It does **NOT** claim resistant superiority (still 0 clean across both scales).
* It does **NOT** claim LiveCodeBench is "exhausted" or "contaminated" — it remains
  a clean instrument; the issue is that its newest FUNCTIONAL problems (Apr-2025) no
  longer post-date the reachable frontier models' cutoffs (the instrument supply
  lags the model frontier), now verified at four upstream surfaces.
* It does **NOT** treat the OpenRouter "2025-06" Mistral Small 4 figure as a KNOWN
  cutoff — it is non-primary AND, even taken at face value, post-dates the Apr-2025
  frontier (C2-exposed). A primary card disclosing a cutoff ≤ 2025-01 is required.

## Honest reading

W116 is the honest aggressive supply-side move: it attacks the upstream instrument
supply and the model-disclosure side at their sources, confirms the blocker is real
and current at four surfaces (and sharper on the model side, with Mistral Small 4 now
real-and-uncertifiable), and operationalises the next clean shot via a durable
upstream-admission pipeline. It buys $0 NIM because no admissible certifiable shot
exists — discipline, not omission. The bounded two-retirement contamination-EXPOSED-
HumanEval-family-at-70B claim remains the registered truth floor.
