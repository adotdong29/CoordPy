# Frontier-relevance audit — W117 (supplement)

> Extends `FRONTIER_RELEVANCE_AUDIT_W116_V1.md`; all prior classifications remain in
> force. Classifies what W117 produced as active-frontier arsenal vs baseline vs
> historical vs dead vs anti-pattern, so the next milestone selects the right surface.

## Active frontier arsenal (W117 additions)

* **`coordpy.upstream_derived_instrument_construction_v1`** — the durable
  upstream-DERIVED CONSTRUCTION/admission pipeline. The W118 (and beyond) push-button
  asset: update ONE `UpstreamProvenanceSnapshotV1` and re-run
  `run_upstream_construction_v1`. Components:
  * `assess_construction_admissibility_v1` (the construction rule: A1..A5 reused ∧ B1
    authoritative-LCB-provenance ∧ B2 no-operator-curation — the gate that REFUSES
    raw-contest hand-assembly / aggregator / mirror constructions);
  * `construct_upstream_derived_candidate_v1` (the candidate-instrument constructor —
    derives a post-v6 observation when a realizable LCB-published artifact exists; else
    names the exact missing artifact);
  * `ProvenanceSurfaceObservationV1` / `UpstreamProvenanceSnapshotV1` (the eight-surface
    construction-provenance state as DATA);
  * `W117_DISCLOSURE_MATRIX` + `disclosure_delta_since_w116_v1` (the sharpened
    per-model disclosure matrix + the newly-disclosed-since-W116 detector);
  * `W118FireConditionV1` (the exact next-milestone trigger — packaged /
    construction-provenance / cutoff).
* **The eight-surface construction-provenance verification method** (HF commit/revision
  log / refs / discussions, GitHub commits / tags / repo pipeline structure, README
  provenance, runner loader) — the reusable primary-source construction-attack recipe
  (deeper than W116's four release-label surfaces; checks the revision history +
  collection mechanism, not just the label).
* **The construction rule B1 + B2** — the reusable anti-cherry-pick criteria: a
  candidate instrument is construction-admissible only if its problem set is defined by
  an LCB-PUBLISHED artifact (B1) reproducibly with no operator discretion (B2). The
  durable discipline that keeps a hand-assembled slice from being sold as
  LiveCodeBench-grade.
* **`scripts/run_w117_upstream_construction_v1.py`** — the push-button driver +
  SHA-pinned histogram re-verification + decision-CID-drift guard (asserts the
  byte-identical `258b6ed7` invariant) + the eight-surface + construction-attempt
  report.

## Baseline-only / context (unchanged)

* `coordpy.upstream_instrument_admission_v1` (W116) — now the packaged-admission LAYER
  that W117 wraps (still active; reused, never forked).
* `coordpy.frontier_certification_pipeline_v1` (W115) + `…stronger_model_cutoff_
  certification_v1` (W114) + the W113 registry — the reused certification chain
  (decision CID `258b6ed7`).
* `bounded_window_baseline_v{1,2,3}` — still falsifier targets, not the frontier.

## Historical artifacts (unchanged)

W95–W112 cross-modal + MBPP-family + APPS-exposed-control lines stay as classified.

## Dead directions (W117 additions)

* **Raw-contest hand-assembly as a "resistant LiveCodeBench instrument"** — DEAD. It
  fails the construction rule (A1 ∧ B1 ∧ B2) and would be vibes-based cherry-picking.
  Not a path to a post-v6 instrument.
* **Waiting passively for a packaged `release_v7`** — superseded. W117 shows the
  construction supply is absent at the provenance level (no published pipeline /
  manifest), so even active construction cannot proceed without LCB publishing. The
  honest move is the push-button detector, not waiting and not hand-assembling.

## Anti-patterns (reinforced)

* bounded-context / compaction / token-compression / "truncate better" REMAIN explicit
  anti-patterns, not the frontier path. W117 reinforces this: the frontier move was an
  aggressive primary-source construction attack + executable construction machinery,
  NOT prose summarization or context trimming.

## What the next milestone should select

* **If** an LCB-published post-v6 provenance artifact appears (a dataset
  revision/commit/PR, OR a published collection pipeline + manifest) **or** a packaged
  `release_v7`+ **or** a reachable stronger-than-Maverick primary-KNOWN cutoff ≤
  2025-01 → re-run `run_upstream_construction_v1`; if a model certifies on a
  construction-admissible instrument, run the cheapest-honest pilot (W118).
* **Else** → the bounded ceiling STANDS; resistant-code NIM is BLOCKED on the named
  missing construction-provenance artifact; a genuinely different non-code superiority
  axis may be selected (the frozen / closed lines stay closed). `COO-9` stays lead.
