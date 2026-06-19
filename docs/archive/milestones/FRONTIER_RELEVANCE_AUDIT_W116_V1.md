# Frontier-relevance audit — W116 (supplement)

> Extends `FRONTIER_RELEVANCE_AUDIT_W115_V1.md`; all prior classifications remain in
> force. Classifies what W116 produced as active-frontier arsenal vs baseline vs
> historical vs dead vs anti-pattern, so the next milestone selects the right surface.

## Active frontier arsenal (W116 additions)

* **`coordpy.upstream_instrument_admission_v1`** — the durable upstream-ADMISSION
  pipeline. The W117 (and beyond) push-button asset: update ONE
  `UpstreamSupplySnapshotV1` and re-run `run_upstream_admission_v1`. Components:
  * `assess_instrument_admissibility_v1` (A1..A5 admissibility rule — the supply
    gate that REFUSES aggregator / mirror / website-intro / rumor instruments);
  * `detect_upstream_change_v1` (multi-surface upstream-change detector — the W117
    update signal, richer than W115's single boolean);
  * `build_certifiable_slice_candidate_v1` (reuses the W114 gate);
  * `W116_DISCLOSURE_MATRIX` + `disclosure_matrix_summary_v1` (the four-way
    per-model disclosure-status matrix);
  * `W117FireConditionV1` (the exact next-milestone trigger).
* **The four-surface upstream-verification method** (lite tree / loader
  `ALLOWED_FILES` + `release_latest` / full dataset / GitHub) — the reusable
  primary-source supply-attack recipe (sharper than W115's single-surface check).
* **`scripts/run_w116_upstream_admission_v1.py`** — the push-button driver +
  SHA-pinned histogram re-verification + decision-CID-drift guard (asserts the
  byte-identical `258b6ed7` invariant).

## Useful baseline / reference (unchanged)

* The W115 `frontier_certification_pipeline_v1` — W116 reuses it wholesale (no
  duplication); it remains the certification-matrix engine the W116 pipeline wraps.
* The W113 registry + W114 `certify_model_v1` gate + the LCB loader — the reused
  substrate.

## Historical artifacts (unchanged)

* W108/W110/W111/W112/W113 resistant-code pilots + the M3 patcher probe — the
  empirical record that resistant superiority is 0 clean across both scales.
* The W89/W105 retirement benches — the two confirmed retirements (the truth floor).

## Dead / closed directions (reaffirmed)

* 405B expensive runs (404×6; CLOSED unless reachability changes + a pre-committed
  gate clears).
* The Llama-3.1 rescue branch (W106 NO-GO).
* MBPP+ V2 (W102 cap); the frozen cross-modal lines (RealWorldQA at 11B).
* A second Maverick resistant reflexion rerun on the SAME `release_v6` slice
  (W113-settled; redundant — no verdict-changing power).
* **NEW (W116)**: treating a "planned"/rumored/aggregator-only/website-only
  instrument or cutoff as admissible — the A1..A5 rule REFUSES them. The "planned
  v7" is a W117 watch signal, NOT supply.

## Anti-patterns (reaffirmed — explicitly NOT the frontier path)

* **Bounded-context / compaction / token-compression / "truncate better"** remain
  explicit anti-patterns. W116 is the OPPOSITE: a certification-supply pipeline on
  real dated instruments + primary-source disclosures — not a truncation trick.
* **Selling a dirty / contamination-EXPOSED benchmark as a frontier win** (the W112
  exposure lesson) — W116 buys $0 NIM rather than run an uncertifiable/exposed pilot.
* **Treating an aggregator cutoff as KNOWN** — the OpenRouter "2025-06" Mistral
  Small 4 figure is non-primary AND C2-exposed; the rule requires a primary-KNOWN
  cutoff ≤ 2025-01.
* **Counting a close / confounded / rumored edge as success** — W116 records the
  exact missing-upstream-release / missing-primary-cutoff / missing-certifiable-slice
  condition as load-bearing, not a vague "needs more work".

## Net

W116's frontier-relevant output is the upstream-ADMISSION pipeline + the four-surface
supply-attack method + the sharpened (now-real Mistral Small 4) disclosure matrix.
The bounded two-retirement ceiling stands; the path forward is operationalised
(W117 push-button), not narrative. `COO-9` stays lead.
