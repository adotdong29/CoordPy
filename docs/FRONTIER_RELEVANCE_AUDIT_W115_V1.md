# Frontier-relevance audit — W115 (supplement)

Extends the W114 V1 audit (`docs/FRONTIER_RELEVANCE_AUDIT_W114_V1.md`); all prior
classifications remain in force. This supplement classifies the W115 artifacts so
the active frontier stays clean and the bounded ceiling is easy to defend.

## Active frontier arsenal (W115 additions)

* **`coordpy.frontier_certification_pipeline_v1`** — the durable future-fire
  supply-chain pipeline. The reusable, primary-source-grounded instrument that, from
  a `FrontierSnapshotV1` (external state as data), produces: a latest-official-
  release detector (`newer_release_available`), a generalised frontier-date summary
  + threshold table, a per-model go/no-go matrix (reusing the W114
  `certify_model_v1` / `decide_certification_v1` gate, no duplication), a disclosure-
  consistency guard, and a structured `W116FireConditionV1`. **This is the durable
  asset W115 leaves behind: W116 re-runs it against an updated snapshot to decide
  spend push-button.** It SUPERSEDES the W114 one-shot certification as the active
  entry point (W114's module remains the underlying gate it wraps).
* **`scripts/run_w115_frontier_certification_v1.py`** — the NIM-free driver;
  re-verifies the instrument histogram against the SHA-pinned corpus and emits
  `results/w115/frontier_certification/frontier_certification_verdict.json`
  (result CID `6890419c…`; decision CID `258b6ed7…` = the W114 decision).
* **The live-re-verification discipline** — re-checking the external frontier from
  primary sources each milestone (not relying on a prior snapshot) is now a
  first-class frontier habit: it caught the DeepSeek V4 card's *publication*
  (2026-04-27) and confirmed it still discloses no cutoff.

## Useful baseline-only / context

* The W113/W114 instruments (`livecodebench_resistant_slice_v1`,
  `cross_scale_resistant_interpretation_v1`, `tier2_readiness_v1`,
  `stronger_model_cutoff_certification_v1`) remain the resistant-slice + cross-scale-
  interp + tier-2-ranking + per-model-certification machinery; W115 imports them
  unchanged (no duplication).
* The W112 reachability facts (Maverick reachable; 405B 404×6; tier-2 reachable)
  are carried as a FIXED PRIOR — reachability is not the binding gate in W115, so it
  was not re-probed.

## Dead / closed directions (reaffirmed, NOT reopened)

* No reopening MBPP+ V2 (W102), frozen cross-modal lines (RealWorldQA @ 11B), the
  closed Llama-3.1 rescue branch (W106), or APPS main-lane NIM (exposed control
  only).
* No 70B resistant reflexion de-noise (W109 rule).
* No second Maverick resistant reflexion rerun on the same `release_v6` instrument
  (redundant; no verdict-changing power).
* No 405B expensive run (404×6; gate closed).
* No dirty / contamination-EXPOSED benchmark sold as a frontier win.

## Anti-patterns (reaffirmed)

* bounded-context / compaction / prose-summary / "truncate better" REMAIN explicit
  anti-patterns, NOT the frontier path. W115's lanes are a LIVE external-frontier
  re-verification + a certification-supply pipeline on real dated instruments — the
  opposite of a token-compression trick.
* "Count a close / confounded / exposed edge as a win" — explicitly rejected: W115
  earns NO pilot because none is genuinely certifiable, rather than buying a
  redundant or uncertifiable run. "Count 'no new pilot' as acceptable when the
  frontier actually moved" — explicitly guarded against: W115 re-verified LIVE that
  it did NOT move.

## Net

W115 adds one durable frontier asset (the future-fire supply-chain pipeline) and
one live-re-verified blocker (the external frontier is unchanged; no certifiable
slice). The active code-line frontier is now: **the bounded ceiling stands +
re-run the push-button pipeline against an updated snapshot the moment a newer
official instrument arrives or a stronger model discloses a KNOWN cutoff (W116)**.
`COO-9` stays lead. W89 + W105 remain the only two retirements.

Anchors: `docs/RESULTS_W115_FRONTIER_CERTIFICATION_V1.md`,
`docs/RESULTS_W115_MILESTONE_SUMMARY_V1.md`, `docs/RUNBOOK_W115.md`.
