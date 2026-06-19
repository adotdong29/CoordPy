# W96-D — RealWorldQA cheap preflight V1 (PASS at both scales; D2 preflight-earned)

> **2026-05-25 — D2 (RealWorldQA) preflight PASSed composite at
> BOTH 11B and 90B on the canonical ``lmms-lab/RealWorldQA``
> test split (765 problems; 2 SHA-anchored parquet shards;
> shard SHAs
> `0ed8b555...e64952` +
> `7dcb3ac3...c74d0`).  All 4 P-probes PASS at both scales;
> the W93 5-gate composite PASSes; P3 A1@K=5 residual is
> 26.56 pp at 11B (well above floor) and 20.51 pp at 90B (just
> above floor).  Per the W96-D runbook, D2 is now
> preflight-earned for a NIM smoke test + 1-seed × 30-problem
> Phase 2 cheap pilot.  Phase 2 is OUT OF SCOPE for this
> milestone unless explicitly authorised.  No NIM spend.**

## Configuration

| Field | Value |
|---|---|
| Battlefield | RealWorldQA test (lmms-lab/RealWorldQA) |
| Parquet URLs | `data/test-00000-of-00002.parquet`, `data/test-00001-of-00002.parquet` |
| Shard SHA-256 (shard 0) | `0ed8b555586923099bd5d6ba5dd8b656b403ccfc418881facd237a6d6fe64952` |
| Shard SHA-256 (shard 1) | `7dcb3ac3483362ca082cd4cddd0ab1389e9a276f1aefd4603fde0a2ce6bc74d0` |
| Total parquet bytes | `678,342,154` |
| n_problems | `765` |
| Corpus Merkle root | `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab` |
| Candidate VLM family | `meta/llama-3.2-{11b,90b}-vision-instruct` |
| Decomposition argument | 1083 chars; D2-B0 (W95-B0 scene-port) |
| P3 ceiling | 80.00 % (W95 default) |

## Probe verdicts

### 11B (`meta/llama-3.2-11b-vision-instruct`)

| Probe | Verdict | Detail |
|---|---|---|
| P1 corpus integrity | **PASS** | 678 MB in [200, 900] MB; 765 problems in [700, 800]; n_with_image=765; n_with_answer=765; both shards SHA-anchored |
| P2 executor self-test | **PASS** | 765/765 = 100.00 % gold-as-prediction (threshold 98 %) |
| P3 A1 saturation | **PASS** | published single-shot 50.00 % → A1@K=5 estimate **73.44 %** → residual **26.56 pp** (ceiling 80 %; PASS by 6.56 pp room) |
| P4 decomposition argument | **PASS** | 1083 chars (threshold 200); 100 % multimodal-completeness (image + question + answer) in first 500 rows |
| W93 G1 hypothesis | PASS | 692 chars |
| W93 G2 sidecar evidence | PASS | W95-B0 cross-benchmark prior |
| W93 G3 adversarial ablation | PASS | reader-removal coherence |
| W93 G4 budget accounting | PASS | 5 model calls / problem; matches K=5 |
| W93 G5 benchmark justification | PASS | 647 chars; non-HumanEval-Visual |
| **Overall composite** | **PASS** | All 9 gates clear |

Run dir: `results/w96/realworldqa_preflight/20260525T172611Z/`
Composite verdict CID: `07f9298c613e77653c6c42cc0cd28063c199ea5d8dc157be534135b45a645ccd`

### 90B (`meta/llama-3.2-90b-vision-instruct`)

| Probe | Verdict | Detail |
|---|---|---|
| P1 corpus integrity | **PASS** | identical to 11B |
| P2 executor self-test | **PASS** | identical to 11B |
| P3 A1 saturation | **PASS** | published single-shot 60.00 % → A1@K=5 estimate **79.49 %** → residual **20.51 pp** (ceiling 80 %; PASS by 0.51 pp room — *narrow*) |
| P4 decomposition argument | **PASS** | identical to 11B |
| W93 G1..G5 | PASS (all 5) | identical hypothesis / evidence / ablation / budget / justification |
| **Overall composite** | **PASS** | All 9 gates clear; P3 margin at 90B is narrow (0.51 pp above floor) |

Run dir: `results/w96/realworldqa_preflight/20260525T172641Z/`
Composite verdict CID: `1e38d04b97c69d8257d6eff6b30992d5ed57bbac06c1a648178b464951ebe4af`

## Cross-scale interpretation

* **Both 11B and 90B PASS** — the strongest possible preflight
  result.  Per the W96-D runbook's cross-scale rule, the
  candidate is entitled to Phase 2 cheap pilot at the chosen
  scale.
* **The 90B margin is structurally narrow** (0.51 pp above the
  P3 floor).  This is the W96-A lesson applied a-priori: a
  stronger unified VLM tends to saturate A1@K=5.  At 90B the
  expected B-team room is ~20 pp residual — meaningful but not
  generous; if the published single-shot for 90B is 2 pp
  higher than recorded (e.g., 62 % rather than 60 %), P3 at 90B
  would FAIL.
* The W96-D 90B Phase 2 cheap pilot would therefore be the
  cross-scale stress test that the runbook anchors: a +5 pp
  margin at 90B with A1@K=5 ≤ 90 % would be a genuinely strong
  signal; a narrower margin would still be informative but
  would not earn Phase 3 by the W96-C cross-scale rule.

## What this preflight earns

* **D2-B0 (W95-B0 scene-port) is preflight-earned** for a
  NIM smoke test against the 11B and 90B Vision-Instruct
  endpoints, and (per the runbook) a 1-seed × 30-problem
  Phase 2 cheap pilot at one or both scales.
* The seed identity for the W96-D D2 default slice is
  **96_504_002** per the W96-D runbook pre-commit; if the
  runbook lock holds, that pid list will be SHA-anchored
  before any NIM call.

## What this preflight does NOT yet entitle

* **No expensive run** is in scope for this milestone.  Phase 2
  is a 330-NIM-call probe, ~20-30 min wall at 11B; the runbook
  explicitly defers the launch decision until budget approval +
  pre-commit of the Phase 2 evidence shape.
* **No carry-forward retirement.**  The W95 / W96-A / W96-C
  carry-forwards remain.
* **No claim about RealWorldQA team superiority.**  The
  preflight passes structural integrity, not empirical
  superiority.  Only a Phase 2 + Phase 3 bench can claim that.

## Mechanism inference (a priori)

The W96-D D2-B0 hypothesis is structurally similar to W95-B0:
the VLM does scene perception; the text LM does inference over
the structured extraction.  RealWorldQA's structural feature is
the **entanglement of perception and reasoning** — many problems
ask "what is the spatial relation between X and Y" or "how many
red cars" or "what does the sign say".  The cleanest team
advantage exists when:

* The scene extraction is structurally lossless for the
  question's relevant features (`vlm_scene_reader` at T=0
  produces a bullet list with all relevant objects + relations).
* The text reasoner can compose the answer from the bullet list
  alone, without needing to re-see the image.

The risk is that real-world scenes are too entangled —
extraction may lose subtle visual cues that the unified VLM
preserves implicitly in its hidden states.  This is exactly
what the Phase 2 pilot would test cheaply.

## Anti-cheat (carry-forward from W88–W96-C)

* Both parquet shards SHA-anchored at preflight time (recorded
  above).
* No selective retries (no NIM yet).
* Executor truth = `evaluate_realworldqa_answer_v1` for every
  arm.  No LLM judge.
* Published-SOTA values are anchored anti-cheat constants in
  `coordpy.realworldqa_preflight_v1.REALWORLDQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL`;
  updates are explicit code changes.
* The W96-D D2 default seed identity (96_504_002) is
  pre-committed in `docs/RUNBOOK_W96D.md` BEFORE any NIM call
  for D2.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New modules `coordpy.realworldqa_loader_v1`,
  `coordpy.realworldqa_executor_v1`,
  `coordpy.realworldqa_preflight_v1` remain explicit-import
  only.

## Carry-forwards added

None at preflight stage.  A Phase 2 outcome (if launched) would
add carry-forwards.

## Carry-forwards retired

None.

## Re-running

```bash
.venv/bin/python scripts/run_w96d_realworldqa_preflight.py \
  --candidate-model meta/llama-3.2-11b-vision-instruct

.venv/bin/python scripts/run_w96d_realworldqa_preflight.py \
  --candidate-model meta/llama-3.2-90b-vision-instruct
```

## The honest claim W96-D D2 preflight earns

**On the canonical `lmms-lab/RealWorldQA` test split (765
problems; 2 SHA-anchored parquet shards), the W96-D D2-B0
(W95-B0-derived scene-extraction team) preflight PASSes ALL 9
composite gates at BOTH 11B and 90B.  P3 A1@K=5 residual is
26.56 pp at 11B and 20.51 pp at 90B (the 90B margin is narrow:
0.51 pp above the +20 pp floor).  The architecture is
hypothesis-coherent, the executor is clean, the corpus is
SHA-anchored, and the budget accounting is K=5 byte-exact.  Per
the W96-D runbook, D2 is now preflight-earned for a NIM smoke
test + Phase 2 cheap pilot (1 seed × 30 problems × K=5; ~330
NIM calls per scale).  No expensive run is in scope for this
milestone.  No carry-forward retirement.  No claim about
RealWorldQA team superiority; the preflight verdict is the
deliverable.**
