# Frontier-relevance audit — W131 (model-supply census + stronger-generator bench)

Supplements `docs/FRONTIER_RELEVANCE_AUDIT_W130_V1.md`; all prior classifications remain in force.
Classifies every W131-touched surface as active-frontier / baseline-diagnostic / dead-direction /
anti-pattern. Filled from emitted JSON where outcome-dependent.

## Active frontier arsenal (NEW in W131)

* **`coordpy.code_model_supply_census_v1`** — ACTIVE. The first module to unify the THREE
  model-supply surfaces (local-HF transformer runtime probe, local-Ollama OpenAI-compatible, hosted
  NIM via `nim_frontier_text_runtime_v1`) into one machine-checkable reachability/capability matrix,
  cross-cut with the `stronger_model_cutoff_certification_v1` disclosure gate. Carries the locked
  census schema, the same-family code smoke gate (`code_smoke_gate_v1`, 2-case constant-output
  guard), the unified `build_openai_compat_gen_v1` generation seam (drives both Ollama and NIM with
  429-aware backoff), and `normalize_fence_v1`.
* **`coordpy.generator_model_bench_v1`** — ACTIVE. Bridges the supply-census generation seam to BOTH
  the audited W130 generator arms (`stronger_generator_slate_v1`) AND the fixed W129 selector
  (`public_signal_selection_oracle_v1.select_so_v1`, gen=None). Adds the B1_PLAIN arm and the
  same-budget guard (`assert_same_budget_v1`); re-exports the W130 earn gate so W131 applies the
  IDENTICAL +2-spanning bar. The model is the only variable.
* **`normalize_fence_v1`** — ACTIVE (parsing-fairness utility). Moves a misplaced `python` fence
  info-string onto its fence line so the audited `extract_candidate_code_v1` parses Qwen-Coder output
  fairly. **Reported AS a parsing fix** — it removes an UNDER-statement of capability, it does not
  add capability (W130 honesty rule). The raw model output is preserved in the call sidecar.

## Active (prior, carried in force — reused unchanged in W131)

* `public_signal_selection_oracle_v1` (W129) — the fixed NIM-free SOLEAD selector, held FIXED
  downstream so GENERATION is the only W131 variable.
* `stronger_generator_slate_v1` (W130) — GG1/GG2/GGLEAD arms reused as B2_RDIV/B3_GG2/B4_GGLEAD; the
  earn gate reused verbatim.
* `generator_failure_atlas_v1` (W130) — the per-problem dominant-mode atlas reused for the earn-gate
  span check.
* `role_diverse_algorithm_search_v1` (W128) — the generation primitives (`build_analyze_prompt_v1`,
  `RoleArtifactsV1`, `_run_capture_stdout_v1`) reused.
* `stronger_model_cutoff_certification_v1` (W114) — the C1∧C2∧C3∧C4 gate; decision CID `258b6ed7`
  re-derives byte-identically (W114→W131).

## Useful baseline / diagnostic-only

* The census **code-competence prior** (`classify_code_prior_v1`) is a transparent name-keyed
  heuristic — an UPPER BOUND on "code competence" (a code-tuned name ≠ a strong code model). Reported
  as a heuristic, never ground truth.
* The cutoff `cutoff_boundary` values in the certification registry are internal estimates; only
  `verified_confidence ∈ {KNOWN, UNKNOWN}` (primary-source disclosure) is load-bearing.

## Dead directions / confirmed caps

* **Local-HF transformer runtime DEAD** — `transformers_runtime_v1` / `code_substrate_v1` /
  `substrate_adapter_v25` cannot run a model: `torch`/`transformers` are not importable under this
  interpreter (Python 3.14.5; ModuleNotFoundError). Worse than W124's "too old". The local model
  lane is reachable ONLY via the Ollama OpenAI-compatible endpoint.
* **Local strong-code throughput cap** — `qwen2.5-coder:32b` is reachable + code-competent (smoke
  PASS) but CPU-bound on this host (`size_vram=0`, ≈1 tok/s) ⇒ ≈25 min/call ⇒ a 110-call bench is
  impractical. The local lane cannot practically supply a stronger-than-Maverick generator (32B too
  slow on Metal; 7B too weak).
* **Model-axis generation cap** — the strongest reachable code model (Qwen3-Coder-480B-a35b, ≈28×
  Maverick's active params), W129 selector fixed, K=5, produced only the W130 `doubleup` solve and
  **0 genuinely-new** EXPOSED hard-cluster solves ⇒ `W131-L-MODEL-AXIS-GENERATION-CEILING-DEV-BENCH-
  CAP` [confirmed by the hosted-core JSON; full-slate (escalation) + local-7B confirmation in
  `RESULTS_W131`]. The MODEL-axis sibling of the W123→W130 cap taxonomy.
* **Resistant-eligibility supply cap** — FRONTIER_ELIGIBLE = NONE: 13 reachable stronger-than-Maverick
  code models, all UNKNOWN-from-primary on cutoff ⇒ DEV_ONLY. The supply gap MOVED from "no strong
  model exists" (W124) to "no PRIMARY-KNOWN-cutoff stronger model on the ICPC family" (disclosure,
  not existence) ⇒ `W131-L-MODEL-CAPABILITY-LIFTS-EXPOSED-BUT-RESISTANT-INELIGIBLE` applies to any
  DEV_ONLY EXPOSED earn (the W127 contamination/memorization lesson).

## Hosted-controller / substrate stack (KILLED literal bridges, carried forward)

* `controller_native_code_mechanism_v1` / `hosted_cache_aware_planner_v12` /
  `hosted_real_handoff_coordinator_v11` — efficiency-only KV-prefix / substrate-trust constructs;
  the literal code bridges remain KILLED as fake-different (W125/W128/W130; graphify-confirmed no
  ICPC-code path). W131 does not revive them — the model-swap seam is a direct OpenAI-compatible
  chat-completions call, not a controller bridge.

## Anti-patterns (unchanged, reinforced)

Bounded-context / compaction / prose-summary / "cram less / truncate better" REMAIN explicit
anti-patterns. W131 reinforces this: the attack was a real model-capability swap on real code with a
hidden-test-free public-signal selector held fixed — the OPPOSITE of a truncation trick. The
`normalize_fence_v1` fix is a parsing-fairness correction, transparently reported, never a capability
claim.
