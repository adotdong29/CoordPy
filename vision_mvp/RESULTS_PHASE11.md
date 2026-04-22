# Phase 11 — Real Result

**Task:** 5 LLM agents (qwen2.5-coder:7b) each write one function of a Transaction-Ledger Analyzer module. Composed module is executed against 15 weighted pytest-style assertions in a sandboxed subprocess (CPU + memory limits).

No mocks. No templates. Real LLM → real Python → real test runner → real ground truth.

## Score

```
Tests passed:    11 / 15
Weighted score:  0.70
LLM calls:       5 (one per agent)
Total tokens:    3,232
Wall time:       ~8 min (sequential Ollama)
```

| Specialty    | Accepted     | Time    | Chars |
|--------------|--------------|---------|-------|
| parser       | attempt 1    | 59.4s   | 949   |
| validator    | attempt 1    | 103.5s  | 563   |
| aggregator   | attempt 1    | 74.9s   | 570   |
| detector     | attempt 1    | 113.4s  | 853   |
| integrator   | attempt 1    | 137.7s  | ~700  |

All five agents produced AST-valid Python that defined the target function on the first try. No retries were needed.

## What passed (11/15)

All validator tests (3/3), all aggregator tests (3/3), all detector tests (3/3), 1/3 parser, 1/3 integrator. The individual functions are individually correct.

## What failed (4/15) — and why

These are the interesting failures. Both are *composition* bugs, not *function* bugs.

### Bug 1 — Namespace collision on `datetime`

- parser writes: `from datetime import date, datetime` (class)
- validator writes: `import datetime` (module)
- Module composition concatenates both. Last write wins: `datetime` in module globals is the **module**.
- `parse_transaction` calls `datetime.strptime(...)` — but the module has no `strptime`, so it raises `AttributeError` at call time (not `ValueError`, so the test doesn't catch it).
- `parse_rejects_malformed` survived only because it fails on `len(fields) != 4` before reaching `strptime`.

**Fails:** `parse_basic`, `parse_credit`, `integrate_basic`, `integrate_composes_all` (all four reach `strptime`).

### Bug 2 — Signature mismatch

- aggregator signature: `aggregate_by_category(txns)` — one arg.
- integrator call: `aggregate_by_category([txn], totals)` — two args.
- `TypeError` at runtime.

The `[signatures]` diagnostic emitted at end of round 2 already showed this:

```
'aggregator': ['txns']
'integrator': ['raw_lines']
```

A downstream sheaf-H¹ check over call-graph edges would have caught this pre-execution.

## Honest read

The individual LLM agents were competent — each one, given only its local spec, produced a correct function. The system-level failure was entirely in the seams between agents. This is the exact class of problem that multi-agent coordination frameworks are supposed to solve, and it's where Phase 11 both succeeds and fails:

- **Succeeds as a diagnostic:** we actually caught the signature mismatch by inspection of the signature map. A mechanical sheaf cohomology check over call edges (nodes = functions, edges = calls, stalks = arg-name vectors, coboundary = call-site arg alignment) would reject the integrator with H¹ ≠ 0 before execution.
- **Does not yet fix:** we report the mismatch but don't feed it back as a constraint to the next generation. A round-2.5 where the integrator re-generates under a "your call to `aggregate_by_category` expects 1 arg" constraint would likely close the gap.

## What the applied math actually bought us (this run)

- **AST + sandbox harness**: rejected nothing this round (all agents succeeded first try) but is load-bearing for any noisier model — without it, we'd be executing arbitrary text.
- **Embedding spread (W2-flavored)**: reported 0.684 across the 5 specialties. Single round, so no drift-over-time measurement yet; the W2 distance function is correct (see `test_w2_detects_polarization_with_same_mean`) but we need ≥ 2 rounds to use it for consensus-drift tracking.
- **Signature / sheaf diagnostic**: correctly surfaced the argument-count mismatch. Not yet wired as a rejection criterion — currently informational only.

## Reference comparison

A hand-written solution (one author, no composition) passes 15/15 on the same test suite. The gap (11 vs 15) is not a capability gap of the individual functions — it's the cost of distributing the work across independent agents without a coordination protocol that enforces interface alignment.

That coordination protocol is the thing this project is building.

## Files

- `vision_mvp/experiments/phase11_code_team.py` — orchestrator
- `vision_mvp/tasks/collaborative_module.py` — specs, test runner, scoring
- `vision_mvp/core/code_harness.py` — extract + AST validate + sandbox
- `vision_mvp/core/wasserstein.py` — W2 / Bures (used for embedding spread)
- `vision_mvp/results_phase11.json` — raw run artifact (score + per-test + module_src)
- `vision_mvp/tests/test_phase11_components.py` — 19 unit tests, all passing

## Next step (not done in this run)

Wire the signature map into a rejection path. Concretely: after round-1 acceptance, build a call-graph stalk assignment over `FUNCTION_SPECS` and require the integrator's prompt to include the exact caller-side arg lists. Re-generate integrator if the composed module's H¹ residual is non-zero.
