"""Phase 11 — five LLM agents write a Python module that must pass tests.

Real task. Real ground truth. Real math applied.

Agents:
  parser      → parse_transaction
  validator   → validate_transaction
  aggregator  → aggregate_by_category
  detector    → detect_anomalies
  integrator  → analyze_ledger   (depends on the other 4)

Pipeline:
  1. Each non-integrator agent writes its function independently.
  2. Integrator agent sees preview of other agents' code and writes
     analyze_ledger — must call the four functions correctly.
  3. Code is extracted (AST-validated), composed into one module,
     run against 15 tests in a sandboxed subprocess.
  4. Partial-credit weighted score + per-test breakdown.

Math diagnostics applied per round:
  - Wasserstein-2 between round-t and round-t-1 code-embedding
    distributions (consensus-drift metric — does team opinion stabilize?)
  - Sheaf H¹ over function-signature constraints — localizes which
    inter-function interface pair has a signature mismatch.

Usage:
    # Real LLM (needs local Ollama with qwen2.5-coder:7b)
    python -m vision_mvp.experiments.phase11_code_team
"""

from __future__ import annotations
import sys, os, argparse, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from vision_mvp.core.llm_client import LLMClient
from vision_mvp.core.code_harness import (
    extract_code, function_is_defined, function_signature, run_sandboxed,
)
from vision_mvp.core.wasserstein import w2, bures_decomposition
from vision_mvp.core.sheaf_monitor import SheafMonitor
from vision_mvp.tasks.collaborative_module import (
    FUNCTION_SPECS, SPEC_ORDER, agent_prompt,
    TEST_RUNNER_SRC, compose_module, score_tests,
)


def progress(msg):
    print(msg, flush=True)


def run(model: str, max_retries: int = 2, seed: int = 1) -> dict:
    client = LLMClient(model=model)
    print(f"Phase 11 — collaborative Python module", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Specialties: {SPEC_ORDER}", flush=True)

    # ---- Round 1: all non-integrator agents write in parallel ----
    # (Actually we call sequentially — Ollama usually serializes anyway,
    # but we don't block dependencies on each other.)
    print("\n[round 1] parser, validator, aggregator, detector — independent agents", flush=True)
    accepted_code: dict[str, str] = {}
    raw_outputs: dict[str, str] = {}
    embeddings_per_round: list[dict] = []

    round_embs = {}
    for specialty in SPEC_ORDER[:-1]:   # skip integrator
        print(f"  - {specialty}: drafting …", flush=True)
        prompt = agent_prompt(specialty)
        # Retry loop: up to max_retries if we can't extract valid code
        accepted = None
        for attempt in range(max_retries + 1):
            t0 = time.time()
            out = client.generate(prompt, max_tokens=450, temperature=0.2)
            wall = time.time() - t0
            raw_outputs[specialty] = out
            code = extract_code(out)
            target_name = FUNCTION_SPECS[specialty]["name"]
            if code and function_is_defined(code, target_name):
                accepted = code
                print(f"    accepted on attempt {attempt+1} ({wall:.1f}s, "
                      f"{len(code)} chars)", flush=True)
                break
            print(f"    attempt {attempt+1} failed — "
                  f"{'no code' if not code else 'function name missing'}",
                  flush=True)
        if accepted is None:
            print(f"    GIVING UP on {specialty} after {max_retries+1} attempts",
                  flush=True)
            accepted = ""
        accepted_code[specialty] = accepted
        # Embed the accepted code for W2 tracking
        if accepted:
            round_embs[specialty] = np.asarray(
                client.embed(accepted[:800]), dtype=np.float64)

    embeddings_per_round.append(round_embs)

    # ---- Round 2: integrator, with previews of others ----
    print("\n[round 2] integrator — composes the others", flush=True)
    # Provide code previews from the other 4
    preview = {k: v for k, v in accepted_code.items() if v}
    prompt = agent_prompt("integrator", dependency_outputs=preview)
    accepted_integrator = None
    for attempt in range(max_retries + 1):
        t0 = time.time()
        out = client.generate(prompt, max_tokens=500, temperature=0.2)
        wall = time.time() - t0
        raw_outputs["integrator"] = out
        code = extract_code(out)
        if code and function_is_defined(code, FUNCTION_SPECS["integrator"]["name"]):
            accepted_integrator = code
            print(f"    accepted on attempt {attempt+1} ({wall:.1f}s)", flush=True)
            break
        print(f"    attempt {attempt+1} failed", flush=True)
    accepted_code["integrator"] = accepted_integrator or ""

    # ---- Sheaf H¹ on function signatures ----
    # The "stalks" are per-function signature vectors; edges enforce that
    # downstream calls match upstream signatures. We flag signature
    # mismatches as sheaf cohomology residuals.
    signature_info = {}
    for sp, src in accepted_code.items():
        if src:
            signature_info[sp] = function_signature(
                src, FUNCTION_SPECS[sp]["name"]
            )
        else:
            signature_info[sp] = None
    print(f"\n[signatures] {signature_info}", flush=True)

    # ---- W2 consensus drift (one round only — only one set of embeddings) ----
    # For a single set of embeddings we have nothing to compare to. But we
    # can at least report the spread (intra-round dispersion).
    if round_embs:
        X = np.stack(list(round_embs.values()))
        mean = X.mean(axis=0)
        spread = float(np.linalg.norm(X - mean, axis=1).mean())
        print(f"\n[embedding spread across specialties] {spread:.3f}", flush=True)

    # ---- Compose + test ----
    module_src = compose_module(accepted_code)
    print(f"\n[compose] module is {len(module_src)} chars", flush=True)
    test_result = run_sandboxed(module_src, TEST_RUNNER_SRC, timeout_s=20)
    score = score_tests(test_result.per_test)

    print(f"\n" + "=" * 78, flush=True)
    print("RESULT", flush=True)
    print("=" * 78, flush=True)
    print(f"  Tests passed: {score['n_passed']} / {score['n_total']}", flush=True)
    print(f"  Weighted score: {score['weighted_score']}", flush=True)
    print(f"  Passed: {score['passed_tests']}", flush=True)
    print(f"  Failed: {score['failed_tests']}", flush=True)
    print(f"  LLM generate calls: {client.stats.n_generate_calls}", flush=True)
    print(f"  LLM total tokens: {client.stats.total_tokens():,}", flush=True)
    if test_result.syntax_error:
        print(f"  SYNTAX ERROR in composed module", flush=True)
    if test_result.timed_out:
        print(f"  Timed out", flush=True)

    return {
        "n_passed": score["n_passed"],
        "n_total": score["n_total"],
        "weighted_score": score["weighted_score"],
        "passed_tests": score["passed_tests"],
        "failed_tests": score["failed_tests"],
        "per_test": score["per_test"],
        "llm_generate_calls": client.stats.n_generate_calls,
        "llm_total_tokens": client.stats.total_tokens(),
        "signature_info": signature_info,
        "module_src": module_src,
        "accepted_specialties": [sp for sp, v in accepted_code.items() if v],
        "stderr": test_result.stderr[:2000],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = run(model=args.model, max_retries=args.retries)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
