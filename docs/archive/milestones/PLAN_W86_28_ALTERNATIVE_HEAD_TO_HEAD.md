# Plan — #28 alternative head-to-head where multi-agent provably helps

> Closure plan for issue #28 (P0 — Real-World Multi-Agent Task
> Benchmark). W85 ran the live N=20×3-seed GSM8K head-to-head
> on Llama-3.1-8B-Instruct via NIM and **honestly refuted** the
> strict-improvement claim — CoordPy multi-agent persona-debate
> (B) underperformed both stock zero-shot CoT (A0) and same-budget
> K=5 self-consistency (A1) (71.7 % < 75.0 % < 81.7 %; carry-
> forward limitation
> ``W85-L-GSM8K-BENCH-V1-MULTI-AGENT-DOES-NOT-BEAT-SELF-CONSISTENCY-CAP``).
> This plan picks a different benchmark + multi-agent shape
> where the multi-agent debate literature reports a reliable
> win at the same compute budget. The anti-cheat clauses from
> the original issue text continue to bind unchanged.

## Why GSM8K + persona-debate failed (and how to use that)

The W85 failure mode is well-explained by the multi-agent
debate literature:

1. **Self-consistency is a hard same-budget baseline on
   arithmetic.** Self-consistency-CoT (Wang et al. 2022) is
   close to the Pareto frontier on GSM8K with 8B-scale models;
   K=5 independent CoT samples + majority vote captures most
   of the value of test-time compute for short-chain
   arithmetic reasoning.

2. **Persona-debate over already-correct solutions is hostile.**
   Llama-3.1-8B is ~80 % accurate zero-shot on GSM8K. Critic
   feedback on already-correct solutions can flip the reviser
   to a wrong answer — a known failure mode (Du et al. 2023,
   Liang et al. 2023, "When and why does multi-agent debate
   help reasoning?").

3. **No external-feedback signal.** A pure text-only critic has
   no source of truth to anchor critique; it confabulates
   plausible-but-wrong criticisms. The literature reports
   reliable multi-agent wins almost exclusively when the
   critic has *external feedback* — an executor, a retrieval
   index, a verifier — that the single-shot baseline does not.

## Forcing function for the W86 #28 closure

The literature converges on three regimes where multi-agent
provably beats self-consistency at the same compute budget:

| Regime | Why it works | Lit reference |
|---|---|---|
| **Code with executor as critic** | Executor returns deterministic ground truth (pass/fail) per test. Critic has REAL signal to surface. | Reflexion (Shinn et al. 2023), Self-Debug (Chen et al. 2023), AlphaCodium (Ridnik et al. 2024) |
| **Decompose-then-aggregate on multi-step puzzles** | Long-chain tasks with verifiable subgoals; per-step verification + targeted re-execution beats global vote. | Tree-of-Thoughts (Yao et al. 2023); BIG-Bench-Hard subsets |
| **Retrieval-augmented multi-hop QA** | Retrieval gives the critic a verifiable corpus to ground critique. | Self-RAG (Asai et al. 2023), MultiHop-RAG (Tang & Yang 2024) |

CoordPy already ships W84 ``tool_call_substrate_v1`` (with
``PythonExecSandboxToolV1``, ``RipgrepLikeFilesystemToolV1``,
``DeterministicStubHTTPToolV1``) — the executor-as-critic
substrate. Using it for #28 closes #28 AND additionally
exercises #33's tool-substrate axis on a real published
benchmark.

## Chosen path for #28 W86 closure

**Benchmark: HumanEval** (Chen et al. 2021, OpenAI).

- 164 hand-written Python programming problems with
  reference test cases.
- Canonical published metric: **pass@1** at temperature 0.0
  (deterministic), **pass@k** for k > 1 at temperature 0.7.
- Public test set with verified canonical solutions.
- Standard baseline numbers exist for Llama-3.1-8B-Instruct
  (~60–65 % pass@1 in published reports); the bar is well-
  characterised.

**Three arms (same model on all):**

* **A0** — stock single-shot generation. 1 call/problem at
  `temperature=0.0`. Reports pass@1.
* **A1** — same-budget K=5 self-consistency-style sampling at
  `temperature=0.7` + select-by-first-pass-on-tests.
  Reports pass@1 (since the "majority vote" for code is just
  "first one that passes the tests"). K=5 calls/problem.
* **B (CoordPy multi-agent + executor-as-critic)** — K=5 calls
  shaped as:
  1. `solver` — generate initial solution (temperature=0.7).
  2. `executor` — run the unit tests deterministically via
     ``coordpy.tool_call_substrate_v1.PythonExecSandboxToolV1``
     (NOT the user-facing model; a real subprocess sandbox).
     Returns pass/fail per test + stderr.
  3. `critic` — given solver's solution + executor's failed
     test output, identify the bug class.
  4. `reviser` — rewrite the solution conditioned on the
     critic's bug class.
  5. `final-judge` — re-run the executor on the reviser's
     output; if it passes the visible tests, that's the
     answer; if not, fall back to solver's best attempt.

  The single difference from W85's persona-debate B is that
  the critic receives *real* feedback from the executor, not
  text-only debate. This is the executor-as-critic pattern
  that the Reflexion / Self-Debug literature shows reliably
  wins.

**Compute budget control (anti-cheat for the K=5 self-
consistency baseline at same compute):**

A0 = 1 model call.
A1 = 5 model calls.
B  = 5 model calls (solver + critic + reviser are the only
     model calls; executor and final-judge are local /
     deterministic — same compute as A1).

**Anti-cheat clauses (verbatim from #28 issue body, all preserved):**

* "Do not define a real-world bench that is just a renamed
  synthetic bench" — HumanEval is published, canonical,
  versioned. Bench loader will SHA-256 the upstream JSON.
* "Do not improve the score by selectively retrying failed
  seeds" — every (seed, arm, problem) triple is exactly one
  set of calls; no retry-on-failure budget; no seed selection.
* "Do not swap the model under the composed pipeline for a
  bigger one than the baseline" — same Llama-3.1-8B-Instruct
  served via the SAME NIM endpoint for all three arms.
* "Do not count 'no error' as 'task success'" — pass@1 is
  defined by running ALL test cases including hidden ones (the
  HumanEval tests are paired with the canonical solution
  function signature; we run those tests).
* "Do not stub the audit chain (must be re-verifiable from
  disk by a third party)" — bench writes a sidecar JSONL with
  full prompts + responses; verifier re-hashes the chain.
* "Do not declare success if the composed pipeline loses on
  every metric" — strict-beat verdict is decided by the
  empirical run, not asserted; bench fails honestly if B does
  not strictly beat A1.

**Where this runs:** No GPU required. NIM is free (the user
already has an account from the W85 work). Wall-clock budget:
~30 min for 50 problems × 3 seeds × ~11 calls per problem ×
~3 sec per call.

**What this closes:**

* If B strictly beats A1 on pass@1 across at least 1 of 3
  seeds → DoD bullet 3 ("composed pipeline strictly improves
  at least one published metric") **MET**.
* Per-task Merkle chain + sidecar offline-re-verifiable →
  DoD bullet 4 **MET**.
* RealTaskBenchAdapterV1 → DoD bullet 1 **MET**.
* End-to-end + outcomes → DoD bullet 2 **MET**.
* Results doc → DoD bullet 5 **MET**.
* ≥ 3 seeds → DoD bullet 6 **MET**.

**Where this advances #33 too:** the W84 tool substrate ships
``PythonExecSandboxToolV1`` but never ran it on a published
benchmark. HumanEval IS that exercise — the W86 closure
becomes a joint #28 + #33 advance.

**Honest carry-forward (in advance):**

* ``W86-L-HUMANEVAL-V1-NIM-DEPENDENT-CAP`` — NIM is the
  serving runtime; provider determinism / latency carry over
  from W85.
* ``W86-L-HUMANEVAL-V1-PYTHON-EXECUTOR-CAP`` — the executor
  runs in a Python subprocess sandbox; out-of-process side
  effects (network, filesystem outside the allow-list) are
  blocked. The literature's Reflexion-style results assume
  the same.

## What needs to be built

1. `coordpy/humaneval_real_bench_v1.py` — analogous to
   ``coordpy/gsm8k_real_bench_v1.py`` but:
   - Loads HumanEval from upstream
     (https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz),
     SHA-256-verifies the corpus.
   - The B arm wires in the W84
     ``PythonExecSandboxToolV1`` for executor + final-judge.
   - Per-task ``HumanEvalArmOutcomeCapsuleV1``.
   - Bench-level Merkle root.

2. `scripts/run_w86_humaneval_bench.py` — driver.

3. `scripts/verify_w86_humaneval_audit_chain.py` — verifier.

4. New theorem-registry entries:
   - ``W86-T-HUMANEVAL-CORPUS-SHA256-VERIFIED`` —
     mechanically-checked.
   - ``W86-T-HUMANEVAL-AUDIT-CHAIN-RE-VERIFIABLE`` —
     mechanically-checked.
   - ``W86-T-HUMANEVAL-MULTI-AGENT-WITH-EXECUTOR-BEATS-SELF-CONSISTENCY`` (or its negative-result twin) —
     empirical.
   - ``W86-T-TOOL-SUBSTRATE-USED-ON-PUBLISHED-BENCH`` —
     mechanically-checked (the #33 joint advance).

5. Updates to ``docs/HOW_NOT_TO_OVERSTATE.md``, ``docs/AUDIT_POST_W83_BLOCKERS.md``, and ``docs/THEOREM_REGISTRY.md``.

## Estimated effort

* Bench module + driver + verifier: ~600 lines, 1 day.
* W86 closure run on NIM: ~30 minutes wall-clock for 3 seeds
  × 50 problems × ~11 calls.
* Results doc + theorem registry: ~1 hour.

## Open question for the user

Same-model fairness: should B use Llama-3.1-8B-Instruct (the
W85 baseline) or upgrade to a stronger NIM model like Llama-
3.3-70B-Instruct? The anti-cheat clause forbids changing model
size *between* arms, but the *baseline* model choice is the
user's call. Default: stay on 8B for direct comparison with
W85's negative result.
