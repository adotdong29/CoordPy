# W102 — MBPP+ loader V2 schema fix V1

> **2026-05-25.  Critical infrastructure finding + corrective
> action discovered during the W101 deferred operator step.
> The W101 V1 loader's schema assumption (parallel
> `plus_input` / `plus_output` arrays) does NOT match the
> actual EvalPlus release schema (single `test` Python program
> containing an iteration loop over `inputs` / `results`
> parallel arrays).  Against real data, V1 would silently
> emit ZERO plus-assertions and the V1 cheap pilot would
> have degenerated to a base-MBPP run — the exact SATURATED-
> CEILING regime W101 was designed to attack.**
>
> **The V2 loader + V2 executor + V2 preflight P5 + V2
> preflight P6 close this failure mode in W102 before any
> NIM spend.**

## What the W101 V1 loader assumed

`coordpy/mbpp_plus_loader_v1.py` (W101):

```python
plus_in = list(row.get("plus_input") or [])
plus_out = list(row.get("plus_output") or [])
# ...
out.append(MbppPlusProblemV1(
    ...
    plus_input=tuple(_coerce_str(x) for x in plus_in),
    plus_output=tuple(_coerce_str(x) for x in plus_out),
    ...))
```

The V1 executor then reconstructed extra-test assertions from
these arrays:

```python
def build_plus_assertions(problem):
    for i, (inp, exp) in enumerate(zip(
            problem.plus_input, problem.plus_output)):
        out.append(
            f"assert {entry}(*{inp}) == {exp}, ...")
```

If `plus_input == ()` and `plus_output == ()`, `build_plus_assertions`
returns `[]`, and the executor's `n_plus_total = 0`.  Any candidate
that passes the 3 base assertions then PASSes in `base_and_plus`
mode trivially — the EvalPlus extra-test surface is invisible.

## What the actual EvalPlus release ships

W102 fetched the canonical artifact during the operator step:

```bash
$ curl -sIL "https://github.com/evalplus/evalplus/releases/download/v0.2.0/MbppPlus-v0.2.0.jsonl.gz"
HTTP/2 404
```

The W101-pinned URL is gone.  EvalPlus's GitHub releases v0.2.0
+ v0.3.1 contain only **model-output zip files** (per-model
prompt+completion records for the EvalPlus paper's reproduction).
The dataset itself is on Hugging Face:

* **MBPP+** — `https://huggingface.co/datasets/evalplus/mbppplus/resolve/main/data/test-00000-of-00001-d5781c9c51e02795.parquet`
  (378 rows; 1 129 135 bytes; LFS SHA-256
  `dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b`)
* **HumanEval+** — `https://huggingface.co/datasets/evalplus/humanevalplus/resolve/main/test.jsonl`
  (164 rows; 11 452 868 bytes; LFS SHA-256
  `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492`)

The MBPP+ row schema is:

```python
{
    "task_id": 2,                         # integer (NOT "Mbpp/2")
    "prompt": "Write a function ...",
    "code": "def similar_elements(...) ...",  # canonical
    "source_file": "...",
    "test_imports": [],                   # may include `from math import inf`
    "test_list": [                        # the 3 base assertions
        "assert similar_elements((3,4,5,6),(5,7,4,10)) == ...",
        ...,
    ],
    "test": (                             # the EvalPlus extra-test PROGRAM
        "import numpy as np\n"
        "from math import inf\n"
        "def is_floats(x): ...\n"
        "def assertion(out, exp, atol): ...\n"
        "inputs = [...]\n"                # parallel arrays inside
        "results = [...]\n"
        "for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
        "    assertion(similar_elements(*inp), exp, 0)\n"
    ),
}
```

The hidden EvalPlus tests live inside the `test` field's
`inputs` + `results` arrays, iterated by the in-program `for`
loop.  There are NO top-level `plus_input` / `plus_output`
fields.

## What V2 does

* **`coordpy.mbpp_plus_loader_v2`** parses the actual parquet
  via pyarrow; exposes `MbppPlusProblemV2` carrying `task_id`,
  `prompt`, `canonical_code`, `base_test_list`,
  `extra_test_program`, `test_imports`, `entry_point`.  Refuses
  rows whose `extra_test_program` is empty (preventing the
  silent-degeneration case at the loader boundary).
* **`coordpy.mbpp_plus_executor_v2`** concatenates the candidate
  code with the row's `extra_test_program` and runs the combined
  program in a fresh CPython subprocess with `-I` (not `-I -S`,
  because the EvalPlus test programs `import numpy as np`).
  PASSes iff the subprocess exits 0.  Three modes:
  * `base_and_plus` — canonical W102 mode; runs the EvalPlus
    `test` program (which itself iterates base + extra tests).
  * `base_only` — runs only the 3-ish base assertions
    individually; mirrors the W90 base-MBPP executor shape.
  * `plus_only` — alias for `base_and_plus` (the EvalPlus
    iteration order does not cleanly separate base from plus;
    callers needing strict plus-only outcomes should compute
    `base_and_plus PASS AND base_only PASS ⇒ plus-only PASS`
    themselves).
* **`coordpy.mbpp_plus_reflexion_bench_v2`** wires V2 loader +
  V2 executor with the W89 sequential-reflexion mechanism;
  byte-identical A0 / A1 / B shape; per-call sidecars + per-
  seed Merkle + bench Merkle preserved.  Adds
  `per_problem_b_first_pass_idx` field for MLB-1 / MLB-2 sub-
  gate computation.
* **`coordpy.mbpp_plus_preflight_v2`** extends V1 with TWO new
  probes:
  * **P5 — extra-test-surface integrity**: ≥ 95 % of rows must
    carry the canonical EvalPlus iteration pattern
    (`for ... in zip(inputs, results)` AND
    `assertion(<entry_point>(`).  This is the structural guard
    against the V1 silent-degeneration failure mode.
  * **P6 — V1-vs-V2 canonical agreement**: V2 base-only mode
    must PASS canonical solutions at ≥ 95 % (sanity: V2 is a
    strict extension of V1's base behavior).

## V1's status post-W102

V1 stays in-repo per the W102 anti-drift contract (nothing
silently removed), but is marked as:

* **Anti-pattern** for new cheap pilots (silent-degeneration
  failure mode).
* **Historical artifact** for the W101 milestone.
* **Explicit-import-only** (no `__init__.py` reference).

The W101 preflight verdict's empirical content (P3 / P4 /
AddrW101-P1..P4) is preserved verbatim because those probes
are *cross-bench priors* extracted from the W88 / W91 sidecars
— they did not depend on the V1 loader's broken schema.  Only
P1 + P2 (corpus + executor self-test) were affected, and both
were DEFERRED in W101 (never actually ran against the wrong
loader on real data).  So the W101 verdict's empirical content
is honest; the V1 *cheap pilot driver* would have produced
silently-degenerated results had it been launched.

## Re-derivation under V2

W102 V2 preflight verdict
(`results/w102/mbpp_plus_v2_preflight/<RUN>/verdict.json`):

| Probe | Verdict (2026-05-25) | Summary |
|---|---|---|
| P1 corpus integrity (V2 loader) | **PASS** | 367 problems loaded from cached parquet; 367 / 367 have non-empty extra_test_program; SHA matches HF LFS oid. |
| P2 executor self-test (V2 base+plus) | **PASS** | 30 / 30 = 100.00 % canonical pass under `base_and_plus`. |
| P3 A1@K=5 residual estimate | **PASS** | Predicted MBPP+ A1@K=5 = 69.97 %; residual 30.03 pp; saturation margin 20.03 pp. |
| P4 decomposition argument | **PASS** | 1727 chars; W89 retirement as precedent. |
| **P5 extra-test-surface integrity (NEW)** | **PASS** | Iter loop on 364 / 367 = 99.18 %; assertion call on 367 / 367 = 100.00 % (floor 95 %). |
| **P6 V1-vs-V2 canonical agreement (NEW)** | **PASS** | V2 base-only canonical: 30 / 30 = 100.00 %. |
| AddrW101-P1 mechanism-load-bearing prior | **PASS** | W89 rescue 9.76 % ≥ 5 % threshold. |
| AddrW101-P2 per-problem cluster structure | **PASS** | Both partitions well-formed. |
| AddrW101-P3 cross-bench failure-residual stability | **PASS** | Margin 20.03 pp ≥ 10 pp floor. |
| AddrW101-P4 anti-pattern guard | **PASS** | No anti-pattern tokens in V2 bench module. |

**10 of 10 PASS.**  The W102 cheap MBPP+ V2 pilot is genuinely
earned per the pre-committed W102 decision logic.

## What this fix does NOT do

* It does NOT retroactively change any W101 published result
  (W101's empirical preflight was 6/8 PASS; that verdict is
  unchanged because P1 + P2 were DEFERRED, never run against
  the wrong loader on real data).
* It does NOT bump `coordpy.__version__` or `SDK_VERSION`.
* It does NOT publish to PyPI.
* It does NOT delete `coordpy.mbpp_plus_loader_v1` or its
  unit tests (V1 stays in-repo, marked as anti-pattern + W101
  historical artifact).

## Anchors

* `coordpy/mbpp_plus_loader_v2.py` — V2 loader.
* `coordpy/mbpp_plus_executor_v2.py` — V2 executor.
* `coordpy/mbpp_plus_reflexion_bench_v2.py` — V2 bench.
* `coordpy/mbpp_plus_preflight_v2.py` — V2 preflight (P5 + P6
  + V1 verbatim).
* `scripts/run_w102_mbpp_plus_v2_preflight.py` — runner.
* `scripts/run_w102_mbpp_plus_v2_pilot.py` — conditional
  cheap-pilot driver.
* `tests/test_w102_mbpp_plus_v2.py` — V2 unit tests (incl.
  schema bug regression).
* `docs/RUNBOOK_W102.md` — pre-commit contract.
