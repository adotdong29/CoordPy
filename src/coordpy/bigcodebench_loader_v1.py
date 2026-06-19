"""W110 / COO-9 — BigCodeBench loader V1 (SECOND contamination-RESISTANT lane).

BigCodeBench (BigCode, 2024-06; ``bigcode/bigcodebench`` on Hugging Face) is
the W110 second contamination-RESISTANT battlefield, selected over SWE-bench-
lite and LiveBench-coding under the locked W110 selection rule
(``docs/RUNBOOK_W110.md`` § 2):

* **Contamination-RESISTANT (S1).** Released 2024-06 (HF ``createdAt``
  2024-06-05; split ``v0.1.4`` ``lastModified`` 2025-04-30), AFTER the
  ≈2024-01 Llama-3.x training cutoff. The specific composed tasks + their
  hidden ``unittest`` tests are post-cutoff — time-anchored at the
  task+test level, analogous to LiveCodeBench's contest-date anchoring of
  familiar LeetCode primitives. The honest caveat
  (``W110-L-BIGCODEBENCH-RELEASE-DATE-RESISTANCE-NOT-CONTEST-DATE-CAP``): the
  library primitives BigCodeBench composes are obviously in-training; the
  RESISTANCE is the novel composition + 2024-06 release date, not strict
  contest-date anchoring.
* **Clean executor (S2).** Each row carries a deterministic
  ``unittest.TestCase`` ``test``; the executor (``bigcodebench_executor_v1``)
  runs it in a fresh ``-I`` subprocess — no Docker, no LLM judge.
* **Same-budget (S3).** A single ``def {entry_point}(...)`` completion graded
  at K=5 byte-exact — byte-identical A0/A1/B mechanism to W89/W105/W108/W109.
* **Genuinely different (S4).** Library-composition tasks authored by the
  BigCode project — NOT the W108 LiveCodeBench contest slice, NOT a disjoint
  LCB window, NOT APPS.

Real ``bigcode/bigcodebench`` row schema (confirmed via HF datasets-server
probe; binds here):

* ``task_id`` — stable id (e.g. ``BigCodeBench/0``).
* ``complete_prompt`` — signature + imports + docstring spec (model-visible).
* ``code_prompt`` — the bare signature + imports stub (no docstring).
* ``canonical_solution`` — the gold function BODY (the bench appends it to the
  prompt to form the gold program; see ``canonical_program_v1``).
* ``test`` — Python source defining ``class TestCases(unittest.TestCase)`` that
  references the entry point. The deterministic, no-LLM-judge oracle.
* ``entry_point`` — the function the test calls (e.g. ``task_func``).
* ``libs`` — declared third-party libraries (used for slice stratification and
  the gold-green dependency story; see ``docs/RUNBOOK_W110.md`` § 3).

Honest scope (W110)
-------------------

* ``W110-L-BIGCODEBENCH-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP`` — the exact
  field names MUST be confirmed against the live corpus at operator-fetch time;
  until then the loader runs only offline schema-shape self-checks and REFUSES
  on mismatch (the W102 silent-degeneration guard).
* ``W110-L-BIGCODEBENCH-GOLD-GREEN-SUBSET-ONLY-CAP`` — the bench scores only
  the ``gold_green`` subset (problems whose ``canonical_solution`` passes its
  own ``test`` in the run environment). Gold-green is an executor-environment
  property (missing/flaky deps), NOT a model outcome (A0/A1/B have not run), so
  it is anti-cheat-safe — the W108 ``func_name``-resolved / W109 wrapper-
  tolerance analogue.
* ``W110-L-BIGCODEBENCH-RELEASE-DATE-RESISTANCE-NOT-CONTEST-DATE-CAP`` — see
  above; the resistance is release-date + novel-composition, not contest-date.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
from typing import Any

W110_BIGCODEBENCH_LOADER_V1_SCHEMA_VERSION: str = (
    "coordpy.bigcodebench_loader_v1.v1")

BIGCODEBENCH_HF_DATASET: str = "bigcode/bigcodebench"
BIGCODEBENCH_DEFAULT_SPLIT: str = "v0.1.4"

BIGCODEBENCH_REQUIRED_FIELDS: tuple[str, ...] = (
    "complete_prompt", "code_prompt", "canonical_solution", "test",
    "entry_point")
BIGCODEBENCH_ID_FIELD_ALIASES: tuple[str, ...] = ("task_id", "id")


class BigCodeBenchCorpusError(RuntimeError):
    """Raised LOUDLY when the BigCodeBench corpus cannot be loaded / verified /
    schema-matched — never degrade a run on a malformed/unpinned corpus."""


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class BigCodeBenchProblemV1:
    task_id: str
    complete_prompt: str   # signature + imports + docstring (model-visible)
    code_prompt: str       # bare signature + imports stub
    canonical_solution: str  # gold BODY (appended to prompt -> gold program)
    test: str              # unittest.TestCase source (the oracle)
    entry_point: str       # function the test calls (e.g. task_func)
    libs: tuple[str, ...]  # declared third-party libs

    def prompt_cid(self) -> str:
        return hashlib.sha256(
            self.complete_prompt.encode("utf-8")).hexdigest()

    def n_libs(self) -> int:
        return len(self.libs)

    def problem_cid(self) -> str:
        return _sha256_hex({
            "task_id": str(self.task_id),
            "prompt_sha256": self.prompt_cid(),
            "entry_point": str(self.entry_point),
            "test_sha256": hashlib.sha256(
                self.test.encode("utf-8")).hexdigest(),
        })


def canonical_program_v1(problem: BigCodeBenchProblemV1) -> str:
    """The gold program = ``complete_prompt`` (signature + docstring) followed
    by ``canonical_solution`` (the body) — the BigCodeBench convention for the
    ``complete`` split. Used by the preflight gold-green check."""
    prefix = problem.complete_prompt
    if not prefix.endswith("\n"):
        prefix = prefix + "\n"
    return prefix + problem.canonical_solution


def _default_cache_path() -> str:
    return os.environ.get(
        "COORDPY_BIGCODEBENCH_CACHE",
        os.path.expanduser("~/.cache/coordpy/bigcodebench-v0_1_4.jsonl"))


def _resolve_id(row: dict) -> str:
    for k in BIGCODEBENCH_ID_FIELD_ALIASES:
        v = row.get(k)
        if v not in (None, ""):
            return str(v)
    return ""


def assert_bigcodebench_pinned(*, expected_sha256: str | None) -> None:
    """Refuse to operate without an explicit SHA pin (W93 discipline)."""
    pin = expected_sha256 or os.environ.get(
        "BIGCODEBENCH_TRUSTED_SHA256_OVERRIDE")
    if not pin:
        raise BigCodeBenchCorpusError(
            "No BigCodeBench SHA-256 pin provided (arg expected_sha256 or env "
            "BIGCODEBENCH_TRUSTED_SHA256_OVERRIDE). Refusing an unpinned "
            "corpus — set the pin at operator-fetch time after confirming the "
            "bigcode/bigcodebench artifact SHA.")


def validate_row_schema(row: dict) -> tuple[bool, str]:
    """Offline schema-shape check for ONE row (refuse-on-mismatch)."""
    if not isinstance(row, dict):
        return False, "row is not a JSON object"
    for f in BIGCODEBENCH_REQUIRED_FIELDS:
        if f not in row:
            return False, (
                f"missing required field {f!r}; confirm the live "
                "bigcode/bigcodebench schema (W110 loader cap)")
    if _resolve_id(row) == "":
        return False, f"no stable id under any of {BIGCODEBENCH_ID_FIELD_ALIASES}"
    if not str(row.get("entry_point") or "").strip():
        return False, "empty entry_point"
    if not str(row.get("test") or "").strip():
        return False, "empty test source"
    return True, "ok"


def _coerce_libs(raw: Any) -> tuple[str, ...]:
    """Coerce the ``libs`` field to a tuple of names. The upstream parquet
    stores it as a Python-repr STRING (single-quoted, e.g.
    ``"['random', 'itertools']"``), so JSON parsing alone fails — fall back to
    ``ast.literal_eval`` (and finally a comma split)."""
    if isinstance(raw, (list, tuple)):
        return tuple(str(x) for x in raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") or s.startswith("("):
            for parser in (json.loads, ast.literal_eval):
                try:
                    v = parser(s)
                    if isinstance(v, (list, tuple)):
                        return tuple(str(x) for x in v)
                except Exception:  # noqa: BLE001
                    continue
        return tuple(x for x in (p.strip() for p in s.split(",")) if x)
    return ()


def parse_bigcodebench(
        raw: bytes, *, max_problems: int | None = None,
) -> tuple[BigCodeBenchProblemV1, ...]:
    """Parse BigCodeBench JSONL bytes into ``BigCodeBenchProblemV1`` records.

    REFUSES on any row whose top-level schema does not match (no silent
    degradation — the W102 lesson)."""
    body = raw.decode("utf-8", errors="replace")
    out: list[BigCodeBenchProblemV1] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        ok, reason = validate_row_schema(row)
        if not ok:
            raise BigCodeBenchCorpusError(
                f"BigCodeBench row schema mismatch: {reason}. Refusing to "
                "degrade silently (W102 lesson).")
        out.append(BigCodeBenchProblemV1(
            task_id=_resolve_id(row),
            complete_prompt=str(row.get("complete_prompt") or ""),
            code_prompt=str(row.get("code_prompt") or ""),
            canonical_solution=str(row.get("canonical_solution") or ""),
            test=str(row.get("test") or ""),
            entry_point=str(row.get("entry_point") or ""),
            libs=_coerce_libs(row.get("libs"))))
        if max_problems is not None and len(out) >= int(max_problems):
            break
    return tuple(out)


def is_bigcodebench_cached(*, cache_path: str | None = None) -> bool:
    path = cache_path or _default_cache_path()
    return os.path.exists(path) and os.path.getsize(path) > 0


def load_bigcodebench_v1(
        *, cache_path: str | None = None,
        expected_sha256: str | None = None,
        max_problems: int | None = None,
) -> tuple[BigCodeBenchProblemV1, ...]:
    """Load + SHA-verify a pinned BigCodeBench JSONL from the local cache and
    return its records. Does NOT fetch over the network — the operator fetches
    + pins first (W101/W102/W108/W109 operator-fetch pattern)."""
    assert_bigcodebench_pinned(expected_sha256=expected_sha256)
    path = cache_path or _default_cache_path()
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        raise BigCodeBenchCorpusError(
            f"BigCodeBench cache missing at {path}. Operator must fetch + pin "
            "the bigcode/bigcodebench artifact first (see "
            "scripts/fetch_w110_bigcodebench_corpus.py).")
    with open(path, "rb") as f:
        raw = f.read()
    actual = hashlib.sha256(raw).hexdigest()
    expected = (
        expected_sha256
        or os.environ.get("BIGCODEBENCH_TRUSTED_SHA256_OVERRIDE"))
    if str(expected).lower() != actual.lower():
        raise BigCodeBenchCorpusError(
            f"BigCodeBench SHA-256 mismatch: actual={actual} "
            f"expected={expected}. Refusing a possibly-tampered corpus.")
    return parse_bigcodebench(raw, max_problems=max_problems)


__all__ = [
    "W110_BIGCODEBENCH_LOADER_V1_SCHEMA_VERSION",
    "BIGCODEBENCH_HF_DATASET",
    "BIGCODEBENCH_DEFAULT_SPLIT",
    "BIGCODEBENCH_REQUIRED_FIELDS",
    "BigCodeBenchCorpusError",
    "BigCodeBenchProblemV1",
    "canonical_program_v1",
    "assert_bigcodebench_pinned",
    "validate_row_schema",
    "parse_bigcodebench",
    "is_bigcodebench_cached",
    "load_bigcodebench_v1",
]
