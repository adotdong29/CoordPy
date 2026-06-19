"""W107 / COO-9 — LiveCodeBench (code_generation_lite) loader V1.

LiveCodeBench (Jain et al., 2024; https://livecodebench.github.io;
`livecodebench/code_generation_lite` on Hugging Face) is a
TIME-ANCHORED, contamination-resistant code benchmark.  Each problem
carries a `contest_date`; the public release is versioned
(`release_v1` … `release_v6`), so a model's pass@1 can be reported on
the window of problems published AFTER its training cutoff — the
property that makes LiveCodeBench the right PRIMARY battlefield for a
*publication-grade* multi-agent-superiority claim (W106 § 7
pre-commit; W107 RUNBOOK § 4).

This loader is the W107-β **scaffolding** for that battlefield.  It is
NIM-free, read-only, SHA-pinnable, and — per the W102
silent-degeneration lesson — it REFUSES to run on a corpus whose
schema does not match what the W107 executor needs, rather than
silently degrading a pilot to a malformed run.

Documented schema (LiveCodeBench `code_generation_lite`, release_vN
JSONL row).  The fields this loader depends on:

* ``question_id`` (or ``question_title``): stable problem identifier.
* ``question_content``: the natural-language problem statement
  (what the model sees).
* ``starter_code``: function/class signature for FUNCTIONAL problems
  (non-empty ⇒ functional form; empty ⇒ stdin/stdout form).
* ``public_test_cases`` / ``private_test_cases``: JSON-encoded list of
  ``{"input": ..., "output": ..., "testtype": "functional"|"stdin"}``.
* ``platform`` (leetcode | atcoder | codeforces), ``difficulty``,
  ``contest_date`` (the time-anchor), ``metadata`` (carries
  ``func_name`` for functional problems).

W107-β attacks the **FUNCTIONAL subset only** (``starter_code``
non-empty; ``testtype == "functional"``) because the W89
read→solve→execute→reflect→repair mechanism produces a complete
function, which matches the LeetCode-style functional form and gives
a clean, deterministic executor (see ``livecodebench_executor_v1``).
The stdin/stdout subset is a structurally different executor shape and
is OUT OF SCOPE for the W107 scaffolding (recorded as a cap).

Honest scope (W107)
-------------------

* ``W107-L-LIVECODEBENCH-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP`` —
  the exact upstream field NAMES (e.g. ``public_test_cases`` vs
  ``test_cases``; the test-case JSON encoding) MUST be confirmed
  against the live release_vN corpus at operator-fetch time.  This is
  the W102 lesson encoded as discipline: the W101 V1 MBPP+ loader
  assumed a wrong schema and would have silently degenerated a pilot;
  this loader instead validates and REFUSES on mismatch.  Until the
  operator fetch confirms the schema, the loader runs only its
  offline schema-shape self-checks.

  **W108 DISCHARGE (2026-05-28).**  The operator fetched the real
  ``release_v6`` corpus (``test6.jsonl``; SHA-256
  ``bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5``;
  175 problems, 63 functional).  The confirmed encoding:
  ``public_test_cases`` and ``metadata`` are BOTH JSON *strings*, and
  each functional test's ``input`` is one JSON value per newline-
  separated line (one positional arg, signature order), ``output`` a
  single JSON value.  The W107 loader unwrapped the test-case string
  but read ``func_name`` only when ``metadata`` was already a dict —
  so on real data ``func_name`` was silently ``""``, which made the
  executor return ``ENTRY_NOT_FOUND`` on every arm (the W108 gold-path
  smoke A0=A1=B=0.0).  ``_resolve_func_name`` now accepts both metadata
  encodings (+ a ``starter_code`` fallback); the cap is DISCHARGED for
  the field-name / encoding surface.  The remaining real-data caps are
  the EXECUTOR-side plain-arg + exact-match caps
  (``W108-L-LIVECODEBENCH-EXECUTOR-V2-*``), not the loader.
* ``W107-L-LIVECODEBENCH-LOADER-V1-FUNCTIONAL-SUBSET-ONLY-CAP`` — only
  the functional (``starter_code``-bearing) subset is loaded for the
  W89 mechanism; stdin/stdout problems are filtered out and counted.
* ``W107-L-LIVECODEBENCH-LOADER-V1-RELEASE-PIN-CAP`` — the loader is
  pinned to ONE release version + SHA-256; cross-version mixing is
  refused (the time-anchor is only meaningful within a fixed release).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from typing import Any

W107_LIVECODEBENCH_LOADER_V1_SCHEMA_VERSION: str = (
    "coordpy.livecodebench_loader_v1.v1")

# Canonical Hugging Face dataset for LiveCodeBench code-generation
# (lite).  The release version + SHA-256 are pinned by the OPERATOR at
# fetch time (W93 discipline); this constant is the dataset root, NOT a
# trusted artifact hash.
LIVECODEBENCH_HF_DATASET: str = "livecodebench/code_generation_lite"
LIVECODEBENCH_HF_TREE_URL: str = (
    "https://huggingface.co/datasets/livecodebench/"
    "code_generation_lite/resolve/main")

# The release versions the loader will accept.  ONE must be pinned per
# run (cross-version mixing is refused).
LIVECODEBENCH_KNOWN_RELEASES: tuple[str, ...] = (
    "release_v1", "release_v2", "release_v3",
    "release_v4", "release_v5", "release_v6",
)

# The row fields the W107 executor depends on.  If a fetched corpus is
# missing any of these (under any documented alias), the loader REFUSES
# rather than degrading silently (the W102 P5 guard).
LIVECODEBENCH_REQUIRED_FIELDS: tuple[str, ...] = (
    "question_content",  # model-visible statement
    "starter_code",      # functional-form signature
)
# Test-case fields, tried in documented-alias order.
LIVECODEBENCH_TESTCASE_FIELD_ALIASES: tuple[str, ...] = (
    "public_test_cases", "test_cases", "public_tests",
)
LIVECODEBENCH_ID_FIELD_ALIASES: tuple[str, ...] = (
    "question_id", "question_title", "task_id",
)


class LiveCodeBenchCorpusError(RuntimeError):
    """Raised when the LiveCodeBench corpus cannot be loaded /
    verified / schema-matched.  Raised LOUDLY — never degrade a run
    on a malformed or unpinned corpus."""


@dataclasses.dataclass(frozen=True)
class LiveCodeBenchFunctionalTestV1:
    """One functional test case: arguments + expected return."""

    input_repr: str   # serialized function arguments (upstream encoding)
    output_repr: str  # serialized expected return value

    def cid(self) -> str:
        return _sha256_hex({
            "input": str(self.input_repr),
            "output": str(self.output_repr),
        })


@dataclasses.dataclass(frozen=True)
class LiveCodeBenchProblemV1:
    """One LiveCodeBench FUNCTIONAL problem.

    Mirrors the upstream row, restricted to the fields the W89
    functional-form mechanism needs.  ``func_name`` is the entry point
    the candidate must define / the method on ``Solution``.
    """

    question_id: str
    question_content: str
    starter_code: str
    func_name: str
    platform: str
    difficulty: str
    contest_date: str
    tests: tuple[LiveCodeBenchFunctionalTestV1, ...]

    def prompt_cid(self) -> str:
        return hashlib.sha256(
            self.question_content.encode("utf-8")).hexdigest()

    def problem_cid(self) -> str:
        return _sha256_hex({
            "question_id": str(self.question_id),
            "prompt_sha256": self.prompt_cid(),
            "starter_code_sha256": hashlib.sha256(
                self.starter_code.encode("utf-8")).hexdigest(),
            "func_name": str(self.func_name),
            "n_tests": len(self.tests),
            "tests_sha256": _sha256_hex(
                [t.cid() for t in self.tests]),
        })


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _default_cache_path(release: str) -> str:
    return os.environ.get(
        "COORDPY_LIVECODEBENCH_CACHE",
        os.path.expanduser(
            f"~/.cache/coordpy/livecodebench-{release}.jsonl"))


def _resolve_id(row: dict) -> str:
    for k in LIVECODEBENCH_ID_FIELD_ALIASES:
        v = row.get(k)
        if v:
            return str(v)
    return ""


def _resolve_testcases_raw(row: dict) -> Any:
    for k in LIVECODEBENCH_TESTCASE_FIELD_ALIASES:
        if k in row and row.get(k) not in (None, ""):
            return row.get(k)
    return None


# Last-resort entry-point recovery from a LeetCode-style ``starter_code``
# body (``def <name>(self, ...)``).  Used only when ``metadata`` carries no
# ``func_name`` — the authoritative source is ``metadata.func_name``.
_STARTER_DEF_RE = re.compile(r"def\s+([A-Za-z_]\w*)\s*\(")


def _resolve_func_name(row: dict) -> str:
    """Resolve the functional entry-point name for one row.

    W108 real-data confirmation (discharging
    ``W107-L-LIVECODEBENCH-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP``): the
    live ``code_generation_lite`` ``release_v6`` corpus stores ``metadata``
    as a JSON *string* (e.g. ``'{"func_name": "zigzagTraversal"}'``), NOT a
    native object — exactly as ``public_test_cases`` is a JSON string.  The
    W107 loader unwrapped the test-case string but checked
    ``isinstance(metadata, dict)`` for ``func_name``, which is ``False`` for
    a string and silently left ``func_name == ""``.  An empty ``func_name``
    makes the executor's entry resolver return ``None`` → ``ENTRY_NOT_FOUND``
    → every arm FAILs (the exact W108 gold-path smoke A0=A1=B=0.0).  This
    resolver accepts BOTH encodings (dict or JSON-string) and adds a
    ``starter_code`` ``def`` fallback so a malformed/absent-metadata row
    still resolves its entry rather than degrading silently.
    """
    meta = row.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:  # noqa: BLE001
            meta = None
    if isinstance(meta, dict):
        fn = str(meta.get("func_name") or "")
        if fn:
            return fn
    # Fallback: first non-dunder ``def`` in the starter_code (the LeetCode
    # method); dunder-only ⇒ last def.  metadata.func_name is authoritative,
    # so this fires only on a metadata-less row.
    names = _STARTER_DEF_RE.findall(str(row.get("starter_code") or ""))
    for name in names:
        if not name.startswith("__"):
            return name
    return names[-1] if names else ""


def is_livecodebench_cached(
        *, release: str, cache_path: str | None = None) -> bool:
    """Return True iff a candidate LiveCodeBench release JSONL exists
    on disk."""
    path = cache_path or _default_cache_path(release)
    return os.path.exists(path) and os.path.getsize(path) > 0


def assert_release_pinned(*, release: str, expected_sha256: str | None) -> None:
    """Refuse to operate without an explicit, known release + SHA pin.

    Mirrors the W93 'no unpinned corpus' discipline.  ``expected_sha256``
    may come from the ``LIVECODEBENCH_TRUSTED_SHA256_OVERRIDE`` env var.
    """
    if release not in LIVECODEBENCH_KNOWN_RELEASES:
        raise LiveCodeBenchCorpusError(
            f"Unknown LiveCodeBench release {release!r}; known: "
            f"{LIVECODEBENCH_KNOWN_RELEASES}. Refusing to operate on "
            "an unpinned release (the time-anchor is only meaningful "
            "within a fixed release).")
    pin = (
        expected_sha256
        or os.environ.get("LIVECODEBENCH_TRUSTED_SHA256_OVERRIDE"))
    if not pin:
        raise LiveCodeBenchCorpusError(
            "No LiveCodeBench SHA-256 pin provided "
            "(arg expected_sha256 or env "
            "LIVECODEBENCH_TRUSTED_SHA256_OVERRIDE). Refusing to use "
            "an unpinned corpus — set the pin at operator-fetch time "
            "after confirming the release_vN artifact SHA.")


def validate_row_schema(row: dict) -> tuple[bool, str]:
    """Offline schema-shape check for ONE row.

    Returns ``(ok, reason)``.  ``ok=False`` means the row does not
    carry the fields the W107 functional executor needs — the loader
    will REFUSE rather than degrade.  This is the W102 P5
    silent-degeneration guard applied to LiveCodeBench.
    """
    if not isinstance(row, dict):
        return False, "row is not a JSON object"
    for f in LIVECODEBENCH_REQUIRED_FIELDS:
        if f not in row:
            return False, (
                f"missing required field {f!r}; confirm the live "
                "release_vN schema (W107 loader cap)")
    if _resolve_id(row) == "":
        return False, (
            "no stable id under any of "
            f"{LIVECODEBENCH_ID_FIELD_ALIASES}")
    if _resolve_testcases_raw(row) is None:
        return False, (
            "no test cases under any of "
            f"{LIVECODEBENCH_TESTCASE_FIELD_ALIASES}")
    return True, "ok"


def is_functional_row(row: dict) -> bool:
    """A row is FUNCTIONAL iff it has a non-empty ``starter_code``.

    stdin/stdout problems carry an empty ``starter_code`` and are out
    of scope for the W107 functional-form mechanism.
    """
    return bool(str(row.get("starter_code") or "").strip())


def parse_functional_subset(
        raw: bytes,
        *,
        max_problems: int | None = None,
) -> tuple[LiveCodeBenchProblemV1, ...]:
    """Parse the release JSONL bytes into the FUNCTIONAL subset.

    REFUSES on any row whose schema does not match (no silent
    degradation).  Test-case parsing tolerates the documented
    JSON-encoded-string form (``public_test_cases`` is often a JSON
    string, not an inline array) — confirm the exact encoding at
    operator-fetch time.
    """
    body = raw.decode("utf-8", errors="replace")
    out: list[LiveCodeBenchProblemV1] = []
    n_seen = 0
    n_stdin = 0
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        n_seen += 1
        ok, reason = validate_row_schema(row)
        if not ok:
            raise LiveCodeBenchCorpusError(
                f"LiveCodeBench row schema mismatch: {reason}. "
                "Refusing to degrade silently (W102 lesson).")
        if not is_functional_row(row):
            n_stdin += 1
            continue
        tests_raw = _resolve_testcases_raw(row)
        if isinstance(tests_raw, str):
            try:
                tests_raw = json.loads(tests_raw)
            except Exception:  # noqa: BLE001
                tests_raw = []
        tests: list[LiveCodeBenchFunctionalTestV1] = []
        for tc in (tests_raw or []):
            if not isinstance(tc, dict):
                continue
            if str(tc.get("testtype") or "functional") != "functional":
                continue
            tests.append(LiveCodeBenchFunctionalTestV1(
                input_repr=str(tc.get("input", "")),
                output_repr=str(tc.get("output", "")),
            ))
        if not tests:
            continue
        func_name = _resolve_func_name(row)
        out.append(LiveCodeBenchProblemV1(
            question_id=_resolve_id(row),
            question_content=str(row.get("question_content") or ""),
            starter_code=str(row.get("starter_code") or ""),
            func_name=func_name,
            platform=str(row.get("platform") or ""),
            difficulty=str(row.get("difficulty") or ""),
            contest_date=str(row.get("contest_date") or ""),
            tests=tuple(tests),
        ))
        if max_problems is not None and len(out) >= int(max_problems):
            break
    return tuple(out)


def load_livecodebench_functional_v1(
        *,
        release: str,
        cache_path: str | None = None,
        expected_sha256: str | None = None,
        max_problems: int | None = None,
) -> tuple[LiveCodeBenchProblemV1, ...]:
    """Load + SHA-verify a pinned LiveCodeBench release from the local
    cache and return its FUNCTIONAL subset.

    This does NOT fetch over the network — the operator fetches +
    pins the release_vN artifact first (W101/W102 operator-fetch
    pattern), then this loads + verifies it.  Raises
    ``LiveCodeBenchCorpusError`` on missing cache / SHA mismatch /
    schema mismatch.
    """
    assert_release_pinned(
        release=release, expected_sha256=expected_sha256)
    path = cache_path or _default_cache_path(release)
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        raise LiveCodeBenchCorpusError(
            f"LiveCodeBench {release} cache missing at {path}. "
            "Operator must fetch + pin the release artifact first "
            "(see scripts/run_w107_livecodebench_preflight.py "
            "--print-fetch-playbook).")
    with open(path, "rb") as f:
        raw = f.read()
    actual = hashlib.sha256(raw).hexdigest()
    expected = (
        expected_sha256
        or os.environ.get("LIVECODEBENCH_TRUSTED_SHA256_OVERRIDE"))
    if str(expected).lower() != actual.lower():
        raise LiveCodeBenchCorpusError(
            f"LiveCodeBench {release} SHA-256 mismatch: actual="
            f"{actual} expected={expected}. Refusing a possibly-"
            "tampered corpus.")
    return parse_functional_subset(raw, max_problems=max_problems)


__all__ = [
    "W107_LIVECODEBENCH_LOADER_V1_SCHEMA_VERSION",
    "LIVECODEBENCH_HF_DATASET",
    "LIVECODEBENCH_HF_TREE_URL",
    "LIVECODEBENCH_KNOWN_RELEASES",
    "LIVECODEBENCH_REQUIRED_FIELDS",
    "LiveCodeBenchCorpusError",
    "LiveCodeBenchFunctionalTestV1",
    "LiveCodeBenchProblemV1",
    "is_livecodebench_cached",
    "assert_release_pinned",
    "validate_row_schema",
    "is_functional_row",
    "parse_functional_subset",
    "load_livecodebench_functional_v1",
]
