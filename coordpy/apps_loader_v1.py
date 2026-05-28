"""W108 / COO-9 — APPS (call-based / functional subset) loader V1 (BACKUP lane).

APPS (Hendrycks et al., 2021; ``codeparrot/apps`` on Hugging Face) is the
structural-pivot BACKUP battlefield held behind LiveCodeBench (W107 § 4.2;
RUNBOOK_W108 § 6).  It is NOT the primary battlefield: APPS is 2021 vintage and
therefore almost certainly inside the Llama-3.x training corpus, so its
contamination resistance (C7) is grade **C** — a high pass rate could be
memorisation, which would weaken any multi-agent-superiority claim built on
it.  LiveCodeBench's time-anchored ``release_vN`` (C7 = A) is why it leads.
APPS is built to *real* here so a future LiveCodeBench structural failure can
pivot IN-milestone (RUNBOOK_W108 § 6) without a paperwork milestone.

This loader attacks the **call-based (functional) subset only** — rows whose
``input_output`` JSON carries an ``fn_name`` (the LeetCode-style entry-point
form).  That is exactly the W89 read→solve→execute→reflect→repair
complete-function shape and gives a clean deterministic subprocess executor
(``apps_executor_v1``).  The much larger stdin/stdout subset (no ``fn_name``)
is a structurally different executor shape and is OUT OF SCOPE (counted, not
silently degraded) — the same functional-only discipline as the LiveCodeBench
loader.

Documented schema (``codeparrot/apps`` row).  Fields this loader depends on:

* ``problem_id`` — stable id.
* ``question`` — the model-visible statement.
* ``input_output`` — a JSON **string**; for call-based problems it decodes to
  ``{"fn_name": <str>, "inputs": [[args...], ...], "outputs": [expected, ...]}``.
* ``starter_code`` — optional signature.
* ``difficulty`` — introductory | interview | competition.
* ``url`` — provenance.

Honest scope (W108)
-------------------

* ``W108-L-APPS-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP`` — the exact field
  names + the ``input_output`` encoding (and whether ``outputs[i]`` is the
  bare value or a 1-element list wrapper) MUST be confirmed against the live
  ``codeparrot/apps`` corpus at operator-fetch time, exactly like the
  LiveCodeBench loader.  Until then the loader runs only offline schema-shape
  self-checks and REFUSES on mismatch (the W102 silent-degeneration guard).
* ``W108-L-APPS-LOADER-V1-CALL-BASED-SUBSET-ONLY-CAP`` — only the
  ``fn_name`` call-based subset is loaded; stdin/stdout problems are filtered
  and counted.
* ``W108-L-APPS-CONTAMINATION-EXPOSED-2021-VINTAGE-CAP`` — APPS predates the
  Llama-3.x cutoff; any APPS result is contamination-exposed and is BACKUP
  evidence only, never the publication-grade time-anchored claim surface.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Any

W108_APPS_LOADER_V1_SCHEMA_VERSION: str = "coordpy.apps_loader_v1.v1"

APPS_HF_DATASET: str = "codeparrot/apps"
APPS_KNOWN_DIFFICULTIES: tuple[str, ...] = (
    "introductory", "interview", "competition")

APPS_REQUIRED_FIELDS: tuple[str, ...] = ("question", "input_output")
APPS_ID_FIELD_ALIASES: tuple[str, ...] = ("problem_id", "id", "task_id")


class AppsCorpusError(RuntimeError):
    """Raised LOUDLY when the APPS corpus cannot be loaded / verified /
    schema-matched — never degrade a run on a malformed/unpinned corpus."""


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class AppsFunctionalTestV1:
    """One call-based test: positional args + expected return (upstream
    JSON encoding preserved as strings)."""

    args_repr: str    # JSON list of positional arguments
    output_repr: str  # JSON expected return (bare value or 1-elt wrapper)

    def cid(self) -> str:
        return _sha256_hex({"args": self.args_repr,
                            "output": self.output_repr})


@dataclasses.dataclass(frozen=True)
class AppsFunctionalProblemV1:
    problem_id: str
    question: str
    starter_code: str
    fn_name: str
    difficulty: str
    url: str
    tests: tuple[AppsFunctionalTestV1, ...]

    def prompt_cid(self) -> str:
        return hashlib.sha256(self.question.encode("utf-8")).hexdigest()

    def problem_cid(self) -> str:
        return _sha256_hex({
            "problem_id": str(self.problem_id),
            "prompt_sha256": self.prompt_cid(),
            "fn_name": str(self.fn_name),
            "n_tests": len(self.tests),
            "tests_sha256": _sha256_hex([t.cid() for t in self.tests]),
        })


def _default_cache_path() -> str:
    return os.environ.get(
        "COORDPY_APPS_CACHE",
        os.path.expanduser("~/.cache/coordpy/apps-test.jsonl"))


def _resolve_id(row: dict) -> str:
    for k in APPS_ID_FIELD_ALIASES:
        v = row.get(k)
        if v not in (None, ""):
            return str(v)
    return ""


def assert_apps_pinned(*, expected_sha256: str | None) -> None:
    """Refuse to operate without an explicit SHA pin (W93 discipline)."""
    pin = expected_sha256 or os.environ.get("APPS_TRUSTED_SHA256_OVERRIDE")
    if not pin:
        raise AppsCorpusError(
            "No APPS SHA-256 pin provided (arg expected_sha256 or env "
            "APPS_TRUSTED_SHA256_OVERRIDE). Refusing an unpinned corpus — "
            "set the pin at operator-fetch time after confirming the "
            "codeparrot/apps artifact SHA.")


def _parse_io(row: dict) -> dict | None:
    io = row.get("input_output")
    if isinstance(io, str):
        try:
            io = json.loads(io)
        except Exception:  # noqa: BLE001
            return None
    return io if isinstance(io, dict) else None


def validate_row_schema(row: dict) -> tuple[bool, str]:
    """Offline schema-shape check for ONE row (refuse-on-mismatch)."""
    if not isinstance(row, dict):
        return False, "row is not a JSON object"
    for f in APPS_REQUIRED_FIELDS:
        if f not in row:
            return False, (
                f"missing required field {f!r}; confirm the live "
                "codeparrot/apps schema (W108 APPS loader cap)")
    if _resolve_id(row) == "":
        return False, f"no stable id under any of {APPS_ID_FIELD_ALIASES}"
    return True, "ok"


def is_call_based_row(row: dict) -> bool:
    """A row is call-based (functional) iff its ``input_output`` decodes to a
    dict carrying a non-empty ``fn_name``."""
    io = _parse_io(row)
    return bool(io and str(io.get("fn_name") or "").strip())


def parse_call_based_subset(
        raw: bytes, *, max_problems: int | None = None,
) -> tuple[AppsFunctionalProblemV1, ...]:
    """Parse APPS JSONL bytes into the call-based (functional) subset.

    REFUSES on any row whose top-level schema does not match (no silent
    degradation).  stdin/stdout rows (no ``fn_name``) are filtered + counted.
    Each call-based test is ``(args_repr, output_repr)`` where ``inputs[i]``
    is a positional-arg list and ``outputs[i]`` the expected return."""
    body = raw.decode("utf-8", errors="replace")
    out: list[AppsFunctionalProblemV1] = []
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
            raise AppsCorpusError(
                f"APPS row schema mismatch: {reason}. Refusing to degrade "
                "silently (W102 lesson).")
        if not is_call_based_row(row):
            continue
        io = _parse_io(row) or {}
        fn_name = str(io.get("fn_name") or "")
        inputs = io.get("inputs") or []
        outputs = io.get("outputs") or []
        tests: list[AppsFunctionalTestV1] = []
        for i, args in enumerate(inputs):
            if i >= len(outputs):
                break
            tests.append(AppsFunctionalTestV1(
                args_repr=json.dumps(args),
                output_repr=json.dumps(outputs[i])))
        if not tests:
            continue
        out.append(AppsFunctionalProblemV1(
            problem_id=_resolve_id(row),
            question=str(row.get("question") or ""),
            starter_code=str(row.get("starter_code") or ""),
            fn_name=fn_name,
            difficulty=str(row.get("difficulty") or ""),
            url=str(row.get("url") or ""),
            tests=tuple(tests)))
        if max_problems is not None and len(out) >= int(max_problems):
            break
    return tuple(out)


def is_apps_cached(*, cache_path: str | None = None) -> bool:
    path = cache_path or _default_cache_path()
    return os.path.exists(path) and os.path.getsize(path) > 0


def load_apps_call_based_v1(
        *, cache_path: str | None = None,
        expected_sha256: str | None = None,
        max_problems: int | None = None,
) -> tuple[AppsFunctionalProblemV1, ...]:
    """Load + SHA-verify a pinned APPS JSONL from the local cache and return
    its call-based (functional) subset.  Does NOT fetch over the network —
    the operator fetches + pins first (W101/W102 operator-fetch pattern)."""
    assert_apps_pinned(expected_sha256=expected_sha256)
    path = cache_path or _default_cache_path()
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        raise AppsCorpusError(
            f"APPS cache missing at {path}. Operator must fetch + pin the "
            "codeparrot/apps artifact first (see "
            "scripts/run_w108_apps_preflight.py --print-fetch-playbook).")
    with open(path, "rb") as f:
        raw = f.read()
    actual = hashlib.sha256(raw).hexdigest()
    expected = (
        expected_sha256 or os.environ.get("APPS_TRUSTED_SHA256_OVERRIDE"))
    if str(expected).lower() != actual.lower():
        raise AppsCorpusError(
            f"APPS SHA-256 mismatch: actual={actual} expected={expected}. "
            "Refusing a possibly-tampered corpus.")
    return parse_call_based_subset(raw, max_problems=max_problems)


__all__ = [
    "W108_APPS_LOADER_V1_SCHEMA_VERSION",
    "APPS_HF_DATASET",
    "APPS_KNOWN_DIFFICULTIES",
    "APPS_REQUIRED_FIELDS",
    "AppsCorpusError",
    "AppsFunctionalTestV1",
    "AppsFunctionalProblemV1",
    "assert_apps_pinned",
    "validate_row_schema",
    "is_call_based_row",
    "parse_call_based_subset",
    "is_apps_cached",
    "load_apps_call_based_v1",
]
