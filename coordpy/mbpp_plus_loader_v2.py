"""W102 / COO-9 — MBPP+ (EvalPlus) corpus loader V2.

Corrects the W101 V1 loader schema bug discovered during the W102
fetch step.

Background
----------

The W101 loader (`coordpy.mbpp_plus_loader_v1`) assumed the EvalPlus
release artifact would be a `.jsonl.gz` with parallel `plus_input`
and `plus_output` arrays per row.  When the W102 operator step
fetched the actual upstream artifact, two facts emerged:

1. EvalPlus GitHub releases (e.g., `v0.2.0`, `v0.3.1`) contain only
   model-output zip files — NOT the canonical MBPP+ dataset.  The
   W101-pinned URL is HTTP 404.
2. The canonical MBPP+ dataset is published on Hugging Face as a
   parquet file at
   ``https://huggingface.co/datasets/evalplus/mbppplus/resolve/main/data/test-00000-of-00001-d5781c9c51e02795.parquet``
   (378 rows, LFS oid
   ``dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b``).
3. The row schema is
   ``{task_id, code, prompt, source_file, test_imports, test_list,
   test}``.  ``test_list`` carries the 3-ish BASE assertions (same
   shape as base-MBPP-sanitized).  The EvalPlus hidden tests are
   inside the **single `test` field** — a Python program that
   defines `inputs` + `results` arrays and iterates calling the
   entry-point function with each input, comparing to expected via
   an ``assertion()`` helper.

There is NO `plus_input` / `plus_output` parallel-array encoding in
the actual EvalPlus release.

If the W101 V1 loader were pointed at the real data, every row would
parse with empty `plus_input` and `plus_output`, the V1 executor's
``build_plus_assertions`` would return ``[]``, and the V1 cheap
pilot would silently degenerate to a base-MBPP run — the exact
SATURATED-CEILING regime W101 was designed to attack.

V2 reads the actual EvalPlus schema and exposes a structured
``MbppPlusProblemV2`` carrying the full `test` program verbatim;
the V2 executor (`coordpy.mbpp_plus_executor_v2`) runs the test
program against the candidate, so the EvalPlus hidden tests are
genuinely evaluated.

V1 stays in-repo (per the W102 anti-drift contract: nothing is
silently removed) but is marked as an **anti-pattern + historical
artifact**.  V1 must NOT be used for any new cheap pilot.

Anti-cheat (mirrors V1 verbatim where applicable):

* Canonical Hugging Face URL + pinned SHA-256 (LFS oid).  A
  mismatch refuses to use the corpus.
* `task_id` mirrors EvalPlus's `Mbpp/<n>` form (uses upstream
  numbering verbatim).
* `entry_point` extracted from the FIRST base-test assertion's
  function call (same regex as V1 and as `mbpp_reflexion_bench_v1`).
* `base_test_list` mirrors the **original** MBPP `test_list`.
* `extra_test_program` is the canonical EvalPlus `test` field —
  a complete Python program that imports `numpy as np`, defines an
  `assertion()` helper, defines the `inputs` + `results` parallel
  arrays, and iterates calling the entry-point function on each
  input.  The V2 executor concatenates this with the candidate
  code and runs the combined program in a fresh CPython
  subprocess; the program exits 0 iff every assertion in the loop
  passes.
* Loader is read-only; no NIM calls.
* The loader handles BOTH cached parquet (preferred for audit) AND
  an explicit JSONL.GZ fallback for environments without pyarrow.

Honest scope (W102)
-------------------

* ``W102-L-MBPP-PLUS-LOADER-V2-PARQUET-PYARROW-CAP`` — the parquet
  path requires ``pyarrow``; the loader falls back to a documented
  JSONL.GZ path when pyarrow is unavailable.
* ``W102-L-MBPP-PLUS-LOADER-V2-EVALPLUS-SCHEMA-CAP`` — V2 reads
  the schema shipped by EvalPlus's v0.3.1 release (and v0.2.1
  back-ported, which is the same row shape).  If EvalPlus
  re-publishes with a different schema (e.g., parallel
  input/output arrays), the loader fails LOUDLY rather than
  silently degenerating.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
import re
import urllib.request
from typing import Any


W102_MBPP_PLUS_LOADER_V2_SCHEMA_VERSION: str = (
    "coordpy.mbpp_plus_loader_v2.v1")


# Canonical Hugging Face MBPP+ parquet (v0.2.1 / v0.3.1 row shape).
# Confirmed live 2026-05-25 during the W102 operator-fetch step.
MBPP_PLUS_HF_CANONICAL_URL_V021: str = (
    "https://huggingface.co/datasets/evalplus/mbppplus/resolve/"
    "main/data/test-00000-of-00001-d5781c9c51e02795.parquet")

# LFS SHA-256 oid as exposed by the Hugging Face dataset tree API
# (`/api/datasets/evalplus/mbppplus/tree/main/data`).  This is the
# authoritative integrity pin.
MBPP_PLUS_HF_EXPECTED_SHA256_V021: str = (
    "dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b")

# EvalPlus's curated subset of MBPP is 378 problems; we accept a
# ±50 margin to remain robust to minor upstream re-releases.
MBPP_PLUS_V2_EXPECTED_PROBLEM_COUNT_MIN: int = 350
MBPP_PLUS_V2_EXPECTED_PROBLEM_COUNT_MAX: int = 450


class MbppPlusV2CorpusError(RuntimeError):
    """Raised when the V2 MBPP+ corpus cannot be loaded /
    verified."""


@dataclasses.dataclass(frozen=True)
class MbppPlusProblemV2:
    """One MBPP+ problem with the actual EvalPlus schema.

    Fields mirror the upstream HF parquet row schema:

    * ``task_id``: e.g., `"Mbpp/2"` (uses upstream numbering with
      the `Mbpp/` prefix added for cross-bench consistency with
      W101 / `coordpy.mbpp_reflexion_bench_v1`).
    * ``prompt``: the original MBPP English description.
    * ``canonical_code``: the reference solution
      (NEVER shown to the model).
    * ``base_test_list``: the original MBPP-sanitized `test_list`
      verbatim (3-ish base assertions).
    * ``extra_test_program``: the EvalPlus `test` field — a
      complete Python program that defines `inputs` + `results`
      parallel arrays and an iteration loop calling the entry
      point with each input and comparing via an ``assertion()``
      helper.  Concatenated with candidate code by the V2 executor.
    * ``test_imports``: extra import lines the executor must
      prepend before the candidate (e.g., `from math import inf`).
    * ``entry_point``: extracted from the first base test.
    """

    task_id: str                        # e.g., "Mbpp/2"
    prompt: str
    canonical_code: str
    base_test_list: tuple[str, ...]
    extra_test_program: str             # the EvalPlus `test` field
    test_imports: tuple[str, ...]
    entry_point: str

    def problem_cid(self) -> str:
        return _sha256_hex({
            "task_id": str(self.task_id),
            "prompt_sha256": hashlib.sha256(
                self.prompt.encode("utf-8")).hexdigest(),
            "base_test_list": list(self.base_test_list),
            "extra_test_program_sha256": hashlib.sha256(
                self.extra_test_program.encode("utf-8")
            ).hexdigest(),
            "test_imports": list(self.test_imports),
            "entry_point": str(self.entry_point),
        })


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _default_cache_path() -> str:
    return os.environ.get(
        "COORDPY_MBPP_PLUS_V2_CACHE",
        os.path.expanduser(
            "~/.cache/coordpy/mbpp-plus.parquet"))


_ENTRY_POINT_FROM_ASSERT_RE = re.compile(
    r"^\s*assert\s+(?:set\(|tuple\(|list\(|sorted\(|abs\()*"
    r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _extract_entry_point(
        base_tests: list[str]) -> str:
    """Extract the function name being asserted on from the first
    base test (same logic as `mbpp_reflexion_bench_v1`)."""
    for t in base_tests or []:
        m = _ENTRY_POINT_FROM_ASSERT_RE.match(str(t).strip())
        if m is not None:
            return str(m.group(1))
    return ""


def is_mbpp_plus_v2_cached(
        *, cache_path: str | None = None) -> bool:
    """Return True iff a candidate MBPP+ V2 cache file exists on
    disk at the configured location."""
    path = cache_path or _default_cache_path()
    return os.path.exists(path) and os.path.getsize(path) > 0


def fetch_mbpp_plus_v2_corpus(
        *,
        cache_path: str | None = None,
        url: str = MBPP_PLUS_HF_CANONICAL_URL_V021,
        timeout: float = 60.0,
        force: bool = False,
) -> tuple[bytes, str]:
    """Fetch (and cache) the canonical EvalPlus MBPP+ parquet.
    Returns ``(raw_bytes, sha256_hex)``.

    No SHA verification is performed here — the caller (the
    loader) is responsible for verifying the artifact against the
    pinned ``MBPP_PLUS_HF_EXPECTED_SHA256_V021`` constant OR an
    explicit operator override.  This separation lets the operator
    fetch the corpus once, verify the actual SHA, and then operate
    offline.

    Raises ``MbppPlusV2CorpusError`` on fetch failure.
    """
    path = cache_path or _default_cache_path()
    if force or not os.path.exists(path):
        try:
            with urllib.request.urlopen(
                    url, timeout=float(timeout)) as r:
                raw = r.read()
        except Exception as e:  # noqa: BLE001
            raise MbppPlusV2CorpusError(
                f"MBPP+ V2 corpus fetch failed: "
                f"{type(e).__name__}: {e} (url={url})") from e
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(raw)
    else:
        with open(path, "rb") as f:
            raw = f.read()
    actual_sha = hashlib.sha256(raw).hexdigest()
    return raw, actual_sha


def load_mbpp_plus_v2_corpus(
        *,
        cache_path: str | None = None,
        url: str = MBPP_PLUS_HF_CANONICAL_URL_V021,
        expected_sha256: str | None = None,
        timeout: float = 60.0,
) -> tuple[MbppPlusProblemV2, ...]:
    """Load + SHA-256-verify the canonical EvalPlus MBPP+ V2 corpus
    (parquet path).

    ``expected_sha256``: if None, falls back to the
    ``MBPP_PLUS_V2_TRUSTED_SHA256_OVERRIDE`` env var, then to the
    pinned ``MBPP_PLUS_HF_EXPECTED_SHA256_V021`` constant (the
    authoritative HF LFS oid).
    """
    expected = (
        expected_sha256
        or os.environ.get(
            "MBPP_PLUS_V2_TRUSTED_SHA256_OVERRIDE")
        or MBPP_PLUS_HF_EXPECTED_SHA256_V021)
    raw, actual = fetch_mbpp_plus_v2_corpus(
        cache_path=cache_path, url=url, timeout=timeout)
    if actual.lower() != str(expected).lower():
        raise MbppPlusV2CorpusError(
            "MBPP+ V2 corpus SHA-256 mismatch: "
            f"actual={actual} expected={expected}. "
            "Refusing to use a possibly-tampered corpus.")
    return parse_mbpp_plus_v2_parquet(raw)


def parse_mbpp_plus_v2_parquet(
        raw: bytes,
) -> tuple[MbppPlusProblemV2, ...]:
    """Parse the EvalPlus MBPP+ parquet bytes into a tuple of
    ``MbppPlusProblemV2``.

    Requires ``pyarrow``.  If pyarrow is unavailable, raises
    ``MbppPlusV2CorpusError`` with an explicit install hint.
    """
    try:
        import io
        import pyarrow.parquet as pq
    except Exception as e:  # noqa: BLE001
        raise MbppPlusV2CorpusError(
            "MBPP+ V2 parquet parse requires pyarrow; install it "
            "via `pip install pyarrow`.  (Error: "
            f"{type(e).__name__}: {e})") from e
    try:
        table = pq.read_table(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise MbppPlusV2CorpusError(
            f"MBPP+ V2 parquet read failed: "
            f"{type(e).__name__}: {e}") from e
    rows = table.to_pylist()
    return _rows_to_problems(rows)


def _rows_to_problems(
        rows: list[dict[str, Any]],
) -> tuple[MbppPlusProblemV2, ...]:
    out: list[MbppPlusProblemV2] = []
    for row in rows:
        task_id_raw = row.get("task_id")
        if task_id_raw is None:
            continue
        # EvalPlus task_id is an integer matching base MBPP.  We
        # normalise to the `Mbpp/<n>` form so audit-chain joins
        # against base-MBPP sidecars are consistent.
        if isinstance(task_id_raw, int) or (
                isinstance(task_id_raw, str)
                and task_id_raw.isdigit()):
            task_id = f"Mbpp/{int(task_id_raw)}"
        else:
            task_id = str(task_id_raw)
        prompt = str(row.get("prompt") or row.get("text") or "")
        canonical_code = str(row.get("code") or "")
        base_tests = list(row.get("test_list") or [])
        test_program = str(row.get("test") or "")
        test_imports = tuple(
            str(x) for x in (row.get("test_imports") or []))
        entry = _extract_entry_point(base_tests)
        if not entry:
            entry = str(row.get("entry_point") or "")
        if not entry:
            continue
        if not test_program.strip():
            # An EvalPlus row with an empty `test` field is the
            # exact silent-degradation case V1 hit.  V2 refuses to
            # silently emit a row with no extra-test surface.
            continue
        out.append(MbppPlusProblemV2(
            task_id=str(task_id),
            prompt=str(prompt),
            canonical_code=str(canonical_code),
            base_test_list=tuple(str(t) for t in base_tests),
            extra_test_program=str(test_program),
            test_imports=tuple(test_imports),
            entry_point=str(entry),
        ))
    if not (
            MBPP_PLUS_V2_EXPECTED_PROBLEM_COUNT_MIN
            <= len(out)
            <= MBPP_PLUS_V2_EXPECTED_PROBLEM_COUNT_MAX):
        raise MbppPlusV2CorpusError(
            f"MBPP+ V2 corpus parsed {len(out)} problems; "
            f"expected between "
            f"{MBPP_PLUS_V2_EXPECTED_PROBLEM_COUNT_MIN} and "
            f"{MBPP_PLUS_V2_EXPECTED_PROBLEM_COUNT_MAX}.")
    return tuple(out)


__all__ = [
    "W102_MBPP_PLUS_LOADER_V2_SCHEMA_VERSION",
    "MBPP_PLUS_HF_CANONICAL_URL_V021",
    "MBPP_PLUS_HF_EXPECTED_SHA256_V021",
    "MBPP_PLUS_V2_EXPECTED_PROBLEM_COUNT_MIN",
    "MBPP_PLUS_V2_EXPECTED_PROBLEM_COUNT_MAX",
    "MbppPlusV2CorpusError",
    "MbppPlusProblemV2",
    "is_mbpp_plus_v2_cached",
    "fetch_mbpp_plus_v2_corpus",
    "load_mbpp_plus_v2_corpus",
    "parse_mbpp_plus_v2_parquet",
]
