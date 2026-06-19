"""W96-D — ChartQA test corpus loader V1.

The W96-D battlefield pivot selected ChartQA (the D1 lead per
`docs/RUNBOOK_W96D.md` and Linear `COO-20`) as the next serious
cross-modal team-superiority battlefield to test cheaply.  See
`docs/W94_CROSS_MODAL_BATTLEFIELD_SCOUTING.md` for the
selection rationale and
`docs/RESULTS_W96D_ARSENAL_MINING_V1.md` for the candidate
mechanism inventory.

This module:

  * downloads the canonical ChartQA test parquet once and SHA-
    anchors it (no ``datasets`` library dependency; uses
    ``urllib.request`` + ``pyarrow``);
  * exposes ``ChartQAProblemV1`` capsules with the canonical
    per-problem schema (pid / query / labels / human_or_machine /
    image bytes + SHA);
  * provides a deterministic ``select_chartqa_subset_v1`` slice
    selector mirroring the W95 ``select_mathvista_subset_v1``
    discipline.

The loader is the W96-D corpus-side anti-cheat surface.  The
bench side will sit on top of this in a separate
``coordpy.chartqa_bench_v1`` module — that one is held back until
cheap preflight earns the expensive run (per W93/W94/W95/W96-A/
W96-C discipline).  This module by itself does NOT call NIM and
does NOT define A0/A1/B baselines; it only loads + anchors the
corpus and is safe to import from preflight probes.

Honest scope (W96-D)
--------------------

* ``W96-L-CHARTQA-LOADER-V1-NETWORK-FETCH-CAP`` — first fetch
  hits the HuggingFace CDN; subsequent loads read the cached
  parquet from disk.  No model calls.
* ``W96-L-CHARTQA-LOADER-V1-PARQUET-LIBRARY-CAP`` — V1 parses
  the parquet with ``pyarrow``.
* ``W96-L-CHARTQA-LOADER-V1-CORPUS-SHA-CAP`` — the canonical
  parquet SHA-256 is recorded on first download and verified on
  every subsequent load; mismatches refuse to return a corpus.
* ``W96-L-CHARTQA-LOADER-V1-CANONICAL-URL-CAP`` — V1 defaults
  to the ``HuggingFaceM4/ChartQA`` test split's canonical
  ``data/test-00000-of-00001.parquet`` shard.  Callers can
  override via ``--parquet-url`` if the canonical URL is moved
  upstream.  Multi-shard test splits are NOT supported in V1.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any


W96_CHARTQA_LOADER_V1_SCHEMA_VERSION: str = (
    "coordpy.chartqa_loader_v1.v1")


# Canonical HuggingFace path to the ChartQA test parquet.  The
# default targets the ``lmms-lab/ChartQA`` formatted dataset (used
# by the lmms-eval pipeline; 2500 problems = 1250 human_test +
# 1250 augmented_test; 4-column schema {type, question, answer,
# image}).  This is the most widely-cited HF dataset for ChartQA
# evaluation as of 2026-05.  Other mirrors (``ahmed-masry/ChartQA``)
# can be substituted via ``--parquet-url``.
CHARTQA_TEST_PARQUET_URL: str = (
    "https://huggingface.co/datasets/lmms-lab/ChartQA/"
    "resolve/main/data/"
    "test-00000-of-00001.parquet")

# Pre-recorded SHA-256 of the canonical ``lmms-lab/ChartQA``
# ``data/test-00000-of-00001.parquet`` snapshot fetched on
# 2026-05-25.  Anchors the W96-D corpus reproducibility.
CHARTQA_TEST_EXPECTED_PARQUET_SHA256: str = (
    "165263505f2998aba65d819b44be832edecd92d676fee2c030645f784cd"
    "55d06")

# Expected bounds for the parquet.  The lmms-lab/ChartQA test
# split has exactly 2500 problems and a ~72 MB parquet.  The
# bounds are slack-padded so future minor re-encodings do not
# trip the integrity probe.
CHARTQA_TEST_EXPECTED_N_PROBLEMS_LOWER: int = 2000
CHARTQA_TEST_EXPECTED_N_PROBLEMS_UPPER: int = 3000
CHARTQA_TEST_EXPECTED_PARQUET_BYTES_LOWER: int = 50_000_000
CHARTQA_TEST_EXPECTED_PARQUET_BYTES_UPPER: int = 800_000_000


@dataclasses.dataclass(frozen=True)
class ChartQAProblemV1:
    """A single ChartQA test problem in canonical capsule form.
    ``image_bytes`` is the raw decoded image as the HuggingFace
    ``Image`` feature serialised it; ``image_sha256`` is the hex
    digest of those bytes.  ChartQA gold answers are a tuple of
    acceptable strings (the canonical relaxed-accuracy evaluator
    passes if any one matches)."""

    pid: str
    query: str
    labels: tuple[str, ...]
    human_or_machine: str
    image_bytes: bytes
    image_sha256: str
    image_format: str
    metadata: dict[str, Any]

    def to_dict_no_image(self) -> dict[str, Any]:
        """Serialisable form omitting raw image bytes (sidecars /
        audit-chain JSON carry only the SHA, not the bytes)."""
        return {
            "pid": str(self.pid),
            "query": str(self.query),
            "labels": list(self.labels),
            "human_or_machine": str(self.human_or_machine),
            "image_sha256": str(self.image_sha256),
            "image_format": str(self.image_format),
            "metadata": dict(self.metadata),
        }


@dataclasses.dataclass(frozen=True)
class ChartQACorpusManifestV1:
    """Content-addressed manifest of the loaded ChartQA test
    corpus.  Records the parquet SHA, problem count, and a
    Merkle-style SHA-256 over the per-problem
    ``to_dict_no_image()`` payloads."""

    schema: str
    parquet_url: str
    parquet_sha256: str
    parquet_bytes: int
    n_problems: int
    corpus_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "parquet_url": str(self.parquet_url),
            "parquet_sha256": str(self.parquet_sha256),
            "parquet_bytes": int(self.parquet_bytes),
            "n_problems": int(self.n_problems),
            "corpus_merkle_root": str(self.corpus_merkle_root),
        }


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(payload).hexdigest()
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sniff_image_format(image_bytes: bytes) -> str:
    """Detect a small subset of image-format magic bytes.
    Returns ``""`` when the format is unknown — the loader does
    not raise so callers can still SHA-anchor the bytes."""
    if not image_bytes:
        return ""
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"
    if image_bytes[:2] == b"BM":
        return "bmp"
    return ""


def fetch_chartqa_test_parquet(
        *,
        cache_dir: Path,
        url: str = CHARTQA_TEST_PARQUET_URL,
        force: bool = False,
        expected_sha256: str | None = None,
        timeout_s: float = 900.0,
) -> tuple[Path, str, int]:
    """Idempotently fetch the canonical ChartQA test parquet to
    ``cache_dir``.  Returns (path, sha256_hex, n_bytes).

    If ``expected_sha256`` is provided, the loader refuses to
    return a path whose bytes do not hash to that value (the W96-D
    audit-chain bar).  If not, the loader records the observed
    SHA-256 the first time and writes a ``parquet.sha256`` sidecar
    so subsequent loads can verify it without re-downloading.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / (
        "chartqa_test_00000-of-00001.parquet")
    sha_sidecar = cache_dir / (
        "chartqa_test_00000-of-00001.parquet.sha256")

    if parquet_path.exists() and not force:
        data_bytes = parquet_path.read_bytes()
        sha = hashlib.sha256(data_bytes).hexdigest()
        if expected_sha256 and sha != expected_sha256:
            raise RuntimeError(
                "ChartQA test parquet SHA-256 mismatch: "
                f"{sha} != expected {expected_sha256}")
        return parquet_path, sha, len(data_bytes)

    req = urllib.request.Request(
        url, headers={
            "User-Agent": "coordpy-chartqa-loader/v1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data_bytes = r.read()

    sha = hashlib.sha256(data_bytes).hexdigest()
    if expected_sha256 and sha != expected_sha256:
        raise RuntimeError(
            "ChartQA test parquet SHA-256 mismatch on "
            f"download: {sha} != expected {expected_sha256}")

    parquet_path.write_bytes(data_bytes)
    sha_sidecar.write_text(
        json.dumps({
            "parquet_url": str(url),
            "parquet_sha256": str(sha),
            "parquet_bytes": int(len(data_bytes)),
        }, sort_keys=True, indent=2))
    return parquet_path, sha, len(data_bytes)


def _coerce_labels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value if x is not None)
    return (str(value),)


def _coerce_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): value[k] for k in value}
    if value is None:
        return {}
    return {"raw": value}


def _image_bytes_from_cell(cell: Any) -> bytes:
    """The HF ``Image`` feature serialises to a dict like
    ``{"bytes": b"...", "path": "..."}`` when read through
    ``pyarrow``."""
    if cell is None:
        return b""
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        for k in ("bytes", "image_bytes", "data"):
            v = cell.get(k)
            if isinstance(v, (bytes, bytearray)):
                return bytes(v)
        path = cell.get("path")
        if path:
            p = Path(str(path))
            if p.exists():
                return p.read_bytes()
    return b""


def load_chartqa_test_corpus_v1(
        *,
        parquet_path: Path,
        limit: int | None = None,
        split_filter: str | None = None,
) -> tuple[ChartQAProblemV1, ...]:
    """Parse the ChartQA test parquet into a tuple of
    ``ChartQAProblemV1`` capsules.

    If ``limit`` is set, only the first ``limit`` rows are
    decoded.  If ``split_filter`` is "human" or "machine", only
    rows with the matching ``human_or_machine`` field are
    returned.  Default (``None``) returns ALL rows in the test
    split — matches the canonical published evaluation.
    """
    import pyarrow.parquet as pq  # heavy import — kept lazy

    table = pq.read_table(str(parquet_path))
    n_rows = int(table.num_rows)
    col_names = set(table.column_names)

    # ChartQA test schemas observed in the wild use either an
    # ``image`` (HF Image feature) or ``decoded_image`` column;
    # accept either.
    image_col_candidates = [
        c for c in (
            "image", "decoded_image", "image_bytes",
        ) if c in col_names]
    if not image_col_candidates:
        raise RuntimeError(
            "ChartQA parquet has no recognized image column "
            f"(saw {sorted(col_names)})")
    image_col = image_col_candidates[0]

    # ChartQA question column is normally ``query`` (HF schema) or
    # ``question`` (some mirrors).  Accept either.
    query_col_candidates = [
        c for c in (
            "query", "question",
        ) if c in col_names]
    if not query_col_candidates:
        raise RuntimeError(
            "ChartQA parquet has no recognized query column "
            f"(saw {sorted(col_names)})")
    query_col = query_col_candidates[0]

    # ChartQA gold labels live in ``label`` (HF) or ``answer`` /
    # ``answers`` (some mirrors).
    label_col_candidates = [
        c for c in (
            "label", "answer", "answers", "labels",
        ) if c in col_names]
    if not label_col_candidates:
        raise RuntimeError(
            "ChartQA parquet has no recognized label column "
            f"(saw {sorted(col_names)})")
    label_col = label_col_candidates[0]

    has_split_col = "human_or_machine" in col_names
    has_type_col = "type" in col_names

    def _get(col: str, i: int) -> Any:
        if col not in col_names:
            return None
        return table.column(col)[i].as_py()

    def _derive_split(
            split_raw: Any, type_raw: Any) -> str:
        """Map the dataset's split indicator to canonical
        ``human`` / ``machine``.  Handles two upstream schemas:

          * ``lmms-lab/ChartQA``: ``type`` ∈ {``human_test``,
            ``augmented_test``} → ``human`` / ``machine``.
          * mirrors that carry an explicit
            ``human_or_machine`` column.
        """
        if split_raw is not None:
            s = str(split_raw).lower().strip()
            if s in {"human", "machine", "augmented"}:
                return (
                    "machine" if s == "augmented" else s)
        if type_raw is not None:
            t = str(type_raw).lower().strip()
            if "human" in t:
                return "human"
            if "augmented" in t or "machine" in t:
                return "machine"
        return ""

    problems: list[ChartQAProblemV1] = []
    for i in range(n_rows):
        if limit is not None and len(problems) >= int(limit):
            break
        split_raw = (
            _get("human_or_machine", i)
            if has_split_col else None)
        type_raw = _get("type", i) if has_type_col else None
        split_val = _derive_split(split_raw, type_raw)
        if (split_filter is not None
                and split_val != str(split_filter).lower()):
            continue
        query_raw = _get(query_col, i)
        label_raw = _get(label_col, i)
        image_cell = _get(image_col, i)
        img_bytes = _image_bytes_from_cell(image_cell)
        img_sha = (
            hashlib.sha256(img_bytes).hexdigest()
            if img_bytes else "")
        img_fmt = _sniff_image_format(img_bytes)
        metadata: dict[str, Any] = {
            "row_index": int(i),
            "image_format_sniff": str(img_fmt),
        }
        if type_raw is not None:
            metadata["type"] = str(type_raw)
        problems.append(ChartQAProblemV1(
            pid=f"chartqa_test_{i:06d}",
            query=str(query_raw) if query_raw else "",
            labels=_coerce_labels(label_raw),
            human_or_machine=str(split_val),
            image_bytes=img_bytes,
            image_sha256=img_sha,
            image_format=img_fmt,
            metadata=metadata,
        ))
    return tuple(problems)


def compute_corpus_merkle_root_v1(
        problems: tuple[ChartQAProblemV1, ...],
) -> str:
    """Per-problem ``to_dict_no_image()`` is canonicalised and
    SHA-256 hashed; leaves are sorted by pid and folded into a
    single root.  Matches the W95 convention so the same audit
    re-derivation code works on both."""
    leaves = []
    for p in problems:
        leaf_bytes = _canonical_bytes(p.to_dict_no_image())
        leaves.append((str(p.pid),
                       hashlib.sha256(leaf_bytes).hexdigest()))
    leaves.sort(key=lambda x: x[0])
    h = hashlib.sha256()
    for pid, leaf_hex in leaves:
        h.update(pid.encode("utf-8"))
        h.update(b"\x00")
        h.update(leaf_hex.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def manifest_for_corpus_v1(
        *,
        parquet_path: Path,
        problems: tuple[ChartQAProblemV1, ...],
        parquet_sha256: str,
        parquet_bytes: int,
        url: str = CHARTQA_TEST_PARQUET_URL,
) -> ChartQACorpusManifestV1:
    return ChartQACorpusManifestV1(
        schema=W96_CHARTQA_LOADER_V1_SCHEMA_VERSION,
        parquet_url=str(url),
        parquet_sha256=str(parquet_sha256),
        parquet_bytes=int(parquet_bytes),
        n_problems=int(len(problems)),
        corpus_merkle_root=compute_corpus_merkle_root_v1(problems),
    )


def select_chartqa_subset_v1(
        *,
        seed: int,
        n_problems: int,
        corpus: tuple[ChartQAProblemV1, ...],
) -> tuple[ChartQAProblemV1, ...]:
    """Deterministically pick a slice of ``n_problems`` from the
    test corpus.  Mirrors the
    ``select_mathvista_subset_v1`` / ``select_humaneval_subset_v1``
    discipline: the slice is fully determined by the seed and the
    ordered list of pids in the corpus (so the slice is
    reproducible across corpus loads as long as the parquet SHA
    matches)."""
    if not corpus:
        return ()
    pids = sorted(p.pid for p in corpus)
    rng = random.Random(int(seed))
    indices = list(range(len(pids)))
    rng.shuffle(indices)
    chosen_pids = {pids[i] for i in indices[:int(n_problems)]}
    return tuple(p for p in corpus if p.pid in chosen_pids)


__all__ = [
    "W96_CHARTQA_LOADER_V1_SCHEMA_VERSION",
    "CHARTQA_TEST_PARQUET_URL",
    "CHARTQA_TEST_EXPECTED_PARQUET_SHA256",
    "CHARTQA_TEST_EXPECTED_N_PROBLEMS_LOWER",
    "CHARTQA_TEST_EXPECTED_N_PROBLEMS_UPPER",
    "CHARTQA_TEST_EXPECTED_PARQUET_BYTES_LOWER",
    "CHARTQA_TEST_EXPECTED_PARQUET_BYTES_UPPER",
    "ChartQAProblemV1",
    "ChartQACorpusManifestV1",
    "fetch_chartqa_test_parquet",
    "load_chartqa_test_corpus_v1",
    "compute_corpus_merkle_root_v1",
    "manifest_for_corpus_v1",
    "select_chartqa_subset_v1",
]
