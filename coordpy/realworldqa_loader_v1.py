"""W96-D — RealWorldQA test corpus loader V1.

After the W96-D D1 (ChartQA) preflight FAILed at P3 (A1@K=5
saturation: 91.69 % at 11B and 92.75 % at 90B; residual < 20 pp
at both scales), the W96-D battlefield pivots to the D2 backup,
RealWorldQA, per `docs/RUNBOOK_W96D.md`.

This module:

  * downloads the canonical ``lmms-lab/RealWorldQA`` test
    parquet shards (the canonical HF Hub copy is split into
    ``test-00000-of-00002.parquet`` + ``test-00001-of-00002.parquet``)
    and SHA-anchors each shard;
  * exposes ``RealWorldQAProblemV1`` capsules with the canonical
    per-problem schema (pid / question / answer / image bytes +
    SHA);
  * provides a deterministic ``select_realworldqa_subset_v1``
    slice selector mirroring the W95
    ``select_mathvista_subset_v1`` discipline.

The loader is the W96-D corpus-side anti-cheat surface for D2.
The bench side will sit on top of this in a separate
``coordpy.realworldqa_bench_v1`` module — that one is held back
until cheap preflight earns the expensive run.  This module by
itself does NOT call NIM and does NOT define A0/A1/B baselines.

Honest scope (W96-D)
--------------------

* ``W96-L-REALWORLDQA-LOADER-V1-NETWORK-FETCH-CAP`` — first fetch
  hits the HuggingFace CDN; subsequent loads read the cached
  parquet shards from disk.  No model calls.
* ``W96-L-REALWORLDQA-LOADER-V1-PARQUET-LIBRARY-CAP`` — V1 parses
  the parquet shards with ``pyarrow``.
* ``W96-L-REALWORLDQA-LOADER-V1-MULTI-SHARD-CAP`` — V1 supports
  multi-shard test parquets by fetching, SHA-anchoring, and
  concatenating each shard in upstream order.
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


W96_REALWORLDQA_LOADER_V1_SCHEMA_VERSION: str = (
    "coordpy.realworldqa_loader_v1.v1")


# Canonical HuggingFace paths to the RealWorldQA test parquet
# shards.  The default targets ``lmms-lab/RealWorldQA`` which is
# the formatted lmms-eval copy of xAI's original 765-problem
# evaluation set.
REALWORLDQA_TEST_PARQUET_URLS: tuple[str, ...] = (
    "https://huggingface.co/datasets/lmms-lab/RealWorldQA/"
    "resolve/main/data/test-00000-of-00002.parquet",
    "https://huggingface.co/datasets/lmms-lab/RealWorldQA/"
    "resolve/main/data/test-00001-of-00002.parquet",
)

# Pre-recorded SHA-256 of the canonical ``lmms-lab/RealWorldQA``
# test parquet shards as fetched on 2026-05-25.  Anchors the
# W96-D D2 corpus reproducibility.
REALWORLDQA_TEST_EXPECTED_PARQUET_SHA256: tuple[
        str | None, ...] = (
    "0ed8b555586923099bd5d6ba5dd8b656b403ccfc418881facd237a6d6f"
    "e64952",
    "7dcb3ac3483362ca082cd4cddd0ab1389e9a276f1aefd4603fde0a2ce6"
    "bc74d0",
)

# Expected bounds for the test split.  765 problems per
# `lmms-lab/RealWorldQA` dataset_info.
REALWORLDQA_TEST_EXPECTED_N_PROBLEMS_LOWER: int = 700
REALWORLDQA_TEST_EXPECTED_N_PROBLEMS_UPPER: int = 800
REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_LOWER: int = 200_000_000
REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_UPPER: int = 900_000_000


@dataclasses.dataclass(frozen=True)
class RealWorldQAProblemV1:
    """A single RealWorldQA test problem in canonical capsule
    form.  ``image_bytes`` is the raw decoded image as the
    HuggingFace ``Image`` feature serialised it; ``image_sha256``
    is the hex digest of those bytes.  RealWorldQA gold answers
    are a single canonical string (the dataset's ``answer``
    field); some are multi-choice letters (``A``, ``B``, ...)
    and some are free-form short text."""

    pid: str
    question: str
    answer: str
    image_path: str
    image_bytes: bytes
    image_sha256: str
    image_format: str
    metadata: dict[str, Any]

    def to_dict_no_image(self) -> dict[str, Any]:
        return {
            "pid": str(self.pid),
            "question": str(self.question),
            "answer": str(self.answer),
            "image_path": str(self.image_path),
            "image_sha256": str(self.image_sha256),
            "image_format": str(self.image_format),
            "metadata": dict(self.metadata),
        }


@dataclasses.dataclass(frozen=True)
class RealWorldQACorpusManifestV1:
    """Content-addressed manifest of the loaded RealWorldQA
    corpus.  Records each parquet shard's SHA, the total problem
    count, and a Merkle-style SHA-256 over the per-problem
    ``to_dict_no_image()`` payloads."""

    schema: str
    parquet_urls: tuple[str, ...]
    parquet_shard_sha256: tuple[str, ...]
    parquet_total_bytes: int
    n_problems: int
    corpus_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "parquet_urls": list(self.parquet_urls),
            "parquet_shard_sha256": list(
                self.parquet_shard_sha256),
            "parquet_total_bytes": int(self.parquet_total_bytes),
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


def fetch_realworldqa_test_parquets(
        *,
        cache_dir: Path,
        urls: tuple[str, ...] = REALWORLDQA_TEST_PARQUET_URLS,
        force: bool = False,
        expected_sha256: tuple[str | None, ...] = (
            REALWORLDQA_TEST_EXPECTED_PARQUET_SHA256),
        timeout_s: float = 1800.0,
) -> tuple[tuple[Path, ...],
           tuple[str, ...], int]:
    """Idempotently fetch the canonical RealWorldQA test parquet
    shards to ``cache_dir``.  Returns
    (shard_paths, shard_sha256_hex, total_n_bytes).

    Each shard is verified against the corresponding entry of
    ``expected_sha256`` if non-None; mismatches refuse to return
    the corpus.  Observed SHAs are recorded to per-shard sidecars.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    shas: list[str] = []
    total_bytes = 0
    for shard_idx, url in enumerate(urls):
        expected = (
            expected_sha256[shard_idx]
            if shard_idx < len(expected_sha256) else None)
        parquet_path = cache_dir / (
            "realworldqa_test_"
            f"{shard_idx:05d}-of-{len(urls):05d}.parquet")
        sha_sidecar = parquet_path.with_suffix(".parquet.sha256")
        if parquet_path.exists() and not force:
            data_bytes = parquet_path.read_bytes()
            sha = hashlib.sha256(data_bytes).hexdigest()
            if expected and sha != expected:
                raise RuntimeError(
                    f"RealWorldQA shard {shard_idx} SHA mismatch: "
                    f"{sha} != expected {expected}")
            paths.append(parquet_path)
            shas.append(sha)
            total_bytes += len(data_bytes)
            continue
        req = urllib.request.Request(
            url, headers={
                "User-Agent": (
                    "coordpy-realworldqa-loader/v1")})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            data_bytes = r.read()
        sha = hashlib.sha256(data_bytes).hexdigest()
        if expected and sha != expected:
            raise RuntimeError(
                f"RealWorldQA shard {shard_idx} SHA mismatch on "
                f"download: {sha} != expected {expected}")
        parquet_path.write_bytes(data_bytes)
        sha_sidecar.write_text(
            json.dumps({
                "parquet_url": str(url),
                "parquet_sha256": str(sha),
                "parquet_bytes": int(len(data_bytes)),
                "shard_idx": int(shard_idx),
                "n_shards": int(len(urls)),
            }, sort_keys=True, indent=2))
        paths.append(parquet_path)
        shas.append(sha)
        total_bytes += len(data_bytes)
    return tuple(paths), tuple(shas), int(total_bytes)


def _image_bytes_from_cell(cell: Any) -> bytes:
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


def load_realworldqa_test_corpus_v1(
        *,
        parquet_paths: tuple[Path, ...],
        limit: int | None = None,
) -> tuple[RealWorldQAProblemV1, ...]:
    """Parse the RealWorldQA test parquet shards into a tuple of
    ``RealWorldQAProblemV1`` capsules.  Shards are concatenated
    in the order supplied; each row's ``pid`` is a global
    shard-prefixed counter so the corpus order is stable across
    re-runs."""
    import pyarrow.parquet as pq  # heavy import — kept lazy

    problems: list[RealWorldQAProblemV1] = []
    global_idx = 0
    for shard_idx, parquet_path in enumerate(parquet_paths):
        table = pq.read_table(str(parquet_path))
        n_rows = int(table.num_rows)
        col_names = set(table.column_names)

        image_col_candidates = [
            c for c in (
                "image", "decoded_image", "image_bytes",
            ) if c in col_names]
        if not image_col_candidates:
            raise RuntimeError(
                f"RealWorldQA shard {shard_idx} has no image "
                f"column (saw {sorted(col_names)})")
        image_col = image_col_candidates[0]

        question_col_candidates = [
            c for c in (
                "question", "query",
            ) if c in col_names]
        if not question_col_candidates:
            raise RuntimeError(
                f"RealWorldQA shard {shard_idx} has no question "
                f"column (saw {sorted(col_names)})")
        question_col = question_col_candidates[0]

        answer_col_candidates = [
            c for c in (
                "answer", "label", "answers",
            ) if c in col_names]
        if not answer_col_candidates:
            raise RuntimeError(
                f"RealWorldQA shard {shard_idx} has no answer "
                f"column (saw {sorted(col_names)})")
        answer_col = answer_col_candidates[0]

        has_path_col = "image_path" in col_names

        def _get(col: str, i: int) -> Any:
            if col not in col_names:
                return None
            return table.column(col)[i].as_py()

        for i in range(n_rows):
            if limit is not None and len(problems) >= int(limit):
                break
            question_raw = _get(question_col, i)
            answer_raw = _get(answer_col, i)
            path_raw = (
                _get("image_path", i)
                if has_path_col else None)
            image_cell = _get(image_col, i)
            img_bytes = _image_bytes_from_cell(image_cell)
            img_sha = (
                hashlib.sha256(img_bytes).hexdigest()
                if img_bytes else "")
            img_fmt = _sniff_image_format(img_bytes)
            metadata: dict[str, Any] = {
                "shard_idx": int(shard_idx),
                "row_index_in_shard": int(i),
                "image_format_sniff": str(img_fmt),
            }
            problems.append(RealWorldQAProblemV1(
                pid=f"rwqa_test_{global_idx:06d}",
                question=(
                    str(question_raw) if question_raw else ""),
                answer=(
                    str(answer_raw) if answer_raw is not None
                    else ""),
                image_path=(
                    str(path_raw) if path_raw is not None
                    else ""),
                image_bytes=img_bytes,
                image_sha256=img_sha,
                image_format=img_fmt,
                metadata=metadata,
            ))
            global_idx += 1
        if limit is not None and len(problems) >= int(limit):
            break
    return tuple(problems)


def compute_corpus_merkle_root_v1(
        problems: tuple[RealWorldQAProblemV1, ...],
) -> str:
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
        parquet_paths: tuple[Path, ...],
        problems: tuple[RealWorldQAProblemV1, ...],
        parquet_shard_sha256: tuple[str, ...],
        parquet_total_bytes: int,
        urls: tuple[str, ...] = REALWORLDQA_TEST_PARQUET_URLS,
) -> RealWorldQACorpusManifestV1:
    return RealWorldQACorpusManifestV1(
        schema=W96_REALWORLDQA_LOADER_V1_SCHEMA_VERSION,
        parquet_urls=tuple(urls),
        parquet_shard_sha256=tuple(parquet_shard_sha256),
        parquet_total_bytes=int(parquet_total_bytes),
        n_problems=int(len(problems)),
        corpus_merkle_root=compute_corpus_merkle_root_v1(problems),
    )


def select_realworldqa_subset_v1(
        *,
        seed: int,
        n_problems: int,
        corpus: tuple[RealWorldQAProblemV1, ...],
) -> tuple[RealWorldQAProblemV1, ...]:
    if not corpus:
        return ()
    pids = sorted(p.pid for p in corpus)
    rng = random.Random(int(seed))
    indices = list(range(len(pids)))
    rng.shuffle(indices)
    chosen_pids = {pids[i] for i in indices[:int(n_problems)]}
    return tuple(p for p in corpus if p.pid in chosen_pids)


__all__ = [
    "W96_REALWORLDQA_LOADER_V1_SCHEMA_VERSION",
    "REALWORLDQA_TEST_PARQUET_URLS",
    "REALWORLDQA_TEST_EXPECTED_PARQUET_SHA256",
    "REALWORLDQA_TEST_EXPECTED_N_PROBLEMS_LOWER",
    "REALWORLDQA_TEST_EXPECTED_N_PROBLEMS_UPPER",
    "REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_LOWER",
    "REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_UPPER",
    "RealWorldQAProblemV1",
    "RealWorldQACorpusManifestV1",
    "fetch_realworldqa_test_parquets",
    "load_realworldqa_test_corpus_v1",
    "compute_corpus_merkle_root_v1",
    "manifest_for_corpus_v1",
    "select_realworldqa_subset_v1",
]
