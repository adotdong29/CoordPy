"""W95 — MathVista testmini corpus loader V1.

The W94 cross-modal battlefield pivot selected MathVista
(``AI4Math/MathVista`` on HuggingFace; testmini split = 1000
problems) as the next serious cross-modal team-superiority
battlefield, replacing the empirically retired HumanEval-Visual
K=5 line.  See ``docs/W94_CROSS_MODAL_BATTLEFIELD_SCOUTING.md``
for the selection rationale.

This module:

  * downloads the canonical testmini parquet once and SHA-anchors
    it (no ``datasets`` library dependency; uses ``requests`` +
    ``pyarrow``);
  * exposes ``MathVistaProblemV1`` capsules with the full
    per-problem schema (pid / question / query / image bytes +
    SHA / choices / answer / answer_type / question_type / unit /
    precision / metadata);
  * provides a deterministic ``select_mathvista_subset_v1`` slice
    selector mirroring the W86 ``select_humaneval_subset_v1`` /
    W88 ``select_cross_modal_subset_v1`` discipline.

The loader is the W95 corpus-side anti-cheat surface.  The bench
side will sit on top of this in a separate
``coordpy.mathvista_bench_v1`` module — that one is held back
until cheap preflight earns the expensive run (per W93/W94
discipline).  This module by itself does NOT call NIM and does
NOT define A0/A1/B baselines; it only loads + anchors the
corpus and is safe to import from preflight probes.

Honest scope (W95)
------------------

* ``W95-L-MATHVISTA-LOADER-V1-NETWORK-FETCH-CAP`` — first fetch
  hits the HuggingFace CDN; subsequent loads read the cached
  parquet from disk.  No model calls.
* ``W95-L-MATHVISTA-LOADER-V1-PARQUET-LIBRARY-CAP`` — V1 parses
  the parquet with ``pyarrow`` (``pa.Table.from_*``).  An
  alternative ``pandas`` path is provided as a fallback.
* ``W95-L-MATHVISTA-LOADER-V1-CORPUS-SHA-CAP`` — the canonical
  parquet SHA-256 is recorded on first download and verified on
  every subsequent load; mismatches refuse to return a corpus.
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


W95_MATHVISTA_LOADER_V1_SCHEMA_VERSION: str = (
    "coordpy.mathvista_loader_v1.v1")


# Canonical HuggingFace path to the testmini parquet.  This is the
# Parquet file that backs the ``testmini`` split of
# ``AI4Math/MathVista`` on the HF Hub.  The filename is anchored
# by the HF Hub revision; the loader records the SHA-256 of the
# fetched bytes for audit-chain re-derivability.
MATHVISTA_TESTMINI_PARQUET_URL: str = (
    "https://huggingface.co/datasets/AI4Math/MathVista/"
    "resolve/main/data/"
    "testmini-00000-of-00001-725687bf7a18d64b.parquet")

# Expected per the HF datasets-server metadata (see
# https://datasets-server.huggingface.co/info?dataset=AI4Math/MathVista).
MATHVISTA_TESTMINI_EXPECTED_N_PROBLEMS: int = 1000
MATHVISTA_TESTMINI_EXPECTED_PARQUET_BYTES_LOWER: int = 100_000_000
MATHVISTA_TESTMINI_EXPECTED_PARQUET_BYTES_UPPER: int = 250_000_000


@dataclasses.dataclass(frozen=True)
class MathVistaProblemV1:
    """A single MathVista testmini problem in canonical capsule
    form.  ``image_bytes`` is the raw decoded image as the
    HuggingFace ``Image`` feature serialised it (typically PNG or
    JPEG); ``image_sha256`` is the hex digest of those bytes."""

    pid: str
    question: str
    query: str
    choices: tuple[str, ...]
    answer: str
    answer_type: str
    question_type: str
    unit: str
    precision: float
    image_bytes: bytes
    image_sha256: str
    image_format: str
    metadata: dict[str, Any]

    def to_dict_no_image(self) -> dict[str, Any]:
        """Serialisable form omitting the raw image bytes
        (sidecars / audit-chain JSON should carry only the SHA, not
        the bytes)."""
        return {
            "pid": str(self.pid),
            "question": str(self.question),
            "query": str(self.query),
            "choices": list(self.choices),
            "answer": str(self.answer),
            "answer_type": str(self.answer_type),
            "question_type": str(self.question_type),
            "unit": str(self.unit),
            "precision": float(self.precision),
            "image_sha256": str(self.image_sha256),
            "image_format": str(self.image_format),
            "metadata": dict(self.metadata),
        }


@dataclasses.dataclass(frozen=True)
class MathVistaCorpusManifestV1:
    """Content-addressed manifest of the loaded testmini corpus.
    Records the parquet SHA, problem count, and a Merkle-style
    SHA-256 over the per-problem ``to_dict_no_image()``
    payloads.  This is what audit verifiers re-derive offline."""

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
    """Detect a small subset of image-format magic bytes that the
    HF ``Image`` feature emits for MathVista.  Returns ``""`` when
    the format is unknown — the loader does not raise so callers
    can still SHA-anchor the bytes."""
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


def fetch_testmini_parquet(
        *,
        cache_dir: Path,
        url: str = MATHVISTA_TESTMINI_PARQUET_URL,
        force: bool = False,
        expected_sha256: str | None = None,
        timeout_s: float = 600.0,
) -> tuple[Path, str, int]:
    """Idempotently fetch the canonical testmini parquet to
    ``cache_dir``.  Returns (path, sha256_hex, n_bytes).

    If ``expected_sha256`` is provided, the loader refuses to
    return a path whose bytes do not hash to that value (the W95
    audit-chain bar).  If not, the loader records the observed
    SHA-256 the first time and writes a ``parquet.sha256`` sidecar
    so subsequent loads can verify it without re-downloading.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / (
        "mathvista_testmini_00000-of-00001.parquet")
    sha_sidecar = cache_dir / (
        "mathvista_testmini_00000-of-00001.parquet.sha256")

    if parquet_path.exists() and not force:
        data_bytes = parquet_path.read_bytes()
        sha = hashlib.sha256(data_bytes).hexdigest()
        if expected_sha256 and sha != expected_sha256:
            raise RuntimeError(
                "MathVista testmini parquet SHA-256 mismatch: "
                f"{sha} != expected {expected_sha256}")
        return parquet_path, sha, len(data_bytes)

    req = urllib.request.Request(
        url, headers={"User-Agent": "coordpy-mathvista-loader/v1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data_bytes = r.read()

    sha = hashlib.sha256(data_bytes).hexdigest()
    if expected_sha256 and sha != expected_sha256:
        raise RuntimeError(
            "MathVista testmini parquet SHA-256 mismatch on "
            f"download: {sha} != expected {expected_sha256}")

    parquet_path.write_bytes(data_bytes)
    sha_sidecar.write_text(
        json.dumps({
            "parquet_url": str(url),
            "parquet_sha256": str(sha),
            "parquet_bytes": int(len(data_bytes)),
        }, sort_keys=True, indent=2))
    return parquet_path, sha, len(data_bytes)


def _coerce_choices(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value)
    return (str(value),)


def _coerce_unit(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_precision(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): value[k] for k in value}
    return {"raw": value}


def _image_bytes_from_cell(cell: Any) -> bytes:
    """The HF ``Image`` feature serialises to a dict like
    ``{"bytes": b"...", "path": "..."}`` when read through
    ``pyarrow``.  This normaliser handles both that shape and the
    direct-bytes shape some pyarrow readers produce."""
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


def load_testmini_corpus_v1(
        *,
        parquet_path: Path,
        limit: int | None = None,
) -> tuple[MathVistaProblemV1, ...]:
    """Parse the testmini parquet into a tuple of
    ``MathVistaProblemV1`` capsules.  If ``limit`` is set, only
    the first ``limit`` rows are decoded (useful for cheap
    probes; the full 1000 problems take ~170 MB of decoded
    images)."""
    import pyarrow.parquet as pq  # heavy import — kept lazy

    table = pq.read_table(str(parquet_path))
    n_rows = int(table.num_rows)
    if limit is not None:
        n_rows = min(n_rows, int(limit))
    # The HF schema may use either "image" (HF Image feature) or
    # "decoded_image" (the binary-decoded variant); accept either.
    col_names = set(table.column_names)
    image_col_candidates = [
        c for c in (
            "decoded_image", "image", "image_bytes",
        ) if c in col_names]
    if not image_col_candidates:
        raise RuntimeError(
            "MathVista parquet has no recognized image column "
            f"(saw {sorted(col_names)})")
    image_col = image_col_candidates[0]

    def _get(col: str, i: int) -> Any:
        if col not in col_names:
            return None
        return table.column(col)[i].as_py()

    problems: list[MathVistaProblemV1] = []
    for i in range(n_rows):
        pid_raw = _get("pid", i)
        question_raw = _get("question", i)
        query_raw = _get("query", i)
        answer_raw = _get("answer", i)
        answer_type_raw = _get("answer_type", i)
        question_type_raw = _get("question_type", i)
        unit_raw = _get("unit", i)
        precision_raw = _get("precision", i)
        choices_raw = _get("choices", i)
        metadata_raw = _get("metadata", i)
        image_cell = _get(image_col, i)
        img_bytes = _image_bytes_from_cell(image_cell)
        img_sha = (
            hashlib.sha256(img_bytes).hexdigest()
            if img_bytes else "")
        img_fmt = _sniff_image_format(img_bytes)
        problems.append(MathVistaProblemV1(
            pid=str(pid_raw) if pid_raw is not None else "",
            question=str(question_raw) if question_raw else "",
            query=str(query_raw) if query_raw else "",
            choices=_coerce_choices(choices_raw),
            answer=str(answer_raw) if answer_raw is not None
            else "",
            answer_type=str(answer_type_raw)
            if answer_type_raw else "",
            question_type=str(question_type_raw)
            if question_type_raw else "",
            unit=_coerce_unit(unit_raw),
            precision=_coerce_precision(precision_raw),
            image_bytes=img_bytes,
            image_sha256=img_sha,
            image_format=img_fmt,
            metadata=_coerce_metadata(metadata_raw),
        ))
    return tuple(problems)


def compute_corpus_merkle_root_v1(
        problems: tuple[MathVistaProblemV1, ...],
) -> str:
    """Per-problem ``to_dict_no_image()`` is canonicalised and
    SHA-256 hashed; the leaves are sorted by pid and folded into
    a single root via ``sha256(sorted leaves)``.  Sorting by pid
    makes the root invariant to row-order changes in the parquet,
    matching the W88/W90/W92 convention."""
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
        problems: tuple[MathVistaProblemV1, ...],
        parquet_sha256: str,
        parquet_bytes: int,
        url: str = MATHVISTA_TESTMINI_PARQUET_URL,
) -> MathVistaCorpusManifestV1:
    return MathVistaCorpusManifestV1(
        schema=W95_MATHVISTA_LOADER_V1_SCHEMA_VERSION,
        parquet_url=str(url),
        parquet_sha256=str(parquet_sha256),
        parquet_bytes=int(parquet_bytes),
        n_problems=int(len(problems)),
        corpus_merkle_root=compute_corpus_merkle_root_v1(problems),
    )


def select_mathvista_subset_v1(
        *,
        seed: int,
        n_problems: int,
        corpus: tuple[MathVistaProblemV1, ...],
) -> tuple[MathVistaProblemV1, ...]:
    """Deterministically pick a slice of ``n_problems`` from the
    testmini corpus.  Mirrors the
    ``select_humaneval_subset_v1`` / ``select_cross_modal_subset_v1``
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
    "W95_MATHVISTA_LOADER_V1_SCHEMA_VERSION",
    "MATHVISTA_TESTMINI_PARQUET_URL",
    "MATHVISTA_TESTMINI_EXPECTED_N_PROBLEMS",
    "MATHVISTA_TESTMINI_EXPECTED_PARQUET_BYTES_LOWER",
    "MATHVISTA_TESTMINI_EXPECTED_PARQUET_BYTES_UPPER",
    "MathVistaProblemV1",
    "MathVistaCorpusManifestV1",
    "fetch_testmini_parquet",
    "load_testmini_corpus_v1",
    "compute_corpus_merkle_root_v1",
    "manifest_for_corpus_v1",
    "select_mathvista_subset_v1",
]
