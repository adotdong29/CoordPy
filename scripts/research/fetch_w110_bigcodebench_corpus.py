#!/usr/bin/env python3
"""W110-α — BigCodeBench (second contamination-RESISTANT) corpus fetch (NIM-free).

Builds the SHA-pinnable ``~/.cache/coordpy/bigcodebench-v0_1_4.jsonl`` that the
W110 ``bigcodebench_loader_v1`` consumes, from the Hugging Face auto-converted
parquet of ``bigcode/bigcodebench`` (the ``refs/convert/parquet`` data-only
branch — the reproducible source).

Provenance (pinned, honest):

* dataset ``bigcode/bigcodebench`` (BigCode, 2024-06 — contamination-RESISTANT
  by release-date anchoring; HF ``createdAt`` 2024-06-05; split ``v0.1.4``
  ``lastModified`` 2025-04-30; AFTER the ≈2024-01 Llama-3.x cutoff). The honest
  caveat (``W110-L-BIGCODEBENCH-RELEASE-DATE-RESISTANCE-NOT-CONTEST-DATE-CAP``):
  the composed library primitives are in-training; the RESISTANCE is the novel
  composition + 2024-06 release date, not strict contest-date anchoring.
* revision ``refs/convert/parquet``; config ``default``, split ``v0.1.4``;
  shard ``0000.parquet`` (≈2 362 110 B); 1140 problems.

Deterministic build rule (reproducible → stable output SHA):

* keep ALL rows (BigCodeBench has no functional/non-functional split — every
  row is a complete-function task with a ``unittest`` oracle);
* preserve the loader-required fields VERBATIM (``task_id``,
  ``complete_prompt``, ``code_prompt``, ``canonical_solution``, ``test``,
  ``entry_point``, ``libs``) — no re-serialization drift of the source code;
* sort by the integer tail of ``task_id`` (``BigCodeBench/<n>``);
* write one canonical JSON object per line (sorted keys, compact separators) so
  the file hashes identically across machines.

This is the operator-fetch pattern (W101/W102/W107/W108/W109) executed
in-milestone because pyarrow + HF egress are available here. Run once; the
loader + preflight + pilot then read the pinned JSONL.

Usage::

    python scripts/fetch_w110_bigcodebench_corpus.py --download
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

BIGCODEBENCH_PARQUET_URL = (
    "https://huggingface.co/datasets/bigcode/bigcodebench/resolve/"
    "refs%2Fconvert%2Fparquet/default/v0.1.4/0000.parquet")
BIGCODEBENCH_SPLIT = "v0.1.4"

PARQUET_DIR = os.path.expanduser("~/.cache/coordpy/bigcodebench_parquet")
OUT_JSONL = os.path.expanduser("~/.cache/coordpy/bigcodebench-v0_1_4.jsonl")

LOADER_FIELDS = ("task_id", "complete_prompt", "code_prompt",
                 "canonical_solution", "test", "entry_point", "libs")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_shard() -> str:
    os.makedirs(PARQUET_DIR, exist_ok=True)
    dst = os.path.join(PARQUET_DIR, f"{BIGCODEBENCH_SPLIT}_0000.parquet")
    if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
        print(f"  downloading {BIGCODEBENCH_PARQUET_URL} -> {dst}")
        urllib.request.urlretrieve(BIGCODEBENCH_PARQUET_URL, dst)
    return dst


def _id_tail(task_id: str) -> int:
    try:
        return int(str(task_id).rsplit("/", 1)[-1])
    except (TypeError, ValueError):
        return 1 << 62


def _libs_to_jsonable(v):
    """Normalize ``libs`` to a JSON list. Upstream stores a Python-repr STRING
    (single-quoted), so parse with json then ast.literal_eval."""
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") or s.startswith("("):
            for parser in (json.loads, ast.literal_eval):
                try:
                    r = parser(s)
                    if isinstance(r, (list, tuple)):
                        return [str(x) for x in r]
                except Exception:  # noqa: BLE001
                    continue
        return [x for x in (p.strip() for p in s.split(",")) if x]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(
        description="W110 BigCodeBench corpus materializer")
    ap.add_argument("--download", action="store_true",
                    help="fetch the parquet shard first if missing")
    ap.add_argument("--parquet-dir", default=PARQUET_DIR)
    ap.add_argument("--out", default=OUT_JSONL)
    args = ap.parse_args()

    shard = os.path.join(args.parquet_dir,
                         f"{BIGCODEBENCH_SPLIT}_0000.parquet")
    if args.download or not (os.path.exists(shard) and os.path.getsize(shard)):
        shard = _download_shard()
    if not (os.path.exists(shard) and os.path.getsize(shard) > 0):
        raise SystemExit(
            f"missing parquet shard {shard}; run with --download first")
    shard_sha = _file_sha256(shard)
    print(f"  shard {BIGCODEBENCH_SPLIT}/0000.parquet: "
          f"{os.path.getsize(shard)} B  SHA-256 {shard_sha[:12]}…")

    tb = pq.read_table(shard)
    avail = set(tb.column_names)
    missing = [c for c in LOADER_FIELDS if c not in avail and c != "libs"]
    if missing:
        raise SystemExit(
            f"BigCodeBench parquet missing expected columns {missing}; "
            f"available={sorted(avail)} — refusing (schema-confirm-at-fetch).")
    cols = [c for c in LOADER_FIELDS if c in avail]
    data = {c: tb.column(c).to_pylist() for c in cols}

    rows: list[dict] = []
    libs_hist: Counter = Counter()
    for i in range(tb.num_rows):
        rec = {
            "task_id": str(data["task_id"][i]),
            "complete_prompt": data["complete_prompt"][i] or "",
            "code_prompt": data["code_prompt"][i] or "",
            "canonical_solution": data["canonical_solution"][i] or "",
            "test": data["test"][i] or "",
            "entry_point": data["entry_point"][i] or "",
            "libs": _libs_to_jsonable(data.get("libs", [None] * tb.num_rows)[i]),
        }
        rows.append(rec)
        n = len(rec["libs"]) if isinstance(rec["libs"], list) else 0
        libs_hist["libs0_1" if n <= 1 else ("libs2" if n == 2 else "libs3plus")] += 1

    rows.sort(key=lambda r: _id_tail(r["task_id"]))
    out_path = Path(os.path.expanduser(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False))
            f.write("\n")

    out_sha = _file_sha256(str(out_path))
    print("\n=== W110 BigCodeBench corpus materialized ===")
    print(f"  upstream   : bigcode/bigcodebench @ refs/convert/parquet "
          f"{BIGCODEBENCH_SPLIT}")
    print(f"  shard SHA  : {shard_sha}")
    print(f"  rows       : {len(rows)}  (n_libs buckets {dict(libs_hist)})")
    print(f"  out        : {out_path}")
    print(f"  out bytes  : {out_path.stat().st_size}")
    print(f"  out SHA-256: {out_sha}")
    print("\n  export BIGCODEBENCH_TRUSTED_SHA256_OVERRIDE=" + out_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
