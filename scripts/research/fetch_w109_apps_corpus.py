#!/usr/bin/env python3
"""W109-α — APPS contamination-control corpus fetch + materialize (NIM-free).

Builds the SHA-pinnable ``~/.cache/coordpy/apps-test.jsonl`` that the W108
``apps_loader_v1`` consumes, from the Hugging Face auto-converted parquet of
``codeparrot/apps`` (the dataset's ``main`` branch uses a loading script, so
the dataset-viewer/parquet HTTP API refuses it; the ``refs/convert/parquet``
data-only branch is the reproducible source).

Provenance (pinned, honest):

* dataset ``codeparrot/apps`` (Hendrycks et al., 2021 — contamination-EXPOSED,
  2021 vintage, almost certainly inside the Llama-3.x training corpus; C7 = C).
* revision ``refs/convert/parquet`` @ commit
  ``0f10e424e13e1c2a69f851e153097b71b6734a1f`` (the auto-convert branch).
* config ``all``, split ``test`` — shards ``0000.parquet`` (424 202 850 B,
  SHA-256 ``f1c36415…``) + ``0001.parquet`` (319 566 256 B, SHA-256
  ``3b02746c…``); 5 000 test problems total.

Deterministic build rule (reproducible → stable output SHA):

* keep ONLY call-based rows — ``input_output`` (a JSON **string**) decodes to a
  dict with a non-empty ``fn_name`` (38 of 5 000);
* preserve the UPSTREAM field strings VERBATIM (``input_output``,
  ``question``, ``starter_code``) — no re-serialization drift;
* sort by integer ``problem_id``;
* write one canonical JSON object per line (sorted keys, compact separators)
  so the file hashes identically across machines.

This is the operator-fetch pattern (W101/W102/W107/W108) executed in-milestone
because pyarrow + network egress are available here.  Run once; the loader +
preflight + pilot then read the pinned JSONL.

Usage::

    python scripts/fetch_w109_apps_corpus.py            # build from cached parquet
    python scripts/fetch_w109_apps_corpus.py --download  # fetch parquet first
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

APPS_CONVERT_COMMIT = "0f10e424e13e1c2a69f851e153097b71b6734a1f"
APPS_PARQUET_BASE = (
    "https://huggingface.co/datasets/codeparrot/apps/resolve/"
    "refs%2Fconvert%2Fparquet/all/test")
APPS_PARQUET_SHARDS = ("0000.parquet", "0001.parquet")
EXPECTED_SHARD_SHA256 = {
    "0000.parquet":
        "f1c36415a9aabe114558a1b0a943f1dc50f2468f12c741585505f84ace577514",
    "0001.parquet":
        "3b02746c5f8ede88388a4635b8d38ab3a29b31f9f964d4758652bcec9dc85913",
}

PARQUET_DIR = os.path.expanduser("~/.cache/coordpy/apps_parquet")
OUT_JSONL = os.path.expanduser("~/.cache/coordpy/apps-test.jsonl")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_shards() -> None:
    os.makedirs(PARQUET_DIR, exist_ok=True)
    for shard in APPS_PARQUET_SHARDS:
        dst = os.path.join(PARQUET_DIR, f"test_{shard}")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        url = f"{APPS_PARQUET_BASE}/{shard}"
        print(f"  downloading {url} -> {dst}")
        urllib.request.urlretrieve(url, dst)


def _is_call_based(io_str: str) -> dict | None:
    try:
        d = json.loads(io_str)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(d, dict) and str(d.get("fn_name") or "").strip():
        return d
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="W109 APPS corpus materializer")
    ap.add_argument("--download", action="store_true",
                    help="fetch the parquet shards first if missing")
    ap.add_argument("--parquet-dir", default=PARQUET_DIR)
    ap.add_argument("--out", default=OUT_JSONL)
    args = ap.parse_args()

    if args.download:
        _download_shards()

    shard_paths = [os.path.join(args.parquet_dir, f"test_{s}")
                   for s in APPS_PARQUET_SHARDS]
    for s, p in zip(APPS_PARQUET_SHARDS, shard_paths):
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            raise SystemExit(
                f"missing parquet shard {p}; run with --download first")
        actual = _file_sha256(p)
        exp = EXPECTED_SHARD_SHA256[s]
        if actual != exp:
            raise SystemExit(
                f"shard SHA drift for {s}: actual={actual} expected={exp}; "
                "refusing a possibly-tampered upstream artifact")
        print(f"  shard {s}: SHA-256 OK ({actual[:12]}…)")

    rows: list[dict] = []
    diffs: Counter = Counter()
    total = 0
    cols = ["problem_id", "question", "input_output", "starter_code",
            "difficulty", "url"]
    for p in shard_paths:
        tb = pq.read_table(p, columns=cols)
        d = {c: tb.column(c).to_pylist() for c in cols}
        for i in range(tb.num_rows):
            total += 1
            io_str = d["input_output"][i]
            if not isinstance(io_str, str):
                continue
            if _is_call_based(io_str) is None:
                continue
            rows.append({
                "problem_id": int(d["problem_id"][i]),
                "question": d["question"][i] or "",
                "input_output": io_str,              # VERBATIM upstream string
                "starter_code": d["starter_code"][i] or "",
                "difficulty": d["difficulty"][i] or "",
                "url": d["url"][i] or "",
            })
            diffs[d["difficulty"][i] or ""] += 1

    rows.sort(key=lambda r: int(r["problem_id"]))
    out_path = Path(os.path.expanduser(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False))
            f.write("\n")

    out_sha = _file_sha256(str(out_path))
    print("\n=== W109 APPS corpus materialized ===")
    print(f"  upstream         : codeparrot/apps @ refs/convert/parquet "
          f"{APPS_CONVERT_COMMIT[:12]}…")
    print(f"  test rows scanned: {total}")
    print(f"  call-based subset: {len(rows)}  (difficulty {dict(diffs)})")
    print(f"  out              : {out_path}")
    print(f"  out bytes        : {out_path.stat().st_size}")
    print(f"  out SHA-256      : {out_sha}")
    print("\n  export APPS_TRUSTED_SHA256_OVERRIDE=" + out_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
