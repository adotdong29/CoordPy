"""Round-1 draft cache for paired A/B benchmarks.

Phase 18 needs to run two trigger variants against the exact same round-1
drafts. This module handles serializing and loading those drafts so both
trigger strategies see identical starting conditions.

Cache file format (JSON):
  {
    "surface":    str,          # "protocolkit" | "numericledger"
    "model":      str,          # Ollama model tag used for generation
    "timestamp":  str,          # ISO-8601 (UTC)
    "round1": {
      "drafts":     {specialty: src_str, ...},
      "tokens":     {specialty: {"prompt": int, "completion": int}, ...},
      "acceptance": {specialty: {"accepted": bool, "attempts": int}, ...}
    }
  }

The `round1` sub-dict is exactly what `run_round1()` returns in both
phase14_benchmark and phase17_generality harnesses.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


_REQUIRED_ROUND1_KEYS = {"drafts", "tokens", "acceptance"}


def save_round1(
    path: str | Path,
    round1: dict,
    surface: str,
    model: str,
) -> None:
    """Persist a round-1 result dict to *path* in the standard cache format.

    Raises `ValueError` if *round1* is missing expected keys.
    """
    missing = _REQUIRED_ROUND1_KEYS - set(round1)
    if missing:
        raise ValueError(
            f"round1 dict is missing required keys: {sorted(missing)}"
        )
    payload = {
        "surface": surface,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "round1": {
            "drafts": dict(round1["drafts"]),
            "tokens": dict(round1["tokens"]),
            "acceptance": dict(round1["acceptance"]),
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def load_round1(path: str | Path) -> tuple[dict, dict]:
    """Load a round-1 cache from *path*.

    Returns `(round1_dict, metadata)` where:
      - `round1_dict` is ready to pass as `round1=` to a harness `run()`.
      - `metadata` holds surface / model / timestamp strings.

    Raises `ValueError` on format problems, `FileNotFoundError` if the
    file is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"round-1 cache not found: {path}")
    with path.open() as fh:
        raw = json.load(fh)
    for key in ("surface", "model", "timestamp", "round1"):
        if key not in raw:
            raise ValueError(
                f"round-1 cache at {path} is missing key {key!r}"
            )
    r1 = raw["round1"]
    missing = _REQUIRED_ROUND1_KEYS - set(r1)
    if missing:
        raise ValueError(
            f"round1 sub-dict at {path} is missing keys: {sorted(missing)}"
        )
    metadata = {
        "surface": raw["surface"],
        "model": raw["model"],
        "timestamp": raw["timestamp"],
    }
    return r1, metadata
