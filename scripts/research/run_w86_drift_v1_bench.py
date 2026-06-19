#!/usr/bin/env python3
"""W86 / P2 #42 Drift V1 — bench driver."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coordpy.state_drift_detection_v1 import (  # noqa: E402
    run_drift_v1_bench,
)


def _utc_stamp() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=86_042)
    p.add_argument(
        "--finetune-noise-scale", type=float, default=0.05)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else _REPO_ROOT / "results" / "w86" / "drift"
        / f"w86_drift_{_utc_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = run_drift_v1_bench(
        seed=args.seed,
        finetune_noise_scale=args.finetune_noise_scale)
    report_dict = {
        "kind": "w86_drift_v1_bench_report",
        "schema": "coordpy.state_drift_detection_v1.w86_v1",
        "report": rep.to_dict(),
    }
    out_path = out_dir / "drift_v1_bench_report.json"
    out_path.write_text(json.dumps(
        report_dict, indent=2, sort_keys=True))

    print(f"wrote {out_path}")
    summary = {
        k: rep.to_dict()[k] for k in [
            "old_weights_cid",
            "new_weights_cid",
            "drift_score_unchanged",
            "drift_score_changed",
            "threshold",
            "detector_fires_when_changed",
            "detector_does_not_fire_when_unchanged",
            "stale_verdict_marks_old_capsule_stale",
            "stale_verdict_marks_fresh_capsule_fresh",
            "fallback_recommendation_is_recompute_for_stale",
            "new_memory_strictly_beats_stale_on_holdout",
            "stale_holdout_mse",
            "new_holdout_mse",
            "report_cid",
        ]
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")

    closed = (
        rep.detector_fires_when_changed
        and rep.detector_does_not_fire_when_unchanged
        and rep.stale_verdict_marks_old_capsule_stale
        and rep.stale_verdict_marks_fresh_capsule_fresh
        and rep.fallback_recommendation_is_recompute_for_stale
        and rep.new_memory_strictly_beats_stale_on_holdout)
    return 0 if closed else 1


if __name__ == "__main__":
    sys.exit(main())
