#!/usr/bin/env python3
"""W123 Lane-alpha — official ICPC large-n supply census + verdict.

Runs :func:`coordpy.icpc_largen_supply_census_v1.assess_largen_supply_v1` over
the pinned official-org surface census and (by default) RE-VERIFIES the pinned
problem-package counts live via the GitHub API (NIM-free) so the verdict is
evidence-backed, not hand-asserted.  Emits the verdict JSON the W123 runbook
consumes.  No model inference, no spend.

Usage::

    python scripts/run_w123_largen_supply_census_v1.py            # verify live, then emit
    python scripts/run_w123_largen_supply_census_v1.py --offline  # pinned snapshot only
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordpy.icpc_largen_supply_census_v1 import (  # noqa: E402
    OFFICIAL_ICPC_SURFACE_CENSUS_V1,
    assess_largen_supply_v1,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "w123", "largen_supply")
OUT_PATH = os.path.join(OUT_DIR, "supply_census_verdict_v1.json")


def _gh_json(path: str):
    out = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:200])
    return json.loads(out.stdout)


def _live_count(surface) -> int:
    """Re-derive a surface's raw problem-package count from the live org."""
    src = surface.source
    if "@" in src:  # ECNA archive year folder (per-problem *.zip Kattis packages)
        repo, _, year = src.partition("@")
        repo = repo.split("github.com/")[-1]
        entries = _gh_json(f"repos/{repo}/contents/{year}")
        return len([e for e in entries if str(e.get("name", "")).endswith(".zip")])
    repo = src.split("github.com/")[-1]
    tree = _gh_json(f"repos/{repo}/git/trees/HEAD?recursive=1")
    return len([t for t in tree.get("tree", [])
                if str(t.get("path", "")).endswith("problem.yaml")])


def verify_live() -> dict:
    """Compare pinned counts to live counts for package-bearing surfaces."""
    rows, mismatches = [], []
    for s in OFFICIAL_ICPC_SURFACE_CENSUS_V1:
        if not s.package_bearing:
            continue
        try:
            live = _live_count(s)
        except Exception as exc:  # network / gh / rate-limit
            return {"verified_live": False, "error": str(exc), "rows": rows}
        ok = live == s.n_problem_packages
        rows.append({"key": s.key, "pinned": s.n_problem_packages, "live": live, "match": ok})
        if not ok:
            mismatches.append(s.key)
    return {"verified_live": True, "all_match": not mismatches,
            "mismatches": mismatches, "rows": rows}


def main() -> int:
    offline = "--offline" in sys.argv
    provenance = {"method": "pinned snapshot only"} if offline else verify_live()

    verdict = assess_largen_supply_v1()
    verdict["census_provenance"] = provenance

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(verdict, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("=" * 72)
    print("W123 Lane-alpha — official ICPC large-n supply census")
    print("=" * 72)
    r, e = verdict["resistant"], verdict["exposed"]
    print(f"RESISTANT: {r['n_surfaces']} post-cutoff surfaces | "
          f"{r['raw_problem_packages']} raw -> ~{r['est_tier1']} tier-1 | "
          f">=100? {r['reaches_target_even_at_upper_bound']}")
    print(f"EXPOSED  : {e['n_surfaces']} pre-cutoff surfaces  | "
          f"{e['raw_problem_packages']} raw -> ~{e['est_tier1']} tier-1 | "
          f">=100? {e['reaches_target_even_at_upper_bound']}")
    print(f"constructible (both>=100): {verdict['largen_matched_battlefield_constructible']}")
    print(f"large-n spend gate open  : {verdict['largen_spend_gate_open']}")
    print(f"verdict                  : {verdict['verdict']}")
    print(f"census_cid               : {verdict['census_cid']}")
    if not offline:
        print(f"live re-verify           : verified={provenance.get('verified_live')} "
              f"all_match={provenance.get('all_match')}")
    print(f"blocker: {verdict['binding_blocker']}")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
