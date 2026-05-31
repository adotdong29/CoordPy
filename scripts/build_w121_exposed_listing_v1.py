#!/usr/bin/env python3
"""W121 — LIVE builder of the matched EXPOSED official-ICPC control listing (NIM-FREE).

This is the only network/exec step for Lane alpha.  It classifies, problem-by-problem,
the SAME two official ICPC org surface families W120 used for its RESISTANT battlefield
(``icpc/na-ecna-archive`` + ``icpc/na-rocky-mountain-*-public``), but on the
immediately-preceding PRE-cutoff (exposed-for-Maverick) year-drops:

* ECNA archive year folders ``2022-2023`` + ``2023-2024``  (per-problem .zip packages)
* ``icpc/na-rocky-mountain-2022-2023-public``  (Kattis ``problems/<p>`` layout)
* ``icpc/na-rocky-mountain-2023-2024-public``  (minimal ``<p>/data/secret`` layout)

Each problem is classified by a TOTAL deterministic read of its ``problem.yaml`` (when
present) + tree, into the SAME kind taxonomy as W120
(passfail / passfail_float / custom_with_validator / custom_no_validator / interactive /
scoring), the secret ``.in`` cases are counted, accepted Python references are located,
and a NIM-free grader self-test runs an accepted Python solution in a fresh isolated
subprocess against the official secret cases (R8 evidence on the exposed surfaces).

Output: the pinned listing tuples (repo, short, contest_date, kind, n_secret, n_acc_py),
the per-surface self-test summary, and the raw-classification SHA-256 — copied verbatim
into ``coordpy.coordpy_icpc_exposed_control_v1`` as the pinned snapshot (the pure module
is NIM-free / network-free and reuses the W120 machinery).

Usage::

    python scripts/build_w121_exposed_listing_v1.py --work /tmp/w121_icpc \
        [--selftest] [--out results/w121/exposed_control/exposed_listing_live.json]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    KIND_PASSFAIL, KIND_PASSFAIL_FLOAT, KIND_CUSTOM_WITH_VALIDATOR,
    KIND_CUSTOM_NO_VALIDATOR, KIND_INTERACTIVE, KIND_SCORING,
    grade_icpc_candidate_case_v1,
)

# --- the pre-committed exposed surfaces (mirror of W120's 2-ECNA + 2-RMRC, pre-cutoff
# side; ALL four ship the SAME W120 artifact set: a problem_statement/*.tex + an
# executable secret-case grader).  RMRC 2023-2024 is EXCLUDED (its minimal package ships
# secret data but NO problem_statement, so it cannot present a statement to the model — a
# W120-comparability failure); the rule advances to the next pre-cutoff RMRC drop (2021).
# (surface, source_repo, contest_date, rel_root_under_work, structure)
EXPOSED_SURFACES = [
    ("ECNA", "icpc/na-ecna-archive", "2022-11-12", "un/2022-2023", "archive"),
    ("ECNA", "icpc/na-ecna-archive", "2023-11-11", "un/2023-2024", "archive"),
    ("RMRC", "icpc/na-rocky-mountain-2021-public", "2022-03-14",
     "rmrc-2021/problems", "kattis"),
    ("RMRC", "icpc/na-rocky-mountain-2022-2023-public", "2023-02-25",
     "rmrc-2022-2023/problems", "kattis"),
]

_FLOAT_RE = re.compile(r"float_(absolute|relative)?_?tolerance", re.I)


def _read_yaml_fields(yaml_path: Path) -> dict:
    """Minimal, dependency-free read of the few problem.yaml fields we classify on."""
    out = {"validation": "", "validator_flags": "", "type": ""}
    if not yaml_path.exists():
        return out
    txt = yaml_path.read_text(errors="replace")
    for line in txt.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("validation:"):
            out["validation"] = s.split(":", 1)[1].strip().strip('"\'')
        elif low.startswith("validator_flags:"):
            out["validator_flags"] = s.split(":", 1)[1].strip()
        elif low.startswith("type:"):
            out["type"] = s.split(":", 1)[1].strip()
    return out


def _has_validator(pkg: Path) -> bool:
    ov = pkg / "output_validators"
    return ov.is_dir() and any(ov.rglob("*"))


def _classify_kind(pkg: Path) -> str:
    f = _read_yaml_fields(pkg / "problem.yaml")
    val = f["validation"].lower()
    if "interactive" in val or (pkg / "interactor").exists():
        return KIND_INTERACTIVE
    if "scoring" in val or f["type"].lower() == "scoring":
        return KIND_SCORING
    if "custom" in val:
        return KIND_CUSTOM_WITH_VALIDATOR if _has_validator(pkg) else KIND_CUSTOM_NO_VALIDATOR
    # default / absent validation
    if _FLOAT_RE.search(f["validator_flags"]):
        return KIND_PASSFAIL_FLOAT
    return KIND_PASSFAIL


def _secret_cases(pkg: Path) -> list:
    """Return sorted (in_path, ans_path) pairs under data/secret (recursive)."""
    sec = pkg / "data" / "secret"
    pairs = []
    if sec.is_dir():
        for inp in sorted(sec.rglob("*.in")):
            ans = inp.with_suffix(".ans")
            if ans.exists():
                pairs.append((inp, ans))
    return pairs


def _accepted_py(pkg: Path) -> list:
    """Accepted Python references: submissions/accepted/*.py* (Kattis) or solutions/*.py*
    (minimal RMRC reference layout)."""
    out = []
    for sub in (pkg / "submissions" / "accepted", pkg / "solutions"):
        if sub.is_dir():
            for ext in ("*.py", "*.py3", "*.python"):
                out.extend(sorted(sub.glob(ext)))
    return sorted(set(out))


def _find_pkg_dirs(root: Path, structure: str) -> list:
    """Locate the package roots (the dir that directly contains data/secret)."""
    if structure == "kattis":
        cands = [d for d in sorted(root.iterdir()) if d.is_dir()]
    elif structure == "minimal":
        cands = [d for d in sorted(root.iterdir())
                 if d.is_dir() and (d / "data").is_dir()]
    else:  # archive: un/<year>/<zipbase>/<inner>/...  -> find dir with data/secret
        cands = []
        for zd in sorted(root.iterdir()):
            if not zd.is_dir():
                continue
            hit = None
            for sub in [zd] + sorted(zd.rglob("*")):
                if sub.is_dir() and (sub / "data" / "secret").is_dir():
                    hit = sub
                    break
            if hit is not None:
                cands.append(hit)
    return [d for d in cands if (d / "data" / "secret").is_dir()]


def _selftest_problem(pkg: Path, kind: str, *, max_cases: int, timeout_s: float) -> tuple:
    """Run the first accepted Python solution that passes ALL secret cases.
    Returns (n_cases_run, n_cases_passed_best, all_pass)."""
    cases = _secret_cases(pkg)[:max_cases]
    accs = _accepted_py(pkg)
    if not cases or not accs:
        return (0, 0, False)
    best = 0
    for sol in accs:
        code = sol.read_text(errors="replace")
        npass = 0
        for inp, ans in cases:
            r = grade_icpc_candidate_case_v1(
                candidate_code=code, stdin_text=inp.read_text(errors="replace"),
                expected_stdout=ans.read_text(errors="replace"), kind=kind,
                timeout_s=timeout_s)
            if r.passed:
                npass += 1
            else:
                break
        best = max(best, npass)
        if npass == len(cases):
            return (len(cases), len(cases), True)
    return (len(cases), best, False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/w121_icpc")
    ap.add_argument("--selftest", action="store_true",
                    help="run the NIM-free grader self-test (slower)")
    ap.add_argument("--selftest-max-cases", type=int, default=30)
    ap.add_argument("--selftest-timeout", type=float, default=8.0)
    ap.add_argument("--out",
                    default=str(ROOT / "results" / "w121" / "exposed_control"
                               / "exposed_listing_live.json"))
    args = ap.parse_args()
    work = Path(args.work)

    listing = []          # (repo, short, contest_date, kind, n_secret, n_acc_py)
    selftest = {}         # surface_label -> [(problem_id, run, passed, all_pass)]
    for surface, repo, cdate, rel, structure in EXPOSED_SURFACES:
        root = work / rel
        if not root.is_dir():
            print(f"  !! missing {root}", file=sys.stderr)
            continue
        pkgs = _find_pkg_dirs(root, structure)
        skey = f"{surface}:{cdate}"
        selftest.setdefault(skey, [])
        for pkg in pkgs:
            short = pkg.name if structure != "archive" else pkg.name
            kind = _classify_kind(pkg)
            n_secret = len(_secret_cases(pkg))
            n_in_only = len(list((pkg / "data" / "secret").rglob("*.in")))
            n_acc = len(_accepted_py(pkg))
            listing.append((repo, short, cdate, kind, n_in_only, n_acc))
            if args.selftest and kind in (KIND_PASSFAIL, KIND_PASSFAIL_FLOAT) and n_acc:
                run, passed, allp = _selftest_problem(
                    pkg, kind, max_cases=args.selftest_max_cases,
                    timeout_s=args.selftest_timeout)
                pid = f"icpc_{repo.split('/')[-1]}_{short}"
                selftest[skey].append((pid, run, passed, allp))

    listing.sort(key=lambda r: (r[2], r[0], r[1]))
    canon = json.dumps([list(r) for r in listing], sort_keys=True,
                       separators=(",", ":")).encode()
    sha = hashlib.sha256(canon).hexdigest()

    # ---- print the pinned listing (copy verbatim into the module) ----
    print(f"\n# raw_classification_sha256 = {sha}")
    print(f"# {len(listing)} problems classified across {len(EXPOSED_SURFACES)} surfaces\n")
    from collections import Counter
    by_kind = Counter(r[3] for r in listing)
    by_kind_exposed = Counter(r[3] for r in listing)  # all rows are pre-cutoff here
    print("EXPOSED_LISTING_SNAPSHOT_V1 = (")
    for r in listing:
        print(f'    ("{r[0]}", "{r[1]}", "{r[2]}", "{r[3]}", {r[4]}, {r[5]}),')
    print(")")
    print(f"\n# kind histogram: {dict(by_kind)}")
    n_core = by_kind.get(KIND_PASSFAIL, 0)
    print(f"# tier-1 pure pass-fail (core) = {n_core}  (>=30: {n_core >= 30})")

    st_summary = {}
    for skey, rows in selftest.items():
        n_problems = len(rows)
        n_allpass = sum(1 for r in rows if r[3])
        cases_run = sum(r[1] for r in rows)
        cases_pass = sum(r[2] for r in rows)
        st_summary[skey] = {"n_self_tested": n_problems, "n_all_pass": n_allpass,
                            "cases_run": cases_run, "cases_passed": cases_pass,
                            "rows": rows}
        if rows:
            print(f"# self-test {skey}: {n_allpass}/{n_problems} problems all-pass; "
                  f"{cases_pass}/{cases_run} cases")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "coordpy.w121_exposed_listing_live.v1",
        "built_on": _dt.date.today().isoformat(),
        "raw_classification_sha256": sha,
        "n_problems": len(listing),
        "listing": [list(r) for r in listing],
        "kind_histogram": dict(by_kind),
        "n_core_passfail": n_core,
        "selftest": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                     for k, v in st_summary.items()},
        "selftest_rows": {k: v["rows"] for k, v in st_summary.items()},
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
