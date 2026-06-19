"""W127 Lane α — build the resistant capability atlas ($0 NIM).

Reconstructs the 22 uniformly-unsolved resistant ICPC problems (W126 grade cache + the
W120 resistant 30-slice, CID ``01bf9ef869a56e20``; 330 already-paid Maverick generations)
and emits the machine-checkable capability atlas defined in RUNBOOK_W127 § 2.

The atlas separates HARD re-executable signals (failure visibility, typed per-generation
failure-category distribution, generation diversity, best public-sample pass fraction)
from a SOFT transparent lexicon family classifier over PUBLIC inputs only, plus an
analyst-only reference cross-check (NEVER model-facing).  $0 NIM.  Emits
results/w127/atlas/capability_atlas_v1.json.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    classify_battlefield_listing_v1, core_slice_cid_v1, select_battlefield_core_slice_v1)
import coordpy.controller_native_code_mechanism_v1 as M  # noqa: E402
import coordpy.resistant_capability_atlas_v1 as A  # noqa: E402
from coordpy.family_adapted_repair_synthesis_v1 import (  # noqa: E402
    load_exposed_teacher_corpus_v1)
from scripts.run_w120_icpc_pilot import load_pilot_problems  # noqa: E402

EXPECTED_SLICE_CID_30 = "01bf9ef869a56e20"
CORPUS_DIR = os.path.join(ROOT, "results", "w120", "icpc_pilot")
CACHE = os.path.join(ROOT, "results", "w126", "cache", "grade_cache_v1.json")
OUT_DIR = os.path.join(ROOT, "results", "w127", "atlas")
RESISTANT_PKG_ROOTS = ("/tmp/w120_icpc/pkgcache", "/tmp/w122_icpc/pkgcache_resistant")


def _latest_corpus() -> str:
    name = open(os.path.join(CORPUS_DIR, "latest_run.txt")).read().strip()
    return os.path.join(CORPUS_DIR, name, "icpc_reflexion_calls.jsonl")


def _find_reference_texts(short_name: str, *, max_files: int = 8) -> list:
    """Analyst-only: load the resistant target's OWN accepted-solution texts for the
    reference cross-check (RUNBOOK_W127 § 2 item 15).  NEVER fed to a generator."""
    out: list = []
    for root in RESISTANT_PKG_ROOTS:
        for p in sorted(glob.glob(os.path.join(
                root, "**", short_name, "submissions", "accepted", "*"), recursive=True)):
            if os.path.isfile(p):
                try:
                    out.append(open(p, encoding="utf-8", errors="replace").read())
                except OSError:
                    pass
            if len(out) >= max_files:
                return out
    return out


def _teacher_family_count(exposed_root: str) -> tuple[dict, int, int]:
    """Classify each EXPOSED teacher problem with the SAME public classifier used for
    targets (statement + accepted code) so the family space is consistent with what the
    leakage-clean G2 retriever keys on (RUNBOOK_W127 § 2/§ 4)."""
    paths = sorted(glob.glob(os.path.join(exposed_root, "**", "submissions", "accepted",
                                          "*.py"), recursive=True))
    by_problem: dict[str, list] = {}
    for p in paths:
        pdir = os.path.dirname(os.path.dirname(os.path.dirname(p)))
        short = os.path.basename(pdir)
        try:
            code = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        by_problem.setdefault(short, {"dir": pdir, "codes": []})["codes"].append(code)
    counts: dict[str, int] = {f: 0 for f in A.LOCKED_FAMILY_TAXONOMY}
    classified = 0
    for short, info in by_problem.items():
        tex = os.path.join(info["dir"], "problem_statement", "problem.tex")
        statement = open(tex, encoding="utf-8", errors="replace").read() \
            if os.path.isfile(tex) else ""
        fc = A.classify_family_v1(statement=statement, sample_text="",
                                  generation_codes=info["codes"])
        counts[fc.family] += 1
        classified += 1
    return counts, len(by_problem), classified


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposed-root", default="/tmp/w121_icpc")
    args = ap.parse_args()

    full = classify_battlefield_listing_v1()
    slice30 = select_battlefield_core_slice_v1(full, n_problems=30)
    slice_cid = core_slice_cid_v1(slice30)
    if not slice_cid.startswith(EXPECTED_SLICE_CID_30):
        raise SystemExit(f"slice CID {slice_cid[:16]} != {EXPECTED_SLICE_CID_30}; refusing.")
    problems = load_pilot_problems(list(slice30))
    recs = [json.loads(l) for l in open(_latest_corpus())]
    if len(recs) != 11 * len(problems):
        raise SystemExit(f"corpus {len(recs)} != {11*len(problems)}; refusing.")
    pools = [M.build_pool_from_records(recs[i * 11:i * 11 + 11], problems[i].problem_id)
             for i in range(len(problems))]

    cache = json.load(open(CACHE))
    unsolved_ids = set(cache["summary"]["unsolved_ids"])
    print(f"  slice cid={slice_cid[:16]}…  unsolved={len(unsolved_ids)}/30")

    teacher_counts, n_teacher_problems, n_classified = _teacher_family_count(
        args.exposed_root)
    print(f"  teacher family coverage over {n_teacher_problems} exposed problems "
          f"({n_classified} classified): "
          f"{ {k: v for k, v in teacher_counts.items() if v} }")

    entries = []
    for prob, pool in zip(problems, pools):
        if prob.problem_id not in unsolved_ids:
            continue
        cp = cache["problems"].get(prob.problem_id, {})
        gen_codes = [pool.a0_code, *pool.a1_codes, *pool.b_codes]
        ref_texts = _find_reference_texts(prob.short_name)
        e = A.build_atlas_entry_v1(
            problem=prob, generation_codes=gen_codes, cache_problem=cp,
            teacher_family_count=teacher_counts, reference_texts=ref_texts)
        entries.append(e)
        print(f"   {e.short_name:30s} {e.dominant_algorithm_family:18s} "
              f"vis={e.failure_visibility:7s} dig={e.digest_distribution} "
              f"cov={e.teacher_family_coverage} scaf={int(e.scaffoldable_flag)} "
              f"ref={e.reference_family_signal}({int(e.atlas_label_agrees)})")

    atlas = A.build_capability_atlas_v1(entries)
    verdict = {
        "schema": atlas.schema, "lane": "alpha_capability_atlas",
        "verified_on": _dt.date.today().isoformat(),
        "field": "W120 resistant official-ICPC 30-slice (22 uniformly-unsolved)",
        "slice_cid": slice_cid, "nim_spend": 0,
        "teacher_corpus": {"exposed_root": args.exposed_root,
                           "n_teacher_problems": n_teacher_problems,
                           "n_classified": n_classified,
                           "family_counts": {k: v for k, v in teacher_counts.items() if v}},
        "atlas": atlas.to_dict(),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "capability_atlas_v1.json")
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2, default=str)

    print()
    print(f"  HARD failure-mode roll-up: {atlas.failure_mode_summary}")
    print(f"  PUBLIC-label CLUSTER COUNTS: {atlas.cluster_counts}")
    print(f"  PUBLIC dominant cluster: {atlas.dominant_cluster}  "
          f"top2_concentration={atlas.concentration_top2_frac}")
    print(f"  REFERENCE-space (actual-algorithm) clusters: {atlas.reference_cluster_counts}")
    print(f"  label confidence: confirmed={atlas.n_ref_confirmed} "
          f"conflict={atlas.n_ref_conflict} unconfirmed={atlas.n_unconfirmed}  "
          f"agreement={atlas.atlas_label_agreement}")
    print(f"  SCAFFOLDABLE: {atlas.scaffoldable_count}/{atlas.n_problems}  "
          f"by_family={atlas.scaffoldable_by_family}")
    print(f"  atlas_cid={atlas.atlas_cid[:16]}…")
    print(f"  wrote {out}  ($0 NIM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
