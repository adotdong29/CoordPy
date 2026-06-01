"""W128 Lane α — non-scaffoldable hard-cluster protocol + EXPOSED supply census (NIM-free).

Reads the W127 capability atlas, derives the W128 hard-cluster RESISTANT target set
(public family in {graph_flow, simulation_grid}, scaffoldable_flag=False — the operator-named
minimum), writes a STRICT per-cluster protocol (cluster id / dominant evidence / why scaffold
transfer is the wrong mechanism / what search signal might help), runs the EXPOSED
hard-cluster supply census (which families have enough exposed dev-target supply), and records
the honest W79 substrate-controller applicability examination (RDA4 literal-bridge kill).

$0 NIM.  Emits results/w128/cluster_protocol/cluster_protocol_v1.json.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.family_scaffold_generation_v1 as G  # noqa: E402
from coordpy.resistant_capability_atlas_v1 import classify_reference_family_v1  # noqa: E402
import coordpy.role_diverse_algorithm_search_v1 as R  # noqa: E402

ATLAS = os.path.join(ROOT, "results", "w127", "atlas", "capability_atlas_v1.json")
OUT_DIR = os.path.join(ROOT, "results", "w128", "cluster_protocol")
HARD_FAMILIES = ("graph_flow", "simulation_grid")  # operator-named minimum

# Per-cluster STRICT protocol: why a SCAFFOLD is the wrong tool, what SEARCH signal may help.
_PROTOCOL = {
    "graph_flow": {
        "why_scaffold_wrong": (
            "graph_flow has thin EXPOSED teacher coverage (atlas teacher count 1; public "
            "EXPOSED supply 0), so there is no reusable family skeleton to transfer; and the "
            "failures are 95% wrong-algorithm — the model emits a plausible traversal that is "
            "the wrong graph model (wrong edges / wrong objective), which a structural "
            "skeleton cannot correct."),
        "search_signal": (
            "role-diverse ALGORITHM proposal (BFS vs Dijkstra vs flow/matching vs DSU) + "
            "DERIVED counterexamples that exercise the graph model (disconnected / cyclic / "
            "weighted edge cases the samples omit) to ELIMINATE the wrong model before "
            "committing; abstain when the proposals irreconcilably diverge."),
    },
    "simulation_grid": {
        "why_scaffold_wrong": (
            "simulation_grid failures are diverse and theme-biased (chess / maze / spies); a "
            "skeleton encodes I/O + a loop but NOT the problem-specific transition rule, and "
            "3/4 of the resistant simulation_grid problems fail on HIDDEN tests (a wrong rule "
            "that looks right on the public samples), which a scaffold cannot detect."),
        "search_signal": (
            "role-diverse rule HYPOTHESES + DERIVED edge-case inputs (boundary cells / "
            "wrap-around / empty grid) used as a candidate-AGREEMENT oracle to catch the "
            "looks-right-fails-hidden rule; abstain on tie (RDA3) or break the tie with the "
            "model's predicted-expected on the derived cases (RDA4)."),
    },
}


def main() -> int:
    atlas = json.load(open(ATLAS))["atlas"]
    entries = atlas["entries"]

    # --- RESISTANT hard-cluster target set (operator-named minimum) ---
    hard = [e for e in entries if e["dominant_algorithm_family"] in HARD_FAMILIES]
    clusters = {}
    for fam in HARD_FAMILIES:
        members = [e for e in hard if e["dominant_algorithm_family"] == fam]
        clusters[fam] = {
            "n_members": len(members),
            "members": [{
                "short_name": e["short_name"],
                "reference_family_signal": e.get("reference_family_signal"),
                "scaffoldable_flag": e["scaffoldable_flag"],
                "failure_visibility": e["failure_visibility"],
                "best_sample_pass_frac": e.get("best_sample_pass_frac"),
                "n_distinct_digests": e.get("n_distinct_digests"),
            } for e in members],
            "dominant_evidence": {
                "n_hidden": sum(1 for e in members if e["failure_visibility"] == "hidden"),
                "n_visible": sum(1 for e in members if e["failure_visibility"] == "visible"),
                "all_scaffoldable_false": all(not e["scaffoldable_flag"] for e in members),
            },
            "why_scaffold_wrong": _PROTOCOL[fam]["why_scaffold_wrong"],
            "search_signal": _PROTOCOL[fam]["search_signal"],
        }
    target_shorts = sorted(e["short_name"] for e in hard)

    # --- EXPOSED hard-cluster supply census (NIM-free) ---
    probs = G.load_exposed_problems_v1("/tmp/w121_icpc")
    pub = Counter()
    by_family = {}
    for p in probs:
        cls = G.target_family_ranking_v1(p.statement, p.samples)
        pub[cls.family] += 1
        by_family.setdefault(cls.family, []).append(p.short_name)
    census = {
        "exposed_root": "/tmp/w121_icpc", "n_exposed": len(probs),
        "public_family_counts": dict(pub),
        "hard_named_supply": {f: pub.get(f, 0) for f in HARD_FAMILIES},
        "non_scaffoldable_supply": {f: pub.get(f, 0)
                                    for f in R.NON_SCAFFOLDABLE_FAMILIES},
        "non_scaffoldable_total": sum(pub.get(f, 0) for f in R.NON_SCAFFOLDABLE_FAMILIES),
        "non_scaffoldable_members": {f: sorted(by_family.get(f, []))
                                     for f in R.NON_SCAFFOLDABLE_FAMILIES if pub.get(f, 0)},
        "graph_flow_exposed_supply_cap": pub.get("graph_flow", 0) == 0,
    }

    out = {
        "schema": "coordpy.w128_cluster_protocol.v1", "lane": "alpha_cluster_protocol",
        "verified_on": _dt.date.today().isoformat(), "nim_spend": 0,
        "atlas_cid": atlas.get("atlas_cid"),
        "hard_families_named": list(HARD_FAMILIES),
        "resistant_hard_target_shorts": target_shorts,
        "n_resistant_hard_targets": len(target_shorts),
        "clusters": clusters,
        "exposed_supply_census": census,
        "substrate_controller_applicability": R.examine_substrate_controller_applicability_v1(),
        "dev_bench_target_rule": (
            "EXPOSED dev targets = public family in NON_SCAFFOLDABLE_FAMILIES "
            "(simulation_grid PRIORITY = named present cluster); graph_flow EXPOSED supply=0 "
            "=> graph_flow is resistant-probe-only (registered exposed-supply cap)."),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "cluster_protocol_v1.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"resistant hard targets ({len(target_shorts)}): {target_shorts}")
    for fam in HARD_FAMILIES:
        c = clusters[fam]
        print(f"  {fam}: {c['n_members']} members  hidden={c['dominant_evidence']['n_hidden']} "
              f"all_nonscaffold={c['dominant_evidence']['all_scaffoldable_false']}")
    print(f"EXPOSED hard-named supply: {census['hard_named_supply']}  "
          f"(graph_flow exposed cap={census['graph_flow_exposed_supply_cap']})")
    print(f"EXPOSED non-scaffoldable supply: {census['non_scaffoldable_supply']} "
          f"(total {census['non_scaffoldable_total']})")
    print(f"W79 controllers all-substrate-specific (RDA4 literal bridge would be fake): "
          f"{out['substrate_controller_applicability']['all_substrate_specific']}")
    print(f"-> {os.path.join(OUT_DIR, 'cluster_protocol_v1.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
