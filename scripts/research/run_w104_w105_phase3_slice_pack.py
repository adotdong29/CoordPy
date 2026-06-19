#!/usr/bin/env python3
"""W104 — W105 Phase 3 slice pack pre-builder (planning lane).

Pre-builds the 100-problem HumanEval+ Phase 3 slice + 3 seeds so
W105 (if entitled by W104 Branch A) is launched as execution, not
paperwork.

Slice construction (deterministic; locked here):

  1. Inner kernel = W103 helper-anchored 30-problem slice
     (helper-priority order; BYTE-FOR-BYTE from the W103
     provenance JSON).
  2. Mid-shell = extension via `coordpy.code_slice_selector_v1.
     propose_cheap_pilot_slice(bench='humaneval_plus',
     n_problems=120)` (oversize ask); de-duplicated on `task_id`
     against the inner kernel.
  3. Outer top-up = `propose_cheap_pilot_slice(bench='humaneval',
     n_problems=120)` (base HumanEval; carried over the seam
     to HumanEval+ via the EvalPlus identity of HumanEval/n
     task_ids); de-duplicated against everything seen so far.
  4. Corpus-fill = if STILL fewer than 100 task_ids, walk the
     full SHA-pinned HumanEval+ corpus in deterministic order
     (lexicographic by integer suffix of the HumanEval/n
     task_id) and add unseen task_ids until n=100.
  5. The final 100-tuple is the W105 Phase 3 slice (helper-
     priority order).  Slice CID =
     SHA-256(",".join(task_ids).encode("utf-8")).

Seeds for W105 Phase 3 (locked):
  * 105_001 / 105_002 / 105_003 — uniform spacing in the W105
    seed namespace; isolated from W88 / W89 / W101 / W102 /
    W103 / W104 (104_001) namespaces.

Output: `data/w105/phase3_slice_pack/<run_id>/slice_pack.json`
+ `slice_pack.md` + the latest_run pointer.

The driver is NIM-free; it never calls a language model.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.code_slice_selector_v1 import (  # noqa: E402
    load_mining_report,
    propose_cheap_pilot_slice,
)
from coordpy.humaneval_plus_loader_v1 import (  # noqa: E402
    is_humaneval_plus_cached,
    load_humaneval_plus_corpus_v1,
)


# W105 seed namespace — locked here so future seed selection
# cannot collide with prior milestones' audit chains.
W105_PHASE3_SEEDS: tuple[int, ...] = (105_001, 105_002, 105_003)


W104_W105_SLICE_PACK_SCHEMA_VERSION: str = (
    "coordpy.w104_w105_phase3_slice_pack.v1")


# W103 inner-kernel CIDs.  Locked.
W103_INNER_KERNEL_CID_HELPER_PRIORITY: str = (
    "c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2")


def _sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _load_w103_inner_kernel(
        *, provenance_path: Path) -> list[dict]:
    if not provenance_path.exists():
        raise SystemExit(
            f"W104 slice pack: W103 provenance JSON missing at "
            f"{provenance_path}.")
    with open(provenance_path) as f:
        prov = json.load(f)
    inner = list(prov.get("helper_priority_slice") or [])
    if len(inner) != 30:
        raise SystemExit(
            "W104 slice pack: expected 30-problem inner kernel; "
            f"got {len(inner)}")
    # Recompute the CID from the task_ids to be sure.
    tids = [str(e["task_id"]) for e in inner]
    cid = _sha256_hex_bytes(",".join(tids).encode("utf-8"))
    if cid != W103_INNER_KERNEL_CID_HELPER_PRIORITY:
        raise SystemExit(
            "W104 slice pack: W103 inner-kernel CID mismatch: "
            f"got {cid!r} vs locked "
            f"{W103_INNER_KERNEL_CID_HELPER_PRIORITY!r}")
    return inner


def _natural_humaneval_task_id_key(tid: str) -> tuple[int, str]:
    """Sort key for `HumanEval/n` task_ids by integer suffix."""
    s = str(tid)
    if "/" in s:
        head, tail = s.split("/", 1)
        try:
            return int(tail), s
        except ValueError:
            return 10**9, s
    return 10**9, s


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W104 — W105 Phase 3 slice pack pre-builder"))
    ap.add_argument(
        "--mining-report",
        default=str(
            ROOT / "results" / "w102" / "arsenal_mining"
            / "w102_arsenal_20260526T000910Z"
            / "mining_report.json"),
        help="Path to the W102 arsenal-mining report JSON.")
    ap.add_argument(
        "--w103-provenance",
        default=str(
            ROOT / "results" / "w103" / "humaneval_plus_pilot"
            / "w103_humaneval_plus_pilot_meta_llama-3.3-70b-"
              "instruct_20260526T022037Z"
            / "provenance.json"),
        help=(
            "Path to W103 provenance JSON (source of the inner "
            "kernel)."))
    ap.add_argument(
        "--n-problems", type=int, default=100,
        help="Target slice size (default 100 for W105 Phase 3)")
    ap.add_argument(
        "--humaneval-plus-cache", default=None,
        help="HumanEval+ JSONL cache path override")
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "data" / "w105" / "phase3_slice_pack"),
        help="Output root for the slice pack artifact.")
    args = ap.parse_args()
    target = int(args.n_problems)

    print(f"  W104 → W105 Phase 3 slice pack pre-builder")
    print(f"  target n_problems = {target}")

    # 1. Inner kernel from W103 provenance.
    inner = _load_w103_inner_kernel(
        provenance_path=Path(args.w103_provenance))
    final: list[dict] = list(inner)
    seen: set[str] = {str(e["task_id"]) for e in inner}
    print(
        f"  inner kernel (W103 helper-priority order): "
        f"{len(final)} task_ids; CID matches locked constant.")

    # 2. Mid-shell extension via helper, oversize ask.
    mining = load_mining_report(args.mining_report)
    extension_n = max(120, target + 40)
    print(
        f"  mid-shell extension: helper proposal of "
        f"{extension_n} entries on humaneval_plus ...")
    mid_proposal = propose_cheap_pilot_slice(
        mining=mining, bench="humaneval_plus",
        n_problems=extension_n,
        bench_module_name="coordpy.humaneval_plus_reflexion_bench_v1")
    mid_cid = mid_proposal.proposal_cid
    n_added = 0
    for e in mid_proposal.proposal:
        if len(final) >= target:
            break
        tid = str(e.task_id)
        if tid in seen:
            continue
        seen.add(tid)
        final.append({
            "task_id": tid,
            "source": f"humaneval_plus:{e.cluster}:mid_shell",
        })
        n_added += 1
    print(f"    +{n_added} unique task_ids; final = {len(final)}")

    # 3. Outer top-up from base humaneval helper proposal.
    if len(final) < target:
        print(
            f"  outer top-up: helper proposal of "
            f"{extension_n} entries on base humaneval ...")
        outer_proposal = propose_cheap_pilot_slice(
            mining=mining, bench="humaneval",
            n_problems=extension_n,
            bench_module_name="coordpy.humaneval_real_bench_v1")
        outer_cid = outer_proposal.proposal_cid
        n_added = 0
        for e in outer_proposal.proposal:
            if len(final) >= target:
                break
            tid = str(e.task_id)
            if tid in seen:
                continue
            seen.add(tid)
            final.append({
                "task_id": tid,
                "source": f"humaneval:{e.cluster}:outer_top_up",
            })
            n_added += 1
        print(
            f"    +{n_added} unique task_ids; "
            f"final = {len(final)}")
    else:
        outer_cid = ""

    # 4. Corpus-fill if still short.
    if len(final) < target:
        if not is_humaneval_plus_cached(
                cache_path=args.humaneval_plus_cache):
            raise SystemExit(
                "W104 slice pack: HumanEval+ cache absent; "
                "cannot run corpus-fill step.")
        corpus = load_humaneval_plus_corpus_v1(
            cache_path=args.humaneval_plus_cache)
        ordered_corpus = sorted(
            corpus, key=lambda p: _natural_humaneval_task_id_key(
                p.task_id))
        print(
            f"  corpus-fill: walking SHA-pinned HumanEval+ "
            f"corpus ({len(ordered_corpus)} problems) in natural "
            "order ...")
        n_added = 0
        for p in ordered_corpus:
            if len(final) >= target:
                break
            tid = str(p.task_id)
            if tid in seen:
                continue
            seen.add(tid)
            final.append({
                "task_id": tid,
                "source": "humaneval_plus_corpus:corpus_fill",
            })
            n_added += 1
        print(
            f"    +{n_added} unique task_ids; "
            f"final = {len(final)}")

    if len(final) < target:
        raise SystemExit(
            f"W104 slice pack: only assembled {len(final)} "
            f"unique task_ids; target was {target}.")

    task_ids = [str(e["task_id"]) for e in final]
    pack_cid = _sha256_hex_bytes(
        ",".join(task_ids).encode("utf-8"))

    # Anti-pattern guard: refuse if the slice degenerated to
    # only shared_wins or corpus_fill (the structural-defence
    # parallel to the W103 RUNBOOK's helper-anchored slice
    # rule).
    source_kinds = {
        str(e["source"]).split(":")[1]
        if ":" in str(e["source"]) else "kernel"
        for e in final}
    if source_kinds <= {"shared_wins", "corpus_fill"}:
        raise SystemExit(
            "W104 slice pack: degenerated to shared_wins / "
            "corpus_fill only; no rescue or stress surface.")

    run_id = _dt.datetime.now(
        _dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / f"w105_phase3_slice_pack_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pack = {
        "schema": W104_W105_SLICE_PACK_SCHEMA_VERSION,
        "n_problems": int(len(final)),
        "phase3_seeds": list(W105_PHASE3_SEEDS),
        "pack_cid": str(pack_cid),
        "inner_kernel_cid_w103_helper_priority": str(
            W103_INNER_KERNEL_CID_HELPER_PRIORITY),
        "mid_shell_helper_proposal_cid": str(mid_cid),
        "outer_top_up_helper_proposal_cid": str(outer_cid),
        "task_ids_helper_priority": list(task_ids),
        "per_entry": list(final),
        "cluster_mix": {},
        "phase3_per_seed_per_scale_budget_nim_calls": 3300,
        "phase3_total_budget_nim_calls_two_scales": 6600,
        "k_multi_sample": 5,
        "scales_locked": [
            "meta/llama-3.3-70b-instruct",
            "meta/llama-3.1-405b-instruct",
        ],
    }
    # Cluster mix aggregation.
    mix: dict[str, int] = {}
    for e in final:
        src = str(e["source"])
        mix[src] = int(mix.get(src, 0)) + 1
    pack["cluster_mix"] = mix
    pack_path = out_dir / "slice_pack.json"
    with open(pack_path, "w") as f:
        json.dump(pack, f, indent=2, default=str)
    md_path = out_dir / "slice_pack.md"
    lines: list[str] = [
        "# W105 Phase 3 slice pack (pre-built in W104)",
        "",
        f"* schema: `{pack['schema']}`",
        f"* pack_cid: `{pack['pack_cid']}`",
        f"* n_problems: {pack['n_problems']}",
        f"* phase3_seeds: {pack['phase3_seeds']}",
        f"* scales locked: {pack['scales_locked']}",
        f"* per-scale per-seed budget: "
        f"{pack['phase3_per_seed_per_scale_budget_nim_calls']} "
        "NIM calls",
        f"* total Phase 3 budget across two scales: "
        f"{pack['phase3_total_budget_nim_calls_two_scales']} "
        "NIM calls",
        "",
        "## Cluster mix",
        "",
        "| Source | Count |",
        "|---|---:|",
    ]
    for src in sorted(mix.keys()):
        lines.append(f"| `{src}` | {mix[src]} |")
    lines.append("")
    lines.append("## Slice (helper-priority order)")
    lines.append("")
    lines.append("| # | task_id | source |")
    lines.append("|---|---|---|")
    for i, e in enumerate(final):
        lines.append(
            f"| {i+1} | {e['task_id']} | `{e['source']}` |")
    md_path.write_text("\n".join(lines) + "\n")
    latest = out_dir.parent / "latest_run.txt"
    with open(latest, "w") as f:
        f.write(out_dir.name + "\n")
    print()
    print(f"  pack_cid: {pack_cid}")
    print(f"  cluster mix:")
    for k in sorted(mix):
        print(f"    {k}: {mix[k]}")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
