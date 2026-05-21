#!/usr/bin/env python3
"""W87 / P3 #46 — Multi-modal substrate offline re-verifier.

Re-derives the bench_cid, cross-modality Merkle root, and the
per-payload payload_cids from the bench report + sidecars, then
re-asserts every load-bearing closure bool.  Exits 0 iff every
check passes.

Usage::

    python scripts/verify_w87_multi_modal_audit_chain.py \
        --report results/w87/multi_modal/<TS>/multi_modal_v1_bench_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def _canonical_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


LOAD_BEARING_BOOLS = (
    "multimodal_payload_for_three_modalities",
    "vision_adapter_loads_real_vlm_and_reads_hidden_state",
    "code_adapter_loads_real_codemodel_and_reads_hidden_state",
    "composed_pipeline_runs_on_team_with_at_least_two_modalities",
    "audit_chain_captures_all_modalities",
    "merkle_root_verifiable",
    "replay_byte_identity_per_modality_at_precision_floor",
)


def _verify(report_path: pathlib.Path) -> int:
    if not report_path.is_file():
        print(f"FAIL: report not found at {report_path}",
              file=sys.stderr)
        return 1
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    fails = 0
    passes = 0

    def _check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails, passes
        if ok:
            passes += 1
            print(f"PASS {name}{(': ' + detail) if detail else ''}")
        else:
            fails += 1
            print(f"FAIL {name}{(': ' + detail) if detail else ''}",
                  file=sys.stderr)

    # Re-derive bench_cid.
    recorded_bench_cid = str(rep.get("bench_cid", ""))
    rep_for_hash = {**rep, "bench_cid": ""}
    derived_bench_cid = _sha256_hex({
        "kind": "w87_multi_modal_v1_bench_v1",
        "report": rep_for_hash,
    })
    _check(
        "bench_cid",
        recorded_bench_cid == derived_bench_cid,
        f"recorded={recorded_bench_cid[:16]}... "
        f"derived={derived_bench_cid[:16]}...")

    # Re-derive cross-modality Merkle root from sidecar.
    sidecar_root = report_path.parent / (
        "cross_modality_merkle_root.json")
    if sidecar_root.is_file():
        root_obj = json.loads(
            sidecar_root.read_text(encoding="utf-8"))
        leaves = root_obj.get("per_modality_payload_cids", [])
        # Canonical sort by (modality, payload_cid)
        leaves_sorted = sorted(
            [tuple(x) for x in leaves],
            key=lambda t: (t[0], t[1]))
        derived_root = _sha256_hex({
            "kind": "w87_cross_modality_merkle_root_leaves_v1",
            "leaves": [[m, c] for m, c in leaves_sorted],
        })
        _check(
            "cross_modality_merkle_root_cid",
            str(root_obj.get("root_cid", "")) == derived_root,
            f"recorded={root_obj.get('root_cid', '')[:16]}... "
            f"derived={derived_root[:16]}...")
        _check(
            "n_modalities_in_root",
            int(root_obj.get("n_modalities", 0)) >= 2,
            f"n={root_obj.get('n_modalities', 0)}")
        # Confirm every per-modality payload_cid in the report
        # appears in the root.
        recorded_cids = {
            rep.get("text_payload_cid"),
            rep.get("code_payload_cid"),
            rep.get("image_payload_cid"),
        }
        root_cids = {c for _, c in leaves_sorted}
        _check(
            "all_payload_cids_in_root",
            recorded_cids.issubset(root_cids),
            f"missing={recorded_cids - root_cids}")
    else:
        _check(
            "cross_modality_merkle_root_sidecar_present",
            False,
            f"missing {sidecar_root}")

    # Re-derive per-modality payload_cids from payload_extras
    # sidecar.
    sidecar_extras = report_path.parent / "payload_extras.json"
    if sidecar_extras.is_file():
        extras = json.loads(
            sidecar_extras.read_text(encoding="utf-8"))
        for modality, payload_dict in extras.items():
            derived = _sha256_hex({
                "kind": "w87_multi_modal_payload_v1",
                "payload": payload_dict,
            })
            recorded = rep.get(f"{modality}_payload_cid", "")
            _check(
                f"{modality}_payload_cid",
                str(recorded) == derived,
                f"recorded={str(recorded)[:16]}... "
                f"derived={derived[:16]}...")
    else:
        _check(
            "payload_extras_sidecar_present",
            False,
            f"missing {sidecar_extras}")

    # Per-modality precision floor: assert within-tolerance.
    sidecar_floor = report_path.parent / (
        "per_modality_precision_floors.json")
    if sidecar_floor.is_file():
        floors = json.loads(
            sidecar_floor.read_text(encoding="utf-8"))
        for f in floors:
            _check(
                f"{f['modality']}_precision_floor_within_tolerance",
                bool(f.get("floor_within_tolerance_fp32", False)),
                f"floor_fp32={f.get('floor_fp32')} "
                f"tolerance={f.get('tolerance_fp32')}")
    else:
        _check(
            "precision_floor_sidecar_present",
            False,
            f"missing {sidecar_floor}")

    # Anti-cheat: vision encoder_kind must be "hf_vlm" if the
    # vlm_loaded_real flag is True.
    if bool(rep.get("vlm_loaded_real", False)):
        _check(
            "vision_encoder_kind_is_hf_vlm",
            str(rep.get("image_encoder_kind", "")) == "hf_vlm",
            f"encoder_kind={rep.get('image_encoder_kind')}")
    else:
        _check(
            "vision_encoder_kind_is_stub",
            "stub" in str(rep.get("image_encoder_kind", "")),
            f"encoder_kind={rep.get('image_encoder_kind')}")
    # Anti-cheat: code encoder_kind must be "hf_causal_lm" if
    # the code_loaded_real flag is True.
    if bool(rep.get("code_loaded_real", False)):
        _check(
            "code_encoder_kind_is_hf_causal_lm",
            str(rep.get("code_encoder_kind", "")) == "hf_causal_lm",
            f"encoder_kind={rep.get('code_encoder_kind')}")
    # Anti-cheat: code must carry the AST extras.
    _check(
        "code_carries_ast_function_count",
        int(rep.get("code_ast_n_functions", 0)) >= 1,
        f"n_functions={rep.get('code_ast_n_functions')}")

    # Re-assert every load-bearing closure bool.
    for k in LOAD_BEARING_BOOLS:
        _check(k, bool(rep.get(k, False)), str(rep.get(k)))

    print()
    print(f"OVERALL: {'PASS' if fails == 0 else 'FAIL'} "
          f"({passes} passed, {fails} failed)")
    return 0 if fails == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="W87 / P3 #46 — multi-modal audit re-verifier")
    parser.add_argument(
        "--report", required=True,
        help="Path to multi_modal_v1_bench_report.json")
    args = parser.parse_args(argv)
    return _verify(pathlib.Path(args.report).resolve())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
