#!/usr/bin/env python3
"""W87 / P3 #46 — Multi-Modal Substrate bench driver.

Exercises the full multi-modal substrate end-to-end:

  * Text payload from a deterministic embedding (W80 contract).
  * Code payload via ``CodeSubstrateAdapterV1`` (AST-aware reads).
    Uses a real causal LM (default ``distilgpt2``) when
    transformers is present; falls back to the stub_sha256_v1
    encoder otherwise.
  * Image payload via ``VisionSubstrateAdapterV1`` with a real
    open-weight VLM (``vikhyatk/moondream2`` by default — CPU
    runnable; ``HuggingFaceM4/idefics2-8b`` / ``llava-hf/
    llava-1.5-7b-hf`` for the Colab path).  Falls back to the
    stub_clip_style_v1 encoder when transformers is absent.
  * Composed pipeline assembles all per-modality payloads into a
    single cross-modality Merkle root.
  * Per-modality precision floor is measured by re-encoding and
    comparing.

Writes a content-addressed ``multi_modal_v1_bench_report.json``
under ``results/w87/multi_modal/<TS>/``.  Sidecars include the
per-modality payloads, the cross-modality Merkle root, and the
per-modality precision-floor blocks.

Verifier (``scripts/verify_w87_multi_modal_audit_chain.py``)
re-derives every CID and asserts every load-bearing bool.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coordpy.code_substrate_v1 import (
    W87_CODE_SUBSTRATE_V1_SCHEMA_VERSION,
    CodeSubstrateAdapterV1,
)
from coordpy.composed_multimodal_pipeline_v1 import (
    W87_COMPOSED_MULTI_MODAL_PIPELINE_V1_SCHEMA_VERSION,
    MultiModalAgentTurnV1,
    run_composed_multi_modal_pipeline_v1,
    verify_multi_modal_run_report_v1,
)
from coordpy.multi_modal_payload_v1 import (
    W87_MODALITY_PRECISION_FLOOR_DEFAULTS_FP32,
    W87_MULTI_MODAL_V1_SCHEMA_VERSION,
    EncoderFingerprintV1,
    Modality,
    build_multi_modal_payload_v1,
    measure_modality_precision_floor_fp32_v1,
)
from coordpy.vision_substrate_v1 import (
    W87_VISION_SUBSTRATE_V1_SCHEMA_VERSION,
    VisionSubstrateAdapterV1,
    make_tiny_png_v1,
    probe_vision_substrate_capability_v1,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


DEFAULT_VLM = "vikhyatk/moondream2"
DEFAULT_CODE_LM = "distilgpt2"

DEFAULT_SOURCE = """\
def quicksort(items):
    if len(items) < 2:
        return items
    pivot = items[0]
    less = [x for x in items[1:] if x <= pivot]
    greater = [x for x in items[1:] if x > pivot]
    return quicksort(less) + [pivot] + quicksort(greater)


class Sorter:
    def sort(self, xs):
        return quicksort(xs)
"""


def _text_payload(prompt: str) -> Any:
    """Build a deterministic text payload (no real model
    needed — text modality is fully exercised by the W80
    transformers_runtime_v1 path; here we ship a content-
    addressed payload with a hash-derived embedding so the
    multi-modal team has a third anchor)."""
    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    rng = np.random.default_rng(
        int.from_bytes(h[:8], "big") & ((1 << 64) - 1))
    emb = rng.normal(0.0, 1.0, (16, 8)).astype(np.float32)
    enc = EncoderFingerprintV1(
        schema=W87_MULTI_MODAL_V1_SCHEMA_VERSION,
        modality=Modality.TEXT.value,
        encoder_kind="deterministic_sha256_v1",
        model_name="deterministic_sha256",
        model_revision="v1",
        precision_tier="tier_fp32",
        embedding_dim=8)
    return build_multi_modal_payload_v1(
        modality=Modality.TEXT,
        raw_bytes=prompt.encode("utf-8"),
        embedding=emb,
        encoder=enc)


@dataclasses.dataclass(frozen=True)
class MultiModalBenchReportV1:
    schema: str
    timestamp_iso: str
    vlm_model_name: str
    code_model_name: str
    vlm_loaded_real: bool
    code_loaded_real: bool
    n_modalities: int
    cross_modality_root_cid: str
    run_report_cid: str
    text_payload_cid: str
    code_payload_cid: str
    image_payload_cid: str
    text_encoder_kind: str
    code_encoder_kind: str
    image_encoder_kind: str
    code_ast_n_functions: int
    image_hidden_shape: tuple[int, ...]
    code_hidden_shape: tuple[int, ...]
    text_hidden_shape: tuple[int, ...]
    text_floor_fp32: float
    code_floor_fp32: float
    image_floor_fp32: float
    text_floor_within_tolerance: bool
    code_floor_within_tolerance: bool
    image_floor_within_tolerance: bool
    bench_wall_seconds: float
    bench_cid: str
    config_cid: str
    # Load-bearing closure bools.
    multimodal_payload_for_three_modalities: bool
    vision_adapter_loads_real_vlm_and_reads_hidden_state: bool
    code_adapter_loads_real_codemodel_and_reads_hidden_state: bool
    composed_pipeline_runs_on_team_with_at_least_two_modalities: bool
    audit_chain_captures_all_modalities: bool
    merkle_root_verifiable: bool
    replay_byte_identity_per_modality_at_precision_floor: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "timestamp_iso": str(self.timestamp_iso),
            "vlm_model_name": str(self.vlm_model_name),
            "code_model_name": str(self.code_model_name),
            "vlm_loaded_real": bool(self.vlm_loaded_real),
            "code_loaded_real": bool(self.code_loaded_real),
            "n_modalities": int(self.n_modalities),
            "cross_modality_root_cid": str(
                self.cross_modality_root_cid),
            "run_report_cid": str(self.run_report_cid),
            "text_payload_cid": str(self.text_payload_cid),
            "code_payload_cid": str(self.code_payload_cid),
            "image_payload_cid": str(self.image_payload_cid),
            "text_encoder_kind": str(self.text_encoder_kind),
            "code_encoder_kind": str(self.code_encoder_kind),
            "image_encoder_kind": str(self.image_encoder_kind),
            "code_ast_n_functions": int(
                self.code_ast_n_functions),
            "image_hidden_shape": list(
                int(x) for x in self.image_hidden_shape),
            "code_hidden_shape": list(
                int(x) for x in self.code_hidden_shape),
            "text_hidden_shape": list(
                int(x) for x in self.text_hidden_shape),
            "text_floor_fp32": float(self.text_floor_fp32),
            "code_floor_fp32": float(self.code_floor_fp32),
            "image_floor_fp32": float(self.image_floor_fp32),
            "text_floor_within_tolerance": bool(
                self.text_floor_within_tolerance),
            "code_floor_within_tolerance": bool(
                self.code_floor_within_tolerance),
            "image_floor_within_tolerance": bool(
                self.image_floor_within_tolerance),
            "bench_wall_seconds": float(
                self.bench_wall_seconds),
            "bench_cid": str(self.bench_cid),
            "config_cid": str(self.config_cid),
            "multimodal_payload_for_three_modalities": bool(
                self.multimodal_payload_for_three_modalities),
            "vision_adapter_loads_real_vlm_and_reads_hidden_state":
                bool(self.vision_adapter_loads_real_vlm_and_reads_hidden_state),
            "code_adapter_loads_real_codemodel_and_reads_hidden_state":
                bool(self.code_adapter_loads_real_codemodel_and_reads_hidden_state),
            "composed_pipeline_runs_on_team_with_at_least_two_modalities":
                bool(self.composed_pipeline_runs_on_team_with_at_least_two_modalities),
            "audit_chain_captures_all_modalities": bool(
                self.audit_chain_captures_all_modalities),
            "merkle_root_verifiable": bool(
                self.merkle_root_verifiable),
            "replay_byte_identity_per_modality_at_precision_floor":
                bool(self.replay_byte_identity_per_modality_at_precision_floor),
        }

    def cid(self) -> str:
        rd = self.to_dict()
        rd_for_hash = {**rd, "bench_cid": ""}
        return _sha256_hex({
            "kind": "w87_multi_modal_v1_bench_v1",
            "report": rd_for_hash,
        })


def run_multi_modal_bench_v1(
        *, vlm_model_name: str = DEFAULT_VLM,
        code_model_name: str = DEFAULT_CODE_LM,
        require_real_vlm: bool = False,
        require_real_code: bool = False,
        out_dir: str | None = None,
        source: str = DEFAULT_SOURCE,
        embedding_dim: int = 2048,
) -> MultiModalBenchReportV1:
    t_start = time.time()
    # Text payload (always real — hash-deterministic).
    text_p = _text_payload("describe the image and review the code")
    # Code adapter (real if transformers available; stub otherwise).
    try:
        if require_real_code:
            code_adapter = CodeSubstrateAdapterV1(
                model_name=code_model_name,
                require_real_model=True)
            code_loaded_real = True
        else:
            code_adapter = CodeSubstrateAdapterV1(
                model_name=code_model_name,
                require_real_model=False)
            # Try to upgrade to real if possible.
            try:
                code_adapter._load_real_runtime_or_raise()
                code_loaded_real = True
            except Exception:  # noqa: BLE001
                code_loaded_real = False
    except Exception as exc:  # noqa: BLE001
        print(f"[w87/mm] code adapter falling back to stub: {exc}")
        code_adapter = CodeSubstrateAdapterV1(model_name="stub")
        code_loaded_real = False
    code_p = code_adapter.encode(source)
    # Vision adapter (real if transformers + VLM available;
    # stub otherwise).  Auto-pick bf16 on CUDA, fp32 on CPU —
    # avoids OOM for LLaVA-7B / Idefics-2-8B at fp32 on A100.
    cap = probe_vision_substrate_capability_v1()
    vlm_precision_tier = (
        "tier_bf16" if cap.cuda_available else "tier_fp32")
    try:
        if require_real_vlm:
            vision_adapter = VisionSubstrateAdapterV1(
                model_name=vlm_model_name,
                require_real_vlm=True,
                precision_tier=vlm_precision_tier,
                embedding_dim=int(embedding_dim))
            vlm_loaded_real = True
        elif cap.can_load_vlm:
            vision_adapter = VisionSubstrateAdapterV1(
                model_name=vlm_model_name,
                require_real_vlm=False,
                precision_tier=vlm_precision_tier,
                embedding_dim=int(embedding_dim))
            try:
                vision_adapter._load_real_vlm_or_raise()
                vlm_loaded_real = True
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[w87/mm] vision adapter falling back to "
                    f"stub: {exc}")
                vision_adapter = VisionSubstrateAdapterV1()
                vlm_loaded_real = False
        else:
            vision_adapter = VisionSubstrateAdapterV1()
            vlm_loaded_real = False
    except Exception as exc:  # noqa: BLE001
        print(f"[w87/mm] vision adapter falling back to stub: {exc}")
        vision_adapter = VisionSubstrateAdapterV1()
        vlm_loaded_real = False
    png = make_tiny_png_v1(seed=87, n=8)
    image_p = vision_adapter.encode_image(
        png, prompt="What do you see in this image?")
    # Per-modality precision floors: re-encode and compare.
    text_p2 = _text_payload(
        "describe the image and review the code")
    code_p2 = code_adapter.encode(source)
    image_p2 = vision_adapter.encode_image(
        png, prompt="What do you see in this image?")
    text_floor = measure_modality_precision_floor_fp32_v1(
        modality=Modality.TEXT,
        encoder_cid=text_p.encoder.cid(),
        embedding_a=text_p.embedding,
        embedding_b=text_p2.embedding)
    code_floor = measure_modality_precision_floor_fp32_v1(
        modality=Modality.CODE,
        encoder_cid=code_p.encoder.cid(),
        embedding_a=code_p.embedding,
        embedding_b=code_p2.embedding,
        tolerance=(
            5e-1 if code_loaded_real else 5e-3))  # bf16-wide
                                                  # for real;
                                                  # fp32 for stub
    image_floor = measure_modality_precision_floor_fp32_v1(
        modality=Modality.IMAGE,
        encoder_cid=image_p.encoder.cid(),
        embedding_a=image_p.embedding,
        embedding_b=image_p2.embedding,
        tolerance=(5e-1 if vlm_loaded_real else 5e-3))
    # Compose pipeline.
    W87CV = W87_COMPOSED_MULTI_MODAL_PIPELINE_V1_SCHEMA_VERSION
    turns = [
        MultiModalAgentTurnV1(
            schema=W87CV, agent_id="reader", role="reader",
            payload=text_p),
        MultiModalAgentTurnV1(
            schema=W87CV, agent_id="vision_critic",
            role="critic", payload=image_p),
        MultiModalAgentTurnV1(
            schema=W87CV, agent_id="code_implementer",
            role="implementer", payload=code_p),
    ]
    report = run_composed_multi_modal_pipeline_v1(
        run_label="w87_multi_modal_v1_bench",
        agent_turns=turns,
        per_modality_precision_floors=[
            text_floor, code_floor, image_floor])
    ok, detail = verify_multi_modal_run_report_v1(report)
    assert ok, f"composed pipeline did not re-verify: {detail}"
    wall = time.time() - t_start
    # Build the bench report.
    config_cid = _sha256_hex({
        "kind": "w87_multi_modal_v1_bench_config_v1",
        "vlm_model_name": str(vlm_model_name),
        "code_model_name": str(code_model_name),
        "source": str(source),
    })
    rep = MultiModalBenchReportV1(
        schema=W87_MULTI_MODAL_V1_SCHEMA_VERSION,
        timestamp_iso=str(
            _dt.datetime.now(_dt.UTC).isoformat()),
        vlm_model_name=str(vlm_model_name),
        code_model_name=str(code_model_name),
        vlm_loaded_real=bool(vlm_loaded_real),
        code_loaded_real=bool(code_loaded_real),
        n_modalities=int(report.n_modalities),
        cross_modality_root_cid=str(
            report.cross_modality_root.root_cid),
        run_report_cid=str(report.report_cid),
        text_payload_cid=str(text_p.payload_cid()),
        code_payload_cid=str(code_p.payload_cid()),
        image_payload_cid=str(image_p.payload_cid()),
        text_encoder_kind=str(text_p.encoder.encoder_kind),
        code_encoder_kind=str(code_p.encoder.encoder_kind),
        image_encoder_kind=str(image_p.encoder.encoder_kind),
        code_ast_n_functions=int(
            code_p.extras.get("n_functions", 0)),
        image_hidden_shape=tuple(
            int(x) for x in image_p.embedding.shape),
        code_hidden_shape=tuple(
            int(x) for x in code_p.embedding.shape),
        text_hidden_shape=tuple(
            int(x) for x in text_p.embedding.shape),
        text_floor_fp32=float(text_floor.floor_fp32),
        code_floor_fp32=float(code_floor.floor_fp32),
        image_floor_fp32=float(image_floor.floor_fp32),
        text_floor_within_tolerance=bool(
            text_floor.floor_within_tolerance_fp32),
        code_floor_within_tolerance=bool(
            code_floor.floor_within_tolerance_fp32),
        image_floor_within_tolerance=bool(
            image_floor.floor_within_tolerance_fp32),
        bench_wall_seconds=float(wall),
        bench_cid="",
        config_cid=str(config_cid),
        multimodal_payload_for_three_modalities=(
            int(report.n_modalities) >= 3),
        vision_adapter_loads_real_vlm_and_reads_hidden_state=bool(
            vlm_loaded_real and
            image_p.encoder.encoder_kind == "hf_vlm"),
        code_adapter_loads_real_codemodel_and_reads_hidden_state=bool(
            code_loaded_real and
            code_p.encoder.encoder_kind == "hf_causal_lm"),
        composed_pipeline_runs_on_team_with_at_least_two_modalities=(
            int(report.n_modalities) >= 2),
        audit_chain_captures_all_modalities=bool(
            len(report.cross_modality_root.per_modality_payload_cids
                ) == 3),
        merkle_root_verifiable=bool(ok),
        replay_byte_identity_per_modality_at_precision_floor=bool(
            text_floor.floor_within_tolerance_fp32 and
            code_floor.floor_within_tolerance_fp32 and
            image_floor.floor_within_tolerance_fp32),
    )
    cid = rep.cid()
    rep = dataclasses.replace(rep, bench_cid=str(cid))
    # Persist.
    if out_dir is None:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = str(
            REPO_ROOT / f"results/w87/multi_modal/w87_mm_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = pathlib.Path(out_dir) / (
        "multi_modal_v1_bench_report.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(rep.to_dict(), fp,
                  sort_keys=True, separators=(",", ":"))
        fp.write("\n")
    # Sidecars.
    sidecar_root = pathlib.Path(out_dir) / (
        "cross_modality_merkle_root.json")
    with open(sidecar_root, "w", encoding="utf-8") as fp:
        json.dump(report.cross_modality_root.to_dict(), fp,
                  sort_keys=True, indent=2)
    sidecar_floor = pathlib.Path(out_dir) / (
        "per_modality_precision_floors.json")
    with open(sidecar_floor, "w", encoding="utf-8") as fp:
        json.dump(
            [f.to_dict()
             for f in report.per_modality_precision_floors],
            fp, sort_keys=True, indent=2)
    sidecar_extras = pathlib.Path(out_dir) / "payload_extras.json"
    with open(sidecar_extras, "w", encoding="utf-8") as fp:
        json.dump({
            "text": text_p.to_dict(),
            "code": code_p.to_dict(),
            "image": image_p.to_dict(),
        }, fp, sort_keys=True, indent=2)
    print(f"[w87/mm] wrote report → {out_path}")
    print(f"[w87/mm] bench_cid = {cid}")
    return rep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="W87 / P3 #46 — multi-modal substrate bench")
    parser.add_argument(
        "--vlm-model", type=str, default=DEFAULT_VLM,
        help=f"VLM model name (default {DEFAULT_VLM})")
    parser.add_argument(
        "--code-model", type=str, default=DEFAULT_CODE_LM,
        help=f"Code LM name (default {DEFAULT_CODE_LM})")
    parser.add_argument(
        "--require-real-vlm", action="store_true",
        help="fail if a real VLM can't be loaded")
    parser.add_argument(
        "--require-real-code", action="store_true",
        help="fail if a real code-LM can't be loaded")
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="output directory (auto-generated under "
             "results/w87/multi_modal/ if omitted)")
    parser.add_argument(
        "--embedding-dim", type=int, default=2048,
        help="Embedding dim per modality (default 2048; use "
             "4096 for LLaVA-1.5-7B / Idefics-2-8B; large "
             "values are truncated to the model's hidden dim)")
    args = parser.parse_args(argv)
    rep = run_multi_modal_bench_v1(
        vlm_model_name=str(args.vlm_model),
        code_model_name=str(args.code_model),
        require_real_vlm=bool(args.require_real_vlm),
        require_real_code=bool(args.require_real_code),
        out_dir=args.out_dir,
        embedding_dim=int(args.embedding_dim))
    print()
    print("Load-bearing closure bools:")
    for k in (
            "multimodal_payload_for_three_modalities",
            "vision_adapter_loads_real_vlm_and_reads_hidden_state",
            "code_adapter_loads_real_codemodel_and_reads_hidden_state",
            "composed_pipeline_runs_on_team_with_at_least_two_modalities",
            "audit_chain_captures_all_modalities",
            "merkle_root_verifiable",
            "replay_byte_identity_per_modality_at_precision_floor"):
        print(f"  {k}: {getattr(rep, k)}")
    print()
    print(f"Wall time: {rep.bench_wall_seconds:.1f}s")
    print(f"VLM: {rep.vlm_model_name} (real={rep.vlm_loaded_real})")
    print(f"Code LM: {rep.code_model_name} "
          f"(real={rep.code_loaded_real})")
    print(f"Image hidden shape: {rep.image_hidden_shape}")
    print(f"Code AST functions: {rep.code_ast_n_functions}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
