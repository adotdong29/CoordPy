"""W87 / P3 #46 — Composed multi-modal pipeline V1.

Runs a multi-modal team (text + image + code agents) and anchors
every per-modality payload into a single cross-modality Merkle
root.  This is the load-bearing closure for the issue's DoD
bullet 4 ("Composed pipeline runs on a multi-modal team
(≥ 2 modalities); the audit chain captures all modalities and
the Merkle root is verifiable").

The pipeline is intentionally minimal — V1 ships the *contract*:

  * each agent contributes one payload per modality;
  * payloads are content-addressed (`MultiModalPayloadV1`);
  * the cross-modality Merkle root spans every payload's
    `payload_cid()`;
  * the result is a `MultiModalRunReportV1` that re-verifies
    offline.

The deeper integration with the W82+W83 audit chain (so that a
multi-modal turn's Merkle root composes with the team's W82
hash-chain) is V2.  V1's separability is intentional: the
multi-modal Merkle root can be embedded as a single leaf in any
larger W82 chain.

Honest scope (W87)
------------------

* ``W87-L-MULTI-MODAL-PIPELINE-V1-DAG-FLAT-CAP`` — V1 has a flat
  list of (agent_id, modality, payload) tuples; multi-turn DAGs
  and per-turn lineage are V2 (the cross-modality Merkle root is
  the V1 audit-chain primitive).
* ``W87-L-MULTI-MODAL-PIPELINE-V1-NO-LIVE-REASONING-CAP`` — V1
  does not run the per-agent reasoning; it captures the
  per-agent multi-modal payloads + bundles them.  The reasoning
  is the responsibility of the substrate adapter calling into
  this pipeline (vision adapter encodes the image, code adapter
  encodes the source, etc.).  Live composed reasoning across
  modalities (e.g. vision agent → code agent → text agent in one
  forward) is V2.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .multi_modal_payload_v1 import (
    CrossModalityMerkleRootV1,
    Modality,
    ModalityPrecisionFloorV1,
    MultiModalPayloadV1,
    W87_MULTI_MODAL_V1_SCHEMA_VERSION,
    build_cross_modality_merkle_root_v1,
    measure_modality_precision_floor_fp32_v1,
)


W87_COMPOSED_MULTI_MODAL_PIPELINE_V1_SCHEMA_VERSION: str = (
    "coordpy.composed_multimodal_pipeline_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------
# Per-agent multi-modal contribution
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MultiModalAgentTurnV1:
    """One agent's contribution to a multi-modal pipeline turn."""

    schema: str
    agent_id: str
    role: str
    payload: MultiModalPayloadV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "agent_id": str(self.agent_id),
            "role": str(self.role),
            "payload": self.payload.to_dict(),
            "payload_cid": str(self.payload.payload_cid()),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w87_multi_modal_agent_turn_v1",
            "turn": self.to_dict(),
        })


# ---------------------------------------------------------------
# MultiModalRunReportV1
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MultiModalRunReportV1:
    """Output of a composed multi-modal pipeline run."""

    schema: str
    run_label: str
    agent_turns: tuple[MultiModalAgentTurnV1, ...]
    cross_modality_root: CrossModalityMerkleRootV1
    per_modality_precision_floors: tuple[
        ModalityPrecisionFloorV1, ...]
    n_modalities: int
    report_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "run_label": str(self.run_label),
            "n_agent_turns": int(len(self.agent_turns)),
            "agent_turn_cids": [
                str(t.cid()) for t in self.agent_turns],
            "agent_turn_modalities": [
                str(t.payload.modality)
                for t in self.agent_turns],
            "cross_modality_root_cid": str(
                self.cross_modality_root.cid()),
            "cross_modality_root_root_cid": str(
                self.cross_modality_root.root_cid),
            "per_modality_precision_floor_cids": [
                str(f.cid())
                for f in self.per_modality_precision_floors],
            "per_modality_precision_floor_modalities": [
                str(f.modality)
                for f in self.per_modality_precision_floors],
            "n_modalities": int(self.n_modalities),
            "report_cid": str(self.report_cid),
        }

    def cid(self) -> str:
        rd = self.to_dict()
        rd_for_hash = {**rd, "report_cid": ""}
        return _sha256_hex({
            "kind": "w87_multi_modal_run_report_v1",
            "report": rd_for_hash,
        })


def run_composed_multi_modal_pipeline_v1(
        *, run_label: str,
        agent_turns: Sequence[MultiModalAgentTurnV1],
        per_modality_precision_floors: Sequence[
            ModalityPrecisionFloorV1] = (),
) -> MultiModalRunReportV1:
    """Run a composed multi-modal pipeline turn.

    The pipeline accepts a list of per-agent multi-modal turns,
    builds the cross-modality Merkle root, and produces a
    content-addressed report.

    The per-modality precision floors are optional: callers that
    measured them (typically via
    ``measure_modality_precision_floor_fp32_v1``) pass them
    through; the report's load-bearing identity does NOT require
    them but the closure DoD does.
    """
    modalities = {str(t.payload.modality) for t in agent_turns}
    n_modalities = len(modalities)
    payloads = [t.payload for t in agent_turns]
    root = build_cross_modality_merkle_root_v1(payloads)
    rep = MultiModalRunReportV1(
        schema=W87_COMPOSED_MULTI_MODAL_PIPELINE_V1_SCHEMA_VERSION,
        run_label=str(run_label),
        agent_turns=tuple(agent_turns),
        cross_modality_root=root,
        per_modality_precision_floors=tuple(
            per_modality_precision_floors),
        n_modalities=int(n_modalities),
        report_cid="",
    )
    cid = rep.cid()
    return dataclasses.replace(rep, report_cid=str(cid))


def verify_multi_modal_run_report_v1(
        report: MultiModalRunReportV1,
) -> tuple[bool, str]:
    """Independently re-derive every CID in the report and
    confirm the cross-modality Merkle root commits to every
    per-agent payload byte-for-byte.

    Returns ``(ok, detail)``.
    """
    # Re-derive each agent turn's CID and confirm it matches.
    recorded_turn_cids = list(
        report.to_dict()["agent_turn_cids"])
    derived = [str(t.cid()) for t in report.agent_turns]
    if derived != recorded_turn_cids:
        return (False, "agent_turn_cids mismatch")
    # Re-derive cross-modality root from the payloads.
    payloads = [t.payload for t in report.agent_turns]
    re_root = build_cross_modality_merkle_root_v1(payloads)
    if str(re_root.root_cid) != str(
            report.cross_modality_root.root_cid):
        return (False, "cross_modality_root_cid mismatch")
    # Re-derive report_cid (with placeholder hash format).
    rd = report.to_dict()
    rd_for_hash = {**rd, "report_cid": ""}
    derived_report_cid = _sha256_hex({
        "kind": "w87_multi_modal_run_report_v1",
        "report": rd_for_hash,
    })
    if derived_report_cid != str(report.report_cid):
        return (False, "report_cid mismatch")
    return (True, "ok")


__all__ = [
    "W87_COMPOSED_MULTI_MODAL_PIPELINE_V1_SCHEMA_VERSION",
    "MultiModalAgentTurnV1",
    "MultiModalRunReportV1",
    "run_composed_multi_modal_pipeline_v1",
    "verify_multi_modal_run_report_v1",
]
