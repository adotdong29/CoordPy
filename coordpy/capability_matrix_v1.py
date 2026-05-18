"""W80 / P0 #17 — Living Hosted-vs-Local Capability Matrix V1.

CoordPy's hosted-real-substrate boundary has been a single
snapshot-style wall (``hosted_real_substrate_boundary_v12``).
That makes the *hosted side* of the picture explicit but leaves
the local side scattered across substrate-adapter probes,
controlled-runtime axes, and façade side-channel notes. The P0
#17 ask is for a single, *living*, machine-readable capability
matrix that spans:

* third-party hosted APIs (text-only / logprobs / prefix-cache)
* the local OpenAI-compatible façade (substrate side-channel)
* the in-repo NumPy controlled runtime
* the HF transformers controlled runtime (P0 #5)
* future second backends (llama.cpp / vLLM / MLX)

V1 is **living**: every refresh re-probes the local plane and
re-reads the hosted boundary, so the matrix never becomes stale
prose. The matrix is **machine-readable**: a single
``CapabilityMatrixV1`` dataclass carries every cell, with CIDs
and a stable schema. The matrix is **honest**: it preserves
the W79 wall semantics (hosted plane explicitly blocks the
substrate axes) and surfaces backend asymmetry (HF runtime
marks attention bias steering as ``BACKEND_SPECIFIC``).

V1 also unifies axis naming. The W80 instrumentation contract
gives us a canonical axis set; W79 had a separate set of
hosted-blocked axes (e.g. ``controlled_runtime_hidden_state``).
V1 cross-links them so callers can ask either question:

* what does this backend expose?
* what does the hosted plane block?

Honest scope (W80)
------------------

* ``W80-L-CAPABILITY-MATRIX-V1-LIVING-CAP`` — V1 *re-probes*
  on every build_capability_matrix_v1 call; it is not a static
  snapshot.
* ``W80-L-CAPABILITY-MATRIX-V1-HOSTED-WALL-PRESERVED-CAP`` —
  the W79 frontier-blocked-axes set is forwarded unchanged.
* ``W80-L-CAPABILITY-MATRIX-V1-NEW-BACKENDS-PLUGGABLE-CAP`` —
  adding a new backend slots into the matrix without schema
  change; the matrix CID changes deterministically.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any

from .hosted_real_substrate_boundary_v12 import (
    HostedRealSubstrateBoundaryV12,
    build_default_hosted_real_substrate_boundary_v12,
)
from .runtime_instrumentation_v1 import (
    CapabilityTag,
    InstrumentationAxis,
    W80_INSTRUMENTATION_AXES_ALL,
)
from .runtime_parity_matrix_v1 import (
    RuntimeParityMatrixV1,
    build_runtime_parity_matrix_v1,
)


W80_CAPABILITY_MATRIX_V1_SCHEMA_VERSION: str = (
    "coordpy.capability_matrix_v1.v1")


# ---------------------------------------------------------------
# Cross-link: W80 instrumentation axes <-> W79 boundary axes
# ---------------------------------------------------------------

W80_AXIS_TO_W79_HOSTED_BLOCKED_AXIS: dict[str, str] = {
    InstrumentationAxis.READ_HIDDEN_STATE.value:
        "controlled_runtime_hidden_state",
    InstrumentationAxis.READ_KV_CACHE.value:
        "controlled_runtime_kv_cache",
    InstrumentationAxis.READ_ATTENTION_PROBS.value:
        "controlled_runtime_attention_probs",
    InstrumentationAxis.WRITE_ATTENTION_BIAS.value:
        "controlled_runtime_per_head_attention_bias",
    InstrumentationAxis.INJECT_PREFIX_STATE.value:
        "controlled_runtime_prefix_state_inject",
    InstrumentationAxis.READ_PER_LAYER_LOGITS.value:
        "controlled_runtime_per_layer_logits",
    InstrumentationAxis.REPLAY_FROM_KV.value:
        "controlled_runtime_replay_from_kv",
}


# ---------------------------------------------------------------
# Surface descriptors
# ---------------------------------------------------------------

W80_SURFACE_HOSTED_THIRD_PARTY: str = "hosted_third_party"
W80_SURFACE_LOCAL_OPENAI_FACADE: str = "local_openai_facade"
W80_SURFACE_CONTROLLED_RUNTIME_NUMPY: str = (
    "controlled_runtime_numpy")
W80_SURFACE_CONTROLLED_RUNTIME_TRANSFORMERS: str = (
    "controlled_runtime_transformers")


W80_CAPABILITY_MATRIX_V1_AXES: tuple[str, ...] = (
    *W80_INSTRUMENTATION_AXES_ALL,
    # Transport / deployment axes that complete the picture.
    "transport_text_completion",
    "transport_logprobs_optional",
    "transport_prefix_cache_optional",
    "side_channel_available",
    "deployment_mode",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(
        _canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SurfaceCapabilityV1:
    """One surface's per-axis declared capability tags."""

    schema: str
    surface_id: str
    surface_label: str
    transport_mode: str
    deployment_mode: str
    per_axis: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]
    refreshed_at_ns: int

    def axis_tag(self, axis: str) -> str:
        for a, t in self.per_axis:
            if a == axis:
                return t
        return CapabilityTag.UNAVAILABLE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "surface_id": str(self.surface_id),
            "surface_label": str(self.surface_label),
            "transport_mode": str(self.transport_mode),
            "deployment_mode": str(self.deployment_mode),
            "per_axis": [
                [str(a), str(t)] for a, t in self.per_axis],
            "notes": [str(n) for n in self.notes],
            "refreshed_at_ns": int(self.refreshed_at_ns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_surface_capability_v1",
            "surface": self.to_dict()})


def _hosted_third_party_surface(
        *, boundary: HostedRealSubstrateBoundaryV12,
) -> SurfaceCapabilityV1:
    """Build the hosted-third-party surface capability descriptor."""

    u = CapabilityTag.UNAVAILABLE.value
    a = CapabilityTag.AVAILABLE.value
    per_axis = []
    blocked = set(boundary.blocked_axes)
    for ax in W80_INSTRUMENTATION_AXES_ALL:
        w79_ax = W80_AXIS_TO_W79_HOSTED_BLOCKED_AXIS.get(ax, "")
        if w79_ax and w79_ax in blocked:
            per_axis.append((ax, u))
        elif ax == (
                InstrumentationAxis.READ_FINAL_LOGITS.value):
            # Logits are optional via the logprobs tier.
            per_axis.append((
                ax, CapabilityTag.BEST_EFFORT.value))
        elif ax == (
                InstrumentationAxis.DETERMINISTIC_REPLAY.value):
            per_axis.append((
                ax, CapabilityTag.BEST_EFFORT.value))
        elif ax == (
                InstrumentationAxis.CONTENT_ADDRESSED_TRACE.value):
            per_axis.append((
                ax, CapabilityTag.BEST_EFFORT.value))
        else:
            per_axis.append((ax, u))
    # Transport / deployment axes.
    per_axis.append(("transport_text_completion", a))
    per_axis.append((
        "transport_logprobs_optional",
        CapabilityTag.BEST_EFFORT.value))
    per_axis.append((
        "transport_prefix_cache_optional",
        CapabilityTag.BEST_EFFORT.value))
    per_axis.append(("side_channel_available", u))
    per_axis.append(("deployment_mode", a))
    return SurfaceCapabilityV1(
        schema=W80_CAPABILITY_MATRIX_V1_SCHEMA_VERSION,
        surface_id=W80_SURFACE_HOSTED_THIRD_PARTY,
        surface_label="hosted third-party LLM HTTP API",
        transport_mode="https",
        deployment_mode="third_party_cloud",
        per_axis=tuple(per_axis),
        notes=(
            "blocked on every controlled-runtime substrate axis",
            "carries text, optional logprobs, optional prefix-"
            "cache accounting only",
            f"hosted-blocked-axis count at probe: "
            f"{len(blocked)}",
        ),
        refreshed_at_ns=int(time.time_ns()),
    )


def _local_openai_facade_surface() -> SurfaceCapabilityV1:
    a = CapabilityTag.AVAILABLE.value
    per_axis = []
    for ax in W80_INSTRUMENTATION_AXES_ALL:
        # The W79 façade reroutes to the in-repo controlled
        # runtime, so it exposes every substrate axis in-process.
        per_axis.append((ax, a))
    per_axis.append(("transport_text_completion", a))
    per_axis.append(("transport_logprobs_optional", a))
    per_axis.append((
        "transport_prefix_cache_optional",
        CapabilityTag.BACKEND_SPECIFIC.value))
    per_axis.append(("side_channel_available", a))
    per_axis.append(("deployment_mode", a))
    return SurfaceCapabilityV1(
        schema=W80_CAPABILITY_MATRIX_V1_SCHEMA_VERSION,
        surface_id=W80_SURFACE_LOCAL_OPENAI_FACADE,
        surface_label=(
            "local OpenAI-compatible façade over controlled "
            "runtime"),
        transport_mode="in_process",
        deployment_mode="local_facade",
        per_axis=tuple(per_axis),
        notes=(
            "OpenAI-shaped HTTP surface; in-process façade "
            "reroutes calls to the in-repo controlled runtime",
            "exposes a substrate side-channel that hosted "
            "APIs cannot carry",
        ),
        refreshed_at_ns=int(time.time_ns()),
    )


def _surface_from_parity_descriptor(
        *, parity: RuntimeParityMatrixV1, backend_id: str,
        surface_id: str, surface_label: str,
        transport_mode: str, deployment_mode: str,
        notes: tuple[str, ...],
) -> SurfaceCapabilityV1:
    declared = {}
    available = False
    for b in parity.backends:
        if b.backend_id == backend_id:
            declared = {a: t for a, t in b.declared_axes}
            available = bool(b.available)
            break
    per_axis = []
    for ax in W80_INSTRUMENTATION_AXES_ALL:
        per_axis.append(
            (ax, declared.get(
                ax, CapabilityTag.UNAVAILABLE.value)))
    a = CapabilityTag.AVAILABLE.value
    u = CapabilityTag.UNAVAILABLE.value
    per_axis.append(("transport_text_completion", a))
    per_axis.append((
        "transport_logprobs_optional", a))
    per_axis.append((
        "transport_prefix_cache_optional",
        CapabilityTag.BACKEND_SPECIFIC.value))
    per_axis.append(("side_channel_available", a))
    per_axis.append(("deployment_mode", a if available else u))
    return SurfaceCapabilityV1(
        schema=W80_CAPABILITY_MATRIX_V1_SCHEMA_VERSION,
        surface_id=surface_id,
        surface_label=surface_label,
        transport_mode=transport_mode,
        deployment_mode=deployment_mode,
        per_axis=tuple(per_axis),
        notes=notes,
        refreshed_at_ns=int(time.time_ns()),
    )


@dataclasses.dataclass(frozen=True)
class CapabilityMatrixV1:
    """The living hosted-vs-local capability matrix.

    Carries:
    * a refreshed-at timestamp (V1 is *living*, not a snapshot);
    * the per-surface descriptors;
    * a cross-link from each W80 instrumentation axis to its
      W79 hosted-blocked counterpart;
    * the hosted boundary V12 CID for tamper-evidence.
    """

    schema: str
    refreshed_at_ns: int
    surfaces: tuple[SurfaceCapabilityV1, ...]
    hosted_boundary_v12_cid: str
    parity_matrix_cid: str
    axes_order: tuple[str, ...]
    axis_cross_link: tuple[tuple[str, str], ...]

    def by_surface(self) -> dict[str, SurfaceCapabilityV1]:
        return {s.surface_id: s for s in self.surfaces}

    def to_dict(self) -> dict[str, Any]:
        cells: dict[str, dict[str, str]] = {}
        for s in self.surfaces:
            cells[s.surface_id] = {
                ax: s.axis_tag(ax)
                for ax in self.axes_order
            }
        return {
            "schema": str(self.schema),
            "refreshed_at_ns": int(self.refreshed_at_ns),
            "surfaces": [s.to_dict() for s in self.surfaces],
            "hosted_boundary_v12_cid": str(
                self.hosted_boundary_v12_cid),
            "parity_matrix_cid": str(self.parity_matrix_cid),
            "axes_order": list(self.axes_order),
            "axis_cross_link": [
                [str(a), str(b)]
                for a, b in self.axis_cross_link],
            "cells": cells,
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_capability_matrix_v1",
            "matrix": self.to_dict()})

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(), indent=int(indent),
            sort_keys=True, default=str)

    def render_markdown(self) -> str:
        """Render the capability matrix as a Markdown table.

        Surfaces are columns; axes are rows. Each cell is the
        declared capability tag.
        """
        header = "| Axis | " + " | ".join(
            s.surface_id for s in self.surfaces) + " |"
        sep = "|------|" + "|".join(
            "------" for _ in self.surfaces) + "|"
        rows: list[str] = []
        for ax in self.axes_order:
            cells = [
                s.axis_tag(ax) for s in self.surfaces]
            rows.append(
                "| " + ax + " | " + " | ".join(cells)
                + " |")
        return "\n".join([header, sep, *rows])

    def axes_universal_on_local(
            self,
    ) -> tuple[str, ...]:
        """Axes available on every *local* surface (façade +
        in-repo numpy + transformers)."""
        local_ids = (
            W80_SURFACE_LOCAL_OPENAI_FACADE,
            W80_SURFACE_CONTROLLED_RUNTIME_NUMPY,
            W80_SURFACE_CONTROLLED_RUNTIME_TRANSFORMERS,
        )
        # The matrix always carries the full surface set, even when an
        # optional runtime is unavailable on this machine. "Universal on
        # local" should quantify over the local runtimes that are
        # actually reachable in this build, not let an unavailable
        # optional backend collapse the whole set to zero.
        active_local_ids = {
            s.surface_id
            for s in self.surfaces
            if s.surface_id in local_ids
            and s.axis_tag("deployment_mode")
            != CapabilityTag.UNAVAILABLE.value
        }
        out: list[str] = []
        for ax in W80_INSTRUMENTATION_AXES_ALL:
            ok = True
            for sid in local_ids:
                if sid not in active_local_ids:
                    continue
                surface = next(
                    (s for s in self.surfaces
                     if s.surface_id == sid), None)
                if surface is None:
                    continue
                tag = surface.axis_tag(ax)
                if tag in (
                        CapabilityTag.UNAVAILABLE.value,
                        "skip"):
                    ok = False
                    break
            if ok:
                out.append(ax)
        return tuple(out)

    def axes_blocked_on_hosted(self) -> tuple[str, ...]:
        """Axes blocked on the hosted third-party surface."""
        hosted = next(
            (s for s in self.surfaces
             if s.surface_id == W80_SURFACE_HOSTED_THIRD_PARTY),
            None)
        if hosted is None:
            return ()
        return tuple(
            ax for ax in W80_INSTRUMENTATION_AXES_ALL
            if hosted.axis_tag(ax)
            == CapabilityTag.UNAVAILABLE.value)

    def asymmetry_report(self) -> dict[str, Any]:
        """Concise hosted-vs-local asymmetry summary."""
        hosted_blocked = self.axes_blocked_on_hosted()
        local_universal = self.axes_universal_on_local()
        # Axes that are hosted-blocked AND local-universal: the
        # load-bearing W80 asymmetry surface.
        asym = tuple(
            ax for ax in hosted_blocked
            if ax in local_universal)
        return {
            "schema": str(self.schema),
            "matrix_cid": str(self.cid()),
            "n_hosted_blocked_axes": int(len(hosted_blocked)),
            "n_local_universal_axes": int(
                len(local_universal)),
            "n_asymmetry_axes": int(len(asym)),
            "hosted_blocked_axes": list(hosted_blocked),
            "local_universal_axes": list(local_universal),
            "asymmetry_axes": list(asym),
        }


def build_capability_matrix_v1(
        *, include_transformers: bool = True,
        transformers_model_name: str | None = None,
        prompt: str = "capability matrix probe",
) -> CapabilityMatrixV1:
    """Build the *living* capability matrix.

    Every call re-probes:
    * the hosted boundary V12,
    * the local OpenAI-compatible façade descriptor,
    * the in-repo NumPy controlled runtime,
    * the HF transformers runtime (if installed),
    and emits a fresh matrix with refreshed timestamps and CIDs.
    """

    boundary = (
        build_default_hosted_real_substrate_boundary_v12())
    parity = build_runtime_parity_matrix_v1(
        include_transformers=include_transformers,
        transformers_model_name=transformers_model_name,
        prompt=prompt)
    surfaces: list[SurfaceCapabilityV1] = []
    surfaces.append(_hosted_third_party_surface(
        boundary=boundary))
    surfaces.append(_local_openai_facade_surface())
    surfaces.append(_surface_from_parity_descriptor(
        parity=parity,
        backend_id="coordpy.controlled_runtime_substrate_v1",
        surface_id=W80_SURFACE_CONTROLLED_RUNTIME_NUMPY,
        surface_label=(
            "in-repo NumPy controlled runtime "
            "(controlled_runtime_substrate_v1)"),
        transport_mode="in_process",
        deployment_mode="in_repo",
        notes=(
            "small NumPy transformer, frozen Xavier-init "
            "weights, byte-identical replay",
            "exposes every W80 substrate axis as AVAILABLE",
        ),
    ))
    surfaces.append(_surface_from_parity_descriptor(
        parity=parity,
        backend_id="coordpy.transformers_runtime_v1",
        surface_id=(
            W80_SURFACE_CONTROLLED_RUNTIME_TRANSFORMERS),
        surface_label=(
            "HF transformers controlled runtime "
            "(transformers_runtime_v1)"),
        transport_mode="in_process",
        deployment_mode="local_pretrained",
        notes=(
            "pretrained transformers backend, lazy-loaded",
            "attention-bias steer is BACKEND_SPECIFIC via "
            "attention_mask augmentation",
            "byte-identical replay-from-KV under fp32 CPU",
        ),
    ))
    return CapabilityMatrixV1(
        schema=W80_CAPABILITY_MATRIX_V1_SCHEMA_VERSION,
        refreshed_at_ns=int(time.time_ns()),
        surfaces=tuple(surfaces),
        hosted_boundary_v12_cid=str(boundary.cid()),
        parity_matrix_cid=str(parity.cid()),
        axes_order=tuple(W80_CAPABILITY_MATRIX_V1_AXES),
        axis_cross_link=tuple(
            (str(k), str(v))
            for k, v in (
                W80_AXIS_TO_W79_HOSTED_BLOCKED_AXIS.items())),
    )


@dataclasses.dataclass(frozen=True)
class CapabilityMatrixV1WitnessV1:
    """Witness emitted from the capability matrix V1."""

    schema: str
    matrix_cid: str
    refreshed_at_ns: int
    n_surfaces: int
    n_axes: int
    n_hosted_blocked_axes: int
    n_local_universal_axes: int
    n_asymmetry_axes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "matrix_cid": str(self.matrix_cid),
            "refreshed_at_ns": int(self.refreshed_at_ns),
            "n_surfaces": int(self.n_surfaces),
            "n_axes": int(self.n_axes),
            "n_hosted_blocked_axes": int(
                self.n_hosted_blocked_axes),
            "n_local_universal_axes": int(
                self.n_local_universal_axes),
            "n_asymmetry_axes": int(self.n_asymmetry_axes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_capability_matrix_v1_witness",
            "witness": self.to_dict()})


def emit_capability_matrix_v1_witness(
        matrix: CapabilityMatrixV1,
) -> CapabilityMatrixV1WitnessV1:
    rep = matrix.asymmetry_report()
    return CapabilityMatrixV1WitnessV1(
        schema=W80_CAPABILITY_MATRIX_V1_SCHEMA_VERSION,
        matrix_cid=str(matrix.cid()),
        refreshed_at_ns=int(matrix.refreshed_at_ns),
        n_surfaces=int(len(matrix.surfaces)),
        n_axes=int(len(matrix.axes_order)),
        n_hosted_blocked_axes=int(
            rep["n_hosted_blocked_axes"]),
        n_local_universal_axes=int(
            rep["n_local_universal_axes"]),
        n_asymmetry_axes=int(rep["n_asymmetry_axes"]),
    )


__all__ = [
    "W80_CAPABILITY_MATRIX_V1_SCHEMA_VERSION",
    "W80_AXIS_TO_W79_HOSTED_BLOCKED_AXIS",
    "W80_SURFACE_HOSTED_THIRD_PARTY",
    "W80_SURFACE_LOCAL_OPENAI_FACADE",
    "W80_SURFACE_CONTROLLED_RUNTIME_NUMPY",
    "W80_SURFACE_CONTROLLED_RUNTIME_TRANSFORMERS",
    "W80_CAPABILITY_MATRIX_V1_AXES",
    "SurfaceCapabilityV1",
    "CapabilityMatrixV1",
    "CapabilityMatrixV1WitnessV1",
    "build_capability_matrix_v1",
    "emit_capability_matrix_v1_witness",
]
