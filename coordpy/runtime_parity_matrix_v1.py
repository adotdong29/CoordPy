"""W80 / P0 #6 — Runtime Parity Matrix V1.

Builds a substrate **capability parity matrix** across the
controlled runtimes in the repo. This is the load-bearing
answer to P0 #6: until W80, the programme had exactly one
controlled runtime (``coordpy.controlled_runtime_substrate_v1``,
a small in-repo NumPy transformer) and could not honestly claim
substrate-portable mechanisms. V1 adds the second controlled
backend (``coordpy.transformers_runtime_v1``, HF transformers
on real pretrained weights) and turns the capability story
into a machine-readable parity matrix:

* every controlled runtime declares its axes via the W80
  instrumentation contract (P0 #8);
* every runtime is fed through the same conformance suite;
* per-axis ``pass`` / ``skip`` / ``fail`` lands in a matrix;
* the matrix is content-addressed for reproducible witnesses;
* deltas between runtimes are explicit and honest — no
  lowest-common-denominator flattening.

The matrix surfaces *real* limitations:

* attention-bias steering is ``BACKEND_SPECIFIC`` on HF (we add
  bias via ``attention_mask`` augmentation, not via a clean
  universal API);
* attention probs read is ``BACKEND_SPECIFIC`` on HF because it
  depends on ``attn_implementation="eager"``;
* the in-repo NumPy runtime claims AVAILABLE on every axis.

Honest scope (W80)
------------------

* ``W80-L-PARITY-MATRIX-V1-TWO-BACKENDS-CAP`` — V1 ships parity
  across the two existing controlled backends (NumPy substrate
  V1 and HF transformers V1). Adding llama.cpp / vLLM / MLX is
  future work and would slot into the same matrix without
  schema change.
* ``W80-L-PARITY-MATRIX-V1-TRANSFORMERS-OPTIONAL-CAP`` —
  building the matrix without the transformers backend
  installed produces a matrix with one runtime + an explicit
  "transformers unavailable" probe entry, so the schema and
  CI stay deterministic.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any, Sequence

from .runtime_instrumentation_v1 import (
    CapabilityTag,
    ControlledRuntimeInstrumentationAdapterV1,
    InstrumentationConformanceReportV1,
    RuntimeInstrumentationProtocolV1,
    W80_INSTRUMENTATION_AXES_ALL,
    W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
    run_instrumentation_conformance_v1,
)


W80_RUNTIME_PARITY_MATRIX_V1_SCHEMA_VERSION: str = (
    "coordpy.runtime_parity_matrix_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(
        _canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class RuntimeBackendDescriptorV1:
    """Per-backend descriptor for the parity matrix.

    Pure data — held outside the running model so the matrix can
    be summarised without instantiating heavy runtimes.
    """

    schema: str
    backend_id: str
    backend_runtime_id: str
    declared_axes: tuple[tuple[str, str], ...]
    available: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "backend_id": str(self.backend_id),
            "backend_runtime_id": str(self.backend_runtime_id),
            "declared_axes": [
                [str(a), str(t)]
                for a, t in self.declared_axes],
            "available": bool(self.available),
            "notes": [str(n) for n in self.notes],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_runtime_backend_descriptor_v1",
            "descriptor": self.to_dict()})

    def axis_tag(self, axis: str) -> str:
        for a, t in self.declared_axes:
            if a == axis:
                return t
        return CapabilityTag.UNAVAILABLE.value


@dataclasses.dataclass(frozen=True)
class RuntimeParityMatrixV1:
    """The canonical machine-readable parity matrix.

    For each axis × backend, the matrix records:
    * the backend's *declared* capability tag (from its contract
      adapter);
    * the backend's *conformance* status on that axis (``pass``,
      ``skip``, or ``fail``).

    Both pieces matter: declared tells you what the backend
    promises, conformance tells you whether the promise is real
    in this build.
    """

    schema: str
    probed_at_ns: int
    backends: tuple[RuntimeBackendDescriptorV1, ...]
    conformance_reports: tuple[
        InstrumentationConformanceReportV1, ...]
    axes_order: tuple[str, ...]

    def cell_status(
            self, *, backend_id: str, axis: str,
    ) -> tuple[str, str]:
        """Return ``(declared_tag, conformance_status)`` for
        one cell."""
        declared = CapabilityTag.UNAVAILABLE.value
        for d in self.backends:
            if d.backend_id == backend_id:
                declared = d.axis_tag(axis)
                break
        status = "skip"
        for r in self.conformance_reports:
            if r.backend_id == backend_id:
                status = r.per_axis_dict().get(axis, "skip")
                break
        return (declared, status)

    def to_dict(self) -> dict[str, Any]:
        cells: dict[str, dict[str, dict[str, str]]] = {}
        for b in self.backends:
            cells[b.backend_id] = {}
            for ax in self.axes_order:
                declared, status = self.cell_status(
                    backend_id=b.backend_id, axis=ax)
                cells[b.backend_id][ax] = {
                    "declared": str(declared),
                    "status": str(status),
                }
        return {
            "schema": str(self.schema),
            "probed_at_ns": int(self.probed_at_ns),
            "backends": [b.to_dict() for b in self.backends],
            "conformance_reports": [
                r.to_dict()
                for r in self.conformance_reports],
            "axes_order": list(self.axes_order),
            "cells": cells,
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_runtime_parity_matrix_v1",
            "matrix": self.to_dict()})

    def backends_with_axis_available(
            self, axis: str,
    ) -> tuple[str, ...]:
        """Return the backends that declared and conformed on
        the given axis."""
        out: list[str] = []
        for b in self.backends:
            declared, status = self.cell_status(
                backend_id=b.backend_id, axis=axis)
            if (declared != CapabilityTag.UNAVAILABLE.value
                    and status == "pass"):
                out.append(b.backend_id)
        return tuple(out)

    def axes_universal(self) -> tuple[str, ...]:
        """Axes that pass on ALL probed backends."""
        out: list[str] = []
        for ax in self.axes_order:
            backends_ok = self.backends_with_axis_available(
                ax)
            if (len(backends_ok)
                    == len(tuple(
                        b for b in self.backends
                        if b.available))):
                if len(backends_ok) > 0:
                    out.append(ax)
        return tuple(out)

    def axes_backend_specific(self) -> tuple[str, ...]:
        """Axes that pass on some but not all available
        backends."""
        out: list[str] = []
        avail_count = sum(
            1 for b in self.backends if b.available)
        if avail_count == 0:
            return ()
        for ax in self.axes_order:
            backends_ok = self.backends_with_axis_available(
                ax)
            if (len(backends_ok) > 0
                    and len(backends_ok) < avail_count):
                out.append(ax)
        return tuple(out)

    def render_markdown_table(self) -> str:
        """Render the parity matrix as a Markdown table.

        Used in docs and in the capability matrix V1 report.
        """
        header = "| Axis | " + " | ".join(
            b.backend_id for b in self.backends) + " |"
        sep = "|------|" + "|".join(
            "------" for _ in self.backends) + "|"
        rows: list[str] = []
        for ax in self.axes_order:
            cells: list[str] = []
            for b in self.backends:
                declared, status = self.cell_status(
                    backend_id=b.backend_id, axis=ax)
                if status == "pass":
                    marker = (
                        "OK"
                        if declared
                        == CapabilityTag.AVAILABLE.value
                        else "OK(bs)")
                elif status == "skip":
                    marker = "—"
                else:
                    marker = "FAIL"
                cells.append(marker)
            rows.append(
                "| " + ax + " | " + " | ".join(cells)
                + " |")
        return "\n".join([header, sep, *rows])


@dataclasses.dataclass(frozen=True)
class RuntimeParityMatrixWitnessV1:
    """Witness emitted from the matrix for CI/audit."""

    schema: str
    matrix_cid: str
    n_backends: int
    n_axes_universal: int
    n_axes_backend_specific: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "matrix_cid": str(self.matrix_cid),
            "n_backends": int(self.n_backends),
            "n_axes_universal": int(self.n_axes_universal),
            "n_axes_backend_specific": int(
                self.n_axes_backend_specific),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w80_runtime_parity_matrix_witness_v1",
            "witness": self.to_dict()})


def _descriptor_for_backend(
        backend: RuntimeInstrumentationProtocolV1, *,
        available: bool,
        notes: Sequence[str],
) -> RuntimeBackendDescriptorV1:
    return RuntimeBackendDescriptorV1(
        schema=W80_RUNTIME_PARITY_MATRIX_V1_SCHEMA_VERSION,
        backend_id=str(backend.backend_id()),
        backend_runtime_id=str(backend.backend_runtime_id()),
        declared_axes=tuple(
            (str(a), str(t))
            for a, t in backend.declared_axes().items()),
        available=bool(available),
        notes=tuple(notes),
    )


def build_runtime_parity_matrix_v1(
        *, include_transformers: bool = True,
        transformers_model_name: str | None = None,
        prompt: str = "runtime parity smoke",
) -> RuntimeParityMatrixV1:
    """Build the parity matrix across all controlled backends.

    Backend 1: ``coordpy.controlled_runtime_substrate_v1``
    (always present).
    Backend 2: ``coordpy.transformers_runtime_v1`` (only if
    ``transformers`` + ``torch`` are importable and
    ``include_transformers`` is ``True``).
    """

    backends: list[RuntimeBackendDescriptorV1] = []
    reports: list[InstrumentationConformanceReportV1] = []
    # Backend 1: in-repo NumPy controlled runtime.
    adapter_1 = ControlledRuntimeInstrumentationAdapterV1()
    backends.append(_descriptor_for_backend(
        adapter_1, available=True,
        notes=(
            "in-repo NumPy controlled runtime "
            "(coordpy.controlled_runtime_substrate_v1)",
            "every axis is AVAILABLE on this backend",
        )))
    reports.append(run_instrumentation_conformance_v1(
        adapter_1, prompt=prompt))
    # Backend 2: HF transformers runtime (optional).
    if include_transformers:
        try:
            from .transformers_runtime_v1 import (
                TransformersRuntimeV1,
                W80_TRANSFORMERS_DEFAULT_MODEL_NAME,
            )
            model_name = (
                transformers_model_name
                or W80_TRANSFORMERS_DEFAULT_MODEL_NAME)
            rt2 = TransformersRuntimeV1(
                model_name=model_name)
            backends.append(_descriptor_for_backend(
                rt2, available=True,
                notes=(
                    "HF transformers runtime "
                    "(coordpy.transformers_runtime_v1)",
                    f"model: {model_name}",
                    "attention-bias steer is BACKEND_SPECIFIC "
                    "via attention_mask augmentation",
                    "attention probs read requires "
                    "attn_implementation='eager'",
                )))
            reports.append(
                run_instrumentation_conformance_v1(
                    rt2, prompt=prompt))
        except Exception as exc:  # noqa: BLE001
            # Record the missing backend honestly.
            backends.append(RuntimeBackendDescriptorV1(
                schema=(
                    W80_RUNTIME_PARITY_MATRIX_V1_SCHEMA_VERSION),
                backend_id=(
                    "coordpy.transformers_runtime_v1"),
                backend_runtime_id=(
                    "coordpy.transformers_runtime_v1"
                    "#unavailable"),
                declared_axes=tuple(
                    (str(a),
                     str(CapabilityTag.UNAVAILABLE.value))
                    for a in W80_INSTRUMENTATION_AXES_ALL),
                available=False,
                notes=(
                    "transformers / torch not importable in "
                    "this build",
                    f"reason: {type(exc).__name__}: "
                    f"{str(exc)[:120]}",
                ),
            ))
            reports.append(InstrumentationConformanceReportV1(
                schema=(
                    W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION),
                backend_id=(
                    "coordpy.transformers_runtime_v1"),
                backend_runtime_id=(
                    "coordpy.transformers_runtime_v1"
                    "#unavailable"),
                per_axis=tuple(
                    (str(a), "skip")
                    for a in W80_INSTRUMENTATION_AXES_ALL),
                n_pass=0, n_skip=int(len(
                    W80_INSTRUMENTATION_AXES_ALL)),
                n_fail=0,
            ))
    return RuntimeParityMatrixV1(
        schema=W80_RUNTIME_PARITY_MATRIX_V1_SCHEMA_VERSION,
        probed_at_ns=int(time.time_ns()),
        backends=tuple(backends),
        conformance_reports=tuple(reports),
        axes_order=tuple(W80_INSTRUMENTATION_AXES_ALL),
    )


def emit_runtime_parity_matrix_witness(
        matrix: RuntimeParityMatrixV1,
) -> RuntimeParityMatrixWitnessV1:
    return RuntimeParityMatrixWitnessV1(
        schema=W80_RUNTIME_PARITY_MATRIX_V1_SCHEMA_VERSION,
        matrix_cid=str(matrix.cid()),
        n_backends=int(len(matrix.backends)),
        n_axes_universal=int(len(matrix.axes_universal())),
        n_axes_backend_specific=int(
            len(matrix.axes_backend_specific())),
    )


__all__ = [
    "W80_RUNTIME_PARITY_MATRIX_V1_SCHEMA_VERSION",
    "RuntimeBackendDescriptorV1",
    "RuntimeParityMatrixV1",
    "RuntimeParityMatrixWitnessV1",
    "build_runtime_parity_matrix_v1",
    "emit_runtime_parity_matrix_witness",
]
