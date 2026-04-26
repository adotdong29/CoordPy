"""Wevra run spec + high-level ``run()`` entry point.

Single stable entrypoint for programmatic Wevra use:

    >>> from vision_mvp.wevra import RunSpec, run
    >>> report = run(RunSpec(profile="local_smoke", out_dir="/tmp/wevra"))
    >>> report["readiness"]["ready"]
    True
    >>> report["provenance"]["schema"]
    'wevra.provenance.v1'

Slice 2 additions:

  * ``acknowledge_heavy``: first-class cost gate for real-LLM runs.
    When False (default) and the profile's sweep is ``mode="real"``,
    Wevra stages the launch command instead of starting the heavy
    run. When True, the sweep runs in-process against the configured
    endpoint.

  * ``report_sinks``: list of names of registered
    ``wevra.extensions.ReportSink`` instances to emit to after the
    standard artifact set lands. Each sink receives the same
    ``product_report.json`` dict.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .config import WevraConfig


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """Declarative description of one Wevra run.

    Fields
    ------
    profile           : name of a registered profile.
    out_dir           : output directory; created if missing.
    jsonl_override    : optional override for the profile's JSONL path.
    skip_sweep        : run readiness only.
    force_sweep       : run the sweep even if readiness fails.
    acknowledge_heavy : explicit cost gate for real-LLM sweeps.
                        Required for in-process execution of real
                        profiles (see ``wevra.runtime``).
    report_sinks      : names of registered ReportSink extensions to
                        emit to after the standard artifacts land.
    config            : optional ``WevraConfig``.
    """

    profile: str
    out_dir: str
    jsonl_override: str | None = None
    skip_sweep: bool = False
    force_sweep: bool = False
    acknowledge_heavy: bool = False
    allow_unsafe_sandbox: bool = False
    report_sinks: tuple[str, ...] = ()
    config: WevraConfig | None = None
    # SDK v3.1 — capsule-native runtime is the default. Setting this
    # to False reverts to the legacy post-hoc fold path (capsules
    # describe the run, but do not gate it). The two paths produce
    # CID-equivalent ledgers for non-ARTIFACT kinds (Theorem W3-34).
    capsule_native: bool = True


def run(spec: RunSpec) -> dict[str, Any]:
    """Execute a ``RunSpec`` and return the product report dict.

    Side effects: writes ``product_report.json``,
    ``product_summary.txt``, ``readiness_verdict.json``,
    ``provenance.json`` (and, for executed sweeps,
    ``sweep_result.json``) into ``spec.out_dir``.

    If ``spec.report_sinks`` is non-empty, each named sink is resolved
    via ``wevra.extensions.get_report_sink`` and called with the final
    report dict. Sink results are aggregated under
    ``report["sink_emissions"]``.
    """
    from vision_mvp.product.runner import run_profile

    report = run_profile(
        spec.profile,
        out_dir=spec.out_dir,
        jsonl_override=spec.jsonl_override,
        force_sweep=spec.force_sweep,
        skip_sweep=spec.skip_sweep,
        acknowledge_heavy=spec.acknowledge_heavy,
        allow_unsafe_sandbox=spec.allow_unsafe_sandbox,
        capsule_native=spec.capsule_native,
    )

    if spec.report_sinks:
        from .extensions import get_report_sink
        emissions: list[dict[str, Any]] = []
        for sink_name in spec.report_sinks:
            sink = get_report_sink(sink_name)
            try:
                result = sink.emit(report)
                emissions.append({
                    "sink": sink_name, "ok": True, "result": result})
            except Exception as ex:
                emissions.append({
                    "sink": sink_name, "ok": False,
                    "error": f"{type(ex).__name__}: {ex}"})
        report["sink_emissions"] = emissions

    return report


__all__ = ["RunSpec", "run"]
