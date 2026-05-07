"""CoordPy run spec + high-level ``run()`` entry point.

Single stable entrypoint for programmatic CoordPy use:

    >>> from coordpy import RunSpec, run
    >>> report = run(RunSpec(profile="local_smoke", out_dir="/tmp/coordpy"))
    >>> report["readiness"]["ready"]
    True
    >>> report["provenance"]["schema"]
    'coordpy.provenance.v1'

Slice 2 additions:

  * ``acknowledge_heavy``: first-class cost gate for real-LLM runs.
    When False (default) and the profile's sweep is ``mode="real"``,
    CoordPy stages the launch command instead of starting the heavy
    run. When True, the sweep runs in-process against the configured
    endpoint.

  * ``report_sinks``: list of names of registered
    ``coordpy.extensions.ReportSink`` instances to emit to after the
    standard artifact set lands. Each sink receives the same
    ``product_report.json`` dict.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .config import CoordPyConfig


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """Declarative description of one CoordPy run.

    Fields
    ------
    profile             : name of a registered profile.
    out_dir             : output directory; created if missing.
                          Accepts ``str`` or ``os.PathLike``
                          (e.g. ``pathlib.Path``).
    jsonl_override      : optional override for the profile's
                          JSONL path.
    skip_sweep          : run readiness only.
    force_sweep         : run the sweep even if readiness fails.
    acknowledge_heavy   : explicit cost gate for real-LLM sweeps.
                          Required for in-process execution of
                          real profiles (see ``coordpy.runtime``).
    allow_unsafe_sandbox: skip the Docker requirement for profiles
                          marked ``trust=untrusted`` (e.g.
                          ``public_jsonl``). Only use for JSONL
                          you have audited yourself.
    report_sinks        : names of registered ReportSink
                          extensions to emit to after the
                          standard artifacts land.
    config              : optional ``CoordPyConfig``.
    capsule_native      : when True (default), capsules drive the
                          runtime; when False, capsules describe
                          the run post-hoc. The two paths produce
                          CID-equivalent ledgers for non-ARTIFACT
                          kinds (Theorem W3-34).
    deterministic       : when True, strip per-run wall-clock
                          fields from PROVENANCE / RUN_REPORT /
                          JSONL paths so two runs of the same
                          mock-mode profile produce identical
                          full-DAG CIDs (Theorem W3-41).
    """

    profile: str
    out_dir: str
    jsonl_override: str | None = None
    skip_sweep: bool = False
    force_sweep: bool = False
    acknowledge_heavy: bool = False
    allow_unsafe_sandbox: bool = False
    report_sinks: tuple[str, ...] = ()
    config: CoordPyConfig | None = None
    # SDK v3.1 — capsule-native runtime is the default. Setting this
    # to False reverts to the legacy post-hoc fold path (capsules
    # describe the run, but do not gate it). The two paths produce
    # CID-equivalent ledgers for non-ARTIFACT kinds (Theorem W3-34).
    capsule_native: bool = True
    # SDK v3.3 — deterministic-mode replay. When True, the runtime
    # strips per-run timestamps (PROVENANCE.timestamp_utc,
    # RUN_REPORT.wall_seconds, JSONL absolute paths) from the
    # capsules' payloads so two runs of the same deterministic
    # profile (mock mode, ``in_process``/``subprocess`` sandbox,
    # frozen JSONL) produce identical full-DAG CIDs (Theorem
    # W3-41). Default False preserves the legacy timestamp-bearing
    # CIDs for normal operations. Determinism is on the capsule
    # graph, not on wall clock.
    deterministic: bool = False

    def __post_init__(self) -> None:
        # Validate the load-bearing fields up front so a typo is
        # caught at construction, not deferred to profile lookup.
        # ``out_dir`` accepts strings and ``os.PathLike`` (e.g.
        # ``pathlib.Path``); the runner stringifies it before use.
        import os as _os
        if not isinstance(self.profile, str) or not self.profile:
            raise TypeError(
                f"RunSpec.profile must be a non-empty str (the name "
                f"of a registered profile); got "
                f"{type(self.profile).__name__}={self.profile!r}"
            )
        if not isinstance(self.out_dir, (str, _os.PathLike)) or (
            isinstance(self.out_dir, str) and not self.out_dir
        ):
            raise TypeError(
                f"RunSpec.out_dir must be a non-empty str or os.PathLike; "
                f"got {type(self.out_dir).__name__}={self.out_dir!r}"
            )


def run(spec: RunSpec) -> dict[str, Any]:
    """Execute a ``RunSpec`` and return the product report dict.

    Side effects: writes ``product_report.json``,
    ``product_summary.txt``, ``readiness_verdict.json``,
    ``provenance.json`` (and, for executed sweeps,
    ``sweep_result.json``) into ``spec.out_dir``.

    If ``spec.report_sinks`` is non-empty, each named sink is resolved
    via ``coordpy.extensions.get_report_sink`` and called with the final
    report dict. Sink results are aggregated under
    ``report["sink_emissions"]``.
    """
    from coordpy._internal.product.runner import run_profile
    from .config import CoordPyConfig

    if spec.config is None:
        resolved_config = CoordPyConfig.from_env()
    else:
        resolved_config = CoordPyConfig.from_env(**spec.config.as_dict())

    # Surface output-path problems as a clear ValueError before
    # any work happens, instead of bubbling a bare OSError /
    # FileNotFoundError out of one of the seven write sites.
    # Auto-creates the out_dir and any missing parents so a
    # relative path like ``./relative/path`` works the same as
    # an absolute one.
    import os as _os
    out_dir = spec.out_dir
    try:
        _os.makedirs(out_dir, exist_ok=True)
        _probe = _os.path.join(out_dir, ".coordpy_write_probe")
        with open(_probe, "w") as _fh:
            _fh.write("ok")
        _os.remove(_probe)
    except OSError as e:
        raise ValueError(
            f"RunSpec.out_dir is not writable: {out_dir!r} "
            f"({type(e).__name__}: {e}). Pick a writable path."
        ) from e

    report = run_profile(
        spec.profile,
        out_dir=spec.out_dir,
        jsonl_override=spec.jsonl_override or resolved_config.jsonl,
        force_sweep=spec.force_sweep,
        skip_sweep=spec.skip_sweep,
        acknowledge_heavy=spec.acknowledge_heavy,
        allow_unsafe_sandbox=spec.allow_unsafe_sandbox,
        config=resolved_config,
        capsule_native=spec.capsule_native,
        deterministic=spec.deterministic,
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
