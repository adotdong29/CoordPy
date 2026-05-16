"""W69 H4 — Hosted Provider Filter V2 (Plane A).

Strictly extends W68's ``coordpy.hosted_provider_filter``. V2 adds:

* **Compositional filters** — V1 was a single declarative spec. V2
  supports ANY/ALL chaining of multiple filter specs.
* **Per-tier weighting** — V2 attaches a weight to each allowed
  tier so that downstream callers (cost planner V2, router V2)
  can prefer specific tiers.
* **Filter audit chain** — V2 records the filter chain CID at each
  step.

Honest scope (W69): same as V1; data policies are declared.
``W69-L-HOSTED-V2-DATA-POLICY-DECLARED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hosted_provider_filter import (
    HostedProviderFilterSpec, filter_hosted_registry,
)
from .hosted_router_controller import HostedProviderRegistry
from .tiny_substrate_v3 import _sha256_hex


W69_HOSTED_PROVIDER_FILTER_V2_SCHEMA_VERSION: str = (
    "coordpy.hosted_provider_filter_v2.v1")


@dataclasses.dataclass(frozen=True)
class HostedProviderFilterSpecV2:
    inner_specs: tuple[HostedProviderFilterSpec, ...]
    combine: str   # "all" or "any"
    tier_weights: dict[str, float] = dataclasses.field(
        default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema":
                W69_HOSTED_PROVIDER_FILTER_V2_SCHEMA_VERSION,
            "inner_specs": [
                s.to_dict() for s in self.inner_specs],
            "combine": str(self.combine),
            "tier_weights": {
                k: float(round(v, 12))
                for k, v in sorted(self.tier_weights.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_provider_filter_spec_v2",
            "spec": self.to_dict()})


def filter_hosted_registry_v2(
        registry: HostedProviderRegistry,
        spec_v2: HostedProviderFilterSpecV2,
) -> tuple[HostedProviderRegistry, dict[str, Any]]:
    """Compositional filter. Returns (filtered_registry,
    report_v2)."""
    inner_reports: list[dict[str, Any]] = []
    if str(spec_v2.combine) == "all":
        cur = registry
        for s in spec_v2.inner_specs:
            cur, rep = filter_hosted_registry(cur, s)
            inner_reports.append(dict(rep))
        filtered = cur
    elif str(spec_v2.combine) == "any":
        kept = set()
        union_providers = []
        for s in spec_v2.inner_specs:
            sub, rep = filter_hosted_registry(registry, s)
            inner_reports.append(dict(rep))
            for p in sub.providers:
                if p.name not in kept:
                    kept.add(p.name)
                    union_providers.append(p)
        filtered = HostedProviderRegistry(
            providers=tuple(union_providers))
    else:
        raise ValueError(
            f"unknown combine={spec_v2.combine!r}")
    report_v2 = {
        "schema":
            W69_HOSTED_PROVIDER_FILTER_V2_SCHEMA_VERSION,
        "kind": "filter_v2_report",
        "spec_v2_cid": str(spec_v2.cid()),
        "n_inner_specs": int(len(spec_v2.inner_specs)),
        "combine": str(spec_v2.combine),
        "n_kept": int(len(filtered.providers)),
        "n_dropped": int(
            len(registry.providers) - len(filtered.providers)),
        "inner_reports": inner_reports,
    }
    return filtered, report_v2


__all__ = [
    "W69_HOSTED_PROVIDER_FILTER_V2_SCHEMA_VERSION",
    "HostedProviderFilterSpecV2",
    "filter_hosted_registry_v2",
]
