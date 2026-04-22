"""Distributed document-analysis task — GENUINE collaboration.

Each agent sees ONE small chunk of a long report. No agent alone can see the
whole thing. The team must answer questions that require cross-chunk
synthesis.

The document here is a **fictional** internal review of a made-up company
("Orion Systems"). It is invented specifically so the LLM cannot retrieve
the answer from pre-training. The ground truth — what the top systemic
risks are — is set by how facts are distributed across chunks.

Key property: **the top risks only emerge from cross-chunk patterns**.
- "Supplier concentration": three separate incidents in three separate
  chunks all trace to the same vendor (NordAxis).
- "Incident-response delay": mean detection-to-mitigation time is
  buried in different chunks and only shows a pattern when summed.
- "Documentation gaps": mentioned in passing in multiple chunks, only
  rises to 'systemic' when counted together.

A single agent who reads only 1 chunk cannot see these patterns. The team
must pool. This is what 'collaboration' actually means.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# -- Fully invented corpus. Names/products/dates not drawn from anywhere real.

ORION_DOCUMENT = """
ORION SYSTEMS — INTERNAL INCIDENT REVIEW — Q3 2025 (CONFIDENTIAL)

Prepared for: Executive Risk Committee
Prepared by: Reliability Office

Section 1. Overview

Orion Systems operates three product lines: PulseCore (industrial sensors),
HelixQ (enterprise data platform), and NovaTrack (logistics software). In
Q3 2025 the Reliability Office recorded eight customer-impacting incidents
across these lines. This document summarizes each incident, its root cause,
and the mitigation actions taken, so the Executive Risk Committee can
decide on systemic follow-ups.

Section 2. Incident OS-0301 — PulseCore P7 sensor firmware corruption

On 2025-07-04 the P7 sensor series at three customer sites in Frankfurt
began returning out-of-range temperature readings. Initial triage blamed a
corrupted firmware image shipped with the 2.4.1 OTA update. The OTA server
had pulled firmware from vendor NordAxis, the contracted signing-key
manager for Orion's OTA channel. Investigation revealed that the firmware
image signature verification had a bypass flaw. Detection-to-mitigation
time: 48 hours. Documentation of the OTA chain was found to be four
months stale and did not describe NordAxis's signing-key rotation.

Section 3. Incident OS-0302 — HelixQ Europe-West latency spike

On 2025-07-11 the HelixQ Europe-West region experienced a six-hour
latency spike, exceeding SLA thresholds for 1 200 enterprise tenants. Root
cause: a config rollout pushed an incorrect throttling policy from the
control plane. Mitigation time: 3 hours active response, 6 hours total
impact. Post-incident review flagged that the runbook for Europe-West
config rollouts was missing — on-call engineers had relied on a colleague's
personal notes. No vendor dependency was implicated in this incident.

Section 4. Incident OS-0303 — NovaTrack billing mis-ledger

On 2025-07-19 NovaTrack's billing engine mis-ledgered 2.4 percent of
subscription invoices over a three-day window. Root cause: a timezone
conversion bug in the new invoicing microservice. Detection-to-mitigation
time: 72 hours. The bug had been caught in pre-release QA but was marked
'known minor' and not blocked. Post-mortem recommended blocking any
release with an open 'known minor' on a billing path.

Section 5. Incident OS-0304 — PulseCore certificate outage

On 2025-08-02 the P7 and P8 sensor lines in North America lost mutual-TLS
connectivity after a scheduled certificate rotation. Investigation traced
the outage to an expired intermediate CA cert delivered by vendor NordAxis
on behalf of Orion's PKI. The intermediate had been valid for 90 days and
an auto-renewal hook failed silently. Detection: 4 hours to first alert,
14 hours to full mitigation. The PKI rotation runbook existed but had not
been updated since before the NordAxis contract began.

Section 6. Incident OS-0305 — HelixQ APAC data-store desync

On 2025-08-17 the HelixQ Singapore and Sydney data-stores desynchronised
by up to 27 minutes for a 4-hour window. Root cause: a network partition
between two availability zones coinciding with a rolling upgrade. No
customer-visible data loss, but three enterprise customers filed Sev-2
tickets. Detection-to-mitigation: 2 hours. Vendor NordAxis provided the
network-observability service; its monitoring alerts did not fire during
the partition due to an instrumentation gap.

Section 7. Incident OS-0306 — NovaTrack mobile-app crash loop

On 2025-08-29 the NovaTrack Android app went into a crash loop on 14
percent of devices for eight hours following a hot-fix push. Root cause:
a null-pointer exception in the route-optimization module, triggered by
an unexpected response shape from an upstream API. Detection: 30 minutes.
Mitigation: 8 hours (phased rollback). No vendor involved. The mobile
crash-recovery runbook referenced an old deployment tool and was out of
date.

Section 8. Incident OS-0307 — PulseCore telemetry leak

On 2025-09-11 the Reliability Office detected that PulseCore fleet
telemetry, including device serial numbers and firmware versions, was
being forwarded to a third-party ingest endpoint operated by NordAxis as
part of a fleet-health analytics program. The data contract had been
signed in 2024 but the forwarding was not documented in the current
customer-facing data-flow diagrams. Detection: 21 days from first
forwarded packet. Mitigation: immediate pause of the forwarding, legal
review pending.

Section 9. Incident OS-0308 — HelixQ regional failover miss

On 2025-09-26 the HelixQ US-East failover test failed to complete within
the 15-minute target during a scheduled chaos-engineering exercise.
Failover completed in 38 minutes. Root cause: a region-specific feature
flag had drifted from the configuration-as-code baseline. No customer
impact — the exercise was read-only. The failover playbook referenced an
older feature-flag service that had been deprecated eight months earlier.

Section 10. Reliability Office closing note

The Reliability Office is tabling this review without editorial summary
so that the Executive Risk Committee can form its own systemic reading.
Individual incident owners have submitted remediation plans for their
respective items; cross-incident items remain open. A follow-up session is
scheduled for the first week of Q4 2025.
""".strip()


# Each chunk is ~60-100 words. 16 chunks cover the whole doc.
def split_into_chunks(text: str, n_chunks: int) -> list[str]:
    """Split by section markers if possible, otherwise by word count."""
    # Split by blank-line paragraphs, then greedily group into n_chunks buckets.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= n_chunks:
        # Further split big paragraphs into sentences to reach n_chunks
        words_per_chunk = max(30, len(text.split()) // n_chunks)
        all_words = text.split()
        chunks = []
        for i in range(0, len(all_words), words_per_chunk):
            chunks.append(" ".join(all_words[i:i + words_per_chunk]))
        return chunks[:n_chunks] if len(chunks) >= n_chunks else chunks
    # More paragraphs than chunks — distribute them round-robin
    chunks = [""] * n_chunks
    for i, p in enumerate(paragraphs):
        chunks[i % n_chunks] += (("\n\n" + p) if chunks[i % n_chunks] else p)
    return chunks


@dataclass
class DistributedSummaryTask:
    document: str = ORION_DOCUMENT
    question: str = (
        "What are the top 3 systemic risks facing this company, and which "
        "specific incidents support each? Answer concisely in bullet form."
    )
    # Keywords that indicate each risk has been identified. The team scores
    # +1 for each risk *whose keyword set* appears in their answer.
    systemic_risks: dict[str, list[str]] = field(default_factory=lambda: {
        "supplier_concentration": [
            "nordaxis", "supplier concentration", "vendor concentration",
            "single supplier", "single vendor", "vendor dependency",
        ],
        "documentation_gaps": [
            "documentation gap", "out of date runbook", "stale runbook",
            "outdated runbook", "documentation debt", "runbook", "doc debt",
            "stale documentation",
        ],
        "detection_delays": [
            "detection delay", "slow detection", "detection time",
            "slow to mitigate", "slow response", "incident response",
            "mean time to", "mttr", "mttd", "mitigation delay",
        ],
    })

    def score(self, answer: str) -> dict:
        low = answer.lower()
        hits = {
            risk: any(k in low for k in keywords)
            for risk, keywords in self.systemic_risks.items()
        }
        return {
            **hits,
            "n_risks_identified": sum(hits.values()),
            "total_risks": len(self.systemic_risks),
        }

    def make_chunks(self, n_chunks: int) -> list[str]:
        return split_into_chunks(self.document, n_chunks)
