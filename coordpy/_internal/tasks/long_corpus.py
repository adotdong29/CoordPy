"""Longer-than-context corpus — stress-tests single-agent limits.

Generates a fictional multi-section corpus that deliberately exceeds the
default Ollama context window (4096 tokens ≈ ~3000 words). The document
is constructed so:

  - Each individual section fits in a small chunk a single agent can hold.
  - Cross-section patterns (the systemic risks we score) are embedded in
    sections no single agent can read simultaneously.
  - Fictional names / events are invented so no pre-training match
    exists.

Default size: 40 sections × 350 words ≈ 14 000 words. That's well past a
4k-context window and still short enough to keep wall time reasonable
on a 7B model.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field


VENDORS_REAL_ISSUES = ["NordAxis", "Keystone-TLS", "GridFlux"]
VENDORS_NOISE = ["Parallax-Net", "DeltaLake", "HexaSign", "VertexMQ",
                 "OakTrail", "BlueVale", "RipplePoint", "Cedarform"]

PRODUCTS = ["PulseCore", "HelixQ", "NovaTrack", "AetherLogs",
            "BluePeak", "Quanta-IO", "ShieldDB", "MagnetView"]

ROOT_CAUSES_DOC = [
    "runbook was last updated eight months before the incident and referenced a deprecated service",
    "the runbook for this region was missing entirely; on-call engineers relied on a colleague's personal notes",
    "documentation for the signing flow was four months stale",
    "no runbook existed for this failure path",
    "the playbook referenced an older feature-flag service that had been deprecated",
]

ROOT_CAUSES_DETECTION = [
    "detection-to-mitigation: 48 hours",
    "detection took 21 days from first forwarded packet",
    "mitigation time: 72 hours",
    "detection: 4 hours to first alert, 14 hours to full mitigation",
    "failover completed in 38 minutes — 23 minutes past target",
    "detection-to-mitigation: 2 hours",
]

ROOT_CAUSES_OTHER = [
    "a config rollout pushed an incorrect throttling policy",
    "a timezone conversion bug in the new invoicing microservice",
    "a null-pointer exception triggered by an unexpected response shape",
    "a region-specific feature flag had drifted from the configuration-as-code baseline",
    "a silent failure in the auto-renewal hook for an intermediate CA cert",
]


@dataclass
class LongCorpusTask:
    n_sections: int = 40
    target_words_per_section: int = 350
    seed: int = 13
    document: str = ""
    systemic_risks: dict[str, list[str]] = field(default_factory=lambda: {
        "supplier_concentration": [
            "nordaxis", "keystone-tls", "gridflux", "supplier concentration",
            "vendor concentration", "single supplier", "single vendor",
            "vendor dependency", "third-party",
        ],
        "documentation_gaps": [
            "documentation gap", "out of date runbook", "stale runbook",
            "outdated runbook", "runbook", "doc debt",
            "stale documentation", "out-of-date documentation",
        ],
        "detection_delays": [
            "detection delay", "slow detection", "detection time",
            "mttr", "mttd", "time to detect", "time to mitigat",
            "hours to mitigate", "days to detect",
        ],
    })

    def generate(self) -> None:
        rng = random.Random(self.seed)
        sections = [self._header()]

        # Decide how many sections carry each systemic signal
        # We want ~40% vendor signals, ~40% doc signals, ~40% detection signals,
        # with overlap allowed.
        n_vendor = int(self.n_sections * 0.4)
        n_doc = int(self.n_sections * 0.4)
        n_det = int(self.n_sections * 0.45)

        vendor_sections = set(rng.sample(range(self.n_sections), n_vendor))
        doc_sections = set(rng.sample(range(self.n_sections), n_doc))
        det_sections = set(rng.sample(range(self.n_sections), n_det))

        for i in range(self.n_sections):
            section_text = self._generate_section(
                i, rng,
                has_vendor_signal=i in vendor_sections,
                has_doc_signal=i in doc_sections,
                has_det_signal=i in det_sections,
            )
            sections.append(section_text)
        sections.append(self._footer())
        self.document = "\n\n".join(sections)

    def _header(self) -> str:
        return (
            "ORION SYSTEMS — EXTENDED INCIDENT REVIEW — Q3–Q4 2025 (CONFIDENTIAL)\n\n"
            "Prepared for: Executive Risk Committee\n"
            "Prepared by: Reliability Office\n\n"
            "Section 0. Overview\n\n"
            "The Reliability Office has consolidated this extended review of "
            "customer-impacting incidents across our three principal product "
            "lines. This edition of the review is intentionally long: the "
            "Committee asked us to include every incident on file from the "
            "quarter, with each incident documented in full. The document is "
            "provided without editorial commentary; cross-incident synthesis "
            "is the Committee's responsibility."
        )

    def _generate_section(self, idx: int, rng: random.Random,
                          has_vendor_signal: bool,
                          has_doc_signal: bool,
                          has_det_signal: bool) -> str:
        # Base incident details
        incident_id = f"OS-{2025}{idx:04d}"
        product = rng.choice(PRODUCTS)
        region = rng.choice(["Frankfurt", "Singapore", "Sydney", "São Paulo",
                              "Mumbai", "Dublin", "Reykjavík", "Santiago",
                              "Cape Town", "Wellington"])
        severity = rng.choice(["Sev-1", "Sev-2", "Sev-3"])
        date = f"2025-{rng.randint(7, 12):02d}-{rng.randint(1, 28):02d}"

        # Vendor mention — either a real-issue vendor or noise
        if has_vendor_signal:
            vendor = rng.choice(VENDORS_REAL_ISSUES)
            vendor_sentence = (
                f"Vendor {vendor} was implicated: the incident chain "
                "traced back to components delivered under the current "
                f"{vendor} services agreement."
            )
        else:
            vendor = rng.choice(VENDORS_NOISE)
            vendor_sentence = (
                f"Vendor {vendor} provides auxiliary services in this "
                "region but was not part of the incident chain."
            )

        # Doc / runbook
        if has_doc_signal:
            doc_sentence = ("On review, " + rng.choice(ROOT_CAUSES_DOC) + ".")
        else:
            doc_sentence = (
                "Runbooks and playbooks for this path were current and "
                "followed as written."
            )

        # Detection time
        if has_det_signal:
            det_sentence = "The incident log records " + rng.choice(ROOT_CAUSES_DETECTION) + "."
        else:
            det_sentence = (
                "Detection was immediate and the mitigation runbook was "
                "applied within minutes."
            )

        # A generic root cause
        other_cause = rng.choice(ROOT_CAUSES_OTHER)

        # Narrative padding so each section reaches target word count
        padding_bits = []
        padding_templates = [
            (f"The {product} platform's on-call rotation was engaged and "
             "an internal Sev review was convened with stakeholders from "
             "product, SRE and customer support."),
            (f"Customer communications were issued within two hours of "
             "first customer contact, using the standard advisory template."),
            (f"A cross-functional retrospective is scheduled with product, "
             "SRE, and engineering within the next two weeks."),
            (f"Key metrics — including tenant impact, API error rate, "
             "and end-to-end latency — all returned to baseline before "
             "the next scheduled customer status call."),
            (f"Customer-success teams proactively reached out to tenants "
             "on enterprise contracts; no Sev-1 escalations resulted."),
            (f"Standard Sev review is tabled for the next rotation review; "
             "the incident owner has filed all required data."),
            (f"Internal tooling captured the incident timeline end-to-end; "
             "the playback is archived in the Reliability Office vault."),
            (f"Preliminary remediation steps include hardening, additional "
             "telemetry, and a playbook update — all tracked in Jira."),
        ]
        while sum(len(p.split()) for p in padding_bits) < self.target_words_per_section // 2:
            padding_bits.append(rng.choice(padding_templates))

        body = (
            f"Section {idx+1}. Incident {incident_id} — {product} {severity} event in {region}\n\n"
            f"On {date} the {product} platform experienced a {severity} "
            f"customer-impacting event in the {region} region. Triage was "
            "initiated within the standard SLA window. "
            f"{vendor_sentence} The proximate technical cause was: {other_cause}. "
            f"{doc_sentence} {det_sentence} "
            + " ".join(padding_bits)
        )
        return body

    def _footer(self) -> str:
        return (
            "Section 99. Reliability Office closing note\n\n"
            "This review is tabled without editorial synthesis. The "
            "Executive Risk Committee is asked to identify the top three "
            "systemic risks emerging across these incidents and to approve "
            "cross-incident remediation items. Individual incident owners "
            "have remediation plans on file for their respective items."
        )

    # ---- Scoring ----

    def score(self, answer: str) -> dict:
        low = answer.lower()
        hits = {
            risk: any(k in low for k in kws)
            for risk, kws in self.systemic_risks.items()
        }
        return {
            **hits,
            "n_risks_identified": sum(hits.values()),
            "total_risks": len(self.systemic_risks),
        }

    def chunk(self, n_chunks: int) -> list[str]:
        """Split document into n_chunks chunks."""
        words = self.document.split()
        size = max(1, len(words) // n_chunks)
        chunks = []
        for i in range(0, len(words), size):
            chunks.append(" ".join(words[i:i + size]))
        return chunks[:n_chunks] if len(chunks) >= n_chunks else chunks

    @property
    def word_count(self) -> int:
        return len(self.document.split())
