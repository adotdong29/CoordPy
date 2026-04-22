"""Hardened CASR router — Wave-4 primitives composed under the Phase-14 API.

What this wraps:
  - Cuckoo filter (D7) replaces the Bloom backing of the footprint. Same
    interface (`__contains__`), but adversarially-robust and with an
    explicit false-positive-rate budget.
  - HashChainLog (D8) records every delivery decision into a signed chain
    per recipient. Post-hoc `audit()` verifies tamper-evidence.
  - Merkle-DAG (D3) content-addresses every payload; recipients can prove
    they received identical bulletins by hash.

What it does NOT add:
  - Differential privacy (H4). Codesign payloads are code strings; additive
    noise would destroy them. Skipped honestly — DP applies to aggregate
    numeric statistics, not this task's content.
  - Control barrier function (E5). No continuous state to bound here.
  - VRF committee (D9). The Phase-14 task has all tiered agents participate
    anyway; committee-gating would withhold required bulletin items. Kept
    out of the critical path.

Drop-in replacement API: `route(messages, recipient_id) -> (delivered, stats)`
— same signature and semantics as `CASRRouter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .casr_router import CASRRouter, RouterMessage, RoutingStats, VALID_MODES
from .causal_footprint import CausalFootprint
from .cuckoo_filter import CuckooFilter
from .merkle_dag import MerkleDAG, content_hash
from .peer_review import HashChainLog, verify_log


@dataclass
class HardeningStats:
    """Per-router counters that let us assert the hardening ran."""

    cuckoo_lookups: int = 0
    cuckoo_false_positives: int = 0       # delivered through cuckoo but not in true footprint
    chain_entries_written: int = 0
    merkle_blobs_stored: int = 0
    audits_passed: int = 0
    audits_failed: int = 0


class HardenedCASRRouter:
    """Drop-in router with cuckoo-backed footprints, hash-chain logs, and a
    Merkle store for content addressing.

    Behaviour-identical to `CASRRouter` when the cuckoo filter has zero false
    positives (the typical case at 16-bit fingerprints). Tokens accounting
    is preserved so Phase 14's pre-registered claims remain measurable.
    """

    def __init__(
        self,
        mode: str,
        footprints: dict[str, CausalFootprint],
        cuckoo_fingerprint_bits: int = 16,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.footprints = footprints
        self.cuckoo_fingerprint_bits = cuckoo_fingerprint_bits

        # Build per-recipient cuckoo filters that mirror the true footprints.
        # Capacity = 4× the largest footprint size to keep load factor low.
        max_fp = max((len(fp.members()) for fp in footprints.values()), default=1)
        cap = max(32, max_fp * 4)
        self._cuckoo: dict[str, CuckooFilter] = {}
        for recipient, fp in footprints.items():
            cf = CuckooFilter(
                capacity=cap,
                fingerprint_bits=self.cuckoo_fingerprint_bits,
                seed=hash(recipient) & 0x7FFFFFFF,
            )
            for member in fp.members():
                cf.insert(member)
            self._cuckoo[recipient] = cf

        # Per-recipient hash chain of delivery decisions.
        self._chains: dict[str, HashChainLog] = {
            r: HashChainLog(agent_id=f"router::{r}") for r in footprints
        }
        # Content-addressed store of every payload we've seen.
        self._merkle = MerkleDAG()
        self.stats = HardeningStats()

    # ---------------- public routing API (matches CASRRouter) ---------------

    def route(
        self, messages: list[RouterMessage], recipient_id: str,
    ) -> tuple[list[RouterMessage], RoutingStats]:
        delivered: list[RouterMessage] = []
        stats = RoutingStats()

        if self.mode == "full":
            use_filter = False
        else:
            use_filter = True

        true_fp = self.footprints.get(recipient_id)
        cuckoo = self._cuckoo.get(recipient_id)

        for msg in messages:
            if msg.source_id == recipient_id:
                stats.dropped += 1
                stats.dropped_tokens += msg.tokens
                continue

            if not use_filter:
                keep = True
            else:
                if cuckoo is None:
                    keep = False
                else:
                    self.stats.cuckoo_lookups += 1
                    keep = msg.source_id in cuckoo
                    if keep and true_fp is not None and msg.source_id not in true_fp.members():
                        # Cuckoo false positive — the filter admitted a message
                        # that is NOT in the true causal footprint.
                        self.stats.cuckoo_false_positives += 1

            if keep:
                delivered.append(msg)
                stats.delivered += 1
                stats.delivered_tokens += msg.tokens
                # Hardening side-effects: content-address + sign the decision
                blob_hash = self._merkle.put(
                    {"source": msg.source_id, "payload": msg.payload}
                )
                self.stats.merkle_blobs_stored += 1
                self._chains[recipient_id].append({
                    "source": msg.source_id,
                    "tokens": msg.tokens,
                    "blob_hash": blob_hash,
                    "decision": "deliver",
                })
                self.stats.chain_entries_written += 1
            else:
                stats.dropped += 1
                stats.dropped_tokens += msg.tokens
                self._chains[recipient_id].append({
                    "source": msg.source_id,
                    "tokens": msg.tokens,
                    "decision": "drop",
                })
                self.stats.chain_entries_written += 1

        return delivered, stats

    # ---------------- audit / inspect ----------------

    def audit(self) -> dict:
        """Verify every per-recipient chain and return a summary."""
        results = {}
        for rid, log in self._chains.items():
            ok, reason = verify_log(
                log.entries(), log.public_key, f"router::{rid}",
            )
            results[rid] = {"ok": ok, "reason": reason, "entries": len(log.entries())}
            if ok:
                self.stats.audits_passed += 1
            else:
                self.stats.audits_failed += 1
        return results

    def content_addressed(self, payload: Any) -> str:
        return content_hash(payload)

    def chain_length(self, recipient_id: str) -> int:
        return len(self._chains[recipient_id].entries())
