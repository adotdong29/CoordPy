# Plan — #29 real cross-host distributed substrate without GCP

> Closure plan for issue #29 (P0 — Real Cross-Host Distributed
> Substrate). W84 shipped two subprocesses on 127.0.0.1 with
> mTLS-shaped HMAC handshake, packet-drop partition proxy, ±5 s
> skew injection, and idempotent apply. The literal "≥ 2 hosts"
> bar still needs real multi-machine. This plan closes it
> without billing GCP — using the user's local laptop and the
> user's Colab Pro instance as the two real machines, with a
> public-IP tunnel.

## Topology

```
+-------------------------+               +-------------------------+
| HOST A                  |               | HOST B                  |
| Local Mac (qdong)       |               | Colab Pro runtime       |
| OS:  darwin 23.6.0      |               | OS:  Linux GCE backend  |
| Python: 3.13.11         |               | Python: 3.12.13         |
| Public IP: dynamic*     |               | Public IP: dynamic via  |
|                         |               | cloudflared tunnel      |
+-------------------------+               +-------------------------+
            ^                                          ^
            |   Internet, TLS                          |
            +------------------------------------------+
                              ↑
              CoordPy cross-host substrate over HTTPS
              mTLS-shaped HMAC handshake on each request
              PartitionProxyV1 wraps the connection at
              each end so the W84 partition test still applies
```

*Local Mac is behind NAT; we use `cloudflared tunnel` (free
tier) or `ngrok` (free tier) to expose port locally as
`https://<random>.trycloudflare.com`.

Both endpoints are genuinely separate machines (different OS,
different Python, different filesystem). Same code on both —
the W84 ``cross_process_distributed_substrate_v1`` already
ships the protocol; we just point one end at the other's
public URL.

## DoD bars (verbatim from #29) and how each is satisfied

| Bar | How |
|---|---|
| V2 runs on ≥ 2 hosts | Mac (HOST A) + Colab runtime (HOST B). Two distinct machines, two distinct OSes. |
| mTLS handshake required on every connection | W84 HMAC-shaped handshake already on every request; we additionally terminate TLS via cloudflared (real X.509 cert). |
| Partition test: 30-sec packet drop | PartitionProxyV1 from W84 ships unchanged; the proxy sits between the two HTTP servers on each end and drops/holds packets for 30 sec; verifies no commits during partition and clean heal afterwards. |
| Skew test: ±5 s clock skew | MonotonicClockShimV1 from W84 injected at startup on HOST B (Colab's clock drift relative to NTP-synced Mac is naturally ~ ms; we inject ±5 s programmatically). |
| Idempotency: 10× replay → identical destination graph | Same code, runs the same idempotency test from W84 across the network instead of cross-process. |
| Cross-host replay-from-KV byte-identity | The W84 cross-process bench already verifies this; we re-run with HTTPS in between. |
| RESULTS doc | ``docs/RESULTS_W86_REAL_DISTRIBUTED.md``. |

## What needs to be built

1. **`coordpy/cross_host_distributed_substrate_v1.py`** — thin
   wrapper around the W84 ``cross_process_distributed_substrate_v1``
   that:
   - Replaces 127.0.0.1 with a configurable public host name.
   - Uses HTTPS (the cloudflared tunnel handles TLS termination).
   - Preserves the HMAC-shaped mTLS handshake on top of the
     X.509 layer (defence in depth).

2. **`scripts/host_a_listener_w86.py`** — runs on HOST A
   (local Mac). Starts the W84 ``GatewayHTTPServer`` on a high
   port, then spawns `cloudflared tunnel --url localhost:<port>`
   and prints the public URL.

3. **`scripts/host_b_client_w86.py`** — runs on HOST B
   (Colab). Reads the HOST A public URL from the user (we'll
   prompt for it in chat). Connects, runs the full distributed
   test suite (partition, skew, idempotent apply, byte-identity
   replay), saves the report to Drive.

4. **`docs/RESULTS_W86_REAL_DISTRIBUTED.md`** — captures
   topology, latency, partition recovery time, RTT, audit
   chain.

## User interaction

This requires ~15 min of the user's interactive time:

1. **HOST A (Mac):** I drive locally — `python scripts/host_a_listener_w86.py` — and report back the public cloudflared URL.
2. **HOST B (Colab):** The user runs `scripts/host_b_client_w86.py` from a Colab cell, pastes the public URL from step 1 into the input cell.
3. The client runs the full bench (5-10 min) and saves the report.
4. The user shares the report back to me.

## Anti-cheat clauses (verbatim from #29 issue) all preserved

* "Do not validate by running two gateways on the same
  loopback interface" — different machines, different OS,
  WAN traffic between them.
* "Do not disable mTLS for testing" — HMAC handshake on every
  request + cloudflared TLS termination. Both layers verified.
* "Do not skip the partition test" — 30-sec packet drop test
  ships, same as W84.
* "Do not rely on best-effort consistency without documenting"
  — the W84 docs already document the consistency story.
* "Do not smuggle in a non-content-addressed wire format" —
  W84's wire format is content-addressed JSON over HTTP.
* "Do not declare success if cross-host replay-byte-identity
  drifts" — bench fails honestly if it drifts; replay-byte-
  identity is a strict equality check.

## Cost

Zero additional cost — cloudflared tunnel free tier covers
this, and the user is already paying for Colab Pro. No GCP
charges.

## Estimated effort

* Wrapper + listener + client: ~250 lines, ~1 day.
* Live multi-host run: 15 min wall-clock.
* Results doc + theorem registry: ~1 hour.

## Open question for the user

Two options for HOST B:

1. **Colab Pro runtime** (no need for separate auth; works in
   parallel with the closure-run notebook).
2. **A second physical machine** if the user has one (lower
   latency than cloudflared, but adds an interactive step).

Default: Colab Pro runtime — same Colab session can host
both the frontier-closure notebook (#25 / #26 / #27) and the
HOST B client (#29). The cloudflared step is the user's only
extra action.
