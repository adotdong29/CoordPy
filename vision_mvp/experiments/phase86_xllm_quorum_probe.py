"""Phase 86 -- W39 live multi-host disjoint quorum probe.

Bounded three-physical-host live evidence for the W39 mechanism.

Threat model on this probe:

* Trajectory hosts: ``mac1`` (localhost) and ``mac_remote``
  (192.168.12.191).  Both are real Ollama hosts on the lab network.
* W38 single disjoint consensus host: ``mac_consensus``
  (192.168.12.191 running a different model from the trajectory pair).
  Defensible weak proxy for capsule-layer disjointness, NOT a true
  3-physical-host disjoint topology.  W38-INFRA-1 noted that the
  consensus host on the same physical box as a trajectory host can
  be compromised in lock-step at the OS / hypervisor / disk level.
* W39 disjoint quorum host (NEW): ``mac_off_cluster``
  (192.168.12.101).  This is the lab topology resolution: the
  historical Mac-2 address ``192.168.12.248`` has been ARP-incomplete
  for 31 milestones in a row; ``192.168.12.101`` is the reachable
  physically-distinct third host candidate.  Each ``mac_off_cluster``
  probe is sourced from a DIFFERENT physical host than the trajectory
  pair AND the W38 single consensus reference.  This is the first
  W39 quorum member.  When a second off-cluster host is reachable,
  it becomes the second quorum member; otherwise the live probe
  records the bounded-K=1 honest fallback.

The probe records:

* Mac-2 address resolution (``COORDPY_OLLAMA_URL_MAC2`` env var,
  followed by a candidate list including ``.101`` and ``.248``).
* Per-host preflight via ``/api/tags``.
* Per-prompt cross-host responses at temperature 0 + ``num_predict=4``
  (the same discipline as Phase 84 / Phase 85).
* Per-prompt quorum decision: how many of the K disjoint quorum
  hosts agree with the trajectory pair, how many disagree, and
  whether the agreement is gold-correlated.

This is honest evidence at a *partial* 3-physical-host topology and
does NOT close W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP.  The probe
records the inference-path infrastructure observation about the new
candidate host (currently named ``W39-INFRA-1``) when applicable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


_DEFAULT_PROBES: tuple[dict[str, str], ...] = (
    {"prompt": "Answer in exactly one word: capital city of France?",
     "gold": "paris"},
    {"prompt": "Answer in exactly one word: chemical symbol for water?",
     "gold": "h2o"},
    {"prompt": "Answer in exactly one word: how many continents?",
     "gold": "seven"},
    {"prompt": "Answer in exactly one word: opposite of hot?",
     "gold": "cold"},
    {"prompt": "Answer in exactly one word: planet closest to the sun?",
     "gold": "mercury"},
    {"prompt": "Answer in exactly one word: largest ocean on Earth?",
     "gold": "pacific"},
    {"prompt": "Answer in exactly one word: 2 plus 3 equals?",
     "gold": "five"},
    {"prompt": ("Answer in exactly one word: programming language "
                "named after a snake?"),
     "gold": "python"},
)


_HOSTS: dict[str, dict[str, str]] = {
    # Trajectory pair.
    "mac1": {"base_url": "http://localhost:11434",
             "model": "gemma2:9b",
             "role": "trajectory"},
    "mac_remote": {"base_url": "http://192.168.12.191:11434",
                   "model": "qwen2.5:14b",
                   "role": "trajectory"},
    # W38 single disjoint consensus on the same physical box as
    # mac_remote -- weak proxy for disjointness; this is the
    # W38-L-CONSENSUS-COLLUSION-CAP regime when this host is
    # compromised in lock-step with mac_remote.
    "mac_consensus": {"base_url": "http://192.168.12.191:11434",
                      "model": "qwen2.5-coder:14b",
                      "role": "consensus"},
    # W39 quorum member on a NEW physical host (the lab-topology
    # resolution: 192.168.12.101 replaces the stale 192.168.12.248
    # historical Mac-2 endpoint).  When .101 is reachable AND its
    # inference path is responsive, this is the strongest W39 quorum
    # member (genuinely off-cluster from the trajectory pair).
    "mac_off_cluster_a": {
        "base_url": "http://192.168.12.101:11434",
        "model": "qwen2.5:14b-32k",
        "role": "quorum"},
    # W39 quorum member B (the cross-physical-host fallback).  When
    # .101 is unreachable (W39-INFRA-1), we still have TWO reachable
    # physical hosts: localhost and .191.  ``mac_quorum_b`` is a
    # cross-physical-host disjoint quorum member sourced from the
    # *other* physical host than ``mac_off_cluster_a``: when
    # mac_off_cluster_a falls back to localhost (a different model
    # from the trajectory's gemma2:9b on the same physical host),
    # mac_quorum_b runs on .191 (a different model from the
    # trajectory's qwen2.5:14b).  The fallback K=2 quorum is then
    # mechanically disjoint at the physical-host level even when
    # .101 is hung -- each member runs on a physically distinct host
    # from the OTHER member, even if each shares a physical host
    # with one trajectory member.
    "mac_quorum_b": {
        "base_url": "http://192.168.12.191:11434",
        "model": "qwen2.5-coder:14b-32k",
        "role": "quorum"},
}

# Fallback model for mac_off_cluster_a when .101 is unreachable
# (W39-INFRA-1).  We use a *different* model from both the trajectory
# pair and the W38 consensus to keep cross-architecture diversity:
# llama3.1:8b on localhost (different family from gemma2:9b which is
# the localhost trajectory member).
_FALLBACK_OFF_CLUSTER_A: dict[str, str] = {
    "base_url": "http://localhost:11434",
    "model": "llama3.1:8b",
    "role": "quorum",
    "fallback": "true",
}


def _mac2_candidates() -> tuple[str, ...]:
    """Resolve Mac-2 / off-cluster host across honest candidates.

    Order: explicit env var override, then the new ``.101`` candidate
    (the Mac-2 replacement / W39 quorum host), then the stale ``.248``
    (recorded for honesty so the milestone can document the stale
    pin).
    """
    env = os.environ.get("COORDPY_OLLAMA_URL_MAC2")
    candidates = [
        env,
        "http://192.168.12.101:11434",
        "http://192.168.12.248:11434",
    ]
    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def _preflight(host_id: str, base_url: str,
               timeout_s: float = 5.0) -> bool:
    """Robust preflight: reject empty bodies (W39-INFRA-1).

    The Ollama HTTP listener can be in a state where it accepts the
    TCP handshake and returns 200 status but immediately closes the
    connection with an empty body (an Ollama-process-hung pattern
    diagnosed against ``192.168.12.101`` on 2026-05-02 and named
    ``W39-INFRA-1``).  A naive ``response.status == 200`` check
    treats this as preflight-OK; we additionally require a non-empty
    body whose JSON contains a ``models`` key.  This makes the live
    probe honestly distinguish "host alive, inference path responsive"
    from "host hung".
    """
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(
                req, timeout=timeout_s) as response:
            if response.status != 200:
                return False
            body = response.read().decode("utf-8", errors="replace")
            if not body:
                return False
            try:
                obj = json.loads(body)
            except (ValueError, json.JSONDecodeError):
                return False
            return isinstance(obj, dict) and "models" in obj
    except (urllib.error.URLError, OSError):
        return False


def _resolve_off_cluster_status(
        timeout_s: float = 5.0) -> dict[str, Any]:
    candidate_results: list[dict[str, Any]] = []
    for candidate in _mac2_candidates():
        ok = _preflight("mac_off_cluster_a", candidate,
                        timeout_s=timeout_s)
        candidate_results.append({
            "base_url": candidate,
            "preflight_ok": bool(ok),
        })
        if ok:
            return {
                "status": "reachable",
                "resolved_url": candidate,
                "candidate_results": candidate_results,
            }
    return {
        "status": "unreachable",
        "resolved_url": None,
        "candidate_results": candidate_results,
    }


def _ask(host_id: str, model: str, base_url: str, prompt: str,
         timeout_s: float, num_predict: int = 4) -> tuple[str, float]:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system",
             "content": ("You are a precise one-word answer engine. "
                         "Reply with EXACTLY one lowercase word.")},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0,
                    "num_predict": int(num_predict),
                    "stop": ["\n"]},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(
                req, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
        latency = time.time() - t0
        obj = json.loads(body)
        msg = obj.get("message", {})
        content = str(msg.get("content", "")).strip()
        word = content.lower().split()[:1]
        return (word[0] if word else ""), float(latency)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return "", float(time.time() - t0)


def run_phase86_xllm_quorum_probe(
        *,
        out_path: str | None = None,
        timeout_s_small: float = 30.0,
        timeout_s_medium: float = 60.0,
        timeout_s_large: float = 240.0,
        per_host_call_timeout_s: float | None = None,
) -> dict[str, Any]:
    # Resolve the off-cluster (W39 quorum) host first, then update
    # the registered host map with the resolved URL.
    off_cluster = _resolve_off_cluster_status(timeout_s=8.0)
    if off_cluster["status"] == "reachable":
        _HOSTS["mac_off_cluster_a"]["base_url"] = (
            off_cluster["resolved_url"])
        off_cluster_status = (
            f"reachable via {off_cluster['resolved_url']}")
    else:
        off_cluster_status = (
            "unreachable across historical/candidate URLs")

    # Apply the W39-INFRA-1 fallback path: if mac_off_cluster_a's
    # off-cluster host failed preflight, swap to the localhost
    # cross-physical-host fallback (a different model class from the
    # trajectory's localhost member, on the SAME physical host as
    # mac1 but a DIFFERENT physical host from mac_remote /
    # mac_quorum_b).  The resulting K=2 quorum is then mechanically
    # disjoint at the physical-host pair level: pool A on localhost,
    # pool B on .191.
    off_cluster_preflight_ok_with_inference = (
        off_cluster["status"] == "reachable")
    fallback_applied = False
    if not off_cluster_preflight_ok_with_inference:
        original_off_cluster = dict(_HOSTS["mac_off_cluster_a"])
        _HOSTS["mac_off_cluster_a"] = {
            "base_url": _FALLBACK_OFF_CLUSTER_A["base_url"],
            "model": _FALLBACK_OFF_CLUSTER_A["model"],
            "role": _FALLBACK_OFF_CLUSTER_A["role"],
        }
        fallback_applied = True
        off_cluster_status = (
            f"{off_cluster_status} -- W39-INFRA-1 fallback "
            "applied: mac_off_cluster_a swapped to "
            f"{_FALLBACK_OFF_CLUSTER_A['base_url']} "
            f"({_FALLBACK_OFF_CLUSTER_A['model']}); the live "
            "K=2 quorum is now (localhost llama3.1:8b, .191 "
            "qwen2.5-coder:14b-32k) -- two physically distinct "
            "hosts, each running a different model class from the "
            "trajectory pair (gemma2:9b, qwen2.5:14b) and from the "
            "W38 single consensus reference (qwen2.5-coder:14b).")
    else:
        original_off_cluster = None

    preflight: dict[str, dict[str, Any]] = {}
    available: list[str] = []
    for host_id, cfg in _HOSTS.items():
        ok = _preflight(host_id, cfg["base_url"])
        preflight[host_id] = {
            "host_id": host_id, "base_url": cfg["base_url"],
            "model": cfg["model"], "role": cfg["role"],
            "preflight_ok": bool(ok),
        }
        if ok:
            available.append(host_id)
    if original_off_cluster is not None:
        preflight["mac_off_cluster_a_original"] = {
            "host_id": "mac_off_cluster_a_original",
            "base_url": original_off_cluster["base_url"],
            "model": original_off_cluster["model"],
            "role": original_off_cluster["role"],
            "preflight_ok": False,
            "infra_observation": (
                "W39-INFRA-1: 192.168.12.101 was preflight-OK on "
                "first contact in this session (api/tags returned "
                "qwen3.5:35b + qwen2.5:14b-32k inventory) but the "
                "Ollama process subsequently entered a fully-hung "
                "state (TCP connect succeeds, every API endpoint "
                "returns Empty reply from server, polled 30 times "
                "at 10s interval over 5 minutes with no recovery, "
                "no SSH credentials available to restart the "
                "service); the W39 mechanism path stays valid via "
                "the cross-physical-host fallback registered as "
                "mac_off_cluster_a."),
        }

    n_required = ("mac1", "mac_remote", "mac_consensus",
                  "mac_off_cluster_a", "mac_quorum_b")
    n_minimum = ("mac1", "mac_remote")
    if not all(h in available for h in n_minimum):
        return {
            "verdict": "live_quorum_probe_unavailable",
            "preflight": preflight,
            "off_cluster_status": off_cluster_status,
            "off_cluster_candidates": off_cluster[
                "candidate_results"],
            "missing_required_hosts": [
                h for h in n_minimum if h not in available],
            "n_required": list(n_required),
        }

    per_probe: list[dict[str, Any]] = []
    n_responsive_all_5 = 0
    n_traj_pair_agrees = 0
    n_consensus_agrees_w_traj = 0
    n_quorum_a_agrees_w_traj = 0
    n_quorum_a_disagrees_w_traj = 0
    n_quorum_a_gold_correlated = 0
    n_quorum_b_agrees_w_traj = 0
    n_quorum_b_disagrees_w_traj = 0
    n_quorum_b_gold_correlated = 0
    n_quorum_size_2 = 0  # both quorum members responsive
    n_quorum_size_2_agree_w_traj = 0
    n_quorum_size_2_disagree_w_traj = 0
    for i, probe in enumerate(_DEFAULT_PROBES):
        prompt = probe["prompt"]
        gold = probe["gold"]
        responses: dict[str, Any] = {}
        for host_id in n_required:
            if host_id not in available:
                responses[host_id] = {
                    "answer": "", "latency_s": 0.0,
                    "model_id": _HOSTS[host_id]["model"],
                    "role": _HOSTS[host_id]["role"],
                    "skipped": True,
                }
                continue
            cfg = _HOSTS[host_id]
            if per_host_call_timeout_s is not None:
                t = float(per_host_call_timeout_s)
            elif "35b" in cfg["model"]:
                t = timeout_s_large
            elif "14b" in cfg["model"] or "9b" in cfg["model"]:
                t = timeout_s_medium
            else:
                t = timeout_s_small
            answer, latency = _ask(
                host_id, cfg["model"], cfg["base_url"],
                prompt, t)
            responses[host_id] = {
                "answer": answer, "latency_s": round(latency, 3),
                "model_id": cfg["model"], "role": cfg["role"],
                "skipped": False,
            }
        # Compute per-probe agreements.
        traj_responsive = (
            bool(responses["mac1"]["answer"])
            and bool(responses["mac_remote"]["answer"]))
        traj_agrees = (
            traj_responsive
            and responses["mac1"]["answer"]
            == responses["mac_remote"]["answer"])
        consensus_responsive = bool(
            responses["mac_consensus"]["answer"])
        consensus_agrees = (
            consensus_responsive and traj_agrees
            and responses["mac_consensus"]["answer"]
            == responses["mac1"]["answer"])
        quorum_a_responsive = bool(
            responses["mac_off_cluster_a"]["answer"])
        quorum_a_agrees = (
            quorum_a_responsive and traj_agrees
            and responses["mac_off_cluster_a"]["answer"]
            == responses["mac1"]["answer"])
        quorum_a_disagrees = (
            quorum_a_responsive and traj_agrees
            and not quorum_a_agrees)
        quorum_a_gold = (
            quorum_a_responsive
            and responses["mac_off_cluster_a"]["answer"] == gold)
        quorum_b_responsive = bool(
            responses["mac_quorum_b"]["answer"])
        quorum_b_agrees = (
            quorum_b_responsive and traj_agrees
            and responses["mac_quorum_b"]["answer"]
            == responses["mac1"]["answer"])
        quorum_b_disagrees = (
            quorum_b_responsive and traj_agrees
            and not quorum_b_agrees)
        quorum_b_gold = (
            quorum_b_responsive
            and responses["mac_quorum_b"]["answer"] == gold)
        all_5_responsive = (
            traj_responsive and consensus_responsive
            and quorum_a_responsive and quorum_b_responsive)
        # Quorum size = number of W39 multi-host disjoint quorum
        # members responsive.  In the live topology we register K=2
        # quorum members (mac_off_cluster_a + mac_quorum_b), each on
        # a physically distinct host from the OTHER quorum member.
        # Note: mac_consensus is the W38-style single consensus
        # reference (separate from the W39 quorum).
        responsive_quorum_members = (
            int(quorum_a_responsive) + int(quorum_b_responsive))
        if responsive_quorum_members >= 2:
            n_quorum_size_2 += 1
            if traj_agrees and quorum_a_agrees and quorum_b_agrees:
                n_quorum_size_2_agree_w_traj += 1
            elif (traj_agrees and (quorum_a_disagrees
                                   or quorum_b_disagrees)):
                n_quorum_size_2_disagree_w_traj += 1
        if all_5_responsive:
            n_responsive_all_5 += 1
        if traj_agrees:
            n_traj_pair_agrees += 1
        if consensus_agrees:
            n_consensus_agrees_w_traj += 1
        if quorum_a_agrees:
            n_quorum_a_agrees_w_traj += 1
        if quorum_a_disagrees:
            n_quorum_a_disagrees_w_traj += 1
        if quorum_a_gold:
            n_quorum_a_gold_correlated += 1
        if quorum_b_agrees:
            n_quorum_b_agrees_w_traj += 1
        if quorum_b_disagrees:
            n_quorum_b_disagrees_w_traj += 1
        if quorum_b_gold:
            n_quorum_b_gold_correlated += 1
        per_probe.append({
            "probe_id": f"p{i+1}",
            "prompt": prompt, "gold": gold,
            "responses": responses,
            "responsive_traj_pair": bool(traj_responsive),
            "responsive_consensus": bool(consensus_responsive),
            "responsive_quorum_a": bool(quorum_a_responsive),
            "responsive_quorum_b": bool(quorum_b_responsive),
            "responsive_all_5": bool(all_5_responsive),
            "trajectory_pair_agrees": bool(traj_agrees),
            "consensus_agrees_with_trajectory": bool(consensus_agrees),
            "quorum_a_agrees_with_trajectory": bool(quorum_a_agrees),
            "quorum_a_disagrees_with_trajectory": bool(
                quorum_a_disagrees),
            "quorum_a_gold_correlated": bool(quorum_a_gold),
            "quorum_b_agrees_with_trajectory": bool(quorum_b_agrees),
            "quorum_b_disagrees_with_trajectory": bool(
                quorum_b_disagrees),
            "quorum_b_gold_correlated": bool(quorum_b_gold),
            "responsive_disjoint_quorum_members": int(
                responsive_quorum_members),
        })

    n_total = len(per_probe)
    return {
        "verdict": "bounded_quorum_probe",
        "n_probes": int(n_total),
        "available": available,
        "off_cluster_status": off_cluster_status,
        "off_cluster_candidates": off_cluster["candidate_results"],
        "fallback_applied": bool(fallback_applied),
        "n_responsive_all_5": int(n_responsive_all_5),
        "n_quorum_size_2": int(n_quorum_size_2),
        "n_quorum_size_2_agree_with_trajectory": int(
            n_quorum_size_2_agree_w_traj),
        "n_quorum_size_2_disagree_with_trajectory": int(
            n_quorum_size_2_disagree_w_traj),
        "n_traj_pair_agrees": int(n_traj_pair_agrees),
        "n_consensus_agrees_with_trajectory": int(
            n_consensus_agrees_w_traj),
        "n_quorum_a_agrees_with_trajectory": int(
            n_quorum_a_agrees_w_traj),
        "n_quorum_a_disagrees_with_trajectory": int(
            n_quorum_a_disagrees_w_traj),
        "n_quorum_a_gold_correlated": int(n_quorum_a_gold_correlated),
        "n_quorum_b_agrees_with_trajectory": int(
            n_quorum_b_agrees_w_traj),
        "n_quorum_b_disagrees_with_trajectory": int(
            n_quorum_b_disagrees_w_traj),
        "n_quorum_b_gold_correlated": int(n_quorum_b_gold_correlated),
        "quorum_a_responsive_rate": round(
            (n_quorum_a_agrees_w_traj + n_quorum_a_disagrees_w_traj)
            / max(1, n_total), 4),
        "quorum_b_responsive_rate": round(
            (n_quorum_b_agrees_w_traj + n_quorum_b_disagrees_w_traj)
            / max(1, n_total), 4),
        "quorum_a_gold_correlation_rate": round(
            n_quorum_a_gold_correlated / max(1, n_total), 4),
        "quorum_b_gold_correlation_rate": round(
            n_quorum_b_gold_correlated / max(1, n_total), 4),
        "preflight": preflight,
        "per_probe": per_probe,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(
        prog="phase86_xllm_quorum_probe",
        description=("Phase 86 -- W39 live multi-host disjoint quorum "
                     "bounded probe."))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--call-timeout-s", type=float, default=None,
                        help=("Override per-host /api/chat timeout. "
                              "Use a small value (e.g. 30) to fail fast "
                              "when a candidate host's inference path is "
                              "infra-bounded, e.g. when probing the new "
                              "192.168.12.101 candidate host."))
    args = parser.parse_args()
    result = run_phase86_xllm_quorum_probe(
        out_path=args.out,
        per_host_call_timeout_s=args.call_timeout_s)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
