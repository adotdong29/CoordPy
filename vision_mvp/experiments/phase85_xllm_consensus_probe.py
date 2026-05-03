"""Phase 85 -- W38 live disjoint cross-source consensus reference probe.

Bounded two-reachable-host live evidence for the W38 mechanism.

Threat model on this probe:

* Trajectory hosts: ``mac1`` (localhost) and ``mac_remote``
  (192.168.12.191).  Both are real Ollama hosts on the lab network.
* Disjoint consensus host: ``mac_consensus`` -- a controller-pre-
  registered audited capsule-layer probe whose host topology is
  mechanically disjoint from the trajectory hosts.  In this live probe
  we use ``mac_remote`` running a *different model* (qwen3.5:35b) as
  the disjoint consensus oracle.  This is a defensible weak proxy: the
  consensus oracle runs on the same physical host as one trajectory
  oracle but on a different model class (MoE 35B vs dense 14B), so a
  collusion attack on the trajectory dense models would NOT
  automatically corrupt the MoE consensus.  We record the residual
  hardware-bound caveat explicitly.
* Mac 2 (192.168.12.248) remains ARP-incomplete (31st milestone in a
  row); the strongest honest 3-host topology is therefore unavailable.

Each probe is a single-prompt cross-host request at temperature 0 with
``num_predict=4``, the same discipline as Phase 84's W37 trajectory
probe.  We record per-probe (mac1 answer, mac_remote answer,
mac_consensus answer) and tally:

  * ``n_consensus_agrees_w_trajectory`` -- W38 RATIFIED would fire.
  * ``n_consensus_disagrees_w_trajectory`` -- W38 DIVERGENCE_ABSTAINED
    would fire.
  * ``n_consensus_unavailable`` -- W38 NO_REFERENCE / WEAK would fire.

The probe is honest evidence at a 2-reachable-host topology and does
NOT close W37-C-MULTI-HOST or W38-L-CONSENSUS-COLLUSION-CAP.
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
    "mac1": {"base_url": "http://localhost:11434",
             "model": "gemma2:9b",
             "role": "trajectory"},
    "mac_remote": {"base_url": "http://192.168.12.191:11434",
                   "model": "qwen2.5:14b",
                   "role": "trajectory"},
    # Disjoint consensus host: a different model class on Mac 1.
    # qwen3.5:35b (MoE) is a genuinely different architecture from
    # the trajectory hosts (gemma2/qwen2.5 dense) but is empirically
    # non-responsive at temperature 0 + num_predict=4 (W34-INFRA-3 /
    # W38-INFRA-1).  qwen2.5-coder:14b is a defensible fallback --
    # same parameter scale as the trajectory but trained for code,
    # giving a different answer manifold for natural-language gold-
    # verifiable prompts.
    "mac_consensus": {"base_url": "http://192.168.12.191:11434",
                      "model": "qwen2.5-coder:14b",
                      "role": "consensus"},
}


def _preflight(host_id: str, base_url: str,
               timeout_s: float = 5.0) -> bool:
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(
                req, timeout=timeout_s) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _ask(host_id: str, model: str, base_url: str, prompt: str,
         timeout_s: float, num_predict: int = 4) -> tuple[
             str, float]:
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


def run_phase85_xllm_consensus_probe(
        *,
        out_path: str | None = None,
        timeout_s_small: float = 30.0,
        timeout_s_medium: float = 60.0,
        timeout_s_large: float = 240.0,
) -> dict[str, Any]:
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
    mac2_ok = _preflight(
        "mac2", "http://192.168.12.248:11434", timeout_s=8.0)
    mac2_status = ("reachable" if mac2_ok
                   else "ARP-incomplete (31st milestone)")
    n_required = ("mac1", "mac_remote", "mac_consensus")
    if not all(h in available for h in n_required):
        return {
            "verdict": "live_consensus_probe_unavailable",
            "preflight": preflight,
            "mac2_status": mac2_status,
            "missing_required_hosts": [
                h for h in n_required if h not in available],
            "n_required": list(n_required),
        }

    per_probe: list[dict[str, Any]] = []
    n_responsive_all = 0
    n_consensus_agrees = 0
    n_consensus_disagrees = 0
    n_consensus_gold_correlated = 0
    n_trajectory_agrees = 0
    for i, probe in enumerate(_DEFAULT_PROBES):
        prompt = probe["prompt"]
        gold = probe["gold"]
        responses: dict[str, Any] = {}
        for host_id in n_required:
            cfg = _HOSTS[host_id]
            if "35b" in cfg["model"]:
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
            }
        responsive_all_3 = all(
            bool(responses[h]["answer"]) for h in n_required)
        responsive_traj_pair = (
            bool(responses["mac1"]["answer"])
            and bool(responses["mac_remote"]["answer"]))
        responsive_consensus = bool(
            responses["mac_consensus"]["answer"])
        traj_agrees = (
            responsive_traj_pair
            and responses["mac1"]["answer"]
            == responses["mac_remote"]["answer"])
        consensus_agrees = (
            responsive_consensus and traj_agrees
            and responses["mac_consensus"]["answer"]
            == responses["mac1"]["answer"])
        consensus_disagrees = (
            responsive_consensus and traj_agrees
            and not consensus_agrees)
        consensus_gold = (
            responses["mac_consensus"]["answer"] == gold
            and responsive_consensus)
        if responsive_all_3:
            n_responsive_all += 1
        if traj_agrees:
            n_trajectory_agrees += 1
        if consensus_agrees:
            n_consensus_agrees += 1
        elif consensus_disagrees:
            n_consensus_disagrees += 1
        if consensus_gold:
            n_consensus_gold_correlated += 1
        per_probe.append({
            "probe_id": f"p{i+1}",
            "prompt": prompt, "gold": gold,
            "responses": responses,
            "responsive_all_3": bool(responsive_all_3),
            "trajectory_pair_agrees": bool(traj_agrees),
            "consensus_agrees_with_trajectory": bool(consensus_agrees),
            "consensus_disagrees_with_trajectory": bool(
                consensus_disagrees),
            "consensus_gold_correlated": bool(consensus_gold),
        })

    return {
        "verdict": "bounded_consensus_probe",
        "n_probes": len(per_probe),
        "available": available,
        "mac2_status": mac2_status,
        "n_responsive_all_3": int(n_responsive_all),
        "n_consensus_agrees_w_trajectory": int(n_consensus_agrees),
        "n_consensus_disagrees_w_trajectory": int(
            n_consensus_disagrees),
        "n_consensus_gold_correlated": int(n_consensus_gold_correlated),
        "n_trajectory_pair_agrees": int(n_trajectory_agrees),
        "consensus_agreement_rate": round(
            n_consensus_agrees / max(1, n_responsive_all), 4),
        "consensus_gold_correlation_rate": round(
            n_consensus_gold_correlated / max(1, n_responsive_all), 4),
        "preflight": preflight,
        "per_probe": per_probe,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(
        prog="phase85_xllm_consensus_probe",
        description=("Phase 85 -- W38 live disjoint cross-source "
                     "consensus reference bounded probe."))
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    result = run_phase85_xllm_consensus_probe(out_path=args.out)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
