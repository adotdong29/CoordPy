"""Phase-45 product profiles.

A profile is a frozen dict that resolves to concrete arguments for
the readiness validator and the parser sweep. Profiles are stable,
versioned, and explicit about which parts of the pipeline they
exercise.

Design rules:
  * Adding a new profile must not silently change another profile.
  * A profile never reads env vars at definition time.
  * A profile is a *declaration*, not a function. The runner is the
    sole place that consumes these dicts.

Canonical profiles (stable API surface):
  * ``local_smoke``        — mock mode, ≤8 instances, in-process sandbox.
  * ``bundled_57``         — full 57-instance readiness run, subprocess.
  * ``aspen_mac1_coder``   — ASPEN macbook-1, qwen2.5-coder:14b.
  * ``aspen_mac2_frontier``— ASPEN macbook-2, qwen3.5:35b.
  * ``public_jsonl``       — template; operator swaps ``jsonl`` at call.
"""

from __future__ import annotations

import copy
import os
from typing import Any

SCHEMA_VERSION = "phase45.profile.v1"

_BUNDLED_BANK = os.path.join(
    os.path.dirname(__file__), "..", "tasks", "data",
    "swe_lite_style_bank.jsonl")

_PROFILES: dict[str, dict[str, Any]] = {
    "local_smoke": {
        "description": (
            "Mock-mode smoke: ≤8 instances, in-process sandbox, "
            "strict+robust parser. Target wall: <10 s. CI-friendly."),
        "readiness": {
            "jsonl": _BUNDLED_BANK,
            "limit": 8,
            "sandbox_name": "in_process",
        },
        "sweep": {
            "mode": "mock",
            "model": None,
            "ollama_url": None,
            "jsonl": _BUNDLED_BANK,
            "n_instances": 8,
            "n_distractors": [6],
            "parser_modes": ["strict", "robust"],
            "apply_modes": ["strict"],
            "sandbox": "in_process",
            "enable_raw_capture": False,
        },
    },
    "bundled_57": {
        "description": (
            "Full bundled-bank release-candidate readiness. 57 rows, "
            "subprocess sandbox. Saturation gate: 57/57 on all five "
            "checks."),
        "readiness": {
            "jsonl": _BUNDLED_BANK,
            "limit": None,
            "sandbox_name": "subprocess",
        },
        "sweep": None,
    },
    "bundled_57_mock_sweep": {
        "description": (
            "Full bundled-bank mock oracle sweep + readiness. No LLM. "
            "Pass@1 saturation under oracle is the Theorem P41-1 "
            "reproduction."),
        "readiness": {
            "jsonl": _BUNDLED_BANK,
            "limit": None,
            "sandbox_name": "subprocess",
        },
        "sweep": {
            "mode": "mock",
            "model": None,
            "ollama_url": None,
            "jsonl": _BUNDLED_BANK,
            "n_instances": None,
            "n_distractors": [6],
            "parser_modes": ["strict", "robust"],
            "apply_modes": ["strict"],
            "sandbox": "subprocess",
            "enable_raw_capture": False,
        },
    },
    "aspen_mac1_coder": {
        "description": (
            "ASPEN macbook-1 (192.168.12.191) qwen2.5-coder:14b "
            "real-LLM sweep, 57 instances, strict+robust parser, "
            "raw capture on. Canonical coder-class cell (Phase 44)."),
        "readiness": {
            "jsonl": _BUNDLED_BANK,
            "limit": None,
            "sandbox_name": "subprocess",
        },
        "sweep": {
            "mode": "real",
            "model": "qwen2.5-coder:14b",
            "ollama_url": "http://192.168.12.191:11434",
            "jsonl": _BUNDLED_BANK,
            "n_instances": None,
            "n_distractors": [6],
            "parser_modes": ["strict", "robust"],
            "apply_modes": ["strict"],
            "sandbox": "subprocess",
            "enable_raw_capture": True,
        },
    },
    "aspen_mac2_frontier": {
        "description": (
            "ASPEN macbook-2 (192.168.12.248) qwen3.5:35b "
            "real-LLM sweep, 57 instances, strict+robust parser, "
            "raw capture on. Canonical frontier cell (Phase 44)."),
        "readiness": {
            "jsonl": _BUNDLED_BANK,
            "limit": None,
            "sandbox_name": "subprocess",
        },
        "sweep": {
            "mode": "real",
            "model": "qwen3.5:35b",
            "ollama_url": "http://192.168.12.248:11434",
            "jsonl": _BUNDLED_BANK,
            "n_instances": None,
            "n_distractors": [6],
            "parser_modes": ["strict", "robust"],
            "apply_modes": ["strict"],
            "sandbox": "subprocess",
            "enable_raw_capture": True,
        },
    },
    "public_jsonl": {
        "description": (
            "Template for a public SWE-bench-Lite drop-in JSONL. "
            "Runs readiness only by default; operator supplies "
            "--jsonl to override the bundled path."),
        "readiness": {
            "jsonl": None,
            "limit": None,
            "sandbox_name": "subprocess",
        },
        "sweep": None,
    },
    "aspen_mac1_coder_70b": {
        "description": (
            "Phase-46 frontier slot: a ≥70B coder-finetuned model "
            "hosted on ASPEN macbook-1 (192.168.12.191). The slot "
            "is declared so adding the model is a pure config "
            "change — populate ``sweep.model`` with the Ollama tag "
            "(e.g. ``qwen2.5-coder:70b`` or ``deepseek-coder-v3:70b``) "
            "once the model is resident on the cluster."),
        "requires_model_availability": True,
        "readiness": {
            "jsonl": _BUNDLED_BANK,
            "limit": None,
            "sandbox_name": "subprocess",
        },
        "sweep": {
            "mode": "real",
            "model": "qwen2.5-coder:70b",
            "ollama_url": "http://192.168.12.191:11434",
            "jsonl": _BUNDLED_BANK,
            "n_instances": None,
            "n_distractors": [6],
            "parser_modes": ["strict", "robust"],
            "apply_modes": ["strict"],
            "sandbox": "subprocess",
            "enable_raw_capture": True,
        },
    },
}

_MODEL_CAPABILITY = {
    # model -> suitable-for tags
    "oracle/mock": ("smoke", "readiness", "null_control"),
    "qwen2.5-coder:7b": ("parser_dominant", "smoke"),
    "qwen2.5-coder:14b": (
        "parser_dominant", "serious", "canonical_coder"),
    "qwen3.5:35b": ("semantic_headroom", "frontier", "serious"),
    "gemma2:9b": ("parser_dominant_negative_control",),
    # Phase-46 frontier slot. Not-yet-resident-on-cluster; the entry
    # declares the intended capability class so the CI gate + reports
    # can reason about the slot before the model lands.
    "qwen2.5-coder:70b": (
        "frontier", "canonical_coder", "semantic_headroom",
        "slot_pending_availability"),
    "deepseek-coder-v3:70b": (
        "frontier", "canonical_coder", "semantic_headroom",
        "slot_pending_availability"),
}


def model_availability(model: str | None) -> dict[str, object]:
    """Return a metadata dict used by the runner / CI gate to record
    whether a model is resident on the cluster.

    The check is deliberately *declarative* here — a future upgrade
    can probe the Ollama endpoint. For now the slot is marked
    ``pending_availability`` for the 70B tags and ``assumed_resident``
    for the Phase-42..44 canonical tags.
    """
    if model is None:
        return {"model": None, "availability": "n/a"}
    tags = _MODEL_CAPABILITY.get(model, ())
    if "slot_pending_availability" in tags:
        return {"model": model, "availability": "pending_availability",
                "tags": list(tags)}
    return {"model": model, "availability": "assumed_resident",
            "tags": list(tags)}


def list_profiles() -> list[str]:
    return sorted(_PROFILES.keys())


def get_profile(name: str) -> dict[str, Any]:
    if name not in _PROFILES:
        raise KeyError(
            f"unknown profile: {name!r}; known: {list_profiles()}")
    return copy.deepcopy(_PROFILES[name])


def model_capability_table() -> dict[str, tuple[str, ...]]:
    return dict(_MODEL_CAPABILITY)
