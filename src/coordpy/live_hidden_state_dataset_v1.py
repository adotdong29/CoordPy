"""W84 / P0 #26 — Live LLM training dataset builder.

Issue #26 asks for live-LLM training of the W83 composed
learned memory. This module ships the **dataset builder
infrastructure**; it does NOT close the issue (which requires
a real pretrained model running forward to extract hidden
states; that requires torch + transformers, which is the same
hardware blocker as #25).

Anti-cheat:

* The builder does NOT generate synthetic data labelled as
  "live". If torch / transformers are unavailable, the builder
  raises ``LiveTrainingBlockedOnHardwareError`` with a
  structured technical-gap message — it never silently falls
  back to the W83 synthetic dataset.
* The dataset is content-addressed by (prompt-corpus CID, model
  CID, layer index, precision tier).
* Held-out splits are enforced by prompt-CID disjointness; no
  prompt appears in both train and eval.

Honest scope
------------

* ``W84-L-LIVE-DATASET-V1-RESEARCH-ONLY-CAP`` — explicit-import
  only.
* ``W84-L-LIVE-DATASET-V1-NO-LIVE-EXTRACTION-WITHOUT-HF-CAP`` —
  the builder requires transformers + torch. If they are not
  available, the builder raises rather than mocking.
* ``W84-L-LIVE-DATASET-V1-ONE-LAYER-CAP`` — V1 extracts a
  single layer's hidden state (the layer is content-addressed
  in the dataset CID). Multi-layer training is V2.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence


W84_LIVE_DATASET_V1_SCHEMA_VERSION: str = (
    "coordpy.live_hidden_state_dataset_v1.v1")


class LiveTrainingBlockedOnHardwareError(RuntimeError):
    """Raised when the live-training path cannot run.

    Carries a structured technical-gap message so a caller
    knows exactly what is missing.
    """


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class TrainingTraceWitnessV1:
    """Per-training-run witness capsule (content-addressed).

    Records the seed, optimiser config, loss curve CID, fitted-
    module CID, and the dataset CID. A third party can re-verify
    that the same dataset + seed + optimiser produces the same
    fitted module.
    """

    schema: str
    seed: int
    optimiser_config_cid: str
    loss_curve_cid: str
    fitted_module_cid: str
    dataset_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "optimiser_config_cid": str(
                self.optimiser_config_cid),
            "loss_curve_cid": str(self.loss_curve_cid),
            "fitted_module_cid": str(self.fitted_module_cid),
            "dataset_cid": str(self.dataset_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_training_trace_witness_v1",
            "witness": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class LiveHiddenStateDatasetCapsuleV1:
    """Content-addressed dataset capsule.

    Fields:

    * ``prompt_corpus_cid``: SHA-256 of the canonical prompt
      corpus bytes.
    * ``model_cid``: SHA-256 of the model identifier + the
      precision tier.
    * ``layer_index``: which transformer layer's hidden state
      was extracted.
    * ``precision_tier``: the W84 precision-tier identifier.
    * ``n_prompts_train`` / ``n_prompts_eval``: held-out split
      sizes.
    * ``train_prompt_cids`` / ``eval_prompt_cids``: per-prompt
      CIDs (sets must be disjoint).
    """

    schema: str
    prompt_corpus_cid: str
    model_cid: str
    layer_index: int
    precision_tier: str
    n_prompts_train: int
    n_prompts_eval: int
    train_prompt_cids: tuple[str, ...]
    eval_prompt_cids: tuple[str, ...]

    def __post_init__(self) -> None:
        # Enforce held-out disjointness (anti-cheat).
        train_set = set(self.train_prompt_cids)
        eval_set = set(self.eval_prompt_cids)
        overlap = train_set & eval_set
        if overlap:
            raise ValueError(
                f"train/eval prompts overlap "
                f"(anti-cheat: held-out split must be "
                f"disjoint); overlap={overlap}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "prompt_corpus_cid": str(self.prompt_corpus_cid),
            "model_cid": str(self.model_cid),
            "layer_index": int(self.layer_index),
            "precision_tier": str(self.precision_tier),
            "n_prompts_train": int(self.n_prompts_train),
            "n_prompts_eval": int(self.n_prompts_eval),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_live_hidden_state_dataset_capsule_v1",
            "dataset": self.to_dict(),
            "train_prompt_cids": list(self.train_prompt_cids),
            "eval_prompt_cids": list(self.eval_prompt_cids),
        })


def _prompt_corpus_cid(
        *, prompts: Sequence[str]) -> str:
    """Content-address the prompt corpus (order-sensitive)."""
    return _sha256_hex({
        "kind": "w84_prompt_corpus_v1",
        "n_prompts": int(len(prompts)),
        "prompt_cids": [
            hashlib.sha256(p.encode("utf-8")).hexdigest()
            for p in prompts],
    })


def _prompt_cid(prompt: str) -> str:
    return hashlib.sha256(
        prompt.encode("utf-8")).hexdigest()


def build_live_hidden_state_dataset_v1(
        *,
        prompts_train: Sequence[str],
        prompts_eval: Sequence[str],
        model_name: str,
        layer_index: int = 4,
        precision_tier: str = "tier_fp32",
) -> LiveHiddenStateDatasetCapsuleV1:
    """Build the dataset CAPSULE — does NOT run the model.

    The capsule is the metadata. Actually populating
    ``(input_features, hidden_state_targets)`` requires running
    the model forward, which lives in
    ``materialise_live_hidden_state_dataset_v1``.

    Anti-cheat: held-out disjointness is enforced in
    ``__post_init__``.
    """
    if not prompts_train or not prompts_eval:
        raise ValueError(
            "both train and eval prompt sets must be non-empty")
    corpus_cid = _prompt_corpus_cid(
        prompts=list(prompts_train) + list(prompts_eval))
    model_cid = _sha256_hex({
        "kind": "w84_model_cid_v1",
        "model_name": str(model_name),
        "precision_tier": str(precision_tier),
    })
    return LiveHiddenStateDatasetCapsuleV1(
        schema=W84_LIVE_DATASET_V1_SCHEMA_VERSION,
        prompt_corpus_cid=str(corpus_cid),
        model_cid=str(model_cid),
        layer_index=int(layer_index),
        precision_tier=str(precision_tier),
        n_prompts_train=int(len(prompts_train)),
        n_prompts_eval=int(len(prompts_eval)),
        train_prompt_cids=tuple(
            _prompt_cid(p) for p in prompts_train),
        eval_prompt_cids=tuple(
            _prompt_cid(p) for p in prompts_eval),
    )


def materialise_live_hidden_state_dataset_v1(
        *,
        capsule: LiveHiddenStateDatasetCapsuleV1,
        prompts_train: Sequence[str],
        prompts_eval: Sequence[str],
        model_name: str,
) -> dict[str, Any]:
    """Actually run the model forward and extract hidden states.

    Raises ``LiveTrainingBlockedOnHardwareError`` if torch /
    transformers are not available. Never falls back to
    synthetic data.
    """
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM, AutoTokenizer)
    except Exception as exc:  # noqa: BLE001
        raise LiveTrainingBlockedOnHardwareError(
            "live hidden-state extraction requires torch + "
            "transformers; install ``coordpy[heavy]`` and a "
            "compatible torch wheel. blocked_reason="
            f"{type(exc).__name__}: {exc}") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_name))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_name), torch_dtype=torch.float32)
    model.eval()
    train_hidden = []
    eval_hidden = []
    for split, prompts, out in (
            ("train", list(prompts_train), train_hidden),
            ("eval", list(prompts_eval), eval_hidden)):
        for prompt in prompts:
            ids = tokenizer(
                prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                outs = model(
                    ids, output_hidden_states=True)
            layer_h = outs.hidden_states[
                int(capsule.layer_index)]
            # (batch, seq, hidden) → (seq, hidden); take final
            # token as the W84 V1 target.
            target = layer_h[0, -1, :].float().numpy()
            out.append(target)
    return {
        "dataset_cid": str(capsule.cid()),
        "n_train": int(len(train_hidden)),
        "n_eval": int(len(eval_hidden)),
        "train_hidden_array_shape": (
            tuple(train_hidden[0].shape)
            if train_hidden else ()),
        "eval_hidden_array_shape": (
            tuple(eval_hidden[0].shape)
            if eval_hidden else ()),
    }


__all__ = [
    "W84_LIVE_DATASET_V1_SCHEMA_VERSION",
    "LiveTrainingBlockedOnHardwareError",
    "TrainingTraceWitnessV1",
    "LiveHiddenStateDatasetCapsuleV1",
    "build_live_hidden_state_dataset_v1",
    "materialise_live_hidden_state_dataset_v1",
]
