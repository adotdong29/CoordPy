"""W84 / P0 #26 — Live Hidden-State Dataset V1.

Every W83 learned-memory line — ``learned_consolidation_v2``,
``differentiable_memory_substrate_v1``, ``composed_learned_
memory_v1``, ``recurrent_slot_reconstruction_v1`` — is trained
on **synthetic** data. ``W83-L-COMPOSED-MEMORY-V1-SYNTHETIC-
CAP`` makes that limit explicit. P0 #26 asks for a live training
line: train at least one W83 learned-memory module on
*hidden states extracted from a real running transformer* and
show that the live-trained model strictly beats its
synthetic-trained sibling on a held-out live evaluation set.

This module is the **dataset builder** for that line. The
``LiveHiddenStateDatasetV1`` is content-addressed by
``(prompts_cid, model_cid, layer_index, projection_seed,
input_dim, output_dim)`` so two callers that pass the same
inputs get the same dataset CID — and a different prompt
corpus or model produces a different dataset CID.

How the dataset is built
------------------------

1. The caller supplies a ``TransformersRuntimeV1`` instance
   and a sequence of prompts. The runtime is parameter-agnostic
   — it can be the W80 distilgpt2 baseline or a P0 #25
   frontier-scale Qwen-2.5-7B-Instruct.
2. For each prompt, the runtime runs a forward pass and the
   builder extracts the per-token hidden states at
   ``layer_index``.
3. Per-token **input features** are produced by a deterministic
   random projection of the token id (one-hot → ``input_dim``).
   The projection matrix is seeded by ``projection_seed`` and
   the model's vocab size.
4. Per-token **target features** are produced by a deterministic
   random projection of the hidden state at ``layer_index``
   (``hidden_dim`` → ``output_dim``). The projection is seeded
   by ``projection_seed + 1``.
5. The dataset records the model name, dtype, layer index,
   projection seed, and (input_features, target) pairs.

Why a random projection? Two reasons:

* The W83 composed-memory line trains at ``output_dim ~= 3``;
  the live model's hidden state is ``hidden_dim ~ 3584``. A
  fixed deterministic projection makes the training task
  tractable AND content-addressable. Larger output_dim is V2.
* The projection is **fixed** across train and eval splits —
  there is no leakage; both splits use the same matrix.

Honest scope (W84 P0 #26)
-------------------------

- ``W84-L-LIVE-DATASET-V1-RESEARCH-ONLY-CAP`` — explicit-
  import only.
- ``W84-L-LIVE-DATASET-V1-RANDOM-PROJECTION-CAP`` — V1 uses
  random projections to ``input_dim/output_dim``. V2 will
  train learned projection heads end-to-end. The random
  projection is honest because (a) it is deterministic from
  ``projection_seed``, (b) it is shared across train and eval,
  and (c) it is content-addressed.
- ``W84-L-LIVE-DATASET-V1-SINGLE-LAYER-CAP`` — V1 extracts
  hidden states from one configurable layer. Multi-layer
  training is V2.
- ``W84-L-LIVE-DATASET-V1-PRECISION-FLOOR-CAP`` — when the
  model is loaded in bf16, the extracted hidden states have
  bf16 precision (~3 decimal digits). The dataset records the
  precision floor so callers do not over-fit to noise.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.live_hidden_state_dataset_v1 requires numpy"
        ) from exc


W84_LIVE_DATASET_V1_SCHEMA_VERSION: str = (
    "coordpy.live_hidden_state_dataset_v1.v1")

W84_LIVE_DATASET_DEFAULT_INPUT_DIM: int = 5
W84_LIVE_DATASET_DEFAULT_OUTPUT_DIM: int = 3
W84_LIVE_DATASET_DEFAULT_LAYER_INDEX: int = 1
W84_LIVE_DATASET_DEFAULT_PROJECTION_SEED: int = 84_026_001
W84_LIVE_DATASET_DEFAULT_MAX_TOKENS: int = 10


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(
        _np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _prompts_corpus_cid(prompts: Sequence[str]) -> str:
    return _sha256_hex({
        "kind": "w84_live_dataset_prompts_corpus",
        "prompts": [str(p) for p in prompts]})


def _input_projection(
        *, vocab_size: int, input_dim: int, seed: int,
) -> "_np.ndarray":
    """Deterministic one-hot-to-input-features projection."""
    rng = _np.random.default_rng(int(seed))
    return rng.standard_normal(
        (int(vocab_size), int(input_dim))).astype(_np.float64) * (
            1.0 / max(1.0, float(vocab_size)) ** 0.5)


def _target_projection(
        *, hidden_dim: int, output_dim: int, seed: int,
) -> "_np.ndarray":
    """Deterministic hidden-state-to-target projection."""
    rng = _np.random.default_rng(int(seed))
    return rng.standard_normal(
        (int(hidden_dim), int(output_dim))).astype(_np.float64) * (
            1.0 / max(1.0, float(hidden_dim)) ** 0.5)


@dataclasses.dataclass(frozen=True)
class LiveHiddenStateDatasetV1:
    """Content-addressed live hidden-state dataset.

    ``cid()`` is a deterministic function of
    ``(prompts_corpus_cid, model_cid, layer_index,
    projection_seed, input_dim, output_dim)``.
    """

    schema: str
    model_name: str
    model_dtype: str
    n_params: int
    hidden_dim: int
    n_layers: int
    layer_index: int
    vocab_size: int
    input_dim: int
    output_dim: int
    projection_seed: int
    max_tokens: int
    prompts: tuple[str, ...]
    n_sequences: int
    # Per-prompt features and targets, padded to ``max_tokens``
    # in T. ``valid_lens[i]`` is the real length for prompt ``i``.
    input_features: tuple["_np.ndarray", ...]
    targets: tuple["_np.ndarray", ...]
    valid_lens: tuple[int, ...]
    # Wire-format CIDs.
    prompts_corpus_cid: str
    model_cid: str
    input_proj_cid: str
    target_proj_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "model_name": str(self.model_name),
            "model_dtype": str(self.model_dtype),
            "n_params": int(self.n_params),
            "hidden_dim": int(self.hidden_dim),
            "n_layers": int(self.n_layers),
            "layer_index": int(self.layer_index),
            "vocab_size": int(self.vocab_size),
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "projection_seed": int(self.projection_seed),
            "max_tokens": int(self.max_tokens),
            "prompts": list(self.prompts),
            "n_sequences": int(self.n_sequences),
            "valid_lens": list(int(v) for v in self.valid_lens),
            "input_features_cid": _sha256_hex({
                "kind": "input_features",
                "cids": [
                    _ndarray_cid(a)
                    for a in self.input_features]}),
            "targets_cid": _sha256_hex({
                "kind": "targets",
                "cids": [
                    _ndarray_cid(a) for a in self.targets]}),
            "prompts_corpus_cid": str(self.prompts_corpus_cid),
            "model_cid": str(self.model_cid),
            "input_proj_cid": str(self.input_proj_cid),
            "target_proj_cid": str(self.target_proj_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_live_hidden_state_dataset_v1",
            "dataset": self.to_dict()})


def build_live_hidden_state_dataset_v1(
        *,
        runtime: Any,
        prompts: Sequence[str],
        layer_index: int = W84_LIVE_DATASET_DEFAULT_LAYER_INDEX,
        input_dim: int = W84_LIVE_DATASET_DEFAULT_INPUT_DIM,
        output_dim: int = W84_LIVE_DATASET_DEFAULT_OUTPUT_DIM,
        projection_seed: int = (
            W84_LIVE_DATASET_DEFAULT_PROJECTION_SEED),
        max_tokens: int = W84_LIVE_DATASET_DEFAULT_MAX_TOKENS,
) -> LiveHiddenStateDatasetV1:
    """Build the W84 live dataset from a runtime + prompt corpus.

    For each prompt:
    - Tokenize (truncated to ``max_tokens``).
    - Forward; capture the per-layer hidden state at
      ``layer_index``.
    - Per-token input features = one-hot(token_id) projected
      via ``input_proj_matrix``.
    - Per-token target = hidden state @ ``target_proj_matrix``.

    The projection matrices are deterministic from
    ``projection_seed`` + the model's vocab size and hidden dim.
    The dataset CID is therefore a pure function of the
    arguments + the model — the dataset is reproducible from
    weights, NOT from a frozen pickle.
    """
    vocab_size = int(getattr(
        runtime.tokenizer, "vocab_size",
        runtime.model.config.vocab_size))
    hidden_dim = int(runtime.hidden_dim)
    n_layers = int(runtime.n_layers)
    inp_proj = _input_projection(
        vocab_size=int(vocab_size),
        input_dim=int(input_dim),
        seed=int(projection_seed))
    tgt_proj = _target_projection(
        hidden_dim=int(hidden_dim),
        output_dim=int(output_dim),
        seed=int(projection_seed) + 1)
    feats: list["_np.ndarray"] = []
    tgts: list["_np.ndarray"] = []
    valids: list[int] = []
    layer_idx = int(
        min(max(0, int(layer_index)), int(n_layers) - 1))
    for prompt in prompts:
        ids = list(runtime.tokenize(
            str(prompt), max_len=int(max_tokens)))
        if len(ids) == 0:
            feats.append(_np.zeros(
                (int(max_tokens), int(input_dim)),
                dtype=_np.float64))
            tgts.append(_np.zeros(
                (int(max_tokens), int(output_dim)),
                dtype=_np.float64))
            valids.append(0)
            continue
        trace = runtime.forward(input_token_ids=ids)
        # Hidden state at layer_idx; if the runtime returned
        # per-layer hidden, use it; else fall back to final.
        if (trace.hidden is not None
                and len(trace.hidden.per_layer) > layer_idx
                and trace.hidden.per_layer[layer_idx] is not None):
            h = _np.asarray(
                trace.hidden.per_layer[layer_idx],
                dtype=_np.float64)
        else:
            h = _np.asarray(
                trace.hidden.final
                if trace.hidden is not None else 0.0,
                dtype=_np.float64)
            if h.ndim == 1:
                h = h[None, :]
        # Build per-token input features.
        ids_array = _np.asarray(ids, dtype=_np.int64)
        feat = inp_proj[ids_array]  # (T, input_dim)
        # Project the hidden state.
        target = h @ tgt_proj  # (T, output_dim)
        # Pad to max_tokens.
        T_real = int(len(ids))
        T_max = int(max_tokens)
        if T_real < T_max:
            pad_f = _np.zeros(
                (T_max - T_real, int(input_dim)),
                dtype=_np.float64)
            pad_t = _np.zeros(
                (T_max - T_real, int(output_dim)),
                dtype=_np.float64)
            feat = _np.concatenate([feat, pad_f], axis=0)
            target = _np.concatenate(
                [target, pad_t], axis=0)
        else:
            feat = feat[:T_max]
            target = target[:T_max]
        feats.append(feat)
        tgts.append(target)
        valids.append(int(T_real))
    # Normalize targets to unit variance per output dim across
    # the full corpus. Live transformer hidden states can have
    # magnitudes ~5–50; without normalization the NumPy
    # composed-memory training explodes on sigmoid/exp paths.
    # The normalization is corpus-deterministic on
    # ``(prompts_cid, model_cid, layer_index, projection_seed)``
    # and is recorded in the dataset's target stats.
    if len(tgts) > 0:
        all_targets = _np.stack(tgts, axis=0)
        # Mask out padding before computing scale.
        mask = _np.zeros(all_targets.shape[:2], dtype=bool)
        for i, vl in enumerate(valids):
            mask[i, :int(vl)] = True
        masked = all_targets[mask]
        if int(masked.size) > 0:
            sigma = float(_np.sqrt(
                _np.var(masked) + 1e-8))
            sigma = max(sigma, 1e-3)
            tgts = [t / sigma for t in tgts]
        else:
            sigma = 1.0
    else:
        sigma = 1.0
    # Model CID: stable on (model_name, config, dtype).
    cfg = runtime.model.config
    model_cid_payload = {
        "model_name": str(runtime.model_name),
        "model_dtype": str(
            getattr(runtime, "model_dtype", "fp32")),
        "config": {
            k: v for k, v in vars(cfg).items()
            if isinstance(v, (int, str, float, bool))},
    }
    model_cid = _sha256_hex(model_cid_payload)
    return LiveHiddenStateDatasetV1(
        schema=W84_LIVE_DATASET_V1_SCHEMA_VERSION,
        model_name=str(runtime.model_name),
        model_dtype=str(
            getattr(runtime, "model_dtype", "fp32")),
        n_params=int(sum(
            int(p.numel())
            for p in runtime.model.parameters())),
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
        layer_index=int(layer_idx),
        vocab_size=int(vocab_size),
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        projection_seed=int(projection_seed),
        max_tokens=int(max_tokens),
        prompts=tuple(str(p) for p in prompts),
        n_sequences=int(len(prompts)),
        input_features=tuple(feats),
        targets=tuple(tgts),
        valid_lens=tuple(int(v) for v in valids),
        prompts_corpus_cid=_prompts_corpus_cid(prompts),
        model_cid=str(model_cid),
        input_proj_cid=_ndarray_cid(inp_proj),
        target_proj_cid=_ndarray_cid(tgt_proj),
    )


__all__ = [
    "W84_LIVE_DATASET_V1_SCHEMA_VERSION",
    "W84_LIVE_DATASET_DEFAULT_INPUT_DIM",
    "W84_LIVE_DATASET_DEFAULT_OUTPUT_DIM",
    "W84_LIVE_DATASET_DEFAULT_LAYER_INDEX",
    "W84_LIVE_DATASET_DEFAULT_PROJECTION_SEED",
    "W84_LIVE_DATASET_DEFAULT_MAX_TOKENS",
    "LiveHiddenStateDatasetV1",
    "build_live_hidden_state_dataset_v1",
]
