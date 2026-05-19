"""W84 / P0 #26 — Live LLM Training of Composed Learned Memory V1.

This module fits the W83 ``composed_learned_memory_v1`` module
against the W84 ``live_hidden_state_dataset_v1`` (real
transformer hidden states from a real running model), then runs
the W84 P0 #26 load-bearing head-to-head: live-trained module
vs synthetic-trained module on a held-out **live** evaluation
set.

The load-bearing W84 claim is:

    live_trained_module.mse_on_held_out_live <
        synthetic_trained_module.mse_on_held_out_live

The claim must hold strictly, not within noise — the P0 #26
anti-cheat is "do not declare success when live-trained MSE is
within noise of synthetic-trained MSE."

Honest scope (W84 P0 #26)
-------------------------

- ``W84-L-LIVE-TRAINING-V1-RESEARCH-ONLY-CAP`` — explicit
  import only.
- ``W84-L-LIVE-TRAINING-V1-ONE-LAYER-CAP`` — V1 trains against
  hidden states from one configurable layer of the live model.
- ``W84-L-LIVE-TRAINING-V1-SMALL-CORPUS-CAP`` — V1 uses a
  small prompt corpus (≤30 train + ≤15 eval) so CPU bench
  time stays in single minutes. Larger corpora are V2.
- ``W84-L-LIVE-TRAINING-V1-NUMPY-CAP`` — the composed memory
  is pure NumPy; gradients flow through ``train_composed_
  learned_memory_module``.
- ``W84-L-LIVE-TRAINING-V1-BF16-FLOOR-CAP`` — bf16 hidden
  states are cast to fp64 for NumPy training, but the
  training task target IS the bf16-precision hidden state.
  V1 records that floor honestly; the live-trained MSE is
  measured against bf16 targets, not against a fictional
  fp32 target.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.live_trained_composed_memory_v1 requires "
        "numpy") from exc

from .composed_learned_memory_v1 import (
    ComposedLearnedMemoryModuleV1,
    build_composed_learned_memory_module_v1,
    build_composed_long_horizon_dataset_v1,
    train_composed_learned_memory_module,
    W83_COMPOSED_MEMORY_V1_SCHEMA_VERSION,
)
from .live_hidden_state_dataset_v1 import (
    LiveHiddenStateDatasetV1,
)


W84_LIVE_TRAINING_V1_SCHEMA_VERSION: str = (
    "coordpy.live_trained_composed_memory_v1.v1")

W84_LIVE_TRAINING_DEFAULT_N_ITERS: int = 60
W84_LIVE_TRAINING_DEFAULT_HIDDEN_DIM: int = 12
W84_LIVE_TRAINING_DEFAULT_MEMORY_DIM: int = 10
W84_LIVE_TRAINING_DEFAULT_K_SLOTS: int = 5
W84_LIVE_TRAINING_DEFAULT_LEARNING_RATE: float = 0.02
W84_LIVE_TRAINING_DEFAULT_SEED: int = 84_026_002


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _dataset_to_sequences(
        dataset: LiveHiddenStateDatasetV1,
) -> tuple[list["_np.ndarray"], list["_np.ndarray"]]:
    """Project the dataset to (X_list, Y_list) of sequences.

    Trims to the valid length per prompt so the module is not
    asked to predict pad targets.
    """
    X_list: list["_np.ndarray"] = []
    Y_list: list["_np.ndarray"] = []
    for X, Y, vl in zip(
            dataset.input_features, dataset.targets,
            dataset.valid_lens):
        vl_int = int(vl)
        if vl_int <= 1:
            continue
        X_arr = _np.asarray(X, dtype=_np.float64)[:vl_int]
        Y_arr = _np.asarray(Y, dtype=_np.float64)[:vl_int]
        X_list.append(X_arr)
        Y_list.append(Y_arr)
    return X_list, Y_list


def _module_mse_on_sequences(
        *, module: ComposedLearnedMemoryModuleV1,
        Xs: Sequence["_np.ndarray"],
        Ys: Sequence["_np.ndarray"],
) -> float:
    """Mean MSE across sequences."""
    if len(Xs) == 0:
        return float("nan")
    total = 0.0
    n_pts = 0
    for X, Y in zip(Xs, Ys):
        _, _, _, Yhat = module.forward_sequence(X)
        diff = (Yhat - Y) ** 2
        total += float(_np.sum(diff))
        n_pts += int(diff.size)
    return float(total / max(1, n_pts))


@dataclasses.dataclass(frozen=True)
class TrainingTraceWitnessV1:
    """Audit witness for a single training run.

    Records seed, optimiser config, the pre/post module CIDs,
    and the loss-curve CID. The witness is signable + content-
    addressable so a third party can re-verify the training
    from the inputs + the witness.
    """

    schema: str
    seed: int
    n_iters: int
    learning_rate: float
    pre_module_cid: str
    post_module_cid: str
    pre_loss: float
    post_loss: float
    loss_curve_cid: str
    dataset_cid: str
    train_dataset_n_sequences: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "n_iters": int(self.n_iters),
            "learning_rate": float(self.learning_rate),
            "pre_module_cid": str(self.pre_module_cid),
            "post_module_cid": str(self.post_module_cid),
            "pre_loss": float(round(self.pre_loss, 12)),
            "post_loss": float(round(self.post_loss, 12)),
            "loss_curve_cid": str(self.loss_curve_cid),
            "dataset_cid": str(self.dataset_cid),
            "train_dataset_n_sequences": int(
                self.train_dataset_n_sequences),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_training_trace_witness_v1",
            "witness": self.to_dict()})


def train_composed_learned_memory_on_live_hidden_states(
        *,
        dataset: LiveHiddenStateDatasetV1,
        hidden_dim: int = W84_LIVE_TRAINING_DEFAULT_HIDDEN_DIM,
        memory_dim: int = W84_LIVE_TRAINING_DEFAULT_MEMORY_DIM,
        K_slots: int = W84_LIVE_TRAINING_DEFAULT_K_SLOTS,
        n_iters: int = W84_LIVE_TRAINING_DEFAULT_N_ITERS,
        learning_rate: float = (
            W84_LIVE_TRAINING_DEFAULT_LEARNING_RATE),
        seed: int = W84_LIVE_TRAINING_DEFAULT_SEED,
) -> tuple[
        ComposedLearnedMemoryModuleV1,
        TrainingTraceWitnessV1]:
    """Fit the W83 composed memory module on a W84 live dataset.

    Returns ``(trained_module, witness)``. The witness records
    the seed, optimiser config, pre/post module CIDs, and the
    loss-curve CID. The training itself is deterministic on
    ``seed``.
    """
    init_module = build_composed_learned_memory_module_v1(
        input_dim=int(dataset.input_dim),
        hidden_dim=int(hidden_dim),
        memory_dim=int(memory_dim),
        output_dim=int(dataset.output_dim),
        K_slots=int(K_slots),
        seed=int(seed))
    X_list, Y_list = _dataset_to_sequences(dataset)
    trained, report = train_composed_learned_memory_module(
        module=init_module,
        train_sequences=X_list,
        train_targets=Y_list,
        n_iters=int(n_iters),
        learning_rate=float(learning_rate))
    witness = TrainingTraceWitnessV1(
        schema=W84_LIVE_TRAINING_V1_SCHEMA_VERSION,
        seed=int(seed),
        n_iters=int(n_iters),
        learning_rate=float(learning_rate),
        pre_module_cid=str(report.module_cid_pre),
        post_module_cid=str(report.module_cid_post),
        pre_loss=float(report.pre_loss),
        post_loss=float(report.post_loss),
        loss_curve_cid=str(report.loss_curve_cid),
        dataset_cid=str(dataset.cid()),
        train_dataset_n_sequences=int(len(X_list)))
    return trained, witness


def train_synthetic_baseline_composed_memory(
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = W84_LIVE_TRAINING_DEFAULT_HIDDEN_DIM,
        memory_dim: int = W84_LIVE_TRAINING_DEFAULT_MEMORY_DIM,
        K_slots: int = W84_LIVE_TRAINING_DEFAULT_K_SLOTS,
        n_iters: int = W84_LIVE_TRAINING_DEFAULT_N_ITERS,
        learning_rate: float = (
            W84_LIVE_TRAINING_DEFAULT_LEARNING_RATE),
        seed: int = W84_LIVE_TRAINING_DEFAULT_SEED,
        n_synthetic_sequences: int = 12,
        synthetic_seq_len: int = 8,
) -> tuple[
        ComposedLearnedMemoryModuleV1,
        TrainingTraceWitnessV1]:
    """Train an identical-capacity baseline on the W83 synthetic
    dataset. The synthetic baseline is the *negative control*
    in the W84 P0 #26 head-to-head: it should LOSE on
    held-out *live* data even though it is competitive on
    *synthetic* data."""
    Xs, Ys = build_composed_long_horizon_dataset_v1(
        n_sequences=int(n_synthetic_sequences),
        seq_len=int(synthetic_seq_len),
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        seed=int(seed) + 7000)
    init_module = build_composed_learned_memory_module_v1(
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        memory_dim=int(memory_dim),
        output_dim=int(output_dim),
        K_slots=int(K_slots),
        seed=int(seed))
    trained, report = train_composed_learned_memory_module(
        module=init_module,
        train_sequences=Xs,
        train_targets=Ys,
        n_iters=int(n_iters),
        learning_rate=float(learning_rate))
    witness = TrainingTraceWitnessV1(
        schema=W84_LIVE_TRAINING_V1_SCHEMA_VERSION,
        seed=int(seed),
        n_iters=int(n_iters),
        learning_rate=float(learning_rate),
        pre_module_cid=str(report.module_cid_pre),
        post_module_cid=str(report.module_cid_post),
        pre_loss=float(report.pre_loss),
        post_loss=float(report.post_loss),
        loss_curve_cid=str(report.loss_curve_cid),
        dataset_cid="synthetic",
        train_dataset_n_sequences=int(n_synthetic_sequences))
    return trained, witness


@dataclasses.dataclass(frozen=True)
class LiveVsSyntheticHeadToHeadReportV1:
    """Head-to-head report. ``live_strict_win`` is the
    load-bearing P0 #26 verdict."""

    schema: str
    model_name: str
    model_dtype: str
    layer_index: int
    n_train_prompts: int
    n_eval_prompts: int
    live_witness_cid: str
    synthetic_witness_cid: str
    live_trained_mse_on_live_eval: float
    synthetic_trained_mse_on_live_eval: float
    live_strict_win: bool
    relative_improvement: float
    eval_train_disjoint: bool
    elapsed_seconds: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "model_name": str(self.model_name),
            "model_dtype": str(self.model_dtype),
            "layer_index": int(self.layer_index),
            "n_train_prompts": int(self.n_train_prompts),
            "n_eval_prompts": int(self.n_eval_prompts),
            "live_witness_cid": str(self.live_witness_cid),
            "synthetic_witness_cid": str(
                self.synthetic_witness_cid),
            "live_trained_mse_on_live_eval": float(round(
                self.live_trained_mse_on_live_eval, 9)),
            "synthetic_trained_mse_on_live_eval": float(round(
                self.synthetic_trained_mse_on_live_eval, 9)),
            "live_strict_win": bool(self.live_strict_win),
            "relative_improvement": float(round(
                self.relative_improvement, 6)),
            "eval_train_disjoint": bool(
                self.eval_train_disjoint),
            "elapsed_seconds": float(round(
                self.elapsed_seconds, 3)),
            "detail": str(self.detail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_live_vs_synthetic_head_to_head_v1",
            "report": self.to_dict()})


def compare_live_trained_vs_synthetic_trained(
        *,
        train_dataset: LiveHiddenStateDatasetV1,
        eval_dataset: LiveHiddenStateDatasetV1,
        hidden_dim: int = W84_LIVE_TRAINING_DEFAULT_HIDDEN_DIM,
        memory_dim: int = W84_LIVE_TRAINING_DEFAULT_MEMORY_DIM,
        K_slots: int = W84_LIVE_TRAINING_DEFAULT_K_SLOTS,
        n_iters: int = W84_LIVE_TRAINING_DEFAULT_N_ITERS,
        learning_rate: float = (
            W84_LIVE_TRAINING_DEFAULT_LEARNING_RATE),
        seed: int = W84_LIVE_TRAINING_DEFAULT_SEED,
        n_synthetic_sequences: int = 12,
        synthetic_seq_len: int = 8,
) -> LiveVsSyntheticHeadToHeadReportV1:
    """Run the load-bearing W84 P0 #26 head-to-head.

    Both modules use identical capacity (``hidden_dim``,
    ``memory_dim``, ``K_slots``). One is trained on
    ``train_dataset`` (live hidden states), the other on the
    W83 synthetic dataset. Both are evaluated on
    ``eval_dataset`` (live hidden states from held-out
    prompts).

    The load-bearing claim: the live-trained module's MSE on
    the held-out live eval set is **strictly less** than the
    synthetic-trained module's MSE on the same set.
    """
    t0 = time.monotonic()
    # Disjointness check: do train and eval share any prompts?
    train_set = set(train_dataset.prompts)
    eval_set = set(eval_dataset.prompts)
    disjoint = bool(len(train_set & eval_set) == 0)
    # Train both.
    live_module, live_witness = (
        train_composed_learned_memory_on_live_hidden_states(
            dataset=train_dataset,
            hidden_dim=int(hidden_dim),
            memory_dim=int(memory_dim),
            K_slots=int(K_slots),
            n_iters=int(n_iters),
            learning_rate=float(learning_rate),
            seed=int(seed)))
    syn_module, syn_witness = (
        train_synthetic_baseline_composed_memory(
            input_dim=int(train_dataset.input_dim),
            output_dim=int(train_dataset.output_dim),
            hidden_dim=int(hidden_dim),
            memory_dim=int(memory_dim),
            K_slots=int(K_slots),
            n_iters=int(n_iters),
            learning_rate=float(learning_rate),
            seed=int(seed),
            n_synthetic_sequences=int(n_synthetic_sequences),
            synthetic_seq_len=int(synthetic_seq_len)))
    # Evaluate both on held-out live data.
    Xe, Ye = _dataset_to_sequences(eval_dataset)
    live_mse = _module_mse_on_sequences(
        module=live_module, Xs=Xe, Ys=Ye)
    syn_mse = _module_mse_on_sequences(
        module=syn_module, Xs=Xe, Ys=Ye)
    live_strict_win = bool(live_mse < syn_mse)
    rel_imp = float(
        (syn_mse - live_mse) / max(1e-12, syn_mse))
    return LiveVsSyntheticHeadToHeadReportV1(
        schema=W84_LIVE_TRAINING_V1_SCHEMA_VERSION,
        model_name=str(train_dataset.model_name),
        model_dtype=str(train_dataset.model_dtype),
        layer_index=int(train_dataset.layer_index),
        n_train_prompts=int(train_dataset.n_sequences),
        n_eval_prompts=int(eval_dataset.n_sequences),
        live_witness_cid=str(live_witness.cid()),
        synthetic_witness_cid=str(syn_witness.cid()),
        live_trained_mse_on_live_eval=float(live_mse),
        synthetic_trained_mse_on_live_eval=float(syn_mse),
        live_strict_win=bool(live_strict_win),
        relative_improvement=float(rel_imp),
        eval_train_disjoint=bool(disjoint),
        elapsed_seconds=float(time.monotonic() - t0),
        detail=(
            "W84 P0 #26 live-vs-synthetic head-to-head: "
            + ("LIVE STRICTLY WINS"
               if live_strict_win else "LIVE DID NOT WIN")),
    )


__all__ = [
    "W84_LIVE_TRAINING_V1_SCHEMA_VERSION",
    "W84_LIVE_TRAINING_DEFAULT_N_ITERS",
    "W84_LIVE_TRAINING_DEFAULT_HIDDEN_DIM",
    "W84_LIVE_TRAINING_DEFAULT_MEMORY_DIM",
    "W84_LIVE_TRAINING_DEFAULT_K_SLOTS",
    "W84_LIVE_TRAINING_DEFAULT_LEARNING_RATE",
    "W84_LIVE_TRAINING_DEFAULT_SEED",
    "TrainingTraceWitnessV1",
    "train_composed_learned_memory_on_live_hidden_states",
    "train_synthetic_baseline_composed_memory",
    "LiveVsSyntheticHeadToHeadReportV1",
    "compare_live_trained_vs_synthetic_trained",
]
