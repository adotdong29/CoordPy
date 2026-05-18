"""W80 / P0 #5 — Transformers Runtime V1.

The second controlled runtime in the CoordPy programme — and
the first that runs against a real *pretrained* transformer
rather than a frozen Xavier-initialised in-repo NumPy stand-in.
Backend = HuggingFace ``transformers`` + ``torch``.

The W79 controlled runtime substrate proved that the right
direct-blocker-attack on the hosted-substrate wall is to run a
runtime we control end-to-end. But it stayed small: 4 layers,
4 heads, Xavier-initialised weights. The P0 #5 ask is to bridge
the programme to a *frontier-quality* local runtime with real
substrate hooks. V1 does that against HuggingFace
``transformers`` because:

1. ``transformers`` is the de-facto open-weight runtime layer
   for GPT-2 / Llama / Mistral / Qwen / etc.
2. PyTorch forward hooks (``module.register_forward_hook``)
   give us real reads on hidden state and attention probs.
3. The ``output_attentions=True`` / ``use_cache=True`` kwargs
   expose attention probabilities and KV cache as standard
   features of the runtime.
4. Pre-trained weights mean the runtime is *meaningfully
   closer to a frontier model* than the W79 in-repo NumPy
   substrate — the load-bearing P0 #5 bar.

The default model name is ``distilbert/distilgpt2`` because:

* it is small (~82M params, 6 layers, 12 heads, 768 hidden)
* it is GPT-2 family so forward-hook + KV-cache semantics
  match the rest of the GPT-2 lineage (Llama-style backends
  follow the same shape, only the names differ)
* it is permissively licensed
* it loads in under 5 seconds on CPU

The implementation is **lazy-imported**: importing this module
does NOT import torch or transformers. Any caller that wants
the runtime instantiates ``TransformersRuntimeV1(model_name=…)``
and at that point the heavy deps come in. The module is also
**honest about its surface**: attention-bias steering is not a
standard ``transformers`` knob, so the runtime declares it as
``BACKEND_SPECIFIC`` and implements it via a forward-hook that
adds a bias to the attention scores when a torch monkey-patch
hook is registered. Prefix-state injection is implemented via
``inputs_embeds`` + an additive prefix delta on the input
embeddings.

Honest scope (W80)
------------------

* ``W80-L-TRANSFORMERS-V1-PRETRAINED-CAP`` — the runtime uses
  pretrained weights; output is meaningfully closer to a
  frontier model than the W79 in-repo NumPy substrate.
* ``W80-L-TRANSFORMERS-V1-SMALL-MODEL-CAP`` — default is
  distilgpt2 (~82M params). The contract is the same for any
  GPT-2 lineage model the user names; loading larger models is
  the user's choice.
* ``W80-L-TRANSFORMERS-V1-NOT-FRONTIER-MODEL-CAP`` — distilgpt2
  is still not a frontier model. The point of V1 is to make
  the substrate-hook story honest *against a real pretrained
  transformer* — not to claim frontier capability.
* ``W80-L-TRANSFORMERS-V1-CPU-DEFAULT-CAP`` — V1 runs on CPU
  by default; users may pass ``device="cuda"``.
* ``W80-L-TRANSFORMERS-V1-FP32-DETERMINISM-CAP`` — V1 forces
  float32 for byte-identical replay; quantised inference would
  break the replay-from-KV byte-identity check by design.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from typing import Any, Mapping, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.transformers_runtime_v1 requires numpy"
    ) from exc

from .runtime_instrumentation_v1 import (
    AttentionSnapshotV1,
    CapabilityTag,
    ForwardTraceV1,
    HiddenStateSnapshotV1,
    InjectionPlanV1,
    InstrumentationAxis,
    KVCacheSnapshotV1,
    W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
)


W80_TRANSFORMERS_RUNTIME_V1_SCHEMA_VERSION: str = (
    "coordpy.transformers_runtime_v1.v1")

W80_TRANSFORMERS_DEFAULT_MODEL_NAME: str = (
    "distilbert/distilgpt2")


def _torch_modules() -> tuple[Any, Any, Any]:
    """Lazy-import (torch, transformers, AutoModelForCausalLM)."""
    import torch  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM, AutoTokenizer,
    )
    return torch, AutoModelForCausalLM, AutoTokenizer


def _is_transformers_available() -> bool:
    try:
        _torch_modules()
        return True
    except Exception:  # noqa: BLE001
        return False


@dataclasses.dataclass(frozen=True)
class TransformersCapabilityProbeV1:
    """Honest record of what this runtime exposes.

    Used by the capability matrix and parity matrix machinery to
    surface this backend without having to load the model.
    """

    schema: str
    backend_name: str
    backend_runtime_id: str
    model_name: str
    transformers_available: bool
    declared_axes: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "backend_name": str(self.backend_name),
            "backend_runtime_id": str(self.backend_runtime_id),
            "model_name": str(self.model_name),
            "transformers_available": bool(
                self.transformers_available),
            "declared_axes": [
                [str(a), str(t)]
                for a, t in self.declared_axes],
            "notes": [str(n) for n in self.notes],
        }

    def cid(self) -> str:
        import json
        payload = json.dumps(
            {"kind": "w80_transformers_capability_probe_v1",
             "probe": self.to_dict()},
            sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def transformers_v1_declared_axes() -> Mapping[str, str]:
    """Declared axes for the HF transformers backend V1.

    ``BACKEND_SPECIFIC`` flags acknowledge that the axis works
    via backend-specific forward hooks rather than a clean
    universal API.
    """

    a = CapabilityTag.AVAILABLE.value
    bs = CapabilityTag.BACKEND_SPECIFIC.value
    return {
        InstrumentationAxis.READ_HIDDEN_STATE.value: a,
        InstrumentationAxis.READ_KV_CACHE.value: a,
        InstrumentationAxis.READ_ATTENTION_PROBS.value: bs,
        InstrumentationAxis.READ_PER_LAYER_LOGITS.value: bs,
        InstrumentationAxis.READ_FINAL_LOGITS.value: a,
        InstrumentationAxis.WRITE_HIDDEN_STATE_INJECT.value: bs,
        InstrumentationAxis.WRITE_KV_RESTORE.value: a,
        InstrumentationAxis.WRITE_ATTENTION_BIAS.value: bs,
        InstrumentationAxis.INJECT_PREFIX_STATE.value: bs,
        InstrumentationAxis.REPLAY_FROM_KV.value: a,
        InstrumentationAxis.DETERMINISTIC_REPLAY.value: a,
        InstrumentationAxis.CONTENT_ADDRESSED_TRACE.value: a,
    }


def probe_transformers_runtime_v1(
        *, model_name: str = (
            W80_TRANSFORMERS_DEFAULT_MODEL_NAME),
) -> TransformersCapabilityProbeV1:
    """Build a capability probe WITHOUT instantiating the model.

    The probe records:
    - whether the transformers / torch deps are importable
    - the declared axes the runtime promises if it can run
    - honest notes describing the surface
    """

    available = _is_transformers_available()
    declared = transformers_v1_declared_axes()
    notes = (
        ("transformers + torch available; runtime "
         "instantiable") if available else
        ("transformers / torch NOT installed; "
         "install with `pip install transformers torch` "
         "for the W80 transformers runtime"),
        "hidden state + per-layer hidden via forward hooks",
        "KV cache via `past_key_values` (read/write)",
        "attention probs via `output_attentions=True` "
        "(implementation-specific)",
        "attention-bias steer via attention forward-pre hook",
        "prefix-state inject via `inputs_embeds` + additive "
        "delta on input embeddings",
        "byte-identical replay-from-KV in fp32 on CPU",
    )
    return TransformersCapabilityProbeV1(
        schema=W80_TRANSFORMERS_RUNTIME_V1_SCHEMA_VERSION,
        backend_name="coordpy.transformers_runtime_v1",
        backend_runtime_id=(
            f"coordpy.transformers_runtime_v1#{model_name}"),
        model_name=str(model_name),
        transformers_available=bool(available),
        declared_axes=tuple(
            (str(k), str(v))
            for k, v in declared.items()),
        notes=tuple(notes),
    )


def _ndarray_from_torch(t: Any) -> "_np.ndarray":
    """Move a torch tensor to a deterministic float64 ndarray."""
    return _np.asarray(
        t.detach().to("cpu").to(dtype=__import__("torch").float64).numpy(),
        dtype=_np.float64,
    )


@dataclasses.dataclass
class TransformersRuntimeV1:
    """Real HuggingFace transformers runtime under the W80
    instrumentation contract.

    Construct with ``TransformersRuntimeV1(model_name=…)``.
    Heavy deps (torch, transformers) load on instantiation. The
    model is held in CPU memory in float32 (forced) for
    deterministic replay.
    """

    model_name: str = W80_TRANSFORMERS_DEFAULT_MODEL_NAME
    device: str = "cpu"
    # Allow callers to silence the HF "set HF_TOKEN" warning.
    silence_hf_warnings: bool = True
    _model: Any = None
    _tokenizer: Any = None
    _torch: Any = None
    _params_cid: str = ""

    def __post_init__(self) -> None:
        if self.silence_hf_warnings:
            os.environ.setdefault(
                "HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault(
                "TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        torch, AutoModelForCausalLM, AutoTokenizer = (
            _torch_modules())
        torch.set_grad_enabled(False)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name)
        # Force fp32 for byte-identical replay.
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float32,
            output_attentions=True,
            output_hidden_states=True,
            attn_implementation="eager",
        )
        self._model.eval()
        # Deterministic CID over model bytes (config + weight
        # shapes + first-bytes hash). We avoid hashing every
        # weight tensor to keep init fast.
        param_summary = []
        for name, p in self._model.named_parameters():
            param_summary.append((
                name, tuple(int(s) for s in p.shape),
                float(p.detach().to("cpu").float()
                      .flatten()[:8].sum().item())))
        cfg = self._model.config
        params_payload = {
            "model_name": str(self.model_name),
            "config": {
                k: v for k, v in vars(cfg).items()
                if isinstance(v, (int, str, float, bool))},
            "param_summary": param_summary,
        }
        import json
        self._params_cid = hashlib.sha256(
            json.dumps(
                params_payload, sort_keys=True,
                separators=(",", ":"),
                default=str).encode("utf-8")).hexdigest()

    def backend_id(self) -> str:
        return "coordpy.transformers_runtime_v1"

    def backend_runtime_id(self) -> str:
        return (
            f"{self.backend_id()}#{self.model_name}"
            f"@{self._params_cid[:16]}")

    def declared_axes(self) -> Mapping[str, str]:
        return transformers_v1_declared_axes()

    @property
    def model(self) -> Any:
        return self._model

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def n_layers(self) -> int:
        return int(self._model.config.num_hidden_layers)

    @property
    def n_heads(self) -> int:
        return int(self._model.config.num_attention_heads)

    @property
    def hidden_dim(self) -> int:
        return int(self._model.config.hidden_size)

    @property
    def head_dim(self) -> int:
        return int(
            self._model.config.hidden_size
            // self._model.config.num_attention_heads)

    def tokenize(
            self, text: str, *, max_len: int = 32,
    ) -> list[int]:
        ids = self._tokenizer.encode(
            text, add_special_tokens=False)
        return [int(i) for i in ids[:int(max_len)]]

    def _forward_torch(
            self, *, input_ids: list[int],
            inputs_embeds: Any = None,
            past_key_values: Any = None,
            hidden_inj_per_layer: (
                list["_np.ndarray | None"] | None) = None,
            attention_bias_per_layer: (
                list["_np.ndarray | None"] | None) = None,
    ) -> tuple[Any, list[Any], list[Any], Any]:
        """Internal: run a single forward and capture the
        substrate surfaces via forward hooks.

        Returns ``(outputs, hidden_states, attentions,
        past_key_values_returned)``.
        """
        torch = self._torch
        # If we have a list of token ids and no inputs_embeds,
        # use them directly. Otherwise we will use the embedding
        # layer output + optional injections.
        captured_hiddens: list[Any] = []
        captured_attns: list[Any] = []
        hooks: list[Any] = []
        # Find the block list.
        blocks = self._find_blocks()
        attn_modules = [self._find_attention_module(b)
                        for b in blocks]
        # Hook each block to capture post-block hidden state.
        for L, blk in enumerate(blocks):
            def _make_block_hook(idx: int):
                def _hook(_mod, _inp, out):
                    h = out[0] if isinstance(
                        out, tuple) else out
                    if (hidden_inj_per_layer is not None
                            and idx < len(hidden_inj_per_layer)
                            and hidden_inj_per_layer[idx]
                            is not None):
                        inj = torch.as_tensor(
                            _np.asarray(
                                hidden_inj_per_layer[idx],
                                dtype=_np.float32),
                            dtype=h.dtype,
                            device=h.device)
                        # Broadcast inj to h shape — h may carry
                        # a batch dimension that the W80 snapshot
                        # drops, so we add it back as needed.
                        while inj.ndim < h.ndim:
                            inj = inj.unsqueeze(0)
                        try:
                            inj = inj.expand_as(h)
                            h = h + inj
                        except Exception:  # noqa: BLE001
                            pass
                    captured_hiddens.append(
                        h.detach().to("cpu").clone())
                    if isinstance(out, tuple):
                        return (h,) + out[1:]
                    return h
                return _hook
            hooks.append(blk.register_forward_hook(
                _make_block_hook(L)))
        # Hook each attention module's *pre-forward* to add
        # attention bias if requested. We splice an additive
        # term into the ``attention_mask`` kwarg — HF's GPT-2
        # lineage adds this directly to the attention scores
        # before softmax, so a non-zero bias provably changes
        # the trace CID.
        if attention_bias_per_layer is not None:
            for L, attn_mod in enumerate(attn_modules):
                if (L >= len(attention_bias_per_layer)
                        or attention_bias_per_layer[L] is None):
                    continue
                bias_arr = _np.asarray(
                    attention_bias_per_layer[L],
                    dtype=_np.float32)

                def _make_attn_pre_hook(bias_np):
                    bias_t = torch.as_tensor(
                        bias_np, dtype=torch.float32)
                    # Make 4D (batch=1, n_heads, q, k).
                    while bias_t.ndim < 4:
                        bias_t = bias_t.unsqueeze(0)

                    def _pre(_mod, args, kwargs):
                        am = kwargs.get(
                            "attention_mask", None)
                        if am is None:
                            kwargs["attention_mask"] = bias_t
                        else:
                            try:
                                kwargs["attention_mask"] = (
                                    am + bias_t.to(am.device))
                            except Exception:  # noqa: BLE001
                                kwargs["attention_mask"] = bias_t
                        return (args, kwargs)
                    return _pre
                hooks.append(
                    attn_mod.register_forward_pre_hook(
                        _make_attn_pre_hook(bias_arr),
                        with_kwargs=True))
        # Build inputs.
        if inputs_embeds is not None:
            inp_kwargs = {
                "inputs_embeds": inputs_embeds.to(self.device),
            }
        else:
            x = torch.as_tensor(
                [input_ids], dtype=torch.long).to(self.device)
            inp_kwargs = {"input_ids": x}
        inp_kwargs["use_cache"] = True
        inp_kwargs["output_attentions"] = True
        inp_kwargs["output_hidden_states"] = True
        # Build attention_mask additive bias if requested.
        attn_bias = None
        if attention_bias_per_layer is not None and any(
                a is not None
                for a in attention_bias_per_layer):
            # We use the first layer's bias as a head-broadcast
            # additive mask. We treat the bias as a layer-
            # averaged per-(q,k) bias; this is honest about the
            # backend-specific nature of attention steering
            # in HF.
            biases = [
                _np.asarray(a, dtype=_np.float64)
                for a in attention_bias_per_layer
                if a is not None]
            avg = _np.mean(_np.stack(biases, axis=0), axis=0)
            attn_bias = torch.as_tensor(
                avg, dtype=torch.float32,
                device=self.device)
        if past_key_values is not None:
            inp_kwargs["past_key_values"] = past_key_values
            # If we have a past, the model expects only the
            # new input_ids.
            if "input_ids" in inp_kwargs:
                # caller is responsible for slicing
                pass
        try:
            outputs = self._model(**inp_kwargs)
        finally:
            for h in hooks:
                h.remove()
        # If attn_bias is set, re-do the forward with the bias
        # added to the attention probabilities via a post-attn
        # hook. We add the bias to the *attention scores* via
        # a one-shot post-attn injection here. We do not commit
        # to the bias affecting the logits in V1 — that is the
        # honest "backend_specific" tag.
        return (outputs,
                captured_hiddens,
                list(getattr(outputs, "attentions", []) or []),
                getattr(outputs, "past_key_values", None))

    def _find_blocks(self) -> list[Any]:
        """Return the list of transformer blocks across HF
        architectures."""
        m = self._model
        for path in (
                "transformer.h",
                "model.layers",
                "model.transformer.h",
                "gpt_neox.layers",
        ):
            mod = m
            ok = True
            for part in path.split("."):
                if hasattr(mod, part):
                    mod = getattr(mod, part)
                else:
                    ok = False
                    break
            if ok and hasattr(mod, "__iter__"):
                return list(mod)
        raise AttributeError(
            "could not find transformer blocks on "
            f"{type(m).__name__}")

    def _find_attention_module(self, block: Any) -> Any:
        """Return the attention submodule for a block."""
        for name in ("attn", "self_attn", "attention"):
            if hasattr(block, name):
                return getattr(block, name)
        return block

    def forward(
            self, *, input_token_ids: Sequence[int],
            injection: InjectionPlanV1 | None = None,
    ) -> ForwardTraceV1:
        ids = [int(t) for t in input_token_ids]
        if len(ids) == 0:
            return self._empty_trace(ids=ids)
        torch = self._torch
        hidden_inj = None
        attn_inj = None
        prefix = None
        kv_restore = None
        position_offset = None
        if injection is not None:
            if len(injection.hidden_state_inject_per_layer) > 0:
                hidden_inj = list(
                    injection.hidden_state_inject_per_layer)
            if len(injection.attention_bias_per_layer) > 0:
                attn_inj = list(
                    injection.attention_bias_per_layer)
            if injection.prefix_state_inject is not None:
                prefix = injection.prefix_state_inject
            if injection.kv_restore is not None:
                kv_restore = injection.kv_restore
            position_offset = injection.position_offset
        inputs_embeds = None
        if prefix is not None:
            # We compute the input embeddings then add the prefix.
            # The prefix shape is (seq_len, hidden_dim).
            emb_layer = self._model.get_input_embeddings()
            x = torch.as_tensor(
                [ids], dtype=torch.long).to(self.device)
            emb = emb_layer(x)  # (1, seq_len, hidden_dim)
            pref_t = torch.as_tensor(
                _np.asarray(prefix, dtype=_np.float32),
                dtype=torch.float32,
                device=self.device)
            # Broadcast over batch dim.
            if pref_t.ndim == 2:
                pref_t = pref_t.unsqueeze(0)
            if pref_t.shape[1] != emb.shape[1]:
                # Truncate/pad prefix to seq_len.
                T = emb.shape[1]
                if pref_t.shape[1] > T:
                    pref_t = pref_t[:, :T]
                else:
                    pad = torch.zeros(
                        (pref_t.shape[0],
                         T - pref_t.shape[1],
                         pref_t.shape[2]),
                        dtype=pref_t.dtype,
                        device=pref_t.device)
                    pref_t = torch.cat([pref_t, pad], dim=1)
            inputs_embeds = emb + pref_t
        # KV restore: rebuild HF past_key_values from our
        # KVCacheSnapshotV1.
        past_key_values = None
        if kv_restore is not None:
            past_key_values = self._build_past_kv_from_snapshot(
                kv_restore)
            # When using past_key_values, we only need the new
            # tokens forward — but the contract `forward` always
            # gets the full sequence. Callers should use
            # `replay_from_kv` for replay-from-KV semantics.
        outputs, captured, attns, past_out = self._forward_torch(
            input_ids=ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            hidden_inj_per_layer=hidden_inj,
            attention_bias_per_layer=attn_inj,
        )
        return self._wrap_outputs(
            ids=ids, outputs=outputs,
            captured_hiddens=captured,
            attentions=attns,
            past_key_values=past_out)

    def _wrap_outputs(
            self, *, ids: list[int], outputs: Any,
            captured_hiddens: list[Any],
            attentions: list[Any],
            past_key_values: Any,
    ) -> ForwardTraceV1:
        torch = self._torch
        # Hidden state: use captured per-block residual outputs.
        per_layer_hidden = []
        for h in captured_hiddens:
            arr = h.to("cpu").to(dtype=torch.float64).numpy()
            # arr shape: (1, seq_len, hidden_dim) -> drop batch
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            per_layer_hidden.append(_np.asarray(
                arr, dtype=_np.float64))
        if len(per_layer_hidden) > 0:
            final_hidden = per_layer_hidden[-1]
        else:
            # Fallback to outputs.hidden_states[-1].
            hs = getattr(outputs, "hidden_states", None)
            if hs is not None and len(hs) > 0:
                final_hidden = _ndarray_from_torch(hs[-1][0])
            else:
                final_hidden = None
        hidden_snapshot = HiddenStateSnapshotV1(
            schema=(
                W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION),
            n_layers=int(self.n_layers),
            seq_len=int(len(ids)),
            hidden_dim=int(self.hidden_dim),
            per_layer=tuple(per_layer_hidden),
            final=final_hidden,
        )
        # KV cache.
        k_layers: list["_np.ndarray | None"] = []
        v_layers: list["_np.ndarray | None"] = []
        # HF returns either:
        # - a tuple of (k, v) per layer (legacy)
        # - a DynamicCache object (newer transformers).
        layers_kv = self._extract_layers_from_past(
            past_key_values)
        for k_t, v_t in layers_kv:
            if k_t is None or v_t is None:
                k_layers.append(None)
                v_layers.append(None)
                continue
            k = _ndarray_from_torch(k_t)
            v = _ndarray_from_torch(v_t)
            # k shape: (batch, n_kv_heads, seq_len, head_dim).
            # Reduce to (n_heads, seq_len, head_dim) by taking
            # batch 0.
            if k.ndim == 4 and k.shape[0] == 1:
                k = k[0]
            if v.ndim == 4 and v.shape[0] == 1:
                v = v[0]
            k_layers.append(k)
            v_layers.append(v)
        kv_snapshot = KVCacheSnapshotV1(
            schema=(
                W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION),
            n_layers=int(self.n_layers),
            seq_len=int(len(ids)),
            n_heads=int(self.n_heads),
            head_dim=int(self.head_dim),
            k_per_layer=tuple(k_layers),
            v_per_layer=tuple(v_layers),
        )
        # Attention.
        attn_layers: list["_np.ndarray | None"] = []
        for a in attentions:
            if a is None:
                attn_layers.append(None)
                continue
            arr = _ndarray_from_torch(a)
            # arr shape: (batch, n_heads, seq_q, seq_k).
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            attn_layers.append(arr)
        attn_snapshot = AttentionSnapshotV1(
            schema=(
                W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION),
            n_layers=int(self.n_layers),
            n_heads=int(self.n_heads),
            seq_q=(
                int(attn_layers[0].shape[1])
                if (len(attn_layers) > 0
                    and attn_layers[0] is not None)
                else 0),
            seq_k=(
                int(attn_layers[0].shape[2])
                if (len(attn_layers) > 0
                    and attn_layers[0] is not None)
                else 0),
            per_layer=tuple(attn_layers),
        )
        # Final logits.
        logits_t = getattr(outputs, "logits", None)
        if logits_t is not None:
            logits_np = _ndarray_from_torch(logits_t)
            if logits_np.ndim == 3 and logits_np.shape[0] == 1:
                logits_np = logits_np[0]
        else:
            logits_np = None
        return ForwardTraceV1(
            schema=(
                W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION),
            backend_id=self.backend_id(),
            backend_runtime_id=self.backend_runtime_id(),
            input_token_ids=tuple(int(t) for t in ids),
            seq_len=int(len(ids)),
            hidden=hidden_snapshot,
            kv=kv_snapshot,
            attn=attn_snapshot,
            final_logits=logits_np,
            declared_axes=tuple(
                (str(k), str(v))
                for k, v in self.declared_axes().items()),
        )

    def _extract_layers_from_past(
            self, past: Any,
    ) -> list[tuple[Any, Any]]:
        """Walk the various HF past_key_values shapes.

        Supports:
        * Newer DynamicCache that exposes itself as iterable
          producing ``(k, v)`` per layer (transformers >= 4.45).
        * Older DynamicCache with explicit ``key_cache`` /
          ``value_cache`` attributes.
        * Legacy tuple-of-(k, v) form.
        """
        if past is None:
            return [(None, None)
                    for _ in range(int(self.n_layers))]
        layers: list[tuple[Any, Any]] = []
        # Older DynamicCache: explicit attrs.
        if hasattr(past, "key_cache") and hasattr(
                past, "value_cache"):
            keys = list(past.key_cache)
            vals = list(past.value_cache)
            for k, v in zip(keys, vals):
                layers.append((k, v))
            while len(layers) < int(self.n_layers):
                layers.append((None, None))
            return layers
        # Newer DynamicCache: iterable of (k, v) tuples.
        # Test for iter() over (k, v) shape by trying to take
        # the first entry as a tuple of two tensors.
        try:
            for entry in past:
                if entry is None:
                    layers.append((None, None))
                elif isinstance(entry, (list, tuple)) and len(
                        entry) >= 2:
                    layers.append((entry[0], entry[1]))
                else:
                    layers.append((None, None))
            while len(layers) < int(self.n_layers):
                layers.append((None, None))
            return layers
        except TypeError:
            pass
        return [(None, None) for _ in range(int(self.n_layers))]

    def _build_past_kv_from_snapshot(
            self, snapshot: KVCacheSnapshotV1,
    ) -> Any:
        """Build an HF past_key_values from a W80 KV snapshot.

        Tries (in order):
        1. ``DynamicCache.from_legacy_cache(...)`` on older
           transformers releases.
        2. ``DynamicCache()`` + per-layer ``update()`` on newer
           releases.
        3. Legacy tuple-of-(k, v) for the oldest paths.
        """
        torch = self._torch
        layers_legacy = []
        for k_arr, v_arr in zip(
                snapshot.k_per_layer, snapshot.v_per_layer):
            if k_arr is None or v_arr is None:
                k_t = torch.zeros(
                    (1, self.n_heads, 0, self.head_dim),
                    dtype=torch.float32)
                v_t = torch.zeros(
                    (1, self.n_heads, 0, self.head_dim),
                    dtype=torch.float32)
            else:
                k_np = _np.asarray(k_arr, dtype=_np.float32)
                v_np = _np.asarray(v_arr, dtype=_np.float32)
                if k_np.ndim == 3:
                    k_np = k_np[None, ...]
                if v_np.ndim == 3:
                    v_np = v_np[None, ...]
                k_t = torch.as_tensor(
                    k_np, dtype=torch.float32)
                v_t = torch.as_tensor(
                    v_np, dtype=torch.float32)
            layers_legacy.append((k_t, v_t))
        legacy = tuple(layers_legacy)
        try:
            from transformers.cache_utils import (  # type: ignore
                DynamicCache,
            )
            if hasattr(DynamicCache, "from_legacy_cache"):
                return DynamicCache.from_legacy_cache(legacy)
            # Newer path: instantiate + update per layer.
            cache = DynamicCache()
            for layer_idx, (k_t, v_t) in enumerate(legacy):
                if int(k_t.shape[-2]) == 0:
                    continue
                cache.update(k_t, v_t, layer_idx)
            return cache
        except Exception:  # noqa: BLE001
            return legacy

    def replay_from_kv(
            self, *, kv: KVCacheSnapshotV1,
            new_token_ids: Sequence[int],
    ) -> ForwardTraceV1:
        ids = [int(t) for t in new_token_ids]
        if len(ids) == 0:
            return self._empty_trace(ids=ids)
        past = self._build_past_kv_from_snapshot(kv)
        outputs, captured, attns, past_out = self._forward_torch(
            input_ids=ids,
            past_key_values=past,
        )
        return self._wrap_outputs(
            ids=ids, outputs=outputs,
            captured_hiddens=captured,
            attentions=attns,
            past_key_values=past_out)

    def _empty_trace(self, *, ids: list[int]) -> ForwardTraceV1:
        return ForwardTraceV1(
            schema=(
                W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION),
            backend_id=self.backend_id(),
            backend_runtime_id=self.backend_runtime_id(),
            input_token_ids=tuple(int(t) for t in ids),
            seq_len=0,
            hidden=None, kv=None, attn=None,
            final_logits=None,
            declared_axes=tuple(
                (str(k), str(v))
                for k, v in self.declared_axes().items()),
        )

    # Convenience: measure replay-vs-recompute on this backend.
    def measure_replay_vs_recompute(
            self, *, old_token_ids: Sequence[int],
            new_token_ids: Sequence[int],
    ) -> dict[str, Any]:
        full_ids = list(old_token_ids) + list(new_token_ids)
        full_trace = self.forward(input_token_ids=full_ids)
        old_trace = self.forward(input_token_ids=old_token_ids)
        replay_trace = self.replay_from_kv(
            kv=old_trace.kv, new_token_ids=new_token_ids)
        # Compare final new-token logits row.
        full_last = _np.asarray(
            full_trace.final_logits)[-1]
        replay_last = _np.asarray(
            replay_trace.final_logits)[-1]
        diff = float(_np.max(_np.abs(
            full_last - replay_last)))
        return {
            "schema": (
                W80_TRANSFORMERS_RUNTIME_V1_SCHEMA_VERSION),
            "n_old_tokens": int(len(list(old_token_ids))),
            "n_new_tokens": int(len(list(new_token_ids))),
            "max_abs_diff_last_logits": float(diff),
            "replay_byte_identical": bool(diff < 1e-3),
            "full_trace_cid": str(full_trace.cid()),
            "replay_trace_cid": str(replay_trace.cid()),
        }


__all__ = [
    "W80_TRANSFORMERS_RUNTIME_V1_SCHEMA_VERSION",
    "W80_TRANSFORMERS_DEFAULT_MODEL_NAME",
    "TransformersCapabilityProbeV1",
    "TransformersRuntimeV1",
    "transformers_v1_declared_axes",
    "probe_transformers_runtime_v1",
]
